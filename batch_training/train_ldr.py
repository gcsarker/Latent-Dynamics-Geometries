import os

# For headless MuJoCo rendering on a remote GPU server
# os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass, asdict
import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from torch.distributions import Normal
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
import pandas as pd
from pathlib import Path
import json
import mujoco
import random

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import r2_score, silhouette_score
# import matplotlib.cm as cm



@dataclass
class LDRConfig:
    # SAC and training settings (Xu: configs/agents/hopper/ldr.yaml)
    total_timesteps: int = 1_500_000
    learning_starts: int = 0             # Xu: train_start_iterations = 0, policy acts from the start
    train_freq: int = 2048               # env steps collected per optimize() call
    gradient_steps: int = 2048           # gradient updates per optimize() call (1 per env step)
    batch_size: int = 256
    eval_interval: int = 100_000
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    reward_scale: float = 0.008
    target_update_interval: int = 1
    hidden_dim: int = 256
    seed: int = 1012

    # LDR-specific settings
    n_past: int = 50
    latent_dim: int = 3                  # Xu uses 5 only for Ant
    vae_loss_weight: float = 5.0
    contrastive_loss_weight: float = 1.0
    infonce_temp: float = 1.0
    contrastive_start_iterations: int = 0
    latent_dropout_prob: float = 0.1
    min_positives: int = 1

    # KL annealing (Xu's kl_schedule): beta goes 0 -> beta_max linearly
    # over the first beta_anneal_fraction of training, then stays at beta_max
    beta_max: float = 0.1
    beta_anneal_fraction: float = 0.5



class HistoryQueue:
    def __init__(self, n_envs, n_past, obs_dim, action_dim):
        self.n_envs, self.H, self.obs_dim, self.action_dim = n_envs, n_past, obs_dim, action_dim
        self.data = np.zeros((n_envs, n_past, obs_dim + action_dim), dtype=np.float32)
    
    def get(self):
        return self.data.copy()
    
    def append(self, obs, actions):
        # shifting all data except the last 1 row. then add the new obs and actions to the first row
        self.data[:,1:] = self.data[:,:-1].copy()
        self.data[:,0,:self.obs_dim] = obs
        self.data[:,0,self.obs_dim:] = actions
    
    # def reset_done(self, dones):
    #     self.data[np.asarray(dones, dtype=bool)] = 0

    def reset_done(self, dones, initial_obs):
        dones = np.asarray(dones, dtype=bool)
        self.data[dones] = 0
        if initial_obs is not None:
            self.data[dones, :, :self.obs_dim] = initial_obs[dones][:, None, :]


# Need to send latent_dim from the LDRAgent

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=256, latent_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_norm = nn.LayerNorm(latent_dim)

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Gaussian distribution parameters
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs):
        # obs = [env_obs, z]; normalize z before the MLP
        if self.latent_dim > 0:
            s, z = obs[..., :-self.latent_dim], obs[..., -self.latent_dim:]
            obs = torch.cat([s, self.latent_norm(z)], dim=-1)

        features = self.net(obs)
        mean = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  
        std = log_std.exp()

        return mean, std


    # Sample actions and compute log probabilities
    def actionLogProb(self, obs):

        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        u = dist.rsample() # reparameterization trick for backpropagation
        action = torch.tanh(u) # action between -1 and 1

        # Calculting log \pi(a|S)
        log_prob = dist.log_prob(u)
        log_prob -= 2 * (torch.log(torch.tensor(2.0, device=u.device)) - u - nn.functional.softplus(-2 * u))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def predict(self, obs, deterministic=True):
        """Return normalized actions for inference."""

        mean, std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean)

        dist = Normal(mean, std)
        return torch.tanh(dist.sample())



class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=256, latent_dim=0):
        super().__init__()

        input_dim = obs_dim + action_dim
        self.latent_dim = latent_dim

        # First Q network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Second Q network
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # One LayerNorm per Q head (Xu deep-copies the extractor for each head)
        if latent_dim > 0:
            self.norm1 = nn.LayerNorm(latent_dim)
            self.norm2 = nn.LayerNorm(latent_dim)

    def _input(self, obs, action, norm):
        if self.latent_dim > 0:
            s, z = obs[..., :-self.latent_dim], obs[..., -self.latent_dim:]
            obs = torch.cat([s, norm(z)], dim=-1)
        return torch.cat([obs, action], dim=-1)

    def forward(self, obs, action):
        if self.latent_dim > 0:
            q1 = self.q1(self._input(obs, action, self.norm1))
            q2 = self.q2(self._input(obs, action, self.norm2))
        else:
            x = torch.cat([obs, action], dim=-1)
            q1, q2 = self.q1(x), self.q2(x)

        return q1, q2



class ReplayBuffer:

    def __init__(self, buffer_size, num_envs, obs_dim, action_dim, device, n_past):

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Capacity measured in vectorized steps
        self.capacity = max(buffer_size // num_envs, 1 )

        # Allocate storage
        self.obs = np.zeros((self.capacity, num_envs, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((self.capacity, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.timeouts = np.zeros_like(self.dones)

        # LDR specific
        self.histories = np.zeros((self.capacity, num_envs, n_past, obs_dim + action_dim), dtype=np.float32)
        self.dynamics_ids = np.zeros((self.capacity, num_envs), dtype=np.int64)

        self.pos = 0
        self.full = False

    def push(self, obs, next_obs, actions, rewards, dones, timeouts, history, dynamics_ids):

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.timeouts[self.pos] = timeouts
        self.histories[self.pos] = history
        self.dynamics_ids[self.pos] = dynamics_ids

        # Circular buffer
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    # bookkeeping, length of stored transition
    def __len__(self):
        steps = self.capacity if self.full else self.pos
        return steps * self.num_envs

    def sample(self, batch_size):

        size = self.capacity if self.full else self.pos

        if size == 0:
            raise ValueError("Cannot sample an empty buffer")

        # Random time & environment indices
        batch_indices = np.random.randint(0, size, size=batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=batch_size)

        obs = self.obs[batch_indices, env_indices]
        next_obs = self.next_obs[batch_indices, env_indices]
        actions = self.actions[batch_indices, env_indices]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1) #reshapes into a column vector for each transition
        dones = self.dones[batch_indices, env_indices].reshape(-1, 1)
        timeouts = self.timeouts[batch_indices, env_indices].reshape(-1, 1)
        history= self.histories[batch_indices, env_indices]
        ids = self.dynamics_ids[batch_indices, env_indices]

        # Only true terminations stop bootstrapping
        dones = dones * (1.0 - timeouts)

        def to_tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return {
            "obs": to_tensor(obs),
            "next_obs": to_tensor(next_obs),
            "actions": to_tensor(actions),
            "rewards": to_tensor(rewards),
            "dones": to_tensor(dones),
            "history": to_tensor(history),
            "ids": torch.as_tensor(ids, dtype=torch.long, device=self.device)
        }



class DynamicsEncoder(nn.Module):

    def __init__(
            self, 
            obs_dim, 
            action_dim, 
            H, 
            latent_dim, 
            embed_dim=32, 
            kernel_sizes=(8, 5, 5), 
            strides=(4, 1, 1), 
            log_std_min=-20.0, 
            log_std_max=2.0
            ):
        super().__init__()
        input_dim = obs_dim + action_dim

        # Per-timestep (obs, action) embedding, shared across time; no ReLU on the output
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # TCN: Conv1d + ReLU blocks over the time axis
        tcn_layers = []
        for k, s in zip(kernel_sizes, strides):
            tcn_layers += [nn.Conv1d(embed_dim, embed_dim, kernel_size=k, stride=s), nn.ReLU()]
        self.tcn = nn.Sequential(*tcn_layers)

        # Flattened TCN output size (96 for H=50 with the default kernels/strides)
        with torch.no_grad():
            dummy = torch.zeros(1, H, input_dim)
            flat_size = self.tcn(self.embedding(dummy).permute(0, 2, 1)).reshape(1, -1).size(1)

        self.mean = nn.Linear(flat_size, latent_dim)
        self.log_std = nn.Linear(flat_size, latent_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, history):
        # history: (B, H, obs+act), newest first -> flip to oldest first
        x = torch.flip(history, dims=[1])
        x = self.embedding(x).permute(0, 2, 1)       # (B, embed_dim, H)
        x = self.tcn(x).reshape(x.size(0), -1)       # (B, flat_size)

        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return Normal(self.mean(x), log_std.exp())




class RunningMeanStd(nn.Module):
    """Running mean/var of the target delta (buffers, not trained)."""

    def __init__(self, shape, epsilon=1e-5, clip_limit=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))
        self.epsilon = epsilon
        self.clip_limit = clip_limit

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        M2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(M2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon), -self.clip_limit, self.clip_limit)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


class DynamicsDecoder(nn.Module):

    def __init__(self, obs_dim, action_dim, latent_dim, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim

        # LayerNorm on z before the MLP (Xu's LatentMLPFeatureExtractor backbone)
        self.latent_norm = nn.LayerNorm(latent_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        # Predicts the NORMALIZED change in observation
        self.mean = nn.Linear(hidden, obs_dim)
        self.log_std = nn.Linear(hidden, obs_dim)

        self.target_scaler = RunningMeanStd(shape=(obs_dim,))

    def forward(self, obs, action, z):
        x = self.net(torch.cat((obs, action, self.latent_norm(z)), dim=-1))
        return Normal(self.mean(x), self.log_std(x).clamp(-20, 2).exp())

    def logProb(self, obs, next_obs, action, z):
        """Log prob of the observed transition, with target normalization (as in Xu)."""
        delta_obs = next_obs - obs
        if self.training:
            self.target_scaler.update(delta_obs.detach())
        return self(obs, action, z).log_prob(self.target_scaler.normalize(delta_obs))

    @torch.no_grad()
    def predictNextObs(self, obs, action, z):
        """Denormalized next-obs prediction (for visualization)."""
        delta = self.target_scaler.denormalize(self(obs, action, z).mean)
        return obs + delta



def multi_positive_infonce(z, ids, temperature):
    z = F.normalize(z, dim=-1) #projecting onto unit sphere

    logits = z @ z.T / temperature  # Matrix of pairwise similarities

    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device) # Self similarities, diagonal of the logits matrix
    positive = (ids[:, None] == ids[None, :]) & ~self_mask
    counts = positive.sum(dim=1)
    valid = counts > 0

    log_den = torch.logsumexp(logits.masked_fill(self_mask, float('-inf')), dim=1)
    log_prob = logits - log_den[:, None]
    per_anchor = -(log_prob.masked_fill(~positive, 0).sum(dim=1) / counts.clamp_min(1))
    
    # Xu averages over ALL anchors; anchors without positives contribute 0
    return per_anchor.mean(), float(valid.float().mean().item())



class LDRAgent:

    def __init__(self, env, config=None, device=None):

        self.env = env
        self.config = config or LDRConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu" ))
        self.num_envs = env.num_envs
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)
        
        # # Safety checks for action space
        # assert isinstance(env.action_space, gym.spaces.Box)
        # assert np.all(np.isfinite(self.action_low))
        # assert np.all(np.isfinite(self.action_high))
        # assert np.all(self.action_high > self.action_low)

        # Training counters
        self.num_timesteps = 0
        self.num_updates = 0

        # Track running returns for each parallel environment.
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.completed_returns = []

        self.last_obs = None #current observations
        self.target_entropy = -float(self.action_dim) # SAC entropy target





        # Actor and Critic Networks 
        input_dim = self.obs_dim + self.config.latent_dim

        self.actor = Actor(
            obs_dim=input_dim, 
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim, 
            latent_dim=self.config.latent_dim
            ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate) 

        self.critic = Critic(
            obs_dim=input_dim, 
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim, 
            latent_dim=self.config.latent_dim
            ).to(self.device)

        self.critic_target = Critic(
            obs_dim=input_dim, 
            action_dim=self.action_dim, 
            hidden_dim=self.config.hidden_dim, 
            latent_dim=self.config.latent_dim
            ).to(self.device)

        # Initially copying online weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target network is not trained by gradient descent rather updated from online Critic using Polyak averaging
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)






        # Encoder & Decoder
        self.encoder = DynamicsEncoder(
            self.obs_dim, 
            self.action_dim, 
            self.config.n_past, 
            self.config.latent_dim
            ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)

        self.decoder = DynamicsDecoder(
            self.obs_dim, 
            self.action_dim, 
            self.config.latent_dim, 
            self.config.hidden_dim
            ).to(self.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config.learning_rate)


        self.history_queue = HistoryQueue(self.num_envs, self.config.n_past, self.obs_dim, self.action_dim)
        
        self.replay_buffer = ReplayBuffer(
            buffer_size = self.config.buffer_size, 
            num_envs = self.num_envs, 
            obs_dim = self.obs_dim, 
            action_dim = self.action_dim, 
            device = self.device, 
            n_past = self.config.n_past
            )

        # Learnable log entropy coefficient
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)


    # polyak averaging for target network update
    def update_target(self):
        tau = self.config.tau
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.mul_(1 - tau)
                target_param.add_(tau * param)

    # Convert normalized actions [-1, 1] to environment bounds
    def _toEnvAction(self, action):
        return (self.action_low + 0.5 * (self.action_high - self.action_low) * (action + 1.0))

    # Convert environment actions back to [-1, 1].
    def _reverseEnvAction(self, action):
        return (2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0)

    
    
    def predict(self, obs, history=None, deterministic=True, train_dropout=False):

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device )

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if history is None:
            history = self.history_queue.get()

        self.actor.eval()
        self.encoder.eval()

        with torch.no_grad():
            z = self._latent(history, train_dropout=train_dropout)
            x = torch.cat([obs, z], dim=-1)
            action = self.actor.predict(x, deterministic=deterministic)

        return self._toEnvAction(action.cpu().numpy())


    def _latent(self, history, train_dropout=False):

        history = torch.as_tensor(history, dtype=torch.float32, device=self.device )

        if history.ndim == 2:
            history = history.unsqueeze(0)

        with torch.no_grad():
            z = self.encoder(history).loc

            if train_dropout and self.config.latent_dropout_prob > 0:
                keep = (
                    torch.rand((z.shape[0], 1), device=self.device)
                    >= self.config.latent_dropout_prob
                ).float()
                z = z * keep

        return z


    def _selectAction(self, obs, deterministic=False):

        if self.num_timesteps < self.config.learning_starts:

            # Random normalized actions during warm-up
            action = np.random.uniform(-1.0, 1.0, size=(self.num_envs, self.action_dim) ).astype(np.float32)

        else:
            # Predict environment-scaled actions
            env_action = self.predict(obs, deterministic=deterministic, train_dropout=not deterministic)

            # Convert back to normalized [-1, 1]
            action = self._reverseEnvAction(env_action)

        return np.clip(action, -1.0, 1.0).astype(np.float32)



    # Collect transitions using vectorized environments
    def explore(self):

        if self.last_obs is None:
            self.last_obs = self.env.reset()

        collected_steps = 0

        while collected_steps < self.config.train_freq:

            history=self.history_queue.get()  # history BEFORE current action
            ids= self.env.dynamics_ids.copy()  # capture pre-step dynamics IDs (auto-reset may change them)

            buffer_actions = self._selectAction(self.last_obs, deterministic=False)
            env_actions = self._toEnvAction(buffer_actions) #scaling to environment action space before applying

            new_obs, rewards, dones, infos = self.env.step(env_actions)

            self.episode_returns += np.asarray(rewards, dtype=np.float64)
            
            for i in np.flatnonzero(dones):
                self.completed_returns.append(float(self.episode_returns[i]))
                self.episode_returns[i] = 0.0

            # 4. Identify time-limit truncations
            timeouts = np.array(
                [
                    info.get("TimeLimit.truncated", False)
                    for info in infos
                ],
                dtype=np.float32
            )

            # 5. Preserve actual terminal observations
            buffer_next_obs = new_obs.copy()

            for i in range(self.num_envs):
                if dones[i]:
                    buffer_next_obs[i] = infos[i]["terminal_observation"]

            
            self.replay_buffer.push(
                obs=self.last_obs,
                next_obs=buffer_next_obs,
                actions=buffer_actions,
                rewards=rewards,
                dones=dones,
                timeouts=timeouts,
                history = history,
                dynamics_ids=ids
            )

            self.history_queue.append(self.last_obs, buffer_actions)
            self.history_queue.reset_done(dones, new_obs)
            self.last_obs = new_obs.copy()
            self.num_timesteps += self.num_envs
            collected_steps += self.num_envs

        return collected_steps

    
    def optimize(self):

        self.actor.train()
        self.critic.train()
        self.encoder.train()
        self.decoder.train()

        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "vae_loss": [],
            "contrastive_loss": [],
            "positive_fraction": []
        }

        for _ in range(self.config.gradient_steps):

            batch = self.replay_buffer.sample(self.config.batch_size)

            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_obs"]
            dones = batch["dones"]

            # LDR-specific
            history = batch["history"]
            dynamics_ids = batch["ids"]

            
            qz = self.encoder(history)
            z = qz.rsample()

            # Latent dropout
            keep = (torch.rand((z.shape[0], 1), device=self.device)  >=   self.config.latent_dropout_prob).float()
            z_rl = z * keep # sampled latent including dropout for RL training

            # Condition observations on latent dynamics
            obs_z = torch.cat([obs, z_rl], dim=-1)
            next_obs_z = torch.cat([next_obs, z_rl.detach()], dim=-1)

            # entropy coefficient
            alpha = self.log_alpha.exp().detach()

            # training critic with target Q-values
            with torch.no_grad():

                next_actions, next_log_prob = (self.actor.actionLogProb(next_obs_z) )
                next_q1, next_q2 = self.critic_target(next_obs_z, next_actions)
                next_q = (torch.min(next_q1, next_q2) - alpha * next_log_prob)

                target_q = (rewards * self.config.reward_scale + self.config.gamma * (1.0 - dones) * next_q )

            
            q1, q2 = self.critic(obs_z.detach(), actions)
            critic_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # training actor with policy gradient
            new_actions, log_prob = (self.actor.actionLogProb(obs_z)  )

            # Freeze critic temporarily
            # for param in self.critic.parameters():
            #     param.requires_grad_(False)

            # (no critic freeze, as in Xu: gradients the actor loss leaves on the critic
            #  are cleared by critic_optimizer.zero_grad() before the next critic update)
            q1_pi, q2_pi = self.critic(obs_z, new_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * log_prob - min_q_pi).mean()



            # VAE reconstruction loss
            recon_loss = -self.decoder.logProb(obs, next_obs, actions, z).mean()

            # KL divergence 
            kl_loss = 0.5 * (qz.loc.square() + qz.scale.square() - 1.0 - 2.0 * qz.scale.log()).sum(dim=-1).mean()
            
            progress = min(self.num_timesteps / self.config.total_timesteps, 1.0)
            beta = self.config.beta_max * min(progress / self.config.beta_anneal_fraction, 1.0)
            vae_loss = recon_loss + beta * kl_loss


            # Contrastive loss (InfoNCE) for latent dynamics
            contrastive_loss = z.sum() * 0.0
            positive_fraction = 0.0

            if (self.num_timesteps >= self.config.contrastive_start_iterations):
                contrastive_loss, positive_fraction = (
                    multi_positive_infonce(z, dynamics_ids, self.config.infonce_temp)
                )

            # Combined total loss for actor, encoder, and decoder
            total_loss = (actor_loss + self.config.vae_loss_weight * vae_loss + self.config.contrastive_loss_weight * contrastive_loss)

            self.actor_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            total_loss.backward()

            self.actor_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Restore critic gradients
            # for param in self.critic.parameters():
            #     param.requires_grad_(True)

            # Update entropy coefficient
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # update target networks using Polyak averaging
            self.num_updates += 1

            if (self.num_updates % self.config.target_update_interval == 0):
                self.update_target()

            # Saving metrics for logging
            metrics["actor_loss"].append(actor_loss.item())
            metrics["critic_loss"].append(critic_loss.item())
            metrics["alpha_loss"].append(alpha_loss.item())
            metrics["alpha"].append(alpha.item())
            metrics["vae_loss"].append(vae_loss.item())
            metrics["contrastive_loss"].append(contrastive_loss.item())

            if torch.is_tensor(positive_fraction):
                positive_fraction = positive_fraction.item()

            metrics["positive_fraction"].append(float(positive_fraction) )

        return {key: float(np.mean(values)) for key, values in metrics.items()}


    # Evaluates Deterministic LDR policy
    def evaluate(self, eval_env, eval_num=1, seed=None):


        self.actor.eval()
        self.encoder.eval()

        num_envs = eval_env.num_envs
        all_returns = []

        for episode in range(eval_num):

            if seed is not None:
                eval_env.seed(seed + episode * num_envs)

            obs = eval_env.reset()

            
            history = HistoryQueue(
                num_envs,
                self.config.n_past,
                self.obs_dim,
                self.action_dim
            )
            history.reset_done(np.ones(num_envs, dtype=bool), obs)

            episode_returns = np.zeros(num_envs, dtype=np.float64)

            # Track which environments have finished
            finished = np.zeros(num_envs, dtype=bool)

            while not np.all(finished):

                # LDR: predict using current obs and past history
                actions = self.predict(
                    obs,
                    history=history.get(),
                    deterministic=True
                )

                next_obs, rewards, dones, infos = eval_env.step(actions)

                episode_returns[~finished] += rewards[~finished]

                # LDR specific: store normalized actions in history
                normalized_actions = self._reverseEnvAction(actions)

                history.append(obs, normalized_actions)
                history.reset_done(dones, next_obs) 

                obs = next_obs
                finished |= dones

            all_returns.extend(episode_returns.tolist())

        return float(np.mean(all_returns))



    def train(self, eval_env=None, eval_num=1, RUN_DIR=Path("hopper_ldr_results")):

        RUN_DIR = Path(RUN_DIR)
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        self.env.seed(self.config.seed)
        self.last_obs = self.env.reset()

        # LDR-specific: Reset history queue
        # self.history_queue.reset_done(np.ones(self.num_envs, dtype=bool))
        self.history_queue.reset_done(np.ones(self.num_envs, dtype=bool), self.last_obs)

        # Reset counters and reward tracking
        self.num_timesteps = 0
        self.num_updates = 0

        self.episode_returns.fill(0.0)
        self.completed_returns.clear()

        # Evaluation tracking
        best_eval_reward = float("-inf")

        next_eval_step = self.config.eval_interval

        training_history = []

        while self.num_timesteps < self.config.total_timesteps:

            collected_steps = self.explore()

            metrics = None

            if (self.num_timesteps >= self.config.learning_starts) and (len(self.replay_buffer) >= self.config.batch_size):
                metrics = self.optimize()

            if metrics is not None:
                metrics["timesteps"] = self.num_timesteps
                training_history.append(metrics)

            if (eval_env is not None) and (self.num_timesteps >= next_eval_step):

                eval_reward = self.evaluate(eval_env, eval_num=eval_num, seed=0)
                print(f"Steps: {self.num_timesteps} | Eval reward: {eval_reward:.2f}")

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save(RUN_DIR / "hopper_ldr_best.pt")
                    print(f"New best reward: {best_eval_reward:.2f}")

                # Advance evaluation threshold
                while next_eval_step <= self.num_timesteps:
                    next_eval_step += self.config.eval_interval

        print("Training completed!")

        return training_history


    
    
    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),

            # LDR networks
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),

            # SAC optimizers
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),

            # LDR optimizers
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),

            # Entropy coefficient
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),

            # Training progress
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,

            # Configuration
            "config": asdict(self.config)

        }, path)


    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        # SAC networks
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(
            checkpoint["critic_target"]
        )

        # LDR networks
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

        # SAC optimizers
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        # LDR optimizers
        self.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer"]
        )
        self.decoder_optimizer.load_state_dict(
            checkpoint["decoder_optimizer"]
        )

        # Entropy coefficient
        with torch.no_grad():
            self.log_alpha.copy_(
                checkpoint["log_alpha"].to(self.device)
            )

        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer"]
        )

        # Training progress
        self.num_timesteps = checkpoint["num_timesteps"]
        self.num_updates = checkpoint["num_updates"]

        # Target critic is not trained directly
        self.critic_target.eval()





class makeVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.dynamics_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def seed(self, seed):
        self._seed = seed

    def reset(self):
        observations = []
        for i, env in enumerate(self.envs):
            seed = getattr(self, "_seed", None)
            obs, _ = env.reset(
                seed=None if seed is None else seed + i
            )
            observations.append(obs)
        self._seed = None
        return np.stack(observations).astype(np.float32)

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            info = dict(info)
            info["TimeLimit.truncated"] = truncated and not terminated

            if done:
                info["terminal_observation"] = obs.copy()
                obs, _ = env.reset()

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos
        )

    def close(self):
        for env in self.envs:
            env.close()




class PhysicsWrapper(gym.Wrapper):
    """Nominal/custom physics, changing physics, or a fixed random anchor."""

    def __init__(
        self,
        env,
        mode="custom",
        mass_scale=1.0,
        damping_scale=1.0,
        torque_scale=1.0,
        multi_episode_dynamics=100,
        required_experience_length=None,   # Xu: change after this many env steps
        mass_range=(0.5, 2.0),
        damping_range=(0.5, 2.0),
        torque_range=(0.5, 1.5),           # Xu: 0.5-1.5 for Hopper
    ):
        super().__init__(env)

        self.mode = mode
        self.model = self.unwrapped.model
        self.base_mass = self.model.body_mass.copy()
        self.base_damping = self.model.dof_damping.copy()
        self.mass_scale = float(mass_scale)
        self.damping_scale = float(damping_scale)
        self.fixed_torque_scale = float(torque_scale)
        self.torque_scale = 1.0
        self.multi_episode_dynamics = int(multi_episode_dynamics)
        self.required_experience_length = required_experience_length
        self.steps_since_change = 0
        self.mass_range = tuple(mass_range)
        self.damping_range = tuple(damping_range)
        self.torque_range = tuple(torque_range)
        self.episodes = 0
        self.last_dynamics_changed = False
        self._rng = np.random.default_rng()

        if self.mode == "custom":
            self._set_physics(
                self.base_mass * self.mass_scale,
                self.base_damping * self.damping_scale,
                self.fixed_torque_scale,
            )

    def _set_physics(self, mass, damping, torque_scale):
        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping
        self.torque_scale = float(torque_scale)
        

    def _sample_physics(self):
        self._set_physics(
            self.base_mass * self._rng.uniform(*self.mass_range, size=self.base_mass.shape),
            self.base_damping * self._rng.uniform(*self.damping_range, size=self.base_damping.shape),
            self._rng.uniform(*self.torque_range),
        )

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.mode == "change" and self.required_experience_length is not None:
            # Xu: start nominal, switch once enough steps have been collected
            changed = self.steps_since_change >= self.required_experience_length
        else:
            changed = (
                (self.mode == "anchor" and self.episodes == 0)
                or (self.mode == "change" and self.episodes % self.multi_episode_dynamics == 0)
            )

        if changed:
            self._sample_physics()
            self.steps_since_change = 0

        obs, info = self.env.reset(**kwargs)
        self.episodes += 1
        self.last_dynamics_changed = changed
        return obs, {**info, "dynamics_changed": changed}

    def step(self, action):
        self.steps_since_change += 1
        return self.env.step(np.asarray(action) * self.torque_scale)

    def getRandomizedParameters(self):
        return np.concatenate((
            self.model.body_mass[1:].copy(),
            self.model.dof_damping[3:].copy(),
            np.array([self.torque_scale]),
        ))

    def getNominalParameters(self):
        return np.concatenate((
            self.base_mass[1:].copy(),
            self.base_damping[3:].copy(),
            np.array([1.0]),
        ))




class GlobalDynamicsIDWrapper:
    """Persistent dynamics IDs: identical parameter vectors share an ID across all modes."""

    MANAGER_DYNAMICS_ID_STR = "dynamics_ids"

    def __init__(self, venv):
        self.venv = venv
        self.envs = venv.envs
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.physics_envs = [self._find_physics_wrapper(env) for env in self.envs]

        # the initial dynamics ID for each environment is set to 0
        self.dynamics_ids = np.zeros(self.num_envs, dtype=np.int64)

        nominal_params = tuple(self.physics_envs[0].getNominalParameters().tolist())
        for physics in self.physics_envs[1:]:
            if tuple(physics.getNominalParameters().tolist()) != nominal_params:
                raise ValueError("All environments must share the same nominal physics")

        self.records = pd.DataFrame({
            "dynamics_id": pd.Series([0], dtype="int64"),
            "parameters": [nominal_params],
        })
        self._initialized = False

    @staticmethod
    def _find_physics_wrapper(env):
        while isinstance(env, gym.Wrapper):
            if isinstance(env, PhysicsWrapper):
                return env
            env = env.env
        raise ValueError("Every environment must contain PhysicsWrapper")

    def seed(self, seed):
        return self.venv.seed(seed)

    def _get_dynamics_id(self, physics):
        params = tuple(physics.getRandomizedParameters().tolist())
        matches = self.records.loc[
            self.records["parameters"].map(lambda p: p == params),
            "dynamics_id",
        ]
        if not matches.empty:
            return int(matches.iloc[0])

        dynamics_id = int(self.records["dynamics_id"].max()) + 1
        self.records.loc[len(self.records)] = [dynamics_id, params]
        return dynamics_id

    def reset(self):
        obs = self.venv.reset()
        for i, physics in enumerate(self.physics_envs):
            if not self._initialized or physics.last_dynamics_changed:
                self.dynamics_ids[i] = self._get_dynamics_id(physics)
        self._initialized = True
        return obs

    def step(self, actions):
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        old_ids = self.dynamics_ids.copy()
        obs, rewards, dones, infos = self.venv.step(actions)
        # makeVecEnv automatically resets done environments.
        for i in np.flatnonzero(dones):
            if self.physics_envs[i].last_dynamics_changed:
                self.dynamics_ids[i] = self._get_dynamics_id(self.physics_envs[i])
        for i, info in enumerate(infos):
            info[self.MANAGER_DYNAMICS_ID_STR] = int(old_ids[i])
            info["next_dynamics_id"] = int(self.dynamics_ids[i])
        return obs, rewards, dones, infos

    def save_dynamics_log(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.records.copy()
        df["parameters"] = df["parameters"].apply(json.dumps)
        df.to_csv(path, index=False)

    def close(self):
        return self.venv.close()

    def __getattr__(self, name):
        return getattr(self.venv, name)


def make_anchored_env_fns(
    num_envs,
    anchor_fraction,
    env_fn,
    seed=0,
    **physics_kwargs,
):

    count = int(np.floor(num_envs * anchor_fraction + 0.5))
    rng = np.random.default_rng(seed)
    anchor_indices = set(map(int, rng.choice(num_envs, size=count, replace=False)))

    env_fns = []
    for i in range(num_envs):
        mode = "anchor" if i in anchor_indices else "change"
        def factory(mode=mode):
            return PhysicsWrapper(env_fn(), mode=mode, **physics_kwargs)
        env_fns.append(factory)
    return env_fns, sorted(anchor_indices)


NUM_ENVS = 16

def make_hopper(required_experience_length, mode = 'change'):
    def _make():
        return PhysicsWrapper(
            gym.make("Hopper-v5"),
            mode=mode,
            required_experience_length=required_experience_length,
        )
    return _make

# Xu: new dynamics every 2000 steps per env
train_env = GlobalDynamicsIDWrapper(makeVecEnv([make_hopper(required_experience_length=2000) for _ in range(NUM_ENVS)]))

# Xu: 16 eval envs, new random dynamics every episode
eval_env = GlobalDynamicsIDWrapper(makeVecEnv([make_hopper(required_experience_length=1) for _ in range(NUM_ENVS)]))

config = LDRConfig()   # use the Xu defaults; don't override them here
agent = LDRAgent(env=train_env, config=config)



### Smoke testing

# import math

# TEST_DIR = Path("smoke_test")
# TEST_DIR.mkdir(exist_ok=True, parents=True)

# NUM_ENVS_TEST = 4

# def make_hopper(required_experience_length):
#     def _make():
#         return PhysicsWrapper(
#             gym.make("Hopper-v5"),
#             mode="change",
#             required_experience_length=required_experience_length,
#         )
#     return _make

# # Physics changes every 50 steps so several dynamics IDs appear quickly
# train_env = GlobalDynamicsIDWrapper(makeVecEnv([make_hopper(50) for _ in range(NUM_ENVS_TEST)]))
# eval_env  = GlobalDynamicsIDWrapper(makeVecEnv([make_hopper(1)  for _ in range(NUM_ENVS_TEST)]))

# smoke_config = LDRConfig(
#     total_timesteps=640,          # 10 collection cycles
#     learning_starts=128,          # first 2 cycles collect only, then training starts
#     train_freq=64,                # 16 vector steps x 4 envs
#     gradient_steps=64,            # UTD = 1, like Xu
#     batch_size=32,
#     eval_interval=320,            # 2 evaluations
#     buffer_size=5_000,
#     contrastive_start_iterations=0,
#     seed=0,
# )

# agent = LDRAgent(env=train_env, config=smoke_config)
# history = agent.train(eval_env=eval_env, eval_num=1, RUN_DIR=TEST_DIR)

# history_df = pd.DataFrame(history)
# history_df.to_csv(TEST_DIR / "history.csv", index=False)

# # ---- checks ----
# assert agent.num_timesteps == 640, agent.num_timesteps
# assert len(history) == 9, len(history)          # cycles 3..10 ran optimize()
# for key in ["actor_loss", "critic_loss", "alpha_loss", "vae_loss", "contrastive_loss"]:
#     assert all(math.isfinite(m[key]) for m in history), f"non-finite {key}"
# assert all(p.requires_grad for p in agent.critic.parameters()), "critic left frozen"
# assert len(train_env.records) > 1, "dynamics never changed"

# print("last metrics:", {k: round(v, 4) for k, v in history[-1].items()})
# print("dynamics IDs seen:", len(train_env.records))

# # Save / load round trip
# agent.save(TEST_DIR / "smoke.pt")
# agent2 = LDRAgent(env=train_env, config=smoke_config)
# agent2.load(TEST_DIR / "smoke.pt")
# for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
#     assert torch.equal(p1, p2)
# assert torch.equal(agent.decoder.target_scaler.mean, agent2.decoder.target_scaler.mean)

# train_env.close()
# eval_env.close()
# print("Smoke test passed")


NUM_ENVS = 16

def make_hopper(required_experience_length, mode = 'change'):
    def _make():
        return PhysicsWrapper(
            gym.make("Hopper-v5"),
            mode=mode,
            required_experience_length=required_experience_length,
        )
    return _make

# Xu: new dynamics every 2000 steps per env
train_env = GlobalDynamicsIDWrapper(makeVecEnv([make_hopper(required_experience_length=2000) for _ in range(NUM_ENVS)]))

# Xu: 16 eval envs, new random dynamics every episode
eval_env = GlobalDynamicsIDWrapper(makeVecEnv([make_hopper(required_experience_length=1) for _ in range(NUM_ENVS)]))

config = LDRConfig()   
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

agent = LDRAgent(env=train_env, config=config)


RUN_DIR = Path("hopper_ldr_results")
RUN_DIR.mkdir(exist_ok=True, parents=True)

try:
    # xu uses eval_num = 1, ours is averaging over 3
    history = agent.train(eval_env=eval_env, eval_num=3, RUN_DIR=RUN_DIR) 
    agent.save(str(RUN_DIR / "hopper_ldr_final.pt"))
    print("Final checkpoint:", RUN_DIR / "hopper_ldr_final.pt")
finally:
    train_env.close()
    eval_env.close()

print("Finished training episodes:", len(agent.completed_returns))
print("Optimization records:", len(history))














PARAM_NAMES = ["mass_torso", "mass_thigh", "mass_leg", "mass_foot",
               "damping_thigh", "damping_leg", "damping_foot", "torque_scale"]


class DynamicsRollout:
    """Rolls out the trained LDR policy under a given physics parameter set."""

    def __init__(self, agent, env_id="Hopper-v5"):
        self.agent = agent
        self.env = gym.make(env_id)
        self.model = self.env.unwrapped.model
        self.default_mass = self.model.body_mass.copy()
        self.default_damping = self.model.dof_damping.copy()
        self.torque_scale = 1.0

    def set_params(self, params):
        """Apply a recorded parameter tuple; world mass and root damping stay at defaults."""
        p = np.asarray(params, dtype=np.float64)
        self.model.body_mass[:] = self.default_mass
        self.model.dof_damping[:] = self.default_damping
        self.model.body_mass[1:] = p[0:4]
        self.model.dof_damping[3:] = p[4:7]
        self.torque_scale = float(p[7])

    @torch.no_grad()
    def rollout(self, seed):
        """One deterministic episode; returns (latents (T, latent_dim), return, length)."""
        a = self.agent
        a.encoder.eval()
        a.actor.eval()

        history = HistoryQueue(1, a.config.n_past, a.obs_dim, a.action_dim)
        obs, _ = self.env.reset(seed=seed)
        history.reset_done(np.ones(1, dtype=bool), obs[None])      # same padding as training

        latents, ep_return, done = [], 0.0, False
        while not done:
            h = torch.as_tensor(history.get(), device=a.device)
            o = torch.as_tensor(obs, dtype=torch.float32, device=a.device).reshape(1, -1)
            z = a.encoder(h).loc
            action = a.actor.predict(torch.cat([o, z], dim=-1), deterministic=True)
            z, action = z.cpu().numpy()[0], action.cpu().numpy()[0]
            latents.append(z)

            env_action = a._toEnvAction(action) * self.torque_scale
            next_obs, reward, terminated, truncated, _ = self.env.step(env_action)
            history.append(obs[None], action[None])                # normalized action

            ep_return += reward
            obs = next_obs
            done = terminated or truncated

        return np.array(latents), ep_return, len(latents)

    def close(self):
        self.env.close()


def latents_from_rollouts(agent, wrapper, episodes_per_id=1, skip_first=None, base_seed=0):
    """
    One row per dynamics ID: parameters, mean/std of z, mean return and episode length.
    skip_first: drop the first steps of each episode, where the window is still
                padded with the initial obs (default n_past). Falls back to all
                steps if the episode is shorter than that.
    """
    skip_first = agent.config.n_past if skip_first is None else skip_first
    runner = DynamicsRollout(agent)
    rows = []
    try:
        for _, rec in wrapper.records.iterrows():
            dyn_id, params = int(rec["dynamics_id"]), rec["parameters"]
            runner.set_params(params)

            all_z, returns, lengths = [], [], []
            for ep in range(episodes_per_id):
                z, ret, length = runner.rollout(seed=base_seed + dyn_id * episodes_per_id + ep)
                all_z.append(z[skip_first:] if len(z) > skip_first else z)
                returns.append(ret)
                lengths.append(length)
            all_z = np.concatenate(all_z)

            row = {"dynamics_id": dyn_id, **dict(zip(PARAM_NAMES, params))}
            for k in range(all_z.shape[1]):
                row[f"z{k}"] = all_z[:, k].mean()
                row[f"z{k}_std"] = all_z[:, k].std()
            row.update(n_steps=len(all_z),
                       mean_return=float(np.mean(returns)),
                       mean_length=float(np.mean(lengths)))
            rows.append(row)
    finally:
        runner.close()
    return pd.DataFrame(rows)


result = latents_from_rollouts(agent, train_env, episodes_per_id=1)
result.to_csv(RUN_DIR / "dynamics_latents.csv", index=False)
print(f"Saved {len(result)} dynamics to {RUN_DIR / 'dynamics_latents.csv'}s")