import numpy as np
import torch
from torch import Tensor
from typing import NamedTuple


class ReplayBufferSamples(NamedTuple):
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    dones: Tensor
    rewards: Tensor


class ReplayBuffer:
    def __init__(self, capacity, env_dim, obs_dim, action_dim):
        self.capacity = max(capacity // env_dim, 1)
        self.env_dim = env_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.ptr = 0
        self.full = False
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Allocate memory for experiences
        self.observations = np.zeros((self.capacity, env_dim, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, env_dim, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, env_dim, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, env_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, env_dim), dtype=np.float32)
        self.timeouts = np.zeros((self.capacity, env_dim), dtype=np.float32)

    @property
    def size(self):
        return self.capacity if self.full else self.ptr

    def push(self, obs, next_obs, action, reward, done, timeout):
        """Store transitions from all parallel environments."""
        i = self.ptr

        self.observations[i] = obs
        self.next_observations[i] = next_obs
        self.actions[i] = action.reshape(
            self.env_dim, self.action_dim
        )
        self.rewards[i] = reward
        self.dones[i] = done
        self.timeouts[i] = timeout

        # Circular buffer: overwrite oldest data when full
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        """Sample random transitions for training."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        # Random time indices and environment indices
        batch_inds = np.random.randint(0, self.size, batch_size)
        env_inds = np.random.randint(0, self.env_dim, batch_size)

        obs = self.observations[batch_inds, env_inds]
        next_obs = self.next_observations[batch_inds, env_inds]
        actions = self.actions[batch_inds, env_inds]
        rewards = self.rewards[batch_inds, env_inds].reshape(-1, 1)

        # Ignore episode termination caused by time limits
        dones = (
            self.dones[batch_inds, env_inds]
            * (1 - self.timeouts[batch_inds, env_inds])
        ).reshape(-1, 1)

        return ReplayBufferSamples(
            observations=torch.tensor(obs, device=self.device),
            next_observations=torch.tensor(next_obs, device=self.device),
            actions=torch.tensor(actions, device=self.device),
            dones=torch.tensor(dones, device=self.device),
            rewards=torch.tensor(rewards, device=self.device),
        )

    # def save(self, file_path):
    #     """Save stored transitions to a compressed file."""
    #     arrays = {
    #         name: getattr(self, name)[:self.size]
    #         for name in (
    #             "observations", "next_observations", "actions",
    #             "rewards", "dones", "timeouts"
    #         )
    #     }

    #     config = dict(
    #         capacity=self.size * self.env_dim,
    #         env_dim=self.env_dim,
    #         obs_dim=self.obs_dim,
    #         action_dim=self.action_dim,
    #     )

    #     np.savez_compressed(
    #         file_path,
    #         ptr=self.size,
    #         full=True,
    #         meta=np.array([config], dtype=object),
    #         **arrays,
    #     )

    # @classmethod
    # def load(cls, file_path):
    #     """Restore a replay buffer from a file."""
    #     with np.load(file_path, allow_pickle=True) as data:
    #         config = data["meta"].item()
    #         buffer = cls(**config)

    #         buffer.full = bool(data["full"].item())
    #         buffer.ptr = int(data["ptr"].item())

    #         for name in (
    #             "observations", "next_observations", "actions",
    #             "rewards", "dones", "timeouts"
    #         ):
    #             buffer.__dict__[name][:] = data[name]

    #     return buffer
