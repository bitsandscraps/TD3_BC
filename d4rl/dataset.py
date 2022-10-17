from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import (DataLoader, default_collate,
                              RandomSampler, Sampler, TensorDataset)


def normalize_data(data_dict: Dict[str, NDArray[Any]],
                   epsilon: float = 1e-3,
                   ) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    observations = data_dict['observations']
    mean = observations.mean(0, keepdims=True)
    std = observations.std(0, keepdims=True) + epsilon
    data_dict['observations'] = (observations - mean) / std
    return mean, std


class InfiniteSampler(Sampler):
    def __init__(self,
                 max_value: int,
                 replacement: bool = False,
                 generator=None) -> None:
        self.max_value = max_value
        self.replacement = replacement
        self.generator = generator
        if not isinstance(self.replacement, bool):
            raise TypeError('replacement should be a boolean value, but got '
                            f'replacement={replacement}')

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            while True:
                yield from torch.randint(high=self.max_value,
                                         size=(32,),
                                         dtype=torch.int64,
                                         generator=generator).tolist()
        else:
            while True:
                yield from torch.randperm(self.max_value,
                                          generator=generator).tolist()


class QLearningDataset(TensorDataset):
    def __init__(self,
                 data_dict: Dict[str, np.ndarray],
                 float_dtype: torch.dtype = torch.float32) -> None:
        assert data_dict['timeouts'][-1] or data_dict['terminals'][-1]
        indices = np.flatnonzero(np.logical_not(data_dict['timeouts']))
        observations = torch.as_tensor(data_dict['observations'][indices],
                                       dtype=float_dtype)
        actions = torch.as_tensor(data_dict['actions'][indices],
                                  dtype=float_dtype)
        rewards = torch.as_tensor(data_dict['rewards'][indices],
                                  dtype=float_dtype)
        dones = torch.as_tensor(data_dict['terminals'][indices],
                                dtype=float_dtype)
        next_observations = torch.as_tensor(
            data_dict['observations'][indices + 1], dtype=float_dtype)
        super().__init__(
            observations, actions, rewards, dones, next_observations)


class StateActionDataset(TensorDataset):
    def __init__(self,
                 data_dict: Dict[str, np.ndarray],
                 float_dtype: torch.dtype = torch.float32) -> None:
        observations = torch.as_tensor(data_dict['observations'],
                                       dtype=float_dtype)
        actions = torch.as_tensor(data_dict['actions'],
                                  dtype=float_dtype)
        super().__init__(observations, actions)


class TrajectoryDataset(TensorDataset):
    def __init__(self,
                 data_dict: Dict[str, np.ndarray],
                 dtype: torch.dtype = torch.float32) -> None:
        observations = torch.as_tensor(data_dict['observations'], dtype=dtype)
        actions = torch.as_tensor(data_dict['actions'], dtype=dtype)
        super().__init__(observations, actions)
        dones = np.logical_or(data_dict['terminals'], data_dict['timeouts'])
        self.indices = np.insert(np.flatnonzero(dones) + 1, 0, 0)
        if not dones[-1]:
            self.indices = np.append(self.indices, dones.size)

    def get_loader(self,
                   trajectories_per_batch: int,
                   samples_per_trajectory: int,
                   sampler_kwargs: Optional[Dict[str, Any]] = None,
                   **kwargs):
        if sampler_kwargs is None:
            sampler_kwargs = {}

        def collate_fn(batch):
            # [trajectories_per_batch * samples_per_trajectory, *_dim]
            observations, actions = default_collate(batch)
            batch_size = observations.size(0) // samples_per_trajectory
            observations = observations.view(batch_size,
                                             samples_per_trajectory,
                                             -1)
            actions = actions.view(batch_size,
                                   samples_per_trajectory,
                                   -1)
            return observations, actions

        return DataLoader(
            dataset=self,
            batch_size=trajectories_per_batch * samples_per_trajectory,
            sampler=TrajectorySampler(
                indices=self.indices,
                samples_per_trajectory=samples_per_trajectory,
                **sampler_kwargs),
            collate_fn=collate_fn,
            **kwargs)


class TrajectorySampler(RandomSampler):
    def __init__(self,
                 indices: NDArray[np.integer],
                 samples_per_trajectory: int,
                 **kwargs) -> None:
        self.samples_per_trajectory = samples_per_trajectory
        super().__init__(indices[:-1], **kwargs)
        self.indices = indices[:-1]
        self.samplers = tuple(iter(InfiniteSampler(length))
                              for length in indices[1:] - indices[:-1])

    @property
    def num_samples(self) -> int:
        return super().num_samples * self.samples_per_trajectory

    def __iter__(self) -> Iterator[int]:
        for trajectory in super().__iter__():
            start_index = self.indices[trajectory]
            for _ in range(self.samples_per_trajectory):
                yield start_index + next(self.samplers[trajectory])
