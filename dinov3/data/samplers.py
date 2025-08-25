# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import itertools
import warnings
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from dinov3.distributed import get_rank, get_world_size


class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _get_numpy_dtype(size: int) -> Any:
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}", stacklevel=1)

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}", stacklevel=1)
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


class BalancedSampler(Sampler):
    """
    Balanced sampler for handling class imbalance using weighted random sampling.
    
    This sampler calculates inverse class frequencies as weights and uses 
    WeightedRandomSampler to oversample minority classes.
    """
    
    def __init__(
        self,
        dataset,
        *,
        size: int = -1,
        replacement: bool = True,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset object that implements get_targets() method
            size: Number of samples per epoch (-1 for full dataset)
            replacement: Whether to sample with replacement
            seed: Random seed
            start: Starting rank (for distributed training)
            step: Step size (for distributed training)
            advance: Number of samples to skip initially
        """
        self._dataset = dataset
        self._size = len(dataset) if size == -1 else size
        self._replacement = replacement
        self._seed = seed
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._advance = advance
        self._epoch = 0
        
        # Calculate sample weights based on class frequencies
        self._sample_weights = self._calculate_sample_weights()
        
    def _calculate_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights based on inverse class frequencies."""
        # Get all targets from the dataset
        if hasattr(self._dataset, 'get_targets'):
            targets = self._dataset.get_targets()
        else:
            # Fallback: iterate through dataset to get targets
            targets = []
            for i in range(len(self._dataset)):
                if hasattr(self._dataset, 'get_target'):
                    targets.append(self._dataset.get_target(i))
                else:
                    _, target = self._dataset[i]
                    targets.append(target)
            targets = np.array(targets)
        
        # Calculate class counts
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        
        # Calculate class weights (inverse frequency)
        class_weights = 1.0 / class_counts
        
        # Create sample weights for each data point
        sample_weights = np.zeros(len(targets))
        for class_idx, class_label in enumerate(unique_classes):
            class_mask = targets == class_label
            sample_weights[class_mask] = class_weights[class_idx]
        
        return torch.from_numpy(sample_weights).float()
    
    def __iter__(self):
        """Generate balanced samples."""
        # Create WeightedRandomSampler
        from torch.utils.data import WeightedRandomSampler
        
        # Set random seed for reproducibility
        generator = torch.Generator()
        seed = self._seed * self._epoch if self._seed != 0 else self._epoch
        generator.manual_seed(seed)
        
        weighted_sampler = WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=self._size,
            replacement=self._replacement,
            generator=generator
        )
        
        # Get all indices from weighted sampler
        all_indices = list(weighted_sampler)
        
        # Apply distributed sampling if needed
        if self._step > 1:
            distributed_indices = all_indices[self._start::self._step]
        else:
            distributed_indices = all_indices
        
        # Apply advance offset
        if self._advance > 0:
            distributed_indices = distributed_indices[self._advance:]
        
        yield from distributed_indices
    
    def __len__(self) -> int:
        """Return the number of samples per epoch for this worker."""
        effective_size = self._size - self._advance
        if self._step > 1:
            return (effective_size - self._start + self._step - 1) // self._step
        return effective_size
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible sampling."""
        self._epoch = epoch
        
    def get_class_distribution_info(self) -> dict:
        """Get information about class distribution and weights."""
        if hasattr(self._dataset, 'get_targets'):
            targets = self._dataset.get_targets()
        else:
            targets = []
            for i in range(len(self._dataset)):
                if hasattr(self._dataset, 'get_target'):
                    targets.append(self._dataset.get_target(i))
                else:
                    _, target = self._dataset[i]
                    targets.append(target)
            targets = np.array(targets)
        
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        class_weights = 1.0 / class_counts
        
        return {
            'unique_classes': unique_classes.tolist(),
            'class_counts': class_counts.tolist(),
            'class_weights': class_weights.tolist(),
            'total_samples': len(targets),
            'num_classes': len(unique_classes)
        }
