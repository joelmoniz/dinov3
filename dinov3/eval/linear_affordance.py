#!/usr/bin/env python3
"""
TL;DR: Run this as--
python -m dinov3.eval.linear_affordance --model.dino_hub=dinov3_vits16 --model.pretrained_weights=/Users/joelmoniz/Downloads/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:preprocd_root_dir=/Users/joelmoniz/code/sit/tsv/" --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:preprocd_root_dir=/Users/joelmoniz/code/sit/tsv/" --eval.test_datasets="[AffordanceADE:split=TEST:affordance_type=SIT:preprocd_root_dir=/Users/joelmoniz/code/sit/tsv/]" --output_dir=/Users/joelmoniz/code/sit/dino_out

Linear evaluation for AffordanceADE dataset using DINOv3 features.

This script performs linear classification on highlighted objects from the ADE20K dataset
with affordance annotations (sit/run/grasp capabilities). It uses frozen DINOv3 features
and trains linear classifiers on top.

EMBEDDING CACHING:
This script now includes automatic embedding caching functionality:
- On first run, DINOv3 embeddings are extracted from all datasets and cached to disk
- Subsequent runs automatically detect and load cached embeddings, skipping feature extraction
- Caching is based on dataset configuration, model type, and transform parameters
- Cache files are stored in {output_dir}/embedding_cache/
- Significantly speeds up training and evaluation after the initial embedding extraction
- To disable caching, use --cache_embeddings=false

DEVICE COMPATIBILITY:
This script automatically detects and works on both CPU and GPU:
- GPU: CUDA will be used if available (recommended for performance)
- CPU: Falls back to CPU computation if CUDA is not available
- Distributed training is supported on GPU setups

IMPORTANT SETUP REQUIREMENTS:
1. Ensure you have the AffordanceADE dataset properly set up with:
   - ADE20K base dataset (images and annotations)
   - Affordance annotations data
2. Have a pretrained DINOv3 model checkpoint available
3. Make sure your environment has all required dependencies installed

QUICK START:
Run the example script for guided setup:
    python dinov3/eval/examples/affordance_linear_example.py

Or run with examples:
    python dinov3/eval/examples/affordance_linear_example.py --examples

Usage Examples:

1. Basic training and evaluation on "sit" affordance:
   python -m dinov3.eval.linear_affordance \
     --model.type=vit_large \
     --model.path=/path/to/dinov3_vitl14_pretrain.pth \
     --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --output_dir=/path/to/output

2. Evaluation with different affordance types:
   # For "run" affordance
   python -m dinov3.eval.linear_affordance \
     --model.type=vit_large \
     --model.path=/path/to/dinov3_vitl14_pretrain.pth \
     --train.dataset="AffordanceADE:split=TRAIN:affordance_type=RUN:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.val_dataset="AffordanceADE:split=VAL:affordance_type=RUN:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --output_dir=/path/to/output

   # For "grasp" affordance  
   python -m dinov3.eval.linear_affordance \
     --model.type=vit_large \
     --model.path=/path/to/dinov3_vitl14_pretrain.pth \
     --train.dataset="AffordanceADE:split=TRAIN:affordance_type=GRASP:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.val_dataset="AffordanceADE:split=VAL:affordance_type=GRASP:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --output_dir=/path/to/output

3. Additional test datasets and custom parameters:
   python -m dinov3.eval.linear_affordance \
     --model.type=vit_large \
     --model.path=/path/to/dinov3_vitl14_pretrain.pth \
     --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --eval.test_datasets="AffordanceADE:split=TEST:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.batch_size=64 \
     --train.epochs=20 \
     --transform.crop_size=224 \
     --output_dir=/path/to/output

4. Few-shot evaluation:
   python -m dinov3.eval.linear_affordance \
     --model.type=vit_large \
     --model.path=/path/to/dinov3_vitl14_pretrain.pth \
     --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --few_shot.enable=true \
     --few_shot.k_or_percent=0.1 \
     --few_shot.n_tries=3 \
     --output_dir=/path/to/output

5. With Weights & Biases logging:
   python -m dinov3.eval.linear_affordance \
     --model.type=vit_large \
     --model.path=/path/to/dinov3_vitl14_pretrain.pth \
     --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
     --wandb.enabled=true \
     --wandb.entity=your_entity \
     --wandb.project=dinov3-affordance \
     --wandb.name=sit_vitl14_experiment \
     --output_dir=/path/to/output

Key Parameters:
- model.type: Model architecture (vit_small, vit_base, vit_large, vit_giant2)
- model.path: Path to pretrained DINOv3 checkpoint
- train.dataset: Training dataset specification with affordance type
- train.val_dataset: Validation dataset specification  
- eval.test_datasets: Additional test datasets (optional)
- train.batch_size: Training batch size per GPU (default: 128)
- train.epochs: Number of training epochs (default: 10) 
- transform.crop_size: Input image crop size (default: 224)
- wandb.enabled: Enable Weights & Biases logging (default: true)
- wandb.entity: W&B entity/team name (optional)
- wandb.project: W&B project name (default: dinov3-affordance-linear)
- wandb.name: Experiment name (auto-generated if not provided)
- output_dir: Directory to save results and checkpoints
- cache_embeddings: Enable DINOv3 embedding caching (default: true)

Notes:
- This script uses MEAN_PER_CLASS_ACCURACY (macro-averaged accuracy) as the primary metric
- The affordance dataset has 7 classes: [negative, exception1-5, positive]
- Macro-averaging gives equal weight to each class regardless of frequency
- Results are saved to results-linear-affordance.csv in the output directory
- Automatically detects and uses GPU if available, falls back to CPU otherwise
- For best performance, use GPU with CUDA support
- Embedding caching significantly speeds up repeated runs with the same datasets and model

=============================

# AffordanceADE Linear Evaluation

This module provides linear evaluation capabilities for the AffordanceADE dataset using DINOv3 features.

## Overview

The AffordanceADE dataset contains highlighted objects from ADE20K with affordance annotations for three types of capabilities:
- **SIT**: Whether an object can be sat on
- **RUN**: Whether an object can be run on/through
- **GRASP**: Whether an object can be grasped

Each affordance is classified into 7 classes:
- `0`: Negative (Firmly Negative)
- `1-5`: Exception classes (various obstacle/awkward/forbidden categories)  
- `6`: Positive

The evaluation uses **MEAN_PER_CLASS_ACCURACY** (macro-averaged accuracy) as the primary metric, which gives equal weight to each class regardless of frequency. This is important for affordance classification since the dataset is imbalanced.

## Usage

### Basic Usage

```bash
python -m dinov3.eval.linear_affordance \
  --model.type=vit_large \
  --model.path=/path/to/dinov3_vitl14_pretrain.pth \
  --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --output_dir=/path/to/output
```

### With Test Evaluation

```bash
python -m dinov3.eval.linear_affordance \
  --model.type=vit_large \
  --model.path=/path/to/dinov3_vitl14_pretrain.pth \
  --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --eval.test_datasets="AffordanceADE:split=TEST:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --output_dir=/path/to/output
```

### Few-Shot Evaluation

```bash
python -m dinov3.eval.linear_affordance \
  --model.type=vit_large \
  --model.path=/path/to/dinov3_vitl14_pretrain.pth \
  --train.dataset="AffordanceADE:split=TRAIN:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --train.val_dataset="AffordanceADE:split=VAL:affordance_type=SIT:ade20k_root_dir=/path/to/ade20k:ade_affordance_root_dir=/path/to/affordance" \
  --few_shot.enable=true \
  --few_shot.k_or_percent=0.1 \
  --few_shot.n_tries=3 \
  --output_dir=/path/to/output
```

## Parameters

### Required Parameters

- `--model.type`: DINOv3 model architecture
  - Options: `vit_small`, `vit_base`, `vit_large`, `vit_giant2`
- `--model.path`: Path to pretrained DINOv3 checkpoint
- `--train.dataset`: Training dataset specification
- `--train.val_dataset`: Validation dataset specification  
- `--output_dir`: Output directory for results and checkpoints

### Dataset Parameters

The dataset string format is: `AffordanceADE:split=SPLIT:affordance_type=TYPE:ade20k_root_dir=PATH:ade_affordance_root_dir=PATH`

- `split`: Dataset split (`TRAIN`, `VAL`, `TEST`)
- `affordance_type`: Affordance type to classify (`SIT`, `RUN`, `GRASP`)
- `ade20k_root_dir`: Path to ADE20K dataset root
- `ade_affordance_root_dir`: Path to affordance annotations root
- `preprocd_root_dir`: (Optional) Path to preprocessed TSV files
- `highlight_params`: (Optional) Object highlighting parameters

### Training Parameters

- `--train.batch_size`: Training batch size per GPU (default: 128)
- `--train.epochs`: Number of training epochs (default: 15)
- `--train.learning_rates`: Learning rate grid search values
- `--train.optimizer_type`: Optimizer type (`sgd`, `adamw`)
- `--train.scheduler_type`: Scheduler type (`cosine_annealing`, `one_cycle`)

### Evaluation Parameters

- `--eval.test_datasets`: Additional test dataset paths
- `--eval.batch_size`: Evaluation batch size per GPU (default: 256)

### Transform Parameters

- `--transform.crop_size`: Input image crop size (default: 224)
- `--transform.resize_size`: Input image resize size (default: 256)

### Few-Shot Parameters

- `--few_shot.enable`: Enable few-shot evaluation
- `--few_shot.k_or_percent`: Number of elements or percentage per class
- `--few_shot.n_tries`: Number of few-shot trials

### Weights & Biases Parameters

- `--wandb.enabled`: Enable W&B logging (default: true)
- `--wandb.entity`: W&B entity/team name (optional)
- `--wandb.project`: W&B project name (default: dinov3-affordance-linear)
- `--wandb.name`: Experiment name (auto-generated if not provided)
- `--wandb.tags`: Experiment tags as comma-separated list
- `--wandb.notes`: Experiment description/notes
- `--wandb.resume`: Resume policy ("allow", "must", "never", "auto")

The wandb integration logs:
- Training metrics: loss, learning rate, iteration, epoch
- Validation metrics: accuracy, best classifier performance
- Test metrics: final evaluation results on test datasets
- Configuration: all hyperparameters and model settings
- System info: device type, number of GPUs, distributed training status
- Dataset info: number of classes, dataset sizes
- Final summary: best accuracy, total training time, etc.

## Key Differences from Standard Linear Evaluation

1. **Metric**: Uses `MEAN_PER_CLASS_ACCURACY` instead of `MEAN_ACCURACY` to handle class imbalance
2. **Learning Rates**: Adjusted learning rate range for affordance classification
3. **Epochs**: Increased default epochs (15 vs 10) for better affordance learning
4. **Classes**: Configured for 7 affordance classes by default
5. **Loss**: Cross-entropy loss optimized for affordance classification

## Output Files

- `results-linear-affordance.csv`: Main results file with accuracy metrics
- `results_eval_linear_affordance.json`: Detailed evaluation metrics
- `checkpoints/`: Training checkpoints
- `checkpoints/best/`: Best model checkpoint based on validation accuracy

## Examples

See `examples/affordance_linear_example.py` for more detailed usage examples and helper functions.

## Notes

- The AffordanceADE dataset must be properly set up with both ADE20K base data and affordance annotations
- The evaluation automatically handles the highlighted object images created by the AffordanceADE dataset
- Macro-averaging ensures equal importance for all affordance classes regardless of their frequency
- The system supports distributed training across multiple GPUs
- Checkpointing allows resuming interrupted training runs
- Weights & Biases logging is enabled by default and provides comprehensive experiment tracking
- For best performance, use GPU with CUDA support



"""

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import hashlib

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
from omegaconf import MISSING
from torch.nn.parallel import DistributedDataParallel

# Device detection for CPU/GPU compatibility
def get_device():
    """Get the appropriate device (cuda if available, otherwise cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(tensor_or_module, device=None, non_blocking=False):
    """Move tensor or module to the specified device."""
    if device is None:
        device = get_device()
    if non_blocking and hasattr(tensor_or_module, 'to'):
        return tensor_or_module.to(device, non_blocking=non_blocking)
    else:
        return tensor_or_module.to(device)

def get_current_device():
    """Get current device (cuda device or cpu)."""
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")

def sync_device():
    """Synchronize device (cuda sync if available, otherwise no-op)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Embedding caching functionality
def get_cache_key(dataset_str: str, model_config: 'ModelConfig', transform_config: 'TransformConfig') -> str:
    """Generate a unique cache key based on dataset, model, and transform configurations."""
    # Create a hash of the key components
    key_components = {
        'dataset_str': dataset_str,
        'model_dino_hub': model_config.dino_hub,
        'model_pretrained_weights': model_config.pretrained_weights,
        'model_config_file': model_config.config_file,
        'transform_crop_size': transform_config.crop_size,
        'transform_resize_size': transform_config.resize_size,
    }
    
    # Convert to string and hash
    key_str = json.dumps(key_components, sort_keys=True)
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    return cache_key


def get_cache_path(output_dir: str, cache_key: str, split_name: str) -> Path:
    """Get the cache file path for embeddings."""
    cache_dir = Path(output_dir) / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"embeddings_{split_name}_{cache_key}.pkl"


def save_embeddings_to_cache(embeddings_dict: Dict, cache_path: Path):
    """Save embeddings dictionary to cache file."""
    logger.info(f"Saving embeddings to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    logger.info(f"Embeddings cached successfully")


def load_embeddings_from_cache(cache_path: Path) -> Optional[Dict]:
    """Load embeddings dictionary from cache file."""
    if not cache_path.exists():
        return None
    
    try:
        logger.info(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        logger.info(f"Embeddings loaded from cache successfully")
        return embeddings_dict
    except Exception as e:
        logger.warning(f"Failed to load embeddings from cache: {e}")
        return None


def extract_and_cache_embeddings(
    feature_model: nn.Module,
    data_loader,
    cache_path: Path,
    device
) -> Dict[int, torch.Tensor]:
    """Extract embeddings from the feature model and cache them."""
    logger.info("Extracting embeddings from dataset...")
    
    embeddings_dict = {}
    feature_model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, targets, indices) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(data_loader)}")
            
            data = to_device(data, device, non_blocking=True)
            
            # Extract features
            features = feature_model(data)
            
            # Store embeddings with their indices
            for i, idx in enumerate(indices):
                embeddings_dict[idx.item()] = features[i].cpu()
    
    # Save to cache
    save_embeddings_to_cache(embeddings_dict, cache_path)
    return embeddings_dict


class CachedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that uses cached embeddings instead of running feature extraction."""
    
    def __init__(self, original_dataset, embeddings_dict: Dict[int, torch.Tensor]):
        self.original_dataset = original_dataset
        self.embeddings_dict = embeddings_dict
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # Get the original item (for target/label)
        _, target = self.original_dataset[idx]
        
        # Get cached embedding
        if idx in self.embeddings_dict:
            embedding = self.embeddings_dict[idx]
        else:
            raise KeyError(f"Embedding for index {idx} not found in cache")
            
        return embedding, target

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    CheckpointRetentionPolicy,
    cleanup_checkpoint,
    find_latest_checkpoint,
    keep_last_n_checkpoints,
)
from dinov3.data import SamplerType, make_data_loader, make_dataset
from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
    make_classification_train_transform,
)
from dinov3.eval.data import create_train_dataset_dict, get_num_classes, pad_multilabel_and_collate
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.metrics import ClassificationMetricType, build_classification_metric
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.eval.utils import LossType, ModelWithIntermediateLayers, average_metrics, evaluate
from dinov3.eval.utils import save_results as default_save_results_func
from dinov3.logging import MetricLogger, SmoothedValue
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-linear-affordance.csv"
# Primary metric for affordance classification - per-class accuracy (macro-averaged)
MAIN_METRICS = [".*_per_class_accuracy(_mean)?"]


class OptimizerType(Enum):
    SGD = "sgd"
    ADAMW = "adamw"

    def get_optimizer(self, optim_param_groups):
        if self == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=0)
        else:
            optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        return optimizer


class SchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    ONE_CYCLE = "one_cycle"

    def get_scheduler(self, optimizer, optim_param_groups, epoch_length, epochs, max_iter):
        if self == SchedulerType.ONE_CYCLE:
            lr_list = [optim_param_groups[i]["lr"] for i in range(len(optim_param_groups))]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr_list, steps_per_epoch=epoch_length, epochs=epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        return scheduler


_DEFAULT_LR_LIST: Tuple[float, ...] = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1)


@dataclass
class TrainConfig:
    dataset: str = MISSING  # train dataset path
    val_dataset: str = MISSING  # val dataset path
    val_metric_type: ClassificationMetricType = ClassificationMetricType.MEAN_PER_CLASS_ACCURACY  # Changed to per-class accuracy
    batch_size: int = 128  # batch size (per GPU)
    num_workers: int = 8
    # Linear Head Parameters
    learning_rates: Tuple[float, ...] = _DEFAULT_LR_LIST  # learning rates to grid search
    n_last_blocks_list: Tuple[int] = (1,)  # number of backbone last blocks used for the linear classifier
    loss_type: LossType = LossType.CROSS_ENTROPY
    optimizer_type: OptimizerType = OptimizerType.SGD
    scheduler_type: SchedulerType = SchedulerType.COSINE_ANNEALING
    epochs: int = 15  # Increased epochs for affordance learning
    epoch_length: int = 1250  # length of an epoch in number of iterations
    save_checkpoint_iterations: int | None = (
        None  # number of iterations between two checkpoint saves (default: one epoch)
    )
    eval_period_iterations: int | None = None  # number of iterations between two evaluations (default: one epoch)
    checkpoint_retention_policy: CheckpointRetentionPolicy = CheckpointRetentionPolicy.NONE  # keep checkpoints or not
    resume: bool = True  # whether to resume from existing checkpoints
    classifier_fpath: Optional[str] = None  # path to a file containing pretrained linear classifiers


@dataclass
class EvalConfig:
    test_datasets: Tuple[str, ...] = ()  # additional test dataset paths
    test_metric_types: Tuple[ClassificationMetricType, ...] = (ClassificationMetricType.MEAN_PER_CLASS_ACCURACY,)
    batch_size: int = 256  # batch size (per GPU)
    num_workers: int = 8


@dataclass
class TransformConfig:
    resize_size: int = RESIZE_DEFAULT_SIZE
    crop_size: int = CROP_DEFAULT_SIZE


@dataclass
class FewShotConfig:
    enable: bool = False  # whether to use few-shot evaluation
    k_or_percent: Optional[float] = None  # number of elements or % to take per class
    n_tries: int = 1  # number of tries for few-shot evaluation


@dataclass
class ModelConfig:
    """
    Model configuration for linear evaluation.
    Compatible with the original ModelConfig but with more flexible field requirements.
    """
    # Loading a local file
    config_file: str | None = None
    pretrained_weights: str | None = None
    # Loading a DINOv3 or v2 model from torch.hub
    dino_hub: str | None = None


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True  # whether to use wandb logging
    entity: str | None = None  # wandb entity/team name
    project: str = "dinov3-affordance-linear"  # wandb project name
    name: str | None = None  # experiment name (auto-generated if None)
    tags: Tuple[str, ...] = ()  # experiment tags
    notes: str | None = None  # experiment notes
    resume: str = "allow"  # resume policy: "allow", "must", "never", "auto"


@dataclass
class LinearAffordanceEvalConfig:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    save_results: bool = False  # save predictions and targets in the output directory
    output_dir: str = ""
    cache_embeddings: bool = True  # whether to cache DINOv3 embeddings


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features for affordance classification"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=7):  # 7 affordance classes by default
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    # Handle both distributed and single-device scenarios
    try:
        world_size = distributed.get_world_size() if distributed.is_enabled() else 1
    except:
        world_size = 1
    return learning_rates * (batch_size * world_size) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=7):
    device = get_device()
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = to_device(linear_classifier, device)
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


def make_eval_transform(config: TransformConfig):
    if config.resize_size / config.crop_size != 256 / 224:
        logger.warning(
            f"Default resize / crop ratio is 256 / 224, here we have {config.resize_size} / {config.crop_size}"
        )
    transform = make_classification_eval_transform(resize_size=config.resize_size, crop_size=config.crop_size)
    return transform


def make_eval_data_loader(
    *,
    test_dataset_str,
    transform_config,
    batch_size,
    num_workers,
    metric_type,
    use_enumerated_targets=False,
):
    transform = make_eval_transform(transform_config)
    test_dataset = make_dataset(dataset_str=test_dataset_str, transform=transform)

    class_mapping = None
    if hasattr(test_dataset, "get_imagenet_class_mapping"):
        class_mapping = test_dataset.get_imagenet_class_mapping()

    # Handle distributed and single-device scenarios
    try:
        num_replicas = distributed.get_world_size() if distributed.is_enabled() else 1
    except:
        num_replicas = 1
    
    # Use enumerated targets for caching if requested
    if use_enumerated_targets:
        dataset_with_enum = DatasetWithEnumeratedTargets(test_dataset, pad_dataset=True, num_replicas=num_replicas)
    else:
        dataset_with_enum = DatasetWithEnumeratedTargets(test_dataset, pad_dataset=True, num_replicas=num_replicas)
    
    test_data_loader = make_data_loader(
        dataset=dataset_with_enum,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED if num_replicas > 1 else None,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=pad_multilabel_and_collate if metric_type == ClassificationMetricType.ANY_MATCH_ACCURACY else None,
    )
    return test_data_loader, class_mapping


@dataclass
class Evaluator:
    batch_size: int
    num_workers: int
    transform_config: TransformConfig
    dataset_str: str
    metric_type: ClassificationMetricType
    metrics_file_path: str
    training_num_classes: int
    save_results_func: Optional[Callable]
    use_cached_embeddings: bool = False
    cached_embeddings: Optional[Dict[int, torch.Tensor]] = None

    def __post_init__(self):
        # Only create data loader if not using cached embeddings
        if not self.use_cached_embeddings:
            self.data_loader, self.class_mapping = make_eval_data_loader(
                test_dataset_str=self.dataset_str,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                transform_config=self.transform_config,
                metric_type=self.metric_type,
                use_enumerated_targets=True,  # Always use enumerated for potential caching
            )
        else:
            self.class_mapping = None
            # Create a dummy data loader for cached embeddings
            self._create_cached_data_loader()
        
        # Use per-class accuracy as main metric name
        self.main_metric_name = f"{self.dataset_str}_per_class_accuracy"
    
    def _create_cached_data_loader(self):
        """Create a data loader using cached embeddings."""
        if self.cached_embeddings is None:
            raise ValueError("Cached embeddings not provided")
        
        # Create the original dataset to get targets
        transform = make_eval_transform(self.transform_config)
        original_dataset = make_dataset(dataset_str=self.dataset_str, transform=transform)
        
        # Create cached dataset
        cached_dataset = CachedDataset(original_dataset, self.cached_embeddings)
        
        # Handle distributed and single-device scenarios
        try:
            num_replicas = distributed.get_world_size() if distributed.is_enabled() else 1
        except:
            num_replicas = 1
        
        self.data_loader = make_data_loader(
            dataset=cached_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler_type=SamplerType.DISTRIBUTED if num_replicas > 1 else None,
            drop_last=False,
            shuffle=False,
            persistent_workers=False,
        )

    @torch.no_grad()
    def _evaluate_linear_classifiers(
        self,
        *,
        feature_model,
        linear_classifiers,
        iteration,
        prefixstring="",
        best_classifier_on_val=None,
        accumulate_results=False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        logger.info("running affordance validation!")

        num_classes = len(self.class_mapping) if self.class_mapping is not None else self.training_num_classes
        metric = build_classification_metric(self.metric_type, num_classes=num_classes)
        postprocessors = {
            k: LinearPostprocessor(v, self.class_mapping) for k, v in linear_classifiers.classifiers_dict.items()
        }
        metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

        if self.use_cached_embeddings:
            # Use cached embeddings - directly pass them to evaluate function
            _, results_dict_temp, accumulated_results = self._evaluate_with_cached_embeddings(
                linear_classifiers,
                postprocessors,
                metrics,
                accumulate_results=accumulate_results,
            )
        else:
            # Use original evaluation with feature model
            _, results_dict_temp, accumulated_results = evaluate(
                feature_model,
                self.data_loader,
                postprocessors,
                metrics,
                get_current_device(),
                accumulate_results=accumulate_results,
            )

        logger.info("")
        results_dict = {}
        max_accuracy = 0
        best_classifier = ""
        for _, (classifier_string, metric) in enumerate(results_dict_temp.items()):
            logger.info(f"{prefixstring} -- Affordance Classifier: {classifier_string} * {metric}")
            if (
                best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
            ) or classifier_string == best_classifier_on_val:
                max_accuracy = metric["top-1"].item()
                best_classifier = classifier_string

        results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}

        logger.info(f"best affordance classifier: {results_dict['best_classifier']}")

        accumulated_best_results = None
        if accumulated_results is not None:
            accumulated_best_results = accumulated_results[best_classifier]

        if distributed.is_main_process():
            with open(self.metrics_file_path, "a") as f:
                f.write(f"iter: {iteration}\n")
                for k, v in results_dict.items():
                    f.write(json.dumps({k: v}) + "\n")
                f.write("\n")

        return results_dict, accumulated_best_results

    def _evaluate_with_cached_embeddings(
        self,
        linear_classifiers,
        postprocessors,
        metrics,
        accumulate_results=False,
    ):
        """Evaluate using cached embeddings instead of running feature extraction."""
        device = get_current_device()
        
        # Collect all predictions and targets
        all_predictions = {k: [] for k in postprocessors.keys()}
        all_targets = {k: [] for k in postprocessors.keys()}
        
        for embeddings_batch, targets_batch in self.data_loader:
            embeddings_batch = to_device(embeddings_batch, device, non_blocking=True)
            targets_batch = to_device(targets_batch, device, non_blocking=True)
            
            # Get classifier outputs
            classifier_outputs = linear_classifiers(embeddings_batch)
            
            # Apply postprocessors and collect results
            for classifier_name, postprocessor in postprocessors.items():
                output = postprocessor(classifier_outputs[classifier_name], targets_batch)
                all_predictions[classifier_name].append(output["preds"])
                all_targets[classifier_name].append(output["target"])
        
        # Concatenate all batches and compute metrics
        results_dict = {}
        accumulated_results = {} if accumulate_results else None
        
        for classifier_name in postprocessors.keys():
            preds = torch.cat(all_predictions[classifier_name], dim=0)
            targets = torch.cat(all_targets[classifier_name], dim=0)
            
            # Update metric
            metrics[classifier_name].update(preds, targets)
            results_dict[classifier_name] = metrics[classifier_name].compute()
            
            if accumulate_results:
                accumulated_results[classifier_name] = {
                    "preds": preds,
                    "target": targets,
                }
        
        return None, results_dict, accumulated_results

    def evaluate_and_maybe_save(
        self,
        feature_model,
        linear_classifiers,
        iteration: int,
        best_classifier_on_val: Optional[Any] = None,
        save_filename_suffix: str = "",
        prefixstring: str = "",
        use_wandb: bool = False,
    ):
        logger.info(f"Testing affordance classification on {self.dataset_str}")
        save_results = self.save_results_func is not None
        full_results_dict, accumulated_best_results = self._evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=remove_ddp_wrapper(linear_classifiers),
            iteration=iteration,
            prefixstring=prefixstring,
            best_classifier_on_val=best_classifier_on_val,
            accumulate_results=save_results,
        )
        if self.save_results_func is not None:
            self.save_results_func(
                filename_suffix=f"{self.dataset_str}{save_filename_suffix}", **accumulated_best_results
            )

        results_dict = {
            self.main_metric_name: 100.0 * full_results_dict["best_classifier"]["accuracy"],
            "best_classifier": full_results_dict["best_classifier"]["name"],
        }
        
        # Log evaluation metrics to wandb
        if use_wandb and distributed.is_main_process():
            # Extract dataset type from dataset string (val, test, etc.)
            dataset_parts = self.dataset_str.split(":")
            dataset_name = dataset_parts[0] if dataset_parts else self.dataset_str
            split_type = "unknown"
            for part in dataset_parts:
                if "split=" in part:
                    split_type = part.split("=")[1].lower()
                    break
            
            affordance_type = "unknown"
            for part in dataset_parts:
                if "affordance_type=" in part:
                    affordance_type = part.split("=")[1].lower()
                    break
            
            # Log the main accuracy metric
            wandb.log({
                f"eval/{split_type}/{affordance_type}_accuracy": results_dict[self.main_metric_name],
                f"eval/{split_type}/best_classifier": results_dict["best_classifier"],
                "train/iteration": iteration,
            }, step=iteration)
            
            # Also log overall accuracy if this is validation
            if "val" in split_type.lower():
                wandb.log({
                    f"val/accuracy": results_dict[self.main_metric_name],
                }, step=iteration)
        
        return results_dict


def make_evaluators(
    eval_config: EvalConfig,
    val_metric_type: ClassificationMetricType,
    val_dataset: str,
    transform_config: TransformConfig,
    metrics_file_path: str,
    training_num_classes: int,
    save_results_func: Optional[Callable],
    cached_embeddings: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
):
    test_metric_types = eval_config.test_metric_types
    if len(test_metric_types) == 0:
        test_metric_types = (val_metric_type,) * len(eval_config.test_datasets)
    else:
        assert len(test_metric_types) == len(eval_config.test_datasets)
    
    # Determine cache usage
    use_cached = cached_embeddings is not None
    
    evaluator_params = []
    dataset_strings = (val_dataset,) + tuple(eval_config.test_datasets)
    metric_types = (val_metric_type,) + tuple(test_metric_types)
    
    for dataset_str, metric_type in zip(dataset_strings, metric_types):
        params = {
            "dataset_str": dataset_str,
            "batch_size": eval_config.batch_size,
            "num_workers": eval_config.num_workers,
            "transform_config": transform_config,
            "metric_type": metric_type,
            "metrics_file_path": metrics_file_path,
            "training_num_classes": training_num_classes,
            "save_results_func": save_results_func,
            "use_cached_embeddings": use_cached,
        }
        
        if use_cached and dataset_str in cached_embeddings:
            params["cached_embeddings"] = cached_embeddings[dataset_str]
        
        evaluator_params.append(params)
    
    evaluators = [Evaluator(**params) for params in evaluator_params]
    val_evaluator = evaluators[0]
    test_evaluators = evaluators[1:]
    
    return val_evaluator, test_evaluators


def setup_linear_training(
    *,
    config: TrainConfig,
    sample_output: torch.Tensor,
    training_num_classes: int,
    checkpoint_output_dir: str,
):
    device = get_device()
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        config.n_last_blocks_list,
        config.learning_rates,
        config.batch_size,
        training_num_classes,
    )
    max_iter = config.epochs * config.epoch_length
    optimizer = config.optimizer_type.get_optimizer(optim_param_groups=optim_param_groups)
    scheduler = config.scheduler_type.get_scheduler(
        optimizer=optimizer,
        optim_param_groups=optim_param_groups,
        epoch_length=config.epoch_length,
        epochs=config.epochs,
        max_iter=max_iter,
    )
    start_iter = 0
    best_accuracy = -1
    if config.resume and (
        last_checkpoint_dir := find_latest_checkpoint(config.classifier_fpath or checkpoint_output_dir)
    ):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        checkpoint = torch.load(last_checkpoint_dir / "checkpoint.pth", map_location=device)
        start_iter = checkpoint.get("iteration", -1) + 1
        best_accuracy = checkpoint.get("best_accuracy", -1)
        linear_classifiers.load_state_dict(checkpoint["linear_classifiers"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if config.loss_type == LossType.BINARY_CROSS_ENTROPY:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # Use weighted cross-entropy to handle class imbalance in affordance data
        criterion = nn.CrossEntropyLoss()

    return (
        linear_classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        best_accuracy,
    )


def train_linear_classifiers(
    *,
    feature_model,
    train_dataset,
    train_config: TrainConfig,
    training_num_classes: int,
    val_evaluator: Evaluator,
    checkpoint_output_dir: str,
    use_wandb: bool = False,
    use_cached_embeddings: bool = False,
):
    device = get_device()
    
    # Setup differs based on whether we're using cached embeddings
    if use_cached_embeddings:
        # For cached embeddings, we need to get a sample from the dataset directly
        sample_embedding = train_dataset[0][0].unsqueeze(0)  # First item is embedding
        sample_output = sample_embedding  # Already processed
    else:
        # Original behavior - extract features from sample
        sample_output = feature_model(to_device(train_dataset[0][0].unsqueeze(0), device))
    
    (linear_classifiers, start_iter, max_iter, criterion, optimizer, scheduler, best_accuracy,) = setup_linear_training(
        config=train_config,
        sample_output=sample_output,
        training_num_classes=training_num_classes,
        checkpoint_output_dir=checkpoint_output_dir,
    )
    checkpoint_period = train_config.save_checkpoint_iterations or train_config.epoch_length
    eval_period = train_config.eval_period_iterations or train_config.epoch_length

    sampler_type = SamplerType.INFINITE
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )

    iteration = start_iter
    logger.info("Starting affordance linear training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6g}"))
    metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.3f}"))
    header = "Affordance Training"
    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = to_device(data, device, non_blocking=True)
        labels = to_device(labels, device, non_blocking=True)

        # Get features - either from cache or feature model
        if use_cached_embeddings:
            features = data  # Data is already the embeddings
        else:
            features = feature_model(data)
        
        outputs = linear_classifiers(features)

        if len(labels.shape) > 1:
            labels = labels.float()
        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            sync_device()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            # Calculate train accuracy for logging
            train_accuracy = None
            if use_wandb and distributed.is_main_process():
                with torch.no_grad():
                    # Get predictions from all classifiers and compute accuracy
                    total_correct = 0
                    total_samples = 0
                    for classifier_name, output in outputs.items():
                        if len(labels.shape) == 1:  # Classification labels
                            predictions = torch.argmax(output, dim=1)
                            correct = (predictions == labels).sum().item()
                        else:  # Multi-label case
                            predictions = (torch.sigmoid(output) > 0.5).float()
                            correct = (predictions == labels).all(dim=1).sum().item()
                        
                        total_correct += correct
                        total_samples += labels.size(0)
                    
                    # Average accuracy across all classifiers
                    train_accuracy = (total_correct / total_samples) / len(outputs) if total_samples > 0 else 0.0
                    metric_logger.update(acc=train_accuracy)
            
            # Log to wandb if enabled
            if use_wandb and distributed.is_main_process():
                log_dict = {
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/iteration": iteration,
                    "train/epoch": iteration / train_config.epoch_length,
                }
                if train_accuracy is not None:
                    log_dict["train/accuracy"] = train_accuracy * 100.0  # Convert to percentage
                
                wandb.log(log_dict, step=iteration)

        # Checkpointing
        is_last_iteration = (iteration + 1) == max_iter
        is_ckpt_iteration = ((iteration + 1) % checkpoint_period == 0) or is_last_iteration
        if is_ckpt_iteration:
            ckpt_dir = Path(checkpoint_output_dir).expanduser()
            if distributed.is_subgroup_main_process():
                ckpt_sub_dir = "final" if is_last_iteration else str(iteration)
                (ckpt_dir / ckpt_sub_dir).mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "iteration": iteration,
                    "linear_classifiers": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }
                torch.save(checkpoint, ckpt_dir / ckpt_sub_dir / "checkpoint.pth")
                keep_last_n_checkpoints(ckpt_dir, train_config.checkpoint_retention_policy.max_to_keep)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            val_results_dict = val_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                prefixstring=f"ITER: {iteration}",
                iteration=iteration,
                use_wandb=use_wandb,
            )
            val_accuracy = val_results_dict[val_evaluator.main_metric_name]
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                (ckpt_dir / "best").mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "iteration": iteration,
                    "linear_classifiers": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }
                torch.save(checkpoint, ckpt_dir / "best" / "checkpoint.pth")
                
                # Log best accuracy to wandb
                if use_wandb and distributed.is_main_process():
                    wandb.log({
                        "val/best_accuracy": best_accuracy,
                        "train/iteration_at_best": iteration,
                    }, step=iteration)
            
            # Only barrier if distributed is enabled
            if distributed.is_enabled():
                torch.distributed.barrier()

        iteration = iteration + 1

    return feature_model, linear_classifiers, iteration


def make_train_transform(config: TransformConfig):
    train_transform = make_classification_train_transform(crop_size=config.crop_size)
    return train_transform


def make_train_dataset(train_dataset: str, transform_config: TransformConfig, cached_embeddings: Optional[Dict[int, torch.Tensor]] = None):
    if cached_embeddings is not None:
        # Create original dataset to get structure
        train_transform = make_train_transform(transform_config)
        original_dataset = make_dataset(dataset_str=train_dataset, transform=train_transform)
        # Return cached dataset
        return CachedDataset(original_dataset, cached_embeddings)
    else:
        # Original behavior
        train_transform = make_train_transform(transform_config)
        return make_dataset(dataset_str=train_dataset, transform=train_transform)


def eval_linear_with_model(*, model: torch.nn.Module, autocast_dtype, config: LinearAffordanceEvalConfig):
    start = time.time()
    device = get_device()
    
    # Initialize wandb if enabled
    use_wandb = config.wandb.enabled and distributed.is_main_process()
    if use_wandb:
        # Generate experiment name if not provided
        run_name = config.wandb.name
        if run_name is None:
            # Extract key info from dataset and model config for auto-naming
            try:
                dataset_parts = config.train.dataset.split(":")
                affordance_type = "unknown"
                for part in dataset_parts:
                    if "affordance_type=" in part:
                        affordance_type = part.split("=")[1].lower()
                        break
                model_info = config.model.dino_hub or "custom"
                run_name = f"linear_{affordance_type}_{model_info}"
            except:
                run_name = "linear_affordance_eval"
        
        # Initialize wandb
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=run_name,
            tags=list(config.wandb.tags),
            notes=config.wandb.notes,
            resume=config.wandb.resume,
            config={
                # Model configuration
                "model_type": config.model.dino_hub or "custom",
                "model_pretrained_weights": config.model.pretrained_weights,
                "model_config_file": config.model.config_file,
                
                # Training configuration
                "train_dataset": config.train.dataset,
                "val_dataset": config.train.val_dataset,
                "batch_size": config.train.batch_size,
                "learning_rates": list(config.train.learning_rates),
                "n_last_blocks_list": list(config.train.n_last_blocks_list),
                "epochs": config.train.epochs,
                "epoch_length": config.train.epoch_length,
                "optimizer_type": config.train.optimizer_type.value,
                "scheduler_type": config.train.scheduler_type.value,
                "loss_type": config.train.loss_type.value,
                "val_metric_type": config.train.val_metric_type.value,
                
                # Transform configuration
                "crop_size": config.transform.crop_size,
                "resize_size": config.transform.resize_size,
                
                # Evaluation configuration
                "test_datasets": list(config.eval.test_datasets),
                "eval_batch_size": config.eval.batch_size,
                
                # Few-shot configuration
                "few_shot_enabled": config.few_shot.enable,
                "few_shot_k_or_percent": config.few_shot.k_or_percent,
                "few_shot_n_tries": config.few_shot.n_tries,
                
                # Caching configuration
                "cache_embeddings": config.cache_embeddings,
                
                # System information
                "device": str(device),
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "distributed": distributed.is_enabled(),
            }
        )
        logger.info(f"Initialized wandb run: {wandb.run.name}")
    
    # Only set cudnn benchmark if using CUDA
    if device.type == "cuda":
        cudnn.benchmark = True

    # Initialize embeddings cache
    cached_embeddings = {}
    use_cached_embeddings = config.cache_embeddings
    
    # Generate cache keys for all datasets
    cache_keys = {}
    dataset_strings = [config.train.dataset, config.train.val_dataset] + list(config.eval.test_datasets)
    for dataset_str in dataset_strings:
        cache_key = get_cache_key(dataset_str, config.model, config.transform)
        cache_keys[dataset_str] = cache_key
    
    if use_cached_embeddings:
        logger.info("Checking for cached embeddings...")
        
        # Check if all required caches exist
        all_caches_exist = True
        for dataset_str in dataset_strings:
            cache_key = cache_keys[dataset_str]
            split_name = "train" if "TRAIN" in dataset_str.upper() else ("val" if "VAL" in dataset_str.upper() else "test")
            cache_path = get_cache_path(config.output_dir, cache_key, split_name)
            
            embeddings = load_embeddings_from_cache(cache_path)
            if embeddings is not None:
                cached_embeddings[dataset_str] = embeddings
                logger.info(f"Loaded cached embeddings for {dataset_str}")
            else:
                all_caches_exist = False
                logger.info(f"No cached embeddings found for {dataset_str}")
        
        if not all_caches_exist:
            logger.info("Not all caches available. Generating embeddings...")
            use_cached_embeddings = False
            cached_embeddings = {}

    # Create datasets and extract embeddings if needed
    if not use_cached_embeddings:
        # Create the feature model for embedding extraction
        n_last_blocks = max(config.train.n_last_blocks_list)
        
        # Set up autocast context based on device type
        if device.type == "cuda":
            autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
        else:
            # For CPU, we can use autocast but it's less critical
            autocast_ctx = partial(torch.autocast, device_type="cpu", enabled=True)
        
        feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
        
        # Extract embeddings for all datasets
        for dataset_str in dataset_strings:
            cache_key = cache_keys[dataset_str]
            split_name = "train" if "TRAIN" in dataset_str.upper() else ("val" if "VAL" in dataset_str.upper() else "test")
            cache_path = get_cache_path(config.output_dir, cache_key, split_name)
            
            # Create data loader for embedding extraction
            data_loader, _ = make_eval_data_loader(
                test_dataset_str=dataset_str,
                transform_config=config.transform,
                batch_size=config.eval.batch_size,
                num_workers=config.eval.num_workers,
                metric_type=config.train.val_metric_type,
                use_enumerated_targets=True,
            )
            
            # Extract and cache embeddings
            embeddings = extract_and_cache_embeddings(feature_model, data_loader, cache_path, device)
            cached_embeddings[dataset_str] = embeddings
        
        use_cached_embeddings = True
        logger.info("All embeddings extracted and cached.")

    # Create datasets using cached embeddings
    train_dataset = make_train_dataset(config.train.dataset, config.transform, 
                                     cached_embeddings.get(config.train.dataset))
    training_num_classes = get_num_classes(train_dataset)
    logger.info(f"AffordanceADE dataset has {training_num_classes} classes")
    logger.info(f"Using device: {device}")
    logger.info(f"Using cached embeddings: {use_cached_embeddings}")
    
    # Log dataset information to wandb
    if use_wandb:
        wandb.log({
            "dataset/num_classes": training_num_classes,
            "dataset/train_size": len(train_dataset),
            "dataset/using_cached_embeddings": use_cached_embeddings,
        })
    
    train_dataset_dict = create_train_dataset_dict(
        train_dataset,
        few_shot_eval=config.few_shot.enable,
        few_shot_k_or_percent=config.few_shot.k_or_percent,
        few_shot_n_tries=config.few_shot.n_tries,
    )
    
    # Create feature model (may not be used if cached)
    n_last_blocks = max(config.train.n_last_blocks_list)
    
    # Set up autocast context based on device type
    if device.type == "cuda":
        autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    else:
        # For CPU, we can use autocast but it's less critical
        autocast_ctx = partial(torch.autocast, device_type="cpu", enabled=True)
    
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    save_results_func = None
    if config.save_results:
        save_results_func = partial(default_save_results_func, output_dir=config.output_dir)

    metrics_file_path = os.path.join(config.output_dir, "results_eval_linear_affordance.json")
    val_evaluator, test_evaluators = make_evaluators(
        eval_config=config.eval,
        val_metric_type=config.train.val_metric_type,
        val_dataset=config.train.val_dataset,
        transform_config=config.transform,
        metrics_file_path=metrics_file_path,
        training_num_classes=training_num_classes,
        save_results_func=save_results_func,
        cached_embeddings=cached_embeddings if use_cached_embeddings else None,
    )
    
    results_dict = {}
    checkpoint_output_dirs: list = []
    for _try in train_dataset_dict.keys():
        if len(train_dataset_dict) > 1:
            checkpoint_output_dir = os.path.join(config.output_dir, f"checkpoints_{_try}")
            save_filename_suffix = f"_{_try}"
        else:
            checkpoint_output_dir = os.path.join(config.output_dir, "checkpoints")
            save_filename_suffix = ""
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        feature_model, linear_classifiers, iteration = train_linear_classifiers(
            feature_model=feature_model,
            train_dataset=train_dataset_dict[_try],
            train_config=config.train,
            training_num_classes=training_num_classes,
            val_evaluator=val_evaluator,
            checkpoint_output_dir=checkpoint_output_dir,
            use_wandb=use_wandb,
            use_cached_embeddings=use_cached_embeddings,
        )
        checkpoint_output_dirs.append(checkpoint_output_dir)
        results_dict[_try] = val_evaluator.evaluate_and_maybe_save(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            iteration=iteration,
            save_filename_suffix=save_filename_suffix,
            use_wandb=use_wandb,
        )
        for test_evaluator in test_evaluators:
            eval_results_dict = test_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                iteration=iteration,
                best_classifier_on_val=results_dict[_try]["best_classifier"],
                save_filename_suffix=save_filename_suffix,
                use_wandb=use_wandb,
            )
            results_dict[_try] = {**eval_results_dict, **results_dict[_try]}

    if len(train_dataset_dict) > 1:
        results_dict = average_metrics(results_dict, ignore_keys=["best_classifier"])
    else:
        results_dict = {**results_dict[_try]}

    for checkpoint_output_dir in checkpoint_output_dirs:
        if distributed.is_subgroup_main_process():
            cleanup_checkpoint(checkpoint_output_dir, config.train.checkpoint_retention_policy)

    # Log final results to wandb
    if use_wandb:
        final_results = {}
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                # Clean up key names for better wandb organization
                if "per_class_accuracy" in key:
                    clean_key = key.replace("_per_class_accuracy", "").replace(":", "_")
                    final_results[f"final/{clean_key}_accuracy"] = value
                elif "best_classifier" not in key:  # Skip non-numeric classifier names
                    final_results[f"final/{key}"] = value
        
        wandb.log(final_results)
        
        # Create a summary table
        wandb.summary.update({
            "final_validation_accuracy": results_dict.get(val_evaluator.main_metric_name, 0),
            "best_classifier_name": results_dict.get("best_classifier", "unknown"),
            "total_training_time_seconds": int(time.time() - start),
            "training_iterations": iteration,
            "used_cached_embeddings": use_cached_embeddings,
        })
        
        logger.info(f"Wandb run completed: {wandb.run.url}")

    logger.info("Affordance Test Results Dict " + str(results_dict))
    logger.info(f"Affordance linear evaluation done in {int(time.time() - start)}s")
    return results_dict


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=LinearAffordanceEvalConfig)
    model, model_context = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    results_dict = eval_linear_with_model(
        model=model, config=dataclass_config, autocast_dtype=model_context["autocast_dtype"]
    )
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    
    # Fix CLI argument keys by removing '--' prefixes
    # OmegaConf.from_cli creates keys with '--' prefixes that don't match our dataclass fields
    processed_args = {}
    for key, value in eval_args.items():
        if key.startswith('--'):
            # Remove the '--' prefix to match dataclass field names
            clean_key = key[2:].replace('-', '_')  # Also convert dashes to underscores
            processed_args[clean_key] = value
        else:
            processed_args[key] = value
    
    # Determine if distributed training should be enabled
    # Enable distributed only if CUDA is available and multiple GPUs are detected
    # This ensures CPU-only setups and single-GPU setups work correctly
    device = get_device()
    distributed_enabled = device.type == "cuda" and torch.cuda.device_count() > 1
    
    with job_context(distributed_enabled=distributed_enabled, output_dir=processed_args["output_dir"]):
        benchmark_launcher(processed_args)
    return 0


if __name__ == "__main__":
    main()
