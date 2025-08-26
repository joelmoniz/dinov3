# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
AffordanceADE dataset for DINOv3.

This dataset loads instances from ADE20K with affordance annotations and creates
highlighted images for classification. Each sample contains a highlighted object
and its affordance label (sit/run/grasp capability).
"""

import os
import logging
import warnings
import numpy as np
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union, Any
from PIL import Image
from io import BytesIO

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder

logger = logging.getLogger("dinov3")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val" 
    TEST = "test"

    @property
    def length(self) -> int:
        # TODO: These would need to be updated based on actual dataset sizes
        return {"train": 177231, "val": 22725, "test": 21873}[self.value]


class _AffordanceType(Enum):
    SIT = "sit"
    RUN = "run"
    GRASP = "grasp"


class AffordanceADE(ExtendedVisionDataset):
    """
    Extended affordance dataset following DINOv3 patterns.
    
    This dataset loads instances from ADE20K with affordance annotations and creates
    highlighted images for classification. Each sample contains a highlighted object
    and its affordance label (sit/run/grasp capability).
    """
    
    Split = _Split
    AffordanceType = _AffordanceType

    def __init__(
        self,
        *,
        split: "AffordanceADE.Split",
        affordance_type: "AffordanceADE.AffordanceType",
        root: str = None,
        extra: str = None,
        ade20k_root_dir: str = None,
        ade_affordance_root_dir: str = None,
        preprocd_root_dir: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        highlight_params: Optional[dict] = None,
    ) -> None:
        """
        Initialize the AffordanceADE dataset.
        
        Args:
            split: Dataset split (train/val/test)
            affordance_type: Type of affordance to classify (sit/run/grasp)
            root: Root directory (for compatibility with torchvision)
            extra: Extra directory for cached data
            ade20k_root_dir: Path to ADE20K dataset
            ade_affordance_root_dir: Path to ADE affordance annotations
            preprocd_root_dir: Path to preprocessed TSV files
            transforms: Transforms to apply to (image, target) pairs
            transform: Transforms to apply to images only
            target_transform: Transforms to apply to targets only
            highlight_params: Parameters for object highlighting
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder
        )
        
        self._extra_root = extra
        self._split = split
        self._affordance_type = affordance_type
        
        # Default highlighting parameters
        self._highlight_params = {
            'alpha': 0,
            'border_thickness': 10,
            'contour_color': (255, 0, 255),  # Magenta
            'crop': False
        }
        if highlight_params:
            self._highlight_params.update(highlight_params)
        
        # Initialize dataset with lazy loading approach
        self._entries = None
        self._dataset_initialized = False
        
        # Store dataset parameters for lazy initialization
        self._ade20k_root_dir = ade20k_root_dir
        self._ade_affordance_root_dir = ade_affordance_root_dir
        self._preprocd_root_dir = preprocd_root_dir
        
    def _initialize_dataset(self):
        """Lazy initialization of the dataset to avoid import issues."""
        if self._dataset_initialized:
            return
            
        try:
            # Import the existing unified dataset infrastructure
            import sys
            import os
            
            # Add the AffordanceADE directory to the path
            affordance_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'AffordanceADE')
            if affordance_path not in sys.path:
                sys.path.append(affordance_path)
            
            from affordance.constants import AFFORDANCE_MAPPING
            from unified.dataset_loader import Dataset
            from ade20k.image_loader import loadAde20K
            from ade20k.image_drawer import highlight_object
            
            # Store these for later use
            self._AFFORDANCE_MAPPING = AFFORDANCE_MAPPING
            self._Dataset = Dataset
            self._loadAde20K = loadAde20K
            self._highlight_object = highlight_object
            
        except ImportError as e:
            logger.warning(f"Could not import affordance dataset dependencies: {e}")
            # Create fallback implementations
            self._AFFORDANCE_MAPPING = ["negative", "exception1", "exception2", "exception3", "exception4", "exception5", "positive"]
            self._Dataset = None
            self._loadAde20K = None
            self._highlight_object = None
            
        # Load the underlying dataset
        if self._Dataset is not None:
            if self._preprocd_root_dir:
                self._dataset = self._Dataset(preprocd_root_dir=self._preprocd_root_dir, split=self._split.value)
            else:
                self._dataset = self._Dataset(
                    ade20k_root_dir=self._ade20k_root_dir,
                    ade_affordance_root_dir=self._ade_affordance_root_dir,
                    split=self._split.value
                )
            
            # Filter instances that have the target affordance type
            self._entries = self._filter_affordance_entries()
        else:
            # Fallback to empty dataset
            self._dataset = None
            self._entries = []
            
        self._dataset_initialized = True
        
    def _filter_affordance_entries(self) -> List:
        """Filter dataset entries to only include instances with the target affordance."""
        if self._dataset is None:
            return []
            
        filtered_entries = []
        
        for item in self._dataset.data:
            if not item.affordance:
                continue
                
            # Get the affordance value based on type
            if self._affordance_type == _AffordanceType.SIT:
                aff_values = item.affordance.sit_value
            elif self._affordance_type == _AffordanceType.RUN:
                aff_values = item.affordance.run_value
            elif self._affordance_type == _AffordanceType.GRASP:
                aff_values = item.affordance.grasp_value
            else:
                continue
                
            # Skip if no affordance values or invalid values
            if not aff_values or aff_values[0] in [None, -1]:
                continue
                
            filtered_entries.append(item)
            
        logger.info(f"Filtered {len(filtered_entries)} entries with {self._affordance_type.value} affordance from {len(self._dataset.data)} total entries")
        return filtered_entries

    @property
    def split(self) -> "AffordanceADE.Split":
        return self._split
    
    @property
    def affordance_type(self) -> "AffordanceADE.AffordanceType":
        return self._affordance_type

    def get_image_data(self, index: int) -> bytes:
        """Get highlighted image data as bytes (required by ExtendedVisionDataset)."""
        self._initialize_dataset()
        
        if not self._entries or index >= len(self._entries):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._entries)}")
            
        entry = self._entries[index]
        
        if self._loadAde20K is None or self._highlight_object is None:
            # Fallback: create a dummy image
            dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img_buffer = BytesIO()
            dummy_image.save(img_buffer, format='JPEG', quality=95)
            return img_buffer.getvalue()
        
        # Load ADE20K image and mask data
        image_props = self._loadAde20K(entry.img_name)
        
        # Load the original image
        with Image.open(image_props['img_name']) as pil_img:
            image = np.array(pil_img)
        
        # Create binary mask for the instance
        binary_mask = (image_props['instance_mask'] == entry.instance_idx).astype(np.uint8)
        
        # Highlight the object
        highlighted_image = self._highlight_object(
            image, 
            binary_mask,
            text=entry.instance_class,
            **self._highlight_params
        )
        
        # Convert back to PIL and then to bytes
        highlighted_pil = Image.fromarray(highlighted_image)
        img_buffer = BytesIO()
        highlighted_pil.save(img_buffer, format='JPEG', quality=95)
        
        return img_buffer.getvalue()

    def get_target(self, index: int) -> int:
        """Get the affordance classification target (required by ExtendedVisionDataset)."""
        self._initialize_dataset()
        
        if not self._entries or index >= len(self._entries):
            return 0
            
        entry = self._entries[index]
        
        # Get the affordance value based on type
        if self._affordance_type == _AffordanceType.SIT:
            aff_values = entry.affordance.sit_value
        elif self._affordance_type == _AffordanceType.RUN:
            aff_values = entry.affordance.run_value
        elif self._affordance_type == _AffordanceType.GRASP:
            aff_values = entry.affordance.grasp_value
        else:
            return 0
        
        # Target: Affordance class
        if aff_values and len(aff_values) > 0:
            return aff_values[0]
        return 0
    
    def get_targets(self) -> np.ndarray:
        """Get all targets as numpy array."""
        targets = []
        for i in range(len(self)):
            targets.append(self.get_target(i))
        return np.array(targets)

    def get_class_name(self, index: int) -> str:
        """Get the object class name."""
        self._initialize_dataset()
        
        aff_value = self.get_target(index)
        if 0 <= aff_value < len(self._AFFORDANCE_MAPPING):
            return self._AFFORDANCE_MAPPING[aff_value]
        return "unknown"

    def get_affordance_explanation(self, index: int) -> str:
        """Get the affordance explanation if available."""
        self._initialize_dataset()
        
        if not self._entries or index >= len(self._entries):
            return ""
            
        entry = self._entries[index]
        
        if self._affordance_type == _AffordanceType.SIT:
            explanations = entry.affordance.sit_explanation
        elif self._affordance_type == _AffordanceType.RUN:
            explanations = entry.affordance.run_explanation
        elif self._affordance_type == _AffordanceType.GRASP:
            explanations = entry.affordance.grasp_explanation
        else:
            return ""
            
        if explanations and len(explanations) > 0:
            return explanations[0]
        return ""

    def __len__(self) -> int:
        """Get the number of items in the dataset (required by ExtendedVisionDataset)."""
        self._initialize_dataset()
        return len(self._entries) if self._entries else 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a single item from the dataset."""
        try:
            # Get highlighted image data
            image_data = self.get_image_data(index)
            
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e
        
        # Get target
        target = self.get_target(index)
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        elif self.transform is not None:
            image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return image, target

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        self._initialize_dataset()
        
        targets = self.get_targets()
        
        # Categorize affordance values:
        # 0: Negative (Firmly Negative)
        # 1-5: Exception classes (various obstacle/awkward/forbidden categories)
        # 6: Positive
        positive_count = np.sum(targets == 6)
        negative_count = np.sum(targets == 0)
        exception_count = np.sum((targets >= 1) & (targets <= 5))
        
        # Detailed class distribution
        class_counts = {}
        for i in range(len(self)):
            class_name = self.get_class_name(i)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Category-specific counts
        category_counts = {
            'positive': int(positive_count),
            'negative': int(negative_count),
            'exception': int(exception_count)
        }
        
        return {
            'total_samples': len(self),
            'positive_affordance': int(positive_count),
            'negative_affordance': int(negative_count), 
            'exception_affordance': int(exception_count),
            'positive_ratio': float(positive_count) / len(targets) if len(targets) > 0 else 0.0,
            'negative_ratio': float(negative_count) / len(targets) if len(targets) > 0 else 0.0,
            'exception_ratio': float(exception_count) / len(targets) if len(targets) > 0 else 0.0,
            'category_counts': category_counts,
            'class_distribution': class_counts,
            'affordance_type': self._affordance_type.value,
            'split': self._split.value
        }

    def save_sample_images(self, output_dir: str, num_samples: int = 10):
        """Save sample highlighted images for inspection."""
        self._initialize_dataset()
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self) == 0:
            logger.warning("No samples available to save")
            return
            
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        
        for i, idx in enumerate(indices):
            entry = self._entries[idx]
            target = self.get_target(idx)
            explanation = self.get_affordance_explanation(idx)
            
            # Get highlighted image data
            image_data = self.get_image_data(idx)
            image = Image.open(BytesIO(image_data))
            
            # Save with descriptive filename
            filename = f"sample_{i:03d}_idx{idx}_{entry.instance_class}_aff{target}_{self._affordance_type.value}.jpg"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            # Save metadata
            metadata_file = filepath.replace('.jpg', '_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Original image: {entry.img_name}\n")
                f.write(f"Instance: {entry.instance_idx}\n")
                f.write(f"Class: {entry.instance_class}\n")
                f.write(f"Subclass: {entry.instance_subclass}\n")
                f.write(f"Affordance type: {self._affordance_type.value}\n")
                f.write(f"Affordance value: {target}\n")
                if explanation:
                    f.write(f"Explanation: {explanation}\n")

    def _get_extra_full_path(self, extra_path: str) -> str:
        """Get full path for extra files (following ImageNet pattern)."""
        return os.path.join(self._extra_root, extra_path) if self._extra_root else extra_path

    def _load_extra(self, extra_path: str) -> np.ndarray:
        """Load extra data files (following ImageNet pattern)."""
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        """Save extra data files (following ImageNet pattern)."""
        extra_full_path = self._get_extra_full_path(extra_path)
        if self._extra_root:
            os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        """Path to cached entries file."""
        return f"affordance_{self._affordance_type.value}_{self._split.value}_entries.npy"

    @property
    def _targets_path(self) -> str:
        """Path to cached targets file."""
        return f"affordance_{self._affordance_type.value}_{self._split.value}_targets.npy"

    def dump_extra(self) -> None:
        """Dump dataset to extra files for faster loading (following ImageNet pattern)."""
        self._initialize_dataset()
        
        if not self._extra_root:
            logger.warning("No extra root directory specified, cannot dump extra files")
            return
            
        if not self._entries:
            logger.warning("No entries to dump")
            return
            
        # Create entry data
        entry_data = []
        target_data = []
        
        for i, entry in enumerate(self._entries):
            entry_dict = {
                'img_name': entry.img_name,
                'instance_idx': entry.instance_idx,
                'instance_class': entry.instance_class,
                'instance_subclass': entry.instance_subclass,
                'is_occluded': entry.is_occluded
            }
            entry_data.append(entry_dict)
            target_data.append(self.get_target(i))
        
        # Save as numpy arrays
        entry_array = np.array(entry_data, dtype=object)
        target_array = np.array(target_data)
        
        self._save_extra(entry_array, self._entries_path)
        self._save_extra(target_array, self._targets_path)
        
        logger.info(f"Dumped {len(entry_array)} entries and targets to {self._extra_root}")
