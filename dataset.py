#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset module for DGCD-3D.

Provides data loading utilities for paired low-field (0.23T) and high-field (3T)
DWI images with anatomical masks.
"""

import os
import re
from glob import glob
from typing import Optional, Callable, Dict, Tuple, List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class NiftiPairImageGenerator(Dataset):
    """
    Dataset for paired DWI image enhancement.
    
    Loads paired 0.23T (low-field) and 3T (high-field) DWI images along with
    anatomical masks for training conditional diffusion models.
    
    Expected file naming convention:
        - Low-field images: {subject_id}_0.23T.nii.gz
        - High-field images: {subject_id}_3T.nii.gz
        - Masks: {subject_id}_0.23T.nii.gz (in mask_folder)
    """
    
    def __init__(
        self,
        input_folder: str,
        target_folder: str,
        mask_folder: str,
        input_size: int = 256,
        depth_size: int = 13,
        input_channel: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (0.0, 1.0)
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            input_folder: Path to directory containing low-field (0.23T) images.
            target_folder: Path to directory containing high-field (3T) images.
            mask_folder: Path to directory containing anatomical masks.
            input_size: Spatial resolution (height/width) of the images.
            depth_size: Number of slices in the depth dimension.
            input_channel: Number of input channels.
            transform: Transform to apply to input images and masks.
            target_transform: Transform to apply to target images.
            normalize: Whether to normalize images using percentile-based normalization.
            normalize_range: Range for normalization (min, max).
        """
        super().__init__()
        
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.mask_folder = mask_folder
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        self.normalize_range = normalize_range
        
        # Find and pair files
        self.pair_files = self._pair_files()
        
        if len(self.pair_files) == 0:
            raise ValueError(
                "No matching file pairs found. Please check your data folders and file naming convention."
            )
    
    def _pair_files(self) -> List[Tuple[str, str, str]]:
        """
        Find and pair input, target, and mask files.
        
        Returns:
            List of tuples (input_path, target_path, mask_path).
        """
        # Get all nii.gz files
        input_files = sorted(glob(os.path.join(self.input_folder, '*.nii.gz')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.nii.gz')))
        mask_files = sorted(glob(os.path.join(self.mask_folder, '*.nii.gz')))
        
        # Extract subject IDs from filenames
        def extract_id(filepath: str, pattern: str) -> Optional[str]:
            match = re.match(pattern, os.path.basename(filepath))
            return match.group(1) if match else None
        
        input_dict = {
            extract_id(f, r"(.*)_0\.23T\.nii\.gz"): f 
            for f in input_files 
            if extract_id(f, r"(.*)_0\.23T\.nii\.gz")
        }
        target_dict = {
            extract_id(f, r"(.*)_3T\.nii\.gz"): f 
            for f in target_files 
            if extract_id(f, r"(.*)_3T\.nii\.gz")
        }
        mask_dict = {
            extract_id(f, r"(.*)_0\.23T\.nii\.gz"): f 
            for f in mask_files 
            if extract_id(f, r"(.*)_0\.23T\.nii\.gz")
        }
        
        # Find common subjects
        common_keys = set(input_dict.keys()) & set(target_dict.keys()) & set(mask_dict.keys())
        pairs = [
            (input_dict[k], target_dict[k], mask_dict[k]) 
            for k in sorted(common_keys)
        ]
        
        return pairs
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.pair_files)
    
    def _load_nifti(self, filepath: str) -> np.ndarray:
        """
        Load a NIfTI file and return as numpy array.
        
        Args:
            filepath: Path to the NIfTI file.
            
        Returns:
            Image data as numpy array.
        """
        img = nib.load(filepath).get_fdata()
        return img
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply percentile-based normalization.
        
        Args:
            data: Input data array.
            
        Returns:
            Normalized data array.
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val == min_val:
            return np.full_like(data, 0.5, dtype=np.float32)
        
        low, high = self.normalize_range
        normalized = (data - min_val) / (max_val - min_val) * (high - low) + low
        return normalized.astype(np.float32)
    
    def _process_mask(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Process mask by thresholding.
        
        Args:
            mask: Input mask array.
            threshold: Threshold value for binarization.
            
        Returns:
            Binarized mask array.
        """
        return (mask > threshold).astype(np.float32)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample.
            
        Returns:
            Dictionary containing:
                - 'input': Low-field (0.23T) image tensor.
                - 'target': High-field (3T) image tensor.
                - 'mask': Anatomical mask tensor.
        """
        input_path, target_path, mask_path = self.pair_files[index]
        
        # Load images
        input_img = self._load_nifti(input_path)
        target_img = self._load_nifti(target_path)
        mask_img = self._load_nifti(mask_path)
        
        # Normalize if requested
        if self.normalize:
            input_img = self._normalize(input_img)
            target_img = self._normalize(target_img)
            mask_img = self._normalize(mask_img)
        
        # Process mask
        mask_img = self._process_mask(mask_img)
        
        # Apply transforms
        if self.transform is not None:
            input_img = self.transform(input_img)
            mask_img = self.transform(mask_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        
        return {
            'input': input_img,
            'target': target_img,
            'mask': mask_img
        }
    
    def get_pair_info(self, index: int) -> Dict[str, str]:
        """
        Get file path information for a given index.
        
        Args:
            index: Index of the sample.
            
        Returns:
            Dictionary with file paths.
        """
        input_path, target_path, mask_path = self.pair_files[index]
        return {
            'input_path': input_path,
            'target_path': target_path,
            'mask_path': mask_path
        }