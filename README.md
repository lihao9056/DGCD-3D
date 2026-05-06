# DGCD-3D: Difference-Guided Conditional Diffusion Model for Low-Field 3D MRI Enhancement to Assist Stroke Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DGCD-3D is a **difference-guided conditional 3D diffusion model** for enhancing low-field (0.23T) DWI (Diffusion-Weighted Imaging) MRI scans to high-field (3T) quality. The model leverages anatomical masks as additional guidance to preserve stroke structural details during the enhancement process.

## Note

This code repository is released to support reproducibility during the peer-review process.
The code and documentation will be further cleaned and improved after the paper is accepted.

## Data Preparation

### Directory Structure

Organize your data as follows:

```
dataset/
├── data/
│   ├── 0.23T_DWI/           # Low-field images
│   │   ├── subject001_0.23T.nii.gz
│   │   ├── subject002_0.23T.nii.gz
│   │   └── ...
│   ├── 3T_DWI/              # High-field images (ground truth)
│   │   ├── subject001_3T.nii.gz
│   │   ├── subject002_3T.nii.gz
│   │   └── ...
│   └── mask_train_0.23T/    # Anatomical masks
│       ├── subject001_0.23T.nii.gz
│       ├── subject002_0.23T.nii.gz
│       └── ...
```

### File Naming Convention

- **Low-field images**: `{subject_id}_0.23T.nii.gz`
- **High-field images**: `{subject_id}_3T.nii.gz`
- **Masks**: `{subject_id}_0.23T.nii.gz`

The dataset loader automatically pairs files based on matching `{subject_id}`.

## Training

### Basic Training (Single GPU)

```bash
python train.py \
    -i ./dataset/data/0.23T_DWI \
    -t ./dataset/data/3T_DWI \
    -m ./dataset/data/mask_train_0.23T \
    --input_size 256 \
    --depth_size 13 \
    --batch_size 1 \
    --epochs 800 \
    --save_pt_dir ./checkpoints/dgcd-3d
```

### Distributed Training (Multi-GPU)

```bash
# 2 GPUs
python -m torch.distributed.launch --nproc_per_node=2 train.py \
    -i ./dataset/data/0.23T_DWI \
    -t ./dataset/data/3T_DWI \
    -m ./dataset/data/mask_train_0.23T \
    --batch_size 1 \
    --epochs 800

# 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 train.py \
    ...
```
