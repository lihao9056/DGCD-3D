# DGCD-3D: Deep Generative Conditional Diffusion for 3D DWI Enhancement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DGCD-3D is a **conditional 3D diffusion model** for enhancing low-field (0.23T) DWI (Diffusion-Weighted Imaging) MRI scans to high-field (3T) quality. The model leverages anatomical masks as additional guidance to preserve structural details during the enhancement process.

### Key Features

- **Conditional Diffusion**: Uses low-field images as conditioning input for guided generation
- **Anatomical Mask Guidance**: Incorporates brain structure masks to preserve anatomical boundaries
- **3D Architecture**: Processes volumetric data directly, capturing spatial context in all dimensions
- **Perceptual Loss**: Combines pixel-level, perceptual, and adversarial losses for realistic outputs
- **Distributed Training**: Supports multi-GPU training with DDP (Distributed Data Parallel)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DGCD-3D Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input (0.23T) ──► Condition Encoder ──► Multi-scale Features   │
│                         │                                         │
│  Mask ──► Mask Encoder ──┤                                         │
│                         ▼                                         │
│              ┌───────────────────────┐                           │
│              │   3D UNet (Denoiser)  │                           │
│              │  ┌─────────────────┐  │                           │
│              │  │ Input Blocks    │  │  ◄── Timestep Embedding   │
│              │  │ (Downsampling)  │  │                           │
│              │  ├─────────────────┤  │                           │
│              │  │ Middle Block    │  │  ◄── Attention            │
│              │  ├─────────────────┤  │                           │
│              │  │ Output Blocks   │  │  ◄── Feature Fusion       │
│              │  │ (Upsampling)    │  │                           │
│              │  └─────────────────┘  │                           │
│              └───────────────────────┘                           │
│                         │                                         │
│                         ▼                                         │
│  Output (3T) ◄────── Enhanced DWI                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Description |
|-----------|-------------|
| **Condition Encoder** | Lightweight 3D encoder extracting 4-scale features from low-field images |
| **Mask Encoder** | 2-scale encoder for anatomical mask features |
| **3D UNet** | Main denoising network with attention and feature fusion |
| **Diffusion Scheduler** | Exponential noise schedule with 250 timesteps |
| **Perceptual Loss** | UNet-based feature matching loss |

## Requirements

### System Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM (recommend ≥16GB for 3D training)
- **OS**: Linux (Ubuntu 18.04+), macOS, or Windows with WSL2
- **Python**: 3.8 or higher

### Dependencies

```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
nibabel>=3.2.0
einops>=0.3.0
tqdm>=4.60.0
monai>=0.8.0
pywt>=1.0.0  # PyWavelets for DWT/IDWT

# Optional (for mixed precision training)
# apex  # Install from source: https://github.com/NVIDIA/apex

# Development
# tensorboard  # For training visualization
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dgcd-3d.git
   cd dgcd-3d
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

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
- **Masks**: `{subject_id}_0.23T.nii.gz` (same naming as low-field)

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

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input_folder` | `./dataset/data/0.23T_DWI` | Path to low-field images |
| `-t, --target_folder` | `./dataset/data/3T_DWI` | Path to high-field images |
| `-m, --mask_folder` | `./dataset/data/mask_train_0.23T` | Path to anatomical masks |
| `--input_size` | 256 | Spatial resolution (height/width) |
| `--depth_size` | 13 | Number of slices in depth |
| `--num_channels` | 64 | Base channels in UNet |
| `--num_res_blocks` | 2 | Residual blocks per level |
| `--batch_size` | 1 | Batch size per GPU |
| `--epochs` | 800 | Number of training epochs |
| `--timesteps` | 250 | Diffusion timesteps |
| `--lr` | 1e-4 | Learning rate |
| `--gradient_accumulate_every` | 2 | Gradient accumulation steps |
| `--ema_decay` | 0.995 | EMA decay rate |
| `--save_pt_dir` | `./checkpoint/dgcd-3d` | Checkpoint save directory |
| `--save_and_sample_every` | 10 | Save every N epochs |
| `--resume_path` | None | Path to resume training |

### Resuming Training

```bash
python train.py \
    -i ./dataset/data/0.23T_DWI \
    -t ./dataset/data/3T_DWI \
    -m ./dataset/data/mask_train_0.23T \
    --resume_path ./checkpoints/dgcd-3d/model/model-10.pt
```

## Inference

### Basic Inference

```python
import torch
from diffusion_model.trainer_brats import GaussianDiffusion
from diffusion_model.unet_brats import create_model

# Load model
model = create_model(
    image_size=256,
    num_channels=64,
    num_res_blocks=2,
    in_channels=1,
    out_channels=1
).cuda()

# Load weights
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# Create diffusion model
diffusion = GaussianDiffusion(
    model=model,
    image_size=256,
    depth_size=13,
    timesteps=250,
    with_condition=True
).cuda()

# Run inference
with torch.no_grad():
    output = diffusion.sample(
        condition_tensors=low_field_image,  # (B, 1, D, H, W)
        mask_tensors=anatomical_mask       # (B, 1, D, H, W)
    )
```

## Project Structure

```
DGCD-3D/
├── train.py                      # Main training script
├── dataset.py                    # Dataset loader
├── README.md                     # This file
├── requirements.txt              # Dependencies
│
├── diffusion_model/
│   ├── __init__.py
│   ├── trainer_brats.py          # Diffusion model & trainer
│   ├── unet_brats.py             # 3D UNet architecture
│   ├── modules.py                # Building blocks (ResBlock, Attention, etc.)
│   ├── encoder.py                # Condition & mask encoders
│   ├── fp16_util.py              # Mixed precision utilities
│   ├── adversarial_loss.py       # Adversarial loss (MONAI)
│   ├── patchgan_discriminator.py # PatchGAN discriminator (MONAI)
│   │
│   ├── DWT_IDWT/                 # Discrete Wavelet Transform
│   │   ├── __init__.py
│   │   ├── DWT_IDWT_layer.py     # 3D DWT/IDWT layers
│   │   └── DWT_IDWT_Functions.py # Custom autograd functions
│   │
│   ├── pyiqa2/                   # Perceptual quality assessment
│   │   ├── __init__.py
│   │   ├── inference_model.py    # LPIPS metric
│   │   ├── lpips_arch.py         # LPIPS network
│   │   └── ...
│   │
│   └── vae/                      # VAE-based perceptual loss
│       ├── __init__.py
│       ├── unetloss.py           # UNet perceptual loss
│       └── ...
│
└── utils/
    ├── __init__.py
    ├── dtypes.py                 # Data type enums
    └── lowpass_filter.py         # Low-pass filtering utilities
```

## Loss Function

The total training loss combines multiple components:

```
L_total = L_v + λ_x0 × w(t) × L_x0 + λ_lpips × w(t) × L_lpips
```

Where:
- **L_v**: V-prediction loss (diffusion objective)
- **L_x0**: Direct x_0 prediction loss (λ_x0 = 0.2)
- **L_lpips**: Perceptual loss using UNet features (λ_lpips = 0.2)
- **w(t)**: Time-dependent weight (higher for smaller timesteps)

## Technical Details

### Diffusion Process

The model uses a variance-preserving diffusion process with:

- **Noise Schedule**: Exponential eta schedule
- **Timesteps**: 250 (configurable)
- **Parameterization**: V-prediction (velocity prediction)
- **Sampling**: DDIM for faster inference

### Uncertainty-Aware Generation

The model computes an uncertainty map from the condition image and mask:

```
un = |condition × (1 - mask)|
```

This uncertainty modulates the noise level during both training and inference, allowing the model to focus generation on uncertain regions.

### Feature Fusion

Multi-scale features from the condition and mask encoders are fused into the main UNet at appropriate resolutions using 1×1×1 convolutions for channel adjustment.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dgcd3d,
  title={DGCD-3D: Deep Generative Conditional Diffusion for 3D DWI Enhancement},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/dgcd-3d}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MONAI](https://monai.io/) for medical imaging components
- [Guided Diffusion](https://github.com/openai/guided-diffusion) for diffusion model architecture
- [PyWavelets](https://pywavelets.readthedocs.io/) for wavelet transform utilities

## Contact

For questions and collaborations, please open an issue or contact the maintainers.