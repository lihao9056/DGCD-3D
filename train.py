#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGCD-3D: Deep Generative Model for Low-Field to High-Field DWI Enhancement

Training script for 3D diffusion model with conditional guidance.
Converts 0.23T DWI images to 3T DWI images using anatomical masks.
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import NiftiPairImageGenerator
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.diffusion_unet import create_model


def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}'), local_rank


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DGCD-3D model for DWI image enhancement'
    )
    
    # Data parameters
    parser.add_argument('-i', '--input_folder', type=str, 
                        default='/data/lihao/SR 数据_256_天坛/训练数据/0.23T_DWI_1',
                        help='Path to low-field (0.23T) DWI images')
    parser.add_argument('-t', '--target_folder', type=str, 
                        default='/data/lihao/SR 数据_256_天坛/训练数据/3T_DWI_1',
                        help='Path to high-field (3T) DWI images')
    parser.add_argument('-m', '--mask_folder', type=str, 
                        default='/data/lihao/SR 数据_256_天坛/训练数据/MASK',
                        help='Path to anatomical masks')
    
    # Model architecture parameters
    parser.add_argument('--input_size', type=int, default=256,
                        help='Spatial resolution (height/width)')
    parser.add_argument('--depth_size', type=int, default=13,
                        help='Number of slices in depth dimension')
    parser.add_argument('--num_channels', type=int, default=64,
                        help='Base number of channels in UNet')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='Number of residual blocks per level')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of training epochs')
    parser.add_argument('--timesteps', type=int, default=250,
                        help='Number of diffusion timesteps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gradient_accumulate_every', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='Exponential moving average decay')
    
    # Checkpoint and logging
    parser.add_argument('--save_pt_dir', type=str, 
                        default='./checkpoint/dgcd-3d',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_and_sample_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to resume training from checkpoint')
    
    # Other parameters
    parser.add_argument('--with_condition', type=str, default='True',
                        help='Use conditional generation')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup distributed training
    device, local_rank = setup_distributed()
    args.local_rank = local_rank
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # Print configuration
    if local_rank == 0:
        print("=" * 50)
        print("DGCD-3D Training Configuration")
        print("=" * 50)
        print(f"Input folder: {args.input_folder}")
        print(f"Target folder: {args.target_folder}")
        print(f"Mask folder: {args.mask_folder}")
        print(f"Image size: {args.input_size}x{args.input_size}x{args.depth_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Diffusion timesteps: {args.timesteps}")
        print("=" * 50)
    
    # Create dataset
    transform = lambda t: torch.tensor(t).float().permute(2, 0, 1).unsqueeze(0)
    
    dataset = NiftiPairImageGenerator(
        input_folder=args.input_folder,
        target_folder=args.target_folder,
        mask_folder=args.mask_folder,
        input_size=args.input_size,
        depth_size=args.depth_size,
        transform=transform,
        target_transform=transform
    )
    
    if local_rank == 0:
        print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create data loader
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )
    
    # Create model
    model = create_model(
        image_size=args.input_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        in_channels=1,
        out_channels=1
    ).to(device)
    
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    
    if local_rank == 0:
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create diffusion model
    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type='l2',
        with_condition=args.with_condition == 'True',
        channels=1
    ).cuda()
    
    # Create trainer
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        dataloader=dataloader,
        image_size=args.input_size,
        depth_size=args.depth_size,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        fp16=args.fp16,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=args.save_pt_dir,
        with_condition=args.with_condition == 'True',
        distributed=True,
        rank=local_rank,
        resume_path=args.resume_path
    )
    
    # Start training
    if local_rank == 0:
        print("Starting training...")
    
    trainer.train()
    
    if local_rank == 0:
        print("Training completed!")


if __name__ == '__main__':
    main()