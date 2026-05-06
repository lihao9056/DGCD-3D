#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Encoder modules for DGCD-3D.

Provides lightweight 3D encoders for condition images and anatomical masks.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block with group normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        return self.activation(residual + out)


class Lightweight3DEncoder(nn.Module):
    """
    Lightweight 3D encoder for condition images.
    
    Extracts multi-scale features with progressive downsampling.
    Output features: [F1, F2, F3, F4] at different resolutions.
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            ResidualBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)),
            ResidualBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)),
            ResidualBlock(base_channels * 2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU()
        )
        
        self.down3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ResidualBlock(base_channels * 2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, C, D, H, W).
            
        Returns:
            List of feature tensors at different scales.
        """
        features = []
        
        x1 = self.init_conv(x)
        features.append(x1)
        
        x2 = self.down1(x1)
        features.append(x2)
        
        x3 = self.down2(x2)
        features.append(x3)
        
        x4 = self.down3(x3)
        features.append(x4)
        
        return features


class Mask3DEncoder(nn.Module):
    """
    Lightweight 3D encoder for anatomical masks.
    
    Extracts multi-scale features with fewer levels than condition encoder.
    Output features: [F1, F2] at different resolutions.
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels),
        )
        
        # Downsampling block
        self.down1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(base_channels),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, C, D, H, W).
            
        Returns:
            List of feature tensors at different scales.
        """
        features = []
        
        x1 = self.init_conv(x)
        features.append(x1)
        
        x2 = self.down1(x1)
        features.append(x2)
        
        return features