#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D UNet architecture for DGCD-3D diffusion model.

Implements a conditional 3D UNet with:
- Multi-scale encoder for condition images
- Multi-scale encoder for anatomical masks
- Attention mechanisms at multiple resolutions
- Timestep embedding for diffusion process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .modules import (
    ResBlock, AttentionBlock, TimestepEmbedSequential,
    normalization, timestep_embedding, conv_nd, linear, zero_module
)
from .encoder import Lightweight3DEncoder, Mask3DEncoder

NUM_CLASSES = 1


class UNetModel(nn.Module):
    """
    3D Conditional UNet for diffusion-based image enhancement.
    
    Architecture:
    - Encoders for condition image and mask
    - Input blocks (downsampling path)
    - Middle block (bottleneck with attention)
    - Output blocks (upsampling path with skip connections)
    """
    
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 3,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        
        # Encoders for condition and mask
        self.condition_encoder = Lightweight3DEncoder(in_channels=1)
        self.mask_encoder = Mask3DEncoder(in_channels=1)
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # Input blocks (downsampling path)
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, 64, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        # Middle block (bottleneck)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        
        # Output blocks (upsampling path)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        
        # Output layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        
        # Feature fusion convolutions
        self._setup_feature_fusion()
    
    def _setup_feature_fusion(self):
        """Setup convolution layers for feature fusion."""
        # Fusion layers for different feature map sizes
        self.fusion_convs = nn.ModuleDict({
            '64_13': nn.Conv3d(96, 64, kernel_size=1),      # (B, 96, 13, 256, 256) -> (B, 64, 13, 256, 256)
            '64_7': nn.Conv3d(128, 64, kernel_size=1),       # (B, 128, 7, 128, 128) -> (B, 64, 7, 128, 128)
            '128_7': nn.Conv3d(256, 128, kernel_size=1),     # (B, 256, 7, 64, 64) -> (B, 128, 7, 64, 64)
            '128_7_v2': nn.Conv3d(256, 128, kernel_size=1),  # (B, 256, 7, 32, 32) -> (B, 128, 7, 32, 32)
            '160': nn.Conv3d(160, 128, kernel_size=1),       # (B, 160, 7, 128, 128) -> (B, 128, 7, 128, 128)
            '192': nn.Conv3d(192, 128, kernel_size=1),       # (B, 192, 7, 256, 256) -> (B, 128, 7, 256, 256)
        })
    
    def convert_to_fp16(self):
        """Convert model to FP16 precision."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """Convert model to FP32 precision."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the UNet.
        
        Args:
            x: Input noisy image tensor (B, C, D, H, W).
            timesteps: Timestep indices (B,).
            condition: Condition image (low-field) (B, C, D, H, W).
            mask: Anatomical mask (B, C, D, H, W).
            y: Class labels (B,) - optional.
            
        Returns:
            Output tensor (B, C, D, H, W).
        """
        assert (y is not None) == (self.num_classes is not None), \
            "Must specify y if and only if model is class-conditional"
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        # Encode condition and mask
        cond_features = self.condition_encoder(condition)  # [c1, c2, c3, c4]
        mask_features = self.mask_encoder(mask)            # [m1, m2]
        
        # Process through input blocks
        hs = []
        h = x.type(self.dtype)
        
        for module in self.input_blocks:
            h = module(h, emb)
            h = self._apply_feature_fusion(h, cond_features, mask_features)
            hs.append(h)
        
        # Middle block
        h = self.middle_block(h, emb)
        
        # Process through output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        h = h.type(x.dtype)
        return self.out(h)
    
    def _apply_feature_fusion(
        self,
        h: torch.Tensor,
        cond_features: list,
        mask_features: list
    ) -> torch.Tensor:
        """
        Apply feature fusion based on feature map size.
        
        Args:
            h: Current feature map.
            cond_features: Condition encoder features.
            mask_features: Mask encoder features.
            
        Returns:
            Fused feature map.
        """
        B, C, D, H, W = h.shape
        shape_key = f"{C}_{D}" if D != H else f"{C}"
        
        # Apply fusion based on feature map characteristics
        if D == 13 and W == 256 and C == 64:
            # Early stage - fuse mask features
            h = torch.cat([h, mask_features[0]], dim=1)
            h = self.fusion_convs['64_13'](h)
        elif D == 7 and W == 128 and C == 64:
            # Second stage - fuse mask features
            h = torch.cat([h, mask_features[1]], dim=1)
            h = self.fusion_convs['64_7'](h)
        elif D == 7 and W == 64 and C == 128:
            # Third stage - fuse condition features
            h = torch.cat([h, cond_features[2]], dim=1)
            h = self.fusion_convs['128_7'](h)
        elif D == 7 and W == 32 and C == 128:
            # Fourth stage - fuse condition features
            h = torch.cat([h, cond_features[3]], dim=1)
            h = self.fusion_convs['128_7_v2'](h)
        elif D == 7 and W == 128 and C == 160:
            # Upsampling stage
            h = torch.cat([h, cond_features[1]], dim=1)
            h = self.fusion_convs['160'](h)
        elif D == 13 and W == 256 and C == 128:
            # Final upsampling stage
            h = torch.cat([h, cond_features[0]], dim=1)
            h = self.fusion_convs['192'](h)
        elif D == 7 and W == 256 and C == 192:
            # Alternative path
            h = torch.cat([h, mask_features[1]], dim=1)
            h = self.fusion_convs['192'](h)
        
        return h


class Upsample(nn.Module):
    """3D upsampling module with shape-aware interpolation."""
    
    def __init__(self, channels: int, use_conv: bool, dims: int = 3, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample the input tensor.
        
        Args:
            x: Input tensor (B, C, D, H, W).
            
        Returns:
            Upsampled tensor.
        """
        assert x.shape[1] == self.channels
        
        if self.dims == 3:
            # Shape-aware upsampling
            D, H, W = x.shape[-3:]
            new_D, new_H, new_W = self._get_upsampled_shape(D, H, W)
            x = F.interpolate(x, size=(new_D, new_H, new_W), mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        if self.use_conv:
            x = self.conv(x)
        
        return x
    
    def _get_upsampled_shape(self, D: int, H: int, W: int) -> Tuple[int, int, int]:
        """Get target shape for upsampling."""
        # Double spatial dimensions, adjust depth based on spatial size
        # When spatial dimensions reach 256, also upsample depth to match encoder
        if H >= 128 and W >= 128:
            # Final upsampling stage - restore depth to 13 to match encoder
            new_D = 13
        else:
            new_D = D  # Keep depth same for earlier stages
        new_H = H * 2
        new_W = W * 2
        return new_D, new_H, new_W


class Downsample(nn.Module):
    """3D downsampling module."""
    
    def __init__(self, channels: int, use_conv: bool, dims: int = 3, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        
        if use_conv:
            self.op_stride2 = conv_nd(3, channels, out_channels, 3, stride=(2, 2, 2), padding=1)
            self.op_stride1 = conv_nd(3, channels, out_channels, 3, stride=(1, 2, 2), padding=1)
        else:
            self.op_stride2 = nn.AvgPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)
            self.op_stride1 = nn.AvgPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample the input tensor.
        
        Args:
            x: Input tensor (B, C, D, H, W).
            
        Returns:
            Downsampled tensor.
        """
        assert x.shape[1] == self.channels
        
        D, H, W = x.shape[-3:]
        if D >= 13 and H >= 256:
            # First downsampling - reduce all dimensions
            return self.op_stride2(x)
        else:
            # Subsequent downsampling - only reduce spatial dimensions
            return self.op_stride1(x)


def create_model(
    image_size: int,
    num_channels: int,
    num_res_blocks: int,
    channel_mult: str = "",
    learn_sigma: bool = False,
    class_cond: bool = False,
    use_checkpoint: bool = False,
    attention_resolutions: str = "16",
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0.0,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    in_channels: int = 2,
    out_channels: int = 1,
) -> UNetModel:
    """
    Factory function to create a UNet model.
    
    Args:
        image_size: Spatial size of input images.
        num_channels: Base number of channels.
        num_res_blocks: Number of residual blocks per level.
        channel_mult: Channel multiplier for each level.
        learn_sigma: Whether to learn variance.
        class_cond: Use class conditioning.
        use_checkpoint: Use gradient checkpointing.
        attention_resolutions: Resolutions to apply attention.
        num_heads: Number of attention heads.
        num_head_channels: Channels per attention head.
        num_heads_upsample: Number of heads for upsampling.
        use_scale_shift_norm: Use scale-shift normalization.
        resblock_updown: Use resblock for up/down sampling.
        use_fp16: Use FP16 precision.
        use_new_attention_order: Use new attention ordering.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        
    Returns:
        UNetModel instance.
    """
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 4, 8)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4)
        elif image_size == 192:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(m) for m in channel_mult.split(","))
    
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(out_channels if not learn_sigma else 2 * out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )