#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core building blocks for 3D UNet architecture.

This module provides fundamental components used in the diffusion model:
- ResBlock: Residual block with timestep conditioning
- AttentionBlock: Self-attention mechanism
- Upsample/Downsample: Spatial resolution changes
- Utility functions for convolutions, normalization, and timestep embeddings
"""

from abc import abstractmethod
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Gradient Checkpointing
# ============================================================================

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations.
    
    This allows for reduced memory at the expense of extra compute in the backward pass.
    
    Args:
        func: The function to evaluate.
        inputs: The argument sequence to pass to func.
        params: A sequence of parameters func depends on but does not explicitly take.
        flag: If False, disable gradient checkpointing.
    
    Returns:
        The result of calling func with the given inputs.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    """Custom autograd function for gradient checkpointing."""
    
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    
    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


# ============================================================================
# Utility Functions
# ============================================================================

def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D convolution module.
    
    Args:
        dims: Number of spatial dimensions (1, 2, or 3).
        *args: Positional arguments for the convolution layer.
        **kwargs: Keyword arguments for the convolution layer.
    
    Returns:
        Appropriate convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Module:
    """Create a linear (fully connected) module."""
    return nn.Linear(*args, **kwargs)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module.
    
    This is useful for initializing residual connections to have no effect initially.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels: int) -> nn.Module:
    """
    Create a group normalization layer.
    
    Args:
        channels: Number of input channels.
    
    Returns:
        GroupNorm module with 32 groups.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 1000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: A 1-D tensor of N indices, one per batch element.
        dim: The dimension of the output embedding.
        max_period: Controls the minimum frequency of the embeddings.
    
    Returns:
        An [N x dim] tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================================
# Basic Modules
# ============================================================================

class GroupNorm32(nn.GroupNorm):
    """Group normalization with float32 casting for stability."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class TimestepBlock(nn.Module):
    """
    Abstract base class for modules that take timestep embeddings as input.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the module to x given timestep embedding emb.
        
        Args:
            x: Input tensor of shape [N, C, ...].
            emb: Timestep embedding tensor of shape [N, emb_channels].
        
        Returns:
            Output tensor of shape [N, C, ...].
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to children that support it.
    """
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ============================================================================
# Residual Block
# ============================================================================

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    
    The block consists of:
    - Input layers (norm -> silu -> conv)
    - Optional up/downsampling
    - Timestep embedding conditioning
    - Output layers (norm -> silu -> dropout -> conv with zero init)
    - Skip connection
    """
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 3,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Input layers
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        
        # Upsampling/downsampling
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        # Timestep embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        # Output layers
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        
        # Skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to x, conditioned on timestep embedding emb.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    
    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # Input path
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # Timestep conditioning
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # Output path
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        # Skip connection
        return self.skip_connection(x) + h


# ============================================================================
# Attention Block
# ============================================================================

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    
    Uses QKV attention with layer normalization.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, \
                f"channels {channels} not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# ============================================================================
# QKV Attention Variants
# ============================================================================

class QKVAttentionLegacy(nn.Module):
    """
    QKV attention that splits heads before splitting q, k, v.
    """
    
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    QKV attention that splits q, k, v before splitting heads.
    """
    
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


# ============================================================================
# Upsample and Downsample
# ============================================================================

class Upsample(nn.Module):
    """
    3D upsampling module.
    
    Uses trilinear interpolation for 3D data.
    """
    
    def __init__(self, channels: int, use_conv: bool, dims: int = 3, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        
        if self.dims == 3:
            # Shape-aware 3D upsampling
            D, H, W = x.shape[-3:]
            new_D = D if D >= 7 else D * 2
            new_H = H * 2
            new_W = W * 2
            x = F.interpolate(x, size=(new_D, new_H, new_W), mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    3D downsampling module.
    
    Uses strided convolution or average pooling.
    """
    
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
        assert x.shape[1] == self.channels
        
        D, H, W = x.shape[-3:]
        if D >= 13 and H >= 256:
            return self.op_stride2(x)
        else:
            return self.op_stride1(x)