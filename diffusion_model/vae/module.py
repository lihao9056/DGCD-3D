import importlib.util
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution

# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

# TODO: Use MONAI's optional_import
# from monai.utils import optional_import
# xformers, has_xformers = optional_import("xformers.ops", name="xformers")

__all__ = ["AutoencoderKL"]


class Upsample(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
       
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"up::{x.shape}")
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        
        if x.shape == torch.Size([1,64,7,128,128]):
            x = F.interpolate(x, (13,256,256), mode="trilinear",align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=(1,2,2), mode="trilinear",align_corners=True)

        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        # self.pad = (0, 1) * spatial_dims

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=(2,2,2),
            kernel_size=(3,3,1),
            padding=(1,1,0),
            conv_only=True,
        )
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=(1,2,2),
            kernel_size=(1,3,3),
            padding=(0,1,1),
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = nn.functional.pad(x, (0,1,0,1), mode="constant", value=0.0)
        # print(f"xxx:{x.shape}")
        if  x.size(2) == 13:
            x = self.conv(x)
        else:
            x = self.conv1(x)
        # print(f"d_x:{x.shape}")
        return x


class ResBlock(nn.Module):

    def __init__(
        self, spatial_dims: int, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
  
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        # self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        # self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual


class Encoder(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels = num_channels#32,64,64
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks #1
        self.norm_num_groups = norm_num_groups #32
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        output_channel = num_channels[0]
        for i in range(len(num_channels)):  #num_channels = (32,64,128)
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(self.num_res_blocks[i]):
        
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:  #(False, False, True)
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(Downsample(spatial_dims=spatial_dims, in_channels=input_channel))

        # Non-local attention block
        blocks.append(
            ResBlock(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                out_channels=num_channels[-1],
            )
        )
        downsample = Downsample(spatial_dims=spatial_dims, in_channels=input_channel)
        blocks.append(
            downsample
        )

        blocks.append(
            AttentionBlock(
                spatial_dims=spatial_dims,
                num_channels=num_channels[-1],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_flash_attention=use_flash_attention,
            )
        )
        blocks.append(
            ResBlock(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                out_channels=num_channels[-1],
            )
        )
      

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        slice = [4,5]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in slice:
                features.append(x)
            
        return features


class Decoder(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],  # (False, False, True)
        
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        blocks.append(
            ResBlock(
                spatial_dims=spatial_dims,
                in_channels=reversed_block_out_channels[0],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                out_channels=reversed_block_out_channels[0],
            )
        )
        blocks.append(
            AttentionBlock(
                spatial_dims=spatial_dims,
                num_channels=reversed_block_out_channels[0],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_flash_attention=False,
            )
        )
        blocks.append(
            ResBlock(
                spatial_dims=spatial_dims,
                in_channels=reversed_block_out_channels[0],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                out_channels=reversed_block_out_channels[0],
            )
        )
        upsample = Upsample(spatial_dims=spatial_dims, in_channels=128)
        blocks.append(
            upsample
        )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=False,
                        )
                    )

            if not is_final_block:
                blocks.append(
                    Upsample(spatial_dims=spatial_dims, in_channels=block_in_ch)
                )

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # count=0
        for block in self.blocks:
            x = block(x)
            
            # print(f"decoder:{x.shape}")
        # x = F.interpolate(x, size=(13, 256, 256), mode='trilinear', align_corners=True)
        # print(f"1111decoder:{x.shape}")
        #     if x.shape == torch.Size([1,32,512,512,12]):
        #         count +=1
        #         if count ==2:
        #             x = F.interpolate(x,(512,512,13),mode='trilinear')
        
        return x