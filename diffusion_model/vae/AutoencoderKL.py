# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
from .module import Encoder, Decoder


class AutoencoderKL(nn.Module):
   
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = 1,
        num_channels: Sequence[int] = (32, 64, 128),
        attention_levels: Sequence[bool] = (False, False, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("AutoencoderKL expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects num_channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
        )
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=128,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
                  
        )
    
    


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        reconstruction = self.decoder(h)
        return reconstruction


