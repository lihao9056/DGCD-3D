#-*- coding:utf-8 -*-
#
#
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .modules import *
from PIL import Image
# from .Adal_N import AdaIN3D

from .encoder import Lightweight3DEncoder,Mask3DEncoder

NUM_CLASSES = 1

class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
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
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.encoder = Lightweight3DEncoder(in_channels=1)
        self.maskencoder = Mask3DEncoder(in_channels=1)

        # self.hr_branch1 = HighFrequencyBranch3D(32,64)
        # self.hr_branch2 = HighFrequencyBranch3D(64,128)
        # self.hr_branch3 = HighFrequencyBranch3D(128,256)
        # self.Downsample_fre1 = Downsample_fre(64,64)
        # self.Downsample_fre2 = Downsample_fre(128,128)
        # self.Downsample_fre3 = Downsample_fre(256,256)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

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

#-----------------------————————————————————————————————————————————————————————————————————

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
#--------------------------------------------------------------------------------------------
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

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        
        self.conv1 = nn.Conv3d(
            in_channels=96,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
           )
        self.conv2 = nn.Conv3d(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
           )
        self.conv3 = nn.Conv3d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
           )
        self.conv4 = nn.Conv3d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
           )
        self.conv5 = nn.Conv3d(
            in_channels=160,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
           )
        self.conv8 = nn.Conv3d(
            in_channels=192,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
           )
        self.conv9 = nn.Conv3d(
            in_channels=192,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
           )


    def convert_to_fp16(self):
     
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):

        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x,timesteps, c, mask, y=None):
 
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        # hs_fre = []
        # Image.fromarray((c[0,0,5,:,:]*255).cpu().numpy().astype(np.uint8)).save('./1.png')
        # Image.fromarray((mask[0,0,5,:,:]*255).cpu().numpy().astype(np.uint8)).save('./mask.png')

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        # print(f"ccc:{c.shape}")
        
        cond_input = self.encoder(c)
        mask_input = self.maskencoder(mask)
        # cond_input[0] = torch.cat([cond_input[0],mask],dim=1)
        # cond_input[1] = torch.cat([cond_input[1],mask],dim=1)
        # cond_input[2] = torch.cat([cond_input[2],mask],dim=1)
        # cond_input[3] = torch.cat([cond_input[3],mask],dim=1)
        
        # print(f"nihaonihao:{mask_input[0].shape}")
        # print(f"nihaonihao:{mask_input[1].shape}")
        # print(f"nihaonihao_cond:{cond_input[0].shape}")
        # print(f"nihaonihao_cond:{cond_input[1].shape}")
        count=0
    
        for module in self.input_blocks:
            # print(f"{h.shape}")

            h = module(h, emb)
            
            if h.shape == th.Size([1,64,13,256,256]):
                count+=1
                if count == 2:
                    # print("5")
                    h = th.cat([h, mask_input[0]], dim=1)
                    h = self.conv1(h)
                    
                    # h_fre = self.hr_branch1(cond_input[0])
                    # h_fre_down = self.Downsample_fre1(h_fre)
                    # hs_fre.append(h_fre_down)
            if h.shape == th.Size([1,64,7,128,128]):
                count+=1
                if count == 5:
                    # print("6")
                    h = th.cat([h, mask_input[1]], dim=1)
                    h = self.conv2(h)
                    # h_fre = self.hr_branch2(cond_input[1])
                    # h_fre_down = self.Downsample_fre2(h_fre)
                    # hs_fre.append(h_fre_down)

            if h.shape == th.Size([1,128,7,64,64]):
                count+=1
                if count == 8:
                    # print("7")
                    h = th.cat([h, cond_input[2]], dim=1)
                    h = self.conv3(h)
                    # h_fre = self.hr_branch3(cond_input[2])
                    # h_fre_down = self.Downsample_fre3(h_fre)
                    # hs_fre.append(h_fre_down)

            if h.shape == th.Size([1,128,7,32,32]):
                count+=1
                if count == 11:
                    # print("8")
                    h = th.cat([h, cond_input[3]], dim=1)
                    h = self.conv4(h)
                    # h_fre = self.hr_branch4(cond_input[2])
            # print(f"hhh:{h.shape}")

            hs.append(h)
                
        # for i in range(len(hs)):
        #     print(f"{hs[i].shape}")
        # for i in range(len(hs_fre)):
        #     print(f"-----{hs_fre[i].shape}")

        h1 = hs[-1]    
        h = self.middle_block(h, emb)
        
        # print(f'midd_h = {h.shape}')
        count1 = 0
        for module in self.output_blocks:
            
            h = th.cat([h, hs.pop()], dim=1)

            # print(f"----h.{h.shape}")
            
            if h.shape == th.Size([2,128,7,32,32]):
                # print("1")
                h = th.cat([h, mask_input[1]], dim=1)
                h = self.conv(h)
                # h = th.cat([h, hs_fre[2]], dim=1)
                
            if h.shape == th.Size([1,384,4,128,128]):
                # print("2")
                h = th.cat([h, cond_input[2]], dim=1)
                # h = th.cat([h, hs_fre[1]], dim=1)
                
            if h.shape == th.Size([2,192,7,256,256]):
                # print("3")
                # h = th.cat([h, cond_input[1]], dim=1)
                h = th.cat([h, mask_input[1]], dim=1)
                # h = th.cat([h, hs_fre[0]], dim=1)
                
            if h.shape == th.Size([1,128,7,128,128]):
                
                count1+=1
                if count1 == 2:
                    # print("4")
                    h = th.cat([h, cond_input[1]], dim=1)
                    h = self.conv8(h)
            if h.shape == th.Size([1,128,13,256,256]):
                
                count1+=1
                if count1 == 4:
                    # print("5")
                    h = th.cat([h, cond_input[0]], dim=1)
                    h = self.conv9(h)
            
            h = module(h, emb)
            # print(f'out_h = {h.shape}')
            if h.shape == th.Size([2,128,7,128,128]):
                # print(f"6")
                h = th.cat([h, mask_input[0]], dim=1)
                h = self.conv5(h)
                    
            # print(f"——output:{h.shape}")
        h = h.type(x.dtype)
        # print(f"---ouyput3.shape:{h.shape}")
        # return self.out(h),h1
        return self.out(h)


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=2,
    out_channels=1,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4,4)
            channel_mult = (1,1,2,4,8)
        elif image_size == 256:
            # channel_mult = (1, 2, 2, 4)
            channel_mult = (1, 1, 2, 2,4)
        elif image_size == 192:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    # print(f"attention_ds:{attention_ds}")
    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
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
