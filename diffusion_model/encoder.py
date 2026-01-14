import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Lightweight3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()        
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1,3,3), padding=(0,1,1)),
            ResidualBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=(3,1,1), stride=(2,2,2),padding=(1,0,0)),
            ResidualBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0)),
            ResidualBlock(base_channels*2),
            nn.GroupNorm(8, base_channels*2),
            nn.GELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(
            in_channels=128,          
            out_channels=128,        
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2), 
            padding=(0, 0, 0)
        ),
        ResidualBlock(base_channels*2),
        nn.GroupNorm(8, 128),       
        nn.GELU()
    )
 

    def forward(self, x):
        x1 = self.init_conv(x) 
        # print(f"x1:{x1.shape}")
        
        x2 = self.down1(x1) 
        
        
        #print(f"x2:{x2.shape}")
        x3 = self.down2(x2)     
        x4 = self.down3(x3) 
        # to_png(x4)
        # print(f"x1:{x1.shape}he{x2.shape}")
        # print(f"x3:{x3.shape}he{x4.shape}")
        
        return [x1,x2,x3, x4]

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        
        a = self.block(x)
        # print(f"block:{x.shape}he{a.shape}")
        return self.activation(x + a)



# class Mask3DEncoder(nn.Module):
#     def __init__(self, in_channels=1, base_channels=32):
#         super().__init__()        
#         self.init_conv = nn.Sequential(
#             nn.Conv3d(in_channels, base_channels, kernel_size=(1,3,3), padding=(0,1,1)),
#             ResidualBlock(base_channels),
#             nn.GroupNorm(8, base_channels),
#             nn.GELU()
#         )
        
#         self.down1 = nn.Sequential(
#             nn.Conv3d(base_channels, base_channels, kernel_size=(3,2,2), stride=(2,2,2),padding=(1,0,0)),
#             ResidualBlock(base_channels),
#             nn.GroupNorm(8, base_channels),
#             nn.GELU()
#         )
        
#         self.down2 = nn.Sequential(
#             nn.Conv3d(base_channels, base_channels*2, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0)),
#             ResidualBlock(base_channels*2),
#             nn.GroupNorm(8, base_channels*2),
#             nn.GELU()
#         )
#     #     self.down3 = nn.Sequential(
#     #         nn.Conv3d(
#     #         in_channels=128,         
#     #         out_channels=128,        
#     #         kernel_size=(1, 2, 2), 
#     #         stride=(1, 2, 2), 
#     #         padding=(0, 0, 0)
#     #     ),
#     #     ResidualBlock(base_channels*2),
#     #     nn.GroupNorm(8, 128),        
#     #     nn.GELU()
#     # )
 

#     def forward(self, x):
#         x1 = self.init_conv(x) 
#         # print(f"x1:{x1.shape}")
#         x2 = self.down1(x1) 
#         # print(f"x2:{x2.shape}")
#         # x3 = self.down2(x2)     
#         # x4 = self.down3(x3) 
#         # print(f"x3:{x3.shape}he{x4.shape}")
        
        return [x1, x2]

# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=8):
#         super().__init__()
#         self.channel_att = ChannelAttention(channels, reduction)
#         self.spatial_att = SpatialAttention()
    
#     def forward(self, x):
#         x = self.channel_att(x) * x
#         x = self.spatial_att(x) * x
#         return x

class Mask3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels),
            # CBAM(base_channels)
        )
        
        self.down1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(base_channels),
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            # CBAM(base_channels*2)
        )

    def forward(self, x):
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        # print(f"xx2:{x2.shape}")
        # to_png(x2)
        return [x1, x2] 

def to_png(x):
    feature_2d = x[0,:,5,:,:]
    selected_channel = feature_2d.mean(dim=0)
    print(f"se:{selected_channel.min()}")
    depth_slice_np = selected_channel.detach().cpu().numpy()
    depth_slice_np = np.flip(depth_slice_np, axis=0) 

    min_val = np.min(depth_slice_np)
    max_val = np.max(depth_slice_np)
    if max_val != min_val:
        normalized = (depth_slice_np - min_val) / (max_val - min_val) * 255
    else:
        normalized = np.zeros_like(depth_slice_np, dtype=np.uint8)
    
    normalized = normalized.astype(np.uint8)

    img = Image.fromarray(normalized)
    img.save("feature_slice.png")

    

