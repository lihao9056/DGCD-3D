#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image

import torchio as tio
import numpy as np
import torch
import re
import os
from sklearn.preprocessing import MinMaxScaler


class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            mask_folder:str,
            input_size: int,
            depth_size: int,
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            # full_channel_mask=False,
            combine_output=False,
          
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.mask_folder = mask_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        # self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
       

    def pair_file(self):
      
        input_files = sorted(glob(os.path.join(self.input_folder, '*.nii.gz')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.nii.gz')))
        mask_files = sorted(glob(os.path.join(self.mask_folder, '*.nii.gz')))

        input_dict = {re.match(r"(.*)_0.23T.nii.gz",os.path.basename(f)).group(1):f for f in input_files if re.match(r"(.*)_0.23T.nii.gz",os.path.basename(f))}
        target_dict = {re.match(r"(.*)_3T.nii.gz",os.path.basename(f)).group(1):f for f in target_files if re.match(r"(.*)_3T.nii.gz",os.path.basename(f))}
        mask_dict = {re.match(r"(.*)_0.23T.nii.gz",os.path.basename(f)).group(1):f for f in mask_files if re.match(r"(.*)_0.23T.nii.gz",os.path.basename(f))}

        common_keys = set(input_dict.keys()) & set(target_dict.keys()) & set(mask_dict.keys())
        pairs = [(input_dict[k],target_dict[k],mask_dict[k]) for k in common_keys]
 
        return pairs
 
    def sample_conditions(self, batch_size: int):
        index = np.random.randint(0, len(self))  
        input_file = self.pair_files[index][0]
        
        input_img = self.read_image(input_file)
        
        if self.transform is not None:
            input_img = self.transform(input_img).unsqueeze(0)  
        
        return input_img.cuda()

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            img = self.percentile_normalize(img) # 0 -> 1 scale
        return img

    def percentile_normalize(self, data, low=0, high=1):
        min_val = np.min(data)
        max_val = np.max(data)
    
        if max_val == min_val:
            return np.full_like(data, 0.5)
        normalization = (data - min_val) / (max_val - min_val) * (high - low) + low
        
        return normalization
        
    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file, mask_file = self.pair_files[index]
        
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)
        mask_img = self.read_image(mask_file)
        mask_img = np.where(mask_img > 0.5, 1, 0)
        

        # with open('min_val.txt','a') as f:
        #     f.write(f"{input_img.min()}he{target_img.min()}he{mask_img.min()}\n")
        
        if self.transform is not None:
            input_img = self.transform(input_img)
            mask_img = self.transform(mask_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
            
        if self.combine_output:
            return torch.cat([target_img, input_img], 0)
 
        return {'input':input_img, 'target':target_img, 'mask':mask_img}
