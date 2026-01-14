from monai.networks.nets import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


class UNETPerceptualLoss(nn.Module):
    def __init__(self, 
                 spatial_dims: int = 3,
                 feature_layers: Optional[List[str]] = None,
                 weights: Optional[List[float]] = None):
    
        super().__init__()
        self.feature_extractor = UNet(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(1, 2, 2, 2),
        ).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_layers = feature_layers or [
            "model.0",                 
            "model.1.submodule.0",      
            "model.1.submodule.1.submodule.0",
        ]
        self.weights = weights or [0.2, 1, 0.5]
        
        self.features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(layer_name):
            def hook(module, input, output):
                self.features[layer_name] = output 
            return hook
        
        for layer_name in self.feature_layers:
            parts = layer_name.split('.')
            module = self.feature_extractor
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            module.register_forward_hook(get_hook(layer_name))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        
        pad_d = (8 - target.shape[2] % 8) % 8
        if pad_d > 0:
            padding = (0, 0, 0, 0, 0, pad_d)
            target = F.pad(target, padding)
            output = F.pad(output, padding)

        with torch.no_grad():
            self.features.clear()
            _ = self.feature_extractor(target)
            target_features = {k: v.detach() for k, v in self.features.items()}
        
        self.features.clear()
        _ = self.feature_extractor(output)
        output_features = self.features.copy()
        
        loss = 0.0
        for layer, weight in zip(self.feature_layers, self.weights):
            loss += weight * F.l1_loss(
                output_features[layer], 
                target_features[layer]
            )
        
        return loss