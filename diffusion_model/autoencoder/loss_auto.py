import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

class PerceptualLoss(nn.Module):
    def __init__(self, vae_model, return_feature_only=False):
        super().__init__()
        self.encoder = vae_model.encoder
        self.return_feature_only = return_feature_only
        self.freeze()


    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
    def extract_features(self, x, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.encoder(x)
        else:
            return self.encoder(x)

    def forward(self, output, target):

        feat_out = self.extract_features(output, no_grad=False)
        feat_tgt = self.extract_features(target, no_grad=True)
            
        loss = 0.0
        for f_out, f_tgt in zip(feat_out, feat_tgt):
            loss += F.l1_loss(f_out, f_tgt)
    
        return loss



