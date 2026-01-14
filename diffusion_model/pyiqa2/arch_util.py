#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
=================================================
@FileName : arch_util.py
@IDE      : PyCharm
@Author   : SS(sammeanshaw@qq.com)
@Date     : 2025/7/22 15:35
=================================================
"""
import torch
from collections import OrderedDict


def clean_state_dict(state_dict):
    """Clean checkpoint by removing .module prefix from state dict if it exists from parallel training."""
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict

def load_pretrained_network(net: torch.nn.Module,
                            model_path: str,
                            strict: bool = True,
                            weight_keys: str = None) -> None:
        """
        Load a pretrained network from a given model path.

        Args:
            net (torch.nn.Module): The network to load the weights into.
            model_path (str): Path to the model weights file. Can be a URL or a local file path.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by net's state_dict(). Default is True.
            weight_keys (str, optional): Specific key to extract from the state_dict. Default is None.

        Returns:
            None
        """
        print(f"Loading pretrained model {net.__class__.__name__} from {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        if weight_keys is not None:
            state_dict = state_dict[weight_keys]
        state_dict = clean_state_dict(state_dict)
        net.load_state_dict(state_dict, strict=strict)