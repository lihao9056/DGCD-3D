#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
=================================================
@FileName : loss_util.py
@IDE      : PyCharm
@Author   : SS(sammeanshaw@qq.com)
@Date     : 2025/7/22 15:36
=================================================
"""
import functools
from torch.nn import functional as F

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        
        return loss
    elif reduction_enum == 1:
        print(f"------333------")
        return loss.mean()
    else:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss