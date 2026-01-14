#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
=================================================
@FileName : api_helpers.py
@IDE      : PyCharm
@Author   : SS(sammeanshaw@qq.com)
@Date     : 2025/7/22 15:38
=================================================
"""
from pyiqa2.default_model_configs import DEFAULT_CONFIGS
from pyiqa2.inference_model import InferenceModel

def create_metric(metric_name, as_loss=False, device=None, **kwargs):
    assert metric_name in DEFAULT_CONFIGS.keys(), f'Metric {metric_name} not implemented yet.'
    metric = InferenceModel(metric_name, as_loss=as_loss, device=device, **kwargs)
    return metric
