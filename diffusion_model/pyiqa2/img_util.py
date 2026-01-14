#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
=================================================
@FileName : img_util.py
@IDE      : PyCharm
@Author   : SS(sammeanshaw@qq.com)
@Date     : 2025/7/22 15:42
=================================================
"""
import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
import io

from PIL import Image
import torchvision.transforms.functional as TF

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in Image.registered_extensions())

def imread2tensor(img_source, rgb=False):
    """Read image to tensor.

    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
    """
    if type(img_source) == bytes:
        img = Image.open(io.BytesIO(img_source))
    elif type(img_source) == str:
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    img_tensor = TF.to_tensor(img)
    return img_tensor