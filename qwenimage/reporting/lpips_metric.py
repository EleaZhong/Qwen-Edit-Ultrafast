import math
from pathlib import Path
from collections import defaultdict
import statistics
from typing import Literal

from pydantic import BaseModel
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import lpips
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


_transforms = T.Compose([
    T.ToImage(),
    T.RGB(),
    T.ToDtype(torch.float32, scale=True), # [0,1]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1,1]
])

def compare_lpips(loss_fn, image1, image2, resize=False, device="cuda", to_item=True):
    if image1.size != image2.size:
        if resize:
            image2 = image2.resize(image1.size, Image.LANCZOS)
        else:
            raise ValueError(f"Got mismatch {image1.size=} {image2.size=}")

    im1_t = _transforms(image1).unsqueeze(0).to(device=device)
    im2_t = _transforms(image2).unsqueeze(0).to(device=device)
    
    with torch.no_grad():
        score = loss_fn(im1_t, im2_t)
    
    if to_item:
        return score.float().item()
    return score

class LpipsCalculator:
    def __init__(self, resize=False, device="cuda", to_item=True):
        self.resize = resize
        self.to_item = to_item
        self.loss_fn = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.loss_fn = self.loss_fn.to(device=self.device)
    
    def __call__(self, image1, image2, resize=None, to_item=None):
        if resize is None:
            resize = self.resize
        if to_item is None:
            to_item = self.to_item
        return compare_lpips(self.loss_fn, image1, image2, resize=resize, device=self.device, to_item=to_item)
