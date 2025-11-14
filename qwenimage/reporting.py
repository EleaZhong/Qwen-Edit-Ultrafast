import math
from pathlib import Path
from collections import defaultdict
import statistics

from pydantic import BaseModel
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import lpips
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

from qwenimage.experiment import ExperimentConfig
from qwenimage.experiments.experiments_qwen import ExperimentRegistry


class ExperimentSet(BaseModel):
    original: str
    comparisons: list[str]

    @classmethod
    def create(cls, *names):
        if len(names)<2:
            raise ValueError(f"{len(names)=}")
        orig = names[0]
        comp = names[1:]
        return cls(original=orig, comparisons=comp)

class SetData:
    def __init__(self, name: str):
        self.name=name
        report_dir = ExperimentConfig().report_dir
        output_dir = report_dir / f"{name}_outputs"
        self.image_paths = sorted(list(output_dir.glob("*.jpg")))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, ind):
        return Image.open(self.image_paths[ind])


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
