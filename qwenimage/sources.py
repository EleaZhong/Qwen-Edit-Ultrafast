

import csv
from pathlib import Path
import random
from typing import Literal

from PIL import Image
import torch
from datasets import concatenate_datasets, load_dataset, interleave_datasets

from qwenimage.types import DataRange
from wandml.core.datamodels import SourceDataType
from wandml.core.source import Source

def parse_datarange(dr: DataRange, length: int, return_as: Literal['list', 'range']='list'):
    if not isinstance(length, int):
        raise ValueError()
    left, right = dr
    if left is None:
        left = 0
    if right is None:
        right = length
    if (isinstance(left, float) or isinstance(right, float)) and (left<1 and right<1):
        left = left * length
        right = right * length
    if return_as=="list":
        return list(range(left, right))
    elif return_as=="range":
        return range(left, right)
    else:
        raise ValueError()


class StyleSource(Source):
    _data_types = [
        SourceDataType(name="image", type=Image.Image),
        SourceDataType(name="text", type=str),
    ]
    def __init__(self, data_dir, prompt, set_len=None):
        data_dir = Path(data_dir)
        self.images = list(data_dir.iterdir())
        self.prompt = prompt
        self.set_len = set_len
    
    def __len__(self):
        if self.set_len is not None:
            return self.set_len
        else:
            return len(self.images)
    
    def __getitem__(self, idx):
        idx = idx % len(self.images)
        im_pil = Image.open(self.images[idx]).convert("RGB")
        return im_pil, self.prompt

class StyleSourceWithRandomRef(Source):
    _data_types = [
        SourceDataType(name="image", type=Image.Image),
        SourceDataType(name="text", type=str),
        SourceDataType(name="reference", type=Image.Image),
    ]
    def __init__(self, data_dir, prompt, ref_dir, set_len=None):
        data_dir = Path(data_dir)
        self.images = list(data_dir.iterdir())
        self.ref_images = list(Path(ref_dir).iterdir())
        self.prompt = prompt
        self.set_len = set_len
    
    def __len__(self):
        if self.set_len is not None:
            return self.set_len
        else:
            return len(self.images)
    
    def __getitem__(self, idx):
        idx = idx % len(self.images)
        im_pil = Image.open(self.images[idx]).convert("RGB")
        rand_ref = random.choice(self.ref_images)
        ref_pil = Image.open(rand_ref).convert("RGB")
        return im_pil, self.prompt, ref_pil


class StyleImagetoImageSource(Source):
    _data_types = [
        SourceDataType(name="text", type=str),
        SourceDataType(name="image", type=Image.Image),
        SourceDataType(name="reference", type=Image.Image),
    ]
    def __init__(self, csv_path, base_dir, style_title=None, data_range:DataRange|None=None):
        self.csv_path = Path(csv_path)
        self.base_dir = Path(base_dir)
        self.style_title = style_title
        self.data = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.style_title is not None and row['style_title'] != self.style_title:
                    continue
                
                input_image = self.base_dir / row['input_image']
                output_image = self.base_dir / row['output_image_path']
                self.data.append({
                    'input_image': input_image,
                    'output_image': output_image,
                    'style_title': row['style_title'],
                    'prompt': row['prompt']
                })
        
        if data_range is not None:
            indexes = parse_datarange(data_range, len(self.data))
            self.data = [self.data[i] for i in indexes]

        print(f"{self.__class__} of len{len(self)}")
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        input_pil = Image.open(item['input_image']).convert("RGB")
        output_pil = Image.open(item['output_image']).convert("RGB")
        return prompt, output_pil, input_pil


class RegressionSource(Source):
    _data_types = [
        SourceDataType(name="data", type=dict),
    ]

    def __init__(self, data_dir, gen_steps=50, data_range:DataRange|None=None):
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_paths = list(data_dir.glob("*.pt"))
        if data_range is not None:
            indexes = parse_datarange(data_range, len(self.data_paths))
            self.data_paths = [self.data_paths[i] for i in indexes]
        self.gen_steps = gen_steps
        self._len = gen_steps * len(self.data_paths)
        print(f"{self.__class__} of len{len(self)}")
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        data_idx = idx // self.gen_steps
        step_idx = idx % self.gen_steps
        out_dict = torch.load(self.data_paths[data_idx])
        t = out_dict.pop(f"t_{step_idx}")
        latents_start = out_dict.pop(f"latents_{step_idx}_start")
        noise_pred = out_dict.pop(f"noise_pred_{step_idx}")
        out_dict["t"] = t
        out_dict["latents_start"] = latents_start
        out_dict["noise_pred"] = noise_pred
        return [out_dict,]

        
class EditingSource(Source):
    _data_types = [
        SourceDataType(name="text", type=str),
        SourceDataType(name="image", type=Image.Image),
        SourceDataType(name="reference", type=Image.Image),
    ]
    EDIT_TYPES = [
        "color",
        "style",
        "replace",
        "remove",
        "add",
        "motion change",
        "background change",
    ]
    def __init__(self, data_dir:Path, total_per=1, data_range:DataRange|None=None):
        data_dir = Path(data_dir)
        self.join_ds = self.build_dataset(data_dir, total_per)

        if data_range is not None:
            indexes = parse_datarange(data_range, len(self.join_ds))
            self.join_ds = self.join_ds.select(indexes)

        print(f"{self.__class__} of len{len(self)}")

    def build_dataset(self, data_dir:Path, total_per:int):
        all_edit_datasets = []
        for edit_type in self.EDIT_TYPES:
            to_concat = []
            for ds_n in range(total_per):
                ds = load_dataset("parquet", data_files=str(data_dir/f"{edit_type}_{ds_n:05d}.parquet"), split="train")
                to_concat.append(ds)
            edit_type_concat = concatenate_datasets(to_concat)
            all_edit_datasets.append(edit_type_concat)
        # consistent ordering for indexing, also allow extension by increasing total_per
        join_ds = interleave_datasets(all_edit_datasets)
        return join_ds
    
    def __len__(self):
        return len(self.join_ds)
    
    def __getitem__(self, idx):
        data = self.join_ds[idx]
        reference = data["input_img"]
        image = data["output_img"]
        text = data["instruction"]
        return text, image, reference
