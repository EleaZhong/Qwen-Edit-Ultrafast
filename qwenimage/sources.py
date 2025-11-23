

import csv
from pathlib import Path
import random

from PIL import Image
from wandml.core.datamodels import SourceDataType
from wandml.core.source import Source

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
    def __init__(self, csv_path, base_dir, style_title=None, data_range:tuple[int|float,int|float]|None=None):
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
            left, right = data_range
            if (isinstance(left, float) or isinstance(right, float)) and (left<1 and right<1):
                left = left * len(self.data)
                right = right * len(self.data)
            remain_data = []
            for i, d in enumerate(self.data):
                if left <= i and i < right:
                    remain_data.append(d)
            self.data = remain_data
        
        print(f"{self.__class__} of len{len(self)}")
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        input_pil = Image.open(item['input_image']).convert("RGB")
        output_pil = Image.open(item['output_image']).convert("RGB")
        return prompt, output_pil, input_pil