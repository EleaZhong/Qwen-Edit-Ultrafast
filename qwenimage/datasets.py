

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
