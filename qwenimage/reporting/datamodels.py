from pydantic import BaseModel
from PIL import Image

from qwenimage.experiment import ExperimentConfig


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
