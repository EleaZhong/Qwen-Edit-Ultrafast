from PIL import Image
import torch
from torchvision.transforms import v2 as T

from wandml.core.datamodels import SourceDataType
from wandml.core.task import Task
from wandml.data.transforms import RemoveAlphaTransform, RandomDownsize


image_transforms = T.Compose([
    RemoveAlphaTransform(bg_color_rgb=(34, 34, 34)),
    T.ToImage(),
    T.RGB(),
    RandomDownsize(sizes=(384, 512, 768)),
    T.ToDtype(torch.float, scale=True),
])

class TextToImageTask(Task):
    data_types = [
        SourceDataType(name="text", type=str),
        SourceDataType(name="image", type=Image.Image),
    ]
    type_transforms = [
        Task.identity,
        image_transforms,
    ]
    sample_input_dict = {
        "prompt": SourceDataType(name="text", type=str),
    }


class TextToImageWithRefTask(Task):
    data_types = [
        SourceDataType(name="text", type=str),
        SourceDataType(name="image", type=Image.Image),
        SourceDataType(name="reference", type=Image.Image),
    ]
    type_transforms = [
        Task.identity,
        image_transforms,
        image_transforms,
    ]
    sample_input_dict = {
        "prompt": SourceDataType(name="text", type=str),
        "image": SourceDataType(name="reference", type=Image.Image),
    }
