import enum
from typing import Literal

import torch
from diffusers.image_processor import PipelineImageInput
from pydantic import BaseModel, ConfigDict, Field

from wandml.foundation.datamodels import FluxInputs
from wandml.trainers.datamodels import ExperimentTrainerParameters


class QwenInputs(BaseModel):
    image: PipelineImageInput | None = None
    prompt: str| list[str] | None = None
    height: int|None = None
    width: int|None = None
    negative_prompt: str| list[str] | None = None
    true_cfg_scale: float = 1.0
    num_inference_steps: int = 50
    generator: torch.Generator | list[torch.Generator] | None = None
    max_sequence_length: int = 512
    vae_image_override: int | None = 512 * 512
    latent_size_override: int | None = 512 * 512

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # extra="allow",
    )


class QuantOptions(str, enum.Enum):
    INT8WO = "int8wo"
    INT4WO = "int4wo"
    FP8ROW = "fp8row"


class QwenConfig(ExperimentTrainerParameters):
    load_multi_view_lora: bool = False
    train_max_sequence_length: int = 512
    train_dist: str = "linear" # "logit-normal"
    train_shift: bool = True
    inference_dist: str = "linear"
    inference_shift: bool = True
    static_mu: float | None = None
    loss_weight_dist: str | None = None # "scaled_clipped_gaussian", "logit-normal"

    vae_image_size: int = 512 * 512
    offload_text_encoder: bool = True
    quantize_text_encoder: bool = False
    quantize_transformer: bool = False

    source_type: str = "im2im"
    style_title: str|None = None
    base_dir: str|None = None
    csv_path: str|None = None
    data_dir: str|None = None
    ref_dir: str|None = None
    prompt: str|None = None
    train_range: tuple[int|float,int|float]|None=None
    test_range: tuple[int|float,int|float]|None=None
    val_with: str = "train"

