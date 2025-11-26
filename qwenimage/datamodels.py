import enum
from pathlib import Path
from typing import Any, Literal

import torch
from diffusers.image_processor import PipelineImageInput
from pydantic import BaseModel, ConfigDict, Field

from qwenimage.types import DataRange
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

class TrainingType(str, enum.Enum):
    IM2IM = "im2im"
    NAIVE = "naive"
    REGRESSION = "regression"

    @property
    def is_style(self):
        return self in [TrainingType.NAIVE, TrainingType.IM2IM]

class QuantOptions(str, enum.Enum):
    INT8WO = "int8wo"
    INT4WO = "int4wo"
    FP8ROW = "fp8row"

LossTermSpecType = int|float|dict[str,int|float]|None

class QwenLossTerms(BaseModel):
    mse: LossTermSpecType = 1.0
    triplet: LossTermSpecType = 0.0
    negative_mse: LossTermSpecType = 0.0
    distribution_matching: LossTermSpecType = 0.0
    pixel_triplet: LossTermSpecType = 0.0
    pixel_lpips: LossTermSpecType = 0.0
    pixel_mse: LossTermSpecType = 0.0
    pixel_distribution_matching: LossTermSpecType = 0.0
    adversarial: LossTermSpecType = 0.0
    teacher: LossTermSpecType = 0.0

    triplet_margin: float = 0.0
    triplet_min_abs_diff: float = 0.0
    teacher_steps: int = 4

    @property
    def pixel_terms(self) -> bool:
        return ("pixel_lpips", "pixel_mse", "pixel_triplet", "pixel_distribution_matching",)

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
    vae_tiling: bool = False


    train_loss_terms:QwenLossTerms = Field(default_factory=QwenLossTerms)
    validation_loss_terms:QwenLossTerms = Field(default_factory=QwenLossTerms)

    training_type: TrainingType|None=None
    train_range: DataRange|None=None
    val_range: DataRange|None=None
    test_range: DataRange|None=None

    style_title: str|None = None
    style_base_dir: str|None = None
    style_csv_path: str|None = None
    style_data_dir: str|None = None
    style_ref_dir: str|None = None
    style_val_with: str = "train"
    naive_static_prompt: str|None = None

    regression_data_dir: str|Path|None = None
    regression_gen_steps: int = 50
    editing_data_dir: str|Path|None = None
    editing_total_per: int = 1
    regression_base_pipe_steps: int = 8

    name_suffix: dict[str,Any]|None = None

    def add_suffix_to_names(self):
        if self.name_suffix is None:
            return
        suffix_sum = ""
        for suf_name,suf_val in self.name_suffix.items():
            suffix_sum += "_" + suf_name
            suf_val = str(suf_val)
            suffix_sum += "_" + suf_val
        self.run_name += suffix_sum
        self.output_dir = self.output_dir.removesuffix("/") # in case
        self.output_dir += suffix_sum


