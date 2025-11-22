


from diffusers.image_processor import PipelineImageInput
from pydantic import BaseModel, ConfigDict, Field
import torch

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # extra="allow",
    )


class QwenConfig(ExperimentTrainerParameters):
    load_multi_view_lora: bool = False
    train_max_sequence_length: int = 512
    train_dist: str = "linear" # "logit-normal"
    train_shift: bool = True
    inference_dist: str = "linear"
    inference_shift: bool = True
    static_mu: float | None = None
    loss_weight_dist: str | None = None # "scaled_clipped_gaussian", "logit-normal"

    vae_image_size: int = 1024 * 1024
