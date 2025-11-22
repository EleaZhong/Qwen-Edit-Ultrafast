# # %%
# %cd /home/ubuntu/Qwen-Image-Edit-Angles

# %%
import os
import subprocess
from pathlib import Path
import argparse

from ruamel.yaml import YAML
import diffusers


from wandml.trainers.experiment_trainer import ExperimentTrainer
from wandml import WandDataPipe
import wandml

from qwenimage.finetuner import QwenLoraFinetuner


# %%


# %%


from qwenimage.datasets import StyleSourceWithRandomRef


src = StyleSourceWithRandomRef("/data/styles-finetune-data-artistic/tarot", "<0001>", "/data/image", set_len=1000)

# %%
from qwenimage.task import TextToImageWithRefTask

task = TextToImageWithRefTask()

# %%
dp = WandDataPipe()
dp.add_source(src)
dp.set_task(task)

# %%
from qwenimage.datamodels import QwenConfig
from qwenimage.foundation import QwenImageFoundation

config = QwenConfig(
    # preprocessing_epoch_len=0,
)
foundation = QwenImageFoundation(config=config)

# %%
finetuner = QwenLoraFinetuner(foundation, config)
finetuner.load(None)

# %%
trainer = ExperimentTrainer(foundation,dp,config)

# %%
trainer.train()

# %%



