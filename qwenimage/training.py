import os
import subprocess
from pathlib import Path
import argparse

import yaml
import diffusers


from wandml.trainers.experiment_trainer import ExperimentTrainer
from wandml import WandDataPipe
import wandml
from wandml import WandAuth
from wandml import utils as wandml_utils
from wandml.trainers.datamodels import ExperimentTrainerParameters
from wandml.trainers.experiment_trainer import ExperimentTrainer


from qwenimage.finetuner import QwenLoraFinetuner
from qwenimage.datasets import StyleSourceWithRandomRef
from qwenimage.task import TextToImageWithRefTask
from qwenimage.datamodels import QwenConfig
from qwenimage.foundation import QwenImageFoundation


wandml_utils.debug.DEBUG = True

def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base

def run_training(config_path: Path | str, update_config_paths: list[Path] | None = None):
    WandAuth(ignore=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if update_config_paths is not None:
        for update_config_path in update_config_paths:
            with open(update_config_path, "r") as uf:
                update_cfg = yaml.safe_load(uf)
            if isinstance(update_cfg, dict):
                config = _deep_update(config if isinstance(config, dict) else {}, update_cfg)

    config = QwenConfig(
        **config,
    )

    # Data
    src = StyleSourceWithRandomRef("/data/styles-finetune-data-artistic/tarot", "<0001>", "/data/image", set_len=1000)
    task = TextToImageWithRefTask()
    dp = WandDataPipe()
    dp.add_source(src)
    dp.set_task(task)


    # Model
    foundation = QwenImageFoundation(config=config)
    finetuner = QwenLoraFinetuner(foundation, config)
    finetuner.load(None)


    trainer = ExperimentTrainer(foundation,dp,config)
    trainer.train()




