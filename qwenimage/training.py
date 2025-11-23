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
from qwenimage.sources import StyleSourceWithRandomRef, StyleImagetoImageSource
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
    dp = WandDataPipe()
    dp.set_task(TextToImageWithRefTask())
    dp_test = WandDataPipe()
    dp_test.set_task(TextToImageWithRefTask())
    if config.source_type == "naive":
        src = StyleSourceWithRandomRef(
            config.data_dir, config.prompt, config.ref_dir, set_len=config.max_train_steps
        )
        dp.add_source(src)
    elif config.source_type == "im2im":
        src = StyleImagetoImageSource(
            csv_path=config.csv_path,
            base_dir=config.base_dir,
            style_title=config.style_title,
            data_range=config.train_range,
        )
        dp.add_source(src)
        src_test = StyleImagetoImageSource(
            csv_path=config.csv_path,
            base_dir=config.base_dir,
            style_title=config.style_title,
            data_range=config.test_range,
        )
        dp_test.add_source(src_test)
    else: 
        raise ValueError()


    # Model
    foundation = QwenImageFoundation(config=config)
    finetuner = QwenLoraFinetuner(foundation, config)
    finetuner.load(None)


    if len(dp_test) == 0:
        dp_test = None
    if config.val_with == "train":
        dp_val = dp
    elif config.val_with == "test":
        dp_val = dp_test
    else:
        raise ValueError()
    trainer = ExperimentTrainer(
        model=foundation,
        datapipe=dp,
        args=config,
        validation_datapipe=dp_val,
        test_datapipe=dp_test,
    )
    trainer.train()




