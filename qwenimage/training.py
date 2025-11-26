import os
import subprocess
from pathlib import Path
import argparse
import warnings

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
from qwenimage.sources import EditingSource, RegressionSource, StyleSourceWithRandomRef, StyleImagetoImageSource
from qwenimage.task import RegressionTask, TextToImageWithRefTask
from qwenimage.datamodels import QwenConfig
from qwenimage.foundation import QwenImageFoundation, QwenImageRegressionFoundation


wandml_utils.debug.DEBUG = True

def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base

def styles_datapipe(config):
    dp = WandDataPipe()
    dp.set_task(TextToImageWithRefTask())
    dp_test = WandDataPipe()
    dp_test.set_task(TextToImageWithRefTask())
    if config.training_type == "naive":
        src = StyleSourceWithRandomRef(
            config.style_data_dir,
            config.naive_static_prompt,
            config.style_ref_dir,
            set_len=config.max_train_steps,
        )
        dp.add_source(src)
    elif config.training_type == "im2im":
        src = StyleImagetoImageSource(
            csv_path=config.style_csv_path,
            base_dir=config.style_base_dir,
            style_title=config.style_title,
            data_range=config.train_range,
        )
        dp.add_source(src)
        src_test = StyleImagetoImageSource(
            csv_path=config.style_csv_path,
            base_dir=config.style_base_dir,
            style_title=config.style_title,
            data_range=config.test_range,
        )
        dp_test.add_source(src_test)
    else: 
        raise ValueError()
    if config.style_val_with == "train":
        dp_val = dp
    elif config.style_val_with == "test":
        dp_val = dp_test
    else:
        raise ValueError()
    return dp, dp_val, dp_test

def regression_datapipe(config):
    dp = WandDataPipe()
    dp.set_task(RegressionTask())
    dp_val = WandDataPipe()
    dp_val.set_task(RegressionTask())
    dp_test = WandDataPipe()
    dp_test.set_task(TextToImageWithRefTask())

    src_train = RegressionSource(
        data_dir=config.regression_data_dir,
        gen_steps=config.regression_gen_steps,
        data_range=config.train_range,
    )
    dp.add_source(src_train)

    src_val = RegressionSource(
        data_dir=config.regression_data_dir,
        gen_steps=config.regression_gen_steps,
        data_range=config.val_range,
    )
    dp_val.add_source(src_val)

    src_test = EditingSource(
        data_dir=config.editing_data_dir,
        total_per=config.editing_total_per,
        data_range=config.test_range,
    )
    dp_test.add_source(src_test)
    
    return dp, dp_val, dp_test

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
    config.add_suffix_to_names()

    # Data
    if config.training_type.is_style:
        dp, dp_val, dp_test = styles_datapipe(config)
    elif config.training_type == "regression":
        dp, dp_val, dp_test = regression_datapipe(config)
    else:
        raise ValueError(f"Got {config.training_type=}")

    # Model
    if config.training_type.is_style:
        foundation = QwenImageFoundation(config=config)
    elif config.training_type == "regression":
        foundation = QwenImageRegressionFoundation(config=config)
    else:
        raise ValueError(f"Got {config.training_type=}")
    finetuner = QwenLoraFinetuner(foundation, config)
    finetuner.load(config.resume_from_checkpoint, config.lora_rank)

    if len(dp_test) == 0:
        warnings.warn("Test datapipe is removed for being len=0")
        dp_test = None
    if len(dp_val) == 0:
        warnings.warn("Validation datapipe is removed for being len=0")
        dp_val = None
    trainer = ExperimentTrainer(
        model=foundation,
        datapipe=dp,
        args=config,
        validation_datapipe=dp_val,
        test_datapipe=dp_test,
    )
    trainer.train()




