import os
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import fal
import modal
import torch
import tqdm
from datasets import concatenate_datasets, load_dataset, interleave_datasets

from qwenimage.datamodels import QwenConfig
from qwenimage.foundation import QwenImageFoundationSaveInterm

REQUIREMENTS_PATH = os.path.abspath("requirements.txt")
WAND_REQUIREMENTS_PATH = os.path.abspath("scripts/wand_requirements.txt")

local_modules = ["qwenimage","wandml","scripts"]



EDIT_TYPES = [
    "color",
    "style",
    "replace",
    "remove",
    "add",
    "motion change",
    "background change",
]

modalapp = modal.App("next-stroke")
modalapp.image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install_from_requirements(REQUIREMENTS_PATH)
    .pip_install_from_requirements(WAND_REQUIREMENTS_PATH)
    .add_local_python_source(*local_modules)
)

@modalapp.function(
    gpu="H100",
    max_containers=8,
    timeout=1 * 60 * 60,
    volumes={
        "/data/wand_cache": modal.Volume.from_name("FLUX_MODELS"),
        "/data/checkpoints": modal.Volume.from_name("training_checkpoints", create_if_missing=True),
        "/root/.cache/torch/hub/checkpoints": modal.Volume.from_name("torch_hub_checkpoints", create_if_missing=True),

        "/root/.cache/huggingface/hub":  modal.Volume.from_name("hf_cache", create_if_missing=True),
        "/root/.cache/huggingface/datasets":  modal.Volume.from_name("hf_cache_datasets", create_if_missing=True),

        "/data/regression_data": modal.Volume.from_name("regression_data"),
        "/data/edit_data": modal.Volume.from_name("edit_data"),
    },
    secrets=[
        modal.Secret.from_name("wand-modal-gcloud-keyfile"),
        modal.Secret.from_name("elea-huggingface-secret"),
    ],
)
def generate_regression_data(start_index=0, end_index=None, imsize=1024, indir="/data/edit_data/CrispEdit", outdir="/data/regression_data/regression_output_1024", total_per=10):

    all_edit_datasets = []
    for edit_type in EDIT_TYPES:
        to_concat = []
        for ds_n in range(total_per):
            ds = load_dataset("parquet", data_files=f"{indir}/{edit_type}_{ds_n:05d}.parquet", split="train")
            to_concat.append(ds)
        edit_type_concat = concatenate_datasets(to_concat)
        all_edit_datasets.append(edit_type_concat)
    join_ds = interleave_datasets(all_edit_datasets)

    save_base_dir = Path(outdir)
    save_base_dir.mkdir(exist_ok=True, parents=True)

    foundation = QwenImageFoundationSaveInterm(QwenConfig(vae_image_size=imsize * imsize))

    if end_index is None:
        end_index = len(join_ds)
    dataset_to_process = join_ds.select(range(start_index, end_index))
    
    for idx, input_data in enumerate(tqdm.tqdm(dataset_to_process), start=start_index):

        output_dict = foundation.base_pipe(foundation.INPUT_MODEL(
            image=[input_data["input_img"]],
            prompt=input_data["instruction"],
            vae_image_override=imsize * imsize,
            latent_size_override=imsize * imsize,
        ))

        torch.save(output_dict, save_base_dir/f"{idx:06d}.pt")


@modalapp.local_entrypoint()
def main(start:int, end:int, num_workers:int):
    per_worker_load = (end - start) // num_workers
    remainder = (end - start) % num_workers
    if remainder > 0:
        per_worker_load += 1
    worker_load_starts = []
    worker_load_ends = []
    cur_start = start
    for worker_idx in range(num_workers):
        if worker_idx < num_workers -1:
            cur_end = cur_start + per_worker_load
        else:
            cur_end = end # pass last worker less
        worker_load_starts.append(cur_start)
        worker_load_ends.append(cur_end)
        cur_start += per_worker_load

    print(f"loads: {list(zip(worker_load_starts, worker_load_ends))}")
    outputs = list(generate_regression_data.map(worker_load_starts, worker_load_ends))
    print(outputs)

    
    

