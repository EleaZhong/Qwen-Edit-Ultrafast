import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional
import uuid
import sys
sys.path.append(str(Path(__file__).parent.parent))

import fal
import modal

from wandml.utils.storage import download_http, upload_gcs
from wandml.services.fal.utils import get_commit_hash, get_requirements, install_wandml
from wandml import WandAuth

from qwenimage.training import run_training
REQUIREMENTS_PATH = os.path.abspath("requirements.txt")
WAND_REQUIREMENTS_PATH = os.path.abspath("scripts/wand_requirements.txt")

local_modules = ["qwenimage","wandml","scripts"]

## Fal zone
@fal.function(
    machine_type="GPU-H100",
    requirements=get_requirements(REQUIREMENTS_PATH, WAND_REQUIREMENTS_PATH),
    local_python_modules = local_modules,
    max_concurrency=16,
    request_timeout=6*60*60,
)
def run_training_on_fal(**kwargs):
    install_wandml(commit_hash=kwargs["commit_hash"])
    cfg_dest = Path("/tmp") / kwargs["yaml_file_url"].split("/")[-1]
    cfg_downloaded = download_http(kwargs["yaml_file_url"], cfg_dest)
    if cfg_downloaded is None:
        raise RuntimeError("Failed to download training config file")
    config_path = cfg_dest
    update_paths = []
    if "update_yaml_file_urls" in kwargs and kwargs["update_yaml_file_urls"] is not None:
        for idx, url in enumerate(kwargs["update_yaml_file_urls"]):
            upd_dest = Path("/tmp") / f"update_{idx}_{url.split('/')[-1]}"
            upd_downloaded = download_http(url, upd_dest)
            if upd_downloaded is None:
                raise RuntimeError(f"Failed to download update config file {url}")
            update_paths.append(upd_dest)
    return run_training(config_path, update_config_paths=update_paths if update_paths else None)
## End Fal zone


## Modal zone


modalapp = modal.App("next-stroke")
modalapp.image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install_from_requirements(REQUIREMENTS_PATH)
    .pip_install_from_requirements(WAND_REQUIREMENTS_PATH)
    .add_local_python_source(*local_modules)
)


@modalapp.function(
    gpu="B200",
    max_containers=1,
    timeout=4 * 60 * 60,
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
def run_training_on_modal(yaml_file_url: str, update_yaml_file_urls: Optional[list[str]] = None):
    config_path = Path("/tmp")/yaml_file_url.split("/")[-1]
    download_http(yaml_file_url, config_path)
    update_paths = []
    if update_yaml_file_urls is not None:
        for idx, url in enumerate(update_yaml_file_urls):
            update_path = Path("/tmp")/f"update_{idx}_{url.split('/')[-1]}"
            download_http(url, update_path)
            update_paths.append(update_path)
    return run_training(config_path, update_config_paths=update_paths if update_paths else None)

@modalapp.local_entrypoint()
def run_modal_local(yaml: str, update: Optional[str] = None):
    WandAuth(ignore=True)
    if not yaml.startswith("http"):
        yamlp = Path(yaml)
        name = yamlp.stem + str(uuid.uuid4())[:8] + yamlp.suffix
        yaml_file_url: str = upload_gcs(yaml, "wand-finetune", name, public=True)  # pyright: ignore
    else:
        yaml_file_url = yaml
    update_urls: Optional[list[str]] = None
    if update is not None and len(update) > 0:
        update_list = update.split("|")
        update_urls = []
        for upd in update_list:
            if not upd.startswith("http"):
                up = Path(upd)
                uname = up.stem + str(uuid.uuid4())[:8] + up.suffix
                update_url = upload_gcs(str(up), "wand-finetune", uname, public=True)  # pyright: ignore
                if update_url is None:
                    raise RuntimeError(f"Failed to upload {upd} to GCS")
                update_urls.append(update_url)
            else:
                update_urls.append(upd)
    return run_training_on_modal.remote(yaml_file_url, update_yaml_file_urls=update_urls) 

## End modal zone


def parse_args():
    parser = argparse.ArgumentParser(description="Run training.")
    parser.add_argument("config",type=str,help="Path or Url to YAML configuration file")
    parser.add_argument("--update", type=str, action="append", help="Optional secondary YAML with overrides (path or URL). Can be specified multiple times.")
    parser.add_argument("--where", choices=["local", "fal", "modal"])
    parser.add_argument("-d", "--detached", action="store_true", default=False, help="Run Modal in detached mode (-d). Only valid when --where modal")
    args = parser.parse_args()
    if args.detached and args.where != "modal":
        parser.error("--detached is only valid when --where modal")
    if args.where == "local" and args.config.startswith("http"):
        local_path = Path("/tmp") / args.config.split("/")[-1]
        download_http(args.config, local_path)
        args.config = local_path
    elif args.where != "local" and not args.config.startswith("http"):
        yamlp = Path(args.config)
        name = yamlp.stem + str(uuid.uuid4())[:8] + yamlp.suffix
        yaml_file_url = upload_gcs(args.config, "wand-finetune", name, public=True)
        if yaml_file_url is None:
            raise RuntimeError(f"Failed to upload {args.config} to GCS")
        args.config = yaml_file_url
    # Handle update paths/urls depending on where
    if args.update is not None and len(args.update) > 0:
        processed_updates = []
        for upd in args.update:
            if args.where == "local" and upd.startswith("http"):
                up_local_path = Path("/tmp") / upd.split("/")[-1]
                download_http(upd, up_local_path)
                processed_updates.append(up_local_path)
            elif args.where != "local" and not upd.startswith("http"):
                up = Path(upd)
                uname = up.stem + str(uuid.uuid4())[:8] + up.suffix
                up_url = upload_gcs(str(up), "wand-finetune", uname, public=True)
                if up_url is None:
                    raise RuntimeError(f"Failed to upload {upd} to GCS")
                processed_updates.append(up_url)
            else:
                processed_updates.append(upd)
        args.update = processed_updates
    return args

if __name__ == "__main__":
    WandAuth()

    args = parse_args()

    if args.where == "fal":
        out = run_training_on_fal(
            yaml_file_url=args.config,
            commit_hash=get_commit_hash(),
            update_yaml_file_urls=args.update,
        )
    elif args.where == "modal":
        cmd = ["modal", "run"]
        if args.detached:
            cmd.append("-d")
        cmd += [os.path.abspath(__file__), "--yaml", args.config]
        if args.update is not None and len(args.update) > 0:
            update_str = "|".join(str(upd) for upd in args.update)
            cmd += ["--update", update_str]
        out = subprocess.run(cmd)
    elif args.where == "local":
        update_paths = [Path(u) for u in args.update] if args.update is not None and len(args.update) > 0 else None
        out = run_training(args.config, update_config_paths=update_paths)
    else:
        raise ValueError()
    print(out)
