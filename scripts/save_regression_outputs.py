# 
# %cd /home/ubuntu/Qwen-Image-Edit-Angles

import torch
import huggingface_hub 
import tqdm

from qwenimage.datamodels import QwenConfig
from qwenimage.foundation import QwenImageFoundationSaveInterm
from datasets import concatenate_datasets, load_dataset, interleave_datasets

repo_tree = huggingface_hub.list_repo_tree(
    "WeiChow/CrispEdit-2M",
    "data",
    repo_type="dataset",
)

all_paths = []
for i in repo_tree:
    all_paths.append(i.path)

parquet_prefixes = set()
for path in all_paths:
    if path.endswith('.parquet'):
        filename = path.split('/')[-1]
        if '_' in filename:
            prefix = filename.split('_')[0]
            parquet_prefixes.add(prefix)

print("Found parquet prefixes:", sorted(parquet_prefixes))



total_per = 10

EDIT_TYPES = [
    "color",
    "style",
    "replace",
    "remove",
    "add",
    "motion change",
    "background change",
]




all_edit_datasets = []
for edit_type in EDIT_TYPES:
    to_concat = []
    for ds_n in range(total_per):
        ds = load_dataset("parquet", data_files=f"/data/CrispEdit/{edit_type}_{ds_n:05d}.parquet", split="train")
        to_concat.append(ds)
    edit_type_concat = concatenate_datasets(to_concat)
    all_edit_datasets.append(edit_type_concat)

# consistent ordering for indexing, also allow extension
join_ds = interleave_datasets(all_edit_datasets)








from pathlib import Path


save_base_dir = Path("/data/regression_output")
save_base_dir.mkdir(exist_ok=True, parents=True)





foundation = QwenImageFoundationSaveInterm(QwenConfig())





for idx, input_data in enumerate(tqdm.tqdm(join_ds)):

    output_dict = foundation.base_pipe(foundation.INPUT_MODEL(
        image=[input_data["input_img"]],
        prompt=input_data["instruction"],
    ))

    torch.save(output_dict, save_base_dir/f"{idx:06d}.pt")



