import argparse
from pathlib import Path

import torch
import tqdm
from datasets import concatenate_datasets, load_dataset, interleave_datasets

from qwenimage.datamodels import QwenConfig
from qwenimage.foundation import QwenImageFoundationSaveInterm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

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

    save_base_dir = Path("/data/regression_output")
    save_base_dir.mkdir(exist_ok=True, parents=True)

    foundation = QwenImageFoundationSaveInterm(QwenConfig())

    dataset_to_process = join_ds.select(range(args.start_index, len(join_ds)))
    
    for idx, input_data in enumerate(tqdm.tqdm(dataset_to_process), start=args.start_index):

        output_dict = foundation.base_pipe(foundation.INPUT_MODEL(
            image=[input_data["input_img"]],
            prompt=input_data["instruction"],
        ))

        torch.save(output_dict, save_base_dir/f"{idx:06d}.pt")


if __name__ == "__main__":
    main()
