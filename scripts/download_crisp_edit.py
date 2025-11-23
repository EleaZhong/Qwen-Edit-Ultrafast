#!/usr/bin/env python3
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Known edit types and how many parquet files exist for each
EDIT_TYPES = {
    "color": 1984,
    "motion change": 128,
    "style": 1600,
    "replace": 1566,
    "remove": 1388,
    "add": 1213,
    "background change": 1091,
}

BASE_URL = "https://huggingface.co/datasets/WeiChow/CrispEdit-2M/resolve/main/data"
MAX_WORKERS = 8  # parallel downloads


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CrispEdit-2M parquet files with wget in parallel."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of parquet files per edit type (default: all available)",
    )
    parser.add_argument(
        "-e",
        "--edit-type",
        choices=EDIT_TYPES.keys(),
        nargs="+",
        help="Edit types to download (default: all types)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to save downloaded files (default: current directory)",
    )
    return parser.parse_args()


def validate_args(args):
    if args.n is not None and args.n <= 0:
        print("Error: -n must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    selected_types = args.edit_type or list(EDIT_TYPES.keys())

    # Ensure n is not larger than available parquet count for any selected type
    if args.n is not None:
        for et in selected_types:
            max_files = EDIT_TYPES[et]
            if args.n > max_files:
                print(
                    f"Error: requested -n={args.n} but edit type '{et}' "
                    f"only has {max_files} parquet files.",
                    file=sys.stderr,
                )
                sys.exit(1)

    return selected_types


def download_file(url: str, dest: Path):
    # if dest.exists():
    #     print(f"[skip] {dest} already exists")
    #     return

    cmd = ["wget", "-O", str(dest), "--progress=bar:force", "-c", url]
    print(" ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"wget failed for {url} (exit code {result.returncode})")


def main():
    args = parse_args()
    edit_types = validate_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    # Build download jobs: one per (edit_type, index)
    for et in edit_types:
        num_files = args.n if args.n is not None else EDIT_TYPES[et]
        for idx in range(num_files):
            filename = f"{et}_{idx:05d}.parquet"
            url = f"{BASE_URL}/{filename}"
            dest = out_dir / filename
            jobs.append((url, dest))

    print(f"Downloading {len(jobs)} files for edit types: {', '.join(edit_types)}")

    # Parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(download_file, url, dest): (url, dest) for url, dest in jobs}

        for future in as_completed(future_to_job):
            url, dest = future_to_job[future]
            try:
                future.result()
            except Exception as e:
                print(f"[error] {url} -> {dest}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()