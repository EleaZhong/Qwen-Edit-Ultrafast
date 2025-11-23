#!/bin/bash
# Usage: ./download_parts.sh -d <dataset> -o <output_dir> -p <num_processes>
# Datasets: hqedit (1-100), omniedit (1-50), ultraedit (1-4)

set -e

# Defaults
OUTPUT_DIR="./GPT-Image-Edit-1.5M/gpt-edit"
NUM_PROC=1
DATASET=""

while getopts "d:o:p:" opt; do
  case $opt in
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    p) NUM_PROC="$OPTARG" ;;
    *) echo "Usage: $0 -d <dataset> -o <output_dir> -p <num_processes>"; exit 1 ;;
  esac
done

if [ -z "$DATASET" ]; then
  echo "Error: dataset must be specified with -d (hqedit | ultraedit | omniedit)"
  exit 1
fi

# Select dataset config
case "$DATASET" in
  hqedit)
    BASE_URL="https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M/resolve/main/gpt-edit/hqedit.tar.gz.part"
    RANGE=$(seq -w 001 100)
    ;;
  omniedit)
    BASE_URL="https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M/resolve/main/gpt-edit/omniedit.tar.gz.part"
    RANGE=$(seq -w 001 175)
    ;;
  ultraedit)
    BASE_URL="https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M/resolve/main/gpt-edit/ultraedit.tar.gz.part"
    RANGE=$(seq -w 001 004)
    ;;
  *)
    echo "Error: invalid dataset '$DATASET'. Choose from: hqedit, omniedit, ultraedit"
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR/$DATASET"

# Download in parallel with resume support
echo "Downloading $DATASET into $OUTPUT_DIR/$DATASET with $NUM_PROC parallel jobs..."
echo "$RANGE" | parallel --lb -j "$NUM_PROC" \
  "wget --progress=bar:force -c '${BASE_URL}{}?download=true' -O '${OUTPUT_DIR}/${DATASET}/${DATASET}.tar.gz.part{}'"

echo "Download completed for $DATASET."
echo "To merge and extract, run:"
echo "cat ${OUTPUT_DIR}/${DATASET}/${DATASET}.tar.gz.part* > ${OUTPUT_DIR}/${DATASET}/${DATASET}.tar.gz"
echo "tar -xzvf ${OUTPUT_DIR}/${DATASET}/${DATASET}.tar.gz -C ${OUTPUT_DIR}/${DATASET}"