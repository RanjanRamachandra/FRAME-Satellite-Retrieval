#!/bin/bash

set -e

BASE="s3://noaa-himawari8/AHI-L1b-FLDK/2022/01/15"

TIMES=(
  "0000"
  "0300"
  "0600"
  "0900"
  "1200"
  "1500"
  "1800"
  "2100"
)

mkdir -p data/raw/multiday

for TIME in "${TIMES[@]}"
do
    echo "Downloading observation: $TIME"

    mkdir -p "data/raw/multiday/$TIME"

    aws s3 cp \
      "$BASE/$TIME/" \
      "data/raw/multiday/$TIME/" \
      --no-sign-request \
      --recursive \
      --exclude "*" \
      --include "*_B07_*" \
      --include "*_B08_*" \
      --include "*_B13_*"

    echo "Completed: $TIME"
done

echo "All downloads completed!"
