#!/bin/bash

set -euo pipefail

BASE="s3://noaa-himawari8/AHI-L1b-FLDK/2022/01"
OUTPUT="data/expanded_2022/raw"

TIMES=("0000" "0300" "0600" "0900" "1200" "1500" "1800" "2100")

for DAY in $(seq -w 1 28); do

    DATE="202201${DAY}"

    if [[ "$DAY" -ge 15 && "$DAY" -le 21 ]]; then
        echo "Skipping ${DATE} - existing processed observations will be reused"
        continue
    fi

    for TIME in "${TIMES[@]}"; do

        echo "Downloading: ${DATE} ${TIME}"

        mkdir -p "${OUTPUT}/${DATE}_${TIME}"

        aws s3 cp \
            "${BASE}/${DAY}/${TIME}/" \
            "${OUTPUT}/${DATE}_${TIME}/" \
            --no-sign-request \
            --recursive \
            --exclude "*" \
            --include "*_B07_*" \
            --include "*_B08_*" \
            --include "*_B13_*"

        echo "Completed: ${DATE} ${TIME}"

    done
done

echo "Expanded download completed!"
