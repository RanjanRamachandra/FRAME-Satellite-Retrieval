
#!/bin/bash
set -euo pipefail

BASE="s3://noaa-himawari8/AHI-L1b-FLDK/2022/01"
DAYS=("16" "17" "18" "19" "20" "21")
TIMES=("0000" "0300" "0600" "0900" "1200" "1500" "1800" "2100")

mkdir -p data/raw/multiday

for DAY in "${DAYS[@]}"; do
    for TIME in "${TIMES[@]}"; do

        DATE="202201${DAY}"
        DEST="data/raw/multiday/${DATE}_${TIME}"

        echo "Checking ${DATE} ${TIME}"

        mkdir -p "$DEST"

        aws s3 cp "${BASE}/${DAY}/${TIME}/" "$DEST/" \
            --no-sign-request \
            --recursive \
            --exclude "*" \
            --include "*_B07_*" \
            --include "*_B08_*" \
            --include "*_B13_*" \
            || echo "WARNING: Download issue for ${DATE} ${TIME}"

        COUNT=$(find "$DEST" -name "*.bz2" | wc -l | tr -d ' ')

        echo "${DATE} ${TIME}: ${COUNT} files downloaded"

    done
done

echo "Download process completed."

