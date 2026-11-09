#!/bin/bash

set -euo pipefail

INPUT="data/expanded_2022/raw"
OUTPUT="data/expanded_2022/processed"

for DIR in "$INPUT"/*; do

    [ -d "$DIR" ] || continue

    FOLDER=$(basename "$DIR")

    COUNT=$(find "$DIR" -maxdepth 1 -type f -name '*.bz2' | wc -l | tr -d ' ')

    if [ "$COUNT" -eq 0 ]; then
        echo "SKIP $FOLDER - no compressed files"
        continue
    fi

    echo "Processing $FOLDER"

    mkdir -p "$OUTPUT/$FOLDER"

    for SRC in "$DIR"/*.bz2; do

        [ -f "$SRC" ] || continue

        DEST="$OUTPUT/$FOLDER/$(basename "${SRC%.bz2}")"

        if [ -f "$DEST" ]; then
            continue
        fi

        bzip2 -dc "$SRC" > "$DEST"

    done

    echo "Completed $FOLDER"

done

echo "Expanded decompression completed!"
