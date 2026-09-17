#!/usr/bin/env bash
set -euo pipefail

CORPUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/corpus"
mkdir -p "$CORPUS_DIR"

# 1. Silesia Compression Corpus (12 files, ~212 MB)
SILESIA_FILES=("dickens" "mozilla" "mr" "nci" "ooffice" "osdb" "reymont" "samba" "sao" "webster" "xml" "x-ray")
MISSING_SILESIA=0
for f in "${SILESIA_FILES[@]}"; do
    if [ ! -f "$CORPUS_DIR/$f" ]; then
        MISSING_SILESIA=1
        break
    fi
done

if [ "$MISSING_SILESIA" -eq 1 ]; then
    echo "Downloading Silesia Compression Corpus..."
    ZIP_FILE=$(mktemp /tmp/silesia.XXXXXX.zip)
    curl -sSL "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip" -o "$ZIP_FILE"
    echo "Extracting Silesia corpus files into $CORPUS_DIR..."
    unzip -q -o "$ZIP_FILE" -d "$CORPUS_DIR"
    rm -f "$ZIP_FILE"
    echo "Silesia corpus downloaded successfully."
else
    echo "Silesia corpus files already present."
fi

# 2. enwik8 Holdout Corpus (100 MB)
if [ ! -f "$CORPUS_DIR/enwik8" ]; then
    echo "Downloading enwik8 holdout corpus (100 MB)..."
    ENWIK_ZIP=$(mktemp /tmp/enwik8.XXXXXX.zip)
    curl -sSL "http://mattmahoney.net/dc/enwik8.zip" -o "$ENWIK_ZIP"
    echo "Extracting enwik8 into $CORPUS_DIR..."
    unzip -q -o "$ENWIK_ZIP" -d "$CORPUS_DIR"
    rm -f "$ENWIK_ZIP"
    echo "enwik8 downloaded successfully ($(stat -c%s "$CORPUS_DIR/enwik8") bytes)."
else
    echo "enwik8 already present."
fi

