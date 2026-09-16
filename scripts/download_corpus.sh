#!/usr/bin/env bash
set -euo pipefail

# Download and extract the official Silesia Compression Corpus (211.9 MB)
CORPUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/corpus"

if [ -d "$CORPUS_DIR" ] && [ "$(ls -A "$CORPUS_DIR" 2>/dev/null)" ]; then
    echo "Silesia corpus already exists in $CORPUS_DIR"
    exit 0
fi

mkdir -p "$CORPUS_DIR"
echo "Downloading Silesia Compression Corpus..."
ZIP_FILE=$(mktemp /tmp/silesia.XXXXXX.zip)

curl -sSL "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip" -o "$ZIP_FILE"
echo "Extracting corpus files into $CORPUS_DIR..."
unzip -q -o "$ZIP_FILE" -d "$CORPUS_DIR"
rm -f "$ZIP_FILE"

echo "Silesia corpus downloaded and verified successfully ($(ls -1 "$CORPUS_DIR" | wc -l) files)."
