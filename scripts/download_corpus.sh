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
    # Two hosts carry it; a host that answers with a page instead of the
    # zip (it has) fails the check and the next one is tried.
    for url in "http://mattmahoney.net/dc/enwik8.zip" "https://cs.fit.edu/~mmahoney/compression/enwik8.zip"; do
        curl -fsSL --retry 3 "$url" -o "$ENWIK_ZIP" && unzip -tq "$ENWIK_ZIP" >/dev/null 2>&1 && break
        echo "enwik8: $url did not give a zip, trying the next" >&2
    done
    echo "Extracting enwik8 into $CORPUS_DIR..."
    unzip -q -o "$ENWIK_ZIP" -d "$CORPUS_DIR"
    rm -f "$ENWIK_ZIP"
    echo "enwik8 downloaded successfully ($(stat -c%s "$CORPUS_DIR/enwik8") bytes)."
else
    echo "enwik8 already present."
fi

# 3. Extended corpus: real-world formats beyond Silesia/enwik8, used by
# `examples/v7_bench.rs` for gate G2 (v7 ratio >= zstd -3 ratio per file).
# Opt-in (EXT_CORPUS=1): it is over 1 GB and CI does not need it. Each
# entry is best-effort: a failed download is reported and skipped rather
# than aborting the rest of the script (no single flaky host should block
# the others).
if [ "${EXT_CORPUS:-0}" = "1" ]; then
mkdir -p "$CORPUS_DIR/ext"

# GitHub Archive: one hour of events, JSON lines (gzip)
[ -f "$CORPUS_DIR/ext/gharchive.json" ] || { echo "Downloading GitHub Archive sample..."; curl -fsSL https://data.gharchive.org/2024-01-15-12.json.gz | gunzip -c > "$CORPUS_DIR/ext/gharchive.json"; } || { echo "WARNING: gharchive.json download failed, skipping (network?)." >&2; rm -f "$CORPUS_DIR/ext/gharchive.json"; }

# NASA HTTP logs, July 1995 (gzip)
[ -f "$CORPUS_DIR/ext/nasa_access.log" ] || { echo "Downloading NASA HTTP logs..."; curl -fsSL ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz | gunzip -c > "$CORPUS_DIR/ext/nasa_access.log"; } || { echo "WARNING: nasa_access.log download failed, skipping (network?)." >&2; rm -f "$CORPUS_DIR/ext/nasa_access.log"; }

# NYC yellow taxi, one month, Parquet (TLC public bucket)
[ -f "$CORPUS_DIR/ext/yellow_tripdata.parquet" ] || { echo "Downloading NYC yellow taxi trip data..."; curl -fsSL https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet -o "$CORPUS_DIR/ext/yellow_tripdata.parquet"; } || { echo "WARNING: yellow_tripdata.parquet download failed, skipping (network?)." >&2; rm -f "$CORPUS_DIR/ext/yellow_tripdata.parquet"; }

# Linux kernel source tarball, uncompressed, first 64 MB. `head -c` closing
# early after 64 MB makes curl/xz report a broken-pipe error even though the
# truncated file is exactly what we want, so success is judged by the
# output file's size rather than the pipeline's exit code.
if [ ! -f "$CORPUS_DIR/ext/linux.tar" ]; then
    echo "Downloading Linux kernel source (first 64 MB)..."
    if curl -fsSL https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.tar.xz | xz -dc | head -c 67108864 > "$CORPUS_DIR/ext/linux.tar"; then :; fi
    if [ -s "$CORPUS_DIR/ext/linux.tar" ]; then
        echo "linux.tar downloaded successfully."
    else
        echo "WARNING: linux.tar download failed, skipping (network?)." >&2
        rm -f "$CORPUS_DIR/ext/linux.tar"
    fi
fi

# OpenStreetMap PBF, a small region (Geofabrik)
[ -f "$CORPUS_DIR/ext/liechtenstein.osm.pbf" ] || { echo "Downloading OpenStreetMap Liechtenstein extract..."; curl -fsSL https://download.geofabrik.de/europe/liechtenstein-latest.osm.pbf -o "$CORPUS_DIR/ext/liechtenstein.osm.pbf"; } || { echo "WARNING: liechtenstein.osm.pbf download failed, skipping (network?)." >&2; rm -f "$CORPUS_DIR/ext/liechtenstein.osm.pbf"; }

# TPC-H lineitem (sf=1), generated locally via DuckDB -- only if duckdb is
# on PATH; there is no download URL for this one.
if command -v duckdb >/dev/null 2>&1; then
    if [ -f "$CORPUS_DIR/ext/lineitem.parquet" ] && [ -f "$CORPUS_DIR/ext/lineitem.csv" ]; then
        echo "TPC-H lineitem already present."
    else
        echo "Generating TPC-H lineitem (sf=1) via DuckDB..."
        duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=1); COPY lineitem TO '$CORPUS_DIR/ext/lineitem.parquet'; COPY lineitem TO '$CORPUS_DIR/ext/lineitem.csv'" \
            || echo "WARNING: DuckDB TPC-H generation failed, skipping." >&2
    fi
else
    echo "NOTE: duckdb not found on PATH, skipping TPC-H lineitem generation."
fi

# vmlinux: needs an ELF kernel build with debug symbols on hand; there is no
# generic download URL for it, so it is skipped unless one is already
# present in the ext corpus.
if [ -f "$CORPUS_DIR/ext/vmlinux" ]; then
    echo "vmlinux already present."
else
    echo "NOTE: skipping vmlinux (no local kernel build available)."
fi
else
    echo "Skipping the extended corpus (corpus/ext); run with EXT_CORPUS=1 to download it."
fi
