#!/usr/bin/env bash
# The telemetry set for record mode (corpus/ext2/): slices of public
# datasets whose rows are measurements, the shape of exported metrics and
# structured logs. Each is the first 200 MB of the source's gzip stream
# (a gzip prefix decodes as a prefix), cut to 300 MB of text; the JSON
# lines twins hold the same rows as JSON objects (scripts/ext2_jsonl.py),
# the shape of telemetry exported or logged as JSON. Every entry is
# best-effort.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/corpus/ext2"
mkdir -p "$DIR"
CAP=$((300 * 1024 * 1024))

slice() { # slice <dest> <url> [tar]
    local dest="$1" url="$2" tar="${3:-}"
    if [ -s "$dest" ]; then echo "have $(basename "$dest")"; return; fi
    echo "downloading $(basename "$dest") from $url"
    if [ "$tar" = "tar" ]; then
        curl -fsSL --retry 3 -r 0-209715199 "$url" | { gzip -dc 2>/dev/null || true; } | { tar -xOf - 2>/dev/null || true; } | head -c "$CAP" > "$dest.part"
    else
        curl -fsSL --retry 3 -r 0-209715199 "$url" | { gzip -dc 2>/dev/null || true; } | head -c "$CAP" > "$dest.part"
    fi
    if [ -s "$dest.part" ]; then mv "$dest.part" "$dest"; else echo "WARNING: $dest failed" >&2; rm -f "$dest.part"; fi
}

# Alibaba cluster trace 2018: machine usage, one row per machine per
# 10 s (machine id, timestamp, cpu %, mem %, ..., net in/out, disk %).
slice "$DIR/alibaba_machine_usage.csv" http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz tar
# NOAA GHCN daily 2023: station, date, element, value, flags.
slice "$DIR/ghcn_2023.csv" https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2023.csv.gz
# Common Crawl index (CDX): one JSON object per line after a key and a
# timestamp; hash-heavy, at its entropy floor, kept as a control.
slice "$DIR/cc_index_cdx.txt" https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2024-10/indexes/cdx-00000.gz

# JSON lines twins of the two measurement sets (128 MB each).
[ -s "$DIR/alibaba_machine_usage.jsonl" ] || python3 "$(dirname "${BASH_SOURCE[0]}")/ext2_jsonl.py" "$DIR/alibaba_machine_usage.csv" "$DIR/alibaba_machine_usage.jsonl" machine_id,time_stamp,cpu_util_percent,mem_util_percent,mem_gps,mkpi,net_in,net_out,disk_io_percent
[ -s "$DIR/ghcn_2023.jsonl" ] || python3 "$(dirname "${BASH_SOURCE[0]}")/ext2_jsonl.py" "$DIR/ghcn_2023.csv" "$DIR/ghcn_2023.jsonl" id,date,element,value,m_flag,q_flag,s_flag,obs_time

# NYC taxi trips (corpus/bench's Parquet) exported to CSV, the first
# 1.5 M rows, when pyarrow is installed: timestamps with fractions,
# fares and distances with decimals.
if [ ! -s "$DIR/yellow_tripdata_2024-02.csv" ]; then
    python3 - "$DIR" <<'PY' || echo "WARNING: taxi CSV export needs pyarrow (pip install pyarrow)" >&2
import sys, pyarrow.parquet as pq, pyarrow.csv as pcsv
t = pq.read_table(sys.argv[1] + "/../bench/yellow_tripdata_2024-02.parquet").slice(0, 1_500_000)
pcsv.write_csv(t, sys.argv[1] + "/yellow_tripdata_2024-02.csv")
PY
fi
ls -la "$DIR"
