#!/usr/bin/env bash
# The benchmark corpus for scripts/bench_suite: real logs, JSON, database
# exports and already-compressed Parquet, about 7 GB uncompressed, in
# corpus/bench/. Every entry is best-effort (a failed download is
# reported and skipped). Small-object dictionaries are trained on
# files in corpus/bench/train/, never on the measured files.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/corpus/bench"
mkdir -p "$DIR/train"

get() { # get <dest> <url> [gunzip]
    local dest="$1" url="$2" gz="${3:-}"
    if [ -s "$dest" ]; then echo "have $(basename "$dest")"; return; fi
    echo "downloading $(basename "$dest") from $url"
    if [ "$gz" = "gz" ]; then
        curl -fsSL --retry 3 "$url" | gunzip -c > "$dest.part" && mv "$dest.part" "$dest" || { echo "WARNING: $dest failed" >&2; rm -f "$dest.part"; }
    else
        curl -fsSL --retry 3 "$url" -o "$dest.part" && mv "$dest.part" "$dest" || { echo "WARNING: $dest failed" >&2; rm -f "$dest.part"; }
    fi
}

# JSON events (GH Archive, one hour each, ~1 GB raw): two hours measured,
# a third (another day) for training small-object dictionaries.
get "$DIR/gharchive-2024-01-15-12.json" https://data.gharchive.org/2024-01-15-12.json.gz gz
get "$DIR/gharchive-2024-01-16-12.json" https://data.gharchive.org/2024-01-16-12.json.gz gz
get "$DIR/gharchive-2024-01-15-18.json" https://data.gharchive.org/2024-01-15-18.json.gz gz
get "$DIR/train/gharchive-2024-01-14-12.json" https://data.gharchive.org/2024-01-14-12.json.gz gz

# Web server access logs (NASA and ClarkNet, 1995; ~700 MB): NASA July
# measured, August for training.
get "$DIR/nasa-access-jul95.log" ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz gz
get "$DIR/train/nasa-access-aug95.log" ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz gz
get "$DIR/clarknet-access-aug28.log" ftp://ita.ee.lbl.gov/traces/clarknet_access_log_Aug28.gz gz
get "$DIR/clarknet-access-sep4.log" ftp://ita.ee.lbl.gov/traces/clarknet_access_log_Sep4.gz gz

# Wikipedia pageview logs (one hour each, ~350 MB raw): two measured, one
# for training.
get "$DIR/pageviews-20240115-10.log" https://dumps.wikimedia.org/other/pageviews/2024/2024-01/pageviews-20240115-100000.gz gz
get "$DIR/pageviews-20240115-11.log" https://dumps.wikimedia.org/other/pageviews/2024/2024-01/pageviews-20240115-110000.gz gz
get "$DIR/pageviews-20240115-20.log" https://dumps.wikimedia.org/other/pageviews/2024/2024-01/pageviews-20240115-200000.gz gz
get "$DIR/train/pageviews-20240114-10.log" https://dumps.wikimedia.org/other/pageviews/2024/2024-01/pageviews-20240114-100000.gz gz

# Database exports: Wikipedia SQL dumps (MySQL INSERT statements).
for t in pagelinks page categorylinks templatelinks; do
    get "$DIR/simplewiki-$t.sql" "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-$t.sql.gz" gz
done
get "$DIR/enwiki-redirect.sql" https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-redirect.sql.gz gz
get "$DIR/enwiki-page_props.sql" https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page_props.sql.gz gz

# Already-compressed columnar data: NYC TLC trip records (Parquet, Snappy).
get "$DIR/fhvhv_tripdata_2024-01.parquet" https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet
get "$DIR/fhvhv_tripdata_2024-02.parquet" https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-02.parquet
get "$DIR/yellow_tripdata_2024-02.parquet" https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet

echo; du -sh "$DIR"; ls -l "$DIR" "$DIR/train"
