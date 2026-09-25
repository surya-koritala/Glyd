#!/usr/bin/env bash
# The savings calculator's rows (docs/savings.html), measured: for each
# row, its raw bytes, and per codec the bytes it stores and the CPU
# seconds its command line spends decompressing them to /dev/null.
# Every Glyd output is decoded and compared with its input.
#   scripts/calc_rows.sh CORPUS_DIR > rows.tsv
# CORPUS_DIR holds bench/ (scripts/download_corpus.sh and the Parquet
# files), ext2/ (scripts/download_ext_corpus.sh) and versions/
# (scripts/download_versions.sh). GLYD, LZ4 name the binaries.
set -u
D="${1:?corpus directory}"
G="${GLYD:-./target/release/glyd}"; LZ4="${LZ4:-lz4}"
W="$(mktemp -d)"; mkdir -p "$W/in"
trap 'rm -rf "$W"' EXIT
# The telemetry CSVs as 128 MiB slices, cut at a line end.
for f in alibaba_machine_usage ghcn_2023 yellow_tripdata_2024-02; do head -c 134217728 "$D/ext2/$f.csv" | awk '{print}' > "$W/in/$f.csv"; done
cp "$D/ext2/alibaba_machine_usage.jsonl" "$D/ext2/ghcn_2023.jsonl" "$W/in/"
B="$D/bench"
declare -A ROWS
ROWS[logs]="$B/nasa-access-jul95.log $B/clarknet-access-aug28.log $B/clarknet-access-sep4.log $B/pageviews-20240115-10.log $B/pageviews-20240115-11.log $B/pageviews-20240115-20.log"
ROWS[sql]="$B/enwiki-page_props.sql $B/enwiki-redirect.sql $B/simplewiki-categorylinks.sql $B/simplewiki-page.sql $B/simplewiki-pagelinks.sql $B/simplewiki-templatelinks.sql"
ROWS[json]="$B/gharchive-2024-01-15-12.json $B/gharchive-2024-01-15-18.json $B/gharchive-2024-01-16-12.json"
ROWS[parquet]="$B/fhvhv_tripdata_2024-01.parquet $B/fhvhv_tripdata_2024-02.parquet"
ROWS[parquet_snappy]="$B/fhvhv_tripdata_2024-01.snappy.parquet $B/yellow_tripdata_2024-02.snappy.parquet"
ROWS[telemetry_csv]="$W/in/alibaba_machine_usage.csv $W/in/ghcn_2023.csv $W/in/yellow_tripdata_2024-02.csv"
ROWS[telemetry_json]="$W/in/alibaba_machine_usage.jsonl $W/in/ghcn_2023.jsonl"
cpu() { /usr/bin/time -f "%U %S" "$@" 2>&1 >/dev/null | tail -n 1 | awk '{print $1 + $2}'; }
for row in logs sql json parquet parquet_snappy telemetry_csv telemetry_json; do
  raw=0; for f in ${ROWS[$row]}; do raw=$((raw + $(stat -c %s "$f"))); done
  for codec in lz4 gzip6 zstd3 zstd19 max maxrec ultra ultrarec; do
    size=0; cpus=0
    for f in ${ROWS[$row]}; do
      o="$W/$(basename "$f").$codec"
      case $codec in
        lz4) "$LZ4" -1 -q -f "$f" "$o"; c=$(cpu "$LZ4" -d -q -c "$o") ;;
        gzip6) gzip -6 -c "$f" > "$o"; c=$(cpu gzip -d -c "$o") ;;
        zstd3) zstd -3 -T0 -q -f "$f" -o "$o"; c=$(cpu zstd -d -q -c "$o") ;;
        zstd19) zstd -19 -T0 -q -f "$f" -o "$o"; c=$(cpu zstd -d -q -c "$o") ;;
        max) "$G" -9 "$f" -o "$o"; c=$(cpu "$G" -d "$o" -o /dev/null) ;;
        maxrec) "$G" -9 -r "$f" -o "$o"; c=$(cpu "$G" -d "$o" -o /dev/null) ;;
        ultra) "$G" --ultra "$f" -o "$o"; c=$(cpu "$G" -d "$o" -o /dev/null) ;;
        ultrarec) "$G" --ultra -r "$f" -o "$o"; c=$(cpu "$G" -d "$o" -o /dev/null) ;;
      esac
      case $codec in max|maxrec|ultra|ultrarec) "$G" -d "$o" -o "$W/back" 2>/dev/null; cmp -s "$W/back" "$f" || echo "MISMATCH $row $codec $f" >&2; rm -f "$W/back" ;; esac
      size=$((size + $(stat -c %s "$o"))); cpus=$(echo "$cpus + $c" | bc); rm -f "$o"
    done
    printf "%s\t%s\t%s\t%s\t%s\n" $row $codec $raw $size $cpus
  done
done
# Versions of one object: a Wikipedia page table a month on, against
# the last (zstd's --patch-from, Glyd's --base; LZ4 and gzip alone).
V="$D/versions"; a="$V/simplewiki-20260801-page.sql"; b="$V/simplewiki-20260901-page.sql"; raw=$(stat -c %s "$b")
"$LZ4" -1 -q -f "$b" "$W/v.lz4"; gzip -6 -c "$b" > "$W/v.gz"
zstd -3 -T0 -q -f --patch-from="$a" "$b" -o "$W/v.z3"; zstd -19 -T0 -q -f --patch-from="$a" "$b" -o "$W/v.z19"
"$G" --max --base "$a" "$b" -o "$W/v.max"; "$G" --ultra --base "$a" "$b" -o "$W/v.ultra"
for x in lz4:v.lz4 gzip6:v.gz zstd3:v.z3 zstd19:v.z19 max:v.max ultra:v.ultra; do printf "versions\t%s\t%s\t%s\t\n" "${x%%:*}" "$raw" "$(stat -c %s "$W/${x#*:}")"; done
