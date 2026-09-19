#!/usr/bin/env bash
# One real object-storage workflow, end to end, per codec: compress the
# dataset -> upload to S3 -> download to a fresh directory -> decompress
# -> verify every file against the original (sha256). Wall and CPU time
# of each step, bytes stored, and the monthly cost of keeping the
# dataset in S3 Standard and reading it back R times, at the region's
# public prices (constants below, dated).
#
#   scripts/s3_workflow.sh <bucket> <dataset dir> [codec ...]
#
# Codecs (each a CLI on PATH, all at the machine's thread count):
#   raw            no compression (the baseline the savings are against)
#   glyd-max       glyd --max -m         zstd-3   zstd -3 -T0
#   glyd-ultra     glyd --ultra -m       zstd-19  zstd -19 -T0
#   lz4            lz4 -1 (one thread: the format's frame is decoded on one)
# Results: s3_workflow.jsonl (one row per codec) and a table on stdout.
set -euo pipefail
BUCKET="$1"; DATA="$2"; shift 2
CODECS=("$@"); [ ${#CODECS[@]} -gt 0 ] || CODECS=(raw glyd-max glyd-ultra zstd-3 zstd-19 lz4)
RUN="s3wf-$(date -u +%Y%m%dT%H%M%SZ)-$$"
WORK="${S3WF_WORK:-$(mktemp -d)}"
OUT="${S3WF_OUT:-s3_workflow.jsonl}"
THREADS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
INSTANCE="${S3WF_INSTANCE:-$(curl -s -m 1 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null | xargs -I{} curl -s -m 1 -H 'X-aws-ec2-metadata-token: {}' http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)}"
INSTANCE="${INSTANCE:-local}"

# us-east-1 public on-demand prices, USD, read 2026-09-19 from
# aws.amazon.com/s3/pricing and aws.amazon.com/ec2/pricing/on-demand.
# EC2<->S3 transfer in one region is free; internet egress is listed
# separately because a workload that serves the data out pays it.
S3_GB_MONTH=0.023      # S3 Standard, first 50 TB
S3_PUT_PER_1000=0.005  # PUT, COPY, POST, LIST
S3_GET_PER_1000=0.0004 # GET, SELECT
EGRESS_PER_GB=0.09     # internet egress, first 10 TB/month
case "$INSTANCE" in
    c7g.2xlarge) EC2_PER_HOUR=0.2890 ;;
    c7i.2xlarge) EC2_PER_HOUR=0.3570 ;;
    c7g.xlarge)  EC2_PER_HOUR=0.1445 ;;
    c7i.xlarge)  EC2_PER_HOUR=0.1785 ;;
    *) EC2_PER_HOUR="${S3WF_EC2_PER_HOUR:-0.30}" ;;
esac

if stat -c%s /dev/null >/dev/null 2>&1; then STATSZ='-c%s'; else STATSZ='-f%z'; fi
files=()
while IFS= read -r f; do files+=("$f"); done < <(find "$DATA" -maxdepth 1 -type f ! -name '*.part' | sort)
[ ${#files[@]} -gt 0 ] || { echo "no files in $DATA" >&2; exit 1; }
raw_bytes=0; for f in "${files[@]}"; do raw_bytes=$((raw_bytes + $(stat $STATSZ "$f"))); done
n_files=${#files[@]}
echo "dataset: $n_files files, $raw_bytes bytes; instance $INSTANCE ($THREADS threads); run $RUN; bucket s3://$BUCKET" >&2
mkdir -p "$WORK"
if stat -c%s /dev/null >/dev/null 2>&1; then STATSZ='-c%s'; else STATSZ='-f%z'; fi
if command -v sha256sum >/dev/null; then sha() { sha256sum "$1" | cut -d' ' -f1; }; else sha() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi
now() { python3 -c 'import time; print(time.time())'; }
MANIFEST="$WORK/orig.sha256"; : > "$MANIFEST"
for f in "${files[@]}"; do echo "$(sha "$f")  $(basename "$f")" >> "$MANIFEST"; done

# time_cmd <var-prefix> <cmd...>: wall and CPU (user+sys) seconds.
time_cmd() {
    local prefix="$1"; shift
    local t0 t1 c0 c1
    t0=$(now); cpu_seconds; c0=$CPU_NOW
    "$@"
    t1=$(now); cpu_seconds; c1=$CPU_NOW
    printf -v "${prefix}_wall" '%.3f' "$(echo "$t1 - $t0" | bc -l)"
    printf -v "${prefix}_cpu" '%.3f' "$(echo "$c1 - $c0" | bc -l)"
}
# CPU seconds consumed by this shell's children so far, into CPU_NOW (the
# `times` builtin must run in this shell, not a subshell).
cpu_seconds() {
    times > "$WORK/.times"
    CPU_NOW=$(awk 'function sec(t) { m = 0; if (index(t, "m")) { split(t, a, "m"); m = a[1]; t = a[2] } sub("s", "", t); return m * 60 + t } NR == 2 { printf "%.3f", sec($1) + sec($2) }' "$WORK/.times")
}
dir_bytes() { find "$1" -type f -exec stat $STATSZ {} + | awk '{s+=$1} END {print s+0}'; }

compress_all() { # compress_all <codec> <src dir> <dst dir>
    local codec="$1" src="$2" dst="$3" f b
    mkdir -p "$dst"
    for f in "$src"/*; do
        [ -f "$f" ] || continue; b="$(basename "$f")"
        case "$codec" in
            raw) ln "$f" "$dst/$b" 2>/dev/null || cp "$f" "$dst/$b" ;;
            glyd-max) glyd --max -m -c "$f" -o "$dst/$b.glyd" ;;
            glyd-ultra) glyd --ultra -m -c "$f" -o "$dst/$b.glyd" ;;
            zstd-3) zstd -q -3 -T0 "$f" -o "$dst/$b.zst" ;;
            zstd-19) zstd -q -19 -T0 "$f" -o "$dst/$b.zst" ;;
            lz4) lz4 -q -1 "$f" "$dst/$b.lz4" ;;
        esac
    done
}
decompress_all() { # decompress_all <codec> <src dir> <dst dir>
    local codec="$1" src="$2" dst="$3" f b
    mkdir -p "$dst"
    for f in "$src"/*; do
        [ -f "$f" ] || continue; b="$(basename "$f")"
        case "$codec" in
            raw) ln "$f" "$dst/$b" 2>/dev/null || cp "$f" "$dst/$b" ;;
            glyd-*) glyd -d -m "$f" -o "$dst/${b%.glyd}" ;;
            zstd-*) zstd -q -d "$f" -o "$dst/${b%.zst}" ;;
            lz4) lz4 -q -d "$f" "$dst/${b%.lz4}" ;;
        esac
    done
}

printf '%-11s %14s %8s %9s %9s %9s %9s %9s  %s\n' codec stored_bytes ratio comp_s up_s down_s decomp_s cpu_s verified
for codec in "${CODECS[@]}"; do
    C="$WORK/$codec"; rm -rf "$C"; mkdir -p "$C/comp" "$C/down" "$C/out"
    time_cmd comp compress_all "$codec" "$DATA" "$C/comp"
    stored=$(dir_bytes "$C/comp")
    time_cmd up aws s3 cp --quiet --recursive "$C/comp" "s3://$BUCKET/$RUN/$codec/"
    time_cmd down aws s3 cp --quiet --recursive "s3://$BUCKET/$RUN/$codec/" "$C/down"
    time_cmd decomp decompress_all "$codec" "$C/down" "$C/out"
    verified=true
    while read -r want b; do
        [ "$(sha "$C/out/$b")" = "$want" ] || { verified=false; echo "MISMATCH: $codec $b" >&2; }
    done < "$MANIFEST"
    rm -rf "$C/down" "$C/out"
    aws s3 rm --quiet --recursive "s3://$BUCKET/$RUN/$codec/"
    ratio=$(echo "$raw_bytes / $stored" | bc -l)
    gb=$(echo "$stored / 1000000000" | bc -l)
    # Monthly cost of holding the dataset and reading it R times: storage +
    # PUTs once + GETs and decompression per read + compression once.
    cost() { # cost <reads per month> -> USD
        echo "$gb * $S3_GB_MONTH + $n_files * $S3_PUT_PER_1000 / 1000 + $1 * $n_files * $S3_GET_PER_1000 / 1000 + ($comp_cpu + $1 * $decomp_cpu) / 3600 * $EC2_PER_HOUR" | bc -l
    }
    egress=$(echo "$gb * $EGRESS_PER_GB" | bc -l)
    c1=$(cost 1); c10=$(cost 10); c100=$(cost 100)
    printf '%-11s %14d %8.3f %9.2f %9.2f %9.2f %9.2f %9.2f  %s\n' "$codec" "$stored" "$ratio" "$comp_wall" "$up_wall" "$down_wall" "$decomp_wall" "$(echo "$comp_cpu + $decomp_cpu" | bc -l)" "$verified"
    printf '{"kind":"s3","run":"%s","instance":"%s","threads":%s,"codec":"%s","files":%d,"raw_bytes":%d,"stored_bytes":%d,"ratio":%.4f,"compress_wall_s":%s,"compress_cpu_s":%s,"upload_wall_s":%s,"download_wall_s":%s,"decompress_wall_s":%s,"decompress_cpu_s":%s,"verified":%s,"usd_month_1_read":%.4f,"usd_month_10_reads":%.4f,"usd_month_100_reads":%.4f,"usd_egress_per_read":%.4f,"prices":{"s3_gb_month":%s,"put_per_1000":%s,"get_per_1000":%s,"ec2_per_hour":%s,"egress_per_gb":%s}}\n' \
        "$RUN" "$INSTANCE" "$THREADS" "$codec" "$n_files" "$raw_bytes" "$stored" "$ratio" "$comp_wall" "$comp_cpu" "$up_wall" "$down_wall" "$decomp_wall" "$decomp_cpu" "$verified" "$c1" "$c10" "$c100" "$egress" "$S3_GB_MONTH" "$S3_PUT_PER_1000" "$S3_GET_PER_1000" "$EC2_PER_HOUR" "$EGRESS_PER_GB" >> "$OUT"
done
echo "rows in $OUT" >&2
