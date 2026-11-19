#!/usr/bin/env bash
# moe_src.tar from this repository's HEAD (committed files only): gpu/, glyd-gpu-py/, job/ (the tables' scripts, these ones), COMMIT. Run it in a checkout of the tree to be measured.
#   bash make_tar.sh [OUT.tar]          then, on the machine: the tar and moe_job.sh in ~, bash ~/moe_job.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
WT=${WT:-$(git -C "$HERE" rev-parse --show-toplevel)}
OUT=${1:-$HERE/moe_src.tar}
T=$(mktemp -d)
git -C "$WT" archive HEAD gpu glyd-gpu-py | tar -x -C "$T"
mkdir -p "$T/job"
cp "$HERE/sweep_summary.py" "$HERE/tile_verdict.py" "$HERE/ncu_summary.py" "$T/job/"
echo "$(git -C "$WT" branch --show-current) $(git -C "$WT" log -1 --format='%h (%s)') on private/main $(git -C "$WT" rev-parse --short private/main)" > "$T/COMMIT"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$T" -cf "$OUT" gpu glyd-gpu-py job COMMIT 2> /dev/null || tar -C "$T" -cf "$OUT" gpu glyd-gpu-py job COMMIT
rm -rf "$T"
echo "$OUT: $(du -h "$OUT" | cut -f1), $(tar -tf "$OUT" | wc -l) files, $(tar -xOf "$OUT" COMMIT)"
