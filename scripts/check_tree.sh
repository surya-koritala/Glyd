#!/usr/bin/env bash
# The source tree stays source: data files in benchmarks/, research/ and condor/results/ are at most 64 KB, and no tracked
# file is over 1 MiB, unless scripts/large-files.allow lists it (a path, or a prefix ending in /).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
allow=()
while IFS= read -r l; do [[ -n "$l" && "$l" != \#* ]] && allow+=("$l"); done < scripts/large-files.allow
allowed() { local p; for p in "${allow[@]}"; do [[ "$1" == "$p" || ( "$p" == */ && "$1" == "$p"* ) ]] && return 0; done; return 1; }
bad=0
while IFS= read -r -d '' f; do
  [[ -f "$f" ]] || continue
  size=$(wc -c < "$f")
  case "$f" in
    benchmarks/*|research/*|condor/results/*)
      case "${f##*.}" in json|jsonl|txt|log|csv|tsv|out|err|eval|npz|npy|bin|f32|parquet|safetensors)
        if (( size > 65536 )) && ! allowed "$f"; then echo "data file over 64 KB: $f ($size bytes)"; bad=1; fi ;;
      esac ;;
  esac
  if (( size > 1048576 )) && ! allowed "$f"; then echo "file over 1 MiB: $f ($size bytes)"; bad=1; fi
done < <(git ls-files -z)
if (( bad )); then
  echo "Keep summaries in the tree and raw outputs out of it (benchmarks/README.md); or list a file in scripts/large-files.allow."
  exit 1
fi
# every step of the M1 gate script is named on the guide's gate page (docs/guide/src/gate.md)
python3 -I scripts/check_gate_docs.py > /dev/null
echo "tree ok"
