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
# every #[ignore]d test of glyd-engine and glyd-gpu is on glyd-engine/tests/ignored-in-gate.txt, as `run` (the gate runs it: scripts/test_ignored.sh) or as `skip ... because ...`;
# a name there that is no longer ignored fails too
ign=$(git ls-files glyd-engine glyd-gpu | grep '\.rs$' | xargs awk '/^[ \t]*#\[.*ignore/ { i = 1; next }
  i && /^[ \t]*(pub[ \t]+)?fn[ \t]+[A-Za-z_]/ { sub(/^[ \t]*(pub[ \t]+)?fn[ \t]+/, ""); sub(/[^A-Za-z0-9_].*/, ""); print; i = 0 }' | sort)
lst=$(awk '$1 == "run" || $1 == "skip" { print $2 }' glyd-engine/tests/ignored-in-gate.txt | sort)
if [[ "$ign" != "$lst" ]]; then
  echo "glyd-engine/tests/ignored-in-gate.txt and the #[ignore]d tests differ (< ignored, > listed):"; diff <(echo "$ign") <(echo "$lst") | grep '^[<>]' || true
  echo "List a new ignored test there as \`run NAME\` (the gate runs it) or \`skip NAME because REASON\`; take a name off it when the test is no longer ignored."
  exit 1
fi
# every step of the M1 gate script is named on the guide's gate page (docs/guide/src/gate.md)
python3 -I scripts/check_gate_docs.py > /dev/null
# the guide's generated pages still read their sources (scripts/build_site.sh runs both): a refactor of support.rs, levels.rs or machine.rs that gen_models.py cannot
# read fails here, on every pull request, not only on the Pages build; they write to a temporary folder, offline
guide=$(mktemp -d)
trap 'rm -rf "$guide"' EXIT
python3 -I scripts/gen_models.py "$guide/models.md" > /dev/null
python3 -I scripts/gen_reference.py "$guide" > /dev/null
echo "tree ok"
