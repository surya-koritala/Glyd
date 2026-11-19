#!/usr/bin/env bash
# results/export on a pod: a job's big files (hbr40's codes, ~42 GB, from tools/cards/nemotron_quality_job.sh). rp_run.sh leaves the directory out of the minute's tar of ~/results (that would copy all of it
# every minute) and calls pull_export after each pull instead: every finished file is copied once, by its own ssh call, into $OUT/export. A job writes a big file as NAME.tmp and renames it when it is whole (those
# are skipped); a big file already here at the pod's size is kept; a small one (meta.json, rewritten after each layer) is taken again. Needs SSH (the ssh command, an array) and OUT; run in the pod's home.
#   bash tools/runpod/pull_export.sh selftest      pull_export against a local directory standing in for the pod
LIST='cd results/export 2> /dev/null && find . -type f ! -name "*.tmp" | while read -r f; do printf "%s\t%s\n" "${f#./}" "$(wc -c < "$f" | tr -d " ")"; done'
pull_export() {
  local f s
  [ "${PULL_EXPORT:-1}" = 0 ] && return 0   # (the job sends its export to the private bucket instead: nemotron_quality.sh with S3)
  "${SSH[@]}" "$LIST" 2> /dev/null | while IFS=$'\t' read -r f s; do
    [ "$s" -gt 1000000 ] && [ "$(wc -c 2> /dev/null < "$OUT/export/$f" | tr -d ' ')" = "$s" ] && continue
    mkdir -p "$OUT/export/$(dirname "$f")"
    "${SSH[@]}" "cat 'results/export/$f'" < /dev/null > "$OUT/export/$f.part" 2> /dev/null && mv "$OUT/export/$f.part" "$OUT/export/$f"
  done
}
if [ "${1:-}" = selftest ] && [ "$(basename "$0")" = pull_export.sh ]; then
  set -eu
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  ok() { "$@" || { echo "pull_export selftest FAILED: $*"; exit 1; }; }
  P=$T/pod/results/export; mkdir -p "$P/sub" "$T/out"; OUT=$T/out; SSH=(bash -c)
  head -c 2500000 /dev/urandom > "$P/layer-000.safetensors"; head -c 2500000 /dev/urandom > "$P/sub/layer-001.safetensors"
  head -c 2500000 /dev/urandom > "$P/layer-002.safetensors.tmp"; echo '{"complete": false}' > "$P/meta.json"
  cd "$T/pod"; pull_export
  ok cmp "$P/layer-000.safetensors" "$OUT/export/layer-000.safetensors"; ok cmp "$P/sub/layer-001.safetensors" "$OUT/export/sub/layer-001.safetensors"; ok cmp "$P/meta.json" "$OUT/export/meta.json"
  ok test ! -e "$OUT/export/layer-002.safetensors.tmp"; ok test ! -e "$OUT/export/layer-002.safetensors"
  head -c 2500000 /dev/urandom > "$P/other"; mv "$P/other" "$P/layer-000.safetensors"   # the same size, other bytes: kept, a big file is written once
  mv "$P/layer-002.safetensors.tmp" "$P/layer-002.safetensors"; echo '{"complete": true}' > "$P/meta.json"   # the finished file is taken, the small one again
  pull_export
  ok bash -c "! cmp -s '$P/layer-000.safetensors' '$OUT/export/layer-000.safetensors'"; ok cmp "$P/layer-002.safetensors" "$OUT/export/layer-002.safetensors"; ok cmp "$P/meta.json" "$OUT/export/meta.json"
  head -c 2600000 /dev/urandom > "$P/layer-000.safetensors"; pull_export   # another size: taken again
  ok cmp "$P/layer-000.safetensors" "$OUT/export/layer-000.safetensors"; ok test -z "$(find "$OUT" -name '*.part')"
  echo "pull_export selftest ok: files copied whole, .tmp skipped, a big file kept at its size, a small one or a resized one taken again"
fi
