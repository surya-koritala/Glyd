#!/usr/bin/env bash
# Model weights (safetensors): each file with zstd -3, zstd -19 and
# Glyd -9; each checkpoint against the one before it with zstd -19
# --patch-from and Glyd --max --base. Bytes, write and read seconds
# (wall, all cores); every Glyd output decoded and compared.
#   scripts/bench_weights.sh MODEL_DIR > rows.tsv
# MODEL_DIR holds the files named below (Pythia-410M checkpoints from
# EleutherAI/pythia-410m revisions step1000..step143000, Qwen2.5-0.5B
# and Qwen2.5-0.5B-Instruct, each as one model.safetensors renamed).
set -u
A="${1:?model directory}"; G="${GLYD:-./target/release/glyd}"; W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
now() { date +%s.%N; }
row() { printf "%s\t%s\t%s\t%s\t%.1f\t%.1f\t%s\n" "$@"; }
for m in pythia-410m-step1000 pythia-410m-step70000 pythia-410m-step143000 Qwen2.5-0.5B; do
  f="$A/$m.safetensors"; raw=$(stat -c %s "$f")
  for c in zstd3 zstd19 glyd9; do
    t0=$(now)
    case $c in zstd3) zstd -3 -T0 -q -f "$f" -o "$W/o" ;; zstd19) zstd -19 -T0 -q -f "$f" -o "$W/o" ;; glyd9) "$G" -9 "$f" -o "$W/o" ;; esac
    t1=$(now)
    case $c in glyd9) "$G" -d "$W/o" -o "$W/back" ;; *) zstd -d -q -f "$W/o" -o "$W/back" ;; esac
    t2=$(now); cmp -s "$W/back" "$f" && ok=exact || ok=MISMATCH
    row "$m" "$c" "$raw" "$(stat -c %s "$W/o")" "$(echo "$t1 - $t0" | bc)" "$(echo "$t2 - $t1" | bc)" "$ok"; rm -f "$W/o" "$W/back"
  done
done
for pair in pythia-410m-step1000:pythia-410m-step2000 pythia-410m-step70000:pythia-410m-step71000 pythia-410m-step142000:pythia-410m-step143000 Qwen2.5-0.5B:Qwen2.5-0.5B-Instruct; do
  a="$A/${pair%%:*}.safetensors"; b="$A/${pair#*:}.safetensors"; raw=$(stat -c %s "$b"); n="${pair#*:} against ${pair%%:*}"
  t0=$(now); zstd -19 -T0 -q -f --long=31 --patch-from="$a" "$b" -o "$W/o"; t1=$(now)
  zstd -d -q -f --long=31 --patch-from="$a" "$W/o" -o "$W/back"; t2=$(now); cmp -s "$W/back" "$b" && ok=exact || ok=MISMATCH
  row "$n" zstd19patch "$raw" "$(stat -c %s "$W/o")" "$(echo "$t1 - $t0" | bc)" "$(echo "$t2 - $t1" | bc)" "$ok"; rm -f "$W/o" "$W/back"
  t0=$(now); "$G" --max --base "$a" "$b" -o "$W/o"; t1=$(now)
  "$G" -d --base "$a" "$W/o" -o "$W/back"; t2=$(now); cmp -s "$W/back" "$b" && ok=exact || ok=MISMATCH
  row "$n" glydbase "$raw" "$(stat -c %s "$W/o")" "$(echo "$t1 - $t0" | bc)" "$(echo "$t2 - $t1" | bc)" "$ok"; rm -f "$W/o" "$W/back"
done
