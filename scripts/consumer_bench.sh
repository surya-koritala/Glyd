#!/usr/bin/env bash
# P20: the consumer benchmarks of a release on this computer's GPU. Each row of benchmarks/consumer/matrix.tsv for this GPU's memory tier, on every engine
# here, measured by the one harness (`glyd-engine bench`, docs/bench.md) on the same GPU: Glyd at the row's level, llama.cpp on a GGUF of the same weights at
# about the same bits, Ollama and LM Studio on that same GGUF file. Each result keeps its command lines, the engines' versions and the server's log beside it
# (NAME.json, NAME.cmd, NAME.log); a side that could not run says why (NAME.skip). At the end the comparison page: `glyd-engine bench --report OUT`.
# Glyd's server is told to take the steps its plan marks (GLYD_DEGRADE=yes, as `--yes`: a mixture's experts in the computer's memory where the card is
# too small), and a row whose server loaded another level than the row's is not a result (NAME.json becomes NAME.json.other-level, which the page does not read, with a NAME.skip).
# It exits 1 where a Glyd row did not run, or where other engines were asked for and none of them ran a row.
#   scripts/consumer_bench.sh OUT
# Environment:
#   GLYD_ENGINE     the glyd-engine program (default: target/release/glyd-engine, then PATH)
#   ENGINES         which sides to run (default "glyd llama.cpp ollama lmstudio")
#   TIER            the tier's rows to run (default: this GPU's memory, 8 / 12 / 16 / 24 / 32 GB); MATRIX the rows' file; ONLY a pattern (grep -E) of the models to run
#   LLAMA_SERVER    llama.cpp's server (default: built at LLAMA_CPP_REF in WORK/llama.cpp); LLAMA_ARGS added to its command line
#   GGUF_DIR        where the GGUF files are, as MODELNAME-QUANT.gguf (default WORK/gguf); CONVERT=1 makes a missing one from the Hub checkpoint with
#                   llama.cpp's convert_hf_to_gguf.py and llama-quantize (needs about three times the bf16 checkpoint's size free, removed after)
#   OLLAMA_URL      a running Ollama (default: `ollama serve` started here where ollama is installed)
#   LMSTUDIO_URL, LMSTUDIO_MODEL   a running LM Studio server with the row's GGUF loaded (LM Studio has no headless start here)
#   LOADS (short:1,short:4,4k:1), RUNS (3), MAX_TOKENS (256), WORK (OUT/.work)
#   SLOTS (4) the requests llama.cpp and Ollama take at once; SLOT_CTX (8192) the context each of them has
set -uo pipefail
[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 2; }
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MATRIX=${MATRIX:-$ROOT/benchmarks/consumer/matrix.tsv}
WORK=${WORK:-$OUT/.work}; mkdir -p "$WORK"
GE=${GLYD_ENGINE:-$ROOT/target/release/glyd-engine}; [ -x "$GE" ] || GE=$(command -v glyd-engine) || { echo "no glyd-engine (GLYD_ENGINE=...)"; exit 2; }
ENGINES=${ENGINES:-glyd llama.cpp ollama lmstudio}
LOADS=${LOADS:-short:1,short:4,4k:1}; RUNS=${RUNS:-3}; MAX_TOKENS=${MAX_TOKENS:-256}
SLOTS=${SLOTS:-4}; SLOT_CTX=${SLOT_CTX:-8192}   # (llama.cpp's -c is shared by its slots: -c 8192 -np 4 left each 2,048 tokens, and the 4K prompt was refused)
LLAMA_CPP_REF=${LLAMA_CPP_REF:-0b51bb2fed322bc884f96f98db340e2b0f6ea176}   # (llama.cpp of 2026-11-22)
GGUF_DIR=${GGUF_DIR:-$WORK/gguf}; mkdir -p "$GGUF_DIR"
GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
MIB=$(echo "$GPU" | cut -d, -f2 | tr -dc 0-9)
TIER=${TIER:-$(awk -v m="$MIB" 'BEGIN { t = 0; n = split("8 12 16 24 32", ts, " "); for (i = 1; i <= n; i++) if (m / 1024 + 0.5 >= ts[i]) t = ts[i]; print t }')}
echo "consumer bench: $GPU, tier $TIER GB; engines: $ENGINES; loads $LOADS, $RUNS runs; out $OUT"
on() { case " $ENGINES " in *" $1 "*) return 0;; esac; return 1; }
has() { command -v "$1" > /dev/null 2>&1; }
up() { for _ in $(seq 1 360); do curl -sf "$1" > /dev/null && return 0; [ -n "${2:-}" ] && ! kill -0 "$2" 2> /dev/null && return 1; sleep 5; done; return 1; }
versions() { echo "gpu: $GPU"; echo "glyd-engine: $("$GE" --version 2> /dev/null) $(git -C "$ROOT" rev-parse --short=10 HEAD 2> /dev/null)"; echo "date: $(date -u +%FT%TZ)"; }
bench() {  # bench NAME ARGS...: the harness's run, its command kept
  local name=$1; shift
  { versions; echo "bench: $GE bench $* --loads $LOADS --runs $RUNS --max-tokens $MAX_TOKENS"; } >> "$OUT/$name.cmd"
  "$GE" bench "$@" --loads "$LOADS" --runs "$RUNS" --max-tokens "$MAX_TOKENS" --out "$OUT/$name.json" >> "$OUT/$name.log" 2>&1
  local rc=$?
  echo "  $name: rc=$rc $(tail -n 4 "$OUT/$name.log" | tr '\n' ' ' | cut -c1-200)"
  # (a side that did not run, as a model the engine refuses on this card, is on the page with the log's last words)
  [ -s "$OUT/$name.json" ] || { echo "did not run: $(tail -n 2 "$OUT/$name.log" | tr '\n' ' ' | cut -c1-300)" > "$OUT/$name.skip"; }
  # (counted for the end: a Glyd row that failed, and the rows another engine ran)
  case $name in
    glyd-*) { [ $rc = 0 ] && [ -s "$OUT/$name.json" ]; } || GLYD_FAILED="$GLYD_FAILED $name" ;;
    *) [ -s "$OUT/$name.json" ] && OTHERS=$((OTHERS + 1)) ;;
  esac
}
GLYD_FAILED=""; OTHERS=0
llama_server() {
  [ -n "${LLAMA_SERVER:-}" ] && { echo "$LLAMA_SERVER"; return; }
  local s=$WORK/llama.cpp/build/bin/llama-server
  if [ ! -x "$s" ]; then
    ( rm -rf "$WORK/llama.cpp" && git clone -q https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp" && cd "$WORK/llama.cpp" && git checkout -q "$LLAMA_CPP_REF" \
      && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)" -DLLAMA_CURL=OFF > /dev/null \
      && cmake --build build -j "$(nproc)" --target llama-server llama-quantize > "$WORK/llama-build.log" 2>&1 ) || { echo "llama.cpp: the build failed ($WORK/llama-build.log)" >&2; return 1; }
  fi
  echo "$s"
}
gguf() {  # gguf REPO QUANT[@SOURCE]: the GGUF file of the row (made where CONVERT=1), or nothing and the reason in WORK/gguf-NAME.why
  local q=${2%%@*} src=$1; [ "$2" != "$q" ] && src=${2#*@}
  local base=${src##*/}-$q; local f=$GGUF_DIR/$base.gguf why=$WORK/gguf-$base.why log=$WORK/gguf-$base
  rm -f "$why"
  [ -s "$f" ] && { echo "$f"; return; }
  [ "${CONVERT:-0}" = 1 ] || { echo "no GGUF $q of ${src##*/} in $GGUF_DIR (CONVERT=1 makes one)" > "$why"; return 1; }
  llama_server > /dev/null || { echo "llama.cpp did not build, so no GGUF could be made ($WORK/llama-build.log)" > "$why"; return 1; }
  local hf=$WORK/hf-src venv=$WORK/convert-venv step=venv
  # (llama.cpp pins a transformers that cannot read the newest tokenizers: gemma-4's stopped it with "'list' object has no attribute 'keys'"; the newest one is put over it)
  [ -x "$venv/bin/python" ] && [ -e "$venv/.glyd-ready" ] || { python3 -m venv "$venv" && "$venv/bin/pip" install -q -r "$WORK/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub \
    && "$venv/bin/pip" install -q -U transformers && touch "$venv/.glyd-ready"; } > "$log.venv.log" 2>&1 \
    && step=download && "$venv/bin/hf" download "$src" --local-dir "$hf" > "$log.download.log" 2>&1 \
    && step=convert && "$venv/bin/python" "$WORK/llama.cpp/convert_hf_to_gguf.py" "$hf" --outtype bf16 --outfile "$WORK/bf16.gguf" > "$log.convert.log" 2>&1 \
    && step=quantize && "$WORK/llama.cpp/build/bin/llama-quantize" "$WORK/bf16.gguf" "$f.part" "$q" > "$log.quantize.log" 2>&1 && mv "$f.part" "$f"
  local rc=$?; rm -rf "$hf" "$WORK/bf16.gguf" "$f.part"
  [ $rc = 0 ] && { echo "$f"; return; }
  echo "no GGUF $q of ${src##*/}: llama.cpp's $step step failed: $(grep -v '^[[:space:]]*$' "$log.$step.log" 2> /dev/null | tail -n 1 | cut -c1-200) ($log.$step.log)" > "$why"
  return 1
}
bits_of() { awk -v b="$(stat -c %s "$1")" -v p="$2" 'BEGIN { printf "%.2f", b * 8 / (p * 1e9) }'; }   # (the file's bytes over the parameters: embeddings and all)

while IFS=$'\t' read -r tier model released params level quant; do
  [ "${tier:0:1}" = "#" ] || [ -z "$tier" ] || [ "$tier" != "$TIER" ] && continue
  [ -n "${ONLY:-}" ] && ! echo "$model" | grep -Eq "$ONLY" && continue
  short=${model##*/}; echo "== $short ($params B, released $released): glyd $level, GGUF $quant"
  common=(--params "$params" --source "$model")
  if on glyd; then
    name=glyd-$short-$level
    echo "env: GLYD_DEGRADE=yes (the server takes the steps its plan marks, as --yes; a lower level than $level is not taken as this row's result)" >> "$OUT/$name.cmd"
    GLYD_DEGRADE=yes bench "$name" "$model:$level" --no-draft --engine glyd --label "glyd $level" "${common[@]}"
    # (with the steps taken the plan may also step down a level: the result says which level the server loaded, and only the row's own counts)
    if [ -s "$OUT/$name.json" ] && ! grep -Eq "\"level\": *\"$level\"" "$OUT/$name.json"; then
      got=$(grep -Eo '"level": *"[a-z]+"' "$OUT/$name.json" | head -1 | cut -d'"' -f4)
      mv "$OUT/$name.json" "$OUT/$name.json.other-level"
      echo "did not run at $level: the server loaded ${got:-another level} ($name.json.other-level)" > "$OUT/$name.skip"
      echo "  $name: loaded ${got:-another level}, not $level"
      GLYD_FAILED="$GLYD_FAILED $name"
    fi
  fi
  [ "$quant" = "-" ] && continue
  q=${quant%%@*}; file=$(gguf "$model" "$quant")
  if [ -z "$file" ]; then
    src=$model; [ "$quant" != "$q" ] && src=${quant#*@}
    reason=$(cat "$WORK/gguf-${src##*/}-$q.why" 2> /dev/null); echo "  ${reason:-no GGUF $q of $short}"
    for e in llama.cpp ollama lmstudio; do on $e && echo "${reason:-no GGUF $q of $short in $GGUF_DIR}" > "$OUT/$e-$short-$q.skip"; done
    continue
  fi
  b=$(bits_of "$file" "$params")
  if on llama.cpp && S=$(llama_server); then
    name=llama.cpp-$short-$q
    cmd=("$S" -m "$file" --fit on -fa on -c $((SLOT_CTX * SLOTS)) -np "$SLOTS" --jinja --host 127.0.0.1 --port 8091 ${LLAMA_ARGS:-})
    echo "server: ${cmd[*]}" > "$OUT/$name.cmd"; echo "llama.cpp: $LLAMA_CPP_REF" >> "$OUT/$name.cmd"
    "${cmd[@]}" > "$OUT/$name.log" 2>&1 & pid=$!
    if up http://127.0.0.1:8091/health $pid; then
      bench "$name" --against http://127.0.0.1:8091 --engine llama.cpp --bits "$b" --label "llama.cpp $q" "${common[@]}"
    else echo "llama-server did not start ($name.log)" > "$OUT/$name.skip"; fi
    kill $pid 2> /dev/null; wait $pid 2> /dev/null
  fi
  if on ollama && { [ -n "${OLLAMA_URL:-}" ] || has ollama; }; then
    name=ollama-$short-$q; url=${OLLAMA_URL:-http://127.0.0.1:11434}; opid=
    if [ -z "${OLLAMA_URL:-}" ] && ! curl -sf "$url/api/version" > /dev/null; then
      OLLAMA_NUM_PARALLEL=$SLOTS OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH=$SLOT_CTX ollama serve > "$OUT/$name.log" 2>&1 & opid=$!
      up "$url/api/version" $opid
    fi
    printf 'FROM %s\nPARAMETER temperature 0\n' "$file" > "$WORK/Modelfile"
    { echo "ollama: $(curl -s "$url/api/version")"; echo "ollama create glyd-$short-$q -f Modelfile: $(cat "$WORK/Modelfile" | tr '\n' ' ')"; echo "server env: OLLAMA_NUM_PARALLEL=$SLOTS OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH=$SLOT_CTX"; } > "$OUT/$name.cmd"
    if ollama create "glyd-$short-$q" -f "$WORK/Modelfile" >> "$OUT/$name.log" 2>&1; then
      bench "$name" --against "$url" --model "glyd-$short-$q" --engine ollama --bits "$b" --label "ollama $q" "${common[@]}"
      ollama rm "glyd-$short-$q" > /dev/null 2>&1
    else echo "ollama create failed ($name.log)" > "$OUT/$name.skip"; fi
    [ -n "$opid" ] && { kill $opid 2> /dev/null; wait $opid 2> /dev/null; }
  elif on ollama; then echo "no ollama here (OLLAMA_URL=...)" > "$OUT/ollama-$short-$q.skip"; fi
  if on lmstudio && [ -n "${LMSTUDIO_URL:-}" ]; then
    echo "LM Studio at $LMSTUDIO_URL, model ${LMSTUDIO_MODEL:-?} (loaded by hand: $file)" > "$OUT/lmstudio-$short-$q.cmd"
    bench "lmstudio-$short-$q" --against "$LMSTUDIO_URL" --model "${LMSTUDIO_MODEL:-}" --engine lmstudio --bits "$b" --label "LM Studio $q" "${common[@]}"
  elif on lmstudio; then echo "no LM Studio server given (LMSTUDIO_URL, LMSTUDIO_MODEL)" > "$OUT/lmstudio-$short-$q.skip"; fi
done < "$MATRIX"

"$GE" bench --report "$OUT" --title "Consumer benchmarks, $(echo "$GPU" | cut -d, -f1) ($TIER GB)" > "$OUT/README.md" || { echo "the report failed"; exit 1; }
echo "wrote $OUT/README.md"
# (the run fails where a Glyd row failed, and where engines besides Glyd were asked for and none ran a row: a page with nothing to compare is no comparison)
[ -z "$GLYD_FAILED" ] || { echo "Glyd's rows that did not run:$GLYD_FAILED"; exit 1; }
asked=0; for e in llama.cpp ollama lmstudio; do on "$e" && asked=1; done
[ $asked = 0 ] || [ "$OTHERS" -gt 0 ] || { echo "no engine besides Glyd ran a row (their .skip files say why)"; exit 1; }
