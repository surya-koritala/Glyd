#!/usr/bin/env bash
# One model split over two GPUs (Linear GLY-279): Qwen3.8-Flash-Next, fast mode, on GPU 0 alone and on --gpus 0,1, on a pod of two (rp_run.sh contract).
# Uploaded beside it (the pod's home): glyd-engine (a build of the branch, archs 80 86 89 90a), libglyd_gpu_cuda13.so, det.tmpl and def.tmpl (the requests, @OUT@ a run's prefix: the Strata chunk's
# teacher-forced logits as kld_engine.py asks for them and 4 rows a pass, plain and drafted decodes; det.tmpl sets promote 0 first, def.tmpl ends with kld_engine.py's decode speed), test_server.py (the
# public repo's tests/).
# ~/results: summary.txt (read it first), NAME.out (one reply a line), NAME.log (the engine's lines: the parts, places, cache hits, ms a token), gpu-NAME.txt (memory a card, every 5 s), test_server.txt; DONE.
#   D=dir of the files; GPU="NVIDIA GeForce RTX 4090" RP_GPUS=2 RP_DISK=200 MAX_MIN=75 NAME=mgpu UPLOAD="$D/glyd-engine $D/libglyd_gpu_cuda13.so $D/det.tmpl $D/def.tmpl $D/test_server.py" \
#     bash tools/runpod/rp_try.sh tools/runpod/jobs/multigpu_job.sh
set -u
R=$HOME/results; mkdir -p "$R"; cd "$HOME"
trap 'touch "$R/DONE"' EXIT
log() { echo "$(date -u +%T) $*" | tee -a "$R/summary.txt"; }
E=$HOME/glyd-engine; chmod +x "$E"; export GLYD_GPU_LIB=$HOME/libglyd_gpu_cuda13.so
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > "$R/gpus.txt" 2>&1; nvidia-smi topo -m > "$R/topo.txt" 2>&1
log "GPUs: $(tail -n +2 "$R/gpus.txt" | tr '\n' ';'); $(free -g | awk '/^Mem:/ {print $2}') GiB of RAM"
"$E" qwenget glyd/Qwen3.8-Flash-Next-hummingbird "$HOME/m" --fast > "$R/download.log" 2>&1 || { log "download failed: $(tail -c 300 "$R/download.log")"; exit 1; }
M=$HOME/m/model; P=$HOME/m/fast
# a run: NAME, the template, then the engine's flags
run() {
  local n=$1 t=$2; shift 2
  sed "s#@OUT@#$R/$n#g" "$t" > "$R/$n.req"
  (while sleep 5; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$R/gpu-$n.txt"; done) & local w=$!
  "$E" qwenserve "$M" "$P" --ctx 32768 "$@" < "$R/$n.req" > "$R/$n.out" 2> "$R/$n.log"; local rc=$?
  kill $w 2> /dev/null
  local most; most=$(awk -F', ' '{if ($2 > m[$1]) m[$1] = $2} END {for (g in m) printf "GPU %s %d MiB; ", g, m[g]}' "$R/gpu-$n.txt" 2> /dev/null)
  log "$n ($*): exit $rc, $(grep -c '"error"' "$R/$n.out") errors; most in use: $most"
  grep -E "GPU [0-9]+: layers|the GPU's places|loaded in|ms a token|in places" "$R/$n.log" | cut -c1-260 | sed 's/^/    /' >> "$R/summary.txt"
}
# one GPU and two with the same places a layer (--cache-gb: all the parts' places), the prompts in chunks (--whole 0 --no-stream: the prompt path whose sums do not depend on the order threads finish in, so that one
# GPU's own runs are the same to the byte; the default path's streamed experts are summed with atomics): the logits of two GPUs are to be the same as one's, byte for byte
for r in "one --gpus 0" "two --gpus 0,1" "p2p --gpus 0,1 --p2p"; do
  set -- $r; n=$1; shift
  run "$n" det.tmpl "$@" --cache-gb 10 --whole 0 --no-stream
done
for b in two p2p; do
  for k in r1 r4; do
    if cmp -s "$R/one.$k.f32" "$R/$b.$k.f32"; then log "$b $k logits: the same as one GPU's, byte for byte ($(stat -c %s "$R/$b.$k.f32") bytes)"; else log "$b $k logits: DIFFER from one GPU's in $(cmp -l "$R/one.$k.f32" "$R/$b.$k.f32" 2> /dev/null | wc -l) bytes"; fi
  done
  log "$b: $(python3 -c 'import json, sys; a, b = ([json.loads(l) for l in open(f)] for f in sys.argv[1:]); print(sum((x.get("ids"), x.get("lps")) == (y.get("ids"), y.get("lps")) for x, y in zip(a, b)), "of", len(a), "replies the same")' "$R/one.out" "$R/$b.out")"
done
rm -f "$R"/*.f32
# what each holds by default (the places one GPU has room for, and two), the default prompt path, and the decode speed (kld_engine.py's: 512 tokens after 64, greedy, drafts; best of 3)
run onefull def.tmpl --gpus 0
run twofull def.tmpl --gpus 0,1
rm -f "$R"/*.f32
for n in onefull twofull; do
  log "$n decode: $(python3 -c 'import json, sys; r = [json.loads(l) for l in open(sys.argv[1])][-6:]; print(round(max(511 / ((r[i + 1]["ms"] - r[i]["ms"]) / 1e3) for i in (0, 2, 4)), 1), "tok/s")' "$R/$n.out")"
done
# the HTTP server split over the two, the public repo's checks
"$E" qwenserve "$M" "$P" --ctx 32768 --port 8080 --gpus 0,1 > "$R/http.log" 2>&1 & S=$!
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health > /dev/null && break; sleep 5; done
python3 test_server.py --url http://127.0.0.1:8080 > "$R/test_server.txt" 2>&1
log "test_server.py on --gpus 0,1: $(tail -1 "$R/test_server.txt")"
kill $S; wait $S 2> /dev/null
log "done"
