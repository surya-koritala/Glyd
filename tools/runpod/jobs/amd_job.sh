#!/usr/bin/env bash
# AMD bring-up (Linear GLY-344), one session on a rented RX 7900 XTX (gfx1100): the engine's HIP build from a tar of the branch, the GPU library, fast's download, the
# engine's checks, one Strata chunk's teacher-forced logits (for a KL against the CUDA run's), tests/test_server.py, and bench/compare.py's chat tok/s and prompt times.
# Not a Runpod pod (Runpod has no Radeon cards): a Vast.ai or TensorDock instance, Ubuntu 22.04 or 24.04 with ROCm 6.4 or newer (7.x preferred), /dev/kfd and /dev/dri
# usable, at least 100 GB of disk and 64 GB of RAM. The contract is rp_run.sh's (~/results, DONE however it ends), and ~/results.tar.gz is left to copy back.
# Uploaded beside it (the instance's home), made on the PC or the laptop:
#   amd_src.tar     git archive -o amd_src.tar HEAD Cargo.toml Cargo.lock src third_party glyd-store glyd-gpu glyd-engine gpu research/lossy-2026 research/qwen-kld
#   pub.tar         tar -cf pub.tar -C ~/glyd-ibis-next tools tests/test_server.py bench/compare.py   (the public repo: the chat page and the checks)
#   strata_v1.json  ~/glyd-scratch/qwen-users/kld/strata_v1.json (its first chunk is read as kld_engine.py reads it)
#   scp -P PORT tools/runpod/jobs/amd_job.sh amd_src.tar pub.tar strata_v1.json USER@HOST: && ssh -p PORT USER@HOST 'nohup bash amd_job.sh > job.log 2>&1 &'
#   then, once ~/results/DONE is there: scp -P PORT USER@HOST:results.tar.gz . ; and destroy the instance.
# MAX_MIN (default 150): the job's own cap; a step that would not end in time is cut or skipped, and the results are tarred whatever happens. GLYD_HIP_ARCHS: the
# targets to build (default: the first gfx11 or gfx12 the box's GPU names, gfx1100 on an RX 7900 XTX). ROCM_PATH: where ROCm is (default /opt/rocm, else /opt/rocm-X.Y.Z).
# ~/results: summary.txt (read it first), system.txt, build-engine.log, build-lib.log, download.log, doctor.txt, kld.txt with chunk0.f32 and chunk0.f32.json (chunk 0's
# logits, 499 rows of 248,320 floats, and its NLL and decode tok/s), serve.log, test_server.txt, compare.txt (the prompt-time table), gpu.txt (VRAM every 5 s); DONE.
set -u
R=$HOME/results; mkdir -p "$R"; cd "$HOME"
T0=$(date +%s); MAX_MIN=${MAX_MIN:-150}
trap 'kill $(jobs -p) 2> /dev/null; tar -czf "$HOME/results.tar.gz" -C "$HOME" results 2> /dev/null; touch "$R/DONE"' EXIT
exec 3>&1  # (the job's own output: log lines go there and to summary.txt, never into a step's redirected log)
log() { echo "$(date -u +%T) $*" | tee -a "$R/summary.txt" >&3; }
left() { echo $((MAX_MIN * 60 - ($(date +%s) - T0))); }
# step NAME SECONDS CMD...: a step under a time limit, never past the cap (skipped with less than 2 minutes left), its exit status logged
step() {
    local n=$1 s=$2 l rc; shift 2
    l=$(left)
    if [ "$l" -lt 120 ]; then log "$n: skipped (the cap)"; return 99; fi
    [ "$s" -gt $((l - 60)) ] && s=$((l - 60))
    timeout "$s" "$@"; rc=$?
    log "$n: exit $rc$([ $rc = 124 ] && echo ' (timed out)')"
    return $rc
}
RP=${ROCM_PATH:-$(ls -d /opt/rocm /opt/rocm-[0-9]* 2> /dev/null | head -1)}
export PATH=$RP/bin:$HOME/.cargo/bin:$PATH LD_LIBRARY_PATH=$RP/lib:${LD_LIBRARY_PATH:-}

# ---- the box
as_root() { if [ "$(id -u)" = 0 ]; then "$@"; elif sudo -n true 2> /dev/null; then sudo "$@"; else return 1; fi; }
{
    echo "== os"; head -3 /etc/os-release; uname -r
    echo "== rocm"; cat "$RP/.info/version" 2> /dev/null; ls -d /opt/rocm* 2> /dev/null; hipcc --version 2>&1 | head -3
    echo "== gpu"; rocminfo 2>&1 | grep -E "Marketing Name|Name: +gfx|Compute Unit|Wavefront Size" | head -12; ls -l /dev/kfd /dev/dri 2>&1
    echo "== user"; id; as_root true 2> /dev/null && echo "root or sudo: yes" || echo "root or sudo: no"
    echo "== cpu, ram, disk"; nproc; free -g | head -2; df -h "$HOME" | tail -1
    echo "== tools"; for t in cc gcc python3 curl tar cargo; do printf "%s: %s\n" "$t" "$(command -v $t || echo none)"; done
} > "$R/system.txt" 2>&1
ROCM=$(cut -d- -f1 "$RP/.info/version" 2> /dev/null)
ARCH=${GLYD_HIP_ARCHS:-$(rocminfo 2> /dev/null | grep -o -m1 "gfx1[12][0-9][0-9]")}
log "box: $(rocminfo 2> /dev/null | grep -m1 'Marketing Name' | sed 's/.*Name: *//'), $ARCH, ROCm ${ROCM:-none}, $(nproc) threads, $(free -g | awk '/^Mem:/ {print $2}') GiB of RAM, $(df -h "$HOME" | awk 'END {print $4}') free"
[ -n "$ROCM" ] && [ "$(printf '%s\n' 6.4 "$ROCM" | sort -V | head -1)" = 6.4 ] || { log "stop: ROCm 6.4 or newer is needed (the engine's driver table takes hipEventRecordWithFlags)"; exit 1; }
[ -n "$ARCH" ] || { log "stop: rocminfo names no gfx11 or gfx12 GPU (set GLYD_HIP_ARCHS, and see system.txt)"; exit 1; }
[ -r /dev/kfd ] && [ -w /dev/kfd ] || { log "stop: /dev/kfd is not usable by $(id -un) (the render and video groups, or the container's --device flags)"; exit 1; }
# a C compiler for the Rust build (crates with C in them, the linker) and Python for the checks: installed where the box lets us, else ROCm's clang as the compiler
if ! command -v cc > /dev/null || ! command -v python3 > /dev/null; then
    as_root apt-get update -qq > /dev/null 2>&1 && as_root apt-get install -y -qq build-essential python3 curl ca-certificates > "$R/apt.log" 2>&1
fi
if ! command -v cc > /dev/null && [ -x "$RP/llvm/bin/clang" ]; then
    export CC=$RP/llvm/bin/clang CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$RP/llvm/bin/clang
fi
command -v python3 > /dev/null || { log "stop: no python3 and no way to install it"; exit 1; }
if ! command -v cargo > /dev/null; then
    step rustup 600 sh -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1' > "$R/rustup.log" 2>&1 || exit 1
fi

# ---- the builds: the engine (its kernels for $ARCH), then fast's download beside the library's build
log "source: amd_src.tar sha256 $(sha256sum amd_src.tar | cut -c1-16)"
rm -rf src pub && mkdir -p src pub && tar -xf amd_src.tar -C src && tar -xf pub.tar -C pub || { log "stop: the tars did not unpack"; exit 1; }
cd "$HOME/src/glyd-engine"
step build-engine 1800 env GLYD_HIP=1 GLYD_HIP_ARCHS="$ARCH" HIPCC="$(command -v hipcc)" cargo build --release > "$R/build-engine.log" 2>&1 || { tail -30 "$R/build-engine.log" >> "$R/summary.txt"; exit 1; }
cd "$HOME"
E=$HOME/src/target/release/glyd-engine
("$E" qwenget glyd/Qwen3.8-Flash-Next-hummingbird "$HOME/m" --fast > "$R/download.log" 2>&1; echo $? > "$R/download.rc") &
DL=$!
step build-lib 2400 env HIPCC="$(command -v hipcc)" GLYD_HIP_ARCHS="$ARCH" bash "$HOME/src/gpu/build_lib_hip.sh" "$HOME/lib" > "$R/build-lib.log" 2>&1 || { tail -30 "$R/build-lib.log" >> "$R/summary.txt"; exit 1; }
export GLYD_GPU_LIB=$HOME/lib/libglyd_gpu_hip.so
HF_HUB_OFFLINE=1 timeout 120 "$E" doctor > "$R/doctor.txt" 2>&1
log "doctor: $(grep -E '^  (GPU|GPU library) ' "$R/doctor.txt" | tr -s ' ' | tr '\n' ';')"
while kill -0 $DL 2> /dev/null; do
    if [ "$(left)" -lt 1800 ]; then kill $DL; log "download: stopped (the cap: 30 minutes kept for the runs)"; exit 1; fi
    sleep 10
done
[ "$(cat "$R/download.rc" 2> /dev/null)" = 0 ] || { log "download failed: $(tail -c 300 "$R/download.log")"; exit 1; }
log "download: $(du -sh "$HOME/m" | cut -f1) in $HOME/m"
M=$HOME/m/model; P=$HOME/m/fast

# ---- on the GPU
(while sleep 5; do echo "$(date -u +%T) $( (amd-smi metric --mem-usage 2> /dev/null || rocm-smi --showmeminfo vram 2> /dev/null) | grep -iE 'used' | tr -s ' ' | tr '\n' ' ')" >> "$R/gpu.txt"; done) &
# chunk 0 of the Strata texts, teacher-forced as kld_engine.py reads every chunk (its first half the prompt, the rest one row a pass), and its decode speed (512 tokens after 64, best of 3)
python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); json.dump({"docs": d["docs"][:1], "names": d.get("names", [])[:1]}, open(sys.argv[2], "w"))' strata_v1.json chunk0.json
step kld 1800 python3 "$HOME/src/research/qwen-kld/kld_engine.py" --tokens chunk0.json --out "$R/chunk0.f32" -- "$E" qwenserve "$M" "$P" > "$R/kld.txt" 2>&1
log "kld: $(grep -E '^chunk|^decode' "$R/kld.txt" | tr '\n' ' ')"
# the HTTP server: the public repo's checks and the prompt-time table (compare.py --speed-only: a short chat, a ~4K and a ~30K prompt)
"$E" qwenserve "$M" "$P" --ctx 32768 --port 8080 --page "$HOME/pub/tools" > "$R/serve.log" 2>&1 &
S=$!
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8080/health > /dev/null && break; kill -0 $S 2> /dev/null || break; sleep 5; done
if curl -sf http://127.0.0.1:8080/health > /dev/null; then
    step test_server 1200 python3 "$HOME/pub/tests/test_server.py" --url http://127.0.0.1:8080 > "$R/test_server.txt" 2>&1
    log "test_server.py: $(tail -1 "$R/test_server.txt")"
    (cd "$R" && step compare 2400 python3 "$HOME/pub/bench/compare.py" --url http://127.0.0.1:8080 --model qwen3.8-flash-next --speed-only --note "RX 7900 XTX, HIP $ARCH, fast" > "$R/compare.txt" 2>&1)
    grep -E '^\| ' "$R/compare.txt" | sort -u >> "$R/summary.txt"
else
    log "the server did not come up: $(tail -c 400 "$R/serve.log")"
fi
kill $S 2> /dev/null; wait $S 2> /dev/null
grep -E "loaded in|places|ms a token|error|panick" "$R/serve.log" | cut -c1-260 | head -20 | sed 's/^/    /' >> "$R/summary.txt"
log "done in $((($(date +%s) - T0) / 60)) min"
