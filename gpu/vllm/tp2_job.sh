#!/usr/bin/env bash
# The embedding and the LM head packed under vLLM (v0.28) over two GPUs, and the LM head's product on this GPU type: unattended, on an x86_64 or
# aarch64 host with two NVIDIA GPUs (an A100 SXM4 or H100 SXM pair is the one that matters: the smallest layout is the slow one there today) whose driver
# runs CUDA 13 (580 or newer). About 26 minutes, 29 at most: no step starts past minute 25 and every step is bounded. In ~: this, and full33_src.tar (the
# full33 branch's tree with its COMMIT: gpu/ and glyd-gpu-py/). Its steps, in order:
#   env    uv; a venv with vLLM (vllm==0.30.0, from PyPI) and pip's nvcc for torch's CUDA; the models download meanwhile (Qwen3-8B's head, Qwen3-4B), timed
#   build  the library for this GPU alone (build_lib.sh's flags); glyd_gpu runs from the tree's sources (PYTHONPATH), vLLM's plugin entry point from a stub
#   sweep  gpu/vllm/head_sweep.py on Qwen3-8B's head (151,936 x 4,096) on the first GPU: 1 to 1,024 rows and more, the smallest and the 12-bit layout against
#          bf16's F.linear and against the matrix decoded in 64 MB pieces then cuBLAS; the embedding's lookup for 1 to 8,192 tokens
#   tiny   gpu/vllm/tables_check.py --tp 2 on two tiny random Qwen3-architecture models (tied and untied, a vocabulary of 32,001 that vLLM pads to 32,064 and
#          shards into 16,032-row shards, the second loading 15,969 rows): each table packed, every lookup the checkpoint's row bit for bit on each rank,
#          each product within 1e-2, bf16's tokens and prompt logprobs, exact mode bit for bit (the config tied with a copy in the checkpoint, and the
#          vocabularies of 32,000 and 32,064, are checked on one GPU: tables_check.py)
#   real   gpu/vllm/check_vllm.py --brief --tp 2 on Qwen3-4B (tied): the weights and KV cache against bf16's over two GPUs, the tables, exact eager
# results/summary.txt rewritten after every step; results/DONE from the exit trap however the job ends.
#   bash ~/tp2_job.sh                              (every step)
#   TJ_STEPS=sweep,tiny bash ~/tp2_job.sh          (some of them)
# Env: TJ_STEPS (sweep,tiny,real), TJ_TP (2: the GPUs the tiny and real steps use; 1 on one GPU, as a dry run), TJ_BUDGET (1500 s), TJ_END (1740 s),
# TJ_PY (a Python that has vLLM, torch and safetensors: no venv is made), TJ_LIB (a built library: no build), FILES (~), R (~/results), W (~/tp2w),
# HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/tp2w}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W/tmp"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${TJ_BUDGET:-1500}
END=${TJ_END:-1740}
STEPS=${TJ_STEPS:-sweep,tiny,real}
TP=${TJ_TP:-2}
echo "full33 two-GPU job: about 26 minutes planned (env and build 8, head sweep 3, tiny models over $TP GPUs 7, Qwen3-4B over $TP GPUs 8), 29 at most: no step starts past minute $(( BUDGET / 60 )), every step is bounded"
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
left() { [ "$(el)" -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
want() { case ",$STEPS," in *",$1,"*) return 0 ;; esac; return 1; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() {  # results/summary.txt from what is there
  { cat "$R/machine-short.txt" 2> /dev/null; cat "$R/steps.txt" 2> /dev/null; echo
    for f in "$R"/sweep.txt "$R"/tables/report.txt "$R"/real/report.txt; do
      [ -f "$f" ] || continue
      echo "== ${f#$R/}"; cat "$f"; echo
    done
  } > "$R/summary.tmp" 2>&1; mv "$R/summary.tmp" "$R/summary.txt"
}
done_() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
fail() { done_ "FAIL: $*"; exit 1; }

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; lscpu | grep -E "Model name|Architecture"; nproc
  free -g | head -2; df -h "$HOME" | tail -1; date -u; echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH  # the host's CUDA libraries never ahead of the venv's own
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NG=$(nvidia-smi -L | grep -c "^GPU")
[ "$NG" -ge "$TP" ] || fail "TJ_TP $TP, but $NG GPUs here"
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/full33_src.tar" || fail "no full33_src.tar in $FILES"
echo "$NG x $NAME (compute $CC, sm_$ARCH, $MIB MiB), $(uname -m) host, $(nproc) CPUs; tree $(cat "$W/src/COMMIT"); steps $STEPS; TP $TP" | tee "$R/machine-short.txt"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ] && echo "WARNING: other processes on the GPU" | tee -a "$R/machine-short.txt"
done_ "machine"

step "uv, then the downloads (in the background, one after another)"
export HF_HOME=${HF_HOME:-$HOME/hf} TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=${UV_CACHE_DIR:-$W/uvcache}
if [ -z "${TJ_PY:-}" ]; then
  UV=$(command -v uv || ls "$HOME/tools/uv/uv" "$W/uv/uv" 2> /dev/null | head -1)
  if [ -z "$UV" ]; then
    mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv
  fi
  [ -x "$UV" ] || fail "no uv"
  [ -x "$W/dl/bin/python" ] || ( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet ) > "$R/log/dl-env.txt" 2>&1 || fail "no download env (log/dl-env.txt)"
  DLPY=$W/dl/bin/python
else
  DLPY=$TJ_PY
fi
dl() {  # REPO [PATTERN]: its snapshot (only the PATTERN's files, and the json), log/dl-NAME.txt its time and size
  local n t; n=$(basename "$1"); t=$(date +%s)
  timeout 1200 "$DLPY" -c "
import json, os, sys
from huggingface_hub import hf_hub_download, snapshot_download
repo = sys.argv[1]
if len(sys.argv) > 2:  # (only the shard that holds the LM head: the sweep reads that one matrix)
    idx = json.load(open(hf_hub_download(repo, 'model.safetensors.index.json')))['weight_map']
    snapshot_download(repo, allow_patterns=['*.json', idx['lm_head.weight']])
else:
    snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
" "$@" > "$W/$n.out" 2> "$R/log/dl-$n.txt"
  local e=$?; echo "exit $e: $(du -sbL "$HF_HOME/hub/models--${1/\//--}" 2> /dev/null | cut -f1) bytes in $(( $(date +%s) - t )) s (at +$(el) s)" >> "$R/log/dl-$n.txt"
}
( want sweep && dl Qwen/Qwen3-8B head; want real && dl Qwen/Qwen3-4B ) &
got() {  # REPO: 0 once its download is done and whole (1 if it failed, or past the budget)
  local n; n=$(basename "$1")
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 3; done
  grep -q "^exit 0" "$R/log/dl-$n.txt"
}

if [ -n "${TJ_PY:-}" ]; then
  PY=$TJ_PY
  done_ "env: $TJ_PY"
else
  step "env: vLLM from PyPI (vllm==0.30.0) and nvcc"
  t=$(date +%s)
  if [ -x "$W/venv/bin/python" ] && "$W/venv/bin/python" -c "import vllm" 2> /dev/null; then
    echo "an earlier job's venv, $W/venv" > "$R/log/env-vllm.txt"
  else
    { rm -rf "$W/venv" && "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0"; } > "$R/log/env-vllm.txt" 2>&1 || fail "vLLM did not install (log/env-vllm.txt)"
  fi
  PY=$W/venv/bin/python
  MM=$("$PY" -c "import torch; print(torch.version.cuda)")
  "$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || fail "nvcc did not install (log/env-nvcc.txt)"
  CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
  mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
  export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
  done_ "env: vLLM and nvcc in $(( $(date +%s) - t )) s"
fi
"$PY" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__, '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1 || fail "no vLLM on the GPU ($(tail -2 "$R/env.txt" | tr '\n' ' '))"
cat "$R/env.txt"

# glyd_gpu from the tree's sources; vLLM finds its plugin by an entry point, which a stub's metadata gives
mkdir -p "$W/stub/glyd_gpu-0.0.0.dist-info"
printf 'Metadata-Version: 2.1\nName: glyd-gpu\nVersion: 0.0.0\n' > "$W/stub/glyd_gpu-0.0.0.dist-info/METADATA"
printf '[vllm.general_plugins]\nglyd = glyd_gpu.vllm_entry:register\n' > "$W/stub/glyd_gpu-0.0.0.dist-info/entry_points.txt"
export PYTHONPATH=$W/stub:$W/src/glyd-gpu-py PYTHONSAFEPATH=1

step "build: the library for sm_$ARCH"
t=$(date +%s)
if [ -n "${TJ_LIB:-}" ]; then
  export GLYD_GPU_LIB=$TJ_LIB
else
  CU=${CUDA_HOME:-}
  MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
  F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
     -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
  mkdir -p "$W/lib"; rm -f "$W/lib/libglyd_gpu_cuda$MAJOR.so"
  { timeout "$(tmo 900)" nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/libglyd_gpu_cuda$MAJOR.so" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
  [ -f "$W/lib/libglyd_gpu_cuda$MAJOR.so" ] || fail "the library did not build (log/build.txt)"
  export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so
fi
"$PY" -c "from glyd_gpu import _lib, kernels as g; g.lib(); print('library: C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" >> "$R/env.txt" 2>&1 || fail "the library does not load ($(tail -1 "$R/env.txt"))"
done_ "build: sm_$ARCH in $(( $(date +%s) - t )) s ($(tail -1 "$R/env.txt"))"

if want sweep && left; then
  step "sweep: head_sweep.py on Qwen3-8B's head, one GPU (waiting for its download)"
  got Qwen3-8B || fail "no Qwen3-8B head: $(tail -2 "$R/log/dl-Qwen3-8B.txt" 2> /dev/null | tr '\n' ' ')"
  t=$(date +%s)
  ( cd "$R" && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 timeout "$(tmo 420)" "$PY" "$W/src/gpu/vllm/head_sweep.py" Qwen/Qwen3-8B --rows 1,2,4,8,16,32,64,65,128,256,512,768,1024,2048 --tokens 1,8,64,256,1024,4096 ) > "$R/sweep.txt" 2>&1
  e=$?; done_ "sweep: exit $e in $(( $(date +%s) - t )) s ($(head -1 "$R/sweep.txt"))"
fi
if want tiny && left; then
  step "tiny: tables_check.py --tp $TP on two tiny models"
  t=$(date +%s)
  ( cd "$R" && HF_HUB_OFFLINE=1 timeout "$(tmo 600)" "$PY" "$W/src/gpu/vllm/tables_check.py" --tp "$TP" --out "$R/tables" to uo ) > "$R/tables.txt" 2>&1
  e=$?; mkdir -p "$R/tables"; [ -f "$R/tables/report.txt" ] || cp "$R/tables.txt" "$R/tables/report.txt"
  done_ "tiny: exit $e in $(( $(date +%s) - t )) s ($(tail -1 "$R/tables.txt"))"
fi
if want real && left; then
  step "real: check_vllm.py --brief --tp $TP on Qwen3-4B (waiting for its download)"
  got Qwen3-4B || fail "no Qwen3-4B: $(tail -2 "$R/log/dl-Qwen3-4B.txt" 2> /dev/null | tr '\n' ' ')"
  t=$(date +%s)
  ( cd "$R" && HF_HUB_OFFLINE=1 timeout "$(tmo 780)" "$PY" "$W/src/gpu/vllm/check_vllm.py" --brief --tp "$TP" --out "$R/real" Qwen/Qwen3-4B ) > "$R/real.txt" 2>&1
  e=$?; mkdir -p "$R/real"; [ -f "$R/real/report.txt" ] || cp "$R/real.txt" "$R/real/report.txt"
  done_ "real: exit $e in $(( $(date +%s) - t )) s ($(tail -1 "$R/real.txt"))"
fi
for f in "$R"/log/dl-*.txt; do [ -f "$f" ] && echo "download $(basename "$f" .txt): $(tail -1 "$f")"; done | grep -v "dl-env" >> "$R/steps.txt"
step "done in $(el) s"
done_ "done"
