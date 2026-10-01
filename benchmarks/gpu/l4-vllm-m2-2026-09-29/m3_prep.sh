#!/bin/bash
# The vllm-plugin branch with main's v0.25.1 (~/vllm-work/glyd, its COMMIT) on the L4, each step under the box's lock:
#   lib     its library for sm_89 (build_lib.sh's flags), in ~/vllm-work/lib-v0251
#   dbg4    dbg4.sh on it: compiled with inductor deterministic, bf16 twice, Glyd fused twice, Glyd exact
#   yi      01-ai/Yi-1.5-6B-Chat (LlamaForCausalLM, Apache-2.0) downloaded, timed; check_vllm.py --saves on it
#   san     opcheck.py under compute-sanitizer: racecheck, synccheck, initcheck, memcheck
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m3prep; L=$W/lib-v0251; mkdir -p $O $W/tmp
source ~/gpuenv/cuda.sh
export HF_HOME=~/hf TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
step() { echo "== $(date -u +%T) $*"; }
if [ ! -f $L/libglyd_gpu_cuda13.so ]; then
  step "lib: $(cat $W/glyd/COMMIT) for sm_89"
  CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd); mkdir -p $L
  F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
     -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden)
  flock ~/.glyd-box.lock bash -c "nvcc ${F[*]} -c -o $L/glyd_gpu.o $W/glyd/gpu/glyd_gpu.cu && nvcc ${F[*]} -shared -Xlinker --exclude-libs,ALL -cudart static -L$CU/lib -L$CU/lib64 -o $L/new.so $L/glyd_gpu.o && mv $L/new.so $L/libglyd_gpu_cuda13.so" > $O/build.txt 2>&1; echo "exit $?"
fi
export GLYD_GPU_LIB=$L/libglyd_gpu_cuda13.so
if [ ! -f $W/m2/dbg4/summary.txt ]; then
  step "dbg4: inductor deterministic, compiled with CUDA graphs"
  flock ~/.glyd-box.lock bash $W/dbg4.sh > $O/dbg4.txt 2>&1; echo "exit $?"; cat $W/m2/dbg4/summary.txt
fi
M=01-ai/Yi-1.5-6B-Chat
if [ ! -f $O/dl-yi.txt ]; then
  step "yi: $M downloaded"
  t=$(date +%s)
  flock ~/.glyd-box.lock ~/gpuenv/bin/python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', 'tokenizer.model', 'NOTICE']))" $M > $O/dl-yi.log 2>&1
  e=$?; d=$(tail -1 $O/dl-yi.log); echo "exit $e: $d, $(du -sbL "$d" | cut -f1) bytes in $(( $(date +%s) - t )) s" | tee $O/dl-yi.txt
fi
if [ ! -f $O/check-Yi-1.5-6B-Chat/report.txt ]; then
  step "yi: check_vllm.py --saves $M"
  (cd $O && HF_HUB_OFFLINE=1 GLYD_SAVE_PYTHON=$HOME/gpuenv/bin/python flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py --saves --out $O/check-Yi-1.5-6B-Chat $M) > $O/check-Yi-1.5-6B-Chat.txt 2>&1; echo "exit $?"
  grep "PASS\|FAIL\|check_vllm:" $O/check-Yi-1.5-6B-Chat.txt | cut -c1-220
fi
for tool in racecheck synccheck initcheck memcheck; do
  [ -f $O/san-$tool.txt ] && continue
  step "san: opcheck.py under compute-sanitizer --tool $tool"
  (cd $O && OPCHECK_SHAPES="2048,2048;1024,4096" OPCHECK_MS="1,16,17,33,64,65,129,300,1024" PYTORCH_NO_CUDA_MEMORY_CACHING=1 flock ~/.glyd-box.lock timeout 2400 /usr/local/cuda-13.0/bin/compute-sanitizer --tool $tool --print-limit 20 $V/python $W/opcheck.py) > $O/san-$tool.tmp 2>&1; echo "exit $?"; mv $O/san-$tool.tmp $O/san-$tool.txt
  grep -E "ERROR SUMMARY|RACECHECK SUMMARY|opcheck: done|Hazard|Barrier" $O/san-$tool.txt | head -5
done
step done
