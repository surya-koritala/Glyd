# Where Hopper's prompts got slower: the same H100, prompts of Qwen3-8B and Qwen3-32B through the library built at four
# points (ebc8f91 measured by its author; bc146c3 its merge of main; 53b1659 main after #42, with review 2's fixes;
# 4fd327d main now), bf16 once. Results in ~/results; DONE at the end.
set -x
R=~/results; mkdir -p $R
source ~/gpuenv/cuda.sh
export HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $R/clocks-start.txt 2>&1
( hf download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B > /dev/null 2>&1; hf download Qwen/Qwen3-32B --local-dir ~/models/Qwen3-32B > /dev/null 2>&1 ) &
for r in ebc8f91 bc146c3 53b1659 4fd327d; do mkdir -p ~/b/$r && tar -C ~/b/$r -xf ~/src-$r.tar && ( cd ~/b/$r/gpu && bash build_lib.sh ~/b/$r/lib > $R/build-$r.txt 2>&1 ); done
wait
nvidia-smi dmon -s pc -d 5 > $R/dmon.txt 2>&1 & DMON=$!
cd ~/b/4fd327d/gpu && timeout 20m python e2e.py ~/models/Qwen3-8B --format auto --fused --merge --baseline --tokens 8 --batch 1 --prefill 512,1024,2048 > $R/bf16-8b.txt 2>&1
cd ~/b/4fd327d/gpu && timeout 25m python e2e.py ~/models/Qwen3-32B --format auto --fused --merge --baseline --tokens 8 --batch 1 --prefill 512,2048 > $R/bf16-32b.txt 2>&1
for pass in 1 2; do for r in ebc8f91 bc146c3 53b1659 4fd327d; do
  L=$(ls ~/b/$r/lib/libglyd_gpu_cuda13.so)
  ( cd ~/b/$r/gpu && PYTHONPATH=~/b/$r/bindings/python GLYD_GPU_LIB=$L timeout 15m python e2e.py ~/models/Qwen3-8B --format auto --fused --merge --tokens 8 --batch 1 --prefill 512,1024,2048 ) > $R/g8b-$r-$pass.txt 2>&1
  ( cd ~/b/$r/gpu && PYTHONPATH=~/b/$r/bindings/python GLYD_GPU_LIB=$L timeout 20m python e2e.py ~/models/Qwen3-32B --format auto --fused --merge --tokens 8 --batch 1 --prefill 512,2048 ) > $R/g32b-$r-$pass.txt 2>&1
done; done
kill $DMON
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > $R/clocks-end.txt 2>&1
touch $R/DONE
