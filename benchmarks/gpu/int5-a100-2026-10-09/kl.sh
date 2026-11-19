# After the benchmarks: the KL study (research/lossy-2026/kld_study.py) of the five-bit layout as packed (bf16 scale) and the best-fit
# bf16 scales of five and six bits, against BF16 over the pinned WikiText-2 positions; the checkpoint from the fresh check download.
set -u
P=~/i5/kl-steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > ~/i5/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
step venv bash -c "python3 -m venv ~/i5/venv && ~/i5/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && ~/i5/venv/bin/pip install -q \"transformers==5.18.0\" \"accelerate>=1.1\" \"safetensors>=0.8\" \"huggingface_hub>=1.5,<2\" pyarrow flash-linear-attention"
cd ~/i5/src/research/lossy-2026
step selftest ~/i5/venv/bin/python kld_study.py selftest
until grep -q "^done2" ~/i5/steps.txt 2>/dev/null; do sleep 15; done
M=$(ls -d /root/fresh3/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)
step ref ~/i5/venv/bin/python kld_study.py ref $M ~/i5/out
step score ~/i5/venv/bin/python kld_study.py score $M ~/i5/out int:5:32:absmax:bf16 int:5:32:mse:bf16 int:6:32:mse:bf16
echo done >> $P
