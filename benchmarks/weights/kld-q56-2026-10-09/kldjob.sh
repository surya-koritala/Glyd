# The KL-divergence study of Qwen3.8-27B's weight representations on one 80 GB GPU (research/lossy-2026/kld_study.py).
# Steps go to ~/k/steps.txt; results to ~/k/out/results.jsonl.
set -u
K=~/k; O=$K/out; S=$K/steps.txt; mkdir -p $O; : > $S
step() { local name=$1; shift; local t=$(date +%s); "$@" > $K/$name.log 2>&1; echo "$name: rc=$? $(( $(date +%s) - t )) s" >> $S; }
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
step venv bash -c 'python3 -m venv $HOME/k/venv && $HOME/k/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && $HOME/k/venv/bin/pip install -q "transformers==5.18.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2" pyarrow flash-linear-attention'
PY=$K/venv/bin/python; HF=$K/venv/bin/hf
step selftest bash -c "cd $K/src && $PY kld_study.py selftest"
step download bash -c "$HF download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir $K/bf16 > /dev/null && $HF download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --local-dir $K/fp8 > /dev/null"
cd $K/src
step ref $PY kld_study.py ref $K/bf16 $O
step base $PY kld_study.py score $K/bf16 $O bf16 frac:3 frac:2
step groups $PY kld_study.py score $K/bf16 $O frac:2@embed=3 frac:2@head=3 frac:2@attn_qkv=3 frac:2@attn_o=3 frac:2@lin_in=3 frac:2@lin_out=3 frac:2@mlp_gate_up=3 frac:2@mlp_down=3
step ints $PY kld_study.py score $K/bf16 $O int:6:32:mse int:5:32:mse int:5:64:mse int:4:32:mse int:6:32:absmax int:5:32:absmax
step fp8 $PY kld_study.py score $K/bf16 $O fp8 --fp8-dir $K/fp8
echo done >> $S
