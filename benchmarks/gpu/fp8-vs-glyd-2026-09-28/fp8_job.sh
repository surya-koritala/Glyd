# bf16 against FP8 and Glyd on Qwen3-4B-Instruct-2507, one mode a process, on the box; then the table. Models from
# ~/p1/hf (the FP8 release downloaded there, removed at the end). Each run waits for a GPU no other process holds.
set -x
H=~/p5fp8; mkdir -p $H/out
export HF_HOME=~/p1/hf TMPDIR=~/p1/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/p3final/wv/bin/python
$PY -c "import datasets" 2> /dev/null || ~/p3final/uvd/uv pip install --python $PY datasets > $H/out/install.txt 2>&1
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done; }
for m in bf16 glyd-exact glyd fp8; do
  idle; ( cd $H && timeout 40m $PY fp8_compare.py $m out/$m.pt ) > $H/out/$m.txt 2>&1; echo "exit $?" >> $H/out/$m.txt
done
$PY $H/fp8_compare.py --compare $H/out > $H/out/table.txt 2>&1
rm -rf ~/p1/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8
touch $H/out/DONE
