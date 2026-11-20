cd ~/glyd-scratch/ttft; exec 9> ~/glyd-scratch/gpu.lock; flock 9
while docker ps --format "{{.Image}}" | grep -q nvidia/cuda; do sleep 5; done
docker run --rm --gpus all -v $PWD/out:/out -v $HOME/glyd-demo/hf:/hf -v $HOME/.cache/glyd:/root/.cache/glyd -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e LD_LIBRARY_PATH=/out -e GLYD_GPU_LIB=/out/libglyd_gpu_cuda13.so nvidia/cuda:13.0.3-devel-ubuntu24.04 bash -c '/usr/local/cuda/bin/ncu --kernel-name regex:"gemm3|gemm_silu|unpack|gemm" --launch-skip 60 --launch-count 12 --csv --metrics gpu__time_duration.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__grid_size,sm__warps_active.avg.pct_of_peak_sustained_active /out/glyd-engine-new bench Qwen/Qwen3-8B --prompt 2000 --new 2 --ctx 4096 --chunk 2048 --reps 1' 2>&1 | grep -E '^"[0-9]+"' | python3 -c "
import sys,csv
rows=list(csv.reader(sys.stdin))
cur={}
for r in rows:
    k=(r[0],r[4][:60]) ; cur.setdefault(k,{})[r[-3]]=(r[-1],r[-2])
for k,v in cur.items():
    print(k[1], {m.split('.')[0][:30]:(val) for m,(val,u) in v.items()})
"
