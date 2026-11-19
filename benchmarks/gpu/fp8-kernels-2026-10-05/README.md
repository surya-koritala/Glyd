# Lossless FP8 (e4m3) kernels: checks and unpack speed (Runpod, 2026-10-05)

The GPU runs behind `gpu/fp8/README.md`'s "Checks" and "Speed" (branch `fp8-lossless`, Task 2 of the FP8 plan): the GPU's pack and
unpack bit for bit against the numpy reference on every case and on the 50 real samples of five 2026 FP8 checkpoints, strides,
lists, CUDA graphs, compute-sanitizer; and the unpack's bandwidth. Report: `report.md` (the worker's).

| GPU | unpack (7168 x 18432) | e4m3 written | with the packed bytes read | pack |
| :--- | ---: | ---: | ---: | ---: |
| RTX 4090 (`f8k-ada1`) | 265.3 us | 498 GB/s | 912 GB/s of 1,008 | 0.5 ms |
| H100 80GB HBM3 (`f8k-hop1`) | 178.0 us | 742 GB/s | 1,359 GB/s of 3,350 | 0.4 ms |
| H100, the dropped variant (`f8k-hop2`) | 189 us | 699 GB/s | | |

```sh
(cd f8k && ./mktar.sh)     # f8k_src.tar: the worktree's gpu/ and glyd-gpu-py/, COMMIT
GPU="NVIDIA GeForce RTX 4090" MAX_MIN=40 NAME=f8k-ada1 UPLOAD="f8k_job.sh f8k_src.tar" bash rp_run.sh f8k_ada.sh
GPU="NVIDIA H100 80GB HBM3"   MAX_MIN=40 NAME=f8k-hop1 UPLOAD="f8k_job.sh f8k_src.tar" bash rp_run.sh f8k_hop.sh   # hop2: the variant's commit
```

3 pods, 13 pod minutes. Files: `f8k-*/` (each run's results: `job.log`, the checks' logs, `gpu.txt`), `job/` (the job, its two
wrappers, `mktar.sh`, `COMMIT`, the launcher's output `rp_run-*.log`).
