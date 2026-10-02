# vLLM with `--quantization glyd` over two H100s: Qwen2.5-72B-Instruct (2026-10-02)

`vllm bench serve`, bf16 against Glyd, on Qwen2.5-72B-Instruct with tensor parallel over two NVIDIA H100 SXM, from v0.26.0's
released code: one unattended job (`big72_job.sh`, 1,125 s) on one instance.

## Setup

- **Machine:** Lambda `gpu_2x_h100_sxm5`: 2x NVIDIA H100 80GB HBM3 SXM (81,559 MiB each, 700 W, 1,980 MHz at most, NVLink
  between them), driver 580.126.20, Xeon Platinum 8480+ (52 CPUs), x86_64 (`machine.txt`).
- **Software:** Glyd v0.26.0, the tree at 10e8caea (C API 7), with the library built there for sm_90a; vLLM 0.30.0 from PyPI
  (torch 2.13.0+cu130, transformers 5.18.0), nvcc 13.0 (`env.txt`).
- **Model:** Qwen/Qwen2.5-72B-Instruct (145,424,086,864 bytes), downloaded in 129 s with no token (`log/`).
- **Servers:** Glyd's, then bf16's, one at a time, each started on an empty compile cache with `--max-model-len 4096
  --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --max-num-seqs 128`.
- **Load:** the random dataset, 1,024 tokens in and 256 out, `--ignore-eos`, `--seed 0`: 64 requests at 1 request a second,
  then 192 sent at once. One run each.
- **Glyd's layout:** the 12-bit one (`best_layout` on Hopper; `glyd: mma12 layout` in the server's log).

## Results (`bench-Qwen2.5-72B-Instruct/`)

|      | Weights, each GPU | KV cache memory, each GPU |                KV cache | Requests of 4,096 tokens at once |
| :--- | ----------------: | ------------------------: | ----------------------: | -------------------------------: |
| bf16 |         67.80 GiB |                  1.34 GiB |            8,800 tokens |                             2.15 |
| Glyd |         52.16 GiB |                 16.99 GiB | 111,328 tokens (12.65x) |                            27.18 |

With the servers up, `nvidia-smi` shows 73,621 MiB in use on each GPU with Glyd and 73,649 MiB with bf16, of 81,559
(`kv-glyd.txt`, `kv-bf16.txt`).

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | SM clock, temperature |
| -----------: | :--- | ---------: | --------------: | -------------------: | -------------------: | :-------------------- |
|            1 | bf16 |       0.76 |           193.4 |       2,948 / 11,701 |          30.9 / 40.0 | 1980 MHz, 49 C        |
|            1 | glyd |       0.90 |           231.3 |            245 / 509 |          29.0 / 32.5 | 1980 MHz, 54 C        |
|          inf | bf16 |       0.87 |           223.5 |    105,775 / 212,657 |          31.5 / 51.3 | 1980 MHz, 55 C        |
|          inf | glyd |       4.24 |          1085.9 |      12,041 / 29,751 |         71.9 / 131.9 | 1950 MHz, 58 C        |

TTFT is the time to the first token, TPOT the time per output token after it. Ratios are Glyd's over bf16's, from the runs'
JSON before rounding (so 0.90 over 0.76 reads 1.20x), as `summary.txt` prints them:

| Glyd against bf16 | Rate 1, 64 requests | Saturated, 192 requests |
| :---------------- | ------------------: | ----------------------: |
| Requests/s        |               1.20x |                   4.86x |
| TTFT mean         |               0.08x |                   0.11x |
| TPOT mean         |               0.94x |                   2.28x |

The KV cache's ratio is 12.65x and the weights' 0.77x.

- **Capacity:** each GPU holds 52.16 GiB of weights against 67.80, and vLLM's KV cache is 111,328 tokens against
  8,800 (12.65x): room for 27.18 requests of 4,096 tokens at once against 2.15.
- **At 1 request a second:** 1.20x the requests a second, the first token in 245 ms against
  2,948 (0.08x), each token 0.94x the time. In the servers' logs (every 10 s)
  bf16's KV cache was 54.6 to 98.9% used with up to 8 requests waiting; Glyd's was at most 8.3% used, with none waiting.
- **Saturated:** 4.86x the requests a second (4.24 against 0.87), 1,085.9 against 223.5 output tokens a
  second, the first token 0.11x the time. bf16 ran 6 to 8 requests at once with up to 185 waiting;
  Glyd ran 56 to 103, with up to 95 waiting.
- **Slower:**
  - Each token took 2.28x the time saturated (71.9 against 31.5 ms), with 56 to 103 requests running at once
    against 6 to 8.
  - A server started on an empty compile cache took about 409 s with Glyd and about 95 s with bf16 (`summary.txt`).
- **The prefix cache:** both servers ran with vLLM's prefix caching on, its default, and both runs use `--seed 0`. Glyd's
  server log shows its hit rate at 0.0% through the rate-1 run and 24.1% in its last line, after the saturated run; bf16's
  shows 0.0% throughout (its KV cache, 8,800 tokens, holds fewer tokens than the 64 earlier prompts).

## Reproduce

With two H100 80 GB, a driver of 580 or newer, and vLLM 0.30.0 with Glyd v0.26.0 (`pip install "glyd[vllm]==0.26.0"`), one
mode at a time (bf16: the same without `--quantization glyd`, and `bf16-rate...` in the file names):

```bash
# the server, on an empty compile cache
VLLM_CACHE_ROOT=$(mktemp -d) vllm serve Qwen/Qwen2.5-72B-Instruct --quantization glyd \
  --max-model-len 4096 --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --port 8012 \
  --max-num-seqs 128 --compilation-config '{"max_cudagraph_capture_size":128}'

# once it answers: 64 requests at 1 a second, then 192 at once
for r in "1 64" "inf 192"; do set -- $r
  vllm bench serve --backend vllm --model Qwen/Qwen2.5-72B-Instruct --port 8012 --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --ignore-eos --num-prompts $2 --request-rate $1 --seed 0 \
    --save-result --result-dir results --result-filename glyd-rate$1.json \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99
done
```

`gpu/vllm/bench_serve.sh` runs both for a mode, and `gpu/vllm/bench_summary.py results` prints the tables from the files.
The job did this on the instance: `big72_job.sh` (as it ran) installs vLLM and nvcc in a venv, builds the library for the GPU
from v0.26.0's tree and installs its `glyd` package, downloads the model, and runs `bench_serve.sh` for Glyd and then for
bf16. It reads `~/big72_src.tar`, v0.26.0's tree as a tar with its commit id in `COMMIT`:

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
git archive -o ~/big72_src.tar v0.26.0
echo "10e8caea (v0.26.0)" > COMMIT && tar -rf ~/big72_src.tar COMMIT
bash benchmarks/gpu/h100x2-qwen2.5-72b-2026-10-02/big72_job.sh
```

## Files

- `bench-Qwen2.5-72B-Instruct.txt`: the console. `bench-Qwen2.5-72B-Instruct/`: for each mode and rate, the server's log
  (`serve-MODE.txt`), `kv-MODE.txt`, `MODE-rateR.json`, `bench-MODE-rateR.txt` and `smi-MODE-rateR.csv` (nvidia-smi's
  temperature, SM clock and power draw each second, for both GPUs), then `summary.txt`.
- `summary.txt`: the job's table and ratios. `steps.txt`, `machine.txt`, `machine-short.txt`, `env.txt`, `job.txt` (`job.log`
  as written, renamed because the repository ignores `*.log`), `log/` (the model's download, the environment, the library's
  build).
- `big72_job.sh`: the job as it ran.
