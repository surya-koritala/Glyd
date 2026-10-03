# vLLM with `--quantization glyd` over two H100s: Qwen2.5-72B-Instruct (2026-10-03)

`vllm bench serve`, bf16 against Glyd, on Qwen2.5-72B-Instruct with tensor parallel over two NVIDIA H100 SXM: one
unattended job (`big72_job.sh`, 1,053 s) on one instance, on v0.26.0's tree with the bench scripts of 6a82f3bc.

## Setup

- **Machine:** Lambda `gpu_2x_h100_sxm5`: 2x NVIDIA H100 80GB HBM3 SXM (81,559 MiB each, 700 W, 1,980 MHz at most, NVLink
  between them), driver 580.126.20, Xeon Platinum 8480+ (52 CPUs), x86_64 (`machine.txt`).
- **Tree:** Glyd v0.26.0 (10e8caea, C API 7) with `gpu/vllm/bench_serve.sh` and `gpu/vllm/bench_summary.py` of 6a82f3bc,
  which is in main since #76 (a seed for each pass, and the prefix cache's hit rate in the summary). The job's `COMMIT` file
  says so, in the first line of `summary.txt`; every other file the job reads is v0.26.0's. The library was built there for
  sm_90a.
- **Software:** vLLM 0.30.0 from PyPI (torch 2.13.0+cu130, transformers 5.18.0), nvcc 13.0 (`env.txt`).
- **Model:** Qwen/Qwen2.5-72B-Instruct (145,424,086,864 bytes), downloaded in 125 s with no token (`log/`).
- **Servers:** Glyd's, then bf16's, one at a time, each started on an empty compile cache with `--max-model-len 4096
  --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --max-num-seqs 128`.
- **Load:** the random dataset, 1,024 tokens in and 256 out, `--ignore-eos`: 64 requests at 1 request a second, then 192
  sent at once. Each pass draws its prompts with a seed of its own (0, then 1), the same for both servers. One run each.
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
|            1 | bf16 |       0.76 |           194.8 |       2,802 / 11,249 |          30.6 / 39.7 | 1980 MHz, 58 C        |
|            1 | glyd |       0.90 |           231.3 |            245 / 512 |          29.0 / 32.6 | 1980 MHz, 59 C        |
|          inf | bf16 |       0.88 |           225.9 |    104,694 / 210,482 |          31.2 / 50.8 | 1980 MHz, 61 C        |
|          inf | glyd |       3.59 |           918.4 |      18,737 / 37,592 |         79.3 / 134.2 | 1590 MHz, 65 C        |

TTFT is the time to the first token, TPOT the time per output token after it. Ratios are Glyd's over bf16's, from the
runs' JSON before rounding (so 0.90 over 0.76 reads 1.19x), as `summary.txt` prints them:

| Glyd against bf16 | Rate 1, 64 requests | Saturated, 192 requests |
| :---------------- | ------------------: | ----------------------: |
| Requests/s        |               1.19x |                   4.07x |
| TTFT mean         |               0.09x |                   0.18x |
| TPOT mean         |               0.95x |                   2.54x |

The KV cache's ratio is 12.65x and the weights' 0.77x.

- **Capacity:** each GPU holds 52.16 GiB of weights against 67.80, and vLLM's KV cache is 111,328 tokens against 8,800
  (12.65x): room for 27.18 requests of 4,096 tokens at once against 2.15.
- **At 1 request a second:** 1.19x the requests a second, the first token in 245 ms against 2,802 (0.09x), each token
  0.95x the time. In the servers' logs (every 10 s) bf16's KV cache was 40.6 to 95.4% used with up to 7 requests
  waiting; Glyd's was at most 11.7% used, with none waiting.
- **Saturated:** 4.07x the requests a second (3.59 against 0.88), 918.4 against 225.9 output tokens a second, the first
  token 0.18x the time. bf16 ran up to 8 requests at once with up to 186 waiting; Glyd ran up to 98, with up to 102
  waiting.
- **Slower:**
  - Each token took 2.54x the time saturated (79.3 against 31.2 ms), with up to 98 requests running at once against 8.
  - A server started on an empty compile cache took about 349 s with Glyd and about 85 s with bf16 (`summary.txt`).
  - At saturation the GPUs drew a median of 695 W with Glyd and 537 W with bf16, of their 700 W, at a median SM clock of
    1590 and 1980 MHz (nvidia-smi's samples above the midpoint of the least and the most power drawn,
    `bench_summary.py`'s way).
- **The prefix cache:** both servers ran with vLLM's prefix caching on, its default. bf16's hit rate in its log is 0.0%
  in every line. Glyd's is 0.0% in every line of the rate-1 pass and 1.2, 0.6, 0.6, 0.5, 0.4 and 0.4% in the six of the
  saturated pass, so 0.4% over the whole run: about one prompt of the 256 sent. `summary.txt` marks the pair NOT
  COMPARABLE, as the bench does for any hit rate above 1%: Glyd's 1.2%.

## Reproduce

With two H100 80 GB, a driver of 580 or newer, and vLLM 0.30.0 with Glyd v0.26.0 (`pip install "glyd[vllm]==0.26.0"`),
one mode at a time (bf16: the same without `--quantization glyd`, and `bf16-rate...` in the file names). A pass's seed is
its place in the list of rates, 0 and then 1:

```bash
# the server, on an empty compile cache
VLLM_CACHE_ROOT=$(mktemp -d) vllm serve Qwen/Qwen2.5-72B-Instruct --quantization glyd \
  --max-model-len 4096 --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --port 8012 \
  --max-num-seqs 128 --compilation-config '{"max_cudagraph_capture_size":128}'

# once it answers: 64 requests at 1 a second (seed 0), then 192 at once (seed 1)
for r in "1 64 0" "inf 192 1"; do set -- $r
  vllm bench serve --backend vllm --model Qwen/Qwen2.5-72B-Instruct --port 8012 --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --ignore-eos --num-prompts $2 --request-rate $1 --seed $3 \
    --save-result --result-dir results --result-filename glyd-rate$1.json \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99
done
```

`gpu/vllm/bench_serve.sh` (as of 6a82f3bc) runs both for a mode, with those seeds, and `gpu/vllm/bench_summary.py results`
prints the tables from the files. The job did this on the instance: `big72_job.sh` (as it ran, with `~/results` emptied
first and its console in `~/vj-72b.log`) installs vLLM and nvcc in a venv, builds the library for the GPU from the tree and
installs its `glyd` package, downloads the model, and runs `bench_serve.sh` for Glyd and then for bf16. It reads
`~/big72_src.tar`: the files of v0.26.0 it needs, with the two bench scripts of 6a82f3bc in place of v0.26.0's, and the
tree's name in `COMMIT`. These commands make the 61 files of the job's own tarball, byte for byte:

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
mkdir ~/tree
git archive v0.26.0 COPYING LICENSE bindings/python glyd-store/LICENSE gpu | tar -x -C ~/tree
git archive 6a82f3bc gpu/vllm/bench_serve.sh gpu/vllm/bench_summary.py | tar -x -C ~/tree
echo "10e8caea (v0.26.0) + gpu/vllm/bench_serve.sh and bench_summary.py of bench-distinct-prompts 6a82f3bc (a seed for each pass, the prefix cache hit rate)" > ~/tree/COMMIT
tar -C ~/tree -cf ~/big72_src.tar .
bash benchmarks/gpu/h100x2-qwen2.5-72b-2026-10-03/big72_job.sh
```

## Files

- `bench-Qwen2.5-72B-Instruct.txt`: the console. `bench-Qwen2.5-72B-Instruct/`: for each mode and rate, the server's log
  (`serve-MODE.txt`), `kv-MODE.txt`, `MODE-rateR.json`, `bench-MODE-rateR.txt` and `smi-MODE-rateR.csv` (nvidia-smi's
  temperature, SM clock and power draw each second, for both GPUs), then `summary.txt`.
- `summary.txt`: the job's table and ratios. `steps.txt`, `steps-wrapper.txt`, `machine.txt`, `machine-short.txt`,
  `env.txt`, `vj-72b.txt` (`vj-72b.log` as written, the job's console, renamed because the repository ignores `*.log`),
  `log/` (the model's download, the environment, the library's build).
- `big72_job.sh`: the job as it ran. Left out: `job.log` (empty: the console went to `vj-72b.log`) and the empty markers
  `DONE` and `ALLDONE`.
