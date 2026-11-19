# Results for v0.1.0 / v0.1.1

Measured on the release build, installed fresh with `./setup.sh`, on an RTX 4080 SUPER 16 GB, Ryzen 9 7950X3D and 64 GB of RAM (Ubuntu),
on 2026-10-14. `a10/` is the same release on an NVIDIA A10 24 GB (Ubuntu 22.04).

| | Precise (default) | Fast |
|---|--:|--:|
| GSM8K | 148 / 150 | 148 / 150 |
| MMLU-Pro | 124 / 150 | 118 / 150 |
| Chat speed (default settings) | 117 tokens/s | 135 tokens/s |

**How:** [Inspect AI](https://inspect.aisi.org.uk/)'s `gsm8k` and `mmlu_pro` tasks as they ship, every second question of each dataset (the
first 150), greedy, thinking off, at most 3,072 tokens an answer, against the running server. Answers that reach the cap count as wrong
(16 in precise MMLU-Pro, 21 in fast).

**Files, for each mode:**
- `gsm8k.json`, `mmlu_pro.json`: the totals.
- `gsm8k-0/` ... `mmlu_pro-125/`: Inspect AI's logs, one per 25 questions, with every question, the model's answer and the score.
  Open them with `inspect view` (`pip install inspect-ai`).
- `speed.md`: `tools/bench.sh`'s report. `test_server.txt`: the API checks.

**Run them yourself:** start the server (`./setup.sh` or `tools/serve.sh`), then `bench/run_bench.sh` (about an hour per mode; it makes
its own Python environment for Inspect AI).
