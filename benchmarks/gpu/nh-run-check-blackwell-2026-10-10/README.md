# Nemotron 3 Super at wren on one RTX PRO 6000 Blackwell (2026-10-10)

`tools/cards/nh_run_check_job.sh` on v030 f3eeced7: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120, 28 CPUs), 30 minutes of
job (about $1.25). The same job and level as the A100 run (benchmarks/gpu/nh-run-check-wren-2026-10-10): the routed experts in four-bit
codes, every other matrix in six-bit ones. It checks the engine's kernels on Blackwell (sm_120), the family of the RTX 5090 and of the DGX
Spark's GB10 (sm_121, not run). Linear GLY-140.

| step | result |
|---|---|
| the grouped products over Int6/Int5/Int4 experts (`moecheck relu2-22of512`) | 24 checks ok, none failed |
| pack (`glyd-engine pack --layout i4`) | 241.34 GB of bf16 to 70.49 GB in 14 shards; 703 s on 28 CPUs |
| verify (`glyd-gpu verify --device cuda:0`) | every pack decodes to glyd.json's sha256 |
| load (`generate`) | 28.5 s: 65.18 GiB of weights, 67.15 of 94.97 GiB in use at 4,096 tokens of context |
| the whole model against the bf16 checkpoint | top-1 token the same as bf16's at 100.0% of 97 positions; KL 0.0012 nats on average (worst 0.0181); the decode graph's logits equal the eager steps' exactly |
| six layers against the reference, on the weights the packs hold | 200 checks ok, none failed |
| `generate` (greedy, decode graph) | 128 tokens in 1.27 s |
| `serve`, one user (speed1.py: 1,050-token prompt, 256 new, greedy, median of 3) | 99.5 tokens a second, first token 243 ms; 67.4 GiB (68,991 MiB) of the card in use, 32,768 tokens of context |
| `run --chat` (the checkpoint's sampling defaults) | 39.8 to 52.7 tokens a second (sampled on the host: GLY-139, fixed after this tree) |

How to read it:
- **Blackwell runs the engine as it is:** every check of the A100 run passes on sm_120, with the build for the card's own architecture.
- **The KL is the reference's on this card:** the bf16 checkpoint is run layer by layer on the same GPU, so the reference's own
  arithmetic differs a little from card to card (0.0016 on the A100 with the same packed model).
- **Speed:** 99.5 tokens a second greedy for one user (GDDR7, about 1.8 TB/s), against 71.1 on the A100 80 GB and 96.6 at linnet on the H200.

Files: steps.txt, summary.txt, moecheck.log, logits-check.log and logits-run.log, logits/, layers-check.log, layers-run.log and
layers-spec.json, generate.log, run-prompt.log, run-chat.log, serve-chat.log, speed.json, serve-mem.txt, pack.txt and pack.log, verify.log,
versions.log, machine.txt, gpu.txt.
