# Nemotron 3 Super at wren on one A100 80 GB (2026-10-10)

`tools/cards/nh_run_check_job.sh` on v030 21c8176b: one NVIDIA A100-SXM4-80GB (sm_80, 14 CPUs), 55 minutes of job (about $1.70). Model:
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a, packed by the engine at wren: the 40,960 routed experts in four-bit codes
(4.53 bits a weight), the other 258 matrices in six-bit ones; run from that directory. Linear GLY-132: Nemotron on an 80 GB card at 4 bits.

| step | result |
|---|---|
| the grouped products over Int4 experts (`moecheck relu2-22of512`) | all ok against the CPU reference, decode graph replays bit for bit |
| pack (`glyd-engine pack --layout i4`) | 241.34 GB of bf16 to 70.49 GB in 14 shards: 40,960 packs at i4, 258 at i6, 409 small tensors kept; 1,561 s on 14 CPUs |
| verify (`glyd-gpu verify --device cuda:0`) | every pack decodes to glyd.json's sha256 |
| load (`generate`) | 45.7 s: 65.18 GiB of weights, 67.01 of 79.25 GiB in use at 4,096 tokens of context |
| the whole model against the bf16 checkpoint | top-1 token the same as bf16's at 100.0% of 97 positions; KL 0.0016 nats on average (worst 0.0398); the decode graph's logits equal the eager steps' exactly |
| six layers against the reference, on the weights the packs hold | 200 checks ok, none failed |
| `generate` (greedy, decode graph) | 128 tokens in 1.81 s |
| `serve`, one user (speed1.py: 1,050-token prompt, 256 new, greedy, median of 3) | 71.1 tokens a second, first token 458 ms; 67.2 GiB (68,843 MiB) of the card in use, 32,768 tokens of context |
| `run --chat` (the checkpoint's sampling defaults) | 25.5 to 38.6 tokens a second (sampled on the host: GLY-139) |

How to read it:
- **It fits an 80 GB card with room for context:** 65.2 GiB of weights and 32,768 tokens of context in 67.2 GiB.
- **The whole-model check** is the same as the linnet run's (benchmarks/gpu/nh-run-check-2026-10-10): 160 tokens of one text against the
  bf16 checkpoint run layer by layer on the same GPU. It is a correctness check, not the held-out quality measurement; on these 97
  positions wren's KL (0.0016) is not above linnet's (0.0022 on the H200), which says the text is too short to rank the two levels. The
  held-out measurement of four-bit experts is benchmarks/weights/moe-quality-2026-10-10/nemotron (int4: KL 0.0133 including a 0.0047 floor).
- **Speed** is the A100's (2.0 TB/s): 71.1 tokens a second greedy for one user, against linnet's 96.6 on the H200 (4.8 TB/s).
- **Answers** are coherent (run-prompt.log, run-chat.log, serve-chat.log).

Files: steps.txt, summary.txt, moecheck.log, logits-check.log and logits-run.log, logits/, layers-check.log, layers-run.log and
layers-spec.json, generate.log, run-prompt.log, run-chat.log, serve-chat.log, speed.json, serve-mem.txt, pack.txt and pack.log, verify.log,
versions.log, machine.txt, gpu.txt.
