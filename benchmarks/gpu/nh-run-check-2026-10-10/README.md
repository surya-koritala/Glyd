# Nemotron 3 Super at swift on one H200 (2026-10-10)

`tools/cards/nh_run_check_job.sh` on v030 d09fa951: one NVIDIA H200 (141 GB, sm_90, 21 CPUs), 34 minutes of job (about $3). Model:
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a, packed by the engine at swift (every matrix five-bit, 5.53 bits a weight) and
run from that directory. Linear GLY-132.

| step | result |
|---|---|
| pack (`glyd-engine pack --layout i5`) | 241.34 GB of bf16 to 83.61 GB in 16 shards: 41,234 packs checked, 409 small tensors kept; 784 s |
| verify (`glyd-gpu verify --device cuda:0`) | every pack decodes to glyd.json's sha256 |
| load (`generate`) | 66.6 s: 77.40 GiB of weights, 79.43 of 139.80 GiB in use at 4,096 tokens of context |
| the whole model against the bf16 checkpoint | top-1 token the same as bf16's at 97.9% of 97 positions; KL 0.0022 nats on average (worst 0.0764); the decode graph's logits equal the eager steps' exactly |
| six layers against the reference, on the weights the packs hold | 200 checks ok, none failed (Mamba-2 layers 0 and 40, mixture of experts 1 and 37, attention 7 and 36; whole prompt, chunks, steps, graph replays) |
| `generate` (greedy, decode graph) | 128 tokens in 1.29 s |
| `serve`, one user (speed1.py: 1,050-token prompt, 256 new, greedy, median of 3) | 96.6 tokens a second, first token 321 ms; 79.6 GiB (81,557 MiB) of the card in use, 32,768 tokens of context |
| `run --chat` (the checkpoint's sampling defaults) | 33.5 to 46.9 tokens a second |

How to read it:
- **The whole-model check** runs the bf16 checkpoint layer by layer on the same GPU (tools/nemotron_ref.py `full`) on 160 tokens of one
  text: 64 prefilled, then 96 teacher-forced steps. It is a correctness check of the engine at swift, not the held-out quality
  measurement of benchmarks/weights/moe-quality-2026-10-10/nemotron (other text, experts only).
- **Answers** are coherent: `run` answers "The capital of France is Paris.", and the chat writes a working Fibonacci function with a
  docstring and explains Rayleigh scattering (run-chat.log, serve-chat.log).
- **Sampled decode is about half the greedy speed.** The chat's 33.5-46.9 tokens a second against serve's greedy 96.6: each sampled step
  copies the logits (131,072 of them) to the host and samples there (`Gen for EngineNh`, as Qwen3.5's engine does). Greedy steps take the
  token from the GPU.
- **Not run here:** kestrel (Int6, about 98.6 GB), tool calls (not parsed in this format yet: GLY-138), contexts past 4,096 tokens in
  the checks.

Files: steps.txt, summary.txt, logits-check.log and logits-run.log (the whole-model check), logits/ (its token ids and the engine's
tokens), layers-check.log, layers-run.log and layers-spec.json, generate.log, run-prompt.log, run-chat.log, serve-chat.log, speed.json,
serve-mem.txt, pack.txt and pack.log, verify.log, versions.log, machine.txt, gpu.txt.
