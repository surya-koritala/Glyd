# How fast it responds: time to first token and tokens a second, 2026-09-29

What a user of `generate()` waits for, bf16 against Glyd, on one GPU at a time: `gpu/respond.py` in four modes, each
in a process of its own, the same prompts in each (a fixed English text's first N tokens; B copies for B sequences),
greedy decoding, every reply forced to its length (`min_new_tokens`: no early end of sequence):

- **bf16**: transformers' own model (`AutoModelForCausalLM`, bf16), its `generate()` as it runs by default (eager, a
  dynamic cache).
- **bf16 compiled**: the same bf16 model, its `generate()` compiled as Glyd's default is (`fast_generate`:
  transformers' static cache and `torch.compile`'s CUDA graphs for the same calls, the rest eager): bf16 like for like.
- **Glyd (default)**: `glyd.from_pretrained(MODEL)` as it loads by default: the layout the GPU's (`"auto"`: tiered on
  Ada, 12-bit on an A10, A100 and H100, where it fits), `generate()` compiled (a static cache, CUDA graphs) where a
  call's cache holds at most 2048 positions in all (1280 on a GeForce card), else eager.
- **Glyd exact**: `glyd.from_pretrained(MODEL, exact=True)`: every matrix decoded whole, then `F.linear`, the logits
  bf16's bit for bit; eager.

Each call is timed by a streamer (transformers hands it the prompt, then each step's tokens, syncing the GPU a step):
**time to first token** from the call to the first new token, **tokens a second** after it (new tokens less the first,
times the sequences, over the first token's put to the last), and the call's **total**. The measurements, in this
order in each process: time to first token for prompts of 128, 512, 2048 and 8192 tokens (16 new tokens each); the
chat mix (a 200-token prompt, a 300-token reply); tokens a second at 1, 8 and 32 sequences (a 128-token prompt, 256 new
tokens); the long-document mix (2000 and 200). Each configuration's first call is a warm-up, kept apart (in the default
mode the first one compiles); then the median of its repeats (5 for the time to first token, 3 for the rest). The
GPU's clocks and power are sampled while each mode runs (`nvidia-smi`, once a second).

| file | what |
| :--- | :--- |
| `gpu/respond.py` | one model in one mode: `python respond.py MODEL --mode bf16|bf16c|glyd|exact --out RESULT.json` |
| `resp_job.sh` | the modes in turn, the library built for the GPU, the models downloaded: unattended, 20 minutes at most on a cloud GPU (an even share of the time left a mode; what does not fit its share is recorded so) |
| `resp_summary.py` | the tables (summary.txt) and every result in one JSON (respond.json) |

## An L4 (l4/; AWS g6.4xlarge, AMD EPYC 7R13, 16 vCPUs; 2026-09-29)

`l4.sh`: resp_job.sh on the tree at d7b6ddf, the models from the machine's cache, Qwen3-8B then
Qwen3-4B-Instruct-2507, every configuration at its full repeats. The L4 held its 72 W cap in every mode, its SM clock
lower under Glyd's kernels (Qwen3-8B, medians while busy: bf16 1575 MHz, Glyd 1260, exact 1365). Glyd's layout there
is the tiered one (Ada). Qwen3-8B (summary.txt has both models):

| | bf16 | Glyd (default) | Glyd exact |
| :-- | --: | --: | --: |
| time to first token, 128-token prompt | 85 ms | 98 ms (1.15x) c | 179 ms (2.09x) |
| time to first token, 512-token prompt | 193 ms | 242 ms (1.25x) c | 310 ms (1.60x) |
| time to first token, 2048-token prompt | 662 ms | 915 ms (1.38x) | 833 ms (1.26x) |
| time to first token, 8192-token prompt | 3023 ms | 6356 ms (2.10x) | 3328 ms (1.10x) |
| chat: 200-token prompt, 300-token reply | 105 ms / 19.81 s | 128 ms / 15.09 s c | 207 ms / 45.29 s |
| tokens/s, 1 sequence (256 new) | 15.1 | 20.0 (1.32x) c | 6.6 (0.44x) |
| tokens/s, 8 sequences (256 new) | 109.6 | 140.0 (1.28x) | 48.9 (0.45x) |
| tokens/s, 32 sequences (256 new) | 358.8 | 438.9 (1.22x) | 180.1 (0.50x) |
| long: 2000-token prompt, 200-token reply | 712 ms / 14.26 s | 910 ms / 11.70 s | 819 ms / 31.14 s |

c: the call ran compiled. The mixes: time to first token / total. On the GPU after the load: bf16 16.38 GB, Glyd 11.45
GB, exact 12.41 GB. The first call of each process (excluded above): bf16 1.8 s, Glyd 33.4 s (its first compile; the
512-token configuration compiled again, 45.1 s, as its static cache grew), exact 3.1 s. Greedy tokens as bf16's: exact
in all 9 configurations of each model; Glyd's default in 6 of 9 (the three tokens-a-second runs, 256 new tokens,
diverge: its fused products sum in another order than cuBLAS), each configuration's repeats the same tokens.

## The L4 again: bf16 compiled, and the L4's prompt routes (l4-routes/; 2026-09-29)

`l4b.sh` and `l4c.sh`: resp_job.sh on branch l4-routes (72a46e2: an L4's prompts decoded for cuBLAS from 896 tokens
tiered and 2560 12-bit; benchmarks/gpu/l4-routes-2026-09-29 there) with this branch's respond.py (bcb540e; glyd12
5639409). The modes, the same machine and cache:
- bf16 eager.
- bf16 compiled, as Glyd's default compiles (the same calls: to 2048 positions).
- Glyd's default (tiered on the L4).
- Glyd in the 12-bit layout (glyd12).
Qwen3-8B (summary.txt has Qwen3-4B-Instruct-2507 too):

| | bf16 eager | bf16 compiled | Glyd (default) | Glyd 12-bit | Glyd / eager | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 86 ms | 95 ms c | 100 ms c | 85 ms c | 1.16x | 1.05x |
| 512 | 205 ms | 208 ms c | 246 ms c | 204 ms c | 1.20x | 1.18x |
| 2048 | 693 ms | 698 ms | 804 ms | 777 ms | 1.16x | 1.15x |
| 8192 | 3141 ms | 3155 ms | 3276 ms | 3297 ms | 1.04x | 1.04x |
| tokens/s, 1 sequence (256 new) | 15.1 | 15.7 c | 20.0 c | 19.7 c | 1.32x | 1.27x |
| 8 sequences | 109.7 | 109.6 | 139.7 | 140.8 | 1.27x | 1.27x |
| 32 sequences | 359.3 | 358.9 | 437.8 | 449.4 | 1.22x | 1.22x |
| chat: 200-token prompt, 300-token reply (first token / total) | 105 ms / 19.88 s | 113 ms / 19.12 s c | 128 ms / 15.08 s c | 104 ms / 15.23 s c | 0.76x | 0.79x |
| long: 2000 + 200 | 723 ms / 14.29 s | 716 ms / 14.34 s | 787 ms / 11.59 s | 766 ms / 11.64 s | 0.81x | 0.81x |

The ratio columns are Glyd's default over each bf16. For a time to first token or a mix's total, under 1 is sooner;
for tokens a second, over 1 is faster.

- **Before the L4's routes** (l4/, v0.25.0), Glyd's time to first token was 1.38x bf16 eager's at 2048 tokens and
  2.10x at 8192; now 1.16x and 1.04x.
- **Compiling bf16:** it gains 4% on one sequence and nothing at 8 and 32, which run eager in both (past the 2048
  positions). Its compiled calls take 8-10 ms longer to their first token (the static cache).
- **The 12-bit layout on the L4:** bf16's time to first token to 512 tokens (0.98-0.99x bf16 eager's), and the
  tiered layout's tokens a second (0.98-1.03x), for 9% more memory (12.54 GB against 11.45 on the GPU after the load).
- **First calls (the compile):** bf16 compiled 36.5-37.3 s. Glyd's took 9.1-9.5 s here, 33 s in l4/: PyTorch's
  compile cache in /tmp still held its graphs from the day's earlier runs.
- **Tokens:** bf16 compiled's greedy tokens are bf16 eager's in 8 of 9 configurations.
