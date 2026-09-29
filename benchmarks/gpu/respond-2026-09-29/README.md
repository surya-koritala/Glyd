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
