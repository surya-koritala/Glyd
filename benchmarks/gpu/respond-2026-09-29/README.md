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
chat mix (a 200-token prompt, a 300-token reply); tokens a second at 1 sequence (a 128-token prompt, 256 new tokens);
the long-document mix (2000 and 200); tokens a second at 8 and 32 sequences (the l4/ and l4-routes/ runs below had
them before the long mix). Each configuration's first call is a warm-up, kept apart (in the default mode the first one
compiles); then the median of its repeats (5 for the time to first token, 3 for the rest, below; resp_job.sh since
b39563d: 3, and at least one then as many as fit 12 s). The GPU's clocks and power are sampled while each mode runs
(`nvidia-smi`, once a second).

| file | what |
| :--- | :--- |
| `gpu/respond.py` | one model in one mode: `python respond.py MODEL --mode bf16|bf16c|glyd|exact --out RESULT.json` |
| `resp_job.sh` | the modes in turn, the library built for the GPU, the models downloaded: unattended, 35 minutes at most on a cloud GPU. The GPU's plan: Qwen3-8B and, by its memory, Qwen3-32B (a GH200 or H100; an A100 of 60 GB or more) or Qwen3-14B (an A100 of 40 GB; an A10, where its bf16 does not fit), each model's Glyd, bf16 compiled and bf16 eager, then each one's exact. Each run's deadline leaves the later runs their expected time (a GH200's run for the hopper plan, scaled for the others); within a run, repeats go first (at least one each, more while they fit 12 s), then its last configurations: exact's 8 and 32 sequences, left out of its expected time, first |
| `resp_summary.py` | the tables (summary.txt) and every result in one JSON (respond.json) |
| `gh200-v0.25.1/` | resp_job.sh at 15d9c04 (v0.25.1) on a GH200, Qwen3-8B and Qwen3-32B, below |
| `a10-v0.25.1/` | resp_job.sh at 15d9c04 (v0.25.1) on an A10, Qwen3-8B and Qwen3-14B (Glyd's alone fits), below |
| `a100-v0.25.1/` | resp_job.sh at 15d9c04 (v0.25.1's routes) on an A100 SXM4 40 GB, Qwen3-8B and Qwen3-14B, below |
| `l4-smoke/` | resp_job.sh at b39563d on the AWS dev L4, Qwen3-0.6B's four modes (`run.sh`: 15 minutes, each run's expected time given): every configuration of every mode run, in 10.4 minutes with the library's build; a check that the job runs, not a result |

## An L4 (l4/; AWS g6.4xlarge, AMD EPYC 7R13, 16 vCPUs; 2026-09-29)

`l4.sh`: resp_job.sh on the tree at d7b6ddf, the models from the machine's cache, Qwen3-8B then
Qwen3-4B-Instruct-2507, every configuration at its full repeats. The L4 held its 72 W cap in every mode, its SM clock
lower under Glyd's kernels (Qwen3-8B, medians while busy: bf16 1575 MHz, Glyd 1260, exact 1365). Glyd's layout there
is the tiered one (Ada). Qwen3-8B (summary.txt has both models):

| | bf16 | Glyd (default) | Glyd exact |
| :-- | --: | --: | --: |
| time to first token, 128-token prompt | 85 ms | 98 ms c | 179 ms (2.09x) |
| time to first token, 512-token prompt | 193 ms | 242 ms c | 310 ms (1.60x) |
| time to first token, 2048-token prompt | 662 ms | 915 ms (1.38x) | 833 ms (1.26x) |
| time to first token, 8192-token prompt | 3023 ms | 6356 ms (2.10x) | 3328 ms (1.10x) |
| chat: 200-token prompt, 300-token reply | 105 ms / 19.81 s | 128 ms / 15.09 s c | 207 ms / 45.29 s |
| tokens/s, 1 sequence (256 new) | 15.1 | 20.0 c | 6.6 (0.44x) |
| tokens/s, 8 sequences (256 new) | 109.6 | 140.0 (1.28x) | 48.9 (0.45x) |
| tokens/s, 32 sequences (256 new) | 358.8 | 438.9 (1.22x) | 180.1 (0.50x) |
| long: 2000-token prompt, 200-token reply | 712 ms / 14.26 s | 910 ms / 11.70 s | 819 ms / 31.14 s |

c: the call ran compiled (no ratio there: this run had no bf16 compiled, which l4-routes/ below has; the ratios are
over bf16 eager where both ran eager). The mixes: time to first token / total. On the GPU after the load: bf16 16.38 GB, Glyd 11.45
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

| | bf16 eager | bf16 compiled | Glyd (default) | Glyd 12-bit | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 86 ms | 95 ms c | 100 ms c | 85 ms c | 1.05x |
| 512 | 205 ms | 208 ms c | 246 ms c | 204 ms c | 1.18x |
| 2048 | 693 ms | 698 ms | 804 ms | 777 ms | 1.15x |
| 8192 | 3141 ms | 3155 ms | 3276 ms | 3297 ms | 1.04x |
| tokens/s, 1 sequence (256 new) | 15.1 | 15.7 c | 20.0 c | 19.7 c | 1.27x |
| 8 sequences | 109.7 | 109.6 | 139.7 | 140.8 | 1.27x |
| 32 sequences | 359.3 | 358.9 | 437.8 | 449.4 | 1.22x |
| chat: 200-token prompt, 300-token reply (first token / total) | 105 ms / 19.88 s | 113 ms / 19.12 s c | 128 ms / 15.08 s c | 104 ms / 15.23 s c | 0.79x |
| long: 2000 + 200 | 723 ms / 14.29 s | 716 ms / 14.34 s | 787 ms / 11.59 s | 766 ms / 11.64 s | 0.81x |

The ratio column is Glyd's default over bf16 compiled: like for like, the same calls compiled and the rest eager in
both. None is given over bf16 eager, whose one-sequence calls run eager where Glyd's default compiles them. For a time
to first token or a mix's total, under 1 is sooner;
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

## A GH200 on v0.25.1 (gh200-v0.25.1/; Lambda, 2026-09-30)

resp_job.sh at 15d9c04 (v0.25.1: Hopper's whole-matrix decode with its low bytes first), riding along with a Hopper
session: a GH200 480GB (96 GB, 132 SMs; a 64-core Neoverse V2 host), Qwen3-8B and Qwen3-32B, each model's four modes,
34.4 minutes with the library's build. Every configuration ran but Qwen3-32B exact's 32 sequences, which the plan cuts
first. The six runs but exact's took their expected times to within 3% (steps.txt); exact's ran their several-sequence
rates in the time left (Qwen3-8B's both, Qwen3-32B's at 8). Repeats: 3 for the time to first token, 1 to 3 for the
rest (as many as fit 12 s: one for each eager call of 200-300 tokens). Glyd's layout there is the 12-bit one.

Eager `generate()` is bound by the host on this machine: bf16 eager makes 21.9 tokens a second on one Qwen3-8B
sequence, bf16 compiled 119.1. The ratio column is therefore Glyd's default over bf16 compiled, the same calls
compiled, the rest eager in both (c: the call ran compiled):

| Qwen3-8B | bf16 eager | bf16 compiled | Glyd (default) | Glyd exact | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 48 ms | 55 ms c | 61 ms c | 58 ms | 1.12x |
| 512 | 49 ms | 62 ms c | 52 ms c | 58 ms | 0.85x |
| 2048 | 69 ms | 70 ms | 81 ms | 80 ms | 1.17x |
| 8192 | 291 ms | 294 ms | 310 ms | 302 ms | 1.05x |
| tokens/s, 1 sequence (256 new) | 21.9 | 119.1 c | 125.9 c | 18.0 | 1.06x |
| 8 sequences | 174.3 | 178.7 | 217.1 | 144.3 | 1.21x |
| 32 sequences | 701.1 | 716.5 | 857.7 | 579.1 | 1.20x |
| chat: 200-token prompt, 300-token reply (first token / total) | 54 ms / 13.66 s | 55 ms / 2.57 s c | 53 ms / 2.43 s c | 62 ms / 16.70 s | 0.95x |
| long: 2000 + 200 | 69 ms / 9.21 s | 70 ms / 9.00 s | 82 ms / 7.46 s | 80 ms / 11.22 s | 0.83x |

| Qwen3-32B | bf16 eager | bf16 compiled | Glyd (default) | Glyd exact | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 81 ms | 96 ms c | 85 ms c | 104 ms | 0.88x |
| 512 | 79 ms | 106 ms c | 107 ms c | 119 ms | 1.00x |
| 2048 | 270 ms | 269 ms | 329 ms | 313 ms | 1.22x |
| 8192 | 1141 ms | 1145 ms | 1270 ms | 1185 ms | 1.11x |
| tokens/s, 1 sequence (256 new) | 13.1 | 39.3 c | 46.8 c | 10.1 | 1.19x |
| 8 sequences | 105.9 | 100.3 | 133.1 | 80.3 | 1.33x |
| 32 sequences | 421.4 | 400.6 | 522.8 | (cut) | 1.30x |
| chat: 200 + 300 (first token / total) | 84 ms / 22.72 s | 102 ms / 7.69 s c | 85 ms / 6.51 s c | 106 ms / 29.76 s | 0.85x |
| long: 2000 + 200 | 259 ms / 15.41 s | 266 ms / 16.04 s | 328 ms / 12.49 s | 308 ms / 20.24 s | 0.78x |

- **Memory on the GPU after the load:** Qwen3-8B bf16 16.38 GB, Glyd 12.54, exact 13.51; Qwen3-32B bf16 65.52 GB,
  Glyd 49.50, exact 50.78.
- **First calls (the compile, excluded above):** bf16 compiled 36.7 s (8B) and 64.8 s (32B), Glyd's default 33.9 and
  55.5 s.
- **Tokens do not repeat from call to call here.** On the GH200, bf16 eager's own greedy tokens (the first sequence's)
  differed between its first call and its repeat at 8 and 32 sequences, for both models, and bf16 compiled's and
  Glyd's default's at one sequence (rate 1). The eager calls at one sequence repeated, in every mode. The earlier
  GH200 run (tree cad1d8c, v0.25.0) shows the same (bf16 eager at 8 sequences: 3 different outputs in 4 calls). The
  L4's runs repeated in every configuration of every mode. So at 8 and 32 sequences on this GPU, no mode's tokens can
  be compared with bf16 eager's; in the 7 configurations of each model where bf16 eager's calls repeated, exact's
  tokens were bf16 eager's in all 7. benchmarks/gpu/repro-2026-09-30 finds which operation does not repeat.

## An A10 on v0.25.1 (a10-v0.25.1/; Lambda, 2026-09-30)

resp_job.sh at 15d9c04 (the plan a10): an A10 (24 GB, 150 W; an Intel Xeon Platinum 8358 host, 30 vCPUs), Qwen3-8B and
Qwen3-14B, 26 minutes with the library's build. Glyd's layout there is the 12-bit one for Qwen3-8B and the tiered one
for Qwen3-14B (the one that fits). Each run waited its 20 s for the GPU to cool and started at 45-61 C (its idle 35 C).

Qwen3-14B fits where bf16 does not: Glyd's default loaded at 20.54 GB on the GPU (22.4 at its peak) and ran every
configuration but the 8192-token prompt (out of memory); bf16, eager and compiled, and Glyd exact did not load (out of
memory). Its time to first token 138 / 355 / 1261 ms at 128 / 512 / 2048 tokens, 18.3 tokens a second at one sequence,
131.2 at 8 and 345.0 at 32, the chat mix 16.26 s and the long one 12.64 s.

| Qwen3-8B | bf16 eager | bf16 compiled | Glyd (default) | Glyd exact | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 54 ms | 65 ms c | 59 ms c | 117 ms | 0.91x |
| 512 | 154 ms | 152 ms c | 169 ms c | 209 ms | 1.11x |
| 2048 | 568 ms | 561 ms | 591 ms | 618 ms | 1.05x |
| 8192 | 2409 ms | 2391 ms | 2395 ms | 2443 ms | 1.00x |
| tokens/s, 1 sequence (256 new) | 24.6 | 26.1 c | 34.1 c | 9.6 | 1.31x |
| 8 sequences | 180.7 | 181.4 | 170.8 | 74.4 | 0.94x |
| 32 sequences | 599.3 | 601.1 | 668.9 | 274.1 | 1.11x |
| chat: 200 + 300 (first token / total) | 74 ms / 12.21 s | 79 ms / 11.69 s c | 82 ms / 8.77 s c | 138 ms / 31.12 s | 0.75x |
| long: 2000 + 200 | 575 ms / 9.01 s | 567 ms / 8.97 s | 592 ms / 7.17 s | 642 ms / 21.67 s | 0.80x |

- **At 8 sequences Glyd's default was slower, 0.94x**, both running eager there (past the compiled calls' 2048
  positions). The GPU's samples show why it was not the GPU's work: while bf16 compiled's 8 sequences ran, the GPU was
  busy 96-98% of each second at 1365-1455 MHz; while Glyd's ran, 68-73% at 1605-1680 MHz, both at the 150 W cap
  (smi-Qwen3-8B-*.csv). So about a quarter of each of Glyd's steps the GPU waited, on the host's launches; the job
  does not sample the CPU, so the logs do not show the host's own load. At 32 sequences Glyd's GPU was busy 81-97%
  (rising with the context), and Glyd was 1.11x.
- **Memory on the GPU after the load:** Qwen3-8B bf16 16.38 GB, Glyd 12.67, exact 13.51.
- **First calls (the compile, excluded above):** bf16 compiled 50.3 s, Glyd's default 54.0 s (Qwen3-14B's 35.9 s).
- **Tokens:** every mode's own calls repeated in every configuration here; exact's were bf16 eager's in all 9.
- **The plan's times:** each run's own time against the plan's expected time (steps.txt gives each run's window: the
  time it was given, not the time it took): Qwen3-8B Glyd 246 s (230 expected), bf16 compiled 272 (250), bf16 eager
  170 (200), exact 348 (240, its 8 and 32 sequences run in the time left); Qwen3-14B Glyd 338 (365); bf16, bf16
  compiled and exact did not fit, 30, 30 and 61 s.

## An A100 SXM4 40 GB on v0.25.1 (a100-v0.25.1/; Lambda, 2026-09-30)

resp_job.sh at 15d9c04 (the plan a100): an A100-SXM4-40GB (400 W; an AMD EPYC 7J13 host, 30 vCPUs), Qwen3-8B and
Qwen3-14B, 29.4 minutes with the library's build, every configuration of every mode. These are v0.25.1's routes: from
v0.26.0 an A100 SXM's 12-bit prompts of 769-4096 tokens take the route SPLIT (option 2), which this run predates.
Glyd's layout there is the 12-bit one. Every mode's own calls repeated in every configuration, and exact's tokens were
bf16 eager's in all 9 of each model.

| Qwen3-8B | bf16 eager | bf16 compiled | Glyd (default) | Glyd exact | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 47 ms | 56 ms c | 54 ms c | 65 ms | 0.96x |
| 512 | 53 ms | 64 ms c | 76 ms c | 77 ms | 1.18x |
| 2048 | 184 ms | 185 ms | 201 ms | 208 ms | 1.09x |
| 8192 | 749 ms | 747 ms | 772 ms | 776 ms | 1.03x |
| tokens/s, 1 sequence (256 new) | 23.0 | 58.8 c | 72.1 c | 16.7 | 1.23x |
| 8 sequences | 181.7 | 184.7 | 194.1 | 132.0 | 1.05x |
| 32 sequences | 725.5 | 746.4 | 773.4 | 531.4 | 1.04x |
| chat: 200 + 300 (first token / total) | 48 ms / 13.08 s | 56 ms / 5.13 s c | 55 ms / 4.22 s c | 64 ms / 17.98 s | 0.82x |
| long: 2000 + 200 | 183 ms / 8.84 s | 184 ms / 8.73 s | 202 ms / 8.30 s | 208 ms / 12.17 s | 0.95x |

| Qwen3-14B | bf16 eager | bf16 compiled | Glyd (default) | Glyd exact | Glyd / compiled |
| :-- | --: | --: | --: | --: | --: |
| time to first token, 128-token prompt | 53 ms | 62 ms c | 60 ms c | 90 ms | 0.97x |
| 512 | 84 ms | 94 ms c | 123 ms c | 129 ms | 1.31x |
| 2048 | 299 ms | 300 ms | 357 ms | 345 ms | 1.19x |
| 8192 | 1257 ms | 1256 ms | 1353 ms | 1308 ms | 1.08x |
| tokens/s, 1 sequence (256 new) | 20.4 | 36.9 c | 46.8 c | 12.7 | 1.27x |
| 8 sequences | 161.5 | 163.7 | 171.6 | 98.5 | 1.05x |
| 32 sequences | 646.2 | 653.3 | 687.0 | 372.7 | 1.05x |
| chat: 200 + 300 (first token / total) | 56 ms / 14.70 s | 70 ms / 8.22 s c | 76 ms / 6.50 s c | 103 ms / 23.58 s | 0.79x |
| long: 2000 + 200 | 299 ms / 9.99 s | 301 ms / 9.86 s | 358 ms / 9.51 s | 346 ms / 16.11 s | 0.96x |

- **Memory on the GPU after the load:** Qwen3-8B bf16 16.38 GB, Glyd 12.54, exact 13.51; Qwen3-14B bf16 29.54 GB,
  Glyd 22.43, exact 23.71.
- **First calls (the compile, excluded above):** bf16 compiled 48.4 s (8B) and 53.8 s (14B), Glyd's default 43.7 and
  48.7 s.
- **The plan's times** (from steps.txt's times of day; its "its time" is each run's window): Qwen3-8B Glyd 216 s (200
  expected), bf16 compiled 227 (220), bf16 eager 137 (180), exact 190 with its several-sequence rates (145 without);
  Qwen3-14B 251 (235), 264 (255), 162 (205), exact 260 with them (210 without).
