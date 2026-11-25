# The quality bar of the lossy levels

Kestrel, swift, wren and hummingbird change the weights; penguin does not. Before a pack of a lossy level is published, one suite measures how far
its outputs are from the lossless level of the same model, and the level's bar decides: publish or refuse. The code is `glyd-engine/src/quality.rs`;
the bars are in `glyd-engine/src/levels.rs` (`Bar`).

## The suite

`glyd-engine quality MODEL[:LEVEL]` starts `glyd-engine serve` of the model and measures it; `--against URL` measures any OpenAI-compatible server
(llama.cpp, vLLM, SGLang, Ollama) the same way. Everything runs at temperature 0 with thinking off. The result is one JSON file, `glyd-quality/1`.

| Part | What | Size |
| :--- | :--- | :--- |
| Next-token KL | after fixed prefixes of four texts written for this suite (prose, a Python program, instructions, a dialogue), the top 20 log-probabilities of a one-token raw completion | 48 prefixes a text, 192 in all |
| Reasoning | word problems of two or three steps (five kinds), made from a fixed seed; the answer is the reply's last number | 40 (`--n`) |
| Code | what a short Python program prints (six kinds: loops, list comprehensions, a Collatz count, string tests, nested functions, a digit sum), worked out by the suite, never run | 40 (`--n`) |
| Long context | four lockers' six-digit codes among 4K or 16K tokens of filler words (`--long 4k,16k,32k`), and the question of one, whose code sits at 10%, 35%, 65% or 90% of the text | 4 a length |

The tasks are made from a seed, not read from a dataset: no download, no license, and the same problems every run. They check that a level did not
break what the model could do; they are not a measurement of the model's ability. The server the suite starts gets a context that holds the longest
item (`--ctx` says otherwise).

```bash
glyd-engine quality Qwen/Qwen3.8-27B:penguin --out ref.json     # the lossless level: the reference
glyd-engine quality Qwen/Qwen3.8-27B:swift --out swift.json
glyd-engine quality --card ref.json swift.json --out card.json  # exit 0: publish, 1: refuse (the reasons on stderr)
```

## The card and the bar

`quality --card REF CAND` compares a candidate's result with the lossless level's: the mean KL over the 192 prefixes (each the KL of the reference's
20 likeliest tokens and the rest as one bin, a lower bound of the full divergence), the share of prefixes whose best token is the reference's, each
text's figures, and each task's accuracy beside the reference's and how many answers were the same. The card (`glyd-quality-card/1`) says
**publish** where all of these are within the level's bar, else **refuse** with the reasons. A reference that is not the lossless level is refused;
a text or task the candidate lacks is a refusal, not a pass.

| Level | Mean KL at most | Best token the same at least | A task's accuracy at most this much under the lossless level's |
| :--- | --: | --: | --: |
| kestrel (6.5 bits) | 0.02 | 94% | 5 points |
| swift (5.5) | 0.05 | 91% | 7.5 points |
| wren (4.5) | 0.08 | 86% | 10 points |
| hummingbird (experts at 2-4) | 0.12 | 90% | 10 points |

The bars come from what the levels' layouts measured on released models with `research/lossy-2026/kld_study.py` and the MoE study, against bf16:
six-bit codes 0.0048 to 0.0147 (the Qwen3.5, 3.6 and 3.8 cards; top-1 96.9% on Qwen3.8-27B), five-bit 0.0139 to 0.0373 (top-1 94.3%), four-bit
without calibration 0.0534 (top-1 88.9%, `benchmarks/weights/kld-q56-2026-10-09`), hummingbird's hb40 recipe 0.0831 (top-1 94.0%,
`benchmarks/weights/moe-quality-2026-10-10/nemotron-hummingbird`). Each bar is about 1.4 times the largest KL and three to four points under the
lowest agreement. Those studies read WikiText-2 with 64 to 128 tokens a position; the suite reads its own texts with 20. The first cards the suite
writes are the check of the bars against its own figures, and a bar that does not hold is changed in `levels.rs` with the cards that show it.

## Publish or refuse

`glyd-engine ship MODEL LEVEL DIR` runs the suite in its `quality` step: on the model's lossless level (or `--quality-ref FILE`, a result taken on a
GPU that holds the lossless level) and on the pack, and writes the card into the pack as `quality.json`, so it is uploaded with it, and as a
"Quality" section of the pack's README. Its `upload` step refuses a lossy pack whose `quality.json` is missing, is of another level, or says refuse.
A penguin pack needs no card: its weights are the release's, bit for bit (`glyd verify`).

A hummingbird pack goes the same way. Its experts are coded from a corpus pass's counts, not by a layout of every matrix, so `ship`'s pack step
takes the counts: `glyd-engine calibrate-experts MODEL usage.json`, then `glyd-engine ship MODEL hummingbird DIR --usage usage.json`. Its quality
step and the upload's refusal are every lossy level's.

## On random models, in the GPU job

The bars above are for released models, measured with the suite. The GPU job holds every level's arithmetic on a small random model of each family
with `glyd-engine levelcheck` (`glyd-engine/src/logits.rs`): the model at penguin and at each lossy level its family runs, the same tokens, and the
KL of the level's logits from penguin's against a threshold of that family and level, twice the family's first measured run. Qwen3.5, Gemma 4 and
the state-space case are held at kestrel, swift and wren; DeepSeek-V4 and Muse Spark at hummingbird, on a directory the check makes with the two
commands above (the corpus pass over made-up documents). A level that does not load fails the check. A random model's weights do not compress as a
trained model's do, so these thresholds say nothing of a release: they catch a level whose kernels or packer changed.

Kimi-K3 and GLM-5.3-Flash are not held at hummingbird yet: on a random model the corpus pass counts none of Kimi-K3's layers, and GLM-5.3-Flash's
hummingbird directory does not load. The check prints both as open with what stops each, and fails the day one of them runs without a threshold.

## In `fit`

`glyd fit MODEL` prints, under the plan, one line a level: penguin is lossless; a lossy level shows its card's figures and verdict (a local pack's
`quality.json`, the level's pre-packed repo in the Hub cache, or the card in `cards/NAME-LEVEL.json`), the earlier figures of a card made before the
suite (KL from bf16 on WikiText-2, GSM8K against bf16, marked as having no verdict), or "not measured". `fit --json` has the same under `quality`.

## Comparisons at equal bits

The suite measures any server, so another format of the same model is measured the same way and its card taken against Glyd's lossless level, held
to the bar of the level of about its bits (`--level` names the format, `--bar` the level):

```bash
llama-server -m model-Q4_K_M.gguf --port 8001 &                 # about 4.9 bits a weight; wren is 4.53
glyd-engine quality --against http://127.0.0.1:8001 --engine llama.cpp --bits 4.89 --out q4km.json
glyd-engine quality --card ref.json q4km.json --level Q4_K_M --bar wren --out card-q4km.json
glyd-engine quality --table card-*.json                          # the cards side by side, by bits, as Markdown
```

The pairs: penguin (about 10.8 bits, lossless) against FP8 (8, vLLM `--quantization fp8` or a released FP8 checkpoint) and Q8_0 (8.5); kestrel
(6.53) against Q6_K (6.56); swift (5.53) against Q5_K_M (about 5.7); wren (4.53) against Q4_K_M (about 4.9), Q4_0 (4.5) and a 4-bit AWQ checkpoint (about 4.2
with group 128). The token texts of the log-probabilities are compared as the servers give them; llama.cpp's shape (`logprobs.content[0].top_logprobs`)
and the completions API's (`logprobs.top_logprobs[0]`) are both read. Published comparisons live under `benchmarks/quality/`, each with its
commands, versions and logs; none is published until it was measured.
