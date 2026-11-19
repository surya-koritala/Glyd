# Qwen3.8-27B across two consumer GPUs (2026-10-22)

GLY-365: `glyd/Qwen3.8-27B-swift` (18.7 GB, 64 layers) split over two Runpod pods, each with one
RTX 4090 (24 GB), in EU-RO-1 on Runpod's private network (0.4-0.6 ms ping). It was set up with the
two commands from PR #151: `starlings up --machines 2` on pod A and `starlings join CODE` on pod B.
The split `up` picked from free memory was 0..32 on A and 32..64 on B, about 9.1 GiB each.

From empty state directories to both machines ready took 25 s. That covers enrollment, both
codes, and loading both halves; the model was already on disk. The join code was 4219 characters
and the approval code 428.

## The answers equal the whole model's

- `glyd-engine rangecheck` on the 27B, split at layer 32, passed on both pods ("all checks
  passed"). It covers stages in one process and as worker processes, verify chunks with every
  keep, sampled runs, the head on 8 rows, and with `--sessions 2` (pod B) two sessions per stage.
  Output: `rangecheck-a.txt`, `rangecheck-b.txt`.
- Through the daemons, each answer's text equals the whole model's text for the same prompt, all
  256 tokens. That holds plain, drafted, and for two requests at once (drafted). The whole model
  ran alone on pod A (`pipeline_tool reference`): `ref-long.json`, `ref-list.json`.

## Speed

Greedy, 256 tokens, prompts of 77 and 76 tokens, `--sessions 2`.

| run | tok/s, one request | first text ms | two at once, tok/s each |
| --- | --- | --- | --- |
| whole model, one RTX 4090 | 46.0 (21.72 ms a step) | | |
| two pods, plain | 43.5 (38.6 on the first request after start) | 134-216 | 42.6 and 42.8 |
| two pods, drafts 7 | 51.9 and 53.8 | 134-233 | 52.1 and 49.4 |

- Plain, the two engines spend 22.2 ms a token between them (10.6 + 11.5); the link and the
  daemons add about 1.2 ms.
- Drafted, 11-12% of the guesses held: 196 and 187 decode steps for 256 tokens. A verify pass of
  8 rows costs 23.97 ms against 21.72 for one step, so the pipeline beats the whole model on one
  card by 13-17%.
- Two requests at once each run at about the speed of one alone, about 100 tok/s together. While
  one request is on a GPU, the other request is on the other GPU.

## Builds

- engine `glyd-engine-chat` (main with #232, #234, #236), sha256 `f1f8cfdf…`, built for sm_89;
- GPU library sha256 `b452dd11…`;
- `starlings` from PR #151, release build, sha256 `1fb2c5be…`.

The pods ran from 7:54 to 8:25 PM Eastern at $0.89/h each, about $0.92 in all.

## Limits

- One run, one pair of pods. The prompts are short, and the context was 4096.
- Runpod has no 16 GB consumer cards in its private-network data centers today. So these are
  24 GB cards, which would also fit the whole 27B. A pair of 16 GB cards (about 10 GB a stage)
  is the case this is for, and it has not been run.
- Not run: a LAN between two home machines, and sampled requests through the daemons (rangecheck
  covers the sampler).
