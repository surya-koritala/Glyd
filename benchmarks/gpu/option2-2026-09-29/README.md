# The long-prompt mode (option 2), 2026-09-29

The long-prompt mode (`GLYD_GPU_WITH_SPLIT`, opt-in) is for 12-bit prompts on an A100 SXM, a GH200 and an H100 SXM.
These are the measurements that set where it applies: models' forward passes with it and without it (v0.25.0's
behavior, then v0.25.1's) in one process, a layer's products, and a stress check. Every session ran unattended
(`jobs/`), its tree a `git archive` of this branch with a COMMIT file.

| script | what |
| :--- | :--- |
| `gpu/e2e.py MODEL --baseline --format mma12 --fused --merge --prefill LENGTHS --without-split --breakdown LENGTHS` | bf16, then Glyd (the Linears merged: q, k, v and gate, up one product each): a forward pass over a prompt of each length and generate() to its first token, each the mean of 3 after 2; again with the mode off in the same process (`--rounds N`: N times each way in turn, the mode first, then the other first, ...); then `--breakdown`: the host's time to issue a pass against the pass's, and a profile of one pass each way |
| `layer.py MODEL` | layer 10's products through GLinear in a prompt's order, 8 layers' Linears over the same packs a pass, each pass timed whole, the median of 5: the mode, today's and bf16; the mode's outputs within 1e-2 of fp32 and the same bits pass to pass |
| `gpu/split_stress.py` | every Qwen3 layer's matrices (0.6B-32B, weights of a trained matrix's spread, a few far out) through the mode at 769-4096 tokens, 36 passes, then GLinear's recording pass and 6 after: every product the same bits across layers and passes, within 1e-2 of fp32 |
| `ring_model.py` | a model check of the mode's ordering over random schedules (`ring_model.txt`: 1,500 schedules) |
| `jobs/` | `o2_job.sh` (the steps; `o2_summary.py` writes summary.txt), `o2_a100.sh` and `o2_hopper.sh` (a GPU class's lists); `o3_job.sh` and `o3_summary.py`, the settle on one Hopper GPU against v0.25.1's (a GH200 or an H100 SXM: the mode as shipped; any other: forced on, a measurement; the H100 SXM's session ran it forced, before the mode was extended to it: `h100-sxm-measure/o3_job.sh` is the script as it ran), its summary a DECIDES line per model and length; `o4_job.sh`, the same on an A100 (Qwen3-8B and 14B at 769-8192 tokens, 8192 forced on: a measurement), with the checks, a second stress run and test_gpu.py whole |

## The sessions

| dir | GPU | tree | what |
| :--- | :--- | :--- | :--- |
| `a100-1` | A100-SXM4-40GB | 7165037 | the first: a pass kept 1-50% of the layers' gain |
| `a100-2` | A100-SXM4-40GB | 4516ea2 | stopped at layer.py's 14B check |
| `l4-stress` | L4 | e744311 | split_stress.py on this tree and on 4516ea2 |
| `a100-3` | A100-SXM4-40GB | 8f10750 | the mode on the A100: checks, the stress, layer.py, e2e.py with its breakdown |
| `gh200` | GH200 480GB | 8f10750 | the mode on Hopper: checks, the stress, layer.py, e2e.py with its breakdown |
| `gh200-settle` | GH200 480GB | 9c218a1 | the settle against v0.25.1's: checks, the stress, e2e.py 3 rounds each way in turn, layer.py |
| `a100-settle` | A100-SXM4-40GB | 4687dd4 | the settle against v0.25.1's (o4_job.sh): checks, the stress, e2e.py 3 rounds each way in turn at 769-8192 (8192 forced on), layer.py, test_gpu.py whole |
| `l4-v0.25.1/review-1` | L4 | d26efb4 | review 1's fixes: check_capi, test_gpu.py whole, the stress at its default setting and two others (30,336 products) |
| `h100-sxm-measure` | H100 80GB HBM3, the SXM5 | 7fe66a2 | the settle on Qwen3-14B against v0.25.1's (o3_job.sh), the mode forced on (the tree's behavior for an H100 SXM was v0.25.1's): checks, the stress, e2e.py 3 rounds each way in turn at 2048-8192, layer.py |

## Results

A forward pass with the mode over v0.25.0's, the same model and prompt in one process (`a100-3/`, `gh200/`: e2e-*.txt,
summary.txt):

| GPU | model | 769 | 1024 | 2048 | 4096 | 8192 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-40GB | Qwen3-8B | 0.887 | 0.851 | 0.952 | 0.964 | 0.988 |
| A100-SXM4-40GB | Qwen3-14B | 0.832 | 0.849 | 0.904 | 0.935 | 1.000 |
| GH200 | Qwen3-8B | | 1.106 | 0.978 | 0.991 | 0.992 |
| GH200 | Qwen3-32B | | 1.012 | 0.895 | 0.935 | 0.938 |

On the A100, Qwen3-8B's pass at 1024 tokens took 95.5 ms with the mode against 112.2 ms with v0.25.0's and 90.0 ms in
bf16; 14B's gate and up (178 M weights) kept v0.25.0's behavior past 4096 tokens (their layer 1.021x with the mode at
8192). On the GH200, 8B's pass at 1024 was issued by the host in 47.6 of its 48.0 ms (its breakdown): the mode's calls
cost the host more than the GPU saved; 32B's at 1024 1.012x. v0.25.1 left an A100's behavior as it was (v0.25.0's and
v0.25.1's libraries choose the same for its GPU; the one kernel change is Hopper's alone), so the A100's ratios stand
against v0.25.1.

The settle on the GH200 against v0.25.1's (`gh200-settle/`: e2e-*.txt, summary.txt), a forward pass's time over
theirs, the median of 3 rounds each way in turn (each round's):

| model | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: |
| Qwen3-8B | 0.978 (0.973, 0.978, 0.980) | 0.994 (0.994, 0.991, 0.994) | 0.994 (0.994, 0.995, 0.990) |
| Qwen3-32B | 0.909 (0.913, 0.909, 0.908) | 0.940 (0.940, 0.941, 0.940) | 0.952 (0.942, 0.952, 0.953) |

The GPU was at its power cap in 96-100% of the samples of both ways' phases, their clocks alike (8B's 1721 MHz on
average with the mode against 1713, 32B's 1580 against 1617).

The settle on the A100 the same way (`a100-settle/`, at 4687dd4; 8192 forced on then, a measurement):

| model | 769 | 1024 | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 0.899 (0.886, 0.899, 0.899) | 0.876 (0.867, 0.879, 0.876) | 0.959 (0.951, 0.959, 0.962) | 0.971 (0.965, 0.971, 0.972) | 0.994 (0.987, 0.994, 0.995) |
| Qwen3-14B | 0.845 (0.834, 0.845, 0.846) | 0.864 (0.857, 0.865, 0.864) | 0.916 (0.905, 0.917, 0.916) | 0.947 (0.941, 0.947, 0.947) | 0.968 (0.965, 0.968, 0.968) |

The first session on an H100 SXM (`h100-sxm-measure/`, tree 7fe66a2, Qwen3-14B, the mode forced on by
`GLYD_SPLIT_MIN=2048`, which is the shipped rule's own choice for Qwen3-14B's matrices) the same way, 3 rounds each way
in turn:

| model | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: |
| Qwen3-14B | 0.893 (0.883, 0.895, 0.893) | 0.937 (0.937, 0.946, 0.924) | 0.938 (0.938, 0.940, 0.935) |

A pass took 119.2, 245.7 and 504.5 ms in bf16 at those lengths, 150.5, 279.0 and 551.4 with v0.25.1's and 134.4,
260.4 and 518.2 with the mode. The GPU was at its 700 W cap in 98% of the samples of the mode's phases and 100% of
v0.25.1's (SM clock 1,696 MHz on average against 1,729). Qwen3-32B was not run there.

Where a pass took at least 2% less time with the mode: an A100 SXM's prompts from 769 to 4096 tokens, and to 8192 for
Qwen3-14B (Qwen3-8B at 8192: 0.6%, not taken); a GH200's and an H100 SXM's from 2048 to 8192 for Qwen3-32B on the
GH200 and Qwen3-14B on the H100 SXM (Qwen3-8B on the GH200: 2.2% at 2048 alone, not taken). An H200, an H100 NVL and
the PCIe cards stay on v0.25.1's behavior until measured; nothing past 8192.

The stress (`l4-stress/`, `a100-3/`, `gh200/`, `h100-sxm-measure/` split_stress.txt): 16,512 products, no failure, on
each GPU. The ordering model held in 1,500 schedules (`ring_model.txt`) and in every stress run.
