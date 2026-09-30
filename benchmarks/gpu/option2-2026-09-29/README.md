# The route SPLIT (option 2), 2026-09-29

A 12-bit prompt's matrices decoded ahead on SMs set apart by the driver's green contexts, while cuBLAS multiplies from
a ring of slots on the other SMs. The measurements that set its routes: models' forward passes by it and by the routes
without it (v0.25.0's, then v0.25.1's) in one process, a layer's products, where a pass's time goes, and a stress check
of the ring. Every session ran
unattended (`jobs/`), its tree a `git archive` of this branch with a COMMIT file.

| script | what |
| :--- | :--- |
| `gpu/e2e.py MODEL --baseline --format mma12 --fused --merge --prefill LENGTHS --without-split --breakdown LENGTHS` | bf16, then Glyd (the Linears merged: q, k, v and gate, up one product each): a forward pass over a prompt of each length and generate() to its first token, each the mean of 3 after 2; again with the route SPLIT off (the routes without it) in the same process (`--rounds N`: N times each way in turn, SPLIT first, then the other first, ...); then `--breakdown`: the host's time to issue a pass against the pass's, and a profile of one pass each way, the GPU's idle time and its kernels by kind (GEMMs, the decode beside GEMMs, beside the rest and alone, attention, the rest) |
| `layer.py MODEL` | layer 10's products through GLinear in a prompt's order, 8 layers' Linears over the same packs a pass, each pass timed whole, the median of 5: the route SPLIT, today's route and bf16; the route's outputs within 1e-2 of fp32 and the same bits pass to pass |
| `gpu/split_stress.py` | every Qwen3 layer's matrices (0.6B-32B, weights of a trained matrix's spread, a few far out) through the ring at 769-4096 tokens, rings of 3-16 slots of three sizes, the order queued whole or a few ahead, 36 passes, then GLinear's recording pass and 6 after: every product the same bits across layers, passes and slot counts, within 1e-2 of fp32 |
| `ring_model.py` | a model of the ring's ordering (glyd_gpu.cu's ring_pump, ring_restart, ring_queue, mma12_ring_linear) on CUDA's stream and event rules, random schedules: every slot written after its last reader's product and read after its decode |
| `jobs/` | `o2_job.sh` (the steps, their budget; `o2_summary.py` writes summary.txt), `o2_a100.sh` and `o2_hopper.sh` (a GPU class's lists); `o3_job.sh` and `o3_summary.py`, the settle on one Hopper GPU against v0.25.1's routes (a GH200: the route as shipped; any other: forced on, a measurement, not a route), its summary a DECIDES line per model and length; `o4_job.sh`, the same on an A100 (Qwen3-8B and 14B at 769-8192 tokens, 8192 forced past the route's end: a measurement), with the checks, the stress skewed both ways and test_gpu.py whole |

## The sessions

| dir | GPU | tree | what |
| :--- | :--- | :--- | :--- |
| `a100-1` | A100-SXM4-40GB | 7165037 | the first: the decode as soon as a slot came free; a pass kept 1-50% of the layers' gain |
| `a100-2` | A100-SXM4-40GB | 4516ea2 | each decode waiting for its gate (the product of the same matrix of the layer before) to start; layer.py's 14B check stopped it (below) |
| `l4-stress` | L4 | e744311 | split_stress.py on this tree and on 4516ea2 |
| `a100-3` | A100-SXM4-40GB | 8f10750 | the A100's routes: checks, the stress, layer.py, e2e.py with its breakdown |
| `gh200` | GH200 480GB (132 SMs) | 8f10750 | Hopper's routes: checks, the stress, layer.py, e2e.py with its breakdown, the decode's SMs at 1024, the scheduling before the gates |
| `gh200-settle` | GH200 480GB (132 SMs) | 9c218a1 | the settle against v0.25.1's routes: checks, the stress, e2e.py 3 rounds each way in turn, layer.py |
| `a100-settle` | A100-SXM4-40GB (108 SMs) | 4687dd4 | the settle against v0.25.1's routes (o4_job.sh): checks, the stress and the stress skewed, e2e.py 3 rounds each way in turn at 769-8192 (8192 forced past the route), layer.py, test_gpu.py whole |
| `l4-v0.25.1/review-1` | L4 | d26efb4 | review 1's fixes: check_capi, test_gpu.py whole, the stress at the default split and skewed both ways (`--sms=-2,1`: 30,336 products), a probe of Split.measured and the cuBLAS lookup |

## Results

A forward pass by the route SPLIT against v0.25.0's routes (today's), the same model and prompt in one process
(`a100-3/`, `gh200/`: e2e-*.txt, summary.txt):

| GPU | model | 769 | 1024 | 2048 | 4096 | 8192 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-40GB | Qwen3-8B | 0.887 | 0.851 | 0.952 | 0.964 | 0.988 |
| A100-SXM4-40GB | Qwen3-14B | 0.832 | 0.849 | 0.904 | 0.935 | 1.000 |
| GH200 | Qwen3-8B | | 1.106 | 0.978 | 0.991 | 0.992 |
| GH200 | Qwen3-32B | | 1.012 | 0.895 | 0.935 | 0.938 |

On the A100, Qwen3-8B's pass at 1024 tokens took 95.5 ms by the route against 112.2 ms by today's and 90.0 ms in
bf16; 14B's gate and up (178 M weights) took today's route past 4096 tokens (their layer 1.021x by the route at 8192).
On the GH200, 8B's pass at 1024 was issued by the host in 47.6 of its 48.0 ms (its breakdown): the route's calls cost
the host more than the GPU saved; 32B's at 1024 1.012x. v0.25.1 left an A100's routes as they were (v0.25.0's and
v0.25.1's libraries: every route and its last token count the same for its code; the one kernel change, the decode's
load order, Hopper's alone), so the A100's ratios stand against v0.25.1.

The settle on the GH200 against v0.25.1's routes (`gh200-settle/`: e2e-*.txt, summary.txt), a forward pass's time
over theirs, the median of 3 rounds each way in turn (each round's):

| model | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: |
| Qwen3-8B | 0.978 (0.973, 0.978, 0.980) | 0.994 (0.994, 0.991, 0.994) | 0.994 (0.994, 0.995, 0.990) |
| Qwen3-32B | 0.909 (0.913, 0.909, 0.908) | 0.940 (0.940, 0.941, 0.940) | 0.952 (0.942, 0.952, 0.953) |

The GPU was at its power cap in 96-100% of the samples of both ways' phases, their SM clocks alike (8B's 1721 MHz on
average by SPLIT against 1713, 32B's 1580 against 1617; the tree then gave 8B's matrices the route too, O and K at least
4096).

The settle on the A100 the same way (`a100-settle/`, at 4687dd4; 8192 forced past the route's end then, a measurement):

| model | 769 | 1024 | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 0.899 (0.886, 0.899, 0.899) | 0.876 (0.867, 0.879, 0.876) | 0.959 (0.951, 0.959, 0.962) | 0.971 (0.965, 0.971, 0.972) | 0.994 (0.987, 0.994, 0.995) |
| Qwen3-14B | 0.845 (0.834, 0.845, 0.846) | 0.864 (0.857, 0.865, 0.864) | 0.916 (0.905, 0.917, 0.916) | 0.947 (0.941, 0.947, 0.947) | 0.968 (0.965, 0.968, 0.968) |

The routes, where a pass took at least 2% less time: an A100 SXM's prompts from 769 to 4096 tokens, every matrix, and
to 8192 for a matrix whose O and K are both at least 5120, as Qwen3-14B's (8B's, 4096 on a side: 0.6% at 8192, not
taken); a GH200's from 2048 to 8192 for such a matrix, as Qwen3-32B's (8B's: 2.2% at 2048 alone, for a ring of 600
MiB, not taken). An H100 SXM, an H200 and the PCIe cards on v0.25.1's routes until measured; nothing past 8192.

Where the time went before the gates (`a100-1` against a layer's products): the rest of a Qwen3-8B pass at 1024 tokens
(norms, activations, rotary, attention) took 43.5 ms by the route against 28.5 in bf16 and 28.9 by today's route: the
decode ran as soon as a slot came free, the end of a product, so beside those memory-bound kernels, taking a third of
the A100's bandwidth. Gated (`a100-3`, breakdown at 1024): the decode beside them 0.8 of its 43.8 ms, the rest 29.9 ms
against 29.5 without the route; on the GH200 the scheduling before the gates (GLYD_SPLIT_SLOTS=3) had the decode beside
them 11.6 of its 19.8 ms.

The stress (`l4-stress/`, `a100-3/`, `gh200/` split_stress.txt): 16,512 products, no failure, on each GPU. On the tree
before 244baf4, GLinear's passes after its first were not its bits (96 of 1,984 on the L4 and the A100, 24 on the
GH200): a device's first prompt, the order being recorded, cut its matrices in 100 MiB row chunks, the prompts after
it in the planned slots' (Qwen3-14B's 178 MB, 32B's 262 MB), other cuBLAS calls; since, a device's ring keeps one slot
size. layer.py's reference in `a100-2` was such a recording pass (its Split dropped between passes), the "differ pass
to pass" that stopped it; the ring's ordering held in ring_model.py's 1,500 schedules (`ring_model.txt`; with the
decode's wait on its slot's last product taken out, 142 of 300 fail, with the product's on its decode, 273) and in every
stress run.
