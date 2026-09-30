# Changelog

All notable user-facing changes. Measurement history, floors and refuted
ideas live in [CHANGELOG-BENCH.md](CHANGELOG-BENCH.md).
Versioning follows [SemVer](https://semver.org); the on-disk format has its
own version in every block header (v6, v7) and every release decodes
every earlier format.

## v0.26.0 (Unreleased)

- Long prompts on an A100 SXM and a GH200 decode on SMs set apart (the
  route SPLIT): each 12-bit matrix is decoded ahead into a ring of slots on
  a few SMs the driver's green contexts set apart, while cuBLAS multiplies
  from the ring on the others, told how many. Each decode waits for the
  product of the same matrix of the layer before to start, so it runs
  beside the products and not beside the norms, activations and attention
  between them. One forward pass against the routes without it, the same
  model and prompt in one process (`e2e.py --prefill --merge`): on an
  A100-SXM4-40GB against v0.25.0's routes (v0.25.1 left an A100's as they
  were), Qwen3-8B's 0.887 / 0.851 / 0.952 / 0.964 / 0.988 at 769 / 1024 /
  2048 / 4096 / 8192 tokens (95.5 ms at 1024 against 112.2, bf16's 90.0)
  and 14B's 0.832 / 0.849 / 0.904 / 0.935 / 1.000; on a GH200 against
  v0.25.1's routes, the median of 3 rounds each way in turn, Qwen3-32B's
  0.909 / 0.940 / 0.952 at 2048 / 4096 / 8192 and 8B's 0.978 / 0.994 /
  0.994. So the routes, where a pass took at least 2% less time: an A100
  SXM's 12-bit prompts from 769 to 4096 tokens, a GH200's from 2048 to
  8192 for a matrix whose O and K are both at least 5120, as Qwen3-32B's
  (8B's matrices, 4096 on a side, not taken: 2.2% at 2048 alone, for a
  ring of 600 MiB; at 1024 tokens the GH200's first session had the route
  lose, 1.106 and 1.012 of v0.25.0's routes' time). An H100 SXM, an H200
  and the PCIe cards keep v0.25.1's routes until a session measures them,
  and nothing past 8192 tokens takes it, not measured. The ring holds a
  layer's chunks ahead: 600 MiB for Qwen3-8B, 1.0 GiB for 14B, 1.5 GiB for
  32B, and a 32 MiB cuBLAS workspace. Its products are cuBLAS's own on the
  decoded bf16, a row chunk a call: the same bits run to run and prompt to
  prompt, not bit for bit a whole-matrix product, so `exact=True` never
  takes it. Where it cannot run (the JIT build, a driver before CUDA 12.4,
  MIG or MPS, too little memory, a CUDA graph capture, a torch.compile
  graph's node) a prompt takes the routes before it; `GLYD_SPLIT_MIN=-1`
  turns it off, `GLYD_SPLIT_MIN`, `GLYD_SPLIT_MAX` and `GLYD_SPLIT_SMS` move
  it. A stress check (`gpu/split_stress.py`, in test_gpu.py quick): every
  Qwen3 layer's matrices, 0.6B-32B, at 769-4096 tokens, rings of 3-16
  slots, 36 passes; 16,512 products the same bits across layers, passes
  and slot counts, within 1e-2 of fp32, on an L4, an A100 and a GH200
  ([benchmarks/gpu/option2-2026-09-29](benchmarks/gpu/option2-2026-09-29)).
- C API version 6: the ring (`glyd_gpu_ring_create`, `_destroy`, `_split`,
  `_reset`, `glyd_gpu_mma12_ring_queue`, `glyd_gpu_mma12_ring_linear`,
  with the caller's cuBLAS as `glyd_gpu_blas`: the library does not link
  it), its decode (`glyd_gpu_mma12_unpack_split`), the route
  `GLYD_GPU_ROUTE_SPLIT` and its SMs (`glyd_gpu_mma12_split_sms`), a GPU's
  PCIe and GH200 classes in its code (`GLYD_GPU_PCIE`, `GLYD_GPU_GH200`:
  an A100 PCIe 5080, an H100 PCIe 5090, a GH200 6090) and
  `GLYD_GPU_WITH_SPLIT` (a code's routes with SPLIT). The route is opt-in:
  only a code with that flag gets it, which the glyd package's Linears ask
  for where the split can run; `glyd_gpu_*_linear`'s own route (-1) never
  is, so the C API's other callers (the glyd-gpu crate, the vLLM plugin)
  keep v0.25.1's routes. `glyd_gpu_mma12_linear` given SPLIT takes it by the
  prompt kernel. v0.25's libraries (version 5) are refused by this package
  and the glyd-gpu crate (`Route::Split`, `PCIE`, `GH200`, `WITH_SPLIT`,
  `Library::split_sms`; the ring declared, not wrapped yet).
- `gpu/e2e.py --without-split` times a prompt again with the route off in
  the same process (`--rounds N`: N times each way in turn); `--breakdown`
  gives a pass's host time and its GPU time by kind of kernel.

## v0.25.1 — 2026-09-29

- On Hopper the 12-bit layout's decode of a whole matrix (for cuBLAS,
  prompts past wgmma's 1024 tokens; `exact=True`'s steps and prompts; a
  mixture of experts' exact decode) loads a step's low bytes and
  exception bounds first, then its codes once those are in. v0.25.0
  issued all three at once, and its decode there took 3.0-6.0% longer
  than the 12-bit layout's before split byte (H100 SXM, GH200). On a
  GH200, layer 10 of Qwen3-8B, 14B and 32B, over the decode before split
  byte: 0.964-0.975 (v0.25.0 1.035-1.060). Measured on a GH200 alone;
  the H100 SXM, where v0.25.0's 3.0-5.8% was measured, and the H100 PCIe
  were not run again. The decode ahead of a prompt's products (a few
  warps an SM: GeForce Ada's, an A10's and an L40S's prompts) keeps
  v0.25.0's loads, the faster for it (0.703-0.725 on the GH200; the new
  order 1.058-1.068), as does every other GPU's decode (on an L4 the new
  order took 1.6-1.9% longer than all three at once; the others were not
  measured). The same bits: on the GH200 the self-test and xcheck.py with
  73b9560's order 3, whose sm_90a instructions this release's two decodes
  have (sass-final.txt); on the L4 with this release's library
  ([benchmarks/gpu/decode-fix-2026-09-29](benchmarks/gpu/decode-fix-2026-09-29)).
- An L4's prompts decode each matrix for cuBLAS first, on the current
  stream, from 896 tokens in the tiered layout (the L4's default) and 2560
  in the 12-bit one; `exact=True`'s prompts as before. At its 72 W cap the
  fused prompt kernel lost to the decode from those lengths, by more the
  longer the prompt (Qwen3-4B-Instruct-2507's tiered two were even at 1024
  tokens), and a decode ahead beside cuBLAS (the A10's route) gained
  nothing (within 1% at 4096-8192 tokens, 1-7% slower at 896-2048).
  Qwen3-8B, one forward pass, over bf16's time in the same run at 1024 /
  2048 / 4096 / 8192 tokens: tiered +25.5 / +11.3 / +8.4 / +4.7% (were
  +27.8 / +31.0 / +37.9 / +98.4%), 12-bit at 4096 / 8192 +9.3 / +4.4%
  (were +19.8 / +105.0%); to 895 and 2559 tokens as before. The time to
  the first token through `generate()` moves with the pass. The tiered
  layout stays the L4's default (33% less memory); for the fastest short
  prompts, at 25% less, load with `layout="mma12"`: its prompts took 5-20%
  less time than the tiered layout's to 1536 tokens, and about the same
  from 1792 (Qwen3-8B and Qwen3-4B-Instruct-2507). The L4 is a class of
  its own in the library's GPU codes (`GLYD_GPU_L4`, 3000: "L4" in the
  name as a word; an L4's code is 3089), so that the L40 and RTX 6000 Ada,
  which share its compute capability and were not measured, keep their
  routes; the glyd package and the glyd-gpu crate have it too (`L4`).
  `GLYD_DEC_MIN` still sets any GPU's 12-bit threshold. check_capi pins
  the L4's routes and an L40's
  ([benchmarks/gpu/l4-routes-2026-09-29](benchmarks/gpu/l4-routes-2026-09-29)).
- An L40S's prompts decode each matrix ahead of its product, beside the
  products before it (the route AHEAD, as an A10's), from 1024 tokens in
  the tiered layout and 2048 in the 12-bit one; `exact=True`'s prompts as
  before. Qwen3-8B on an AWS g6e.xlarge, one forward pass, over bf16's
  time in the same run at 1024 / 2048 / 3072 / 4096 / 8192 tokens: tiered
  +30.3 / +11.9 / +8.0 / +10.9 / +3.8% (were +38.1 / +41.2 / +37.3 /
  +39.4 / +35.0%), 12-bit at 2048 / 3072 / 4096 / 8192 +12.9 / +7.7 /
  +11.3 / +3.9% (were +15.5 / +14.9 / +20.0 / +18.2%); to 1023 and 2047
  tokens as before. The decode ahead took 0.4-5.1% less time than a
  decode first at 1024-3072 and 8192 tokens and 0.8-1.0% more at 4096.
  The scratch buffer holds two matrices there (Qwen3-8B's 0.40 GB, was
  0.27). The L40S's 12-bit prompts took 17.3 / 18.1 / 12.8 / 4.0% less
  time than the tiered layout's at 512 / 768 / 1024 / 1536 tokens (its
  fused kernel at 512 tokens -0.3% over bf16's time, the tiered one's
  +20.6%) and were within 0.9% of them from 2048; the tiered layout stays
  its default (33% less memory; `layout="mma12"` for 25%). The L40S is a
  class of its own (`GLYD_GPU_L40S`, 4000: "L40S" in the name as a word;
  an L40S's code is 4089; the package and the crate have it too), so the
  L40 and RTX 6000 Ada keep their routes until measured; check_capi pins
  its routes.
- `glyd_gpu_*_route` at M = INT64_MAX tokens gives the route before it,
  as `last` says (with `GLYD_WG_MAX` below INT64_MAX - 1; v0.25.0 gave
  AHEAD in the tiered layout and DECODE in the 12-bit one there, on every
  GPU).
- This release's code (dc490e4) built by build_lib.sh and checked on an
  L4: the self-test, xcheck.py, check_capi.py (also with `GLYD_DEC_MIN`
  1000 and 3000), test_gpu.py and the crate's tests pass
  ([benchmarks/gpu/decode-fix-2026-09-29/l4-head](benchmarks/gpu/decode-fix-2026-09-29/l4-head)).

## v0.25.0 — 2026-09-29

- The 12-bit layout is split byte: a weight's low byte (the exponent's
  lowest bit and the mantissa) kept as it is, its high byte (the sign and
  the exponent's other 7 bits) a 4-bit code, the sign and an offset 0-7
  from the matrix's base `hb` (exponents 2·hb to 2·hb + 15: the window of
  16 from an even exponent holding the most weights), any other weight in
  its step's exception list as before. Its
  decode is an AND and an add for four weights and a byte permute for two,
  with no table (8.5 integer instructions a k-block in SASS against the
  15-exponent code's 22.0), in every kernel of the layout: the step,
  mid, prompt, Hopper TMA and wgp, A100 mid, mixture-of-experts and
  decode kernels. The same size (12.04-12.07 bits a weight) and the same
  bits decoded, so the same products: on an L4, an A10, an A100 SXM4 40
  GB, an H100 PCIe and an H100 SXM (2026-09-29) every output of the
  layout's kernels was main's bits; models' logits and greedy tokens,
  fused and exact, the same as main's: Qwen3-1.7B and
  granite-3.1-3b-a800m-instruct on the L4, A10, A100 and H100 PCIe
  (round 1), and those two and Qwen3-4B-Instruct-2507 on an L4 on the
  release candidate. A layer's time against the 12-bit layout's
  before, main's library and this release's in one process (layer 10 of
  Qwen3-8B with 4B-Instruct-2507, 14B or 32B, two runs each): faster on
  the H100 SXM, 0.933-0.969 at 32-1024 tokens (wgmma: 3.1-6.7% less; the
  H100 PCIe 0.942-0.991) and 0.986-0.992 at 1-16; faster on the A100,
  0.923-0.987 at 64-128 (its mid kernel: 5.5-7.7% less at 128),
  0.969-0.982 at 256-768 (prompts) and 0.976-0.998 at 1-16; the same
  to 1.5% faster at 32 (0.985-1.003); the same on the A10 and the L4, 0.989-1.010 at
  1-1024 tokens (the A10's from 640 by `linear`'s prompt kernel, where
  glyd.gpu decodes those prompts ahead, below). Slower: a matrix decoded
  whole, for cuBLAS and exact mode, takes 3.0-5.8% longer on the H100 SXM
  and 1.0-1.6% on the A10 (0.998-1.012 on the A100, the same on the L4).
  That decode is in Hopper's prompts past wgmma's 1024 tokens, the A100's
  past 768, an A10's 12-bit prompts from 640 tokens (decoded ahead beside
  cuBLAS; the prompt's time not measured) and every step of exact mode
  ([benchmarks/gpu/splitbyte-2026-09-29](benchmarks/gpu/splitbyte-2026-09-29)).
  The 12-bit layout's bytes and words change from 0.24's (a code into the
  15 commonest exponents, v0.19-v0.24): the library refuses 0.24's words,
  so pack those models again. The C API is version 5: a 12-bit
  pack's `sym[4]` holds its base, `hb` in each byte of `sym[0]` and the
  other three zero; any other words, 0.24's exponents among them, are
  refused with `cudaErrorInvalidValue`. glyd.json's 12-bit packs
  (glyd-v3) carry `hb`; a glyd-v3 save of the layout before (its `sym`) is
  refused as it loads: save it again. `glyd pack --layout mma12` packs
  split byte on the CPU, byte for byte as `pack_mma12` (the glyd-gpu
  crate's test holds its bytes to a CPU port of it, every bf16 bit pattern
  and the base at 0 and 120 among its matrices), and `glyd verify` decodes
  it and refuses a 12-bit pack without `hb`. Tiered saves (glyd-v1,
  glyd-v2) are unchanged.
- Saved models in the 12-bit layout too: `glyd.save_pretrained(model,
  path, layout="mma12")`, `python -m glyd.gpu pack MODEL OUT --layout
  mma12` and `glyd pack MODEL OUT --layout mma12` write glyd-v3, the packs
  as an A10, A100 or H100 runs them (each pack's `.glyd_data`, `.glyd_exc`,
  `.glyd_exc_base`; its base, `hb`, in glyd.json), which
  `from_pretrained` loads as saved where the 12-bit layout is the one
  (else decodes and packs again, as it does a tiered save there; glyd 0.24
  and before refuse glyd-v3 by its format). The same bytes from Rust and
  Python for the five models of `glyd pack` below, and `glyd verify` reads
  it. Loaded for the 12-bit layout on an RTX 4080 SUPER
  (`layout="mma12"`, warm cache, three fresh processes each; measured on
  the 12-bit layout before split byte, whose load reads the same buffers
  and decodes nothing):
  granite-3.1-3b-a800m-instruct in 0.49 s against 1.32-1.33 s from the
  tiered save and 1.44-1.45 s from the bf16 checkpoint,
  Qwen3-4B-Instruct-2507 in 0.84-0.85 s against 1.76-1.77 and 1.95,
  Qwen3-8B in 1.17-1.18 s against 3.14-4.55 and 3.54-3.61 (2.7-3.9x
  faster than packing again; 2.1-3.9x across the three), its peak 0.56 GB
  below the tiered save's
  (14.34 GB against 14.90)
  ([benchmarks/gpu/rtx4080s-rust-2026-09-28](benchmarks/gpu/rtx4080s-rust-2026-09-28)).
- glyd.json holds the sha256 of every tensor saved as it is too (its
  `tensors`: the norms, biases, an embedding or output layer not packed),
  and verify checks them: `python -m glyd.gpu verify`,
  `from_pretrained(verify=True)` and `glyd verify` hold each file's
  tensors back to back to its end, every tensor to a pack's buffer or one
  of those sha256, and each pack's tensors to its own (its module's; a
  merged group's q, k, v or gate, up), so a byte changed outside the packs,
  bytes appended to a shard or a renamed member fails them. Both savers
  write it alike; glyd 0.24 and before load such a save and ignore it. verify
  requires the map where the save's format or glyd says it is there
  (glyd-v3, and a save of glyd 0.25 on) and refuses a key glyd.json does
  not have in a save of this glyd or an older one, so a damaged map cannot
  pass for an older save (a newer glyd's key is skipped with a warning,
  the rest checked); a save of glyd 0.24 or before verifies as it did, its
  other tensors counted unchecked.
- `glyd pack MODEL OUT` and `glyd verify PATH` in the Rust CLI: a bf16
  checkpoint (a directory, or a repo in the local Hugging Face cache)
  packed on the CPU and saved as glyd-v1 (glyd-v2 with a mixture of
  experts' packs, glyd-v3 in the 12-bit layout) with no Python, PyTorch or
  GPU, byte for byte as `python -m glyd.gpu pack` saves it: every file of
  Qwen3-0.6B, 1.7B, 4B-Instruct-2507 and 8B and of
  granite-3.1-3b-a800m-instruct (its experts glyd-v3's mixture of
  experts) is Python's in both layouts, the 12-bit one in split byte
  (their sha256; glyd.json, the shards, the index; three rounds each),
  each pack decoded back and checked as it is made, and each Rust save
  verified by `glyd verify` on the CPU and the GPU and by `python -m
  glyd.gpu verify`. On the AWS dev machine (a g6.4xlarge: 16 vCPUs of an
  AMD EPYC 7R13, an NVIDIA L4), 16 threads, three rounds each: 1.55 /
  1.55-1.58 / 1.58-1.60 / 0.71-0.88 / 1.44-1.45 GB/s of bf16 tiered and
  1.64-1.67 / 1.88-1.91 / 2.01-2.03 / 0.67-0.72 / 1.78-1.80 GB/s in the
  12-bit layout (Qwen3-8B's 16.4 GB in 18.5-23.0 s and 22.8-24.6 s, at
  the pace of the disk its shards were written to: 3.2-6.4 of the CPUs
  busy). A save holds two shards (about 5 GB each) and at most a shard's
  worth of weights in flight past the one it waits for (about 15 GB at
  most for a model of several shards; Qwen3-8B measured 11.4-13.6 GB peak
  RSS on 16 threads). `glyd verify` checks a save
  as `python -m glyd.gpu verify` does, its packs decoded on the CPU or
  (`--device cuda:0`) on the GPU by the library (Qwen3-8B's 253 packed
  tensors and 146 saved as they are in 11.6-11.8 s tiered, 8.8-9.0 s
  12-bit, on 16 threads)
  ([benchmarks/gpu/l4-rust-2026-09-29](benchmarks/gpu/l4-rust-2026-09-29)).
  The families are written out as transformers 5.17 holds them: Qwen3,
  Qwen2, Llama, Mistral, Granite and GraniteMoe for now (tiny random
  checkpoints of each family save the same bytes too, both layouts),
  anything else refused with Python's command. The commands are
  the `glyd-gpu` program's, under the Business Source License as the rest
  of the GPU code, which the glyd CLI runs (it ships beside glyd; a file
  named `pack` or `verify` is compressed as `./pack`).
- The `glyd-gpu` crate: Rust over the GPU library's C API, the library
  loaded at run time and refused where its API version is not the
  crate's, each function typed (a test holds the declarations, the routes'
  numbers and the GPU classes to `glyd_gpu.h`), device pointers and
  streams the caller's, a product's workspace query first, errors as
  `Result`, and the few CUDA driver calls a caller without a runtime of
  its own needs (a context stays on the thread that made it; copies take
  plain integers and floats); no dependency but sha2, and nothing linked
  at build time. Its examples decode a saved model's packs on the GPU
  against the bf16 checkpoint, bit for bit (`unpack.rs`, the C example in
  Rust), and multiply by `linear` (`linear.rs`); its `pack` module packs a
  matrix in either layout on the CPU, byte for byte as glyd.gpu's
  `pack_mma` and `pack_mma12` do on the GPU.
- The kernel a product for M tokens takes on a GPU is the library's:
  `glyd_gpu_mma_route` and `glyd_gpu_mma12_route` give it (and the last
  token count that takes it), as glyd.gpu 0.24 chose it (a check against
  0.24's rule on eleven GPU codes, 0-5000 tokens, both layouts): Hopper's
  12-bit steps and prompts to 1024 tokens by wgmma, an A10's prompts
  decoded ahead from 512 tokens tiered and 640 12-bit (not exact),
  GeForce Ada's past 512 and 1792. `GLYD_WG_MIN`, `GLYD_WG_MAX`,
  `GLYD_MID_MIN` and `GLYD_DEC_MIN` (any GPU's 12-bit prompts) are read
  there, once a process, at the library's first route (when the first
  model is loaded or compressed; the prebuilt library and the JIT build
  each at their own): set them in the environment before that. A later
  change has no effect, nor has assigning glyd.gpu.model's `WG_MIN`,
  `WG_MAX`, `MID_MIN` or `DEC_MIN`, or a GLinear's `dec` or `mid`, which
  are gone (`GLYD_AHEAD_MIN` is read at import as before, and a GLinear's
  `ahead` can still be set, then `lin.step = lin._step()`). A value that is
  not a whole number fails the import of glyd.gpu.model, as it did at
  `int()` (the library alone takes it as unset). A GPU's code is its compute capability plus a class where the
  name tells GPUs apart (`GLYD_GPU_GEFORCE`; `GLYD_GPU_A10`, an A10 and
  not an A10G, A40 or A6000). `glyd_gpu_mma_linear` and
  `glyd_gpu_mma12_linear` run a route's kernel (where glyd decodes the
  matrix for cuBLAS, the prompt kernel, on every GPU; where K is not a
  multiple of 64, past 64 tokens (12-bit: also from `GLYD_DEC_MIN` tokens
  where that is lower), they refuse those routes with
  `cudaErrorNotSupported`: decode it there for a GEMM of your own; the WG
  route's done counters at least 1024, as `glyd_gpu_mma12_gemm_wg`'s);
  the glyd package's Linears take their routes from the library and
  multiply by `linear` in their one C call, so every caller routes alike.
  C API version 5 (4 with the 12-bit layout before split byte: builds of
  main alone, refused). The same bits (check_capi on an RTX 4080 SUPER: 6975
  calls through both hosts, bit for bit, and 220044 routes as 0.24's
  rule) and the same speed (generate()
  eager, before the merge with 0.24: main's package and library and these
  in turn, four rounds, RTX 4080 SUPER: Qwen3-1.7B and
  Qwen3-4B-Instruct-2507 at 1, 8 and 32 sequences in both layouts, each
  round -1.1% to +1.3% of main's, their means -0.3% to +0.5%).

## v0.24.0 — 2026-09-28

- `generate()` on a model from `glyd.from_pretrained` or
  `glyd.gpu.compress` runs compiled by default on PyTorch 2.13.0 or later
  (measured on 2.14; below it, a 2.13 pre-release included, it stays
  eager, as in 0.23), as `generate(..., cache_implementation="static")`
  asks transformers to run it (a static cache, the forward under
  `torch.compile` with CUDA graphs): a step's host time goes. Tokens/s
  generating 128 tokens at 1 / 8 sequences on an RTX 4080 SUPER, plain
  `generate()`, each in a process of its own: Qwen3-1.7B 184.6 / 1260 (was
  96.6 / 772; asked for with `cache_implementation="static"` 184.9 /
  1260), Qwen3-4B-Instruct-2507 94.3 / 610 (was 75.1 / 563), Qwen3-8B 55.3
  / 386 (was 48.6 / 365), granite-3.1-3b-a800m-instruct 232.3 / 1547 (was
  90.3 / 699). The first call compiles: Qwen3-8B's took 17.5 s with
  PyTorch's compile caches empty, 6.7 s in a later process: warm up with
  one short `generate()` before serving (a streamer's consumer waits
  through it). Greedy tokens compiled can differ from 0.23's eager loop's,
  as a compiled bf16 model's can from its eager ones (the first 8 of 32
  the same on Qwen3-0.6B, 17 on Qwen3-1.7B, 32 on
  granite-3.1-3b-a800m-instruct: check_api). In the default mode they can
  also vary within a process, between calls whose cache sizes compile
  differently (Qwen3-1.7B's first compiled call and its later ones, after
  a longer cache, shared 13 of 32 in one run of check_api:
  benchmarks/gpu/rtx4080s-fastloop-2026-09-28/checks-merge-6e17b3f);
  `exact=True` is never compiled and stays bit-identical to bf16.
  `compile=False` (`from_pretrained`, `compress`) or `GLYD_COMPILE=0` runs
  it eager, as before; so do `exact=True` (its tokens are bf16's eager
  ones), a family transformers does not compile whole (its
  `_can_compile_fullgraph`), a model over several GPUs, and a transformers
  whose generation helpers are not as 5.17 has them (one warning at the
  load). Only greedy and sampled calls compile: a call runs as
  transformers runs it if it uses several beams, an assistant or another
  assisted mode (prompt lookup, early exit, `use_mtp`), its own cache or a
  `cache_implementation`, `use_cache=False`, `return_dict_in_generate`,
  attentions or hidden states, `custom_generate`, or
  `disable_compile=True` (one call eager). A call whose static cache would
  hold more positions in all (its sequences times the prompt and
  `max_new_tokens`, or `max_cache_len` where longer) than 1280 on a
  GeForce card and 2048 on another (`GLYD_COMPILE_MAX` sets it) runs eager
  too: the static cache holds every position the call may reach from its
  first step and each step's attention reads all of it, and past that the
  eager loop was as fast, sooner with a desktop's CPU (Qwen3-8B, a step's
  ms compiled against eager with 1024 / 2048 / 4096 positions held and 64
  used: 20.4 / 23.0 / 27.6 against 20.5 on an RTX 4080 SUPER with a Ryzen
  9 7950X3D, 31.3 / 34.9 / 42.8 against 36.9 / 37.2 / 33.6 on an A10 with
  a Xeon Platinum 8358). Where transformers 5.17's static cache fails
  (bf16's too) it runs eager from the start: Llama 4, and a model with
  multi-head latent attention whose config has fewer key/value heads than
  heads (tiny DeepSeek V2, V3, Kimi Linear and AXK1 test models; the
  released checkpoints compile). A call whose forward fails to compile
  anyway runs again eager from its start (a streamer gets only what the
  failed attempt had not streamed, and a sampled call draws again from the
  random state it started with: its tokens and text are the eager run's),
  and so do the model's later calls, with one warning; any other error,
  out of memory included, is the call's own, and the next call compiles.
  Each model's forward compiles to a graph of its own, so Glyd's compiled
  calls run with `torch._dynamo.config.recompile_limit` at 64 at least,
  for those calls alone (dynamo compiles 8 graphs a frame by default and
  runs the rest uncompiled; ten Qwen3-0.6B models one after another in a
  process all compiled (graphs 2 to 11), the second to tenth at
  296.5-301.3 tokens/s against 99.6 eager; the process's own setting is
  left as it is). A model that has generated compiled is freed at `del`,
  as an eager one (its compiled forward does not refer to it, as
  transformers' own does). How: the model's class's `generate` and
  `get_compiled_call` are taken over once for the process; a model of the
  class Glyd did not set up (bf16, or `compile=False`) runs transformers'
  own, and `compile=False` or `GLYD_COMPILE=0` takes nothing over.
  transformers sets `TOKENIZERS_PARALLELISM=0` for the process where it
  compiles; Glyd puts back the value it had, or its absence, after each
  compiled call (per call: two calls at once in two threads can leave it
  0, as transformers' own compiled calls do).
- Prompts on an A10 (150 W, full-rate tensor cores), but `exact=True`'s,
  decode each matrix ahead of its product, beside the products before it,
  from 640 tokens in the 12-bit layout and 512 in the tiered one, as
  GeForce Ada's do: the fused kernel's decode costs the A10 clock at its
  power cap, more the longer the prompt. Qwen3-8B, one forward pass, over
  bf16's time at 1024 / 2048 / 4096 tokens: +10.0 / +5.2 / +2.6% (were
  +30.3 / +38.8 / +50.9%); to 639 tokens as before (+2.2% at 128, +15.8%
  at 512). The scratch buffer holds two matrices there (Qwen3-8B's 0.40
  GB, was 0.27). The A10G (half-rate tensor cores, its fused prompts at
  most +5.3% over bf16's to 4096 tokens) and the L4, L40S and RTX 6000 Ada
  (half the A10's bandwidth a FLOP) keep their routes until measured.
- Prompts of 129-1024 tokens on Hopper multiply in a new kernel,
  `mma12_wgp_kernel` (`mma_gemm_wg` past 128 tokens; `GLYD_WG_MAX` is
  1024, was 512): a block an SM staying for the whole product,
  warp-specialized as CUTLASS 3.x's and vLLM's Hopper mixed-input main
  loops are (a TMA warp filling a ring of stages, two consumer warpgroups
  decoding a k-block at a time into wgmma's registers while the k-blocks
  before it multiply), in clusters of two sharing X's tiles by TMA
  multicast, the last wave's tiles split by stages, each over a whole
  number of clusters where that idles at most a sixth of them. On an H100
  SXM (benchmarks/gpu/h100-hopper2-val-2026-09-28) Qwen3-8B's decoder
  layer (q, k, v and gate, up merged) takes 1.16 / 1.24 / 1.43 / 1.38x
  cuBLAS's time at 129 / 256 / 512 / 1024 tokens, and one forward pass
  over 1024 tokens (`gpu/e2e.py --prefill --merge`) 45.0 ms against bf16's
  36.9; past 1024 tokens the matrices are decoded for cuBLAS as before. In
  an earlier run (benchmarks/gpu/h100-hopper2-2026-09-28), with the kernel
  as first written (before whole tiles and the zeroing below), main's path
  took Qwen3-8B's layer 1.31 / 1.33 / 1.55 / 1.64x there and its pass 48.4
  ms (bf16 36.4); the kernel took Qwen3-14B's layer 1.14 / 1.28 / 1.44 /
  1.41 / 1.33x at 129 / 256 / 512 / 768 / 1024 tokens (main's 1.14 / 1.30
  / 1.45 / 1.92 / 1.68x) and Qwen3-32B's 1.12 / 1.19 / 1.39 / 1.36 / 1.35x
  (main's 1.17 / 1.24 / 1.42 / 1.88 / 1.68x), Qwen3-32B's pass over 1024
  tokens 166.3 ms (main's 194.9; bf16 135.5) and over 512 90.9 (94.2;
  73.7); every layer faster than main's but Qwen3-14B's at 129-160 tokens,
  level, while Qwen3-8B's and 14B's o alone were 6-16% slower than in the
  old kernel at 129-512 tokens (8B's at all six lengths measured, 14B's at
  129-256), and a few other products by 4% at most. Whole tiles, timed in
  a second run against the kernel before them
  (benchmarks/gpu/h100-hopper2-cu12-2026-09-28), take 8-19% off Qwen3-8B's
  o at 129-1024 tokens and 6-10% off 14B's at 129-256, about what they had
  been slower, 3-11% off 14B's q, k, v at 129-512 and 4-7% off 32B's o at
  129-256; they also change the splits of Qwen3-8B's gate_up at 129-512
  tokens and its down at 129-512 and 1024 (in the 8B layer above), and of
  14B's and 32B's down at 129-256 and 32B's q, k, v at 1024, not timed.
  Where a tile splits differently its sums add in another order, so some
  outputs past 128 tokens differ from before in their last bits.
  Generation runs the same machine code as before but for the TMA kernel's
  accumulator (below), whose products at 17-128 tokens took as long as
  before in the second run's CUDA 13 libraries (a median 0.0% apart, 1.1%
  less to 2.2% more). A layer still takes more than cuBLAS's time: the
  decode's integer instructions cost the tensor cores a quarter to a third
  more time even beside them (`gpu/README.md`).
- On Hopper, the CUDA 12 library (`libglyd_gpu_cuda12.so`, built with CUDA
  12.8; the wheels load it for PyTorch built for CUDA 12) ran the 12-bit
  layout's tensor-core products one at a time, the TMA kernel's in v0.22.0
  and v0.23.0 too: the accumulator was left unset until a tile's first
  product, and for that CUDA 12.8's ptxas serialized every wgmma (its
  warning C7515; CUDA 13's did not). It is now zeroed first. In the second
  run's libraries (on an H100 SXM,
  benchmarks/gpu/h100-hopper2-cu12-2026-09-28), the CUDA 12 library's
  products (Qwen3-8B's four and Qwen3-32B's o and gate_up at 17-1024
  tokens) took a median 4% less time than before, up to 9%, Qwen3-8B's
  layer 2-8% less (1.58x cuBLAS's time at 1024 tokens before, 1.45x with
  the zeroing alone), as fast as the CUDA 13 library's (a median 0.3%
  apart); the CUDA 13 library's took as long as before (a median 0.2%
  apart), and every output was the same, bit for bit. As committed, the
  CUDA 12 library's Qwen3-8B layer takes 0.94-1.42x cuBLAS's time at
  17-1024 tokens, within 0.8% of the CUDA 13 library's
  (benchmarks/gpu/h100-hopper2-val-2026-09-28).
- Validated as committed on an H100 SXM, with the library built for CUDA
  12.8 and for CUDA 13.0 (benchmarks/gpu/h100-hopper2-val-2026-09-28): the
  full self-test passed through both, a K = 128 matrix whose tiles of two
  stages split over clusters included; the two libraries' 342
  `mma_gemm_wg` outputs on the self-test's matrices (1-2100 tokens) are
  the same, bit for bit; and `gpu/e2e.py --exact` gives bf16's logits bit
  for bit through both, 8 of 8 tokens.
- `GLYD_DEC_MIN` (a prompt's products decoded for cuBLAS from that many
  tokens, 12-bit layout) now applies on any GPU where it is set (on Hopper
  the 12-bit layout's prompts to `GLYD_WG_MAX` tokens are still wgmma's);
  unset, an A100's prompts are decoded from 769 tokens as before, and
  elsewhere none.
- The C API is version 3: `glyd_gpu_mma12_gemm_wg` takes at least 1024
  done counters (as many as O / 64 where that is more); the package's
  calls always gave it that many.

## v0.23.0 — 2026-09-28

- Prompts on GeForce Ada (RTX 40) multiply faster, with the same bits:
  `mma_gemm_big`'s products there keep their four producer warps and have
  eight consumers of 64 tokens by 32 rows in place of four of 64 by 64
  (two a scheduler, each as lean), in blocks of 256 tokens in both layouts
  and of 128 in the 12-bit one. On an RTX 4080 SUPER a Qwen3-1.7B, 4B or 8B
  layer's products take 1.7-4.3% less time in the 12-bit layout at 256-4096
  tokens (Qwen3-4B-Instruct-2507's 1.03-1.04x cuBLAS's at 512-4096, were
  1.06-1.08x) and about 1-2% less in the tiered layout; the 12-bit
  layout's prompts are now fused to 1792 tokens (were to 1023), then
  decoded ahead, the length that loses least across Qwen3-1.7B, 4B and 8B
  (Qwen3-8B's fused pass is 1.0-4.6% slower than decoded ahead at six
  lengths of nine to there, Qwen3-1.7B's 3.5-4.0% faster at 1793-2047).
  One forward pass (`gpu/e2e.py --prefill --merge`), 12-bit,
  Qwen3-4B-Instruct-2507 at 256 / 384 / 512 / 640 / 768 tokens: 28.3 /
  40.8 / 50.9 / 64.2 / 75.2 ms against bf16's 28.1 / 39.5 / 49.1 / 62.7 /
  73.0 (were 28.9 / 41.9 / 51.8 / 65.8 / 76.6); Qwen3-1.7B at 384 / 640 /
  1024 / 1536: 18.6 / 26.8 / 42.5 / 63.6 against 18.7 / 25.9 / 42.3 / 63.0
  (were 19.2 / 27.8 / 43.5 / 67.8). Generation to 64 sequences runs the
  same machine code as before; a step of 65 sequences or more multiplies
  by the prompt kernel (the same bits): at 128 sequences
  Qwen3-4B-Instruct-2507 makes 5069 tokens/s 12-bit (were 4977; bf16
  4702) and 4971 tiered (were 4950), Qwen3-1.7B 8845 and 8384 (were 8699
  and 8368; bf16 8452).

- The GPU kernels as a library of their own, for engines in C, C++, Rust
  or any language with a C FFI, with no Python or PyTorch: every release
  carries `glyd-gpu-TAG-linux-ARCH-cudaN.tar.gz` for x86_64 and aarch64,
  CUDA 12 and 13 (the library, its header, the example below,
  `gpu/LICENSE` and a README with the example's build line), each with its
  `.sha256`. `gpu/glyd_gpu.h` declares the C API: its 37 functions,
  `GLYD_GPU_API_VERSION` (2), the arrays of each packed layout, the
  workspace queries, the stream and the return codes. `glyd_gpu.cu`
  includes it, so nvcc holds each definition to its declaration (the
  library's build and the JIT's), and `bindings/python/test_gpu.py` holds
  the package's ctypes calls to it. `gpu/examples/unpack.c` reads a
  matrix of a model saved by `glyd.save_pretrained`, decodes it on the GPU
  with the library and checks it against the bf16 checkpoint: on an RTX
  4080 SUPER every one of Qwen3-0.6B's 112 packs (its 196 Linears, q, k,
  v and gate, up merged) decodes to the checkpoint's bits.

## v0.22.0 — 2026-09-28

- Prompts of 129-512 tokens on Hopper multiply straight from the packed
  weights (`mma_gemm_wg`: `GLYD_WG_MAX` is 512, was 128), where each
  matrix was decoded for cuBLAS first: tiles of 192 or 256 tokens, each
  weight decoded once a tile, a launch's tokens in chunks of a tile (two
  where O / 64 is even); past 512 tokens as before. On an H100 PCIe
  (`gpu/e2e.py --prefill --merge`), Qwen3-32B's prompts of 129 / 256 /
  384 / 512 tokens take 63.1 / 79.7 / 118.3 / 141.0 ms against bf16's
  58.6 / 70.3 / 89.3 / 113.1 (were 131.1 / 142.1 / 165.3 / 189.8),
  Qwen3-8B's of 384 / 512 tokens 33.2 / 39.2 against 25.9 / 32.6 (were
  41.6 / 47.7); Qwen3-8B's pass at 129-256 tokens is mostly the host's
  launches, 22.3-28.7 ms in four runs against bf16's 22.7-51.5 (was
  33.4-36.7). A layer's products take 1.12-1.97x cuBLAS's time at
  129-512 tokens (weights read from memory).
- Steps of 17-128 tokens on Hopper faster: the TMA kernel's products no
  longer wait on one another (ptxas had serialized every wgmma), a
  stage's exceptions are taken 32 at a time, tiles of 96 and 112 tokens
  as well as 128, and a unit's parts counted by one thread. On an H100
  PCIe (`gpu/e2e.py --merge --profile`), a step's GPU time at 1 / 8 / 32
  / 64 sequences is 11.13 / 12.32 / 13.48 / 14.64 ms for Qwen3-8B
  against bf16's 12.30 / 13.34 / 14.57 / 15.66 (were 11.24 / 12.45 /
  15.55 / 17.57), 34.65 / 37.34 / 41.24 / 44.41 for Qwen3-32B against
  42.44 / 44.40 / 47.10 / 49.56 (were 34.75 / 37.48 / 44.97 / 49.84); a
  layer's products (weights read from memory) take 0.85x / 0.86x / 0.96x
  / 1.05x cuBLAS's time at 32 / 64 / 96 / 128 tokens for Qwen3-8B (were
  0.92x / 1.00x / 1.20x / 1.20x), 0.78x / 0.81x / 0.95x / 1.08x for
  Qwen3-32B, whose layer at 17-128 tokens takes 2-6% more than an
  earlier build of these changes measured (the cause not found). Steps
  of 1-16 tokens (on other GPUs to 64) take the exceptions of a layer
  that has many 32 at a time (Qwen3-8B's gate, up and down in layers
  1-3: 4-5 a step): those layers 210 us at 8 tokens against 242-269 on
  an H100 PCIe, 2-3% faster on an RTX 4080 SUPER, the others as before;
  outputs bit for bit as before.
- The prebuilt library carries native Blackwell code (sm_100, sm_120)
  where nvcc has it (CUDA 12.8 on: `gpu/build_lib.sh` builds without it
  with an older nvcc). The Hopper kernel is sm_90a code alone: launched
  from a build without it (PTX compiled for an H100) it traps, where it
  returned its output untouched, and `GLYD_GPU_ARCH=sm_90` builds
  sm_90a. `glyd_gpu_mma12_gemm_wg` and its workspace query return
  cudaErrorInvalidValue for O under 64 (O = 0 returned success,
  launching nothing).
- Short prompts on GeForce Ada faster: a prompt's fused product is one C
  call, as a generation step's (it had cost 12 us of host time a call,
  twice F.linear's), and on GeForce Ada runs by stream-K
  (`mma_gemm_sk_kernel`: as many blocks as the GPU holds, each an equal
  share of the tiles' stages, a tile several share summed by the last of
  them in their order), where K was split and its parts summed by a
  second kernel (as still on other GPUs, until measured there). On an RTX 4080
  SUPER (`gpu/e2e.py --prefill --merge`), Qwen3-1.7B's prompts of 128 /
  256 / 512 tokens take 10.3 / 14.6 / 23.8 ms tiered and 10.2 / 14.5 /
  23.7 12-bit against bf16's 10.8 / 14.0 / 24.3 (0.21.0: 11.4 / 15.8 /
  25.4 and 11.4 / 15.4 / 24.7), Qwen3-4B-Instruct-2507's 128 / 256 / 512
  tokens 19.4 / 29.3 / 52.6 and 19.2 / 28.7 / 51.5 against 20.7 / 28.0 /
  49.1 (were 19.7 / 29.9 / 53.3 and 18.6 / 29.3 / 52.0); at 384 tokens
  Qwen3-1.7B's 20.1 / 19.2 against 18.7 (were 23.9 / 23.2), Qwen3-4B's
  44.4 / 41.7 against 39.5 (were 50.2 / 49.3); the time to the first
  token with them (Qwen3-1.7B at 128 tokens 11.6 / 11.8 ms against 13.0
  / 12.8); generation as before. In the 12-bit layout the decode ahead
  starts at 1024 tokens (was past 640) where the fused kernel takes the
  prompt: it is now the faster one to there (exact and unfused products
  past 640, as before). The C API's `glyd_gpu_mma_gemm_big` and
  `glyd_gpu_mma12_gemm_big` take a product's done counters, and
  `glyd_gpu_api_version` tells the C API's version (2; the package
  refuses a library of another).
- Products on two streams of one GPU at once no longer share done
  counters (a set a stream, as the workspace): a small product's outputs
  could come out wrong that way.
- Long prompts on GeForce Ada within 0.2-0.7% of bf16's time from 2048
  tokens, 1.5-2.9% at 1024 (were 5-10% behind): past 512 tokens (from
  1024 in the 12-bit layout) each matrix is decoded once, for cuBLAS, on
  a second stream beside the products before it, a few warps an SM
  beside cuBLAS's blocks, where the fused kernel decoded each weight
  again for every 256 tokens. On an RTX 4080 SUPER
  (`gpu/e2e.py --prefill --merge`), Qwen3-4B-Instruct-2507's prompts of
  1024 / 2048 / 4096 tokens take 97.5 / 199.5 / 447.5 ms tiered against
  bf16's 96.1 / 198.3 / 445.4 (were 102.8 / 211.5 / 476.5), Qwen3-1.7B's
  43.3 / 86.8 / 186.0 against 42.1 / 86.6 / 185.3 (were 45.9 / 91.5 /
  190.1), Qwen3-8B's 175.2 / 345.1 / 767.6 (were 185.6 / 370.8 / 811.3);
  the time to the first token with them; generation as before. With
  `exact=True` the same path, the logits bf16's bit for bit. On GeForce
  Ada the fused kernel runs blocks of 128 tokens where the last block of
  256 would be half empty or less, to 1024 tokens tiered and 4224 12-bit
  (300 tokens: 0.79-0.89x the time; past 1024 the tiered layout's cost
  1.3-6.9% more from 1600 tokens, the 12-bit's 0.8-4.5% less to 4224;
  elsewhere as before, until measured). The decode ahead is off until
  measured on an H100 and the L4, L40S and RTX 6000 Ada
  (`GLYD_AHEAD_MIN=513` takes it); on an A100 it is off, as measured
  (below).
- On an A100, batched steps of 65-128 tokens and prompts faster in the
  12-bit layout. A step of 65-128 tokens runs `mma_gemm_mid`'s A100
  kernel in one launch (units of two row blocks by 96 or 128 tokens), was
  `mma_gemm_big`, and is one C call; its consumers no longer hold the
  next stage's fragments through a unit's sums (the 64-token kernel had
  spilled). A prompt runs `mma_gemm_big` in blocks of 256 tokens by two
  row blocks with eight consumer warps (a weight decoded once for 256
  tokens, X's tile read once for 128 rows; `variant=3`), in blocks of 128
  where the last of 256 would be half empty or less, to 640 tokens; from
  769 tokens (`GLYD_DEC_MIN`) each matrix is decoded for cuBLAS. On an
  A100-SXM4-40GB (`gpu/e2e.py --merge --fused --profile 16`), Qwen3-8B's
  GPU time a step at 32 / 64 / 128 sequences is 18.54 / 21.05 / 28.17
  ms against bf16's 21.12 / 21.13 / 25.71 (was 19.14 / 21.69 / 29.45),
  Qwen3-14B's 28.70 / 31.81 / 42.41 against 32.89 / 36.01 / 41.15 (was
  29.24 / 32.95 / 49.23); Qwen3-8B generates 2993.6 tokens/s at 128
  sequences against bf16's 2997.4 (was 2747.0). Qwen3-8B's prompts of 128
  / 512 / 1024 / 2048 / 4096 tokens (`--prefill`) take 40.9 / 64.5 /
  112.0 / 195.7 / 368.2 ms against bf16's 41.0 / 48.4 / 90.1 / 174.3 /
  347.9 (were 44.9 / 68.3 / 120.7 / 237.1 / 489.2), Qwen3-14B's 512 /
  1024 / 2048 / 4096 tokens 111.7 / 198.6 / 347.5 / 653.7 against 82.7 /
  151.2 / 291.1 / 584.6 (were 113.1 / 207.3 / 416.6 / 855.4). The decode
  ahead was slower there than each matrix decoded before its product at
  every length measured (Qwen3-8B's 2048 tokens 214.8-274.3 ms at 2 to 4
  warps an SM, against 195.7), so it stays off on an A100.
- `glyd.save_pretrained` saves a mixture of experts: glyd-v1 holds each
  layer's experts as one pack (their matrices stacked, under the module
  holding them), and `glyd.json` the sha256 of each weight as the model
  holds it; such a checkpoint's format is glyd-v2, which glyd 0.21 refuses
  (a dense model's stays glyd-v1). `from_pretrained(path)` loads the packs
  as saved (the 12-bit layout packed again from them), `verify=True`
  checks every tensor, and `python -m glyd.gpu pack` and `verify` take
  one. On an RTX 4080 SUPER, granite-3.1-3b-a800m saves in 7.4 s to 4.66
  GB of safetensors (6.60 GB in bf16) and loads from them in 0.4 s (3.4 s
  verified; 3.9 s from its bf16 checkpoint), its logits bit for bit the
  model packed as it loaded.
- Every mixture-of-experts family of transformers 5.17 packs its experts:
  the 54 whose Experts modules transformers runs through an experts
  implementation (run by `glyd`), and those whose own code runs them,
  taken over where it multiplies: Llama 4, DBRX, Aria, JetMoE (its
  attention experts too), Step 3.7 and LongCat-Flash, each one op under
  `torch.compile` (but JetMoE, whose router calls `.tolist()`, as bf16's
  does not compile); Switch Transformers' and NLLB-MoE's experts are
  Linears, packed as such. Llama 4's experts stayed bf16, and transformers
  runs every expert on every token there; Glyd runs each token's chosen
  one alone: on an RTX 4080 SUPER a Scout MoE block at its real sizes
  (hidden 5120, 16 experts of 8192) takes 0.66 ms at one token against
  bf16's 6.17, 1.91 against 6.26 at 8 and 6.40 against 23.4 at 512, its
  experts 2.70 GB against 4.03, exact bit for bit.
  `bindings/python/test_gpu.py` builds each family as a tiny model on the
  GPU: its experts packed, its logits as near fp32's as bf16's, exact bit
  for bit, saved and loaded in both layouts. Along the way: a packed
  Linear's or embedding's `.weight` reads as a model's own code reads it
  (Llama 4's embedding device, Gemma 4's pad row: these models failed
  before), a Linear subclass with a forward of its own is left as it is
  (Llama 4's router), a weight a model keeps in fp32 is loaded as it is
  (HunYuan V4's output layer: it failed to load), and a model with packed
  experts let go of is freed at once (it held reference cycles).
- `best_layout()` takes a mixture of experts into account: the tiered
  layout on an A10 as on Ada, where its decode keeps up (on an A10 1-6%
  less GPU time a step than the 12-bit one, on an RTX 4080 SUPER 6-7%)
  and it is 10-11% smaller; the 12-bit one on an A100 and an H100.
- `gpu/sizes.py` counts granite's and Mixtral's checkpoint names for their
  experts (granite-3.1-3b-a800m: 3.22 B weights, was 0.20 B); `gpu/e2e.py`
  frees the Glyd model before bf16's profile (Qwen3-30B-A3B ran out of
  memory there on an H100).

## v0.21.0 — 2026-09-27

- `pip install "glyd[gpu]"`: the Linux wheels (x86_64 and aarch64,
  manylinux_2_28) carry the GPU kernels built for CUDA 12.8 and 13.0,
  `libglyd_gpu_cuda12.so` and `libglyd_gpu_cuda13.so`, the one for
  PyTorch's CUDA taken; no compiler and no checkout. The extra installs
  PyTorch 2.5+, transformers 5.17+, accelerate, safetensors and
  huggingface_hub ([bindings/python](bindings/python/README.md#on-the-gpu-a-models-weights-held-compressed-bit-for-bit)).
- `glyd.from_pretrained("Qwen/Qwen3-8B")`: transformers loads the
  checkpoint and every Linear's weight is packed on the GPU as it
  arrives (a quantizer registered as `glyd`), in `best_layout()`'s
  layout for the GPU, q, k, v and gate, up as one product each. On an
  RTX 4080 SUPER, Qwen3-0.6B's logits from the fused kernels are nearer
  the fp32 model's than bf16's are (mean |difference| 0.030 against
  0.041; at the first token bf16 rounds " with" and " in" to a tie that
  fp32 and Glyd both break toward " with"), Qwen3-1.7B's as near (0.034
  against 0.032).
- `glyd.from_pretrained` packs a mixture of experts. transformers 5.17
  keeps a layer's experts as 3-D parameters of an Experts module (OLMoE,
  granite MoE, Qwen3-MoE, Qwen3-Next, Gemma 4, GLM-4.5, Mixtral, gpt-oss
  ...), and they stayed bf16; now each is packed as one matrix of its
  experts as it arrives and run by `glyd`, an experts implementation
  registered with transformers: each token's choices sorted by expert on
  the GPU, a layer's experts in one grouped product for gate and up (the
  activation applied as it is written out) and one for down (the routing
  weights applied), no host sync; a long prompt's in tiles of 128 tokens an
  expert. On an RTX 4080 SUPER (`gpu/e2e.py --from-pretrained --baseline
  --prompts --merge --profile --prefill`), granite-3.1-3b-a800m holds 4.61
  GB against 6.60 GB in bf16 and generates 90.0 tokens/s at one sequence
  against bf16's 71.1 and 600.9 at eight different prompts against 195.6
  (a step's GPU time 5.0 ms against 6.8, and 9.5 against 39.4); OLMoE-1B-7B,
  measured before a step's product was one C call, holds 9.28 GB against
  13.84 and generates 135.6 against 99.1 and 607.9 against 142.1 (4.3 ms
  against 8.8, and 11.8 against 55.7). Prompts of 16 to 2048 tokens take
  less time than bf16's (2048 tokens: granite 91.2 ms against 100.8, OLMoE
  98.7 against 110.5). Compiled (`generate(...,
  cache_implementation="static")` or `torch.compile(model.forward,
  mode="reduce-overhead", fullgraph=True)`), a packed Experts module is one
  op of the graph (`glyd::experts`), no graph break, its kernels in the
  CUDA graph: granite generates 225.1 tokens/s at one sequence against
  bf16's 170.0 compiled the same way, and 909.9 at eight prompts against
  175.2 (`e2e.py --compile`). With `exact=True` the experts the tokens are
  routed to are decoded and the path bf16 takes runs on them, grouped_mm
  and, while `generate()` decodes, batched_mm (transformers switches
  bf16's so): the logits are bf16's bit for bit at every step (both
  models, one sequence and eight different prompts, 16 steps). torch's
  grouped_mm runs on the GPU alone only on compute capability 9.x and
  10.x (10.x from torch 2.9, as its source reads); on any other GPU it
  copies to the host, which a CUDA graph's capture refuses (on an RTX 4080
  SUPER bf16's own `torch.compile(model.forward, ...)` of a mixture of
  experts stops there), so there exact runs batched_mm while a graph
  captures. glyd-v1 (`save_pretrained`) holds no packed experts yet, and
  a model with them can't be copied or pickled (`copy.deepcopy`,
  `torch.save`): load it again.
- `gpu/e2e.py` takes every timing before its first profile (`--profile`):
  a profiler session leaves each CUDA launch after it slower for the rest
  of the process, and Glyd, timed after bf16's profile, lost some 20% of
  its tokens/s (granite-3.1-3b-a800m at one sequence on an RTX 4080 SUPER:
  68.6 against 85.4 without it); bf16's profile now runs last, on the
  model loaded again. `--prompts` generates for different prompts, not
  copies of one; `--from-pretrained` times the model `glyd.from_pretrained`
  loads; `--prefill` draws its tokens within the model's vocabulary.
- `exact=True`: every product decodes its matrix and multiplies by
  `F.linear` as `nn.Linear` does, so the logits are bf16's bit for bit
  (Qwen3-0.6B and 1.7B, 32 of 32 tokens as bf16's; `e2e.py --exact` the
  same for the scripts).
- `glyd.save_pretrained(model, path)` writes the glyd-v1 format (the
  packs as safetensors, `glyd.json` with every tensor's sha256), which
  `from_pretrained(path)` loads packed (Qwen3-1.7B in 2.52 GB of
  safetensors); `verify=True` decodes every tensor and checks it.
  `glyd.fit("Qwen/Qwen3-32B", gpu="48GB")` answers from the config and
  the checkpoint's metadata, the measured sizes where there are some;
  `python -m glyd.gpu fit|pack|verify`.
- The kernels behind a C API (`gpu/build_lib.sh`), called through ctypes:
  Qwen3-8B generates 48.4 tokens/s at one sequence through the library as
  through a local build of the extension (363.6 and 363.4 at eight); on
  models under 2B, where a step is mostly Python, 3-4% fewer.
- Compiled: `model.generate(..., cache_implementation="static")`
  (transformers' compiled forward) and `torch.compile(model.forward,
  mode="reduce-overhead", fullgraph=True)` take every GLinear and
  GEmbedding as one op of the graph (`glyd::linear`, `glyd::embedding`),
  with no graph break, and the CUDA graph captures Glyd's kernels: on an
  RTX 4080 SUPER Qwen3-1.7B generates 187.8 tokens/s at one sequence
  against bf16's 151.9 compiled the same way (1282 against 1020 at eight),
  Qwen3-4B-Instruct-2507 95.2 against 73.9, Qwen3-8B 55.6. Eager, a step's
  product is one C call, what does not change between calls made once:
  4.7 us of host time against `F.linear`'s 5.4 (8.7 before), a Qwen3-1.7B
  layer 295 us against bf16's 322 (344 before), so eager `generate()` is
  faster than bf16's where the host is the bottleneck too (Qwen3-1.7B 98.3
  tokens/s against 88.9, 86.8 before; Qwen3-4B-Instruct-2507 75.7 against
  61.5). On a server's CPU too (AWS g5.2xlarge, an A10G, users' path:
  the wheel installed, `generate()` a fresh process each): Qwen3-8B 26.1
  tokens/s at one sequence against bf16's 23.0 and 201.1 against 182.3
  at eight, compiled 37.1 against 27.1 and 256.5 against 197.5;
  Qwen3-4B-Instruct-2507 compiled 58.3 against 45.8; granite-3.1-3b-a800m
  (a mixture of experts) 31.4 against 25.3.
- On an A100, steps of 17-64 tokens (batched generation) through a
  kernel of its own behind `mma_gemm_mid` (compute capability 8.0 only):
  producer warps copy each stage's compressed step and its exceptions by
  cp.async several stages ahead into a ring in shared memory, consumer
  warps decode and multiply by mma.sync, stream-K with a fixed-order sum.
  Qwen3-8B's layer at 17-64 tokens takes 0.87-0.98x cuBLAS's GPU time
  (was 0.96-1.32x), Qwen3-32B's 0.79-0.87x; a Qwen3-8B step at 32
  sequences 19.79 ms against bf16's 20.82 (was 21.27), at 64 21.80
  against 21.67 (was 25.96). Every other GPU keeps its kernels bit for
  bit ([benchmarks/gpu/a100-mid-2026-09-27](benchmarks/gpu/a100-mid-2026-09-27)).
- `glyd.save_pretrained` copies the source's tokenizer files once and
  their content only: the Hub's cache keeps them read-only, and
  `tokenizer.model`, which two patterns match, failed on its second copy
  (Mistral 7B). `gpu/check_models.py`: each model against bf16 and the
  fp32 model over eight prompts, exact mode and save/verify; nine dense
  models from Qwen3-0.6B to Llama 3.1 8B pass on an RTX 4080 SUPER.
- The PyPI page leads with the models on the GPU (`pip install
  "glyd[gpu]"`, `from_pretrained`), with the project's links, keywords
  and classifiers.
- Blackwell, measured on an RTX PRO 6000 Blackwell Server Edition
  (compute capability 12.0, AWS g7e) with the wheel's library as it
  ships (its compute_80 PTX, compiled by the driver): `gpu/check_api.py`
  passes (dense, compiled, mixture of experts); Qwen3-8B's logits against
  the fp32 model 0.0509 (bf16's 0.0517), `exact=True` bit for bit;
  `generate()` 53.0 tokens/s at one sequence against bf16's 50.0,
  compiled 93.0 against 75.9; a step's GPU time 11.22 ms against 13.42 at
  one sequence, 16.34 against 16.35 at 64. Prompts are slower there (512
  tokens 41.2 ms against 33.3, 2048 tokens 130.5 against 103.2). A
  library of its own for compute 12.0 needs the wgmma kernels kept to
  Hopper's code first (ptxas refuses wgmma for sm_120).
- `mma12_gemm_wg` on compute capability 9.0 alone (its code is sm_90a);
  later GPUs take the kernels they would without it.
- Homebrew installs the release's binaries on Apple silicon and Linux
  (x86_64, arm64) in seconds; an Intel Mac and `--HEAD` build from
  source. `scripts/bump_formula.py` points the formula at a release.

## v0.20.0 — 2026-09-27

- The Python package on PyPI: `pip install glyd` (wheels for Linux x86_64
  and aarch64 and macOS arm64), published from the release workflow by
  trusted publishing.
- Nine more open models measured, nineteen in all: GLM-4.5-Air, Llama 4
  Scout, Qwen3-Next 80B-A3B, Muse Glimmer 30B, Qwen3.8 27B, Gemma 4
  26B-A4B, Gemma 3 12B, Qwen3 4B 2507 and Llama 3.2 3B, every projection's
  matrix 32.2-33.0% smaller in the tiered layout, bit for bit. `sizes.py`
  counts a layer's experts kept as one tensor (Gemma 4, Llama 4) a matrix
  an expert, and every projection (the linear-attention inputs of
  Qwen3-Next and Qwen3.8); `e2e.py` runs Gemma 3 and 4 (the decoder under
  `language_model`, the scaled embedding), Qwen3.5-style linear-attention
  layers and checkpoints that load only with their vision tower. Qwen3.8
  27B, the highest-scoring open model that fits one GPU, uses 41,071 MiB
  with Glyd against bf16's 51,771 (under a 48 GB card's 49,140),
  perplexity 15.1941 against 15.1946, MMLU answers as bf16's on 99.67%
  of 300 ([gpu/](gpu/README.md#popular-models)).
- On Hopper, steps of 17-128 tokens from the 12-bit layout by the copy
  engine and wgmma (`mma_gemm_wg`, [gpu/](gpu/README.md#many-tokens-a-step-on-an-h100-the-copy-engine-and-wgmma)):
  TMA bulk copies of the compressed steps and a tensor map for X's tiles
  into a ring in shared memory, the weights decoded straight into
  wgmma's registers, stream-K with a fixed-order sum (the same result
  every run). On an H100 SXM, Qwen3-32B's MLP matrices at 32 and 64
  tokens take 83 and 92 us against cuBLAS's 90-96 (the `mma_gemm` kernel
  105-140); at 128 tokens 121-124 against 98-99 (decoded for cuBLAS
  328-335). Small matrices still cost more than cuBLAS's, so end to end
  32 and 64 sequences take 37.71 and 43.04 ms of GPU time a step against
  bf16's 32.44 and 35.50 (Qwen3-32B, 25% less memory); one sequence
  26.14 against 28.23. `pack_mma12` pads the exception list to a
  multiple of four entries; `e2e.py --profile` measures every `--batch`
  size.
- The TMA kernel's stages decode with their eight loads in flight at once
  and one run of exceptions: with q, k, v and gate, up as one product
  each (`e2e.py --merge`, bf16 alike), Qwen3-32B on an H100 takes 24.50 /
  26.75 / 33.46 / 37.35 ms of GPU time a token at 1 / 8 / 32 / 64
  sequences against bf16's 27.90 / 29.78 / 31.44 / 33.52.
- The layouts measured on an RTX 4080 SUPER, an A10, an A100 and an H100
  ([gpu/](gpu/README.md#which-layout-on-which-gpu)); `best_layout()` and
  `e2e.py --format auto` take the faster for the GPU (the tiered layout on
  Ada and wherever only it fits). `mma_gemm_mid` for 17-64 tokens on GDDR
  Ampere and Ada.
- Side by side with DFloat11 and ZipServ ([README](README.md#related-work)).

## v0.19.0 — 2026-09-26

- A second GPU layout, `mma12` ([gpu/](gpu/README.md#two-layouts-the-most-memory-or-the-lightest-decode)):
  a weight's exponent a 4-bit code into the tensor's 15 commonest, a
  step's exceptions in a list; four weights are three byte permutes, so
  the decode keeps up with an H100's HBM3 where the tiered one is bound by
  arithmetic. 12.04 bits a weight, 25% under bf16, bit for bit. On an H100
  SXM, **Qwen3-32B in 49.23 GB at 26.39 ms of GPU time a token against
  bf16's 65.52 GB and 28.22 ms** (the tiered layout: 44.45 GB, 40.58 ms);
  Qwen2.5-7B 7.35 ms against 7.47; the products at 1-16 tokens 1.1-1.2x
  faster than cuBLAS's (Qwen3-32B's MLP at one token: 75-79 us against
  86-90); MMLU 78.1% (bf16 78.3%). On an RTX 4080 SUPER, where memory is
  the limit, the tiered layout stays the faster at 1-32 sequences and
  `mma12` leads at 64 (2,455.9 tokens/s against 2,244.7; bf16 2,160.0).
  `pack_mma12`; `mma_gemm`, `mma_gemm_big` and `mma_unpack` take either;
  `e2e.py --format mma12`; the kernels are written once over both layouts.

## v0.18.0 — 2026-09-26

- The KV cache compressed in GPU memory ([gpu/kv.py](gpu/kv.py)), bit
  for bit: `GlydKVCache(config)` for a Hugging Face model keeps each
  layer's newest tokens as they are and packs every full page of 64 in
  the mma layout's tiered code (keys by token, values transposed); with
  `fused=True` a step of one new token a sequence runs `attn_decode`,
  attention straight from the packed pages (keys and values decoded in
  registers, both products on the tensor cores, an online softmax, a
  fixed order: the same result every run). Qwen2.5-7B-Instruct on an RTX
  4080 SUPER: **the cache 31% smaller** (16K tokens: 651 MB for 947),
  peak memory below the plain cache's, a step as fast (21.7 ms against
  21.6 at 16K; 19.2 against 18.3 at 1K); decoded, the cache is the plain
  one's bit for bit (the same tokens); through `attn_decode`, 256 tokens
  fed one at a time give perplexity 2.4778 against 2.4809 (16K).
- Larger models on rented GPUs, bf16 and Glyd in the same runs, MMLU on
  1,000 questions ([gpu/README.md](gpu/README.md#larger-models)):
  **Qwen3-32B on one 48 GB RTX A6000** (44.45 GB; bf16 65.52 GB across
  two) at 1.24-1.28x bf16's tokens/s for 1-8 sequences, MMLU 78.0%
  (bf16 78.5%); **Qwen2.5-72B on three** (97.80 GB; bf16 145.41 GB
  across four) at 1.40-1.42x, MMLU 81.8% (81.9%); on an H100 SXM
  Qwen3-32B in 44.45 GB with MMLU 78.2% as bf16's, its products slower
  than cuBLAS's (40.6 ms of GPU time a token against 28.2).
- The README leads with the AI work: `nvidia-smi` from the runs
  (`e2e.py --smi`, drawn by `scripts/term_svg.py`), the measured limits
  (about 34% off bf16 weights or KV cache for any lossless code, 18% off
  FP8, 7% off NVFP4).
- `e2e.py --mmlu N`, `--kv LENGTHS`, `--smi PREFIX`; `pack_mma` takes
  given tiers and a chunk size, `mma_cat` appends packs;
  `gpu_lambda.sh` takes models as `[org/]name[:B[:G]]` (side by side on
  B GPUs, then Glyd alone on G) and follows the run as it goes.

## v0.17.0 — 2026-09-26

- Model weights on the GPU ([gpu/](gpu/README.md)): the `mma` layout's
  exponents in tiers of 2-bit digits — the tensor's 3 commonest
  exponents, digit 3 going on to the next 3, then the next 3, then the
  exponent itself: **10.80 bits a weight over Qwen2.5-7B's matrices,
  32.5% under bf16** (was 11.25), every tensor bit for bit. A step's
  escapes are decoded by the whole warp (each lane 16 of its tier-2
  digits, placed by warp scans, through shared memory), so no lane
  waits on another's. Qwen2.5-7B-Instruct on an RTX 4080 SUPER: 10.61
  GB where bf16 takes 15.25 (was 11.05); **1.25-1.32x bf16's tokens/s
  at 1 to 32 sequences** (one: 55.7, bf16 43.4; 32: 1,518.9, bf16
  1,153.7), 1.13x at 48, 1.04x at 64; prompts of 16 to 128 tokens
  19-29 ms (bf16 24-29), 256 to 4096 within 5-10%. Perplexity as
  before (Wikipedia, 64-token windows 17.0052 vs bf16's 17.0015,
  512-token 7.5660 vs 7.5677).
- The GPU extension builds for the GPU it runs on (`sm_90a` on Hopper);
  on Hopper `e2e.py` multiplies prompts past 64 tokens by decoding then
  cuBLAS; `e2e.py --profile N` splits GPU time by kernel.
  `scripts/gpu_lambda.sh` runs `gemm.py` and `e2e.py` on one Lambda
  Cloud GPU instance launched for the run (a time cap, terminated and
  checked on exit).
- gpu/ is under the Business Source License 1.1 from this release (the
  store's terms); the codec stays BSD-3-Clause OR GPL-2.0.

## v0.16.0 — 2026-09-25

- Model weights on the GPU, several tokens at once
  ([gpu/](gpu/README.md)): the `mma` layout (`pack_mma`), the fast
  format's 3-bit codes into the tensor's densest run of 7 exponents,
  each step of 1024 weights one run in the order the tensor cores take
  their operand. `mma_gemm` (1 to 64 tokens) decodes it in registers
  straight into `mma.sync` fragments; `mma_gemm_big` (prompts) is a
  tiled GEMM whose producer warps decode the weights into shared memory
  while its consumer warps multiply. Qwen2.5-7B-Instruct on an RTX 4080
  SUPER, in 11.05 GB where bf16 takes 15.25: **1.23-1.33x bf16's
  tokens/s at 1 to 48 sequences at once** (one: 55.2 tokens/s, bf16
  43.3; 32: 1,528.6, bf16 1,149.0; the batched fast-format product it
  replaces ran 0.72-0.90x); prompts of up to 128 tokens faster than
  bf16 (128: 27 ms, bf16 29; was 60), 256 to 4096 within 5-9% (4096:
  696 ms, bf16 645). The same result every run (sums in a fixed order);
  perplexity as bf16's (Wikipedia, 64-token windows 17.0052 vs 17.0015,
  512-token 7.5660 vs 7.5677).
- `gpu/e2e.py`: `--format mma`, `--batch`, `--gpus N` (layers spread
  over GPUs by their bytes; bf16 by accelerate's device map), `--ppl`;
  every kernel runs on its tensors' device and PyTorch's current stream.
  `gpu/setup_env.sh` builds the environment without root (nvcc pinned
  to PyTorch's CUDA); `scripts/gpu_aws.sh` runs Qwen2.5-32B and 72B on
  4x L40S.

## v0.15.0 — 2026-09-25

- Model weights on the GPU ([gpu/](gpu/README.md), Python and CUDA beside
  the library): a bf16 model's weights held compressed in GPU memory and
  decoded there bit for bit, the sign-and-mantissa byte as it is and the
  exponent coded, in two formats: dense (a prefix code read by counting
  leading zeros, as short as Huffman's; 10.9 bits a weight) and fast
  (3-bit codes into the 7 most common exponents, an escape to the rest;
  11.25). Generation multiplies straight from the packed weights, never
  writing bf16 out. Qwen2.5-7B-Instruct on an RTX 4080 SUPER (16 GB):
  **55.1 tokens/s in 11.05 GB** (fast) and **52.0 in 10.60 GB** (dense),
  against bf16's 43.2 in 15.25 GB; the fast format's 128 tokens as
  bf16's. Prompts of up to 64 tokens multiply on the tensor cores from
  the fast format (35-40 ms, bf16 24-27); longer prompts decode each
  matrix and use PyTorch's matmul (2048 tokens: 337 ms, bf16 301).
- PyTorch checkpoints (`torch.save`): the zip's tensor storages go in as
  byte planes, their element widths read from the checkpoint's pickle
  (a reader for the opcodes `torch.save` writes, no dependency); against
  a base checkpoint (`--base`, the store) each storage the base holds
  under the same name and size goes in as XOR that storage, where that
  is the cheaper (weights move little between checkpoints, Adam's first
  moment as much as it holds). Qwen2.5-0.5B fine-tuned with AdamW
  (fp32 weights and both moments, 5.93 GB a checkpoint): 83.2% of its
  size alone, 77.3% against the checkpoint 50 steps before; zstd -19
  92.2% (its `--patch-from` stops at 2 GB). The store finds a
  checkpoint's predecessor by its storages' names and sizes.

## v0.14.9 — 2026-09-25

- The store: a family's first object sits at the depth cap only until
  its family first needs it; then it is lifted, stored again against a
  shallower base, in place of every new family starting shallow
  (v0.14.8's rule). The terabyte gate, same corpus and instance
  ([report](docs/benchmarks/store-gate-2026-09-24.md)): **43.61 GB,
  3.52× fewer bytes than zstd -3** (44.35 GB, 3.46× in v0.14.8), 27.2×
  against raw; the English Wikipedia tables 8.67 → 7.94 GB, the kernels
  as in v0.14.8 or smaller. Put 374 MB/s, read-back 486 MB/s (zstd -3's
  346), restore 559 MB/s; all 1,192 objects byte-exact, both reads.
- The store finds a model checkpoint's predecessor: a safetensors
  object's fingerprints are its tensors' names, widths and sizes (it
  shares no bytes with the checkpoint before it). Nine Pythia-410M
  checkpoints (14.6 GB): 5.88 GB in the store, where zstd -19 stores
  7.28 GB and zstd -3 8.63 GB; the last two as deltas of 501 MB, every
  object verified ([report](docs/benchmarks/weights-2026-09-25.md)).
- Reads: the decoder no longer spins while a unit waits for the ones
  ahead of it; the unit that completes the run writes it out. Decompress
  CPU at the plain levels fell by up to 65% on 32 threads (logs `--max`
  3.08 → 1.09 s), wall time unchanged.
- The same bytes on every machine: an input is cut into sixteen units
  whatever the core count (it followed the thread count before, so a
  32-thread machine cut twice as finely as a 16-core one). GitHub
  events at `--ultra` 5.4% smaller on 32 threads; unchanged on 16.
- Noise-level bytes of a small alphabet (a model weight's exponent or
  mantissa plane) are coded as literals alone where the parse's short
  matches would cost more: `--max` on Pythia's exponent plane 171.7 →
  140.1 MB (zstd -19: 143.8), on Qwen2.5's 208.3 → 169.8 MB (171.6).
  Text, logs, SQL and kernel tars come out byte-identical in size. The
  store's dense level (stripes on all cores) too: a stored checkpoint
  752 → 697 MB.
- Model weights: a safetensors file is opened, each tensor of 2-, 4- or
  8-byte elements as byte planes (exponents together, mantissas
  together), the header kept; closed byte for byte. Pythia-410M (fp32,
  1,621 MB) at `-9`: 701 MB in 3.5 s, against zstd -19's 809 MB in 92 s
  and zstd -3's 959 MB; Qwen2.5-0.5B (bf16, 988 MB): 663 MB against
  750 and 769. Reads at 1.1 GB/s.
- Checkpoints against checkpoints: in base mode, a safetensors file
  against one holds each tensor the base has under the same name and
  size as that tensor XOR the base's, in planes. Pythia-410M step 71000
  against step 70000: 612 MB in 4.7 s, where zstd -19 `--patch-from`
  stores 805 MB (the raw delta finds nothing to match); step 143000
  against 142000: 501 MB. Qwen2.5-0.5B-Instruct against its base model:
  558 MB.
- The command line asks for `--base` for a container opened against a
  base (a gzip against a gzip, weights against weights): such a file
  started with the container envelope, so decoding it without that
  check failed.
- The savings calculator re-measured at this code (LZ4, gzip, zstd -3
  and -19, four Glyd levels, decompress CPU per row; the bucket row
  from the v0.14.8 gate): [report](docs/benchmarks/savings-2026-09-24.md).

## v0.14.8 — 2026-09-24

- **At a terabyte: 3.46× fewer bytes than zstd -3** (3.32× in
  v0.14.7). The gate's 1,192 objects, 1.18 TB, put through the store
  into S3 from one 16-vCPU instance next to the bucket
  ([report](docs/benchmarks/store-gate-2026-09-24.md)): 44.35 GB
  stored against zstd -3's 153.5 GB, 26.7× against raw; put at 386
  MB/s (372 in v0.14.7, zstd -3's own put 535); every object read back
  by its own process at 464 MB/s (zstd -3's read-back 348), all 1,192
  byte-exact; the whole bucket restored by one process at
  571 MB/s (603 in v0.14.7), all 1,192 byte-exact. Kernel releases 5.15 at 286×,
  6.1 at 406× (155× in v0.14.7), 6.6 at 377× against raw; hourly
  GitHub events 14.1×. One cost in the release's own rule (below):
  the English Wikipedia tables 19.1× against raw, where v0.14.7 kept
  them at 20.9×.
- **Record mode writes 1.6–1.9× faster, the same bytes.** On the
  Ryzen box, output byte-identical to v0.14.7: the NASA access log on
  one core 151 -> 265 MB/s, on all cores 749 -> 1,186 MB/s; a
  Wikipedia table dump 126 -> 198 and 407 -> 790 MB/s. A dictionary
  column's values are hashed once, and its recency list is kept in
  place, searched eight entries at a time and not at all for a value
  it cannot hold; a time value on the last exact value's date is not
  printed back to be checked; field ranges are 32-bit (a log unit's
  ranges had outweighed its text); a delimited line is split in one
  pass; a SQL dump's rows no longer allocate, and its text is crossed
  a word at a time to the next special byte.
- **Parquet files with snappy or zstd pages are opened**
  (`src/parquet.rs`, `src/resnappy.rs`, `src/rezstd/`): the footer's
  column chunks and page headers are read (thrift's compact protocol,
  no dependency), and every page is written back byte for byte by a
  port of the compressor that wrote it, so the page's raw bytes are
  compressed instead of its LZ tokens: google/snappy 1.2 level 1
  (builds differ in their hash, a multiply or the CRC32C instruction,
  and their table, 2^14 entries up to 1.1.10, 2^15 since 1.2.0), and
  zstd 1.5.2 through 1.5.7 at levels 1 and 3 (the fast and
  double-fast finders and their variants past the window's wrap,
  Huffman literals with the previous block's table, FSE sequence
  tables, the capacity rules, the checksum; 1.5.7's pre-block-splitter
  and its two double-fast rules, 1.5.2's three fast-finder rules;
  1.5.4 and 1.5.6 write what 1.5.5 does) as its library's one-shot
  call writes them and as its command line does: the
  `--single-thread` stream of 128 KB chunks, and the default of 2 MB
  jobs from fresh contexts seeded with the 64 KB before each, so a
  file `zstd` wrote opens too. Checked against zstd's own output on
  the fixtures, on a sweep of 3,500 inputs under every version, level
  and writer, and on 200 MB logs and dumps. The opener finds the
  build that wrote a page and keeps a page no build made (zstd 1.4
  and older, other levels, gzip pages). A container under a frame
  opens in turn: the snappy taxi file inside a `zstd -3` frame,
  52.3 MB, comes to 34.8 MB. A whole zstd frame (a `.zst` object) opens the
  same way, a container under it opened in turn: the NASA access log
  as zstd 1.5.5 wrote it at level 1, 22.3 MB, comes to 8.0 MB at
  `--max` (its records modeled), 1.4 s to write and 0.6 s to read
  back, byte-exact.
  A page's plain values are then modeled so the LZ and entropy stages
  see their structure: fixed-width values as byte planes, integers in
  their unit (microseconds that are whole seconds divided down) and
  as deltas, doubles that are decimals as scaled integers, byte
  arrays as lengths then bytes; dictionary-index pages have their
  runs decoded, written again by a port of Arrow's run-length encoder
  and compared, and the indices laid out as planes of the bytes that
  hold them. Each page takes the cheapest of its candidates or stays
  as it is, judged by the max level's own output, and the level
  blocks stay ahead. A NYC taxi month written by pyarrow 21 with
  snappy, 61.7 MB: 174 of 174 pages reproduced; `--max` 34.8 MB in
  0.5 s on ten cores (1.8 s on one), read back in 94 ms (zstd -3 on
  the file 52.3 MB, zstd -19 49.8; the same table's zstd-page file
  50.3; record mode on the table as CSV 37.0), `--ultra` 32.3 MB,
  every decode byte-exact; the same table's zstd-page file, 50.3 MB
  (pyarrow 14, zstd level 1): `--max` 34.8 MB in 0.5 s, read back in
  0.14 s. A month of for-hire trips, 519 MB with snappy, 1,291 pages,
  more of them ids: `--max` 376.5 MB in 3.4 s (zstd -3 on the file
  473.2), read back in 1.0 s; with zstd pages, 472.8 MB: 376.5 MB in
  2.7 s. Pages compressed with gzip, lz4 and brotli are left as they
  are.
  Files polars and DuckDB write open fully too: polars' snappy pages
  are the Rust `snap` crate's (the multiply's older hash, shifted by
  the table's size, which differs on blocks under 8 KB), and each
  writer's run-length encoder for dictionary indices is ported beside
  Arrow's (polars: repeats of more than eight, literal runs of up to
  8192 values packed in blocks of 32; DuckDB: repeats of four or more,
  bit-packed blocks of 256 written whole), the one that writes a
  page's runs again named in its recipe. A bit-packed run's padding
  (a block's earlier values) is taken as the writers leave it. The
  same taxi month at `--max`: polars' snappy file, 86.9 MB, 51.4 ->
  48.1 MB, its zstd file, 57.8 MB, 50.8 -> 48.1 MB; DuckDB's snappy
  file, 61.1 MB, 36.6 -> 35.0 MB, its zstd file, 45.7 MB, 36.6 ->
  35.0 MB; every decode byte-exact.
- **A new family starts shallow.** An object that starts a family (a
  new major release) takes an ancestor at depth 1 or the chain's root
  as its base, so its versions come back to it at the depth cap; one
  that had landed at the cap itself sent them to the chain's root.
  At the gate, 6.1.1 sat at depth 4 on a 5.15 release and every fifth
  6.1 release was a 26 MB delta of 5.15.1. On the Ryzen box, 5.15.1-100
  then the 150 releases of 6.1 in the gate's order: 6.1 1,316 -> 502
  MB, all 250 releases 1,714 -> 899 MB. At the gate the rule also
  moved the monthly Wikipedia page tables, each month a family of its
  own, onto bases two months back and the September one to alone:
  0.73 GB more there, most of the kernels' 0.81 GB gain. The next
  release lifts only a family's first object that sits at the cap,
  when its family first needs it (measured on the box: the kernels
  as here, the Wikipedia tables as in v0.14.7).
- **The store keeps a version's base among its own kind.** An object
  whose base holds under 98% of its fingerprints starts a family (a
  new kernel major holds 0.85–0.94 of the old one's releases; point
  releases hold 0.99–1.00 of the last, a 16-day Ubuntu image 0.97,
  monthly Wikipedia tables 0.69–0.99), and past the depth cap a version's base is its
  family's first object, not the chain's root. At the terabyte gate
  every kernel release sat in one chain rooted at 5.15.1, and every
  fifth 6.1 and 6.6 release was a delta of 5.15.1 at 42 MB against
  2–4 MB within its series: 1.55 of the 6.6 series' 1.84 GB. Measured
  on the Ryzen 9 box, 5.15.1 then the 150 releases of 6.6 in the
  gate's order: 2,024 MB stored before, **643 MB now**, one 42 MB
  delta (6.6.1 itself against 5.15.1). The index line carries the
  family (an eighth field; older lines read as before, their family
  the chain's root). A star-shaped chain tree was tried and dropped:
  6% fewer bytes on kernels, 30% more on monthly tables, where a base
  two months back costs half again the neighbour's.
- **Whether a delta pays is judged on four 8 MB windows spread over
  the object**, each against its own base region, at the same 80% bar
  the whole must meet (the head's 32 MB at a 50% bar before). Across
  the English Wikipedia `page` dump the ratio of delta to alone runs
  56–110% by window, 68% whole; the head's verdict had stored the
  2026-09 dump alone, 1,776 MB where its delta against 2026-08 is
  1,205 MB (v0.12.0 had that delta; v0.13.0's sample lost it). An
  object holding a fingerprint several times now counts once among
  its holders. The four windows run on threads of their own: an hour
  of GitHub events put in 0.50 s on the box against 0.79 s with them
  one after another.
- The gate's read-back compares each object with the corpus file of
  its name, the index line's last field (it read the seventh, which
  the family field made the family's id, and counted every object
  failed); each gate run syncs into a directory of its own (two runs at
  once shared one, and the first to finish stopped the other's wait).

## v0.14.7 — 2026-09-24

- **The store's put is 1.6–4.0× faster, its get 1.1–1.5×.** On a
  Ryzen 9 7950X3D (16 cores), best of three, every object read back
  byte-exact: Linux 6.10.1 as a version of 6.10 (1.5 GB) put at
  1,224 MB/s against 308 before, alone at 2,545 against 668, read
  back at 1,590 against 1,051; an Ubuntu 24.04 cloud root filesystem
  (1.1 GB, gzip inside) as a version 215 against 127, read 287
  against 258; a Wikipedia table a month on (108 MB) as a version 383
  against 233. What changed: `put_file` maps the file and keeps the
  mapping as the cached copy (`put_vec` takes the bytes over; no
  second copy of the object in memory); the container path's deflate
  emulation returns at once when nothing in the object opened; an
  opened base is decoded once and cached; the base region a unit
  searches is cut to what its fingerprints reach (97% of the hits
  kept, never under the unit and 16 MB); each thread keeps one
  region-and-unit buffer; a delta under a thirty-second of the object
  is taken without also compressing the object alone; `get_to` writes
  into the caller's buffer. The Ubuntu root's version is bounded by
  the deflate emulator, which runs at zlib's own search speed per
  thread. At a terabyte, one im4gn.4xlarge next to S3
  ([report](docs/benchmarks/store-gate-2026-09-24.md)): put at 372
  MB/s (243 in the 2026-09-22 run), every object read back by its own
  process at 461 MB/s (166 before; zstd -3's own read-back on that
  instance 341), the bucket restored by one process at 603 MB/s, all
  1,192 objects byte-exact both times; 46.3 GB stored against zstd
  -3's 153.5 GB, 3.32× fewer bytes (3.10× then: the hourly events
  store 9% smaller; one English Wikipedia table of fifteen went alone
  that was a delta before, 0.6 GB, to be looked at).
- **CLI reads: the decoded batch lives on huge pages.** The output
  batch is an anonymous 2 MB-aligned mapping with `MADV_HUGEPAGE`,
  the input is populated on a thread while the first units decode,
  and `-d -s` streams on one thread. One core against `zstd -d -T1`:
  Ryzen 9 7950X3D 1.00–1.30× its speed (0.65–0.81× before), Graviton3
  1.15–1.32× (1.03×), Sapphire Rapids 0.79–0.97× (0.80×). All cores:
  3.9–13.3 GB/s on the Ryzen, 3.6–10.0 GB/s on eight Graviton3 cores,
  1.9–5.1 GB/s on eight Sapphire Rapids cores.
- **CRC-32C in three lanes**: the block checksum runs three CRC
  streams over 1 KB lanes and joins them by table, so the CRC
  instruction's latency overlaps; it was a tenth of a one-core read on
  Sapphire Rapids. Bytes unchanged: the checksum's value is the same.
- **A corrupted unit fails a stream's parallel decode instead of
  hanging it**: the units after a failed one waited for their turn
  for ever; the first error now stops the rest
  (`tests/fuzz_safety.rs`).
- Library: `Store::put_vec`, `Store::put_file`, `Store::get_to`. CI
  keeps the corpus between runs and fetches enwik8 from a second host
  when the first answers with a page.

## v0.14.6 — 2026-09-24

- **Blocks are cut where the bytes' statistics change** (`src/split.rs`,
  the idea of zstd 1.5.7's pre-splitter): before each block's parse,
  sixteen bytes of every 256 in the 256 KB ahead are counted per 16 KB
  segment, and the block ends at the segment boundary where the two
  parts coded on their own statistics beat the whole by more than a few
  blocks' overhead, each part charged for describing its table; never
  under 32 KB, and the search rests after 32 windows without a cut. On
  binaries with sections of different content it pays: Silesia mozilla
  0.9% smaller (now 1.0% under zstd 1.5.5, 0.2% over zstd 1.5.7) for
  5% more write time on that file; JSON events, the NASA log, a table
  dump and enwik8 are unchanged in bytes and time. Every stream decodes
  as before (blocks were always any length up to 256 KB).

## v0.14.5 — 2026-09-24

- **`--max`'s parse is zstd -3's double-fast, no lazy step, with one
  of zstd's rules made stricter.** The lazy compare one byte on cost
  4–8% of the write time for 0.1% (a table dump) to 4% (a log) fewer
  bytes; it is gone. In its place: when only the short table matched,
  the long table's entry one byte on (loaded already) is tried and its
  match taken when it is at least two bytes longer; zstd's
  unconditional form loses 0.7% on the table dump, this one gains on
  every file. The literal Huffman lengths meet their 11-bit limit by
  package-merge (optimal) instead of halving the counts. Bytes against
  zstd -3: GitHub events −7.8%, a Wikipedia table dump −1.2%, the NASA
  log −0.6%, enwik8 −0.5%, Silesia mozilla −0.1% (zstd 1.5.5; against
  1.5.7, whose new block splitter gains 1.2% on mozilla, that file is
  +1.1%). One core on Graviton3 against `zstd -3` as installed: events
  1.05× its speed, mozilla 1.05×, enwik8 1.01×, the log and the dump
  0.89×; against `zstd -3 --single-thread` 1.03–1.26× on all five.
  Eight cores against `zstd -3 -T8`: 1.09×, 1.21×, 0.97×, 1.08×,
  0.83×. On a Ryzen 9 7950X3D (one core, zstd 1.5.7) 1.07–1.26×
  faster on all five. The CLI's one-core path writes on a second
  thread, as zstd's does; the parse loop keeps fewer values live.
  Every stream decodes as before.

## v0.14.4 — 2026-09-23

- **`--max` is now zstd -3's structure: the long-distance matcher is
  opt-in (`--long`, `-L`).** The pass that finds repeats up to 128 MB
  back was a third of the write time on logs and events; zstd -3 has
  no such pass, so `--max` now runs without it and `--max --long` is
  what `--max` was, as `zstd --long`. Library:
  `compress_into_max_long`, `compress_parallel_into_max_long`,
  `compress_records_into_max_long`; C `GLYD_LEVEL_MAX_LONG`; Python
  level `"max-long"`, Go `LevelMaxLong`. `--dense`, `--ultra`, base
  mode and the store keep the long search: their job is the ratio.
  One core on Graviton3 / Sapphire Rapids, `glyd -9` against zstd -3:
  GitHub events 0.93/0.95× the speed at 9.7% fewer bytes, the NASA
  log 0.81/0.80× at 4.1% fewer, a Wikipedia table dump 0.82/0.74× at
  1.0% fewer, Silesia mozilla 0.94/0.87× at 0.5% fewer, enwik8
  0.89/0.80× at 0.3% fewer (v0.14.3 wrote at 0.50–0.83×). Eight cores
  against `zstd -3 -T8`: events 1.08/1.14×, the log 1.07/1.07×,
  mozilla 1.05/0.99×, the dump 0.84/0.85×, enwik8 0.86× (Sapphire
  Rapids). `--max --long` against `zstd -3 --long=27`: 0.83–1.18× the
  speed at 0.3–14% fewer bytes. Every stream decodes as before.

## v0.14.3 — 2026-09-23

- **A repeat offset after zero literals is implied by the literal
  length** (flag `LL0_REP`): a match with no literal before it cannot be
  the last offset going on, so the repeat codes shift and the common
  case — records alternating between two sources — is code 0 every
  time, which is zstd's rule too. Measured with zstd's own sequences in
  both coders: ours was 3.4% behind zstd's on the table dump, now 1.1%
  (table headers). On the servers, one core, the Wikipedia table dump
  32.48 → 31.44 MB per 200 MB (zstd -3 31.18), the NASA log −0.3%,
  GitHub events −0.2%, Silesia mozilla −0.25%, at the same speed.
  Earlier files decode as before.

## v0.14.2 — 2026-09-23

- **The max level's parse takes the last offset one byte on, before
  anything the hash tables say** (zstd's double-fast order): a record
  that differs from the one before it in a byte keeps its offset, which
  codes in a couple of bits. On Graviton3 and Sapphire Rapids, one
  core: a Wikipedia table dump 6.3% smaller (34.67 → 32.48 MB for 200
  MB; zstd -3 31.18), GitHub events 0.8%, the NASA log 1.2%, Silesia
  mozilla 0.2%, at the same speed. The probe steps grow after 256
  misses instead of 64 (as zstd's double-fast).
- **Blocks are checked with CRC-32C** (flag `CRC32C`, the hardware
  instruction on aarch64 and x86-64). The Adler-like sum every earlier
  release wrote kept 16 bits of its weighted half and missed, for one,
  two bytes swapped 8 KB apart — found by the mutation fuzz once the
  parse above changed the bytes it mutates. Earlier files verify as
  before; files written from now on need v0.14.2 or later to verify.

## v0.14.1 — 2026-09-23

- **JPEG on every core.** The range coder's decision is branch-free
  (the mispredicted branch on the bit was the cost); the scan is
  written in bands on every core and, when it has restart intervals,
  parsed in bands at its markers; files of 10 MB and up take eight
  stripes. All cores against v0.13.4 (Lepton, single-threaded), every
  decode byte-exact: the 13.4 MB photo written in 0.62 s and read in
  0.29 (1.77 and 0.91); the 6.4 MB one 0.48 and 0.23 (0.97 and 0.50);
  the three of ~2 MB 0.16–0.21 and 0.07–0.10 (0.37–0.43 and
  0.19–0.22); every one smaller. One core: 12–16% slower than v0.13.4.
  v0.14.0 files read back unchanged (fixtures in `tests/data/legacy`).

## v0.14.0 — 2026-09-23

- **JPEG recoded by Glyd's own model.** `src/jpg/` (design:
  `docs/design/jpeg-recoding.md`) parses a baseline JPEG into its
  markers and coefficients and writes it back bit for bit; the
  coefficients are coded with a range coder under contexts from the
  blocks above and to the left, the first row and column predicted
  from pixel continuity across the block edge, the DC from both
  edges, in four stripes of block rows after a prefix so four cores
  share the work. Stream `GJPG` inside the `GLYDJPEG` envelope and
  inside containers (spec 2c); the kept bytes (EXIF, previews)
  compressed. `lepton_jpeg` stays only to read what v0.13.0–v0.13.4
  wrote (the `jpeg` feature). Against v0.13.4 on this Mac, every
  decode byte-exact: smaller on all five photos (13.4 MB: 9,934,880
  against 9,971,627 bytes; 6.4 MB: 4,997,778 against 5,001,278; the
  three of ~2 MB by 0.4–0.8%), 1.6–1.9× faster to write and 1.6–1.7×
  faster to read on all cores (13.4 MB: 1.02 and 0.54 s against 1.77
  and 0.91); on one core 1.3× slower each way. A progressive JPEG
  stays as it is, as before.
- The binary arithmetic coder is generic over its probability type
  (`reflate::coder::Prob`); the corrections coder is unchanged.

## v0.13.4 — 2026-09-23

- **Containers opened by Glyd's own deflate reconstruction.**
  `src/reflate/` (design: `docs/design/deflate-reconstruction.md`)
  parses a deflate stream into its blocks and tokens, runs zlib's own
  matcher over the plain text — `deflate_fast` and `deflate_slow`,
  the level's chain, lazy and nice limits, the rolling hash, the
  window, memLevel and windowBits detected from the stream — and
  builds each block's Huffman trees the way zlib does, so that for a
  stream zlib made almost nothing needs saying: what differs is coded
  with a binary arithmetic coder. Streams are cut into 1 MB chunks at
  block boundaries, each emulated from the window before it, so they
  open and close on every core. Nothing outside this repository is in
  the codec's path any more; the copy of preflate-rs stays to read
  what v0.12.0 to v0.13.3 wrote and to open the bases their deltas
  were made against. Envelope `GLYDDEF3`; segments 9–12 (spec 2b).
  Against v0.13.3 on this Mac, all cores, every decode byte-exact:
  the NASA gzip 8.19 → 8.17 MB, written in 1.2 s instead of 2.4;
  a PDF (pdfTeX, 4 KB windows) 741 → 728 KB, 0.15 s instead of 0.55,
  read in 0.06 instead of 0.15; a PNG 1.65 → 1.60 MB, 0.16 s instead
  of 0.49; a Guava jar 1.77 → 1.68 MB; a .docx 2.32 → 2.27 MB, 0.25 s
  instead of 0.70; the mixed tar.gz 10.18 → 10.11 MB, 1.8 s instead of
  3.7. On a 1 MB text through zlib at every level and strategy the
  recipe is 63–250 bytes (0.01–0.1% of the stream) where preflate's
  corrections were 28 bytes to 17 KB. Behind on one file: a PDF of
  large images whose streams are level 9 (a 4,096-deep chain per
  token) writes in 3.3 s instead of 2.1 and reads in 1.3 instead of
  0.6 — the matcher's speed on such data is the next piece of work.

## v0.13.3 — 2026-09-22

- **Containers open and close on every core.** A deflate stream of
  16 MB of content or more is cut into 8 MB chunks at block boundaries;
  each is predicted, checked and later re-created by a predictor of its
  own that first learns the 32 KB before it, so every chunk runs on its
  own core and the pieces join bit for bit (segments `DEFLATE_CHUNKED`
  and `DEFLATE_NESTED_CHUNKED`; `preflate-rs` is carried in
  `third_party/` with the chunking added, see its README). Ten cores,
  byte-exact: the 20.7 MB NASA gzip written in 2.3 s instead of 4.6,
  read in 0.26 s instead of 1.56; a 6.8 MB PDF with figures 1.9 s
  instead of 13.7, read in 0.59 s instead of 4.3; a 23 MB tar.gz 3.5 s
  instead of 6.8, read in 0.50 s instead of 1.9. One thread: the same
  as before. Corrections grow by a few hundred bytes per chunk.
- **Content reads.** `glyd -d --content`, `glyd-store --get ID
  --content`, `decompress_content` and `decompress_content_with_base`:
  the content of a gzip or zlib object stored opened — what `gunzip`
  prints, members one after the other, a tar.gz's tar — without
  re-creating the deflate stream, which is 96% of a read. The 20.7 MB
  NASA gzip on one thread: 0.36 s for the content against 1.58 s for
  the gzip back (0.11 s on ten cores; `gunzip` 0.10 s: the stored form
  is record mode, whose decode is the remaining cost). A zip, a PDF, a
  tar of gzips and an object stored closed have no content view.

## v0.13.2 — 2026-09-22

- **The default, fast and turbo levels leave containers closed.** They
  opened gzip, zip, tar, PDF, PNG and JPEG objects like every other
  level, at 0.6–5.5 MB/s on one thread, and on most then wrote the
  closed form anyway, an LZ4-class level on the content losing to the
  file's own deflate; on a jar, a JPEG and a .pptx they kept the opened
  form, whose reads run at 9–20 MB/s. Those levels exist for speed, so
  they no longer open anything: a 20.7 MB gzip goes through the default
  level at 1,529 MB/s instead of 5 (zstd -3: 1,204), a 6.8 MB PDF with
  figures in 0.00 s instead of 11.7. Containers open from `--max` up,
  as before. Files those levels wrote with an envelope still decode.

## v0.13.1 — 2026-09-22

Fixes. Every file v0.12.0 and v0.13.0 wrote reads back with this one.

- **Files that did not decode.** Since v0.12.0 a unit of a multi-unit
  stream could itself be opened as a container: a gzip member (from
  v0.13.0 also a zip, tar or PDF) that began exactly on an internal
  unit boundary got an envelope where a block was due, and the file
  failed to decode ("Implausible block header"): an error, never wrong
  bytes. Seen with v0.12.0's `--max` and v0.13.0's default level and
  `--max` on a file with a gzip member at 8 MB. A part of a stream (a
  unit, a record unit, a trial sample, an opened container's plain
  text) is never opened now, and the decoders read such files: the
  stream is read around the envelope, its inner blocks taken until they
  hold the plain text its recipe needs. Files written by the released
  binaries are in `tests/data/legacy/`, and a test reads them.
- **Containers against a base, and their speed.** v0.13.0 kept a
  container's own bytes (a tar's files and headers, a zip's directory)
  and the corrections in the recipe, out of reach of base mode: an
  Ubuntu image with 6,128 .gz files inside came out of
  `compress_with_base` at 250.7 MB in 17 s, where without opening it is
  25.6 MB. The envelope is now `GLYDDEF2`: the plain text holds the
  streams' content, then every other byte, the corrections and the
  transcoded pictures; the recipe is structure only. The same pair:
  24.8 MB in 3.6 s, decoded in 1.9 s. Tar, zip and PDF entries open on
  every core, and segments close on every core. `GLYDGZIP` (v0.12.0)
  and `GLYDDEFL` (v0.13.0) envelopes still decode, alone and against a
  base.
- When the opened object loses to the closed one, the closed
  compression already made is written instead of being made again.
- A deflate stream expands to at most 200 times its size (at least
  256 MB) before it is left closed.
- `preflate-rs` and `lepton_jpeg` are pinned to exact versions: a base
  must open to the same plain text for as long as its deltas are kept.

## v0.13.0 — 2026-09-22

### Deflate containers opened

- Zip (and so .docx, .xlsx, .pptx, .jar, .apk, .odt), zlib streams and
  PNG join gzip: `src/deflate.rs`, envelope `GLYDDEFL` (replacing
  v0.12.0's `GLYDGZIP`). Every deflate stream inside is decoded with
  what it takes to re-encode it bit for bit; headers, directories,
  stored entries and non-image chunks are kept; a PNG's image stream is
  cut back into its IDAT chunks. Entries preflate cannot reproduce stay
  as they are, and so does an object that would not shrink. The CLI's
  `--max` and `--ultra` take record mode on an opened container where
  it pays. PDF too: every stream whose data is zlib and that ends
  before an `endstream`, found by scanning. The recipe goes into the
  envelope compressed (a PDF's duplicate fonts and a jar's thousand
  entries repeat their corrections), and the object is also compressed
  closed at the same level, the smaller kept. Measured, decodes
  compared: a Guava jar 3.05 → 1.76 MB at `--max` (zstd -19 on the
  jar: 2.70), 1.14 MB cold; a GitHub source zip 2.73 → 2.28, 1.64
  cold; a 60-slide .pptx 88 → 24 KB; a 30,000-row .xlsx 1.34 → 0.47
  MB cold; a pdfTeX paper 2.22 → 0.73 MB (zstd -19: 1.04), a paper
  with figures 6.77 → 4.23 (5.54); a PNG photo 1.83 → 1.64, 1.19 cold.

### JPEG transcoded

- A JPEG is recoded losslessly by Lepton (`lepton_jpeg`, the Rust port
  of Dropbox's, behind the default feature `jpeg`): its DCT
  coefficients under an arithmetic coder with a predictor across
  blocks, envelope `GLYDJPEG`, the identical JPEG back. Six photos,
  35.9 MB: 27.3 MB, 24% fewer bytes (a JPEG XL transcode: 20%), every
  one restored byte for byte; 5–7 MB/s in, 12–14 MB/s out, one core.
  A JPEG Lepton cannot take, or that does not shrink, stays as it is.
  Inside a container too: a JPEG stored in a zip, or deflated (as an
  Office document holds its pictures), is transcoded under its entry;
  any container stored or deflated inside another is opened with a
  recipe of its own nested in the segment, four deep, and tar joins
  the containers. A 12-slide deck of photos, 6.51 MB: 5.29 MB (zstd
  -19: 6.50); a document of six PNG screenshots, 2.51 MB: 2.32 MB at
  `--max`, 1.65 MB cold (zstd -19: 2.50); a tar of six photos, 36.0
  MB: 27.3 MB, and 27.3 MB through a gzip of it; a tar.gz of a gzipped
  log, a PDF, a PNG and a .docx, 22.9 MB: 10.2 MB (zstd -19: 22.9).

### Speed, same bytes

- The store's put, where its time went (`GLYD_STORE_TIMING=1` prints
  it): the fingerprint scan now runs on every core; a 4 GB cache of
  decoded objects serves the next base and every delta on a chain (its
  root decoded once, where each object over 1 GB fetched and decoded
  its base again); a 32 MB sample decides a delta before the whole is
  tried, and its alone size is the estimate (an hour of events shares
  half its fingerprints with the hour before and gained nothing from a
  full delta). Kernels put at 800–1,600 MB/s on this Mac (300–500
  before), events at 1,300–1,900; every stored byte identical. The
  terabyte gate rerun ([report](docs/benchmarks/store-gate-2026-09-22.md)):
  put 243 MB/s against 150, 1 h 21 min against 2 h 12 min, every
  object back byte-exact, the rebuild 51 min against 86.
- Dense max level (`--dense`, `compress_into_max_dense`,
  `compress_max_stream`): units stay at the far matcher's 128 MB
  instead of shrinking to give every core one; a unit's far matches
  are found by one core and its blocks parsed in 16 MB stripes by all
  of them, tables seeded with the 2 MB before each stripe, so the
  bytes are one core's at any core count: 5–9% fewer than the default
  on files of a few hundred MB, the same on files of gigabytes (the
  8.7 GB suite corpus: 3.946 against 3.939). Reads then scale only
  with the units (the suite's 8-thread decode 4.7 GB/s against 10.4),
  so it is opt-in: the CLI's `--dense`, and the store, whose objects
  are written once and read rarely. Raw results of the suite run with
  it: `benchmarks/suite/*-dense/`.

## v0.12.0 — 2026-09-22

### Gzip objects opened

- A gzip object's deflate streams are decoded to their plain text
  with what it takes to re-encode each bit for bit (`preflate-rs`,
  the crate's one dependency, behind the default feature `deflate`);
  the plain text then takes whatever was asked — a level, record
  mode, the cold level, a base — so a gzipped log costs what the log
  costs. Every encode entry point opens gzip input, every decode
  entry point closes it, and an opened object that would cost more
  than the gzip stays as it is. A gzip -6 NASA log: 20.7 MB → 8.2 MB
  at `--max -r`, 6.5 MB cold; a gzipped 512 MB kernel tree: 72.7 MB →
  56.1 MB at `--max`, 42.0 MB ultra; all back byte-exact. Through the
  store too.

### The store at a terabyte

- The gate run ([report](docs/benchmarks/store-gate-2026-09-22.md)):
  1,192 objects, 1.18 TB, put into S3 from one instance, 49.0 GB
  stored against zstd -3's 153.5 GB (3.13× fewer bytes, 24× against
  raw), every object read back byte-exact, the metadata directory
  rebuilt from the bucket and verified.
- The first attempt died of memory on 20 GB objects: put now maps its
  files instead of reading them, and the last object is kept as the
  likeliest next base only up to 1 GB.
- `glyd-store --version`.

## v0.11.2 — 2026-09-21

- A lost metadata directory is rebuilt from the objects: every
  object's index lines now ride beside it in the backend as
  `<id>.index` (a pack's carry its members'), and `--rebuild` (or
  `Store::rebuild_with`) remakes the index from those sidecars and the
  fingerprint table by reading every object back. Checked live: two
  kernel tarballs put to S3, the metadata directory deleted, rebuilt
  in 15 s, verified, read back byte-exact, and a third version then
  stored as a 1.8 MB delta against the second.
- `Backend::list` on both backends.

## v0.11.1 — 2026-09-21

- Multipart upload: objects over 64 MB go to S3 as 64 MB parts on up
  to 8 connections and one completion, so objects up to 640 GB store
  and fat links fill; a failed part or completion aborts the upload,
  leaving no parts behind to be billed. Checked live: a 17 MB object
  in four parts read back whole, undersized parts refused and aborted
  cleanly, the 201 MB kernel object in three parts byte-exact. On a
  home uplink the two-kernel put took 12.2 s against 11.6 s single-put:
  the link, not the client, is the limit there.

## v0.11.0 — 2026-09-21

### The store speaks S3 itself

- `S3Backend` replaces `S3Cli`: PUT, GET, HEAD, DELETE and
  ListObjectsV2 over HTTPS with Signature V4, no AWS CLI on the
  machine. Credentials from the environment, `~/.aws/credentials`
  (`AWS_PROFILE`) or the instance/container role, refreshed before
  they expire; the region from the environment, the profile, or the
  bucket's answer; `AWS_ENDPOINT_URL` for MinIO, R2, B2, Ceph and other
  S3-compatible services. Retries with backoff on 5xx and throttling.
  Two 1.5 GB kernel tarballs put through S3, one as a 3.4 MB delta,
  read back byte-exact and verified; the two-object put took the same
  11.6 s as through the CLI on this link. Single puts up to 5 GB;
  multipart upload is next.
- `--audit s3://...` lists and reads through the same client.
- The `glyd-store` crate takes its first dependencies for this:
  `ureq` (HTTPS through rustls), `sha2`, `hmac`. The `glyd` crate
  stays at zero.

## v0.10.2 — 2026-09-21

- The codec's licenses are now exactly zstd's: BSD 3-Clause (`LICENSE`)
  or GPL version 2 (`COPYING`), at the user's option, replacing
  Apache-2.0 OR GPL-2.0. Whatever may ship zstd may ship Glyd. The store
  stays under BUSL 1.1.
- `glyd.h` and the CLI banner no longer claim GPU or AVX-512 kernels;
  there are none (AVX2 and NEON only).

## v0.10.1 — 2026-09-21

- The codec is dual-licensed, Apache-2.0 or GPL-2.0 at the user's
  option (`LICENSE`, `LICENSE-GPL2`), the choice zstd offers, so GPLv2
  projects can carry it too. The store stays under BUSL 1.1.

## v0.10.0 — 2026-09-21

### Installing
- `brew install surya-koritala/glyd/glyd` (the tap
  github.com/surya-koritala/homebrew-glyd; installs both CLIs and
  `glyd.h`); every release carries prebuilt CLIs, libraries and Python
  wheels for Linux x86_64 / aarch64 and macOS arm64
  (`.github/workflows/release.yml`), and publishes to crates.io and
  PyPI once those tokens are repository secrets.

### The codec under Apache-2.0; the store its own crate under BUSL
- The `glyd` crate (the codec, every level and mode, base mode, packs,
  shape dictionaries, the C ABI, the `glyd` CLI) is licensed under the
  Apache License 2.0. The store moved to the `glyd-store` crate
  (`glyd-store/`, a workspace member) under the Business Source License
  1.1, with its own CLI (`glyd-store DIR --put ...`, `--audit`) and its
  C ABI in `libglyd_store`, which carries the codec's ABI too. The
  bindings load `libglyd_store` when present and `libglyd` otherwise
  (everything but `Store`). `Mapping` moved to `glyd::mmap`.

## v0.9.3 — 2026-09-21

### Adoption: bindings, the audit, the spec, lzbench, packaging
- C ABI, allocating form (`include/glyd.h`): `glyd_compress2` (every
  level, record mode, thread count), `glyd_decompress2` (any stream),
  `glyd_decompressed_len`, base mode, packs (`glyd_pack`,
  `glyd_unpack_object`, `glyd_pack_len`), the store (`glyd_store_*`:
  open with a directory or an S3 url, put, get, id_of, delete,
  compact, flush, rebase, verify, stats, set_level, count), `glyd_free`;
  `glyd_compress_fast` / `_turbo` in the buffer form; `glyd_version`
  reports the crate's version. Every decoder now reads a pack (its
  objects back to back).
- Python (`bindings/python`, ctypes, `pip install`) and Go
  (`bindings/go`, cgo) bindings over that ABI, each with a test that
  round-trips every level, record mode, base mode, packs and the store.
- `glyd --audit DIR|s3://bucket/prefix [--sample N]`: a sample of the
  objects (runs of consecutive names at eight places in the listing)
  through a store, against zstd -3 per object when the CLI is there,
  scaled to the listing with the yearly cost at S3 Standard's list
  price.
- `docs/spec.md`: the formats as a map for decoder writers (block
  framings and flags, every envelope's layout, the store's files).
- `contrib/lzbench`: Glyd in lzbench (`setup.sh <checkout>` builds it
  in; levels 1 default, 2 fast, 3 turbo, 4 max, 5 ultra); run and
  checked on an lzbench checkout.
- Packaging: crates.io metadata (the crate excludes the corpus, the
  benchmarks and the bindings), a Homebrew formula (`Formula/glyd.rb`).

## v0.9.2 — 2026-09-21

### Write speed, and the corpus rerun on AWS
- The suite (`scripts/bench_aws_suite.sh`, both machines, every decode
  checked; [docs/benchmarks/suite-2026-09-21.md](docs/benchmarks/suite-2026-09-21.md)):
  `--max` 3.939 over the 8.7 GB corpus (zstd -3 3.851) at 1,512 MB/s
  on 8 Graviton3 cores against zstd -3's 1,969 — 0.77× (0.61× in the
  previous report), decoding at 10,413 against 1,422; one core 233
  against 313 (0.74×), decode 1,571 against 1,425. Sapphire Rapids:
  1,122 against 1,513 at 8 threads, 266 against 377 on one. `--max -r`
  4.712 at 643 MB/s; `--ultra` 4.663 (zstd -19 4.664); `--ultra -r`
  5.224. In the S3 workflow `--max`'s reads now cost less CPU than
  zstd -3's on Graviton3 (9.7 against 10.7 s over the corpus), so it
  is the cheapest row at every read rate: a terabyte-year at 1 / 10 /
  100 reads a month $71.6 / $81.3 / $178 against zstd -3's $73.3 /
  $84.1 / $191; `--max -r` $61.7 / $82.4 / $289.
- The max level's finder tables are zstd -3's size (17/16 bits, 768
  KB) instead of 2 MB: on Graviton3 and Sapphire Rapids the 2 MB
  tables ran 10–25% slower for 0.5–1.5% fewer bytes (on an M1 the
  difference is 2–7%). One core, `benchmarks/max`: JSON events
  330/463 MB/s against zstd -3's 440/633 (Graviton3 / Sapphire Rapids),
  a table dump 301/420 against 331/462, a root filesystem 178/248
  against 115/294. Where the time goes: the parse 45–80%, the
  long-distance pass 7–45% (JSON), the entropy coder 10–18%.
- The CLI maps its input file instead of reading it first, and at
  `--max` and `--ultra` writes each unit as it finishes from a writer
  thread (`compress_stream`), so the output never sits whole in memory
  and the write overlaps the compressing: a 512 MB file at `--max` on
  ten cores 1,060 → 1,500 MB/s (a root filesystem) and 2,090 → 2,940
  (JSON events); in-process the level runs at 1,760 and 4,200.
- Measured and not taken: probing only the first repeat offset (no
  faster, 2–3% more bytes); 16/15 tables (5–12% faster again, 1–2%
  more bytes).

## v0.9.1 — 2026-09-21

### The store: S3, rebase, a second candidate
- `Backend` trait for the objects' bytes: `LocalBackend` (a directory)
  and `S3Cli` (an S3 bucket through the AWS CLI, `aws s3 cp` per
  object; the library carries no HTTP client); `Store::open_with`.
  CLI `--s3 s3://bucket/prefix`. Metadata (index, table) stays local.
  Round trip through a real bucket verified.
- `rebase(id)` / `--rebase ID`: an object stored alone again, one
  decode to read; a later index line for an id replaces the earlier.
- Two base candidates when the second scores at least half the first:
  both tried on the first 32 MB, the smaller delta wins the object. On
  the bucket: 1,313.6 -> 1,312.1 MB (the single candidate was already
  within a few percent of ideal), put 63 -> 70 s.
- The per-object fingerprint files are gone (the table on disk is the
  record).

## v0.9.0 — 2026-09-21

### The store, complete
- `delete(id)` (a tombstone in the index; the bytes stay while a live
  object's chain runs through them), `compact()` (removes what no live
  object needs: deleted objects off every live chain, packs whose
  members are all deleted; returns the bytes freed), `verify()` (every
  live object read back and checked), `id_of(name)`, `set_level`
  (`Max`, `Ultra`, `Cold` for objects stored alone and packs; deltas at
  the ultra level under `Ultra`). Deleted objects are never chosen as
  bases. CLI: `--find NAME`, `--delete ID`, `--compact`, `--verify`;
  `--ultra` / `--cold` with `--store` set the level.
- Put keeps the last large object in memory as the likeliest next base,
  and judges a delta against the object alone estimated from its first
  64 MB when that settles it either way (a version's delta is a few
  percent of the estimate, an unrelated object's about all of it),
  compressing the whole object alone only in between. The bucket put
  79 -> 63 s (620 MB/s end to end), same bytes; a version of the last
  object put runs at 900 MB/s. An estimate used alone, without the
  exact check in between, chose bases that were not worth it (the
  bucket 1,314 -> 1,890 MB) — measured, and not shipped.

## v0.8.1 — 2026-09-21

### The store at scale
- The fingerprint table lives on disk: an open-addressing hash table
  (12-byte slots, the fingerprint and the object id, linear probing, a
  fingerprint's last eight holders kept) mapped into memory through
  libc's `mmap`, at most half full, doubled in a fresh file when it
  fills. The store's memory no longer grows with what it holds: the
  39 GB bucket's table is 100 MB on disk and the put's memory is the
  object's own working set. Same bytes (1,314 MB, 29.9x), 480 MB/s end
  to end.
- Small objects (under 256 KB) go into packs of about 2 MB, one stored
  object each (`flush` writes the open pack; `Drop` flushes); `get`
  decodes the pack and slices, keeping the last pack decoded. 2,000
  GitHub events put one by one: 9.0x against zstd -3's 3.6x per event.
- Objects stored alone go through record mode where it pays
  (`compress_records_into_max`), so a log or a dump put into a store
  gets its columns.

## v0.8.0 — 2026-09-21

### The store: compression across objects
- `Store::open(dir)`, `put(name, data)`, `get(id)`, `entries`, `stats`;
  CLI `--store DIR --put FILE...`, `--get ID -o`, `--stats`. Each
  object's fingerprints (one sparse anchor in 4 KB) are looked up in
  the store's table (the last eight holders of each); the stored object
  sharing the most is its base, taken when the delta (`--base`) saves a
  fifth or more of the object alone; chains at most four long, the
  chain's root past that. A 39 GB bucket (six Ubuntu image builds, the
  fifteen Linux 6.10 releases, two months of three Wikipedia tables,
  twelve hours of GitHub events; `scripts/download_bucket.sh`): 1,334
  MB against zstd -3's 6,132 MB per object, 4.6x, put at 500 MB/s end
  to end and read back at 270 MB/s with the file written, every object
  byte-exact; by family 13.9x, 5.2x, 2.0x, 1.3x. The
  research pass that led here: experiments/research/README.md, H.

### Packs: small objects as one stream
- `compress_pack(objects, out, level)`, `decompress_pack`,
  `decompress_pack_object(pack, i)`, `pack_len`; CLI `--pack files...`,
  `--unpack dir`. Envelope `GLYDPACK`: the count, the lengths as
  zigzag deltas compressed at the max level, then the concatenation in
  record mode where it pays. 1 MB packs of 1 KB objects at `--max`:
  JSON events 42.4x (zstd -3 + dict per object 10.6x), NASA log 15.2x
  (5.3x), HDFS 16.2x (6.0x), CSV telemetry 8.9x (3.8x), taxi CSV 7.6x
  (3.9x); 40-130 MB/s to pack, an object read back in 0.5-1.7 ms.
  Record mode's pay decision now samples an eighth of a small input
  (at least 256 KB) instead of the whole of it.

### Shape dictionaries: record mode for small objects
- `ShapeDict::train(sample)`, `compress`, `decompress`, `to_bytes`,
  `from_bytes`; CLI `--shape-train`, `--shape`. The dictionary carries
  the shape (delimited, JSON lines, or a log's skeletons: punctuation
  between runs of letters and digits), the frames lines take with a
  column per hole, the columns' types (integers with a recency ring
  and deltas, times, decimals, dictionaries seeded with the sample's
  values by frequency, constants, text) and an LZ `Dict` trained on
  such objects' images. An object is a compact image: a byte per row,
  then every column's values, each column self-delimiting; unknown
  lines stay raw. 1–4 KB objects cut from real files: JSON lines 14.6×
  and 24.3× (zstd -3 + dict 10.6× and 13.0×), CSV telemetry 5.8× and
  7.9× (3.8×, 4.4×), HDFS log 6.7× and 9.7× (6.0×, 7.4×), NASA log
  5.1× and 7.3× (5.3×, 6.4×). 60–150 MB/s to code, 55–430 MB/s to
  decode, one core. The same objects packed into one record-mode
  stream cost 2–4× less than zstd + dict per object.

## v0.7.0 — 2026-09-20

### The cold level: context mixing
- `glyd --cold` (`compress_into_cold`, `compress_parallel_into_cold`,
  `compress_records_into_cold` with `-r`): every bit predicted from
  eleven contexts (byte orders 1-4, 6, 8; the word and the one before;
  the column and the byte above; the JSON key; the longest earlier
  match) through paq-style bit histories, mixed by two networks, two
  SSE stages, a binary arithmetic coder; 32 MB units coded from an
  empty model, in parallel, each with a checksum in the envelope
  (`GLYDCOLD`); every decoder reads it. 64 MB slices, two threads: JSON
  events 22.5x (zstd -19 14.6x, `--ultra` 15.9x, zpaq -m5 22.8x), NASA
  log `-r` 31.1x (15.7x, 26.4x, 31.7x), page_props dump `-r` 11.6x
  (6.2x, 8.6x, 11.1x), webster 7.1x (4.8x, 4.8x, 7.3x); the HDFS and
  Spark logs (128 MB, `-r`) 33.7x and 65.2x against zstd -19's 16.0x
  and 25.2x; 1.2-1.5 MB/s per core each way, 400 MB per thread. Record mode decides whether
  its transform pays at the max level whatever the level.

## v0.6.0 — 2026-09-20

### Base mode: content found wherever it moved, ultra at full speed
- Each unit's region of the base is chosen from a coarse map of the
  base (its sparse anchors, one per KB, found with the matcher's vector
  scan at 3-4 GB/s and sorted by the hash of the 32 bytes at each): the
  96 MB window holding the most of the unit's own anchors, or the base
  around the unit's position when its content is new. A version with
  48 MB inserted before the kernel tree costs 18.4 MB against 33.5 MB
  with the fixed window (zstd -3 --patch-from: 18.7 MB). The consecutive
  pairs are unchanged within 0.5%; `--max --base` runs at 720-1,700
  MB/s on ten M1 cores (was 860-2,020: the map's cost).
- `--ultra --base` inserted each unit's whole 96 MB region into the
  tree finder, which reaches 8 MB back: it now starts at the window's
  edge and runs at the plain `--ultra` speed, 3-11 MB/s on ten M1 cores
  against 1-4 before (the kernel pair 483 s, the new version alone at
  `--ultra` 555 s), the bytes the same.
- Chains measured (`scripts/download_chain.sh`, `scripts/bench_chain.sh`):
  the 15 Linux 6.10 point releases cost 228 MB each against the one
  before (zstd -3 --patch-from 260 MB; stored one by one 3.0-3.2 GB) or
  246 MB each against 6.10 (zstd 265 MB), a step 1.8 MB and the delta
  against a base 14 releases old 3.6 MB.

### Record mode: templates
- Logs whose lines vary in shape (application and system logs) take a
  fourth shape: each line's template (its text with a hole where every
  token holding a digit was) goes into a dictionary, and the tokens
  become typed columns keyed by template and slot; lines past the
  column budget stay raw. loghub 2.0, 128 MB of each, 10 cores: HDFS
  `--max -r` 22.0 against zstd -3's 10.5 and zstd -19's 16.0 (`--ultra
  -r` 27.5), Spark 47.0 against 14.5 and 25.2 (53.6), BGL 15.9 against
  11.0 and 22.4 (28.9), Android 17.9 against 12.9 and 23.0 (25.4);
  writes at 260-460 MB/s, reads at 1,200-1,400 MB/s. The levers were
  sized first (experiments/research/README.md): version chains, these
  log shapes, context mixing for cold data, float columns.

### Reads
- The record-mode rebuild decodes a column at a time into tables and
  assembles the rows by copy (reserved space, no length branch per
  value; integers through a digit-pair table; the minute's prefix of a
  time column kept and copied as a block; a ring for the recency list;
  16-byte padded dict8 entries; varints of up to three bytes from one
  load). One core, M1 Max: int columns 489 -> 810 MB/s, dictionaries
  344 -> 1,076, times 477 -> 1,815; the NASA log 509 -> 734, JSON
  lines 734 -> 1,117, the taxi CSV 260 -> 409. Record images are
  unchanged.
- The S3 workflow rerun on both AWS machines: the CLI's decompress
  CPU per 8.7 GB fell from 14.5 to 11.7 s on Graviton3 for `--max`
  (zstd -3: 10.9) and from 19.2 to 14.2 on Sapphire Rapids (zstd -3:
  10.4); `--ultra` 11.1 against zstd -19's 12.5. A terabyte-year at
  ten reads a month: `--max` $83.1, `--max -r` $83.6, zstd -3 $84.2.
- The reference codecs (zstd, LZ4, Snappy, LZAV) are dev-dependencies:
  the benchmarks link them, the library and CLI carry none.

## v0.5.0 — 2026-09-20

### Base mode
- `glyd --base old new` / `compress_with_base`: a new version of an
  object compressed against the old one, decodable with it (`glyd -d
  --base old`, `decompress_with_base`). Units of 32 MB are parsed with
  the base around their own position as history (32 MB of slack each
  way), the long-distance matcher reaching all of it; the decoder reads
  the base in place. Against zstd 1.5.7 `--patch-from` on the same
  machine, byte-exact: Wikipedia page-table dumps a month apart
  `--max` 1.79 MB at 863 MB/s (zstd -3 patch 3.84 MB at 409, zstd -19
  patch 1.30 MB at 2), `--ultra` 1.23 MB; Ubuntu cloud root filesystems
  16 days apart 5.31 MB at 2,018 MB/s (zstd -3 8.82 MB at 654, zstd -19
  5.61 MB at 39), `--ultra` 4.59 MB; Linux 6.10 -> 6.10.1 3.04 MB at
  1,980 MB/s (zstd -3 3.26 MB at 560, zstd -19 2.58 MB at 30), `--ultra`
  2.04 MB. Plain `--max` on those
  files: 33, 287 and 200 MB. Design notes in docs/design/format-v7.md;
  the measurement that led here in experiments/structure/README.md.
- A far match's cap of 130 bytes per sequence applied to the
  repeat-offset continuation as well; a repeat carries no offset bits,
  so the rest of the match is now one sequence. Plain `--max` gains 1%
  on JSON events.
- The long-distance matcher's table grows with the input (a slot per
  16 bytes, up to 2^25 entries).
- The CLI decodes a batch of units at a time into one reused buffer
  and writes as it goes (`decompress_stream`): the memory is a batch,
  not the file, and no output page is touched for the first time after
  the first batch; the library's parallel paths run on scoped worker
  threads (`set_threads`) instead of rayon.

## v0.4.0 — 2026-09-20

### Long-distance matching
- The max and ultra levels find repeats of 32 bytes or more up to 128 MB
  back (`src/ldm.rs`): one pass over the input before the parse, with
  content-defined anchors (one position in 16, found 16 at a time with
  NEON or AVX2) hashing the 32 bytes after them into a 16 MB table whose
  entries carry a hash check; matches are verified, extended both ways
  and handed to the parse, which takes one wherever it beats the local
  finder. Format v9 offsets grow to 27 bits (30 offset codes; v8 blocks
  keep 26, and a table with fewer symbols than its version allows still
  decodes). A far match is capped at 130 bytes per sequence so the
  decoder's one-load walk holds its extra bits, the rest following as a
  repeat-offset sequence. Parallel units grow to one per core, up to
  128 MB (the matcher's reach is the unit).
- The pass runs at 1.2-2 GB/s on one core. The max level gates it:
  after 4 MB and 16 MB (or half the input) it stops on data whose
  repeats are too few or too near to pay (media, Parquet, most SQL
  dumps), which keep 92-96% of their speed; the ultra level runs it
  whole. Where it stays on, the max level compresses at 63-85% of its
  former speed for 3-16% fewer bytes. One core, M1 Max, 128 MB of
  GitHub Archive JSON: `--max` 11.63 -> 9.73 MB at 577 MB/s (zstd -3
  12.90 MB at 895; `zstd -3 --long=27` 10.20 MB at 433), `--ultra`
  7.78 MB (zstd -19 8.96, `zstd -19 --long=27` 7.80); NASA access
  log 13.22 -> 11.92 MB at 435 MB/s (`zstd -3 --long=27` 13.63 MB at
  383); Silesia `--max` 3.259 -> 3.302 at 247 MB/s (290 before; zstd
  -3 3.205 at 335). The 8.7 GB corpus on 10 cores: `--max` 3.89 ->
  3.96 at 2,000 MB/s (2,400 before; zstd -3 3.85 at 4,000), JSON
  events 11.49 -> 13.26 (zstd -3 10.46); `--ultra` 4.65 (zstd -19
  4.66), JSON events 16.40 (zstd -19 15.07).

### Record mode
- `glyd -r` / `compress_records_with`: delimited lines, SQL dumps and
  JSON lines become typed column streams (integer, decimal and
  date-time deltas, dictionaries with recency ranks, text) before the
  level, in parallel 32 MB units, rebuilt byte for byte; other data is
  left as it is; input the transform does not pay on (API events with
  hashes and free text, binaries) takes the plain parallel path. JSON
  lines: a column per key path, typed values leave holes in a frame of
  the structure, keys and text. Telemetry as rows, 128 MB slices, 10
  cores: a cluster trace as CSV `--max -r` 12.6 against zstd -3's 4.5
  and zstd -19's 6.9, as JSON lines 54.6 against 15.7 and 28.7; daily
  weather 19.6 / 47.0 against 7.0 / 18.7 and 12.0 / 31.6; taxi trips
  exported to CSV 8.9 against 5.6 and 8.4 (`scripts/download_ext_corpus.sh`). Whole 8.7 GB corpus, 10 cores: `--ultra -r`
  5.20 against zstd -19's 4.66 (SQL dumps 1.43x smaller, access logs
  1.54x, JSON 1.09x through the plain level's matcher); `--max -r`
  4.75 at 1,100 MB/s. Design notes in
  docs/design/format-v7.md; the prototypes and measurements that led
  here in experiments/structure/.

### Small objects and dictionaries
- `Dict`: a prepared dictionary (trained content plus entropy tables)
  for small objects; `Dict::train` (cover selection as zstd's fastcover,
  scoring each distinct string once), `to_bytes`/`from_bytes`,
  `compress_with_dict`, `compress_with_dict_ultra`,
  `decompress_with_dict`. The object is parsed in place against the
  dictionary's own seeded tables; the decoder copies from the content
  and borrows the dictionary's built tables.
- Format v9: compact framing for blocks of at most 32 KB (one marker
  byte, varint lengths, a 5-10 byte sub-header, single-stream sections
  under 1,024 symbols, no padding on disk): 207 -> 21 bytes of framing
  on a 4 KB object. Every earlier format decodes unchanged
  (tests/format_compat.rs holds v0.2.0, v0.3.0 and v0.4.0 output).
- The ultra level with a dictionary prices its parse from the
  dictionary's tables.
- Per-object work cut: entropy tables and codes built once per
  dictionary, table costs from a lookup, buffers kept across calls,
  single-stream decode paths (three code chains at once, one-load
  batches).

GitHub Archive JSON objects, Apple M1 Max, one core, 110 KB
dictionaries trained on other objects (zstd's numbers without a
checksum; Glyd writes 4 bytes per object): 1 KB objects `--max` + Dict
ratio 4.75 (zstd -3 + dict 4.96), compress 245 MB/s (454), decode 920
MB/s (1,117); 4 KB 6.28 (6.42), 321 (588), 1,209 (1,440); 16 KB 7.65
(7.66), 394 (635), 1,674 (1,890). `--ultra` + Dict: 5.25 / 7.13 / 8.84
(zstd -19 + dict 5.47 / 7.41 / 8.95). Before this work the same objects
compressed to 2.15 / 3.87 / - with a window-only dictionary at 6 MB/s
and decoded at 170 MB/s.

### Verification and benchmarks
- `scripts/verify_roundtrip.sh` (every level and core mode through the
  CLI, byte-compared; corrupted copies rejected or decoded exactly),
  `scripts/download_bench_corpus.sh` (~9 GB of logs, JSON, SQL dumps
  and Parquet with a separate training set), `examples/bench_suite.rs`
  (Glyd against zstd -3, zstd -19 and LZ4 at a stated thread count,
  every decode checked, peak memory, small-object latencies),
  `scripts/s3_workflow.sh` (compress, upload, download, decompress,
  verify, monthly cost), `scripts/bench_aws_suite.sh` (all of it on
  Graviton3 and Sapphire Rapids), `scripts/report_suite.py`.
- The format-compatibility fixtures are now committed (they were
  ignored by the `*.glyd` rule; CI failed on every push since they were
  added).
- Results of the program on Graviton3 and Sapphire Rapids:
  docs/benchmarks/suite-2026-09.md (raw rows in benchmarks/suite/).

### Parallel paths
- The parallel compressors cut the input into units of at least 2 MB
  (v6 levels), 8 MB (`--max`) and 16 MB (`--ultra`) instead of 256 KB,
  growing with the input (two units per core, up to 64 MB): JSON events
  lost 4.7% to 8 MB units against the sequential ratio, 1% at 64 MB. Each unit is
  still a chain of its own (parallel decode, random access), and the
  ratio now stays within 0.5-0.7% of the sequential path; at 256 KB the
  CLI's multi-core default was giving up 3% (default level), 6.6%
  (`--max`) and 13% (`--ultra`). Multi-core decode of files with few
  units is correspondingly less parallel (Silesia on 10 cores: 31,900
  MB/s against 42,900).

### Ultra level
- Blocks are split where the parse's statistics change
  (`v7_ultra::split_points`); prices carry the parse's own prior at
  weight 2 and half a bit per literal. Silesia 3.925 → 3.946 (zstd -19:
  4.006), decode 2,150 → 2,090 MB/s.
- x86-64: the walk's and the tANS batch's per-stream state through
  memory: +6% max-level decode on Sapphire Rapids.

## v0.3.0 — 2026-09-19

### Format v8
Every level of the entropy-coded family now writes format v8; v7 (and
v6) files from earlier releases decode unchanged (tests/format_compat.rs
holds v0.2.0 output as fixtures).
- 8 MB window (26 offset codes).
- Section layout: 24-bit sub-stream sizes and one padding per section
  instead of per stream; tANS counts as width + mantissa; literal tables
  as nibbles with unused-symbol runs. Per-block overhead 950 → 421
  bytes.
- Length codes with direct codes to 15 (runs) and 34 (matches) and
  short buckets before the log2 ones.

Silesia, M1 Max, same run: `--ultra` 3.80 → **3.93** (zstd -16 3.83,
zstd -19 4.01), decode 2,150 MB/s (1.3× zstd -19); `--max` 3.22 →
**3.25** (zstd -3 3.20), decode 1,890 MB/s (1.27× zstd -3). Design
notes: docs/design/format-v7.md, "Format v8".

### Library
- The ultra finder sizes its tables to the input per call: 2 MB for a
  256 KB chunk, 64 MB for an input that fills the window.
- `bits::Stream` (a sub-stream with its own length and the bytes to the
  section's end) replaces bare slices in the decoders' signatures.

## v0.2.0 — 2026-09-19

### Levels
- **`--ultra` / `-19`** (format v7, same decoder): optimal parse on a
  binary-tree match finder, every position priced in the coder's own
  bits ([design](docs/design/ultra-parse.md)). Silesia ratio 3.80 vs
  `--max`'s 3.22; zstd -16 3.83, zstd -19 4.01 (3.91 inside Glyd's 2 MB
  window). Compresses at 4.8 MB/s; its output decodes at 2,190 MB/s,
  1.3× zstd -19's. `compress_into_ultra`, `compress_parallel_into_ultra`,
  `compress_with_dict_ultra`; C `glyd_compress_ultra`,
  `glyd_compress_ultra_parallel`.

### Platforms
- x86-64 `--max` decoder: an AVX2+BMI2 entry point and loop shapes for
  16 registers (stream-major entropy batches, the NEON copy structure,
  split walk tables). Sapphire Rapids decode 858 → 1,300 MB/s in the
  published run (zstd -3: 1,260 MB/s); default builds gain the same.
- Cross-platform benchmarks re-run; `ultra_bench` added to the script.

### CLI
- `-19` / `--ultra`; `--single-core` is now `-s` (`-1` was `--fast`).

### Fixed
- Nothing user-visible; see CHANGELOG-BENCH.md for the measurement trail.

## v0.1.0 — 2026-09-19

First public release.

### Levels
- **default** (format v6): LZAV-class parse, minimum match 7. Silesia
  ratio 2.19, decode 6,900 MB/s (1.6× liblz4), compress 340 MB/s.
- **`--fast` / `-1`**: LZ4-class finder, minimum match 5. Ratio 2.18,
  decode 4,900 MB/s, compress 550 MB/s.
- **`--turbo` / `-t`**: minimum match 10. Ratio 1.88, decode 9,200 MB/s
  (2.1× liblz4).
- **`--max` / `-9`** (format v7): 8-way interleaved Huffman literals,
  tANS-coded sequences with repeat offsets, 2 MB window, double-fast
  lazy parse; three-pass decoder. Ratio 3.22 vs zstd -3's 3.20, decode
  1,860 MB/s (1.3× zstd -3), compress 300 MB/s.

### Platforms
- aarch64 NEON decoders for v6 and v7; x86-64 AVX2 decoder for v6 and a
  portable scalar path everywhere else.
- Multi-core compression and decompression over independent 256 KB blocks.

### APIs
- Rust: `compress_into{,_fast,_turbo,_max}`, `compress_parallel_into*`,
  `decompress`, `decompress_into`, `decompress_parallel*`,
  `compress_with_dict` / `decompress_with_dict` (v7), `GlydReader` /
  `GlydWriter` (v6 streaming).
- C ABI (`include/glyd.h`, `libglyd`): `glyd_compress*`,
  `glyd_decompress*`, `glyd_max_compressed_len`, `glyd_version`.
- CLI `glyd`: compress/decompress, level flags, pipes, `-b` benchmark.

### Safety
- Every decoder fuzzed with 1,000,000 random mutations per run into
  exact-size buffers with sentinel guards; no per-call allocation in the
  decoder (1.5 MB thread-local scratch for v7).

### Known gaps
- `--max` compresses at ~90% of zstd -3's speed and decodes 1.3× (not 2×).
- `--max` beats zstd -3 on 3 of 5 extended-corpus files (loses 0.6% on
  repetitive JSON).
- No AVX2 v7 decoder yet (scalar on x86); streaming adapters are v6-only.
