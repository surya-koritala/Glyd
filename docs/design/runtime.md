# The M2 runtime: model step, layer kinds, backend seam

Design for #230 (R9), the first issue of M2 (#121). It covers the model step the scheduler drives, the layer-kind interface the model
descriptions compose, the seam where vendors plug in, the scheduler and KV cache, expert tiering, the order in which the per-family
pipelines move onto it, and the per-step ceiling it is measured against. Numbers are measured where a record is named, derived where the
formula is given. Written 2026-11-20 against main ea26239f; revised the same day for the owner's scope rule below.

**Scope.** What M2 tests and optimises for are the models released in the last six months (cutoff 2026-05-20): Qwen3.8 (27B dense,
Flash-Next, 2.4T-A95B), GLM-5.3 (and -Flash), Muse-Spark-Open-816B-A42B, Kimi-K3 (and K3.1-Pro), gemma-4-12B, DeepSeek-V4 (Flash and Pro,
with their DSpark drafters). Qwen3 (2025), Qwen3.5/3.6, Nemotron-3-Super and gemma-4-31B are legacy: they move onto the runtime so their
pipelines can be deleted, with parity and nothing more. Batching and paged KV exist today only on the legacy Qwen3 path; giving them to
the recent families is the first priority.

## Top gaps for the recent families

1. **Qwen3.8 dense (27B) has no lossless level on the engine.** The Qwen3.5 code path it runs on accepts kestrel, swift and wren only
   (`support.rs`, the `Qwen3.5` row: `levels: LOSSY`); penguin runs only through the frozen vLLM plugin. Its penguin pack is ~38 GiB
   (51.7 GiB bf16), so it needs the 40-80 GB cards or a split (#331).
2. **No recent family batches or pages its KV.** Every one runs one sequence per engine: the description executor refuses more than one
   for every family but Qwen3.5 (`dynamic/mod.rs:2352`), and Condor swaps sequences rather than batching them (R4a #138, R4b #141,
   #329).
3. **The lossless KV format covers none of them.** `kv.rs` takes head dim 128 and GQA groups of at most 8 (`kv.rs:224-230`); the recent
   families use head dims 256 and 512 (Qwen3.8, Gemma 4), 64 (Muse), MLA latents (Kimi-K3, GLM-5.3) and DeepSeek-V4's compressed entries (#332).
4. **Recent releases the engine does not load yet:** GLM-5.3 (the full `GlmMoeDsa` model; -Flash runs), Qwen3.8-2.4T-A95B (refused as a
   Qwen3.5 MoE, `qwen35.rs:492`), Kimi-K3.1-Pro, DeepSeek-V4.1-Flash, and the DSpark drafters (parsed MTP is unused on DeepSeek-V4) (#333).

## Decisions

1. **Names.** `ModelStep` is what the scheduler drives (today's `sched::Backend`, renamed). `Backend` is a vendor (CUDA, HIP,
   Metal through MLX, CPU) and implements layer kinds. `Device` is the CUDA/HIP device layer under those two backends (R1, R15).
   `Engine` is the request-level interface the API front ends share (R11); `model::Engine` went with `model.rs` (R3c, #139: Qwen3 dense is a
   Llama-style description on the executor, `src/dynamic/dense.rs`).
2. **One step over a flat batch.** A step takes a list of sequences, each contributing 1 token (decode), 1+k tokens (decode with
   drafts) or a prompt chunk, packed into one token list. There is no prefill or decode mode; a step is the tokens of every row
   the budget admits, decode rows first, then prompt chunks (Sarathi-Serve, vLLM V1).
3. **The step is asynchronous.** `launch` returns a handle and does not wait; `finish` returns the step's sampled token ids. The
   scheduler builds step N+1 while step N runs; a decode row's next input can name step N's output, which the device resolves.
   Sampling runs on the device; only token ids (and log-probabilities when asked) come back.
4. **The vendor seam is the layer kind.** A backend says per kind and weight format whether it supports it (`supports`), and builds
   a `LayerImpl` for each layer of a description. Nothing above the executor is generic over the backend; the executor is generic
   over a backend's associated types, compiled once per backend; layers are `Box<dyn LayerImpl>`, one virtual call per layer per
   step, none under graph replay. Within the CUDA and HIP backends launches are concrete calls on `Device`, with no virtual call.
5. **The unit of a description is a block, not an operator.** A block is pre-norm, mixer or feed-forward, optional post-norm and a
   residual rule (add, gated, multi-stream mixing, or Kimi's depth mix). Backends fuse inside a block (norm into the product's prologue, residual into its
   epilogue); there is no general operator graph. Glyd's formats need their own product kernels on every vendor anyway, so an
   operator graph would buy little and cost the fusions that make the fixed time.
6. **The CPU f32 reference is a backend.** `Ref` implements every kind in plain f32 Rust and runs whole tiny models, so the same
   description is checked kind by kind and end to end (R2, R7). The M5a CPU backend (V4) is a separate, fast backend.
7. **Typed launches from one definition.** Each kernel takes one `#[repr(C)]` argument struct defined in Rust; the build writes the C
   header the `.cu` (and later Metal) sources include, so the types cannot drift (R15).
8. **Memory is arenas, sized by the plan at load.** Weights, KV pages, state slots, the step workspace and pinned host staging are
   arenas; a step allocates nothing. Buffers are bound to a device, not to the thread's current context.
9. **One KV block size (16 tokens) for every attention kind**, per-layer-group page pools, sliding layers recycle pages past the
   window, MLA caches latents in the same pages, recurrent layers (GDN, KDA, Mamba-2) hold per-sequence state slots.
10. **Blocks are named by content**: a 128-bit hash of the parent's hash, the block's tokens and extra keys (salt, LoRA, image
    hashes). The radix index is over blocks. The hash is of tokens, not of stored bytes, so compressed and bf16 pages, transfers
    and router events share one name.
11. **Admission is deterministic for lossless KV.** A sequence is admitted only with its worst-case escape room reserved from the
    model's KV tables; an admitted request never fails because its keys compressed worse than expected.
12. **Expert tiering is part of the MoE kind on every MoE.** An `ExpertStore` holds tiers (GPU slots, pinned RAM, NVMe) and a per-step
    plan: hits on the GPU, misses copied or computed on the CPU, chosen per expert by cost; each missed expert is read once per step
    for all the tokens routed to it. The routing stays on the device where the store has a device path, so the step stays in a graph.
    Dense layers in RAM use the same store as an expert every row routes to.
13. **Migration moves one family at a time** behind R7's logits test and R8's bench gate, no regression beyond the measured noise band,
    and deletes the old pipeline in the PR after the parity record. Order: the foundation (trait rename, context, kinds, the executor
    behind `ModelStep`), then the recent families with batching and paged KV (gemma-4-12B and Qwen3.8 dense with its lossless level,
    then DeepSeek-V4, Kimi-K3, GLM-5.3, Muse with expert tiering, Condor last of them), then the standard families, then the legacy
    pipelines (Qwen3 dense, Qwen3.5/3.6, Nemotron-H, gemma-4-31B) at parity only.

## 1. Inventory

### Two executors and a handful of engines

Today a model runs on one of two kinds of code:

- **Hand-written engines.** `model.rs` (Qwen3 dense), `model35.rs` (Qwen3.5/3.8), `model_gemma4.rs`,
  `model_qwen4x.rs` (Condor's Qwen3.8-Flash-Next) and the GLM development engine (`model_glm.rs`, `mla_glm.rs`, `mtp_glm.rs`). Each has
  its own step, cache and sampling. **As built (R13, #234):** the GLM development engine is gone (`model_glm.rs`, `mla_glm.rs`, `mtp_glm.rs`,
  `kcheck_glm.rs`, `glm_serve.rs`, `glm_cb.rs`, `draft_loop.rs`, the codes block of `moe_glm_cb.rs`): GLM-5.3-Flash runs on the executor
  alone; what the executor and Condor load GLM by is `kern_glm.rs` (the kernel table, the layers' weights, a latent layer's) and
  `moe_glm.rs` (the experts' layer); its experts are tiered by hummingbird and the block levels (R12). Its MTP drafting went with it:
  R6 (#145) puts the MtpHead kind on the executor (its CPU reference is `runtime/reference/mtp.rs`).
- **The description executor** (`dynamic/mod.rs`). A `ModelDesc` (`mod.rs:518-558`) is a list of `Op`s per layer over 25 activation slots;
  `Dyn::op` is a `match` over 60 `Op` variants (`mod.rs:85-482`, `1428-1870`). Ops are named by family more than by kind (`Prep35`,
  `AttnNh`, `V4Attn`, `GlmPre`, `KimiMixNorm`, `MuseQk`; M2 moves them onto the kinds' operations: the residual streams, the depth mix,
  the gated residual, the norms, the gated activation and the experts are one operation each, their family differences the kinds'
  parameters, `StreamsP::rounding` among them). The executor imports the hand-written engines' kernel tables (`Kern35`,
  `KernG4`, `KernNh`, `KernGlm`, `mod.rs:22-24`). Qwen3.5, Gemma 4, Nemotron-H, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4 and Muse Spark run
  on it. It holds one contiguous bf16 cache per sequence and swaps sequences with `select` (`mod.rs:2383-2399`); more than one sequence
  is refused for every family but Qwen3.5 (`mod.rs:2352`). Multi-GPU runs as a layer split with activations handed over through host
  memory (`dynamic/split.rs:1-4`) or as row slices with an all-gather after each product (`dynamic/tp.rs`, `tp.cu`); `range.rs` runs a
  stage as its own process for other machines.

Only `model.rs` batches (up to `MAX_ROWS = 8`, `model.rs:71`) and only it has the paged lossless KV cache. Every engine samples on the host
from GPU top-k candidates (`TopGpu`), and syncs the device every step.

Kernels live in two places. Every `kernels/*.cu` is included into `kernels/engine.cu` (lines 1006-1021) and built into the engine's fatbin;
the `Kern*` structs call it. The packed dense products (`mma`, `mma12` and unpack), FP8 products, the grouped MoE (`moe`, `moe_way`)
and the whole lossless KV cache (`kv_write`, `kv_gather`, `kv_decode`, ...) are in `gpu/glyd_gpu.cu`, reached through the C ABI in
`ffi.rs:395-439` (R17 folds it in).

### Per family

| family (code) | attention | position | norm and residual | FFN | other | KV and state | weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3.8 dense 27B** (also legacy Qwen3.5/3.6) (`model35.rs`, `dynamic/qwen35.rs`) | 3 of 4 layers Gated DeltaNet (conv 4, fp32 state [nv,128,128], chunk 64); 1 of 4 GQA hd 256 with sigmoid output gate | partial rope 64/256, θ 1e7; mrope read as plain for text | (1+w) RMSNorm | SiLU-gated (MoE refused) | MTP and vision not loaded | bf16 contiguous, 1 sequence (Qwen3.5 can hold several and select) | library layouts; **lossy levels only** (no penguin on the engine) |
| **Qwen3.8-Flash-Next** (Condor, `model_qwen4x.rs`) | 36 GDN layers; 12 sparse attention layers (24 q / 2 kv heads, hd 256; indexer picks 2,048 tokens by 4-token blocks; sigmoid gate) | partial rope 64/256, θ 1e7; M-RoPE for images | four residual streams joined by gated residuals, (1+w); n-gram embedding at layer 1 | MoE: softmax top-10 of 512, SwiGLU, Hadamard-rotated input; gated shared expert 640 | 1 MTP layer (verifies up to 8 rows); vision tower (27 blocks) | bf16 KV, indexer keys, GDN state; sequences swapped, not batched | Int6/5/4 dense, int8 shared and residual, codebook experts cb2/3/4/ternary decoded in `qx_cb_*` |
| **gemma-4-12B** (also legacy 31B) (`model_gemma4.rs`, `dynamic/gemma4.rs`) | 5 of 6 sliding (hd 256, ring window+chunk); global hd 512, K=V option; q/k norm, weightless v norm | full θ 1e4 (sliding); proportional 64/256, θ 1e6 (global) | sandwich RMSNorm, per-layer scalar | GELU-tanh-gated | embed x sqrt(hidden), tied head, softcap 30, separate "assistant" draft model | bf16 contiguous, ring for sliding | library layouts; Int4-6 via `skinny.cu` for 1-16 rows |
| **Kimi-K3** (`dynamic/kimi.rs`) | 69 of 93 layers Kimi delta attention (KDA, fp32 state 96x128x128, conv 4); 24 MLA (q 1536, kv 512, nope 128 + 64 shared, v 128, absorbed, sigmoid gate) | none (no rope anywhere) | KimiRMSNorm (two roundings); depth mix: each block's input is a softmax over the embedding and completed 12-layer blocks | layer 0 dense (situ); MoE: sigmoid + bias, top-16 of 896, latent 3584, situ; 2 shared experts as one MLP | - | MLA latent [ctx, 576] bf16 plus KDA state, contiguous | experts MXFP4 as released; rest lossless `Tiered`; expert cache with device-side plan (`dc_plan`, `dc_fill`) |
| **GLM-5.3-Flash** (`dynamic/glm.rs`, `kern_glm.rs`) | 34 of 45 KDA; 11 MLA (nope 256, v 256) with pooled-key indexer (top 2,048) | none (refused) | mHC: 4 streams, Sinkhorn mixing (20 iterations) | layers 0-2 dense SwiGLU clamp 10; MoE: sigmoid + bias, top-8 of 288, x2.5, 1 shared | 1 MTP head (not loaded; R6 #145) | latent 512 + indexer keys + KDA state, contiguous | FP8 128x128 blocks as released; Int6/5/4 or codebook experts on the hand-written path |
| **DeepSeek-V4** Flash, Pro (`dynamic/deepseek_v4.rs`) | one shared KV head of 512; window 128 ring plus a per-layer compressed cache (ratio 4 with an FP4-simulated indexer, or every entry); learned sink; grouped low-rank output | partial rope 64/512; YaRN on compressed layers | mHC, 4 streams; FP8 quantized product inputs | every layer MoE: sqrt-softplus + bias, hash routing (token id) in the first layers, SwiGLU limit, 1 FP8 shared | MTP parsed, unused | window ring (E4M3-simulated) + compressed caches + fp32 pooling state | FP8 large matrices kept, FP4 experts exact |
| **Muse Spark** 816B-A42B (`dynamic/muse.rs`) | GQA hd 64, sliding (rope) and global (no rope); weightless q/k norm, gated output with per-head norm | full rope on sliding layers only | centred (1+w) RMSNorm; learned residual gates X = aX + bO; sandwich | every layer MoE in a latent, sigmoid + bias, per-expert output norm, no shared | logit softcap with multiplier | bf16 rings | bf16, lossless `Tiered` |
| legacy: Qwen3 dense (`model.rs`) | GQA, hd 128, q/k norm fused with rope (`qk_norm_rope`) | rope, full, no scaling (refused) | RMSNorm pre-norm, residual fused in `rmsnorm_add` | SiLU-gated | tied allowed | paged 12-bit lossless, 16-token pages, 8 rows | `mma`/`mma12` decoded in the library product; long prompts unpack then GEMM |
| Nemotron-3-Super / Nemotron-H (`dynamic/nemotron.rs`; the hand-written engine is retired, `kern_nh.rs` holds the kernel table and weights) | pattern of Mamba-2 (SSD chunk 128, fp32 state [heads,64,128]) and GQA hd 128 | none | RMSNorm, `x + mixer(norm(x))` | MoE: sigmoid + bias, top-22 of 512, renorm, x5; latent 1024 (Super); relu^2, no gate; full-width shared expert | MTP not loaded | bf16 contiguous plus SSM and conv state | Int6/5/4, hb40 codebook; expert cache with CPU misses (`moe_nh.rs:273-319`) |

Recent families first, in bold; legacy rows last.

The M2 standard families need fewer kinds than the seven above (from their `config.json` fields):

| family | attention | position | norm and residual | FFN | parameters to carry |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Llama 3.1-3.3 | GQA hd 128 | rope, `llama3` scaling | RMSNorm pre-norm | SiLU-gated | tied for 1B/3B |
| Qwen2/2.5 | GQA with q/k/v bias | rope, YaRN for long context | RMSNorm | SiLU-gated (Qwen3 MoE: softmax top-8) | bias (refused today) |
| Mistral, Ministral | GQA; sliding in Mistral 7B v0.1, interleaved in Ministral | rope | RMSNorm | SiLU-gated | window, layer pattern |
| Phi-4 | GQA | rope; partial rotary and `longrope` (mini) | RMSNorm | SiLU-gated | partial factor, long/short factors |
| Granite 3.x | GQA | rope | RMSNorm | SiLU-gated | embedding, residual, attention and logits multipliers |
| OLMo 2 | MHA/GQA with q/k norm over all heads | rope | post-norm (norm on the sublayer output) | SiLU-gated | norm placement |
| SmolLM 2/3 | GQA | rope (SmolLM3: no rope every 4th layer) | RMSNorm | SiLU-gated | tied, per-layer rope switch |

### Layer kinds by family

Recent: Q38 Qwen3.8 dense (and legacy Qwen3.5/3.6, same code), CO Qwen3.8-Flash-Next (Condor), G4 gemma-4-12B, KI Kimi-K3, GL GLM-5.3-Flash,
DS DeepSeek-V4, MU Muse Spark. Then STD, the M2 standard families, and the legacy Q3 Qwen3 dense and NH Nemotron-H. "x" uses the kind; a
word names the variant.

| kind | Q38 | CO | G4 | KI | GL | DS | MU | STD | Q3 | NH |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| GQA / MHA / MQA attention | gated | - | global | - | - | - | x | x | x | x |
| sliding-window attention | - | - | x | - | - | window 128 | x | Mistral, Ministral | - | - |
| sparse attention (indexer) | - | x | - | - | on MLA | on compressed | - | - | - | - |
| MLA (latent KV) | - | - | - | x | x | single-head latent | - | - | - | - |
| compressed KV entries | - | - | - | - | pooled keys | x | - | - | - | - |
| Gated DeltaNet | x | x | - | - | - | - | - | - | - | - |
| Kimi delta attention | - | - | - | x | x | - | - | - | - | - |
| Mamba-2 | - | - | - | - | - | - | - | Granite 4 (M3) | - | x |
| rope: full / partial / mrope / scaled | partial | partial, mrope | full, proportional | - | - | partial, YaRN | full | llama3, YaRN, longrope, partial | full | - |
| RMSNorm / (1+w) / LayerNorm | 1+w | 1+w | x | two roundings | x | x | centred 1+w | x (vision: LayerNorm) | x | x |
| norm placement: pre / sandwich / post | pre | pre | sandwich | pre | pre | pre | sandwich | pre; OLMo 2 post | pre | pre |
| residual: add / multi-stream / depth mix / gated | add | 4 streams, gated | add, scalar | depth mix | mHC 4 | mHC 4 | gated a, b | add; Granite scaled | add | add |
| dense MLP (SiLU, GELU, relu^2, situ) | SiLU | - | GELU | situ (layer 0) | SiLU clamp | - | - | SiLU | SiLU | - |
| MoE router: softmax / sigmoid+bias / hash | - | softmax | - | sigmoid | sigmoid | sqrt-softplus, hash | sigmoid | Qwen3 MoE softmax | - | sigmoid |
| shared experts | - | gated | - | x | x | x | - | - | - | x |
| latent MoE | - | - | - | x | - | - | x | - | - | Super |
| MTP / draft head | not loaded | x | draft model | - | x | parsed | - | - | - | not loaded |
| vision tower | not loaded | x | - | - | - | - | - | - | - | - |
| logit softcap / scale | - | - | softcap | - | - | - | softcap | Granite scale | - | - |

Read across, the interface needs four mixers (attention with variants, MLA, linear attention, Mamba-2), two feed-forward kinds (MLP and
MoE with router variants), four residual rules (add, multi-stream mixing, depth mix, gated), and a handful of extras (draft heads,
vision, embedding and head options). The variants that look family-specific today (Condor's gated residuals and GLM's and DeepSeek's mHC
streams, the three indexers) are each one kind with parameters.

## 2. The model step

### What the scheduler hands a step

```rust
/// One sequence's part of a step. Rows are packed: decode rows first, then prompt chunks.
pub struct SeqPart {
    pub seq: SeqId,
    pub tokens: Range<u32>,     // into StepBatch::tokens
    pub past: u32,              // tokens of this sequence already in its cache
    pub blocks: Range<u32>,     // into StepBatch::block_table: this sequence's blocks, in order
    pub state: Option<SlotId>,  // recurrent-state slot (GDN, KDA, Mamba-2 layers), if the model has any
    pub sample: Sample,         // None (a prompt chunk that is not the last), Last (one sample), Each (verify drafts)
}

pub enum Tok { Id(u32), Prev(u32) }   // Prev(r): the token step N-1 sampled for its row r, resolved on the device

pub struct StepBatch<'a> {
    pub tokens: &'a [Tok],
    pub positions: &'a [u32],
    pub parts: &'a [SeqPart],
    pub block_table: &'a [BlockId],
    pub sampling: &'a [SamplingParams], // one per part that samples: temperature, top-k/p, penalties, grammar mask id, seed
    pub kv_ops: &'a [KvOp],             // before the step: copy blocks (fork), clear, state snapshot or restore
}

pub struct StepOut {
    pub tokens: Vec<u32>,               // one per sampling row, in order (k+1 for a row that verified k drafts)
    pub accepted: Vec<u8>,              // drafts accepted per verifying row
    pub logprobs: Option<Vec<Lp>>,      // only rows that asked
    pub lost: Lost,                     // lossless KV blocks the step could not store exactly (kv.rs semantics)
    pub experts: Option<ExpertStats>,   // per MoE layer: hits, copies, host-computed, bytes
}

pub trait ModelStep: Send {
    fn spec(&self) -> &StepSpec;                       // budgets and shapes, below
    fn launch(&mut self, b: &StepBatch) -> Result<Pending>;  // uploads metadata, enqueues the step; never waits on the device
    fn finish(&mut self, p: Pending) -> Result<StepOut>;     // waits for that step only
    fn logits(&mut self, row: u32) -> Result<Vec<f32>>;      // debugging and tests; not on the serving path
}

pub struct StepSpec {
    pub max_tokens: u32,       // tokens a step may hold (the budget's upper bound)
    pub max_rows: u32,         // sequences a step may hold; sized by memory, not a constant
    pub block_tokens: u32,     // 16
    pub groups: Vec<KvGroup>,  // one per layer group with its own page pool (below)
    pub state_slot_bytes: u64, // per sequence, 0 for a model without recurrent layers
    pub vocab: u32,
    pub eos: Vec<u32>,
    pub graph_rows: Vec<u32>,  // row counts with a captured graph (1, 2, 4, ... 64); others run eagerly
}
```

`launch` and `finish` are split so the scheduler's CPU work overlaps the device. At 17 ms a decode step on the 4080 SUPER the host has
the whole step to admit, page, hash and build the next batch; today `model.rs:990` syncs the device every step and the dynamic path syncs
inside `sample` (`dynamic/mod.rs:2402-2536`). `Tok::Prev` is how a decode row names a token the host has not seen yet; the device writes
step N's sampled ids into a ring that step N+1's embedding reads. This is SGLang's overlap scheduler and vLLM's async scheduling. A
stop condition found one step late costs one wasted row-step, which the scheduler discards.

### Where the other parts plug in

| part | in the step | owner |
| :--- | :--- | :--- |
| scheduler | builds `StepBatch`, consumes `StepOut` | `runtime::sched` (R4a), model-agnostic |
| KV manager | `block_table`, `kv_ops`; admission and eviction | `runtime::kv` (R4b, R4c), host side; pages live in the backend's arenas |
| recurrent state | `SeqPart::state`, snapshot/restore in `kv_ops` | the KV manager, slots in the backend's state arena |
| expert tiering | inside the MoE layer's `run`; its stats in `StepOut::experts`; placement changes between steps | `runtime::experts` (R12) |
| sampler | last stage of the step, on the device; grammar masks uploaded by the scheduler | `runtime::sample` (R4a, R5a) |
| drafts (MTP, R6) | a draft head is a layer kind; its proposals are the next step's `Tok`s, verified with `Sample::Each` | the description |

### A step inside the executor

```
upload metadata (positions, cu_seqlens, block table, slots, Prev map) -> one H2D copy from pinned memory
embed (resolves Tok::Prev from the sampled-ids ring)
for each block of the description:
    block.run(cx, &mut residual)      // dyn call per block; fused norm, mixer or FFN, residual rule
head -> logits rows of the sampling parts only (prompt chunks that do not sample skip the head)
sample on the device -> sampled-ids ring, token ids to pinned host memory
record the step's fence
```

The executor captures this as a graph per row-count bucket where the backend supports capture and every block is capturable
(`LayerImpl::capturable`). Prompt chunks vary in length; the executor captures decode-only steps by row bucket and runs mixed steps
eagerly or piecewise (attention outside the graph, products inside), as vLLM's piecewise mode does. A block that must read the router's
choices on the host (an expert store without a device path) splits the graph at that layer; `dynamic/mod.rs:1201` already refuses to
capture such a step.

Built (#329): a decode-only step's rows are padded to 2, 4, 8, ... and replay one graph where every operation acts row by row, is an
attention layer, or is a recurrent layer's convolution or recurrence (Gated DeltaNet, Mamba-2), each row through its own block table and
its own state (a table of the rows' state addresses the host writes per step, as vLLM's state indices; a padding row's entry is 0 and
touches no state). Every other pass (a prompt's chunk, prompts together, decode rows with prompt chunks, decode rows of a description with
no whole graph) runs in pieces (`dynamic/pieces.rs`): the runs of row-wise operations between the attention and recurrent operations are
graphs, the rest eager between them. Pieces are captured for a pass's exact row count, not a padded bucket, because a product's kernel
(the library's route, cuBLAS's algorithm) turns on its rows: a replay is the eager pass bit for bit. The one size captured is a full
pass's (the chunk's rows: the prompt budget fills a mixed step to it), when the scheduler's step is made, as the decode graphs are, and for
descriptions whose decode steps take no whole graph (latent attention, picked attention, DeepSeek-V4) the decode steps' of 2 to 16, 32 and 64
rows; the step's metadata (positions, slots, the state table) is one buffer written in one copy, which every graph reads. A pass of another
size runs eager, and so does a pass whose products take cuBLAS (on the A10, full gemma-4-12B passes in pieces took 0.7% more
GPU time than eager, and whether cuBLAS keeps its eager algorithm in a capture is not measured).

## 3. The layer-kind interface

### Kinds are data

```rust
pub enum Mixer {
    Attention(Attn),     // MHA, GQA, MQA: heads, kv_heads, head_dim, v_head_dim, window, sinks, rope, qk/v norm, output gate,
                         // softcap, bias, scale, K=V; indexer: Option<Indexer> (Condor's block indexer)
    Mla(Mla),            // q_lora, kv_lora, nope/rope/v dims, rope or none, absorbed, output gate; indexer: Option<Indexer> (GLM)
    Compressed(Comp),    // DeepSeek-V4: one latent KV head, a sliding window plus per-layer compressed entries (ratio), indexer, sink
    Linear(Linear),      // Gated DeltaNet, Kimi delta attention: heads, key/value dims, conv kernel, gate form, state dtype
    Mamba2(Mamba2),      // heads, head_dim, state size, groups, conv kernel, chunk
}

pub enum Ffn {
    Mlp(Mlp),            // gated (SiLU, GELU-tanh) or not (relu^2, GELU); bias
    Moe(Moe),            // router (softmax or sigmoid, bias correction, group limits, top-k, norm, scaling), experts,
                         // shared experts, latent projection (Kimi-K3), expert format per tier
}

pub enum Residual {
    Add, AddScaled(f32),       // Granite's residual multiplier
    Gated(GateP),              // Muse: X = a X + b O
    Streams(StreamsP),         // n residual streams mixed per block: mHC with Sinkhorn (GLM, DeepSeek-V4), gated residuals (Condor)
    DepthMix(DepthMixP),       // Kimi-K3: softmax over the embedding and completed blocks of 12 layers
}

pub struct Block {
    pub norm_in: Norm,             // RMSNorm, LayerNorm, (1 + w) variants, eps, f32 or bf16 rounding points
    pub body: Body,                // Mixer or Ffn
    pub norm_out: Option<Norm>,    // sandwich (Gemma), post-norm (OLMo 2)
    pub residual: Residual,
}

pub struct ModelDesc {
    pub embed: Embed,              // scale, tied or not
    pub blocks: Vec<Block>,
    pub final_norm: Norm,
    pub head: Head,                // tied, softcap, scale
    pub mtp: Vec<MtpHead>,         // draft heads, each a small list of blocks plus a projection
    pub vision: Option<VisionDesc>,
    pub rope: Option<Rope>,        // default for kinds that do not set their own: theta, span and angles, pairing, mrope, scaling
}
```

`Rope` covers the scaling variants the standard families need: linear, dynamic NTK, YaRN, `llama3`, `longrope` (R2's list). The
description builder (R3a) makes a `ModelDesc` from `config.json` and the safetensors index with no per-family code for the standard
architectures; families with unusual parts (Kimi-K3's depth mix, DeepSeek-V4's indexer) add a kind, not a pipeline. Partial and
proportional rope are different shapes, not one `partial` number: partial rope (Qwen3.8, 64 of 256) rotates the first 64 dimensions in
pairs (i, i + 32), while Gemma 4's proportional rope on global layers rotates the whole head in pairs (i, i + 256) with only a quarter of the
angles non-zero and exponents over the head dim. `Rope` therefore carries the rotated span, the number of angles with a frequency, the
exponents' denominator and the pairing (`rotate_half` or interleaved, as GLM-5.3's and DeepSeek's latent attention use). A residual
rule also sets the width of the stream a block runs on: with mHC (`Streams`) the block's `x` is n x hidden a token and the block takes
its input from the streams and spreads its output back; `Add`, `Gated` and `AddScaled` keep one stream.

### A backend implements kinds

```rust
pub trait Backend: Send + Sync {
    type Queue; type Fence; type Tensor;                      // the executor is generic over these, compiled once per backend
    fn devices(&self) -> &[DeviceInfo];                       // vendor, arch, memory, unified or not, links
    fn supports(&self, kind: &KindRef, fmt: &Format) -> Support;   // Yes { rate }, Slower(reason), No(reason)
    fn build(&self, layer: &LayerDesc, w: WeightSource, place: &Placement, mem: &mut Arenas<Self>)
        -> Result<Box<dyn LayerImpl<Self>>>;
    fn queue(&self, dev: DeviceId) -> Result<Self::Queue>;
    fn capture(&self, q: &Self::Queue, body: &mut dyn FnMut() -> Result<()>) -> Result<Option<Captured>>; // None: no capture here
    fn collective(&self) -> Option<&dyn Collective>;          // all-reduce, all-gather, all-to-all (M4)
}

pub trait LayerImpl<B: Backend>: Send {
    fn workspace(&self, shape: &StepShape) -> u64;            // bytes of the step arena it needs at this shape
    fn capturable(&self) -> bool;
    fn run(&mut self, cx: &mut StepCx<B>, x: &mut B::Tensor) -> Result<()>;  // enqueue only; never waits on the device
}
```

`StepCx` carries the queue, the step's device-side metadata, the KV views of the layer's group, the state slots, the workspace arena, the
expert store and the profiler marks. `supports` is what the planner (P2) asks before it places a layer: a CUDA backend says `No` for a
format it has no kernel for, the CPU backend says `Slower` for attention at long context; the planner then places that layer elsewhere
or refuses with the reason. This is ggml's `supports_op` at the layer-kind level.

`B::Tensor` is a device pointer with a shape on CUDA, HIP and CPU, and an `mlx_array` on Metal. MLX evaluates lazily: its `run` adds
operations to the graph and the step's `finish` calls `mlx_eval`. The split of `launch` and `finish` exists for this too.

### The CPU f32 reference

`Ref` is a `Backend` whose `Tensor` is `Vec<f32>` and whose kinds are written for clarity: one function per kind, the formula from the
model's reference code, rounding points named where the reference rounds (Kimi-K3's RMSNorm rounds twice: `research/kimi-k3.md` §2.1).
Today's references (`qwen35_ref.rs`, `kimi_ref.rs`, `dsv4_ref.rs`, `muse_ref.rs`, `moe_ref.rs`) do not become kinds of it: they
are whole-model emulations of vLLM's or transformers' bf16 pipelines (a rounding after nearly every operation), which is what R7's
random-model logits need until each family moves. `Ref` is f32 with rounding only where a kind names it (norms); each kind is tested
against the per-family reference with a bf16 tolerance and against transformers in float32, and the old reference goes with its
family's pipeline (`glyd-engine/src/runtime/reference`).

- Per kind: random weights at small shapes, the backend's kind against `Ref`'s, a tolerance per kind and format written next to the
  kind (bf16 accumulation order differs; lossless formats must match the bf16 kernel bit for bit before the tolerance applies). On
  CUDA this is `glyd-engine kindcheck` (`dynamic/kindcheck.rs`): every layer of each recent family's random model, its bodies on
  the GPU's own inputs and the residual's operations between them, against `Ref`; the bounds and the worst errors an RTX 4080 SUPER
  measured are the table in `runtime/reference/mod.rs`. The attention's rounding points are a parameter of the kind
  (`AttnRounding`: f32, FlashAttention's bf16 probabilities, transformers' eager bf16 scores), so the reference rounds where the
  family's engine does.
- Per model: tiny random models built from each family's real `config.json` with reduced widths (the `hubslice` approach), whole
  logits against `Ref` (R7).
- Per checkpoint: the R7 digest of a real model's logits on fixed prompts, against the previous release and the old pipeline.

## 4. The backend seam

### Four backends, one contract

| | CUDA | HIP | Metal (MLX) | CPU |
| :--- | :--- | :--- | :--- | :--- |
| `Queue` | stream | stream | MLX stream | ordered task queue on a worker pool |
| `Fence` | event | event | `mlx_eval` completion | a counter |
| capture | CUDA graph | HIP graph | `mx.compile` (fixed shapes) | none |
| memory | device arenas, pinned host | device arenas, pinned host | unified: one pool, host-visible | host |
| kernels | `kernels/*.cu`, `gpu/glyd_gpu.cu` (R17 folds it in) | the same sources, `gpu/hip.h` | MLX ops; Glyd's formats as custom Metal kernels (`mlx_fast_metal_kernel`, V3b) | Rust SIMD (`cpu_*.rs`) |
| libraries | cuBLASLt where present; FlashInfer, CUTLASS and others where faster (V8a, M2) | hipBLASLt, aiter, composable_kernel (V8b, M5b) | MLX | none |

Nothing in the contract assumes a separate device memory, a launch with a grid, or a graph: those are inside the CUDA and HIP
backends. A unified-memory machine (Apple, DGX Spark, GH200, Strix Halo) reports `unified` and the planner counts its memory once; the
expert store's RAM tier and the GPU tier are then the same memory, and a "miss" is a read at that memory's rate.

### Under CUDA and HIP: `Device`

R1a (#134, CUDA in M2), R1b (#446, HIP in M5b) and R15 carry the detail; the design fixes these points.

- **Bound to a device.** `Buf` holds its `DeviceId`; the 56 process-wide globals (R16) become the runtime context. Two models in one
  process each own their arenas and queues.
- **Arenas, not per-call allocation.** The plan sizes weights, KV pages, state slots, the step workspace and host staging at load.
  A step takes workspace from a bump arena that resets when the step's fence passes, so no buffer can be freed while queued work
  reads it, which is what makes the safe API sound without reference counts per launch (R15 lists the unsound functions).
- **Typed launches.** Each kernel takes one argument struct by value. The struct is defined once in Rust (`#[repr(C)]`, `DevPtr<T>`
  fields), and `build.rs` writes `kernels/args.h` that the CUDA and HIP sources include. CUDA and HIP pass kernel parameters up to
  4 KB by value, more than any of today's signatures need; the 255 untyped call sites become one generated function each. Metal gets
  the same structs as argument buffers.
- **Graphs per bucket.** Captured once per row bucket at load; replay costs ~0.5 µs a launch (`docs/ceiling.md`, measured on the 4080).
- **Errors.** Every call returns a typed error carrying the status and the call; a sticky device error stops the model and the
  service restarts it (R16).

### Why dynamic dispatch at the block and nothing generic above it

A gemma-4-12B decode step on the 4080 is a few hundred kernels in one graph; a dyn call per block (48 x 2 per step) costs well under a
microsecond against a 9.6 ms step, and nothing at replay. Generics over the backend in the scheduler, server and planner would multiply
compile time by the backend count and buy no measurable time. #134's "static dispatch on hot paths" holds where it matters: within a
CUDA kind every launch is a direct call on `Device`.

## 5. The scheduler and the KV cache

### What exists

`sched.rs` already does most of the right things for one model: it admits sequences, finds a prompt's pages by a chain of page keys
(`kv::chain_key`, `Pager::lookup`), runs prompt chunks, decodes up to `MAX_ROWS = 8` rows, preempts the newest sequence by recompute
when pages run out, and retries sequences whose lossless blocks were lost (`MAX_TRIES = 6`). Its 20 tests run on a `Fake` backend. Two
things tie it to Qwen3: it imports `Row`, `Sample`, `Reply`, `Finish` and `MAX_ROWS` from `model.rs`, and its round is "one prompt pass,
then one decode step". R4a moves it into `runtime::sched` with its own types and the `ModelStep` above, and keeps the tests.

### A round is one step

The only batched path today is the legacy Qwen3 one, and its 4080 SUPER records of 2026-11-20 measured what the round structure costs
(Qwen3-8B, `benchmarks/gpu/rtx4080s-batched-prefill-2026-11-20`, `rtx4080s-bar-2026-11-20`). The recent families get the rule these
measurements lead to, not the old round:

- A short prompt's pass takes 19-20 ms and a decode step 17 ms, both mostly the 15 ms weight read. Eight short prompts one per round
  cost 8 weight passes and the k-th waited about k x 37 ms: first token 162 ms at 8 users against vLLM's 68.
- Running the prompts of up to 8 sequences in one pass took the 8-user first token to 82 ms; skipping the decode step after a short
  pass while prompts were arriving took the 4-user one from 80 to 51 ms.
- A 2,000-token pass is 350-372 ms of products at 88-94 TFLOPS (compute-bound); letting decode wait for up to two such passes while
  prompts are pending took the 8-user first token from 1.640 to 1.604 s with the same decode rate.

Each of those is a special case of one rule: a step holds the decode rows and as many prompt tokens as the budget allows, all in one
pass, so the weights are read once for both. The budget is per step, in tokens:

```
budget = rows_decoding * (1 + drafts)                    -- always in, oldest first
       + prompt tokens up to max(chunk_min, T_itl * peak / flops_per_token - rows_decoding)
```

`T_itl` is the inter-token latency target (a setting, default from the plan: the decode step's own time x 3), `peak / flops_per_token`
the tokens a millisecond the GPU computes (P2c measures it per GPU). On the 4080 SUPER with gemma-4-12B, 23.8 GFLOP a token at ~104
TFLOPS is ~4.4 tokens a millisecond: a 50 ms target admits a ~210-token chunk beside 8 decode rows, and decoding users see one 50 ms
step instead of a ~470 ms stall for a 2,000-token prompt. With no decode rows (a burst of new prompts) the budget is the largest chunk that fits the
workspace, and several prompts share it (batched prompts). This is Sarathi-Serve's stall-free batching; the record above is the
measurement that sets its constants on our kernels.

Two details the records found that the policy keeps:

- **The last bits of a batched prompt pass.** A pass of more than 64 rows takes the products' route its row count gives, so a reply's
  last bits can depend on what else ran (140 of 140 sequences gave the same token, 136 within 0.19 of the logits alone). The step
  therefore carries the route choice: in deterministic mode (#206) every product takes the route of a fixed row tile whatever the step's
  row count. On the description executor the kernels change at 16 rows and at 64 (`glyd-engine roundcheck`, 2026-11-22): in a step of
  16 rows or fewer each row's logits are its own pass's bit for bit, on every family; in a larger one they differ by the rounding of
  bf16 values (gemma-4-12B at wren, a text's tokens: a KL of 0.002 to 0.005 from the row alone on average, at most 0.04), whether the
  other rows are a prompt's chunk or other sequences that decode. What the other sequences hold never changes a row's bits. So the
  round that is one step re-rolls that rounding for a reply whenever a prompt runs beside it; so does decoding beside more than 16
  others, and so does cutting a prompt into other chunks (the same 224 to 376 token prompt alone, in chunks of 7 to 127 against one
  pass: 0.0008 to 0.05, and 0.36 at one place where it ends).
- **Arrival skew.** Requests of a burst arrive over 40-60 ms (tokenizing one after another). The front end tokenizes on its own threads
  (R11) and the scheduler waits at most one decode step's time for more prompts when the queue is non-empty and no row is decoding.

### Launch and finish (built, R4a #138)

`ModelStep::launch` queues a round's step and `finish` returns its samples; between the two the host does the work of the step before.
`Sched::round_one_step` launches step N+1 before it lands step N (`land`: tokens chosen and sent, ends found, slots given back) where the
step `overlaps` and every sequence that samples in step N takes its largest logit (greedy, no log-probabilities). A decode row whose token
step N chose names its head row (`Row::after`, the `Tok::Prev` above); the executor's `step_ids` kernel makes the step's int64 token ids
from the host's tokens and step N's head (`tok`, the argmax), and the library's `mma_rows` looks the embedding's rows up by them on the GPU,
bit for bit the rows `unpack` gives. A step that draws, or asks for log-probabilities, lands before the next is launched: its rows are
read on the host; its candidates (the 64 largest logits a row, `top`) are queued with the step and copied back with its samples
(`ModelStep::draws`), so the draw waits for nothing more. Every step of the scheduler goes through `launch` and `finish` (a model whose
rounds are two passes launches its prompt pass and its decode step each to run at once). A sequence whose token in flight is its last by its length has no row in the next step; one that ends on a stop token
or the end-of-sequence token has had its row in the next step run, and that row's output is not taken.

On the executor (`src/dynamic/rows.rs` `Lanes`) a step's inputs are copied on the stream from pinned memory (three lanes: one step
running, the next queued, and the one before for its timings), its samples copied back into the lane, and `finish` waits on the lane's
event, not the stream. A block table a step changes is copied on the stream from the lane too (`upload_tables_queued`): a step in flight
reads the tables as they were, and a block given back is written only by the steps after it. No other wait is left in a decode step
that keeps its blocks; a step that grows a sequence's blocks queues the copy and does not wait either. A recurrent model's states at a
prompt's end (`mark`) and at a block boundary (`name`) are copied as the next step is launched, before its work: the scheduler asks then
for the step in flight's sequences (`Sched::launch`), so the copies are of the states after that step's tokens. The 12-bit pages' status
(the void pages a step made) is copied into the lane with the samples and read from there (`kv12_lost`). The operations that read the
tokens (an n-gram embedding, routing by token) read the int32 ids `step_ids` writes beside the embedding's. The executor overlaps on one
GPU where no layer's experts are in a cache and no layer's matrices are in the computer's memory (both do host work inside the pass);
`GLYD_ENGINE_OVERLAP=0` turns it off. `GLYD_PROF=1` and `GLYD_ENGINE_STEP_PROF=1` time
each step's GPU work and the GPU's idle time before it from events at its first and last work (`Gaps`).

The models that ran one request at a time through a loop of their own (`server::work_gen`) are the scheduler's steps of one slot: stages
and layers across GPUs (`ModelStep for Run`: a prompt in one call, a token a step, the logits read for a draw) and a model with a draft
model (`ModelStep::draft`: a decode step is a drafting round, `spec_round`, each of its tokens sent and added as a step's; a request that
asks for log-probabilities or a least length has plain steps). Condor's own engine (`qwenserve`, `qwen4x_serve.rs`'s `Qx` over
`EngineQx`) is not the executor's and keeps its loop until R3d (#140) moves Condor onto the executor.

### Rows sized by memory

`max_rows` comes from the plan: KV pages and state slots that fit after weights and workspace, not a constant. Graphs are captured at row
buckets 1, 2, 4, ..., up to the bucket past `max_rows`; a step pads its decode rows to the bucket (padding costs no weight reads). The
M2 gate is 64 concurrent requests on the 4080 SUPER and the L4. For gemma-4-12B at wren on the 4080 that is ~7.4 GiB of bf16 KV after
6.28 GiB of weights. A token costs 327,680 bytes in the 40 sliding layers (until the 1,024-token window is full) and 8,192 in the 8
global ones, so 64 users get ~370 tokens each, and a user past the window costs only 8 KB a token more: 21 users of 4,096 tokens fit.
Lossless 12-bit KV for head dims 256 and 512 would add a third.

### Paged blocks for every attention kind

One block is 16 tokens for every kind. What a block holds depends on the layer group:

| layer kind | a block holds | per-sequence extra |
| :--- | :--- | :--- |
| GQA, MHA, MQA | K and V of 16 tokens, every KV head of the group's layers | - |
| sliding window | the same; blocks past the window return to the pool | - |
| MLA | 16 tokens of the latent (kv_lora + rope dims) | - |
| sparse (indexer) | K/V as GQA, or the MLA latent, plus the indexer keys of the block | - |
| compressed entries (DeepSeek-V4) | the window's tokens in the window group; compressed entries in a group of their own, 16 entries a block (16 x ratio tokens), addressed by table index / ratio | the pooling state of the unfinished entry |
| GDN, KDA, Mamba-2 | nothing | one state slot (and the conv tail) |
| cross-layer shared KV (Gemma "K=V", KV sharing) | the owning layer's block | - |

Layers whose blocks have the same bytes share a pool; groups whose block bytes differ get pools of their own, and one block id addresses
the same tokens in every pool, so a sequence has one block table (vLLM's hybrid KV manager does the same with its KV cache groups). A
model with only sliding and full layers (Gemma 4) holds per sequence window/16 blocks in the sliding group and context/16 in the full one.
A prompt's room is taken at once as the scheduler lets it in: a full group's blocks to its end and, in a sliding group, the blocks its
first pass reads, as many as the window and a chunk at its end take; the passes after take theirs as the window moves and give back what
it leaves (`Pages::grow_from`). So one sequence alone needs no more of a sliding group than its window and a chunk, which is what the pool
of one user holds. A sequence alone that the step still has no room for ends with an error in words; sent back to wait, it would only be
let in again to the same end.

What is built for the latent kind (`Op::Mla`, `dynamic/mla.rs`): a latent attention layer's cache is a ring of keys alone (`Ring::values`
false: no room in the value pool), kv_lora + rope values a token, and GLM-5.3's indexer a second one of [normed key | gate score] a token.
A pool of `kpool` tokens is keyed when its last token is written, and its key is written over that token's key (only the pooling read it,
in the same call), so a block that a name or another sequence holds is never written. Both rings sit in the full group's pools, so these
models take the same page budget, swap (R4d) and prefix sharing (R4c, at the marks of their delta-attention states) as the dense kinds.
The indexer's pick is one operation and kernel for GLM-5.3's pools and DeepSeek-V4's compressed entries (`IdxPick`, `idx_pick`).

What is built for the compressed kind (`Op::CompKv`, `Compress`, `IdxPick`, `CompAttn`, `dynamic/comp.rs`): DeepSeek-V4's window is a
ring of keys alone in the window's sliding group, and its compressed entries, the attention's and the indexer's, are rings too
(`Ring::entries`: a row an entry of `ratio` tokens). A group is the rings of one window and one ratio; a compressed group's blocks are
16 entries, so a block covers 16 x ratio tokens (64 for the ratio-4 layers, 2,048 for the ratio-128 ones), and its table entry b holds
the entries of tokens 16 ratio b on (`Pages::covering`). The allocator asks every question in tokens and each group answers it in its
own blocks: a sequence of n tokens holds a compressed group's blocks to the one its last token is in; a cut into a block another
sequence holds goes to that block's start; a start is found by name in whole blocks of every group (a compressed block is named for
the 16-token block that ends it), so DeepSeek-V4's starts are found in steps of 2,048 tokens; a share at a mark inside a compressed
block copies that block. The pools' budget gives every group as many tokens (`pool_blocks`). The kernels read and write the window and
the entries through the sequence's block table (`v4_kv_prep`, `v4_comp_out`, `v4_attn_scores`, `v4_attn_out`, `idx_score_bf16`). Its
states are its compressors' open windows alone (fp32 rows and scores of one or two windows a layer), so it keeps marks too, and the page
budget, preemption, swap and prefix sharing apply to it as to every paged model.

### Compressed KV

Glyd's lossless KV stores a 12-bit value per bf16 (per-model tables; values outside a layer's windows go to an escape pool, and a page
that cannot be held is flagged lost, `kv.rs:1-8`). It reads 25% fewer bytes a token than bf16 and holds a third more tokens. The paged
design keeps it, with two changes:

- **Deterministic capacity (R4b).** Admission reserves each block's worst-case escape chunks (the tables' `worst_chunks`) for the tokens the
  request may write; the pool is the sum of reservations, not a share of the cache. An admitted request never fails because its keys
  compressed worse than expected. If the reservation leaves less than the bf16 cache would hold, those layers store bf16 pages instead:
  the planner picks per layer group the format that holds more tokens.
- **Lost blocks become impossible, not rare.** With reservations, `Lost` remains only as a checked invariant: the kernel still writes the
  status word, and a lost block is a bug reported as an error, not a retry.

Built (R4b, the owner's choice of 2026-11-23: reserve the calibrated need, pause the newest request when the spare pool runs out): the
escape pool is made for the calibrated chunks of every block, so a sequence's blocks reserve its calibrated need as it takes them. Values
that compress worse make a page raw where the chunks run out (a slab of the raw pool for the page), and a page is void only where no slab
is free. So a pass may begin only where every raw pool has a slab for every page it writes, beside what the step's growths and the copies
back in flight have promised (`Dyn::kv12_room`, from the pools' status counts); where not, the named blocks no one holds are dropped, and
then `ModelStep::grow` says no and the scheduler pauses the newest sequence (swapped or recomputed), whose pages are cleared as it gives
them up (`glyd_gpu_kv_release`: its chunks and slabs back). Each raw pool holds at least one sequence's pages and a pass more
(`Lossless::alone_slabs`, counted in the pools' budget), so a sequence alone is never refused for its values. Checked by
`incompressible_values_pause_the_newest_and_fail_no_request` (the scheduler on the made-up model, every page raw) and on the GPU by
`usercheck --kv12` with `GLYD_KV12_SHIFT=60 GLYD_KV12_POOL=1,0` (no step taken back, no request ended). Left: a copy back's pages take
their chunks beside the steps that run meanwhile, which may take them first; such a page is void and its sequence taken back as before.

Today's kernels take head dim 128 and GQA groups of at most 8 (`kv.rs:224-230`), which covers no recent family; widening them to head
dims 256 and 512 and to MLA latents (576 values a token for Kimi-K3) is its own piece of work (#332), and until it lands those families
use bf16 pages. **As built for latents (`dynamic/kvpages.rs`):** a latent attention's ring is one stream (`KV_ONE_STREAM`: the keys' tiles
alone), its row in heads of 128 (Kimi-K3's 512 + 64 as 5 with 64 zero channels, GLM-5.3's 512 as 4); the step's
latents go to staging rows and from there to their pages, and the attention reads a sequence's pages unpacked to bf16 rows (listed on the
device, so a decode step's graph holds it), then runs the bf16 ring's kernel: the same values in the same order, the same bits. Every
project gathers a cached latent into a 16-bit workspace for its decompressed prefill this way; the decode's
reading the 12-bit latent in place (vLLM's Triton MLA reusing the loaded tile as V) is not built, so a latent's
decode reads more bytes than bf16's, not fewer. GLM-5.3's indexer keys stay bf16. Models without tables get bf16 pages; FP8 KV (as released or as a lossy level) is a third page format. The page format is a property of
the pool; attention kinds read pages through the pool's decode, as today's `kv_decode` does.

### Content-addressed blocks and the radix index (R4c)

- A full block's id is `h = H128(parent_h, tokens[16], extra)`, where `extra` holds the `cache_salt`, LoRA id and image hashes. The
  hash becomes public with KV events (D2) and transfer (D1), where a 64-bit collision between tenants is not acceptable, so it is 128
  bits. **As built (`kv::block_hash`, `kv::block_root`):** the first 16 bytes, read little-endian, of SHA-256 over the parent's name
  (16 bytes little-endian) and the block's 16 token ids (4 bytes little-endian each); a scope's chains begin at the hash of the label
  `glyd-kv-1` and the scope's key (the `cache_salt` as the server keys it; LoRA ids and image hashes go in here, or with the block
  they are in, when the scheduler serves them). SHA-256 is in the tree already and costs about a microsecond a block; a faster
  function buys nothing at one hash per 16 tokens. Both caches use it: the dense engine's `Pager` (which still compares a page's
  tokens on a hit) and the description executor's pools (`dynamic/pages.rs`).
- The index maps hash to block and keeps parent and child links, which makes it a radix tree over blocks: a lookup walks a prompt's
  hashes and stops at the first miss (one hash per 16 tokens). Eviction is LRU over leaves with no running reference. SGLang's tree is
  per token; per block costs at most 15 tokens of a match and matches the paging unit.
- Hashes are of tokens, so a block has one name whether it is stored 12-bit, bf16 or in transit. Lost or heavy blocks are not
  published (as today).
- **As built on the description executor:** a pool finds its full blocks by name; a named block no sequence holds stays, with its
  bytes, until the pool needs a block (the oldest first; a sequence gives its blocks up last block first, so a chain loses its end
  before its start). `Pages::adopt` gives an empty sequence the longest start a step can go on from: every block before in a full
  group, the window behind in a sliding one. So a start is found whoever holds it, or no one: a system prompt after the conversation
  that ran it has grown far past it (its sliding layers gave those blocks back) or has ended. A block is never written once named:
  a cut back into one takes its name away. A finished sequence still stays in its slot where one is free (it goes on to the token,
  where names go on to the block), and gives the slot up first.
- Recurrent layers cannot resume from a block of KV: their state must be snapshotted. A snapshot is a slot copy at a block boundary, and
  its size decides the policy: Qwen3.5-9B's GDN state is 50 MB a sequence (24 layers x 32 x 128 x 128 fp32), Kimi-K3's KDA state is 434 MB (69 layers x 96 x 128 x 128
  fp32), the KV of ~16,000 MLA tokens. Snapshots are therefore taken only at the end of a request's prompt and of its reply (where the
  next turn resumes), held in the state arena with LRU, and published with the block they end at. vLLM keeps Mamba state only at
  (large) Mamba block boundaries and optionally where a shared prefix diverges; SGLang's unified radix cache copies the state on a match
  and places checkpoints on a grid (section 8).

### Preemption

When pages run out the scheduler preempts the newest sequence. Two ways to give the pages back:

- **Recompute** (today): free the pages, run the prompt and generated tokens again later. Cost: the tokens' prefill flops.
- **Swap**: copy the sequence's pages (and state slot) to pinned host memory, copy back later. Cost: bytes over PCIe twice. Compressed
  pages make swap 25% cheaper than bf16.

The scheduler chooses per sequence by cost: 2,000 tokens of gemma-4-12B are ~49 TFLOP (~470 ms on the 4080) to recompute against 352 MB
of bf16 KV (the sliding layers hold only the last 1,024 tokens; ~14 ms each way at 25 GB/s). For a GDN or KDA model recompute also
replays the whole prefix through the recurrent layers, so swap wins by more. Swap needs host memory the expert store may want; the plan gives the swap area a size.

Built (R4d, #328): `ModelStep::swap_out` / `swap_in` / `swap_ready` / `swap_drop`, implemented once for the executor's paged sequences
(`dynamic/swap.rs`). A swap copies what a step of the sequence reads: in every layer group the blocks of its tokens its table holds (a
sliding group's window, a shared start's floor: `Pages::held_entries`, `restore`), its recurrent states and their copy at its mark. bf16
rings move as their bytes, by one launch of a copy kernel over the blocks of every layer; 12-bit rings move as their pages are, in their 12
bits: an image of each page (its bytes, the escape chunks it holds, its raw slab: `glyd_gpu_kv_save`), loaded into new pages with chunks
and slabs lent to their new blocks, the directory naming them (`glyd_gpu_kv_load`, C API 21); a page that finds none is void, as one
written is. The host end is pinned memory mapped into the device's addresses: no GPU staging. Out: the copy runs beside the next steps in
two parts, the first `head` blocks of each group (the sequences: the most blocks a decode step takes of a group), then the rest and the
states. The blocks given back keep the part that reads them as a fence (`Pages::release_fenced`); a fenced block is taken after every
free block without one, a swapped sequence's first entries first, and the step that takes one waits for that part alone
(`Dyn::fence_wait`); the slot's states are filled when the slot is used again, after the copy. Back: the copy runs on its own stream
beside the other sequences' steps, and the sequence joins them when it is done. The choice (`Sched::swap_pays`): swap where 2 x bytes / link rate (measured
on the swaps, `plan::LINK_GBS` before) is less than the sequence's tokens / the measured prompt-pass rate. The area: `plan::swap_area`,
`SWAP_SHARE` = 2 x the GPU cache it backs (SGLang's `hicache_ratio` default), within the computer's free memory less `HOST_MARGIN`, read
after the expert store's host tier is pinned; a sequence it has no room for is recomputed. Swapped, a sequence's cache is the same bit for
bit: with prompts one a pass every reply under preemption is the reply alone (`usercheck` with `GLYD_ENGINE_PAGES`).

## 6. Expert tiering for every MoE

### What exists

`offload.rs` is the expert store Condor runs on, and Nemotron-H and Kimi-K3 (`tp_cache.rs`) use parts of it:

- **Slots.** Each MoE layer has GPU slots, `cache GB / ((layers + MTP) x expert bytes)` (`qwen4x_serve.rs:290-295`), replaced LRU among the
  experts the current call does not need (`offload.rs:62-110`); a call needing more experts than slots runs in waves (`offload.rs:241`).
- **Misses.** `Mode` Copy, Cpu, Split or Auto (default Auto, `offload.rs:283-337`): `plan_groups` weighs the copy time against the
  host's time per (token, expert) pair, with smoothed measured costs and the copy stream's backlog (`offload.rs:401-497`). Host compute
  is used for calls of at most `CPU_TOKENS = 64` tokens (`offload.rs:53`).
- **Promotion.** A miss seen again within 16 calls is copied in behind the call, at most 4 a call (`offload.rs:300-312`); over the
  record's traces a window of 16 found as many hits as 32 or 128 with fewer copies.
- **Host and disk.** Condor's host tier is the memory-mapped pack (page cache, `will_need` read-ahead); an O_DIRECT pool with 16 reads
  in flight (`disk.rs`) serves hummingbird and Nemotron-H layers kept on NVMe.
- **Host compute.** AVX2/AVX-512/NEON kernels per format (`cpu_cb.rs`, `cpu_hb.rs`, `cpu_experts.rs`, which skips the ~92% zero relu^2
  activations of Nemotron-H).
- **The graph.** A host-side plan needs the router's choices on the host. Condor runs one graph per stretch between host waits, one host
  wait a layer, polling a mapped flag (`run_parts_mapped`, `offload.rs:2558-2700`); the description executor does not capture a step
  with such a block at all (`dynamic/mod.rs:1201`). Kimi-K3's device path plans on the GPU (`dc_plan`, `dc_fill` in `kimi.cu`) and copies
  every miss, so the step stays one graph.

### The design

The expert store becomes part of the MoE kind for every description (R12). The MoE `LayerImpl` owns a handle to an `ExpertStore`:

```rust
pub struct ExpertStore {
    tiers: Tiers,                 // GPU slots per layer, host (pinned or mapped), NVMe (O_DIRECT), each with its format
    place: Vec<LayerPlace>,       // which experts are where; changed between steps, never during one
    policy: Policy,               // promotion window and rate, miss mode, read-ahead
    costs: Costs,                 // measured per box: copy GB/s, host pair time per format, disk GB/s (P2c)
    stats: Vec<LayerStats>,       // per layer: hits, copies, host pairs, bytes, waits; reported in StepOut
}
```

Per MoE layer in a step:

1. The router runs on the device. Its choices are grouped by expert (`moe_way`): for each expert, the rows routed to it.
2. **Hits** run as one grouped product over the slots.
3. **Misses** are grouped by expert too, so each missed expert's bytes are read once a step for all its rows. For each missed expert
   the plan picks the cheaper of: copy it to a staging slot (bytes / link rate, and it may stay as a promotion) or compute its rows on
   the host (bytes / host memory rate, plus the rows' flops at the CPU's rate and their activations over the link). Host time grows
   with the expert's rows and copy time does not, so with many rows on one expert the copy wins and with a few the host does.
4. Host results are written to mapped memory and joined on the device after an event; GPU and host work on the same layer at once.

Two plan locations, one interface:

- **Device plan** (Kimi-K3's `dc_plan`): the plan runs on the GPU from a placement table the host updates between steps; misses are
  copied by the GPU (mapped host memory) or computed by a host thread that polls a mapped ring. The step stays in one graph. This is
  the default where the link is fast enough to make copies the common choice (PCIe 4/5 x16 per GPU, NVLink-C2C).
- **Host plan** (Condor's today): the host reads the choices per layer and plans with full costs. One wait a layer, which at 48 layers
  and ~10-20 µs a wait is 0.5-1 ms a step; kept for consumer cards where host compute is the common choice.

What changes for batching: with B rows a step touches far more experts than one row does. Under uniform routing Kimi-K3 (16 of 896)
touches 120 experts a layer at 8 rows and 613 at 64, against 16 at one row: 194 GB and 990 GB of experts a step against 26 GB. Real
routing is skewed, so fewer, but the point stands: on a box where the experts do not fit, multi-user decode is bounded by the host's
memory rate over the missed experts of a step, and the store must read each missed expert once per step, not once per row. The
scheduler cannot do much here (routes are not known before the step); the levers are the store's grouping and the planner's choice
of a level that fits more experts.

Placement uses per-layer statistics, which the ceiling formula needs (`docs/ceiling.md`: "the per-layer statistics come first"):
`LayerStats` per step, a hit-rate model in the planner calibrated from them, and `dcreplay` to compare the policy to an ideal one
offline.

What is built (R12b, #327):

- **`LayerStats`** (`offload.rs`): each store layer's calls, rows, pairs, experts asked for and found, misses copied and computed
  by the host, the host's pairs, the bytes read for the misses (over the link, by the host's kernels), copies behind the calls, and
  the copy stream's, the host's and the wait's times. The engine ends a step's statistics after each step (`end_step`);
  `ModelStep::expert_stats` returns the last step's and `/stats` every layer's since the process started (`"experts"`).
- **One read per missed expert a step.** A step of B rows is one call a layer; its misses are its distinct experts that have no
  place, each copied once (waves never copy an expert twice). The host's FP4 kernel (`cpu_fp4.rs`: Kimi-K3's experts and
  DeepSeek-V4's release experts, which now have a host half) and the hummingbird places' of every mixture of experts (`cpu_hbg.rs`)
  take an expert's pairs together: its rows are decoded once for every row routed to it, each pair's sums the bits it has alone.
  The legacy route's host kernels (GLM's codebook rows `cpu_cb.rs`, Nemotron-H's `cpu_hb.rs` and `cpu_experts.rs`) still read an
  expert once a pair, and the statistics count their bytes so.
- **The copy-or-host choice by rows.** `Costs` prices a miss on the host as its expert's first row (`pair`: its bytes read, then a
  row's products) plus each further row (`row`: the products alone), both measured from the host's calls by a least-squares step,
  so a miss of many rows goes to a copy where its rows' products outweigh one. A call whose misses outnumber the places its hits leave is
  copied whole in waves where the host's share would take longer (`waves_quicker`, `Auto` only).
- **`storebench`** runs batched decode through the cache and writes each count of rows' statistics; `ceiling --stats` predicts the
  step from them (`docs/ceiling.md`: DeepSeek-V4 and Kimi-K3 at the release's widths, 18 GB each on the 4080 SUPER, at 1, 8 and 32 rows).
- **The plan on the GPU** (`devplan.rs`, kernels/moe.cu `xp_plan`/`xp_fill`/`xp_wait`/`xp_take`; `Offload::device`, a development
  build's `--device-plan`, `tiercheck`/`storebench --device`): a layer whose host half is FP4 experts (DeepSeek-V4's, Kimi-K3's) keeps
  its places in a table on the GPU; a decode call's plan, its misses' copies (read through the host tier's mapped addresses), the
  products, the promotions behind them and the wait for the host's pairs are kernels on the engine's stream, and the host's pairs
  are a thread of the store's that answers the GPU's requests through mapped host memory (a request's number written last, its
  answer's number read back by `xp_wait`). No host waits inside such a call, so a decode step of them is one graph
  (`MoeLayer::reads_host` is false, and the one-row step is captured). The misses are split as the store's `plan` splits them
  (`Params`: the store's `Costs` and mode): the ones with the most pairs copied, as many as make the longer of the copies and the
  host's pairs shortest, the rest the host's; a hot miss the host keeps is promoted while the host's time leaves room for its copy.
  A copy is priced at the link's time for an expert, measured once at load (the GPU's own copies are not timed). Calls of up to
  `DEV_ROWS` (8) rows are planned on the GPU; more rows are the host's plan, whose waves copy every miss where the misses
  outnumber the places. (The first rule, a miss copied wherever a copy cost less than the host's pairs, ignored that the two run
  at once and copied every miss at 1 and 8 rows: 62-80% slower than the host's plan on an A10, session 5z2.) The store's own plan
  (`LayerCache`) still plans a prompt's chunks: before such a call the GPU's table is read into it, after it written back.
  On the Nemotron 3 Super decode traces its clock finds 0.5 to 0.6 points fewer of the pairs in places than the least-recently-used
  cache at one row (60.7, 72.8 and 79.4% at 17, 32 and 46% of the experts), and at 8 rows 5.1 more to 0.6 fewer.
  Its GPU check (`tiercheck --device`, `storebench --device`) is not run yet.
- **The slot policy and the hit-rate model at many rows**, checked on the Nemotron 3 Super decode traces (20 generations, 1 to 20
  rows side by side): a call's experts are asked for in the order of their pairs, fewest first (`by_rows`), so the busiest stay in
  places across waves (1 to 7 points more pairs found at 16 and 20 rows, never fewer); the planner prices a step's misses as its
  distinct missed experts (`plan::step_misses`: the distinct experts of U rows, E (1 - (1 - k/E)^U), less the share the places
  find, f^s at one row and going to f as the rows' experts fill the places), within 19% of the cache's at every point (it had
  priced one row's misses at any rows: 7.4 experts a layer at 20 rows against 215).

Dense layers in RAM (P2d's last rung before refusing) use the same tiers: a dense layer is a single expert every row routes to, so
the store's copy-or-compute choice applies to it unchanged.

Hummingbird's experts are two groups a layer, one a bit width, each with its own slot size and host cost: `hummingbird.md`.

## 7. Migration plan

### The gates every move passes

- **R7 logits test** (#146): greedy tokens identical to the old path on the fixed prompt set; logits within the kind's tolerance; the
  tiny-model test against `Ref`. For a lossless level, the new path matches the old bit for bit wherever the kernels are the same
  ones in the same order, and where they differ the difference is explained in the PR.
- **R8 bench gate** (#133): `glyd bench` on the family's reference GPU at 1, 4 and 8 users (and 64 where the gate asks), decode tokens/s,
  first token and peak memory; no regression beyond the measured noise band (two standard deviations of five runs, the 3% that
  `scripts/gpu_bar.py` uses where no band is measured yet).
- **R10 packs** (#231): every published pack of the family since v0.29 loads and matches its digest.
- **Deletion.** The old pipeline stays selectable (`GLYD_*_DYNAMIC=0` style) only until the parity record is merged; the next PR deletes
  it, its kernels that nothing else uses, its switch and its per-path sizing (P2f). No old path survives a release.

### Order

Recent families first; the legacy pipelines move last, at parity, so they can be deleted.

| step | what moves | issues | why here | deleted after |
| :--- | :--- | :--- | :--- | :--- |
| 0 | spikes: one kind (RMSNorm + GQA attention) on CUDA, `Ref` and MLX through `mlx-c` | #230 | proves the seam before anything depends on it | - |
| 1 | `sched::Backend` renamed `ModelStep`, scheduler types moved out of `model.rs`; `model::Engine` implements it unchanged | R4a #138 | no behaviour change; every later step uses the new names | - |
| 2 | runtime context, typed errors, arenas, typed launches for the kernels the executor uses | R16 #237, R15 #236, R1a #134 | mechanical, benched per PR, needed by the executor | 56 globals, untyped launches |
| 3 | layer-kind library and `Ref`; CUDA kinds wrap today's kernels (no new kernels) | R2 #135 | the vocabulary the descriptions need | the family-named `Op` variants, as kinds replace them |
| 4 | the executor implements `ModelStep` (one sequence first); `work_gen` replaced (done: drafted and multi-GPU models are steps of one slot) | R4a #138 | every description reaches the scheduler | `work_gen` (deleted) |
| 5 | **gemma-4-12B and Qwen3.8 dense**: descriptions from `config.json`, batched rows, paged KV (sliding ring, GQA, GDN state slots), graphs per row bucket; Qwen3.8 dense gains its lossless level | R3a #136, R4b #141, #329; Qwen3.8 penguin (#331) | the recent dense models; the first batched paths besides legacy Qwen3 | - |
| 6 | lossless KV for head dims 256/512 and MLA latents | #332 | the 12-bit format reaches the recent families | - |
| 7 | **DeepSeek-V4, Kimi-K3, GLM-5.3-Flash, Muse Spark**: batched rows, paged MLA latent and compressed-entry groups, KDA state slots, expert tiering on the MoE kind | R12 #232, #327, R13 #234 | the large recent families; single-user parity on their reference boxes first, then rows | `tp_cache.rs` special cases, the GLM development engine |
| 8 | content-addressed blocks and the radix index; state snapshots for GDN/KDA | R4c #142 | needs paged KV on the recent families first | `Pager`'s 64-bit chain |
| 9 | **Qwen3.8-Flash-Next (Condor)** | R3d #140 | the shipping consumer path; within 3% of v0.33.0 | `model_qwen4x.rs`, `Qx`, `qwen4x_serve.rs` sizing |
| 10 | recent releases not loaded yet: GLM-5.3, Qwen3.8-2.4T-A95B, Kimi-K3.1-Pro, DeepSeek-V4.1, DSpark drafters | #333 | descriptions on the kinds steps 5-9 built | - |
| 11 | the standard families: Llama 3, Mistral, Qwen 2/2.5, Phi-4, Granite, OLMo, SmolLM (the M2 gate's list) | F1a #166, F1b #167, F1d #169, F1e #170, F1h #173 | one GQA + SwiGLU kind with parameters; proves "config only" | the vLLM plugin path for them (F7, later) |
| 12 | legacy at parity: Qwen3.5/3.6 and gemma-4-31B (same code as step 5), Nemotron-3-Super/Nemotron-H, Qwen3 dense with its lossless paged KV | R3b #137, R3c #139 | pipelines deleted; no optimisation beyond parity | `model35.rs`, `model_gemma4.rs`, `model_nh.rs`, `model.rs` |

Steps 5 and 6 run in parallel once step 4 has landed, and step 7 can start with 5. The legacy Qwen3 path served until step 12;
its measured numbers (59.9 tok/s at 1 user, 452.7 at 8 on the 4080 SUPER) are the parity bar there and the reference for what batching
must give the recent families in step 5. Step 12's Qwen3 part is R3c (#139): Qwen3 dense, and every config the description builder's
`llama` reading takes, lowered to the shared operations (`lower::dense`) and served by the scheduler with paged keys and values;
`model.rs` and what only it used are deleted. Its parity against that bar is the GPU check's to measure.

## 8. Prior art

Sources read on 2026-11-20 at each project's main branch; several changed shape this year, noted where it matters.

| piece | vLLM | SGLang | llama.cpp / ggml | TensorRT-LLM | MLX / mlx-lm | Glyd (this design) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| step shape | one token budget, no prefill/decode split | `ForwardMode` EXTEND / DECODE / MIXED | logical batch split into ubatches | context then generation requests, packed | per-request generate; `BatchGenerator` | one flat batch, decode rows first |
| overlap | async scheduling with placeholder tokens, depth 2 | overlap event loop, on by default | - | overlap scheduler, on by default | lazy graph | `launch`/`finish`, `Tok::Prev` |
| KV unit | block, one page size across groups | token (page size 1 default) radix | cells in a unified buffer | power-of-two blocks, pool per window and heads | contiguous, grows by 256 | 16-token block, pool per byte size |
| prefix reuse | hash chain (SHA-256 default) over blocks | radix tree | prompt cache, `--cache-reuse` | radix tree over blocks, priority LRU | prompt trie, LRU per role | 128-bit hash chain + block radix |
| recurrent state | state at Mamba block boundaries | unified radix with state copy-on-write | `llama_memory_recurrent`, checkpoints | KV manager V2 for hybrids | `ArraysCache` | slots; snapshots at turn ends |
| preemption | recompute | retract (recompute) | - | evicted blocks to host | - | recompute or swap by cost |
| vendor seam | attention backend + metadata builder | attention backend | `ggml_backend_i` + `supports_op`, graph split across backends | attention backend + metadata | the array library | layer kind + `supports` |
| MoE offload | - | KTransformers kernels | `-ot`, `--cpu-moe`, `--moe-cache-mib` | wide-EP load balancer | - | expert store, copy or host compute |
| graphs | full for decode, piecewise for mixed | per batch size | backend graph plans | padded to captured sizes | `mx.compile` | per row bucket; piecewise for mixed |

**vLLM.** The V1 scheduler gives each request tokens so that its computed count catches up with its known count, which covers chunked
prefill, prefix caching and speculative decoding with one rule under one budget, `max_num_batched_tokens`
([scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)). Preemption frees the blocks and
recomputes. Async scheduling runs step N+1's scheduling while step N runs, with placeholder tokens
([async_scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/async_scheduler.py)). Model Runner V2 (V1
was removed on 2026-11-07) gives each request a permanent row, gathers the step's inputs on the GPU and has no CPU sync points
([model_runner_v2.md](https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md)). Blocks are hashed as
(parent hash, tokens, extra keys: multimodal, LoRA, `cache_salt`), SHA-256 by default, free blocks kept in an LRU list
([kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py),
[block_pool.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)). The hybrid manager splits layers into
KV cache groups that share one page size, with a manager per attention type
([hybrid_kv_cache_manager.md](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)). Attention
backends declare which graph modes they support; the default captures full graphs for uniform decode and piecewise graphs for mixed
batches ([cuda_graphs.md](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md)). KV events are `BlockStored`,
`BlockRemoved`, `AllBlocksCleared` ([kv_events.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_events.py)).
*Taken:* the single token budget, async scheduling, the hash chain with extra keys, KV groups by page size, piecewise graphs for mixed
steps, the event names. *Different:* the hash is over tokens and 128-bit (no pickle); pools are per byte size because 12-bit,
bf16 and FP8 pages coexist; preemption may swap, because a compressed page is cheap to move and recompute is costly for recurrent models.

**SGLang.** RadixAttention keeps a token-level radix tree with LRU eviction and per-node locks
([radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py),
[paper](https://arxiv.org/abs/2312.07104)). Hybrid models moved to one `UnifiedRadixCache` with a component per kind (full, SWA,
Mamba) and copy-on-write of Mamba state on a match (the separate Mamba and SWA radix caches were removed on 2026-09-20,
[#40313](https://github.com/sgl-project/sglang/pull/40313);
[components](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md)).
The overlap event loop processes step N's results while step N+1 runs ([scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)).
*Taken:* one tree for every layer kind with per-kind state handling, the overlap loop. *Different:* the tree's unit is the 16-token block,
which is the paging unit and costs at most 15 tokens of a match; mixing prefill and decode is the default, not an option.

**llama.cpp / ggml.** A backend is a stream interface (`ggml_backend_i`: async tensor copies, graph compute, events) plus a device
interface with `supports_op`, buffer types and host (pinned) buffers
([ggml-backend-impl.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-impl.h)). `ggml_backend_sched` assigns
graph nodes to backends by support and by where the weights live, and copies at the splits
([ggml-backend.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h)). The memory interface has KV, recurrent
and hybrid implementations ([llama-memory.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory.h)). MoE experts go to the
CPU with `-ot` or `--cpu-moe`, and since 2026-10-07 a GPU LRU expert cache (`--moe-cache-mib`) uploads only misses for batches of up to 32
tokens: 1.62x decode on an RTX 4090 at 77% hits and 2.20x on an RTX 5090 at 89% for Qwen3.8-Flash-Next Q4_0
([#29887](https://github.com/ggml-org/llama.cpp/pull/29887)); a CPU share of misses and an LRFU policy followed
([#30966](https://github.com/ggml-org/llama.cpp/pull/30966), [#31233](https://github.com/ggml-org/llama.cpp/pull/31233)).
*Taken:* `supports` as the planner's question, host buffer types for pinned memory, mixing backends at a boundary for experts.
*Different:* the seam is the layer kind, not the operator: Glyd's formats need fused per-format product kernels on every backend,
and a per-operator graph would lose the fusions that set the fixed time (`docs/ceiling.md`). llama.cpp's expert cache is the closest
thing to Condor's; R12 compares the two same-bits, same-GPU.

**TensorRT-LLM.** The PyTorch backend is now the only one; its `PyExecutor` loops fetch, schedule (capacity then micro-batch), forward,
decode, respond, with resource managers that prepare and free per step and an overlap scheduler on by default
([arch_overview.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/torch/arch_overview.md),
[overlap-scheduler.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/overlap-scheduler.md)). Its KV manager (V2 the
default) has a pool per attention window and head count, reuse through a radix tree over blocks, priority-based LRU (0-100), offload of
evicted blocks to host and disk, `cache_salt`, and an event API for routers
([kvcache.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md)). Chunks other than the last are
multiples of the block size ([paged-attention-ifb-scheduler.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md)).
Graphs pad batches to captured sizes. The wide-EP load balancer replicates hot experts across GPUs from routing statistics
([wide_ep](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/wide_ep/ep_load_balancer/README.md)).
*Taken:* chunk lengths in whole blocks, a block-level radix tree with leaf-only eviction, padding decode rows to graph buckets,
routing statistics driving placement. *Different:* placement across memory tiers (VRAM, RAM, NVMe) on one box, not only across GPUs.

**MLX / mlx-lm.** Operations build a lazy graph that `eval` runs; `mx.compile` traces once per shape
([lazy_evaluation.rst](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst),
[compile.rst](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst)). Arrays live in unified memory and an operation
picks its device ([unified_memory.rst](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst)). mlx-lm has one
file per architecture, contiguous and rotating KV caches, quantized KV, `ArraysCache` for recurrent state, and a continuous-batching
`BatchGenerator` ([cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py),
[generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)); `mlx-c` is the C API ([mlx-c](https://github.com/ml-explore/mlx-c)).
*Taken:* the Metal backend is MLX through `mlx-c`, with Glyd's formats as custom kernels. *Different, and why the seam is shaped this way:*
a backend's `run` may only add work (MLX adds to a lazy graph, CUDA enqueues on a stream), and `finish` is where it evaluates; nothing in
the contract hands out raw device pointers or launch grids.

**Rust prior art.** Candle's backend is a large per-operator trait with a closed `Storage` enum and custom ops with per-device forward
functions ([backend.rs](https://github.com/huggingface/candle/blob/main/candle-core/src/backend.rs)); Burn's `Backend` is associated
types plus op traits, composed as decorators, with GPU backends on CubeCL
([base.rs](https://github.com/tracel-ai/burn/blob/main/crates/burn-backend/src/backend/base.rs)). *Taken:* associated types for a backend's
tensor and queue (Burn), compiled once per backend in the executor only. *Different:* the trait's methods are layer kinds, not operators.

**Expert offload.** KTransformers puts experts on the CPU with AMX kernels and defers some experts' results to keep the CPU busy (SOSP'25,
[project page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/),
[repo](https://github.com/kvcache-ai/ktransformers)); expert deferral changes the outputs, so it is not for a lossless level. Fiddler
chooses per expert between computing on the CPU and moving weights to the GPU ([arXiv 2402.07033](https://arxiv.org/abs/2402.07033)), as
`offload.rs` does. MoE-Infinity drives eviction and prefetch from activation traces ([arXiv 2401.14361](https://arxiv.org/abs/2401.14361)).

**Scheduling papers.** Orca introduced iteration-level scheduling ([OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)).
Sarathi-Serve's chunked prefills and stall-free batches are the step policy above ([arXiv 2403.02310](https://arxiv.org/abs/2403.02310)).
DistServe ([arXiv 2401.09670](https://arxiv.org/abs/2401.09670)) and Splitwise ([arXiv 2311.18677](https://arxiv.org/abs/2311.18677))
split prefill and decode across machines; that is D1, after 1.0, and needs nothing from this design beyond content-addressed blocks and
compressed page transfer.

What Glyd does differently because its formats are compressed, in one place:

- Product kernels decode weights in the kernel (lossless 10.75-12 bits, Int4-6, codebooks, FP8/FP4 as released), so kernels are per
  format and the seam must be above them: a layer kind, not an operator.
- KV pages come in more than one byte size (12-bit lossless, bf16, FP8), so pools are per size and admission reserves the lossless
  format's worst case.
- Expert tiers hold different formats per tier (hummingbird bits by use), so the copy-or-compute choice is per expert and per format.
- Content addresses are of tokens, so the same block can be stored, moved and announced in any format.

## 9. Ceilings

The target is the hardware's limit for the step, not another engine. Each reference case below gives the step's bytes and flops, the
time each takes at the hardware's rate, and what the design adds on top. The formulas are `docs/ceiling.md`'s, and each row
shows its arithmetic.

### gemma-4-12B at wren on the RTX 4080 SUPER

Its lossless pack (~18 GiB) does not fit 16 GB, so the 4080 runs the wren level: 6.28 GiB of weights on the GPU. Inputs: 48 layers (40
sliding with 16 query and 8 KV heads of 256, window 1,024; 8 global with one K=V head of 512), hidden 3,840, MLP 15,360, vocab 262,144
tied; 10.9 B body parameters and a 1.0 B head. A token reads 6.74 GB (the tied embedding once, by the head). The card: 702 GB/s measured
copy rate (736 on the spec sheet), ~104 TFLOPS bf16 with fp32 accumulation; wren's 4-bit weights are decoded to bf16 for the tensor
cores, so flops are at the bf16 rate. Flops a token: 23.8 GFLOP, plus attention (0.67 GFLOP a token in the sliding layers once the window
is full, and 0.26 MFLOP per token of context in the global ones). KV a token: 327,680 bytes in the sliding layers (up to the window),
8,192 in the global ones, bf16 (no 12-bit tables for these head dims yet).

| step | bytes (weights + KV) | at 702 GB/s | flops | at 104 TFLOPS | ceiling | measured |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode, 1 row, ~30 tokens of context | 6.74 GB | 9.60 ms | 24 GFLOP | 0.23 ms | 104 tok/s | 11.22 ms, 89 tok/s (86%) |
| decode, 8 rows, 4,096 context | 6.74 + 2.95 GB | 13.81 ms | 0.20 TFLOP | 1.97 ms | 72 tok/s a user, 579 total | not possible today (one sequence) |
| decode, 64 rows, ~370 context | 6.74 + 7.52 GB | 20.32 ms | 1.54 TFLOP | 14.85 ms | 49 tok/s a user, 3,150 total | not possible today (the M2 gate) |
| prompt, 8 x 30 tokens | 6.74 GB | 9.6 ms | 5.7 TFLOP | 55 ms | 55 ms | one prompt at a time |
| prompt, 2,000 tokens | 6.74 GB | 9.6 ms | 49 TFLOP | 473 ms | 473 ms | not recorded |
| prompt, 4,000 tokens | 6.74 GB | 9.6 ms | 100 TFLOP | 959 ms | 959 ms | not recorded |

The one-row measurement is `research/ceiling/gemma4-12b-wren-4080-2026-10-17.txt` (12.47 ms; in git history since the raw-data move,
commit 26faee41) and GLY-312's prefetch change after it (11.32 to 11.22 ms, commit 4f245566). That record's profile: the four products
take 10.18 ms against 8.80 ideal (86% of the copy rate), and norms, attention, prep and the activation 1.42 ms.

- Decode is memory-bound only up to ~42 rows here (6.74 GB / 702 GB/s equals rows x 23.8 GFLOP / 104 TFLOPS), against ~103 for an
  8B model at 10.75 bits: the fewer bits a weight, the sooner compute binds, so at 64 rows the products' tensor rate matters as much
  as their bytes. A tensor path below bf16 for the products is kernel work (V8), not runtime work.
- KV is up to a third of a step's bytes at 8 rows of 4,096 tokens, all bf16, which is why the lossless KV format for head dims 256/512
  is on the list above.
- Batching is where the room is: one row uses 9.6 ms of the card for 104 tokens at most, eight rows 13.8 ms for 579.

Where the runtime spends time per step (target, then how it is held):

| item | target | how |
| :--- | ---: | :--- |
| scheduling (admission, paging, hashing, batch build) | 0 on the device's critical path | overlapped with the previous step (`launch`/`finish`); < 0.2 ms of CPU even at 64 rows |
| metadata upload | ~10 µs | one H2D copy from pinned memory (positions, offsets, block table, `Prev` map) |
| dispatch | < 1 µs | 96 dyn calls a step; none at graph replay |
| launches | ~0.5 µs a kernel in a graph | graphs per row bucket |
| sampling | < 0.1 ms | on the device, over 262,144 logits a row; 4 bytes a row back |
| kernels' fixed time | today 1.6 ms (11.22 - 9.60) | the kernel side's budget, not the runtime's: fusions per kind (the record's ablation: norms 0.45 ms, attention 0.84 at 30 tokens) |

The runtime's own overhead target is under 1% of the ceiling step (0.1 ms here); everything above that is kernel time, measured per
kind by `glyd bench`.

### Kimi-K3 on 8x A100 80 GB and 8x H100 80 GB

Inputs (`research/kimi-k3.md`): 93 layers (69 KDA, 24 MLA), hidden 7,168, 104.2 B active and 2.78 T total parameters. Non-expert
parameters 55.6 B (bf16, held lossless at 68.9% of their bytes: 76.6 GB). Routed experts: 896 a layer in 92 MoE layers, 33.0 M parameters
each in MXFP4 (4.25 bits: 17.5 MB), 1,347 GiB in all; top-16, so 1,472 experts and 25.8 GB a token. The experts do not fit in
8 x 80 GB, so the step has a miss term; the 2026-10-22 run held 574.5 GiB on the GPUs and found 83.8% of its expert reads there
(`research/kimi-k3-a100-2026-10-22b`).

One row (the M2 gate's single-user parity), dense matrices sliced across 8 GPUs:

| term | formula | 8x A100 | 8x H100 |
| :--- | :--- | ---: | ---: |
| weights read from HBM | (76.6 GB + 0.838 x 25.8 GB) / 8 / HBM rate | 6.0 ms (2.04 TB/s) | 3.7 ms (3.35 TB/s) |
| missed experts | 0.162 x 25.8 GB = 4.18 GB / (8 x PCIe all at once + host memory rate) | 10.5 ms (12.3 GB/s measured; 300 GB/s assumed) | 7.0 ms (25 GB/s, 400 GB/s assumed) |
| exchanges | 699 small gathers a token x NVLink latency (~2 µs) | 1.4 ms | 1.4 ms |
| flops | 208 GFLOP / (8 x peak) | 0.08 ms | 0.03 ms |
| KDA state and MLA cache | 434 MB state read and written + 27.6 KB x context | < 0.1 ms | < 0.1 ms |
| ceiling (GPU and host work overlap within a layer) | max(GPU terms, misses) | ~10.5 ms (95 tok/s) | ~7.0 ms (143 tok/s) |

The host memory rates are assumptions to replace with P2c's measurements (Lambda's A100 hosts are two-socket DDR4, H100 hosts DDR5).
The 2026-10-22 run measured 87 ms a token on 8x A100, 12% of this ceiling. Its terms (`kimi-k3-a100-2026-10-22b`): weights and product
starts 22.7 ms, missed experts 42.5 ms (the device path copied every miss; no host compute), exchanges 9.0 ms (12.9 µs a gather), small
kernels 11.8 ms. The design addresses the first two of those directly: the expert store computes misses on the host when that is cheaper
than copying, which takes the miss term from 4.18 GB / 98 GB/s (42.5 ms) toward 4.18 GB / 398 GB/s (10.5 ms) while staying in the
graph; and per-layer statistics let the planner fit placement to the measured routes. Exchanges and small kernels are kernel and
collective work (C2a, the fused layers of `docs/ceiling.md`); the runtime adds nothing to them.

With many rows the same box behaves differently: under uniform routing 64 rows touch 613 of 896 experts a layer, 990 GB a step against
the ~500 GB the GPUs hold, so multi-user throughput is bounded by the host's memory rate over the step's missed experts (~490 GB at
300-400 GB/s, over a second a step). Kimi-K3 serves many users only where its experts fit (8x H200 at a hummingbird level, or B200), or
with expert parallelism across nodes (D4). That is a planner fact, not a runtime limit: the store reads each missed expert once a step.

## 10. Risks and open questions

### Risks

| risk | effect | what we do |
| :--- | :--- | :--- |
| The generic executor loses fusions the hand-written engines have (Qwen3's fused `gemm3f` epilogue, Kimi's six-product KDA) | decode a few percent slower; the bench gate blocks the move | fusions live inside a block's `LayerImpl`, where they are today; a kind may take its neighbour's norm and residual |
| Worst-case escape reservation eats the lossless KV's gain | fewer tokens than bf16 for some models | measure per model in R4b; the planner picks bf16 pages for a group where they hold more |
| A host-planned expert store breaks the step graph | ~10-20 µs a MoE layer on consumer cards | the device plan where copies dominate; the host plan only where host compute is the common case |
| Recurrent-state snapshots are large (434 MB a sequence for Kimi-K3) | prefix reuse costs memory on KDA/GDN models | snapshots at turn ends only, LRU, counted in the plan |
| MLX's lazy graph and our explicit queues disagree on when work runs | the Metal backend is slower than MLX alone | the spike (step 0) runs a kind through `mlx-c` before R2 lands |
| Batched prompt passes change a reply's last bits | replies depend on concurrent load | deterministic mode (#206) fixes the product route; off by default, as today |
| Mixed steps run eagerly or piecewise | launch overhead on mixed steps | decode-only steps (the common case at load) replay full graphs; measure the mixed-step cost in R4a |
| The migration stalls half way with two paths alive | double maintenance | deletion is part of each move's next PR; no old path survives a release |

### Open questions

1. **Experts across 8 GPUs: slices or whole experts?** Today every expert is sliced across GPUs (two gathers per MoE layer, part of 699 a
   token). Expert parallelism (whole experts per GPU, an all-to-all of rows) has fewer exchanges and is how TensorRT-LLM and SGLang serve
   DeepSeek-class models. The step and the store support either; the choice is C2c's (#196), with a measured A/B on Kimi-K3.
2. **Content hash function.** xxh3-128 (fast, not cryptographic) or BLAKE3 truncated to 128 bits (slower, collision-resistant against a
   tenant who chooses tokens). Matters once blocks are shared across tenants or announced to routers; decided in R4c with the event schema.
3. **Draft tokens in the step.** `Sample::Each` verifies k drafts in one row; whether MTP heads run in the same step (Condor today) or in a
   step of their own is R6's (#145) to settle with measurements.
4. **Vision encoders.** An image's embeddings are computed before its prompt chunk; whether the encoder runs in the step or as a separate
   queue on the device is F2a's (#174).
5. **HIP as a build or a run-time choice.** Today `cfg(glyd_hip)`. The seam allows both; P7b's kernel bundles decide.
6. **The spikes.** #230's acceptance asks for one kind on CUDA, CPU and MLX. This document lands the design; the spikes are step 0 of the
   migration and #230 stays open until they run.
