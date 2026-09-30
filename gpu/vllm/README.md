# Glyd in vLLM: `vllm serve MODEL --quantization glyd`

vLLM serves the model with its Linear layers held packed on the GPU, bit for bit, and a mixture of experts' experts
too. Glyd's kernels multiply them from there. vLLM sizes its KV cache after the weights load, so the memory the packs
save becomes KV cache: more requests at once on the same GPU.

```bash
pip install "glyd[vllm]"                                  # vLLM 0.30, and Glyd with its plugin
vllm serve Qwen/Qwen3-8B --quantization glyd              # a bf16 checkpoint, packed as it loads
vllm serve ./qwen3-8b-glyd --quantization glyd            # a glyd save (glyd pack, glyd.save_pretrained), as saved
vllm serve Qwen/Qwen3-8B --quantization glyd --additional-config '{"glyd": {"layout": "mma12"}}'
```

The `glyd` package registers the plugin with vLLM through its `vllm.general_plugins` entry point; nothing else is
needed. It is tested with vLLM 0.30.0, and `glyd[vllm]` pins `vllm>=0.30,<0.31`. With another minor release of vLLM
the entry point logs one line and loads nothing, and `--quantization glyd` stops with why.

## What it does

- **At load.** Each Linear's bf16 weight is held on the meta device. As vLLM's layerwise loading completes a layer,
  its weights are packed on the GPU, so the load peaks at the packs plus one layer. The layout is Glyd's tiered one
  (10.80 bits a weight) or its 12-bit one (12.04). With `verify`, every pack is decoded and compared with its weights.
  - A layer whose checkpoint lacks a piece (a merged qkv's k, say, or an expert's up) is refused, naming the layer and
    the piece. vLLM's bf16 runs such a checkpoint with that piece's memory never written.
  - A model whose packs cannot fit the GPU's free memory is refused before it loads, with the numbers; running out of
    memory while packing says which layer and how much was packed.
- **A glyd save** loads as saved: its packs are the layers' buffers, with no bf16 at any point. Asked for the other
  layout, it is decoded and packed again. A save's packed LM head is decoded into the bf16 weight vLLM's LM head runs
  on. Saves of Qwen3 and Llama models are checked in vLLM; another family's save is refused, naming its bf16 source.
- **Each product** is one op, `glyd::vllm_linear`, which vLLM's torch.compile takes as one node and its CUDA graphs
  capture. The op takes the library's route for the step's tokens on this GPU: the fused kernels (the weights decoded
  in registers, straight into the tensor cores), or, for long prompts where cuBLAS is the faster, each matrix decoded
  and then cuBLAS.
- **A mixture of experts** (a model vLLM runs by its fused MoE layer): each layer's experts are packed as one matrix,
  the experts' stacked.
  - Their products are the library's grouped ones. Each token's choices are sorted by expert on the GPU; gate and up
    run with SiLU applied as their sums are written out; down runs with the router's weights, each token's rows
    added.
  - None of it syncs with the host, so vLLM's CUDA graphs capture it.
  - These stay bf16, with a warning: experts with biases, activations other than SiLU, expert parallelism, and sizes
    off the packs' multiples.
- **Embeddings, the LM head, norms, attention and the KV cache stay vLLM's.**
- **The compile cache.** The options in effect, with a digest of the packs, go into vLLM's `additional_config`, which
  its compile cache is keyed by. Another layout, mode or checkpoint never finds another's compiled graph.

## Options

Each option is read from the first of these that sets it: `--additional-config '{"glyd": {...}}'`, the checkpoint's
`quantization_config` (or `--hf-overrides`'), or the environment (`GLYD_LAYOUT`, `GLYD_EXACT`, `GLYD_VERIFY`). Another
key, or a flag other than true or false (`1`, `true`, `yes`, `on`; `0`, `false`, `no`, `off`), is refused.

| Option | Values | What it does |
| :--- | :--- | :--- |
| `layout` | `auto` (default), `mma`, `mma12` | `auto` takes `best_layout`'s choice for the GPU: the tiered layout on Ada (L4, L40S, RTX 40), for a mixture of experts on an A10 too, and wherever only it fits; else the 12-bit one (A10, A100, H100, GH200). A save loads in its own. |
| `exact` | `false` (default), `true` | Each product's matrix decoded whole, then the GEMM vLLM runs for bf16, so the logits are bf16's bit for bit (below). |
| `verify` | `false` (default), `true` | Every pack decoded at load and compared with its weights bit for bit. A save's packs by glyd.json's sha256, its other tensors too, and a save packed again in the other layout against the save. |

## Exact mode

The fused products sum in another order than cuBLAS's, so logits can differ from bf16's in their last bits. The same
happens between any two GEMM kernels, and within vLLM's own bf16, between CUDA graphs and eager.

With `exact`, each Linear's matrix is decoded into a scratch buffer and multiplied by the GEMM vLLM runs for bf16:
its `UnquantizedLinearMethod`'s, which is F.linear by default, a FlashInfer `--linear-backend`'s where one is asked for,
and the batch-invariant one under `VLLM_BATCH_INVARIANT`. A mixture of experts' layer decodes the experts its tokens are
routed to and runs vLLM's own bf16 MoE kernel.

- **Eager (`--enforce-eager`):** the logits are vLLM's bf16 eager's, bit for bit.
- **Compiled:** in inductor's deterministic mode, the logits are vLLM's compiled bf16's in that mode, bit for bit.
  That is the one mode in which compiled bf16 is itself the same from one run to the next:

  ```bash
  vllm serve MODEL --quantization glyd --additional-config '{"glyd": {"exact": true}}' \
    --compilation-config '{"inductor_compile_config": {"deterministic": true, "combo_kernels": true, "benchmark_combo_kernel": false}}'
  ```

  Otherwise inductor picks some of its kernels' variants by timing them on the GPU, among them the q and k norms and
  rotary embedding before attention. Compiled logits then vary from one process to the next, bf16's too, so exact
  refuses to start compiled without the deterministic mode. It never runs inexact without saying so.
- **Compiled, Linears with biases** (Qwen2.5's q, k and v, for one) are refused: in bf16's graph inductor adds a
  Linear's bias apart from its matmul, rounding before the add, where exact's product adds it in the GEMM. Eager, exact
  gives bf16's bits there too (Qwen2.5-1.5B-Instruct on an L4).
- **`VLLM_BATCH_INVARIANT`** asks for every product's bits not to depend on the batch. The fused kernels are chosen by
  the batch's tokens, so without exact it is refused; with exact the products are vLLM's batch-invariant GEMM on the
  decoded weights.
- **The deterministic mode on its own** makes compiled Glyd, fused too, the same from one run to the next. On an L4
  it cost nothing measurable: Qwen3-8B's tokens/s at 1, 8 and 32 sequences were within 1% of the default's, bf16's
  and Glyd's alike.
- **The cost of exact** is a decode per product.
- **For a mixture of experts,** exact needs vLLM's Triton kernel for bf16's experts, vLLM's pick on the L4. It is
  refused where vLLM picks another, which lays the weights out otherwise.

## Measured

`vllm bench serve`, bf16 against Glyd:

- the same `--gpu-memory-utilization 0.9`;
- servers started warm, on the compile cache their first start filled;
- the random dataset, 1,024 tokens in and 256 out;
- v0.25.1's library.

Low load is 1 request a second, 0.25 on the L4. Saturated is every request sent at once. A ratio or a percentage is
Glyd's against bf16's; for the times, less is better.

| GPU (Glyd's layout) | Model | KV cache | Requests/s, saturated | Low load: first token, each token | Saturated: first token, each token |
| :--- | :--- | ---: | ---: | :--- | :--- |
| L4 (tiered) | Qwen3-8B | 1.89x | 1.33x | +16%, -21% | -25%, +42% |
| A10 (12-bit) | Qwen3-8B | 1.73x | 1.31x | +16%, -21% | -25%, +30% |
| A100 40 GB (12-bit) | Qwen3-8B | 1.14x | 1.19x | +18%, -10% | -32%, -2% |
| A100 40 GB (12-bit) | Qwen3-14B | 1.77x | 1.65x | +18%, -13% | -57%, +4% |
| GH200 (12-bit) | Qwen3-8B | 1.04x | 0.92x | +4%, +1% | +4%, +9% |
| GH200 (12-bit) | Qwen3-32B | 1.66x | 0.88x | +28%, -6% | -52%, +82% |
| 2x RTX A6000, tensor parallel (tiered) | Qwen3-30B-A3B (a mixture of experts) | 1.67x | 0.87x | +30%, -7% | +34%, +28% |

- **Wins:**
  - 1.04-1.89x the KV cache, so more requests at once (1.67x for Qwen3-30B-A3B over two RTX A6000s).
  - 1.19-1.65x the requests a second saturated on the L4, A10 and A100.
  - At saturation, the first token 25-57% sooner there, and 52% sooner on the GH200 with Qwen3-32B.
  - At low load, each token 6-21% sooner, but for Qwen3-8B on the GH200 (1% slower).
- **Losses:**
  - At low load, the first token 4-28% later.
  - On the GH200 at saturation, 0.88-0.92x bf16's requests a second, with Qwen3-32B although bf16 ran short of KV
    cache there. Hopper's gap is not profiled yet.
  - At saturation, each token 30-42% slower on the L4 and A10, where each step carries more requests; within 4% on the
    A100. On the GH200, 9% slower with Qwen3-8B and 82% with Qwen3-32B.
  - Qwen3-30B-A3B over two RTX A6000s: at 1 request a second the same requests a second, each token 7% sooner; from 4
    a second 0.87-0.88x, each token 28-59% later, with the two modes running the same batches. The experts' grouped
    products at those batch sizes are the next work.
- **The L4's pair** ran back to back in one session. An earlier pair, Glyd's run within the hour after bf16's, gave
  1.39x saturated, +17% and -23% at low load, and -28% and +36% saturated (`l4-vllm-m2-2026-09-29`).

Every rate and percentile, and the logs: [L4](../../benchmarks/gpu/l4-vllm-m5-2026-09-30), [2x RTX A6000](../../benchmarks/gpu/vllm-m4-2xa6000-2026-09-30),
[A10](../../benchmarks/gpu/vllm-m3-a10-2026-09-30), [A100](../../benchmarks/gpu/vllm-m3-a100-40gb-2026-09-30),
[GH200](../../benchmarks/gpu/vllm-m3-gh200-2026-09-30).

**Against vLLM's own bf16** (`check_vllm.py`): Qwen3-1.7B, Qwen3-4B-Instruct-2507, Yi-1.5-6B-Chat (the Llama
architecture), granite-3.1-3b-a800m-instruct (a mixture of experts) and Qwen2.5-1.5B-Instruct (Linears with biases) on
an L4, and Qwen3-8B on an L4, A10, A100 and GH200:

- Every pack decodes to its weights bit for bit.
- Every product is within 6.2e-3 (relative) of the same product on its matrix decoded, with the same bits every run.
- The fused tokens are within vLLM's own bf16 noise, bf16 eager's against bf16 with CUDA graphs. Fed the 1,536-token
  continuation bf16 generated, Glyd ranks 0.988-0.997 of its tokens first, and bf16 eager 0.987-0.995.
  - Glyd's share is at or above bf16 eager's for 16 of the 18 model, GPU and layout pairs. The other two are 0.13 and
    0.20 points under: Yi 12-bit on the L4, and Qwen3-8B tiered on the A100.
  - Glyd's mean logprob difference from bf16's is at most 1.01x bf16 eager's.
- Exact mode gives bf16's bits: eager, and compiled in the deterministic mode (refused there for Qwen2.5's biases).
- Over two GPUs (tensor parallel, 2x RTX A6000): Qwen3-8B and granite, every check but one (below); Qwen3-30B-A3B,
  `--brief`, all 5, exact eager bit for bit. The one failed check: exact compiled was refused as it should be, but in
  the workers, where the check did not see Glyd's message.

## Speculative decoding

vLLM's speculation runs on Glyd's packs as on bf16's weights. Qwen3-8B on an L4, one user, greedy, the tiered layout,
in two mixes of five prompts: "edit" (fix, annotate, convert, rewrite or summarize a given text, code or data) and
"chat". The drafts are n-gram (5 tokens) and EAGLE-3 (`RedHatAI/Qwen3-8B-speculator.eagle3`, Apache-2.0, 3 tokens).

| Output tokens/s, one user | Edit | Chat | Draft tokens accepted (edit, chat) |
| :--- | ---: | ---: | :--- |
| bf16 | 16.6 | 16.7 | |
| bf16, n-gram | 27.1 | 16.7 | 40%, 12% |
| bf16, EAGLE-3 (`--gpu-memory-utilization 0.95 --max-num-batched-tokens 2048`) | 43.3 | 30.3 | 73%, 39% |
| Glyd | 21.2 | 21.5 | |
| Glyd, n-gram | 35.5 | 22.1 | 40%, 12% |
| Glyd, EAGLE-3 | 55.0 | 38.9 | 74%, 39% |

- **Speed.** Glyd with EAGLE-3 made 3.3x bf16's tokens a second on the edit mix, and 2.3x on chat. That is 1.27-1.28x
  bf16 with the same draft.
- **Memory.** bf16 with the draft did not fit the L4 at vLLM's defaults: no memory was left for the KV cache.
- **First token.** Glyd's came later on the edit mix's longer prompts (178 against 150 ms) and sooner on chat's (61
  against 82 ms).
- **Exact.** With speculation, exact eager gave bf16 eager's tokens, request for request: 10 of 10 with each draft.
- **Tokens against plain decoding.** A verify step multiplies up to 1 + k tokens at once, and GEMMs and attention round
  by that shape. So speculative decoding's greedy tokens differ from plain decoding's, bf16's too: eager bf16 parted in
  5 of 10 requests with n-gram and 7 of 10 with EAGLE-3. Under `VLLM_BATCH_INVARIANT=1` they are the same: bf16 with
  n-gram, 10 of 10, and Glyd exact with n-gram, 10 of 10.
- **The draft.** Under `--quantization glyd`, vLLM leaves an EAGLE-3 draft bf16. With `"quantization": "glyd"` in
  `--speculative-config` it is packed too, with the same speed and exact's same tokens.

Every run and its log: [benchmarks/gpu/l4-vllm-spec-2026-09-30](../../benchmarks/gpu/l4-vllm-spec-2026-09-30).
`spec_decode.py` runs one configuration; `spec_summary.py` makes the tables.

## Not supported yet

Each of these is refused at start, with a message saying why; none runs wrong. What vLLM's config tells is refused as
the engine builds it, before any worker starts, so over several GPUs too Glyd's message is the error the user sees:

- dual-batch overlap (`--enable-dbo`), whose two streams would share the GPU's done counters;
- LoRA;
- weight offloading (`--cpu-offload-gb`) and sleep mode;
- a glyd save over several GPUs, one with a mixture of experts' packs, or one of a family other than Qwen3's and
  Llama's; their bf16 checkpoints load;
- exact under torch.compile where a packed Linear has a bias (exact eager runs), and fused products under
  `VLLM_BATCH_INVARIANT`.

Tensor parallelism packs each rank's shard (measured over two RTX A6000s above).

## Files here

- `check_vllm.py [MODEL ...]`: the checks against vLLM's bf16.
  - Every pack against its weights.
  - Every layer's product against F.linear on its matrix decoded, and every mixture of experts' layer against its
    experts decoded.
  - Tokens and logprobs against bf16's own noise, and exact mode.
  - The compile cache: a graph for each layout and mode, each loaded again.
  - Flags: `--saves` (saves, as saved and in the other layout), `--quick`, `--brief`, `--tp N`, `--mp` (vLLM's workers
    in processes of their own, as over several GPUs, on one), `--out DIR`.
- `bench_serve.sh [MODEL]`, `bench_summary.py`: `vllm bench serve`, bf16 against Glyd at several rates. `WARM=1`
  notes the cold start and measures warm. The summary adds the GPU's clock and temperature.
- `profile_steps.py [MODEL]`: a step's GPU time by kind of kernel (Glyd's, GEMMs, attention, the rest), bf16 against
  Glyd, at decode steps of B sequences and prompt steps of M tokens.
- `spec_decode.py`, `spec_summary.py`: one user's tokens/s, first token and speculation's acceptance for a configuration
  (bf16 or Glyd, n-gram or EAGLE-3, eager, exact), and the runs' tables and token-for-token comparisons.
- `bindings/python/test_vllm.py`: the plugin's logic that needs no GPU (options, a save's packs by vLLM's layer names,
  the pieces a checkpoint gave, what is refused, the entry point's version rule), with vLLM installed.

## vLLM's internals

The plugin is `bindings/python/glyd/gpu/vllm_plugin.py`, loaded by `vllm_entry.py`, the entry point, which checks
vLLM's version first. Beyond vLLM's plugin entry point and `register_quantization_config`, it relies on these of vLLM
0.30's internals:

- `QuantizationConfig` (`from_config`, `maybe_update_config`, `get_quant_method`).
- `LinearBase` and `LinearMethodBase`; `QKVParallelLinear` and `MergedColumnParallelLinear`, for their loaders' shard ids;
  `UnquantizedLinearMethod`, whose GEMM exact runs.
- `VocabParallelEmbedding` and `UnquantizedEmbeddingMethod`; `ModelWeightParameter` and `set_weight_attrs`.
- The layerwise online processing in `model_loader.reload.layerwise`: `initialize_online_processing`, and
  `get_layerwise_info`'s record of the weights loaded.
- The config through `get_current_vllm_config_or_none`:
  - `additional_config`, the compile cache's key;
  - the parallel, LoRA, offload, model, compilation and scheduler configs;
  - `vllm.envs`: `VLLM_BATCH_INVARIANT` and `VLLM_DISABLE_SHARED_EXPERTS_STREAM`.
- For a mixture of experts (without these, experts stay bf16, with a warning):
  - `RoutedExperts` and its `moe_config`;
  - `OnlineMoEMethodBase`, `FusedMoEQuantConfig.make` and `MoEActivation`;
  - `UnquantizedFusedMoEMethod` (its `unquantized_backend`, `_init_moe_kernel` and `moe_kernel.apply`'s arguments) with
    `UnquantizedMoeBackend`.

It copies none of vLLM's code. Like the rest of Glyd's GPU code it is under the Business Source License 1.1
(`gpu/LICENSE`).
