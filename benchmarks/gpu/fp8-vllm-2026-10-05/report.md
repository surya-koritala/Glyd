# Task 3 report: vLLM FP8 Linear layers through Glyd (prompt path, identity codec)

Status: DONE_WITH_CONCERNS. Worktree `/Users/suryakoritala/glyd-fp8-vllm`, branch `fp8-vllm`, pushed to `private/fp8-vllm` (never `origin`).
The Ada gates pass. Hopper is not verified (an environment error in the one H100 pod, below).

Commits (all `-s`, one line, no AI attribution): abffd276, 433a4f17, ba09f8f6, 39be0d2c, d5527721, 4bfc1eff, 2d5c45fc, 970a0488, 5ddb27f5, edd0f101 (head).
The Ada run that verified the final behaviour (pod 5) ran 970a0488; the two later commits only change docstrings.

## 1. What was built

| file | what |
| :--- | :--- |
| `glyd-gpu-py/glyd_gpu/f8codec.py` (new, 79 lines) | the codec interface, the registry and the one implementation, "identity" |
| `glyd-gpu-py/glyd_gpu/vllm_fp8.py` (new, ~430 lines) | scheme detection and refusals, the wrapped quantization config, `GlydFp8LinearMethod`, the scratch, the op `glyd::f8_linear`, `stats()` |
| `glyd-gpu-py/glyd_gpu/_trace.py` | `fp8_linear_method_apply` (7 lines; the function torch.compile traces has to stay source in the compiled wheel) |
| `glyd-gpu-py/glyd_gpu/vllm_plugin.py` | `GlydConfig.override_quantization_method` (claims an FP8 checkpoint when the user passed `glyd`), `from_config` hands an FP8 dict to `vllm_fp8.config`, `register()` registers the op, `maybe_update_config` refuses a ModelOpt checkpoint whose quantization is in `hf_quant_config.json` alone. Nothing else of the bf16 path changed. |
| `glyd-gpu-py/tests/test_vllm_fp8.py` (new) | 10 no-GPU tests, run against the real vLLM 0.30 code |
| `gpu/fp8/make_slice.py` | few-layer model directory from a real public checkpoint, HTTP range reads, stdlib only (also fetches whole models: `--layers 64`) |
| `gpu/fp8/check_fp8_vllm.py` | stock vs `--quantization glyd` on a model dir: exact, kernel lines, weights, speed; `--diag` for repeated compiles |
| `gpu/fp8/pertensor_slice.py` | requantizes the block-FP8 slice to per-tensor FP8 (same real weights) to reach vLLM's per-tensor path |
| `gpu/fp8/fp8_vllm_job.sh` | the unattended pod job (uv, vLLM 0.30.0, glyd-gpu 0.28.0 wheel for its library, this tree on PYTHONPATH, tests, slices, checks, bf16 smoke) |

## 2. The interface (`f8codec`), and the swap for Task 2

```
p = f8codec.pack(w, codec=None)    # w: contiguous [rows, cols] byte matrix on the GPU (uint8 or float8_e4m3fn); -> Packed
p.codec, p.shape (rows, cols), p.tensors (GPU tensors decode reads), p.meta (ints), p.nbytes()
f8codec.decode(p, out)             # out: contiguous byte matrix of p.shape, made beforehand (the scratch's start); exact bytes of w; capturable
f8codec.register(codec)            # codec: .name, .version, .pack(w) -> Packed, .decode(p, out)
f8codec.DEFAULT = "identity"       # the selection; also $GLYD_F8_CODEC, and --additional-config '{"glyd": {"f8_codec": ...}}'
```

- Swap: Task 2 registers its class and changes `DEFAULT = "identity"` to its name. One line. The codec name and version go into vLLM's additional_config (`f8codec`), which keys the compile cache.
- `identity.pack` keeps `w`'s own storage (no copy). A compressing codec must copy; the weight's old memory is then freed when its Parameter is pointed at the scratch (to verify in Task 2/6 from the "Model loading took" line).
- `decode` runs inside the captured region on fixed pointers: no host sync, no allocation (workspace made in `pack`).
- `Packed.nbytes()` is what `vllm_fp8.stats()` and `check_fp8_vllm.py` report as held.
- Design point for Task 2: the matrix handed to the codec is what vLLM's kernel is fed. On Ada block FP8 and on every MXFP8 that is Marlin's tile-permuted int32 (4 e4m3 per int32, observed `[320, 139264]` int32 for a gate_up); on Hopper block FP8 it is `[N, K]` e4m3 (DeepGEMM-requantized bytes); per-tensor it is the storage of a transposed `[K, N]` view (observed `[5120, 34816]` strides (1, 5120)). A tile-window code was chosen on row-major `[N, K]` bytes; on Marlin's permuted bytes its gain may be smaller. Packing before the repack and repacking at decode is an alternative the codec author should weigh.
- Task 4 (MoE) is not blocked: pack one `Packed` per expert, `decode(p, out_slice)` into a compact scratch; `_Scratch` here is per device and flat, for the dense Linears.

## 3. How it works

- **Claim.** vLLM probes `override_quantization_method` of the custom names first. `GlydConfig` returns `"glyd"` when the user passed `--quantization glyd` and the checkpoint's `quantization_config` is FP8; for a bf16 checkpoint and a glyd save nothing changes (checked against vLLM's own `ModelConfig._verify_quantization` and `VllmConfig._get_quantization_config`, run on the real code on the Mac).
- **Config.** `vllm_fp8.config()` builds vLLM's own class for the checkpoint (the probe vLLM itself would run without `--quantization`: Fp8Config, ModelOptFp8Config, ModelOptMxFp8Config, CompressedTensorsConfig, DeepSeek V4's) and makes the object an instance of a subclass with `GlydFp8Config` in front. So `weight_block_size`, `use_deep_gemm`, `ignored_layers`, `packed_modules_mapping`, KV-cache scales and every `isinstance(quant_config, Fp8Config)` of a model hold. It is picklable (`__reduce__`), as vLLM pickles its config to start the engine. `get_min_capability` is 89.
- **Method.** `get_quant_method` is the inner config's; each FP8 `LinearBase` method (`Fp8LinearMethod`, `ModelOptLinearMethod`, `CompressedTensorsLinearMethod` with a W8A8 FP8 or MXFP8 scheme) is wrapped in `GlydFp8LinearMethod`. Unquantized layers, embeddings, the LM head, attention/KV and MoE methods are returned untouched; Linears under 1 M weights and bmm layers stay vLLM's. `create_weights`, the loaders, `process_weights_after_loading` and the scales are vLLM's. The wrapper is registered for `weight_loader_v2`.
- **Pack.** After the inner `process_weights_after_loading`, `layer.weight` (whatever the kernel will be fed: dtype, shape, strides) is read as the bytes of its dense storage (`_matrix`: contiguous, or the transpose of a contiguous), packed, and the Parameter's `.data` becomes a view of the per-device scratch with the exact shape/strides. A layout it does not know stays vLLM's, with a log line (no such layer was met).
- **Scratch.** One byte buffer per device, grown at load to the largest layer (existing layers re-pointed; during load no graph holds it), counted by vLLM's memory profile (measured: "Model loading took" +0.16 GiB = the scratch; the KV size drops by the same). It cannot grow once the model has run (refused with why).
- **Product.** One op, `glyd::f8_linear(x, bias, layer_id)`: decode into the scratch, then the inner method's own `apply` (which reads `layer.weight`, the scales, the workspace from the layer). Why one op: it makes the order of the scratch's writer and reader independent of what vLLM's torch.compile does with the graph. Fake impl gives the shape for compile. Capturable: verified under vLLM's default FULL_AND_PIECEWISE, CUDA graphs without inductor, compiled, eager.
- **Cost of that choice.** `layer.requires_unquantized_input = True` (no pre-quantized activation from a fused producer), and vLLM's compile fusions around a Linear's input quantization are not applied to these layers. Real consequence below (per-tensor under compile).
- **Refusals at config time** (all with why, all unit-tested): GPU not sm_89/sm_90 (NVML capability; Ampere: "vLLM serves FP8 weight-only; not covered", Blackwell "not measured"); FP8-like quantization_configs not covered (ModelOpt NVFP4/mixed/other algos, FBGEMM FP8, Quark, FP-Quant, PTPC; compressed-tensors with non-8-bit-float weights, W8A16, or mixed FP8/MXFP8 groups); a per-layer scheme that is not FP8; a ModelOpt checkpoint with `hf_quant_config.json` and no `quantization_config` in config.json (the plain `GlydConfig` would be built for it and the bf16 path would load its FP8 weights into bf16 parameters, without their scales); `additional_config["glyd"]` keys of the bf16 path; LoRA, weight offload, sleep mode, DBO/ubatching (they share one scratch across concurrent batches).

## 4. Per-scheme status

| scheme | detected (dict) | inner class | wrapped layout | GPU result |
| :--- | :--- | :--- | :--- | :--- |
| fp8, block 128x128 (Qwen3.8-27B-FP8, GLM-5.3-Flash) | yes | Fp8Config | Ada: Marlin int32 tiles | exact PASS on an RTX 4090: Qwen eager, graph, graphonly; GLM eager, compile |
| fp8, per-tensor | yes | Fp8Config | Ada: CUTLASS, transposed `[K, N]`, shards requantized | eager PASS; compiled with `+quant_fp8` PASS; compiled by default differs (section 6) |
| mxfp8 (MiniMax) | yes | ModelOptMxFp8Config | Marlin int32 (same layout as block FP8 on Ada) | not run on a GPU |
| modelopt FP8 / FP8_PB_WO / per-channel; modelopt MXFP8 | yes | ModelOptFp8Config / ModelOptMxFp8Config | as above | not run on a GPU |
| compressed-tensors W8A8 FP8 (tensor/channel/block), MXFP8 | yes | CompressedTensorsConfig | as above | not run on a GPU |
| DeepSeek V4 FP8 ([32,32] e8m0) | yes (picked by hf_config) | DeepseekV4FP8Config | Marlin MXFP8 | not run |
| block FP8 on Hopper | yes | Fp8Config | `[N, K]` e4m3, DeepGEMM-requantized | NOT VERIFIED (section 5) |

Everything but the first two rows is covered only by the no-GPU tests (detection, instantiation of vLLM's real config classes, wrapping, pickling).

## 5. Tests and GPU results

**No-GPU tests** (`python glyd-gpu-py/tests/test_vllm_fp8.py`; local venv with torch CPU and vLLM 0.30.0's real source; and on every pod): 10 new + the 27 existing tests of `test_vllm.py` = 37 ok. They cover the detection of each scheme from a config dict; the refusals; the probe order against vLLM's real `_verify_quantization`; wrapping of the real `Fp8LinearMethod`/`ModelOptLinearMethod`/`CompressedTensorsLinearMethod` classes; pickling; the layouts (`[N,K]`, transposed, int32 tiles) packed and decoded back through the scratch with the scratch overwritten between products and growing as layers come; `glyd::f8_linear` on CPU and traced by `torch.compile(aot_eager)` with two Linears on one input; the identity codec on all 256 byte patterns incl. NaN bytes and the registry swap. bf16 regression: the 27 old tests pass; a bf16 smoke (`Qwen/Qwen3-0.6B --quantization glyd`, eager) ran OK on pods 1, 2 and 5.

**GPU runs** (RTX 4090 = Ada, vLLM 0.30.0, torch 2.13.0+cu130). Slices: Qwen3.8-27B-FP8 layers 0..3 (81 tensors, 6.16 GiB; 16 FP8 Linears, 1.45 GiB) and GLM-5.3-Flash layers 0..2 (its 3 dense layers: 6 FP8 MLP Linears; the KDA attention is BF16; 3.56 GiB). Final run: `scratchpad/ceiling/results-f8a5/`.

| run | exact (tokens, token logprobs, top-20 prompt logprobs, 4 prompts, 40k values) | weights vLLM logs (stock -> Glyd) |
| :--- | :--- | :--- |
| Qwen4 eager | PASS bit for bit | 6.32 -> 6.48 GiB (+0.16 = scratch 170 MiB) |
| Qwen4 graph (compile, FULL_AND_PIECEWISE, inductor deterministic) | PASS | same |
| Qwen4 graphonly (CUDA graphs, no inductor) | PASS | same |
| GLM3 eager | PASS | 3.58 -> 3.67 (+0.09 = scratch 96 MiB) |
| GLM3 compile | PASS | same |
| Qwen4 per-tensor eager (pod 2) | PASS | 6.29 -> 6.45 |
| Qwen4 per-tensor graphq (compile, `+quant_fp8` in both) | PASS | same |
| Qwen4 per-tensor graph (default compile) | FAIL (differs from stock; stock vs stock again identical) | same |

- Kernel lines are identical in both servers (Marlin on Ada for block FP8, CUTLASS for per-tensor). 16 (Qwen) / 6 (GLM) Linears packed in every run.
- With the identity codec nothing is saved: weights = stock + one scratch. The size gate (>= 15% under) is for the real codec (Tasks 2/6).
- Speed, identity codec (Glyd / stock tokens a second, the codec copies every weight each product): Qwen4 eager prefill 0.99, decode at 1 / 16 / 64 sequences 0.64 / 0.67 / 0.74; graph 0.98, 0.63, 0.66, 0.73. Per-tensor: 0.94-0.97, 0.64, 0.66, 0.71. INFO only: a placeholder cost.
- GLM graph modes: in stock vLLM 0.30 on this Ada GPU GLM's KDA layer fails under any CUDA graph capture (`.tolist()` in `prepare_chunk_indices`, "Cannot copy between CPU and CUDA tensors during CUDA graph capture"), FULL or PIECEWISE, stock and Glyd alike; so GLM was checked eager and compiled without graphs.
- The harness's `--pin` (Triton autotuners keep their first config, `PYTHONHASHSEED=0`) is on in the final run. Triton picked different autotuner configs from one unpinned process to the next (2 to 6 of its 6 autotuners differed in the four comparisons made) without changing a bit, so the pin is for speed (no autotuning) and reproducibility, not a fix.
- One unexplained event: the very first run's `glyd-graph` (pod 1, earlier commit, same op) differed from stock on the long prompt from position 24; stock again was identical. A traced variant (decode op, inner apply traced, x declared mutated; `f8_apply: traced`) gave identical numbers to that run; the op variant matched in every later fresh compile (5 fresh Glyd compiles + 4 cache loads with debug flags: sync, contiguous, clone, weights-as-loaded) while stock matched itself in 8 fresh compiles. The traced variant and its debug flags were removed (a mutation op perturbs inductor's graph; no evidence it can be exact). I could not reproduce the first event; a few more fresh compiles on the final tree would settle whether it was a one-off.

**Hopper: not verified.** The one H100 pod (`results-f8h1/`) failed identically for stock and Glyd: FlashInfer's JIT of `fp8_blockscale_gemm_90` could not link (`ld: cannot find -lnvrtc`: the pip toolkit holds `libnvrtc.so.13` without the `libnvrtc.so` link). Before that: Hopper picks `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` with "DeepGEMM E8M0 enabled", and the Glyd wrapper loaded and packed the DeepGEMM-requantized `[N,K]` weights (Model loading 6.45 GiB vs stock 6.29). I killed the pod when it was clear every remaining item would hit the same error. The fix is committed in `fp8_vllm_job.sh` (links every `lib*.so.N` of the toolkit as `lib*.so`) but untested, because the rule is one H100 run. The whole Qwen3.8-27B-FP8 was not run (the `make_slice.py --layers 64` fetch worked: 27.4 GiB in 6.4 min on that pod). To finish Hopper: `F8_PLAN="qwen4:eager,graph glm3:eager,compile whole:eager,graph"` in `~/f8env`, one H100, about 40 minutes.

## 6. Pods and cost

Runpod, through `rp_run.sh` with `rp_guard.sh` running; each pod deleted by the script; "pods left: []" after each. Five pods, none two at once. Capacity: three attempts (4090, L40S, RTX 6000 Ada) found no instance once and created no pod; the 4090 came back minutes later.

| pod | GPU | minutes | job | results |
| :--- | :--- | ---: | :--- | :--- |
| 1 `f8a1` | RTX 4090 | 15 | Qwen4 eager/graph/graphonly, GLM3 eager (graph modes fail in stock), bf16 smoke | `scratchpad/ceiling/results-f8a1` |
| 2 `f8a2` | RTX 4090 | 23 | `--diag` variants, GLM3, per-tensor eager, bf16 smoke | `results-f8a2` |
| 3 `f8a3` | RTX 4090 | 30 | `--diag` unpinned and pinned (a slow install took 8.6 of the 30 minutes) | `results-f8a3` |
| 4 `f8h1` | H100 80GB HBM3 | 20 | Hopper attempt, FlashInfer JIT link error | `results-f8h1` |
| 5 `f8a5` | RTX 4090 | 16 | the final tree: Qwen4 3 modes, GLM3 eager/compile, per-tensor graph/graphq, bf16 smoke | `results-f8a5` |

Total 104 pod-minutes: 84 on the 4090, 20 on the H100.

## 7. Open issues and concerns

1. **Hopper unverified** (above). The first thing to run once an H100 is allowed.
2. **Per-tensor and per-channel FP8 are not bit for bit under vLLM's default compile.** Their activation quantization is torch ops that inductor fuses into stock's graph; inside the opaque op it runs unfused (the top-20 logprobs differ by up to ~0.09 on the garbage-output slice). Bit for bit in eager, in graphs without inductor, and compiled with `custom_ops=["+quant_fp8"]` (measured); `+quant_fp8` is on by default only for block-scaled weights. The spec's default mode ("within the noise") covers it; the `exact` mode needs this to be told (eager or `+quant_fp8`). Alternative (not built): trace the inner apply in the graph; the one such variant tried was not bit for bit and has an ordering hazard for parallel Linears on one input.
3. The compile fusions around a wrapped Linear's input quantization are lost (`requires_unquantized_input`). Block-scaled Linears on Hopper (FlashInfer takes bf16; DeepGEMM quantizes with a custom op) are less affected. The speed gate has to be measured with the real codec.
4. The codec's input is a kernel-layout byte matrix (Marlin tiles on Ada). Compression gain on those bytes is unmeasured; see section 2.
5. MoE experts, the bf16 parts of an FP8 checkpoint (GLM's attention, `lm_head`, embeddings) and attention stay vLLM's. An FP8 `kv_b_proj` (DeepSeek-V3-style MLA) goes through vLLM's identity-matrix dequantization path (the wrapper is deliberately not an `Fp8LinearMethod`, whose shortcut would read the scratch): untested.
6. A shared scratch cannot be used by two Linears at once: DBO is refused; for MoE (Task 4) the shared experts' side stream (`VLLM_DISABLE_SHARED_EXPERTS_STREAM`) must be dealt with, or those Linears given a scratch of their own.
7. The compiled wheel (`setup.py`, Cython) was not built; `vllm_fp8.py` uses explicit two-argument `super` and no dev blocks; the one function torch.compile traces is in `_trace.py`. A wheel build and `check_wheel.py` are for the release flow.
8. Weights reload paths (`supports_pre_processed_weights`, RL reload, sleep mode) are refused or unsupported, as for the bf16 path.
9. Not run on GPU: mxfp8, modelopt, compressed-tensors, DeepSeek V4, TP > 1, speculative decoding with an FP8 draft.

## 8. Reproduce

```
# no GPU
cd glyd-gpu-py/tests && PYTHONPATH=.. python test_vllm_fp8.py && PYTHONPATH=.. python test_vllm.py      # vLLM 0.30 and torch installed
# a slice, then the check, on a GPU
python gpu/fp8/make_slice.py Qwen/Qwen3.8-27B-FP8 /root/slice-qwen --layers 4
python gpu/fp8/make_slice.py zai-org/GLM-5.3-Flash /root/slice-glm --layers 3
PYTHONPATH=glyd-gpu-py python gpu/fp8/check_fp8_vllm.py /root/slice-qwen --language-model-only --modes eager,graph,graphonly --pin --out out
PYTHONPATH=glyd-gpu-py python gpu/fp8/check_fp8_vllm.py /root/slice-glm --language-model-only --no-bench --modes eager,compile --pin --out out-glm
# a pod: UPLOAD="f8src.tar f8env" bash rp_run.sh gpu/fp8/fp8_vllm_job.sh      (f8src.tar = git archive HEAD glyd-gpu-py gpu/fp8 + COMMIT)
```

---

# Fix round 1

Worktree `glyd-fp8-vllm`, branch `fp8-vllm`, pushed to `private` only. The identity codec is still in; nothing here changes what the codec does. Head: `a8ebabda`. The commits of this round: `1a0ae45d`, `6e7bced1`, `f05874cd`, `b7a4190d`, `8cba0fee`, `6daf0d45`, `3d4a41dc`, `a5840073`, `a8ebabda` (the earlier head was `edd0f101`).

## F1. What changed

| finding | change | where |
| :--- | :--- | :--- |
| Q1 | the `GLYD_F8_CODEC` read is a `# dev:` ... `# end dev` block; the docstring no longer names it. `check_wheel.py`'s env, ident and prose checks on the tree: `env: [] ident: [] prose: {}`; the stripped source ignores the variable | `f8codec.py` (`1a0ae45d`) |
| M10 | `kv_b_proj` is not wrapped (`DIRECT`): vLLM's own `Fp8LinearMethod` keeps its post-load exact dequantization for MLA. Test: an MLA-style config's `kv_b_proj` gets vLLM's method, its siblings Glyd's | `vllm_fp8.py`, `test_vllm_fp8.py::test_wrap` (`6e7bced1`) |
| doc | the experts case is one stacked `[E*O, K]` pack (O a multiple of 64) plus int32 expert ids on the GPU, `decode_experts(p, E, ids, out)`, as `kernels.py`'s `f8_unpack_experts`; the identity codec has it (capturable, no host read) | `f8codec.py` (`1a0ae45d`) |
| S2 | the rule (`exact` on per-tensor or per-channel FP8 is refused under vLLM's default compile unless eager or `custom_ops +quant_fp8`) is in the spec, the plan and the code: `exact` is an option of `--additional-config {"glyd": ...}` (or `GLYD_EXACT`), and the refusal says why (inductor fuses the torch-op quantization into vLLM's graph, so it runs apart inside Glyd's product). Which checkpoints count: fp8-tensor, compressed-tensors with a tensor or channel weight strategy, ModelOpt FP8 | `vllm_fp8.py`, spec, plan (`6e7bced1`) |
| M1 | the op's layer argument is the layer prefix (a string), not a counter; a live collision gets `#n`; the entry is dropped with the layer (weakref) | `vllm_fp8.py` |
| M2 | `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` is set where the model has experts (the shared scratch cannot be used from two streams) | `vllm_fp8.py` |
| M3 | the ModelOpt file is read through `hf_hub_download(..., revision=)`, not by a raw URL | `vllm_fp8.py` |
| M7, M8 | the unused `SCHEMES` is gone; the log line says what it does | `vllm_fp8.py` |
| M6 (part) | new tests: inductor on CPU (besides aot_eager), storage freed with a copying codec, vLLM's real `_verify_quantization` probe, ids by prefix, the poison canary | `test_vllm_fp8.py` (16 tests; 43 ok with the 27 existing) |
| harness | `check_fp8_vllm.py --seeds` (per `PYTHONHASHSEED`: stock and Glyd, fresh compile cache each, plus Glyd with its scratch overwritten after every product and Glyd twice; each comparison printed as it finishes); kernel lines leave out progress bars (the two H100 "FAIL: kernel" lines were DeepGEMM warmup bars); `make_slice.py` reads Mistral's own format (`--weights consolidated`), samples its end check in parallel, and raises after 12 failed tries (it used to return None); `fp8_vllm_job.sh` links every toolkit lib (the `-lnvrtc` fix) and builds FlashInfer's block-scale GEMM once as a Hopper preflight; plan items `seeds`, `minimax3`, `mistral2` | `gpu/fp8/` |

Deliberately left alone: `gpu/fp8/fmt_study.py`, `glyd_gpu/fp8ref.py`, `tests/test_fp8ref.py`.

## F2. GPU results (identity codec)

All with `COMMIT` files in the results directories; logs under `scratchpad/ceiling/`.

| item | result | where |
| :--- | :--- | :--- |
| (a) H100 80GB HBM3, Qwen3.8-27B-FP8 4 layers, eager and default graphs (FULL_AND_PIECEWISE, inductor) | **PASS bit for bit** both (40036 values; stock against itself again identical). Hopper picks `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, "DeepGEMM E8M0 enabled", `[N, K]` e4m3 weights; 16 Linears packed; weights 6.29 -> 6.45 GiB (+0.16 = scratch). Speed, Glyd / stock: eager prefill 0.98, decode 1/16/64 = 0.94/0.96/0.96; graph prefill 0.97, decode 0.63/0.67/0.89 (INFO, identity codec copies every weight each product) | `results-f8h2/qwen4` |
| (a) H100, GLM-5.3-Flash 3 dense layers, eager and compile | **PASS** both (34824 values); 6 Linears packed. bf16 smoke ok | `results-f8h2/glm3`, `bf16` |
| (a) FlashInfer `-lnvrtc` | FlashInfer's FP8 block-scale GEMM built in ~80 s on the pod; Hopper is verified now (section 5's "not verified" is superseded) | `results-f8h2/log` |
| (c) RTX 4090, MiniMax-M3-MXFP8 3 dense layers (12 MXFP8 Linears, Marlin MXFP8 kernel, int32 tiles), eager and graph | **PASS bit for bit** both (34929 values); kernel lines equal; weights 5.72 -> 5.84 GiB | `results-f8a6/minimax3` |
| (b) RTX 4090, Qwen4, seed sweep: `PYTHONHASHSEED` 0..4, a fresh compile cache each (default compile and graphs, Triton autotuners pinned: four unpinned comparisons earlier changed no bit), stock and Glyd; Glyd with its scratch overwritten (0xFF) after every product; Glyd twice | **PASS** 5/5 seeds Glyd = stock; poison run = unpoisoned run; two fresh Glyd compiles identical. All 12 runs (5 stock, 7 Glyd) gave ONE output | `results-f8a7/qwen4-seeds/check.txt`, `seeds.json` |
| (c) Mistral-Small-4-119B first layers, per-tensor static FP8, MLA + experts | **NOT RUN on a GPU** (hold from the coordinator; no fourth pod). A CPU dry run of the real vLLM 0.30 code is in F4 | `scratchpad/t3/dry_mistral.py` |

The first-run mismatch of round 1 (pod 1's `glyd-graph` differed from stock from position 24 of the long prompt) is better understood: the output Glyd gave then (kept as `~/xbits.json`, "X") is the one every run in the sweep gave, stock included (`stock-s0` .. `stock-s4` all equal X, over five hash seeds and five fresh compiles). So X is an output stock itself produces on this tree; the round-1 event was a compile or autotune outcome of that pod, not a Glyd defect. What was not reproduced is the other output (stock on pod 1). It is therefore not "unexplained Glyd drift", but its cause (environment of pod 1: an older install, unpinned autotuners) is not proven. Section 5's wording of that event should read this way.

## F3. Pods and minutes

Runpod through `rp_run.sh`, `rp_guard.sh` running, MAX_MIN 45 each, never two at once, all deleted ("pods left" shows none of mine; a `glyd-bench-f8f-a2` pod in the listing is not mine and was not touched).

| pod | GPU | minutes | job | results |
| :--- | :--- | ---: | :--- | :--- |
| `f8h2` | H100 80GB HBM3 | 27 | (a): Qwen4 eager/graph, GLM3 eager/compile, bf16 smoke | `results-f8h2` |
| `f8a6` | RTX 4090 | 31 | MiniMax-M3 MXFP8 passed; the Mistral slice wait blocked on a sequential end check and the seeds item was skipped (a 18-minute install): killed | `results-f8a6` |
| `f8a7` | RTX 4090 | 21 | seed sweep (b) passed; the Mistral slice crashed on a `get()` that returned None after a throttled request (both fixed locally, `a5840073`, `a8ebabda`) | `results-f8a7` |

This round: 79 pod-minutes (27 H100, 52 4090), three pods of the four allowed. With round 1: 183 pod-minutes, eight pods. The cost of the two 4090 pods that did not finish everything is in the job script's three bugs (the end check, `get()`, the seeds budget), each now fixed and dry-run on this Mac with a stub harness (`scratchpad/t3/dry/dry.sh`).

## F4. Mistral, as far as it goes without a GPU

- vLLM 0.30 resolves Mistral-Small-4-119B-2603 only in Mistral's own format (`params.json` + `consolidated*.safetensors` -> `MistralLarge3ForCausalLM`, a DeepSeek-V3-style MLA with experts, `quantization_config {quant_method: fp8, activation_scheme: static}`); the HF-format route is not supported. `make_slice.py --weights consolidated` builds the text-only slice (the vision encoder dropped, `n_layers` set; 1 layer: 1184 tensors, 5.05 GiB, end check ok on this Mac).
- A CPU dry run (`scratchpad/t3/dry_mistral.py`, the real vLLM 0.30 source, gloo, `VLLM_TARGET_DEVICE=cpu`, `--quantization glyd` through `vp.register()`): `ModelConfig` keeps `glyd`; `initialize_model` builds one layer in which `fused_qkv_a_proj`, `q_b_proj`, `o_proj` and the shared experts' `gate_up_proj`, `down_proj` get `GlydFp8LinearMethod` (static per-tensor, `fp8-tensor`), `kv_b_proj` keeps vLLM's own `Fp8LinearMethod` (M10 on the real model), the experts are vLLM's `Fp8MoEMethod`, the router and `lm_head` are unquantized. `load_weights` of the slice's safetensors loaded all 32 parameter names, none missing. The run stopped in `process_weights_after_loading` on a missing CPU kernel (`_C.static_scaled_fp8_quant`), which a GPU build has.
- Expectation for the GPU run, when allowed: eager and `+quant_fp8` exact; the default compile expected to differ (static per-tensor quantization is a torch op inductor fuses), which S2's refusal covers for `exact`. Use `F8_PLAN="mistral2:eager,graphq,graph"` on a fresh 4090 (the slice builds in ~6 minutes beside the install; keep `seeds` out of that pod).

## F5. Deferred, and concerns

1. **Mistral per-tensor static slice on a GPU** (above). Static activation scales through the wrapper are untested on a GPU.
2. M5: the codec's input is Marlin-permuted int32 tiles on Ada; the compression gain on those bytes is unmeasured (needs the Marlin repack for the real codec).
3. M9: a library-load check for a real codec (the identity codec needs none).
4. `refusals()` repeats `vllm_plugin`'s refusal code; share it when the bf16 plugin next changes. `_HF` (hf_config by quantization config) is never cleared.
5. M6 gaps left: a multi-device scratch, the draft-model `f8_codec` option.
6. Not run on GPU: modelopt FP8 (non-MX), compressed-tensors, DeepSeek V4, TP > 1, speculative decoding with an FP8 draft.
7. Speed with the identity codec is a placeholder (0.63-0.96 of stock; graph decode 1-16 sequences is the low end). The speed gate needs the fused kernel (Task 5), whose own gate has failed on H100 per the coordinator; whether this plugin continues is the owner's decision.
