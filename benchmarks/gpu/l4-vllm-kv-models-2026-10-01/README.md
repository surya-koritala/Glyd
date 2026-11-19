# Other models on the lossless KV cache: first starts (2026-10-01 to 2026-10-02)

vLLM 0.30.0, the dev L4. Models (ungated mirrors): `NousResearch/Meta-Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B`.

`kv: lossless` is calibrated once for each model, on vLLM itself, from a bundled text, and it measures the model's whole vocabulary; the result is kept in `$GLYD_CACHE/kv` ([gpu/vllm/README.md](../../../gpu/vllm/README.md#the-kv-cache-in-12-bits)). This record measures what that adds to a model's first start. Each model's KV cache against stock's, its checks and its floods are in [`l4-vllm-kv-heavy-2026-10-01`](../l4-vllm-kv-heavy-2026-10-01) and [`l4-vllm-kv-wait-2026-10-02`](../l4-vllm-kv-wait-2026-10-02).

## What a first start adds (`firststart/`, `kv_firststart.py`: an empty `GLYD_CACHE`, the whole `LLM(...)` from the start to a first prompt's text, eager, weights in the OS file cache)

| Model | `--max-model-len` | the start | the calibration and the vocabulary sweep (its log) |
| :--- | ---: | ---: | :--- |
| Qwen/Qwen3-4B | 4,096 | 57.8 s | 38 s (12 s of it loading the model) |
| Qwen/Qwen3-8B | 4,096 | 79.8 s | 59 s (13 s of it loading the model) |
| Qwen/Qwen3-8B | 16,384 | 102.2 s | 82 s (16 s of it loading the model) |

## The calibration and the sweep per model (16,384 positions, without the model's load)

| Model | layers x KV heads | the calibration and the sweep | the sweep's part | the model's load in that run |
| :--- | :--- | ---: | ---: | ---: |
| Qwen3-4B | 36 x 8 | 40 s | 23 s | 73 s |
| Qwen3-8B | 36 x 8 | 65 s | 41 s | 36 s |
| Qwen3-14B (16 GB of its weights offloaded to the host: they do not fit the L4) | 40 x 8 | 2,017 s | 1,247 s | 394 s |
| Llama-3.1-8B | 32 x 8 | 64 s | 41 s | 210 s |
| Mistral-7B-v0.3 | 32 x 8 | 61 s | 38 s | 136 s |
| Qwen2.5-7B | 28 x 4 | 59 s | 39 s | 138 s |

The part of the start that the plugin adds is the calibration and the sweep: 40 to 65 s for each of the five models that fit the L4, measured at 16,384 positions, plus a second load of the model, from the file cache then. The model's own load is vLLM's: 136 s (Mistral-7B), 138 s (Qwen2.5-7B) and 210 s (Llama-3.1-8B) from this box's disk when the weights were not cached, so on a cold disk a first start is those plus the calibration, 3.3 to 4.6 minutes for the calibration step alone; with the weights in the file cache it is 58 to 102 s.

Qwen3-14B's calibration was made with its weights streamed from the host, a time that says nothing of a first start on a GPU that holds the model; no engine checks were run on it here.

## Files

`firststart/` (each first start's result file). The tool is [`gpu/vllm/kv_firststart.py`](../../../gpu/vllm/kv_firststart.py). Result files are the harness's own JSON; logs that carry the plugin's internal diagnostics are not part of this record.
