# generate() compiled by default: the box's runs (RTX 4080 SUPER, Ryzen 9 7950X3D, 2026-09-28)

PyTorch 2.14.0 (CUDA 13.0), transformers 5.17.0; the library built from the branch; fresh processes.

- `gen-main.txt`: origin/main (1653818), `gen.py` for bf16 and Glyd, plain `generate()` (eager) and with
  `cache_implementation="static"` (transformers' compiled loop), 128 tokens at 1 and 8 sequences.
- `gen-branch.txt`: the branch at 9611d6c, plain `generate()` (compiled by default), and Qwen3-8B's first call with
  PyTorch's compile caches empty (its last line).
- `steps-by-cache-length.txt`: `long.py`, a step's ms compiled against eager by the static cache's length (a long
  prompt, or a generous `max_new_tokens` stopped after 64 tokens), Qwen3-1.7B, 4B-Instruct-2507, 8B, 1 and 8 sequences
  (the prompts of 4096 tokens at 8 sequences ran out of memory).
- `chat-turns.txt`: `grow.py`, a chat's turns (the context and 64 tokens each), the compile, the recompile, the rest.
- `freed-at-del.txt`: `free3.py`, a model that generated compiled freed at `del` (the branch's compiled forward).
- `checks-5d38f20/`: check_api (Qwen3-0.6B + 1.7B; granite MoE), check_models (Qwen3-1.7B), check_capi, test_gpu,
  `fast.py` (the fast loop's cases), `families.py` (generate() on tiny models of every mixture-of-experts family
  test_gpu packs and 12 dense ones), `many.py` (ten models in one process), at 5d38f20.
- `checks-3a32b9b/`: check_api dense and MoE at 3a32b9b; check_models got to Qwen3-8B's fp32 reference on the CPU,
  when the box reset (17:35 EDT) and the run ended there.
- `checks-7acf9a0/`: at 7acf9a0 (the review's fixes), one at a time: check_capi, test_gpu, the self-test
  (`glyd_gpu.py`), check_api dense (Qwen3-0.6B + 1.7B) and MoE (granite), check_models (Qwen3-1.7B).
