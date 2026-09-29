# generate() compiled by default: the box's runs (RTX 4080 SUPER, Ryzen 9 7950X3D, 2026-09-28)

PyTorch 2.14.0 (CUDA 13.0), transformers 5.17.0; the library built from the branch; fresh processes.

The scripts as they ran, at the commits named; from 5f27509 the compiled forward is in glyd.gpu.model._COMPILED
(the scripts read `glyd_compiled` in the model's `__dict__`).

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
- `checks-d3cbba0/`: at d3cbba0 (the second review's fixes), one at a time: test_gpu, check_api dense (Qwen3-0.6B +
  1.7B) and MoE (granite).
- `checks-merge-6e17b3f/`: main (6e17b3f, gpu-hopper2) merged in (6bef364), the library built from it, one at a
  time: check_capi, test_gpu, the self-test; check_api dense at 6bef364 (`check_api_dense-6bef364.txt`: Qwen3-1.7B's
  compress check failed; `-instrumented`: its first compiled call and its later ones share 13 of 32 tokens, compress's
  model's are the later ones', the eager ones all the same; `compress-apart.txt`, `compress_apart.py`: the two models
  alone, three fresh inductor caches, all the same) and at f4822f5 (`check_api_dense.txt`: the tokens compared eager).
