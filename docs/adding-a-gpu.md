# Adding a GPU to the engine's list

`glyd run` hands a model to `glyd-engine` on the GPUs in `GPUS` (`glyd-engine/src/machine.rs`) and to vLLM on every other. A GPU goes in
where the engine has been measured at least as fast as vLLM at every load it would serve there, with less memory: the speed bar.

1. On the GPU, with nothing else using it, put the same loads on both servers, one after the other, with `glyd-engine bench` (docs/bench.md): it measures any OpenAI-compatible server over HTTP, so
   the two files differ in the server and in nothing else. `glyd-engine` and the pinned vLLM (`glyd-gpu`'s, `vllm>=0.30,<0.31`) are what a user has.

   ```bash
   glyd-engine bench Qwen/Qwen3-8B --suite bar --runs 5 --out engine.json        # starts `glyd-engine serve` itself
   vllm serve Qwen/Qwen3-8B --quantization glyd --port 8001 &                    # vLLM's, then the same loads on it
   glyd-engine bench --against http://127.0.0.1:8001 --suite bar --runs 5 --out vllm.json
   ```

   (1, 4 and 8 requests at once with short, 2,000 and 4,000 token prompts, a warm-up round and five runs of each load, the peak GPU memory sampled while they run.)
2. Put the two files through the bar:

   ```bash
   glyd-engine bench --bar engine.json vllm.json --gpu-name "NVIDIA GeForce RTX 4080 SUPER" --cc 8.9      # or: python3 scripts/gpu_bar.py engine.json vllm.json "NVIDIA GeForce RTX 4080 SUPER" 8.9
   ```

   It prints the table of the loads; at every one the engine's decode tokens/s (a user, and in all) and its time to first token must be at least vLLM's (3%, the
   spread of two runs of one engine, is allowed), and its peak memory lower. A GPU that passes prints its line; one that fails prints each load it failed by how much,
   and no line. Exit status 0 or 1.
3. Add the line to `GPUS`, with the GPU's name as the driver gives it (`nvidia-smi --query-gpu=name`: the word must be a whole word of it) and its compute capability, and
   a test line in `machine.rs` for its neighbours (an RTX 4080 is not an RTX 4080 SUPER). Write the run up in `benchmarks/gpu/<gpu>-<date>/` (a README and the scripts; raw outputs
   are not kept in the tree: `scripts/check_tree.sh`).
4. The acceptance is `glyd run MODEL` using the engine on that GPU with no environment variable set: `glyd run Qwen/Qwen3-8B --prompt hi` says which program runs it.

A GPU that fails stays out; `GLYD_ENGINE=1 glyd run MODEL` still uses the engine on it, by the user's choice.
