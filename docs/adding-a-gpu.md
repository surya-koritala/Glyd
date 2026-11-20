# Adding a (GPU, family) to the engine's list

`glyd run` hands a model to `glyd-engine` where the pair of its GPU and its model family is in `GPUS` (`glyd-engine/src/machine.rs`: one line a GPU, naming the families), and to vLLM on every other. A pair goes in where
the engine has been measured at least as fast as vLLM at every load it would serve there, with less memory: the speed bar. It is per family because families differ in what the engine does with several users: the
Qwen3 dense models (`machine::QWEN3`, the family `glyd run` hands to the engine by default) batch up to 8 requests at once, a family that serves one request at a time (Gemma 4, Qwen3.5 and the other lossy-level
families) loses to vLLM at 4 and 8 users and is not listed. The lossy levels (`:kestrel`, `:swift`) are the engine's whenever the user asks for them (vLLM has none), and the list is not looked at for them.

The model measured must have been released within the last 6 months (the Hub's `createdAt`, https://huggingface.co/api/models/ID): the file records which model and when, and `scripts/gpu_bar.py` refuses an older one
(`--stale-ok` for a family with no newer model, and says so).

1. On the GPU, with nothing else using it, run both programs under the same load, one after the other, more than once (five runs of each: the spread of a run is under 1% for decode and first token). The earlier
   records' `scripts/` have what a run used: `run.sh engine OUT` and `run.sh vllm OUT` (each starts the server, puts 1, 4 and 8 users at once with short and 2,000 and 4,000 token prompts on it, writes `OUT/load.json` and
   stops it; a run during which a CI job was on the GPU is discarded). `glyd-engine` and the pinned vLLM (`glyd-gpu`'s, `vllm>=0.30,<0.31`) are what a user has; where vLLM cannot run the engine's level of the model
   (a lossy level) the comparison is vLLM's own checkpoint of the model nearest it in size.
2. Put the runs' means in a file each and through the bar:

   ```bash
   python3 scripts/gpu_bar.py --summarize e1.json,e2.json,e3.json,e4.json,e5.json --model ORG/NAME --released YYYY-MM-DD --family Qwen3 > engine.json
   python3 scripts/gpu_bar.py --summarize v1.json,...,v5.json > vllm.json
   python3 scripts/gpu_bar.py engine.json vllm.json "NVIDIA GeForce RTX 4080 SUPER" 8.9 Qwen3
   ```

   It prints the table of the loads; at every one the engine's decode tokens/s (a user, and in all) and its time to first token must be at least vLLM's (3%, more than the spread of a run, is allowed), and its peak
   memory lower. A pair that passes prints its line; one that fails prints each load it failed by how much, and no line. Exit status 0 passes, 1 fails the bar, 2 a model older than 6 months.
3. Add the line to `GPUS` (a GPU already there gets the family added to its line), with the GPU's name as the driver gives it (`nvidia-smi --query-gpu=name`: the word must be a whole word of it) and its compute
   capability, and a test line in `machine.rs` for its neighbours (an RTX 4080 is not an RTX 4080 SUPER). Write the run up in `benchmarks/gpu/<gpu>-<date>/` (a README, the two measurement files and the scripts; raw
   outputs are not kept in the tree: `scripts/check_tree.sh`).
4. The acceptance is `glyd run MODEL` using the engine on that GPU with no environment variable set: `glyd-engine preflight MODEL` exits 0.

A pair that fails stays out; `GLYD_ENGINE=1 glyd run MODEL` still uses the engine on it, by the user's choice.
