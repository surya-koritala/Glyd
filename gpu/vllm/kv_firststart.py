"""The first start of kv: lossless on a model, with an empty cache directory: the base tables made by a child process (a model load, the calibration, the vocabulary sweep), then the engine's own load, and a prompt served.

    GLYD_CACHE=/empty/dir VLLM_ENABLE_V1_MULTIPROCESSING=0 python kv_firststart.py MODEL [MAX_MODEL_LEN] [OUT.json]

The seconds of each part come from the plugin's log lines; the total is the time from LLM(...) to the first prompt's text. A second start in the same directory (the tables kept) is timed too."""
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
assert os.environ.get("GLYD_CACHE"), "GLYD_CACHE must name an empty directory"
from vllm import LLM, SamplingParams

model = sys.argv[1]
mml = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
out = sys.argv[3] if len(sys.argv) > 3 else None
res = {"model": model, "max_model_len": mml, "cache_dir_before": sorted(os.listdir(os.path.join(os.environ["GLYD_CACHE"], "kv"))) if os.path.isdir(os.path.join(os.environ["GLYD_CACHE"], "kv")) else []}
t0 = time.time()
llm = LLM(model, dtype="bfloat16", enforce_eager=True, max_model_len=mml, gpu_memory_utilization=0.9, quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless"}})
res["first_start_s"] = round(time.time() - t0, 1)
o = llm.generate(["The history of data compression began"], SamplingParams(temperature=0, max_tokens=24), use_tqdm=False)[0]
res["text"] = o.outputs[0].text
res["cache_files"] = sorted(os.listdir(os.path.join(os.environ["GLYD_CACHE"], "kv")))
print(json.dumps(res), flush=True)
if out:
    json.dump(res, open(out, "w"), indent=1)
