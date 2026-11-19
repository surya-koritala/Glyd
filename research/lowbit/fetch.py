"""Download named files of a public Hugging Face repo with hf_hub_download: no token, a throwaway HF_HOME.

    python fetch.py REPO FILE [FILE ...]      # into $LOWBIT_DATA/REPO (default ./data)
"""
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # never send a token, even a cached one
os.environ.pop("HF_TOKEN", None)
from huggingface_hub import hf_hub_download

repo, files = sys.argv[1], sys.argv[2:]
out = os.path.join(os.environ.get("LOWBIT_DATA", "data"), repo.replace("/", "__"))
for f in files:
    t = time.time()
    p = hf_hub_download(repo, f, local_dir=out, token=False)
    print(f"{repo}/{f}: {os.path.getsize(p) / 1e9:.2f} GB in {time.time() - t:.0f} s -> {p}", flush=True)
