import json, sys, re, struct, urllib.request, time, collections
from probe_repos import get, header
H = "https://huggingface.co"
repo = sys.argv[1]
idx = json.loads(get(f"{H}/{repo}/resolve/main/model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
n, hdr = header(repo, shards[len(shards)//2])
pat = collections.OrderedDict()
for k, v in hdr.items():
    if k == "__metadata__": continue
    key = re.sub(r"\.\d+\.", ".N.", k)
    if key not in pat: pat[key] = (v["dtype"], v["shape"])
for k, (d, s) in list(pat.items())[:60]: print(d, s, k)
