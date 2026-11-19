"""hub.json (each repo's createdAt and downloads: the Hub's `downloads`, the last 30 days, and `downloadsAllTime`, with the date read)
and the safetensors headers + config.json of each repo in repos.txt by HTTP range reads (no weights) into headers/ (resumable),
2.5 s apart.   python hdr.py [hub]   (hub: hub.json only)"""
import json, os, struct, sys, time, urllib.request, urllib.error
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "headers"); os.makedirs(OUT, exist_ok=True)
last = [0.0]
def get(url, rng=None):
    for t in range(8):
        w = 2.5 - (time.time() - last[0])
        if w > 0: time.sleep(w)
        h = {"User-Agent": "glyd-quant/1"}
        if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
        try:
            last[0] = time.time()
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500: time.sleep(30 * (t + 1)); continue
            raise
        except Exception: time.sleep(10 * (t + 1))
    raise RuntimeError(url)
REPOS = open(os.path.join(D, "repos.txt")).read().split()
hub = {}
for repo in REPOS:
    m = json.loads(get(f"https://huggingface.co/api/models/{repo}?expand[]=createdAt&expand[]=downloads&expand[]=downloadsAllTime"))
    hub[repo] = {"created": m["createdAt"][:10], "downloads_30d": m["downloads"], "downloads_all_time": m["downloadsAllTime"],
                 "as_of": time.strftime("%Y-%m-%d", time.gmtime())}
json.dump(hub, open(os.path.join(D, "results", "hub.json"), "w"), indent=1)
for repo in REPOS if sys.argv[1:] != ["hub"] else []:
    p = os.path.join(OUT, repo.replace("/", "__") + ".json")
    if os.path.exists(p): continue
    base = f"https://huggingface.co/{repo}/resolve/main/"
    o = {"repo": repo, "config": json.loads(get(base + "config.json")), "files": {}}
    info = json.loads(get(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    for fn, size in [(x["rfilename"], x.get("size")) for x in info["siblings"] if x["rfilename"].endswith(".safetensors")]:
        b = get(base + fn, (0, 262143))
        n = struct.unpack("<Q", b[:8])[0]
        if n + 8 > len(b): b += get(base + fn, (len(b), n + 7))
        o["files"][fn] = {"size": size, "hlen": n, "header": json.loads(b[8:8 + n])}
    json.dump(o, open(p, "w")); print(repo, len(o["files"]), "files", flush=True)
