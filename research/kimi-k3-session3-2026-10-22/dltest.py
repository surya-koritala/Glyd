#!/usr/bin/env python3
# GLY-356: where a datacenter link's download time goes. The same shard of DeepSeek-V4-Flash read over one connection, then 16, through the Hub's
# bridge (resolve: what glyd's downloader reads) and straight from the xorbs (the reconstruction's ranges: what hf_xet reads). curl does the transfers,
# each stopped at 60 s (the bytes it got by then count).
import json, os, subprocess, sys, time, urllib.request

REPO = os.environ.get("DLTEST_REPO", "deepseek-ai/DeepSeek-V4-Flash")
HUB = "https://huggingface.co"
GB = int(os.environ.get("DLTEST_MB", "1024")) << 20  # (a smaller test where DLTEST_MB says)


def get_json(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60) as r:
        return json.load(r)


def curl(url, a, b):
    # (one connection: bytes a..b inclusive to nowhere; the bytes and seconds curl reports)
    out = subprocess.run(["curl", "-s", "-L", "-m", "60", "-o", "/dev/null", "-H", f"Range: bytes={a}-{b}", "-w", "%{size_download} %{time_total}", url], capture_output=True, text=True, timeout=900).stdout.split()
    return int(out[0]), float(out[1])


def parallel(jobs):
    # (each job a curl of its own, all at once: the bytes over the wall time)
    t0 = time.time()
    ps = [subprocess.Popen(["curl", "-s", "-L", "-m", "60", "-o", "/dev/null", "-H", f"Range: bytes={a}-{b}", "-w", "%{size_download}", u], stdout=subprocess.PIPE, text=True) for u, a, b in jobs]
    got = sum(int(p.communicate(timeout=900)[0] or 0) for p in ps)
    return got, time.time() - t0


def main():
    index = get_json(f"{HUB}/{REPO}/resolve/main/model.safetensors.index.json")
    shard = sorted(set(index["weight_map"].values()))[int(os.environ.get("DLTEST_SHARD", "1"))]
    url = f"{HUB}/{REPO}/resolve/main/{shard}"
    head = subprocess.run(["curl", "-s", "-I", url], capture_output=True, text=True).stdout.lower()
    xet = next(l.split(":", 1)[1].strip() for l in head.splitlines() if l.startswith("x-xet-hash:"))
    tok = get_json(f"{HUB}/api/models/{REPO}/xet-read-token/main")
    rec_req = urllib.request.Request(f"{tok['casUrl']}/reconstruction/{xet}", headers={"Authorization": f"Bearer {tok['accessToken']}", "Range": f"bytes=0-{4 * GB - 1}"})
    with urllib.request.urlopen(rec_req, timeout=60) as r:
        rec = json.load(r)
    ranges = [(f["url"], f["url_range"]["start"], f["url_range"]["end"]) for fs in rec["fetch_info"].values() for f in fs]
    print(f"{shard}: {len(rec['terms'])} terms over {len(ranges)} xorb ranges for its first 4 GiB", flush=True)

    n, s = curl(url, 0, GB - 1)
    print(f"bridge, one connection: {n / 1e6:.0f} MB in {s:.1f} s, {n / s / 1e6:.1f} MB/s", flush=True)
    got, t0, k = 0, time.time(), 0
    while got < GB and k < len(ranges):
        u, a, b = ranges[k]
        got += curl(u, a, b)[0]
        k += 1
    s = time.time() - t0
    print(f"xorbs, one at a time: {got / 1e6:.0f} MB in {s:.1f} s, {got / s / 1e6:.1f} MB/s ({k} ranges)", flush=True)

    part = GB // 4
    n, s = parallel([(url, i * part, (i + 1) * part - 1) for i in range(16)])
    print(f"bridge, 16 connections of GB/4: {n / 1e6:.0f} MB in {s:.1f} s, {n / s / 1e6:.1f} MB/s", flush=True)
    n, s = parallel(ranges[:16])
    print(f"xorbs, 16 at once: {n / 1e6:.0f} MB in {s:.1f} s, {n / s / 1e6:.1f} MB/s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"dltest failed: {e}", file=sys.stderr)
