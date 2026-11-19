"""The tensors of GLM-5.3-Flash at the revision glyd-engine's GLM path is checked against, from its shards' headers, and its config files.

    glm_inventory.py OUT_DIR       OUT_DIR/config.json and OUT_DIR/generation_config.json as the revision has them, and OUT_DIR/glm5_next_tensors.json as tests/fixtures/glm5_next_tensors.json holds it

Two HTTP range reads of each of the 62 shards (8 bytes, then the JSON header they give the length of), the two config files and nothing else: no weights, no index, no token (the repo is ungated; none is read or sent).
A mixture of experts' experts are one run for each layer in the fixture (their prefix, how many, and the dtype and shape of each of their six tensors: every expert of a layer is checked to have the same); every other
tensor is by name. Python's standard library only.
"""
import json
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "zai-org/GLM-5.3-Flash"
REV = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
SHARDS = 62


def get(path, rng=None):
    """Bytes of a file of the repo at the revision, the whole file or a byte range (start, end inclusive)."""
    req = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REV}/{path}", headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            if rng and len(body) != rng[1] - rng[0] + 1:
                raise OSError(f"{path}: {len(body)} bytes, where {rng} was asked")
            return body
        except Exception:
            if attempt == 3:
                raise


def header(shard):
    """{tensor: {dtype, shape, data_offsets}} of a safetensors shard, from its first 8 bytes and its header."""
    n = struct.unpack("<Q", get(shard, (0, 7)))[0]
    h = json.loads(get(shard, (8, 7 + n)))
    h.pop("__metadata__", None)
    return h


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        (out / name).write_bytes(get(name))
    shards = [f"model-{i:05d}-of-{SHARDS:05d}.safetensors" for i in range(1, SHARDS + 1)]
    with ThreadPoolExecutor(8) as ex:
        heads = list(ex.map(header, shards))
    have = {}
    for shard, h in zip(shards, heads):
        for n, t in h.items():
            assert n not in have, f"{n}: in two shards"
            have[n] = (t["dtype"], t["shape"])
    plain, runs = {}, {}
    for n, (d, sh) in sorted(have.items()):
        m = re.fullmatch(r"(.*\.mlp\.experts\.)(\d+)\.((?:gate|up|down)_proj\.weight(?:_scale_inv)?)", n)
        if m:
            run = runs.setdefault(m.group(1), {"count": set(), "tensors": {}})
            run["count"].add(int(m.group(2)))
            assert run["tensors"].setdefault(m.group(3), [d, sh]) == [d, sh], f"{n}: not the shape of the layer's other experts"
        else:
            plain[n] = [d, sh]
    out_runs = []
    for prefix, run in sorted(runs.items()):
        assert run["count"] == set(range(len(run["count"]))), f"{prefix}: experts are not 0..{len(run['count'])}"
        assert len(run["tensors"]) == 6, f"{prefix}: {sorted(run['tensors'])}"
        out_runs.append({"prefix": prefix, "count": len(run["count"]), "tensors": run["tensors"]})
    (out / "glm5_next_tensors.json").write_text(json.dumps({"repo": REPO, "revision": REV, "tensors": plain, "runs": out_runs}, indent=0, separators=(",", ":")))
    print(f"{out}: {len(plain)} tensors by name and {len(out_runs)} runs of experts ({sum(r['count'] * 6 for r in out_runs)} tensors); {len(have)} in all, from {len(shards)} shards")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
