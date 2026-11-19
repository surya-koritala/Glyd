"""Prepare pinned matrix bytes and exact Rust packs in CI, before a GPU allocation."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import urllib.parse

from protocol import MODEL, REVISION, POLICY, TOKENS, check_inputs, digest, plan
STUDY = Path(__file__).resolve().parents[1] / "lossy-2026"
sys.path.insert(0,str(STUDY))
from sample import fetch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.out
    root.mkdir(parents=True)
    (root/"fixtures").mkdir()
    pin = json.loads((STUDY/"models.json").read_text())[MODEL]
    if pin["sha"] != REVISION or not pin["createdAt"].startswith("2026-"):
        raise ValueError("wrong model pin")
    inventory = json.loads((STUDY/"qwen38-inventory.json").read_text())
    cases, unsupported = plan(inventory)
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
    metadata = json.loads(fetch(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"))
    if metadata["sha"] != REVISION or not metadata["createdAt"].startswith("2026-"):
        raise ValueError("wrong Hub revision")
    index = json.loads(fetch(base+"model.safetensors.index.json"))["weight_map"]
    if set(index) != {r["name"] for r in inventory}:
        raise ValueError("pinned inventory changed")
    headers = {}
    source = root/"source.bf16"
    for case in cases:
        filename = index[case["name"]]
        if Path(filename).name != filename:
            raise ValueError("invalid shard name")
        url = base+urllib.parse.quote(filename)
        if filename not in headers:
            head = fetch(url,0,262143)
            n = struct.unpack("<Q",head[:8])[0]
            if n > 16 << 20:
                raise ValueError("oversized header")
            if n+8 > len(head):
                head += fetch(url,len(head),n+7)
            headers[filename] = (n+8,json.loads(head[8:n+8]))
        start, header = headers[filename]
        info = header[case["name"]]
        if info["dtype"] != "BF16" or info["shape"] != case["source_shape"]:
            raise ValueError("pinned tensor changed")
        begin,end = info["data_offsets"]
        if end-begin != 2*case["source_shape"][0]*case["source_shape"][1]:
            raise ValueError("tensor length mismatch")
        count = 2*case["shape"][0]*case["shape"][1]
        with source.open("wb") as out:
            for off in range(0,count,16<<20):
                out.write(fetch(url,start+begin+off,start+begin+min(count,off+(16<<20))-1))
        case["source"] = {"file":filename,"byte_start":start+begin,"bytes":count,"sha256":digest(source),
                          "verification":"pinned revision, checked Content-Range and frozen range digest; not a full-shard digest"}
        subprocess.run([str(args.binary.resolve()),str(source),*map(str,case["shape"]),
                        str(root/"fixtures"/case["id"])],check=True)
        source.unlink()
        print(case["id"],case["name"],case["shape"],flush=True)
    edges = [
        ("edge-integer-small",64,16,"integer",True),
        ("edge-integer-tail",192,256,"integer",True),
        ("edge-integer-k-tail",192,80,"integer",True),
        ("edge-all-compact-words",64,64,"compact",False),
        ("edge-all-bf16-words",64,1024,"all",False),
    ]
    for ident,rows,cols,kind,products in edges:
        if kind == "all":
            words = list(range(65536))
        elif kind == "compact":
            words = list(range(0,65536,16))
        else:
            words = [struct.unpack("<I",struct.pack("<f",float((i*13+i//17)%17-8)))[0] >> 16
                     for i in range(rows*cols)]
        source.write_bytes(struct.pack("<"+"H"*len(words),*words))
        subprocess.run([str(args.binary.resolve()),str(source),str(rows),str(cols),
                        str(root/"fixtures"/ident)],check=True)
        source.unlink()
        cases.append({"id":ident,"kind":kind,"shape":[rows,cols],"products":products,
                      "exact_product":products,"whole_tensor":False})
    for case in cases:
        directory = root/"fixtures"/case["id"]
        case["files"] = {p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(directory.iterdir())}
    manifest = {"schema":1,"model":MODEL,"revision":REVISION,"createdAt":pin["createdAt"],
                "tokens":TOKENS,"policy":POLICY,"cases":cases,"unsupported_tensors":unsupported,
                "coverage":"one matrix per eligible shape; embedding rows explicitly sliced; no whole-model residency"}
    (root/"cases.json").write_text(json.dumps(manifest,indent=2)+"\n")
    check_inputs(root)
    print(json.dumps({"prepared_cases":len(cases),"uncovered_tensors":len(unsupported),
                      "fixture_bytes":sum(f["bytes"] for c in cases for f in c["files"].values())}))

if __name__ == "__main__":
    main()
