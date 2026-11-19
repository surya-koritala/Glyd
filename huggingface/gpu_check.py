"""On one GPU (a pod): this repository's glyd_gpu packs MODEL as from_pretrained does and saves it (save_pretrained, shards
of SHARD bytes), lists every file's and tensor's sha256 (as the CPU pack workflow lists its save), generates greedily, then
loads the save back with verify=True and generates again. The listing is compared with the CPU save's elsewhere."""
import hashlib
import json
import sys
import time
from pathlib import Path


def listing(d):
    files, tensors = [], []
    for f in sorted(Path(d).glob("*.safetensors")):
        data = f.read_bytes()
        files.append({"file": f.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        n = int.from_bytes(data[:8], "little")
        head = json.loads(data[8:8 + n])
        for k, v in head.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            tensors.append({"file": f.name, "name": k, "dtype": v["dtype"], "shape": v["shape"], "sha256": hashlib.sha256(data[8 + n + a:8 + n + b]).hexdigest()})
    for f in sorted(Path(d).iterdir()):
        if f.suffix != ".safetensors":
            files.append({"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    return files, tensors


def main():
    model, revision, out, save = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])  # (the save outside out: out is pulled back)
    import torch
    from transformers import AutoTokenizer
    import glyd_gpu
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    prompt = tok("The three primary colours of light are", return_tensors="pt").to("cuda")

    def generate(m):
        with torch.no_grad():
            return m.generate(**prompt, max_new_tokens=24, do_sample=False)[0].tolist()

    r = {"model": model, "revision": revision, "glyd_gpu": glyd_gpu.__version__, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(), "complete": False}
    t = time.time()
    m = glyd_gpu.from_pretrained(model, revision=revision, layout="mma", verify=True, compile=False)
    r["pack_s"], r["packed_verified"] = round(time.time() - t, 1), m.config.quantization_config.verified
    r["tokens_packed"] = generate(m)
    t = time.time()
    glyd_gpu.save_pretrained(m, str(save), shard_bytes=10**9)
    r["save_s"] = round(time.time() - t, 1)
    del m
    torch.cuda.empty_cache()
    files, tensors = listing(save)
    (out / "gpu-files.json").write_text(json.dumps(files, indent=0) + "\n")
    (out / "gpu-tensors.json").write_text(json.dumps(tensors, indent=0) + "\n")
    for n in ("glyd.json", "model.safetensors.index.json"):
        if (save / n).exists():
            (out / ("gpu-" + n)).write_bytes((save / n).read_bytes())
    t = time.time()
    m = glyd_gpu.from_pretrained(str(save), verify=True, compile=False)
    q = m.config.quantization_config
    r["load_s"], r["loaded_verified"], r["loaded_hashed"], r["loaded_unhashed"] = round(time.time() - t, 1), q.verified, q.hashed, q.unhashed
    r["tokens_loaded"] = generate(m)
    r["tokens_match"] = r["tokens_loaded"] == r["tokens_packed"]
    r["text"] = tok.decode(r["tokens_loaded"])
    r["complete"] = True
    (out / "gpu-check.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("tokens_")}))


if __name__ == "__main__":
    main()
