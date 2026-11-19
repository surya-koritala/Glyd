"""A top-level model.safetensors.index.json for the Hub's model-size panel: every tensor of model/ and precise/ (what precise mode loads)."""
import json, struct, os, glob, sys
root, out = sys.argv[1], sys.argv[2]
os.chdir(root)
wm, total = {}, 0
for d in ["model", "precise"]:
    for f in sorted(glob.glob(d + "/**/*.safetensors", recursive=True)):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__":
                wm[k if d == "model" else f.rsplit(".", 1)[0] + "/" + k] = f
        total += os.path.getsize(f)
meta = {"total_size": total, "note": "The weights Glyd loads in precise mode (model/ and precise/), listed for the model-size panel. Fast mode uses pack/ in place of precise/."}
json.dump({"metadata": meta, "weight_map": wm}, open(out, "w"), indent=1)
print(len(wm), "tensors in", len(set(wm.values())), "files,", round(total / 1e9, 1), "GB")
