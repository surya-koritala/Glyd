"""The samples of the format-33 study: for each 2026 bf16 model, 12-14 tensors over its kinds (attention, linear attention, MLP,
experts, embedding, LM head, vision tower, MTP head), each a slice of whole rows (a multiple of 64, about 4 M weights) at a seeded
random place. Reads the tensor names and shapes of research/inventory-2026's headers (no weights).
    python plan.py HEADERS_DIR > samples.json"""
import json, os, random, re, sys

MODELS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B", "google/gemma-4-12B-it", "google/gemma-4-31B-it",
          "ibm-granite/granite-4.2-8b", "zai-org/GLM-5.3-BF16", "meta-models/Muse-Glimmer-30B"]
TARGET = 4 << 20  # weights a sample


def kind(n):
    n = n.lower()
    if n.startswith("mtp") or ".mtp." in n or "eh_proj" in n:
        return "mtp"
    if any(v in n for v in ("visual", "vision", "embed_audio")):
        return "vision"
    if "lm_head" in n:
        return "lm_head"
    if "embed_tokens" in n:
        return "embed"
    if ".experts." in n:
        return "expert"
    if "linear_attn" in n:
        return "linear_attn"
    if "attn" in n:
        return "attn"
    if "mlp" in n:
        return "mlp"
    return "other"


# samples a kind (the rest of a model's 12 from its biggest kinds)
QUOTA = {"attn": 3, "linear_attn": 2, "mlp": 3, "expert": 4, "embed": 1, "lm_head": 1, "vision": 2, "mtp": 1}


def main():
    hd = sys.argv[1]
    rnd = random.Random(0)
    out = []
    for repo in MODELS:
        h = json.load(open(os.path.join(hd, repo.replace("/", "__") + ".json")))
        groups = {}
        for shard, ts in h["shards"].items():
            for name, (dt, shape) in ts.items():
                if dt != "BF16" or len(shape) != 2 or shape[0] < 64 or shape[1] % 16 or "gate.weight" in name and ".mlp.gate." in name:
                    continue
                # one list per kind and per tensor role (the name without its layer/expert numbers)
                groups.setdefault(kind(name), {}).setdefault(re.sub(r"\.\d+\.", ".N.", name), []).append((name, shard, shape))
        picks = []
        for k, q in QUOTA.items():
            roles = sorted(groups.get(k, {}))
            rnd.shuffle(roles)
            for i in range(min(q, sum(len(v) for v in groups.get(k, {}).values()))):
                picks.append(rnd.choice(groups[k][roles[i % len(roles)]]))
        while len(picks) < 12:  # fill from the attention and MLP roles
            k = rnd.choice([k for k in ("attn", "mlp", "expert", "linear_attn") if k in groups])
            picks.append(rnd.choice(groups[k][rnd.choice(sorted(groups[k]))]))
        for name, shard, (R, C) in picks:
            n = max(64, min(R // 64 * 64, TARGET // C // 64 * 64))
            r0 = rnd.randrange(0, R // 64 * 64 - n + 1) // 64 * 64
            out.append({"repo": repo, "name": name, "kind": kind(name), "shard": shard, "shape": [R, C], "r0": r0, "r1": r0 + n})
    json.dump(out, sys.stdout, indent=0)


if __name__ == "__main__":
    main()
