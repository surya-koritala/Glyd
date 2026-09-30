"""Speculative decoding with Glyd in vLLM, one user: a model (default Qwen/Qwen3-8B) as vLLM serves it, bf16 or
--quantization glyd, with or without vLLM's speculation (n-gram prompt lookup, or an EAGLE-3 draft), greedy, one
request at a time, on two mixes of prompts: "edit" (a given text, code or data to fix, rewrite, convert or summarize,
where n-gram lookup finds the answer in the prompt) and "chat". For each request its tokens, its time to the first
token and its whole time; for each mix vLLM's speculation counters (drafts, draft tokens, accepted tokens). With
"draft": the drafter's layers packed by Glyd, and vLLM's additional_config (Glyd's compile cache key).

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python spec_decode.py OUT.json '{"mode": "glyd", "spec": "ngram"}'

The run's options (JSON): mode (bf16 or glyd), spec (none, "ngram" or "eagle3"), draft_glyd (the EAGLE-3 draft asked
for --quantization glyd too), eager, model, tokens (max output tokens, 400), util (--gpu-memory-utilization, 0.9),
batched (--max-num-batched-tokens, vLLM's default). Glyd's own options from the environment (GLYD_EXACT and the
rest)."""
import json
import sys
import time

EDIT = [
    "Fix the spelling and grammar in this paragraph and return the whole corrected paragraph, nothing else:\n\n"
    "The comittee met on tuesday to discus the new budjet for the libary. Several members argued that the the "
    "reading room needs new chairs, becuase the old ones is broken and uncomfortable. Others said the money would "
    "be better spent on longer opening hours, since many students cant visit during the day. After a long debate, "
    "the chair propose a compromise: half of the funds will go to furniture, and the other half will pay for two "
    "extra evenings each week. The vote was postponed untill next month so that the treasurer can check weather the "
    "numbers add up. Members was asked to send there comments by friday.",
    "Add type hints and a one-line docstring to every function in this Python module. Return the full module.\n\n"
    "```python\nimport json\nimport os\n\n\ndef load(path):\n    with open(path) as f:\n        return json.load(f)\n\n\n"
    "def save(path, data):\n    with open(path, \"w\") as f:\n        json.dump(data, f, indent=2)\n\n\n"
    "def merge(a, b):\n    out = dict(a)\n    for k, v in b.items():\n        if k in out and isinstance(out[k], dict) and isinstance(v, dict):\n"
    "            out[k] = merge(out[k], v)\n        else:\n            out[k] = v\n    return out\n\n\n"
    "def find(root, suffix):\n    hits = []\n    for d, _, files in os.walk(root):\n        for name in files:\n"
    "            if name.endswith(suffix):\n                hits.append(os.path.join(d, name))\n    return sorted(hits)\n\n\n"
    "def count_keys(data):\n    if isinstance(data, dict):\n        return len(data) + sum(count_keys(v) for v in data.values())\n"
    "    if isinstance(data, list):\n        return sum(count_keys(v) for v in data)\n    return 0\n```",
    "Convert this JSON to YAML, keeping every key and value exactly. Return only the YAML.\n\n"
    "{\"service\": \"inventory\", \"version\": \"2.4.1\", \"replicas\": 3, \"port\": 8080, \"debug\": false, "
    "\"database\": {\"host\": \"db.internal\", \"port\": 5432, \"name\": \"inventory\", \"user\": \"svc_inventory\", "
    "\"pool\": {\"min\": 2, \"max\": 20, \"timeout_seconds\": 30}}, \"cache\": {\"enabled\": true, \"ttl_seconds\": 600, "
    "\"backend\": \"redis\", \"host\": \"cache.internal\"}, \"features\": [\"search\", \"bulk_import\", \"audit_log\", "
    "\"low_stock_alerts\"], \"limits\": {\"max_items_per_order\": 500, \"max_orders_per_minute\": 120}, "
    "\"regions\": [{\"name\": \"us-east\", \"weight\": 60}, {\"name\": \"eu-west\", \"weight\": 30}, "
    "{\"name\": \"ap-south\", \"weight\": 10}], \"owner\": {\"team\": \"supply\", \"email\": \"supply-team@example.com\"}}",
    "Rewrite this email to sound more formal and polite. Keep every detail (names, dates, numbers). Return only the "
    "email.\n\nHi Dana,\n\nquick one - the shipment of 240 laptops we talked about is late again. The supplier now "
    "says it'll arrive on March 14 instead of March 3, which messes up our rollout for the Denver office. Can you "
    "ask them for a discount, like 5 percent, and also check if they can send the first 80 units early so the "
    "Denver team isn't stuck? I need an answer by Friday because the office manager, Luis, is planning the setup "
    "weekend around it. Also the invoice number is INV-20931, in case they ask.\n\nThanks,\nSam",
    "Summarize this meeting transcript in bullet points. Quote each decision word for word as it was stated.\n\n"
    "Priya: Let's start. The release is scheduled for the 18th, and QA found two blocking bugs.\n"
    "Tom: Both are in the export feature. One is a crash with empty files, the other is a timeout on large ones.\n"
    "Priya: Can we fix both by the 15th?\nTom: The crash, yes. The timeout needs a redesign of the batching.\n"
    "Mei: Then let's ship without large-file export and add it in the next minor version.\n"
    "Priya: Decision: we ship on the 18th without large-file export.\n"
    "Tom: I'll add a message telling users that files over 2 GB are not supported yet.\n"
    "Priya: Decision: Tom adds the 2 GB warning by the 15th.\n"
    "Mei: What about the docs? They still describe the old settings page.\n"
    "Priya: Decision: Mei updates the settings page docs before the release.\n"
    "Tom: One more thing, the beta users asked for a changelog.\n"
    "Priya: Decision: we publish a changelog with the release, and Tom drafts it by the 16th.",
]
CHAT = [
    "Explain how a refrigerator keeps food cold.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the trade-offs between SQL and NoSQL databases?",
    "Give me a five-day plan to learn the basics of Python.",
    "Why is the sky blue? Explain it for a ten-year-old.",
]
SPEC = {
    "ngram": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4, "prompt_lookup_min": 2},
    "eagle3": {"method": "eagle3", "model": "RedHatAI/Qwen3-8B-speculator.eagle3", "num_speculative_tokens": 3},
}
NAMES = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")


def counters(llm):
    """vLLM's speculation counters so far: drafts, draft tokens, accepted tokens."""
    got = {n: 0 for n in NAMES}
    for m in llm.get_metrics():
        if m.name in got:
            got[m.name] += getattr(m, "value", 0) or 0
    return [got[n] for n in NAMES]


def drafter_packed(llm):
    """The drafter's layers packed by Glyd (glyd_words set), and its Linears, where the engine runs in this process."""
    try:
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        d = next(getattr(runner, n) for n in ("drafter", "speculator") if getattr(runner, n, None) is not None)
        model = d.model
    except Exception as e:  # (no drafter, or another engine layout)
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    mods = list(model.modules())
    return {"packed": sum(getattr(m, "glyd_words", None) is not None for m in mods),
            "linears": sum(type(m).__name__.endswith("Linear") for m in mods)}


def main():
    out, cfg = sys.argv[1], json.loads(sys.argv[2])
    from vllm import LLM, SamplingParams

    spec = None
    if cfg.get("spec"):
        spec = dict(SPEC[cfg["spec"]], **({"quantization": "glyd"} if cfg.get("draft_glyd") else {}))
    t = time.perf_counter()
    llm = LLM(model=cfg.get("model", "Qwen/Qwen3-8B"), quantization="glyd" if cfg["mode"] == "glyd" else None, dtype="bfloat16",
              gpu_memory_utilization=cfg.get("util", 0.9), max_model_len=4096, enforce_eager=cfg.get("eager", False), seed=0, speculative_config=spec,
              disable_log_stats=False, enable_prefix_caching=False, **({"max_num_batched_tokens": cfg["batched"]} if cfg.get("batched") else {}))
    res = {"cfg": cfg, "spec": spec, "load_s": round(time.perf_counter() - t, 1), "mixes": {}}
    vc = llm.llm_engine.vllm_config
    res["glyd"] = vc.additional_config.get("glyd") if isinstance(vc.additional_config, dict) else None
    if spec and spec["method"] != "ngram":
        res["drafter"] = drafter_packed(llm)
    greedy = SamplingParams(temperature=0, max_tokens=cfg.get("tokens", 400))
    chat = lambda p: llm.chat([{"role": "user", "content": p}], greedy, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0]
    for p in (EDIT[0][:200], CHAT[0]):  # (warm-up: the first requests' one-time costs)
        chat(p)
    for name, prompts in (("edit", EDIT), ("chat", CHAT)):
        c0, rows = counters(llm), []
        for p in prompts:
            t = time.perf_counter()
            o = chat(p)
            e2e = time.perf_counter() - t
            m = o.metrics
            rows.append({"tokens": list(o.outputs[0].token_ids), "e2e_s": e2e, "ttft_s": getattr(m, "first_token_latency", None) if m is not None else None})
        c1 = counters(llm)
        res["mixes"][name] = {"rows": rows, "drafts": c1[0] - c0[0], "draft_tokens": c1[1] - c0[1], "accepted": c1[2] - c0[2]}
        n, s = sum(len(r["tokens"]) for r in rows), sum(r["e2e_s"] for r in rows)
        print(f"{name}: {n} tokens in {s:.1f} s, {n / s:.1f} tokens/s; drafts {c1[0] - c0[0]}, accepted {c1[2] - c0[2]} of {c1[1] - c0[1]}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
