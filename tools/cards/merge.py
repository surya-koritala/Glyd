#!/usr/bin/env python3
"""Build or merge a model card's numbers (cards/<model>-<level>.json) from one pod's results directory (numbers_job.sh's files).
  merge.py CARD.json --results DIR --glyd 0.29.3 --commit abc1234 [--kl-jsonl FILE] [--kl-fp8-ref 0.00465]
  merge.py CARD.json --evals DIR... [--key evals|evals_bf16]     (eval_job.sh's evals.json of each shard: one entry per task, the values weighted by n; the key replaced on a rerun)
One GPU entry per name, replaced on a rerun; a kl value not given keeps the card's. merge.py --selftest checks it."""
import argparse, datetime, json, sys
from pathlib import Path

GIB_GB = 1.073741824  # (1 GiB in GB)


def last_json(text):
    rows = [l for l in text.splitlines() if l.startswith("{")]
    return json.loads(rows[-1]) if rows else None


def gpu_entry(name, bench, fits):
    return {"name": name, "tok_s": bench["decode_tokens_per_s"], "first_token_ms": bench["prefill_ms"],
            "mem_gib_4k": bench.get("gpu_mem_used_gib"), "fits_32k": fits == 0}


def merge(card, gpu, bench, size_bytes, kl, kl_fp8, glyd, commit, date):
    card = dict(card or {})
    card["size_gb"] = size_bytes / 1e9 if size_bytes else round(bench["weights_gib"] * GIB_GB, 3)
    card["gpus"] = [g for g in card.get("gpus", []) if g["name"] != gpu["name"]] + [gpu]
    card["kl_bf16"] = kl if kl is not None else card.get("kl_bf16")
    card["kl_fp8_ref"] = kl_fp8 if kl_fp8 is not None else card.get("kl_fp8_ref")
    card["date"], card["versions"] = date, {"glyd": glyd, "commit": commit}
    return {k: card[k] for k in ("size_gb", "gpus", "kl_bf16", "kl_fp8_ref", "date", "versions", "evals", "evals_bf16", "costs") if k in card}  # (costs: glyd-engine costs, docs/cost-profile.md)


def merge_evals(card, shards, key, date):
    """card with `key` set to one entry per (dataset, task_id) over the shards' lists: n summed, value the mean weighted by n."""
    by = {}
    for rows in shards:
        for e in rows:
            by.setdefault((e["dataset"], e["task_id"]), []).append(e)
    out = []
    for (ds, tid), es in by.items():
        assert all(e["harness"] == es[0]["harness"] and e["notes"] == es[0]["notes"] for e in es), f"{tid}: shards differ in harness or notes"
        n = sum(e["n"] for e in es)
        out.append({"dataset": ds, "task_id": tid, "value": round(sum(e["value"] * e["n"] for e in es) / n, 6), "n": n, "shards": len(es), "harness": es[0]["harness"], "notes": es[0]["notes"], "date": date})
    return {**card, key: out}


def selftest():
    b = {"decode_tokens_per_s": 40.5, "prefill_ms": 310.2, "weights_gib": 17.0, "gpu_mem_used_gib": 21.5}
    assert last_json('log\n{"a":1}\n{"decode_tokens_per_s":1}\n')["decode_tokens_per_s"] == 1 and last_json("none") is None
    c = merge(None, gpu_entry("RTX 4090", b, 0), b, None, None, 0.00465, "0.29.3", "abc", "2026-10-09")
    assert c["size_gb"] == round(17.0 * GIB_GB, 3) and c["kl_bf16"] is None and c["kl_fp8_ref"] == 0.00465
    assert c["gpus"] == [{"name": "RTX 4090", "tok_s": 40.5, "first_token_ms": 310.2, "mem_gib_4k": 21.5, "fits_32k": True}]
    b2 = dict(b, decode_tokens_per_s=50.0)
    c = merge(c, gpu_entry("L4", b, 3), b, 18e9, 0.02, None, "0.29.3", "abc", "2026-10-10")
    c = merge(c, gpu_entry("RTX 4090", b2, 0), b, 18e9, None, None, "0.30.0", "def", "2026-10-11")
    assert [g["name"] for g in c["gpus"]] == ["L4", "RTX 4090"] and c["gpus"][1]["tok_s"] == 50.0 and c["gpus"][0]["fits_32k"] is False
    assert c["size_gb"] == 18.0 and c["kl_bf16"] == 0.02 and c["kl_fp8_ref"] == 0.00465 and c["versions"] == {"glyd": "0.30.0", "commit": "def"}
    e = lambda v, n: {"dataset": "openai/gsm8k", "task_id": "gsm8k", "value": v, "n": n, "shard": "1/2", "harness": {"inspect_ai": "1"}, "notes": "x"}
    c = merge_evals(c, [[e(0.9, 100)], [e(0.6, 300)]], "evals", "2026-10-09")
    assert c["evals"][0]["value"] == 0.675 and c["evals"][0]["n"] == 400 and c["evals"][0]["shards"] == 2
    assert merge(c, gpu_entry("L4", b, 0), b, 18e9, None, None, "0.30.0", "d", "2026-10-12")["evals"] == c["evals"]  # (a numbers rerun keeps the evals)
    c["costs"] = {"resident_bytes": 1}
    assert merge(c, gpu_entry("L4", b, 0), b, 18e9, None, None, "0.30.0", "d", "2026-10-12")["costs"] == {"resident_bytes": 1}  # (and the cost profile)
    print("ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("card", type=Path); ap.add_argument("--results", type=Path)
    ap.add_argument("--evals", type=Path, nargs="+"); ap.add_argument("--key", default="evals", choices=("evals", "evals_bf16"))
    ap.add_argument("--glyd"); ap.add_argument("--commit")
    ap.add_argument("--kl-jsonl", type=Path); ap.add_argument("--kl-fp8-ref", type=float)
    a = ap.parse_args()
    if a.evals:
        card = merge_evals(json.loads(a.card.read_text()) if a.card.exists() else {}, [json.loads((d / "evals.json").read_text()) for d in a.evals], a.key, datetime.date.today().isoformat())
        a.card.parent.mkdir(parents=True, exist_ok=True)
        a.card.write_text(json.dumps(card, indent=2) + "\n")
        return print(f"{a.card}: {a.key} " + ", ".join(f"{e['task_id']} {e['value']} (n={e['n']})" for e in card[a.key]))
    if not (a.results and a.glyd and a.commit):
        ap.error("--results, --glyd and --commit (or --evals)")
    r = a.results
    bench = last_json((r / "bench.json").read_text()) if (r / "bench.json").exists() else None
    if not bench:
        sys.exit(f"{r}: no bench line (see job.log, bench.err)")
    name = (r / "gpu.txt").read_text().splitlines()[0].split(",")[0].strip()
    fits = int((r / "fits.txt").read_text().strip())
    size = int((r / "size.txt").read_text()) if (r / "size.txt").exists() else None
    kl = last_json(a.kl_jsonl.read_text())["kl_mean"] if a.kl_jsonl and a.kl_jsonl.exists() else None
    card = json.loads(a.card.read_text()) if a.card.exists() else None
    out = merge(card, gpu_entry(name, bench, fits), bench, size, kl, a.kl_fp8_ref, a.glyd, a.commit, datetime.date.today().isoformat())
    a.card.parent.mkdir(parents=True, exist_ok=True)
    a.card.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{a.card}: {name} {gpu_entry(name, bench, fits)['tok_s']} tok/s")


if __name__ == "__main__":
    main()
