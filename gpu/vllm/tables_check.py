"""The embedding and the LM head packed under vLLM, on tiny random Qwen3-architecture models (two layers, a hidden size of 1,024) made here for the cases
the real checkpoints lack, each in a vLLM of its own, over --tp N GPUs (tensor parallel; 1 by default):

- tt: tied, a vocabulary of 32,064 (vLLM shards it over 2 GPUs into shards of 16,032 rows, not a multiple of 64);
- ut: untied, 32,000 rows; tc_eq and tc_ne: the config ties them and the checkpoint holds an lm_head too (vLLM builds them apart), the same as the
  embedding in tc_eq and another in tc_ne;
- to and uo: tied and untied with a vocabulary of 32,001, which vLLM pads to 32,064 (the checkpoint's tables hold 32,001 rows: on 1 GPU the weight loads
  fewer rows than the layer holds, and over 2 the shards are 16,032 rows, the second loading 15,969).

Per model: bf16 (eager) and Glyd (the layout the GPU's best, compiled with CUDA graphs): every table packed (one pack where tied, or the checkpoint's
two are the same), every lookup of random ids and of the shard's ends the checkpoint's own row bit for bit on each rank, each product within 1e-2 of its
matrix, the same bits twice, the first token of every prompt bf16's where bf16's margin is clear, and prompt logprobs within 0.05 on average of bf16's;
and Glyd's exact mode (eager) bf16 eager's tokens, logprobs and prompt_logprobs bit for bit (a tied model's one table stays as vLLM runs it, an untied
model's embedding is packed and its LM head is not).

    python tables_check.py [--tp N] [--out DIR] [MODEL ...]      (default: all)

Needs vLLM (glyd[vllm]) and the library (GLYD_GPU_LIB); transformers (vLLM's) makes the models. Its results, a JSON file and a log a run, in --out."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"tt": (32064, True, None), "ut": (32000, False, None), "tc_eq": (32000, True, "equal"), "tc_ne": (32000, True, "other"),
          "to": (32001, True, None), "uo": (32001, False, None)}  # vocabulary, tied, the checkpoint's lm_head
PROMPTS = (7, 40, 129, 300)  # tokens
NEW = 24


def make(out, name):
    """The model's directory in out (made once)."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    V, tie, extra = MODELS[name]
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    cfg = Qwen3Config(vocab_size=V, hidden_size=1024, intermediate_size=3072, num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                      max_position_embeddings=4096, tie_word_embeddings=tie, bos_token_id=1, eos_token_id=2, pad_token_id=0, architectures=["Qwen3ForCausalLM"])
    cfg.save_pretrained(d)  # (the architecture named: vLLM runs a language model, not the bare model's pooling)
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return d
    torch.manual_seed(0)
    sd = {k: v.contiguous().clone().to(torch.bfloat16) for k, v in Qwen3ForCausalLM(cfg).state_dict().items()}
    if tie:
        sd.pop("lm_head.weight", None)
        if extra:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone() if extra == "equal" else (torch.randn(V, 1024) * 0.02).to(torch.bfloat16)
    elif "lm_head.weight" not in sd:
        sd["lm_head.weight"] = (torch.randn(V, 1024) * 0.02).to(torch.bfloat16)
    save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    return d


def child(model, mode, path, tp):
    """One vLLM instance: tokens, logprobs and prompt logprobs of PROMPTS (token ids), and the tables."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sys.path.insert(0, HERE)
    from check_vllm import _tables

    os.environ["CHECK_VLLM_CKPT"] = model
    V = json.load(open(os.path.join(model, "config.json")))["vocab_size"]
    kw = dict(model=model, tokenizer=None, skip_tokenizer_init=True, dtype="bfloat16", seed=0, max_model_len=2048, gpu_memory_utilization=0.3, enable_prefix_caching=False,
              tensor_parallel_size=tp, enforce_eager=mode != "glyd")
    if mode != "bf16":
        kw.update(quantization="glyd", additional_config={"glyd": {"kv": "off", **({"exact": True} if mode == "exact" else {})}})
    llm = LLM(**kw)
    g = torch.Generator().manual_seed(1)
    prompts = [TokensPrompt(prompt_token_ids=torch.randint(100, V - 100, (n,), generator=g).tolist()) for n in PROMPTS]
    res = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=NEW, logprobs=2, prompt_logprobs=1, ignore_eos=True), use_tqdm=False)
    out = {"mode": mode, "tp": tp, "tokens": [list(o.outputs[0].token_ids) for o in res],
           "logprobs": [[d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)] for o in res],
           "margin": [[abs(sorted((x.logprob for x in d.values()), reverse=True)[0] - sorted((x.logprob for x in d.values()), reverse=True)[1]) if len(d) > 1 else 9.0 for d in o.outputs[0].logprobs] for o in res],
           "prompt_logprobs": [[None if d is None else d[t].logprob for d, t in zip(o.prompt_logprobs, o.prompt_token_ids)] for o in res]}
    if mode != "bf16":
        ranks = llm.apply_model(_tables)
        out["tables"] = dict(ranks[0], ok=all(r["ok"] for r in ranks), ranks=len(ranks))
    json.dump(out, open(path, "w"))


def run(model, mode, out, tp, name):
    path = os.path.join(out, f"{name}-{mode}.json")
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", GLYD_KV="off")  # (as check_vllm's runs; over several GPUs its workers are processes of their own)
    if tp > 1:
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # (LLM.apply_model sends its function to the workers)
    with open(os.path.join(out, f"{name}-{mode}.log"), "w") as log:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", model, mode, path, str(tp)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    if r.returncode or not os.path.exists(path):
        raise RuntimeError(f"{name} {mode}: exit {r.returncode}; see {name}-{mode}.log")
    return json.load(open(path))


def main():
    a = sys.argv[1:]
    if a[:1] == ["--child"]:
        return child(a[1], a[2], a[3], int(a[4]))
    tp = int(a[a.index("--tp") + 1]) if "--tp" in a else 1
    out = a[a.index("--out") + 1] if "--out" in a else "tables_check_results"
    names = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--tp", "--out"))] or list(MODELS)
    os.makedirs(out, exist_ok=True)
    report, failed = [], []

    def check(ok, what):
        report.append(("PASS " if ok else "FAIL ") + what)
        print(report[-1], flush=True)
        if not ok:
            failed.append(what)

    for name in names:
        V, tie, extra = MODELS[name]
        d = make(out, name)
        tag = f"{name}{f' tp{tp}' if tp > 1 else ''}"
        bf = run(d, "bf16", out, tp, name)
        g = run(d, "glyd", out, tp, name)
        x = run(d, "exact", out, tp, name)
        t, e = g["tables"], x["tables"]
        packs = {r["buffer"] for r in t["tables"] if r["packed"]}
        worst = max(r["worst_rel_error"] for r in t["tables"] if r["packed"])
        check(t["ok"] and all(r["packed"] for r in t["tables"]) and len(packs) == t["expect_packs"],
              f"{tag}: the embedding and the LM head packed ({len(packs)} pack{'s' if len(packs) != 1 else ''}, {t['expect_packs']} expected; shard rows {sorted({r['shard_rows'] for r in t['tables']})}), every lookup the checkpoint's row bit for bit on {t['ranks']} rank{'s' if t['ranks'] != 1 else ''}, each product within 1e-2 ({worst:.1e}) and the same bits twice")
        first = [(a_, b_) for a_, b_, m in zip(bf["tokens"], g["tokens"], [mm[0] for mm in bf["margin"]]) if m > 0.1]
        check(all(a_[0] == b_[0] for a_, b_ in first), f"{tag}: the first token of every prompt bf16's where bf16's margin is over 0.1 ({len(first)} of {len(PROMPTS)})")
        dd = [abs(u - v) for p_, q_ in zip(bf["prompt_logprobs"], g["prompt_logprobs"]) for u, v in zip(p_[1:], q_[1:])]
        check(sum(dd) / len(dd) < 0.05, f"{tag}: prompt logprobs within {sum(dd) / len(dd):.4f} on average of bf16's (at most 0.05; the largest {max(dd):.3f})")
        same = x["tokens"] == bf["tokens"] and x["logprobs"] == bf["logprobs"] and x["prompt_logprobs"] == bf["prompt_logprobs"]
        tied = e["config_ties"]  # (the config ties them: one table, or the checkpoint's own copy of it, which vLLM builds apart: exact keeps both as vLLM runs them)
        check(same and e["ok"] and all(r["packed"] != (r["head"] or tied) for r in e["tables"]),
              f"{tag}: exact: bf16 eager's tokens, logprobs and prompt_logprobs bit for bit; {'the tied model stays as vLLM runs it' if tied else 'the embedding packed, the LM head as vLLM runs it'} ({[(r['name'], r['method']) for r in e['tables']]})")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"tables_check: {'all passed' if not failed else f'{len(failed)} failed'} ({len(report)} checks)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
