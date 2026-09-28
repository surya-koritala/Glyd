"""bf16 against FP8 and Glyd on one model: what each changes in the model's answers.

    python fp8_compare.py MODE OUT.pt      MODE: bf16 | bf16-eager | fp8 | glyd | glyd-exact
    python fp8_compare.py --compare DIR    the table, each mode against bf16

bf16 and Glyd load Qwen/Qwen3-4B-Instruct-2507; fp8 loads Qwen's own FP8 release of it
(Qwen/Qwen3-4B-Instruct-2507-FP8: e4m3 weights in 128x128 blocks, activations quantized per token as they run).
Per mode: the weights' GPU memory; perplexity on WikiText-2's test text (windows of 1024 tokens) and the top token at
each position; greedy continuations of fixed prompts; MMLU (cais/mmlu test, a fixed shuffle, 0-shot, by the answer
letter's logit, as gpu/e2e.py --mmlu)."""
import os, sys, torch

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
WINDOWS, SEQ, MMLU_N, NEW = 64, 1024, 1000, 64
PROMPTS = [
    "Explain why the sky is blue.", "Write a haiku about the ocean.", "What is the capital of Australia?",
    "Summarize the plot of Hamlet in three sentences.", "How does a transistor work?", "Give me a recipe for pancakes.",
    "What are the main causes of inflation?", "Translate 'good morning, how are you?' into French.",
    "Write a Python function that reverses a linked list.", "Why do cats purr?", "What is 17 times 23? Show your work.",
    "Describe the water cycle.", "What is the difference between TCP and UDP?", "Tell me a short story about a robot.",
    "What were the causes of World War I?", "How do vaccines train the immune system?", "List five prime numbers above 100.",
    "What is gradient descent?", "Explain the rules of chess briefly.", "Why is the Great Barrier Reef important?",
    "Write a limerick about a programmer.", "What is a black hole?", "How do I make a good cup of coffee?",
    "Compare Python and Rust in two paragraphs.", "What does DNA stand for and what does it do?",
    "Solve for x: 3x + 7 = 22.", "What is the Pythagorean theorem?", "Give three tips for public speaking.",
    "Explain what an API is to a child.", "What causes earthquakes?", "Write a SQL query that counts rows per country.",
    "Who painted the Mona Lisa and when?", "What is photosynthesis?", "How does compound interest work?",
    "Describe the city of Tokyo.", "What is the speed of light?", "Explain recursion with an example.",
    "What are the benefits of regular exercise?", "How do airplanes stay in the air?", "What is a haiku?",
    "Write an email asking for a day off.", "What is the tallest mountain on Earth?", "Explain supply and demand.",
    "How do I sort a list in JavaScript?", "What is the theory of evolution?", "Name the planets of the solar system.",
    "What is machine learning?", "Why do we have seasons?", "Give a short biography of Marie Curie.",
    "What is the difference between weather and climate?",
]


def load(mode):
    from transformers import AutoModelForCausalLM
    if mode in ("bf16", "bf16-eager"):  # bf16-eager: bf16 through another attention kernel, bf16's own variation
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                    attn_implementation="eager" if mode == "bf16-eager" else None)
    if mode == "fp8":
        return AutoModelForCausalLM.from_pretrained(MODEL + "-FP8", dtype="auto", device_map="cuda")
    import glyd
    return glyd.from_pretrained(MODEL, exact=mode == "glyd-exact")


def run(mode, out):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL)
    torch.cuda.reset_peak_memory_stats()
    model = load(mode).eval()
    res = {"mode": mode, "mem_gb": torch.cuda.memory_allocated() / 1e9}
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][: WINDOWS * SEQ].view(WINDOWS, SEQ)
    nll, tops = 0.0, []
    with torch.no_grad():
        for w in ids:
            x = w[None].cuda()
            lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
            nll += -lp.gather(1, x[0, 1:, None]).sum().item()
            tops.append(lp.argmax(-1).cpu())
        res["ppl"] = float(torch.exp(torch.tensor(nll / (WINDOWS * (SEQ - 1)))))
        res["tops"] = torch.stack(tops)
        gens = []
        for p in PROMPTS:
            x = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
            y = model.generate(x, max_new_tokens=NEW, do_sample=False)
            gens.append(y[0, x.shape[1]:].cpu())
        res["gens"] = gens
        qs = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(MMLU_N))
        letters = [tok(f" {c}", add_special_tokens=False).input_ids[-1] for c in "ABCD"]
        picks, right = [], 0
        for q in qs:
            prompt = f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}.\n\n{q['question']}\n"
            prompt += "".join(f"{c}. {a}\n" for c, a in zip("ABCD", q["choices"])) + "Answer:"
            x = tok(prompt, return_tensors="pt").input_ids.cuda()
            pick = int(model(x, logits_to_keep=1).logits[0, -1, letters].argmax())
            picks.append(pick)
            right += pick == q["answer"]
        res["mmlu_picks"], res["mmlu_acc"] = torch.tensor(picks), right / MMLU_N
    torch.save(res, out)
    print(f"{mode}: {res['mem_gb']:.2f} GB, perplexity {res['ppl']:.4f}, MMLU {100 * res['mmlu_acc']:.2f}%")


def compare(d):
    r = {m: torch.load(os.path.join(d, f"{m}.pt"), weights_only=True) for m in ("bf16", "bf16-eager", "glyd-exact", "glyd", "fp8") if os.path.exists(os.path.join(d, f"{m}.pt"))}
    b = r["bf16"]
    print(f"{'':12} {'weights GB':>10} {'perplexity':>11} {'top token = bf16':>17} {'answers = bf16':>15} {'MMLU':>7} {'MMLU answers changed':>21}")
    for m, x in r.items():
        top = 100 * (x["tops"] == b["tops"]).float().mean().item()
        same = sum(bool(torch.equal(g, h)) for g, h in zip(x["gens"], b["gens"]))
        changed = int((x["mmlu_picks"] != b["mmlu_picks"]).sum())
        print(f"{m:12} {x['mem_gb']:>10.2f} {x['ppl']:>11.4f} {top:>16.2f}% {same:>9} of {len(b['gens']):<3} {100 * x['mmlu_acc']:>6.2f}% {changed:>12} of {len(b['mmlu_picks'])}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2])
    else:
        run(sys.argv[1], sys.argv[2])
