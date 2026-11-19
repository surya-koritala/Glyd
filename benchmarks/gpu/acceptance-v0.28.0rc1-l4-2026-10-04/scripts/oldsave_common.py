"""What the old-save check asks of a model: Qwen3-0.6B, four short prompts, greedy, 32 new tokens at most."""
MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 17 plus 25? Answer with the number only.",
    "Complete the sentence in at most ten words: The quick brown fox",
    "Write one sentence about the sea.",
]
NEW = 32


def versions():
    from importlib.metadata import version
    return {"glyd": version("glyd"), "glyd-gpu": version("glyd-gpu")}


def answers(model, tok):
    """Greedy answers (token ids and text) of a model to PROMPTS, the chat template with thinking off."""
    res = []
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to("cuda:0")
        out = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=NEW, pad_token_id=tok.pad_token_id)
        new = out[0, enc["input_ids"].shape[1]:].tolist()
        res.append({"prompt": p, "tokens": new, "text": tok.decode(new, skip_special_tokens=True)})
    return res


def same(a, b):
    return [x["tokens"] == y["tokens"] for x, y in zip(a, b)]
