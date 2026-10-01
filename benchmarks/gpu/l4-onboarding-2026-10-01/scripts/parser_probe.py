"""How vLLM 0.30's qwen3 reasoning parser splits a stream whose end-of-thinking token arrives alone or with its neighbours."""
from transformers import AutoTokenizer
from vllm.reasoning import ReasoningParserManager

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
text = "<think>\nOkay, so the answer is short.\n</think>\n\nLossless compression is a method."
ids = tok.encode(text, add_special_tokens=False)
print("token pieces:", [tok.decode([i]) for i in ids])


def feed(groups):
    P = ReasoningParserManager.get_reasoning_parser("qwen3")(tok)  # (stateful: one a stream)
    prev_text, prev_ids, out = "", [], []
    for g in groups:
        delta_text = tok.decode(g)
        cur_text, cur_ids = prev_text + delta_text, prev_ids + g
        d = P.extract_reasoning_streaming(prev_text, cur_text, delta_text, prev_ids, cur_ids, g)
        out.append((delta_text, None if d is None else (d.reasoning, d.content)))
        prev_text, prev_ids = cur_text, cur_ids
    return out


k = ids.index(tok.convert_tokens_to_ids("</think>"))
for lo, hi in ((k, k + 1), (k - 1, k + 2), (k, k + 3), (k, k + 2), (k - 1, k + 1), (k + 1, k + 3), (k + 2, k + 4)):
    groups = [[i] for i in ids[:lo]] + [ids[lo:hi]] + [[i] for i in ids[hi:]]
    res = feed(groups)
    print(f"--- tokens {lo}..{hi - 1} in one delta {[tok.decode([i]) for i in ids[lo:hi]]}")
    print("   the deltas around it:", [r for r in res if lo - 1 <= res.index(r) <= hi + 1])
    print("   reasoning:", repr("".join((r[1][0] or "") for r in res if r[1])), " content:", repr("".join((r[1][1] or "") for r in res if r[1])))
