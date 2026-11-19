"""A chat template as transformers renders it, for the Rust renderer's tests (src/chat.rs): the template, cases (messages, options) and their text and token ids.

    python chat_fixtures.py MODEL_DIR ../tests/fixtures/chat_qwen3.json        (where transformers and the kvcal corpus are)
"""
import json
import sys

from transformers import AutoTokenizer

from glyd_gpu import kvcal


def main(model, out):
    tok = AutoTokenizer.from_pretrained(model)
    cases = []
    chats = [json.loads(body) for kind, name, body in kvcal.corpus() if kind == "chat"]
    extra = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "<think>\nplan\n</think>\n\nHello there"}, {"role": "user", "content": "And now?"}],
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Again"}, {"role": "assistant", "content": "<think>\nx\n</think>\n\nSure"}],
        [{"role": "user", "content": "Très bien — 你好"}],
    ]
    for msgs in chats + extra:
        for gen, think in ((False, None), (True, None), (True, False), (True, True)):
            kw = {} if think is None else {"enable_thinking": think}
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen, **kw)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
            cases.append({"messages": msgs, "gen": gen, "thinking": think, "text": text, "ids": ids})
    json.dump({"template": tok.chat_template, "cases": cases}, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
    print(out, len(cases), "cases")


if __name__ == "__main__":
    main(*sys.argv[1:3])
