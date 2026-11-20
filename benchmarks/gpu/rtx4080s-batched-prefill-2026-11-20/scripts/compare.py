#!/usr/bin/env python3
"""Two files of tokens_check.py side by side: per reply, the first token (which the prompt's pass gives) and its logprob, the tokens, and the logprobs.  compare.py A.json B.json"""
import json
import sys

A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for phase in ("cold", "warm"):
    a, b = A[phase], B[phase]
    n = len(a)
    first = sum(1 for x, y in zip(a, b) if x["lp"][0] == y["lp"][0])
    exact = sum(1 for x, y in zip(a, b) if x == y)
    div, worst = [], 0.0
    for x, y in zip(a, b):
        tx, ty = [t for t, _ in x["lp"]], [t for t, _ in y["lp"]]
        k = next((j for j, (p, q) in enumerate(zip(tx, ty)) if p != q), None)
        div.append(-1 if k is None else k)
        worst = max([worst] + [abs(p[1] - q[1]) for p, q in zip(x["lp"][:k], y["lp"])])
    print(f"{phase}: first token and logprob bit-for-bit {first}/{n}; whole reply bit-for-bit {exact}/{n}; same tokens {sum(1 for d in div if d < 0)}/{n} "
          f"(first differing token, -1 none: {div}); largest logprob difference before it {worst:.2e}")
