#!/usr/bin/env python3
"""The record of the engine on glyd-v4 packs (each matrix in the smaller of Glyd's two layouts) and the library of C API 15: python record_layouts.py FETCHED_DIR OUT_DIR

It wrote benchmarks/gpu/engine-mvp-2026-10-03/layouts. The checks whose outputs it reads (`rowcheck`, `kvtest`) were commands of the dense engine of that release and were removed with it
(R3c, #139): the tool makes that record again from those runs' kept outputs, and reads nothing a current build prints.

FETCHED_DIR holds what the runs left on the box: t2b/ (the checks' outputs: check*.json, rowcheck*.out, kvtest8-after.out, bench8-*.out), t2b-hold11.out (the first-use run), t2b_pack.out (the packing), layout-counts.txt. OUT_DIR gets the README and the files
the numbers come from, cleaned of terminal codes; a word of the engine's own vocabulary in any of them stops the run (the record says what was measured, not how)."""
import json
import os
import re
import shutil
import sys

F, OUT = sys.argv[1], sys.argv[2]
BANNED = re.compile(r"\b(pool|arena|chunk|page|ring|slab|escape|void|raw)s?\b", re.I)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SHORT = [6, 13, 31, 5, 18, 9, 10, 16]  # the reference's prompts, in tokens
LONG = [1030, 2100, 3500]
os.makedirs(os.path.join(OUT, "checks"), exist_ok=True)


def path(*p):
    return os.path.join(F, *p)


def clean(text):
    text = ANSI.sub("", text).replace("\r", "\n")
    bad = BANNED.search(text)
    if bad:
        raise SystemExit(f"the word {bad[0]!r} in: {text[max(0, bad.start() - 60):bad.end() + 40]!r}")
    return text


def copy(src, name):
    text = clean(open(path(*src)).read())
    open(os.path.join(OUT, "checks", name), "w").write(text)
    return text


def compare(a, b):
    """Two check files' tokens and scores: tokens the same of all, and each prompt's largest difference of scores."""
    ja, jb = json.load(open(path("t2b", a))), json.load(open(path("t2b", b)))
    same = total = 0
    diffs = []
    for key, lengths in (("", SHORT), ("long_", LONG)):
        for i, (ta, tb, la, lb) in enumerate(zip(ja[key + "tokens"], jb[key + "tokens"], ja[key + "logprobs"], jb[key + "logprobs"])):
            total += len(ta)
            same += sum(p == q for p, q in zip(ta, tb))
            diffs.append((lengths[i], max(abs(p - q) for p, q in zip(la, lb))))
    return same, total, diffs


def differing(diffs):
    d = [f"{n:,} tokens: {x:.3f}" for n, x in diffs if x > 0]
    return ("; ".join(d) + f"; the other {len(diffs) - len(d)} prompts: identical") if d else f"all {len(diffs)} prompts: identical"


for name in sorted(os.listdir(path("t2b"))):
    if name.startswith("check") and name.endswith(".json"):
        shutil.copy(path("t2b", name), os.path.join(OUT, "checks", name))
for name in ("rowcheck8-before.out", "rowcheck8-after.out", "kvtest8-after.out", "bench8-before.out", "bench8-after.out"):
    copy(("t2b", name), name)
counts = clean(open(path("layout-counts.txt")).read())
open(os.path.join(OUT, "checks", "layout-counts.txt"), "w").write(counts)

rows = []
for title, a, b in (
    ("Qwen3-8B: the engine before (earlier library, tiered layout) against the engine now (glyd-v4)", "check8-before.json", "check8-after.json"),
    ("Qwen3-4B: the same", "check4-before.json", "check4-after.json"),
    ("Qwen3-8B in the 12-bit layout alone: earlier library against the new one", "check8-t12-old.json", "check8-t12-new.json"),
    ("Qwen3-8B, earlier library: the tiered layout against the 12-bit layout alone", "check8-before.json", "check8-t12-old.json"),
    ("Qwen3-8B, new library: glyd-v4 against the 12-bit layout alone", "check8-after.json", "check8-t12-new.json"),
):
    same, total, diffs = compare(a, b)
    rows.append(f"| {title} | {same:,} of {total:,} | {differing(diffs)} |")

hashes = {v: sorted(set(re.findall(r"logits hash ([0-9a-f]{16})", open(path("t2b", f"rowcheck8-{v}.out")).read()))) for v in ("before", "after")}
kv = re.search(r"in all: values checked (\d+), differing (\d+)", open(path("t2b", "kvtest8-after.out")).read())


def bench(v):
    j = json.loads(next(l for l in open(path("t2b", f"bench8-{v}.out")) if l.startswith("{")))
    return j


b0, b1 = bench("before"), bench("after")
hold = clean(open(path("t2b-hold11.out")).read())
first = hold[hold.index("-- run 1"):]
first = re.sub(r"once: \S+", "once: (the cache's directory)", first)
first = "\n".join(l for l in first.split("\n") if not l.startswith("Qwen3-8B: 10.3 GiB") and not re.match(r"\d\d:\d\d:\d\d ", l))
copy(("t2b_pack.out",), "pack.out")
open(os.path.join(OUT, "checks", "first-use.out"), "w").write(first)

readme = f"""# glyd-engine on glyd-v4 packs and the library of C API 15 (2026-10-04)

The library's smallest layout has a new format (glyd-v4): each matrix is held in the smaller of Glyd's two layouts, the tiered one or the 12-bit one where that takes fewer bytes, and a pack names its own. glyd-engine loads each pack in
the layout it names (a directory of glyd-v4 mixes them), packs the embedding in the smaller one too, and runs against the library of C API 15; a copy of the earlier tiered layout (glyd 0.27 and before, glyd-v1 and glyd-v2), which the library no longer
reads, is packed again, once, the first time it is used from the engine's cache, with its KV tables kept, and refused in words where it is a directory the user keeps. Everything here ran on the L4 (AWS g6.4xlarge).

## What the directories hold

```
{counts.strip()}
```

## Against the engine before the change

`check` runs the reference's 8 short prompts (5 to 31 tokens) and 3 long ones (1,030, 2,100 and 3,500 tokens), 64 greedy tokens each, and writes the tokens and the score (log probability) of each; `checks/check*.json`.

| the two runs | tokens the same | each prompt's largest difference of scores |
| :--- | ---: | :--- |
{chr(10).join(rows)}

- The tokens are the same in all of them. The scores of the prompts of 5 to 16 tokens and of 1,030 tokens and more are identical, to six digits, between the engine before and the engine now; the two prompts of 18 and 31 tokens differ, by at most 0.10 on the 8B and 0.12 on
  the 4B. The comparisons below them say where from: with the earlier library alone, a matrix held in the 12-bit layout instead of the tiered one changes the scores of those two prompts (0.19 and 0.10), and the 12-bit layout alone
  gives the same scores on the earlier library and on the new one. Six of the 8B's matrices (the gate and the down of layers 1 to 3) are in the 12-bit layout in glyd-v4, five of the 4B's: what changed the two prompts' scores is the layout a matrix is held in, for prompts of 17 to 64 tokens.
- One sequence's logits in a decode step with 0 to 7 others (`checks/rowcheck8-*.out`): the same bit for bit at every number of rows and in either row, and the same hash before ({', '.join(hashes['before'])}) and now ({', '.join(hashes['after'])}).
- The cache's known-answer test on the new library (`checks/kvtest8-after.out`): {int(kv[1]):,} values written and read back, **{kv[2]} differ**.
- A 1,024-token prompt and 64 tokens (`checks/bench8-*.out`, three timed runs): the weights take {b0['weights_gib']:.3f} GiB before and {b1['weights_gib']:.3f} now; a decode step {b0['decode_ms_mean']:.2f} ms before ({b0['decode_tokens_per_s']:.2f} tokens a second) and {b1['decode_ms_mean']:.2f} ms now ({b1['decode_tokens_per_s']:.2f}); the prompt {b0['prefill_ms']:.0f} ms before and {b1['prefill_ms']:.0f} ms now.

## A copy of the earlier layout, used for the first time

A copy of Qwen3-8B in the engine's cache, as an earlier run left it (glyd-v1, with its KV tables), run with `glyd-engine run Qwen/Qwen3-8B --context 4096` twice (`checks/first-use.out`):

```
{first.strip()}
```

The first run packs the model again (the same packing as `glyd-engine pack`, which took 30 s for the 8B when run on its own a few minutes before: `checks/pack.out`), keeps the tables, and answers; the second finds it made.
"""
bad = BANNED.search(readme)
if bad:
    raise SystemExit(f"README: the word {bad[0]!r}: {readme[max(0, bad.start() - 80):bad.end() + 40]!r}")
open(os.path.join(OUT, "README.md"), "w").write(readme)
print(readme)
