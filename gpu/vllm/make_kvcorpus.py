"""Builds glyd-gpu-py/glyd_gpu/kvcorpus.bin (the text zlib-compressed: the wheel ships it so), the text the lossless KV cache's base tables are calibrated on
(glyd_gpu.kvcal: a model run on it once, its keys and values' exponents counted a layer, head and channel). The file is
committed; this is how it was made, and every part of it is public domain or ours:

  prose   Project Gutenberg's A Tale of Two Cities (Charles Dickens, ebook 98), The Adventures of Sherlock Holmes
          (Arthur Conan Doyle, 1661), On the Origin of Species (Charles Darwin, 1228) and Pride and Prejudice (Jane
          Austen, 1342): public domain in the United States; tens of thousands of characters from each, unwrapped into
          paragraphs, without the Project Gutenberg header and licence text
  docs    excerpts of this repository's own documents (docs/engineering-notes.md, gpu/vllm/README.md)
  code    short Python and C modules written for this file
  chat    short conversations written for this file
  table   a CSV table and JSON records generated from fixed seeds

About 40,000 tokens in all: the tables are calibrated over the positions a model serves (a key's exponents drift with its position under
rotary embeddings, so text of a few thousand tokens calibrates the first few thousand positions only): kvcal runs the sections in an order
that puts every kind at every depth, as one sequence of up to 32,768 tokens.

    python make_kvcorpus.py [OUT]          (default: next to glyd_gpu/kvcal.py; the Gutenberg texts are fetched; OUT ending in .txt: the text itself)

The file is sections, each a line "@@@ kind name" and its body; kvcal runs each as a prompt (a chat section's body is a JSON
list of messages, put through the model's chat template where it has one).
"""
import json
import os
import random
import re
import sys
import zlib
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GUTENBERG = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"


def gutenberg(n, start, chars):
    """About `chars` characters of Project Gutenberg ebook n from the line that begins with `start`, its hard-wrapped lines joined
    into paragraphs, cut at the end of a sentence."""
    t = urllib.request.urlopen(GUTENBERG.format(n=n), timeout=60).read().decode("utf-8")
    t = t[t.index("*** START OF THE PROJECT GUTENBERG EBOOK") :]
    t = t[t.index(start) :]
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", t)]
    out, size = [], 0
    for p in paras:
        if not p:
            continue
        out.append(p)
        size += len(p) + 2
        if size >= chars:
            break
    text = "\n\n".join(out)
    cut = max(text.rfind(". "), text.rfind('." '), text.rfind(".”"))
    return text[: cut + 1] if cut > chars // 2 else text


def docs(path, start, chars):
    t = open(os.path.join(ROOT, path)).read()
    t = t[t.index(start) :]
    return t[:chars].rsplit("\n", 1)[0]


CODE = [
    ("python", '''from __future__ import annotations

import argparse
import bisect
import collections
import csv
import sys


class LRUCache:
    """A least recently used cache: a get or a put makes its key the newest; over capacity the oldest goes."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: collections.OrderedDict[str, bytes] = collections.OrderedDict()

    def get(self, key: str) -> bytes | None:
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: str, value: bytes) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


def parse_size(text: str) -> int:
    """Bytes in a size such as 512, 4k, 16M or 2G (powers of 1024)."""
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    text = text.strip().lower()
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile_rank(sorted_values: list[float], x: float) -> float:
    """The share of the values below x, by binary search."""
    return bisect.bisect_left(sorted_values, x) / max(1, len(sorted_values))


def summarize(path: str, column: str) -> dict[str, float]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    values = [float(r[column]) for r in rows if r.get(column)]
    if not values:
        raise SystemExit(f"no values in column {column!r} of {path}")
    return {"count": len(values), "min": min(values), "max": max(values), "median": median(values), "mean": sum(values) / len(values)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a numeric column of a CSV file.")
    parser.add_argument("path")
    parser.add_argument("column")
    parser.add_argument("--cache", default="64k", help="size of the read cache")
    args = parser.parse_args(argv)
    cache = LRUCache(max(1, parse_size(args.cache) // 4096))
    cache.put(args.path, b"")
    for name, value in summarize(args.path, args.column).items():
        print(f"{name:>8}: {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''),
    ("c", '''#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* A ring buffer of bytes: push appends, pop takes the oldest; both return how many bytes they moved. */
typedef struct {
    uint8_t* data;
    size_t capacity, head, used;
} ring;

size_t ring_push(ring* r, const uint8_t* src, size_t n) {
    size_t room = r->capacity - r->used;
    if (n > room) n = room;
    size_t tail = (r->head + r->used) % r->capacity;
    size_t first = r->capacity - tail < n ? r->capacity - tail : n;
    memcpy(r->data + tail, src, first);
    memcpy(r->data, src + first, n - first);
    r->used += n;
    return n;
}

size_t ring_pop(ring* r, uint8_t* dst, size_t n) {
    if (n > r->used) n = r->used;
    size_t first = r->capacity - r->head < n ? r->capacity - r->head : n;
    memcpy(dst, r->data + r->head, first);
    memcpy(dst + first, r->data, n - first);
    r->head = (r->head + n) % r->capacity;
    r->used -= n;
    return n;
}

uint32_t adler32(const uint8_t* p, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; i++) {
        a = (a + p[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}
'''),
]

CODE.append(("python-lexer", '''import re
from dataclasses import dataclass

TOKEN = re.compile(
    r"""\\s*(?:(?P<number>\\d+(?:\\.\\d+)?)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<string>"(?:[^"\\\\]|\\\\.)*")|(?P<op>[-+*/()=<>!,;{}\\[\\]]+))"""
)


@dataclass
class Token:
    kind: str
    text: str
    pos: int


def tokenize(source: str):
    """Yield the tokens of a small expression language; anything else is a syntax error that names its position."""
    pos = 0
    while pos < len(source):
        if source[pos:].strip() == "":
            return
        m = TOKEN.match(source, pos)
        if not m:
            raise SyntaxError(f"unexpected character {source[pos]!r} at {pos}")
        kind = m.lastgroup
        yield Token(kind, m.group(kind), m.start(kind))
        pos = m.end()


class Parser:
    """Precedence climbing over the tokens: numbers, names, calls and the binary operators + - * /."""

    PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}

    def __init__(self, source: str) -> None:
        self.tokens = list(tokenize(source))
        self.i = 0

    def peek(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expression(self, min_prec: int = 1):
        left = self.atom()
        while (tok := self.peek()) and tok.text in self.PRECEDENCE and self.PRECEDENCE[tok.text] >= min_prec:
            op = self.take().text
            right = self.expression(self.PRECEDENCE[op] + 1)
            left = (op, left, right)
        return left

    def atom(self):
        tok = self.take()
        if tok.kind == "number":
            return float(tok.text)
        if tok.kind == "name":
            if (nxt := self.peek()) and nxt.text == "(":
                self.take()
                args = []
                while self.peek() and self.peek().text != ")":
                    args.append(self.expression())
                    if self.peek() and self.peek().text == ",":
                        self.take()
                self.take()
                return ("call", tok.text, args)
            return ("name", tok.text)
        if tok.text == "(":
            inner = self.expression()
            self.take()
            return inner
        raise SyntaxError(f"unexpected {tok.text!r} at {tok.pos}")


def evaluate(node, env):
    if isinstance(node, float):
        return node
    kind = node[0]
    if kind == "name":
        return env[node[1]]
    if kind == "call":
        return env[node[1]](*(evaluate(a, env) for a in node[2]))
    op, a, b = node
    x, y = evaluate(a, env), evaluate(b, env)
    return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[op]
'''))

CHAT = [
    [
        {"role": "user", "content": "What is lossless compression, and how is it different from quantization?"},
        {"role": "assistant", "content": "Lossless compression stores the same bits in fewer bits: decoding gives back exactly what went in, every byte. Quantization changes the values themselves, for example rounding 16-bit weights to 4 bits, so the result is close to the original but not equal to it. A lossless format can be checked with a hash; a quantized one has to be checked by how the model behaves."},
        {"role": "user", "content": "Why do bf16 weights compress at all? They look like noise."},
        {"role": "assistant", "content": "Only part of a bf16 number is noise. Its 7 mantissa bits are close to uniformly random, but its 8 exponent bits are not: trained weights sit in a narrow range, so a handful of exponent values cover almost every weight. The exponent can be coded in about 3 bits instead of 8, which brings a weight from 16 bits to about 11."},
    ],
    [
        {"role": "user", "content": "Write a shell one-liner that lists the five largest files under the current directory."},
        {"role": "assistant", "content": "```bash\nfind . -type f -printf '%s %p\\n' | sort -nr | head -5\n```\n\nfind prints each file's size in bytes and its path, sort -nr orders the lines by size, largest first, and head -5 keeps the first five. On a system whose find has no -printf, `find . -type f -exec ls -l {} + | sort -k5 -nr | head -5` does the same."},
        {"role": "user", "content": "Now only files changed in the last week."},
        {"role": "assistant", "content": "Add a time test to find: `find . -type f -mtime -7 -printf '%s %p\\n' | sort -nr | head -5`. The test -mtime -7 keeps files modified less than seven days ago."},
    ],
    [
        {"role": "user", "content": "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?"},
        {"role": "assistant", "content": "Add the hours first: 3:40 pm plus 2 hours is 5:40 pm. Then add the 35 minutes: 5:40 plus 20 minutes is 6:00 pm, and the remaining 15 minutes make 6:15 pm. The train arrives at 6:15 pm."},
    ],
]


def table(rows_n):
    """A CSV of orders, seeded."""
    r = random.Random(0)
    first = ["Ada", "Bruno", "Chen", "Dalia", "Emil", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kira", "Luca", "Mira", "Noor", "Omar", "Pia"]
    last = ["Ansari", "Berg", "Castillo", "Dubois", "Eriksen", "Fischer", "Gupta", "Haddad", "Ito", "Jovanovic", "Kaya", "Lindqvist", "Moreau", "Nakamura"]
    city = ["Lisbon", "Osaka", "Toronto", "Nairobi", "Hamburg", "Bogota", "Pune", "Oslo", "Seville", "Perth"]
    item = ["USB-C cable", "desk lamp", "notebook", "water bottle", "keyboard", "headphones", "webcam", "monitor stand", "mouse", "charger"]
    rows = ["order_id,customer,city,item,quantity,unit_price,total,shipped,status"]
    for i in range(1, rows_n + 1):
        q, p = r.randint(1, 9), round(r.uniform(3, 120), 2)
        day = f"2025-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
        rows.append(f"{10000 + i * 7},{r.choice(first)} {r.choice(last)},{r.choice(city)},{r.choice(item)},{q},{p:.2f},{q * p:.2f},{day},{r.choice(['delivered', 'delivered', 'in transit', 'returned', 'pending'])}")
    return "\n".join(rows)


def records(n):
    """JSON records, seeded."""
    r = random.Random(1)
    tags = ["gpu", "compression", "serving", "kernel", "benchmark", "format", "cache", "latency"]
    out = []
    for i in range(1, n + 1):
        out.append({"id": i, "title": f"Run {i}: {r.choice(['decode', 'prefill', 'pack', 'verify'])} on {r.choice(['L4', 'A10', 'A100', 'H100'])}", "tokens": r.randint(128, 8192),
                    "ok": r.random() > 0.15, "seconds": round(r.uniform(0.2, 90.0), 3), "tags": r.sample(tags, 3),
                    "gpu": {"memory_gib": r.choice([22, 24, 40, 80]), "clock_mhz": r.randint(1200, 2100), "temp_c": r.randint(40, 83)}})
    return json.dumps(out, indent=2)


def build():
    parts = [
        ("prose", "a-tale-of-two-cities-1", gutenberg(98, "It was the best of times", 24000)),
        ("prose", "sherlock-holmes-1", gutenberg(1661, "To Sherlock Holmes she is always", 20000)),
        ("prose", "origin-of-species-1", gutenberg(1228, "When we look to the individuals of the same variety", 20000)),
        ("prose", "pride-and-prejudice-1", gutenberg(1342, "It is a truth universally acknowledged", 20000)),
        ("prose", "a-tale-of-two-cities-2", gutenberg(98, "A wonderful fact to reflect upon", 16000)),
        ("prose", "sherlock-holmes-2", gutenberg(1661, "II. THE RED-HEADED LEAGUE", 14000)),
        ("prose", "origin-of-species-2", gutenberg(1228, "CHAPTER III.", 14000)),
        ("docs", "engineering-notes-1", docs("docs/engineering-notes.md", "## Format v7", 5200)),
        ("docs", "engineering-notes-2", docs("docs/engineering-notes.md", "## Format v6 (current)", 2400)),
        ("docs", "vllm-readme", docs("gpu/vllm/README.md", "## Local chat, like Ollama", 5200)),
        ("docs", "gpu-readme", docs("gpu/README.md", "## ", 5200)),
        ("docs", "spec", docs("docs/spec.md", "## 1. Block streams", 5200)),
        ("code", CODE[0][0], CODE[0][1]),
        ("code", CODE[1][0], CODE[1][1]),
        ("code", CODE[2][0], CODE[2][1]),
        ("table", "orders-csv", table(150)),
        ("table", "runs-json", records(28)),
    ] + [("chat", f"chat-{i + 1}", json.dumps(m, indent=1)) for i, m in enumerate(CHAT)]
    return "".join(f"@@@ {k} {n}\n{b.strip()}\n" for k, n, b in parts)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "glyd-gpu-py", "glyd_gpu", "kvcorpus.bin")
    text = build()
    with open(out, "wb") as f:
        f.write(text.encode("utf-8") if out.endswith(".txt") else zlib.compress(text.encode("utf-8"), 9))
    print(f"{out}: {len(text):,} characters, {text.count('@@@ ')} sections")
