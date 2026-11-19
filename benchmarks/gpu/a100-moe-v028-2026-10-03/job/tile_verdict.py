"""The wgmma tile rule against the forced tiles of a sweep's JSON (moe_layer.py --json, variants new, off, nt16, nt32, nt64, nt128): for each tokens a step, the
tile the rule takes (the smallest holding the pairs an expert has on average and 1.5 standard deviations more), each forced tile's layer time, the best, and
how far the rule's pick is from it.

    python tile_verdict.py layer.json [E] [k]      (the experts and top k of the JSON's layer, else 128 and 8)
"""
import json
import math
import sys


def rule(T, E, k):
    P = T * k
    hit = max(1, min(E, P))
    mean = P / hit
    need = mean + 1.5 * math.sqrt(mean)
    return 16 if need <= 16 else 32 if need <= 32 else 64 if need <= 64 else 128


def main(path, E=None, k=None):
    d = json.load(open(path))
    rows, E, k = d["rows"], E or d.get("E", 128), k or d.get("k", 8)
    by = {}
    for r in rows:
        by.setdefault(r["T"], {})[r["variant"]] = r["all_us"]
    print("| Tokens | rule's tile | new | nt16 | nt32 | nt64 | nt128 | best tile | the rule vs best | off (the other kernels) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    worst = 0.0
    for T in sorted(by):
        v = by[T]
        tiles = {n: v[f"nt{n}"] for n in (16, 32, 64, 128) if f"nt{n}" in v}
        if not tiles:
            continue
        best = min(tiles, key=tiles.get)
        mine = v.get("new") if "new" in v else tiles.get(rule(T, E, k))
        gap = mine / tiles[best] - 1 if mine else float("nan")
        worst = max(worst, gap) if gap == gap else worst
        cells = " | ".join(f"{tiles[n]:,.0f}" if n in tiles else "-" for n in (16, 32, 64, 128))
        print(f"| {T:,} | {rule(T, E, k)} | {mine:,.0f} | {cells} | {best} | {100 * gap:+.1f}% | {v.get('off', float('nan')):,.0f} |")
    print(f"\nthe rule's layer time is at most {100 * worst:.1f}% over the best forced tile at the tokens measured")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:4]))
