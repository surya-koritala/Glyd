import sys, src_show
steps = float(sys.argv[2])
d = src_show.parse(sys.argv[1])
f = [k for k in d if k.endswith("glyd_gpu.cu")][0]
L = d[f]
# (name, first line, last line) of the source regions (glyd_gpu.cu v0.27.0 main)
regions = [
 ("tier-1 group loop (L833-839: table, scratch read, PRMT)", 833, 839),
 ("tier2/3 digits to bytes (tier23, group, advance: L694-720, L784-792)", 694, 720),
 ("tier23 body", 784, 792),
 ("scans + window reads (L726-755)", 721, 755),
 ("exponents(): counts, shuffles, positions (L798-832)", 798, 832),
 ("pairs (L844-850)", 844, 850),
 ("load_step (L765-779)", 765, 779),
 ("mma_steps: loop, X loads, MMA (L1017-1048)", 1017, 1048),
 ("kernel prologue/epilogue (L1094-1162)", 1094, 1162),
 ("Tiered::decode wrappers (L853-870)", 853, 870),
]
tot = 0
seen = set()
for name, a, b in regions:
    s = sum(n for ln, (n, _) in L.items() if a <= ln <= b and ln not in seen)
    seen |= {ln for ln in L if a <= ln <= b}
    tot += s
    print(f"{s / steps:8.1f}  {name}")
rest = sum(n for ln, (n, _) in L.items() if ln not in seen)
print(f"{rest / steps:8.1f}  other lines of glyd_gpu.cu")
print(f"{(tot + rest) / steps:8.1f}  total")
