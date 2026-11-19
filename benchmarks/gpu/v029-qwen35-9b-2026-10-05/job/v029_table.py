import json, os, sys, re
S = sys.argv[1]
for n in ("v29-4090", "v29-5090"):
    R = f"{S}/ceiling/results-{n}"
    print(f"== {n}")
    for f in sorted(os.listdir(R)):
        if f.startswith("speed-") and f.endswith(".json"):
            d = json.load(open(os.path.join(R, f)))
            print(f"  {f[6:-5]:30} TTFT {d['ttft_ms_median']:7.1f} ms  decode {d['decode_tok_s_median']:7.2f} tok/s")
    for l in open(f"{R}/numbers.txt") if os.path.exists(f"{R}/numbers.txt") else []:
        if l.strip().endswith(":"): continue
        print("  ", l.strip())
