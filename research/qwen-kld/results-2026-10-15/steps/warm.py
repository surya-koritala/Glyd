# Warm speed from the server logs of chain8 (the second pass of compare.py's speed part): ms a token, tokens a pass, GPU hit rate per prompt.
import re, sys
for n in sys.argv[1:]:
    lines = open(f"/home/surya-koritala/glyd-scratch/qwen-users/srv_{n}.log").read().splitlines()
    places = next((l.split("places: ")[1].split(",")[0] + " " + l.split(",")[-1].strip() for l in lines if "places:" in l), "?")
    vram = next((l.split(": ", 2)[-1] for l in lines if "in use" in l), "?")
    rows = [l for l in lines if " prompt tokens (" in l]
    out = []
    for size in ("33", "4016", "29663"):
        r = [l for l in rows if re.search(rf"\] {size} prompt tokens", l)][-1:]
        if r:
            m = re.search(r"then ([\d.]+) ms a token\): (\d+) of (\d+) guesses right, ([\d.]+) tokens a pass; the cache: ([\d.]+)%", r[0])
            out.append(f"{size}: {m.group(1)} ms ({m.group(4)} a pass, {m.group(5)}% hits)" if m else f"{size}: ?")
    print(f"{n:10} {' | '.join(out)} | places {places} | {vram}")
