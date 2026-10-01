# The spike's runs against their bf16 run: greedy tokens (the matching prefix a prompt, as check_api counts them),
# the chosen tokens' logprobs where both have them (bit-identical, and the mean absolute difference over the shared
# prefix), and the KV blocks and tokens/s side by side.
#   python compare.py RESULTS_DIR
import glob, json, os, sys

R = sys.argv[1]
runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "*.json"))) if not p.endswith("reply.json")}
for name, r in runs.items():
    base = runs.get(name.split("-glyd")[0] + "-bf16" + ("-eager" if name.endswith("-eager") else "")) if "-glyd" in name else None
    line = f"{name}: blocks {r['num_gpu_blocks']} (x{r['block_size']}), load {r['load_s']} s, tokens/s 1/8/32: {r['tokens_per_s_1']} / {r['tokens_per_s_8']} / {r['tokens_per_s_32']}, {r['cudagraph_mode']}{' eager' if r['eager'] else ''}"
    if base:
        same, exact, diffs = [], 0, []
        for a, b, la, lb in zip(r["tokens"], base["tokens"], r["logprobs"], base["logprobs"]):
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            same.append(n)
            exact += la[:n] == lb[:n] and n == len(a) == len(b)
            diffs += [abs(x - y) for x, y in zip(la[:n], lb[:n])]
        line += f"; against bf16: tokens the same {same} of {len(r['tokens'][0])}; prompts bit-identical (tokens and logprobs) {exact} of {len(same)}; mean |logprob diff| on the shared prefix {sum(diffs) / max(1, len(diffs)):.2e}; blocks x{(r['num_gpu_blocks'] or 0) / (base['num_gpu_blocks'] or 1):.2f}"
    print(line)
