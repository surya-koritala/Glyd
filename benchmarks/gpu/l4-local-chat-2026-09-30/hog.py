"""Holds this GPU's memory but FREE_MIB (default: an RTX 4080 SUPER's 16,376 MiB less 500 for a desktop), so a server
started after it sees what a 16 GB GeForce card would leave free; prints what it left, then sleeps until killed.
    python hog.py [FREE_MIB]"""
import sys
import time

import torch

want = int(sys.argv[1]) if len(sys.argv) > 1 else 16376 - 500
held = []
torch.cuda.init()
while True:
    free = torch.cuda.mem_get_info()[0] >> 20
    if free <= want:
        break
    held.append(torch.empty(min(free - want, 256) << 20, dtype=torch.uint8, device="cuda"))
free, total = (v >> 20 for v in torch.cuda.mem_get_info())
print(f"hog: {sum(t.numel() for t in held) >> 20} MiB held, {free} of {total} MiB free", flush=True)
while True:
    time.sleep(3600)
