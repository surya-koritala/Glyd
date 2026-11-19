"""Fixtures of the pack's four formats made by the pack writer itself (glm_pack.pack_rows): python fmtfix.py OUT_DIR [ROWS=128]
fmt-{tern,cb2,cb3,cb4}.safetensors: codes (U8 [rows, row bytes], the writer's bytes), codebook (BF16 [rows, k], ascending), idx (U8 [rows, 4096], the codes before packing)."""
import os, sys
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.expanduser("~/glyd-scratch/glm"))
import glm_pack as P

out = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 128
os.makedirs(out, exist_ok=True)
g = torch.Generator().manual_seed(7)
for fmt, k in P.K_ENTRIES.items():
    idx = torch.randint(0, k, (rows, 4096), generator=g, dtype=torch.int64).to(torch.uint8)
    codes = P.pack_rows(idx, fmt).contiguous()
    book = (torch.randn(rows, k, generator=g) * 0.05).sort(dim=1).values.to(torch.bfloat16).contiguous()
    assert torch.equal(P.unpack_rows(codes, fmt, 4096), idx)
    save_file({"codes": codes, "codebook": book, "idx": idx.contiguous()}, f"{out}/fmt-{fmt}.safetensors")
    print(fmt, tuple(codes.shape), tuple(book.shape))
