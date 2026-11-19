"""One GPU comparison of existing expert-scale selection and the implemented small-route decoder.
Synthetic byte and 16-bit scale stacks; kernel/selection evidence only, not whole-model speed.
"""
import json
import os
import statistics
import sys
from pathlib import Path

import torch
from glyd_gpu import kernels
from glyd_gpu.vllm_scales import _Experts

out_dir = Path(sys.argv[1])
out_dir.mkdir(exist_ok=True)
torch.manual_seed(20261006)
rows = []

def graph(work):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            work()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        work()
    torch.cuda.synchronize()
    return gr

def timing(gr):
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(200):
        gr.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 200

for E, shapes, dt, kinds in ((256, ((128, 128), (64, 128)), torch.float16, (0, 2)),
                            (256, ((512, 128), (256, 128)), torch.uint8, (0,)),
                            (128, ((128, 128), (64, 128)), torch.uint8, (1,))):
    packs, parts, want, at = [], [], [], 0
    for O, K in shapes:
        if dt == torch.float16:
            m = (torch.rand(E * O, K, device="cuda") * .004 + .01).to(dt)
        else:
            m = torch.randint(110, 119 if kinds == (0,) else 113, (E * O, K), dtype=dt, device="cuda")
        p = kernels.sc_pack(m, kinds=kinds)
        packs.append(p)
        n = m.numel() * m.element_size()
        parts.append(("s", at, n, None))
        want.append(m.view(torch.uint8 if p.lo is None else torch.int16).reshape(E, O, K))
        at += n
    e = _Experts(E, packs, torch.device("cuda"))
    e.parts = parts
    from types import SimpleNamespace
    e.scratch = SimpleNamespace(buf=torch.empty(at, dtype=torch.uint8, device="cuda"), sealed=False)
    for count in (0, 1, 8, 32, 64, 256, 1024, 2048):
        for id_dtype in (torch.int32, torch.int64):
            ids = torch.arange(count, dtype=id_dtype, device="cuda") % E
            if count > 2:
                ids[1] = -1
                ids[2] = E
            def work(enabled):
                e.routed = enabled
                e.decode(ids)
            gs = [graph(lambda enabled=enabled: work(enabled)) for enabled in (False, True)]
            snapshots = []
            for gr in gs:
                e.scratch.buf.fill_(0xA5)
                gr.replay()
                torch.cuda.synchronize()
                snapshots.append(e.scratch.buf.clone())
            assert torch.equal(*snapshots), (E, shapes, str(dt), count, str(id_dtype), "output differs")
            selected = {int(i) for i in ids.cpu().tolist() if 0 <= int(i) < E}
            for p, (_, offset, n, _), original in zip(packs, parts, want):
                view = snapshots[1][offset:offset+n].view(original.dtype).reshape(original.shape)
                for i in selected:
                    assert torch.equal(view[i], original[i]), (E, count, i, "weight bits differ")
            times = [[], []]
            for _ in range(3):
                for v in (0, 1, 1, 0):
                    times[v].append(timing(gs[v]))
            a, b = map(statistics.median, times)
            row = dict(experts=E, shapes=shapes, scale_dtype=str(dt), kinds=[p.kind for p in packs],
                       route_count=count, route_dtype=str(id_dtype), old_us=a, candidate_us=b,
                       ratio=a/b, direct=count<=64, exact=True, trials_us=times)
            rows.append(row)
            print(json.dumps(row), flush=True)
    del packs, want, gs, snapshots, e, m
    torch.cuda.empty_cache()
(out_dir / "summary.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
    scope="synthetic scale selection and decoding only; no whole-model inference claim", rows=rows), indent=2))
print("ROUTED EXACT PASS", flush=True)
