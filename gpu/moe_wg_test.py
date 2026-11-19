"""The wgmma route's pieces that run on any GPU (the kernel itself is Hopper's, sm_90a): the tiles of a plan, the pairs' rows of X in the
order's, and the unit's epilogue (its sums out of a thread's registers: the exchange between the gate's and the up's warpgroups, the
bias, SiLU, the router's weights, the rows each warpgroup writes), each against a torch reference. Builds the library's pybind module
for this GPU (PyTorch's JIT, nvcc).

    python moe_wg_test.py
"""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glyd-gpu-py")))
import torch
import torch.nn.functional as F

from glyd_gpu_dev import _jit

BF = torch.bfloat16


def route(ext, ids, E):
    plan = torch.empty(2 + 2 * E + ids.numel(), dtype=torch.int32, device="cuda")
    ext.moe_way(ids, E, plan)
    return plan


def reference_tiles(plan, E, NT):
    h = int(plan[0])
    out = []
    for i in range(h):
        e, p0, p1 = int(plan[1 + i]), int(plan[1 + E + i]), int(plan[2 + E + i])
        n = p1 - p0
        for c in range((n + NT - 1) // NT):
            out.append((e, p0 + c * NT, min(NT, n - c * NT), int(n <= NT)))
    return out


def test_tiles(ext):
    gen = torch.Generator(device="cuda").manual_seed(1)
    cases = [(16, 2, 40), (128, 8, 232), (128, 8, 1152), (8, 2, 1), (4096, 8, 2048), (64, 1, 3000), (5, 2, 500)]  # (E, k, T): few hits, many (past the scan's 1,024), one expert with most
    for E, k, T in cases:
        logits = torch.randn(T, E, device="cuda", generator=gen)
        if (E, T) == (5, 500):
            logits[:, 0] += 6.0  # one expert holds most of the pairs: many chunks
        ids = torch.topk(logits, k, -1).indices.contiguous()
        plan = route(ext, ids, E)
        P = T * k
        for NT in (16, 32, 64, 128):
            tiles = ext.moe_wg_tiles(plan, E, NT, P).cpu()
            ref = reference_tiles(plan.cpu(), E, NT)
            n = int(tiles[0, 0])
            assert n == len(ref), (E, k, T, NT, n, len(ref))
            assert tiles[1 : 1 + n].tolist() == [list(r) for r in ref], (E, k, T, NT)
            assert 1 + n <= tiles.shape[0], "the records fit the workspace"
    print("tiles: passed", flush=True)


def test_gather(ext):
    gen = torch.Generator(device="cuda").manual_seed(2)
    for E, k, T, K in [(16, 2, 40, 64), (128, 8, 232, 2048), (64, 4, 300, 768)]:
        ids = torch.topk(torch.randn(T, E, device="cuda", generator=gen), k, -1).indices.contiguous()
        ids[0, 0] = -1  # a pair routed nowhere (the plan leaves it out)
        plan = route(ext, ids, E)
        x = (torch.randn(T, K, device="cuda", generator=gen) * 0.1).to(BF)
        out = ext.moe_wg_gather(x, k, plan, E)
        h = int(plan[0])
        pairs = int(plan[1 + E + h])
        order = plan[2 + 2 * E : 2 + 2 * E + pairs].long()
        assert pairs == T * k - 1, pairs
        assert torch.equal(out[:pairs].view(torch.int16), x[order // k].view(torch.int16)), (E, k, T, K)
        assert not out[pairs:].view(torch.int16).any(), "rows past the routed pairs untouched"
    print("gather: passed", flush=True)


def test_walk(ext):
    """The producer's walk over a block's stages and the consumers' loop (the kernel's own functions, signatures in place of the copies and the
    products): every stage's tile record, row unit, row blocks' first steps and bounds as the consumers expect them, a unit's last record, the
    stages counted, over rings of 2 to 8 slots (the phases wrap) and blocks from 1 to more than the units."""
    gen = torch.Generator(device="cuda").manual_seed(4)
    E, k = 24, 4
    for T, skew in ((50, 0.0), (700, 0.0), (400, 3.0)):
        logits = torch.randn(T, E, device="cuda", generator=gen)
        logits[:, :2] += skew
        ids = torch.topk(logits, k, -1).indices.contiguous()
        plan = route(ext, ids, E)
        P = T * k
        for NT in (16, 64):
            tiles = ext.moe_wg_tiles(plan, E, NT, P)
            TU = int(tiles[0, 0])
            tl = tiles[1 : 1 + TU].cpu().tolist()
            for act, O in ((1, 512), (0, 512), (0, 704)):  # (704: 11 row blocks, the last row unit's second one past)
                RB = O // 64
                RU = O // 128 if act else (RB + 1) // 2
                for S in (1, 2, 3, 12):
                    KS = 4 * S
                    exc_base = torch.cumsum(torch.randint(0, 5, (E * RB * KS + 8,), device="cuda", generator=gen), 0).to(torch.int32)
                    total = RU * TU
                    for nb in (1, 3, 40, total, total + 5):
                        max_units = -(-total // nb)
                        for ns in (2, 3, 5, 8):
                            res = ext.moe_wg_walk_test(tiles, exc_base, RB, KS, RU, S, nb, ns, act, max_units).cpu()
                            tag = (T, NT, act, O, S, nb, ns)
                            assert int(res[0]) == 0, ("the walk's checks failed (-1: a wait did not end)", tag, int(res[0]))
                            for b in range(nb):
                                units = [u for u in range(b, total, nb)]
                                for uk, u in enumerate(units):
                                    q, ru = divmod(u, RU)
                                    e, row0, np_, _ = tl[q]
                                    got = res[8 + 8 * (b * max_units + uk) : 8 + 8 * (b * max_units + uk) + 5].tolist()
                                    assert got == [row0, np_, e, ru, (uk + 1) * S], (tag, b, uk, got, [row0, np_, e, ru, (uk + 1) * S])
    print("walk: passed", flush=True)


def test_epilogue(ext):
    gen = torch.Generator(device="cuda").manual_seed(3)
    E, k, T = 16, 4, 80
    ids = torch.topk(torch.randn(T, E, device="cuda", generator=gen), k, -1).indices.contiguous()
    plan = route(ext, ids, E)
    P = T * k
    order = plan[2 + 2 * E : 2 + 2 * E + P].long()
    for NT in (16, 32, 64, 128):
        for act in (0, 1):
            for O in ((256, 512) if act else (640, 704)):  # (704 / 64 = 11 row blocks: the last row unit has one)
                RB = O // 64
                RU = O // 128 if act else (RB + 1) // 2
                for with_bias in (False, True):
                    for weights in ((None,) if act else ("none", "bf16", "fp32")):
                        for ru, e, np_, row0 in ((0, 3, NT, 0), (RU - 1, 5, max(1, NT // 3), 7), (RU // 2, 0, 1, P - 1)):
                            if row0 + np_ > P:
                                continue
                            mat = torch.randn(2, 64, NT, device="cuda", generator=gen)
                            bias = (torch.randn(E, O, device="cuda", generator=gen) * 0.1).to(BF) if with_bias else torch.empty(0, dtype=BF, device="cuda")
                            wt = torch.empty(0, device="cuda")
                            if weights == "bf16":
                                wt = torch.rand(P, device="cuda", generator=gen).to(BF)
                            elif weights == "fp32":
                                wt = torch.rand(P, device="cuda", generator=gen)
                            sentinel = 7.0
                            if act:
                                Y = torch.full((P, O // 2), sentinel, dtype=BF, device="cuda")
                                Y32 = torch.empty(0, device="cuda")
                            elif weights == "none":
                                Y = torch.full((P, O), sentinel, dtype=BF, device="cuda")
                                Y32 = torch.empty(0, device="cuda")
                            else:
                                Y = torch.empty(0, dtype=BF, device="cuda")
                                Y32 = torch.full((P, O), sentinel, device="cuda")
                            before = (Y.clone(), Y32.clone())
                            ext.moe_wg_epilogue_test(mat, [row0, np_, e, ru], NT, act, O, E, plan, bias, wt, Y, Y32)
                            torch.cuda.synchronize()
                            # the reference: the same cells, written by torch
                            expect = [before[0].clone(), before[1].clone()]
                            b = bias.float() if with_bias else torch.zeros(E, O, device="cuda")
                            for m in range(np_):
                                i = row0 + m
                                if act:
                                    o = ru * 64 + torch.arange(64, device="cuda")
                                    gate = mat[0, :, m] + b[e, o]
                                    up = mat[1, :, m] + b[e, O // 2 + o]
                                    expect[0][i, o] = (F.silu(gate) * up).to(BF)
                                else:
                                    for wg in range(2):
                                        rb = 2 * ru + wg
                                        if rb >= RB:
                                            continue
                                        o = rb * 64 + torch.arange(64, device="cuda")
                                        y = mat[wg, :, m] + b[e, o]
                                        if weights == "none":
                                            expect[0][i, o] = y.to(BF)
                                        else:
                                            j = int(order[i])
                                            expect[1][j, o] = y * wt[j].float()
                            tag = (NT, act, O, with_bias, weights, ru, e, np_, row0)
                            if act:
                                a, c = Y.float(), expect[0].float()
                                assert torch.allclose(a, c, rtol=2e-2, atol=1e-3), (tag, (a - c).abs().max().item())
                            elif weights == "none":
                                assert torch.equal(Y.view(torch.int16), expect[0].view(torch.int16)), tag
                            else:
                                assert torch.equal(Y32, expect[1]), (tag, (Y32 - expect[1]).abs().max().item())
    print("epilogue: passed", flush=True)


if __name__ == "__main__":
    assert torch.cuda.get_device_capability()[0] >= 8, "Ampere or later"
    ext = _jit()
    test_tiles(ext)
    test_gather(ext)
    test_walk(ext)
    test_epilogue(ext)
    print("moe_wg_test: all passed")
