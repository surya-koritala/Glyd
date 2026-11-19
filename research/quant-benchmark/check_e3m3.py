"""CI-only check of the e3m3 transform against an independent nearest-value search; no model or GPU."""
from fractions import Fraction
from pathlib import Path
import sys
import torch
from model_adapter import LEGACY, round_e3m3  # LEGACY puts matched_quality on the path
from matched_quality import round_candidate_a

GRID = sorted({(Fraction(m, 8) * Fraction(1, 4) if e == 0 else Fraction(8 + m, 8) * Fraction(2) ** (e - 3), e * 8 + m)
               for e in range(8) for m in range(8)})  # (non-negative value, code)


def expected(values, block):
    """Nearest e3m3 value times the smallest power-of-two scale covering each block, ties to the even code."""
    out = []
    for b in range(0, len(values), block):
        part = [Fraction(v) for v in values[b:b+block]]
        peak = max((abs(v) for v in part), default=Fraction(0))
        k = 0
        if peak:
            k = -200
            while peak > 30 * Fraction(2) ** k:
                k += 1
        scale = Fraction(2) ** k
        for v in part:
            q = abs(v) / scale
            best = min(GRID, key=lambda g: (abs(g[0] - q), g[1] % 2))
            out.append((best[0] * scale) * (1 if v >= 0 else -1))
    return out


def check(words, block):
    got = round_e3m3(words, block).view(torch.bfloat16).double().tolist()
    source = words.view(torch.bfloat16).double()
    for r in range(words.shape[0]):
        want = expected(source[r].tolist(), block)
        for c, (g, w) in enumerate(zip(got[r], want)):
            if Fraction(g) != w:
                raise ValueError(f'e3m3 rounding differs at row {r} column {c}: {g} vs {float(w)}')


def main():
    every = round_candidate_a(torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
    finite = (every.view(torch.int16).to(torch.int32) & 0x7f80) != 0x7f80
    words = every[finite].view(torch.int16)
    rows = words[: words.numel() // 128 * 128].reshape(-1, 128)
    check(rows[::7], 128)  # a spread of rows over every Candidate A word
    torch.manual_seed(5)
    for scale in (1e-3, 1.0, 40.0):
        noise = round_candidate_a((torch.randn(8, 300) * scale).to(torch.bfloat16)).view(torch.int16)
        check(noise, 128)  # includes a partial last block
    special = torch.tensor([[0x7f80, 0x7fc0, 0x3f80, 0x0000]], dtype=torch.int32).to(torch.int16)
    if not torch.equal(round_e3m3(special, 128)[0, :2], special[0, :2]):
        raise ValueError('non-finite words changed')
    print('e3m3 transform matches the independent nearest-value search')
    peak = rows.view(torch.bfloat16).float().abs().amax(dim=1)
    kept = rows[(peak >= 2.0 ** -90) & (peak <= 2.0 ** 7)]  # the storage's scale range; real weights use 2^-38..2^-2
    check_seven_bit(kept[: kept.shape[0] // 64 * 64], torch)


def check_seven_bit(words, torch):
    """The GPU harness's seven-bit storage (research/candidate-a-gpu/a7_variant.py) holds exactly this transform."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'candidate-a-gpu'))
    from a7_variant import ORDER, decode, encode, exps_blocks, fragments, pack7, unpack7
    torch.manual_seed(9)
    noise = round_candidate_a((torch.randn(64, 512) * 3.0).to(torch.bfloat16)).view(torch.int16)
    for w in (words, noise):
        codes, exps = encode(w)
        if not torch.equal(decode(codes, exps).to(torch.bfloat16).view(torch.int16), round_e3m3(w, 128)):
            raise ValueError('seven-bit encoding differs from the e3m3 transform')
        peak = torch.ldexp(w.view(torch.bfloat16).float().abs().view(w.shape[0], -1, 128).amax(dim=2), -(exps.int() - 4))
        if not bool(((peak <= 30) & ((peak > 15) | (peak == 0))).all()):
            raise ValueError('block scale is not the smallest covering one')
        blocks = exps_blocks(exps)
        for r, c in ((0, 0), (63, 0), (9, exps.shape[1] - 1), (w.shape[0] - 1, exps.shape[1] // 2)):
            if blocks[r // 64, c, r % 8, r % 64 // 8] != exps[r, c]:
                raise ValueError('block exponent order differs')
        frag = fragments(codes)
        packed = pack7(frag)
        if packed.numel() * 8 != codes.numel() * 7 or not torch.equal(unpack7(packed, *frag.shape[:2]), frag):
            raise ValueError('seven-bit packing does not round-trip')
        back = torch.empty_like(codes)  # the device unpack's scatter: register 2nn + h of lane 4g + t
        rb, npairs = frag.shape[:2]
        for i in range(16):
            for lane in range(32):
                for j in range(4):
                    r = torch.arange(rb) * 64 + i // 2 * 8 + lane // 4
                    c = torch.arange(npairs) * 32 + i % 2 * 16 + ORDER[4 * (lane % 4) + j]
                    back[r[:, None], c[None, :]] = frag[:, :, i, lane, j]
        if not torch.equal(back, codes):
            raise ValueError('fragment order does not invert')
    print('seven-bit storage holds the e3m3 transform exactly and its packing round-trips')


if __name__ == '__main__':
    main()
