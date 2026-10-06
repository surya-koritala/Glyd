import json, os, struct, sys, math, glob, collections
sys.path.insert(0, os.path.join(os.environ.get("CODECW_SCRATCH", "."), "gguf"))
from gguflib import TYPES
sys.path.insert(0, "tools")
from build_corpus import gguf_skip_kv, ESIZE

def check_st(p):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
    size = os.path.getsize(p); end = 0; by = collections.Counter()
    for k, v in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = v["data_offsets"]; assert a == end, (k, a, end); end = b
        assert math.prod(v["shape"]) * ESIZE[v["dtype"]] == b - a, (k, v)
        by[v["dtype"]] += b - a
    assert 8 + n + end == size, (8 + n + end, size)
    return len(h), n + 8, dict(by)

def check_gguf(p):
    size = os.path.getsize(p); f = open(p, "rb")
    hb = f.read(min(size, 32 << 20)); assert hb[:4] == b"GGUF"
    ver, nt, nkv = struct.unpack_from("<IQQ", hb, 4)
    pos = gguf_skip_kv(hb, nkv, 24)
    infos = []
    for _ in range(nt):
        ln = struct.unpack_from("<Q", hb, pos)[0]; pos += 8; name = hb[pos:pos + ln].decode(); pos += ln
        nd = struct.unpack_from("<I", hb, pos)[0]; pos += 4
        dims = struct.unpack_from("<" + "Q" * nd, hb, pos); pos += 8 * nd
        ty, off = struct.unpack_from("<IQ", hb, pos); pos += 12
        infos.append((name, dims, ty, off))
    data = (pos + 31) // 32 * 32
    by = collections.Counter(); end = 0
    for name, dims, ty, off in infos:
        tn, blck, tsz = TYPES[ty]
        assert dims[0] % blck == 0
        nb = math.prod(dims) // blck * tsz
        assert off == end, (name, off, end)
        end = off + nb; end += -end % 32
        by[tn] += nb
    assert data + end >= size - 31 and data + end - size < 32, (data, end, size)
    return nt, data, dict(by)

for p in sorted(glob.glob("corpus/*.safetensors") + glob.glob("corpus/*.gguf")):
    r = check_st(p) if p.endswith(".safetensors") else check_gguf(p)
    print(f"{os.path.basename(p):42s} {os.path.getsize(p)/1e6:8.1f} MB  tensors {r[0]:4d} header {r[1]/1e6:6.2f} MB  {({k: round(v/1e6,1) for k,v in r[2].items()})}")
