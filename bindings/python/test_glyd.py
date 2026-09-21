import os, shutil, tempfile
import glyd

def test_round_trips():
    data = b"".join(b'{"ts": %d, "host": "h%d", "n": %d}\n' % (1700000000 + i, i % 20, (i * 7919) % 1000) for i in range(20000))
    for level in ["default", "fast", "max", "ultra"]:
        assert glyd.decompress(glyd.compress(data, level=level)) == data
    c = glyd.compress(data, records=True)
    assert glyd.decompress(c) == data and len(c) < len(glyd.compress(data))
    assert glyd.decompressed_len(c) == len(data)
    assert glyd.decompress(glyd.compress(b"")) == b""
    v2 = data[:5000] + b"changed\n" + data[5000:]
    assert glyd.decompress_with_base(data, glyd.compress_with_base(data, v2)) == v2
    objs = [data[i * 1000:(i + 1) * 1000] for i in range(50)]
    p = glyd.pack(objs)
    assert glyd.pack_len(p) == 50 and glyd.unpack(p, 7) == objs[7] and glyd.decompress(p) == b"".join(objs)
    d = tempfile.mkdtemp()
    try:
        with glyd.Store(d) as s:
            i = s.put("v1", data)
            j = s.put("v2", v2)
            assert s.get(i) == data and s.get(j) == v2 and s.id_of("v2") == j
            k = s.put("small", b"tiny")
            assert s.get(k) == b"tiny"
            raw, stored = s.stats()
            assert stored * 4 < raw
            assert s.verify() == 0
        with glyd.Store(d) as s:
            assert s.get(j) == v2 and len(s) >= 3
    finally:
        shutil.rmtree(d)
    print("glyd", glyd.version(), "python binding ok")

if __name__ == "__main__":
    test_round_trips()
