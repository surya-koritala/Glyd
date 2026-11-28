"""glyd-v7 (the six- and five-bit block layouts) in format.py, what needs no GPU and no vLLM: read_manifest reads it, manifest() labels an
Int save and names its level, glyd.json's "level" is a key verify accepts, and the loaders' refusal says it runs on the Glyd engine.

    python test_format_v7.py              (or pytest test_format_v7.py)

The glyd_gpu it runs: the installed one, or this checkout's sources with PYTHONPATH=.."""
import json
import os
import tempfile

from glyd_gpu import format as fmt


def test_manifest_and_read():
    for layout, level in (("i6", "kestrel"), ("i5", "linnet")):
        e = fmt.entry([64, 32], [(4 if layout == "i6" else 5) << 24, 0, 0], [("m.weight", [64, 32], "00")], layout=layout)
        assert e["layout"] == layout and e["tiers"][0] >> 24 in (4, 5)
        m = fmt.manifest({"repo": None, "revision": None}, {"m": e}, "0.30.0", layout, {})
        assert (m["format"], m["layout"], m["level"]) == ("glyd-v7", layout, level) and all(k in fmt.KEYS for k in m)
        with tempfile.TemporaryDirectory() as d:
            json.dump(m, open(os.path.join(d, fmt.MANIFEST), "w"))
            assert fmt.read_manifest(d) == m
            try:
                fmt.refuse_int(fmt.read_manifest(d), "vLLM")
            except ValueError as err:
                assert "glyd-v7" in str(err) and "Glyd engine" in str(err) and "glyd run" in str(err) and level in str(err), str(err)
            else:
                raise AssertionError("not refused")
    assert fmt.manifest(None, {}, "0.30.0")["format"] == "glyd-v4" and "level" not in fmt.manifest(None, {}, "0.30.0")
    assert "calibration" in fmt.KEYS  # (glyd-gpu pack --codes records how a save's codes were made: a key verify accepts)
    fmt.refuse_int(fmt.manifest(None, {}, "0.30.0"), "vLLM")  # (a lossless save is not refused)


def test_row_layouts_are_refused():
    """A glyd-v7 save may hold a pack in a row layout ("i6r", "i5r", "i4r": the codes and scales of a block layout for a weight's transpose, a row at a time; the Glyd engine reads it and glyd-gpu verify decodes it
    on the CPU). The Python package lists the layouts it can load (format.LAYOUTS): it refuses such a save, in its loaders by the format and in check_files by the pack's buffers, which it does not count as a
    pack's, a ValueError each; it never reads one as a block layout's."""
    for lay, tag in (("i6r", 8), ("i5r", 9), ("i4r", 10)):
        assert lay not in fmt.LAYOUTS and lay not in fmt.WORDS
        e = {"layout": lay, "shape": [256, 128], "tiers": [tag << 24, 0, 0], "tensors": [{"name": "m.down_proj.weight", "shape": [128, 256], "sha256": "00"}]}
        m = fmt.manifest({"repo": None, "revision": None}, {"m.down_proj": e}, "0.30.0", "i5", {})
        assert m["format"] == "glyd-v7"
        with tempfile.TemporaryDirectory() as d:
            json.dump(m, open(os.path.join(d, fmt.MANIFEST), "w"))
            try:
                fmt.refuse_int(fmt.read_manifest(d), "vLLM")
            except ValueError as err:
                assert "glyd-v7" in str(err) and "Glyd engine" in str(err), str(err)
            else:
                raise AssertionError("not refused by the loader")
            # the pack's three buffers in the safetensors, as the save writes them
            names = [f"m.down_proj.glyd_{b}" for b in fmt.BUFFERS]
            dtypes = {"data": ("U8", 100), "blocks": ("U8", 8), "block_base": ("I32", 33)}
            header, at, body = {"__metadata__": {"format": "pt"}}, 0, b""
            for b, n in zip(fmt.BUFFERS, names):
                dt, k = dtypes[b]
                size = k * (4 if dt == "I32" else 1)
                header[n] = {"dtype": dt, "shape": [k], "data_offsets": [at, at + size]}
                body += bytes(size)
                at += size
            h = json.dumps(header).encode()
            h += b" " * (-len(h) % 8)
            open(os.path.join(d, "model.safetensors"), "wb").write(len(h).to_bytes(8, "little") + h + body)
            try:
                fmt.check_files(d, m)
            except ValueError as err:
                assert "in the safetensors, but glyd.json neither packs it nor holds its sha256" in str(err), str(err)
            else:
                raise AssertionError("not refused by check_files")


def test_codebook_row_layouts_are_refused():
    """A glyd-v7 save at the Hummingbird level (layout "hb", the level "hummingbird") holds packs in the codebook row layouts "c3r" and "c2r" (tags 11 and 12; the buffers data, blocks and codebook: a matrix of neuron rows
    held as indices into a codebook of each row's own; the Glyd engine reads them and glyd-gpu verify decodes them on the CPU). The Python package lists none of it (format.LAYOUTS): its loaders refuse the save by the
    format, naming the level and the engine, and check_files raises on the pack's buffers, a ValueError each; it never reads one as a block layout's."""
    for lay, tag in (("c3r", 11), ("c2r", 12)):
        assert lay not in fmt.LAYOUTS and lay not in fmt.WORDS
        e = {"layout": lay, "shape": [64, 128], "tiers": [tag << 24, 0, 0], "transposed": True, "tensors": [{"name": "m.down_proj.weight", "shape": [128, 64], "sha256": "00"}]}
        m = fmt.manifest({"repo": None, "revision": None}, {"m.down_proj": e}, "0.30.0", "i6", {})
        m["layout"], m["level"] = "hb", "hummingbird"
        assert m["format"] == "glyd-v7" and all(k in fmt.KEYS for k in m)
        with tempfile.TemporaryDirectory() as d:
            json.dump(m, open(os.path.join(d, fmt.MANIFEST), "w"))
            try:
                fmt.refuse_int(fmt.read_manifest(d), "vLLM")
            except ValueError as err:
                assert "glyd-v7" in str(err) and "Glyd engine" in str(err) and "hummingbird" in str(err), str(err)
            else:
                raise AssertionError("not refused by the loader")
            # the pack's three buffers in the safetensors, as the save writes them
            names = [f"m.down_proj.glyd_{b}" for b in ("data", "blocks", "codebook")]
            header, at, body = {"__metadata__": {"format": "pt"}}, 0, b""
            for n, k in zip(names, (64 * (48 if lay == "c3r" else 32), 64 * 8, 64 * (16 if lay == "c3r" else 8))):
                header[n] = {"dtype": "U8", "shape": [k], "data_offsets": [at, at + k]}
                body += bytes(k)
                at += k
            h = json.dumps(header).encode()
            h += b" " * (-len(h) % 8)
            open(os.path.join(d, "model.safetensors"), "wb").write(len(h).to_bytes(8, "little") + h + body)
            try:
                fmt.check_files(d, m)
            except ValueError as err:
                assert "in the safetensors, but glyd.json neither packs it nor holds its sha256" in str(err), str(err)
            else:
                raise AssertionError("not refused by check_files")


if __name__ == "__main__":
    test_manifest_and_read()
    print("test_manifest_and_read ok")
    test_row_layouts_are_refused()
    print("test_row_layouts_are_refused ok")
    test_codebook_row_layouts_are_refused()
    print("test_codebook_row_layouts_are_refused ok")
