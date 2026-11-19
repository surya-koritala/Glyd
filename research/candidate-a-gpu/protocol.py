"""Immutable input and completion checks; no CUDA or model imports."""
import hashlib
import json
import math
from pathlib import Path

TOKENS = [1, 2, 8, 16, 17, 32, 33, 64, 65, 127, 128, 129, 256, 512, 1024, 2048]
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
POLICY = {
    "max_abs_error_fraction_of_reference_max": 0.01,
    "max_abs_error_absolute_allowance": 1e-6,
    "max_normalized_rms_error": 0.004,
    "exact_integer_products": True,
    "decoded_weights_bit_exact": True,
    "repeated_products_bit_exact": True,
    "repetitions": 10,
    "timing_groups": 5,
    "warmup": 3,
    # Each timed operation reads its own weight copy; the copies span at least this many bytes (the RTX 4090's
    # L2 holds 72 MiB), so timed weights come from device memory. Tiny matrices are capped and stay L2-resident.
    "rotation_bytes": 192 << 20,
    "max_copies": 256,
    "big_variants": [1, 2],
}
TIMED = ("compact_ms", "expanded_ms", "bf16_ms", "unpack_ms")

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(8 << 20):
            h.update(data)
    return h.hexdigest()

def plan(inventory):
    """One complete tensor per eligible shape; the large embedding uses a declared row slice."""
    by_shape = {}
    unsupported = []
    for row in sorted(inventory, key=lambda r: r["name"]):
        shape = row["shape"]
        if row["dtype"] != "BF16" or len(shape) != 2 or shape[0] % 64 or shape[1] % 16:
            unsupported.append(row)
            continue
        if tuple(shape) not in by_shape or "embed_tokens" in row["name"]:
            by_shape[tuple(shape)] = row
    cases = []
    for i, (_, row) in enumerate(sorted(by_shape.items())):
        rows, cols = row["shape"]
        tested_rows = min(rows, 4096) if "embed_tokens" in row["name"] else rows
        cases.append({
            "id": f"model-{i:02d}", "name": row["name"],
            "source_shape": [rows, cols], "shape": [tested_rows, cols],
            "source_row_start": 0, "whole_tensor": rows == tested_rows,
            "kind": "model", "products": True, "exact_product": False,
        })
    return cases, unsupported

def relative_file(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("invalid bundle-relative file")
    path = root / name
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing or unsafe input file")
    return path

def check_inputs(root):
    root = Path(root)
    manifest = json.loads((root / "cases.json").read_text())
    if (manifest.get("schema"), manifest.get("model"), manifest.get("revision")) != (1, MODEL, REVISION):
        raise ValueError("wrong input identity")
    if manifest["tokens"] != TOKENS or manifest["policy"] != POLICY:
        raise ValueError("validation protocol changed")
    cases = manifest["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("empty or duplicate cases")
    for case in cases:
        rows, cols = case["shape"]
        if min(rows, cols) <= 0 or max(rows, cols) > 1 << 20 or rows % 64 or cols % 16 or rows * cols > 1 << 31:
            raise ValueError("unsupported matrix shape")
        files = case["files"]
        required = {"rounded.bf16", "compact.data", "expanded.data", "blocks.bin", "bases.i32", "pack.json"}
        if set(files) != required:
            raise ValueError("missing matrix buffer")
        for name, meta in files.items():
            path = relative_file(root, f"fixtures/{case['id']}/{name}")
            if path.stat().st_size != meta["bytes"] or digest(path) != meta["sha256"]:
                raise ValueError("input length or digest mismatch")
        p = json.loads((root / "fixtures" / case["id"] / "pack.json").read_text())
        if [p["rows"],p["cols"]] != [rows,cols] or p["exact_after_rounding"] is not True:
            raise ValueError("pack shape or host validation mismatch")
        steps = rows * cols // 1024
        if p["tiers"][0] >> 24 != (3 if p["compact"] else 2) or p["base_tiers"][0] >> 24 != 2:
            raise ValueError("unexpected representation tag")
        expected = {"rounded.bf16": rows*cols*2, "bases.i32": (steps+1)*4,
                    "compact.data": steps*(768 if p["compact"] else 1280),
                    "expanded.data": steps*1280}
        if any(files[n]["bytes"] != size for n,size in expected.items()):
            raise ValueError("matrix buffer geometry mismatch")
        shared = files["blocks.bin"]["bytes"] + files["bases.i32"]["bytes"]
        if p["weight_buffer_bytes"] != files["compact.data"]["bytes"] + shared or p["base_buffer_bytes"] != files["expanded.data"]["bytes"] + shared:
            raise ValueError("weight accounting mismatch")
        if case["products"] and not p["compact"]:
            raise ValueError("fallback fixture must use decode-only checks")
    return manifest

def expected_products(case):
    return [(m,bias) for m in TOKENS for bias in (False,True)] if case["products"] else []

def complete_report(manifest, rows):
    expected = {c["id"]: c for c in manifest["cases"]}
    if len(rows) != len(expected) or {r["case"] for r in rows} != set(expected):
        raise ValueError("missing or duplicate device case")
    for row in rows:
        case = expected[row["case"]]
        if row.get("complete") is not True or row.get("unpack_bit_exact") is not True:
            raise ValueError("incomplete device evidence")
        results = row["products"]
        identities = [(r["tokens"], r["bias"]) for r in results]
        if len(identities) != len(set(identities)) or set(identities) != set(expected_products(case)):
            raise ValueError("missing product or refusal case")
        for r in results:
            unsupported = r["tokens"] > 64 and case["shape"][1] % 64 != 0
            if r["status"] != ("expected-refusal" if unsupported else "pass"):
                raise ValueError("invalid product result")
            if not unsupported:
                if r.get("repeat_bit_exact") is not True or r.get("reference_pass") is not True:
                    raise ValueError("product correctness failed")
                for prefix in ("", "base_"):
                    error, nrms = r[prefix+"max_abs_error"], r[prefix+"normalized_rms_error"]
                    maximum = r["reference_max"]
                    if any(not math.isfinite(x) or x < 0 for x in (error,nrms,maximum)) or error > POLICY["max_abs_error_fraction_of_reference_max"]*maximum+POLICY["max_abs_error_absolute_allowance"] or nrms > POLICY["max_normalized_rms_error"]:
                        raise ValueError("product error exceeds frozen bounds")
                if case["exact_product"] and (r["integer_or_reference_bit_exact"] is not True or r["base_reference_bit_exact"] is not True):
                    raise ValueError("integer product is not bit exact")
                checked = [r[name] for name in TIMED]
                if r["tokens"] > 64:
                    variants = r.get("variants", {})
                    if set(variants) != {str(v) for v in POLICY["big_variants"]}:
                        raise ValueError("missing large-product variant")
                    for v in variants.values():
                        if v.get("repeat_bit_exact") is not True or v.get("reference_pass") is not True:
                            raise ValueError("large-product variant failed")
                        checked.append(v["ms"])
                if "fp8_ms" in r:
                    checked.append(r["fp8_ms"])
                elif not isinstance(r.get("fp8_error"), str):
                    raise ValueError("FP8 comparison missing without a recorded reason")
                if any(len(s) != POLICY["timing_groups"] or any(not math.isfinite(x) or x <= 0 for x in s) for s in checked):
                    raise ValueError("missing or invalid timing samples")
    return {
        "complete": True, "cases": len(rows),
        "scope": "isolated matrices and synthetic edge cases; not whole-model VRAM, native FP8 serving or end-to-end performance",
        "model": MODEL, "revision": REVISION,
        "results": rows,
    }
