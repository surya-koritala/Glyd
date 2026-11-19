"""Model-free checks for a prepared native runner; never execute its module body."""
import ast
import hashlib
import json
import pathlib
import sys
import tempfile

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "call":
        args = [arg.value if isinstance(arg, ast.Constant) else None for arg in node.args]
        assert args[1:3] != ["model", "pin"], "full catalog verification belongs in offline preparation"

check = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_reference_pin")
namespace = {"hashlib": hashlib, "json": json}
exec(compile(ast.Module(body=[check], type_ignores=[]), "pin-check", "exec"), namespace)
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    worker = root / "worker"
    (worker / "cache/models").mkdir(parents=True)
    manifest = "a" * 64
    lease = "00000000-0000-4000-8000-000000000001"
    receipt = root / "direct-pin.json"
    pin = worker / "cache/models" / f"pin-{lease}.json"
    namespace.update(ROOT=root, worker=worker, worker_route={"qualification": {"replica": {"manifest": manifest}}})
    for bad in [None, "manifest", "binding", "hash", "missing"]:
        receipt.write_text(json.dumps({"pin": lease, "manifest": "b" * 64 if bad == "manifest" else manifest, "retained": True}))
        pin.write_text(json.dumps({"version": 1, "manifest": manifest, "lease": "wrong" if bad == "binding" else lease}))
        namespace["prep"] = {"direct_pin_record_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(), "direct_pin_file_sha256": hashlib.sha256(pin.read_bytes()).hexdigest()}
        if bad == "hash":
            pin.write_text("{}")
        if bad == "missing":
            pin.unlink()
        try:
            namespace["verify_reference_pin"]()
        except (AssertionError, FileNotFoundError):
            assert bad is not None
        else:
            assert bad is None, f"accepted invalid {bad} pin"

result = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
manifest_value = next(value for key, value in zip(result.value.keys, result.value.values) if isinstance(key, ast.Constant) and key.value == "manifest")
assert eval(compile(ast.Expression(manifest_value), "manifest-check", "eval"), {"worker_route": namespace["worker_route"], "model": "backend-alias"}) == manifest
print("prepared ownership and immutable manifest checks passed; no native module body executed")
