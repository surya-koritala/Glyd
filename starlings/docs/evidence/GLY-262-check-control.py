"""Model-free function checks; never execute a native runner module body."""
import ast
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

original, comparison, control = [ast.parse(pathlib.Path(path).read_text()) for path in sys.argv[1:]]
old_wait = next(node for node in ast.walk(original) if isinstance(node, ast.Lambda) and ast.unparse(node).startswith("lambda: report(coordinator"))
try:
    eval(compile(ast.Expression(old_wait), "old-resource-wait", "eval"), {"report": lambda *args: None, "coordinator": None, "ids": {"worker": "synthetic"}})()
except TypeError:
    pass
else:
    raise AssertionError("old null-resource dereference did not fail")

for tree in [comparison, control]:
    chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"call", "ready_resource"}]
    assert len(chosen) == 2
    clock = [0.0]
    starts = []
    outcome = [subprocess.CompletedProcess([], 0, b"ok", b"")]

    def run(*args, **kwargs):
        starts.append((clock[0], kwargs["timeout"]))
        if isinstance(outcome[0], Exception):
            raise outcome[0]
        return outcome[0]

    namespace = {"time": SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay)), "subprocess": SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired), "BIN": pathlib.Path("synthetic"), "START": 0, "control_events": [], "control_next": {}, "resource_history": [], "coordinator": pathlib.Path("coordinator"), "ids": {"worker": "synthetic"}}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), "control-functions", "exec"), namespace)
    for value in [None, {"sample": {"sequence": "1", "capabilities": {"state": "unavailable"}}, "age_ms": 0}, {"sample": {"sequence": "2", "capabilities": {"state": "ready"}}, "age_ms": 0}]:
        namespace["report"] = lambda *args, value=value: value
        assert namespace["ready_resource"]() == (value if value is not None and value["sample"]["capabilities"]["state"] == "ready" else None)
    assert namespace["resource_history"][0] is None
    call = namespace["call"]
    state = pathlib.Path("worker")
    for _ in range(12):
        assert call(state, "cluster", "status") == "ok"
    assert all(b[0] - a[0] >= .2 - 1e-9 for a, b in zip(starts, starts[1:]))
    assert all(timeout == 3 for _, timeout in starts)
    # Model preparation is outside the listener bucket; different listeners are independent.
    before = clock[0]
    call(state, "model", "route-digest", "private-file")
    call(pathlib.Path("coordinator"), "cluster", "status")
    assert clock[0] == before
    for message, code in [(b"REFUSED: CONTROL_UNAVAILABLE: secret", "CONTROL_OUTCOME_UNKNOWN"), (b"REFUSED: ROUTE_ACTIVATION_REFUSED private-path", "ROUTE_ACTIVATION_REFUSED"), (b"SECURITY_Capacity: private-path", "CONTROL_CAPACITY"), (b"unknown secret payload", "UNKNOWN_CONTROL_ERROR")]:
        outcome[0] = subprocess.CompletedProcess([], 1, b"secret output", message)
        before = len(starts)
        try:
            call(state, "cluster", "route", "private-file", "--approve", "private-digest")
        except RuntimeError as error:
            assert str(error) == "CONTROL_COMMAND_FAILED"
        else:
            raise AssertionError("failed control command accepted")
        assert len(starts) == before + 1, "control mutation was retried"
        event = namespace["control_events"][-1]
        assert event == {"operation": "cluster/route", "role": "worker", "returncode": 1, "code": code}
        assert "secret" not in json.dumps(event) and "private" not in json.dumps(event)
    outcome[0] = subprocess.TimeoutExpired("private-command", 3, output=b"secret")
    before = len(starts)
    try:
        call(state, "cluster", "route", "private-file")
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("timeout accepted")
    assert len(starts) == before + 1
    assert namespace["control_events"][-1]["code"] == "CONTROL_TIMEOUT"
    phases = {node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and any(isinstance(target, ast.Name) and target.id == "phase" for target in node.targets)}
    assert {"coordinator_route_hash", "coordinator_route_activation", "coordinator_route_status", "coordinator_resource_readiness"} <= phases

assert not any(isinstance(node, ast.FunctionDef) and node.name in {"chat", "send", "generate", "decode_stream"} for node in control.body)
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "generate" for node in ast.walk(control))
for node in ast.walk(control):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "call":
        args = [arg.value if isinstance(arg, ast.Constant) else None for arg in node.args]
        assert args[1:3] != ["cluster", "token"], "control diagnostic must create no inference token"
result = next(node.value for node in ast.walk(control) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
values = {key.value: value for key, value in zip(result.keys, result.values)}
for name in ["generations_requested", "count_call_upper_bound", "output_tokens_total_cap"]:
    assert isinstance(values[name], ast.Constant) and values[name].value == 0
assert values["json_sse_semantic_parity"].value is False and values["matched_direct_proxy"].value is False
print("old null readiness fails; corrected readiness/pacing/private error/timeout/no-retry/phase checks pass; control driver has zero generation/count/token paths")
