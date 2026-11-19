"""Synthetic SSE checks; extract functions without executing the native runner."""
import ast
import io
import json
import pathlib
import sys
import time
from types import SimpleNamespace


def functions(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"chat", "decode_stream", "generate"}]
    assert len(selected) == 3
    scope = {"json": json, "time": time, "stage": "direct", "model": "synthetic", "requests": 0, "runs": [], "wait_for": lambda check, seconds: check(), "direct_idle": lambda: True, "settled": lambda: True}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "synthetic-stream", "exec"), scope)
    return scope


def response(body, usage):
    events = [{"choices": [{"delta": {"content": "", "reasoning_content": None}, "finish_reason": "length"}]}]
    if body.get("stream_options", {}).get("include_usage"):
        events.append({"choices": [], "usage": usage})
    data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
    result = io.BytesIO(data)
    result.status = 200
    result.headers = {"X-Compute-Session-Id": "synthetic"}
    return result


original, corrected = map(functions, sys.argv[1:])
valid = {"prompt_tokens": 17, "completion_tokens": 8, "total_tokens": 25}
original["send"] = lambda body: response(body, valid)
try:
    original["generate"](True)
except TypeError:
    assert len(original["runs"]) == 0
else:
    raise AssertionError("original runner did not reproduce missing-usage TypeError")

assert corrected["chat"](True)["stream_options"] == {"include_usage": True}
assert "stream_options" not in corrected["chat"](False)
corrected["send"] = lambda body: response(body, valid)
assert corrected["generate"](True)["usage"] == valid
for usage in [None, {}, {**valid, "prompt_tokens": "17"}, {**valid, "completion_tokens": True}, {**valid, "completion_tokens": 9}, {**valid, "total_tokens": 24}]:
    corrected["send"] = lambda body, usage=usage: response(body, usage)
    try:
        corrected["generate"](True)
    except RuntimeError as error:
        assert str(error) == "USAGE_MISSING_OR_INVALID"
    else:
        raise AssertionError("accepted missing/invalid usage")
assert len(corrected["runs"]) == 1
corrected["stage"] = "proxy"
assert "stream_options" not in corrected["chat"](True), "strict client protocol cannot carry backend-only options"
corrected["send"] = lambda body: response({"stream_options": {"include_usage": True}}, valid)
assert corrected["generate"](True)["usage"] == valid
for tail in [b"", b"data: {}\n\n"]:
    data = b"data: " + json.dumps({"choices": []}).encode() + b"\n\n"
    if tail:
        data += b"data: [DONE]\n\n"
    try:
        corrected["decode_stream"](io.BytesIO(data + tail))
    except (AssertionError, RuntimeError):
        pass
    else:
        raise AssertionError("accepted missing DONE or output after DONE")
print("original missing-usage failure reproduced; corrected direct/proxy usage and negative SSE checks pass; no native module body executed")

tree = ast.parse(pathlib.Path(sys.argv[2]).read_text())
groups = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "server_groups" for target in node.targets))
assert eval(compile(ast.Expression(groups), "group-check", "eval"), {"groups": {1, 2, 3, 4}, "children": {name: SimpleNamespace(pid=pid) for name, pid in [("direct", 1), ("worker", 2), ("coordinator", 3)]}}) == {1, 4}
print("direct runtime and worker child groups counted; daemon wrapper groups excluded")
