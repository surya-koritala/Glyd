#!/usr/bin/env python3
"""The HTTP tests of the Open Responses compliance suite (openresponses/openresponses, bin/compliance-test.ts: basic-response, system-prompt, tool-calling, multi-turn, streaming-response, and image-input where the
model reads images), run in Python against a running `glyd serve`: each response, and each event of a stream, is checked against the specification's OpenAPI schema (tests/clients/openresponses/openapi.json, release 2026-04-24).
The suite's WebSocket and compaction tests are not run.

    pip install requests jsonschema
    python3 tests/clients/openresponses_check.py http://127.0.0.1:8238 [--key KEY] [--images]

Every check prints PASS or FAIL; the exit status is the number of failures.
"""
import argparse
import json
import os
import sys
import traceback

import requests
from jsonschema import Draft202012Validator

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "openresponses", "openapi.json")))
failures = []
NOTHINK = {"reasoning": {"effort": "none"}}


DEFS = json.loads(json.dumps(SPEC["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))


def validator(name):
    return Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": DEFS})


def errors(v, doc):
    out = []
    for e in sorted(v.iter_errors(doc), key=lambda e: list(e.path)):
        out.append("/".join(str(p) for p in e.path) + ": " + e.message[:200])
    return out[:12]


def check(fn):
    try:
        fn()
        print(f"PASS {fn.__name__}")
    except Exception:
        failures.append(fn.__name__)
        print(f"FAIL {fn.__name__}")
        traceback.print_exc()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key", default="none")
    ap.add_argument("--images", action="store_true")
    a = ap.parse_args()
    url = a.base.rstrip("/") + "/v1/responses"
    h = {"Authorization": f"Bearer {a.key}"}
    resource = validator("ResponseResource")
    stream_events = {}
    for k, s in SPEC["components"]["schemas"].items():
        t = s.get("properties", {}).get("type", {}).get("enum") or ([s["properties"]["type"]["const"]] if "const" in s.get("properties", {}).get("type", {}) else None)
        if k.endswith("StreamingEvent") and t:
            stream_events[t[0]] = validator(k)

    def post(body):
        r = requests.post(url, json={"model": "m", **body}, headers=h, timeout=300)
        assert r.status_code == 200, r.text
        return r.json()

    def whole(name, body, output_type=None):
        def run():
            d = post({**NOTHINK, "max_output_tokens": 300, **body})
            e = errors(resource, d)
            assert not e, e
            assert d["output"], d
            if output_type:
                assert any(i["type"] == output_type for i in d["output"]), [i["type"] for i in d["output"]]
            else:
                assert d["status"] == "completed", d["status"]

        run.__name__ = name
        check(run)

    msg = lambda role, content: {"type": "message", "role": role, "content": content}
    whole("basic_response", {"input": [msg("user", "Say hello in exactly 3 words.")]})
    whole("system_prompt", {"input": [msg("system", "You are a pirate. Always respond in pirate speak."), msg("user", "Say hello.")]})
    whole("multi_turn", {"input": [msg("user", "My name is Alice."), msg("assistant", "Hello Alice! Nice to meet you. How can I help you today?"), msg("user", "What is my name?")]})
    whole("tool_calling", {"input": [msg("user", "What's the weather like in San Francisco?")], "tool_choice": "required", "tools": [{
        "type": "function", "name": "get_weather", "description": "Get the current weather for a location",
        "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}}, "required": ["location"]}}]}, "function_call")
    if a.images:
        whole("image_input", {"input": [msg("user", [{"type": "input_text", "text": "What do you see in this image? Answer in one sentence."}, {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}])]})

    @check
    def streaming_response():
        r = requests.post(url, json={"model": "m", **NOTHINK, "max_output_tokens": 300, "stream": True, "input": [msg("user", "Count from 1 to 5.")]}, headers=h, stream=True, timeout=300)
        assert r.status_code == 200, r.text
        evs, bad = [], []
        for line in r.iter_lines():
            if line.startswith(b"data: "):
                e = json.loads(line[6:])
                evs.append(e)
                v = stream_events.get(e["type"])
                if v is None:
                    bad.append(f"{e['type']}: an event the specification does not have")
                else:
                    bad += [f"{e['type']} {x}" for x in errors(v, e)]
        assert evs and evs[0]["type"] == "response.created" and evs[-1]["type"] == "response.completed", [e["type"] for e in evs][:5]
        assert [e["sequence_number"] for e in evs] == list(range(len(evs)))
        assert not bad, bad[:12]

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
