#!/usr/bin/env python3
"""Open WebUI (its container, set up as `glyd connect open-webui` says) against `glyd serve`: its model list has the Glyd model through the Ollama connection and the OpenAI one, and a chat through each answers.

    python3 tests/clients/openwebui_check.py http://127.0.0.1:3000

Open WebUI's first account is its admin: this makes one (a throwaway), so run it on a fresh container.
Every check prints PASS or FAIL; the exit status is the number of failures.
"""
import json
import sys
import time
import traceback

import requests

failures = []


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
    base = sys.argv[1].rstrip("/")
    for _ in range(240):
        try:
            if requests.get(base + "/health", timeout=3).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        print("FAIL open_webui_did_not_start")
        sys.exit(1)
    s = requests.Session()
    r = s.post(base + "/api/v1/auths/signup", json={"name": "glyd", "email": "glyd@example.com", "password": "glyd-test-password"}, timeout=60)
    assert r.status_code == 200, r.text
    s.headers["Authorization"] = "Bearer " + r.json()["token"]
    models = {}

    @check
    def the_model_list_has_the_glyd_model_from_both_connections():
        for _ in range(30):
            models.update({m["id"]: m for m in s.get(base + "/api/models", timeout=60).json()["data"]})
            owners = {m.get("owned_by") for m in models.values()}
            if {"ollama", "openai"} <= owners:
                break
            time.sleep(2)
        assert {"ollama", "openai"} <= {m.get("owned_by") for m in models.values()}, json.dumps(list(models.values()))[:2000]

    for owner in ("ollama", "openai"):
        @check
        def chat_through(owner=owner):
            ms = [m for m in models.values() if m.get("owned_by") == owner]
            assert ms, owner
            r = s.post(base + "/api/chat/completions", json={"model": ms[0]["id"], "stream": False, "messages": [{"role": "user", "content": "Reply with the single word: pong"}], "max_tokens": 300,
                                                          **({"chat_template_kwargs": {"enable_thinking": False}} if owner == "openai" else {})}, timeout=300)
            assert r.status_code == 200, r.text
            m = r.json()["choices"][0]["message"]["content"]
            assert m and m.strip(), r.text

        chat_through.__name__ = f"chat_through_{owner}"

    @check
    def a_streamed_chat():
        ms = [m for m in models.values() if m.get("owned_by") == "ollama"]
        r = s.post(base + "/api/chat/completions", json={"model": ms[0]["id"], "stream": True, "messages": [{"role": "user", "content": "Count: one two three"}], "max_tokens": 300}, stream=True, timeout=300)
        assert r.status_code == 200, r.text
        events = [l for l in r.iter_lines() if l.startswith(b"data:")]
        assert len(events) > 2 and events[-1].strip() == b"data: [DONE]", events[-3:]

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
