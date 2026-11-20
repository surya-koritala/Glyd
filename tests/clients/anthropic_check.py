#!/usr/bin/env python3
"""The official `anthropic` Python SDK against a running `glyd serve`: messages, streams, thinking, tool use and results, stop sequences, counts, errors.

    pip install anthropic
    python3 tests/clients/anthropic_check.py http://127.0.0.1:8238 [--key KEY]

Every check prints PASS or FAIL; the exit status is the number of failures. The model's own words are not checked, only what the SDK parses of the reply.
"""
import argparse
import sys
import traceback

import anthropic

TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key", default="none")
    a = ap.parse_args()
    c = anthropic.Anthropic(base_url=a.base.rstrip("/"), api_key=a.key, max_retries=0, timeout=300)
    model = "claude-sonnet-5-5"
    # (the current SDK has no `temperature` argument: a request that wants one sends it in the body)
    det = {"extra_body": {"temperature": 0}}
    ask = [{"role": "user", "content": "Reply with the single word: pong"}]

    @check
    def messages():
        r = c.messages.create(model=model, max_tokens=300, messages=ask, system=[{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}], **det)
        assert r.type == "message" and r.role == "assistant" and r.stop_reason in ("end_turn", "max_tokens")
        assert r.content and r.content[0].type == "text" and r.usage.input_tokens > 0 and r.usage.output_tokens > 0

    @check
    def stream():
        with c.messages.stream(model=model, max_tokens=300, messages=ask, **det) as s:
            kinds = [e.type for e in s]
            text = s.get_final_text()
            final = s.get_final_message()
        assert kinds[0] == "message_start" and kinds[-1] == "message_stop", kinds
        assert "content_block_start" in kinds and "content_block_delta" in kinds and "content_block_stop" in kinds and "message_delta" in kinds
        assert text and final.stop_reason in ("end_turn", "max_tokens") and final.usage.output_tokens > 0

    @check
    def thinking_blocks():
        r = c.messages.create(model=model, max_tokens=1500, messages=ask, thinking={"type": "enabled", "budget_tokens": 1024})
        assert r.content[0].type == "thinking" and r.content[0].thinking, r.content
        with c.messages.stream(model=model, max_tokens=1500, messages=ask, thinking={"type": "enabled", "budget_tokens": 1024}) as s:
            kinds = [(e.type, getattr(getattr(e, "delta", None), "type", None)) for e in s]
            final = s.get_final_message()
        assert ("content_block_delta", "thinking_delta") in kinds and ("content_block_delta", "signature_delta") in kinds, kinds
        assert final.content[0].type == "thinking" and final.content[0].thinking

    @check
    def stop_sequence():
        r = c.messages.create(model=model, max_tokens=200, messages=[{"role": "user", "content": "Count from 1 to 9 separated by spaces, nothing else."}], stop_sequences=["5"], **det)
        assert r.stop_reason == "stop_sequence" and r.stop_sequence == "5", (r.stop_reason, r.stop_sequence, r.content)

    @check
    def tool_use_round_trip():
        r = c.messages.create(model=model, max_tokens=400, messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[TOOL], tool_choice={"type": "any"}, **det)
        tu = next(b for b in r.content if b.type == "tool_use")
        assert r.stop_reason == "tool_use" and tu.name == "get_weather" and tu.input["city"].lower().startswith("paris"), r
        assert tu.id.startswith("toolu_")
        msgs = [{"role": "user", "content": "What is the weather in Paris?"}, {"role": "assistant", "content": r.content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": "18 C, sunny"}]}]
        r2 = c.messages.create(model=model, max_tokens=500, messages=msgs, tools=[TOOL], **det)
        assert r2.stop_reason == "end_turn" and any(b.type == "text" and b.text for b in r2.content), r2

    @check
    def tool_use_streamed():
        with c.messages.stream(model=model, max_tokens=400, messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[TOOL], tool_choice={"type": "tool", "name": "get_weather"}, **det) as s:
            kinds = [getattr(getattr(e, "delta", None), "type", e.type) for e in s]
            final = s.get_final_message()
        assert "input_json_delta" in kinds, kinds
        tu = next(b for b in final.content if b.type == "tool_use")
        assert final.stop_reason == "tool_use" and tu.input["city"].lower().startswith("paris"), final

    @check
    def counts_and_models():
        n = c.messages.count_tokens(model=model, messages=ask, tools=[TOOL])
        assert n.input_tokens > 0
        ms = c.models.list().data
        assert len(ms) == 1 and ms[0].id and ms[0].created_at

    @check
    def errors():
        try:
            c.messages.create(model=model, messages=ask)
            raise AssertionError("no max_tokens answered")
        except TypeError:
            pass  # (the SDK itself wants max_tokens)
        try:
            c._client.post(a.base.rstrip("/") + "/v1/messages", json={"model": model, "messages": ask}, headers={"x-api-key": a.key, "anthropic-version": "2023-06-01"}).raise_for_status()
            raise AssertionError("no max_tokens answered")
        except Exception as e:
            assert getattr(e, "response", None) is not None and e.response.status_code == 400, e
            assert e.response.json()["error"]["type"] == "invalid_request_error", e.response.text
        try:
            c.messages.create(model=model, max_tokens=8, messages=[{"role": "user", "content": "word " * 200000}])
            raise AssertionError("a prompt past the context answered")
        except anthropic.BadRequestError as e:
            assert e.body["error"]["type"] == "invalid_request_error", e.body
        if a.key != "none":
            try:
                anthropic.Anthropic(base_url=a.base.rstrip("/"), api_key="wrong", max_retries=0).messages.create(model=model, max_tokens=8, messages=ask)
                raise AssertionError("a wrong key answered")
            except anthropic.AuthenticationError as e:
                assert e.body["error"]["type"] == "authentication_error", e.body

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
