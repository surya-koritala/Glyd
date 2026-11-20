#!/usr/bin/env python3
"""The official `openai` Python SDK against a running `glyd serve`: chat, stream, tools, response_format, the Responses API, fill in the middle, errors.

    pip install openai
    python3 tests/clients/openai_check.py http://127.0.0.1:8238 [--key KEY]

Every check prints PASS or FAIL; the exit status is the number of failures. The model's own words are not checked, only what the SDK parses of the reply (its shape, its types, its events).
"""
import argparse
import json
import sys
import traceback

import openai
from openai import OpenAI

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
WEATHER_RESP = {"type": "function", "name": "get_weather", "description": "Get the current weather in a city", "parameters": WEATHER["function"]["parameters"]}
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "temp_c": {"type": "integer"}}, "required": ["city", "temp_c"], "additionalProperties": False}

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
    ap.add_argument("--fim", action="store_true", help="the model has fill-in-the-middle tokens (Qwen's)")
    a = ap.parse_args()
    c = OpenAI(base_url=a.base.rstrip("/") + "/v1", api_key=a.key, max_retries=0, timeout=300)
    model = c.models.list().data[0].id
    ask = [{"role": "user", "content": "Reply with the single word: pong"}]

    @check
    def models():
        assert [m.id for m in c.models.list().data] == [model]
        assert c.models.retrieve(model).id == model
        try:
            c.models.retrieve("no-such-model")
            raise AssertionError("retrieve of another model answered")
        except openai.NotFoundError as e:
            assert e.body and e.body["type"] == "invalid_request_error", e.body

    @check
    def chat():
        r = c.chat.completions.create(model=model, messages=ask, max_tokens=64, temperature=0)
        assert r.object == "chat.completion" and r.choices[0].message.role == "assistant"
        assert r.choices[0].finish_reason in ("stop", "length") and r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens

    @check
    def chat_stream_with_usage():
        s = c.chat.completions.create(model=model, messages=ask, max_tokens=64, temperature=0, stream=True, stream_options={"include_usage": True})
        chunks = list(s)
        assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens > 0
        assert all(x.usage is None for x in chunks[:-1])
        assert "".join(x.choices[0].delta.content or "" for x in chunks[:-1] if x.choices)

    @check
    def seed_is_reproducible():
        r = [c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Name a color."}], max_tokens=24, temperature=1.0, seed=7).choices[0].message.content for _ in range(2)]
        assert r[0] == r[1], r

    @check
    def tools():
        r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[WEATHER], tool_choice="required", max_tokens=200, temperature=0)
        call = r.choices[0].message.tool_calls[0]
        assert r.choices[0].finish_reason == "tool_calls" and call.function.name == "get_weather"
        assert json.loads(call.function.arguments)["city"].lower().startswith("paris")
        # and the call's result back
        msgs = [{"role": "user", "content": "What is the weather in Paris?"}, r.choices[0].message, {"role": "tool", "tool_call_id": call.id, "content": "18 C, sunny"}]
        r2 = c.chat.completions.create(model=model, messages=msgs, tools=[WEATHER], max_tokens=200, temperature=0)
        assert r2.choices[0].message.content

    @check
    def tools_streamed_and_parallel_off():
        s = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Weather in Paris and in Rome?"}], tools=[WEATHER], parallel_tool_calls=False, tool_choice="required", max_tokens=300, temperature=0, stream=True)
        names, args = {}, {}
        for x in s:
            for t in (x.choices[0].delta.tool_calls or []) if x.choices else []:
                names.setdefault(t.index, t.function.name or names.get(t.index))
                args[t.index] = args.get(t.index, "") + (t.function.arguments or "")
        assert list(names) == [0] and json.loads(args[0]), (names, args)

    @check
    def response_format_json_schema():
        r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Weather in Paris is 18 degrees. Give it as JSON."}], max_tokens=200, temperature=0,
                                      response_format={"type": "json_schema", "json_schema": {"name": "weather", "schema": SCHEMA, "strict": True}})
        d = json.loads(r.choices[0].message.content)
        assert set(d) == {"city", "temp_c"} and isinstance(d["temp_c"], int), d

    @check
    def responses_text():
        r = c.responses.create(model=model, input="Reply with the single word: pong", instructions="Be brief.", max_output_tokens=64, temperature=0)
        assert r.object == "response" and r.status == "completed" and r.output_text, r
        assert r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens
        assert r.instructions == "Be brief." and r.store is False

    @check
    def responses_input_items():
        r = c.responses.create(model=model, input=[{"role": "developer", "content": "Answer in one word."}, {"role": "user", "content": [{"type": "input_text", "text": "Capital of France?"}]}], max_output_tokens=64, temperature=0)
        assert "paris" in r.output_text.lower(), r.output_text

    @check
    def responses_stream():
        with c.responses.stream(model=model, input="Count: one two three", max_output_tokens=64, temperature=0) as s:
            kinds = [e.type for e in s]
            final = s.get_final_response()
        assert kinds[0] == "response.created" and kinds[-1] == "response.completed", kinds
        assert "response.output_text.delta" in kinds and final.output_text and final.status == "completed"

    @check
    def responses_function_call_round_trip():
        r = c.responses.create(model=model, input="What is the weather in Paris?", tools=[WEATHER_RESP], tool_choice="required", max_output_tokens=200, temperature=0)
        call = next(i for i in r.output if i.type == "function_call")
        assert call.name == "get_weather" and json.loads(call.arguments)["city"].lower().startswith("paris")
        r2 = c.responses.create(model=model, tools=[WEATHER_RESP], max_output_tokens=200, temperature=0, input=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments},
            {"type": "function_call_output", "call_id": call.call_id, "output": "18 C, sunny"},
        ])
        assert r2.output_text, r2

    @check
    def responses_function_call_streamed():
        with c.responses.stream(model=model, input="What is the weather in Paris?", tools=[WEATHER_RESP], tool_choice="required", max_output_tokens=200, temperature=0) as s:
            kinds = [e.type for e in s]
            final = s.get_final_response()
        assert "response.function_call_arguments.delta" in kinds and "response.function_call_arguments.done" in kinds, kinds
        assert any(i.type == "function_call" for i in final.output)

    @check
    def responses_json_schema():
        r = c.responses.create(model=model, input="Weather in Paris is 18 degrees. Give it as JSON.", max_output_tokens=200, temperature=0,
                               text={"format": {"type": "json_schema", "name": "weather", "schema": SCHEMA, "strict": True}})
        d = json.loads(r.output_text)
        assert set(d) == {"city", "temp_c"} and isinstance(d["temp_c"], int), d

    @check
    def responses_is_stateless():
        try:
            c.responses.create(model=model, input="x", previous_response_id="resp_1")
            raise AssertionError("previous_response_id answered")
        except openai.BadRequestError as e:
            assert "previous_response_id" in e.message, e.message

    if a.fim:
        @check
        def fill_in_the_middle():
            r = c.completions.create(model=model, prompt="def add(a, b):\n    return ", suffix="\n\nprint(add(1, 2))\n", max_tokens=16, temperature=0)
            assert r.choices[0].text is not None and r.usage.completion_tokens > 0

    @check
    def errors():
        try:
            c.chat.completions.create(model=model, messages=ask, n=0)
            raise AssertionError("n=0 answered")
        except openai.BadRequestError as e:
            assert e.status_code == 400 and e.body["type"] == "invalid_request_error" and "param" in e.body, e.body
        try:
            c.chat.completions.create(model=model, messages=[{"role": "user", "content": "word " * 200000}], max_tokens=8)
            raise AssertionError("a prompt past the context answered")
        except openai.BadRequestError as e:
            assert e.code == "context_length_exceeded", e.body
        if a.key != "none":
            try:
                OpenAI(base_url=a.base.rstrip("/") + "/v1", api_key="wrong", max_retries=0).models.list()
                raise AssertionError("a wrong key answered")
            except openai.AuthenticationError as e:
                assert e.code == "invalid_api_key" and e.body["type"] == "authentication_error", e.body

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
