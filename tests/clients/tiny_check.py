#!/usr/bin/env python3
"""The clients against `glyd serve` on a random model (`glyd-engine logitsmodel qwen35 DIR --servable`: Qwen3.8's layout, a character-level tokenizer, Qwen3.8's chat template). Its words are random characters, so
what is checked is what a client can rely on whatever the weights write: the shape of every reply and stream, the counts, the usage, and what a grammar holds: a JSON schema's document and a forced tool call's arguments
are valid (checked against the schema), through the OpenAI SDK (chat, Responses), the Anthropic SDK, the `ollama` library and LangChain's three chat models.

    pip install openai anthropic ollama requests jsonschema langchain-openai langchain-anthropic langchain-ollama
    python3 tests/clients/tiny_check.py http://127.0.0.1:8238

Every check prints PASS or FAIL; the exit status is the number of failures.
"""
import argparse
import json
import os
import sys
import traceback

import anthropic
import jsonschema
import ollama
import openai
import requests
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
# (the values are bounded: a model that writes random characters ends a free string once in a long while, and a reply that is cut at its length is not a document)
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"enum": ["Paris", "Rome", "Oslo"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 9},
        "sunny": {"type": "boolean"},
        "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
    },
    "required": ["city", "days", "sunny", "tags"],
    "additionalProperties": False,
}
TOOL_SCHEMA = {"type": "object", "properties": {"city": {"enum": ["Paris", "Rome"]}, "days": {"type": "integer", "minimum": 1, "maximum": 9}}, "required": ["city", "days"]}
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": TOOL_SCHEMA}}
OTHER = {"type": "function", "function": {"name": "get_time", "description": "Get the time", "parameters": {"type": "object", "properties": {"zone": {"enum": ["UTC", "CET"]}}, "required": ["zone"]}}}
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
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


def valid(doc, schema=SCHEMA):
    jsonschema.validate(doc, schema)


def valid_call(args, schema):
    """A forced call's arguments: each one written is of its schema (a call's parameters are the tool's, in any order and any number: `required` is not held)."""
    jsonschema.validate(args, {k: v for k, v in schema.items() if k != "required"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    c = OpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=300)
    model = c.models.list().data[0].id
    ask = [{"role": "user", "content": "What is the weather in Paris?"}]

    # ---- OpenAI: chat ----
    @check
    def openai_models_and_chat():
        assert c.models.retrieve(model).id == model
        r = c.chat.completions.create(model=model, messages=ask, max_tokens=24, extra_body=NOTHINK)
        assert r.object == "chat.completion" and r.choices[0].message.role == "assistant" and r.choices[0].finish_reason in ("stop", "length")
        assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens and 0 < r.usage.completion_tokens <= 24

    @check
    def openai_stream_and_usage():
        chunks = list(c.chat.completions.create(model=model, messages=ask, max_tokens=24, stream=True, stream_options={"include_usage": True}, extra_body=NOTHINK))
        assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens > 0
        assert chunks[0].choices[0].delta.role == "assistant"
        assert any(x.choices and x.choices[0].finish_reason for x in chunks)
        assert "".join(x.choices[0].delta.content or "" for x in chunks if x.choices)

    @check
    def openai_thinking_is_reasoning():
        # (the template opens the thinking: what a random model writes is its reasoning, and none of it is the answer)
        r = c.chat.completions.create(model=model, messages=ask, max_tokens=24)
        m = r.model_dump()["choices"][0]["message"]
        assert not (m.get("content") or "").strip() and r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 24, r.model_dump()
        chunks = list(c.chat.completions.create(model=model, messages=ask, max_tokens=24, stream=True))
        assert not "".join(x.choices[0].delta.content or "" for x in chunks if x.choices).strip()

    @check
    def openai_several_choices_and_a_seed():
        r = c.chat.completions.create(model=model, messages=ask, max_tokens=12, n=2, temperature=1.0, extra_body=NOTHINK)
        assert [x.index for x in r.choices] == [0, 1]
        q = lambda: c.chat.completions.create(model=model, messages=ask, max_tokens=12, temperature=1.0, seed=5, extra_body=NOTHINK).choices[0].message.content
        assert q() == q()

    # ---- OpenAI: what a grammar holds ----
    @check
    def openai_json_schema_is_valid():
        for stream in (False, True):
            r = c.chat.completions.create(model=model, messages=ask, max_tokens=200, temperature=1.0, extra_body=NOTHINK, stream=stream,
                                          response_format={"type": "json_schema", "json_schema": {"name": "w", "schema": SCHEMA, "strict": True}})
            text = "".join(x.choices[0].delta.content or "" for x in r if x.choices) if stream else r.choices[0].message.content
            valid(json.loads(text))

    @check
    def openai_json_object_is_json():
        r = c.chat.completions.create(model=model, messages=ask, max_tokens=200, temperature=1.0, extra_body=NOTHINK, response_format={"type": "json_object"})
        # (free strings are as long as the weights make them: a document is whole where the reply ended, else it was cut at its length)
        if r.choices[0].finish_reason == "stop":
            assert isinstance(json.loads(r.choices[0].message.content), dict)
        else:
            assert r.choices[0].finish_reason == "length" and r.choices[0].message.content.lstrip().startswith("{")

    @check
    def openai_forced_tool_calls_are_valid():
        for choice, names in (({"type": "function", "function": {"name": "get_weather"}}, ["get_weather"]), ("required", ["get_weather", "get_time"])):
            r = c.chat.completions.create(model=model, messages=ask, tools=[TOOL, OTHER], tool_choice=choice, max_tokens=300, temperature=1.0, extra_body=NOTHINK)
            call = r.choices[0].message.tool_calls[0]
            assert r.choices[0].finish_reason in ("tool_calls", "stop") and call.function.name in names, r
            args = json.loads(call.function.arguments)
            valid_call(args, TOOL_SCHEMA if call.function.name == "get_weather" else OTHER["function"]["parameters"])
        s = c.chat.completions.create(model=model, messages=ask, tools=[TOOL], tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=300, temperature=1.0, stream=True, extra_body=NOTHINK)
        name, args = None, ""
        for x in s:
            for t in (x.choices[0].delta.tool_calls or []) if x.choices else []:
                name = name or t.function.name
                args += t.function.arguments or ""
        assert name == "get_weather"
        valid_call(json.loads(args), TOOL_SCHEMA)

    @check
    def openai_tool_result_is_read():
        call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 1})}}
        r = c.chat.completions.create(model=model, tools=[TOOL], max_tokens=16, extra_body=NOTHINK,
                                      messages=ask + [{"role": "assistant", "content": "", "tool_calls": [call]}, {"role": "tool", "tool_call_id": "call_1", "content": "18 C"}])
        assert r.usage.prompt_tokens > 20

    # ---- OpenAI: Responses ----
    @check
    def responses_shapes_and_forced_call():
        r = c.responses.create(model=model, input="Hi", max_output_tokens=16, reasoning={"effort": "none"})
        assert r.object == "response" and r.status in ("completed", "incomplete") and r.output and r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens
        r = c.responses.create(model=model, input="Weather?", max_output_tokens=300, reasoning={"effort": "none"}, tool_choice={"type": "function", "name": "get_weather"},
                               tools=[{"type": "function", "name": "get_weather", "description": "w", "parameters": TOOL_SCHEMA}])
        call = next(i for i in r.output if i.type == "function_call")
        assert call.name == "get_weather"
        valid_call(json.loads(call.arguments), TOOL_SCHEMA)
        r = c.responses.create(model=model, input="Weather?", max_output_tokens=200, reasoning={"effort": "none"},
                               text={"format": {"type": "json_schema", "name": "w", "schema": SCHEMA, "strict": True}})
        valid(json.loads(r.output_text))
        with c.responses.stream(model=model, input="Hi", max_output_tokens=16, reasoning={"effort": "none"}) as s:
            kinds = [e.type for e in s]
        assert kinds[0] == "response.created" and kinds[-1] in ("response.completed", "response.incomplete"), kinds

    @check
    def openai_errors():
        for kwargs in ({"n": 0}, {"temperature": "hot"}):
            try:
                c.chat.completions.create(model=model, messages=ask, **kwargs)
                raise AssertionError(f"{kwargs} answered")
            except openai.BadRequestError as e:
                assert e.body["type"] == "invalid_request_error", e.body

    # ---- Anthropic ----
    ac = anthropic.Anthropic(base_url=base, api_key="none", max_retries=0, timeout=300)
    ATOOL = {"name": "get_weather", "description": "Get the weather", "input_schema": TOOL_SCHEMA}

    @check
    def anthropic_messages_stream_and_counts():
        r = ac.messages.create(model="claude-sonnet-5-5", max_tokens=16, messages=ask)
        assert r.type == "message" and r.stop_reason in ("end_turn", "max_tokens", "stop_sequence") and r.usage.input_tokens > 0 and r.usage.output_tokens > 0
        with ac.messages.stream(model="claude-sonnet-5-5", max_tokens=16, messages=ask) as s:
            kinds = [e.type for e in s]
            final = s.get_final_message()
        assert kinds[0] == "message_start" and kinds[-1] == "message_stop" and "message_delta" in kinds, kinds
        assert final.usage.output_tokens > 0
        assert ac.messages.count_tokens(model="claude-sonnet-5-5", messages=ask, tools=[ATOOL]).input_tokens > 10

    @check
    def anthropic_forced_tool_use_is_valid():
        r = ac.messages.create(model="claude-sonnet-5-5", max_tokens=300, messages=ask, tools=[ATOOL], tool_choice={"type": "tool", "name": "get_weather"})
        tu = next(b for b in r.content if b.type == "tool_use")
        assert tu.name == "get_weather" and tu.id.startswith("toolu_") and r.stop_reason == "tool_use", r
        valid_call(tu.input, TOOL_SCHEMA)
        with ac.messages.stream(model="claude-sonnet-5-5", max_tokens=300, messages=ask, tools=[ATOOL], tool_choice={"type": "any"}) as s:
            kinds = [getattr(getattr(e, "delta", None), "type", e.type) for e in s]
            final = s.get_final_message()
        assert "input_json_delta" in kinds, kinds
        valid_call(next(b for b in final.content if b.type == "tool_use").input, TOOL_SCHEMA)

    @check
    def anthropic_thinking_block():
        r = ac.messages.create(model="claude-sonnet-5-5", max_tokens=1500, messages=ask, thinking={"type": "enabled", "budget_tokens": 1024})
        assert r.content[0].type == "thinking" and r.content[0].thinking.strip(), r.content

    @check
    def anthropic_errors():
        try:
            ac.messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "word " * 5000}])
            raise AssertionError("a prompt past the context answered")
        except anthropic.BadRequestError as e:
            assert e.body["error"]["type"] == "invalid_request_error", e.body

    # ---- Ollama ----
    oc = ollama.Client(host=base)

    @check
    def ollama_chat_generate_and_names():
        r = oc.chat(model="m", messages=ask, options={"num_predict": 16}, think=False)
        assert r.done and r.message.role == "assistant" and r.eval_count > 0 and r.prompt_eval_count > 0 and r.done_reason in ("stop", "length")
        parts = list(oc.chat(model="m", messages=ask, options={"num_predict": 16}, think=False, stream=True))
        assert parts[-1].done and not any(p.done for p in parts[:-1])
        r = oc.generate(model="m", prompt="Hi", options={"num_predict": 16}, think=False)
        assert r.done and r.eval_count > 0
        m = oc.list().models[0]
        assert oc.show(m.model).details is not None and oc.ps().models[0].model == m.model
        assert requests.post(base + "/api/embed", json={"model": "m", "input": "x"}).status_code == 501

    @check
    def ollama_format_schema_is_valid():
        r = oc.chat(model="m", messages=ask, format=SCHEMA, options={"num_predict": 200, "temperature": 1.0}, think=False)
        valid(json.loads(r.message.content))

    # ---- LangChain ----
    from langchain_anthropic import ChatAnthropic
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    def get_weather(city: str, days: int) -> str:
        """Get the weather."""
        return "sunny"

    @check
    def langchain_chat_models_and_forced_tools():
        models = [
            ChatOpenAI(base_url=base + "/v1", api_key="none", model="m", max_tokens=300, max_retries=0, extra_body=NOTHINK),
            ChatAnthropic(base_url=base, api_key="none", model="claude-sonnet-5-5", max_tokens=300, max_retries=0),
            ChatOllama(base_url=base, model="m", num_predict=300, reasoning=False),
        ]
        for m in models:
            assert str(m.invoke("Hi").content).strip(), m
        forced = [
            models[0].bind_tools([get_weather], tool_choice="get_weather"),
            models[1].bind_tools([get_weather], tool_choice={"type": "tool", "name": "get_weather"}),
        ]
        for m in forced:
            call = m.invoke("What is the weather in Paris?").tool_calls[0]
            assert call["name"] == "get_weather"
            valid_call(call["args"], TOOL_SCHEMA)
        out = models[0].with_structured_output(SCHEMA, method="json_schema", strict=True).invoke("Weather in Paris?")
        valid(out)

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
