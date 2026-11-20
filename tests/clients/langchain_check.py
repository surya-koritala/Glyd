#!/usr/bin/env python3
"""LangChain's chat models for OpenAI, Anthropic and Ollama against a running `glyd serve`, each set up as `glyd connect` says.

    pip install langchain-openai langchain-anthropic langchain-ollama
    python3 tests/clients/langchain_check.py http://127.0.0.1:8238 [--key KEY]

Every check prints PASS or FAIL; the exit status is the number of failures.
"""
import argparse
import sys
import traceback

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return "sunny"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key", default="none")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    models = {
        "openai": ChatOpenAI(base_url=base + "/v1", api_key=a.key, model="glyd", temperature=0, max_tokens=300, extra_body=nothink, max_retries=0),
        "anthropic": ChatAnthropic(base_url=base, api_key=a.key, model="claude-sonnet-5-5", max_tokens=300, max_retries=0),
        "ollama": ChatOllama(base_url=base, model="glyd", temperature=0, num_predict=300, reasoning=False, client_kwargs={"headers": {"Authorization": f"Bearer {a.key}"}} if a.key != "none" else {}),
    }
    for kind, m in models.items():
        def invoke(m=m):
            r = m.invoke("Reply with the single word: pong")
            assert str(r.content).strip(), r

        invoke.__name__ = f"{kind}_invoke"
        check(invoke)

        def stream(m=m):
            parts = [c.content for c in m.stream("Reply with the single word: pong")]
            assert "".join(p if isinstance(p, str) else "".join(x.get("text", "") for x in p) for p in parts).strip()

        stream.__name__ = f"{kind}_stream"
        check(stream)

        def tools(m=m):
            r = m.bind_tools([get_weather]).invoke("What is the weather in Paris?")
            assert r.tool_calls and r.tool_calls[0]["name"] == "get_weather" and r.tool_calls[0]["args"]["city"].lower().startswith("paris"), r

        tools.__name__ = f"{kind}_tools"
        check(tools)

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
