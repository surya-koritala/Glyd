"""Constrained decoding of a running `glyd-engine serve`: every reply held to a grammar is valid, and what the masks cost.

    python constrained_check.py http://127.0.0.1:8000 [--key KEY] [--bench [--plain]] [--runs 3] [--tokens 200] [--out results.json]
    python constrained_check.py --selftest            (no server: the script's own validator)

Standard library only (`jsonschema` is used where it is installed). The checks, each at temperature 0 and at 0.8, whole and streamed where it says so; a line `ok` or `FAIL` for each, exit status 1 if one FAILs:
  json_object        the reply parses and is an object
  json_schema        response_format json_schema (strict) with nested objects, arrays, an enum, numbers, a null, required keys: the reply validates against the schema, or the finish reason is `length`
  guided_*           guided_regex, guided_choice, guided_json and structured_outputs, the same
  tools              tool_choice naming a function, and `required` among two: finish_reason tool_calls, a known function, arguments that validate against its schema
  streamed           the streamed reply of a schema is the same kind of reply
  refusals           a schema llguidance cannot compile and two forms at once are 400, in words

--bench: decode tokens per second (first token to last, streamed, the same prompt and length) with no schema (greedy and sampled) and with one (greedy and sampled), `--runs` times each, the median.
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

RESULTS = []
FAILED = []


def say(kind, name, detail=""):
    print(f"{kind:5} {name}" + (f" -- {detail}" if detail != "" else ""), flush=True)
    RESULTS.append({"kind": kind, "name": name, "detail": str(detail)[:2000]})
    if kind == "FAIL":
        FAILED.append(name)


def check(name, ok, detail=""):
    say("ok" if ok else "FAIL", name, detail)
    return ok


# ---- a validator for the schemas used here (jsonschema where it is installed) ----------------------------------------------------------------------

def validate(value, schema, path="$"):
    """The problems of `value` against `schema` (a list of words; empty: valid). The keywords the schemas below use."""
    try:
        import jsonschema

        errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.path))
        return [f"{'/'.join(map(str, e.path)) or '$'}: {e.message}" for e in errs]
    except ImportError:
        pass
    out = []
    t = schema.get("type")
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    if t == "integer":
        if not (isinstance(value, int) and not isinstance(value, bool)):
            return [f"{path}: not an integer"]
    elif t == "number":
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return [f"{path}: not a number"]
    elif t in kinds and not isinstance(value, kinds[t]):
        return [f"{path}: not {t}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: not one of {schema['enum']}")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                out.append(f"{path}: missing {k}")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                out += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                out.append(f"{path}: extra key {k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            out += validate(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path}: above {schema['maximum']}")
    return out


def selftest():
    s = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"enum": ["x", "y"]}, "c": {"type": "array", "items": {"type": "string"}}}, "required": ["a"], "additionalProperties": False}
    assert validate({"a": 1, "b": "x", "c": ["q"]}, s) == []
    assert validate({"b": "x"}, s) and validate({"a": 1.5}, s) and validate({"a": 1, "b": "z"}, s) and validate({"a": 1, "z": 1}, s) and validate({"a": 1, "c": [1]}, s)
    assert validate({"a": True}, s)
    print("selftest ok")


# ---- the server -------------------------------------------------------------------------------------------------------------------------------

class Server:
    def __init__(self, base, key=None):
        self.base, self.key = base.rstrip("/"), key
        self.model = json.load(self._open("/v1/models"))["data"][0]["id"]

    def _open(self, path, body=None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
        return urllib.request.urlopen(req, timeout=600)

    def chat(self, **kw):
        body = dict(model=self.model, chat_template_kwargs={"enable_thinking": False}, **kw)
        try:
            with self._open("/v1/chat/completions", body) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def stream(self, **kw):
        """The events of a streamed chat: (time, chunk) for each, the last one with the usage."""
        body = dict(model=self.model, chat_template_kwargs={"enable_thinking": False}, stream=True, stream_options={"include_usage": True}, **kw)
        out = []
        with self._open("/v1/chat/completions", body) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    out.append((time.perf_counter(), json.loads(line[5:])))
        return out


PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "height_m": {"type": "number"},
        "email": {"type": ["string", "null"]},
        "role": {"enum": ["engineer", "designer", "manager"]},
        "skills": {"type": "array", "items": {"type": "string"}},
        "address": {"type": "object", "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}, "required": ["city", "zip"], "additionalProperties": False},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "height_m", "email", "role", "skills", "address", "active"],
    "additionalProperties": False,
}
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
BOOK = {"type": "function", "function": {"name": "book_table", "description": "Book a table at a restaurant.", "parameters": {"type": "object", "properties": {"restaurant": {"type": "string"}, "people": {"type": "integer"}, "time": {"type": "string"}}, "required": ["restaurant", "people"]}}}


def reply_of(resp):
    c = resp["choices"][0]
    return c["message"].get("content") or "", c["finish_reason"], c["message"].get("tool_calls") or []


def json_ok(text, schema=None):
    try:
        v = json.loads(text)
    except ValueError as e:
        return False, f"not JSON: {e}: {text[:200]!r}"
    if schema is None:
        return isinstance(v, dict), "" if isinstance(v, dict) else f"not an object: {text[:200]!r}"
    problems = validate(v, schema)
    return not problems, f"{problems[:3]}: {text[:200]!r}"


def correctness(s, temperatures=(0.0, 0.8)):
    ask = [{"role": "user", "content": "Invent a person for a staff directory and describe them."}]
    for t in temperatures:
        tag = f"T={t}"
        st, r = s.chat(messages=ask, temperature=t, max_tokens=700, response_format={"type": "json_object"})
        text, fin, _ = reply_of(r) if st == 200 else ("", "", [])
        ok, why = json_ok(text) if fin == "stop" else (fin == "length", f"finish {fin}")
        check(f"json_object {tag}", st == 200 and ok, why if st == 200 else r)

        rf = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": PERSON}}
        for i in range(2 if t else 1):
            st, r = s.chat(messages=ask, temperature=t, max_tokens=900, response_format=rf)
            text, fin, _ = reply_of(r) if st == 200 else ("", "", [])
            ok, why = json_ok(text, PERSON)
            check(f"json_schema strict {tag} #{i}", st == 200 and (ok or fin == "length"), (why, fin) if st == 200 else r)

        st, r = s.chat(messages=[{"role": "user", "content": "Give a US phone number for the office."}], temperature=t, max_tokens=40, guided_regex=r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}")
        text = reply_of(r)[0] if st == 200 else ""
        check(f"guided_regex {tag}", st == 200 and re.fullmatch(r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}", text) is not None, text if st == 200 else r)

        choices = ["positive", "negative", "neutral"]
        st, r = s.chat(messages=[{"role": "user", "content": "Sentiment of: 'I loved it'?"}], temperature=t, max_tokens=20, guided_choice=choices)
        text = reply_of(r)[0] if st == 200 else ""
        check(f"guided_choice {tag}", st == 200 and text in choices, text if st == 200 else r)

        small = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
        for what, extra in (("guided_json", {"guided_json": small}), ("structured_outputs.json", {"structured_outputs": {"json": small}})):
            st, r = s.chat(messages=[{"role": "user", "content": "What is 6 times 7? Answer as JSON."}], temperature=t, max_tokens=60, **extra)
            text, fin, _ = reply_of(r) if st == 200 else ("", "", [])
            ok, why = json_ok(text, small)
            check(f"{what} {tag}", st == 200 and ok, why if st == 200 else r)

        q = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
        named = {"type": "function", "function": {"name": "get_weather"}}
        st, r = s.chat(messages=q, temperature=t, max_tokens=200, tools=[WEATHER, BOOK], tool_choice=named)
        text, fin, calls = reply_of(r) if st == 200 else ("", "", [])
        good = bool(calls) and calls[0]["function"]["name"] == "get_weather"
        why = ""
        if good:
            ok, why = json_ok(calls[0]["function"]["arguments"], WEATHER["function"]["parameters"])
            good = ok
        check(f"tool_choice named {tag}", st == 200 and fin in ("tool_calls", "stop") and good, (fin, calls, why) if st == 200 else r)
        st, r = s.chat(messages=[{"role": "user", "content": "Book a table for 4 at Chez Paul at 8pm."}], temperature=t, max_tokens=200, tools=[WEATHER, BOOK], tool_choice="required")
        text, fin, calls = reply_of(r) if st == 200 else ("", "", [])
        schemas = {x["function"]["name"]: x["function"]["parameters"] for x in (WEATHER, BOOK)}
        good = bool(calls) and calls[0]["function"]["name"] in schemas
        if good:
            good, why = json_ok(calls[0]["function"]["arguments"], schemas[calls[0]["function"]["name"]])
        check(f"tool_choice required {tag}", st == 200 and fin == "tool_calls" and good, (fin, calls) if st == 200 else r)

        ev = s.stream(messages=ask, temperature=t, max_tokens=900, response_format=rf)
        text = "".join((c["choices"][0]["delta"].get("content") or "") for _, c in ev if c.get("choices"))
        fin = [c["choices"][0]["finish_reason"] for _, c in ev if c.get("choices") and c["choices"][0].get("finish_reason")]
        ok, why = json_ok(text, PERSON)
        check(f"json_schema streamed {tag}", ok or fin == ["length"], (why, fin))

    st, r = s.chat(messages=ask, max_tokens=20, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"$ref": "#/nowhere"}}}}})
    check("a schema that cannot be compiled is 400, in words", st == 400 and "grammar" in json.dumps(r), r)
    st, r = s.chat(messages=ask, max_tokens=20, guided_regex="a+", guided_choice=["x"])
    check("two forms at once are 400, in words", st == 400 and "one form" in json.dumps(r), r)
    st, r = s.chat(messages=ask, max_tokens=20, guided_regex="(")
    check("a regular expression that does not parse is 400", st == 400, r)


def rate(ev):
    """Decode tokens per second of a streamed reply: from the first chunk that holds text to the last, over the tokens between (the usage says how many in all)."""
    ts = [t for t, c in ev if c.get("choices") and (c["choices"][0]["delta"].get("content"))]
    usage = [c["usage"] for _, c in ev if c.get("usage")]
    n = usage[0]["completion_tokens"] if usage else len(ts)
    if len(ts) < 3:
        return None, n
    return (len(ts) - 1) / (ts[-1] - ts[0]), n


def bench(s, runs, tokens, plain=False):
    """The same prompt and length each way. A schema's reply must go on for `tokens` tokens: one string of free text."""
    free = {"type": "object", "properties": {"story": {"type": "string"}}, "required": ["story"], "additionalProperties": False}
    ask = [{"role": "user", "content": 'Write a long story about a lighthouse keeper. Reply as JSON: {"story": "..."}'}]
    s.stream(messages=ask, temperature=0.0, max_tokens=8)  # (warm)
    rows = {}
    legs = (
        ("no schema, greedy", dict(temperature=0.0)),
        ("no schema, sampled (T 0.7, top_k 40)", dict(temperature=0.7, top_k=40)),
        ("schema, greedy", dict(temperature=0.0, response_format={"type": "json_schema", "json_schema": {"name": "story", "strict": True, "schema": free}})),
        ("schema, sampled (T 0.7, top_k 40)", dict(temperature=0.7, top_k=40, response_format={"type": "json_schema", "json_schema": {"name": "story", "strict": True, "schema": free}})),
    )
    for label, kw in legs[:2] if plain else legs:
        rs = []
        for _ in range(runs):
            ev = s.stream(messages=ask, max_tokens=tokens, **kw, **({} if "response_format" in kw else {"ignore_eos": True}))
            r, n = rate(ev)
            if r:
                rs.append(r)
        rows[label] = statistics.median(rs) if rs else None
        say("BENCH", label, f"{rows[label]:.1f} tok/s (median of {len(rs)}; {tokens} tokens asked)" if rs else "no result")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("--key")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--plain", action="store_true", help="with --bench: the legs with no schema alone (a server that has none)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.url:
        ap.error("a server's URL")
    s = Server(a.url, a.key)
    print(f"model {s.model}")
    if a.bench:
        bench(s, a.runs, a.tokens, a.plain)
    else:
        correctness(s)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(RESULTS, f, indent=1)
    if FAILED:
        print(f"\n{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)


if __name__ == "__main__":
    main()
