# Using it from other apps

The server (`./setup.sh` or `tools/serve.sh`) speaks the OpenAI chat API and the Anthropic messages API on your own machine:

```
OpenAI-compatible      http://127.0.0.1:8080/v1          (chat completions: /v1/chat/completions)
Anthropic-compatible   http://127.0.0.1:8080/v1/messages
model name             qwen3.8-flash-next (any name works)
API key                any
```

If you started it on another port (`tools/serve.sh 8090`), change the port.

It listens on 127.0.0.1 only, so it's reachable from the same machine. Reaching it from another machine or a container isn't set up yet.

## Generic setup

Most OpenAI-compatible apps and SDKs want a base URL and a key. The official OpenAI SDKs read these environment variables:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=local
```

Python:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
r = client.chat.completions.create(model="qwen3.8-flash-next", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

In apps with a settings screen (Open WebUI, Continue, Cline, LibreChat and so on), add an "OpenAI-compatible" provider with the base URL
`http://127.0.0.1:8080/v1`, any key, and model `qwen3.8-flash-next`. I haven't tested each app yet; see below.

For Anthropic-style clients, the base URL is `http://127.0.0.1:8080` (no `/v1`), e.g. `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`.

Thinking is off by default. Send `reasoning_effort` to turn it on.

## What is tested

Two scripts run against a live server. Standard library only:

```
python3 tests/test_server.py --url http://127.0.0.1:8080
python3 tests/test_agents.py --url http://127.0.0.1:8080
```

`tests/test_server.py` (9 checks, all passed in `results/v0.1.0/fast/test_server.txt` and on the A10 run in `results/v0.1.0/a10`):

```
OpenAI     plain answer, streamed answer, one tool call (not streamed), the tool's result sent back and used
Anthropic  plain answer, streamed answer, one tool use (not streamed)
OpenAI     a client that drops a stream halfway; the next request still gets its own answer
```

`tests/test_agents.py` (19 checks). On my RTX 4080 SUPER, precise mode, 2026-10-15. The v0.1.1 column is a build of v0.1.1's
server code (same source as the release, built on that machine), not the release download itself:

```
                                                           v0.1.1    next release
GET /v1/models                                             pass      pass
tool call, streamed (OpenAI)                               pass      pass
tool_use, streamed (Anthropic)                             pass      pass
two tool calls in one answer, whole and streamed           pass      pass
parallel_tool_calls: false keeps one call                  pass      pass
tool_choice "required", one tool                           FAIL      pass
tool_choice "required", two tools                          FAIL      pass
tool_choice "required", streamed                           FAIL      pass
tool_choice "none"                                         FAIL      pass
tool_choice named function                                 pass      pass
Anthropic tool_choice any / tool / none                    FAIL      pass
tool call with thinking on, whole and streamed             pass      pass
12 tool round trips in one conversation                    pass      pass
response_format json_object                                FAIL      pass
response_format json_schema                                FAIL      pass
```

What failed in v0.1.1:

- `tool_choice: "required"` could call a function that isn't in `tools` (it wrote `unknown`, `weather`, `example_function_name`).
- `tool_choice: "none"` put the raw `<tool_call>...</tool_call>` text into the answer instead of answering in words.
- Anthropic `tool_choice: {"type": "tool"}` returned the tool_use block with `stop_reason: "end_turn"` instead of `"tool_use"`.
  (The "any" and "none" rows failed on the made-up names and the raw tool text above.)
- `response_format` with `json_object` or `json_schema` was refused with a 400.

The next release fixes these. With the fix, `required` with two tools called a real one 30 times out of 30 (three prompts, ten
times each, temperature 0.7).

JSON mode is not grammar-constrained decoding. The server puts the instruction (and the schema, if you send one) in the system
prompt, turns thinking off for that request, and starts the answer with `{`. In my run of 10 prompts x 3 (`json_object`, and
`json_schema` with nested arrays, enums and numbers), 30 of 30 answers parsed and had the schema's required keys with the right
types. It can still write JSON that doesn't match a schema; check it if your app depends on it. `response_format` together with
`tools` in one request is refused for now.

The 12-round tool loop reads only the new part of the conversation on each turn (about 98% of the prompt reused from the turn
before), so each round took about 0.25 s here.

## Not tested yet

- Open WebUI, Hermes, Continue, Cline and LibreChat end to end. The official `openai` (3.31.0) and `anthropic` (1.14.0) Python
  SDKs stream tool calls fine, including while their request waits behind another one.
- Hermes Agent needs at least 64K tokens of context (its quickstart says so); the server's default is 32K, so start it with
  `GLYD_CTX=65536 tools/serve.sh` (`$env:GLYD_CTX = '65536'` on Windows).
- `usage.prompt_tokens_details.cached_tokens` is not filled in; the reuse shows in the server's log only.

If you try one, tell me how it went in an issue. This list is tracked in
[#10](https://github.com/surya-koritala/condor/issues/10).

## Things that will trip up agent frameworks

- Up to 4 requests at once (`GLYD_USERS=N tools/serve.sh`, 1 to 8), each streaming from the start; total speed stays about one
  user's, so four agents each see a quarter of it. Past N, requests wait their turn in the order they came: a streamed request
  that is waiting gets its headers within a second and then a `: waiting` comment line every second (Anthropic API: a `ping`
  event), so client read timeouts don't fire; the official SDKs skip these. A client that gives up while waiting is dropped
  before its prompt is read. A new long prompt pauses the other streams while it is read. `tests/test_concurrent.py` and
  `tests/test_parallel.py` check this.
- Context is 32,768 tokens by default. Agent prompts with many tool definitions get long fast; start the server with a bigger context
  (`GLYD_CTX=65536 tools/serve.sh`, up to 262144) if you need it.

If something doesn't work, open an issue with the client, its version, and what the server printed.
