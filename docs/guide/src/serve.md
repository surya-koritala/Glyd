# Serving and the APIs

```bash
glyd serve google/gemma-4-12B-it
```

`serve` loads the model and answers on `http://127.0.0.1:8000`. The chat page is at `/`. `run` starts the same server.

After the release script, a background service is also running on `127.0.0.1:11434`. It loads a model when a request names it
and unloads it when idle, and holds two at a time by default:

```bash
glyd pull gemma4:12b
curl localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:12b","messages":[{"role":"user","content":"hi"}]}'
glyd ps                  # what is loaded, and when it is unloaded
glyd stop gemma4:12b     # unload it now (glyd stop --all)
glyd service install     # run it at every login; glyd service runs it in this terminal
```

The service loads only models already pulled, and `glyd run` chats through it where it runs. Its options (`--keep-alive`,
`--max-loaded`, `--port`) and what it leaves open are in
[the service page](service.md).

## Options

| Flag | What |
| :--- | :--- |
| `--port N` | The port. Without it, 8000, then each next port up to 8020 that is free. With it, that port or a refusal. |
| `--host H` | Listen on H. Without it, only this computer, and a request whose Host or Origin is not this computer's is refused (421, 403). With it, those checks are off. |
| `--context N` | The context in tokens (`--ctx` is the same). Without it, what fits: [Models](models.md) has each family's default. |
| `--users N` | How many requests run at once, 1 to 8, 4 by default. Batching runs on the Qwen3 path only; other families take one request at a time and the rest wait in order. |
| `--host-memory GIB` | The computer's memory a mixture of experts may use for its experts. |
| `MODEL:LEVEL` | A level: [Choosing a level](levels.md). |

## The key

```bash
GLYD_API_KEY=secret glyd serve google/gemma-4-12B-it
```

With a key, every path under `/v1` needs `Authorization: Bearer secret`. `/`, `/health` and `/stats` stay open. Put the key in
the environment: a `--api-key` flag shows in `ps`. With `--host` set to an address other than this computer's, anyone who
can reach it can use the model, and the traffic is plain HTTP: use a key and an SSH tunnel or a VPN.

## What it answers

| Path | What |
| :--- | :--- |
| `POST /v1/chat/completions` | OpenAI chat, JSON or a stream. Tools, `response_format` (`json_object`, `json_schema`), `tool_choice`, and vLLM's `guided_*` fields. |
| `POST /v1/completions` | A plain completion; `suffix` makes it fill in the middle for a model that has the tokens. |
| `POST /v1/responses` | OpenAI's Responses API, stateless. |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic's Messages API, for the Anthropic SDKs and Claude Code: thinking blocks, tools, streaming. |
| `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` | Ollama's API, for programs that talk to Ollama. `--ollama-port N` listens on Ollama's port too, where it is free. |
| `GET /v1/models`, `/v1/models/ID` | The one model. `id` is the name you started it with. |
| `GET /health` | `{"status":"ok"}`, no key needed. |
| `GET /v1/glyd/info` | What the server is and holds: model, level, precision, device, limits, memory. |
| `GET /stats` | The scheduler's counts (lossless engine). |

A first request with `curl`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say pong."}],"max_tokens":64}'
```

A thinking model thinks before it answers. To turn that off for a request, send
`"chat_template_kwargs": {"enable_thinking": false}`; `run` takes `--no-think`.

Errors are OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`. A prompt past the context is 400 with the code
`context_length_exceeded`; a wrong key is 401 `invalid_api_key`.

## The full reference

Every field the server reads, refuses or ignores, the states from loading to failed, the exit codes and the status file are in
the [API reference](api.md) (the contract, `glyd-serve/1`). The background service is in [its own page](service.md).
