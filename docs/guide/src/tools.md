# Connecting tools

`glyd connect [TOOL]` finds the server that runs here (ports 8000 to 8020, then Ollama's 11434, or `--port N`) and prints
what each tool needs, with the real port and model. It changes nothing and never prints a key. Start a server first
(`glyd serve MODEL`).

| Tool | What it is given | `glyd connect` |
| :--- | :--- | :--- |
| OpenAI SDKs, LangChain `ChatOpenAI`, aider | `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`, `OPENAI_API_KEY` (any, or the server's key) | `openai` |
| Claude Code, Anthropic SDKs | `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `ANTHROPIC_API_KEY` | `anthropic` |
| `ollama` libraries, LangChain `ChatOllama` | `OLLAMA_HOST=http://127.0.0.1:8000` (`--ollama-port 11434` for programs that look at Ollama's port) | `ollama` |
| Open WebUI (container) | its Ollama and OpenAI connections set to the server | `open-webui` |
| Continue | `~/.continue/config.yaml`: a chat model through the OpenAI API | `continue` |

The model name can be anything: the server answers with the one it runs. `GET /v1/models` shows it. Without a key set, any
key works; with `GLYD_API_KEY` set on the server, use that key.

An agent that sends many tool definitions needs more context than the default (the [Models](models.md) page has each
family's): start the server with `--context N`.

## Containers

A container on Linux reaches a server on this computer only from the host's network (`docker run --network=host ...`). Docker
Desktop and Podman reach it as `host.docker.internal` and `host.containers.internal`, and the server's Host check answers those
two names and no others. A server started with `--host 0.0.0.0` has the Host and Origin checks off: give it a key.
`glyd connect open-webui` prints the whole `docker run` line.

## What is checked

The clients are checked in [tests/clients/](https://github.com/surya-koritala/Glyd/tree/main/tests/clients): `run.sh URL [KEY]`
runs the OpenAI, Anthropic, Ollama, LangChain and Open Responses checks, each with the newest release of its library, against a
running server, and `openwebui_check.py` checks an Open WebUI container. Claude Code and Open WebUI were run against the
server; the [API reference](api.md), Connecting tools, says on what. Continue's settings are printed by `glyd connect`; a session with Continue
is not recorded.

If you try another tool, say how it went in an [issue](https://github.com/surya-koritala/Glyd/issues).

## Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
model = client.models.list().data[0].id
r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
print(r.choices[0].message.content)
```

