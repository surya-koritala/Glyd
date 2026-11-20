# Connecting tools

Any program that speaks OpenAI's or Anthropic's API on a base URL can use `glyd-engine serve`. The settings are the same for
all of them:

| Setting | Value |
| :--- | :--- |
| OpenAI base URL | `http://127.0.0.1:8000/v1` |
| Anthropic base URL | `http://127.0.0.1:8000` (no `/v1`) |
| API key | Anything, unless the server was started with `GLYD_API_KEY`: then that key |
| Model | The id `GET /v1/models` returns: the name you started the server with, tag included |

```bash
curl -s http://127.0.0.1:8000/v1/models
```

The server runs one model, so a tool that asks for another model name gets a 404 from `/v1/models/ID`. Use the id the list
shows. The context is the family's default from the [Models](models.md) page unless you start the server with `--context N`; an
agent that sends many tool definitions needs more.

## What has been tried

| Tool | Status |
| :--- | :--- |
| OpenAI Python SDK | Checked by `tests/clients/openai_check.py`, which runs chat, streaming, tools, `response_format`, the Responses API, fill in the middle and errors through the official SDK: `pip install openai`, then `python3 tests/clients/openai_check.py http://127.0.0.1:8000`. |
| Open WebUI | Recorded against the earlier Python server (v0.26.0, October 2026, [benchmarks/gpu/l4-onboarding-2026-10-01](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/l4-onboarding-2026-10-01)). Not recorded against `glyd-engine serve`. |
| Continue, Cline, LibreChat | Not tried. Add an OpenAI-compatible provider with the settings above. |
| Claude Code and the Anthropic SDK | Condor's server runs `ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude` ([condor/docs/CLIENTS.md](https://github.com/surya-koritala/Glyd/blob/main/condor/docs/CLIENTS.md)). `glyd-engine serve` has the same `/v1/messages` endpoint, and no Claude Code session against it is recorded. |

If you try one, say how it went in an [issue](https://github.com/surya-koritala/Glyd/issues). The page is updated from the
records, not from hope.

## Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
model = client.models.list().data[0].id
r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
print(r.choices[0].message.content)
```

## Open WebUI

In Open WebUI's admin settings, add an OpenAI-compatible connection with the base URL and key above. If Open WebUI runs in
Docker, the server must be reachable from the container: start it with `--host` set to an address the container reaches, and
use a key (see [Serving](serve.md)).
