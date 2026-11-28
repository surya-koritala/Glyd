# glyd serve: the HTTP API, glyd-serve/1

Notes for a program that starts `glyd serve` (or `glyd-engine serve`) on its own machine and sends it whole requests, as
Glyd-Compute's service adapter does. `glyd-serve/1` is what `GET /v1/glyd/info` says in `api`; a change that breaks a
client of this page gets a new number. The numbers below are from one run on an RTX 4080 SUPER with Qwen3.5-9B at the
linnet level: [benchmarks/gpu/rtx4080s-serve-probe-2026-10-19](../benchmarks/gpu/rtx4080s-serve-probe-2026-10-19/README.md).
The contract tests are in glyd-engine/src/server.rs and cli.rs (the `#[test]`s named below) and glyd-gpu-py/tests/test_onboard.py.

## Start, ready, stop

```
glyd serve Qwen/Qwen3.5-9B:linnet                         # the Python front end: hands a level to the engine
glyd-engine serve Qwen/Qwen3.5-9B:linnet --ctx 4096       # the engine itself
HF_HUB_OFFLINE=1 GLYD_API_KEY=... glyd-engine serve /path/to/models--glyd--Qwen3.5-9B-linnet/snapshots/073a2132... \
    --ctx 4096 --port 8238 --status-file /run/glyd.status  # offline: the directory alone is read
```

- `MODEL:level` takes penguin (lossless), kestrel, linnet, wren and hummingbird (the level of a mixture of experts: Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark, the Nemotron-H layout). A tag that names no level is refused (exit 1). A level the engine does not run
  for the model, or a model that does not fit, is refused with the reason (exit 3, on stderr and in the status file).
  The model and the level are the ones asked for, or the start does not go on without your word: where the level does not load as asked, the first lines say
  which steps would take it there and what each costs, and the start is refused (exit 3) unless `--yes` (or `GLYD_DEGRADE=yes`) takes them; a lower level, or a step that
  costs more than 30% of the speed, is such a step (`glyd run` at a terminal asks instead). Two steps are taken and said in those lines, with no word asked: a shorter
  context than the one asked for, and the experts or the dense layers of a large model kept in the computer's memory where that costs no more than 30%.
- `glyd serve` of a level, of a Gemma 4 model, of a glyd-v7 directory or of a glyd/ repo vLLM cannot read goes to the engine without GLYD_ENGINE=1.
  `glyd serve` of any other lossless model is vLLM's unless GLYD_ENGINE=1, as before.
- A Hub name with a level reads the base model's config.json (the cache, else the Hub) and loads glyd/NAME-LEVEL from the
  cache (else downloads it). Offline, give the snapshot directory: nothing else is read.
- The address is 127.0.0.1:8000, or the next free port up to 8020; `--port N` is that port or a refusal. `--host H`
  listens on H: an address that is not this computer's own (0.0.0.0, a LAN address) needs `GLYD_API_KEY` (or `--api-key`) or `--insecure-no-auth`, else it is a refusal before the model loads. `--allowed-origins URL,URL` lets those pages' scripts call the API (CORS answers for exactly those origins, never `*`; with none listed CORS is off), `--allowed-hosts NAME,NAME` adds names the Host may be. `--ollama-port N` also listens on N (Ollama's own is 11434), only
  where N is free (else it says so and goes on): for the programs that look for Ollama at its port; Ollama's API is at
  the server's own port too.
- The GPU is the one with the most free memory of those CUDA_VISIBLE_DEVICES shows. The context is `--ctx N` (`--context`);
  without it, what fits, up to 32,768 at a level (served to several users, what fits each of them, at least 4,096). `--users N`
  (4 if not given) is how many requests run at once: 1 to 64 for a model on one GPU, as many as the GPU's memory holds (the server says how many); a model
  with a draft model, a model split across GPUs, or one with an operation that takes one sequence, runs one request at a time (the others wait in order).
- The keys and values are in bf16 pages, or in the lossless 12-bit pages where `GLYD_KV=lossless` asks for them (where the model's attention has heads of 128, 256 or 512 channels or is a latent attention, and no draft model runs beside it) and the plan finds that
  they reserve no more than bf16 pages for the context and users asked: both formats are lossless, so the plan never takes a lower level for the 12-bit pages. The plan chooses and the loader
  follows it. The start's `Plan:` line says which (`keys and values in bf16` / `in 12-bit pages`) and `fit --json` has it as `kv_format` (`bf16` or `12-bit`); `/v1/glyd/info`'s `precision.kv` says what was loaded.
- `--with MODEL2` serves a second model from the same process, on the same GPU: MODEL on its port and MODEL2 on `--with-port N`, else the next
  free port after MODEL's. Each is a server of its own (its scheduler, its `/v1/glyd/info`, `/metrics` and requests); the first loads as it would alone and takes
  up to half of the memory left for its keys and values, then the second loads in what is left and takes the rest. Both are models the description executor serves (each loads as `serve` would load it alone, as MODEL:wren).
  The status file says ready when both serve; a model that fails to load, or whose server does not start, ends the start of both. A GPU error after
  which the GPU cannot be used ends the process, both models with it, as it ends a server of one.
- A model too big for one GPU (Kimi-K3, DeepSeek-V4, GLM-5.3-Flash, Muse Spark) uses all the GPUs shown. On datacenter
  GPUs (A100, H100, B200) that reach each other directly, Kimi-K3 and DeepSeek-V4 put every layer across all of them:
  each GPU holds a slice of every big matrix and the slices are gathered GPU to GPU, so every GPU works on every token.
  The start checks two gathers of known values and refuses (exit 3) where the GPUs' writes to each other arrive wrong.
  Elsewhere, or with `--split CUTS --devices GPUS`, the layers are split between the GPUs (one at work at a time). The
  numbers: [research/tp-8xa100-2026-10-22](../research/tp-8xa100-2026-10-22/README.md). The Qwen3.5 layout is cut over the GPUs at every level (the lossy levels in stages, one sequence at a time) and Nemotron-H at penguin.
  Qwen3.8-Flash-Next (penguin, `glyd run`, `serve` and `fit`) is cut over the GPUs too, the same way: one GPU where one holds it, its layers split where
  not (`--split CUTS --devices GPUS` chooses); Condor's `qwenserve CKPT PACK --gpus 0,1` cuts the layers over those cards with each part's experts in a cache
  of its own (one request at a time across GPUs, as the release did).
- The environment it reads: GLYD_API_KEY (or VLLM_API_KEY; `--api-key` works but `ps` shows it), HF_HUB_CACHE or HF_HOME
  for a Hub name, HF_HUB_OFFLINE=1. A directory needs nothing else: a CUDA build loads no GPU library (its kernels are its own),
  so there is none to find (`GLYD_GPU_LIB=PATH` loads Glyd's library for the checks that compare with it; an AMD build looks for
  `libglyd_gpu_hip` beside the program).

States:

| state | how it shows |
| :--- | :--- |
| loading | the process runs; the port is not bound (connections are refused); the status file is touched as it goes |
| ready | the port is bound, `/health` is 200, `/v1/glyd/info` says `"state":"ready"`, the status file says `ready` (2.6 s after the start in the record) |
| failed before ready | exit 3 (1 for a wrong flag or level name), the reason on stderr and in the status file |
| failed after ready | `/v1/glyd/info` says `"state":"failed"` with `error: {at, message}`, the engine's first failure. It stays failed: start the process again |

Stop: SIGTERM drains (Operations, below; a second one ends the process at once, exit 143); SIGINT ends the process at once (exit 130), and a request in flight is cut. The GPU's memory is the
driver's again when the process is gone (the record: 2,223 MiB used before the start and after the stop).

## The key and the address checks

- With a key, every path under `/v1` needs `Authorization: Bearer KEY` (the scheme in any case), else
  `401` (`{"error": {"message": ..., "type": "authentication_error", "param": null, "code": "invalid_api_key"}}`, with `WWW-Authenticate: Bearer`). `/`, `/health`, `/ready` and `/stats` are open; `/metrics` needs the key. The key is compared as SHA-256 digests in constant time.
- On this computer's own address (the default, and a loopback `--host`): a Host that is not localhost, 127.0.0.1, [::1], a container's name for this computer (`host.docker.internal`, `gateway.docker.internal`, `kubernetes.docker.internal`, `host.containers.internal`, `host.lima.internal`, `host.minikube.internal`) or a name in `--allowed-hosts` is 421. On every address an Origin that is not the server's own (`http://` and the Host; `https://` too where the server is open to the network, for a proxy that ends TLS) or one in `--allowed-origins` is 403 (a CORS preflight too). A program that sends no Origin (curl, the OpenAI libraries) is answered.
- Limits: a body of 16 MiB (413), a header block of 64 KiB or 256 headers (431), 50,000 open connections and 8,192 requests in flight (503), the request line and headers within 30 s and the body within 120 s (408), a write to a client that reads nothing 60 s.

## Operations: metrics, readiness and drain

`GET /metrics` (guarded by the key where there is one: Prometheus takes `authorization: {credentials: ...}` in its scrape config) answers, from the first request on:

| metric | kind | what |
|---|---|---|
| `glyd_requests_total{api}` | counter | requests taken for `chat_completions`, `completions`, `messages`, `responses`, `ollama` |
| `glyd_requests_refused_total{code}` | counter | requests refused before any work: 401, 403, 421, 408, 413, 431, 503 (a drain) |
| `glyd_requests_in_flight` | gauge | requests taken and not yet answered to their end |
| `glyd_prompt_tokens_total`, `glyd_prefix_cache_hit_tokens_total`, `glyd_generated_tokens_total` | counter | prompt tokens run or found in the prefix cache, those found, tokens written |
| `glyd_sequences_finished_total{reason}` | counter | `stop`, `length`, `stopped` (the client left or cancelled), `error` |
| `glyd_time_to_first_token_seconds` | histogram | from the sequence reaching the scheduler to its first token (the queue and the prompt included) |
| `glyd_time_per_output_token_seconds` | histogram | between two tokens of one reply |
| `glyd_request_duration_seconds` | histogram | a sequence from the scheduler to its end |
| `glyd_queue_depth`, `glyd_sequences_running` | gauge | sequences waiting for room, sequences running |
| `glyd_kv_cache_pages`, `glyd_kv_cache_pages_used`, `glyd_kv_cache_usage_ratio` | gauge | the KV cache's pages (for a model whose sequences are slots, its slots) |
| `glyd_gpu_memory_used_bytes{gpu}`, `glyd_gpu_memory_total_bytes{gpu}` | gauge | the computer's GPU memory by `nvidia-smi` (read at most every 5 s; left out where there is none) |
| `glyd_ready`, `glyd_draining`, `glyd_uptime_seconds`, `glyd_build_info{version,model}` | gauge | the server's own state |

`glyd service` answers `/ready` and `/metrics` with its own state (`glyd_service_models_loaded`, requests, refusals, in flight); a model's tokens and latencies are on that model's own `glyd serve`.

SIGTERM drains: the server stops taking requests (every new one for an API is `503` with `Retry-After: 1`, `/ready` is `503 draining`; `/health`, the model list and `/metrics` still answer), lets the requests in flight end, and exits 0 when the last has ended, or 1 when `--drain-timeout SECS` (default 30) passes with some still running (they end with the process). A second SIGTERM exits at once (143). Ctrl-C is not a drain. `glyd service` drains the same way and then unloads its models. Not done yet: the experts' cache usage.

Traces: with `OTEL_EXPORTER_OTLP_ENDPOINT` (a base URL; `/v1/traces` is added) or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (the whole URL) set, every sequence sends a span `glyd.request` (attributes `gen_ai.request.model`, `glyd.prompt_tokens`, `glyd.generated_tokens`, `glyd.finish_reason`; status error where it ended in an error) with the children `glyd.queue` (waiting for room), `glyd.prefill` (the prompt, to the first token) and `glyd.decode` (first token to the end), as OTLP over HTTP with JSON, batched by a thread of their own (up to 100 spans or a second). `OTEL_EXPORTER_OTLP_HEADERS` (`k=v,k2=v2`) carries a collector's key, `OTEL_SERVICE_NAME` (default `glyd`) names the service, `OTEL_SDK_DISABLED=true` turns it off. Off with no endpoint: nothing is sent. A collector that is down is said on stderr at most every 30 s, the spans are dropped and counted (`glyd_trace_spans_dropped_total`, `glyd_trace_spans_exported_total`), and no request is slowed. Only `http/json` is sent (the protocol variable is read and said where it asks for another); a request's incoming `traceparent` is not yet its span's parent.

## Connections

The connections are tasks of an async front end (tokio; `glyd-engine/src/server/frontend.rs`), not threads: up to 50,000 are open at once (`MAX_OPEN`), and a connection that waits for its request, for its reply or for a slow client to read costs a task. What the client sends is read within bounds (the head within 30 s of the first byte, in 64 KiB and 256 headers; the body within 120 s; an answer's write within 60 s), and a connection that sends nothing is answered 408 and closed by its time. A client that sends `Connection: keep-alive` and gets an answer with a `Content-Length` keeps the connection for its next request (60 s idle); an answer that streams (server-sent events, Ollama's lines) ends its connection, and a client that sends no word of it is answered and closed, as before. What a client sends while it is answered is held to 64 KiB (past it the connection is let go); the bodies of requests share 1 GiB; there are at most 50,000 open connections and 8,192 requests in flight, and the next is answered 503 with `Retry-After`. A request with a repeated or non-numeric `Content-Length`, or with a `Transfer-Encoding`, is refused 400. A client that disconnects while its reply is being made ends the sequence: the engine does not go on generating for nobody. A client that reads slowly holds back the thread that writes its answer (a pipe of 64 chunks sits between them), not memory.

Every API front end (OpenAI's completions, Anthropic's Messages, OpenAI's Responses, Ollama's) sends its sequences to the scheduler through `server::Remote`; `glyd-engine/src/engine_api.rs` names that as one request-level interface (`submit` a request, read its `Stream`, `cancel` it, `stats`), which no handler calls yet. The load test (`a_thousand_streaming_connections_all_end_without_an_error`) holds 1,000 streaming connections at once on a scheduler over a CPU fake model: all are answered 200 with their events and their `[DONE]`.

## Endpoints

| | key | |
| :--- | :--- | :--- |
| `GET /health` | no | `{"status":"ok"}`: the process is up (a load balancer's liveness probe) |
| `GET /ready` | no | `{"status":"ready"}` where the model is loaded and the server takes requests; 503 `{"status":"loading"}` while it loads (`qwenserve`, which listens while it loads; `serve` binds its port after the load), 503 `{"status":"draining"}` after SIGTERM (readiness probe) |
| `GET /metrics` | with a key | Prometheus text (version 0.0.4): see Operations |
| `GET /v1/models`, `GET /v1/models/ID` | yes | the one model: `id` is MODEL as typed (with its tag, or the directory's path), `max_model_len` the context; another id is 404 |
| `POST /v1/chat/completions` | yes | a chat, JSON or SSE |
| `POST /v1/completions` | yes | a plain completion, JSON or SSE; `suffix` makes it a fill in the middle (see below) |
| `POST /v1/responses` | yes | OpenAI's Responses API, stateless: JSON or SSE (see below) |
| `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, `/api/pull`, `/api/delete` | yes | Ollama's API (see below); `/api/embed` and the model-making calls answer 501 |
| `POST /v1/messages`, `/v1/messages/count_tokens` | yes | Anthropic's Messages API, JSON or SSE (see below; not tracked by id) |
| `POST /v1/glyd/tokenize` | yes | a request's prompt counted, nothing run |
| `GET /v1/glyd/info` | yes | what the server is and holds |
| `GET /v1/glyd/requests/ID` | yes | how a request stands |
| `POST /v1/glyd/requests/ID/cancel` | yes | a request asked to end |
| `GET /v1/glyd/kv/events` | yes | the cache's blocks stored and removed by name, with their tokens (see below; 404 unless the operator turns them on) |
| `GET /stats` | no | the scheduler's counts |

A method an API path does not take is 405 (the health, readiness, metrics, stats and events paths answer 404 to it), a path there is not 404.

Every error is OpenAI's: `{"error": {"message", "type", "param", "code"}}` with the status a client keys on. `type` is `invalid_request_error` (400, 404, 405, 413),
`authentication_error` (401), `rate_limit_error` (429) or `server_error` (5xx); `code` is null except `invalid_api_key` (401) and
`context_length_exceeded` (a prompt, or a prompt and its length, past the context). The Host and Origin refusals (421 and 403) have a body of their own, `{"error": {"message", "type": "Forbidden", "code": 421}}`
with the status as a number in `code` (the last sample of Replies, below): no 403 is `permission_error`. `stream_options.include_usage` puts `"usage": null` in every chunk of a stream and the usage in a
last chunk with no choices, as OpenAI's does. Where the server found some of a prompt's tokens in its cache (a conversation's earlier turns, a prompt it has run before), the usage has
`prompt_tokens_details.cached_tokens`: they are among `prompt_tokens`, and were not run again. The Responses API has them as `input_tokens_details.cached_tokens`.

The cache is one for every caller of a server: a request finds what any request before it left (a conversation's earlier turns, a prompt it has run before), and by what it
finds, or by how soon its first token comes, a caller can tell whether a prompt that begins as its own was run before. Callers who are not to learn that of each other each send a
`cache_salt` (vLLM's field: any string that is not empty; a gateway before the callers sets it): a request with one finds only what requests with the same one left, and what it
leaves is found by them alone; requests with none are a scope of their own. The usage tells a request how many tokens it found only where the request has a `cache_salt`:
the others' usage has no cached tokens, though the cache serves them. A server whose callers are all one user can be told so (`GLYD_ENGINE_CACHED_USAGE=1` in its
environment, which `glyd service` passes to the models it starts) and then tells every request. The server's address does not decide this: one on this computer's own address
may answer callers from anywhere, behind a proxy. The same holds for the server's own count of the tokens found: `prefix_hit_tokens` and `prefix_hit_rate_percent` of `/stats` are
null, and `/metrics` has no `glyd_prefix_cache_hit_tokens_total`, unless the server was told its callers are one user.

What a finished request leaves is kept until the server needs its place, the oldest going first. `cache_pin_seconds` (a number up to 3600) asks that what this request leaves
goes after the others' for that long; it is a preference among what is kept and never makes another request wait. The Anthropic endpoint sets it from `cache_control` (300, or
3600 with `ttl: "1h"`). A request that begins as an earlier one did has that start without running it, where the model allows: on the models whose keys and values are paged (Gemma 4 and Muse Spark)
the server finds the blocks of 16 tokens a prompt begins with by what they hold, whether the request that ran them is still running, is kept, has grown far past them or has ended,
for as long as their memory has not been needed for something else; the Qwen3.5 layout (its Gated DeltaNet layers beside paged attention) finds the start any earlier request ran by the
states it keeps at every chunk of a prompt, in a pool of fixed size made at load (512 MB by default, boundaries every chunk; a development build's `GLYD_ENGINE_STATE_SNAP_MB` and `GLYD_ENGINE_STATE_EVERY` set them), and so does Nemotron-H with its Mamba-2 states.
A request that comes while another still runs the start waits for it rather than running the same tokens beside it. `GLYD_ENGINE_KEEP=0` empties a finished request's slot at once, so
nothing is kept for a conversation's next turn.

## Connecting tools

The default address is `http://127.0.0.1:8000` (`--port N` for another; if 8000 is busy and no port was asked for, the server takes the next free one up to 8020 and says which at its start).
`glyd connect` finds the server that runs here (8000 to 8020, then Ollama's 11434, or `--port N`) and prints the lines each tool needs, with its real port and model; it changes nothing, and never prints a key (it says `$GLYD_API_KEY`).

| tool | what it is given | `glyd connect` |
| :--- | :--- | :--- |
| OpenAI SDKs, LangChain `ChatOpenAI`, aider | `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`, `OPENAI_API_KEY` (any, or the server's key) | `openai` |
| Claude Code, Anthropic SDKs, LangChain `ChatAnthropic` | `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `ANTHROPIC_API_KEY` | `anthropic` |
| `ollama` libraries, LangChain `ChatOllama`, programs with an Ollama setting | `OLLAMA_HOST=http://127.0.0.1:8000` (`--ollama-port 11434` for the programs that look at Ollama's own port) | `ollama` |
| Open WebUI (container) | its Ollama and OpenAI connections set to the server (below) | `open-webui` |
| Continue | `~/.continue/config.yaml`: a chat model through the OpenAI API, autocomplete through Ollama's | `continue` |

A container on Linux reaches a server on this computer's own address only from the host's network: `docker run --network=host -e PORT=3000 -e OLLAMA_BASE_URL=http://127.0.0.1:8000 -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 ... ghcr.io/open-webui/open-webui:main`.
Docker Desktop (macOS, Windows) and Podman reach it as `host.docker.internal` and `host.containers.internal`; the Host check answers those two names (it still refuses every other name and every other site's Origin), and
the address stays this computer's own. A server on another interface (`--host 0.0.0.0`) needs a key (or `--insecure-no-auth`), answers any Host (or those of `--allowed-hosts`) and still refuses another page's Origin. On Linux, Docker's `host.docker.internal:host-gateway` reaches the Docker bridge, not 127.0.0.1: use `--host 172.17.0.1` (with a key) or `--network=host`. WSL reaches a server on Windows' 127.0.0.1 as `localhost` (mirrored networking or localhost forwarding); another name goes in `--allowed-hosts`.

The GPU job runs the clients against a random model made there (`glyd-engine logitsmodel qwen35 DIR --servable`: Qwen3.8's layout, a character-level tokenizer and Qwen3.8's chat template; `tests/clients/tiny_check.py`): the weights write random characters, so it checks what a client can rely on whatever they write (the shape of every reply, stream and count, and, where a grammar holds the reply, a JSON schema's document and a forced tool call's arguments being valid) through the OpenAI SDK (chat and Responses), the Anthropic SDK, the `ollama` library and LangChain. Open WebUI and Continue against a real model are run in the testing phase.

The clients are checked in `tests/clients/`: `run.sh URL [KEY]` runs the OpenAI, Anthropic, Ollama, LangChain and Open Responses checks (the HTTP tests of the Open Responses compliance suite, with every response and stream event checked against the specification's schema in `tests/clients/openresponses/`), each with the newest release of its library, against a running server; `openwebui_check.py`
checks an Open WebUI container. Claude Code and Open WebUI were run against Qwen3-8B on an RTX 4080 SUPER for v0.34.0 (the numbers are in the pull request).

## /v1/responses

The request is read as the chat request it is the same as and run as any chat request is, so its refusals, its grammar and its tools are the chat path's. Read: `input` (a string, or a list
of `message`, `function_call`, `function_call_output` items; `reasoning` items are skipped; `input_text`, `output_text` and `input_image` parts), `instructions`, `tools` (`function` only),
`tool_choice`, `parallel_tool_calls`, `text.format` (`json_schema`, `json_object`, `text`), `max_output_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `reasoning.effort`, `stream`. The server
keeps no conversations: `previous_response_id`, `conversation`, `background`, `item_reference` and the tools that run at OpenAI's end (search, code, computer) are 400 with the reason, and
`store` is ignored (the response says `store: false`). The response has the thinking as a `reasoning` item (`content[].reasoning_text`), the answer as a `message`, each call as a
`function_call`; `status` is `incomplete` with `incomplete_details.reason` `max_output_tokens` where the length cut it. A stream is the events `response.created`, `response.in_progress`,
`response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `.done`, `response.reasoning_text.delta`, `.done`, `response.function_call_arguments.delta`, `.done`,
`response.output_item.done`, then `response.completed` (or `response.incomplete`, `response.failed`), each numbered by `sequence_number`.

## /v1/messages

Anthropic's Messages API for the Anthropic SDKs and Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:8238 ANTHROPIC_API_KEY=any`; with a server key, `x-api-key: KEY` or a bearer token). The request is read as the chat
request it is the same as and run as any chat request is. Read: `system` (a string or blocks), `messages` (`text`, `image`, `tool_use`, `tool_result` blocks; `thinking` blocks of an earlier turn are skipped),
`tools` (custom tools; the ones Anthropic runs, `web_search` and the like, are left out), `tool_choice` (`auto`, `any`, `tool`, `none`, `disable_parallel_tool_use`), `thinking` (`enabled` or `adaptive` shows the model's thinking in
a `thinking` block with an empty signature, which this server cannot make; else it is off), `stop_sequences`, `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stream`; `cache_control` does not choose what is cached: a request finds what the
server holds of its prompt whatever it marks. A request with a marker has what it leaves held before other requests' for the marker's time (5 minutes, an hour with
`ttl: "1h"`) when the server needs a place; never so long that another request waits. The usage says how many tokens a request found (`cache_read_input_tokens`) and, for a request with a `cache_control` marker, how many more it left that a request with the same start would find (`cache_creation_input_tokens`: the cache keeps every prompt, at no price), with `input_tokens` the rest; in a stream, in `message_delta`. Both are told as the cached count is (below). A stream is `message_start`, `ping`, each block's `content_block_start`, `content_block_delta` (`text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`) and
`content_block_stop`, `message_delta` (the `stop_reason`: `end_turn`, `max_tokens`, `stop_sequence` with the string, `tool_use`; the usage) and `message_stop`, with a `ping` while a request waits for its turn. Errors are
`{"type": "error", "error": {"type", "message"}}` (`invalid_request_error`, `authentication_error`, `not_found_error`, `api_error`). `/v1/models` carries `type`, `display_name` and `created_at` beside OpenAI's fields.

## /api (Ollama)

For the programs that talk to Ollama (Open WebUI, Continue, LangChain, the `ollama` libraries). `POST /api/chat` and `/api/generate` stream JSON lines (`application/x-ndjson`; `stream` is on unless it is false) and end with
`done`, `done_reason`, the durations (nanoseconds) and `prompt_eval_count` and `eval_count`; with `tools` the reply comes whole (a call is sent at once, its arguments an object). Read: `messages` (with `images` in base64, `tool_calls`,
and `tool` messages by `tool_name`), `prompt`, `system`, `suffix` (fill in the middle) and `raw`, `tools`, `format` (`"json"` or a JSON schema), `think` (a boolean or a level; the thinking comes in `thinking`),
and `options`: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `num_predict`, `repeat_penalty`, `presence_penalty`, `frequency_penalty` (`num_ctx` is the server's `--context`; the others have no effect).
`keep_alive` and `model` are read and ignored: the server holds one model and every request is answered by it; `/api/tags` and `/api/ps` list it by its short name in the model manifest, as Ollama writes one (`qwen3:8b`; the names `glyd pull` takes), or as `REPO:latest` where the manifest has none, (a GGUF served as a checkpoint is named by its file and its quantization, `qwen3:8b-bf16` or `my-model:f16`, and answers to the same names without the tag) and `/api/pull`, `/api/show` and `/api/delete` take those names and `/api/show` says its `capabilities`
(`completion`, `tools`, `thinking`, `vision`, `insert`) and its `context_length`. `/api/pull` of that model says `success`, of another is 404 with what to run; `/api/delete` of the served model is refused.
Errors are `{"error": "..."}`. The key guards `/api` as it guards `/v1`.

## Tool calls and thinking by family

The server reads each family's tool calls and reasoning from the markup its chat template shows the model, and gives them as OpenAI's `tool_calls` and `reasoning_content` and Anthropic's `tool_use` and `thinking` blocks, whole
and streamed (a call's name when it is complete, its arguments as they are written; the thinking as it comes). A reader (`glyd-engine/src/reader.rs`) turns a family's own markup into the shape the parsers read; the rest are read as they are.

| family | tool calls | reasoning | forced calls (`tool_choice`) |
| :--- | :--- | :--- | :--- |
| Qwen3, Qwen3-Coder | hermes `<tool_call>{json}</tool_call>` | `<think>` | grammar, or the prompt opens the call |
| Qwen3.5, Qwen3.6, Qwen3.8, Nemotron 3 | XML `<tool_call><function=NAME><parameter=KEY>`, typed by the tool's schema | `<think>` | grammar, or the prompt opens the call |
| GLM-5.3 | `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>` | `<think>` | grammar, or the prompt opens the call |
| DeepSeek-V4 | DSML `<｜DSML｜tool_calls>` blocks | `<think>` | grammar, or the prompt opens the call |
| Kimi-K3 | its tags (`xtml::Xtml`) | the think channel | the prompt opens the call |
| Gemma 4 | `<|tool_call>call:NAME{key:<|"|>text<|"|>,...}<tool_call|>` (`reader::Gemma`) | `<|channel>thought\n...<channel|>` | the prompt opens the call |
| Muse | ATEM `<atem:function_calls>` blocks in a message to the tool (`reader::Muse`), typed by the schema | a message to `self` | the prompt opens the message and the block |

Qwen3.8-27B and Qwen3.8-Flash-Next ship the same chat template. A family's test (`glyd-engine/src/families.rs`) renders a conversation with a call and a reasoning through its real chat template and reads both back, whole and in pieces of every size, and again token by token through the server's reader, reasoning split and tool parser with the deltas a client sees; the GPU job serves Gemma 4 at kestrel and runs the client checks against it. Where a call is forced by the prompt, the
model's own arguments are not held to the tool's schema by a grammar.

## Fill in the middle

`POST /v1/completions` with `"prompt"` (the text before the cursor, one string) and `"suffix"` (the text after) is written with the model's own tokens, `<|fim_prefix|>BEFORE<|fim_suffix|>AFTER<|fim_middle|>`
(or StarCoder's `<fim_...>`, DeepSeek-Coder's `<｜fim▁begin｜>`), and the reply is the text between. A model whose tokenizer has none of them refuses `suffix` with 400.

## /v1/glyd/info

```
curl -s -H "Authorization: Bearer $GLYD_API_KEY" http://127.0.0.1:8238/v1/glyd/info
```

From the record (served by its Hub name; the file list cut to two of sixteen):

```json
{"object":"glyd.info","api":"glyd-serve/1",
 "build":{"version":"0.32.1","commit":"86530a29ad1096f8f116aab2578b1a911c64217d","development":true},
 "model":{"id":"Qwen/Qwen3.5-9B:linnet","repo":"glyd/Qwen3.5-9B-linnet","revision":"073a2132abeaa41c119678b9687092871521eb4d",
          "source":{"repo":"Qwen/Qwen3.5-9B","revision":"c202236235762e1c871ad0ccb60c8ee5ba337b9a"},"made_by":"glyd 0.29.3",
          "files":[{"name":"glyd.json","bytes":92205,"sha256":"85b27d2d4a7d31f9c3493a3dbf5f00f00469e0c67b917e8f1374900d674cc671"},
                   {"name":"model-00001-of-00007.safetensors","bytes":1004449488,"sha256":"ea1b678c4787351b59de96915373a4a2aecc9eea49df70a64baf707b3e6f3830"}]},
 "level":{"requested":"linnet","effective":"linnet","lossless":false},
 "precision":{"weights":"linnet level: about 5.5 bits per weight, not lossless","activations":"bf16","sums":"fp32","kv":"bf16",
              "state":"fp32 (the delta rule's), bf16 (the convolution's)"},
 "device":{"ordinal":0,"name":"NVIDIA GeForce RTX 4080 SUPER","compute_capability":"8.9","cuda":"13.2","driver":"595.91.07"},
 "limits":{"context_tokens":4096,"concurrency":1,"max_choices":64,"max_body_bytes":16777216,"max_header_bytes":65536,
           "max_headers":256,"max_connections":null,"read_timeout_seconds":30},
 "fields":{"read":[...38],"no_effect":[...12],"refused":[...38]},
 "state":"ready","error":null,"requests":{"running":0,"cancelling":0},
 "memory":{"gpu":0,"measured_at":1792451421.629,"engine_bytes":6987473708,"weights_bytes":6200345608,"kv_bytes":134217728,
           "state_bytes":51511296,"scratch_bytes":601399076,"peak_bytes":6987473708,"peak_at":1792451421.367,
           "peak_since":1792451419.173,"device_total_bytes":16717119488,"device_free_bytes":7039418368,
           "not_counted":["the CUDA context and the kernels' code","cuBLAS's own workspace","the computer's memory"]}}
```

- `build.commit` is what the build was given in GLYD_SOURCE_COMMIT (release-gpu.yml gives the tag's); null where it was given none.
- `model.repo` and `revision` are read from the Hub cache's layout (`models--ORG--NAME/snapshots/REV`); null for another directory.
  `source` and `made_by` are glyd.json's: the model the level was made from, at its revision, and the Glyd that made it.
  glyd.json holds each tensor's SHA-256 too. A file's `sha256` is the cache's blob name where the file is a link to one
  (the download checked it), else the file read at the start where it is 64 MiB or smaller, else null.
- `level.requested` is the tag, `--lossy` (kestrel), or a glyd/NAME-LEVEL name; null where nothing asked for one (a
  directory runs at its own level). `effective` is the level loaded.
- `precision`: `sums` is the products' accumulation. `kv` and `state` are null for a model across several GPUs, which does not say. `kv` is the format the keys and values were loaded in:
  `bf16`, or `12-bit` where the engine holds them in the lossless 12-bit pages (`GLYD_KV=lossless`, where the plan took them: the plan's `kv_format`). The `costs` profile below counts the keys and values in bf16, its formula, whichever pages are loaded.
- `fields`: `read` are the request fields this server reads (some values of some are refused, below), `no_effect` change
  nothing here, `refused` are refused when set to anything but null or the value that means "not asked" (false, 0, []). The record's lists had 38, 12 and 38
  names; this tree's have 49, 12 and 28 (`DONE`, `NO_EFFECT`, and `NOT_DONE` with `CONSTRAINED`, in `glyd-engine/src/server.rs`).
- Times are Unix seconds, to the millisecond.
- `memory.devices` (not in the record above, added after it): the engine's buffers by GPU, one entry a CUDA context the engine opened or held a buffer
  in (`gpu`, `engine_bytes`, `peak_bytes`, `peak_at`); `engine_bytes` and `peak_bytes` above are every GPU's together.
- `costs` (not in the record above, added after it): the model's cost profile at the level, as `glyd-engine costs DIR` prints it, with `allocated` (the
  engine's counters at load) beside the sizes from the files: what each layer holds by component, the bytes a decoded token reads, the KV and state
  bytes a token and a session. The fields are in [docs/cost-profile.md](cost-profile.md). Where the profile cannot be made (a family it does not
  describe), `costs` is `{"status": "unavailable", "error": "..."}`. A model on the dynamic route (Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark, Qwen3.8-Flash-Next, and the Qwen3.5
  layout and Nemotron-H at penguin) has no `costs` key: it is there for the models the one-GPU path of the levels loads (Gemma 4, the plain dense kind, the lossy levels of the Qwen3.5 layout and Nemotron-H).

### Memory

- `engine_bytes`: every buffer the engine holds on the GPU now, counted where the engine allocates and frees device memory
  (the sizes it asks for, not the driver's rounding).
- `weights_bytes`, `kv_bytes`, `state_bytes`: what the model holds, made once at load and kept: the weights (the same
  number as the start's "GiB of weights" line), the keys and values for the whole context (bf16, or, where `precision.kv` is `12-bit`, the pages with their escape and raw pools and the buffers a pass unpacks them into; `--ctx` positions, used
  or not), the states a sequence carries (zeroed at each request). `scratch_bytes` is the rest of `engine_bytes`: the
  activations, the products' workspaces, the logits, the rotary tables. Null where the engine does not say (a model across several GPUs).
- `peak_bytes`, `peak_at`: the most `engine_bytes` has been since the process started, and when. Never reset.
  `peak_since` is when the load began.
- `device_total_bytes`, `device_free_bytes`: the driver's numbers now, for the whole GPU: every process's memory is in them.
- `not_counted`: in nvidia-smi's number for this process but not in `engine_bytes`. The record: 7,002 MiB in nvidia-smi
  against 6,663.8 MiB of buffers, 338 MiB of context, code and rounding.
- Nothing is allocated for a request: it uses the KV and state made at load (what a finished request leaves there is kept for the
  conversation's next turn while there is room). The counters are the process's.

## Requests

- A field a request sets that `fields.refused` lists is refused, 400, in words. So are: `response_format` of a type other than
  text, json_object and json_schema, `echo` of a chat, `best_of` over `n`, `n` over 64, prompts times `n` over 64, `n` over 1 at temperature 0,
  `stream_options` without `stream`, `continuous_usage_stats`, tools where the template has none.
- A reply can be held to a grammar, on every model the server runs: `response_format` (`json_object`, `json_schema`; the schema holds
  whether or not `strict` is set), vLLM's `structured_outputs` (`json`, `regex`, `choice`, `grammar`, `json_object`) and its
  `guided_json`, `guided_regex`, `guided_choice` and `guided_grammar` (a Lark grammar, or llguidance's grammar list as JSON), and a
  forced tool call (`tool_choice` naming a function, or `required`) in the call formats of Qwen, Nemotron, GLM and DeepSeek-V4 (hermes
  JSON, XML, GLM's `<arg_key>` and DSML): the function is one of the request's and the arguments follow its schema. Kimi's, Gemma's
  and Muse's calls are in their own markup and are opened by the prompt only. The tokens the grammar does not allow are taken out of
  each step's row of logits (llguidance), so the reply is valid or ends with `finish_reason` `length`. A schema or grammar that cannot
  be compiled, and two forms at once, are 400 in words. A request with none is sampled as before. `guided_whitespace_pattern` is refused.
- Every model is served by the scheduler and takes `logprobs` and `min_tokens`: a model on one GPU to `--users` at once
  (`limits.concurrency`); a model with a draft model, a model split across GPUs, or one with an operation that takes one sequence, one request at a time (`concurrency: 1`). `GLYD_ENGINE_BATCH_PREFILL=0` makes every reply the one the same
  request alone gets: prompts run one a pass, and a model with experts (Muse Spark, Kimi-K3, GLM-5.3-Flash) decodes one sequence a step (its
  experts' products differ in the last bits at several rows). There is one exception: a model whose expert cache has the host compute
  some misses (Kimi-K3 at penguin, a Condor codes pack). There, which pairs the host computes follows the copy link's timing. Such a
  reply holds within the kind's bound from one run to the next, not to the bit (docs/design/runtime.md, the expert cache's
  "Run to run").
- A field this server does not know is answered without it, and said once on stderr.
- `model` is not checked: one model is served, and the reply names it.
- The context holds the prompt and the reply. A prompt of the context's size or more is refused; with `max_tokens` (or
  `max_completion_tokens`), a prompt plus that is refused where it is more than the context, not cut short. Without it, a
  chat may run to the context's end, a plain completion stops at 16 tokens (the legacy default).
- Sampling: temperature 0 is greedy; seed, top_p, top_k, min_p, the penalties, logit_bias, allowed_token_ids, stop,
  stop_token_ids, ignore_eos as vLLM takes them.
- Thinking: Qwen3.5's template thinks unless `"chat_template_kwargs":{"enable_thinking":false}` or
  `"reasoning_effort":"none"`. `"include_reasoning":false` leaves the thinking out of the reply (it is still made and counted).

### Counting a prompt

```
curl -s -H "Authorization: Bearer $GLYD_API_KEY" -H 'Content-Type: application/json' http://127.0.0.1:8238/v1/glyd/tokenize \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],"chat_template_kwargs":{"enable_thinking":false}}'
{"prompt_tokens":17,"max_model_len":4096}
```

The body is the request's (`messages`: a chat's; else `prompt`: a plain completion's), read by the same code as the
request: the tokenizer, the chat template, the special tokens and the thinking setting are the reply's. It refuses what the
request would be refused for, runs nothing, and its count is the reply's `usage.prompt_tokens` (17 and 17 in the record).
It does not wait for the replies that run: the count is made on the request's own thread, not in the scheduler's.

### Usage

`prompt_tokens`: the prompt's tokens as the template made it. `completion_tokens`: every token made, the thinking, the
answer and the end token where the reply ends by it. The thinking's share is not given apart.

## Replies

A JSON chat (from the record; DIR is the directory served):

```
curl -s -H "Authorization: Bearer $GLYD_API_KEY" -H 'Content-Type: application/json' http://127.0.0.1:8238/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
{"id":"chatcmpl-8","object":"chat.completion","created":1792451404,"model":"DIR","choices":[{"index":0,"message":{"role":"assistant",
 "content":"The three primary colors in traditional color theory ... (additive color mixing, such"},"logprobs":null,"finish_reason":"length"}],
 "usage":{"prompt_tokens":17,"completion_tokens":64,"total_tokens":81}}
```

With the thinking on, the message has `reasoning_content` and `reasoning` (the same text) beside `content`.

A streamed chat:

```
curl -sN -H "Authorization: Bearer $GLYD_API_KEY" -H 'Content-Type: application/json' http://127.0.0.1:8238/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":1024,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
data: {"id":"chatcmpl-11","object":"chat.completion.chunk","created":1792451405,"model":"DIR","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-11",...,"choices":[{"index":0,"delta":{"reasoning_content":"Thinking","reasoning":"Thinking"},"logprobs":null,"finish_reason":null}]}
...                                         (the thinking, then deltas with "content")
data: {"id":"chatcmpl-11",...,"choices":[{"index":0,"delta":{"content":"\n"},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-11",...,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
data: {"id":"chatcmpl-11",...,"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":662,"total_tokens":682}}
data: [DONE]
```

- The first event of each choice carries the role and `"content":""`. A delta carries `reasoning_content` and `reasoning`
  (the same text), or `content`, or `tool_calls`.
- `finish_reason`: `stop` (the end token, a stop string or stop token, or a cancel: below), `length` (`max_tokens`, or the
  context full), `tool_calls`.
- The usage event (`"choices":[]`) comes only with `stream_options.include_usage`. Then `data: [DONE]`, and the chunked
  body ends.
- While a request waits for its turn or its prompt is read for more than a second, the stream's headers go out and an SSE
  comment (`: waiting`) comes each second.
- An error after the stream began: `data: {"error":{"message":"...","type":"server_error"}}`, then `data: [DONE]`.

Errors before a reply:

```
400 {"error":{"message":"use_beam_search is not supported by this engine yet (GLYD_ENGINE=0 runs vLLM, which has it)","type":"invalid_request_error","param":null,"code":null}}
400 {"error":{"message":"This prompt has 15 tokens and asks for up to 5000 more, 5015 in all: longer than the model's context of 4096 tokens: maximum context length (a longer one: --context N)","type":"invalid_request_error","param":null,"code":"context_length_exceeded"}}
401 {"error":{"message":"Incorrect API key provided: the server's key is needed as `Authorization: Bearer <key>`.","type":"authentication_error","param":null,"code":"invalid_api_key"}}
404 {"error":{"message":"no request has this id (the last 64 that ended are kept)","type":"invalid_request_error","param":null,"code":null}}
421, 403: {"error":{"message":"...","type":"Forbidden","code":421}}     431: a header block past 64 KiB or 256 headers     503: past 50,000 open connections or 8,192 requests in flight
```

## A request by its id: status and cancel

A reply's `id` is `chatcmpl-` (a chat) or `cmpl-` (a plain completion) and the request's `request_id`, or a number the
process counts where it gave none. A request whose id is running refuses a second one with it (400).

```
curl -s -H "Authorization: Bearer $GLYD_API_KEY" http://127.0.0.1:8238/v1/glyd/requests/chatcmpl-rec-1
{"id":"chatcmpl-rec-1","state":"running","cancel":null,"began":1792451411.803,"ended":null,"how":null}
curl -s -X POST -H "Authorization: Bearer $GLYD_API_KEY" http://127.0.0.1:8238/v1/glyd/requests/chatcmpl-rec-1/cancel
{"id":"chatcmpl-rec-1","state":"cancelling","cancel":"asked","began":1792451411.803,"ended":null,"how":null}
... 0.016 s later:
{"id":"chatcmpl-rec-1","state":"ended","cancel":"asked","began":1792451411.803,"ended":1792451414.882,"how":"cancelled"}
```

| state | |
| :--- | :--- |
| `running` | the request runs, or waits for its turn |
| `cancelling` | asked to end (`cancel`: `asked` by the endpoint, or `client gone`), and the engine is not done with it yet |
| `ended` | the engine is done with every sequence of it: what they used is the next request's. `how`: `finished`, `cancelled`, or `failed: ...` |

- A cancel stops each sequence at its next token; one that waits for its turn is not run. The scheduler looks at what is asked to end at the start of each round, a prompt
  chunk or a decode step, so a cancel during a prompt is seen after the chunk in progress. The record's 0.633 s for a cancel 0.2 s into a 3,790-token prompt (0.016 s during a reply) is
  of the engine of that day, which read the whole prompt first; it has not been measured on this one.
- A client that closes its connection is cancelled the same way, seen at the next write or within a second: `ended` 0.031 s
  after curl was killed in the record.
- A cancelled request whose client still reads gets the end of a client's stop: finish_reason `stop`, the usage, `[DONE]`.
  The status says `cancelled`.
- `ended` is said once every choice's thread has seen its sequence's end from the engine, before the reply's last bytes go
  out. Nothing is freed then: the KV and the state are made at load and the next request uses them (what a finished request leaves is kept for its conversation's next turn).
- The last 64 ended requests are kept; an id given again names the new request. Ids are matched as the path has them (no
  percent-decoding): give `request_id`s of letters, digits, `-` and `_`.

## The cache's events

A router that sends a request to the server already holding its start reads what the cache holds by name: each step's blocks
stored (their names, the name of the block before them, their tokens) and removed. Off unless the server's operator sets
`GLYD_ENGINE_KV_EVENTS=1`: an event carries a prompt's tokens, and whoever reads the events reads every caller's prompts (with
no key on the server, whoever reaches its port). The blocks are the description executor's (the families it serves); 16 tokens
a block.

```
curl -s -H "Authorization: Bearer $GLYD_API_KEY" 'http://127.0.0.1:8238/v1/glyd/kv/events?after=41&wait=5'
{"next":43,"oldest":1,"gap":false,"batches":[{"seq":42,"ts":1795366128.125,"events":[
  {"type":"removed","blocks":["9f2c..."],"medium":"gpu"},
  {"type":"stored","blocks":["1a7e...","c03b..."],"parent":null,"tokens":[...32 ids...],"block_size":16,"medium":"gpu"}]}]}
```

- `after`: the newest batch already read (0: all kept); `wait`: seconds to wait for a batch where none is new yet (at most 10).
  Pass `next` as the next call's `after`.
- A block's name is 32 hexadecimal digits: the first 16 bytes of SHA-256 over the name of the block before it and its 16 token
  ids (the earlier block's 16 digest bytes as they came, then each id as 4 bytes, little-endian), the 16 bytes read as a little-endian 128-bit number and written
  as 32 digits, so the digits are the digest's bytes in reverse (`kv::block_hash`). A request's chain begins at its scope's root (`cache_salt`):
  the same hash over the text `glyd-kv-1` and the scope's key as 8 bytes, little-endian (`kv::block_root`; the key is 0 for no `cache_salt`, else the first 8 bytes of the salt's SHA-256 read little-endian, with the lowest bit set: `salt_of` in `src/server.rs`). `parent` null: the first stored block is its sequence's first.
- Batch 1 is `{"type":"cleared"}`: the cache holds nothing when the server starts. The last 1,024 batches are kept; `gap: true`
  says some after `after` are not, and a reader takes what it knew as cleared.
- The schema is part of `glyd-serve/1`: a change that breaks it is a new API version.

## Limits

A body of 16 MiB; a header block of 64 KiB and 256 headers (431 past them, read no further); 50,000 open connections and 8,192 requests in flight (503
past them); the request's head is read within 30 s and its body within 120 s. No time limit on a reply: cancel it, or close the connection.

## Differences from Glyd-Compute's service adapter

What an adapter written for a vLLM-like server must do on purpose here (the adapter's own contract is
[starlings/docs/service-adapter.md](../starlings/docs/service-adapter.md)):

1. Loading: the port is bound only once the model is loaded. Before that a connection is refused; there is no 503 while
   loading. The status file (`--status-file`) says `ready`, or the reason of a refusal.
2. `/health` is open: it answers 200 with or without the key. The refusal without a key holds for every path under `/v1`.
3. There is no `/props`. `/v1/glyd/info` has the build, the context per request (`limits.context_tokens`), the requests it runs
   at once (`limits.concurrency`) and the model; it has no chat-template capability flags.
4. `/v1/models`'s id, and every reply's `model`, is MODEL as typed: the directory's path where a directory is served.
   There is no served-model-name flag, and a request's `model` is not checked.
5. The final usage needs `stream_options: {"include_usage": true}`.
6. A stream that waits sends `: waiting` comment lines.
7. Deltas carry the thinking twice, as `reasoning_content` and `reasoning`.
8. A cancelled stream ends with finish_reason `stop`, the usage and `[DONE]`; `/v1/glyd/requests/ID` says it was cancelled.
9. Cancel by id (`POST /v1/glyd/requests/ID/cancel`), then `ended` before the memory is reused, in place of stopping the
   process; stopping the process still frees everything.
10. The key's variable is GLYD_API_KEY (or VLLM_API_KEY). With the environment not inherited, give HF_HUB_OFFLINE=1 where
    the model is a directory; a CUDA build needs no GPU library, so no path to one.
11. Every engine refuses a `max_tokens` the context has no room
    for, rather than cutting it.

## Known limits

- A model across several GPUs gives no memory parts or KV and state
  precision (null); its `engine_bytes` and peak are counted.
- `/v1/messages` requests are not tracked by id; they stop when their client goes.
- A Hub name needs the network only where the base model's config.json is not in the Hugging Face cache (and a level's pre-packed repo is not either); offline, serve the directory.
- A model split across GPUs is served one request at a time.
- `completion_tokens` is not split into the thinking and the answer.
