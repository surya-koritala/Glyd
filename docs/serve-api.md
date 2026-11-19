# glyd serve: the HTTP API, glyd-serve/1

Notes for a program that starts `glyd serve` (or `glyd-engine serve`) on its own machine and sends it whole requests, as
Glyd-Compute's service adapter does. `glyd-serve/1` is what `GET /v1/glyd/info` says in `api`; a change that breaks a
client of this page gets a new number. The numbers below are from one run on an RTX 4080 SUPER with Qwen3.5-9B at the
swift level: [benchmarks/gpu/rtx4080s-serve-probe-2026-10-19](../benchmarks/gpu/rtx4080s-serve-probe-2026-10-19/README.md).
The contract tests are in glyd-engine/src/server.rs and cli.rs (the `#[test]`s named below) and glyd-gpu-py/tests/test_onboard.py.

## Start, ready, stop

```
glyd serve Qwen/Qwen3.5-9B:swift                         # the Python front end: hands a level to the engine
glyd-engine serve Qwen/Qwen3.5-9B:swift --ctx 4096       # the engine itself
HF_HUB_OFFLINE=1 GLYD_API_KEY=... glyd-engine serve /path/to/models--glyd--Qwen3.5-9B-swift/snapshots/073a2132... \
    --ctx 4096 --port 8238 --status-file /run/glyd.status  # offline: the directory alone is read
```

- `MODEL:level` takes penguin (lossless), kestrel, swift, hummingbird (Nemotron 3 Super), and wren (a development build;
  Gemma 4's QAT checkpoints in a release). A tag that names no level is refused (exit 1). A level the engine does not run
  for the model, or a model that does not fit, is refused with the reason (exit 3, on stderr and in the status file).
  Nothing is changed for you: the model, the level and the precision are the ones asked for, or the start fails.
- `glyd serve` of a level, a glyd-v7 directory or a glyd/ repo vLLM cannot read goes to the engine without GLYD_ENGINE=1.
  `glyd serve` of a lossless model is vLLM's unless GLYD_ENGINE=1, as before.
- A Hub name with a level reads the base model's config.json (the cache, else the Hub) and loads glyd/NAME-LEVEL from the
  cache (else downloads it). Offline, give the snapshot directory: nothing else is read.
- The address is 127.0.0.1:8000, or the next free port up to 8020; `--port N` is that port or a refusal. `--host H`
  listens on H and turns the Host and Origin checks off.
- The GPU is the one with the most free memory of those CUDA_VISIBLE_DEVICES shows. The context is `--ctx N` (`--context`);
  without it, what fits, up to 32,768 at a level. A level's engine runs one request at a time (the others wait in order);
  the lossless engine runs `--users N` at once (1 to 8, 4 if not given).
- The environment it reads: GLYD_API_KEY (or VLLM_API_KEY; `--api-key` works but `ps` shows it), GLYD_GPU_LIB where the
  GPU library is not beside the program, HF_HUB_CACHE or HF_HOME for a Hub name, HF_HUB_OFFLINE=1. A directory needs
  nothing else.

States:

| state | how it shows |
| :--- | :--- |
| loading | the process runs; the port is not bound (connections are refused); the status file is touched as it goes |
| ready | the port is bound, `/health` is 200, `/v1/glyd/info` says `"state":"ready"`, the status file says `ready` (2.6 s after the start in the record) |
| failed before ready | exit 3 (1 for a wrong flag or level name), the reason on stderr and in the status file |
| failed after ready | `/v1/glyd/info` says `"state":"failed"` with `error: {at, message}`, the engine's first failure. It stays failed: start the process again |

Stop: SIGTERM or SIGINT end the process at once (exit 143, 130); a request in flight is cut. The GPU's memory is the
driver's again when the process is gone (the record: 2,223 MiB used before the start and after the stop).

## The key and the address checks

- With a key, every path under `/v1` needs `Authorization: Bearer KEY` (the scheme in any case), else
  `401 {"error":"Unauthorized"}`. `/`, `/health` and `/stats` are open. The key is compared as SHA-256 digests in constant time.
- Without `--host`: a Host that is not localhost, 127.0.0.1 or [::1] is 421, an Origin that is not the server's own is 403
  (a CORS preflight too). A program that sends no Origin (curl, the OpenAI libraries) is answered.

## Endpoints

| | key | |
| :--- | :--- | :--- |
| `GET /health` | no | `{"status":"ok"}` |
| `GET /v1/models` | yes | the one model: `id` is MODEL as typed (with its tag, or the directory's path), `max_model_len` the context |
| `POST /v1/chat/completions` | yes | a chat, JSON or SSE |
| `POST /v1/completions` | yes | a plain completion, JSON or SSE |
| `POST /v1/messages`, `/v1/messages/count_tokens` | yes | Anthropic's Messages API (not tracked by id) |
| `POST /v1/glyd/tokenize` | yes | a request's prompt counted, nothing run |
| `GET /v1/glyd/info` | yes | what the server is and holds |
| `GET /v1/glyd/requests/ID` | yes | how a request stands |
| `POST /v1/glyd/requests/ID/cancel` | yes | a request asked to end |
| `GET /stats` | no | the lossless engine's scheduler counts (zero at a level) |

A method a path does not take is 405, a path there is not 404.

## /v1/glyd/info

```
curl -s -H "Authorization: Bearer $GLYD_API_KEY" http://127.0.0.1:8238/v1/glyd/info
```

From the record (served by its Hub name; the file list cut to two of sixteen):

```json
{"object":"glyd.info","api":"glyd-serve/1",
 "build":{"version":"0.32.1","commit":"86530a29ad1096f8f116aab2578b1a911c64217d","development":true},
 "model":{"id":"Qwen/Qwen3.5-9B:swift","repo":"glyd/Qwen3.5-9B-swift","revision":"073a2132abeaa41c119678b9687092871521eb4d",
          "source":{"repo":"Qwen/Qwen3.5-9B","revision":"c202236235762e1c871ad0ccb60c8ee5ba337b9a"},"made_by":"glyd 0.29.3",
          "files":[{"name":"glyd.json","bytes":92205,"sha256":"85b27d2d4a7d31f9c3493a3dbf5f00f00469e0c67b917e8f1374900d674cc671"},
                   {"name":"model-00001-of-00007.safetensors","bytes":1004449488,"sha256":"ea1b678c4787351b59de96915373a4a2aecc9eea49df70a64baf707b3e6f3830"}]},
 "level":{"requested":"swift","effective":"swift","lossless":false},
 "precision":{"weights":"swift level: about 5.5 bits per weight, not lossless","activations":"bf16","sums":"fp32","kv":"bf16",
              "state":"fp32 (the delta rule's), bf16 (the convolution's)"},
 "device":{"ordinal":0,"name":"NVIDIA GeForce RTX 4080 SUPER","compute_capability":"8.9","cuda":"13.2","driver":"595.91.07"},
 "limits":{"context_tokens":4096,"concurrency":1,"max_choices":64,"max_body_bytes":16777216,"max_header_bytes":65536,
           "max_headers":256,"max_connections":64,"read_timeout_seconds":30},
 "fields":{"read":[...38],"no_effect":[...12],"refused":[...38]},
 "state":"ready","error":null,"requests":{"running":0,"cancelling":0},
 "memory":{"gpu":0,"measured_at":1792451421.629,"engine_bytes":6987473708,"weights_bytes":6200345608,"kv_bytes":134217728,
           "state_bytes":51511296,"scratch_bytes":601399076,"peak_bytes":6987473708,"peak_at":1792451421.367,
           "peak_since":1792451419.173,"device_total_bytes":16717119488,"device_free_bytes":7039418368,
           "not_counted":["the CUDA context and the kernels' code","cuBLAS's own workspace","the computer's memory"]}}
```

- `build.commit` is what the build was given in GLYD_SOURCE_COMMIT (release.yml gives the tag's); null where it was given none.
- `model.repo` and `revision` are read from the Hub cache's layout (`models--ORG--NAME/snapshots/REV`); null for another directory.
  `source` and `made_by` are glyd.json's: the model the level was made from, at its revision, and the Glyd that made it.
  glyd.json holds each tensor's SHA-256 too. A file's `sha256` is the cache's blob name where the file is a link to one
  (the download checked it), else the file read at the start where it is 64 MiB or smaller, else null.
- `level.requested` is the tag, `--lossy` (kestrel), or a glyd/NAME-LEVEL name; null where nothing asked for one (a
  directory runs at its own level). `effective` is the level loaded.
- `precision`: `sums` is the products' accumulation. `kv` and `state` are null for the lossless engine, which does not say.
- `fields`: `read` are the request fields this server reads (some values of some are refused, below), `no_effect` change
  nothing here, `refused` are refused when set to anything but null or the value that means "not asked" (false, 0, []).
- Times are Unix seconds, to the millisecond.
- `costs` (not in the record above, added after it): the model's cost profile at the level, as `glyd-engine costs DIR` prints it, with `allocated` (the
  engine's counters at load) beside the sizes from the files: what each layer holds by component, the bytes a decoded token reads, the KV and state
  bytes a token and a session. The fields are in [docs/cost-profile.md](cost-profile.md). Where the profile cannot be made (a family it does not
  describe), `costs` is `{"status": "unavailable", "error": "..."}`.

### Memory

- `engine_bytes`: every buffer the engine holds on the GPU now, counted where the engine allocates and frees device memory
  (the sizes it asks for, not the driver's rounding).
- `weights_bytes`, `kv_bytes`, `state_bytes`: what the model holds, made once at load and kept: the weights (the same
  number as the start's "GiB of weights" line), the keys and values for the whole context (bf16, `--ctx` positions, used
  or not), the states a sequence carries (zeroed at each request). `scratch_bytes` is the rest of `engine_bytes`: the
  activations, the products' workspaces, the logits, the rotary tables. Null where the engine does not say (lossless).
- `peak_bytes`, `peak_at`: the most `engine_bytes` has been since the process started, and when. Never reset.
  `peak_since` is when the load began.
- `device_total_bytes`, `device_free_bytes`: the driver's numbers now, for the whole GPU: every process's memory is in them.
- `not_counted`: in nvidia-smi's number for this process but not in `engine_bytes`. The record: 7,002 MiB in nvidia-smi
  against 6,663.8 MiB of buffers, 338 MiB of context, code and rounding.
- Nothing is cached between requests and nothing is allocated for one on a level's engine: a request uses the KV and
  state made at load. One GPU a process: the counters are the process's.

## Requests

- A field a request sets that `fields.refused` lists is refused, 400, in words. So are: `response_format` other than text,
  `echo` of a chat, `best_of` over `n`, `n` over 8, prompts times `n` over 64, `n` over 1 at temperature 0,
  `stream_options` without `stream`, `continuous_usage_stats`, tools where the template has none.
- On a level's engine (`concurrency: 1`), `logprobs` and `min_tokens` are refused: it gives no log-probabilities and has no
  least length. `fields.refused` says so.
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
Waiting for a reply that runs: it is answered between replies on a level's engine.

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
400 {"error":{"message":"logprobs is not supported by the engine this model runs on yet","type":"invalid_request_error","code":400}}
400 {"error":{"message":"This prompt has 15 tokens and asks for up to 5000 more, 5015 in all: longer than the model's context of 4096 tokens: maximum context length (a longer one: --context N)","type":"invalid_request_error","code":400}}
401 {"error":"Unauthorized"}
404 {"error":{"message":"no request has this id (the last 64 that ended are kept)","type":"invalid_request_error","code":404}}
421, 403: {"error":{"message":"...","type":"Forbidden","code":421}}     431: a header block past 64 KiB or 256 headers     503: past 64 connections
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

- A cancel stops each sequence at its next token; one that waits for its turn is not run. A level's engine reads the whole
  prompt before it looks at a cancel: 0.633 s for a cancel 0.2 s into a 3,790-token prompt in the record, 0.016 s during a reply.
- A client that closes its connection is cancelled the same way, seen at the next write or within a second: `ended` 0.031 s
  after curl was killed in the record.
- A cancelled request whose client still reads gets the end of a client's stop: finish_reason `stop`, the usage, `[DONE]`.
  The status says `cancelled`.
- `ended` is said once every choice's thread has seen its sequence's end from the engine, before the reply's last bytes go
  out. On a level's engine nothing is freed then: the KV and the state are made at load and the next request uses them.
- The last 64 ended requests are kept; an id given again names the new request. Ids are matched as the path has them (no
  percent-decoding): give `request_id`s of letters, digits, `-` and `_`.

## Limits

A body of 16 MiB; a header block of 64 KiB and 256 headers (431 past them, read no further); 64 connections at once (503
past them); a read of the request's head or body waits 30 s at most. No time limit on a reply: cancel it, or close the connection.

## Differences from Glyd-Compute's service adapter

Against docs/service-adapter.md at Sigbound/glyd-compute 24baad6dba877e37e4d80a53ea8b341bc8115079, what an adapter must do
on purpose:

1. Loading: the port is bound only once the model is loaded. Before that a connection is refused; there is no 503 while
   loading. The status file (`--status-file`) says `ready`, or the reason of a refusal.
2. `/health` is open: it answers 200 with or without the key. The refusal without a key holds for every path under `/v1`.
3. There is no `/props`. `/v1/glyd/info` has the build, the context per request (`limits.context_tokens`), one request
   at a time (`limits.concurrency: 1`) and the model; it has no chat-template capability flags.
4. `/v1/models`'s id, and every reply's `model`, is MODEL as typed: the directory's path where a directory is served.
   There is no served-model-name flag, and a request's `model` is not checked.
5. The final usage needs `stream_options: {"include_usage": true}`.
6. A stream that waits sends `: waiting` comment lines.
7. Deltas carry the thinking twice, as `reasoning_content` and `reasoning`.
8. A cancelled stream ends with finish_reason `stop`, the usage and `[DONE]`; `/v1/glyd/requests/ID` says it was cancelled.
9. Cancel by id (`POST /v1/glyd/requests/ID/cancel`), then `ended` before the memory is reused, in place of stopping the
   process; stopping the process still frees everything.
10. The key's variable is GLYD_API_KEY (or VLLM_API_KEY). With the environment not inherited, give GLYD_GPU_LIB where the
    library is not beside the program, and HF_HUB_OFFLINE=1.
11. A level's engine refuses `logprobs` and `min_tokens`, and every engine refuses a `max_tokens` the context has no room
    for, rather than cutting it.

## Known limits

- The lossless engine (Qwen3 dense; `glyd serve` of Qwen3.5 penguin is vLLM's) gives no memory parts or KV and state
  precision (null); its `engine_bytes` and peak are counted.
- `/v1/messages` requests are not tracked by id; they stop when their client goes.
- A Hub name needs the network for the base model's config.json; offline, serve the directory.
- One process serves one GPU; a model split across GPUs is not served.
- `completion_tokens` is not split into the thinking and the answer.
