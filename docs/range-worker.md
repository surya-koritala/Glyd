# A range of layers as a worker: glyd-range/1

What it is for: one model run across several processes or machines, a contiguous range of its layers in each, when one
GPU cannot hold it or the work is spread over several computers. Starlings (our distributed inference) starts the workers
and carries the tensors between them; any program that speaks the frames below can. Each worker holds only its layers'
weights, keys and values; together they give the whole model's tokens and logits, bit for bit.

`glyd-engine rangeworker` (src/dynamic/range.rs) is the process, in the release build; the engine's stage is
`Dyn::new_stage` (glyd-engine/src/dynamic/mod.rs); the proof is `glyd-engine rangecheck`, a development build's. The numbers are from one run on an
RTX 4080 SUPER with Qwen3.5-9B: [benchmarks/gpu/rtx4080s-range-2026-10-20](../benchmarks/gpu/rtx4080s-range-2026-10-20/README.md).
`glyd-range/1` is what `describe` says in `protocol`; a change that breaks a client of this page gets a new number.

## What a stage is

A stage runs layers `a..b` of the model on one GPU and holds only their weights. The first stage (`a == 0`) holds the
embedding and embeds the tokens; the last (`b == layers`) holds the final norm and the LM head, and picks the greedy
token. A middle stage holds neither. Each stage owns the keys and values of its full-attention layers (bf16, for the
whole context, made at load) and the states of its linear-attention layers (zeroed at a new sequence). Nothing is shared
between stages but the boundary tensors of each chunk.

The whole model on one GPU is the stage `0..layers`: the same code, the same kernels, the same launch order. That is why
the stages reproduce it bit for bit: a cut adds two copies (GPU to host, host to GPU) of tensors the whole model keeps on
the GPU between two layers, and nothing else.

## The boundary (Qwen3.5, 3.6, 3.8 dense)

What the whole-model path has live between layer `l` and layer `l + 1` of Qwen3.5 is a pair, not one tensor:

| order | name | slot | what | shape | dtype |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | residual | X | the residual stream before layer `l`'s MLP output is added | [rows, hidden] | bf16 |
| 2 | mlp_out | D | layer `l`'s MLP down product | [rows, hidden] | bf16 |

Layer `l + 1` begins with `NormAdd`: `x = bf16(x + d)`, `h = bf16(rmsnorm(x) (1 + w))`; the final norm after the last
layer does the same add before its norm. Nothing else a layer wrote is read later (H, O and the attention's slots are
written before they are read in the next layer). So the stage hands over X and D as they are, in bf16, and the next
stage's first operation is the whole model's: the sum is rounded once, in the same kernel, at the same point. One fused
tensor would also be exact (the add is the next layer's first step, rounded to bf16 before anything reads it), but it would
be an extra launch on the sender for nothing; the engine carries the pair.

Byte layout of a chunk's boundary (`rows` tokens): the residual, row-major, `rows x hidden` u16 little-endian (the bf16
bits), then mlp_out the same; `2 x rows x hidden x 2` bytes in all. Qwen3.5-9B (hidden 4,096): 16,384 bytes a token; a
1,024-token chunk is 16 MiB. The last stage takes the pair in and gives the token and its logit back, and the logits of
the last row as a body where asked (`vocab` u16 bf16 words: 496,640 bytes for Qwen3.5-9B). With `head_rows` k it runs the
head on the last k rows of the chunk (a draft's check): a greedy token a row, and the k rows' logits as the body
(`k x vocab` words, row after row) where asked. A stage that is not the last gives no logits.

Positions: explicit. A chunk carries `position_start`; row `i` is position `position_start + i`. Each stage holds the
rotary tables and applies the rotary embedding to its own full-attention layers' queries and keys; nothing about
positions crosses the boundary but that number. A chunk must continue the session (`position_start` is the tokens the
session holds); a chunk at an earlier position is refused (a repeat of an executed step is answered from the kept reply
instead, below).

State ownership and reset: the KV of a stage's full-attention layers lives in the stage (bf16, `context_tokens` positions,
written by position); the convolution state (bf16) and the delta-rule state (fp32) of its linear-attention layers live in
the stage. `begin_session` zeroes the states and sets the position to 0; the KV is overwritten as positions are used.
One session at a time per stage by default; `--sessions 2` gives each of two sessions its own KV, states and graph (below). The first stage also holds the embedding
(packed), the last the final norm and the LM head.

In the engine the same thing in one process is `Dyn::stage_rows(tokens, at, io, head)` (the first stage embeds `tokens`,
a later one reads `io`; a stage that is not the last writes `io`; the last runs the head on the last `head` rows, at most
what `Dyn::set_head_rows` made room for, 1 by default; `Dyn::head_out` reads them),
`Dyn::boundary()` says the slots and widths, and `split::Split` chains stages on threads (one GPU each, or the same GPU
twice, as `rangecheck` does). `Dyn::layer_costs()` gives each layer's bytes.

The other families' stages (Gemma 4, Nemotron, GLM, Muse, DeepSeek-V4 in `Split`) hand over the slots `X, H, D, O, S`
where the model has them, as before this change; their boundary is not reduced to the live pair here.

## The process and its protocol

```
glyd-engine rangeworker --model MODEL[:level] --layers A..B [--ctx N] [--chunk N] [--head-rows N] [--sessions N] [--device D] [--layout L] [--no-graph]
```

MODEL is a directory (a glyd/NAME-LEVEL snapshot, a lossless glyd-v4 save, or a bf16 checkpoint packed at load) or a Hub
id in the cache; `ORG/NAME:level` takes glyd/NAME-level from the cache. `--head-rows N` (1 to 64 and the chunk; 1 by
default): the most rows the last stage's head runs on in one chunk; each row more costs the last stage `vocab x 2 + 8`
bytes and the LM head's workspace for that many rows where it needs more (Qwen3.5-9B linnet, `--head-rows 8`: 3,476,536
bytes more scratch, the workspace unchanged). `--sessions N` (1 or 2; 1 by default): the sessions open at once, each with
its own KV, states and graph, made at `prepare` ([two sessions a stage](#two-sessions-a-stage---sessions-2)). The worker loads nothing until `prepare`. Its
standard input and output carry frames; its standard error is its own lines (the load's, errors'). `--no-graph` is a
development build's.

A frame: `[u32 head bytes][u32 body bytes][head][body]`, the lengths little-endian, the head UTF-8 JSON, the body raw bytes
(the boundary tensors or the logits, never JSON numbers). A head is 1 byte to 1 MiB; a body at most `limits.max_body_bytes`
(describe: the chunk's boundary or `--head-rows` rows of logits, whichever is more, at most 64 MiB). A length past its limit is refused
before anything is allocated, and the worker ends (the stream is not trusted past it). Every request has `op` and `id`
(1 to 128 of letters, digits, `_`, `.`, `-`); every reply carries the `id`, `ok`, and the fields below or `error`.

| op | request | reply |
| :--- | :--- | :--- |
| `describe` | | `protocol`, `build`, `incarnation`, `pid`, `model` (dir, family, layers, hidden, vocab, level), `stage` (layers, first, last, embedding, final_norm, lm_head, sampling), `boundary` (dtype, byte_layout, bytes_per_row, tensors, in, out, positions), `state` (kv, recurrent, reset, verify), `limits` (context_tokens, chunk_tokens, max_head_bytes, max_body_bytes, sessions: `--sessions`, 1 by default, kept_replies, kept_reply_bytes, head_rows: the last stage's `--head-rows`, 0 on the others, verify_rows: 8), `ops`, `device` |
| `inspect` | | `available` false before `prepare`; after it: `layers` [{layer, kind, weights_bytes, kv_bytes_per_token, state_bytes}], `embedding_bytes`, `lm_head_bytes`, `small_tensors_bytes`, `kv_bytes_per_token`, `state_bytes`, `boundary_bytes_per_token`, and the allocation (below) |
| `prepare` | | loads the stage: `instance` (a handle), `load_s`, `describe`, `weights_bytes`, `kv_bytes`, `state_bytes`, `scratch_bytes`, `engine_bytes`, `peak_bytes`, `device` {name, compute_capability, free_bytes, total_bytes}. A second `prepare` is refused until `unload` |
| `commit` | `instance` | `committed`, `ready`. Nothing runs before it |
| `begin_session` | `instance` | `session` (a handle), `context_tokens`, `kv_bytes`, `state_bytes` (the session's own), `tokens` 0. The states zeroed. While fewer than `limits.sessions` are open |
| `run_chunk` | `session`, `phase` (prefill, decode), `position_start`, `head` (true: the last stage runs the head on the last row), the first stage: `tokens` [ids]; a later stage: `rows` and the body (the boundary of the stage before); the last stage: `logits` true for the body, `head_rows` k (1 by default: the head on the last k rows, at most the chunk and `--head-rows`), `sampling` {temperature, top_p, top_k, min_p, seed, step} (below; with head_rows 1); any stage: `verify` true (a verify chunk: phase decode, 1 to `verify_rows` rows), `keep_previous` K (below: drafted tokens) | `rows`, `position_end`; a stage that is not the last: the body (the boundary) and `body` {what, rows, bytes}; the last, with `head`: `token`, `logit` (the last row's: the greedy token, or the drawn one and its logit), `tokens` [the greedy token of each of the k rows] where `head_rows` was given, and the logits' body where asked (`body` {what, dtype, rows k, width, bytes}) |
| `status` | `session` (optional) | `state` (empty, idle, running, cancelling, cancelled), `incarnation`, `instance`, `session`, `session_cancelled`, `tokens` (the named session's, else the earliest begun open one's), `running` {id, op, ms, session} or null, `sessions` [{session, tokens, cancelled}] (every open one, in the order begun), `draws` {from_gpu, read_row} (the sampled tokens this process drew from what its GPU made of the row, and by reading the whole row: none is to be of the second). Answered by the control thread at once, whatever runs |
| `cancel` | `session` | the status (that session's). The session takes no more chunks; a chunk under way runs to its end, its output is withdrawn (its reply is an error), the state has it. Another session is not touched. `end_session` closes it |
| `end_session` | `session` | `ended`, `tokens`, `cancelled`. Waits its turn behind a chunk under way |
| `unload` | `instance` | `unloaded`, `retained_bytes` (the engine's buffers left: 0). Refused while a session is open |
| `exit` | | `bye`; the process ends. The end of the input ends it too, after what is queued |
| `tokenize` | `text`, `chat` (false: the text as glyd serve takes a plain prompt, the tokenizer's special tokens added; true: a chat of one user message through the model's chat template with its generation prompt, as serve renders it); or `messages` [{role: system, user or assistant, content}] (1 to 256 turns, the last a user's): a chat through the template with its generation prompt, as serve renders the same messages (with no template, serve's lines of `role: text` and `assistant:`); `thinking` (a template's `enable_thinking`; the template's default where not given) | `ids`. Answered by the control thread; the first `tokenize` or `detokenize` loads the tokenizer and the template from the model directory (no weights, no `prepare` needed) |
| `detokenize` | `ids`, `skip_special` (true by default, as serve) | `text`: the whole text of the ids. A caller that streams sends the running ids and takes what is new; control thread |

The allocation fields are the engine's own counters (the bytes it asked the driver for, as `/v1/glyd/info` counts them):
`weights_bytes`, `kv_bytes`, `state_bytes`, `scratch_bytes` (the activations for a chunk, the products' workspaces, the
logits, the rotary tables), `engine_bytes` their sum, `peak_bytes` the most so far. The CUDA context and the kernels' code
are not in them: about 300 MiB a process in nvidia-smi (the record: a linnet worker at 3,694 MiB against 3,398 MiB of buffers).
The record's numbers for Qwen3.5-9B: a stage of 16 layers holds 6.03 GB (penguin), 3.10 GB (linnet), 3.66 GB (kestrel) of
weights, 21,233,664 bytes of KV for 1,296 positions, 25,755,648 bytes of states and 416,096,036 bytes of scratch (a 1,024-token
chunk's); the boundary is 16,384 bytes a token.

Threads: a control thread reads the frames and answers `describe`, `status`, `cancel`, `tokenize` and `detokenize` itself; an executor thread owns
the engine and runs the rest in the order they came, one at a time. Replies can interleave: a `status` sent during a
`run_chunk` is answered before the chunk's reply. Each reply is written whole.

Handles and incarnations: `prepare` and `begin_session` give opaque handles, `i-<incarnation>-<n>` and `s-<incarnation>-<n>`,
the incarnation a number of this process start (`describe` says it). A handle of another incarnation is refused and the
refusal says so: a restart invalidates every handle, nothing revives one. A handle is not a pointer, a pid or an address.

Sampling: `sampling` on the last stage's `run_chunk` draws the last row's token with glyd serve's sampler (sampler.rs): the
model's generation_config.json values where a field is not given (serve's defaults), temperature 0 to 2 (above 0 and below
0.01 is 0.01; 0 is greedy), top_p above 0 to 1, top_k (0: all), min_p 0 to 1, `seed` (0 to 2^64 - 1, required) and `step`
(required: the tokens of the answer before this one, 0 for the token after the prompt, at most the context). Serve's
generator starts from the seed and draws once a sampled token; the worker starts it from the seed and moves it on `step`
draws, so the request alone says the token: the same seed, step and logits give serve's token, and a repeat of the request
gives the same. No penalties (repetition, presence, frequency) and no logit_bias: they need the tokens so far; a request
with one is refused, and a model's default repetition_penalty is not applied. Ask for the logits and apply them on your
side if needed. The token is drawn on the worker's GPU as one GPU's scheduler draws it (`sched::draw_row`): from the row's
candidates and, where the cut or the point lies past them, the weights and tokens of ranges of the row, every sum exact in any
order; the row is not read on the host for it (its body only where `logits` is true, and the drawn token's `logit` is the
model's logit of it, whether or not the body comes). `status`'s `draws` counts them.

Repeated requests: the worker keeps the last 256 requests by id with their replies (their bodies up to 64 MiB in all; the
oldest bodies are let go first). The same id with the same request comes back from the kept reply and is never run
again: a `run_chunk` sent twice (a lost reply) runs once, the state moves once. The same id with a different request is
refused ("differs from this one"). A repeat whose reply was let go is refused and says so. A new request at a position the
session has passed is refused ("position_start"). Nothing is retried by the worker; an ambiguous loss is the caller's to
repeat by id, which is safe.

## Drafted tokens: verify chunks and keep_previous

Several drafted tokens per hop: a decode chunk of rows `[t0, d1 .. dk]` (t0 the last confirmed token, d the drafts,
k at most 7) through every stage, then the next chunk keeps the rows that were right and drops the rest.

- `verify: true` on `run_chunk` (phase `decode`, 1 to `limits.verify_rows` rows, 8): a verify chunk. Each row runs as a
  decode step at its position would (the delta rule one row after another, the decode step's attention, products of at
  most 8 rows, which take the one-row kernel), so row `i`'s boundary, token and logits are the plain step's at
  `position_start + i`, bit for bit. Before the pass the stage copies its linear-attention states aside and keeps each
  such layer's inputs for the pass (b, a and the convolution's input rows).
- `keep_previous: K` on the next `run_chunk` (any phase): before it runs, the stage keeps the first K rows of the verify
  chunk (1 to its rows) and drops the rest: the linear-attention states go back to before the verify chunk and the
  convolution and delta rule run again over the K kept rows' inputs; the keys and values past them are written over by
  the next rows before any row reads them. The chunk's `position_start` must be the verify chunk's + K. One request
  does both; a refused request changes nothing.
- Every stage gets the same `verify`, `keep_previous` and `position_start`: first, middle and last alike.
- Refused: `keep_previous` 0 or past the verify chunk's rows, `keep_previous` when the session's last chunk was not a
  verify chunk, `verify` on a prefill chunk or on more than `verify_rows` rows, a `position_start` other than the
  verify chunk's + K. A chunk with no `keep_previous` after a verify chunk keeps all its rows. A verify chunk of 1 row
  is a plain step (no copy); `keep_previous: 1` after it is allowed.

Greedy, the way Starlings uses it: the last stage runs `head_rows: k + 1` and gives each row's greedy token g_i in
`tokens`. a = the longest prefix with g_i == d_(i+1); the new tokens are d1 .. da and g_a. The next chunk is
`[g_a, new drafts]` with `position_start` = start + a + 1 and `keep_previous: a + 1`. Sampling on the last stage stays
head_rows 1 (refused with more rows); draw on the caller's side from the logits' body for anything else.

Cost on the RTX 4080 SUPER, Qwen3.5-9B linnet ([the record](../benchmarks/gpu/rtx4080s-range-rewind-2026-10-22/README.md)).
The whole model in one process (24 linear-attention layers, 64-token prompt): a one-row step 9.28 ms; a verify pass of 8
rows 10.26 ms, of it the states' copy 0.17 ms (51,511,296 bytes); a rewind keeping 1 to 7 of the 8 rows 0.40 to 0.59 ms
(0.03 ms more a kept row; keeping all 8 does nothing). Two worker processes on the one GPU: a plain step 9.89 ms a token;
verify chunks of 8 rows with 0 to 7 drafts kept in turn (4.3 tokens a chunk) 12.80 ms a chunk, 3.02 ms a token. The
first verify chunk allocates the room: a copy of the stage's states, 8 rows of each linear-attention layer's inputs and
the decode attention's parts for 8 rows (a 16-layer stage of Qwen3.5-9B: 25,755,648 + 2,371,584 + 2,113,536 bytes).

In the engine: `Dyn::verify_rows(tokens, at, io, head)` (`stage_rows` as a verify pass), `Dyn::rewind(keep)`,
`Dyn::verified()` (the last chunk's start and rows where it was one), `Dyn::keep_states()` (the copy alone).

## Two sessions a stage: --sessions 2

For a pipeline of two requests: stage 0 runs request B's chunk while stage 1 runs request A's. `--sessions 2` makes room at
`prepare` for two sessions. Each has its own keys and values (the whole context's), its own linear-attention states, its
own copy of them for a verify chunk (made at its first one) and its own one-row graph, captured on its buffers at
`prepare`. The activations, the workspaces, the head's outputs and a verify pass's attention parts are the stage's,
shared: one pass at a time on a stage, the requests run in the order they came, as before. In the engine:
`Dyn::add_seqs(k)`, `Dyn::select(i)` (swaps the sequence's buffers, position, verify state and graph in; nothing is
copied).

- `describe`: `limits.sessions` is N. With `--sessions 1` (the default) describe is what it was, byte for byte.
- `prepare`: `kv_bytes`, `state_bytes` and `engine_bytes` count every session's. `begin_session`'s `kv_bytes` and
  `state_bytes` are the session's own. `inspect`'s per-layer `kv_bytes_per_token` and `state_bytes`, and its
  `state_bytes`, are one session's; its allocation fields are `prepare`'s.
- `begin_session` gives a new handle while fewer than N are open; past that it is refused ("this worker takes N at a
  time"). An ended session's room is the next one's.
- `run_chunk`, `end_session` and `cancel` name their session; the chunk runs on that session's buffers only.
  `position_start`, `verify` and `keep_previous` are each session's own: a verify chunk of A, then chunks of B, then A's
  `keep_previous` is right.
- `cancel` marks the named session only. The other session goes on as if nothing happened: its chunks run, its state is
  not touched, before and after the cancelled one's `end_session`. A chunk of the cancelled session under way runs to
  its end and its reply is an error, as before.
- `end_session` ends that session; the other goes on. `unload` is refused while any is open.
- `status` takes an optional `session`. Its `session`, `session_cancelled` and `tokens` are that session's, or the
  earliest begun open session's where none is named (with one session: that one, as before). `state` is `running` while
  any request runs on the stage (either session's), `cancelling` while a request of this session runs after its cancel,
  `cancelled` where this session is cancelled, else `idle` (`empty` before `prepare`). `running` carries `session` (the
  session the request names, or null). `sessions` lists every open session, {session, tokens, cancelled}, in the order
  begun. A `status` naming a session that is not open is refused, as `cancel` is.
- One change for one session too: `session_cancelled` is true and `state` `cancelled` as soon as a cancel is asked. Before,
  a cancel with no chunk under way left `session_cancelled` false and `state` `idle` until the next request ran.

Cost on the RTX 4080 SUPER, Qwen3.5-9B linnet ([the record](../benchmarks/gpu/rtx4080s-range-sessions-2026-10-22/README.md)).
A session more on a stage of 16 layers: its keys and values (16,384 bytes a token: 134,217,728 at ctx 8,192) and its
states (25,755,648 bytes); prepare's `engine_bytes` 3,677,122,908 with `--sessions 1`, 3,837,096,284 with 2; the
device's free bytes 5.7 MB lower still (the graph, the driver's rounding). Its room and graph are made at `prepare` in
about 12 ms. A pass costs the same: the whole model's one-row step 9.34 ms alone and 9.34 ms with two sequences in turn;
two worker processes on the one GPU, a plain step 9.92 ms with one session, 9.87 ms with two in turn (one client sending
the passes one after another: nothing overlaps here). For Qwen3.8-27B (64 layers: 16 full-attention, 48
linear-attention) a session is 64 KiB a token and 147 MiB of states for the whole model; on a stage of half the layers
32 KiB a token and 73.5 MiB (330 MiB at ctx 8,192; not run here).

## rangecheck

```
glyd-engine rangecheck MODEL[:level] [--prompt 1024] [--new 256] [--cut L | --cuts A,B] [--ctx N] [--chunk N] [--device D] [--no-graph] [--workers [--worker-exe PATH] [--sessions 2]]
```

A development build's command (it loads the model three times and is a proof, not a tool). `--worker-exe`: the workers
from another build of the program (a release build's) against this build's whole model. CI has no GPU: there
`glyd-engine/tests/rangeworker.rs` starts the worker as the build ships it, with no GPU and no weights, and checks
`describe`, `tokenize`, `detokenize`, `inspect` before `prepare`, a refused handle and `exit`.

One process, one GPU: the whole model loaded, a prompt of N random tokens (`report::random_prompt`, the bench's) and
`--new` greedy tokens, every step's logits (the bf16 words) and token kept; the whole model freed; the stages loaded
(`Split`, each stage's weights once, cut before each of `--cuts`, or in the middle), the same run, every step's logits and
token compared bit for bit; with `--workers`, the whole model once more (the scheduler's sampled replies made between the two), and the stages again as `glyd-engine rangeworker` processes chained through the
protocol, compared the same way, then the protocol's rules: a repeated request id, a conflicting repeat, a stale position,
another incarnation's handle, an unknown op, `status` during a chunk, `cancel` during a chunk; then `tokenize` and
`detokenize` (a text there and back, and as a chat; the text is one the model's tokenizer gives back as it was given, so a
random model made with `logitsmodel CASE DIR --servable` is one the check can use), three kinds of seeded sampled runs of up
to 16 tokens each (a temperature with top-p and top-k, a flat top-p past the GPU's candidates, a top-k of 200 with top-p and
min-p) from fresh sessions, with the rows' body asked for and without it: the same tokens and `logit`s both ways, the same as
serve's sampler drawing token after token over the same logits, and one GPU's scheduler's reply to the same request (the
whole model's `sched::Sched`, its tokens and the row each was drawn from) token for token, every row and every reply's `logit`
bit for bit; the last sampled request sent again (the kept reply, not a new draw); the last worker's `draws`: every sampled
token from the GPU's view of the row, none by reading it; a repetition penalty refused with the session as it was; and the head on 8 rows (the prompt, then the first 8 greedy tokens as one chunk with
`head_rows` 8: every row's token and logits equal to the whole model's on the same chunk, bit for bit; the last worker runs
with `--head-rows 8`); drafting (the prompt, then verify chunks of 8 rows whose 7 drafts are the plain run's next a
tokens and then wrong ones, a = 0 to 7 in turn, each next chunk with `keep_previous: a + 1`: every kept row's token and
logits equal to the plain run's, bit for bit) and its refusals; with `--sessions 2` (the workers started with it) two
sessions on every worker, a pass of each in turn (a prompt chunk, a step or a verify chunk): A on the prompt, B on another
(the prompt backwards, a quarter shorter); a third session refused; both plain to half the tokens, then A cancelled (its
status cancelled, B's idle, A's next chunk refused), B 4 steps alone, A ended, a session C on A's room from position 0 in
turn with B to the end; then A drafted with B plain, then both drafted; then both sampled (A's first kind, B's third on its own prompt), each against one GPU's scheduler's reply to its request, and the draws counted again. Every session's tokens and logits against its own
prompt's whole-model run, bit for bit. Then `end_session`, `unload`, `exit`. Before the stages, the whole model times a
step, a verify pass of 8 rows, its states' copy and a rewind to each keep; with `--sessions 2` also a second sequence's
bytes and graph capture, and a step of each in turn against a step alone. Prints "all checks passed", or the first
difference and exits 1.
Each run's SHA-256 of (tokens, logits) is printed. On a random model: `glyd-engine logitsmodel qwen35 DIR --servable` (the
tokenizer and chat template `--workers` needs; `qwen35-wide` is the other dense shape), then `glyd-engine rangecheck DIR:linnet
--workers [--sessions 2]` (the verify chunks are checked at linnet only).

## Against Compute's proposal (partial-worker-contract.md)

| Compute | here |
| :--- | :--- |
| Describe: version/capability, incarnation, model, level, boundary profile, device, pools, ops, limits | `describe`: all of it but the pool inventory (the allocation is `prepare`'s and `inspect`'s, the engine's own counters) and a `content_hash` of the boundary contract (the boundary is the `boundary` object; hash it on your side) |
| InspectModel: a complete cost profile before loading | `inspect` answers from the loaded stage's actual allocations; before `prepare` it says `available: false`. Costs from the config alone are not given |
| Prepare / Commit | `prepare` loads and allocates (the KV for `--ctx`, the states, the scratch); `commit` is a state transition (nothing runs before it). Readiness is `prepare`'s success; there is no partial allocation to roll back |
| BeginSession with client-owned session/generation, sampling settings | `begin_session` gives the worker's session handle; `--sessions` (1 or 2) sessions a stage at a time; the context is the worker's `--ctx`; sampling is per chunk on the last stage: greedy, or `sampling` with a seed and a step (serve's sampler; the logits as a body where asked). Client-named sessions and generations are the caller's mapping onto the handle |
| RunChunk with DataScope/StepIdentity, ordered TensorLayouts | `run_chunk` with `position_start`, `phase`, `head`, the tokens (first stage) or `rows` and one body in the declared order (residual, mlp_out). The layouts are fixed per model (describe) and not sent with each chunk; the worker checks the body's length against `rows` |
| Status / completion, independently responsive | `status` from the control thread while a chunk runs (`running` {id, op, ms}); completion is the chunk's own reply |
| Cancel: withdraw output, report drain after the device is done | `cancel` marks the session; a chunk under way completes (one GPU pass, not interrupted), its reply is an error and its output is withdrawn; `status` says cancelling until then, then cancelled; `end_session` after it |
| EndSession / Unload with retained pools | `end_session` (waits behind a chunk), `unload` (refused while a session is open) with `retained_bytes` |
| Handles scoped to the incarnation; exact repeats answered from history, conflicting repeats fail, retired sequences cannot execute | the same: incarnation-scoped handles, the kept replies by id (256, 64 MiB of bodies), a stale `position_start` refused |
| Frames: 1 MiB JSON, length-prefixed; bodies up to 64 MiB; never JSON floats | the same limits; the body limit is the stage's own (describe) |
| Process ownership, pipes, environment, credentials | the worker is a plain child on pipes; it reads HF_HUB_CACHE/HF_HOME for a Hub id (a directory needs nothing: a CUDA build loads no GPU library). No network, no shared memory, no broker |

## Supported and measured

- Models: Qwen3.5 dense (9B here; 27B, 3.6-27B and 3.8-27B are the same description, and Config35 takes their configs, so
  the worker takes them; the 27Bs are not run here yet) at every level the engine loads:
  penguin (a lossless save, glyd-v4 or glyd-v7 `mma`, or a bf16 checkpoint packed at load), kestrel, linnet, wren. The
  record has penguin, linnet and kestrel on the 4080 SUPER, two stages (0..16, 16..32) and linnet in three (0..11, 11..23, 23..32):
  every step's logits and token equal to the whole model's, bit for bit, in one process and across worker processes.
- Verify chunks and keep_previous: linnet on the 4080 SUPER, two stages and three (0..11, 11..23, 23..32), prompts of 64
  and 1,024 tokens: drafted equal to plain, bit for bit, every keep from 0 to 7 drafts
  ([rtx4080s-range-rewind-2026-10-22](../benchmarks/gpu/rtx4080s-range-rewind-2026-10-22/README.md)).
- Two sessions a stage (`--sessions 2`): linnet on the 4080 SUPER, two stages and three, prompt 1,024 and 768: each
  session's 256 tokens and logits equal to its own whole-model run, bit for bit, with the passes of the two in turn,
  plain, drafted and mixed, and one session cancelled and ended half way
  ([rtx4080s-range-sessions-2026-10-22](../benchmarks/gpu/rtx4080s-range-sessions-2026-10-22/README.md)).
- Other families run as stages in `Split` (Gemma 4, Nemotron, GLM: `splitcheck` across two GPUs); the worker and
  `rangecheck` take Qwen3.5 only for now.
- Speed is not the point here and not tuned: a chunk's boundary goes GPU to host to GPU through pipes (16 KiB a token).

## Known limits

- At most 2 sessions a stage (`--sessions`), fixed at start; their room is made at `prepare` whether used or not. One
  sequence a session. Passes on a stage are one at a time (the sessions share the activations). Sampling on the last stage draws the last row only (head_rows 1), with
  no penalties and no logit_bias; the logits come back for anything else on the caller's side.
- The head on at most 64 rows of a chunk (`--head-rows`, fixed at start): past 64 rows a product decodes its matrix
  first, and the LM head is not one the engine makes room to decode. A prefill chunk's rows are not one-row steps: their logits
  equal the whole model's on the same chunk, not a decode step's (the tokens agreed on 8 of 8 rows in the record run);
  a verify chunk's rows are (above).
- Verify chunks: Qwen3.5 only (the worker's family), at most 8 rows (the products' one-row kernel), greedy checks on the
  caller's side. A rewind reaches back one verify chunk only. The exactness is measured on the 4080 SUPER (Ada) with the
  linnet layout; another GPU or layout whose product of 2 to 8 rows takes another kernel than one row's would not be exact.
- `inspect` is the loaded stage's allocation, not a plan from the config.
- A cancel cannot interrupt a chunk's GPU pass; it takes effect at the chunk's end (a 1,024-token prefill chunk of half
  the 9B model is about 100 ms on the 4080 SUPER).
- Transport security, multi-machine networking and a process owner are the caller's (Compute's); the worker is pipes only.
- `rangecheck` is in a development build only; the release build has the worker. CI checks the worker without a GPU;
  the GPU proof is a `rangecheck --workers` run by hand.
