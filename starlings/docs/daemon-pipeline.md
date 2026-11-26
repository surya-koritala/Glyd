# Daemon pipeline (GLY-188)

One chat request runs through two warm `glyd-engine rangeworker` stages held by two
enrolled worker daemons. The coordinator admits the client, owns the request's plan,
leases and grants, and relays text back as SSE. Boundaries and tokens go straight
between the stages over enrolled TLS. Measured on the PC in
[benchmarks/2026-10-22-daemon-pipeline](../benchmarks/2026-10-22-daemon-pipeline/README.md).

## One command per machine (GLY-362, GLY-363)

Install (Linux x86_64, an NVIDIA driver for CUDA 13, no root): unpack the download and run its
`install.sh`. It checks `SHA256SUMS` and puts starlings and glyd-engine in
`~/.local/lib/starlings`, with the license, checksums and uninstall script retained there,
and the command in `~/.local/bin`. It validates the staged executable and restores the
previous installation if activation fails.
`install.sh --uninstall` removes them
(state and models stay); an upgrade is a new download's `install.sh`.
`scripts/build-starlings-linux.sh OUT_DIR` builds starlings against glibc 2.28 in a Rocky Linux 8
container, as the engine release is built. `scripts/package-beta.sh STARLINGS ENGINE_DIR OUT_DIR`
builds the download from that and the engine release: ENGINE_DIR is an unpacked
`glyd-engine-vX.Y.Z-linux-x86_64` bundle of release-gpu.yml, of which it takes `glyd-engine` and
`LICENSE` (never a hand build). The engine loads no GPU library in a CUDA build, so the download
holds none.

One machine, both stages (on GPU 0 and 1, or both on GPU 0):

    starlings up --model glyd/Qwen3.8-27B-swift [--ctx 4096] [--sessions 1|2] [--drafts 7]

`--model` is a directory (absolute, or starting with `./` or `../`) or a Hub name. A name is fetched
by the engine into the Hugging Face cache, after the operator agrees to the download.
An existing cached revision is reused without fetching a newer `main`.
`--engine` defaults to the one installed beside starlings. `--gpu-lib` is optional and has no
default in the download: without it the stage's engine runs its own kernels. A library named with
it (or found beside starlings) is passed on as the engine's `GLYD_GPU_LIB`.

Two machines on one LAN:

    first:   starlings up --model glyd/Qwen3.8-27B-swift --machines 2 [--host LAN_IP]
    second:  starlings join --model glyd/Qwen3.8-27B-swift [--host LAN_IP]   (then paste the join code)

`up` enrolls a coordinator, its stage worker(s) and a client with the steps below, run as this
binary's own commands. It prints the web UI link and the API address. Both machines need the
same model files and the same engine.

With two machines, two codes are copied by hand:

- The join code goes from the first machine to the second. It holds the cluster's one-time
  worker invitation (valid five minutes), the context, sessions and drafts, and the first GPU's
  free memory.
- The approval code goes from the second machine back to the first. It holds the second
  machine's node, fingerprint, LAN address, engine digest, model digest and free GPU memory.

The join code is pasted at the second machine's prompt, never given on its command line. Byte
arrays in it are base64, so it fits one terminal line (about 2 KB).

Two checks guard the hand-carried codes:
- The cluster check is 16 hex digits of the coordinator's CA certificate. The first machine prints
  it with the join code; the second shows the one its code carries, and the two must match.
- The approval check is 16 hex digits of the second machine's node and key. The second machine
  prints it, and the first machine's operator must type it to approve. A code swapped on the way
  gives another check, and that node is not approved.

The invitation files are deleted once the machine has joined. The first stage's control listener
is on loopback, and every stage drops a control connection that does not come from the
coordinator's address (or loopback) before its handshake, which gets 2 s. Every run prints the
engine (and the GPU library, if one is named) with their digest prefixes, and the model, before it asks.

From the same numbers both machines compute the same split (`split::cut`: it fits both GPUs with
1 GiB to spare and leaves the two equally full). The second machine shows its layers and asks
before it sends its code (`--yes` skips that). The first machine shows both stages and always asks
before it approves the other node. Each machine then runs only the stage its own operator
confirmed.

A rerun of either command reuses its saved enrollment and split after checking the checkpoint
contents and split bounds. `up` also checks the context, sessions, drafts, host and number of
selected GPUs. Incompatible or older saved records require a new `--state-dir`; they are never
silently reapproved. The ports are fixed:
- registry 9741, on the first machine's LAN address;
- API 9750 and UI 9751, loopback;
- stages 9760 and 9770, plus one port per extra session.

Stage addresses must be private LAN addresses. Machines on public addresses, such as cloud pods,
need a private network first.

Limits (GLY-366 review):
- The client API allows 4 connections per source address, and local clients all come from
  127.0.0.1. Another user on the same machine can hold them all.
- Checkpoint contents are hashed at approval and stage load. This reads every weight shard;
  startup cost on large checkpoints still needs native measurement. It does not prevent a
  hostile process running as the same user from modifying files while inference runs.
- The download has SHA256SUMS for corruption but is not signed yet.
- The UI's secret link is recorded in the browser's history when it is opened.

`starlings model split MODEL FREE0_GIB FREE1_GIB [--ctx N] [--sessions N]` prints the split
without starting anything.

## Processes

```
starlings --state-dir COORD daemon --pipeline PIPELINE.json --approve DIGEST
starlings --state-dir S0    daemon --stage STAGE0.json --approve DIGEST
starlings --state-dir S1    daemon --stage STAGE1.json --approve DIGEST
```

`model files-digest MODEL_DIR` computes the content identity needed by a manually prepared
stage record. `model stage-digest FILE` and `model pipeline-digest FILE` print a record's digest;
they approve nothing. `cluster service` on a stage daemon shows its state, request
counts and the engine's device report.

`StageRoute` (worker): generation and data listeners, the engine path and SHA-256, the
model directory and required `model_files_sha256`, layer range, `--ctx`/`--chunk`, the per-request maxima (prompt +
output <= ctx), the approved `describe` digest, profile names, device/pool ID, an
engine-bytes bound, the engine's environment (no `LD_*` or `DYLD_*` keys) and, with
`GLYD_GPU_LIB`, that library's approved SHA-256. At start the daemon spawns the engine
(no shell or PATH, cleared environment, own process group) from bytes that cannot change
after their check: on Linux a sealed memfd (no write, grow, shrink or new seals) run as
`/proc/self/fd/N`, elsewhere a copy in a fresh 0700 directory under the state directory.
The engine's `GLYD_GPU_LIB` names a checked copy there too. Then the daemon checks
`describe` against the approved digest, and loads and commits the stage. The path-independent
model identity hashes the names, sizes and contents of root files and chat templates,
including configuration, tokenizer and every safetensors shard. Hub blob file symlinks work;
template directory symlinks are refused. The files are checked before starting the engine and
again during loading. Two machines may use different absolute model paths.
A load past 10 minutes, or a stop, kills it. The stage stays loaded across requests;
each request opens and ends one engine session.
While warm, the heartbeat sample advertises the backend with `layer_sharding`, so the
coordinator's plan check sees the stage as eligible.

`PipelineRoute` (coordinator): the loopback client listener, a two-stage
`layer_pipeline` plan template, each stage's generation and data address, the prefill
chunk, the output maximum, the lease lifetime (<= 60 s) and the stop token IDs.

## One request

1. `admit` checks the bearer token, the model and the body, as for whole requests.
   Temperature 0 is greedy; above 0 the last stage draws with the engine's sampler
   from (seed, step, logits), so a request alone fixes its tokens.
2. The first stage counts: it tokenizes the system/user/assistant conversation with the model's chat
   template (the engine's `tokenize`, `chat: true`).
3. Per request the coordinator writes a plan (its epoch, a new ID and times), one
   lease and one session per stage, and a `StageIssuer` for the live client
   (`stage_issuer_for`: the same plan/token/permission checks as `stage_issuer`).
   Each stage gets the assignment, then Prepare, Commit and Begin. The stage holds
   its backend reservation right after Prepare and checks the plan's stage entry
   (node, layers, build, backend, device) and the session's counts against its record.
4. `Start` gives both stages the leases, data addresses, chunk size and stop IDs.
   Each renews its two route grants (0→1 and 1→0) now and then every 0.9 s while it
   answers the coordinator's polls; a grant lasts 2 s from its request.
5. The stages connect 0→1, then 1→0. A stage takes its peer's data connection only
   from the peer's data address, only if it speaks within 250 ms, and only after the
   stage handshake for that route's grant; anything else is dropped and counted
   (`refused_connections`) while it waits, up to 5 s. The engine session opens on the
   steps thread, so polls and renewals go on meanwhile. Step k has sequence k and
   invocation ID
   `step_invocation(generation, k)` on both stages. The prompt runs in chunks:
   stage 0 starts each one locally (`enqueue_prompt`, on the 1→0 grant), sends its
   boundary (residual and MLP output as one bf16 `[2, rows, hidden]` tensor) and
   moves on; stage 1 runs each boundary in order. The head step returns its token to
   stage 0 as step k + 1: decode phase, position = start + rows, one u32 row. Stage 0
   detokenizes the answer so far and sends only the new text to the coordinator.
   Each stage opens its next receive (and grants its credit) as soon as its own output
   is on the wire, before that output's receipt returns ([data flow](data-flow.md)), so
   a token costs one round trip between the stages; stage 0 detokenizes after the next
   step has gone out.
6. Both stages stop at the same token: the output maximum or a stop ID. Each ends
   its engine session, checks with `status` that no session is open, and reports
   layers, steps, rows, token counts and digests, engine time and the ended session's
   positions. The coordinator fails the request unless both reports agree (token count
   and digest, stop, prompt count, at most `max_tokens`). It logs `PIPELINE_REPORT`
   without token IDs; the prompt and answer digests are logged only by test-backend
   builds. It takes the next request only after both stages have closed their
   connections, or 5 s after a failure once the stages had started.

## Drafted decode (GLY-360)

With `draft_tokens` k > 0 in the pipeline route, a greedy request's decode steps carry the
last accepted token and up to k guesses after it (at most 7, and a step is at most one chunk).
The last stage's route sets `head_rows` (the engine's `--head-rows`, k + 1 or more) so it can
return a token per row; 0, the default, passes nothing, as before. Both stages make the same
guesses: an n-gram lookup over the prompt's token IDs and the tokens accepted so far. Only a
drafted request (greedy, k > 0) has the first stage return the prompt's IDs at Count and the
coordinator pass them to the last stage at Start; any other request sends none. Each stage can therefore
predict the other's next message exactly, and every receive still admits one exact header.

- Stage 0 runs [t, d1..dk] at position p, as an engine `verify` chunk on both stages.
- Stage 1 runs it with `head_rows` k+1 and returns every row's greedy token g0..gk: k+1
  TokenIds rows at p+1, one per position.
- Both take a, the longest prefix with g(i) = d(i+1). The new tokens are g0..ga, and both stop
  at the same token.
- Stage 0 builds its next input locally (`enqueue_decode`): [g_a, next guesses] at p+a+1. Both
  engines then keep a+1 rows of the verify chunk (`keep_previous`) before running it.
- The core admits a decode step of 1 to 8 rows, and after a multi-row step a start anywhere in
  (start, end]. Sampled requests send no guesses (the engine draws only a chunk's last row).
  Reports count `drafted` and `accepted`.

## When an engine stops answering

Every engine call in a request is bounded: the stage's control loop ends the request
at its lease end (or on a daemon stop) and cancels its grants; data waits end with them.
A chunk still running gets the engine's `cancel`: it finishes, its output is withdrawn,
and the stage stays warm. Only a step still blocked after max(3 s, 4 x the longest chunk
so far) is ended by killing the engine's process group, which closes its pipes. A
watchdog also kills the engine 75 s into any request.
A killed or ambiguous engine leaves the stage `failed`: it stops advertising and takes
no more requests until the daemon restarts; the daemon still stops on request.

## Web UI (GLY-364)

`daemon --pipeline FILE --approve DIGEST --ui 127.0.0.1:PORT --ui-client CLIENT_NODE` also serves a
page (plain `daemon --ui ...` serves it without chat). The daemon prints
`UI_READY: http://127.0.0.1:PORT/#k=SECRET`; open that address.

- Cluster: the stages in plan order, each node's memory pools, state and sample age, the layer
  split with the embedding and head owners, link RTT and bandwidth when a resource sample carries
  them, other nodes, and the join commands while fewer than two workers are online.
- Chat: system prompt, history, temperature and seed. Under each answer: tokens, decode tok/s,
  time to first text, setup, engine time per stage, the rest (pipeline time minus engine time),
  drafts held, and whether both stages reported the same tokens.
- Requests: the last 50 `PIPELINE_REPORT` records (they now carry `id`, `at` and
  `first_text_ms`). The page polls every 2 s, not while its tab is hidden.

Security. The listener binds loopback only. It takes `Host` only as `127.0.0.1:PORT` (or the
bound address) or `localhost:PORT`, refuses any other `Origin` or a cross-site `Sec-Fetch-Site`,
answers only GET and POST (no preflight, no CORS headers), and bounds headers (8 KiB), bodies
(512 KiB), request time (5 s) and connections (8). `/ui/v1/*` needs the launch's random secret in
`X-Starlings-UI`, compared in constant time; the page reads it from the URL fragment, which a
browser never sends, and keeps it in the tab's session storage. The API reads state, apart from
chat. Chat posts to the pipeline's own client listener, so it passes the same admission and
dispatch as any client. It acts as `--ui-client`, an enrolled client that must already hold infer
permission for the model: the coordinator issues that client a 10-minute token in memory (never
written out, renewed after 5 minutes). The UI can do what that client can do. A caller on another
machine reaches neither the listener nor the token, and the client listener still refuses browsers.

## Limits

- Two stages only, with one or two concurrent sessions according to the approved route.
- A stage must connect from the IP of its own data listener; one handshake is tried at a
  time, so a connection from that IP that speaks and then stalls holds the wait for up
  to a second.
- A request must finish inside one 60 s lease; the output maximum is set for that.
- The engine's device bytes in the heartbeat are its report at prepare, not a live sample.
- Text is detokenized from the whole answer at each token (quadratic in its length).
- A bare stage daemon needs an operator restart after engine failure. `up` supervises local
  stages and permits at most three restarts in ten minutes.
- The engine runs as a memfd, so it finds nothing beside its own path: it looks there only
  for an optional cuBLAS (a CUDA build loads no GPU library unless `GLYD_GPU_LIB` names
  one). A host that ships cuBLAS beside the engine sets `GLYD_CUBLAS` in the stage
  environment (not checked).
- Web UI: samples carry no GPU or machine names, so nodes show their ID, address and device
  ID. Worker heartbeats send no link samples yet, so a link shows "no link sample" and, after a
  request, its time outside both engines per step. Requests lists answered requests only; a
  refused or failed one shows only in the chat. No GPU use or temperature.
