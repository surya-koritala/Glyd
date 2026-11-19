# Durable Objects, celld, and where Glyd fits

Internal. Written 2026-10-15. Web research only, nothing built or spent. Competitors are named here and must stay out of public material.
Sources are mostly vendor docs plus press; secondary sources disagree in places and I mark those.

## 1. Facts: Cloudflare Durable Objects (DO)

Model
- One named object = one single-threaded, cooperatively scheduled actor. Globally unique ID (random or user-chosen), addressed from any Worker. The object is created near where it is first requested (location hint available, not in every data center).
- Each object has its own transactional SQLite storage, accessible only to it. Memory is not durable: after idle (a few seconds) it hibernates and in-memory state is lost, so anything needed later must be written to storage. WebSocket Hibernation API keeps sockets open without duration billing.
- Newer "Agents SDK" is a class on top of this: durable identity, local SQL, WebSockets, scheduling, recoverable runs. Defaults to Workers AI, can call OpenAI / Anthropic / Gemini / "any other provider".

Limits (SQLite-backed, docs page 2026-10)
- 10 GB per object (a different docs page footnote says 1 GB currently; uncertain which is live). 2 MB per row/blob/key+value. Soft 1,000 req/s per object. CPU 30 s default, up to 5 min. 6 outgoing connections per request. 32 MiB WebSocket message. 500 classes (paid). Memory limit is not stated (duration is billed on 128 MB).
- Takeaway: an object is a small, strongly consistent state box. It can hold a chat transcript or an agent plan. It cannot hold a KV cache (GBs) or model weights.

Pricing (Workers Paid, $5/mo minimum)
- Requests $0.15/M after 1 M. Duration $12.50 per M GB-s after 400k. SQLite rows read $0.001/M, written $1.00/M after 50 M. Stored data $0.20/GB-month after 5 GB. WebSocket incoming messages count 20:1 as requests. Idle hibernated objects are not billed for duration. SQLite storage billing started Jan 2026.

Cloudflare's AI pieces
- Workers AI: serverless inference, $0.011 per 1,000 neurons, 10k neurons/day free. Per-token examples: gpt-oss-120b $0.35 in / $0.75 out per M; Qwen3.8-27B $0.45 / $3.20; Llama 3.3 70B fp8 $0.293 / $2.253; GLM-5.2 $1.40 / $4.40. LoRA support exists (since 2024), reportedly limited to some base models; I found no way to bring arbitrary weights.
- AI Gateway: proxy with caching, rate limits, logs, retries, fallback across providers. Docs do not say it accepts custom/OpenAI-compatible origins (unverified).
- Containers: beta/GA-ish, billed per 10 ms, biggest instance 4 vCPU / 12 GiB / 20 GB disk. No GPU listed; the product page says "GPU support (coming soon)" (undated). Cloudflare does run GPUs, but only inside Workers AI. So today GPU = Workers AI only.
- R2: $0.015/GB-month standard, free egress, Class A $4.50/M, Class B $0.36/M.

## 2. Facts: celld (Deno / now Cloudflare)

- Open-source daemon (Apache-2.0, Rust, embeds V8) that runs Workers + Durable Objects on your own VMs. Announced 2026-08-05 by Ryan Dahl as v0.1.0 (alpha, "I would not move an important production system to it": Flavio Copes). The site now shows v0.6.2 beta. Repo ~5.6k stars, PRs disabled (patches by email), AI-generated contributions banned. HN thread ~223 points (mirror).
- Architecture: nodes share one S3/GCS/Azure bucket and nothing else. A conditional write (compare-and-swap) in the bucket grants a node a lease on a "cell" (a named object with its own SQLite). One writer per cell. Changes replicate as LTX frames to peers, then the bucket. No membership protocol, failure detector or consensus service.
- Numbers (their own, different conditions): ~4 MB per resident trivial cell, ~1,000 resident cells per 8 GB node, ~4 ms wake, ~90 ms durable regional write (single node waits on the bucket; ~25 ms with a second node holding it; ~600 ms seen to a non-local bucket), ~20 s failover, 0 acknowledged writes lost on node kill in their tests. Cost claim $49/mo vs about $4,150/mo for 1,000 always-active DOs; Cloudflare replied that hibernating objects cost far less (about $20.65/mo for a 100-object example), so treat the claim as only true for always-resident objects.
- Runs now: Workers, DO (SQLite, alarms, hibernating WebSockets), KV, Queues, D1, R2 bindings (against the fleet bucket), Workflows, Cron, static assets; Containers/Sandboxes experimental. (At 0.1.0 KV/R2/Queues/Cron were missing, so it moved fast.)
- Gaps: no public ingress/TLS/DDoS, peer traffic is plain HTTP (needs a private network), not safe for hostile multi-tenant, bucket credentials control the whole fleet. Workers AI, Vectorize, Browser Rendering listed unsupported. The README says anything needing Cloudflare's network or a GPU is out of scope. The compatibility page 404'd for me.
- GPU / local services: celld does not schedule GPUs. A Worker inside it can still `fetch()` any HTTP service reachable from the node, so a local OpenAI-compatible server on the same private network should work (my inference from "Workers fetch"; not verified in their docs).
- Corporate: Cloudflare blog 2026-10-09 (updated 10-13): Deno team joins Cloudflare; "merging workerd and celld"; Dahl and Bert Belder lead making workerd self-hosting first class; more "in the coming months". Per Dahl's post: Deno runtime supported one more year, Deno Deploy shuts after six months, celld framed around running AI agents. Implication: celld as a standalone product is likely to be folded into workerd. Don't build against celld-specific details.

## 3. Where inference meets these runtimes

- Agent / chat state fits DO well (transcripts, tool state, schedules, WebSockets). That is the part Cloudflare owns and does well; it is small data.
- KV-cache / prompt-cache persistence does not fit DO (GB-scale, 2 MB row cap). Prior art lives in the serving stacks: LMCache (CPU, local disk, remote disk, Redis), NVIDIA Dynamo KVBM (host, disk, object storage), SGLang HiCache, FlexKV. Per-session pinning and compression exist in LMCache's control API. Hosted APIs expose this as "prompt caching" billing, not as something you own.
- Weights in R2/S3: storage is cheap; example 27B bf16 ~54 GB = ~$0.81/month on R2, ~33% less after Glyd (about $0.27/month saved per model). Storage dollars are not the pain point. The pain is time to fetch + load on a cold pod and egress on non-R2 clouds (S3 internet egress is roughly $0.09/GB from memory; not checked today).
- Workers/celld apps get inference by `fetch()` to Workers AI, a third-party API, or a self-run endpoint. The last one is what Glyd can be.

## 4. Where Glyd could fit

Short version: do not clone Durable Objects. That actor layer is being given away (celld Apache, workerd merge) and we cannot use their code anyway. The gap nobody fills is the GPU and big-state side: fast cold start, exact weights, and session KV on local disk.

### (c) Inference: Glyd as the GPU backend behind Workers / celld apps -- S
- Build: nothing new in the engine. A documented recipe plus a small compatibility check that our OpenAI-compatible server works from a Worker `fetch()` (streaming SSE, tool calls, auth header, CORS), and as a custom provider behind an AI-gateway-style proxy.
- Why adopt: the user's app logic stays on Workers/celld, and the model runs on their own card or box: private data, flat cost, `exact=True` (bit-identical to bf16, which a hosted API cannot promise), long context on a 24 GB card. Hosted comparison point: Workers AI Qwen3.8-27B is $0.45 in / $3.20 out per M tokens; a self-run endpoint wins only with steady load. The honest weakness: ops, scale-out, and Linux/NVIDIA only.
- Already out there: Workers AI, any hosted API, Ollama/vLLM/llama.cpp-style local servers behind a tunnel.
- First experiment (docs + our own server only): write our own 30-line test Worker (no outside code) and run it with Cloudflare's local dev against `glyd run` on localhost:8000; check streaming + tool calls. Then read AI Gateway docs for custom-origin support.

### (a) Compression: Glyd artifacts in R2/S3 and faster cold starts -- S to M
- Build: ship the packed weights as the stored artifact (so a new pod downloads ~33% fewer bytes and skips repacking), range-readable so loading overlaps download. Plus Glyd's object-storage compressor for the other data in the bucket.
- Why pay: cold start on rented/serverless GPUs and edge boxes is download-bound. 54 GB bf16 at 1 Gbps ~7 min; ~36 GB packed ~5 min, so a 2 minute saving per cold start before overlap tricks. Egress savings if the bucket is not R2.
- Already out there (from memory, not checked today): zstd/safetensors as-is, DFloat11 (lossless bf16), fast model-streaming loaders, GPU-cloud snapshotting and model caches.
- First experiment: arithmetic is above; one measurement on the PC (no cloud spend): time load-from-NVMe of packed vs bf16 for one model, and time a download from our existing S3 bucket (already ours) at the PC's line speed. Also check whether our object-storage benchmark numbers (getglyd.com/benchmarks) can be quoted with a single headline. I could not find them on the README.

### (b) Runtime: stateful session layer for Glyd's server, self-hosted -- M (thin) / L (full DO clone)
- Build (thin): session id -> cached prefix (KV) on NVMe, compressed, evicted on idle, restored instead of re-prefilling. Optional, later: lease via a bucket CAS like celld so several boxes share sessions. Skip actors, SQLite per object, and Workers APIs.
- Why adopt: agents with long histories pay prefill every turn. Our own record: 32K prefill is 8.8 s on the PC. If restoring a saved 32K cache takes about 1 s from NVMe the user feels it in every follow-up turn. Compressed KV on disk is the Glyd angle (we already do KV compression: 1.67x more KV tokens on H100).
- Already out there: LMCache, Dynamo KVBM, SGLang HiCache, FlexKV, vendor prompt caches; Cloudflare Agents SDK for the state half.
- First experiment: on our engine, measure KV bytes for a 32K session (packed vs raw), time save to and restore from NVMe vs re-prefill. Decision rule: build only if restore is at least 5x faster than prefill and fits in under about 1 GB per session.

## 5. Risks and unknowns
- celld is moving (0.1.0 in Aug to 0.6.2 now) and is being merged into workerd; any integration doc may be obsolete in months. Its compatibility/limits pages were unreadable to me.
- GPU on Cloudflare: "coming soon" is undated; if Containers get GPUs, hosted competition for (c) rises. Workers AI may also add bring-your-own-model.
- BUSL licence (glyd-store, v0.29+): some Workers/celld users want Apache. Not new, but this audience cares.
- Linux-only, NVIDIA-only; Workers people are JS/serverless folks, not GPU folks. Adoption path may be through self-hosters and agent builders, not Cloudflare shops.
- Scope: this is a distraction risk (see "ship over nuance", Penguin focus). All three items below are small on purpose.
- Uncertain facts above: DO storage 10 GB vs 1 GB, AI Gateway custom origins, celld outbound fetch to local services, S3 egress price, and the celld cost dispute.
- Public rule: no vendor comparisons in README/site/posts; use only our own numbers.

## 6. Do next (ranked)
1. Compatibility recipe + test for "Glyd behind a Worker / celld app" (option c). 1-2 days, $0. Gives a concrete integration story and finds streaming/tool-call bugs cheaply.
2. KV session save/restore measurement on our engine (option b-thin). 2-4 days, $0 on the PC. Decides whether session persistence is worth building; fits the Penguin/long-context story.
3. Cold-start measurement for packed vs bf16 artifacts (option a). 1 day, $0 (PC + our existing bucket). Turns "33% smaller" into seconds saved. Fold the numbers into the card standard from the community-wants notes.
Do not: build a Durable Objects clone, depend on celld internals, or spend cloud money.

## Sources
- https://celld.dev
- https://github.com/denoland/celld
- https://flaviocopes.com/celld/
- https://blog.cloudflare.com/deno-joins-cloudflare/
- https://deno.com/blog/cloudflare
- https://www.theregister.com/devops/2026/08/12/nodejs-creator-liberates-durable-objects-from-cloudflare-with-celld/5286954 (via search snippet)
- https://betanews.com/article/cloudflare-has-acquired-deno-and-will-use-it-to-boost-workers/ (via search snippet)
- https://aiidelist.com/blog/deno-joins-cloudflare-runtime-sunset-astro-vite-developer-reactions (via search snippet, secondary)
- https://developers.cloudflare.com/durable-objects/platform/limits/
- https://developers.cloudflare.com/durable-objects/platform/pricing/
- https://developers.cloudflare.com/durable-objects/concepts/what-are-durable-objects/
- https://developers.cloudflare.com/agents/
- https://developers.cloudflare.com/workers-ai/platform/pricing/
- https://developers.cloudflare.com/ai-gateway/
- https://developers.cloudflare.com/containers/pricing/
- https://www.cloudflare.com/products/containers.md (via search snippet, GPU "coming soon")
- https://developers.cloudflare.com/r2/pricing/
- https://docs.nvidia.com/dynamo/dev/cli/kv-cache-offloading/overview.md
- https://arxiv.org/abs/2510.09665 (LMCache)
- https://github.com/surya-koritala/Glyd
- https://getglyd.com
