# glyd-engine serve on an RTX 4080 SUPER: the probe, memory and cancel, 2026-10-19

The GLY-238 checks on the owner's PC (Ryzen 9 7950X3D, RTX 4080 SUPER 16 GB, driver 595.91.07): `glyd-engine serve` of
Qwen3.5-9B at the swift level, the snapshot of glyd/Qwen3.5-9B-swift at revision 073a2132abeaa41c119678b9687092871521eb4d
(packed by glyd 0.29.3 from Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a), `--ctx 4096`, built from commit
86530a29 (build.txt). The run held ~/glyd-scratch/gpu.lock; another job's Python process held 1,982 MiB of the GPU
throughout. `record.sh` is the whole run (19:10 Eastern); every file below is its output, as it came. A first run of the
same script three minutes earlier, without the last cancel, gave the same counters and times to the millisecond (not kept).

```
                                nvidia-smi: this     GPU used    the engine's own counters (/v1/glyd/info, bytes)
                                process (MiB)        (MiB)
before the server               -                    2223
loaded, idle                    7002                 9230        engine 6,987,473,708 = weights 6,200,345,608 + KV 134,217,728
                                                                 + state 51,511,296 + working memory 601,399,076; peak 6,987,473,708
during a streamed request       7002                 9230        the same, requests running 1
after its cancel                7002                 9230        the same, running 0
during a second one             7002                 9230        the same, running 1
after its client went away      7002                 9230        the same, running 0
after a cancel during a prompt  7002                 9230        the same, running 0
after the server stopped        -                    2223
```

- Ready 2.6 s after the start (the status file says ready and /health answers; the port is bound only then).
- Cancel by id (`POST /v1/glyd/requests/chatcmpl-rec-1/cancel`, 3 s into the reply): `cancelling` at once, `ended` (every
  sequence's end seen: the engine is done with it) 0.016 s after the cancel was asked. The stream ended with finish_reason
  stop, the usage (333 completion tokens) and `[DONE]` (cancel-stream.txt).
- A client that goes away (curl killed 3 s in): the request `ended` 0.031 s later, `cancel: "client gone"`.
- A cancel 0.2 s into a prompt of 3,790 tokens: `ended` 0.633 s later. This path reads the whole prompt before it looks at
  a cancel, and the token after the prompt goes out before the cancel is seen: the reply had that one token, then
  finish_reason stop (prompt-cancel-stream.txt).
- The process's 7,002 MiB is 338 MiB more than the engine's buffers (6,663.8 MiB): the CUDA context, the kernels' code and
  the driver's rounding, which the counters do not count (`not_counted` in info.json).
- The same model by its Hub name and level, `glyd-engine serve Qwen/Qwen3.5-9B:swift`: the same snapshot from the cache, the
  same 6,987,473,708 bytes, `level.requested: "swift"` (info-hubname.json); ready in 2.5 s. The base model's config.json
  came from the Hub; the directory run was offline (HF_HUB_OFFLINE=1).
- /v1/glyd/tokenize counted 17 tokens for the prompt that chat-json.txt's usage counts as 17.

Nothing is allocated for a request on this path: the KV for the whole context and the state are allocated at load and
reused, so a cancel has nothing to free, and the memory before, during and after is the loaded level.

| file | what |
| :--- | :--- |
| record.sh | the run |
| build.txt | the engine's and the GPU library's SHA-256, the commit, rustc, nvcc, the kernel, the CPU, the driver |
| steps.txt | the steps with their times (ready, cancel to ended, client gone to ended, a cancel during a prompt) |
| nvidia-smi.txt, nvidia-smi-full-before.txt | the GPU and its processes at each step |
| memory.jsonl | /v1/glyd/info's state, requests and memory at the same steps |
| info.json, info-hubname.json | /v1/glyd/info, served from the directory and by the Hub name |
| access.txt, models-hubname.json | what answers without the key (/health) and what does not (401), /v1/models |
| chat-json.txt, chat-length.txt, chat-sse.txt, tokenize.txt | a JSON reply (thinking off, cut at 64 tokens), one cut at 8, a streamed one (thinking, then the answer, stop, usage, [DONE]), the count |
| errors.txt | logprobs at this level, max_tokens past the context, an id no request has, a cancel by GET |
| cancel-stream.txt, status-1-*.json | the stream cancelled by id, its status during, at the cancel and ended |
| disconnect-stream.txt, status-2-ended.json | the stream whose client went away, its status ended |
| tokenize-long.txt, prompt-cancel-stream.txt, status-3-*.json | the 3,790-token prompt, its stream, its status at the cancel and ended |
| server-*.err, status-dir, status-hubname, stats.json | the server's own lines, its status files, /stats (the scheduler's: unused on this path) |

Known limits

- One run on one shared box; the times are wall clock, the status polled every 10 ms.
- The development build (the default features). A release build answers the same but `build.development: false`; not run here.
- `glyd serve MODEL:level` through the Python front end was checked by its tests (test_onboard.py), not on this GPU.
