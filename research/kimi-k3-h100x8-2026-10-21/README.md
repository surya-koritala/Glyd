# Kimi-K3, DeepSeek-V4-Flash and GLM-5.3-Flash on 8x A100 80 GB (Lambda us-east-1), 2026-10-21

8x H100 had no capacity, so it ran on 8x A100 SXM4 80 GB at $22.32/h: 1,771 GiB RAM, 240 CPUs. 16:19-19:10 EDT, about $64. Downloads
by `hf download` (Xet): Kimi-K3 1.56 TB in 587 s, DeepSeek-V4-Flash in 90 s, GLM-5.3-Flash in 171 s.

## Kimi-K3 (moonshotai/Kimi-K3 f831ab66), `glyd serve`, penguin

- Plan: 574.5 GiB on the GPUs; 290 to 815 of a layer's 896 experts in the GPU cache; all 1,347 GiB of experts in host memory.
- out/: the first serve. Load 53 min. Every request failed: `cuLaunchKernel: invalid argument`. The dense latent attention asked
  67,840 bytes of shared memory for a prompt at context 8,192. Fixed in #217; long contexts are GLY-357.
- out2/: the serve again with the fix. Ready in 36m34s.
  - chat: "The capital of France is Paris." (73 tokens with reasoning, about 2 tokens/s).
  - stream: reasoning sent chunk by chunk.
  - tools, tool_choice required, tools with a system prompt: all `get_weather {"city": "Paris", ...}` as OpenAI tool calls.
- The first load spent about half its time in kernel TLB shootdowns from freed buffers (perf). With #217's fix the reload filled the
  GPUs 1.35x faster (376 against 279 MiB/s).
- A process holding 1.35 TB pinned takes about 10 minutes to exit.

## GLM-5.3-Flash, `glyd serve` (plain FP8 on 7 GPUs, 331.7 GiB, ready in 2m38s)

- out2/: chat right. Every tools request failed: `linear: invalid argument`. The library's workspace was sized without the latent
  attention blocks' matrices; prompts past 64 rows needed more. Fixed in this branch.
- out3/: tools and tools with a system prompt return calls. Required put the call into the reasoning: the template opens `<think>`
  with no switch.
- out4/: with a forced call now closing the thinking empty, required returns `get_weather {"city": "Paris"}`.

## DeepSeek-V4-Flash, `glyd serve` (145.3 GiB, ready in 1m27s-1m46s)

- Chat right. Required returns the call. Auto tools return no call: the model writes `{"city": "Paris"}` after its text, and with
  thinking on, `{"function":"get_weather","parameters":{"city":"Paris"}}`.
- out4/: the same prompts sent raw to /v1/completions with special tokens kept (req/ds_*_prompt.txt). The reply has no DSML at all,
  so this is what the model writes for this prompt, not a parser loss. Next: check the prompt against DeepSeek's encoding.

## Downloads

The engine's own download (GLY-356), 7.04 GB slice, adding connections while it helps: 51 s, 138 MB/s. 8 fixed connections gave
18 MB/s on another Lambda box.
