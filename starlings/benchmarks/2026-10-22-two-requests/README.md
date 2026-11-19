# Two requests at once (2026-10-22)

GLY-190 on the real engine: each stage daemon serves two requests at once, in two engine sessions
(glyd-gpu-private#234's `--sessions 2`) on one shared runtime per stage. Qwen3.5-9B swift, layers 0..16
and 16..32 on one RTX 4080 SUPER, drafts 7.

- The greedy-48 request gives the whole model's tokens.
- Two prompts sent at once give the same answers as the same prompts sent one after the other.
- Time: 836 ms for both at once, 861 ms one after the other. Both stages share one GPU here,
  so there is little to overlap; the gain needs two GPUs (GLY-365).

Engine: range-chat-messages build (main with #232 and #234, plus #236), GPU library from the same
`gpu/` source. Command: as benchmarks/2026-10-22-drafted-decode, with `STARLINGS_RANGE_SESSIONS=2`.
Record: `record-greedy-48-drafts-7-sessions-2.json` (`two_at_once`).
