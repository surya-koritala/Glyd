# One request through two machines (2026-10-22)

GLY-188 (M4) on two Runpod pods in different data centers, on Runpod's private network (ping RTT 43 ms):

- pod C (US-WA-1, 10.0.26.59): the coordinator and the stage-0 worker daemon (layers 0..16);
- pod G (US-KS-2, 10.1.54.61): the stage-1 worker daemon (layers 16..32).

Both pods have an RTX 6000 Ada, driver 595.91.07 and Ubuntu 24.04. Qwen3.5-9B (glyd swift, 073a2132…) is at the same path on both. All three daemons were enrolled through the CLI. The chat request went over HTTP to the coordinator on C. Then the daemons were stopped, and the whole model (layers 0..32) ran the same prompt on C with the same chunking and settings.

Result: all three runs give tokens equal to the whole model's (same count, same sha256; the prompt sha256 matches too).

| run | prompt | tokens | equal | pipeline tokens sha256 | whole model sha256 | pipeline ms | pipeline ms / token, cross-datacenter | whole model ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy-48 | 21 | 48 | yes | `6a4659506278…` | `6a4659506278…` | 6908 | 143.9 | 379 |
| long-200 | 104 | 200 | yes | `49bbae4fc6ec…` | `49bbae4fc6ec…` | 27391 | 137.0 | 1583 |
| sampled-64 (T 0.7, top_p 0.9, seed 7) | 21 | 64 | yes | `a0df36acfa3b…` | `a0df36acfa3b…` | 9851 | 153.9 | 1769 |

- **Times are cross-datacenter.** Each token crosses the 43 ms RTT link three times, as a credit, the data and its receipt, in both directions. That is ~140 ms a token, against ~8 ms on one GPU. The engine itself spent 197 + 260 ms on 48 tokens. Speed is not the point of this run; correctness is.
- **Pipeline ms** runs from the first poll to both stages' reports, including prefill.
- **Client ms** (the whole HTTP request) was 8497 for greedy-48 and 11440 for the sampled run. It was not captured for long-200.
- **Stage reports:** each stage reports its own layer range. Rows (68 / 303 / 84) equal the engine's `end_session` count on both stages, the session was closed, and there were 0 refused data connections.
- **Matches the single PC:** greedy-48 and long-200 give the same token sha256 as the single-PC runs in `2026-10-22-daemon-pipeline`. That was another GPU (4080 SUPER) with another engine build (66e6bdbe dev). The sampled run differs from the PC run, but it matches the whole model on this engine build.

Builds:
- engine: glyd-engine release 8c667f40 (sha256 `eb3f4a54…`);
- GPU library: sha256 `f15088de…`;
- starlings: a0c4ce8 plus `examples/pipeline_tool`, release, test-backend (sha256 `772065f1…`). It was built on the PC; it needs GLIBC_2.39 at most.

The full reports and references are in record.json.

## Commands

`m4.sh` is the step script on both pods (`/opt/m4/m4.sh`). It runs from this Mac through `pod.sh` (the Runpod SSH proxy is PTY-only). `c.sh STEP` runs `pod.sh 0w1508oeaawpwg-64411a4a "/opt/m4/m4.sh STEP"`, and `g.sh` does the same for qwp9lan9p376y8-64410aaa.

```
c.sh c-init                                   # coordinator on 10.0.26.59:9741; stage 0 and the client enrolled
c.sh c-invite > invite.txt                    # stage-1 invitation as base64
send-invite.sh                                # to pod G in pieces; G prepares: node, fingerprint
c.sh c-approve NODE FINGERPRINT
g.sh g-join                                   # G joins over the private network
c.sh describe stage0 0 16 10.0.26.59          # approve each stage's describe (pipeline_tool)
g.sh describe stage1 16 32 10.1.54.61
c.sh c-policy MODEL_DIGEST STAGE1_NODE        # permissions, client token, plan (pipeline_tool plan), route
c.sh start coord; c.sh start stage0; g.sh start stage1
c.sh chat greedy-48 "Explain in three sentences why the sky is blue." 48 0 1 7
c.sh chat long-200 "You are reviewing a design ... reduce each." 200 0 1 7
c.sh chat sampled-64 "Explain in three sentences why the sky is blue." 64 0.7 0.9 7
g.sh stop stage1; c.sh stop stage0; c.sh stop coord
c.sh reference greedy-48 "..." 48 0 1 7       # whole model, layers 0..32, on pod C
c.sh reference long-200 "..." 200 0 1 7
c.sh reference sampled-64 "..." 64 0.7 0.9 7
```

`examples/pipeline_tool` (starlings-cli) does three things the daemon test does in Rust:
- approves a stage's `describe`;
- writes a plan's canonical `plan_digest`;
- runs the whole-model reference.

The token digests appear in `PIPELINE_REPORT` only in test-backend builds.
