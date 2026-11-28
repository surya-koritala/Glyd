# Layer ranges against the whole model on an RTX 4080 SUPER: penguin, linnet, kestrel, 2026-10-20

The GLY-235 proof on the owner's PC (Ryzen 9 7950X3D, RTX 4080 SUPER 16 GB, driver 595.91.07, CUDA 13.2): `glyd-engine rangecheck`
of Qwen3.5-9B at three levels, a prompt of 1,024 random tokens and 256 greedy tokens, the whole model against two stages of
layers (0..16 and 16..32) chained in one process, then against the same two stages as `glyd-engine rangeworker` processes chained
through the glyd-range/1 protocol (docs/range-worker.md); every step's logits (the bf16 words of all 248,320) and token compared
bit for bit. Linnet once more in three stages (0..11, 11..23, 23..32). Built from commit dce84eba (build.txt: the binary's
and the library's SHA-256). `record.sh` is the whole run (00:00 to 00:03 Eastern, 2026-10-20), under ~/glyd-scratch/gpu.lock;
every file here is its output, as it came. No other process held the GPU during this run (232 MiB used before and after).
A first run of the same script 8 minutes earlier (the binary of commit 79b01c9, SHA-256 a34d0eab..., whose cancel probe
sent its chunk at a position the context had no room for) gave the same four "all checks passed" and the same digests, with
another process holding 1.9-2.2 GB of the GPU; not kept.

| level | snapshot (glyd/Qwen3.5-9B-LEVEL) | weights on the GPU | whole model | two stages, one process | two worker processes | sha256 (tokens, 256 x 248,320 logits) |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| penguin (lossless, tiered `mma`) | 5985564d007e847509609c23d7647a8688867889 (glyd-v4, glyd 0.29.4) | 12,050,620,953 B | prompt 227.6 ms, step 17.55 ms | 230.1 ms, 17.56 ms; equal | 258.6 ms, 17.98 ms; equal | f60090ec21ab223ee29b89abe1d6381cbe6c32e473c4db1b45f96d13eefd77b3 |
| linnet (i5) | 073a2132abeaa41c119678b9687092871521eb4d (glyd-v7, glyd 0.29.3) | 6,200,345,608 B | 221.0 ms, 9.73 ms | 218.1 ms, 9.85 ms; equal | 252.2 ms, 10.27 ms; equal | 51d48afee8695e0668c64b551dcb5fc26227d231823a12ce32f475e2d09d53aa |
| kestrel (i6) | 4d7e583a7cce212734140927824f91ad9ad11cf7 (glyd-v7) | 7,318,651,912 B | 222.3 ms, 11.13 ms | 219.1 ms, 11.25 ms; equal | 254.3 ms, 11.60 ms; equal | 895a76df93f1156b3e42ddbe93d12c5b9ec0ec28bedb1d202b9d8d38c6d04463 |
| linnet, three stages | as above | | 218.9 ms, 9.73 ms | 219.8 ms, 9.89 ms; equal | 279.7 ms, 10.44 ms; equal (3 processes) | 51d48afee8695e0668c64b551dcb5fc26227d231823a12ce32f475e2d09d53aa |

"equal": every logit the same, bit for bit, 256 steps, 63,569,920 logits; every token the same (the first token after the
prompt and 255 steps). A step's time here includes reading the 496,640-byte logits row each step (and, for the workers, the
frames through the pipes); it is not the decode speed (bench, below). The prompt is `report::random_prompt(1024, vocab)`, the
bench's generator; the greedy tokens of such a prompt repeat (penguin: 220, 220, 17, 15, 16, 24, ...; linnet: 248046, 198, ...),
which changes nothing about the comparison.

The boundary carried at each cut: the residual X and the MLP output D of the layer before the cut, bf16 [rows, 4096] each,
16,384 bytes a token (`boundary: [(X, 4096), (D, 4096)]` in each output). Positions are explicit (`position_start`); the KV and
the linear-attention states stay in their stage.

## Memory

From the engine's own counters (bytes it asked the driver for) and nvidia-smi (every second, smi-during-*.txt):

| | penguin | linnet | kestrel |
| :--- | ---: | ---: | ---: |
| whole model: weights | 12,050,620,953 | 6,200,345,608 | 7,318,651,912 |
| = layers + embedding + lm_head + small tensors | 9,319,363,022 + 1,366,276,916 + 1,362,850,583 + 2,130,432 | 4,791,730,688 + 703,242,244 + 703,242,244 + 2,130,432 | 5,655,757,312 + 830,382,084 + 830,382,084 + 2,130,432 |
| KV (1,296 positions), states, scratch | 42,467,328 / 51,511,296 / 416,096,036 | the same | the same |
| engine bytes, whole model | 12,560,695,613 | 6,710,420,268 | 7,828,726,572 |
| nvidia-smi, the whole model's process | 12,776 MiB | 7,146 MiB | 8,202 MiB |
| worker 0..16: weights / kv / state / scratch | 6,028,596,695 / 21,233,664 / 25,755,648 / 416,096,036 | 3,100,168,760 / 21,233,664 / 25,755,648 / 416,096,036 | 3,659,321,912 / 21,233,664 / 25,755,648 / 416,096,036 |
| worker 16..32: weights / kv / state / scratch | 6,022,024,358 / the same | 3,100,176,948 / the same | 3,659,330,100 / the same |
| nvidia-smi, each worker process | 6,510 MiB | 3,694 MiB | 4,222 MiB |
| nvidia-smi, the GPU's peak during the run | 13,529 MiB | 7,897 MiB | 8,953 MiB |

- A linear-attention layer (24 of 32): weights 294,333,082 (penguin; the tiered packs differ a little by layer, the table in
  rangecheck-penguin.txt has all 32), 151,322,640 (linnet), 178,585,616 (kestrel) bytes; no KV; state 2,146,304 bytes a
  sequence (the convolution's 49,152 bf16 + the delta rule's 2,097,152 fp32). A full-attention layer (8): 282,377,030 /
  144,998,416 / 171,212,816 bytes; KV 4,096 bytes a token (4 heads x 256 x 2 x bf16); no state. The whole model: 32,768 bytes
  of KV a token, 51,511,296 bytes of state a sequence. The small tensors (the norms, the linear attention's parameters):
  2,130,432 bytes in all.
- Scratch is the same for a stage and for the whole model (416,096,036 bytes: a 1,024-token chunk's activations, the products'
  workspaces, the logits, the rotary tables): a stage pays it whole. A worker process is its engine bytes plus about 300 MiB
  (linnet: 3,694 MiB in nvidia-smi against 3,398 MiB of buffers): the CUDA context, the kernels' code, the driver's rounding.
- Penguin's two workers plus the parent's leftover context: 13,529 MiB at the peak here. In the earlier run with another
  process holding 1.9-2.2 GB they fit with 529,137,664 bytes free after the second `prepare` (rangecheck-penguin.txt of that
  run, not kept; this run's: 2,531,262,464 free).
- `unload` leaves 0 engine bytes; the GPU is back at 232 MiB after each run (nvidia-smi.txt).

## The protocol's checks (every run)

On the last worker: the last step's `run_chunk` sent again with the same id came back from the kept reply (position_end
1279, the 496,640-byte logits), the session still at 1,279 tokens, not run again; the same id with another position_start
refused ("differs from this one"); a new request at position 1278 refused ("position_start 1278, where the session holds
1279"); a session handle of another incarnation refused ("another worker incarnation"); an unknown op refused. On the first
worker, in a fresh session: a 1,024-token prefill chunk sent, 20 ms later `status`: answered first, at 20.1 ms, `running`
with the chunk's id and 20.0 ms; the chunk's reply at 124.7-128.6 ms (91.5 ms for 11 layers). Then another fresh session, a
chunk, 20 ms later `cancel`: answered `cancelling` (the chunk under way), the chunk's reply an error ("cancelled during this
chunk; its output is withdrawn"), both within 77-114 ms; `status` then `cancelled`; a chunk after it refused; `end_session`
said `cancelled: true`; `unload` 0 bytes retained; every worker exited 0.

## The whole-model path, unchanged

`glyd-engine bench` of the linnet snapshot (Engine35, `--prompt 1024 --new 256`), the binary built from main (85529126) and this
one, three times each, interleaved (bench-before.txt, bench-after.txt):

```
before: prefill 217.0, 216.9, 216.9 ms (medians of 5)   decode mean 9.353, 9.353, 9.353 ms   106.92 tok/s   7.057 GiB used
after:  prefill 217.1, 216.6, 217.0 ms                  decode mean 9.354, 9.353, 9.353 ms   106.91-106.92  7.057 GiB used
```

`glyd-engine generate` of the same prompt, 64 greedy tokens, before and after: the same text (generate-before.txt,
generate-after.txt, `cmp`). `cargo test --release -p glyd-engine --lib`: 400 passed (pipeline log, not kept; one earlier run of
the same suite had server::tests::a_request_by_its_id_its_cancel_and_its_end fail once at a 404 on a status poll, under
`-j 8`; it passed in the rerun and three times alone).

## Files

| file | what |
| :--- | :--- |
| record.sh | the run |
| build.txt | the commit, the SHA-256 of the engine (this build and the one from main), the library, rustc, nvcc, the GPU, the CPU, the kernel, the snapshots |
| steps.txt | each step with its time and its exit status |
| rangecheck-penguin.txt, -linnet.txt, -kestrel.txt, -linnet-3.txt | each check's output: the whole model's sizes by layer, the three runs' times and digests, the equality lines, every worker's describe, inspect, prepare, begin_session, the probes, end_session, unload |
| rangecheck-*.err | the engines' own lines (the workers' start lines); nothing else |
| smi-during-*.txt | nvidia-smi every second during each check: the GPU's used memory and each process's |
| nvidia-smi.txt, nvidia-smi-full-before.txt, nvidia-smi-full-after.txt | the GPU before and after each check, and in full at the start and the end |
| bench-before.txt, bench-after.txt (.err: the load lines) | the whole-model bench, three times each |
| generate-before.txt, generate-after.txt | the same 64 tokens before and after |

## Known limits

- One GPU, one machine: the stages hand their boundary through pipes on the same box; nothing about a network is measured.
- One run of each check kept; the earlier run gave the same digests.
- The workers' step time (10.3-18.0 ms) is the chain through pipes with the logits read each step, not a speed the worker is
  tuned for.
- The development build; `worker` and `rangecheck` are development commands.
