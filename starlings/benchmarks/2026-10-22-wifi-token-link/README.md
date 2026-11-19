# Token link, Mac to PC over Wi-Fi (2026-10-22)

The GLY-241 token-link profile between two enrolled Starlings nodes. The worker is an M1 Max Mac on Wi-Fi
(5 GHz, channel 44, -49 dBm). The coordinator is the Ubuntu PC on Wi-Fi (6 GHz, 160 MHz, -37 dBm,
`power_save` on; its Ethernet port is unplugged). The traffic is enrolled mTLS through the home router.
Each sample is 16 KiB out and 64 B back (one token's boundary is 16 KiB for Qwen3.5-9B).
`run.sh` restarts both daemons, approves the profile, runs it and saves the record.

| profile | p50 ms | p90 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- |
| every 20 ms, start to start, no keepalive | 8.59 | 35.85 | 354.95 | 508.22 | 28.35 |
| 100 ms idle after each reply, no keepalive | 99.24 | 136.55 | 174.60 | 205.37 | 87.13 |
| 100 ms idle after each reply, 5 ms keepalive | 32.28 | 109.51 | 198.72 | 203.05 | 46.26 |

Against the owner baseline (2026-10-12, Python ping-pong over Wi-Fi):
- Without keepalive, a 100 ms idle gap gives ~98 ms median. Reproduced: 99.2 ms.
- With a 5 ms keepalive the baseline was 7.8 ms. Not reproduced: 32.3 ms here. The difference we can see is
  that the PC is now on 6 GHz with power save on, and the keepalive only flows Mac to PC.

What it means for a split model: at 8.6 ms median per hop, and hundreds of ms at the tail, Wi-Fi costs about
as much as the 9B model's whole decode step (~10 ms on the 4080 SUPER). Two-machine runs need a cable.
These are application round trips on the real enrolled stack, not raw link speed or a model benchmark.
