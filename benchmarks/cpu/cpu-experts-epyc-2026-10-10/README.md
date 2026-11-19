# CPU experts on an x86 server CPU (2026-10-10)

`tools/cards/cpu_experts_job.sh` on v030 64af6278 on a Runpod RTX 4090 pod's CPU: AMD EPYC 7642 (Zen 2, AVX2 and FMA, no AVX-512), 10 CPUs to the
container (the thread counts past 10 run no faster). The CPU half of the sparse experts (GLY-130): a Nemotron 3 Super expert computed from
the row layout, up in full, relu², down for the nonzero neurons only (the probe's distribution of neurons on: mean 91.8% off).

| measure (Int4, 10 threads, AVX2) | result |
|---|---|
| every kernel against the scalar one and float64 (13 unit tests) | ok: AVX2 the scalar kernel's bits; 3.7e-3 of a result's norm at most against the GPU's bf16-rounded weights, 1.5e-7 against the same weights unrounded |
| streaming read, 10 threads | 101-104 GB/s |
| one call of 512 distinct experts | 22.6 ms (22,666 pairs a second, 37.6 GB/s of codes and scales) |
| a decode token with 100 / 230 / 400 missed experts over the 40 layers | 10.6 / 13.3-19.5 / 24.7 ms median (a layer's call 253 / 327-466 / 610 µs) |
| 230 missed experts by threads: 1, 2, 4, 8, 16, 32 | 92.1, 52.5, 27.3, 15.6, 20.2, 17.7 ms |

How to read it: the misses of a token on the RTX 4090 run (about 284 a token, benchmarks/gpu/nh-offload-4090-2026-10-10) take about
15-20 ms on these 10 server cores, where copying them took about 70 ms a token. A desktop CPU with more cores, higher clocks or AVX-512 is
not measured here.

Files: summary.txt, steps.txt, bench4.log, bench5.log, bench6.log, threads_all.log, threads1..32.log, unit.log, cpu.log.
