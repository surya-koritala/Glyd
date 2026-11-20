# Predictions, written before the run (committed 21:36:18 EDT, a202c84d; the launch was at 21:36:47)

Kimi-K3 on 8x A100 80 GB, every layer across the 8 (#229), the expert cache planned on the GPU (#237, GLYD_KIMI_DEVICE=1), the low-latency
gathers (#230), after the fusions of #239 and #241. Branch a100-run: main at c5ca3d39 with #230, #241 and #242 merged in.

The formula, `glyd-engine ceiling moonshotai/Kimi-K3 --box a100x8 --hit 0.85 --bf16-pct 75.2 --t0-us 12.7 --t0-moe-us 12.7 --fixed-us 11800
--exchange-us E`:

| | predicted | how it is checked |
|---|---|---|
| decode, LL gathers (the default) | 70.2 ms a token, 14.2 tokens/s (56-84 ms at ±20%) | the 512-token story's time |
| decode, regular gathers | 80.5 ms a token | not run (one load); tpgather's two kinds |
| exchanges a token | 699 (882 before the fusions) | |
| a small gather at 8 GPUs (1 x 896) | LL ~5 µs (not measured: the PC has no NVLink), regular 19.5 µs (Oct 22) | tpgather |
| found share, device path | 0.85 assumed (host-path runs Oct 21-22: 0.81-0.88) | GLYD_KIMI_STATS at the next prompt |
| tpcheck, 2 GPUs (toy): LL, regular, device path in copy mode | bit for bit | |
| answers | chat: Paris; pass key: 48213; tools: get_weather for Paris | |
| ready | ~35 min | |

Terms at our rates (LL): weights and product starts 22.7 ms, missing experts 31.9, exchanges 3.4, small kernels 11.8. Ceiling at the box's
rates: 18.3 ms.

Not measured on an A100, taken from the 4080 times 2.3 (DeepSeek-V4-Flash's A100 fit to the 4080): a product's start (5.5 µs there, 12.7 here)
and the small kernels (55 µs a layer there; 11.8 ms a token here). The run checks them together, as the step minus the other terms.

The routing of the pass key's 2,472-token prompt is traced (GLYD_KIMI_TRACE) for `dcreplay`: CLOCK against the ideal cache at the run's places.
