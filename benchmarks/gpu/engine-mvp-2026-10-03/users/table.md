| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| glyd-engine | 1 | engineC | 8 | 21.2 | 21.8 | 372, 379 | 45.9, 46.1 | 0.00 |
| glyd-engine | 1 | engineD | 8 | 21.1 | 21.7 | 377, 389 | 46.0, 46.2 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglydA | 8 | 20.6 | 21.2 | 419, 430 | 47.2, 47.3 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglydB | 8 | 20.7 | 21.3 | 407, 425 | 47.0, 47.2 | 0.00 |
| glyd-engine | 4 | engineC | 24 | 73.7 | 19.7 | 953, 1526 | 50.7, 53.2 | 0.00 |
| glyd-engine | 4 | engineD | 24 | 73.7 | 19.7 | 961, 1569 | 50.7, 52.8 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglydA | 24 | 71.9 | 19.4 | 1075, 1630 | 51.6, 54.3 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglydB | 24 | 71.8 | 19.4 | 1082, 1668 | 51.6, 54.3 | 0.00 |
| glyd-engine | 8 | engineC | 32 | 125.5 | 17.0 | 1322, 3362 | 58.7, 62.2 | 0.00 |
| glyd-engine | 8 | engineD | 32 | 124.2 | 16.9 | 1335, 3233 | 59.2, 63.1 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglydA | 32 | 125.8 | 17.6 | 1730, 3020 | 56.9, 61.6 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglydB | 32 | 127.2 | 17.7 | 1669, 2877 | 56.5, 61.3 | 0.00 |

| server | users | runs | output tok/s (mean) | per user tok/s (mean) | time to first token ms (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| glyd-engine | 1 | 2 | 21.2 | 21.8 | 374 |
| vLLM with Glyd (weights and KV cache) | 1 | 2 | 20.6 | 21.2 | 413 |
| glyd-engine | 4 | 2 | 73.7 | 19.7 | 957 |
| vLLM with Glyd (weights and KV cache) | 4 | 2 | 71.8 | 19.4 | 1078 |
| glyd-engine | 8 | 2 | 124.9 | 17.0 | 1329 |
| vLLM with Glyd (weights and KV cache) | 8 | 2 | 126.5 | 17.6 | 1700 |

The engine against vLLM with Glyd, the means of the final runs:

- 1 user: 21.2 tokens a second against 20.6 (+2.6%; the runs of each within 0.3% and 0.5% of each other), the first token after 374 ms against 413 (-9%)
- 4 users: 73.7 tokens a second against 71.8 (+2.6%; the runs of each within 0.0% and 0.1% of each other), the first token after 957 ms against 1078 (-11%)
- 8 users: 124.9 tokens a second against 126.5 (-1.3%; the runs of each within 1.0% and 1.1% of each other), the first token after 1329 ms against 1700 (-22%)

The build between (the engine thread's work after each step moved to the GPU, a request's first token still sent after its round's decode step; one run, the vLLM run beside it is kvglydA):

| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| glyd-engine (sampling on the GPU, first token not yet sent early) | 1 | engineA | 8 | 21.1 | 21.8 | 421, 435 | 45.8, 46.1 | 0.00 |
| glyd-engine (sampling on the GPU, first token not yet sent early) | 4 | engineA | 24 | 72.3 | 19.5 | 1043, 1620 | 51.4, 53.9 | 0.00 |
| glyd-engine (sampling on the GPU, first token not yet sent early) | 8 | engineA | 32 | 122.5 | 16.7 | 1411, 3459 | 59.9, 63.4 | 0.00 |

The first measurement, before the engine thread's work after each step was cut (same client, same prompts):

| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| glyd-engine (first measurement) | 1 | engine1 | 8 | 20.5 | 21.1 | 376, 386 | 47.5, 47.8 | 0.00 |
| glyd-engine (first measurement) | 1 | engine2 | 8 | 20.4 | 21.0 | 384, 393 | 47.6, 47.8 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglyd1 | 8 | 20.6 | 21.3 | 411, 422 | 47.0, 47.2 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglyd2 | 8 | 20.6 | 21.2 | 417, 432 | 47.2, 47.3 | 0.00 |
| glyd-engine (first measurement) | 4 | engine1 | 24 | 65.1 | 17.3 | 997, 1577 | 57.7, 60.0 | 0.00 |
| glyd-engine (first measurement) | 4 | engine2 | 24 | 64.9 | 17.3 | 1001, 1584 | 57.9, 60.2 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglyd1 | 24 | 72.1 | 19.4 | 1059, 1604 | 51.5, 54.0 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglyd2 | 24 | 71.8 | 19.4 | 1073, 1624 | 51.6, 54.3 | 0.00 |
| glyd-engine (first measurement) | 8 | engine1 | 32 | 102.6 | 13.7 | 1362, 3380 | 72.7, 76.1 | 0.00 |
| glyd-engine (first measurement) | 8 | engine2 | 32 | 102.5 | 13.7 | 1359, 3381 | 72.9, 76.3 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglyd1 | 32 | 126.1 | 17.6 | 1719, 2920 | 56.8, 61.8 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglyd2 | 32 | 125.7 | 17.6 | 1734, 3073 | 56.9, 61.8 | 0.00 |

| server | users | runs | output tok/s (mean) | per user tok/s (mean) | time to first token ms (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| glyd-engine (first measurement) | 1 | 2 | 20.5 | 21.0 | 380 |
| vLLM with Glyd (weights and KV cache) | 1 | 2 | 20.6 | 21.2 | 414 |
| glyd-engine (first measurement) | 4 | 2 | 65.0 | 17.3 | 999 |
| vLLM with Glyd (weights and KV cache) | 4 | 2 | 72.0 | 19.4 | 1066 |
| glyd-engine (first measurement) | 8 | 2 | 102.5 | 13.7 | 1360 |
| vLLM with Glyd (weights and KV cache) | 8 | 2 | 125.9 | 17.6 | 1727 |
