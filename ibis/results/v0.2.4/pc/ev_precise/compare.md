```
compare.py compare-v1, 2026-10-20 23:29 EDT
Server:  http://127.0.0.1:8099, model qwen3.8-flash-next
Note:    precise
GPU:     NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
Driver:  595.91.07
PCIe:    now gen 4 x16, max gen 4 x16
CPU:     AMD Ryzen 9 7950X3D 16-Core Processor
RAM:     61 GiB
OS:      Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |
|---|--:|--:|--:|--:|--:|
| Chat (short prompt) | 33 | 2.39 | 14 | 32.9 | 512 |
| ~4K prompt | 4016 | 6.48 | 620 | 72.4 | 180 |
| ~30K prompt | 29663 | 7.36 | 4030 | 85.0 | 196 |
GSM8K     148 / 150   (0 hit the 3,072 cap, mean 179 tokens an answer)
MMLU-Pro  127 / 150   (13 hit the 3,072 cap, mean 1101 tokens an answer)
```
