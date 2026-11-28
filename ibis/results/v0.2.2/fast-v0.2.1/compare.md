```
compare.py compare-v1, 2026-10-19 23:43 EDT
Server:  http://127.0.0.1:8096, model q
Note:    v0.2.1 fast experts, control
GPU:     NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
Driver:  595.91.07
PCIe:    now gen 4 x16, max gen 4 x16
CPU:     AMD Ryzen 9 7950X3D 16-Core Processor
RAM:     61 GiB
OS:      Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |
|---|--:|--:|--:|--:|--:|
| Chat (short prompt) | 33 | 0.11 | 297 | 120.3 | 512 |
| ~4K prompt | 4016 | 1.40 | 2860 | 117.3 | 189 |
| ~30K prompt | 29663 | 7.36 | 4033 | 140.8 | 178 |
GSM8K     147 / 150   (0 hit the 3,072 cap, mean 157 tokens an answer)
MMLU-Pro  124 / 150   (18 hit the 3,072 cap, mean 1202 tokens an answer)
```
