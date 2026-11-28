```
compare.py compare-v1, 2026-10-21 00:21 EDT
Server:  http://127.0.0.1:8099, model qwen3.8-flash-next
Note:    best
GPU:     NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
Driver:  595.91.07
PCIe:    now gen 4 x16, max gen 4 x16
CPU:     AMD Ryzen 9 7950X3D 16-Core Processor
RAM:     61 GiB
OS:      Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |
|---|--:|--:|--:|--:|--:|
| Chat (short prompt) | 33 | 3.16 | 10 | 26.5 | 512 |
| ~4K prompt | 4016 | 14.37 | 279 | 40.6 | 238 |
| ~30K prompt | 29663 | 7.97 | 3724 | 75.2 | 216 |
GSM8K     147 / 150   (0 hit the 3,072 cap, mean 177 tokens an answer)
MMLU-Pro  131 / 150   (14 hit the 3,072 cap, mean 1131 tokens an answer)
```
