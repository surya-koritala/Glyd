```
compare.py compare-v1, 2026-10-18 13:59 EDT
Server:  http://127.0.0.1:8096, model q
Note:    v0.2.1 release gate, best
GPU:     NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
Driver:  595.91.07
PCIe:    now gen 4 x16, max gen 4 x16
CPU:     AMD Ryzen 9 7950X3D 16-Core Processor
RAM:     61 GiB
OS:      Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |
|---|--:|--:|--:|--:|--:|
| Chat (short prompt) | 33 | 0.22 | 150 | 104.5 | 512 |
| ~4K prompt | 4016 | 2.19 | 1832 | 91.6 | 207 |
| ~30K prompt | 29663 | 7.48 | 3967 | 97.0 | 222 |
GSM8K     149 / 150   (0 hit the 3,072 cap, mean 175 tokens an answer)
MMLU-Pro  128 / 150   (15 hit the 3,072 cap, mean 1086 tokens an answer)
```
