```
compare.py compare-v1, 2026-10-18 09:55 EDT
Server:  http://127.0.0.1:8096, model q
Note:    v0.2 release gate, fast
GPU:     NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
Driver:  595.91.07
PCIe:    now gen 4 x16, max gen 4 x16
CPU:     AMD Ryzen 9 7950X3D 16-Core Processor
RAM:     61 GiB
OS:      Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |
|---|--:|--:|--:|--:|--:|
| Chat (short prompt) | 33 | 0.10 | 319 | 144.5 | 512 |
| ~4K prompt | 4016 | 1.39 | 2887 | 136.9 | 189 |
| ~30K prompt | 29663 | 7.33 | 4046 | 156.3 | 152 |
GSM8K     150 / 150   (0 hit the 3,072 cap, mean 162 tokens an answer)
MMLU-Pro  118 / 150   (24 hit the 3,072 cap, mean 1265 tokens an answer)
```
