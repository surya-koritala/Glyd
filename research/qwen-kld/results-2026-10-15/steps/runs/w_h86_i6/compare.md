```
compare.py compare-v1, 2026-10-15 16:44 EDT
Server:  http://127.0.0.1:8093, model q
Note:    (none given)
GPU:     NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
Driver:  595.91.07
PCIe:    now gen 4 x16, max gen 4 x16
CPU:     AMD Ryzen 9 7950X3D 16-Core Processor
RAM:     61 GiB
OS:      Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |
|---|--:|--:|--:|--:|--:|
| Chat (short prompt) | 33 | 0.23 | 143 | 104.6 | 512 |
| ~4K prompt | 4016 | 1.85 | 2176 | 102.6 | 233 |
| ~30K prompt | 29663 | 7.48 | 3964 | 111.4 | 222 |
Quality: not run (--speed-only)
```
