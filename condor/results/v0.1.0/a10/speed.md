# Qwen3.8-Flash-Next on Glyd: speed on an NVIDIA A10

A clean Lambda instance, 2026-10-14: NVIDIA A10 24 GB (Ampere), Ubuntu 22.04, driver 595.91; precise mode (the default), context 32,768.
`git clone` + `./setup.sh --yes` from the public repositories; tests/test_server.py 9 of 9; the model loaded in 52 s.
The numbers below are tools/bench.sh's lines from the run's output (the report file was not kept).

| Test | Reading the prompt (s) | Writing (tokens/s) |
|---|--:|--:|
| Chat (short prompt) | | 101.1 |
| 4K prompt | 2.3 | 62.4 |
| 16K prompt | 6.6 | 74.4 |
| 32K prompt | 11.6 | 76.9 |
