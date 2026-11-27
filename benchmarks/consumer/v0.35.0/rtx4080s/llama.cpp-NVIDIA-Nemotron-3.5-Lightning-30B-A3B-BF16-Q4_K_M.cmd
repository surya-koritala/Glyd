server: /cache/llama.cpp-build/llama.cpp/build/bin/llama-server -m /cache/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-Q4_K_M.gguf --fit on -fa on -c 32768 -np 4 --jinja --host 127.0.0.1 --port 8091
llama.cpp: 0b51bb2fed322bc884f96f98db340e2b0f6ea176
gpu: NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 595.99.02
glyd-engine: glyd-engine 0.35.0 
date: 2026-11-27T19:59:32Z
bench: /cache/target-sm89/release/glyd-engine bench --against http://127.0.0.1:8091 --engine llama.cpp --bits 6.18 --label llama.cpp Q4_K_M --params 32.9 --source nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --loads short:1,short:4,4k:1 --runs 3 --max-tokens 256
