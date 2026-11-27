server: /cache/llama.cpp-build/llama.cpp/build/bin/llama-server -m /cache/gguf/gemma-4-12B-it-Q6_K.gguf --fit on -fa on -c 32768 -np 4 --jinja --host 127.0.0.1 --port 8091
llama.cpp: 0b51bb2fed322bc884f96f98db340e2b0f6ea176
gpu: NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 595.99.02
glyd-engine: glyd-engine 0.35.0 
date: 2026-11-27T19:50:31Z
bench: /cache/target-sm89/release/glyd-engine bench --against http://127.0.0.1:8091 --engine llama.cpp --bits 6.52 --label llama.cpp Q6_K --params 12 --source google/gemma-4-12B-it --loads short:1,short:4,4k:1 --runs 3 --max-tokens 256
