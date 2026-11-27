- The consumer benchmark's llama.cpp side is measured as intended: each of llama.cpp's 4 slots has 8,192 tokens (its context is shared by its
  slots, so `-c 8192 -np 4` refused the 4K prompt), a GGUF that could not be made says why in the row's `.skip`, Gemma 4 converts (the newest
  transformers over the one llama.cpp pins), and `glyd-engine bench --against` no longer marks a result contended for the memory of the server it
  measures. Nemotron 3.5 Lightning is counted at its 32.9 billion parameters.
