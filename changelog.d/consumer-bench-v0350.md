- The first consumer benchmark of a release against llama.cpp is in the tree: v0.35.0 on an RTX 4080 SUPER (`benchmarks/consumer/v0.35.0/rtx4080s`),
  every side's command lines and logs beside the page. Decode, one user: Gemma 4 12B 65.6 tokens/s at kestrel against llama.cpp's 61.7 at Q6_K, and
  90.5 at wren against 82.1 at Q4_0 of Google's QAT checkpoint; Nemotron 3.5 Lightning 126.1 at wren against 85.1 at Q4_K_M (a file of 6.18 bits a
  weight against wren's 4.53). llama.cpp answers the first token of a 4K prompt sooner on Gemma at four bits (1.00 s against 1.09) and uses less
  GPU memory on Gemma (9.8 against 11.8 GB at four bits).
