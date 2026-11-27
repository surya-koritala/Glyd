- Three more families run: Domyn Edge, Granite Swash (dense) and Mellum 2. `glyd fit`, `run` and `serve` take them on one GPU: Domyn Edge and Granite Swash
  at penguin, kestrel, swift and wren; Mellum 2 at kestrel, swift and wren (its experts are not held lossless on one GPU yet).
- The engine's dense layer takes more of what recent families are built of: a norm after each block as well as before it, queries normed with no weight,
  a learned sink a head in the attention (in the attention kernels), a scaled embedding, residual and logits, a mixture of experts as a layer's
  feed-forward, and a rotary scaled the same at every length (yarn, linear, Llama 3's). A family built of these needs a description and nothing else:
  it is sized from its description and loaded by the dense loader.
