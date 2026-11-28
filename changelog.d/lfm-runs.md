- LFM runs. `glyd fit`, `run` and `serve` take the family on one GPU at penguin, kestrel, linnet and wren: its short convolution layers and its attention
  gated a head are on the engine's executor, in kernels of their own, and each sequence's convolution state is counted in the plan.
- A layer of the engine's dense kind may now be a gated short convolution in the attention's place; a convolution with biases is refused by name.
