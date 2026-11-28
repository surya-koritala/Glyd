- Mistral Large's layers are on the engine's executor: the latent attention with its queries as one product scaled by the logarithm of their position,
  a rotary of YaRN's frequencies times the model's learned factors, layers with no positions beside layers with them, layers that attend to an earlier
  layer's latents with their own expansion, and an indexer that reads the layer's input through a LayerNorm and scales its queries by position; beside
  them the dense MLP, softmax-routed experts with a shared MLP, and the head capped at 30 tanh(x / 30). A small random model of the layout is a case of
  `kindcheck` and of the logits checks, held to the CPU reference.
- Mistral Large does not run from its release yet: the family is in no row of what `fit`, `run` and `serve` take. Its release's FP8 matrices keep their
  scales under a name the executor's loaders do not read, its draft model and its vision side are unread, and a layer that attends to another layer's
  latents is refused by name where those are held in 12-bit pages. A model split across GPUs keeps such a layer with the layer it reads.
