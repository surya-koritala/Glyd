- Ling's layers are on the engine's executor: the Kimi delta attention on plain residuals (each sequence's states counted in the plan), the latent
  attention with its queries as one product and its output gated by one value a head, experts chosen inside the best groups, and the SwiGLU limit on
  the activated gate a layer, the shared MLP's another than its experts'. A small random model of the layout is a case of `kindcheck` and of the logits
  checks, held to the CPU reference.
- Ling does not run from its release yet: the family is in no row of what `fit`, `run` and `serve` take. Its draft head is refused by name, and
  neither its FP8 matrices nor its experts in the computer's memory have been loaded on a GPU.
