- Mistral Large is described: `glyd-engine intake` reads its config and its checkpoint whole (a latent attention with a sparse indexer in every layer,
  a layer in four without positions, layers that attend to an earlier layer's latents, softmax-routed experts with a shared one, a capped head). The
  latent attention gains four forms, each computed by the CPU reference: queries scaled by the logarithm of their position, latents shared across
  layers, learned rotary frequencies, and an indexer that reads the layer's input. It does not run yet: the executor refuses each form by name.
