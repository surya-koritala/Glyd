- Hunyuan is described: `glyd-engine intake` reads every key of its config (the text model's and the drafter's), holds each to the form the family
  computes, and says its layers (latent attention with a compressed indexer and in a window, a dense feed-forward and experts in a latent, residual
  streams, an n-gram memory); the released checkpoint's text model is read whole. The layer kinds say what its layers add, each with its CPU reference:
  a window on the latent attention that is all a query reads, residual streams that do not mix with each other, an n-gram memory whose compressed
  vocabulary is in the checkpoint and whose rows and value are normed, and an indexer that pools each closed group of tokens into one key and always
  keeps the query's own group. The drafter and the vision tower are parts the description leaves, by name: the text model runs without them, and
  drafting is not described. The family is described, not run: the executor refuses each of these parts by its name.
