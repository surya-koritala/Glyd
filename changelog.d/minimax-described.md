- MiniMax is described: `glyd-engine intake` reads its config and its checkpoint whole but the vision tower and its projectors (layers of a
  grouped-query attention with a sparse indexer, three dense MLPs then sigmoid-routed experts beside a shared expert). Two forms are new to the
  layer kinds and computed by the CPU reference: an indexer on grouped-query attention that keeps blocks of positions by their best key, a choice a
  key head, and the SwiGLU whose gate's sigmoid has a gain and whose up is shifted by 1. Its stored formats are named, FP8 with a scale a run of 32
  and 4-bit experts with FP8 scales a run of 16 under one scale a matrix; no loader reads them yet. It does not run yet: the executor refuses the
  indexer and the activation by name.
