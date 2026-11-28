- LFM is described: `glyd-engine intake` reads its config and its checkpoint whole (layers of three sorts: a gated short convolution, attention over
  the whole context, attention with a window; a norm after each part as well as before it; each attention head's output gated by one value). The gated
  short convolution is a new layer kind, computed by the CPU reference. It does not run yet: the executor refuses the convolution and the gate by name.
