- Qwen3.8-Flash-Next (Condor's model) loads on the description executor with its routed experts from a pack of codebook rows (`glyd-hbr-codes/2`: cb2, cb3,
  cb4 or ternary per expert) in the shared expert store: a layer's places on the GPU sized by the room left after the always-on matrices and each user's
  states, the pack's file mapped as their host tier, misses computed by the host. Before, the executor refused the released model (its experts do not fit a
  16 GB GPU). The fast, precise and best packs take the same sidecar and skipping on both loaders (one rule). The executor takes the experts at any
  level, lossless at penguin. Part of #140.
- `paritycheck` and `logitscheck` hold the family's executor to the CPU reference of the layer kinds, not to Condor's own engine: the two run different
  kernels of the delta rule, so their bits part there. The family's two random cases are held at the bound the Qwen3.5 layout's fixtures hold (KL mean
  1e-4, max 2.5e-4); the 5e-4 the first case's fixture carried was fitted to a run of the executor and is withdrawn. A router's choice taken the other
  way is no longer covered by a wider bound: wherever the executor's choice is not the reference's own, the reference's router run on the executor's
  own input must make the same choice or one a near tie from it, and the logits are held to the reference run again on the executor's choices; the
  figures against the reference's own choices are printed beside them. At penguin the checks also read the experts back from the GPU and hold them
  to the checkpoint's bits, and the second random case's experts are no longer on the six-bit grid, so a lossy load under the lossless level's name
  fails. Part of #140.
