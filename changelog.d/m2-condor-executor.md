- Qwen3.8-Flash-Next (Condor's model) loads on the description executor with its routed experts from a pack of codebook rows (`glyd-hbr-codes/2`: cb2, cb3,
  cb4 or ternary per expert) in the shared expert store: a layer's places on the GPU sized by the room left after the always-on matrices and each user's
  states, the pack's file mapped as their host tier, misses computed by the host. Before, the executor refused the released model (its experts do not fit a
  16 GB GPU). The fast, precise and best packs take the same sidecar and skipping on both loaders (one rule). The executor takes the experts at any lossy
  level, and `paritycheck qwen4x` holds it to Condor's own engine on the same weights before that engine is retired. Part of #140.
