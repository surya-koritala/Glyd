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
- Where the engine computes experts on the host (an expert cache whose mode gives the host its misses: cpu, split or auto, and the codebook cache),
  `usercheck` with prompts one a pass, `keepcheck` and `roundcheck` hold each row run beside others to the same row alone at the kind's lossless bound (KL
  mean 1e-4, max 2.5e-4; the best token the same wherever the lead clears 0.1), not bit for bit, and say why: the host and the GPU sum a pair in different
  orders, and which share of a call the host takes turns on what shares the call and on the places' state. Each row is judged by the first discrete
  choice the two runs made apart in its sequence's history (an expert route or an indexer's pick of pools, every layer and step, in causal order): none
  apart, the row is held to the bound; a near tie (below 1e-2 on each run's own input), the row and every later one are reported, not failed, with that
  choice; anything wider fails. The lines count rows held, reported and failed apart. Everywhere else (every expert on the GPU, copy mode) they stay
  bit for bit. Before, those checks failed on the codebook cache's bits alone. Part of #140.
- `fit --machine NAME` no longer turns on the computer that plans it: a preset says whether it has NVIDIA's cuBLAS (none of them: the engine never ships
  it), and a machine described in JSON says so with `cublas`. DGX Spark's plans had a long prompt's chunk of 4,096 tokens when planned on a computer with
  cuBLAS and 2,048 elsewhere. Part of #140.
- Qwen3.8-Flash-Next's MTP layer is a draft head on the description executor: its config's `mtp.*` are described (`MtpInput::Streams`: the model's last
  streams normed as one vector and each stream projected by `fc_hidden`, the next token's embedding normed and projected by `fc_embedding` added to each,
  an indexed attention and its experts in gated residuals as the model's layers, its own final mixer into the model's LM head), the CPU reference runs
  it, and the executor loads it as a draft part (its experts held as the model's, its keys and values in rings after the model's, the model's LM head
  and embedding shared). A drafting round checks the drafts in one verify pass whose rows are the steps', then goes back to the rows kept: the delta
  rule's and its convolution's states and now the n-gram embedding's last tokens and convolution. `glyd-engine draftcheck qwen4x` holds a drafted
  greedy generation to the model's alone bit for bit and the head to the CPU reference at the family's bound. Part of #140.
