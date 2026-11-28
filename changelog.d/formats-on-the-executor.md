- A MiMo release is read by the loader of the engine's executor, and is no longer refused. Its two stored forms are read where the loader reads a
  matrix, by the scales the model's description names: the fused attention matrices' FP8 with block scales that start again at each shard, and the
  experts' packed 4-bit codes. Each such matrix is decoded on the host to the bf16 its codes and scales stand for, and then goes the way a bf16
  matrix does: packed at the plan's level (kestrel, linnet, wren), or held lossless at penguin. The 4-bit experts decode exactly; a fused FP8
  weight is the bf16 nearest its code times its block's fp32 scale. A small model in the release's stored forms loads to the same bytes as its bf16
  twin at every level, on the host. It is a case of the logits checks and of `kindcheck` (`mimo-released`), held to the CPU reference. No GPU had
  run that case, or loaded a release, when this was written.
- `glyd fit` sizes a MiMo release as the bf16 model its description makes, at each level's bits, not at its files' bytes: the release's files are
  172.9 GB, and the engine holds it in 174.9 GB at wren, 213.5 GB at linnet, 252.1 GB at kestrel and 416.9 GB at penguin. It plans linnet on a GPU
  of 80 GB with 200 GB of memory, its experts in the computer's memory, and holds no level on a GPU of 16 or 24 GB. A release stored quantized is
  not held as released: penguin is not smaller than bf16 would be.
- FP8 with fp32 scales under a release's own name and in blocks of any size (Mistral Large's), FP8 with a byte's scale a run of 32 and NVFP4 experts
  (MiniMax's) are read by the same loader code, and held to their bf16 twins at the loader. Those two families' layouts do not run on the executor
  yet, so their releases do not load.
- Still not done for MiMo: its draft layers are not read, so it runs without drafting; `glyd pack` of a release stored in these forms is refused by
  name (the release runs as it is); its layers are not cut over several GPUs.
