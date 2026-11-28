- The engine's reader of stored weights decodes two more forms of FP8 on the CPU reference, by the scales a model's description names rather than by a
  fixed suffix, the block read from the scales' shape and the config's `weight_block_size`: fp32 scales a block under another name (Mistral Large's
  `X.qscale_weight`), and a byte's power-of-two scale a run of 32 columns of a row (MiniMax M3's products: the weight is its E4M3 value times
  2^(scale - 127)). The executor's loaders read `weight_scale_inv` with fp32 scales, as before.
- FP8 block scales that start again at each shard of a fused matrix are read on the CPU reference: MiMo V2's `qkv_proj`, cut for its key heads'
  shards, holds a grid of scales a shard (4 x 27 rows of blocks where one grid over the matrix would be 106), and each shard is decoded with its own.
