- The engine's reader of stored weights decodes two more forms of FP8 on the CPU reference, by the scales a model's description names rather than by a
  fixed suffix, the block read from the scales' shape and the config's `weight_block_size`: fp32 scales a block under another name (Mistral Large's
  `X.qscale_weight`), and a byte's power-of-two scale a run of 32 columns of a row (MiniMax M3's products: the weight is its E4M3 value times
  2^(scale - 127)). The executor's loaders read `weight_scale_inv` with fp32 scales, as before.
