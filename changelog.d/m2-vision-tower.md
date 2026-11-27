- The vision tower is a kind of the description (`VisionDesc`, the Qwen-VL vision transformer: Qwen3.8-Flash-Next's) and an input stage of the executor that
  any family's description can use: the description builder reads a multimodal config's `vision_config` and image token and names the tower's tensors
  (a tower with deepstack layers, a text rotary whose M-RoPE sections do not interleave, or a key it does not know is refused in words); the executor
  opens the tower from the description, runs a prompt's images through it when a pass reaches their rows, writes its rows over the image tokens'
  embedding rows, and turns every row by its M-RoPE position (an image's tokens (p, p + row, p + column), the text after it from p + the larger side) in
  the attention and the mean-pooled indexer, as the release's engine did. Images come through the common server and scheduler (OpenAI image parts,
  Anthropic image blocks, `--image-pixels N`), and `glyd run --image PATH --prompt TEXT` reads them too. A prompt with images finds in the cache only
  what prompts with the same images left. The tower has a CPU reference at the release's rounding points. Before, the family's engine on the executor
  served text alone since the release's own engine was deleted. Part of #140.
