- A prompt chunk of more than 64 tokens through the expert cache's streaming prefill (Qwen3.8-Flash-Next's codebook experts) gives the same bits every run:
  each pair's row is written once at its place in the router's order and each token's rows are summed in that order, where before the experts' products
  were added into the tokens' rows with atomic adds in whatever order the GPU ran them. Which experts the host computes beside the GPU is the call's plan
  alone, no longer moved by the calls' measured times. A long prompt, and every image prompt, read the same logits on one GPU and across GPUs.
- Images on a model cut across GPUs are read again (`qwenserve --gpus 0,1`, `glyd run --split`, `serve`): the first stage runs the vision tower, every
  stage turns its rows by the images' positions, and the replies are one GPU's to the bit. `visioncheck repeat` holds an image prompt and a long text
  prompt to one GPU's first pass on one GPU, on a split and with the experts streamed instead of held. Part of #140.
