# The dynamic engine (GLY-326): design, 2026-10-17

Goal (milestone 0.33.0): one engine runs the popular families with no quality loss at 95%+ of each card's practical ceiling;
a family is a description, not a hand-written engine. Checkpoints GLY-327..331.

## What changes

Today each family has its own engine (`model.rs` Qwen3, `model35.rs` Qwen3.5, `model_nh.rs` Nemotron-H, `model_glm.rs`,
`model_qwen4x.rs`, `model_gemma4.rs`): its own config parser, weight loader, buffers, forward pass, graphs, drafting and serve
loop, 1,000 to 5,000 lines each. The kernels are mostly shared already (`kernels/*.cu`, `skinny.rs`, the library's products);
what is not shared is the code that strings them together.

The dynamic engine keeps the kernels and replaces the stringing:

1. `dynamic::blocks`: each kernel family behind one Rust type with one call shape, `run(&Ctx, rows, pos)`, reading and writing
   named activation slots (x, h, proj, att, o, gu, m, d, ...). A block owns its weights (PackBufs) and its parameters
   (eps, head dims, window, rotary, activation). First set (checkpoint 1, from Gemma 4's engine): Embed (scale), AddNorm
   (sandwich, scalar, next norm), Linear (every layout: skinny for few rows, the library otherwise, the L2 prefetch),
   QkPrep (q/k/v norms, rotary kinds, KV ring write), Attention (decode parts + merge, prompt tensor cores, window),
   Gate (gelu-tanh, silu; gated product), Head (final norm, LM head, soft-cap, argmax / top-k).
2. `dynamic::desc`: a `ModelDesc` built from config.json by a short per-family function: the activation slots and their widths,
   then each layer as a list of blocks with the tensor names they load and their parameters. No forward code in a family.
3. `dynamic::exec`: one executor. Loads the weights a description names (`weights.rs` as today, any layout, penguin
   included), sizes the slots and the KV rings, runs the blocks in order for n rows at positions, captures one CUDA graph
   per (rows, kind of pass) as the engines do today, and implements `conv::Gen` (run, serve) and the drafted round
   (a draft model is a second description; verification rows go through the same blocks, so drafted equals plain).
4. The ceiling report from the description: bytes a token (each block's weight bytes + KV/state per token), the practical
   ceiling from `pkcheck`'s stage costs, written beside every measurement.

## Rules

- Bit for bit: a family moved onto the engine gives the same tokens and the same logits as its old engine (the kernels
  are the same; the executor only orders them). The old engine stays until the new one matches, then goes.
- A block is added only when a family needs an operation no block does (Gemma's soft-cap, Qwen's n-gram embeddings,
  GLM's KDA); a block's kernel can carry a hand-tuned override per shape or card where it measurably wins.
- No per-family forward code in `exec`; families differ only in their descriptions.

## Checkpoint 1 (Oct 24): the core and Gemma 4 12B

1. blocks + desc + exec for the dense decoder shape (Gemma 4: sliding/global attention, sandwich norms, per-layer scalar,
   gelu-tanh, soft-capped tied head), prompt chunks and one-row steps, graphs.
2. `desc_gemma4(config) -> ModelDesc`.
3. Checks: g4check through the new engine gives the old engine's top-1 and KL to the digit; `glyd verify` passes on the
   penguin repo; g4spec drafted = plain; wren step <= 11.22 ms on the RTX 4080 SUPER.
4. The penguin product's rate against the copy rate, 4080 SUPER and 4090: the number that decides "lossless at the ceiling".
