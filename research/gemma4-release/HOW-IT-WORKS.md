# How Gemma 4 runs in Glyd

## The weights

Google released Gemma 4 31B and 12B twice: in bf16, and as QAT ("quantization-aware trained") checkpoints. In a QAT checkpoint
every matrix was trained so that its weights are a 4-bit whole number times one scale per group of 32 weights. Google ships those
checkpoints in bf16 (`-qat-q4_0-unquantized`), so they take as much memory as the bf16 model, but the information in them is
4-bit.

Glyd's wren level on a QAT checkpoint stores exactly that: the 4-bit numbers, and the scale of each group of 32 in 16 bits. That is
4.5 bits per weight. Nothing is rounded. When Glyd packs a matrix it decodes it again and compares every value with Google's; if
one differs, the pack stops with an error instead of saving an approximation.

So `glyd/gemma-4-31B-wren` is Google's QAT model, bit for bit, in 16.2 GiB instead of 58. It is not the original bf16 Gemma 4:
the QAT model is Google's own variant of it, and how close it is to bf16 is Google's work, not ours.

On other models wren rounds bf16 weights to 4 bits, like any 4-bit quant. That is why the card says which source it was packed from.

## Checking it yourself

`glyd verify glyd/gemma-4-31B-wren` decodes every tensor on the CPU and compares its sha256 with the one recorded from Google's
files when it was packed. It takes 17 s for the 31B on a 16-core CPU. Flip one bit anywhere in the files and it fails, naming the
tensor.

## Running it

The engine reads the 4-bit weights on the GPU directly and turns them into bf16 inside the matrix multiply. The model never takes its
bf16 size in GPU memory.

Gemma 4 alternates two kinds of attention layers: most look at the last 1,024 tokens only, and every sixth layer looks at the whole
context. The engine keeps the short ones in a fixed-size buffer, so a long context mostly costs memory in the few full layers. Long
prompts are read on the tensor cores in blocks of 512 tokens.

## The draft model

Google also released a small draft model for each size (the `-assistant` repos), trained to guess what the big model will write
next. It reads the big model's own state, so it is small (0.85-1.0 GB) and fast.

Each round the draft model guesses up to 8 tokens. The big model then checks all of them in one pass, which costs about the same as
writing one token, and keeps the guesses up to the first one it disagrees with, plus its own token at that point. With greedy
decoding the output is the same token for token as without the draft model. When sampling, a guess is kept with the probability
the big model gives it, so the text follows the same distribution.

How much it helps depends on the text. On chat-style answers the big model keeps about 2 guesses a round (1.7-2x faster). On
predictable text like code or lists it keeps 4-5 (up to 3.8x). `--no-draft` turns it off.

## What is not done yet

- Gemma 4 26B-A4B (mixture of experts) and E4B / E2B: not supported yet. 26B-A4B's QAT experts are not exactly 4-bit, so it can't
  be stored exactly this way.
- Lossless (penguin) for Gemma 4's bf16 models: not yet.
- Image and audio input: not yet. Text only.
- One request at a time.
