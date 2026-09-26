# Glyd weights on the GPU

A bf16 model's weights held compressed in VRAM, decoded on the GPU bit
for bit. A bf16 weight is a sign, 8 exponent bits and 7 mantissa bits;
in a trained model the sign and mantissa are noise, and the exponent
carries about 2.6 bits of information. Both formats keep the sign and
mantissa byte as it is and code the exponent:

| Format | Exponent | Bits a weight (Qwen2.5) | Decode |
| :--- | :--- | ---: | :--- |
| `huffman` (`pack`) | per-tensor prefix code read by counting leading zeros (as short as Huffman's on every tensor measured), 32 streams a tile | 10.88 | a lane decodes its stream in turn |
| `fast` (`pack_fast`) | 3-bit code into the tensor's 7 most common exponents, an escape to the exponent itself | 11.25 | bit operations, every weight in parallel |
| `mma` (`pack_mma`) | 2-bit digits in tiers: the tensor's 3 commonest exponents, digit 3 going on to the next 3, then the next 3, then the exponent itself; laid out in the order the tensor cores take their operand | 10.80 | in registers, straight into the tensor cores' operands |

The floor for any code that sees each tensor's exponents on their own
is about 10.6 bits a weight.

For one-token steps (generation) the product is fused: `fast_gemv` and
`gemv` read the packed weights, decode them in registers and multiply,
never writing bf16 out. Otherwise a matrix is decoded into one scratch
buffer and multiplied by PyTorch (`unpack`, `fast_unpack`).

## Measured

RTX 4080 SUPER (16 GB), PyTorch 2.14, CUDA 13.0; Qwen2.5-7B-Instruct,
greedy (`e2e.py MODEL --format mma --fused --baseline`): generating for
1 to 48 sequences at once, a prompt's forward pass, and perplexity on
Wikipedia text (enwik8 from its 10th MB, 200 windows of 64 tokens):

| | bf16 | Glyd `mma` |
| :--- | ---: | ---: |
| Peak VRAM | 15.25 GB | **10.61 GB** |
| 1 sequence | 43.4 tokens/s | **55.7** (1.28x) |
| 4 sequences | 167.9 | **217.5** (1.30x) |
| 16 sequences | 652.0 | **814.2** (1.25x) |
| 32 sequences | 1153.7 | **1518.9** (1.32x) |
| 48 sequences | 1664.8 | **1882.5** (1.13x) |
| 64 sequences | 2160.0 | **2244.7** (1.04x) |
| Prompt of 16 tokens | 24 ms | **19 ms** |
| Prompt of 64 tokens | 27 ms | **26 ms** |
| Prompt of 128 tokens | 29 ms | 29 ms |
| Prompt of 256 tokens | 43 ms | 45 ms |
| Prompt of 512 tokens | 79 ms | 85 ms |
| Prompt of 1024 tokens | 154 ms | 163 ms |
| Prompt of 2048 tokens | 301 ms | 330 ms |
| Prompt of 4096 tokens | 645 ms | 702 ms |
| Perplexity, 64-token windows | 17.0015 | 17.0052 |
| Perplexity, 512-token windows | 7.5677 | 7.5660 |

The weights are the model's to the bit; the product sums in another
order than cuBLAS, which moves the logits by a rounding: the next token
chosen is bf16's 98.13% of the time (98.49% on the 512-token windows).
bf16 against itself, two windows a pass instead of one: perplexity
17.0153, the same next token 98.33% of the time.

`mma_gemm` (1 to 64 tokens): a warp step is 1024 weights, 64 rows by
16 columns, in the order `mma.sync.m16n8k16` takes its B operand: 1280
bytes (each lane's 32 tier-1 digits, then its 32 sign-and-mantissa
bytes) and a block of the step's escapes (its tier-2 digits in words of
16 back from the block's end, its tier-3 digits and exponent bytes from
its start). A lane's weights arrive in four loads and two words of the
block's ends; one warp scan places the tier-2 digits, each lane decodes
16 of them (and their tier-3 digits, placed by a second scan) into
shared memory, and each lane's tier-1 digits take their symbols or
those bytes in one table lookup and one byte permute for 4 weights. A
pair of bf16s is then one permute and one rotate (each weight's byte
carries its pair's other sign). No lane waits on another's escapes:
a step decodes in the same instructions however its escapes fall. The
steps are split evenly over the blocks (stream-K), a row block's parts
added in a fixed order by its last block: the same result every run. On
7B's matrices it reads the packed weights at 94% of the bandwidth bf16's
product reaches, 1.30-1.34x faster than bf16 at one token on the MLP's,
1.06-1.19x at 64.

`mma_gemm_big` (more tokens: a prompt) is a tiled GEMM with its warps
split: four produce, copying X's tile of each stage (64 columns) into
shared memory by `cp.async` and decoding W's steps of it into B
fragments there; four consume, each 64 tokens by 64 rows on the tensor
cores with their fragments double-buffered, never waiting on a decode.
Three stages are in flight, passed between the two by named barriers. A
block is 128 tokens by 128 rows of W, or past 128 tokens 256 by 64 (a
weight decoded once for twice the tokens); where the blocks would not
fill the GPU, K is split and the parts added in a fixed order. Past 128
tokens the product is bound by the tensor cores, not by memory, so the
most it can be is bf16's time; it is within 5-10% of it. Measured on
Qwen2.5-7B's matrices: the consumers alone come within 1-4% of cuBLAS
(one warp an SM quarter keeps the tensor cores full: 106 TFLOPS, as
cuBLAS's kernel); the rest is the producers' decoding sharing the SM.

The other formats, 128 new tokens, one sequence:

| | Peak VRAM | Tokens/s | Tokens as bf16's |
| :--- | ---: | ---: | ---: |
| bf16 | 15.25 GB | 43.3 | |
| `fast`, fused | 11.05 GB | 55.1 | 128 of 128 |
| `huffman`, fused | 10.60 GB | 52.0 | 54 of 128 |
| `fast`, decoded then PyTorch's matmul | 11.05 GB | 18.3 | 128 of 128 |

The fused `fast` product on 7B's matrices (`shapes.py`): 662–695 GB/s of
packed weights, 1.29–1.48x bf16's matrix-vector time (the small key and
value projections, which sit in L2, 0.66x). The fused `huffman` product:
1.17–1.46x (575–690 GB/s of packed weights). Profiled (Nsight Compute),
its integer pipe had been the limit (83% busy, 44 instructions a
weight): the reader is two words and a funnel shift, a code's rank is its
value plus a stored base (taken mod 32 by the shuffle), a weight's float
is assembled by byte permutes around an exponent kept in a float's place,
and the next stream word is asked for a word ahead. Rows longer than a
tile are split across tiles, their parts added in fp32 by the last tile
of the row. Tried and dropped: two interleaved streams a lane, a table
giving up to three codes a lookup. The fused product sums in
another order than cuBLAS, as any two GEMM kernels do; the decoded path
multiplies with PyTorch's own kernel and gives bf16's logits bit for bit
where a matrix is decoded whole (Qwen2.5-0.5B: logits and 128 tokens
identical; the 7B output layer is decoded in blocks to cap the scratch).

Several tokens at once in the fast format: `fast_gemm` reads it,
decodes each 64-weight step of a 64-row tile into shared memory once for
all the tile's tokens and multiplies on the tensor cores; K is split
across blocks where W has few rows, the parts added in a fixed order (the
same result every run). `e2e.py` uses it for prompts of up to 64 tokens;
longer ones decode each matrix and use PyTorch's matmul. A prompt's
forward pass, Qwen2.5-7B (`--prefill`):

| Prompt | bf16 | Glyd `fast` |
| ---: | ---: | ---: |
| 16 tokens | 24 ms | 35 ms (was 61) |
| 64 tokens | 27 ms | 40 ms (was 61) |
| 128 tokens | 29 ms | 61 ms |
| 512 tokens | 79 ms | 113 ms |
| 2048 tokens | 301 ms | 337 ms |

Past 64 tokens the fused kernel is not yet faster than decoding the matrix
and multiplying (profiled: its loads queue up and stall, at 29%
occupancy); a GEMM at cuBLAS's efficiency is what closes that.

## Larger models

On rented GPUs (`scripts/gpu_lambda.sh`: one Lambda Cloud instance a
run, terminated at the end; raw logs in `benchmarks/gpu/lambda-*`), bf16
and Glyd in the same run, 64 new tokens a sequence, MMLU on the same
1,000 questions (0-shot):

| Model, GPUs | Weights | Tokens/s at 1 / 8 / 32 / 64 sequences | Prompt of 2048 | Perplexity | MMLU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B, bf16, 2x A6000 48 GB | 65.52 GB | 9.5 / 74.4 / 275.9 / 488.4 | 1332 ms | 17.0796 | 78.5% |
| Qwen3-32B, Glyd, **1x** A6000 | **44.45 GB** | **11.8 / 94.9 / 289.9** / 400.5 | 2082 ms | 17.0788 | 78.0% |
| Qwen2.5-72B, bf16, 4x A6000 | 145.41 GB | 4.5 / 35.0 / 134.6 / 254.3 | 2689 ms | 10.6035 | 81.9% |
| Qwen2.5-72B, Glyd, **3x** A6000 | **97.80 GB** | **6.4 / 49.0 / 153.8** / 218.1 | 4195 ms | 10.5996 | 81.8% |
| Qwen3-32B, bf16, H100 SXM 80 GB | 65.52 GB | 13.4 / 123.2 / 502.0 / 988.4 | 262 ms | 17.0814 | 78.2% |
| Qwen3-32B, Glyd, H100 SXM | **44.45 GB** | 19.9 / 157.1 / 465.1 / 789.5 | 305 ms | 17.0849 | 78.2% |
| Qwen2.5-7B, bf16, H100 SXM | 15.23 GB | 25.9 / 207.1 / 823.0 / 1670.2 | 55 ms | 17.0178 | 73.3% |
| Qwen2.5-7B, Glyd, H100 SXM | **10.32 GB** | 57.4 / 455.5 / 1655.5 / 2807.1 | 64 ms | 17.0162 | 73.4% |

The MMLU answers are bf16's on 99.2% (Qwen3-32B, A6000), 99.4%
(Qwen2.5-72B), 100% (Qwen3-32B, H100) and 99.9% (Qwen2.5-7B, H100) of
the questions. Across GPUs the A6000s run Glyd's model on fewer of them,
so their pipeline has fewer stages. On the H100 Hugging Face's generation
loop is bound by the CPU at few sequences (Qwen3-32B, profiled over 16
tokens: 54.8 ms a token for bf16, 55.8 for Glyd), and the GPU's work is
not Glyd's gain there: 40.6 ms of GPU time a token against bf16's 28.2,
Qwen3-32B's MLP matrices 128 us against cuBLAS's 90 at one token (the
decode is bound by arithmetic when memory moves 3.35 TB/s).

## The KV cache

The keys and values a model keeps for the tokens it has seen are bf16
like its weights, and as compressible: on Qwen2.5-7B over 2,048 tokens of
enwik8 the exponent carries 2.80 bits in the keys (2.59 given the
channel) and 2.67 in the values, a floor of 10.5-10.6 bits a value, 34%
under bf16. `kv.py` holds a Hugging Face model's cache that way:
`GlydKVCache(config)` keeps each layer's newest tokens as they are and
packs every full page of 64 in the mma layout's tiered code, keys by
token and values transposed (the operands of attention's two products),
the layer's tiers taken from its first page. With `fused=True` and
`use_fused_attention(model)`, a step of one new token a sequence runs
`attn_decode` on the packed pages: a block per KV head and run of pages,
a page a warp, keys and values decoded in registers, the head's queries
as one tensor-core tile, an online softmax, the blocks merged in a fixed
order. Other steps (a prompt) get the keys and values decoded exactly.

Qwen2.5-7B-Instruct, weights in the mma layout, RTX 4080 SUPER, an
enwik8 prompt then 128 new tokens (`e2e.py --kv 1024,4096,16384`):

| Prompt | KV cache, plain | Packed | A step, plain | Fused | Peak memory, plain | Packed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 tokens | 66 MB | **46 MB** (70.4%) | 18.3 ms | 19.2 ms | 10.80 GB | 10.78 GB |
| 4,096 tokens | 242 MB | **167 MB** (69.1%) | 19.1 ms | 19.6 ms | 11.41 GB | 11.34 GB |
| 16,384 tokens | 947 MB | **651 MB** (68.7%) | 21.6 ms | 21.7 ms | 13.87 GB | 13.58 GB |

Decoded back, the cache is the plain cache's bit for bit: the same 128
tokens. Through `attn_decode` the sums run in another order than
FlashAttention's, as with the weights: 256 tokens of the text fed one at
a time after the prompt, perplexity 2.9895 / 4.3106 / 2.4778 against the
plain cache's 2.9950 / 4.3130 / 2.4809, the next token the plain cache's
97.3% / 99.6% / 99.6% of the time.

## Running

Needs PyTorch with CUDA and nvcc (the extension builds on first import):

    python check.py model.safetensors         # every tensor packed, unpacked, compared; speeds
    python shapes.py MODEL_DIR                # fused product against bf16, one layer's matrices
    python gemm.py MODEL_DIR 1,16,64          # several tokens: every product against bf16, one layer's matrices
    python e2e.py MODEL_DIR --format mma --fused --baseline [--batch 1,8,32] [--prefill 16,64] [--ppl TEXT] [--mmlu 1000] [--kv 1024,4096]
    python kv.py                              # the KV cache packed and decoded bit for bit; attn_decode against SDPA

## License

The files under gpu/ are under the [Business Source License 1.1](LICENSE):
source available, free for personal, educational, research and other
non-commercial use; any commercial production use needs a license
(suryakoritala1324@gmail.com); each version converts to Apache-2.0 four
years after its release. The Glyd codec is BSD-3-Clause OR GPL-2.0.
