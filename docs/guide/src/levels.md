# Choosing a level

A level is how a model's weights are stored on the GPU. You pick one with a tag: `glyd-engine run Qwen/Qwen3.8-27B:swift`.
With no tag, `run` uses penguin, and refuses a model that has none. `fit` lists every level and picks the first that fits:
lossless if it fits, then the highest lossy level that does, then experts in RAM for a mixture of experts. `--lossy` is the
same as `:kestrel`.

| Level | Bits a weight | Lossless | What it is |
| :--- | :--- | :--- | :--- |
| penguin | about 10.8 | yes | The release's weights, bit for bit, packed smaller than bf16. |
| kestrel | 6.5 | no | The closest lossy level. |
| swift | 5.5 | no | Smaller again. |
| wren | 4.5 | no | The smallest dense level. Exact on a quantization-aware checkpoint; a development build otherwise. |
| hummingbird | 2 to 4 on the experts | no | A mixture of experts' experts at 2 to 4 bits by use. Pre-packed. |

Which families run at which level is on the [Models](models.md) page. A level a family does not run is refused.

## How to choose

1. Run `glyd-engine fit MODEL`. It lists every level with its size on your GPU and says which fit.
2. Take penguin if it fits. Nothing changes.
3. If not, take the highest level that fits. Each step down changes more of the model's answers than the one above it.
4. For code and long tool loops, stay at kestrel or above where you can.

## What each costs

`fit` shows the memory for your GPU ([First run](first-run.md) has an example). Quality is measured as the KL divergence of
the model's next-token distribution from bf16's, and how often its top token is bf16's. The records, each with its machine,
build and log, are in [benchmarks/](https://github.com/surya-koritala/Glyd/tree/main/benchmarks): start with
[benchmarks/weights/kld-q56-2026-10-09](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/weights/kld-q56-2026-10-09)
for the dense levels and
[benchmarks/cards](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/cards) for the cards.
