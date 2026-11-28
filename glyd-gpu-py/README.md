# glyd-gpu

The GPU half of [Glyd](https://getglyd.com): a bf16 model's weights held compressed in GPU memory, bit for bit, and multiplied
from there (about a third less GPU memory), for PyTorch and transformers, and as a vLLM quantization method.

    pip install "glyd[gpu]"      # glyd.from_pretrained, glyd.fit
    pip install "glyd[vllm]"     # vllm serve MODEL --quantization glyd; glyd run MODEL

This package is what those two extras install: `glyd[gpu]` is `glyd-gpu[gpu]` (PyTorch, transformers, accelerate) and `glyd[vllm]` is
`glyd-gpu[vllm]` (vLLM 0.30); installed on its own it needs none of them. It ships as compiled wheels for Linux with an NVIDIA GPU (Ampere or later; CUDA 12
or 13 builds of PyTorch). Import name: `glyd_gpu`; `python -m glyd_gpu` has `fit`, `pack` and `verify`.

## The engine

`glyd run` starts Glyd's own engine, `glyd-engine` (a program in this package, beside the libraries), where it will certainly run the model (checked before anything is downloaded), and vLLM where it will not,
as before. A plain dense model at penguin goes to the engine only on a GPU where it is measured at least as fast as vLLM and with less memory (`GPUS` in `glyd-engine/src/machine.rs`: the RTX 4080 SUPER so far); every other family and level is the
engine's whatever that list says. `glyd serve` of a lossless model other than Gemma 4 runs vLLM, which serves more users at once than the 8 this tool allows the engine (the engine itself takes up to 64); a level, a Gemma 4 model and a glyd-v7 directory
go to the engine. Where the engine does not get up (it refuses, crashes, is killed or goes silent) the command goes on with vLLM, with one line saying why. `GLYD_ENGINE=0` keeps vLLM, and
`GLYD_ENGINE=1` asks for the engine for `glyd serve`. The engine needs only the NVIDIA driver.

`glyd run MODEL:LEVEL` holds the weights in fewer bits, not lossless, on the engine only (Gemma, Qwen and Nemotron-H have `kestrel`, `linnet` and `wren`; Nemotron-H, Kimi, GLM, DeepSeek and Muse also `hummingbird`): `kestrel`
is 6.53 bits per weight (`--lossy` is the same) and `linnet` 5.53; `penguin` is lossless, the default. Qwen3.8-27B's weights are 20.48 GiB
at kestrel and 17.35 GiB at linnet. A level the engine can't run on this GPU stops with a message rather than going to vLLM.

## C, C++, Rust

The wheel carries the GPU libraries (`libglyd_gpu_cuda12.so` and `libglyd_gpu_cuda13.so`: they need only the NVIDIA driver, and are
built with the CUDA 12 and 13 runtimes linked in) and their C header, `glyd_gpu.h`, in the package's directory:

    python -c "import glyd_gpu, os; print(os.path.dirname(glyd_gpu.__file__))"

For a program that multiplies by the matrices of a model saved by `glyd pack` (or `glyd.save_pretrained`), or decodes them back to bf16
bit for bit, without Python: the functions the header declares take the arrays of a pack as the save names them, passed as they are,
and choose how to run each product themselves. Linking: `gcc prog.c -I DIR -I CUDA/include -L DIR -lglyd_gpu_cuda13 -L CUDA/lib64 -lcudart -Wl,-rpath,DIR`.

License: the Apache License 2.0 (LICENSE), as all of Glyd.
