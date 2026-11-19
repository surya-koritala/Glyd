# glyd-gpu

The GPU half of [Glyd](https://getglyd.com): a bf16 model's weights held compressed in GPU memory, bit for bit, and multiplied
from there (about a third less GPU memory), for PyTorch and transformers, and as a vLLM quantization method.

    pip install "glyd[gpu]"      # glyd.from_pretrained, glyd.fit
    pip install "glyd[vllm]"     # vllm serve MODEL --quantization glyd; glyd run MODEL

This package is what those two extras install: `glyd[gpu]` is `glyd-gpu[gpu]` (PyTorch, transformers, accelerate) and `glyd[vllm]` is
`glyd-gpu[vllm]` (vLLM 0.30); installed on its own it needs none of them. It ships as compiled wheels for Linux with an NVIDIA GPU (Ampere or later; CUDA 12
or 13 builds of PyTorch). Import name: `glyd_gpu`; `python -m glyd_gpu` has `fit`, `pack` and `verify`.

## The engine

`glyd run` starts Glyd's own engine, `glyd-engine` (a program in this package, beside the libraries), for a model it runs on a GPU where it is measured at least as fast as vLLM and with less memory, and where it will
certainly run the model (checked before anything is downloaded), and vLLM everywhere else, as before: it is measured on no GPU yet, so until it is every model runs on vLLM. Lossless, the 2026 models (Qwen3.5, Qwen3.8, Gemma 4) run on vLLM: the engine does not run them yet. `glyd serve` runs vLLM, which serves more
users at once than the engine's 8. Where the engine does not get up (it refuses, crashes, is killed or goes silent) the command goes on with vLLM, with one line saying why. `GLYD_ENGINE=0` keeps vLLM, and
`GLYD_ENGINE=1` asks for the engine for `glyd serve`. It needs only the NVIDIA driver and this package's library.

`glyd run MODEL:LEVEL` holds the weights in fewer bits, not lossless, on the engine only (the Qwen3.5 family, Qwen3.8 included): `kestrel`
is 6.53 bits per weight (`--lossy` is the same) and `swift` 5.53; `penguin` is lossless, the default. Qwen3.8-27B's weights are 20.48 GiB
at kestrel and 17.35 GiB at swift. A level the engine can't run on this GPU stops with a message rather than going to vLLM.

## C, C++, Rust

The wheel carries the GPU libraries (`libglyd_gpu_cuda12.so` and `libglyd_gpu_cuda13.so`: they need only the NVIDIA driver, and are
built with the CUDA 12 and 13 runtimes linked in) and their C header, `glyd_gpu.h`, in the package's directory:

    python -c "import glyd_gpu, os; print(os.path.dirname(glyd_gpu.__file__))"

For a program that multiplies by the matrices of a model saved by `glyd pack` (or `glyd.save_pretrained`), or decodes them back to bf16
bit for bit, without Python: the functions the header declares take the arrays of a pack as the save names them, passed as they are,
and choose how to run each product themselves. Linking: `gcc prog.c -I DIR -I CUDA/include -L DIR -lglyd_gpu_cuda13 -L CUDA/lib64 -lcudart -Wl,-rpath,DIR`.

License: Business Source License 1.1 (LICENSE), as all of Glyd from v0.29.0. Free forever for personal and non-commercial use (including education, research and nonprofits) on computers you own or rent for yourself, and for anyone to try, test and develop with. Commercial use needs a license: suryakoritala@getglyd.com. Each version becomes Apache-2.0 four years after its release.
