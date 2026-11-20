# Install

## What each machine gets today

| Machine | What you get |
| :--- | :--- |
| Linux x86_64, NVIDIA Ampere or newer, driver 580 or newer | The engine: `run`, `serve`, `fit`, `doctor`. |
| Linux x86_64, NVIDIA older than Ampere (Turing and before) | Refused: "older than the engine runs on". |
| Linux aarch64 (Grace, Jetson, DGX Spark) | Refused: the engine has not run on an aarch64 computer. File compression only. |
| macOS, Apple GPU | Refused: Apple GPUs are not supported yet. File compression only. |
| Windows | Nothing is built or tested. [Condor](condor.md) has PowerShell scripts for NVIDIA GPUs on Windows. |
| AMD (RDNA 3 and 4) | A development build: `GLYD_HIP=1 cargo build --release -p glyd-engine`. Not in releases. |
| No GPU | Refused: "no GPU was found: the engine's CPU backend is not built yet". |

The refusals are what `glyd-engine run` prints (exit 3). `glyd-engine doctor` says what the machine has.

## Build from source

This is the way that works on main today. It needs Linux x86_64, git, the Rust toolchain (the repository pins 1.97.1 and
rustup fetches it), and for the GPU kernels the CUDA toolkit with `nvcc` (12.8 or newer).

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
cargo build --release -p glyd-engine
bash gpu/build_lib.sh target/release
target/release/glyd-engine doctor
```

- `cargo build` finds `nvcc` on the PATH, or at `$NVCC`. Without one it builds an engine with no CUDA kernels and warns.
  The kernels are built for Ampere, Ada and Hopper (`GLYD_ENGINE_ARCHS="80 86 89 90a"` is the default); set
  `GLYD_ENGINE_ARCHS="89"` to build for one architecture, which is quicker.
- `gpu/build_lib.sh DIR` writes the GPU library, `libglyd_gpu_cuda13.so`, into `DIR`. The engine looks for it beside
  its own program, then in the loader's path; `GLYD_GPU_LIB=/path/to/it` names it.
- The engine needs the NVIDIA driver only. It does not link the CUDA runtime.

## The release script

```bash
curl -LsSf https://getglyd.com/install.sh | sh
```

On Linux with an NVIDIA GPU, the current release (v0.32.1) installs the Python tool with vLLM and PyTorch (about 8 GB) as an
isolated uv tool. In that release `glyd run` goes through vLLM, except on the GPUs listed on the [Models](models.md) page,
where it hands the model to the engine. On a Mac or a machine with no NVIDIA GPU it installs the compression program.

## Not on main yet

v0.33.0 is the first release built only from this repository, with the engine installed without Python
([#129](https://github.com/surya-koritala/Glyd/issues/129)). Until it is out, build from source for the engine.
