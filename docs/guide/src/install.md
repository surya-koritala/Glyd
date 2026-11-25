# Install

## What each machine gets today

| Machine | What you get |
| :--- | :--- |
| Linux x86_64, NVIDIA Ampere or newer, driver 535 or newer (570 for Blackwell; 580 for a build with CUDA 13 kernels alone) | The engine: `run`, `serve`, `fit`, `doctor`. |
| Linux x86_64, NVIDIA older than Ampere (Turing and before) | Refused: "older than the engine runs on". |
| Linux aarch64 (Grace, Jetson, DGX Spark) | Refused: the engine has not run on an aarch64 computer. File compression only. |
| macOS, Apple GPU | Refused: Apple GPUs are not supported yet. File compression only. |
| Windows x86_64, NVIDIA Ampere or newer, driver 535 or newer | From v0.34.0 a zip with `glyd.exe` (the engine linked in) and the GPU libraries, and `install.ps1` (below). The release workflow builds it and installs it on a runner with no GPU; a first chat on a Windows GPU is still to be shown ([#159](https://github.com/surya-koritala/Glyd/issues/159)). [Condor](condor.md) has PowerShell scripts for its own model. |
| AMD (RDNA 3 and 4) | A development build: `GLYD_HIP=1 cargo build --release -p glyd --features engine`. Not in releases. |
| No GPU | Refused: "no GPU was found: the engine's CPU backend is not built yet". |

The refusals are what `glyd run` prints (exit 3). `glyd doctor` says what the machine has.

## The release script

```bash
curl -LsSf https://getglyd.com/install.sh | sh
```

On Linux x86_64 with an NVIDIA GPU, the script installs `glyd` with the engine in one folder (its kernels are in the program: no GPU library)
(about 100 MB, no Python, no PyTorch), checks each download against the release's sha256, and ends with `glyd doctor`. It
also sets up the background service as a user unit of your account (see [Serving](serve.md)); `GLYD_NO_SERVICE=1` in front of
`sh` leaves that out. `GLYD_WITH_VLLM=1` adds the Python tool with vLLM and PyTorch (about 8 GB), for the bf16 models the
engine does not run.

On a Mac, or a machine with no NVIDIA GPU, it installs the file compression program only.

The copy of the script at getglyd.com is replaced with the v0.33.0 one when the site is next published from this repository.
Until then it installs v0.32.1, which goes through vLLM. To install v0.33.0 now, take the script from its tag:

```bash
curl -LsSf https://raw.githubusercontent.com/surya-koritala/Glyd/v0.33.0/scripts/install.sh | sh
```

The release has the engine bundles `glyd-engine-v0.33.0-linux-x86_64.tar.gz` and its `.sha256`
([releases](https://github.com/surya-koritala/Glyd/releases/tag/v0.33.0)).

## Windows

```powershell
irm https://getglyd.com/install.ps1 | iex
```

The script downloads the release's `glyd-engine-vX.Y.Z-windows-x86_64.zip`, checks it against the `.sha256` beside it, puts `glyd.exe` and the GPU
libraries (CUDA 13 and CUDA 12; the engine loads the one the driver's version asks for) in `%LOCALAPPDATA%\Glyd`, adds that folder to your user PATH, and runs
`glyd doctor`. No administrator rights. `$env:GLYD_RELEASE_URL` (a mirror or a folder), `$env:GLYD_VERSION`, `$env:GLYD_HOME` and `$env:GLYD_NO_PATH = '1'`
change where it reads from and writes to.

## Unsigned downloads

The Windows and macOS programs of a release are not code-signed or notarized: the certificates are the owner's to get ([#223](https://github.com/surya-koritala/Glyd/issues/223)).
Expect these warnings, which do not mean the file is bad:

- Windows SmartScreen, "Windows protected your PC", for `glyd.exe` or the installer: More info, then Run anyway.
- macOS Gatekeeper, "cannot be opened because the developer cannot be verified", for the `glyd` program of the tarball: `xattr -d com.apple.quarantine glyd`, or allow it in
  System Settings, Privacy & Security. (Homebrew's `glyd` is not quarantined.)

Check the file instead of trusting a warning: every release file is attested and signed by the workflow that built it, with no key of ours
(`gh attestation verify`, or `cosign` with the `.sigstore.json` beside the file). [Verifying a release](../../verify-releases.md) has the commands.

## Build from source

Linux x86_64, git, the Rust toolchain (the repository pins 1.97.1 and rustup fetches it), and for the GPU kernels the CUDA
toolkit with `nvcc` (12.8 or newer).

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
cargo build --release -p glyd --features engine
bash gpu/build_lib.sh target/release
target/release/glyd doctor
```

Or let cargo build and install the one program, the engine linked in (`glyd-engine` is `glyd` under another name: make it a link if a tool asks for it):

```bash
cargo install --git https://github.com/surya-koritala/Glyd glyd --features engine --locked
ln -s glyd "$(dirname "$(command -v glyd)")/glyd-engine"    # optional
glyd --version                                       # says `engine: linked`
```

Without `--features engine` the install is the file compression program alone.

- `-p glyd --features engine` builds `glyd` with the engine linked in: `run`, `serve`, `pull`, `ls`, `fit`, `doctor` and the
  rest are its own commands. `cargo build --release -p glyd-engine` builds the engine's own program, `glyd-engine`, which has
  the older command line without `pull`, `ls`, `rm` and the service.
- The build finds `nvcc` on the PATH, or at `$NVCC`. Without one it builds without CUDA kernels and warns. The kernels are
  built for Ampere, Ada and Hopper (`GLYD_ENGINE_ARCHS="80 86 89 90a"` is the default); `GLYD_ENGINE_ARCHS="89"` builds for one
  architecture, which is quicker.
- The engine's GPU kernels are in its own program: it loads no GPU library. `gpu/build_lib.sh DIR` writes the GPU library,
  `libglyd_gpu_cuda13.so`, into `DIR`, for the Python package and for `glyd-engine foldcheck` (a development build's
  comparison of the engine's kernels with the library's), which `GLYD_GPU_LIB=/path/to/it` names to the engine.
- The engine needs the NVIDIA driver only. It does not link the CUDA runtime.

The rest of the guide writes `glyd`. With a source build, run it as `target/release/glyd`.
