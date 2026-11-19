#!/bin/sh
# Glyd's installer:  curl -LsSf https://getglyd.com/install.sh | sh
#
# On Linux with an NVIDIA GPU it installs `glyd run MODEL`: Glyd with vLLM and PyTorch (about 8 GB) as an isolated tool of uv, on a Python
# 3.12 that uv manages, so the system's Python is left alone: no virtual environment, no pip refusing to install (PEP 668), no Python
# version to choose, no C headers to find. Where the machine has no gcc or clang, which vLLM's Triton builds its launchers with, it adds
# ziglang, a compiler from PyPI, so no sudo is needed. It ends with `glyd doctor`.
# On a Mac, or a Linux machine with no NVIDIA GPU, there is no `glyd run` to install: it installs the compression program (glyd FILE -o
# OUT) from the release's tarball, checked against the sha256 the release lists, and says what `glyd run` needs.
#
# All of it is under your home directory, and there is no sudo. What it puts there:
#   uv       where there is none: uv's own installer, at the version pinned below and checked against its sha256, told not to edit your
#            shell's startup files; into ~/.local/bin
#   Glyd     the release pinned below, with the versions of its ~200 packages that the acceptance run installed (the list at the end of
#            this file), in uv's tool directory (~/.local/share/uv/tools/glyd) and uv's cache (~/.cache/uv); a link ~/.local/bin/glyd
#   PATH     where ~/.local/bin is not on it: `uv tool update-shell` adds a line to your shell's startup file, and says which one
# It stops, before it installs Glyd, where ~/.local/bin/glyd is not Glyd's Python tool: it does not replace a program it did not make.
# The whole script is one function, called on its last line, so a download that is cut short runs nothing.
#
#   GLYD_VERSION      the release to install (default below; a pre-release is named here, e.g. 0.26.0rc3, and only that one is taken)
#   GLYD_SPEC         the package to install instead, as uv takes it: a wheel with its extra ("/path/glyd-...whl[vllm]"), for another
#                     build (the list below still applies to it: a build that needs other versions says GLYD_CONSTRAINTS=none)
#   GLYD_CONSTRAINTS  none: resolve the packages fresh, where one of the versions listed below has been withdrawn from PyPI
set -eu

GLYD_VERSION="${GLYD_VERSION:-0.29.1}"
PYTHON=3.12
DRIVER_MIN=580  # the NVIDIA driver vLLM 0.30's PyTorch (2.13, CUDA 13.0) runs on
ZIGLANG=0.16.0  # the C compiler from PyPI that stands in where the machine has none (the version glyd run was tried with)
UV_VERSION=0.12.21  # the uv the acceptance run installed; its installer (https://astral.sh/uv/$UV_VERSION/install.sh) names the sha256 of each download
UV_INSTALLER_SHA256=0722d6c438395e39e1c27a86a79054d3b2820dd9399c7f8b0f6f84cd27ce36c3
UV_MIN_MINOR=7  # uv 0.7: `uv tool install --managed-python` is there in 0.6.17 and not in 0.6.0
DISK_GB=10      # free disk the install needs: about 8 GB, 16 where uv's cache and its tool directory are on different disks
REPO=https://github.com/surya-koritala/Glyd

say() { printf '==> %s\n' "$*"; }
warn() { printf 'glyd install: %s\n' "$*" >&2; }
die() { printf 'glyd install: %s\n' "$*" >&2; exit 1; }

tmp=
resume=
finish() {
  rc=$?
  [ -z "$tmp" ] || rm -rf "$tmp"
  if [ "$rc" != 0 ] && [ -n "$resume" ]; then warn "the install did not finish (see above). Run the same command again: $resume"; fi
}

# HTTPS only, and a failing status is a failure (a 404 page is not a file)
fetch() { curl --proto '=https' --tlsv1.2 -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d ' ' -f 1
  else return 1
  fi
}

# Where another glyd comes first on PATH than $1/glyd, the one just installed (the compression program, if it is the Rust one, takes
# `run` for a file name): say what to do about it. $2 is a command to try by its path.
check_first() {
  first=$(command -v glyd 2>/dev/null || true)
  [ -n "$first" ] || return 0
  if [ "$first" = "$1/glyd" ] || [ "$first" -ef "$1/glyd" ] 2>/dev/null; then return 0; fi
  warn "another glyd comes first on your PATH: $first. Typing glyd reaches that one, not $1/glyd, the one just installed."
  warn "  Put its folder first: add this line to your shell's startup file (~/.zshrc for zsh, ~/.bashrc for bash) and open a new terminal:"
  warn "      export PATH=\"$1:\$PATH\""
  warn "  Or run it by its path:  $1/glyd $2"
}

# The compression program, for a machine that has no use for glyd run.
install_cli() {
  case "$os-$arch" in
    Linux-x86_64) plat=linux-x86_64 ;;
    Linux-aarch64 | Linux-arm64) plat=linux-aarch64 ;;
    Darwin-arm64) plat=macos-arm64 ;;
    *) die "there is no prebuilt compression program for $os on $arch. Homebrew builds one (brew install surya-koritala/glyd/glyd), or: cargo install --git $REPO glyd glyd-store" ;;
  esac
  bin=${XDG_BIN_HOME:-$HOME/.local/bin}
  cli=${XDG_DATA_HOME:-$HOME/.local/share}/glyd/cli
  name=glyd-v$GLYD_VERSION-$plat
  url=$REPO/releases/download/v$GLYD_VERSION
  for f in glyd glyd-store; do  # (before anything is downloaded: only a link of this installer's own is replaced)
    if [ -e "$bin/$f" ] || [ -L "$bin/$f" ]; then
      case $(readlink "$bin/$f" 2>/dev/null || true) in
        "$cli"/*) ;;
        *) die "$bin/$f exists, and this installer did not make it: it does not replace a program it did not make. Move it away (or set XDG_BIN_HOME to another folder on your PATH), then run this again." ;;
      esac
    fi
  done
  say "Installing the compression program, glyd $GLYD_VERSION for $plat (the release's tarball, checked against its sha256)"
  fetch "$url/$name.tar.gz" "$tmp/$name.tar.gz" || die "could not download $url/$name.tar.gz: check the network, and that release $GLYD_VERSION has a $plat tarball."
  fetch "$url/$name.tar.gz.sha256" "$tmp/$name.sha256" || die "could not download the release's checksum ($url/$name.tar.gz.sha256)."
  want=$(cut -d ' ' -f 1 "$tmp/$name.sha256")
  got=$(sha256_of "$tmp/$name.tar.gz") || die "there is no sha256sum or shasum here to check the download with."
  [ -n "$want" ] && [ "$got" = "$want" ] || die "the download is not the file the release lists (its sha256 is $got, the release says ${want:-nothing}). Run this again; if it happens again, tell the Glyd project."
  mkdir "$tmp/x"
  tar -xzf "$tmp/$name.tar.gz" -C "$tmp/x" "$name/glyd" "$name/glyd-store" "$name/LICENSE" \
    || die "the release's tarball is not laid out as this installer expects ($name/glyd, ...): run this again, or take the program from $REPO/releases."
  mkdir -p "$cli.new" "$bin"  # (tried where it will live: a temporary folder can be noexec)
  cp "$tmp/x/$name"/* "$cli.new"/
  "$cli.new/glyd" --version >/dev/null 2>&1 \
    || { rm -rf "$cli.new"; die "the compression program from the release does not start on this machine (the Linux x86_64 one needs a CPU with AVX2 and BMI2). Build it: cargo install --git $REPO glyd"; }
  rm -rf "$cli"
  mv "$cli.new" "$cli"
  for f in glyd glyd-store; do ln -sf "$cli/$f" "$bin/$f"; done
  say "$("$bin/glyd" --version | head -n 1) is installed in $cli, linked from $bin (glyd, glyd-store)"
  case ":$PATH:" in
    *":$bin:"*) ;;
    *) say "$bin is not on your PATH. Add this line to your shell's startup file (~/.zshrc for zsh, ~/.bashrc for bash) and open a new terminal:  export PATH=\"$bin:\$PATH\"" ;;
  esac
  check_first "$bin" --version
  say "Next: glyd --help, or glyd FILE -o FILE.glyd. glyd run needs Linux with an NVIDIA GPU (Ampere or newer); run this installer on that machine for it."
}

main() {
  os=$(uname -s)
  arch=$(uname -m)
  case "$os" in
    Linux | Darwin) ;;
    *) die "this installer is for Linux and macOS (on Windows, use WSL2 with the NVIDIA driver of Windows)." ;;
  esac
  case "${HOME:-}" in
    /*) [ -d "$HOME" ] || die "HOME ($HOME) is not a folder: this installs under your home directory." ;;
    *) die "HOME is not set to your home directory: this installs under it." ;;
  esac
  command -v curl >/dev/null 2>&1 || die "curl is needed (Ubuntu: sudo apt install curl)."
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' HUP TERM
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/glyd-install.XXXXXX") || die "could not make a temporary folder in ${TMPDIR:-/tmp}."

  gpu=no
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then gpu=yes; fi

  stack=no  # whether vLLM and PyTorch go in, which is what glyd run uses
  if [ -n "${GLYD_SPEC:-}" ]; then
    spec=$GLYD_SPEC
    case "$spec" in *'[vllm]') stack=yes ;; esac
  elif [ "$os" = Darwin ]; then
    say "This computer is a Mac: glyd run (the local chat) needs Linux with an NVIDIA GPU. The compression commands work here."
    install_cli
    return 0
  elif [ "$gpu" = no ]; then
    say "No NVIDIA GPU answered here (nvidia-smi): glyd run (the local chat) needs Linux with an NVIDIA GPU. The compression commands work without one."
    say "If this machine has an NVIDIA GPU, install its driver first (Ubuntu: sudo apt install nvidia-driver-$DRIVER_MIN, then reboot) and run this again."
    install_cli
    return 0
  elif [ "$arch" = x86_64 ] || [ "$arch" = aarch64 ]; then
    spec="glyd[vllm]==$GLYD_VERSION"
    stack=yes
  else
    say "There is no vLLM build for $arch: glyd run (the local chat) is not available here. The compression commands work."
    install_cli
    return 0
  fi

  if [ "$stack" = yes ] && [ -z "${UV_CACHE_DIR:-}${UV_TOOL_DIR:-}" ]; then  # (uv's folders placed by the user are the user's to size)
    free_kb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR == 2 { print $4 }') || free_kb=
    case "$free_kb" in
      '' | *[!0-9]*) ;;
      *) if [ "$free_kb" -lt $((DISK_GB * 1024 * 1024)) ]; then
           die "$HOME has $((free_kb / 1024 / 1024)) GB free, and the install needs about 8 GB (16 GB where uv's cache and its tool folder are on different disks). Free some space, or set UV_CACHE_DIR and UV_TOOL_DIR to a folder on a bigger disk, and run this again."
         fi ;;
    esac
  fi

  if [ "$gpu" = yes ]; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)
    major=${driver%%.*}
    case "$major" in
      '' | *[!0-9]*) ;;
      *) if [ "$major" -lt "$DRIVER_MIN" ]; then
           warn "your NVIDIA driver ($driver) is older than $DRIVER_MIN, which vLLM's PyTorch needs. Installing anyway; before you run a model, update it (Ubuntu: sudo apt install nvidia-driver-$DRIVER_MIN, then reboot)."
         fi ;;
    esac
  fi

  uv=
  for c in "$(command -v uv 2>/dev/null || true)" "${XDG_BIN_HOME:-$HOME/.local/bin}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -n "$c" ] && [ -x "$c" ]; then uv=$c; break; fi
  done
  if [ -z "$uv" ]; then
    say "Installing uv $UV_VERSION, which manages Python for Glyd (its own installer, from astral.sh, checked against its sha256; it edits no shell startup file)"
    fetch "https://astral.sh/uv/$UV_VERSION/install.sh" "$tmp/uv-installer.sh" || die "could not download uv's installer from astral.sh: check the network, then run this again."
    got=$(sha256_of "$tmp/uv-installer.sh") || die "there is no sha256sum or shasum here to check uv's installer with. Install uv yourself (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
    [ "$got" = "$UV_INSTALLER_SHA256" ] || die "uv's installer is not the file this script was written for (its sha256 is $got, expected $UV_INSTALLER_SHA256). Install uv yourself (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
    UV_NO_MODIFY_PATH=1 sh "$tmp/uv-installer.sh" || die "uv's installer failed (see above); install uv another way (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
    for c in "${XDG_BIN_HOME:-$HOME/.local/bin}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "${UV_INSTALL_DIR:-/nonexistent}/uv" "$(command -v uv 2>/dev/null || true)"; do
      if [ -n "$c" ] && [ -x "$c" ]; then uv=$c; break; fi
    done
    [ -n "$uv" ] || die "uv installed, but it is not where its installer says; open a new terminal and run this again."
  fi
  uvv=$("$uv" --version 2>/dev/null | cut -d ' ' -f 2) || uvv=
  uvmajor=${uvv%%.*}
  uvminor=${uvv#*.}
  uvminor=${uvminor%%.*}
  case "$uvmajor.$uvminor" in
    . | .* | *.) ;;  # (a version this cannot read: go on)
    *[!0-9.]*) ;;
    *) if [ "$uvmajor" -eq 0 ] && [ "$uvminor" -lt "$UV_MIN_MINOR" ]; then
         die "uv $uvv is older than 0.$UV_MIN_MINOR, which this installer needs. Update it ($uv self update, or the way you installed it) and run this again."
       fi ;;
  esac

  bin=$("$uv" tool dir --bin 2>/dev/null) || die "uv could not say where it puts programs (uv tool dir --bin): update it ($uv self update, or the way you installed it) and run this again."
  if { [ -e "$bin/glyd" ] || [ -L "$bin/glyd" ]; } && ! "$uv" tool list 2>/dev/null | grep -q '^glyd '; then
    die "$bin/glyd is not Glyd's Python tool (it is the compression program, a pip install of glyd, or a program of your own). The tool would replace it, and this installer does not replace a program it did not make.
  To keep it, put the tool in another folder, which this installer then puts first on your PATH:
      curl -LsSf https://getglyd.com/install.sh | UV_TOOL_BIN_DIR=\$HOME/.glyd/bin sh
  Or remove $bin/glyd, and run this again."
  fi

  set -- tool install --managed-python --python "$PYTHON"
  constrained=
  if [ "$stack" = yes ] && [ "$arch" = x86_64 ] && [ "${GLYD_CONSTRAINTS:-}" != none ]; then
    constraints > "$tmp/constraints.txt"  # (what the acceptance run installed: for aarch64, none was run)
    set -- "$@" --constraints "$tmp/constraints.txt"
    constrained=yes
  fi
  if [ "$stack" = yes ]; then
    say "Installing $spec and Python $PYTHON (PyTorch and vLLM: several GB, a few minutes)"
    if [ -z "${CC:-}" ] && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
      set -- "$@" --with "ziglang==$ZIGLANG"
      say "No C compiler found, and vLLM needs one: adding ziglang, a compiler from PyPI (no sudo)"
    fi
  else
    say "Installing $spec and Python $PYTHON"
  fi
  resume="uv keeps what it downloaded."
  if [ -n "$constrained" ]; then resume="$resume If uv reports that versions conflict, the list at the end of this script is for glyd $GLYD_VERSION; GLYD_CONSTRAINTS=none in front of sh resolves the packages fresh."; fi
  "$uv" "$@" "$spec"
  resume=
  [ -x "$bin/glyd" ] || die "uv installed Glyd, but there is no $bin/glyd: run this again."

  case ":$PATH:" in
    *":$bin:"*) check_first "$bin" "run MODEL" ;;
    *) say "$bin is not on your PATH: adding it, which edits your shell's startup file (uv tool update-shell says which)"
       "$uv" tool update-shell || true
       say "Open a new terminal (or run: export PATH=\"$bin:\$PATH\") so that glyd is found." ;;
  esac

  say "Checking this machine (glyd doctor)"
  if "$bin/glyd" doctor; then
    [ "$stack" != yes ] || say "Next: glyd run Qwen/Qwen3.5-9B"
  else
    say "Fix the lines marked NO above, then run: glyd doctor"
  fi
}

# The versions the acceptance run installed (x86_64 Linux, Python 3.12, glyd[vllm] and ziglang): what is not listed is glyd itself.
# Regenerate from a run's logs/freeze.txt:  python3 scripts/install_constraints.py FREEZE
constraints() {
  cat <<'CONSTRAINTS'
agent-detector==2.0.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==1.11.0
anyio==4.15.1
apache-tvm-ffi==0.1.11
astor==0.8.1
attrs==26.1.0
blake3==1.0.10
cachetools==7.2.0
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
compressed-tensors==0.17.0
cryptography==50.0.2
cuda-bindings==13.4.3
cuda-core==1.2.1
cuda-pathfinder==1.8.2
cuda-python==13.4.1
cuda-tile==1.6.0
cuda-toolkit==13.0.3.0
depyf==0.20.0
detect-installer==0.2.1
dill==0.4.1
dnspython==2.8.0
docstring-parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.136.3
fastapi-cli==0.0.32
fastapi-cloud-cli==0.26.0
fastar==0.12.0
fastsafetensors==0.4.0
filelock==4.0.7
flashinfer-python==0.6.18.post1
frozenlist==1.8.0
fsspec==2026.9.0
googleapis-common-protos==1.75.5
grpcio==1.84.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpcore2==2.13.1
httptools==0.8.0
httpx==0.28.1
httpx2==2.13.1
huggingface-hub==1.33.0
humming-kernels==0.1.12
idna==3.20
ijson==3.5.1
instanttensor==0.2.0
interegular==0.3.3
jinja2==3.1.6
jiter==0.17.0
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==1.7.6
llvmlite==0.47.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
markupsafe==3.0.3
mcp==2.2.0
mcp-types==2.2.0
mdurl==0.1.2
mistral-common==1.12.0
ml-dtypes==0.6.0
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgspec==0.22.0
multidict==6.9.1
nccl4py==0.6.0
networkx==3.7
ninja==1.13.2
numba==0.65.0
numpy==2.3.5
nvidia-cublas==13.1.1.3
nvidia-cuda-cccl==13.3.4.3.1
nvidia-cuda-crt==13.4.92
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvcc==13.4.92
nvidia-cuda-nvdisasm==13.4.92
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cudnn-frontend==1.30.0
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-cutlass-dsl==4.7.1
nvidia-cutlass-dsl-libs-base==4.7.1
nvidia-cutlass-dsl-libs-core==4.7.1
nvidia-cutlass-dsl-libs-cu12==4.7.1
nvidia-cutlass-dsl-libs-cu13==4.7.1
nvidia-ml-py==13.615.71
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvvm==13.4.92
nvtx==0.2.15
openai==3.22.1
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
opentelemetry-api==1.45.0
opentelemetry-exporter-http-transport==0.66b0
opentelemetry-exporter-otlp==1.45.0
opentelemetry-exporter-otlp-common==0.66b0
opentelemetry-exporter-otlp-proto-common==1.45.0
opentelemetry-exporter-otlp-proto-grpc==1.45.0
opentelemetry-exporter-otlp-proto-http==1.45.0
opentelemetry-proto==1.45.0
opentelemetry-sdk==1.45.0
opentelemetry-semantic-conventions==0.66b0
opentelemetry-semantic-conventions-ai==0.5.1
outlines-core==0.2.14
packaging==26.3
partial-json-parser==0.2.1.1.post7
pillow==12.3.0
prometheus-client==0.26.0
prometheus-fastapi-instrumentator==8.1.0
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
py-cpuinfo==9.0.0
pybase64==1.5.0
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.5
pydantic-core==2.46.5
pydantic-extra-types==2.11.1
pydantic-settings==2.15.0
pygments==2.21.0
pyjwt==2.15.1
pynvvideocodec==2.0.4
python-dotenv==1.2.3
python-json-logger==4.2.0
python-multipart==0.0.32
pyyaml==6.0.3
pyzmq==27.2.0
quack-kernels==0.6.5
referencing==0.37.0
regex==2026.9.29
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.5
rignore==0.8.1
rpds-py==2026.6.3
safetensors==0.8.0
sentencepiece==0.2.2
sentry-sdk==2.71.0
setproctitle==1.3.7
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sse-starlette==3.5.0
starlette==1.7.0
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tiktoken==0.14.0
tilelang==0.1.12
tokenizers==0.23.2
tokenspeed-mla==0.1.8
tokenspeed-triton==3.8.10.post20260920
torch==2.13.0
torch-c-dlpack-ext==0.1.5
torchaudio==2.11.0
torchcodec==0.17.0
torchvision==0.28.0
tqdm==4.70.1
transformers==5.18.0
triton==3.7.1
truststore==0.10.4
typer==0.27.2
typing-extensions==4.16.0
typing-inspection==0.4.4
urllib3==2.8.0
uvicorn==0.54.0
uvloop==0.23.0
vllm==0.30.0
watchfiles==1.3.0
websockets==17.1
xgrammar==0.2.8
yarl==1.25.1
z3-solver==4.15.4.0
ziglang==0.16.0
CONSTRAINTS
}

main "$@" </dev/null
