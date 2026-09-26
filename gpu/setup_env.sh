#!/usr/bin/env bash
# The Python environment gpu/ runs in, without root: uv, Python 3.12,
# PyTorch (CUDA 13), and nvcc with its headers pinned to the CUDA PyTorch
# was built with (the extension builds on first import).
#   bash gpu/setup_env.sh [ENV_DIR]      then: source ENV_DIR/cuda.sh
set -euo pipefail
ENV="${1:-$HOME/gpuenv}"
mkdir -p "$HOME/tools/uv"
[ -x "$HOME/tools/uv/uv" ] || curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1 -C "$HOME/tools/uv"
UV="$HOME/tools/uv/uv"
"$UV" venv -q --python 3.12 "$ENV"
"$UV" pip install -q --python "$ENV/bin/python" torch safetensors numpy transformers accelerate hf_transfer ninja datasets
CUDA_MM=$("$ENV/bin/python" -c "import torch; print(torch.version.cuda)")  # e.g. 13.0
"$UV" pip install -q --python "$ENV/bin/python" "nvidia-cuda-nvcc==$CUDA_MM.*" "nvidia-cuda-cccl==$CUDA_MM.*" "nvidia-cuda-crt==$CUDA_MM.*" "nvidia-nvvm==$CUDA_MM.*"
CUDA_HOME="$("$ENV/bin/python" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$CUDA_MM'.split('.')[0]))")"
mkdir -p "$CUDA_HOME/lib64"
ln -sf "../lib/$(ls "$CUDA_HOME/lib" | grep -m1 '^libcudart.so')" "$CUDA_HOME/lib64/libcudart.so"
cat > "$ENV/cuda.sh" <<EOF
export CUDA_HOME=$CUDA_HOME
export PATH=$ENV/bin:\$CUDA_HOME/bin:\$PATH
EOF
source "$ENV/cuda.sh"
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.device_count(), 'GPUs', torch.cuda.get_device_name(0))"
nvcc --version | tail -1
