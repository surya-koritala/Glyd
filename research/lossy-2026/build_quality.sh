#!/usr/bin/env bash
# Build the prepared A100 study artifacts on CI, before any paid GPU session.
set -euo pipefail
DEST=${1:?output directory}
: "${GITHUB_SHA:?CI source revision required}"
mkdir -p "$DEST"
export GLYD_ENGINE_ARCHS=80
cargo test --release -p glyd-engine
cargo test --release --manifest-path research/lossy-2026/quality/Cargo.toml
cargo build --release -p glyd-engine
cargo build --release --manifest-path research/lossy-2026/quality/Cargo.toml
cargo build --release --manifest-path research/lossy-2026/rust/Cargo.toml --bin glyd-lossy-inventory
CU=$(dirname "$(command -v nvcc)")/..
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include"
   -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode arch=compute_80,code=sm_80 -Xcompiler -fPIC,-fvisibility=hidden)
nvcc "${F[@]}" -c gpu/glyd_gpu.cu -o "$DEST/glyd_gpu.o"
nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib64" -o "$DEST/libglyd_gpu_cuda13.so" "$DEST/glyd_gpu.o"
rm "$DEST/glyd_gpu.o"
cp target/release/glyd-engine "$DEST/"
cp research/lossy-2026/quality/target/release/glyd-lossy-quality "$DEST/"
cp research/lossy-2026/rust/target/release/glyd-lossy-inventory "$DEST/"
printf '%s\n' "$GITHUB_SHA" > "$DEST/COMMIT"
cp research/lossy-2026/{models.json,quality_job.sh,quality_reference.py,quality_compare.py,quality_env.sh,quality_env.py,quality-requirements.txt,pod-image.txt,matched_quality.py,quality_match_job.sh,check_matched.py,qwen38-inventory.json,qwen38-config.json} "$DEST/"
(cd "$DEST" && sha256sum glyd-engine glyd-lossy-quality libglyd_gpu_cuda13.so COMMIT matched_quality.py quality_match_job.sh check_matched.py qwen38-inventory.json qwen38-config.json glyd-lossy-inventory models.json quality_job.sh quality_reference.py quality_compare.py quality_env.sh quality_env.py quality-requirements.txt pod-image.txt > SHA256SUMS)
