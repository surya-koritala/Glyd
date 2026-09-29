#!/usr/bin/env bash
# bash cu12_fetch.sh DIR: NVIDIA's CUDA 12.8.2 redistributables (nvcc 12.8.93, cudart and cccl 12.8.90: the CUDA 12
# the release builds with, nvidia/cuda:12.8.2-devel) into DIR, each archive checked against the release's manifest
# sha256. A build tool only. Exit 1 if any step fails.
set -u
D=$1; B=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p "$D/dl"
curl -sfL --retry 3 -o "$D/dl/manifest.json" "$B/redistrib_12.8.2.json" || { echo "cu12_fetch: no manifest"; exit 1; }
for c in cuda_nvcc cuda_cudart cuda_cccl; do
  set -- $(python3 -c "import json; x = json.load(open('$D/dl/manifest.json'))['$c']['linux-x86_64']; print(x['relative_path'], x['sha256'])")
  curl -sfL --retry 3 -o "$D/dl/$c.tar.xz" "$B/$1" || { echo "cu12_fetch: $c download failed"; exit 1; }
  echo "$2  $D/dl/$c.tar.xz" | sha256sum -c --quiet - || { echo "cu12_fetch: $c sha256 mismatch"; exit 1; }
  tar -xJf "$D/dl/$c.tar.xz" -C "$D" --strip-components=1 || { echo "cu12_fetch: $c extract failed"; exit 1; }
done
rm -rf "$D/dl"
"$D/bin/nvcc" --version | tail -2
