#!/bin/bash
# dk.sh without the box lock (for a batch that holds it)
OPTS=(); while [ "$1" != "--" ]; do OPTS+=("$1"); shift; done; shift
exec docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --ipc=host --user $(id -u):$(id -g) \
  -e HOME=/work -e UV_CACHE_DIR=/work/uv-cache -v $HOME/accept/bin/uv:/usr/local/bin/uv:ro -v $HOME/accept/work:/work -v $HOME/hf:/hf -e HF_HOME=/hf "${OPTS[@]}" glyd-accept:ubuntu26.04 bash "$@"
