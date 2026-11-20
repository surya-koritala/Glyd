#!/usr/bin/env bash
# The engine and vLLM under the same load, alternated (engine1 was run first; then): vllm1, engine2, vllm2 (two of each, to show how far one run is from the next), each only with no CI job on the GPU.
#   pair.sh OUTROOT
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for run in vllm1 engine2 vllm2; do bash "$HERE/until_clean.sh" "${run%[0-9]}" "$1/$run"; done
echo all done
