#!/usr/bin/env bash
# The model from Hugging Face (no account needed), glyd/Qwen3.8-Flash-Next-hummingbird: model/ (Qwen's tokenizer and configs, and the weights
# outside the experts, 37 GB) and the experts for one mode: precise-v2/ (45 GB, the default), with --fast fast/ (29 GB), with --best best-v2/ (53 GB). The files go to the
# Hugging Face cache (~/.cache/huggingface, or $HF_HUB_CACHE), which other tools share; DIR gets links to them. It resumes if it stops.
#   tools/download.sh [--fast | --best] [DIR]        (default DIR $GLYD_QWEN/hf)
set -euo pipefail
. "${GLYD_QWEN:-$HOME/.glyd-qwen-3.8-flash}/env.sh"
MODE=--precise; case "${1:-}" in --fast) MODE=; shift;; --best) MODE=--best; shift;; esac
# Hugging Face counts downloads by requests for the top-level config.json, so ask for it once (a HEAD request, nothing else).
curl -sfI -o /dev/null https://huggingface.co/glyd/Qwen3.8-Flash-Next-hummingbird/resolve/main/config.json || true
exec "$GLYD_ENGINE" qwenget glyd/Qwen3.8-Flash-Next-hummingbird "${1:-$GLYD_QWEN/hf}" $MODE
