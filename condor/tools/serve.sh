#!/usr/bin/env bash
# The chat page and the API (OpenAI and Anthropic) on http://127.0.0.1:PORT (default 8080).
#   tools/serve.sh [--fast | --best] [PORT]   precise mode by default; --fast or --best uses those experts (tools/download.sh --fast or --best)
#   GLYD_CTX=65536 tools/serve.sh         the context in tokens (default 32768, up to 262144)
#   GLYD_USERS=2 tools/serve.sh           requests answered at once (default 4, 1 to 8)
#   GLYD_GPUS=0,1 tools/serve.sh          split the layers over two cards (the second card's VRAM holds more experts)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "${GLYD_QWEN:-$HOME/.glyd-qwen-3.8-flash}/env.sh"
DIR=${GLYD_HF:-$GLYD_QWEN/hf}
F=; case "${1:-}" in --fast | --best) F=$1; shift;; esac
# (precise and best keep the always-on matrices at six bits: their calibrated experts were packed for that)
case $F in
  --fast) PACK=$DIR/fast; MODE=;;
  --best) PACK=$DIR/best; MODE="--no-side --layout i6 --hi i6";;
  *) PACK=$DIR/precise; MODE="--precise --layout i6 --hi i6";;
esac
# (the 3- and 4-bit experts are precise-v2 and best-v2: those where downloaded, else the earlier ones)
[ "$F" != --fast ] && [ -e "$PACK-v2" ] && PACK=$PACK-v2
[ -e "$PACK" ] || { echo "$PACK is missing: run tools/download.sh${F:+ $F} first." >&2; exit 1; }
exec "$GLYD_ENGINE" qwenserve "$DIR/model" "$PACK" $MODE --ctx "${GLYD_CTX:-32768}" --port "${1:-8080}" --page "$HERE" \
  ${GLYD_USERS:+--users "$GLYD_USERS"} ${GLYD_GPUS:+--gpus "$GLYD_GPUS"}
