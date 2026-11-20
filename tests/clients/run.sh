#!/bin/bash
# The client matrix against one running server: every client's own check, each with its current release (pip's newest), in a virtual environment.
#
#   tests/clients/run.sh http://127.0.0.1:8238 [KEY]    (--fim: the model has fill-in-the-middle tokens, as Qwen's)
#
# Open WebUI (a container) is tests/clients/openwebui_check.py, run against a container set up as `glyd connect open-webui` says.
# Claude Code is run by hand: ANTHROPIC_BASE_URL=URL ANTHROPIC_API_KEY=any claude -p "Read hello.txt" --allowedTools Read
set -u
URL=$1; KEY=${2:-}; EXTRA="${*:3}"
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$(mktemp -d)/venv}
python3 -m venv "$VENV" && "$VENV/bin/pip" install -q openai anthropic ollama requests langchain-openai langchain-anthropic langchain-ollama || exit 99
KEYARG=(); [ -n "$KEY" ] && KEYARG=(--key "$KEY")
bad=0
for c in openai anthropic ollama langchain; do
  echo "== $c"
  case $c in openai | ollama) extra="$EXTRA" ;; *) extra="" ;; esac
  "$VENV/bin/python" "$HERE/${c}_check.py" "$URL" "${KEYARG[@]}" $extra || bad=$((bad + $?))
done
echo "== $bad failed in all"
exit $((bad > 0))
