#!/usr/bin/env bash
# The quality benchmarks behind the README's numbers: GSM8K 150 and MMLU-Pro 150 (Inspect AI's tasks
# as shipped, every second question, the first 150), greedy, thinking off, a 3,072-token cap; against a server from tools/serve.sh.
#   bench/run_bench.sh [BASE_URL] [OUT]       (defaults http://127.0.0.1:8080/v1 and ./bench-out)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:-http://127.0.0.1:8080/v1}; OUT=${2:-./bench-out}
. "${GLYD_QWEN:-$HOME/.glyd-qwen-3.8-flash}/env.sh"
EV=$GLYD_QWEN/venv-eval
[ -x "$EV/bin/python" ] || { python3 -m venv "$EV" && "$EV/bin/pip" install -q inspect-ai==0.3.278 inspect-evals==0.24.0 openai==3.28.0; }
for task in gsm8k mmlu_pro; do
  for s in 0 25 50 75 100 125; do
    GLYD_API_KEY=none "$EV/bin/python" "$HERE/eval_slice.py" "$BASE" qwen3.8-flash-next $task 1/2 150 $s $((s + 25)) "$OUT/$task-$s"
  done
  "$EV/bin/python" "$HERE/merge_slices.py" "$OUT/$task.json" "$OUT/$task-"*
done
