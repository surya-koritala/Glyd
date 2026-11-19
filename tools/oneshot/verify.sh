#!/bin/bash
# The release checks on the clean install (reltest2): precise mode (already running on 8080), then fast mode.
set -u
R=$HOME/glyd-scratch/reltest2; Q=$HOME/glyd-scratch/qwen; cd $R/repo; . $R/home/env.sh; O=$R/out; mkdir -p $O
wait_ready() { for i in $(seq 120); do curl -s localhost:8080/v1/glyd/status | grep -q '"state":"idle"' && return; sleep 2; done; }
run_mode() {
  m=$1; wait_ready
  echo "== $m $(date +%T)"
  python3 tests/test_server.py --url http://127.0.0.1:8080 > $O/$m-test_server.txt 2>&1; tail -1 $O/$m-test_server.txt
  python3 $Q/cancel_test.py http://127.0.0.1:8080 > $O/$m-cancel.txt 2>&1; tail -1 $O/$m-cancel.txt
  (cd $O && bash $R/repo/tools/bench.sh > $m-bench.txt 2>&1 && mv bench-report.md $m-bench-report.md); cat $O/$m-bench.txt
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"glyd_ask","arguments":{"prompt":"2+2? Just the number.","max_tokens":10}}}' | $GLYD_ENGINE qwenmcp --repo $R/repo > $O/$m-mcp.txt 2>&1; cat $O/$m-mcp.txt
  GLYD_QWEN=$R/home bash bench/run_bench.sh http://127.0.0.1:8080/v1 $O/bench-$m > $O/$m-run_bench.txt 2>&1
  for t in gsm8k mmlu_pro; do python3 -c "import json; r=json.load(open('$O/bench-$m/$t.json')); print('$m $t', r if not isinstance(r, list) else r[0])" 2>&1 | cut -c1-300; done
}
run_mode precise
for p in $(pgrep -f "reltest2/home/bin/glyd-engine qwenserve"); do kill $p; done; sleep 8
(GLYD_QWEN=$R/home nohup flock -w 7200 $HOME/glyd-scratch/gpu.lock $Q/gpurun.sh bash tools/serve.sh --fast 8080 > $R/serve-fast.log 2>&1 < /dev/null &)
sleep 20; run_mode fast
echo VERIFY-DONE
