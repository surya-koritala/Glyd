#!/usr/bin/env bash
# the respond job's smoke test on the dev L4: Qwen3-0.6B's four modes, 15 minutes, the models from the cache
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
echo "queued $(date -u +%T)" > ~/resptest/queue.log
flock ~/.glyd-box.lock bash -c 'echo "started $(date -u +%T)" >> ~/resptest/queue.log; touch ~/.glyd-busy
  cd ~/resptest && HF_HOME=~/hf HF_HUB_OFFLINE=1 FILES=~/resptest R=~/resptest/results W=~/resptest/w RESP_END=900 \
    RESP_RUNS="Qwen/Qwen3-0.6B:glyd:150 Qwen/Qwen3-0.6B:bf16c:150 Qwen/Qwen3-0.6B:bf16:60 Qwen/Qwen3-0.6B:exact:60" \
    bash ~/resptest/resp_job.sh > ~/resptest/run.log 2>&1
  echo "ended $(date -u +%T) exit $?" >> ~/resptest/queue.log'
