#!/bin/bash
# The L4 runs one after another, each queued on the box's lock when the one before it has ended (not holding it between them,
# so another helper's job can take its turn): check_vllm.py --quick --fraction 0.5 on Qwen3-8B, then the L4 sanity sweep (bf16,
# glyd@0, glyd@0.5, glyd@1), then the GH200 job's dry run. Each stage whatever the one before gave.
B=~/budget
echo "== $(date -u +%T) chain: check (waiting for the lock)"
flock ~/.glyd-box.lock bash $B/l4_check.sh 0.5 Qwen/Qwen3-8B --quick > $B/chain-check.out 2>&1; echo "check exit $? at $(date -u +%T)"
echo "== $(date -u +%T) chain: sweep (waiting for the lock)"
flock ~/.glyd-box.lock bash $B/l4_sweep.sh > $B/chain-sweep.out 2>&1; echo "sweep exit $? at $(date -u +%T)"
echo "== $(date -u +%T) chain: dry run (waiting for the lock)"
flock ~/.glyd-box.lock bash $B/dry.sh > $B/chain-dry.out 2>&1; echo "dry exit $? at $(date -u +%T)"
echo "== $(date -u +%T) chain: done"
