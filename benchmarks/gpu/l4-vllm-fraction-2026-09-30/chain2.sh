#!/bin/bash
# After chain.sh: check_vllm.py --brief --fraction 0.5 on granite-3.1-3b-a800m-instruct (a mixture of experts: its layers'
# experts follow the rule), then a fraction Glyd refuses. Each queued on the box's lock.
B=~/budget
echo "== $(date -u +%T) chain2: granite check (waiting for the lock)"
flock ~/.glyd-box.lock bash $B/l4_check.sh 0.5 ibm-granite/granite-3.1-3b-a800m-instruct --brief > $B/chain2-granite.out 2>&1; echo "granite exit $? at $(date -u +%T)"
echo "== $(date -u +%T) chain2: a bad fraction (waiting for the lock)"
flock ~/.glyd-box.lock bash $B/bad_fraction.sh > $B/chain2-bad.out 2>&1; echo "bad exit $? at $(date -u +%T)"
echo "== $(date -u +%T) chain2: done"
