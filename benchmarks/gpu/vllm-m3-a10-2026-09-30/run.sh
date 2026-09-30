#!/usr/bin/env bash
# vLLM M3 on this GPU: the steps in turn (check bench), each with its own results dir, then all of them in ~/results and ALLDONE.
rm -rf ~/results; for s in check bench; do R=$HOME/results-$s VJ_STEPS=$s bash ~/vllm_job.sh > ~/vj-$s.log 2>&1; echo "$(date -u +%T) $s exit $?" >> ~/vj-steps.txt; done
mkdir -p ~/results; for s in check bench; do cp -r ~/results-$s ~/results/$s 2>/dev/null; done; cp ~/vj-*.log ~/vj-steps.txt ~/results/ 2>/dev/null; touch ~/results/DONE ~/results/ALLDONE
