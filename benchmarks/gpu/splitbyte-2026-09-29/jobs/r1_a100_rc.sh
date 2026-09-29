#!/usr/bin/env bash
# The release candidate on an A100, the final tree in r1_src.tar: r1_job.sh's self-test, xcheck (synthetic, then every
# model's: Qwen3-0.6B, 1.7B, 4B-Instruct-2507, 8B and granite-3.1-3b-a800m-instruct, downloaded whole) and layer.py's
# runs 1 and 2 (Qwen3-8B, 14B's layer 10), alone; then the decoder test's follow-up (dec_more.sh, results/dec). No
# check starts past 7 minutes (BUDGET) or runs past 9 (END) from the environment, the decoder's end by 15: about 11
# minutes, 25 at most.
#   bash ~/r1_a100_rc.sh     (in ~: this, r1_a100.sh, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py,
#                             r1_rust.sh, dec_more.sh, dec_job.sh, dec_src.tar)
export R=${R:-$HOME/results}
trap 'mkdir -p "$R" && touch "$R/DONE"' EXIT
export ONLY=selftest,xcheck,xmodels,layer,layer2,dec DEC=1 BUDGET=420 END=540
bash "${FILES:-$HOME}/r1_a100.sh"
