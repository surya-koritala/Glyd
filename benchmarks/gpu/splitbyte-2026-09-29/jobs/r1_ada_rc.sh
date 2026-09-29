#!/usr/bin/env bash
# The release candidate on Ada (an AWS g6's L4), the final tree in r1_src.tar: r1_job.sh's layer.py runs 1 and 2 and the
# attribution (Qwen3-8B and Qwen3-4B-Instruct-2507, layer 10 alone the downloads), alone. No step starts past 6 minutes
# (BUDGET) or runs past 8 (END) from the environment (made here in about half a minute on a Deep Learning AMI): about
# 5 minutes, under 12.
#   bash ~/r1_ada_rc.sh      (in ~: this, r1_ada.sh, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py, r1_rust.sh)
export R=${R:-$HOME/results}
trap 'mkdir -p "$R" && touch "$R/DONE"' EXIT
export ONLY=layer,layer2,attr MODELS= LAYER_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507" BUDGET=360 END=480
bash "${FILES:-$HOME}/r1_ada.sh"
