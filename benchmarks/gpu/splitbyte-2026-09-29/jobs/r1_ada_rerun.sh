#!/usr/bin/env bash
# Round 1's Ada rerun on an AWS g6 (L4), the release tree at 8a7769e: xcheck on granite (xmodels), test_gpu (the host's
# LD_LIBRARY_PATH unset), the attribution (attr) and the routes check, alone; granite whole and Qwen3-8B's layer 10 the
# only downloads. About 10 minutes, the checks ending by 21 minutes (END) past the environment.
#   bash ~/r1_ada_rerun.sh   (in ~: this, r1_ada.sh, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py, r1_rust.sh)
export ONLY=routes,test_gpu,attr,xmodels MODELS=ibm-granite/granite-3.1-3b-a800m-instruct LAYER_MODELS=Qwen/Qwen3-8B TIME_MODELS=Qwen/Qwen3-8B BUDGET=1080 END=1260
exec bash "${FILES:-$HOME}/r1_ada.sh"
