#!/usr/bin/env bash
# Round 1 (v0.25.0) on Ada: an AWS g6's L4 (its code 89: the grid prompt kernels, as an L40S
# takes them), or a GeForce card (RTX 4080 SUPER, 4090: the stream-K kernels, then the grid kernels through a patched
# build, step k): r1_job.sh with this class's lists (sb_ada.sh's) and the attribution (ATTR=1, sb_job.sh's attr step).
# 29 minutes at most.
#   bash ~/r1_ada.sh         (in ~: this, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py, r1_rust.sh)
export CLASS=ada TIME_MODELS=${TIME_MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507"} MS=${MS:-1,8,32,256,1024}
export LAYER_MODELS=${LAYER_MODELS:-""} ATTR=${ATTR:-1}
exec bash "${FILES:-$HOME}/r1_job.sh"
