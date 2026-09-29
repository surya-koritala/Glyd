#!/usr/bin/env bash
# Round 1 (v0.25.0) on Hopper (a GH200, else an H100 PCIe or SXM): r1_job.sh with this class's lists (sb_hopper.sh's),
# then the decoder test's follow-up (DEC=1: dec_more.sh). About 40 minutes: the checks 29 at most, the decoder's 6-8
# (14.5 at most). The release candidate's run: r1_hopper_rc.sh.
#   bash ~/r1_hopper.sh      (in ~: this, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py, r1_rust.sh,
#                             dec_more.sh, dec_job.sh, dec_src.tar)
export CLASS=hopper TIME_MODELS=${TIME_MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B"} MS=${MS:-1,16,32,64,128,256,512,1024}
export LAYER_MODELS=${LAYER_MODELS-"Qwen/Qwen3-14B Qwen/Qwen3-32B"} DEC=${DEC:-1}
exec bash "${FILES:-$HOME}/r1_job.sh"
