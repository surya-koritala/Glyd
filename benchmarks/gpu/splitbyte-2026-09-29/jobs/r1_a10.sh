#!/usr/bin/env bash
# Round 1 (v0.25.0) on an A10 (24 GB; its route class, 2086, checked on the device): first, where dec_job.sh and
# dec_src.tar are in ~ and DEC_QUICK is not 0, the decoder test's 5-minute smoke run (QUICK=1 MODELS=Qwen3-8B, the GPU
# to itself); then r1_job.sh with this class's lists (sb_a10.sh's) and the attribution (ATTR=1, sb_job.sh's attr step).
# About 35 minutes: the smoke run 6, the checks 29.
#   bash ~/r1_a10.sh         (in ~: this, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py, r1_rust.sh;
#                             dec_job.sh and dec_src.tar for the smoke run)
export CLASS=a10 TIME_MODELS=${TIME_MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507"} MS=${MS:-1,8,32,256,640,1024}
export LAYER_MODELS=${LAYER_MODELS:-""} DEC_QUICK=${DEC_QUICK:-1} ATTR=${ATTR:-1}
exec bash "${FILES:-$HOME}/r1_job.sh"
