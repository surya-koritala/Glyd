#!/usr/bin/env bash
# The release candidate on Hopper (a GH200 or H100 SXM: the decoder's follow-up wants a full-power Hopper), the final
# tree in r1_src.tar: r1_job.sh's self-test, xcheck (synthetic, then Qwen3-0.6B's), check_capi, the Rust step (the
# crate's tests, glyd pack against Python's, verify, the unpack and linear examples) and layer.py's runs 1 and 2
# (Qwen3-8B, 14B and 32B, layer 10 alone), alone; then the decoder test's follow-up (dec_more.sh, results/dec). No
# check starts past 7 minutes (BUDGET) or runs past 9 (END) from the environment, the decoder's end by 15: about 12
# minutes, 25 at most.
#   bash ~/r1_hopper_rc.sh   (in ~: this, r1_hopper.sh, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py,
#                             r1_rust.sh, dec_more.sh, dec_job.sh, dec_src.tar)
export R=${R:-$HOME/results}
trap 'mkdir -p "$R" && touch "$R/DONE"' EXIT
export ONLY=selftest,xcheck,capi,rust,xmodels,layer,layer2,dec MODELS=Qwen/Qwen3-0.6B LAYER_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B" DEC=1 BUDGET=420 END=540
bash "${FILES:-$HOME}/r1_hopper.sh"
