#!/usr/bin/env bash
# Option 2 built (gpu-option2) on Hopper (an H100 SXM or a GH200; an H100 PCIe runs it too, its route at 1024 alone):
# o2_job.sh with this class's lists. Qwen3-8B and Qwen3-32B, whole (about 82 GB): layer 10 per layer at 1023-8192
# tokens (1023 below the route), end to end at 1024-8192 with the breakdown at 1024 and 4096; Qwen3-8B again with the
# last session's scheduling (GLYD_SPLIT_SLOTS=3) and with the decode on 12 and 28 SMs at 1024. About 20 minutes (25
# where the environment is made here); 28 at most.
# The stress check first (split_stress.py; quick, on the tree before its fix, with o2_old_src.tar).
#   bash ~/o2_hopper.sh      (in ~: this, o2_job.sh, o2_src.tar, o2_old_src.tar, o2_summary.py)
export CLASS=hopper MODELS=${MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-32B"}
export LAYER_MS=${LAYER_MS:-1023,1024,2048,4096,8192} E2E_MS=${E2E_MS:-1024,2048,4096,8192} BREAK_MS=${BREAK_MS:-1024,4096}
export SWEEP_SMS=${SWEEP_SMS:-"12 28"} SWEEP_MS=${SWEEP_MS-1024}
exec bash "${FILES:-$HOME}/o2_job.sh"
