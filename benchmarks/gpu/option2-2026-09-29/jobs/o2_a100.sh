#!/usr/bin/env bash
# Option 2 built (gpu-option2) on an A100 (40 or 80 GB): o2_job.sh with this class's lists. The stress check
# (split_stress.py) on this tree and, quick, on the tree before its fix (o2_old_src.tar: it fails there); Qwen3-8B and
# Qwen3-14B, whole (about 46 GB): layer 10 per layer at 768-8192 tokens (768 below the route, 4097 past 14B's gate and
# up by it), end to end at 769-8192 with the breakdown at 1024 and 2048. The last session's scheduling and SM sweep
# are left out (SKIP=slots3, no SWEEP_MS: measured). About 15 minutes (20 where the environment is made here); 28 at most.
#   bash ~/o2_a100.sh        (in ~: this, o2_job.sh, o2_src.tar, o2_old_src.tar, o2_summary.py)
export CLASS=a100 MODELS=${MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-14B"}
export LAYER_MS=${LAYER_MS:-768,769,1024,2048,4096,4097,8192} E2E_MS=${E2E_MS:-769,1024,2048,4096,8192} BREAK_MS=${BREAK_MS:-1024,2048}
export SWEEP_SMS=${SWEEP_SMS:-"8 16"} SWEEP_MS=${SWEEP_MS-} SKIP=${SKIP-slots3}
exec bash "${FILES:-$HOME}/o2_job.sh"
