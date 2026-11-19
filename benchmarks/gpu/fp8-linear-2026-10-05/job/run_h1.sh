#!/usr/bin/env bash
S=/private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad
cat > $S/f8f/job_h1.sh <<'J'
export F8_STEPS="bench" F8_BENCH_ARGS="--no-ours"
exec bash ~/f8f_job.sh
J
GPU="NVIDIA H100 80GB HBM3" MAX_MIN=45 NAME=f8f-h1 UPLOAD="$S/f8f/f8f_job.sh $S/f8f/f8f_src.tar" bash $S/rp_run.sh $S/f8f/job_h1.sh
