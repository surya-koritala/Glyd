#!/usr/bin/env bash
# Ada: the checks (with the unpack's bandwidth) and the bench on an RTX 4090 (else an L40S, else an RTX 6000 Ada), one pod at a time
S=/private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad
cat > $S/f8f/job_a2.sh <<'J'
export F8_STEPS="check bench" F8_CHECK_ARGS="" F8_BENCH_ARGS="--numerics --variant GLYD_F8_LIN_CTAS=3"
exec bash ~/f8f_job.sh
J
for gpu in "NVIDIA GeForce RTX 4090" "NVIDIA L40S" "NVIDIA RTX 6000 Ada Generation"; do
  GPU="$gpu" MAX_MIN=45 NAME=f8f-a2 UPLOAD="$S/f8f/f8f_job.sh $S/f8f/f8f_src.tar $S/f8f/ncu_f8.py" bash $S/f8f/rp_run_a2.sh $S/f8f/job_a2.sh && break
  echo "== $gpu: no run, the next"
  sleep 20
done
