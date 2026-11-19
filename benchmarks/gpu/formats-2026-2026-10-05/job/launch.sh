#!/usr/bin/env bash
# Tries GPUs in order until a pod is created; writes launch.status: running / done / failed.
S=/private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad
echo running > $S/fmt/launch.status
for G in "NVIDIA RTX 4000 Ada Generation" "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA GeForce RTX 4090" "NVIDIA L40"; do
  rm -rf $S/ceiling/results-fmt2026
  echo "== trying $G"
  GPU="$G" MAX_MIN=90 NAME=fmt2026 MIN_DRIVER=550 RP_CUDA=12.8 RP_DISK=60 UPLOAD="$S/fmt/fmt_measure.py" bash $S/rp_run.sh $S/fmt/fmt_job.sh; rc=$?
  echo "== rc $rc ($G)"
  if [ -f $S/ceiling/results-fmt2026/DONE ]; then echo done > $S/fmt/launch.status; exit 0; fi
done
echo failed > $S/fmt/launch.status
