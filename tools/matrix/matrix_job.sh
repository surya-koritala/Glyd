#!/usr/bin/env bash
# One GPU class's cells of the release matrix on one pod (rp_run.sh contract: results in ~/results, ~/results/DONE at the end). Uploaded beside it: matrix.env
# (VERSION, MODELS, LEVELS, PROMPT, CAP_MIN). Installs the tag's release as a user does, then `glyd run MODEL:LEVEL --no-think --context CTX -p PROMPT` for each
# model x level x context (4096, 32768): c<N>.out / c<N>.err per cell and a line in cells.tsv (model level ctx exit seconds N); a cell that would end past CAP_MIN is skipped.
set -u
R=~/results; mkdir -p $R; cd ~
. ~/matrix.env
trap 'touch $R/DONE' EXIT
T0=$(date +%s)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $R/gpu.txt
curl -LsSf "https://raw.githubusercontent.com/surya-koritala/Glyd/v$VERSION/scripts/install.sh" | GLYD_VERSION=$VERSION sh > $R/install.log 2>&1; echo "install: rc=$?" > $R/steps.txt
export PATH=$HOME/.local/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
glyd --version >> $R/steps.txt 2>&1
: > $R/cells.tsv; n=0
for m in $MODELS; do for l in $LEVELS; do for ctx in 4096 32768; do
  n=$((n + 1)); left=$(( CAP_MIN * 60 - ($(date +%s) - T0) ))
  [ $left -gt 120 ] || { echo "skipped cell $n ($m:$l $ctx): cap reached" >> $R/steps.txt; continue; }
  t=$(date +%s)
  timeout -k 20 $(( left < 1500 ? left : 1500 )) glyd run "$m:$l" --no-think --context $ctx -p "$PROMPT" > $R/c$n.out 2> $R/c$n.err < /dev/null
  echo -e "$m\t$l\t$ctx\t$?\t$(( $(date +%s) - t ))\t$n" >> $R/cells.tsv
done; done; done
nvidia-smi > $R/gpu-final.txt
