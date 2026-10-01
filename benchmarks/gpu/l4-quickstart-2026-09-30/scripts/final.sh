#!/bin/bash
# the merged release tree's wheel (7f2b218, 0.26.0rc2) through the merged tree's acceptance.sh: a 16 GB card, then no hog (24 GB class)
export HF_HOME=$HOME/hf PATH=$HOME/accept/bin:$PATH
unset PYTHONPATH
cd ~/accept/repo-final/gpu/vllm
W=$HOME/accept/wheels-final/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl
OUT=~/accept/final.out; : > $OUT
echo "=== 16 GB card, both Open WebUI routes ($(date -u +%T))" >> $OUT
bash acceptance.sh --wheel $W --card 4080s --work ~/accept/work > ~/accept/acc-final.out 2>&1
mkdir -p ~/accept/acc-final-logs && cp -a ~/accept/work/logs/server.log ~/accept/work/logs/install.log ~/accept/work/logs/freeze.txt ~/accept/work/logs/hog.log ~/accept/work/logs/chat.log ~/accept/work/logs/webui-*.log ~/accept/work/logs/webui-*.check.log ~/accept/work/summary.txt ~/accept/work/serve.sh ~/accept/acc-final-logs/ 2>/dev/null
cat ~/accept/acc-final.out >> $OUT
echo "=== no hog, 0.88 of the L4's 22 GB, the plugin reading GeForce Ada, no Open WebUI ($(date -u +%T))" >> $OUT
bash acceptance.sh --wheel $W --geforce --webui none --work ~/accept/work > ~/accept/acc-final-24.out 2>&1
mkdir -p ~/accept/acc-final-24-logs && cp -a ~/accept/work/logs/server.log ~/accept/work/logs/install.log ~/accept/work/logs/freeze.txt ~/accept/work/logs/chat.log ~/accept/work/summary.txt ~/accept/work/serve.sh ~/accept/acc-final-24-logs/ 2>/dev/null
cat ~/accept/acc-final-24.out >> $OUT
echo "done $(date -u +%T)" >> $OUT
