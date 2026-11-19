#!/usr/bin/env bash
# ncu on one launch of a kernel of the harness: ncu_h.sh OUT KERNEL_REGEX(demangled) PACK M WHICH [SKIP=2] [BIN]
source ~/gpuenv/cuda.sh
OUT=$1; K=$2; PACK=$3; M=${4:-1}; V=${5:-l}; SKIP=${6:-2}; BIN=${7:-$HOME/tf/harness2}
NCU=/usr/local/cuda-13.2/bin/ncu
cd ~/tf
sudo -n -E env "PATH=$PATH" ROUNDS=1 $NCU --cache-control none --clock-control none --kernel-name-base demangled \
  -k regex:"$K" --launch-skip $SKIP --launch-count 1 --set full --import-source yes -f -o $OUT \
  $BIN $PACK $M 1 $V 9>&- > $OUT.log 2>&1
sudo -n -E env "PATH=$PATH" $NCU -i $OUT.ncu-rep --page raw --csv > $OUT.raw 2>/dev/null
sudo -n -E env "PATH=$PATH" $NCU -i $OUT.ncu-rep --page raw --csv --print-metric-instances details --metrics sass__inst_executed_per_opcode,sass__inst_executed_per_opcode_pipeline > $OUT.mix 2>/dev/null
sudo -n -E env "PATH=$PATH" $NCU -i $OUT.ncu-rep --page source --csv --print-source cuda,sass > $OUT.src2 2>/dev/null
sudo -n chown $USER $OUT.* 2>/dev/null
