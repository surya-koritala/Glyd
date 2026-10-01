#!/bin/bash
# usage: rv.sh ENVDIR NAME FREE_MIB UTIL [-e VAR=val ...]    one server start in the clean container, the hog, the probe, memwatch
ENVDIR=$1; NAME=$2; FREE=$3; UTIL=$4; shift 4
W=$HOME/accept/work/logs
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > $W/$NAME.csv &
MW=$!
cd ~/accept && ${DK:-./dk.sh} -e PYTHONPATH=/work/probe -e GLYD_PROBE_GEFORCE=1 -e GLYD_PROBE=1 -e GLYD_PROBE_SNAP=1 "$@" -- /work/run1.sh $ENVDIR $NAME $FREE $UTIL > $W/$NAME.out 2>&1
kill $MW
T=22565
awk -F", " -v T=$T "{u=\$2+0; if (u>m) m=u} END {printf \"memwatch: most used %d MiB of %d, so least free %d MiB\n\", m, T, T-m}" $W/$NAME.csv | tee -a $W/$NAME.out
cat $W/$NAME.out | cut -c1-260
grep -E "\[probe\] (at the end|packed model.layers.35.self_attn.qkv)" $W/$NAME.log | cut -c1-260
