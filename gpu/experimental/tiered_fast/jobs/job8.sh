#!/usr/bin/env bash
cd ~/tf
H=./harness26
SW="o152u3w4b4s,o168u1w2b8s,o296u1w2b8s,o184u3w2b8s,o152u3w2b8s,o552u1w2b8s,o552u1w4b4s,o552u1w1b16s,o552u1w3b5s,o520u1w2b8s,o568u3w2b8s,o568u3w4b4s,o536u3w4b4s,"
NS="o152u3w4b4,o152u3w2b8,o184u3w2b8,o552u1w2b8,o552u1w4b4,o568u3w4b4,o568u3w2b8,"
ROUNDS=9 $H packs/o10 1 150 "n,l,o40u1w4,$SW" > ~/tf/j8_o10_m1.txt 2>&1
ROUNDS=7 $H packs/o10 8 150 "n,l,o40u1w4,$SW" > ~/tf/j8_o10_m8.txt 2>&1
ROUNDS=7 $H packs/o10 16 150 "n,l,o40u1w4,$NS" > ~/tf/j8_o10_m16.txt 2>&1
ROUNDS=5 $H packs/g1 1 150 "l,o40u1w4,o552u1w2b8s,o552u1w4b4s,o568u3w2b8s,o568u3w4b4s,o536u3w4b4s,o552u1w2b8,o568u3w4b4," > ~/tf/j8_g1_m1.txt 2>&1
COLD=1 ROUNDS=5 $H packs/g10 1 100 "n,l,o552u1w2b8s,o552u1w4b4s,o568u3w2b8s,o568u3w4b4s," > ~/tf/j8_g10_cold.txt 2>&1
COLD=1 ROUNDS=5 $H packs/u10 1 60 "n,l,o552u1w2b8s,o552u1w4b4s,o568u3w2b8s,o568u3w4b4s," > ~/tf/j8_u10_cold.txt 2>&1
for v in o568u3w2b8s o552u1w2b8s; do
  ./ncu_h.sh $HOME/tf/ncu/s_$v "step_kernel" packs/o10 1 "$v," 2 $HOME/tf/harness26
done
echo done > ~/tf/j8_done.txt
