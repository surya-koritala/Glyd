#!/usr/bin/env bash
cd ~/tf
H=./harness19
SW="o168u1w4b4s,o168u1w4b5s,o168u1w4b6s,o168u1w2b8s,o168u1w2b10s,o168u1w2b12s,o168u1w1b16s,o168u1w1b20s,o168u1w1b24s,o232u1w4b5s,o152u3w4b4s,o152u3w4b3s,o184u3w4b4s,o1064u1w4b5s,o1064u1w4b6s,o7208u1w4b5s,o7208u1w4b6s,o8u1w4b5s,o40u1w4b4s,"
NS="o168u1w4b4,o168u1w4b5,o168u1w4b6,o168u1w2b8,o168u1w2b10,o168u1w2b12,o168u1w1b16,o168u1w1b20,o168u1w1b24,o152u3w4b4,o152u3w4b3,o184u3w4b4,o184u3w4b3,o1064u1w4b5,o7208u1w4b5,"
ROUNDS=7 $H packs/o10 1 150 "n,l,o8u1,o40u1,o8u1w4,o40u1w4,o24u3w4b3,$NS$SW" > ~/tf/j5_o10_m1.txt 2>&1
ROUNDS=5 $H packs/o10 8 150 "n,l,o8u1,o40u1,o8u1w4,o8u1w8b0s,$SW" > ~/tf/j5_o10_m8.txt 2>&1
ROUNDS=5 $H packs/o10 16 150 "n,l,o8u1,o40u1,o8u1w4,o40u1w4,$NS" > ~/tf/j5_o10_m16.txt 2>&1
COLD=1 ROUNDS=5 $H packs/g10 1 100 "n,l,o40u1w4,o168u1w4b5,o168u1w2b10,o168u1w1b20,o168u1w4b5s,o168u1w2b10s,o168u1w1b20s,o152u3w4b4s,o1064u1w4b5s," > ~/tf/j5_g10_cold.txt 2>&1
COLD=1 ROUNDS=5 $H packs/u10 1 60 "n,l,o40u1w4,o168u1w4b5,o168u1w2b10,o168u1w4b5s,o168u1w2b10s,o168u1w1b20s,o1064u1w4b5s," > ~/tf/j5_u10_cold.txt 2>&1
for v in o168u1w4b5s o168u1w4b4 o24u3w4b3 o168u1w1b20s; do
  ./ncu_h.sh $HOME/tf/ncu/q_$v "step_kernel" packs/o10 1 "$v," 2 $HOME/tf/harness19
done
echo done > ~/tf/j5_done.txt
