#!/usr/bin/env bash
cd ~/tf
H=./harness24
SW="o152u3w4b4s,o264u1w4b4s,o296u1w4b4s,o296u1w4b5s,o296u1w4b6s,o296u1w2b8s,o296u1w8b0s,o280u3w4b4s,o280u3w4b3s,o312u3w4b4s,o312u3w4b3s,"
NS="o152u3w4b4,o264u1w4b4,o296u1w4b4,o296u1w4b5,o296u1w2b8,o280u3w4b4,o280u3w4b3,o312u3w4b4,"
TS="o33064u1w4b4s,o33064u1w4b5s,o33064u1w2b8s,o33048u3w4b4s,o16680u1w4b2s,o296u1w4b2s,o16664u3w4b2s,o280u3w4b2s,"
TN="o33064u1w4b4,o33048u3w4b4,o16680u1w4b2,o296u1w4b2,"
XF="o8200u1,o8232u1w4b4,o8488u1w4b4,o8472u3w4b4,"
ROUNDS=9 $H packs/o10 1 150 "n,l,o8u1,o40u1w4,$NS$SW$TS$TN" > ~/tf/j6_o10_m1.txt 2>&1
ROUNDS=7 $H packs/o10 8 150 "n,l,o8u1,o40u1w4,$SW$TS" > ~/tf/j6_o10_m8.txt 2>&1
ROUNDS=7 $H packs/o10 16 150 "n,l,o8u1,o40u1w4,$NS$TN$XF" > ~/tf/j6_o10_m16.txt 2>&1
ROUNDS=5 $H packs/o10 64 100 "n,l,o8u1,o8200u1," > ~/tf/j6_o10_m64.txt 2>&1
ROUNDS=5 $H packs/g1 1 150 "l,o40u1w4,o296u1w4b4s,o296u1w4b5s,o312u3w4b4s,o280u3w4b4,o33064u1w4b4s," > ~/tf/j6_g1_m1.txt 2>&1
COLD=1 ROUNDS=5 $H packs/g10 1 100 "n,l,o40u1w4,o296u1w4b4s,o296u1w4b5s,o312u3w4b4s,o152u3w4b4s,o33064u1w4b4s,o33048u3w4b4s," > ~/tf/j6_g10_cold.txt 2>&1
COLD=1 ROUNDS=5 $H packs/u10 1 60 "n,l,o40u1w4,o296u1w4b4s,o296u1w4b5s,o312u3w4b4s,o33064u1w4b4s," > ~/tf/j6_u10_cold.txt 2>&1
for v in o312u3w4b4s o296u1w4b4s o33064u1w4b4s; do
  ./ncu_h.sh $HOME/tf/ncu/r_$v "step_kernel" packs/o10 1 "$v," 2 $HOME/tf/harness24
done
./ncu_h.sh $HOME/tf/ncu/r_m16_o296u1w4b4 "step_kernel" packs/o10 16 "o296u1w4b4," 2 $HOME/tf/harness24
./ncu_h.sh $HOME/tf/ncu/r_m16_xf "step_kernel" packs/o10 16 "o8488u1w4b4," 2 $HOME/tf/harness24
echo done > ~/tf/j6_done.txt
