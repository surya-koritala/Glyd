#!/usr/bin/env bash
cd ~/tf
H=./harness15
W="n,l,o0u1,o8u1,o32u1,o40u1,o72u1,o104u1,o64u1,o8u1w4,o8u1w8,o40u1w4,o24u3w4,o56u3w4,o24u3w8,o24u3w4b4,o56u3w4b4,o24u3w4b3,o24u3w8b2,o8u1w8b0s,o8u1w4b0s,o8u1w4b5s,o40u1w4b4s,o24u3w4b3s,o24u3w4b4s,o56u3w4b4s,o24u3w8b2s,"
ROUNDS=7 $H packs/o10 1 150 "$W" > ~/tf/j4_o10_m1.txt 2>&1
ROUNDS=5 $H packs/o10 8 150 "n,l,o8u1,o8u1w8b0s,o8u1w4b5s,o24u3w4b4s,o56u3w4b4s,o24u3w4b3s," > ~/tf/j4_o10_m8.txt 2>&1
ROUNDS=5 $H packs/o10 16 150 "n,l,o8u1,o40u1,o8u1w4,o24u3w4,o56u3w4,o24u3w4b4,o56u3w4b4,o24u3w8b2," > ~/tf/j4_o10_m16.txt 2>&1
ROUNDS=5 $H packs/o10 64 100 "n,l,o8u1,o8u1w4,o24u3w4,o24u3w4b4," > ~/tf/j4_o10_m64.txt 2>&1
COLD=1 ROUNDS=5 $H packs/g10 1 100 "n,l,o8u1,o8u1w4,o24u3w4,o24u3w4b4,o8u1w4b5s,o24u3w4b4s," > ~/tf/j4_g10_cold.txt 2>&1
ROUNDS=5 $H packs/g1 1 150 "l,o8u1,o24u3w4,o24u3w4b4,o24u3w4b4s," > ~/tf/j4_g1_m1.txt 2>&1
./ncu_h.sh $HOME/tf/ncu/p4 "step_kernel<Tiered2o<\(int\)24>, \(int\)1, \(int\)3, \(int\)4, \(int\)3, \(bool\)true" packs/o10 1 "o24u3w4b3s," 2 $HOME/tf/harness15
./ncu_h.sh $HOME/tf/ncu/p4b4 "step_kernel<Tiered2o<\(int\)24>, \(int\)1, \(int\)3, \(int\)4, \(int\)4, \(bool\)true" packs/o10 1 "o24u3w4b4s," 2 $HOME/tf/harness15
./ncu_h.sh $HOME/tf/ncu/nib2 "mma_gemm_kernel<Nib, \(int\)1>" packs/o10 1 "n," 2 $HOME/tf/harness15
echo done > ~/tf/j4_done.txt
