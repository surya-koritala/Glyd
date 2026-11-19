#!/usr/bin/env bash
cd ~/tf
W="n,l,o0u1,o8u1,o9u1,o11u1,o15u1,o8u2,o15u2,o1u1,o2u1,o4u1,o24u3,o27u3,"
ROUNDS=7 ./harness6 packs/o10 1 150 "$W" > ~/tf/j3_o10_m1.txt 2>&1
PAD=40000 ROUNDS=5 ./harness6 packs/o10 1 150 "n,o0u1,o15u1,o15u2," > ~/tf/j3_pad40.txt 2>&1
ROUNDS=5 ./harness6 packs/o10 16 150 "$W" > ~/tf/j3_o10_m16.txt 2>&1
ROUNDS=5 ./harness6 packs/o10 64 100 "n,l,o0u1,o8u1,o15u1,o15u2,o27u3," > ~/tf/j3_o10_m64.txt 2>&1
COLD=1 ROUNDS=5 ./harness6 packs/g10 1 100 "n,l,o0u1,o8u1," > ~/tf/j3_g10_cold.txt 2>&1
for M in 128 256 1024; do ROUNDS=3 ./harness6 packs/o10 $M 20 "n," > ~/tf/j3_o10_big$M.txt 2>&1; done
./ncu_h.sh $HOME/tf/ncu/v2lib "mma_gemm_kernel<Tiered2, \(int\)1>" packs/o10 1 "l,"
./ncu_h.sh $HOME/tf/ncu/v1lib "mma_gemm_kernel<Tiered, \(int\)1>" packs/o10 1 "l,"
./ncu_h.sh $HOME/tf/ncu/nib "mma_gemm_kernel<Nib, \(int\)1>" packs/o10 1 "n,"
./ncu_h.sh $HOME/tf/ncu/o0 "step_kernel<Tiered2o<\(int\)0>, \(int\)1" packs/o10 1 "o0u1,"
./ncu_h.sh $HOME/tf/ncu/o15 "step_kernel<Tiered2o<\(int\)15>, \(int\)1" packs/o10 1 "o15u1,"
./ncu_h.sh $HOME/tf/ncu/o27 "step_kernel<Tiered2o<\(int\)27>, \(int\)1" packs/o10 1 "o27u3,"
echo done > ~/tf/j3_done.txt
