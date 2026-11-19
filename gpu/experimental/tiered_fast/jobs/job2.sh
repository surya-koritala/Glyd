#!/usr/bin/env bash
# one slot of the GPU: the harness's variants, then ncu of the library's v2 step kernel
cd ~/tf
./harness2 packs/o10 1 200 all > ~/tf/j2_o10_m1.txt 2>&1
./harness2 packs/o10 16 200 all > ~/tf/j2_o10_m16.txt 2>&1
./ncu_h.sh $HOME/tf/ncu/v2_o10 "mma_gemm_kernel<Tiered2, 1>" packs/o10 1 l
./ncu_h.sh $HOME/tf/ncu/v1_o10 "mma_gemm_kernel<Tiered, 1>" packs/o10 1 l
echo done > ~/tf/j2_done.txt
