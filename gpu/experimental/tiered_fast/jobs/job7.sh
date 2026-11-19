#!/usr/bin/env bash
cd ~/tf
H=./harness25
for M in 17 32 48 64; do
  ROUNDS=5 $H packs/o10 $M 100 "n,l,o8u1,o40u1," > ~/tf/j7_o10_m$M.txt 2>&1
done
COLD=1 ROUNDS=5 $H packs/u10 32 60 "n,l,o8u1," > ~/tf/j7_u10_m32.txt 2>&1
COLD=1 ROUNDS=5 $H packs/u10 64 60 "n,l,o8u1," > ~/tf/j7_u10_m64.txt 2>&1
echo done > ~/tf/j7_done.txt
