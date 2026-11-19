#!/usr/bin/env bash
cd ~/tf
H=./harness29
for M in 17 24 32 48 64; do
  ROUNDS=5 $H packs/o10 $M 100 "n,l,o8u1,o40u1,o552u1w4b2,o552u1w4b3,o552u1w8b1,o552u1w2b4,o552u1w2b6," > j13_o10_m$M.txt 2>&1
done
echo done > j13_done.txt
