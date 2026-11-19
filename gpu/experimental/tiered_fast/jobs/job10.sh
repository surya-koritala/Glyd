#!/usr/bin/env bash
cd ~/tf
H=$HOME/tf-job/bin/harness_89
V="n,l,o40u1w4,o552u1w2b8s,o552u1w4b4s,o520u1w2b8s,o568u3w4b4s,o296u1w8b0s,o552u1w1b16s,o552u1w3b5s,"
for p in c1 c2 o10; do
  ROUNDS=9 $H packs/$p 1 200 "$V" > j10_${p}_m1.txt 2>&1
  ROUNDS=7 $H packs/$p 8 200 "$V" > j10_${p}_m8.txt 2>&1
done
echo done > j10_done.txt
