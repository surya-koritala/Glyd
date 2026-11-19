#!/usr/bin/env bash
cd ~/tf-job
bash ./tf_job.sh > rehearsal.log 2>&1
cd ~/tf
H=$HOME/tf-job/bin/harness_89
ROUNDS=7 $H packs/g1 1 150 "l,o552u1w4b4s,o33320u1w4b4s,o33320u1w2b8s,o16936u1w4b2s,o16936u1w2b4s,o16952u3w4b2s,o33336u3w4b4s," > j9_g1_rep.txt 2>&1
ROUNDS=9 $H packs/o10 1 150 "n,l,o552u1w4b4s,o33320u1w4b4s,o296u1w4b2s,o16936u1w4b2s,o552u1w2b8s,o33320u1w2b8s,o16936u1w2b4s,o552u1w1b16s," > j9_o10_rep_m1.txt 2>&1
ROUNDS=7 $H packs/o10 8 150 "n,l,o552u1w4b4s,o33320u1w4b4s,o16936u1w4b2s," > j9_o10_rep_m8.txt 2>&1
ROUNDS=7 $H packs/o10 16 150 "n,l,o552u1w4b4,o33320u1w4b4,o16936u1w4b2," > j9_o10_rep_m16.txt 2>&1
echo done > j9_done.txt
