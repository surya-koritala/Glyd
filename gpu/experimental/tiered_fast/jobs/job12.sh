#!/usr/bin/env bash
cd ~/tf-job
bash ./tf_job.sh > rehearsal2.log 2>&1
cd ~/tf
H=$HOME/tf-job/bin/harness_89
for N in 0 100 20 10 5 2 1; do
  FLAGEVERY=$N ROUNDS=7 $H packs/o10 1 150 "l,o552u1w2b8s,o552u1w4b4s,o568u3w4b4s," > j11_o10_m1_every$N.txt 2>&1
done
for N in 0 20 5; do
  FLAGEVERY=$N ROUNDS=5 $H packs/o10 8 150 "l,o552u1w2b8s,o552u1w4b4s,o568u3w4b4s," > j11_o10_m8_every$N.txt 2>&1
done
echo done > j12_done.txt
