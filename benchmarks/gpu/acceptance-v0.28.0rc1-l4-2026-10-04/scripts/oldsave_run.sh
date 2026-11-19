#!/bin/bash
# the old-save check of the v0.28.0rc1 acceptance on the L4 (each step under the GPU lock): a smallest-layout save of Qwen3-0.6B by glyd-gpu 0.27.0 (glyd-v1), loaded by 0.28.0rc1 with verify=True and a short greedy answer compared;
# a glyd-v4 save made by 0.28.0rc1 (save_pretrained, and glyd pack) and refused by 0.27.0 (from_pretrained, and glyd verify)
O=/opt/dlami/nvme/rel028/oldsave; cd $O || exit 1
unset PYTHONPATH LD_LIBRARY_PATH GLYD_GPU_LIB GLYD_VERSION GLYD_KV
export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=false
cp /tmp/oldsave_common.py /tmp/oldsave_make.py /tmp/oldsave_load.py /tmp/oldsave_refuse.py $O/
mkdir -p logs; touch ~/.glyd-busy
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); echo "== $(date -u +%T): $free GB free on /"; [ "$free" -ge 30 ] || { echo "== not run: under 30 GB free"; exit 3; }
step() {  # step NAME CMD...: CMD under the lock, its output in logs/NAME.txt
  n=$1; shift
  echo "== $(date -u +%T) $n: $*"
  flock -o ~/.glyd-box.lock "$@" > logs/$n.txt 2>&1; rc=$?
  echo "== $n: exit $rc"; sed 's/^/    /' logs/$n.txt | grep -v -E '^\s*$' | cut -c1-260 | tail -40
  return $rc
}
step 1-make-027      $O/v027/bin/python $O/oldsave_make.py $O;   r1=$?
step 2-load-028rc1   $O/v028rc1/bin/python $O/oldsave_load.py $O; r2=$?
step 3-refuse-027    $O/v027/bin/python $O/oldsave_refuse.py $O/save028-v4; r3=$?
step 4-verify-027    $O/v027/bin/glyd verify $O/save028-v4; r4=$?
rm -rf $O/save028-v4-cli
step 5-pack-028rc1   $O/v028rc1/bin/glyd pack Qwen/Qwen3-0.6B $O/save028-v4-cli; r5=$?
step 6-verify-028rc1 $O/v028rc1/bin/glyd verify $O/save028-v4-cli; r6=$?
step 7-verify-027-cli-save $O/v027/bin/glyd verify $O/save028-v4-cli; r7=$?
echo "== exits: make-027=$r1 load-028rc1=$r2 refuse-027(python)=$r3 verify-027(glyd verify, an API save)=$r4 pack-028rc1=$r5 verify-028rc1=$r6 verify-027(a glyd pack save)=$r7"
echo "OLDSAVE DONE"
pgrep -f '[f]lock .*glyd-box.lock' > /dev/null || rm -f ~/.glyd-busy
