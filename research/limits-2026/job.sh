#!/usr/bin/env bash
# limits-2026 on a pod's CPU (no GPU used): tools, the self-test, the plan from the headers, measure.py over it, report.py.
# ~/results is pulled by rp_run.sh every minute (the records only; the arrays stay in ~/work). lm.tgz: mktar.sh.
cd ~; mkdir -p results work; trap 'touch ~/results/DONE' EXIT
tar xzf lm.tgz
{ echo "nproc $(nproc)"; echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2> /dev/null)"; echo "memory.max $(cat /sys/fs/cgroup/memory.max 2> /dev/null)"
  echo "cfs quota $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2> /dev/null) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2> /dev/null)"
  free -g | head -2; df -h ~ | tail -1; python3 -c "import sys, numpy; print(sys.version.split()[0], 'numpy', numpy.__version__)"; } > results/env.txt 2>&1
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xz-utils zstd zpaq gcc) > results/apt.txt 2>&1
{ xz --version | head -1; zstd --version; zpaq 2>&1 | head -1; gcc --version | head -1; } >> results/env.txt 2>&1
cd lm/research/limits-2026
gcc -O2 -shared -fPIC -o libcm.so cm.c -lm > ~/results/build.txt 2>&1 || { echo "build failed" >> ~/results/build.txt; exit 1; }
python3 measure.py --selftest > ~/results/selftest.txt 2>&1 || { echo "selftest failed" >> ~/results/selftest.txt; exit 1; }
python3 plan.py ~/lm/headers ~/results/plan > ~/results/plan.txt 2>&1 || { echo "plan failed" >> ~/results/plan.txt; exit 1; }
ARRAYS=~/work/arrays python3 measure.py ~/results/plan ~/results/out > ~/results/run.log 2>&1
echo "exit $?" >> ~/results/run.log
python3 report.py ~/results/out ~/results/report > ~/results/report.log 2>&1
echo "exit $?" >> ~/results/report.log
