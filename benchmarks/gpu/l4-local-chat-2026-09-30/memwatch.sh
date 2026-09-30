#!/bin/bash
# The GPU's memory.used every 0.1 s (nvidia-smi) into memwatch-NAME.csv while a command runs, then its floor: the CUDA
# total less the most used (what a desktop's new window would meet), and the samples under 512 MiB free.
#   bash memwatch.sh NAME command...
Q=~/quick; N=$1; shift
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > $Q/memwatch-$N.csv &
W=$!
"$@"
kill $W
T=$($Q/venv/bin/python -c "import torch; print(torch.cuda.mem_get_info()[1] >> 20)")
awk -F', ' -v T=$T '{u = $2 + 0; if (u > m) m = u; if (T - u < 512) c++} END {printf "memwatch: %d samples; most used %d MiB of %d, so least free %d MiB; %d samples under 512 MiB free\n", NR, m, T, T - m, c + 0}' $Q/memwatch-$N.csv | tee $Q/memwatch-$N.txt
