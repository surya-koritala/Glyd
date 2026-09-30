#!/usr/bin/env bash
# the crossover, finer: fused against decoded between the lengths run 1 measured, main (before the L4 routes)
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
flock $HOME/.glyd-box.lock env FILES=$HOME/l4routes R=$HOME/l4routes/results-main-fine ROUTE_ARGS="--lengths 896,1024,1280,1536,1792,2048,2304,2560,3072 --routes fused,decoded --reps 5" bash $HOME/l4routes/l4_routes.sh
