#!/bin/bash
# Restart the Rust HTTP server on port 8092 with the latest dev build (target2), fast pack, the public repo's page; wait until it's ready.
B=$HOME/glyd-scratch/qwen-build; . $B/env.sh
for p in $(pgrep -f "^/home/.*/glyd-engine-http qwenserve"); do kill $p; done
sleep 4
cp $B/target2/release/glyd-engine $B/bin/glyd-engine-http
cd $B/pubtest
nohup flock -w 7200 $HOME/glyd-scratch/gpu.lock $HOME/glyd-scratch/qwen/gpurun.sh $B/bin/glyd-engine-http qwenserve $HOME/glyd-scratch/qwen/slim $HOME/glyd-scratch/qwen/packed/${PACK:-fast_x} --ctx 65536 --port 8092 --page tools "$@" > $HOME/glyd-scratch/qwen/http.log 2>&1 < /dev/null &
for i in $(seq 120); do grep -q "^Ready" $HOME/glyd-scratch/qwen/http.log && break; sleep 2; done
tail -2 $HOME/glyd-scratch/qwen/http.log
