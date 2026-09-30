#!/bin/bash
# The box's lock held (and ~/.glyd-busy) while ~/quick/HOLD exists: the quickstart test's steps run inside it.
touch ~/quick/HOLD
(nohup flock ~/.glyd-box.lock bash -c 'touch ~/.glyd-busy; echo "held $(date -u +%T)" > ~/quick/held; while [ -f ~/quick/HOLD ]; do sleep 5; done; rm -f ~/.glyd-busy ~/quick/held' > /dev/null 2>&1 &)
for i in $(seq 1 720); do [ -f ~/quick/held ] && { cat ~/quick/held; exit 0; }; sleep 5; done
echo "lock not held after an hour"; exit 1
