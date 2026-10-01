#!/bin/bash
# acc-post.sh NAME WORKDIR ARGS...: gpu/vllm/acceptance.sh of the box's post-release-0.26.0 checkout under the GPU queue's lock, in WORKDIR; its output in
# ~/accept-post/NAME.out, the run's summary and logs kept in ~/accept-post/results/NAME
NAME=$1; WORK=$2; shift; shift
mkdir -p ~/accept-post/results
exec > ~/accept-post/$NAME.out 2>&1
echo "== queued $(date -u +%T): $*"
exec flock ~/.glyd-box.lock bash -c 'touch ~/.glyd-busy; echo "== started $(date -u +%T)"; cd ~/onb/post/gpu/vllm && git -C ~/onb/post log -1 --format="== checkout %h %s" | cut -c1-90; bash acceptance.sh --work "$0" "${@:2}"; rc=$?; rm -f ~/.glyd-busy; rm -rf ~/accept-post/results/$1; mkdir -p ~/accept-post/results/$1; cp -r "$0/summary.txt" "$0/logs" ~/accept-post/results/$1/ 2>/dev/null; cp -r "$0"/home/.local/state/glyd/logs ~/accept-post/results/$1/glyd-logs 2>/dev/null; echo "== finished $(date -u +%T) rc=$rc"' "$WORK" "$NAME" "$@"
# ---- post-chain.sh
#!/bin/bash
# the second post-release run (24 GB) after the first has finished
while ! grep -q "^== finished" ~/accept-post/r16.out 2>/dev/null; do sleep 20; done
~/onb/acc-post.sh r24 ~/accept-post/work --rust-glyd /home/ubuntu/onb/post-rust/glyd --install-url https://raw.githubusercontent.com/surya-koritala/Glyd/v0.26.0/scripts/install.sh --webui none
# ---- post-chain2.sh
#!/bin/bash
# the third post-release run (16 GB, the script getglyd.com serves) after the second has finished
while ! grep -q "^== finished" ~/accept-post/r24.out 2>/dev/null; do sleep 20; done
~/onb/acc-post.sh r16served ~/accept-post/work --rust-glyd /home/ubuntu/onb/post-rust/glyd --served --install-url https://raw.githubusercontent.com/surya-koritala/Glyd/v0.26.0/scripts/install.sh --card 4080s --webui all --pip-refusal
