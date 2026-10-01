#!/bin/bash
# The post-release acceptance's box side: the post-release-0.26.0 checkout, the release's own Rust glyd (its tarball, sha256 checked), and the lock wrapper.
set -e
rm -rf ~/onb/post && git clone -q --depth 1 --branch post-release-0.26.0 https://github.com/surya-koritala/Glyd ~/onb/post
git -C ~/onb/post log -1 --format='post-release-0.26.0 at %h %s' | cut -c1-100
mkdir -p ~/onb/post-rust && cd ~/onb/post-rust
B=https://github.com/surya-koritala/Glyd/releases/download/v0.26.0
curl -fsSL -o glyd-v0.26.0-linux-x86_64.tar.gz $B/glyd-v0.26.0-linux-x86_64.tar.gz
curl -fsSL -o glyd-v0.26.0-linux-x86_64.tar.gz.sha256 $B/glyd-v0.26.0-linux-x86_64.tar.gz.sha256
cat glyd-v0.26.0-linux-x86_64.tar.gz.sha256
sha256sum -c glyd-v0.26.0-linux-x86_64.tar.gz.sha256
tar -xzf glyd-v0.26.0-linux-x86_64.tar.gz glyd-v0.26.0-linux-x86_64/glyd && cp glyd-v0.26.0-linux-x86_64/glyd ./glyd && ./glyd --version
cat > ~/onb/acc-post.sh <<'EOA'
#!/bin/bash
# acc-post.sh NAME WORKDIR ARGS...: gpu/vllm/acceptance.sh of the box's post-release-0.26.0 checkout under the GPU queue's lock, in WORKDIR; its output in
# ~/accept-post/NAME.out, the run's summary and logs kept in ~/accept-post/results/NAME
NAME=$1; WORK=$2; shift; shift
mkdir -p ~/accept-post/results
exec > ~/accept-post/$NAME.out 2>&1
echo "== queued $(date -u +%T): $*"
exec flock ~/.glyd-box.lock bash -c 'touch ~/.glyd-busy; echo "== started $(date -u +%T)"; cd ~/onb/post/gpu/vllm && git -C ~/onb/post log -1 --format="== checkout %h %s" | cut -c1-90; bash acceptance.sh --work "$0" "${@:2}"; rc=$?; rm -f ~/.glyd-busy; rm -rf ~/accept-post/results/$1; mkdir -p ~/accept-post/results/$1; cp -r "$0/summary.txt" "$0/logs" ~/accept-post/results/$1/ 2>/dev/null; cp -r "$0"/home/.local/state/glyd/logs ~/accept-post/results/$1/glyd-logs 2>/dev/null; echo "== finished $(date -u +%T) rc=$rc"' "$WORK" "$NAME" "$@"
EOA
chmod +x ~/onb/acc-post.sh
