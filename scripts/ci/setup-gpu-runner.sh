#!/usr/bin/env bash
# Sets this Ubuntu machine up as Glyd's self-hosted GPU runner (issue S2, #127):
#   1. checks the NVIDIA driver (580 or newer, for the engine's CUDA 13 kernels);
#   2. installs Docker and the NVIDIA Container Toolkit if missing, and checks a container sees the GPU
#      (the GPU jobs run in nvidia/cuda:13.0.3-devel-ubuntu24.04 with --gpus all);
#   3. creates a dedicated user, glyd-runner, and installs the GitHub Actions runner in /opt/glyd-runner;
#   4. registers it with surya-koritala/Glyd (labels: self-hosted, Linux, X64, gpu, glyd-pc, sm<compute capability>)
#      and runs it as a system service, so it starts with the machine;
#   5. sets the repository variables GPU_RUNNER and GPU_RUNNER_ARCHS, which turn the GPU workflow on.
# Run it as your own user (it asks for sudo):
#   curl -fsSL https://raw.githubusercontent.com/surya-koritala/Glyd/main/scripts/ci/setup-gpu-runner.sh -o setup-gpu-runner.sh && bash setup-gpu-runner.sh
# Needs: gh logged in to an account that administers the repository (gh auth login), or RUNNER_TOKEN set from
# GitHub > Settings > Actions > Runners > New self-hosted runner. Safe to run again.
set -euo pipefail
REPO=surya-koritala/Glyd
NAME=${RUNNER_NAME:-glyd-$(hostname -s)}
RUSER=glyd-runner
DIR=/opt/glyd-runner

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "this script is for x86_64 Linux"
command -v sudo > /dev/null || die "sudo is needed"

say "NVIDIA driver"
command -v nvidia-smi > /dev/null || die "nvidia-smi was not found: install the NVIDIA driver (580 or newer) first"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 580 ] || die "driver $drv: the engine's CUDA 13 kernels need driver 580 or newer"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')

say "Docker"
if ! command -v docker > /dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker.io
fi
sudo systemctl enable --now docker

say "NVIDIA Container Toolkit"
if ! command -v nvidia-ctk > /dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker > /dev/null
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi -L || die "a container cannot see the GPU"

say "Runner user and files ($RUSER, $DIR)"
id "$RUSER" > /dev/null 2>&1 || sudo useradd --system --create-home --shell /bin/bash "$RUSER"
sudo usermod -aG docker "$RUSER"
sudo mkdir -p "$DIR"
sudo chown "$RUSER": "$DIR"
if [ ! -x "$DIR/config.sh" ]; then
  ver=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p')
  [ -n "$ver" ] || die "could not find the runner's latest release"
  curl -fsSL -o /tmp/actions-runner.tar.gz "https://github.com/actions/runner/releases/download/v$ver/actions-runner-linux-x64-$ver.tar.gz"
  sudo -u "$RUSER" tar -xzf /tmp/actions-runner.tar.gz -C "$DIR"
  rm -f /tmp/actions-runner.tar.gz
  sudo "$DIR/bin/installdependencies.sh" > /dev/null
fi

say "Registration token"
TOKEN=${RUNNER_TOKEN:-}
if [ -z "$TOKEN" ] && command -v gh > /dev/null && gh auth status > /dev/null 2>&1; then
  TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
fi
[ -n "$TOKEN" ] || die "no token: run 'gh auth login' first, or set RUNNER_TOKEN from GitHub > Settings > Actions > Runners > New self-hosted runner"

say "Registering $NAME (labels: self-hosted, Linux, X64, gpu, glyd-pc, sm$cc)"
if [ -f "$DIR/.service" ]; then (cd "$DIR" && sudo ./svc.sh stop || true); fi
sudo -u "$RUSER" bash -c "cd '$DIR' && ./config.sh --unattended --replace --url 'https://github.com/$REPO' --token '$TOKEN' --name '$NAME' --labels 'gpu,glyd-pc,sm$cc' --work _work"
cd "$DIR"
[ -f .service ] || sudo ./svc.sh install "$RUSER"
sudo ./svc.sh start

say "Repository variables"
if command -v gh > /dev/null && gh auth status > /dev/null 2>&1; then
  gh variable set GPU_RUNNER --repo "$REPO" --body '["self-hosted","glyd-pc","gpu"]'
  gh variable set GPU_RUNNER_ARCHS --repo "$REPO" --body "$cc"
  echo "GPU_RUNNER and GPU_RUNNER_ARCHS=$cc are set"
else
  echo "Set these in GitHub > Settings > Secrets and variables > Actions > Variables:"
  echo "  GPU_RUNNER       = [\"self-hosted\",\"glyd-pc\",\"gpu\"]"
  echo "  GPU_RUNNER_ARCHS = $cc"
fi

say "Done. GitHub > Settings > Actions > Runners lists $NAME as Idle; the next push to main that touches the engine runs the GPU tests here."
