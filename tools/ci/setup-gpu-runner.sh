#!/usr/bin/env bash
# Register this Ubuntu machine's NVIDIA GPU as a self-hosted GitHub Actions runner and install it as a service.
#
# One runner per repository (a personal repository cannot share an organization's runner), each in its own folder under
# ~/actions-runners/<owner>-<repo>, labeled self-hosted, linux, gpu (plus --labels). The GPU jobs (.github/workflows/gpu-tests.yml)
# run in an nvidia/cuda container with --gpus all, so the machine needs the NVIDIA driver, Docker and the NVIDIA container toolkit;
# --install-deps installs the last two.
#
# Usage (as the user the runner should run as, not root; sudo is used where needed):
#   tools/ci/setup-gpu-runner.sh [--install-deps] [--labels a,b] [--name NAME] OWNER/REPO [OWNER/REPO ...]
#
# Registration tokens: taken from `gh` when it is signed in with admin rights on the repository; otherwise set
# RUNNER_TOKEN (one repository at a time) from Settings > Actions > Runners > New self-hosted runner.
#
# Afterwards, in each repository: Settings > Secrets and variables > Actions > Variables:
#   GPU_RUNNER        ["self-hosted","linux","gpu"]
#   GPU_RUNNER_ARCHS  the compute capability this script prints (e.g. 89, 120)
#
# Re-running is safe: a repository whose runner folder is already configured is skipped (its service is (re)started).
# To remove one: cd ~/actions-runners/<owner>-<repo> && sudo ./svc.sh uninstall && ./config.sh remove --token <removal token>
set -euo pipefail

INSTALL_DEPS=0
EXTRA_LABELS=""
NAME="$(hostname -s)-gpu"
REPOS=()
ROOT="${RUNNER_ROOT:-$HOME/actions-runners}"

die() { echo "setup-gpu-runner: $*" >&2; exit 1; }
say() { echo "==> $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --install-deps) INSTALL_DEPS=1 ;;
    --labels) EXTRA_LABELS="${2:?}"; shift ;;
    --name) NAME="${2:?}"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    */*) REPOS+=("$1") ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
[ ${#REPOS[@]} -gt 0 ] || die "name at least one repository, e.g. surya-koritala/glyd (see --help)"
[ "$(id -u)" -ne 0 ] || die "run as the runner's user, not root (the script uses sudo where it must)"
[ "$(uname -s)" = Linux ] || die "Linux only"
[ -n "${RUNNER_TOKEN:-}" ] && [ ${#REPOS[@]} -gt 1 ] && die "RUNNER_TOKEN is for one repository; name one at a time or use gh"

case "$(uname -m)" in
  x86_64) RARCH=x64 ;;
  aarch64) RARCH=arm64 ;;
  *) die "unsupported CPU: $(uname -m)" ;;
esac

# --- The GPU, Docker and the NVIDIA container toolkit -------------------------------------------------------------------
command -v nvidia-smi > /dev/null || die "nvidia-smi not found: install the NVIDIA driver first (e.g. sudo ubuntu-drivers install), then reboot"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"

if [ "$INSTALL_DEPS" = 1 ]; then
  if ! command -v docker > /dev/null; then
    say "Installing Docker"
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io
    sudo systemctl enable --now docker
  fi
  if ! command -v nvidia-ctk > /dev/null; then
    say "Installing the NVIDIA container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
fi
command -v docker > /dev/null || die "docker not found (re-run with --install-deps)"
command -v nvidia-ctk > /dev/null || die "the NVIDIA container toolkit is not installed (re-run with --install-deps)"
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
  say "Adding $USER to the docker group (the runner service picks it up when it starts)"
  sudo usermod -aG docker "$USER"
fi
say "Checking a container sees the GPU"
sudo docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi -L || die "a container could not reach the GPU: check the driver and 'sudo nvidia-ctk runtime configure --runtime=docker'"

# --- The runner package (one download, unpacked into each repository's folder) ------------------------------------------
say "Finding the latest runner release"
VER="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -1)"
[ -n "$VER" ] || die "could not read the runner's latest release from api.github.com"
PKG="actions-runner-linux-$RARCH-$VER.tar.gz"
CACHE="$ROOT/.download"
mkdir -p "$CACHE"
if [ ! -s "$CACHE/$PKG" ]; then
  say "Downloading runner $VER"
  curl -fsSL -o "$CACHE/$PKG.part" "https://github.com/actions/runner/releases/download/v$VER/$PKG"
  mv "$CACHE/$PKG.part" "$CACHE/$PKG"
fi

token_for() {
  if [ -n "${RUNNER_TOKEN:-}" ]; then echo "$RUNNER_TOKEN"; return; fi
  command -v gh > /dev/null || die "no RUNNER_TOKEN and no gh: set RUNNER_TOKEN from $1's Settings > Actions > Runners > New self-hosted runner"
  gh api -X POST "repos/$1/actions/runners/registration-token" --jq .token \
    || die "gh could not get a registration token for $1 (sign in with 'gh auth login' as a repository admin, or set RUNNER_TOKEN)"
}

LABELS="gpu,linux${EXTRA_LABELS:+,$EXTRA_LABELS}"
for repo in "${REPOS[@]}"; do
  dir="$ROOT/${repo//\//-}"
  say "$repo -> $dir"
  mkdir -p "$dir"
  cd "$dir"
  if [ ! -f .runner ]; then
    tar xzf "$CACHE/$PKG"
    sudo ./bin/installdependencies.sh > /dev/null
    ./config.sh --unattended --replace \
      --url "https://github.com/$repo" \
      --token "$(token_for "$repo")" \
      --name "$NAME" \
      --labels "$LABELS" \
      --work _work
  else
    echo "    already configured; keeping it"
  fi
  if [ ! -f .service ]; then
    sudo ./svc.sh install "$USER"
  fi
  sudo ./svc.sh start
  sudo ./svc.sh status | grep -E "Active:" | sed 's/^ */    /' || true
done

cat <<EOF

Done. In each repository, set these Actions variables (Settings > Secrets and variables > Actions > Variables):
  GPU_RUNNER        ["self-hosted","linux","gpu"]
  GPU_RUNNER_ARCHS  $CC
Then run the GPU workflow by hand (Actions > GPU > Run workflow) to check it.
EOF
