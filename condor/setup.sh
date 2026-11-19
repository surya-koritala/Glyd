#!/usr/bin/env bash
# Qwen3.8-Flash-Next on Glyd in one command: checks this computer, installs the engine, downloads the model, asks two questions, then starts the
# chat page and the API. Run it again to start it next time (nothing is downloaded twice).   ./setup.sh   (or ./setup.sh --yes for the defaults)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${GLYD_QWEN:-$HOME/.glyd-qwen-3.8-flash}; YES=0; [ "${1:-}" = --yes ] && YES=1
say() { printf '\033[1m%s\033[0m\n' "$*"; }
ask() { local q=$1 d=$2 a; if [ "$YES" = 1 ]; then echo "$d"; return; fi; read -r -p "$q [$d]: " a < /dev/tty; echo "${a:-$d}"; }

say "Condor: Qwen3.8-Flash-Next (Hummingbird level) on your GPU"
command -v nvidia-smi > /dev/null || { echo "No NVIDIA driver found (nvidia-smi). Install the NVIDIA driver 580 or newer, then run this again."; exit 1; }
read -r GPU VRAM DRV < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2, $3}')
RAM=$(free -g | awk '/^Mem:/ {print $2}'); DISK=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
echo "  GPU ${GPU//_/ }, $(((VRAM + 512) / 1024)) GB, driver $DRV;  RAM $RAM GiB;  free disk $DISK GB"
[ "$VRAM" -ge 15000 ] || echo "  Note: less than 16 GB on the GPU: fewer experts fit there (on a 16 GB card held to 12 GB, about 10% slower). Under 12 GB is untested."
[ "$RAM" -ge 60 ] || echo "  Note: less than 64 GB of RAM: fast mode is the one to pick (precise's experts do not fit in RAM and run about 40% slower); the first answer after a start is slower while experts load from the disk."
[ "${DRV%%.*}" -ge 580 ] || { echo "  The NVIDIA driver is $DRV; 580 or newer is needed. On Ubuntu: sudo apt install nvidia-driver-595 (or sudo ubuntu-drivers install), reboot, then run this again."; exit 1; }

# (the engine again when Glyd's latest release is not the one installed: these scripts may need it; offline, or an engine built here
# with tools/install.sh --source, the installed one stays)
NEW=$(curl -fsSLI --max-time 10 -o /dev/null -w '%{url_effective}' https://github.com/surya-koritala/Glyd/releases/latest 2> /dev/null | sed -n 's#.*/tag/##p' || true)
OLD=$(cat "$H/ENGINE_RELEASE" 2> /dev/null || true)
if [ ! -x "$H/bin/glyd-engine" ] || { [ -n "$NEW" ] && [ "$OLD" != source ] && [ "$NEW" != "$OLD" ]; }; then
  say "Installing the engine (about 50 MB)"; bash "$HERE/tools/install.sh"
fi
. "$H/env.sh"
DEF=precise; [ "$RAM" -ge 60 ] || DEF=fast
MODE=$(ask "Mode: precise (accurate, good for code; wants 64 GB of RAM), best (more accurate still, a little slower, 8 GB more to download; 64 GB of RAM) or fast (quicker answers, a smaller download)" $DEF)
case $MODE in precise) PART=precise-v2; GB=82;; best) PART=best-v2; GB=90;; fast) PART=fast; GB=66;; *) echo "Answer precise, best or fast."; exit 1;; esac
F=; [ "$MODE" = precise ] || F=--$MODE
if [ ! -f "$H/hf/model/config.json" ] || [ ! -e "$H/hf/$PART" ]; then
  [ "$DISK" -ge $((GB + 7)) ] || { echo "About $((GB + 7)) GB of free disk is needed for $MODE mode; $DISK GB is free."; exit 1; }
  say "Downloading the model (about $GB GB; it resumes if it stops)"
  bash "$HERE/tools/download.sh" $F
fi
CTX=$(ask "Context (tokens the model keeps in mind): 32768 is fastest; 65536, 131072 or 262144 hold more, with less room for the model on the GPU" 32768)
say "Starting: chat at http://127.0.0.1:8080/  ·  API at http://127.0.0.1:8080/v1 (OpenAI and Anthropic)  ·  Ctrl+C to stop"
( sleep 3; command -v xdg-open > /dev/null && xdg-open http://127.0.0.1:8080/ > /dev/null 2>&1 || true ) &
GLYD_CTX=$CTX exec bash "$HERE/tools/serve.sh" $F 8080
