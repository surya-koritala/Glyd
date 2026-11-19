#!/bin/bash
# Installs this directory's starlings, glyd-engine and GPU library for the current user (no root):
# files in ~/.local/lib/starlings, the command in ~/.local/bin. `install.sh --uninstall` removes
# them (state in ~/.local/state/starlings and downloaded models are kept). Only a directory this
# script installed (it holds .starlings-install) is ever replaced or removed.
set -euo pipefail
lib=${STARLINGS_LIB:-$HOME/.local/lib/starlings}
bin=${STARLINGS_BIN:-$HOME/.local/bin}
ours() { [ ! -e "$lib" ] || [ -f "$lib/.starlings-install" ]; }
ours || { echo "$lib is not a Starlings install; not touching it" >&2; exit 1; }
if [ "${1:-}" = --uninstall ]; then
    [ "$(readlink "$bin/starlings" 2>/dev/null)" = "$lib/starlings" ] && rm -f "$bin/starlings"
    rm -rf "$lib"
    echo "removed $lib and its command"
    exit 0
fi
here=$(cd "$(dirname "$0")" && pwd)
(cd "$here" && sha256sum -c --quiet SHA256SUMS)
command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver (nvidia-smi) found" >&2; exit 1; }
if [ -e "$bin/starlings" ] && [ "$(readlink "$bin/starlings" 2>/dev/null)" != "$lib/starlings" ]; then
    echo "$bin/starlings exists and is not ours; not touching it" >&2
    exit 1
fi
rm -rf "$lib.new"
mkdir -p "$lib.new" "$bin"
cp "$here/starlings" "$here/glyd-engine" "$here/libglyd_gpu_cuda13.so" "$lib.new/"
touch "$lib.new/.starlings-install"
rm -rf "$lib"
mv "$lib.new" "$lib"
ln -sfn "$lib/starlings" "$bin/starlings"
echo "installed: $bin/starlings ($("$lib/starlings" --version))"
case ":$PATH:" in *":$bin:"*) ;; *) echo "add $bin to PATH" ;; esac
echo "start: starlings up --model glyd/Qwen3.8-27B-swift"
