#!/bin/bash
# Installs this directory's starlings and glyd-engine for the current user (no root):
# files in ~/.local/lib/starlings, the command in ~/.local/bin. `install.sh --uninstall` removes
# them (state in ~/.local/state/starlings and downloaded models are kept). Only a directory this
# script installed (it holds .starlings-install) is ever replaced or removed.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --uninstall ) ]] || { echo "usage: $0 [--uninstall]" >&2; exit 2; }
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
if { [ -e "$bin/starlings" ] || [ -L "$bin/starlings" ]; } && [ "$(readlink "$bin/starlings" 2>/dev/null)" != "$lib/starlings" ]; then
    echo "$bin/starlings exists and is not ours; not touching it" >&2
    exit 1
fi
mkdir -p "$(dirname "$lib")" "$bin"
stage=$(mktemp -d "$lib.new.XXXXXX")
previous=
activated=false
cleanup() {
    if [[ -n "$previous" && -d "$previous" ]]; then
        rm -rf "$lib"
        mv "$previous" "$lib"
    elif $activated; then
        rm -rf "$lib"
    fi
    rm -rf "$stage"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp "$here/starlings" "$here/glyd-engine" \
    "$here/LICENSE-glyd-engine" "$here/install.sh" "$here/SHA256SUMS" "$stage/"
touch "$stage/.starlings-install"
# A broken binary or unsupported host must not remove a working installation.
version=$("$stage/starlings" --version)
if [[ -d "$lib" ]]; then
    previous=$stage.previous
    mv "$lib" "$previous"
fi
mv "$stage" "$lib"
activated=true
ln -sfn "$lib/starlings" "$bin/starlings"
[[ -z "$previous" ]] || rm -rf "$previous"
previous=
activated=false
echo "installed: $bin/starlings ($version)"
case ":$PATH:" in *":$bin:"*) ;; *) echo "add $bin to PATH" ;; esac
echo "start: starlings up --model glyd/Qwen3.8-27B-swift"
