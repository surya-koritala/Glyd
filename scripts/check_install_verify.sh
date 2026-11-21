#!/usr/bin/env bash
# install.sh's signature check, on a release folder made here (GLYD_RELEASE_URL=file://...), on a machine with no NVIDIA GPU (the compression program's install):
# an older release without a signature installs with a warning, a signed release's tarball without one is refused (GLYD_SKIP_VERIFY=1 installs it), and with a
# cosign that accepts or refuses the bundle the tarball is installed or refused, with exactly the release workflow at this version's tag as the identity asked for.
#   scripts/check_install_verify.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) plat=linux-x86_64 ;;
  Linux-aarch64 | Linux-arm64) plat=linux-aarch64 ;;
  Darwin-arm64) plat=macos-arm64 ;;
  *) echo "no prebuilt platform here: nothing to check"; exit 0 ;;
esac
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

release() { # release VERSION [bundle]: a release folder in $work/rel-VERSION
  local v=$1 name=glyd-v$1-$plat d=$work/rel-$1
  mkdir -p "$d" "$work/pkg/$name"
  printf '#!/bin/sh\necho glyd %s\n' "$v" > "$work/pkg/$name/glyd"
  cp "$work/pkg/$name/glyd" "$work/pkg/$name/glyd-store"
  echo license > "$work/pkg/$name/LICENSE"
  chmod +x "$work/pkg/$name/glyd" "$work/pkg/$name/glyd-store"
  tar -czf "$d/$name.tar.gz" -C "$work/pkg" "$name"
  sha "$d/$name.tar.gz" > "$d/$name.tar.gz.sha256"
  if [ "${2:-}" = bundle ]; then echo '{"not":"a real bundle"}' > "$d/$name.tar.gz.sigstore.json"; fi
}
install() { # install VERSION [env...]: runs install.sh in a fresh home, prints its output, returns its status
  local v=$1; shift
  rm -rf "$work/home"; mkdir -p "$work/home"
  env -i PATH="$PATH" HOME="$work/home" TMPDIR="$work" GLYD_VERSION="$v" GLYD_RELEASE_URL="file://$work/rel-$v" "$@" sh "$ROOT/scripts/install.sh" 2>&1
}
fail() { echo "FAIL: $*"; exit 1; }

release 0.33.1
out=$(install 0.33.1) || fail "an older release without a signature did not install: $out"
echo "$out" | grep -q "older than the signed releases" || fail "no warning for the older release: $out"

release 0.34.0
if out=$(install 0.34.0); then fail "an unsigned 0.34.0 tarball was installed"; fi
echo "$out" | grep -q "has no signature" || fail "the refusal does not say why: $out"
out=$(install 0.34.0 GLYD_SKIP_VERIFY=1) || fail "GLYD_SKIP_VERIFY=1 did not install: $out"
echo "$out" | grep -q "GLYD_SKIP_VERIFY=1" || fail "no warning for GLYD_SKIP_VERIFY=1: $out"

release 0.34.0 bundle
mkdir -p "$work/bin-refuse" "$work/bin-accept"
printf '#!/bin/sh\necho "cosign: no matching signature" >&2\nexit 1\n' > "$work/bin-refuse/cosign"
printf '#!/bin/sh\necho "$@" > "%s/cosign-args"\nexit 0\n' "$work" > "$work/bin-accept/cosign"
chmod +x "$work/bin-refuse/cosign" "$work/bin-accept/cosign"
if out=$(install 0.34.0 PATH="$work/bin-refuse:$PATH"); then fail "a tarball cosign refused was installed"; fi
echo "$out" | grep -q "is not signed by Glyd's release workflows" || fail "the refusal does not say why: $out"
out=$(install 0.34.0 PATH="$work/bin-accept:$PATH") || fail "a tarball cosign accepted was not installed: $out"
echo "$out" | grep -q "Signature of glyd-v0.34.0-$plat.tar.gz checked" || fail "no line for the checked signature: $out"
grep -q -- '--certificate-identity https://github.com/surya-koritala/Glyd/.github/workflows/release.yml@refs/tags/v0.34.0 ' "$work/cosign-args" && grep -q 'token.actions.githubusercontent.com' "$work/cosign-args" || fail "cosign was not asked for release.yml at the tag v0.34.0: $(cat "$work/cosign-args")"
echo "install.sh signature checks: ok"
