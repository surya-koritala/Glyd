#!/usr/bin/env bash
# Is a commit ready to be tagged as a release of glyd? Run before tagging:
#
#     scripts/release_ready.sh 0.30.0 [COMMIT]      # COMMIT: default HEAD (needs git, cargo, curl and gh signed in)
#
# It says, of COMMIT: CI's (ci.yml) result on that exact commit, in a push or a run by hand (release.yml refuses a tag whose commit
# CI has not passed), whether its version files say the version (cargo xtask bump-version --check, COMMIT's own xtask on COMMIT's own files), whether
# the tag exists already elsewhere, and says whether glyd-gpu VERSION is on PyPI already (release-gpu.yml publishes it on the same tag;
# the extras pin glyd-gpu==VERSION, and install-check.yml retries until both are there).
# Exit 0: ready to tag.
set -euo pipefail
[ $# -ge 1 ] && [ $# -le 2 ] || { sed -n '2,9p' "$0"; exit 2; }
cd "$(dirname "$0")/.."
sha=$(git rev-parse --verify "${2:-HEAD}^{commit}")
here=surya-koritala/Glyd
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
ready=true
no() { ready=false; echo "  NOT READY: $*"; }

# CI on a commit: "success" (a push's or a by-hand run; a pull request's run tests a merge, not the commit), or its latest run
ci() {
  local runs="repos/$here/actions/workflows/ci.yml/runs?head_sha=$1" n
  n=$(gh api "$runs&status=success&per_page=100" --jq '[.workflow_runs[] | select(.event != "pull_request")] | length') || { echo "unreadable (gh api failed)"; return; }
  [ "$n" -gt 0 ] && { echo success; return; }
  n=$(gh api "$runs&per_page=1" --jq '.workflow_runs[0] // empty | "\(.event) \(.status) \(.conclusion // "-") \(.html_url)"') || { echo "unreadable (gh api failed)"; return; }
  echo "${n:-no run}"
}

# this commit's own version files, read as its own bump-version lists them (xtask/ built alone, outside the workspace: the standard library only)
mkdir "$tmp/task" "$tmp/tree"
git archive "$sha" xtask | tar -x -C "$tmp/task"
bump() { cargo run --quiet --manifest-path "$tmp/task/xtask/Cargo.toml" -- bump-version "$@"; }
files=$(bump --files)
# shellcheck disable=SC2086 # (one path a line, none with a space)
git archive "$sha" $files | tar -x -C "$tmp/tree"
py=$(bump --py "$1")

echo "$here at ${sha:0:12} ($(git log -1 --format=%s "$sha")), release v$py:"
c=$(ci "$sha"); echo "  CI: $c"
[ "$c" = success ] || no "CI has not succeeded on this commit: run it on a branch or tag at this commit (gh workflow run ci.yml --ref REF) and check again"
if out=$(bump --check "v$py" --root "$tmp/tree" 2>&1); then echo "  $out"; else echo "$out" | sed 's/^/  /'; no "the version files do not say v$py"; fi
t=$(gh api "repos/$here/commits/v$py" --jq .sha 2>/dev/null || true)
if [[ "$t" =~ ^[0-9a-f]{40}$ ]]; then
  [ "$t" = "$sha" ] && echo "  (tag v$py exists already, at this commit)" || no "tag v$py exists already in $here at ${t:0:12}, not at this commit"
fi
echo "glyd-gpu $py (the extras' pin; release-gpu.yml publishes it on this tag):"
if curl -fsS -o /dev/null "https://pypi.org/pypi/glyd-gpu/$py/json" 2>/dev/null; then echo "  on PyPI already"; else echo "  not on PyPI yet: the tag publishes it (its PyPI publisher must name $here release-gpu.yml, environment pypi)"; fi

if $ready; then echo "READY: tag v$py at $sha"; else echo "not ready to tag v$py"; exit 1; fi
