#!/usr/bin/env bash
# One command for a Glyd release, from the version commit to the fresh-install check. Each step of the runbook is a named step; each
# prints the commands it will run and the check that proves it worked, then runs them:
#
#     scripts/release.sh 0.30.0 --private-notes FILE --public-notes FILE [--post FILE] [--dry-run] [--from STEP] [--only STEP]
#
# Run it from the glyd-gpu worktree on the release's pull request branch (not main; it commits the version files there). It needs git,
# python3, curl, and gh signed in. FILE: the CHANGELOG entry that goes under `## Unreleased` (the private one may link and name
# formats and kernels; the public one never does), and the public notes are also the GitHub release's. --post: the site's release post
# (an .mdx with a description of at most 155 characters). Steps, in order:
#     private-bump  public-pr  shim  private-merge  private-tag  pypi-gpu  public-merge  public-tag  release-page  pypi-glyd
#     formula  tap  site  install-check
# Finished steps are written to ${XDG_STATE_HOME:-~/.local/state}/glyd-release/VERSION, one a line; a rerun skips them (--from STEP
# runs STEP and every later step again, --only STEP runs just that one). --dry-run prints every step and runs nothing that writes
# (no state file either); a value it cannot know is shown as <name>. Two steps stop for your "go" (read from the terminal): the
# public merge and the site merge. The remotes: `private` is Sigbound/glyd-gpu-private, `origin` is surya-koritala/Glyd; to `origin`
# this pushes only the public release branch, the public tag and the formula branch.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=(private-bump public-pr shim private-merge private-tag pypi-gpu public-merge public-tag release-page pypi-glyd formula tap site install-check)
PRIV=Sigbound/glyd-gpu-private; PUB=surya-koritala/Glyd; TAP=surya-koritala/homebrew-glyd; SITE_REPO=Sigbound/glyd-website
SITE=${GLYD_SITE:-$HOME/glyd-website}
ID=(-c user.name="Surya Koritala" -c user.email="suryakoritala1324@gmail.com")

die() { echo "release.sh: $*" >&2; exit 1; }
usage() { sed -n '2,19p' "$0"; exit 2; }
DRY=false FROM="" ONLY="" PRIV_NOTES="" PUB_NOTES="" POST="" V=""
while [ $# -gt 0 ]; do
  case $1 in
    --private-notes) PRIV_NOTES=${2:?}; shift;; --public-notes) PUB_NOTES=${2:?}; shift;; --post) POST=${2:?}; shift;;
    --from) FROM=${2:?}; shift;; --only) ONLY=${2:?}; shift;; --dry-run) DRY=true;;
    -h|--help) usage;; -*) die "unknown option $1";; *) [ -z "$V" ] || usage; V=$1;;
  esac; shift
done
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
[ -n "$PRIV_NOTES" ] && [ -n "$PUB_NOTES" ] || usage
for s in "$FROM" "$ONLY"; do [ -z "$s" ] || [[ " ${STEPS[*]} " == *" $s "* ]] || die "no step $s (steps: ${STEPS[*]})"; done
[ -e "$PRIV_NOTES" ] && [ -e "$PUB_NOTES" ] || die "notes file not found"
T=v$V
STATE_DIR=${XDG_STATE_HOME:-$HOME/.local/state}/glyd-release
STATE=$STATE_DIR/$V
PUBWT=$STATE_DIR/$V.public   # the public tree: a worktree of this repository, from origin/main
SITEWT=$STATE_DIR/$V.site    # the site's branch, a worktree of $SITE

# --- the machinery -------------------------------------------------------------------------------------------------------------
run() { printf '  +'; printf ' %q' "$@"; echo; $DRY || "$@"; }
check() { local d=$1; shift; echo "  check: $d"; $DRY || "$@" || die "check failed: $d"; }
# KEY=value for a later step (a rerun reads it back); a dry run shows <key>
put() { $DRY || { mkdir -p "$STATE_DIR"; echo "$1=$2" >> "$STATE.vars"; }; }
get() { if $DRY; then echo "<$1>"; else sed -n "s/^$1=//p" "$STATE.vars" 2>/dev/null | tail -1; fi; }
rec() { local k=$1; shift; printf '  + %s=$(' "$k"; printf '%q ' "$@"; echo ')'; $DRY || put "$k" "$("$@")"; }
approve() { echo "  STOP: $1"; $DRY && { echo "  (dry run: would wait for \"go\")"; return; }
  local a; read -r -p "  type go to continue: " a < /dev/tty || a=""; [ "$a" = go ] || die "not approved; the step stays unfinished"; }
# wait_for WHAT TRIES SECONDS CMD...: CMD until it succeeds
wait_for() { local what=$1 n=$2 s=$3; shift 3; echo "  wait (up to $((n * s / 60)) min, every ${s} s): $what"; $DRY && return
  while ! "$@"; do n=$((n - 1)); [ "$n" -gt 0 ] || die "gave up waiting: $what"; sleep "$s"; done; }
# the newest run of a workflow on a ref, watched to its end
watch_run() { local id; wait_for "a $2 run on $3 in $1" 30 10 bash -c "gh run list -R $1 -w $2 --branch $3 --limit 1 --json databaseId --jq '.[0].databaseId' | grep -q ."
  $DRY && { echo "  + gh run watch <run> -R $1 --exit-status"; return; }
  id=$(gh run list -R "$1" -w "$2" --branch "$3" --limit 1 --json databaseId --jq '.[0].databaseId'); gh run watch "$id" -R "$1" --exit-status >/dev/null; }
pypi_has() { # PACKAGE: this version is on the JSON API and in the simple index (its files are named with - or _)
  local p=$1; curl -fsS -o /dev/null "https://pypi.org/pypi/$p/$V/json" && curl -fsS "https://pypi.org/simple/$p/" | grep -qE "${p//-/[_-]}-${V//./\\.}[-.]"; }
changelog_insert() { # CHANGELOG NOTES: the notes go under `## Unreleased` (bump_version.py turns that heading into the release's)
  python3 - "$1" "$2" <<'PY'
import sys
path, notes = sys.argv[1:]
text = open(path).read()
head = "## Unreleased\n"
if head not in text: sys.exit("CHANGELOG.md has no '## Unreleased' heading")
body = open(notes).read().strip("\n")
if not body: sys.exit(notes + " is empty")
open(path, "w").write(text.replace(head, head + "\n" + body + "\n", 1))
PY
}
clean() { [ -z "$(git status --porcelain)" ]; }
tag_is() { [ "$(git rev-parse "$T^{commit}")" = "$1" ]; } # the local tag points at this commit
remote_is() { [ "$(git ls-remote "$1" "$2" | cut -f1)" = "$3" ]; } # REMOTE REF SHA
ready() { local out; out=$("$@" 2>&1) && return 0; echo "$out" | grep -q "CI: .*failure" && die "CI failed:"$'\n'"$out"; return 1; }
tag_install_is() { cmp "$1" <(git show "$T:scripts/install.sh"); } # FILE is the tag's install.sh
pr_merged() { [ "$(gh pr view "$1" -R "$2" --json state --jq .state)" = MERGED ]; }
site_install_ok() { curl -fsSL https://getglyd.com/install.sh | cmp -s - <(git show "$T:scripts/install.sh"); }

# --- the steps -----------------------------------------------------------------------------------------------------------------
step_private_bump() { # the version commit on the release PR branch, and its pull request
  local br; br=$(git branch --show-current)
  [ -n "$br" ] && [ "$br" != main ] && [ "$br" != glyd-shim ] || die "run this on the release's pull request branch, not '$br'"
  check "the tree is clean" clean
  run changelog_insert CHANGELOG.md "$PRIV_NOTES"
  run python3 scripts/bump_version.py "$V"
  check "every version file says $T" python3 scripts/bump_version.py --check "$T"
  run git "${ID[@]}" commit -s -am "$T: CHANGELOG and versions"
  rec PRIV_SHA git rev-parse HEAD
  run git push private "HEAD:refs/heads/$br"
  run gh pr create -R "$PRIV" --base main --head "$br" --title "$T: CHANGELOG and versions" --body ""
  rec PRIV_PR gh pr view "$br" -R "$PRIV" --json number --jq .number
  check "the pull request exists" test -n "$(get PRIV_PR)"
}
step_public_pr() { # the public release branch, from origin/main, with the public notes
  run git fetch origin main
  run git worktree add -B "public-release-$V" "$PUBWT" origin/main
  run changelog_insert "$PUBWT/CHANGELOG.md" "$PUB_NOTES"
  run python3 "$PUBWT/scripts/bump_version.py" "$V"
  check "the public tree says $T" python3 "$PUBWT/scripts/bump_version.py" --check "$T"
  check "its glyd[gpu]/glyd[vllm] extras pin glyd-gpu $V" grep -q "glyd-gpu.*$V" "$PUBWT/bindings/python/pyproject.toml"
  run git -C "$PUBWT" "${ID[@]}" commit -s -am "$T: CHANGELOG and versions"
  rec PUB_SHA git -C "$PUBWT" rev-parse HEAD
  run git -C "$PUBWT" push origin "HEAD:refs/heads/release-$V"
  run gh pr create -R "$PUB" --base main --head "release-$V" --title "$T: CHANGELOG and versions" --body ""
  rec PUB_PR gh pr view "release-$V" -R "$PUB" --json number --jq .number
}
step_shim() { # glyd-shim to the public release commit BEFORE private CI runs (test_onboard compares install.sh's pin to glyd-shim's version)
  run git push private "$(get PUB_SHA):refs/heads/glyd-shim"
  check "private glyd-shim is at the public release commit" remote_is private refs/heads/glyd-shim "$(get PUB_SHA)"
}
step_private_merge() { # no review needed; the exact head commit; then the redundant CI runs go
  run gh pr merge "$(get PRIV_PR)" -R "$PRIV" --merge --match-head-commit "$(get PRIV_SHA)"
  rec PRIV_MAIN gh pr view "$(get PRIV_PR)" -R "$PRIV" --json mergeCommit --jq .mergeCommit.oid
  echo "  + cancel every CI run on $PRIV main that is not at the merge commit"
  if ! $DRY; then
    gh run list -R "$PRIV" -w ci.yml --branch main --limit 20 --json databaseId,headSha,status \
      --jq ".[] | select(.status != \"completed\" and .headSha != \"$(get PRIV_MAIN)\") | .databaseId" |
      while read -r id; do gh run cancel "$id" -R "$PRIV"; done
  fi
  check "the PR is merged" pr_merged "$(get PRIV_PR)" "$PRIV"
}
step_private_tag() { # after ci.yml passed on that exact commit; release.yml then publishes glyd-gpu
  local sha; sha=$(get PRIV_MAIN)
  run git fetch private main
  wait_for "scripts/release_ready.sh $V $sha says READY" 60 60 ready scripts/release_ready.sh "$V" "$sha"
  run git "${ID[@]}" tag -a "$T" "$sha" -m "$T"
  run git push private "refs/tags/$T"
  check "the private tag is at $sha" remote_is private "refs/tags/$T^{}" "$sha"
}
step_pypi_gpu() { wait_for "glyd-gpu $V on PyPI (JSON and the simple index; release.yml takes about 20 minutes)" 90 60 pypi_has glyd-gpu; }
step_public_merge() { # after the release is on PyPI, the owner's go
  check "glyd-gpu $V is on PyPI" pypi_has glyd-gpu
  echo "  PR checks (shown only):"; $DRY || gh pr checks "$(get PUB_PR)" -R "$PUB" || true
  approve "merge $PUB PR #$(get PUB_PR) ($T: CHANGELOG and versions) with --admin"
  run gh pr merge "$(get PUB_PR)" -R "$PUB" --merge --admin
  run git fetch origin main
  rec PUB_MAIN gh pr view "$(get PUB_PR)" -R "$PUB" --json mergeCommit --jq .mergeCommit.oid
  run git push private "$(get PUB_MAIN):refs/heads/glyd-shim"
  check "private glyd-shim is at the public merge" remote_is private refs/heads/glyd-shim "$(get PUB_MAIN)"
}
step_public_tag() { # `git fetch private` brings the PRIVATE tag back: the local one is deleted, made again on the public merge, and checked
  local sha; sha=$(get PUB_MAIN)
  run git -C "$PUBWT" checkout --detach "$sha"
  wait_for "$PUBWT/scripts/release_ready.sh $V $sha says READY" 60 60 ready "$PUBWT/scripts/release_ready.sh" "$V" "$sha"
  echo "  + git tag -d $T   (if the private tag is here)"; $DRY || git tag -d "$T" 2>/dev/null || true
  run git "${ID[@]}" tag -a "$T" "$sha" -m "$T"
  check "git rev-parse $T^{commit} is the public merge $sha" tag_is "$sha"
  run git push origin "refs/tags/$T"
  check "the public tag is at $sha" remote_is origin "refs/tags/$T^{}" "$sha"
}
step_release_page() { # right away: the Release workflow's attach job needs it
  run gh release create "$T" -R "$PUB" --verify-tag --title "$T" --notes-file "$PUB_NOTES"
  watch_run "$PUB" release.yml "$T"
}
step_pypi_glyd() { wait_for "glyd $V on PyPI (JSON and the simple index; the index can lag)" 60 30 pypi_has glyd; }
step_formula() { # the formula branch, its PR, admin-merged once CI passes
  run git -C "$PUBWT" fetch origin main
  run git -C "$PUBWT" checkout -B "formula-$V" origin/main
  run python3 "$PUBWT/scripts/bump_formula.py" "$V"
  check "the formula says $V" grep -q "version \"$V\"" "$PUBWT/Formula/glyd.rb"
  run git -C "$PUBWT" "${ID[@]}" commit -s -am "glyd $V: Homebrew formula"
  run git -C "$PUBWT" push origin "HEAD:refs/heads/formula-$V"
  run gh pr create -R "$PUB" --base main --head "formula-$V" --title "glyd $V: Homebrew formula" --body ""
  rec FORMULA_PR gh pr view "formula-$V" -R "$PUB" --json number --jq .number
  run gh pr checks "$(get FORMULA_PR)" -R "$PUB" --watch --fail-fast
  run gh pr merge "$(get FORMULA_PR)" -R "$PUB" --merge --admin
}
step_tap() { # Formula/glyd.rb into the tap, then the Install check on this version
  local d; d=${TMPDIR:-/tmp}/glyd-tap-$V
  run rm -rf "$d"
  run gh repo clone "$TAP" "$d" -- -o tap
  run cp "$PUBWT/Formula/glyd.rb" "$d/Formula/glyd.rb"
  run git -C "$d" "${ID[@]}" commit -s -am "glyd $V"
  run git -C "$d" push tap HEAD:main
  run gh workflow run install-check.yml -R "$PUB" -f "version=$V"
  watch_run "$PUB" install-check.yml main
}
step_site() { # install.sh from the tag byte for byte, release.json, the post; the owner's go before the merge
  local br=release-$V
  run git -C "$SITE" fetch origin main
  run git -C "$SITE" worktree add -B "$br" "$SITEWT" origin/main
  run sh -c "git show '$T:scripts/install.sh' > '$SITEWT/public/install.sh'"
  check "public/install.sh is the tag's, byte for byte" tag_install_is "$SITEWT/public/install.sh"
  check "it installs $V" grep -q "^GLYD_VERSION=.*$V" "$SITEWT/public/install.sh"
  run sh -c "cd '$SITEWT' && npm run sync"
  run python3 - "$SITEWT/data/release.json" "$T" <<'PY'
import json, sys
path, tag = sys.argv[1:]
d = json.load(open(path)); d["release"] = d["installScript"] = tag
json.dump(d, open(path, "w"), indent=2, ensure_ascii=False); open(path, "a").write("\n")
PY
  approve "edit $SITEWT/data/release.json by hand (highlights, history, gaps: RELEASING.md step 2) and data/releases/$T.json if a GPU run was made"
  if [ -n "$POST" ]; then
    check "the post's description is at most 155 characters" bash -c "d=\$(sed -n 's/^description: *//p' '$POST' | head -1 | sed -E 's/^[\"]|[\"]\$//g'); [ -n \"\$d\" ] && [ \${#d} -le 155 ]"
    run cp "$POST" "$SITEWT/src/content/blog/$(basename "$POST")"
  else echo "  (no --post: RELEASING.md step 4 wants a release post; add one by hand)"; fi
  run git -C "$SITEWT" add -A
  run git -C "$SITEWT" "${ID[@]}" commit -s -m "$T: release data, post and install.sh"
  run git -C "$SITEWT" push origin "HEAD:refs/heads/$br"
  run gh pr create -R "$SITE_REPO" --base main --head "$br" --title "$T: release data, post and install.sh" --body ""
  rec SITE_PR gh pr view "$br" -R "$SITE_REPO" --json number --jq .number
  run gh pr checks "$(get SITE_PR)" -R "$SITE_REPO" --watch --fail-fast
  approve "merge $SITE_REPO PR #$(get SITE_PR) (deploys getglyd.com)"
  run gh pr merge "$(get SITE_PR)" -R "$SITE_REPO" --merge
  wait_for "https://getglyd.com/install.sh is the tag's, byte for byte" 40 30 site_install_ok
}
step_install_check() { # printed, never run: it costs a pod
  cat <<EOT
  Fresh GPU pod (tools/runpod): save this as /tmp/accept-$V.sh, then run the launcher; \$MODEL:\$LEVEL e.g. Qwen/Qwen3.5-9B:kestrel
      #!/usr/bin/env bash
      R=\$HOME/results; mkdir -p \$R; trap 'touch \$R/DONE' EXIT
      curl -LsSf https://getglyd.com/install.sh | sh 2>&1 | tee \$R/install.log
      export PATH=\$HOME/.local/bin:\$PATH; glyd --version | tee \$R/version.txt
      glyd run \$MODEL:\$LEVEL -p "Say hello in five words." 2>&1 | tee \$R/run.log
  launch:   NAME=accept-$V MAX_MIN=40 bash tools/runpod/rp_try.sh /tmp/accept-$V.sh
  check:    results-accept-$V/ has install.log ending in glyd doctor's all-clear, version.txt saying $V, and run.log with an answer
EOT
}

# --- the order -----------------------------------------------------------------------------------------------------------------
started=false
for s in "${STEPS[@]}"; do
  if [ -n "$ONLY" ]; then [ "$s" = "$ONLY" ] || continue
  elif [ -n "$FROM" ]; then [ "$s" = "$FROM" ] && started=true; $started || continue
  elif [ -f "$STATE" ] && grep -qx "$s" "$STATE"; then echo "== $s: done already"; continue; fi
  echo "== $s"
  "step_${s//-/_}"
  $DRY || { mkdir -p "$STATE_DIR"; echo "$s" >> "$STATE"; }
done
$DRY && echo "dry run: nothing was run"
echo "release $T: ${ONLY:-all steps} finished"
