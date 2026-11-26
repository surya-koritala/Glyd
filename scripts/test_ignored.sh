#!/usr/bin/env bash
# The #[ignore]d tests the gate runs: the `run` lines of glyd-engine/tests/ignored-in-gate.txt, each once and each passing. A name that matches no ignored test, or a name that
# also matches another, fails here (the count of tests run is the count of names). Extra arguments go to cargo (the target dir is CARGO_TARGET_DIR's).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
names=$(awk '$1 == "run" { print $2 }' glyd-engine/tests/ignored-in-gate.txt)
[[ -n "$names" ]] || { echo "no run line in glyd-engine/tests/ignored-in-gate.txt"; exit 1; }
log=$(mktemp); trap 'rm -f "$log"' EXIT
rc=0
# shellcheck disable=SC2086
cargo test --profile ci -p glyd-engine --lib "$@" -- --ignored $names 2>&1 | tee "$log" || rc=$?
n=$(wc -w <<< "$names"); ran=$(grep -cE '^test .* \.\.\. (ok|FAILED)$' "$log" || true)
for t in $names; do grep -qE "^test .*::$t \.\.\. ok$" "$log" || { echo "ignored test $t did not run and pass"; rc=1; }; done
(( ran == n )) || { echo "ran $ran ignored tests for $n names"; rc=1; }
exit $rc
