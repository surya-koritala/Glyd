# GLY-267 exact cached Windows toolchain reuse

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).

PR111 source02ff760 / CI38960549383 passed all seven checks before merge
7e8ff9d416502a63b7736e6f2aaab0207fa07eb3. Its separate automatic main
[CI38961787964](https://github.com/Sigbound/glyd-compute/actions/runs/38961787964)
is canceled at the ten-minute Windows backend cap; the aggregate fails and the
other five partitions pass. GitHub check117178386345 explicitly annotates
“The job has exceeded the maximum execution time of 10m0s”. This is not an
application assertion failure or a qualified main result.

The completed log shows all nine production modes passing in179.33s and recovery
passing in61.71s. The lifecycle step and checkout cleanup report success before
the final cancellation. The pinned toolchain cache restored an exact hit, then
the unconditional installer spent14s checking the same unchanged Rust1.97.1,
rustfmt and Clippy. Raw work/m6-pr111-main-windows-backend.log SHA256
014ac2ae2f3b494b3d57206c40a12338396f7c22b2db34f59b70b7deff3cda68;
job/annotation JSON is retained alongside it. A passing test log does not turn
the canceled job or failed aggregate green.

Use the existing exact cache-hit output to skip only that redundant Windows
installation. The [pinned cache implementation's documented output](https://github.com/actions/cache/blob/caa296126883cff596d87d8935842f9db880ef25/README.md#outputs)
distinguishes an exact hit from a miss or partial match. The toolchain cache has
no fallback keys and is saved only after installing the pinned compiler and both
components. Its version/architecture/config key and paths stay unchanged.
Missing/nonexact Windows caches and all Linux/macOS jobs keep installation.
Every formatting, lint, build and test command still explicitly uses Rust1.97.1;
missing or unusable cached components fail those required commands, without an
automatic repair/retry or a false success. No test is moved or skipped.

This removes observed repeated work; it does not promise a fixed total CI speedup
or explain historical Transport failures. Seven checks, warning denial, platform
coverage, all test commands, harness scheduling and ten-minute caps stay unchanged.
No new cache, dependency, job, local service/socket/model/instrumentation test or
unchanged failure rerun is introduced. Separate complexity review keeps one native
condition. Correctness review covers exact/nonexact/missing cache outputs across
platforms, fail-closed component use and unchanged validation gates. Native Ruby
YAML parsing and all16 shell-block syntax checks pass. Parsed baseline comparison
proves the condition is the sole workflow change; test commands, matrix, caches,
caps and warning policy are identical. Diff whitespace checks pass. No local Cargo
build or runtime test was needed. Full exact-source CI is required before a
reviewed current-main merge; automatic main remains separate. GLY267 and native
parents stay open. Independent milestone/security review is deferred, not approved.
