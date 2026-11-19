# GLY-290 — Offline benchmark evidence validation

Parent: [GLY-178](https://linear.app/getglyd/issue/GLY-178); related preparation:
[GLY-175](https://linear.app/getglyd/issue/GLY-175). Baseline main:
`a52da22627d267a66bb853e9d1b52a37f40858c2`, qualified by all seven automatic
[CI 38749294633 checks](https://github.com/Sigbound/glyd-compute/actions/runs/38749294633).

The existing `BenchmarkRun` v1 draft had no typed Rust contract, inspector or
in-repository collector/record. This delivery finalizes its first implementation:
explicit fixture marking, strict schema/semantic checks, and an offline inspection
command that reuses the scoreboard's bounded file/hash verification. It reads no
model and starts no workload. Earlier native evidence formats remain unchanged.
See [the contract and limitations](../benchmark-inspection.md).

Controlled tests use only explicitly synthetic in-memory/temporary evidence.
Protocol checks cover round trips, fixture refusal, measured/failed consistency,
missing observations for pre-request failures, successful requests followed by
failed cleanup/collection, finite/coherent token timings, floating-point rounding,
identity collisions/placeholders, canonical u64 overflow, version/unknown fields,
date-time and portable-path refusal. A real CLI check runs in default and
development modes, verifies fixture gating, exact input/evidence hashes, retained
failure counts/IDs/codes, no private-note echo, tampered evidence refusal and
traversal rejection. No fabricated measured example or native result is supplied.

Initial focused checks passed: protocol 0.03 seconds, default CLI 0.71 seconds,
development CLI 0.69 seconds. Raw combined log SHA-256:
`e21149262e0b2252c7de82aaeb8f03580e1866341ed760ebfc7d99da071b1fc1`.
After final semantic checks, formatting, both warning-denied workspace/all-target
lint modes, the complete protocol tests and both-mode CLI library/binary/CLI tests
pass. Existing native-only tests remain ignored. Raw validation log SHA-256:
`cd2919c440a92277a68ac65bb608e3446f7f87c7d80d9c301a7f61b6ac3d97cd`.
The final cleanup-failure assertion passes (0.02 seconds), followed by both
warning-denied lint modes and formatting. Raw final-controls log SHA-256:
`844b2d89830a51be023ac41367d4bec03af86d1a3d41629498b1cb0e8a1303af`.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test -p compute-protocol --locked --offline
cargo test -p compute-cli --lib --bins --test cli --locked --offline
cargo test -p compute-cli --lib --bins --test cli --all-features --locked --offline
```

Correctness/security review traces strict parsing through schema/semantic checks,
then bounded regular-file/hash verification before any receipt is printed. Errors
are static; notes and raw contents are not echoed. The inspector never converts
failed runs into successful measurements, computes concurrent throughput from
summed durations, qualifies a backend or treats hashes as truthful collection.
It deliberately uses the same stable-directory boundary as the existing offline
scoreboard; concurrent hostile filesystem mutation is not newly solved here.

Separate complexity review reuses the protocol schema/parser, artifact reader,
existing JSON output path and standard collections. No new dependency, background
worker, schema framework, test runner, CI job/filter or time-budget change. All
seven latest-source checks and the clean/current-main review gate remain required
before authorized merge, followed by automatic main verification. Independent
milestone review remains deferred, not approved.

GLY-178's real direct/proxy and one/two-replica measurements remain open. Provider
GLY-238, reviewed native profiles, approved devices/workloads and M2 acceptance
are not completed by this software portion. Historical Windows initiating causes
remain unproved. No native/model/GPU/Ubuntu/SSH workload, download, spend, old-state
recovery, quarantine reset, license, repository setting or publication change.
