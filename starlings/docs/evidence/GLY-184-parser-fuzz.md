# GLY184 parser fuzzing

Issue: https://linear.app/getglyd/issue/GLY-184

Oracle/runtime source `99b9757253a0732a693291de738ec7a2aec200f7`, initially
based on mainc55c8d. Signed integrationb42bd4f incorporates PR93/main669a614;
protocol/fuzz source and lockfiles are verified unchanged by that merge.
Corpus source `d84d50157b372403fc68c48975e7767e25797242` adds a valid maximum-size
header. The task PR records final source/CI separately. This is model-free testing.

The target reuses DataPrefix/DataHeader and one shared assertion function; ordinary
CI compiles and replays the same function using all six synthetic kinds and the
maximum header size. Assertions
check structural bounds and semantic round trips, not membership or execution.
An isolated optional runtime avoids five repeated instrumented platform builds.
The existing dependency job audits both locked graphs with warnings denied;
formatting covers both manifests. No job, production API or parser change.

## Observed local checks

* Original seed replay:1 passed,0.02s. Final full protocol suites:13 passed,
  including seven corpus seeds/the same oracle after actual main integration.
* Both warning-denied workspace lint modes, optional-runtime Clippy (19.12s),
  formatting and diff checks pass.
* Optional dependency audit:advisories/bans/licenses/sources all pass. The generator's
  versionless local dependency was rejected before CI and given an explicit version;
  no policy was relaxed. The narrow fuzz graph adds NCSA only for libfuzzer-sys0.4.13.
* Installed cargo-fuzz0.13.2/nightly-2026-07-20 builds the AddressSanitizer/coverage
  target in40.64s with two build jobs and cached dependencies.
* First campaign completes10,000 inputs/6s without a parser crash, but the sandboxed
  atos subprocess emits symbolization warnings. Its complete log is retained and
  is not reported warning-free. Static symbol lookup succeeds.
* Same code/binary, fresh corpus, explicit installed atos and host permission:
  **10,000 inputs/7s, exit0, no warnings/errors/crashes**. Boundaries:30s total,
  10,000 runs,2s per input,1024MiB RSS limit,16,400-byte maximum input, seed184.
* Final campaign adds the seventh valid16KiB-header seed, with the same bounds and
  unchanged binary: **10,000 inputs/7s, exit0, no warnings/errors/crashes**.
  Reported `cov:28037`, `ft:51289`, corpus165/168Kb, length limit16400 and
  `rss:184Mb` include dependency/runtime effects; they are not backend memory or
  exhaustive parser coverage. Standard library not rebuilt. Immutable identities,
  bounds and observations are in [the record](parser-fuzz-2026-10-16.json).

Actual instrumented binary SHA256:
`70d88dd65a98e028032f0f3732cb4b6fc5ef4cb84732268c39e9965d21220c5d`.

| Raw workspace log | SHA256 |
| --- | --- |
| m3-parser-fuzz-seeds.log |15873d49df3902a0119417cdfac6795c32b4d8a52eb864b8ac1405a87385238f|
| m3-parser-fuzz-protocol-suite.log |ab2e361925bbd19bf06383e94aa9d99ca158710f486c312798447da3685b3f3f|
| m3-parser-fuzz-policy.log |f1a0fca39d4280363937aabd77783990ea6480bd9ca257816de3b68fc8efa845|
| m3-parser-fuzz-build.log |a1373c345506a35073d898bd4a9211583500934230c20babc8d81c8840aa1eca|
| m3-parser-fuzz-run.log (warnings retained) |5cc228db2dcc02ff783ca135372f438d12c3409c0f99692861108bc7a79ca9c4|
| m3-parser-fuzz-host-run.log |5bd71e5e0433c7c8102004db9385de57e7c21d5910c45d8275b0e801b0c0e095|
| m3-parser-fuzz-boundary-run.log |92729df488ba2c21fbca575ff960f293a148cee8c7ccadf3bb65c0266aa9708c|
| m3-parser-fuzz-boundary-suite.log |81ea933d81022ea3fc00ae5097fc50defc7253a286650ce21a7f1e1035fb7c54|
| m3-parser-fuzz-runtime-lint.log |f2f1fb132ec803bee0bd61d1f267daeb262bd6d7196868354332d840378801ce|

Correctness/security review checks bounded input slicing, strict decoding, seed
validity, canonical semantic equality, isolated outputs/dependencies and unchanged
production authority. The separate complexity review retains a direct shared
function and native fuzz runtime, with no new parser, framework or configuration
surface in the application. GLY184's combined fuzz/memory subtask stays unchecked:
stateful reassembly/allocator/TLS-copy and physical links remain. No model, native
inference/drain, Exo or milestone qualification is inferred. Mesh license undecided;
independent milestone review deferred, not approved.
