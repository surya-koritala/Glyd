# GLY-267 fixture startup validation

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).
Delivery: [PR100](https://github.com/Sigbound/glyd-compute/pull/100).

Merged maina97b971 independently passes all seven CI38927095681 checks. Windows
backend production190.06s/recovery58.23s pass; raw SHA256
`49594a705ce40cbdb8e14e3b57fdd32193f4f99f121f2b3799568af968e8acda`.
This qualifies that software gate, not native inference or historical causes.

PR100 source913dac4/CI38927209799 fails Windows backend/aggregate; five other checks
pass. Production fails at daemon.rs294 after183.84s: the12s service readiness
observation sees loading/executable_digest with registry online. Unwind later sees
ready. Raw SHA256
`1ff612386f4165dc399fb752ee855069ebff2f5284751cbc805abbe39d9a726d`.
That phase snapshot does not prove CPU hashing time, file I/O or OS scheduling.
The failed source is preserved and not retried unchanged or merged.

Source tracing proves an extra full executable hash in the development fixture
wrapper between catalog validation and the fixture's final Service launcher.
Reuse the shared approved-settings/path validator there, retaining its original
64KiB and all other field/path checks. The final launcher still performs the full
validation and executable-byte check immediately before process creation. Catalog
validation and the normal production launcher remain full checks. No file hash is
cached across launches, native check skipped, deadline/TTL/durability relaxed or
test assertion removed. This removes a proved redundant hash, without claiming
the entire Windows startup failure's initiating cause.

One exact test uses a13-byte non-executable file and four fixed refusal variants.
The original wrapper fails the phase assertion0.00s by hashing even before the
fixture precondition refusal. The revised test passes0.02s: absent fixture mode,
changed approved configuration, oversized original report value that the fixture
would overwrite, and wrong executable bytes all fail at the expected boundary.
No case reaches process creation or network use. Correctness review rejected a
digest-only shortcut because it would not independently retain all original
settings constraints; the shared extraction preserves them.

Both warning-denied workspace/all-target lints pass2.93s/2.68s; fmt/diff pass.
Only basic bounded Mac file/compile checks, max two build jobs and cached offline
assets. No sockets/services, native/model workloads or host instrumentation/
inspection. Separate Ponytail complexity review keeps the existing validator and
launcher, one settings extraction and one bounded check; no receipt framework,
dependency or new CI job. Independent milestone/release review remains deferred.
Final combined source CI and current-main/clean/review gates remain required.

Combined source1b4d7a6 checks pass: refusal control0.01s, archive fixture0.11s,
five planner/oracle cases0.39s and offline CLI default/development0.73s/0.77s.
Both warning-denied workspace lint modes pass0.84s/0.63s. Formatting initially
rejects the shared phase-array layout, then is corrected and passes; no semantic
test/assertion change. The default Rust CLI release build also passes with
warnings denied152s/two jobs. Static candidate commands execute no service/model.
