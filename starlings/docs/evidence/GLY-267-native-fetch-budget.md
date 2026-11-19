# GLY-267 native fetch and fixed CI budget

PR107 source82d8cd964d21e1703ae2b26428d32f6862b59be7 / CI38948219733
is CANCELLED Windows backend, with five other partitions PASS and aggregate FAIL.
Raw work/m2-pr107-82d8cd9-windows-backend.log SHA256
2881de324628b34b96b07a3bf4394708277b0cacbc6bd7c11ec1d8adff1dcd31.
This does not qualify the complete source, though Linux/Windows default alias and
original preflight controls pass. No unchanged-source retry is used.

The job starts18:33:16UTC and reaches the existing ten-minute limit. Eight short
cases pass62.05s; the nine-mode production matrix passes241.86s. Recovery begins
18:43:19 and is canceled roughly five seconds later. There is no observed recovery
assertion or hang; it does not finish, so recovery is not qualified. Expected
caught publication/unwind panics belong to passing controls.

The exact425MiB dependency/build cache restores successfully. Existing locked
fetch takes55.6s; lint21.25s, prepared test compilation40.57s, later Cargo entries
0.30–0.65s confirm binary reuse. The fixture does not run nine times by accidental
duplicate selection: it deliberately serializes nine distinct Windows scenarios.
The total includes slow execution and all-target dependency preparation, not just
compilation. No unmeasured OS/security-software cause is asserted.

Use [Cargo's native target selection](https://doc.rust-lang.org/cargo/commands/cargo-fetch.html)
with --target host-tuple in the existing offline and online fetch commands. The
pinned1.97.1 offline invocation succeeds locally and leaves Cargo.lock unchanged.
No dependency download, compilation, process inspection or network test is needed
for that command. Native jobs still run their exact complete tests/lints offline;
all platforms, seven checks, warning/error policy, serial execution, test selection,
cache-source integrity policy and job/deadline limits remain unchanged.

Bounded offline all-feature Cargo tree comparison finds235 all-target unique
package/version entries versus195 for Windows:40 unrelated entries. This is
dependency-graph evidence, not a controlled fetch-time or job-speed measurement.
Raw work/m2-pr107-dependency-tree-all.txt SHA256
e3f229d6c82ebe9b1e668a7d48d1af0f2b582c864ba044e057f7475cb36ddbc6;
Windows f8ff46d530e54f7bb91b0214c3659fb2fd9385f7e68215c3ea32e08a2106d6a5.
Both reads are capped20s/1MiB per output, existing cached inputs only.

The same existing Windows nine-mode loop adds static start/done and monotonic
elapsed-ms markers. A later canceled run can identify its unfinished mode without
host/process instrumentation, secrets, new cases, retries or longer waits. Local
Mac verification compiles/lints only; it does not execute the socket/daemon matrix.

Separate complexity review: two native fetch flags and bounded existing-loop
markers, no scheduler/cache framework or extra job. Correctness review checks
native-host coverage, identical lock/test/lint modes, no source-cache policy change
and no timing/admission/quarantine relaxation. Final exact-source and separate main
qualification remain required; no speedup or historical cause is claimed yet.

PR107 final source1a785fc / CI38950518749 passed all seven checks before merge
05aa80908a72387f9ffca96c607d414ca2d3dc32. Source Windows short cases55.07s,
production nine modes166.02s and recovery51.45s pass; raw backend SHA256
935c84df94b7fa9545359f9ed0e3e6fa91e24e2f29458eb06924b5c6977fedd4.
Its separate automatic main CI38952049958 is CANCELLED Windows backend and
aggregateFAIL, other five partitionsPASS. Main is not separately qualified.
Raw work/m2-pr107-main-windows-backend.log SHA256
a8f938e1ca9f5e0de9d8dcaed66644b537f698d4919b75d5debbc0e2fa4d2710.

On main the backend starts19:03:26UTC; short eight casesPASS60.94s, production
all nine modesPASS183.77s and recoveryPASS56.17s. The last test passes19:13:31,
then the job is canceled at the unchanged ten-minute cap before normal completion.
There is no observed failed assertion or unfinished test in this run. Fetch40s,
lint16.31s, test compilation31.26s and the earlier serialized test groups also
consume the total budget. Expected caught panics remain passing controls.
Windows default completes all its tests at19:09:34, then cache/post-job work
finishes; raw work/m2-pr107-main-windows-default.log SHA256
c214b87e61a34b92568bf54351dff9789b0b41d6695f1ea8167d013a37a153d7.

Move only the existing eight-case short catalog command to a step on the existing
Windows default partition. Its command, all-features selection and serialized
execution remain byte-identical. The two long catalog commands remain on Windows
backend; non-Windows execution is unchanged. This balances already required work
without new jobs, omitted cases, higher caps, retries or changes to production
timeouts. Additional all-features compilation on default is expected and must be
measured by actual CI; the observed sixty-second group is not a guaranteed saving.

File-only verification compares the full Cargo test command multiset before/after,
checks the moved step's Windows-only partition and serialization, unchanged caps,
matrix entries and warning policy, and parses every existing multiline shell
block with bash -n. No Mac daemon/socket tests or host instrumentation execute.
Separate complexity review keeps one moved command/existing partition; correctness
review covers exact coverage, feature-mode builds and aggregate completion. Exact
new-source CI and separate automatic main remain required; GLY311 integrated gate
and historical267 attribution remain open. No unchanged-source rerun is requested.
