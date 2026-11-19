# GLY-183 — bounded invocation flow software

Issue: https://linear.app/getglyd/issue/GLY-183

The [flow contract](../data-flow.md) adds reserved direct reassembly and an
exclusive credit/receipt sender to authenticated GCP1. Both admission owners
derive an immutable local scratch allowance from the actual ready plan and
recheck it through the existing live binding. No incoming record chooses a
template, grants capacity, starts native work or releases a backend reservation.
The wire format, cryptography, dependency set, workflow and existing caps remain.

Pool/peer/session/lease budgets include handed-off input ownership. Cross-peer
session accounting and cross-session lease accounting cannot evade the limits.
Allocation happens before credit; chunks read directly into reserved boxed
tensors without another frame payload Vec. Scoped record/header/deadline caps,
strict offsets/order/credit, monotonically advancing receipts, poisoned-stream
refusal and final live input-state handoff fail closed. No partial tensor escapes.
The independent native owner still must retain inputs through trusted drain.

## Controlled findings during implementation

The draft initially accepted two1MiB tensors when the caller allowed2MiB but
the actual ready lease allowed only1MiB of scratch. The exact regression FAILS
before the fix (0.19s, raw SHA256
af700a773fc1bd06a62632740ac42428bfae71d246bfbd3b666c3728035a9377).
Admission now carries the plan-derived local allowance and the aggregate
constructor/ledger enforce it before allocation. The corrected controls pass
(raw8e25a4b682541524efcdae14d4aa6583daea86bf42d8399e01c717cc7ac85576).

An independent cross-peer/session case FAILS the draft peer-plus-session map
(0.02s, raw6ab30ecdde3b2b044665090124747d4d1373cf99f27876f3745958a98bb6d617).
Use one session key across peers, plus a distinct actual-lease ledger across
sessions/peers. Its corrected case passes in the final suite. These are draft
flow defects caught before delivery, not claimed causes of archived M2 failures.
An initial large-fixture setup also correctly refused an inconsistent lease
reservation; the fixture now computes both capacity and lease bytes from its
approved plan, preserving the production consistency check.

## Validation

Pinned Rust1.97.1, model-free loopback and synthetic bytes only:

```sh
cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo test -p compute-security --lib --locked --offline
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
```

Full security suite109 PASS/one unchanged ignored capacity case,18.18s;
raw49da9e591191b04883542deacf4b673598fa8fade290943e32c0304667304e0b.
Eleven new cases cover independent aggregate limits and bounded identity maps,
local lease allowance, all four raw dtype vectors, ordered hidden/residual chunks,
a16MiB tensor in two8MiB frames, no-credit/reordered refusal with omitted payloads,
duplicate chunks/forged receipts, truncation/drop and a separately pending step
between receipt and handoff. The pending step and core reservation remain
unchanged when handoff fails; only unexecuted host assembly ownership is freed.

The original child-process StageWorker test keeps every raw framing assertion
and also runs two-chunk credited flow on both actual data endpoints/all four
dtypes, checking owned byte equality, held/released payload totals and assembly
counts. Coordinator-only files remain absent from both device directories.
Worker snapshots remain unchanged: neither raw bytes, receipts nor flow finish
execute anything. The targeted process pass takes0.44s, raw
c64db178ab0ef1bbbf6d235f28c92690aa6f5cd746ba6f51e64bbe7c22ee583d;
it is also included in the final full-suite pass above.

An unused test-only import was caught by normal-library lint and moved into its
test module; no warning suppression is used. Final default lint rawdfa1d153f90ef5d4e2951024991914849cca1504af0f9a16045028e8d88ee222;
all-feature lint4cf86278c1e6d659943a46a3d7e3da429ac15be90f0f17742e24ef85228e7d3e.
The final bounded large-payload assertion also passes1.11s,
raw1b169d53665b20cd90a3a6c4fd91b9b67ddabc8dab5ecda10001af1f4706b63c;
its failure reports no payload dump and keeps exact byte equality.
Record exact source/PR/CI and separate automatic-main outcomes in Linear as they occur.

Correctness/security review covers pre-credit allocation, local authenticated
allowances, independent shared quotas, receiver order/extent/role, header budget
before variable header allocation, record/deadline immutability, incomplete/
failed buffer discard, final live input-state handoff and preservation of native
reservations. Separate complexity review removes duplicate template storage,
receiver state and repeated schema validation; it reuses borrowed admissions,
owned standard-library buffers and existing frame poisoning/gates. No new
framework, generic backend trait, retry path, scheduler or background collector.

## Acceptance still open

This is a bounded flow library with enrolled TLS and process evidence;
it is not a deployed partial-model runtime. Keep GLY-183 In Progress for driven
daemon pipeline/owner integration, slow-reader/ambiguous-execution acceptance and
qualified cleanup. GLY-181/185/186/187 and external235/236 delivery remain required.
The core's native StartStep/CompleteStep boundary is not relaxed. No native
workload, model, device, download, GPU, SSH, spend or milestone qualification.
Historical GLY-263/267 causes are not explained by these new controls or green CI.

## First source Windows result and immediate-failure correction

Source5fd2986105645e6a4180418f0c068b8c9803dee3,
[CI38871419428](https://github.com/Sigbound/glyd-compute/actions/runs/38871419428),
is CANCELLED, not qualified. Windows default116876149871 hits the unchanged10m
cap; its aggregate fails and five other platform/policy checks pass. All eleven
new flow controls and the extended child-process stage case report PASS on
Windows. The existing enrolled_worker_applies_whole_requests_once_and_retains_buffers_until_local_drain
case reports FAILED08:13:47Z, but its captured panic is lost when cancellation
prevents the final test summary. Raw log SHA256
7752ad72c58ba232d27fbf19445fc1d0e1f5bf4bc8cbc5b5336e54070aba5796.
Neither the failing assertion nor its initiating cause can be inferred from this.

The confirmed diagnostic gap is corrected: default suites use nocapture, and the
known failing Windows case runs first with the existing redacted generation-gate
trace plus a static four-case marker. The remaining default selection skips that
one exact full name, so it does not execute twice. All existing assertions,
platforms, warning policies, job count and10m cap remain. Passing controlled
fault tests may print caught panic text; process/test results determine failure.

Local traced lifecycle control passes all four cases1.01s, raw
7c46b7aa357140aca62bbcdd0f5013fb18c49d32cf13014b08522cf4235e45ce.
Actual compiled test lists prove174 total=1 early+173 remaining, disjoint and
complete on the local platform; raw selection proof
70892ef851a62b4618a2dec343ce419ce24b1ffe505c003751fbffd9d85b32f4.
The same static full-name predicate preserves Windows-specific cases. This fixes
visibility/failure ordering, not the unknown Windows lifecycle cause. Keep PR86
unmerged until the final exact source passes every check and its review gates.
