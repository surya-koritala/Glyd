# GLY-184 exact header rejection and transport deadlines

[Automatic main39047592576](https://github.com/Sigbound/glyd-compute/actions/runs/39047592576)
at7bc090090374cd5042a2f58eeb1f3d547e2421c1 failed macOS job117439339269:
`malformed_or_unadmitted_headers_poison_the_stream_before_payload_read` observed
Transport instead of Denied. The security target reported127passed/1failed/
2ignored in31.28s. The raw job log is preserved locally with SHA256
bed5e6a7e21700e596f22322cbe2a99fb05138213ae3907cc91286860ead2d98.
Its assertion did not identify which session/sequence/dtype case failed or the
underlying I/O cause. Earlier exact-source39046952297 passed all eight checks;
that does not qualify the failed main.

The test conflated two obligations: exact semantic rejection of a complete
header, and fail-closed transport when a100ms absolute deadline can win before
that header is available. An I/O failure cannot promise a later semantic error.
The existing socket-free frame-clock control proves first errors are preserved,
late successes expire and expired operations never start (1PASS/0.00s, cached
offline two-job build10.77s). This is not attribution of the observed I/O failure.

The private header reader accepts standard `Read`; its body is unchanged. A
follow-up review found that manually composing parsing and admission in the
memory test could miss an ordering regression in the real receiver. The existing
receiver body is now one private `read_frame`: initial admission, bounded header
read, header admission, deadline check, payload allocation/read. The actual
receive path and the memory control both call it. Production still supplies
enrolled PeerStream and the same DataAdmission check; binding/frame deadlines and
the final locked enqueue/recheck remain outside it in their original order.
Flow callers keep their existing bounded header/credit/admission path.

The existing eight-case network fixture first feeds header-only Cursor bytes
through that actual receive body and admission gate. Every malformed/unadmitted
case must return exact Invalid/Denied independently of socket timing. A valid
header with absent payload must reach payload reading and return Transport.
Reading payload before header authorization fails this shared-path control.

The original network calls,100ms limit, panic-on-delivery and exact poisoned-stream
Expired assertions remain. Transport/Expired can satisfy the network refusal
only when observed after that deadline; otherwise the exact expected error is
required. Case, actual/expected errors and elapsed microseconds identify a future
failure. No production error mapping, timeout, payload/authorization rule, native
acceptance or numeric tolerance changes. No extra connection or traffic is added.

The bounded memory-only parser/frame control passes0.02s: complete header without
payload, incomplete-header Transport, malformed-prefix Invalid, header-budget
Capacity and actual frame admission before payload reading. A deliberate local
mutation moving header admission after payload reading fails in0.03s with
Transport instead of Denied. Restoring the exact ordered source passes0.02s
(build1.29s); the mutation was never committed or pushed. This proves regression
sensitivity, not the macOS I/O cause. Warning-denied security/all-target/
all-feature lint passes2.94s; formatting/diff checks pass. The same combined
network fixture runs only in existing CI. No local socket, service, CLI, model,
device, pressure or host inspection test, dependency, job, cap or retry is added.

Separate complexity review retains one direct extraction of the existing
read/authorize body and standard Read/Cursor; no duplicated production ordering
in the test or new transport hook. Correctness review traces framed/flow callers,
all admission and deadline checks, the mutation failure, no payload delivery and
permanent stream poisoning. Exact-source CI and separate automatic main remain
required. GLY184's physical/native/instrumentation and independent review stay
open; this corrects a test assumption and coverage gap, not the unknown macOS I/O
delay or historical Windows causes.
