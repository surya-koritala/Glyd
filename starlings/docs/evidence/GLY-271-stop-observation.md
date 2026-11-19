# Observe completion after a started stop outlives its reply

[GLY-271](https://linear.app/getglyd/issue/GLY-271) corrects the lifecycle fixture's
stop acknowledgement assumption. [PR70](https://github.com/Sigbound/glyd-compute/pull/70)
source `a383b9731be52ba4729fa132898c57a2a5d1f0af`,
[CI38666785333](https://github.com/Sigbound/glyd-compute/actions/runs/38666785333),
Windows backend job116200450536 fails at `ok(cluster stop)`, not an admission or
interface assertion. Raw log SHA-256:
`145f839991f554865c5ff9475a27dacbf4ed6d785ce82bedab811ac3377de33b`.

The daemon records executing-phase reply expiry, execution lasting 3688 ms and
then STOPPED. The existing three-second RPC correctly returns outcome unknown;
the fixture panics before its existing owned-process completion observation.
The client-grant maintenance-gap flags in this record occur during stop. They
do not establish the initiating cause of any earlier active registry/queue gap,
nor does the record identify which OS operation made stop take 3688 ms.

All 29 fixture stop submissions now reuse the existing `mutation_once` helper.
It permits only success or the exact started-unknown result, never retries, and
rejects queued-not-executed or unrelated errors. Every following process wait,
successful or expected failure exit status, transient endpoint removal, retained
configuration, allocation and quarantine assertion remains in place. Unknown
acknowledgement is not successful completion; the original twelve-second process
observation still decides that. The existing unqualified-drain failure loop is
unchanged.

The existing real authenticated lost-reply case now also receives one Stop
command, discards its reply, calls the same mutation helper and checks that no
second request or offline mutation occurs. Full lifecycle cases separately
observe actual owned-process exit. This extends an existing case without a new
framework, sleep, job, timeout or production branch.

The local revised fixture passes all 18 catalog lifecycle cases in 56.57 seconds
and all ten default daemon cases in 28.35 seconds. Formatting and both workspace/
all-target Clippy modes pass with warnings denied. Correctness/security review
checks all stop callers retain their completion/cleanup assertions; separate
complexity review reuses the existing mutation helper without another wrapper.
The initial failed source remains failed. Final revised-source CI and review
are recorded with PR70 and Linear. No production control result, gate, retry,
cleanup or authorization semantics change. Broader GLY-267 attribution remains
open, provider-dependent M2 acceptance is incomplete, and independent M3 work may
proceed under the updated project direction. No native/model/GPU/PC/Ubuntu work,
downloads, spending or old quarantine recovery is performed.
