# GLY-196 current support classifications

Issues: [GLY-196](https://linear.app/getglyd/issue/GLY-196),
[GLY-199](https://linear.app/getglyd/issue/GLY-199).
Initial base: mainb5c75baafb30d3e2863637b2b14d453b63f23724.

Reuse the operator guide's existing support table. Explicitly distinguish
experimental control-plane software, historical task-only CPU evidence,
unqualified delivery/installation targets and currently unsupported GPU routing
or single-model pipeline execution. Define layer-ready against actual provider,
allocation, conformance and drain evidence; no current configuration receives
that classification. Split the former combined unsupported/unqualified row so
missing evidence does not imply a backend cannot implement the capability.

Every row links existing source contracts or exact historical evidence. No
upstream performance number, model support, current approval, Exo win, memory
saving or release qualification is introduced. Native artifacts, matched 2026
model/level/context trials, review and actual installation remain open. The
existing collector already implements scoped-thread batches and batch-wall
timing; do not add a second concurrent collector for this documentation work.

Correct the README's stale pre-PR108 CI ordering: the Windows default job now
runs the unchanged short all-feature catalog group, independently of the backend
job's long cases. Backend/CLI execution is serialized; the other non-CLI default
tests retain normal Rust harness scheduling. The aggregate
requires both; a short-case failure cannot stop the other job from starting.
This support-table preparation changes no runtime permission, dependency or test
selection. The later registry diagnostic correction is recorded separately.

Separate complexity review keeps one existing table and a corrected paragraph.
Correctness review compares each status against the linked source/evidence,
including CPU-only production admission, proposal-versus-native boundaries,
consumed historical approvals, WSL2 uncertainty and inactive exports. File-link
and diff checks are the local validation; no service, model, instrumentation,
network test or security campaign runs. All52 relative file links exist and the
diff whitespace check passes. Exact-source/current-main/review/CI and
separate automatic-main gates remain required before qualification.

This prepares the support-table part of GLY-196/199; neither full issue closes.
Independent milestone/release review is deferred, not approved. The license and
publication decision are unchanged.
