# GLY-260: observe invitation commit after a lost control reply

[GLY-260](https://linear.app/getglyd/issue/GLY-260) separates fixture contract
handling from the unproved Windows commit slowdown. PR61 source
`98768fe7280dd92f1c42a647d7519b3aafb89e5f`,
[run38597196561](https://github.com/Sigbound/glyd-compute/actions/runs/38597196561),
Windows backend job115966860686 failed; the Windows aggregate also failed, and
the other five checks passed. Keep that source/run failed rather than rerunning
unchanged source or substituting a later green run.

## Genuine phase evidence

The full production routing matrix passed. The service survival/restart fixture
then failed issuing the startup-crash worker's invitation. Its CLI reported
`REFUSED: CONTROL_UNAVAILABLE: outcome unknown; check status`; the coordinator
recorded `CONTROL_WAIT_EXPIRED phase=executing invitation_step=commit`.
Raw log: `work/m2-pr61-windows-job-failure.log`, SHA-256
`633afa9e912e64bfeddb01f77f4fc27533663eca4b43952b11d8b8dec96d8a17`.

`InvitationIssuer::invite_with` places that marker immediately before SQLite
`transaction.commit()`. This identifies the observed wait phase; it does not
prove an OS scheduler, VFS or flush cause, whether commit eventually succeeded,
or the initiating causes of historical GLY-248/251/253 events. No SQLite
durability, locking, production timeout or authorization behavior is changed.

## Demonstrated fixture cause and correction

The positive invitation fixture calls `Daemon::admin`, which requires immediate
CLI success. A started control mutation may correctly finish after its response
wait or lose its reply. Reissuing could duplicate the mutation; generic cluster
status and file existence cannot establish commit. The existing
[`invite-status` contract](GLY-247.md) already verifies the protected invitation's
authority, token row, role, expiration and current admission state without mutation.

The six positive invitation call sites now use the existing mutation classifier,
issue exactly once, and require one authenticated `invite_status` reply with
exact code `INVITATION_STATUS` and message `active=true` before enrolling. Inactive,
missing, malformed, unauthorized, transport-failed and refused outcomes do not
authorize progress or retry. No observation loop or new deadline is added;
the existing three-second control reply and twelve-second CLI watchdog remain.
The roles fixture's duplicate active check is removed because the shared helper
retains that exact assertion through authenticated IPC. Consumption/refusal checks
and every service/registry/ownership assertion remain.

The shared mutation helper takes a state path instead of a whole daemon; existing
permissions/revocation callers keep the same behavior. No new product API,
abstraction, dependency, CI job or invocation retry is introduced.

## Controlled verification

`catalog_service_invitation_reply_loss_requires_active_observation_without_retry`
(subsequently broadened to token replies and renamed
`catalog_service_publication_reply_loss_requires_active_observation_without_retry`)
uses the existing authenticated AdminListener and real CLI, with no model service.
It receives exactly one invitation mutation and deliberately drops its reply.
The shared classifier admits only the exact established started-unknown outcome.
One read-only query accepts a committed/active reply and rejects an inactive reply
even with a synthetic published file. No second mutation remains on the listener.
The original strict-success assertion would reject the same lost reply; the genuine
Windows run records that failure. This reproduces the contract mismatch, not
Windows disk behavior or a real delayed SQLite commit.

The controlled check passes locally in1.58s. The existing mutation classifier
negative test rejects not-executed, policy-refused and similarly worded errors.
All15 catalog lifecycle cases pass in210.54s in an explicitly serialized local
run, including the production matrix and survival/restart case. The default
roles/enrollment/drain/restart case passes in8.33s after removal of its duplicate
query. Final formatting and both workspace/all-target warning-denied lint modes
pass. These local durations are not a CI speedup measurement. Record final
exact-head seven-check CI/merge identities in Linear. The native SSE model-free
checks remain separate.

Correctness/security review traces all positive callers, single mutation/observation,
strict code/active checks, deny paths and preserved live revocation/quarantine gates.
Separate Ponytail review reuses the existing classifier, IPC and read-only contract;
there is no timing workaround. Independent milestone review remains deferred,
not approved. M2 provider/native/physical qualification remains open; M3 unstarted.
