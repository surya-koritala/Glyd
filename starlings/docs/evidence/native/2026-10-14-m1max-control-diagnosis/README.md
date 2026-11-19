# Native startup and control diagnosis passed

The separately approved single-use [GLY-261](https://linear.app/getglyd/issue/GLY-261)
control diagnosis passed on 14 October 2026 in 30.023 seconds. The worker reached
nonfixture qwen35 Ready, route hashing/activation/status passed, ready resource
telemetry reported sequence 12 and age 195 ms, and the coordinator retained zero
requests/body bytes. [Raw result](daemon-result.json) and [run log](approved-run.log)
remain unchanged. Two initial DAEMON_UNAVAILABLE status observations preceded daemon
readiness; a transient loading probe Unavailable then cleared before Ready. No
terminal fault occurred. Neither observation establishes the earlier failure cause.

One cached Mac CPU service start used two threads, zero generation/count API calls,
zero output tokens and no inference client token. The attempt cost $0. Sampled owned
RSS peaked at 1,688,109,056 bytes at 100 ms intervals, under the monitored 4 GiB cap;
sampling is not an allocation guarantee. One native server group was observed and
none remained after shutdown. Token rows were zero, all protected ownership hashes
matched, and new/old pins and quarantine files remain intact. Group absence does
not qualify allocation cleanup.

[Approval](user-approval.json) binds the [historical preapproval plan](approved-plan.md).
The [single-use marker](started-once) is consumed; no automatic repeat occurred.
[Preparation](preparation.json) retains its historical false preapproval flags.
[Worker route](worker-route.proposed.json), [route template](coordinator-route.template.json),
[UUID identities](identities.json), [pin receipt](direct-pin.json) and
[preflight](preflight-result.json) bind the actual isolated inputs. The runner is
reconstructible byte for byte from the prior frozen comparison runner using the
[GLY-262 control patch](../../GLY-262-run_control.py.patch): copy that runner to
run_control.py first, then apply the patch with git apply --unidiff-zero.
Runner SHA256 f447da3193204f430ba08ba3cb5294481343db89c0faeb012dd7c31ff7b0d8cc.
CLI SHA256 46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c.
Implementation 78b737495ff52fc69e05b928a460f400fcd4af70 matches production
Rust/manifests/lockfile in merged PR #62, 55989bde757be6e4839e2038e50678f102eb9466;
all seven source checks passed in
[CI 38609367481](https://github.com/Sigbound/glyd-compute/actions/runs/38609367481).
Runtime/model/configuration identities and exact limits are in the bound plan.

The earlier 93.235-second failed comparison remains failed. This successful
fresh sequence does not identify its discarded control response. GLY-261 attribution
stays open. Full direct/proxy JSON/SSE parity, installed Glyd integration, conversion
fidelity, two-worker qualification and the Exo scoreboard remain unqualified.
M2 remains open; M3 has not started. Correctness/privacy review checks hashes,
approval binding, limits, safe result fields and preserved ownership. Separate
complexity review reuses the existing runner/patch rather than duplicating code.
Independent milestone review remains deferred, not approved.
