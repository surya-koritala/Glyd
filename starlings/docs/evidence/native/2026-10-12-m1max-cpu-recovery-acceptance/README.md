# Native CPU recovery acceptance with separate final verification

PR #36 implementation `419a6fdbb6e26d008d7e407ce3c4121738dbd86f` was reviewed and
merged as `e13418f0d397c5ca131f1a87c8ab7159a2a950a8`. Corrected control runner
PR #37 merged as `ab743b518ba6352bfeefe8b067cecd9140776772` after [all five checks](https://github.com/Sigbound/glyd-compute/actions/runs/38331446482)
passed on `1ad905020c1d6aa7dcabb4dc59269aae67352efd`: Linux x64 3m46, ARM64
3m55, macOS 3m35, Windows 6m32, dependency policy 19 s. Warnings were errors.

The renewed single-use approved experiment ran two native requests on M1 Max CPU,
two threads/one slot, ≤8 output tokens each, count-call upper bound four, cached
BF16 artifact/F32 KV, logical context 128/allocated 512, loopback only, $0. Native
run 43.396 s; sampled owned peak RSS 1,716,879,360 bytes, no owned group remaining.

Both native requests and recovery assertions completed: JSON usage/drain, SSE
revocation after real text with no DONE, qualified whole-service cleanup and zero
failed reservation, exact durable receipt, wrong-lease refusal, explicit admin
recovery/repeat, recovered state/no pin and offline repeat. Original raw runner
reports FAILURE at the final `model paths` read-only cache verification because
its three-second subprocess deadline was too short to hash the 1.52 GB artifact.
Do not describe that raw runner as passed or change its false result flags.

Separate bounded read-only verification completed in 11.102 s with a 30-second
command deadline and no further model start, generation or count call. It verified
every asset, exact recovered lease/configuration, missing matching pin, offline
idempotence, zero token rows and all six older ownership files unchanged. Combined
acceptance is recorded separately in `post-run-acceptance.json`. The marker/raw
failure/terminal record remain preserved. No model retry, restart, GC or old
quarantine recovery occurred. Approval is consumed.

`file-verification-runner.py` changes only that read-only command deadline to
30 seconds for future approved runs. It was not run against a model; this evidence
is the original unchanged runner plus separate verification, not a retroactive
pass for a modified script. No new native run is needed to establish the observed
recovery assertions. File hashing is separate from generation/count deadlines.

The source/build cleanup review is bound before launch. This is controlled
CPU/no-device/no-escape policy, not attestation or a general memory/production
profile. No arbitrary-prompt envelope, actual ENOSPC/power-cut, GPU/escaped-work/
Windows cleanup, base conversion fidelity, physical Pi or real two-worker claim.
M2 remains open, M3 unstarted; independent review deferred, not approved.
