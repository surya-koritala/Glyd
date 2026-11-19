# Native recovery attempt and model-free control correction

The one explicitly approved experiment used reviewed implementation
`419a6fdbb6e26d008d7e407ce3c4121738dbd86f` after PR #36 merged as
`e13418f0d397c5ca131f1a87c8ab7159a2a950a8`. [All five checks](https://github.com/Sigbound/glyd-compute/actions/runs/38330352244)
passed: Linux x64 3m49, ARM64 3m54, macOS 2m57, Windows 5m34, dependency 19 s.

The attempt FAILED in 29.030 s after readiness and before generation/count calls.
The original runner captured only RuntimeError, so the exact rejected operation
is unknown. It did not exercise midstream revocation or explicit recovery. Raw
result remains unchanged; no native recovery qualification is claimed. Peak
sampled owned RSS was 1,684,930,560 bytes, no observed owned group remained, and
spend was $0. Two threads/one slot/five-minute/4 GiB monitored limits were retained;
no download, GPU, Ubuntu/Pi, retry, second model start or old-quarantine reset.

The new exact launch produced a durable `drained` receipt on idle shutdown and
kept its pin. This is an idle cleanup observation, not failed-request/recovery
acceptance. Four historical ownership files remain byte-for-byte unchanged.
The runner suppressed a failed token-withdrawal command; read-only verification
found its row still present. A control-only coordinator restart then revoked the
token and stopped normally, without any worker/model/route activation. Final
verification shows zero token rows. The marker and all ownership records remain.

A model-free admin reproduction explains a plausible harness failure: eight
unpaced status commands succeed and the next four encounter DAEMON_UNAVAILABLE
at the eight-per-address admission limit; ten paced reads succeed. This is not
proof of the original missing error. The corrected runner paces local cluster
commands at 0.2 s per state, records only operation/error codes and failure stage,
waits for a non-null ready resource observation and treats token/stop command
failure as failure. Admission limits are unchanged; no request retry is added.
Its exact control helper is exercised against a bare coordinator, plus closed
endpoint and missing-resource checks. It starts no model. A further native
attempt needs separate reviewed preparation and explicit single-use approval.

The cleanup policy is trusted operator review for the pinned CPU/no-device/
no-escape topology, not attestation or a general profile. Source/build and
executable hashes are recorded. GGML_NO_BACKTRACE=1 skips the diagnostic fork;
subprocess and all device backends are disabled. No GPU/escaped-child/Windows,
base conversion fidelity, arbitrary-prompt memory or physical Pi claim follows.
M2 remains open; M3 unstarted. Independent review deferred, not independently approved.
