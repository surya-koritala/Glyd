# GLY-272 — Keep optional cache writes within the CI budget

Main `744a40b6071caae6f50fbe418cd5df77af0aa6d0` automatic CI
[38672809547](https://github.com/Sigbound/glyd-compute/actions/runs/38672809547)
cancelled both Windows jobs during optional cache compression after their tests
passed. The aggregate correctly failed; this main run is not green.

| Job | Completed tests | Last test / cache start / cancellation (UTC) | Raw log SHA-256 |
| --- | --- | --- | --- |
| Windows backend116219601540 | All library, CLI and8+1+1 catalog groups pass | 02:56:26 / 02:56:38 / 02:57:05 | 7e8041e1799caa231defbd72c566eeaec95a1908ca069e4feee72c1f3db58197 |
| Windows default116219601537 | All default tests pass; one existing ignored case | 02:56:36 / 02:56:49 / 02:57:07 | cba7e5d3eb6688d556d793756573715745e5bbf0cdca16fedd8676f27837a061 |

The backend trimmed1.7GiB to1.5GiB before starting archive compression with less
than30seconds left. A previous source-qualified Windows save took about11seconds;
that does not guarantee a later host's compression/network duration.

The existing first step now records the job's initial wall time. Optional cache
preparation is skipped once seven minutes have elapsed. Save eligibility is
checked again after preparation, reserving roughly three minutes of the unchanged
ten-minute cap when a save starts; setup precedes the first-step timestamp.
Missing start time also skips optional work.
The gate avoids the proved late-start case; the job cap still bounds any later
stall. Existing restore keys, Cargo validation, cold-cache trimming and saves
after early test failures remain. No test, platform, assertion, job, dependency,
deadline or warning check changes. No blanket error suppression is added.

Controlled checks execute the actual extracted workflow block with cleanup
commands mocked: adequate budget performs preparation and sets readiness; late
or missing start performs no cleanup and no save; preparation consuming the
reserve cannot set readiness. A retained sentinel survives all four cases.
Machine-readable check SHA-256:
`3f5036d0495f21773bfbddbd138b021a85c81bd8750b31bda6221b45354e4806`.
Formatting/diff and correctness/complexity review accompany the PR; exact-source
seven-check CI and post-merge verification are required.

This is a proved optional-cache timing defect. The previously captured panic
messages in passing short controls are intentional caught-panic tests, not a new
initialization failure. Historical GLY263/267 causes remain separate and open.
GLY270's offline foundation retains its successful source qualification; native
and provider acceptance remain open.
