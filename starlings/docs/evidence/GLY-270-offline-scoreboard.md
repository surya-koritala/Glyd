# GLY-270 — Offline scoreboard foundation

Issue: [GLY-270](https://linear.app/getglyd/issue/GLY-270).
Parent [GLY-245](https://linear.app/getglyd/issue/GLY-245) retains actual build,
measurement and native acceptance. This change supplies strict records, bounded
artifact verification and an offline comparison command, with no dependencies,
backend launches, fixture qualification or CI job changes.

The [contract](../scoreboard.md) records all fields, metric directions, bounds,
five-repeat pairing and the conservative comparison rule. The example retains
all 22 planned cells as unmeasured. Upstream research records immutable official
Exo source/file pins and measurement differences; no implementation was copied
and no working-build or performance claim is made.

Controlled unit checks cover both comparison directions, exact speed thresholds,
overlap, incomplete/failing repeats, units and build/match bindings, duplicate
repeats, invalid numbers, missing metric kinds, fixture isolation, raw tampering,
unknown/duplicate JSON fields, nonregular files, path traversal and size limits.
The real CLI check preserves all missing rows, rejects implicit fixtures and
checks explicit fixture mode only exists in the development build. It creates
no runtime state. Final command outcomes and exact-source CI are recorded with
the PR in Linear.

Review boundaries: hashes establish supplied integrity and identity, not whether
a measurement is true. The tool reads a stable operator-supplied directory; it
does not confer runtime authority or harden a concurrently hostile filesystem.
Missing qualification stays `not_assessed`, even for successful comparisons.
Native collection, best-working Exo recipes, same-type multi-node hardware,
provider integration, independent milestone review and publication remain open.
Independent milestone review is deferred under the recorded authorization,
not represented as independent approval.
