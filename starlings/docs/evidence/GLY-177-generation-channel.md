# GLY-177 generation trust boundary

The remote worker previously had no usable generation channel: ordinary peer
streams require the authority's live policy database. A dedicated bounded TLS
channel now lets a worker authenticate the enrolled coordinator using only its
own identity, the CA/public-key pin and its independently maintained registry
fence. Both endpoints agree on immutable model/profile and full authority epoch.
See [the contract](../generation-channel.md) for exact limits and obligations.

No new dependency, runtime, background thread, daemon listener or real workload
is introduced. Existing Rustls, identity storage, listener budgets and registry
state are reused. Client-token policy stays on the coordinator. The larger
generation-record limit does not widen existing control/admin limits.

Synthetic enrolled loopback tests cover large records, bidirectional exchange,
wrong model/profile/epoch, missing model permission, key/name mismatch, valid
certificate refresh, revocation, drain, offline fencing, duplicate/malformed JSON,
oversized length prefixes, exhausted record budgets and cancellation of a reader.
Workers in the test have no authority database or CA private-key file.

Validation uses the pinned toolchain with warnings denied: workspace formatting,
all-target/default and all-feature Clippy, and compute-security tests. Required
four-platform CI plus dependency policy must pass before merge; exact head and
run URLs are recorded in Linear. Correctness/security review and Ponytail
complexity review are separate passes; independent approval is not claimed.

GLY-177 remains in progress. The daemon generation route, client-token binding,
worker lifecycle/service integration, cache pins, resource qualification and
native end-to-end acceptance are not completed by a transport test. M3 remains
closed until M2 acceptance is complete.
