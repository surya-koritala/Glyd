# Immutable model catalog

`model seal DRAFT SOURCE_DIR` streams every declared file through SHA-256
with a 64 KiB buffer, checks its exact declared length, fills metadata hashes and
computes the existing RFC 8785 manifest identity. Paths are schema/semantically
validated before opening files through capability directory handles. The source is
read-only to this operation; read-only preloaded directories are supported without
permission changes. Escaping symlinks/reparse paths, ambiguous paths and non-regular
files are refused. Cache state uses stricter owner-only protected files.
The JSON input/output is capped at 1 MiB. No network request occurs.

The manifest pins a full 40-character source revision. Mutable branch aliases
are refused. Source revision, creation evidence, license, architecture and layer
inventory are supplied provenance requiring operator review; a content hash does
not independently authenticate a publisher or prove a complete tensor inventory.
`model resolve REPOSITORY REVISION` explicitly queries the Hub API and returns
a full commit and repository creation evidence. This does not approve the source,
license, inventory or model. No other local command resolves a branch implicitly.

Each storage shard must be referenced by a layer/component. The catalog requires
one config/tokenizer/template, contiguous declared layers, unique components,
executed first/last components, and explicit scope/precision/transformation fields.
It preserves recurrent layer profiles and excluded/unsupported components.
It does not parse tensor formats, infer backend support, or certify lossless
execution. Text-only scope and lossy transforms remain visible, never promoted
to whole-model equivalence. Native adapter conformance supplies that later evidence.

After profile initialization, use the following local commands:

```text
starlings model inspect manifest.json
starlings model register manifest.json
starlings model approve MANIFEST_DIGEST
starlings model check MANIFEST_DIGEST
starlings model show MANIFEST_DIGEST
starlings model revoke MANIFEST_DIGEST
```

`--state-dir` selects an existing profile; its configured cache owns `models/`.
Registration grants no permission. Approval is an explicit local decision bound
to the exact manifest digest, stored separately under protected durable atomic
files. `approve ... --redistribute` additionally requires the manifest's explicit
redistribution declaration; local-use approval never enables transfer by default.
Re-registration preserves approval. Revocation preserves metadata for inspection.
One standard file lock excludes other processes; mutable Rust access serializes
in-process writers and GC. Interrupted writes never
publish partial policy. Same-user/root compromise is outside this boundary.

`check` reports approval only. `ready` rehashes all stored objects and checks their
exact sizes, still reporting inference validation false. Consumers check approval
on each new operation; previously returned metadata is not a durable grant. Live
execution revocation and automatic lease pinning remain adapter integration gates.

## Verified objects and explicit network access

```text
starlings model import MANIFEST_DIGEST SOURCE_DIR
starlings model fetch MANIFEST_DIGEST
starlings model ready MANIFEST_DIGEST
starlings model cost MANIFEST_DIGEST FIRST_LAYER END_EXCLUSIVE
starlings model storage
starlings model quota BYTES
starlings model pin MANIFEST_DIGEST LEASE_UUID
starlings model unpin LEASE_UUID
starlings model gc
```

Import streams through protected temporary files, validates exact byte length and
SHA-256, then durably publishes `object-SHA256`. An existing verified object is
reused without reading the source again. The source cannot supply cache filenames.
All import/readiness operations work offline. Readiness hashes whole files once
in preparation; it must not be called per token. No automatic backend starts.

Only explicit `resolve`/`fetch` commands create a provider HTTP client. Fetch uses
the approved manifest's pinned revision and paths. It allows HTTPS on the Hub and
four named artifact hosts, with at most four redirects and a 60-second per-asset
transfer deadline. Environment proxies and compression are disabled. Unknown
hosts, credentials in URLs, non-443 ports and malformed redirect URLs are denied.
A supplied library credential is sent only to the Hub host and is never logged or
forwarded to artifact hosts. The CLI currently supports public repositories only;
it does not discover credentials from the environment.

Interrupted transfers retain a bounded prefix. The next explicit fetch sends Range;
an exact Content-Range resumes, a full 200 response restarts, and any other range
response fails. Final hashing includes both prefix and suffix. Hash mismatch drops
the invalid prefix. There is no automatic retry loop or hidden background download.
Errors omit tokens and signed URLs. Tests use synthetic content and mocked HTTP;
no real model download or provider availability claim is made.

## Storage limits and recovery

The default import budget is 16 GiB. Admission charges all current cache files,
including metadata, partials and orphaned temporary files, then conservatively
reserves two full object lengths for staging/checkpoint publication. This can
refuse an import that would fit after cleanup; raise the explicit quota or run GC.
The budget governs artifact writes. Small administrative policy/quota records can
be written beyond that budget so storage pressure does not prevent revocation.
The directory has a 16,384-entry bound, including temporary files; new writes
leave room for atomic publication. Actual filesystem exhaustion can still occur
because other programs consume space. Failed writes never mark an object ready.

Durable pins bind a lease UUID to one manifest. A conflicting rebind fails. GC
retains every pinned object, including after revocation/restart; pins are released
only explicitly after backend drain. GC removes unpinned objects, partials and
orphaned atomic files under exclusive ownership. There is no second database index
to reconcile. A crash during deletion may retain extra files/pins and does not
justify releasing backend allocations. Adapter-managed pin lifetimes are still
required before live inference. Local filesystem power-cut guarantees and actual
OS disk-full fault injection remain unmeasured.

Model-free fault checks cover quota exhaustion, protected publication/write
refusal, restart/orphan cleanup, an injected StorageFull reader error and exact
recovery receipt/pin preservation. Filesystem seek, digest, enumeration, deletion
and directory-sync errors report Storage rather than network Transport. A failed
staging hash/seek does not trigger another checkpoint write onto a failing volume.
These checks do not fill the host disk, mount a filesystem or relax private-file
permissions in production.

`cost` reports unique whole-file storage needed for a selected layer range,
including required first/last components and metadata. Storage shards are not
execution partitions: one selected layer can require an entire multi-layer file.
Declared resident weights are reported separately. The transfer buffer is 64 KiB;
backend peak host memory remains null until a qualified backend measures it.

`max_asset_import_reserve_bytes` is twice the largest selected unique asset's
declared size: conservative additional quota for its staging and checkpoint.
Imports are sequential; this is not twice the whole model, current cache usage,
observed free disk or backend memory. Already verified objects need no fresh
import reserve. Admission also charges all existing metadata, objects, partials
and orphans. Arithmetic overflow is refused. The command is a metadata report,
never qualification or permission to load an execution partition. Exact backend/
workload allocation profiles belong to GLY-175; file bytes and declared resident
weights cannot establish peak host RAM.
