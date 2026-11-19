# GLY-204 private candidate build receipts

Issues: [GLY-204](https://linear.app/getglyd/issue/GLY-204),
[GLY-198](https://linear.app/getglyd/issue/GLY-198).
Initial base: main48c18b32e15aa5985a7867de3fb34d6c9f6a964e.

Reuse the candidate builder's existing pinned/default release compilation with
native Cargo json-render-diagnostics output. CARGO-BUILD.jsonl joins the fixed
payload and inner/outer checksums. There is no extra build/resolver, downloaded
tool, parser framework or metadata execution. Source/target/flags and the lock
remain recorded; source is checked clean/unchanged before and after compilation.
Warnings remain denied and compiler diagnostics still reach stderr; failure
cannot stage an old executable as a new candidate. Empty receipts are refused.

The [Cargo message contract](https://doc.rust-lang.org/cargo/reference/external-tools.html#json-messages)
provides package/artifact identities, features/profile, fresh-cache markers and
build-script data for the actual invocation. It is private review input, not a
complete linked-binary/system-library/SDK/backend/model SBOM, license/notice
clearance or reproducibility attestation. Local paths and build-script values
require review before any approved distribution; raw receipts are not uploaded.
Arbitrary compiler/macro text must not be executed or treated as trusted commands.

One bounded zero-dependency Rust experiment on the Mac verifies the pinned option:
an intentional unused variable fails under -Dwarnings, its human diagnostic stays
on stderr and stdout ends build-finished=false; clean source then produces an
artifact and build-finished=true. Each invocation has a20s cap/two jobs; no binary
is executed, dependency downloaded, service/model started or process inspected.
Private raw files in work/ have SHA256:

- m6-cargo-receipt-negative.jsonl:7c4af99fd1ea2a26d5425d34c5279d692c40dff9303bccfa457c8b1058299ee8.
- m6-cargo-receipt-negative.stderr:5413ceb6de412104edc3a6c35744c75926a310fa19512377991a3c21b88751a7.
- m6-cargo-receipt-positive.jsonl:2257d5a8050425bf52fd77a6a03f631f4a8ef63c96ea0ca85ce08d47778aea08.
- m6-cargo-receipt-positive.stderr:9624157acc7e62ddd2c7591e14bcfe7c0aec5ea536cb5a60ad3f54dd22543852.

The existing Unix archive fixture passes0.13s (4.43s cached compilation). It checks
all seven entries, all five payload hashes, binary mode, private-data exclusion,
binary tampering and receipt-only tampering after restoring the binary, plus
existing/symlink/unsafe-output refusals. Shell syntax and diff checks pass. No new
test binary/workflow/job. Initial native experiment/fixture proof is not a newly
built real candidate, supported installation, native inference or full SBOM audit.

Separate complexity review keeps one native output flag and one fixed payload
file, reusing existing checksums/fixture. Correctness review covers stdout/stderr,
unchanged denied warnings/offline flags/source guard, exact archive membership,
independent tamper detection and private/provenance limits. Current-main/exact
source CI/review/merge and post-merge main checks remain required; independent
milestone/security review is deferred, not approved. License/publication unchanged.

A real private candidate was then built from clean merged source
3074642974f954c82866817023b8ef55242a8abd (including main05aa809 / PR107).
The one offline native aarch64-apple-darwin release build finished in153s under
the280s compilation cap, with two jobs and denied warnings. No application or
model was executed. File-only verification checked the outer checksum, exact
seven-entry archive, all five inner payloads, binary mode, source/target/flags,
empty feature selection and successful optimized non-test CLI artifact.
The233680-byte receipt contains248 Cargo records,216 artifact records, zero
fresh-cache artifacts and zero non-JSON lines. Its actual invocation is not a
claim about future cache reuse, reproducibility or supported installation.

- Archive SHA256:bf25159dc4474d488df472d323c559e026d1388d5e178558233b08d61aba9175.
- Binary SHA256:832d5d1039bb668bce40f65b0dabd6c12a45ea3f872ff28aab7647a80d0ba82e.
- Receipt SHA256:d214d8c3f4a002329064cfb34ba5952c4e6a814416bb52387865af97e278d828.

The archive, receipt and verification report remain private local outputs;
only these hashes and bounded summary are recorded here. Spend was $0.
BUILD.json retains qualification=not_assessed and license=undecided. This checks
the builder, not final-source CI, native inference, a full SBOM or release gates.
