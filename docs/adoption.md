# What adoption takes

What a Linux distribution, a cloud storage system or a large
deployment needs before it will store data in a format, and where Glyd
stands on each (v0.14.4, September 2026). Ratio is not on the list
because ratio is what gets a codec looked at; the items below are what
gets it kept. Each line says done, partly, or not, and what the gap is.

## Distributions, kernels, core tools

The hardest audience and the one that makes the others possible: what
`apt install` and `tar` carry is what everyone else trusts.

| Need | Why | State |
| :--- | :--- | :--- |
| A frozen format with a written spec and test vectors | Data stored today must decode in twenty years; a second implementation must be possible from the document alone. | **Partly.** `docs/spec.md` describes every envelope and block; every release still decodes every earlier format (`tests/data/legacy`). The format is not frozen: four block flags and one stream tag were added this month. A 1.0 needs a freeze, a spec someone else can implement from, and vectors for each block kind and flag. |
| A C library with a stable ABI | The consumers are C: libarchive, language runtimes, the kernel's tooling. | **Partly.** `include/glyd.h` exposes compress/decompress at every level, parallel and sequential, bases, packs and the store; the Rust crate builds the library. Missing: a streaming (chunk in, chunk out) C interface, an ABI version and a promise of what stays. |
| Builds without the Rust toolchain for the user | Many distributions will package a Rust crate but every downstream `configure` expects a `.so` and a header. | **Partly.** Release assets carry the library; a `pkg-config` file, `.deb`/`.rpm` from the release workflow and a distro-style build (no network at build time) are not done. |
| Bounded, declared memory in the decoder | A decoder that allocates whatever the frame asks for is a denial of service; a frame must say its window and the decoder must refuse more than it was told to spend. | **Partly.** The window is fixed (8 MB, 2^23 offsets); parallel units are 8 MB; the decoder allocates the output and a batch. Missing: a declared per-frame memory bound and a decode API with a caller-set ceiling. |
| Streaming both ways with a small buffer | Pipes, `tar`, backups, anything larger than memory. | **Partly.** `compress_stream` / `decompress_stream` in Rust feed a sink in order with a batch of memory; the CLI uses them. Missing: a true chunked reader/writer (input arriving in pieces), and the same in C. |
| Continuous fuzzing, no panics on any input | The decoder is attack surface. A maintainer's first question. | **Partly.** Mutation fuzz tests run in `cargo test` (they caught the checksum weakness this month); corrupted-input tests cover every envelope. Missing: fuzzing as a job that runs all the time (cargo-fuzz / OSS-Fuzz), a panic-free guarantee documented and enforced (`panic = abort` audit, no `unwrap` on input-derived values). |
| A security process | Who to tell, how fast it is fixed, how it is announced. | **Done** in outline (`SECURITY.md`); no advisory has been issued yet, so the process is untested. |
| License everyone can carry | | **Done.** Codec `BSD-3-Clause OR GPL-2.0-only`, zstd's pair. The store is BUSL and stays out of distributions. |
| Interop plumbing | `file` magic, `tar --use-compress-program`, `Content-Encoding` registration, libarchive, Python `tarfile`, editors. | **Not.** Each is small; together they are what "supported" means. Magic bytes are stable (`GLYD*`, `SIMD`, `G`). |

## Cloud storage systems and pipelines

| Need | Why | State |
| :--- | :--- | :--- |
| Cost per TB stored and read, on their machines | Storage is bought in $/TB-year after CPU. | **Done and reproducible** on Graviton3 and Sapphire Rapids (`benchmarks/`, `scripts/aws_workbench.sh`); the numbers in the README are from those runs and the scripts that made them. Every claim must stay that way. |
| Decode speed and CPU per read | Written once, read many times. | **Done as a codec** (1.3–2.4× zstd -d on 8 vCPU to memory); **partly** as a tool: with the output written to a filesystem, reads are ~1× zstd on a saturated box because the write itself is the cost (see the v0.14.2 notes). |
| Write speed at the default level, on all cores | The first number a platform team checks: `zstd -3 -T8` on their box. | **Partly.** Since v0.14.4 (`--max` without the far pass, as zstd -3): 0.99–1.14× zstd -3 -T8 on events, logs and mozilla, 0.84–0.86× on a table dump and enwik8; one core 0.74–0.95×. What is left is the parse loop and the entropy coder each running more instructions a byte than zstd's on the same work. This is the open engineering item. |
| Random access into an object | Range reads, one record of a pack, one file of an archive, without decoding the rest. | **Partly.** Packs index their objects; parallel units are independent streams. Missing: a first-class per-frame index (unit offsets in the frame header) and a range-read API. |
| Corruption detection they can trust | Every block checked, no known blind spots. | **Done** since v0.14.2 (CRC-32C per block; the earlier sum had a blind spot). Records and cold envelopes still use the older sum. |
| Predictable behaviour | No level that is sometimes 3× slower; no input class that surprises. | **Partly.** Container and JPEG opening, the far pass and record mode are each gated; the gates are measured but the slow paths exist. Needs a documented worst case per level and a switch to turn each opener off. |

## Large deployments

| Need | Why | State |
| :--- | :--- | :--- |
| Never a byte lost | | **Done** as far as testing goes: every write verified against its read where a reconstruction is involved (containers, JPEG), checksums on every block, legacy fixtures, mutation fuzz. Continuous fuzzing (above) is the gap. |
| Speed parity at the level they would use | | **Not** (write speed, above). |
| Bindings that install from the usual places | PyPI, Maven Central, NuGet, Go modules, `apt`. | **Partly.** Python and Go bindings exist and wheels are built by the release workflow; nothing is on PyPI or crates.io yet (the workflow is ready, the tokens are not set); JVM and .NET do not exist. |
| A versioning promise | Semver on the format, deprecation policy, long-lived releases. | **Not** written. The practice is right (every release decodes every earlier format); the promise is not stated. |

## The order

1. **Write speed**: the parse loop and the entropy coder taken to
   zstd's instruction counts (the far pass is opt-in since v0.14.4).
   This is the one objection every audience raises, and it is
   measurable.
2. **The 1.0 track**, in this order: memory bound declared and enforced;
   chunked streaming in Rust and C; a fuzzing job; a panic audit; the
   spec rewritten so it can be implemented from; the format frozen with
   test vectors; ABI version; `pkg-config`, `.deb`, `.rpm`; PyPI and
   crates.io.
3. Interop plumbing (`file`, libarchive, `tarfile`, `Content-Encoding`)
   once 2 is done — there is no point registering a format that may
   still change.

Container opening, JPEG recoding, record mode and the store are what
sets Glyd apart; they come after the two above, because a format that
is not adoptable does not get to show them.
