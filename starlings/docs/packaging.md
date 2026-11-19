# Private development candidates

GLY-198's candidate builder stages the default CLI, build facts and checksums.
It does not establish an installed desktop release. Service installation,
clean-machine lifecycle/upgrade/rollback, WSL2 GPU/network behavior and native
backend qualification remain open. The Mesh license and public release method
are undecided; staging a private candidate grants no publication or distribution
approval.

From a clean, committed checkout with the pinned Rust1.97.1 toolchain, native
linker and locked dependencies already available:

```sh
mkdir -p target/candidates
bash scripts/build-candidate.sh "$PWD/target/candidates/first-check"
python3 -I scripts/verify-candidate.py "$PWD/target/candidates/first-check"
```

The parent directory must exist; the output directory must be new. The builder
uses locked offline Cargo, two build jobs, release mode and no default CLI features.
Its compiler flags are fixed to `-Dwarnings`, overriding caller Rust flags for
this build; those flags are recorded in the manifest.
It checks the clean source revision before and after compilation. It starts no
application, service or inference backend. Only the native compiler host is built;
candidate target allowlisting covers Linux GNU x64/ARM64 and macOS x64/ARM64.
That list is not a qualified deployment/support matrix. WSL2 uses the Linux build;
native Windows packaging is not implemented here. No cross-compilation, model
download, package upload or installer runs.

The private output directory contains one `starlings-VERSION-TARGET.tar.gz`
and its `SHA256SUMS`. Its single top-level folder contains:

* `starlings`: default CLI executable, with executable permissions.
* `BUILD.json`: version, target, full source revision, toolchain, empty feature
  CLI feature selection, compiler flags, `fixture: false`,
  `qualification: not_assessed`, `license: undecided`.
* `Cargo.lock` and `RUSTC.txt`: exact dependency lock and compiler facts.
* `CARGO-BUILD.jsonl`: Cargo's output from this exact build invocation, including
  artifact package IDs, target/profile/features, cache reuse and build-script data.
* `SHA256SUMS`: hashes of those five files.

The [native Cargo message stream](https://doc.rust-lang.org/cargo/reference/external-tools.html#json-messages)
comes from the existing build, without a second dependency resolver or extra build.
`json-render-diagnostics` keeps compiler warnings/errors visible on stderr while
artifact records are captured; warnings remain denied and failure stages no
candidate. A nonempty receipt is required. Fresh/cached artifacts remain marked;
the receipt is not proof that every unit was recompiled. Checksums cover its exact
bytes, not a parsed or executed interpretation.

This is private review input, not a complete binary SBOM or license clearance.
It can contain local paths and build-script environment values. Review/redaction,
actual license/notice files, system-library/SDK and external-backend/model
inventories still gate approved distribution. No raw receipt is automatically
published or uploaded; arbitrary compiler/macro output is not trusted executable
input. The Mesh license stays undecided.

Only these fixed files are copied; platform extended metadata is not serialized
into the archive. Source file attributes and OS security controls are unchanged.
Profiles, identities, credentials, runtime
state, caches and model files are not package inputs. Existing outputs and symlink
inputs are refused. A failed/interrupted attempt can leave an incomplete output
directory; it must not be treated as a candidate without successful verification.
The builder does not overwrite it on retry. Native tar timestamps are retained;
byte-identical archives across builds are not claimed.

The verifier requires Python3.9+ for development checks, the same clean source
checkout and its existing native release executable. It checks outer/inner hashes,
exact archive members and modes, size limits, source/compiler/lock identity and
the optimized non-test default CLI artifact in the Cargo receipt. Receipt paths
are compared with trusted checkout paths; no path from a receipt is opened.
Duplicate JSON keys, nonfinite JSON, fixture manifests and failed/incomplete
build receipts are refused. It decompresses at most128MiB in memory, reads at
most seven logical tar members and never extracts or executes a payload. The
executable is limited to96MiB and the receipt to16MiB; larger candidates need
reviewed limits. Checksums detect corruption; these unsigned, same-user build
facts do not authenticate a publisher or protect against a compromised build.
The verifier prints only a pass/refusal code, not private build metadata.
Minimum OS/libc/shared-library requirements, dependency notices and release
provenance still need actual artifact review before approved distribution.

`bash scripts/build-candidate.sh --self-test` checks a bounded synthetic archive,
hashes, tamper detection, executable permissions, excluded identity data and
existing/symlink/path refusal. `python3 -I scripts/verify-candidate.py --self-test`
also exercises receipt/identity, hash, mode, link, duplicate-member and JSON
refusals with synthetic bytes. It has no production fixture-acceptance option.
Neither self-test compiles or executes a candidate. The existing Unix CLI
integration check runs both in default CI; Python is not a product dependency.
These controls are not installation, real release-build, WSL2 or model evidence.

## Opt-in native Linux build qualification

The separate `Native candidate qualification` workflow builds and file-verifies
Linux GNU x64 and ARM64 candidates. It runs only on a deliberately pushed
`qualify-candidate/FULL_SOURCE_SHA` branch whose name exactly matches its commit.
First qualify that main commit through the ordinary source and automatic-main CI
gates, then obtain approval for the bounded extra CI batch before pushing the ref.
No repository setting/default-branch change or automatic PR/main release build
is required. An example ref push, **only after those gates and approval**, is:

```sh
git push origin QUALIFIED_FULL_MAIN_SHA:refs/heads/qualify-candidate/QUALIFIED_FULL_MAIN_SHA
```

Each of the two native hosted-runner jobs has a ten-minute cap and two Cargo jobs.
The approved batch installs the pinned compiler and fetches only the native
dependencies from the existing lockfile, then builds offline. It uses no cache
action: a push to a sibling branch cannot restore main's cache when main is not
the repository default, under GitHub's [cache scope rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache).
No repository setting changes are needed and no release cache is saved. The
existing eight ordinary CI checks and their caps are unchanged. Compiler output
and build receipts stay private on the ephemeral runner; logs contain only phase/refusal
outcomes. No artifact upload, executable invocation, installation, service/model
workload or publication runs. The candidates disappear with the runner; a green
batch establishes native build/archive verification only, not retained review
artifacts, deployment support, ABI/minimum-libc, WSL2 or inference acceptance.

The approved [Linux x64/ARM64 batch](evidence/GLY-198-native-candidate-verifier.md#actual-linux-native-builds--17-october-2026)
passed native build/archive verification at exact main
`2a71bbae5c12c6508069bad4e3877adc240e5e58`. It ran no candidate executable and
retained no downloadable artifacts. One earlier local Mac ARM64 candidate passed
build/archive/hash and static CLI checks; its
[historical evidence](evidence/GLY-198-candidate-packaging.md#actual-mac-candidate-check)
identifies different source and is not current-revision executable acceptance.
Neither record establishes installed desktop or inference qualification.

## Private source staging with Cargo

For an already configured development checkout, Cargo's native
[install](https://doc.rust-lang.org/cargo/commands/cargo-install.html) and
[uninstall](https://doc.rust-lang.org/cargo/commands/cargo-uninstall.html)
commands can manage a separate private prefix. This source workflow uses the
existing pinned compiler, native linker/SDK and cached locked dependencies.
It grants no distribution approval or installed-backend qualification.

Run from the workspace root. The subshell stops if the prefix already exists;
its variables and command-local PATH do not change the parent shell:

```sh
(
  set -eu
  umask 077
  STARLINGS_PREFIX="$PWD/target/private-prefix-check"
  mkdir -p target
  mkdir -m 700 "$STARLINGS_PREFIX"
  mkdir -m 700 "$STARLINGS_PREFIX/bin"
  PATH="$STARLINGS_PREFIX/bin:$PATH" CARGO_ENCODED_RUSTFLAGS=-Dwarnings \
    rustup run 1.97.1 cargo install --path crates/starlings-cli \
      --bin starlings --root "$STARLINGS_PREFIX" --target host-tuple \
      --target-dir "$PWD/target" --no-default-features --locked --offline --jobs 2
  rustup run 1.97.1 cargo install --list --root "$STARLINGS_PREFIX" --locked --offline
  "$STARLINGS_PREFIX/bin/starlings" --version
  "$STARLINGS_PREFIX/bin/starlings" doctor
  rustup run 1.97.1 cargo uninstall --root "$STARLINGS_PREFIX" starlings-cli --locked --offline
  rustup run 1.97.1 cargo install --list --root "$STARLINGS_PREFIX" --locked --offline
)
```

Only the local Compute binary is selected for installation; its cached build
dependencies are still required. No inference backend or model is installed.
Cargo's tracking stays enabled; no force
overwrite is requested. The private bin directory is on PATH only for the install
command, and the two CLI calls use its absolute path. `--version` prints the crate
version; `doctor` prints compiled protocol/OS/architecture and
`inference=unavailable`, without starting a service or probing a device.
After removal, the installed-package list is empty; Cargo metadata and prefix
directories may remain. Preserve review files instead of deleting an existing
profile or cache to repeat the check. A failed check can leave a partial prefix;
inspect it before deciding on cleanup, rather than forcing another installation.

The [bounded Mac check](evidence/GLY-198-private-prefix.md) verifies source staging,
static CLI output and tracked removal. It is not clean-machine installation,
version upgrade/rollback, service-manager lifecycle, WSL2 or native inference
acceptance. Those GLY-198 gates remain open; the license stays undecided.

## Inactive service descriptor export

For an existing initialized profile, export one descriptor into a **new private
directory** whose private parent already exists:

```sh
starlings --state-dir /absolute/private/profile service template /absolute/private/review-copy
```

This reads the bounded profile config and creates review files only. It never
loads identity keys, contacts a daemon, installs into OS service directories or
calls a service manager. It does not initialize a profile, enable/start a service
or approve a backend. The success line is
`SERVICE_TEMPLATE: installed=false started=false`. Existing output directories,
symlinks and unsafe/non-private paths are refused, without overwriting existing
files. A write failure can leave an incomplete new directory; there is no automatic
retry or replacement. The export directory is mode700 and its single file mode600.

The descriptor captures the current executable's canonical absolute path and the
selected profile's canonical absolute path. Its only arguments are
`--state-dir PROFILE daemon`: foreground control-plane operation using the saved
config/cache, without backend warm-up, routing approval or discovery flags. An
export from a development checkout still points at that development executable;
it does not install/copy a stable binary. Moving the binary/profile requires a new
export. Editing a config later changes what a future daemon reads; the descriptor
does not pin an immutable config or grant service execution. The stable filename
uses the enrolled node UUID; copying a profile does not create a new identity.
UTF-8 path strings are bounded to4096bytes and control characters are refused;
existing private-filesystem path restrictions remain in force.

Linux exports `com.getglyd.compute.NODE.service` for a **user** manager, with
`Type=exec`, `UMask=0077`, `NoNewPrivileges=yes`, `Restart=no` and no install target.
Arguments use the native quoting/escaping grammar, literal percent signs are
doubled and the colon prefix disables environment expansion. It uses no shell,
root user, capability grant, environment file or restart loop. These choices
follow the upstream [service command grammar](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml),
[quoting rules](https://github.com/systemd/systemd/blob/main/man/systemd.syntax.xml)
and [execution options](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml).

macOS exports `com.getglyd.compute.NODE.plist` as a per-user LaunchAgent
descriptor. The installed `/usr/bin/plutil` converts bounded JSON to XML; no
custom XML serializer or new dependency is added. `ProgramArguments` is an array,
`Umask` is decimal63 (octal077), and both `RunAtLoad` and `KeepAlive` are false.
No identity/credentials, environment values or automatic launch triggers are
embedded. Apple's [LaunchAgent guidance](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)
and [property-list reference](https://github.com/apple-oss-distributions/launchd/blob/main/man/launchd.plist.5)
describe the per-user boundary and these keys.

Exported files stay in the review directory. There is no installer, activation
command or supported background-service lifecycle in this delivery. Native
Windows export is refused; WSL2 can render the Linux format but systemd
availability and lifecycle are unqualified. The file-only check covers literal
paths, native plist round-trip, private modes and output refusal without launching
any daemon/service.

Linux CI additionally checks the actual generated unit with the native
`systemd-analyze --user --man=no --generators=no --recursive-errors=yes verify`
parser: valid input must be quiet/successful, and an unknown-directive control
must fail. This development check requires `/usr/bin/systemd-analyze` version250
or newer; it is not an application/runtime dependency. Its private runtime and
empty environment-generator search directory are scoped to the child process.
The [upstream verifier options](https://github.com/systemd/systemd/blob/v255/man/systemd-analyze.xml)
make syntax warnings fail; the [test-mode implementation](https://github.com/systemd/systemd/blob/v255/src/analyze/analyze-verify-util.c)
does not open an executor. Unit generators and man-page execution are disabled;
the [environment-generator path override](https://github.com/systemd/systemd/blob/v255/src/basic/path-lookup.c)
uses an empty private directory. It invokes no service manager operation and
installs or starts no service. This is syntax/executable/dependency validation,
not installed lifecycle or exact effective-argv qualification.
Login/logout, graceful stop/drain,
installed upgrade/rollback/uninstall and clean-machine qualification remain open.
Neither descriptor makes service-manager startup an inference-readiness signal.
