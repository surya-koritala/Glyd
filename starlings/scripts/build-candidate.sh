#!/usr/bin/env bash
set -euo pipefail
umask 077

checksum() {
    if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}

archive_candidate() (
    local input=$1 output=$2 name=$3 file stage
    [[ "$name" =~ ^starlings-[A-Za-z0-9.+_-]+$ ]] || exit 2
    for file in starlings BUILD.json Cargo.lock RUSTC.txt CARGO-BUILD.jsonl; do
        [[ -f "$input/$file" && ! -L "$input/$file" ]] || exit 2
    done
    mkdir -m 700 "$output" || exit 2 # Also refuse when invoked as an if-condition.
    output=$(cd "$output" && pwd)
    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    mkdir "$stage/$name"
    for file in starlings BUILD.json Cargo.lock RUSTC.txt CARGO-BUILD.jsonl; do
        cp "$input/$file" "$stage/$name/$file"
        chmod 644 "$stage/$name/$file"
    done
    chmod 755 "$stage/$name/starlings"
    (cd "$stage/$name" && checksum starlings BUILD.json Cargo.lock RUSTC.txt CARGO-BUILD.jsonl > SHA256SUMS)
    COPYFILE_DISABLE=1 tar --no-xattrs -czf "$output/$name.tar.gz" -C "$stage" "$name"
    (cd "$output" && checksum "$name.tar.gz" > SHA256SUMS)
)

if [[ ${1:-} == --self-test && $# == 1 ]]; then
    scratch=$(mktemp -d "${TMPDIR:-/tmp}/glyd candidate.XXXXXX")
    trap 'rm -rf "$scratch"' EXIT
    mkdir "$scratch/input" "$scratch/unpacked"
    printf 'synthetic bytes; never executed\n' > "$scratch/input/starlings"
    printf '{"fixture":true,"qualification":"not_assessed"}\n' > "$scratch/input/BUILD.json"
    printf 'synthetic lock\n' > "$scratch/input/Cargo.lock"
    printf 'synthetic compiler\n' > "$scratch/input/RUSTC.txt"
    printf '{"reason":"build-finished","success":true,"fixture":true}\n' > "$scratch/input/CARGO-BUILD.jsonl"
    printf 'must not be packaged\n' > "$scratch/input/identity.json"
    archive_candidate "$scratch/input" "$scratch/output" starlings-fixture
    (cd "$scratch/output" && checksum --check SHA256SUMS)
    actual=$(tar -tzf "$scratch/output/starlings-fixture.tar.gz" | sed 's:/$::' | LC_ALL=C sort)
    expected=$(printf '%s\n' starlings-fixture starlings-fixture/{BUILD.json,CARGO-BUILD.jsonl,Cargo.lock,RUSTC.txt,SHA256SUMS,starlings} | LC_ALL=C sort)
    [[ "$actual" == "$expected" ]]
    tar -xzf "$scratch/output/starlings-fixture.tar.gz" -C "$scratch/unpacked"
    (cd "$scratch/unpacked/starlings-fixture" && checksum --check SHA256SUMS)
    [[ ! -e "$scratch/unpacked/starlings-fixture/identity.json" ]]
    [[ -x "$scratch/unpacked/starlings-fixture/starlings" ]]
    cmp "$scratch/input/starlings" "$scratch/unpacked/starlings-fixture/starlings"
    printf 'tampered\n' >> "$scratch/unpacked/starlings-fixture/starlings"
    if (cd "$scratch/unpacked/starlings-fixture" && checksum --check SHA256SUMS) >/dev/null 2>&1; then exit 1; fi
    cp "$scratch/input/starlings" "$scratch/unpacked/starlings-fixture/starlings"
    printf 'changed receipt\n' >> "$scratch/unpacked/starlings-fixture/CARGO-BUILD.jsonl"
    if (cd "$scratch/unpacked/starlings-fixture" && checksum --check SHA256SUMS) >/dev/null 2>&1; then exit 1; fi
    printf 'keep\n' > "$scratch/output/keep"
    if archive_candidate "$scratch/input" "$scratch/output" starlings-fixture >/dev/null 2>&1; then exit 1; fi
    [[ $(cat "$scratch/output/keep") == keep ]]
    ln -s output "$scratch/linked-output"
    if archive_candidate "$scratch/input" "$scratch/linked-output" starlings-fixture >/dev/null 2>&1; then exit 1; fi
    if archive_candidate "$scratch/input" "$scratch/bad" ../invalid >/dev/null 2>&1; then exit 1; fi
    [[ ! -e "$scratch/bad" ]]
    rm "$scratch/input/starlings"
    ln -s BUILD.json "$scratch/input/starlings"
    if archive_candidate "$scratch/input" "$scratch/symlink" starlings-fixture >/dev/null 2>&1; then exit 1; fi
    [[ ! -e "$scratch/symlink" ]]
    printf 'candidate archive checks passed\n'
    exit
fi

if [[ $# != 1 || ${1:-} == -* ]]; then
    printf 'usage: bash scripts/build-candidate.sh NEW_OUTPUT_DIRECTORY | --self-test\n' >&2
    exit 2
fi
output=$1
[[ "$output" == /* ]] || output="$PWD/$output"
[[ ! -e "$output" && ! -L "$output" ]] || { printf 'NEW_OUTPUT_REQUIRED\n' >&2; exit 2; }
[[ -d $(dirname "$output") ]] || { printf 'OUTPUT_PARENT_REQUIRED\n' >&2; exit 2; }
# (the repository root: Starlings' crates are members of its workspace, with the one Cargo.lock and target directory)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
[[ -z $(git status --porcelain) ]] || { printf 'CLEAN_SOURCE_REQUIRED\n' >&2; exit 2; }
revision=$(git rev-parse --verify HEAD)
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 2
package=$(rustup run 1.97.1 cargo pkgid -p starlings-cli --locked --offline)
version=${package##*#}
version=${version##*@}
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+[-A-Za-z0-9.+]*$ ]] || exit 2
compiler=$(mktemp -d)
trap 'rm -rf "$compiler"' EXIT
rustup run 1.97.1 rustc -vV > "$compiler/RUSTC.txt"
target=$(sed -n 's/^host: //p' "$compiler/RUSTC.txt")
case "$target" in
    x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu|x86_64-apple-darwin|aarch64-apple-darwin) ;;
    *) printf 'UNSUPPORTED_CANDIDATE_TARGET\n' >&2; exit 2 ;;
esac
CARGO_ENCODED_RUSTFLAGS=-Dwarnings rustup run 1.97.1 cargo build -p starlings-cli --bin starlings --release --no-default-features --locked --offline --jobs 2 --target "$target" --target-dir "$root/target" --message-format=json-render-diagnostics > "$compiler/CARGO-BUILD.jsonl"
[[ -s "$compiler/CARGO-BUILD.jsonl" ]] || { printf 'BUILD_RECEIPT_REQUIRED\n' >&2; exit 2; }
[[ -z $(git status --porcelain) && $(git rev-parse HEAD) == "$revision" ]] || { printf 'SOURCE_CHANGED_DURING_BUILD\n' >&2; exit 2; }
cp "$root/target/$target/release/starlings" "$compiler/starlings"
cp Cargo.lock "$compiler/Cargo.lock"
printf '{"version":"%s","target":"%s","source_revision":"%s","toolchain":"1.97.1","features":[],"rustflags":["-Dwarnings"],"fixture":false,"qualification":"not_assessed","license":"undecided"}\n' "$version" "$target" "$revision" > "$compiler/BUILD.json"
archive_candidate "$compiler" "$output" "starlings-$version-$target"
printf 'Development candidate staged; qualification, license and publication approval remain required.\n'
