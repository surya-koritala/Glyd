#!/usr/bin/env python3
"""Check private native candidates without extracting or executing their contents."""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zlib


LIMIT = 128 * 1024 * 1024
FILES = {
    "starlings": (0o755, 96 * 1024 * 1024),
    "BUILD.json": (0o644, 16384),
    "Cargo.lock": (0o644, 1024 * 1024),
    "RUSTC.txt": (0o644, 65536),
    "CARGO-BUILD.jsonl": (0o644, 16 * 1024 * 1024),
    "SHA256SUMS": (0o600, 4096),
}
TARGETS = {
    "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin", "aarch64-apple-darwin",
}


class Refused(Exception):
    pass


def require(condition, code):
    if not condition:
        raise Refused(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded_file(path, limit):
    require(stat.S_ISREG(path.lstat().st_mode), "REGULAR_FILE_REQUIRED")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit, "FILE_SIZE_LIMIT")
    return data


def unique_object(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "DUPLICATE_JSON_KEY")
    return result


def invalid_constant(_):
    raise Refused("NONFINITE_JSON")


def parse_json(data):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def manifest(context):
    return {
        "version": context["version"], "target": context["target"],
        "source_revision": context["revision"], "toolchain": "1.97.1",
        "features": [], "rustflags": ["-Dwarnings"], "fixture": False,
        "qualification": "not_assessed", "license": "undecided",
    }


def command(root, *args):
    return subprocess.run(args, cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=15).stdout


def context_for(root):
    require(not command(root, "git", "status", "--porcelain"), "CLEAN_SOURCE_REQUIRED")
    revision = command(root, "git", "rev-parse", "--verify", "HEAD").decode().strip()
    require(re.fullmatch("[0-9a-f]{40}", revision), "SOURCE_REVISION_REQUIRED")
    package = command(root, "rustup", "run", "1.97.1", "cargo", "pkgid", "-p",
                      "starlings-cli", "--locked", "--offline").decode().strip()
    version = package.rsplit("#", 1)[-1].rsplit("@", 1)[-1]
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+[-A-Za-z0-9.+]*", version), "VERSION_REQUIRED")
    compiler = command(root, "rustup", "run", "1.97.1", "rustc", "-vV")
    require(compiler.startswith(b"rustc 1.97.1 "), "PINNED_COMPILER_REQUIRED")
    hosts = [line[6:] for line in compiler.decode().splitlines() if line.startswith("host: ")]
    require(len(hosts) == 1 and hosts[0] in TARGETS, "NATIVE_TARGET_REQUIRED")
    target = hosts[0]
    return {"root": root, "version": version, "revision": revision, "package": package,
            "compiler": compiler, "target": target,
            "lock": bounded_file(root / "Cargo.lock", FILES["Cargo.lock"][1]),
            "executable": root / "target" / target / "release/starlings"}


def verify(output, context):
    info = output.lstat()
    require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
            and info.st_uid == os.getuid(), "PRIVATE_DIRECTORY_REQUIRED")
    base = f'starlings-{context["version"]}-{context["target"]}'
    archive_name = base + ".tar.gz"
    require({p.name for p in output.iterdir()} == {archive_name, "SHA256SUMS"}, "OUTPUT_MEMBERS")
    compressed = bounded_file(output / archive_name, LIMIT)
    require(bounded_file(output / "SHA256SUMS", 4096)
            == f"{sha(compressed)}  {archive_name}\n".encode(), "OUTER_CHECKSUM")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(LIMIT + 1)
    except zlib.error:
        raise Refused("ARCHIVE_COMPRESSION") from None
    require(len(raw) <= LIMIT, "ARCHIVE_SIZE_LIMIT")
    payload = {}
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            require(len(seen) < 7 and member.name not in seen, "ARCHIVE_MEMBERS")
            seen.add(member.name)
            if member.name == base:
                require(member.isdir() and member.mode == 0o700 and member.size == 0, "ARCHIVE_DIRECTORY")
                continue
            name = member.name.removeprefix(base + "/")
            require(member.name == base + "/" + name and name in FILES, "ARCHIVE_MEMBERS")
            mode, limit = FILES[name]
            require(member.isfile() and not member.issparse() and member.mode == mode
                    and 0 < member.size <= limit, "ARCHIVE_FILE")
            payload[name] = archive.extractfile(member).read(limit + 1)
            require(len(payload[name]) == member.size, "ARCHIVE_FILE_SIZE")
    require(seen == {base} | {base + "/" + name for name in FILES}, "ARCHIVE_MEMBERS")
    expected_sums = "".join(f"{sha(payload[name])}  {name}\n" for name in FILES if name != "SHA256SUMS")
    require(payload["SHA256SUMS"] == expected_sums.encode(), "INNER_CHECKSUM")
    # Serialized comparison distinguishes false from 0 and [] from other JSON types.
    require(json.dumps(parse_json(payload["BUILD.json"]), sort_keys=True)
            == json.dumps(manifest(context), sort_keys=True), "BUILD_IDENTITY")
    require(payload["Cargo.lock"] == context["lock"], "LOCK_IDENTITY")
    require(payload["RUSTC.txt"] == context["compiler"], "COMPILER_IDENTITY")
    lines = payload["CARGO-BUILD.jsonl"].splitlines()
    require(0 < len(lines) <= 8192 and all(0 < len(line) <= 1024 * 1024 for line in lines), "RECEIPT_LIMIT")
    records = [parse_json(line) for line in lines]
    require(all(isinstance(record, dict) for record in records), "RECEIPT_RECORD")
    finished = [record for record in records if record.get("reason") == "build-finished"]
    require(finished == [records[-1]] and records[-1].get("success") is True, "BUILD_NOT_FINISHED")
    artifacts = [record for record in records if record.get("reason") == "compiler-artifact"
                 and record.get("target", {}).get("name") == "starlings"]
    require(len(artifacts) == 1, "CLI_ARTIFACT_REQUIRED")
    artifact = artifacts[0]
    root = context["root"]
    require(artifact.get("package_id") == context["package"]
            and artifact.get("manifest_path") == str(root / "crates/starlings-cli/Cargo.toml")
            and artifact.get("target", {}).get("kind") == ["bin"]
            and artifact["target"].get("src_path") == str(root / "crates/starlings-cli/src/main.rs")
            and artifact.get("executable") == str(context["executable"])
            and artifact.get("features") == []
            and artifact.get("profile", {}).get("opt_level") == "3"
            and artifact["profile"].get("test") is False, "CLI_BUILD_IDENTITY")
    # Never follow a path supplied by a receipt. This path comes from the checkout.
    require(payload["starlings"] == bounded_file(context["executable"], FILES["starlings"][1]),
            "EXECUTABLE_IDENTITY")
    magic = b"\x7fELF" if context["target"].endswith("linux-gnu") else b"\xcf\xfa\xed\xfe"
    require(payload["starlings"].startswith(magic), "NATIVE_FILE_REQUIRED")


def self_test():
    with tempfile.TemporaryDirectory(prefix="glyd candidate verifier ") as temporary:
        root = Path(temporary)
        executable = root / "target/x86_64-unknown-linux-gnu/release/starlings"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"\x7fELFsynthetic bytes; never executed\n")
        context = {"root": root, "version": "0.1.0", "target": "x86_64-unknown-linux-gnu",
                   "revision": "a" * 40, "package": "synthetic-package",
                   "compiler": b"synthetic compiler\n", "lock": b"synthetic lock\n",
                   "executable": executable}
        artifact = {"reason": "compiler-artifact", "package_id": context["package"],
                    "manifest_path": str(root / "crates/starlings-cli/Cargo.toml"),
                    "target": {"name": "starlings", "kind": ["bin"],
                               "src_path": str(root / "crates/starlings-cli/src/main.rs")},
                    "executable": str(executable), "features": [],
                    "profile": {"opt_level": "3", "test": False}}
        original = {"starlings": executable.read_bytes(),
                    "BUILD.json": json.dumps(manifest(context)).encode(),
                    "Cargo.lock": context["lock"], "RUSTC.txt": context["compiler"],
                    "CARGO-BUILD.jsonl": (json.dumps(artifact) + '\n'
                                         + '{"reason":"build-finished","success":true}\n').encode()}
        output = root / "output"
        output.mkdir(mode=0o700)
        base = "starlings-0.1.0-x86_64-unknown-linux-gnu"
        archive_path = output / (base + ".tar.gz")

        def write(payload, change_member=None, corrupt_sums=False):
            payload = dict(payload)
            payload["SHA256SUMS"] = "".join(f"{sha(payload[name])}  {name}\n"
                                          for name in FILES if name != "SHA256SUMS").encode()
            if corrupt_sums:
                payload["SHA256SUMS"] = b"wrong inner checksum\n"
            with tarfile.open(archive_path, "w:gz") as archive:
                directory = tarfile.TarInfo(base)
                directory.type, directory.mode = tarfile.DIRTYPE, 0o700
                archive.addfile(directory)
                for name, data in payload.items():
                    member = tarfile.TarInfo(base + "/" + name)
                    member.mode, member.size = FILES[name][0], len(data)
                    if change_member:
                        change_member(name, member)
                    archive.addfile(member, io.BytesIO(data) if member.isfile() else None)
            (output / "SHA256SUMS").write_text(f"{sha(archive_path.read_bytes())}  {archive_path.name}\n")

        def refused():
            try:
                verify(output, context)
            except Refused:
                return
            raise AssertionError("corrupt candidate accepted")

        write(original)
        verify(output, context)
        for name, data in [
            ("BUILD.json", json.dumps({**manifest(context), "fixture": True}).encode()),
            ("BUILD.json", json.dumps({**manifest(context), "fixture": 0}).encode()),
            ("BUILD.json", json.dumps({**manifest(context), "source_revision": "b" * 40}).encode()),
            ("BUILD.json", original["BUILD.json"][:-1] + b',"fixture":false}'),
            ("Cargo.lock", b"wrong lock"), ("RUSTC.txt", b"wrong compiler"),
            ("starlings", b"\x7fELFchanged executable"),
            ("CARGO-BUILD.jsonl", original["CARGO-BUILD.jsonl"].replace(b'"success":true', b'"success":false')),
            ("CARGO-BUILD.jsonl", original["CARGO-BUILD.jsonl"].replace(b'"features": []', b'"features": ["test-backend"]')),
            ("CARGO-BUILD.jsonl", original["CARGO-BUILD.jsonl"].replace(b'"test": false', b'"test": true')),
            ("CARGO-BUILD.jsonl", original["CARGO-BUILD.jsonl"].replace(str(executable).encode(), b"/never/read/receipt/path")),
            ("CARGO-BUILD.jsonl", original["CARGO-BUILD.jsonl"] + b'NaN\n'),
        ]:
            write({**original, name: data})
            refused()
        for attribute, value in [("name", base + "/identity.json"), ("name", "../outside"),
                                 ("name", base + "/Cargo.lock"), ("mode", 0o777),
                                 ("type", tarfile.SYMTYPE), ("type", tarfile.LNKTYPE)]:
            write(original, lambda name, member: setattr(member, attribute, value)
                  if name == "starlings" else None)
            refused()
        write(original)
        verify(output, context)
        write(original, corrupt_sums=True)
        refused()
        write(original)
        # Valid gzip header, reserved deflate block type; outer integrity still matches.
        archive_path.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x07")
        (output / "SHA256SUMS").write_text(f"{sha(archive_path.read_bytes())}  {archive_path.name}\n")
        refused()
        write(original)
        (output / "SHA256SUMS").write_text("wrong checksum\n")
        refused()
        write(original)
        (output / "identity.json").write_text("must be refused")
        refused()
    print("candidate verifier checks passed")


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    require(len(sys.argv) == 2 and not sys.argv[1].startswith("-"), "CANDIDATE_DIRECTORY_REQUIRED")
    root = Path(__file__).resolve().parent.parent
    context = context_for(root)
    verify(Path(sys.argv[1]), context)
    require(not command(root, "git", "status", "--porcelain")
            and command(root, "git", "rev-parse", "HEAD").decode().strip() == context["revision"],
            "SOURCE_CHANGED_DURING_VERIFICATION")
    print("Candidate archive and native build receipt verified; qualification remains not_assessed.")


if __name__ == "__main__":
    try:
        main()
    except Refused as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, EOFError, RecursionError, KeyError, TypeError, AttributeError,
            tarfile.TarError, subprocess.SubprocessError):
        print("CANDIDATE_VERIFICATION_FAILED", file=sys.stderr)
        sys.exit(2)
