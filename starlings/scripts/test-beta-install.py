#!/usr/bin/env python3
"""Exercise the beta installer with synthetic files, never a model or GPU."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    scripts = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="starlings-installer-") as tmp:
        root = Path(tmp).resolve()
        bundle, mock = root / "bundle", root / "mock"
        bundle.mkdir()
        mock.mkdir()
        shutil.copy2(scripts / "install.sh", bundle / "install.sh")
        (bundle / "glyd-engine").write_text("synthetic engine; never executed\n")
        (bundle / "libglyd_gpu_cuda13.so").write_bytes(b"synthetic library")
        (bundle / "LICENSE-glyd-engine").write_text("synthetic notice\n")
        (mock / "nvidia-smi").write_text("#!/bin/sh\nexit 99\n")
        (mock / "nvidia-smi").chmod(0o755)
        lib, bin_dir = root / "prefix/lib", root / "prefix/bin"
        state = root / "prefix/state/keep"
        state.parent.mkdir(parents=True)
        state.write_text("retained")
        env = dict(os.environ, STARLINGS_LIB=str(lib), STARLINGS_BIN=str(bin_dir),
                   PATH=str(mock) + os.pathsep + os.environ["PATH"])

        def binary(version):
            (bundle / "starlings").write_text(f"#!/bin/sh\nprintf 'starlings {version}\\n'\n")
            (bundle / "starlings").chmod(0o755)

        def sums():
            names = ("starlings", "glyd-engine", "libglyd_gpu_cuda13.so", "LICENSE-glyd-engine", "install.sh")
            (bundle / "SHA256SUMS").write_text("".join(
                hashlib.sha256((bundle / name).read_bytes()).hexdigest() + "  " + name + "\n"
                for name in names))

        def install(*args, script=None, success=True, overrides=None):
            result = subprocess.run(["bash", str(script or bundle / "install.sh"), *args],
                                    env=env | (overrides or {}), capture_output=True, text=True, timeout=10)
            assert (result.returncode == 0) == success, result

        binary("0.1.0")
        sums()
        (mock / "ln").write_text("#!/bin/sh\nexit 1\n")
        (mock / "ln").chmod(0o755)
        install(success=False)
        assert not lib.exists()
        (mock / "ln").unlink()
        bin_dir.mkdir(exist_ok=True)
        (bin_dir / "starlings").symlink_to(root / "missing-foreign-target")
        install(success=False)
        assert (bin_dir / "starlings").readlink() == root / "missing-foreign-target"
        (bin_dir / "starlings").unlink()
        install()
        assert (lib / "LICENSE-glyd-engine").read_text() == "synthetic notice\n"
        assert (lib / "install.sh").is_file() and (lib / "SHA256SUMS").is_file()
        assert (bin_dir / "starlings").resolve() == lib / "starlings"
        binary("0.2.0")
        sums()
        install()
        previous = (lib / "starlings").read_bytes()
        assert b"0.2.0" in previous
        (bundle / "starlings").write_text("tampered")
        install(success=False)
        assert (lib / "starlings").read_bytes() == previous
        (bundle / "starlings").write_text("#!/bin/sh\nexit 1\n")
        sums()
        install(success=False)
        assert (lib / "starlings").read_bytes() == previous

        # Failure after activation also restores the old installation.
        binary("0.3.0")
        sums()
        (mock / "ln").write_text("#!/bin/sh\nexit 1\n")
        (mock / "ln").chmod(0o755)
        install(success=False)
        assert (lib / "starlings").read_bytes() == previous
        (mock / "ln").unlink()
        assert not list(lib.parent.glob("lib.new.*"))

        foreign = root / "foreign"
        foreign.mkdir()
        (foreign / "keep").write_text("untouched")
        install(success=False, overrides={"STARLINGS_LIB": str(foreign)})
        assert (foreign / "keep").read_text() == "untouched"
        install("--unknown", success=False)
        install("--uninstall", script=lib / "install.sh")
        assert not lib.exists() and not (bin_dir / "starlings").is_symlink()
        assert state.read_text() == "retained"

        # Packaging uses GNU tar and targets Linux; other Unix systems check installation.
        if sys.platform.startswith("linux"):
            engine = root / "engine"
            engine.mkdir()
            for name in ("glyd-engine", "libglyd_gpu_cuda13.so"):
                shutil.copy2(bundle / name, engine / name)
            shutil.copy2(bundle / "LICENSE-glyd-engine", engine / "LICENSE")
            output = root / "output"
            result = subprocess.run(["bash", str(scripts / "package-beta.sh"),
                                     str(bundle / "starlings"), str(engine), str(output)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result
            archive = next(output.glob("*.tar.gz"))
            checksum = archive.with_name(archive.name + ".sha256").read_text().split()[0]
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
            with tarfile.open(archive) as packed:
                names = {Path(name).name for name in packed.getnames()}
                assert {"starlings", "glyd-engine", "libglyd_gpu_cuda13.so",
                        "LICENSE-glyd-engine", "SHA256SUMS", "install.sh"} <= names
    print("beta installer checks passed")


if __name__ == "__main__":
    main()
