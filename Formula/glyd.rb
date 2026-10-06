# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.29.1"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.1/glyd-v0.29.1-macos-arm64.tar.gz"
      sha256 "39c8a728a0e3c83fa220e6817eeca8f3ce013791cd6ce5c3056a8df22393adfe"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.29.1.tar.gz"
      sha256 "4fd8c96e84418a5a2eea6ac8867d11b0dbf9fb73e1fddd5f84324d00106e2372"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.1/glyd-v0.29.1-linux-x86_64.tar.gz"
      sha256 "bf25e3f921e44387fbff6e18267a4c347fd1f4f7f4e1c7d3726f2dbfc4b074ec"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.1/glyd-v0.29.1-linux-aarch64.tar.gz"
      sha256 "3aa806d88b78db4d8fee3512b292b0b634553ce041ff7cf8c81d47053cbf1a75"
    end
  end

  def install
    if File.exist?("Cargo.toml")
      system "cargo", "install", *std_cargo_args
      system "cargo", "install", *std_cargo_args(path: "glyd-store")
      include.install "include/glyd.h"
    else
      bin.install "glyd", "glyd-store"
      include.install "glyd.h"
    end
  end

  test do
    (testpath/"a.txt").write("hello hello hello hello glyd\n" * 100)
    system bin/"glyd", "--max", testpath/"a.txt", "-o", testpath/"a.glyd"
    system bin/"glyd", "-d", testpath/"a.glyd", "-o", testpath/"b.txt"
    assert_equal (testpath/"a.txt").read, (testpath/"b.txt").read
  end
end
