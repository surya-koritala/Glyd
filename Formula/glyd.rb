# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.31.0"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.31.0/glyd-v0.31.0-macos-arm64.tar.gz"
      sha256 "6dec17e80165dde58ec0fdd3ef2ee9bce84d47d48fa7eb45c4598aa5a7a4446e"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.31.0.tar.gz"
      sha256 "f3b4b6ed18bbffb5bdf2d85c10917ac9daf975bbf763bc1a6312cced8f0b53e5"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.31.0/glyd-v0.31.0-linux-x86_64.tar.gz"
      sha256 "280e31b463163d431b2db33e50b80f4e2161b4fa088ff66835bec9ffbe107fb5"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.31.0/glyd-v0.31.0-linux-aarch64.tar.gz"
      sha256 "c5bd8ea24ae51746e89ce52a4016ff0a7b134a9651a7ee21c5f877bf8c3a766d"
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
