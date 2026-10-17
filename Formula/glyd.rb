# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.32.1"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.32.1/glyd-v0.32.1-macos-arm64.tar.gz"
      sha256 "ab9d99cf36c5cf2d4f2ca300e23368899f341301f584fbb6613a163a5daa6924"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.32.1.tar.gz"
      sha256 "7d29191eb55872ea8a2102c23614c9492810a3bc749a99194088f75b0eafb075"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.32.1/glyd-v0.32.1-linux-x86_64.tar.gz"
      sha256 "46fa77fc5a1f49b85c84cb72531b5428b1d3c97fd76894c4fb50fc8f89b2d9b4"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.32.1/glyd-v0.32.1-linux-aarch64.tar.gz"
      sha256 "a237f75e5fcb54fbf14e30f37a9a2147a74afaf5fd8c430acf9a3334355a3564"
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
