# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.31.1"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.31.1/glyd-v0.31.1-macos-arm64.tar.gz"
      sha256 "19f26e50d68ff9bb4f74278e2d495e3df9df2c7a1a9f3a881f06d96ff0fa0368"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.31.1.tar.gz"
      sha256 "46b4b34b9a41d9eaddc0f35e3e4177f9d468b5df2681ca92749520c8707cf878"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.31.1/glyd-v0.31.1-linux-x86_64.tar.gz"
      sha256 "f8cac806444e2e900635296f02c6a15fc37cc6b4c255c9ce1b09a1541c7f2023"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.31.1/glyd-v0.31.1-linux-aarch64.tar.gz"
      sha256 "a2b74e1d72046dbea429676ce8b19b8430293e3fa073fb0b0ca8832f1911e2cc"
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
