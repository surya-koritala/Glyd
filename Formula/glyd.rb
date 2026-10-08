# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.29.2"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.2/glyd-v0.29.2-macos-arm64.tar.gz"
      sha256 "f396ed4f9b0f45d4037d7f657fdeffd325cf7adf12c1e9f4ff3b1addefc615cb"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.29.2.tar.gz"
      sha256 "99f7e736ac3f203c0ea22fb71ddc6dc1304acf28028f6e23ad09ec5a475593ef"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.2/glyd-v0.29.2-linux-x86_64.tar.gz"
      sha256 "3b424f0f0b1db339c79b7dac304329154a06562fac3c80c2bda4e25b2a5336b5"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.2/glyd-v0.29.2-linux-aarch64.tar.gz"
      sha256 "9d485fdc40ad0426f081b8b77aa04ade7581a288cfcccdbb3dfa3b5025bd8e73"
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
