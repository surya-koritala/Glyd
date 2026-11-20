# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.33.1"
  # Apache-2.0 from the release after v0.32.1 (v0.29.0 to v0.32.1: BUSL-1.1; v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "Apache-2.0"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.33.1/glyd-v0.33.1-macos-arm64.tar.gz"
      sha256 "4c2cf4ba6f6f21a58fbe0c24984b5c914269a26c202f967152424bcdeeabe2e8"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.33.1.tar.gz"
      sha256 "e6ee92c86e850909cf3d8fafbba5450f08d460526ae1dd84b45034ab0d345319"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.33.1/glyd-v0.33.1-linux-x86_64.tar.gz"
      sha256 "890c3e309cf58672bc03dc1692d7c364af3a7f6a40ef32e93dee6b4c8e39221b"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.33.1/glyd-v0.33.1-linux-aarch64.tar.gz"
      sha256 "714b7224d39e5ef6b315ff46731d8438cb63c3f09a9ba7a34ea382cc5f4036b2"
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
