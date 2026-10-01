# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.26.0"
  # The codec and CLI: BSD-3-Clause or GPL-2.0; the glyd-store binary: BUSL-1.1.
  license all_of: [
    { any_of: ["BSD-3-Clause", "GPL-2.0-only"] },
    "BUSL-1.1",
  ]

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.26.0/glyd-v0.26.0-macos-arm64.tar.gz"
      sha256 "80df429eea41859f0a2faf21eea7dc3140f2985230527c3c9d406febc03fb37f"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.26.0.tar.gz"
      sha256 "ed6592c4ef3d324c8b1aec476a1eedac3df383e8713f9953837bc02d9b0695f0"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.26.0/glyd-v0.26.0-linux-x86_64.tar.gz"
      sha256 "160b81b88789ede86553925dc361e4211e68822c2aef3906c9566be7ea5c0bd5"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.26.0/glyd-v0.26.0-linux-aarch64.tar.gz"
      sha256 "7d5d9e26cc1f4b6f38dc64c0c4371e307c63de1dd4fdccbc93d55ba1f0128b8b"
    end
  end

  def install
    if File.exist?("Cargo.toml")
      system "cargo", "install", *std_cargo_args
      system "cargo", "install", *std_cargo_args(path: "glyd-store")
      system "cargo", "install", *std_cargo_args(path: "glyd-gpu")
      include.install "include/glyd.h"
    else
      bin.install "glyd", "glyd-store", "glyd-gpu"
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
