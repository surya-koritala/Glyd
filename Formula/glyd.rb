# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.29.4"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.4/glyd-v0.29.4-macos-arm64.tar.gz"
      sha256 "1721d635d6bd521ee9e5f6d10c97c9a536673c0d534aafc15e281a2a291ecd68"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.29.4.tar.gz"
      sha256 "0d8d361f4484b60c505bdfb911e5b01beaad4f3e0526715740ea4e7bb24ee29f"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.4/glyd-v0.29.4-linux-x86_64.tar.gz"
      sha256 "c0ae8092adbb679fd49fe34b408a23865ee89c0576d641a80e879e664ebddc7d"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.29.4/glyd-v0.29.4-linux-aarch64.tar.gz"
      sha256 "a44fe20e57ba15251760a37ad60c2f0064f46568bbcc820ece9685fce36e1efd"
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
