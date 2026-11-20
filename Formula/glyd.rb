# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.33.0"
  # Apache-2.0 from the release after v0.32.1 (v0.29.0 to v0.32.1: BUSL-1.1; v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "Apache-2.0"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.33.0/glyd-v0.33.0-macos-arm64.tar.gz"
      sha256 "a78efc6452d57924b1e9c8d008817a145c7da6ea0cef44131abdae1b76e5cc61"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.33.0.tar.gz"
      sha256 "84cfe1b5479be6fa675866ab2d348a33e337561a7edec88e939430482c8fdd7c"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.33.0/glyd-v0.33.0-linux-x86_64.tar.gz"
      sha256 "9355e62ac07c45248e8739c02b7f8e9438fc86fb122491de7911b00cc496dbb4"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.33.0/glyd-v0.33.0-linux-aarch64.tar.gz"
      sha256 "c6e06e12f84bed1f18042eef14e53febc539257cc6342f18bff3b389448714ff"
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
