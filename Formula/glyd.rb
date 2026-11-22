# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.34.2"
  # Apache-2.0 from the release after v0.32.1 (v0.29.0 to v0.32.1: BUSL-1.1; v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "Apache-2.0"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.34.2/glyd-v0.34.2-macos-arm64.tar.gz"
      sha256 "b7116e9e44f7bc679e1f5fd28a4975b9af715a8a6568b4eadc1f92ef6f00f62e"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.34.2.tar.gz"
      sha256 "7835a1bb2f4e5b444fb760c5cfd365b87d35ca42bcd16699f9500c685bb307c9"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.34.2/glyd-v0.34.2-linux-x86_64.tar.gz"
      sha256 "ed019ddcb2c33daeb95c710e4d0d44e77426a89ae1a348208518e00afffe340b"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.34.2/glyd-v0.34.2-linux-aarch64.tar.gz"
      sha256 "40c342701f3f9f4f118a19031c7f58dd383250de5cb8db8465562e65d8d09037"
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
