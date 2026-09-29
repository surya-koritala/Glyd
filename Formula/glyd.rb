# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.24.0"
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
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.24.0/glyd-v0.24.0-macos-arm64.tar.gz"
      sha256 "4c75f63a293a7dcaef524c5eee09c31982405c90ad1d911949575a9e85ce1ec6"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.24.0.tar.gz"
      sha256 "53439c8a6e2d5e0964bb417a8b652958c3f44285aef44579410d3d122e97ddb1"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.24.0/glyd-v0.24.0-linux-x86_64.tar.gz"
      sha256 "f82e2e988d6b568868d1c4c74e159a9832bc1b377c62804a06289587ef47edd5"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.24.0/glyd-v0.24.0-linux-aarch64.tar.gz"
      sha256 "a26d9c64522ea2f667edc6164913f647f5f6253640a4b6d7fb86aa5e4e7e5c81"
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
