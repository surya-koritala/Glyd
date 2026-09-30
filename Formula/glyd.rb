# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.25.0"
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
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.25.0/glyd-v0.25.0-macos-arm64.tar.gz"
      sha256 "611e74ff076b624a964e612f5c8a5fd4eb687a3be094da30bf2a2fb62712d6ad"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.25.0.tar.gz"
      sha256 "a8cb165eb10fe2937cb1c4243193a3f496c707388b95920fe15e55073ba47d22"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.25.0/glyd-v0.25.0-linux-x86_64.tar.gz"
      sha256 "ac91d0f4c933e95d64b582a6cd6cd2ad208b73a68c9c670a3786e9187cd7a48f"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.25.0/glyd-v0.25.0-linux-aarch64.tar.gz"
      sha256 "2d43d284b74ad2b32eb9454642fd4ee60e05f8500e22c2623173ef91a2c2fc5d"
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
