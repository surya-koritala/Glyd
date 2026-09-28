# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.22.0"
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
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.22.0/glyd-v0.22.0-macos-arm64.tar.gz"
      sha256 "b17a97264ce8e82ead35c48a94017af28c10b0a1920b70c4d44ab366864b7531"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.22.0.tar.gz"
      sha256 "4cb29315d856a50e5e407c93549fdc999deeb2e8d66468632282d1ccb1c2e943"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.22.0/glyd-v0.22.0-linux-x86_64.tar.gz"
      sha256 "e4f15d69f308eb784a94ec26a093ee94bd1d74237bd2a96ac332062ba04925ea"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.22.0/glyd-v0.22.0-linux-aarch64.tar.gz"
      sha256 "c59420e2745d5c0b266373fd6a965ada1e32f21c6b9424802503c6fd1e2c47f2"
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
