# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.25.1"
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
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.25.1/glyd-v0.25.1-macos-arm64.tar.gz"
      sha256 "b4566a2bcd362375ba3886baeaf375df2796cf113ad81fdb2bd69f674f56e399"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.25.1.tar.gz"
      sha256 "7b1661b8fe4d0ee2914c21d3fe2629ae83f727bd1e09bbe3b56b308b9bcf33d2"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.25.1/glyd-v0.25.1-linux-x86_64.tar.gz"
      sha256 "514735940d91dc608f3b746c49f10b85a5d916c66457c2d68ec71dea7ad8383a"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.25.1/glyd-v0.25.1-linux-aarch64.tar.gz"
      sha256 "13806de69b8fb87a5a2af9707d4d5c03c8b320084b85ffb736f24520f1f680ec"
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
