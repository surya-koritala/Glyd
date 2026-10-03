# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.27.0"
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
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.27.0/glyd-v0.27.0-macos-arm64.tar.gz"
      sha256 "293db8b7baf70df6abbe6dc5aecb6062320faf65ff038d3651d9eabb94246fd1"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.27.0.tar.gz"
      sha256 "31a55cb52087642064f714aa82a7fb3824ae5a061ddfb830239b70ed21e09eb6"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.27.0/glyd-v0.27.0-linux-x86_64.tar.gz"
      sha256 "ae25084c632b54bf4c54083556f5dea058ff43567a93a58bd62d3d15e61cf7d9"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.27.0/glyd-v0.27.0-linux-aarch64.tar.gz"
      sha256 "92de378d5c0cc8e03dd23990c143c7cb154eb82602339cd182b0cc235518858f"
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
