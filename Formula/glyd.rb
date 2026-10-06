# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file).
# Apple silicon and Linux (x86_64, arm64) install the release's prebuilt
# binaries in seconds; an Intel Mac and --HEAD build from source with cargo.
# Each release: python3 scripts/bump_formula.py X.Y.Z, then copy this file to the tap.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://getglyd.com"
  version "0.28.0"
  # BUSL-1.1 from v0.29.0, the glyd and glyd-store programs both (v0.28.0 and earlier: glyd BSD-3-Clause or GPL-2.0, glyd-store BUSL-1.1).
  license "BUSL-1.1"

  head do
    url "https://github.com/surya-koritala/Glyd.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.28.0/glyd-v0.28.0-macos-arm64.tar.gz"
      sha256 "81057758708c07259680c0f2ad5bd39ed1548e65b6185f31ff4420db58f83128"
    end
    on_intel do
      url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.28.0.tar.gz"
      sha256 "d47cc6185cc00bb2ff3638ba41eda9690893c0da7d8b26c98cd357fa2ffa6e6a"
      depends_on "rust" => :build
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.28.0/glyd-v0.28.0-linux-x86_64.tar.gz"
      sha256 "2dcb0ea6943ffa009666ed9a3c030aa83ef20fb3761656a59f2c58516c37044c"
    end
    on_arm do
      url "https://github.com/surya-koritala/Glyd/releases/download/v0.28.0/glyd-v0.28.0-linux-aarch64.tar.gz"
      sha256 "5f7e11329e58439fe0e3de3119ffda301933f7a1153bf6c13b430276d2064c76"
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
