# Homebrew formula: `brew install surya-koritala/glyd/glyd` (the tap
# github.com/surya-koritala/homebrew-glyd carries a copy of this file),
# or `brew install --build-from-source Formula/glyd.rb` from a checkout.
class Glyd < Formula
  desc "Compression for object storage: record mode, packs, a cross-object store"
  homepage "https://github.com/surya-koritala/Glyd"
  url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.14.7.tar.gz"
  sha256 "b98eb87274fd0a8c86e70c5538a73e0ace2f1ff4a6f3eada34a11777a8391bfd"
  # The codec and CLI: BSD-3-Clause or GPL-2.0; the glyd-store binary: BUSL-1.1.
  license all_of: [
    { any_of: ["BSD-3-Clause", "GPL-2.0-only"] },
    "BUSL-1.1",
  ]
  head "https://github.com/surya-koritala/Glyd.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
    system "cargo", "install", *std_cargo_args(path: "glyd-store")
    include.install "include/glyd.h"
  end

  test do
    (testpath/"a.txt").write("hello hello hello hello glyd\n" * 100)
    system bin/"glyd", "--max", testpath/"a.txt", "-o", testpath/"a.glyd"
    system bin/"glyd", "-d", testpath/"a.glyd", "-o", testpath/"b.txt"
    assert_equal (testpath/"a.txt").read, (testpath/"b.txt").read
  end
end
