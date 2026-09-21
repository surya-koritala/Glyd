# Homebrew formula: `brew install surya-koritala/glyd/glyd` once this file
# sits in a tap repository (github.com/surya-koritala/homebrew-glyd,
# Formula/glyd.rb), or `brew install --build-from-source Formula/glyd.rb`
# from a checkout.
class Glyd < Formula
  desc "Compression for the data that fills object storage: record mode, packs, a store that compresses across objects"
  homepage "https://github.com/surya-koritala/Glyd"
  url "https://github.com/surya-koritala/Glyd/archive/refs/tags/v0.10.0.tar.gz"
  sha256 "REPLACED-BY-THE-RELEASE-SCRIPT"
  license "Apache-2.0"
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
