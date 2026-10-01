#!/bin/bash
# the Rust glyd of the onboarding branch, for acceptance.sh --rust-glyd (CPU only: no GPU lock)
set -e
export PATH=$HOME/.cargo/bin:$PATH
if ! command -v cargo > /dev/null; then curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable > ~/onb/rustup.txt 2>&1; fi
if [ -d ~/onb/rust-src/.git ]; then git -C ~/onb/rust-src pull -q; else git clone -q --depth 1 --branch onboarding https://github.com/surya-koritala/Glyd ~/onb/rust-src; fi
cd ~/onb/rust-src && git log -1 --format="%h %s" | cut -c1-90
cargo build --release --bin glyd 2>&1 | tail -5
cp target/release/glyd ~/onb/glyd-rust && ls -la ~/onb/glyd-rust && ~/onb/glyd-rust --version
