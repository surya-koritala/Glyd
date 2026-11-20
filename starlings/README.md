# Starlings

One model across several GPUs: two cards in one machine, or machines on one LAN. Each GPU holds a contiguous range of
layers; tokens cross between stages once per step, and the output is the same tokens the whole model gives on one card.
Becomes `glyd cluster` ([ROADMAP.md](../ROADMAP.md), M5).

**Beta:** 2 GPUs, 2 machines, NVIDIA on Linux, Qwen3.5-family models. Guide: [docs/beta.md](docs/beta.md).

```bash
starlings up --model glyd/Qwen3.8-27B-swift                 # one machine, its GPUs
starlings up --model glyd/Qwen3.8-27B-swift --machines 2    # first machine; prints a join code
starlings join --model glyd/Qwen3.8-27B-swift               # second machine; enter the code, approve on the first
```

Then a chat page and an OpenAI-style `/v1/chat/completions` on loopback. Two RTX 4090 machines (0.4-0.6 ms apart):
43.5 tokens/s split against 46.0 on one card; 51.9-53.8 with drafting ([benchmarks/](benchmarks/)).

## How it works

- Each stage daemon runs `glyd-engine rangeworker` for its layers over a framed stdio protocol
  ([../docs/range-worker.md](../docs/range-worker.md), [docs/partial-worker-contract.md](docs/partial-worker-contract.md)).
- Machines enroll with one-time codes and key fingerprints; traffic is mutual TLS over TCP with credit flow control;
  mDNS hints are optional and never enroll anything ([docs/enrollment.md](docs/enrollment.md), [docs/daemon.md](docs/daemon.md),
  [docs/discovery.md](docs/discovery.md)).
- The split fits each GPU with 1 GiB to spare ([docs/daemon-pipeline.md](docs/daemon-pipeline.md)).

## Crates

| Crate | What |
| :--- | :--- |
| `starlings-protocol` | wire types and schemas |
| `starlings-core` | control-plane state, admission, placement |
| `starlings-adapters` | backends: the Glyd range worker, external services, a fake for tests |
| `starlings-security` | identities, TLS, authorization, revocation, listeners |
| `starlings-cli` | the `starlings` program |

## Build and test

Starlings' crates are members of the repository's workspace (one `Cargo.lock`, one `target/` at the root), so `cargo test --workspace` at the root covers them. Rust 1.97.1 is pinned in the root `rust-toolchain.toml`; formatting follows the `rustfmt.toml` in this directory.

```bash
S="-p starlings-protocol -p starlings-core -p starlings-adapters -p starlings-cli -p starlings-security"
cargo build $S --locked
cargo test $S --locked                   # a few tests need IPv6 loopback and systemd-analyze >= 250
cargo clippy $S --all-targets --all-features --locked -- -D warnings
```

Design and contracts: [docs/architecture.md](docs/architecture.md) and the rest of [docs/](docs/).
