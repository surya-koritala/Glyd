# Contributing

Thanks for helping. Bug reports, speed results from your hardware and fixes are all welcome.

## What is in this repository

- `setup.sh` and `tools/`: the installer, the download and start scripts, and the chat app (`tools/chat.html`, `tools/web/`).
- `bench/`: the quality benchmarks behind the README's numbers. `results/`: their logs.
- `tests/`: `node tests/test_flow.js` checks the flowchart renderer; `python3 tests/test_server.py --url http://127.0.0.1:8080` checks
  a running server end to end.

Condor is a folder of the Glyd repository: the engine it runs is [glyd-engine/](../glyd-engine) with its CUDA kernels (the GPU library in
[gpu/](../gpu) is the Python package's), all Apache-2.0. Fixes to the engine are pull requests to this same repository.

## Reporting a problem

Open an [issue](https://github.com/surya-koritala/Glyd/issues) with your GPU, driver, RAM and Linux version, what you
ran, and what it printed. For speed, run `tools/bench.sh` and paste the block it prints at the end.

## Pull requests

- Keep a change small and about one thing, and say how you tested it.
- Sign off your commits (`git commit -s`): it certifies the [Developer Certificate of Origin](https://developercertificate.org/), that you
  wrote the change or have the right to submit it under this repository's license (Apache-2.0).

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
