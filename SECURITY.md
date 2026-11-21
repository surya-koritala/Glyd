# Security policy

This covers everything in this repository: the codec and store (`glyd`, `glyd-store`), the inference engine
(`glyd-engine`, `glyd serve`), the GPU library, the installer, and Starlings (`starlings/`, beta).

## Supported versions

Fixes go into the latest release only. There are no long-term branches. Streams written by any earlier release keep
decoding with the latest one (see [docs/codec/spec.md](docs/codec/spec.md)), so upgrading is the fix for every version.

Starlings is a beta. Report problems in it the same way, but expect the fix in the next release, not a backport.

## Reporting a vulnerability

Please do not open a public issue. Report it privately, either way:

- GitHub: the repository's **Security** tab, **Report a vulnerability**
- email: suryakoritala@getglyd.com

Include the version (`glyd --version`, `glyd-engine --version`), what you ran or sent, and what happens. A model file,
request or stream that triggers it is the most useful thing to attach. Do not send anyone else's data.

## Response times

Glyd has one maintainer today, so these are targets, not guarantees:

| Step | Target |
| :--- | :--- |
| Acknowledge the report | 7 days |
| Say whether it is a vulnerability, and how severe | 14 days |
| Release a fix for a confirmed one | 30 days; 90 days at the latest |

If a fix needs longer, we say so in the report thread and agree a date with you. Please give us the 90 days before
publishing details.

## After a report

The fix ships in a release with a GitHub security advisory that names the affected versions. A fix for something
already exploited goes out as a patch release at once, without waiting for the next planned one. Reporters are
credited in the advisory unless they ask not to be.

## What is in scope

**The HTTP server** (`glyd serve`, `glyd-engine serve`). It reads requests from the network and runs on your machine.

- By default it listens on `127.0.0.1` only (port 8000, or the next free port up to 8020), and refuses a `Host` that
  is not a loopback or container name (421) and an `Origin` that is not its own page (403). CORS is off: no
  `Access-Control-Allow-Origin` is sent to any page the user has not listed (`--allowed-origins`, never `*`). These checks
  stay on for a loopback `--host` too. An address that is not loopback (`--host 0.0.0.0`) does not start without an API key
  unless `--insecure-no-auth` says an open server is meant; the Origin check stays on there too, and `--allowed-hosts`
  limits the names a request may use. The service (`glyd service`) and the `--ollama-port` listener follow the same rules.
- CI scans a default install for a listener that is not on this computer's own address: the service in `glyd-engine/tests/listeners.rs` on every push, and a default `glyd serve` with its `--ollama-port` listener in the `gpu` job (`scripts/listener_scan.py`, `tests/ops_check.py`).
- Image URLs in a request are fetched only from public addresses: the address a name resolves to is checked where the
  connection is made, so a name that answers a public address first and a private one next (DNS rebinding) is refused.
- An API key (`GLYD_API_KEY`, or `--api-key`) guards the `/v1` routes. It is compared as SHA-256 digests in constant
  time. The chat page, `/health`, `/ready` and `/stats` do not need the key; `/metrics` does.
- In scope: reading or using the model without the key where one is set; getting past the loopback checks on an
  address the server chose itself; a request that crashes, hangs or exhausts memory beyond the limits it sets
  (request body 16 MiB, header block 64 KiB, 256 headers, 64 connections, 30 s for the request line and headers, 120 s for the body, 60 s for a write to a client that reads nothing: a request that is too slow is answered 408); an out-of-bounds read or
  write while parsing a request; reading a file outside the chat page's own directory.
- Image URLs in chat requests: the server fetches only `http(s)` URLs whose host resolves to a public address, at
  every redirect (up to 5), with a 64 MiB and 30 s limit. `file:` and other schemes are refused. A way to make it
  fetch from this computer, its network or a cloud metadata service is in scope.

**Model files and packs.** Checkpoints (`safetensors`, `config.json`, tokenizer files) and the packed directories
`glyd pack` writes (`glyd.json`, `engine.safetensors`, packs) are read by the engine and the GPU library. A model
from an untrusted source must not be able to run code, or read or write memory outside its own buffers. In scope:
memory-unsafe behavior or code execution on a crafted file. A clean error or refusal on a malformed file is the
expected result. A panic or crash of one `glyd run` on a malformed file is a bug: please report it, publicly if you
like. A malformed file that can also hurt other users of the same machine or server is a vulnerability.

**The codec decoders** read untrusted bytes, so on any input these are security bugs: a crash or panic, a hang, an
out-of-bounds read or write, memory growing without bound, or a round trip that returns different bytes without an
error. The same holds for `glyd-store` reading its metadata and objects.

**Downloads.** `glyd fetch`, `run` and `serve` download models from the Hugging Face Hub (or `HF_ENDPOINT`). Files are
checked by size and, where the Hub gives one, by SHA-256. The Hub token goes only to the Hub's own host, never to
the storage a redirect points to. In scope: sending the token elsewhere, or accepting a file whose size or hash does
not match. A SHA-256 shows the bytes are the ones the Hub lists, not that the publisher can be trusted.

**The installer** (`scripts/install.sh`, `curl ... | sh`). It uses HTTPS only (TLS 1.2 or later) and checks each
tarball against the SHA-256 the release lists. The `uv` installer it may run is pinned to a version and a SHA-256.
In scope: a way to make it install a file that fails the check, run code from a source other than the pinned
release, or write outside the directories it names. The checksum is published with the tarball, so it guards
against a damaged or swapped download, not against a compromised release.

**The GPU library** (`libglyd_gpu_cuda12.so`, `libglyd_gpu_cuda13.so`, `gpu/`, `glyd-gpu/`). It is loaded with
`dlopen` by name from the loader's path, or from the file `GLYD_GPU_LIB` names. In scope: memory-unsafe behavior in
the library on input from a model file or the codec's streams. Whoever can set `GLYD_GPU_LIB`, or write into the
loader's search path, can already run code as you, so that is not a vulnerability.

**Starlings.** Machines enroll with one-time codes (256-bit, valid about five minutes) and key fingerprints, and
traffic between them is mutual TLS. The enrollment address is `127.0.0.1:9741` unless you choose a LAN address;
wildcard and multicast addresses are refused. In scope: joining a cluster without a valid code and an approval,
reading or altering traffic between enrolled machines, using a revoked identity, and parsing bugs in anything read
from the network. Enrolled machines are trusted with their share of the model and the prompts: Starlings does not
protect against an enrolled machine that computes wrong answers or leaks what it is given. The full list of boundaries is in
[starlings/docs/security-boundaries.md](starlings/docs/security-boundaries.md).

## What is not a vulnerability

- Starting the server with `--host` on a non-loopback address and `--insecure-no-auth` (it does not start without a key
  or that flag). It prints a warning that anyone who can reach the machine can use the model, and that the traffic is plain HTTP. Use a VPN or an SSH tunnel, or put a
  TLS proxy in front. A key given as `--api-key` shows in `ps`: the warning says to use the environment variable.
- The chat page, `/health` and `/stats` answering without a key when a key is set. The key guards `/v1`.
- Plain HTTP on the server's own port. The server has no TLS.
- Anything that needs you to run code already: a hostile `GLYD_GPU_LIB`, a modified installer you ran by hand, a
  build from a source tree someone changed.
- What a model says. Wrong, harmful or manipulated output, prompt injection into an application built on top, and
  jailbreaks are model behavior, not Glyd's. The same goes for a model's license or training data.
- Slowness or memory use from a client that holds the key and sends many large requests. Limit access with the key
  and a proxy.
- Bugs in vLLM, PyTorch, CUDA, ROCm, drivers or other dependencies. Report those upstream. If Glyd ships or pins a
  version with a known issue, tell us and we will update it.
- Reports from automated scanners with no demonstration that the problem can be reached.

When you are not sure which side a report falls on, send it privately. We will tell you if it belongs in a public issue.
