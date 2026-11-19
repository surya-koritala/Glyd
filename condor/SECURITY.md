# Security

Please report a security problem privately, not in a public issue: email **suryakoritala@getglyd.com** with what you found and how to
reproduce it. You will get an answer within a few days.

Things worth knowing:

- The server listens on `127.0.0.1` (this computer only) by default, and refuses requests from other web pages' scripts.
- With `--host 0.0.0.0` it is open to your network. Set an API key first (`GLYD_API_KEY=... tools/serve.sh`); the traffic is plain HTTP,
  so use a VPN or an SSH tunnel across networks you don't trust.
- The chat app's live preview runs the model's pages in a sandbox with no network access.
- The engine and its library are checked against `SHA256SUMS` when they are installed.
