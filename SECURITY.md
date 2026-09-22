# Security policy

## Supported versions

Fixes go into the latest release. Streams written by any earlier release
keep decoding with it (see [docs/spec.md](docs/spec.md)), so upgrading is
the fix for every version.

## Reporting a vulnerability

Please do not open a public issue. Report it privately, either way:

- GitHub: the repository's **Security** tab, **Report a vulnerability**
- email: suryakoritala1324@gmail.com

Include the version (`glyd --version`), the input that triggers it (or how
to make it), and what happens.

The decoders read untrusted bytes, so on any input these are security
bugs: a crash or panic, a hang, an out-of-bounds read or write, memory
growing without bound, or a round trip that returns different bytes without
an error. The same holds for `glyd-store` reading its metadata and objects.

## After a report

The fix ships in a release with a GitHub security advisory describing it.
Reporters are credited in the advisory unless they ask not to be.
