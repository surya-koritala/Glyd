# Governance

How Glyd is run today: who decides, how, and how that can change. Glyd has one maintainer. This page is written so that
a second and third one can join without rewriting it.

## Roles

- **Contributor.** Anyone who sends an issue, a pull request, a review or a measurement. Contributions are certified with
  the [Developer Certificate of Origin](DCO) (`git commit -s`) and licensed under the Apache License 2.0
  ([CONTRIBUTING.md](CONTRIBUTING.md)).
- **Maintainer.** Can merge pull requests, triage issues and speak for an area of the project. Listed in
  [MAINTAINERS](MAINTAINERS).
- **Owner.** Surya Koritala, the founder and the only maintainer today. The owner holds the repository, the release
  accounts (GitHub releases, PyPI, crates.io, Homebrew, Hugging Face, the site) and the project name.

## How decisions are made

- **Small changes** (a fix, a test, a doc, a flag) are decided in the pull request. A maintainer reviews and merges it.
- **Larger changes** (a new model family or backend, a change to a command line, an API or a file format, a new
  dependency) start as an issue. [CONTRIBUTING.md](CONTRIBUTING.md) asks for this. The issue says what changes and why, and what
  it costs. Anyone can comment. A maintainer decides when the discussion has settled, and writes the decision in the
  issue.
- **Compatibility is a rule, not a vote.** A change may not make files that an earlier version wrote unreadable
  (see [docs/spec.md](docs/spec.md)). A claim about speed, size or quality needs a measurement others can repeat.
- **Disagreement.** With one maintainer, the maintainer decides. With several, they try to agree in the issue. If they
  cannot after seven days, a majority of the maintainers decides. A tie goes to the owner.
- **Security reports** are handled privately by the maintainers, as [SECURITY.md](SECURITY.md) describes.
- **Conduct** follows the [code of conduct](CODE_OF_CONDUCT.md).

## Releases

A release is cut by a maintainer following [RELEASING.md](RELEASING.md): one version number, one tag, one
GitHub release. Publishing to PyPI, crates.io, Homebrew, Hugging Face and the site needs accounts that only the owner
holds today. A maintainer who needs to release without the owner asks the owner to add them to those accounts, or to
publish for them.

## Licence

The code is under the Apache License 2.0, with no separate contributor agreement and no right to relicense a
contribution. Releases v0.29.0 to v0.32.1 were published under the Business Source License 1.1 ([NOTICE](NOTICE)).
Changing the license of the code in this repository would need the agreement of everyone whose contribution is in it.

## Becoming a maintainer

Maintainers are chosen for work they have already done in the project, not for a promise.

1. **What we look for.** Over about three months, several merged pull requests of some size in one area (engine,
   kernels, cluster, docs, installers or CI), reviews of other people's pull requests, and answers to issues. The
   changes follow the project's rules: measured claims, compatibility, small scope.
2. **How to be nominated.** A maintainer nominates you in an issue, or you ask in one. Say which area you would
   look after.
3. **How it is decided.** The existing maintainers agree. With one maintainer, that maintainer decides.
4. **What changes.** You get write access, you are added to [MAINTAINERS](MAINTAINERS), and you may merge in your
   area. A maintainer does not merge their own pull request without a second review where there is a second
   maintainer to give it.

A maintainer who is not active for six months, or who asks to step down, moves to an "emeritus" list in MAINTAINERS and
loses write access. They can come back by the same route. A maintainer who breaks the code of conduct can be removed by
the other maintainers; the owner can remove one at any time and says why in an issue.

The areas where the project wants maintainers beyond the owner: the engine and its runtime, GPU kernels, cluster
(Starlings), documentation, and CI and release engineering.

## Name and logo (draft)

This section is a draft. It becomes the policy when the owner approves it as final.

The Apache License 2.0 gives you the code. It does not give you the right to use the project's name or logo (section 6
of the license). This is what the project asks.

**You may, without asking:**

- Say that your software works with Glyd, uses Glyd, or is built on Glyd, if it does.
- Name Glyd in an article, a talk, a comparison or a benchmark, with the version you measured.
- Redistribute an unmodified release of Glyd under its name.
- Use the name for a plugin, adapter or tool, in the form "X for Glyd" or "X for use with Glyd", where X is your name.
- Use the logo, where the project publishes one, unmodified, to point to this project.

**Ask first (open an issue, or write to the address in [SECURITY.md](SECURITY.md)):**

- A product, company, package, domain or social account that has Glyd as its name or the first word of its name.
- A modified build or fork that you distribute under the name Glyd. A fork may keep the license and the history, but
  needs its own name.
- Any use that says or suggests the project endorses you, or that you are the project.
- Model repositories on Hugging Face or elsewhere under the `glyd/` name, which are the project's.

**Do not:**

- Use the name or logo to sell something in a way that makes it look like the project's own.
- Change the logo, or combine it with another mark.

Using the name in the ways above, or in a way that does not mislead anyone, does not need a license from the owner.
If you are not sure, ask in an issue.
