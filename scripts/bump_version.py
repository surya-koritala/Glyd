#!/usr/bin/env python3
"""Set, or check, the version in every file that carries it.

    python3 scripts/bump_version.py 0.26.0            # a release: the code's versions, the docs', the CHANGELOG heading dated
    python3 scripts/bump_version.py 0.26.0rc3         # a pre-release: the code's versions only (the docs stay at the last release)
    python3 scripts/bump_version.py --check           # every file says the same version (ci.yml)
    python3 scripts/bump_version.py --check v0.26.0   # ... and it is this tag's (release.yml, before anything is built)
    python3 scripts/bump_version.py --selftest        # this script's own checks (ci.yml)

PEP 440 (Python: 0.26.0rc3) and semver (Cargo: 0.26.0-rc.3) spell a pre-release differently and name the
same version. The argument or tag may be in either spelling (a leading v is allowed); each file gets its
own ecosystem's, and --check fails on a file in the other's. Only X.Y.Z and its alpha, beta and rc
pre-releases (X.Y.Zrc3, X.Y.Za1, X.Y.Zb2) exist here.

What carries it: the two Cargo.toml (and glyd-store's dependency on glyd), the two glyd entries in
Cargo.lock, pyproject.toml (and its extras' pins of glyd-gpu), glyd/__init__.py and scripts/install.sh (the release it installs), in every version. In a release (X.Y.Z) also ROADMAP.md,
docs/adoption.md and docs/index.html, where they name the current release, and CHANGELOG.md's heading
(`## Unreleased` or `## vX.Y.Z (Unreleased)` becomes `## vX.Y.Z` with the date). Formula/glyd.rb is not
here: it needs the release's files to exist, see scripts/bump_formula.py.
"""
import argparse
import datetime
import pathlib
import re
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE = {"a": "alpha", "b": "beta", "rc": "rc"}  # PEP 440's letter, and Cargo's word for it
LONG = {"a": "a", "alpha": "a", "b": "b", "beta": "b", "rc": "rc"}
DASH = "—"
CHANGELOG = "CHANGELOG.md"

# (file, ecosystem, pattern with the version in group v, matches expected)
CODE = [
    ("Cargo.toml", "cargo", r'^version = "(?P<v>[^"]+)"', 1),
    ("glyd-store/Cargo.toml", "cargo", r'^version = "(?P<v>[^"]+)"', 1),
    ("glyd-store/Cargo.toml", "cargo", r'^glyd = \{ version = "(?P<v>[^"]+)"', 1),
    ("Cargo.lock", "cargo", r'^name = "glyd(?:-store)?"\nversion = "(?P<v>[^"]+)"', 2),
    ("bindings/python/pyproject.toml", "py", r'^version = "(?P<v>[^"]+)"', 1),
    ("bindings/python/pyproject.toml", "py", r'^gpu = \["glyd-gpu\[gpu\]==(?P<v>[^"]+)"\]', 1),  # the extras pin the glyd-gpu of the same version
    ("bindings/python/pyproject.toml", "py", r'^vllm = \["glyd-gpu\[vllm\]==(?P<v>[^"]+)"\]', 1),
    ("bindings/python/glyd/__init__.py", "py", r'^__version__ = "(?P<v>[^"]+)"', 1),
    ("scripts/install.sh", "py", r'^GLYD_VERSION="\$\{GLYD_VERSION:-(?P<v>[^}]+)\}"', 1),  # the release the installer pins
]
# where a release is named in prose: the text (version in group v) and what a release writes there
DOCS = [
    ("ROADMAP.md", r"^## Where it stands \(v(?P<v>[\d.]+), \d{4}-\d{2}-\d{2}\)$", "## Where it stands (v{v}, {date})"),
    ("docs/adoption.md", r"stands on each \(v(?P<v>[\d.]+), [A-Z][a-z]+ \d{4}\)", "stands on each (v{v}, {month})"),
    ("docs/index.html", r"Rust, C ABI, CLI; v(?P<v>[\d.]+), [A-Z][a-z]+ \d{4}\.", "Rust, C ABI, CLI; v{v}, {month}."),
]


def parse(s):
    """'v0.26.0-rc.2', '0.26.0rc2' -> (0, 26, 0, 'rc', 2); '0.26.0' -> (0, 26, 0)."""
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-.]?(a|alpha|b|beta|rc)[-.]?(\d+))?", s)
    if not m:
        sys.exit(f"{s!r} is not a version here: X.Y.Z, or X.Y.Z with rcN, aN or bN")
    x, y, z, pre, n = m.groups()
    return (int(x), int(y), int(z)) + ((LONG[pre], int(n)) if pre else ())


def spell(v, ecosystem):
    s = "%d.%d.%d" % v[:3]
    if len(v) > 3:
        s += f"{v[3]}{v[4]}" if ecosystem == "py" else f"-{PRE[v[3]]}.{v[4]}"
    return s


def read(root, path):
    return (root / path).read_text(encoding="utf-8")


def swap(pattern, text, new):
    """text with the version (group v) of every match of pattern replaced by new; and the number of matches."""
    return re.subn(pattern, lambda m: m.group(0)[: m.start("v") - m.start()] + new + m.group(0)[m.end("v") - m.start():], text, flags=re.M)


def current(root):
    path, _, pat, _ = CODE[0]
    m = re.search(pat, read(root, path), re.M)
    return parse(m.group("v")) if m else sys.exit(f"{path}: no version")


def problems(root, want=None):
    """What is wrong with the versions under root: every file must say `want` (default: Cargo.toml's), in its own spelling."""
    want = want or current(root)
    bad = []
    for path, eco, pat, n in CODE:
        vs = [m.group("v") for m in re.finditer(pat, read(root, path), re.M)]
        if len(vs) != n:
            bad.append(f"{path}: {len(vs)} matches of {pat!r}, expected {n} (this script's table is out of date)")
        bad += [f"{path}: {v}, expected {spell(want, eco)}" for v in vs if v != spell(want, eco)]
    if len(want) == 3:  # a release: the docs name it
        v = spell(want, "py")
        for path, pat, _ in DOCS:
            m = re.search(pat, read(root, path), re.M)
            if not m or m.group("v") != v:
                bad.append(f"{path}: does not name v{v} ({m.group(0) if m else 'the text this script looks for is gone'})")
        if not re.search(rf"^## v{re.escape(v)} {DASH} \d{{4}}-\d{{2}}-\d{{2}}$", read(root, CHANGELOG), re.M):
            bad.append(f"{CHANGELOG}: no heading '## v{v} {DASH} DATE' (still Unreleased?)")
    return bad


def bump(root, version, date):
    """Set every file to version (a release: dated `date`); the files changed."""
    want = parse(version)
    files = {}
    for path, eco, pat, n in CODE:
        text, k = swap(pat, files[path] if path in files else read(root, path), spell(want, eco))
        if k != n:
            sys.exit(f"{path}: {k} matches of {pat!r}, expected {n} (this script's table is out of date)")
        files[path] = text
    if len(want) == 3:
        v = spell(want, "py")
        for path, pat, tmpl in DOCS:
            text, k = re.subn(pat, lambda m: tmpl.format(v=v, date=date.isoformat(), month=date.strftime("%B %Y")), read(root, path), flags=re.M)
            if k != 1:
                sys.exit(f"{path}: {k} matches of {pat!r}, expected 1 (this script's table is out of date)")
            files[path] = text
        head = rf"^## (?:Unreleased|v{re.escape(v)} \(Unreleased\)|v{re.escape(v)} {DASH} \d{{4}}-\d{{2}}-\d{{2}})$"
        files[CHANGELOG], k = re.subn(head, f"## v{v} {DASH} {date.isoformat()}", read(root, CHANGELOG), flags=re.M)
        if k != 1:
            sys.exit(f"{CHANGELOG}: {k} headings '## Unreleased' or '## v{v} (Unreleased)', expected 1")
    changed = [p for p, t in files.items() if t != read(root, p)]
    for p in changed:
        (root / p).write_text(files[p], encoding="utf-8")
    return changed


def selftest():
    def tree(d):
        root = pathlib.Path(d)
        for p in {p for p, *_ in CODE} | {p for p, *_ in DOCS} | {CHANGELOG}:
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / p, root / p)
        return root

    for s, same in [("0.26.0rc2", "0.26.0-rc.2"), ("0.26.0rc2", "v0.26.0-rc2"), ("0.26.0a1", "0.26.0-alpha.1"), ("0.26.0", "v0.26.0")]:
        assert parse(s) == parse(same), (s, same)
    assert spell(parse("0.26.0-alpha.1"), "py") == "0.26.0a1" and spell(parse("0.26.0b2"), "cargo") == "0.26.0-beta.2"
    for s in ["0.26", "0.26.0.dev1", "0.26.0rc", "0.26.0-rc.1+x", "0.26.0.post1", "rc1", ""]:
        try:
            parse(s)
        except SystemExit:
            continue
        raise AssertionError(f"{s!r} was taken for a version")
    day = datetime.date(2031, 2, 3)
    with tempfile.TemporaryDirectory() as d:
        root = tree(d)
        released = re.sub(r"^## (?:(v\S+) \(Unreleased\)|Unreleased)$", lambda m: f"## {m.group(1) or 'v0.0.0'} {DASH} 2000-01-01", read(root, CHANGELOG), flags=re.M)
        (root / CHANGELOG).write_text("## v9.9.9 (Unreleased)\n\n" + released, encoding="utf-8")  # (whatever the real one says, now)
        docs = {p: read(root, p) for p, *_ in DOCS}

        bump(root, "9.9.9rc1", day)  # a pre-release: the code only, in each ecosystem's spelling
        assert not problems(root) and not problems(root, parse("v9.9.9-rc.1")) and problems(root, parse("9.9.9rc2"))
        assert 'version = "9.9.9-rc.1"' in read(root, "Cargo.toml") and read(root, "Cargo.lock").count('version = "9.9.9-rc.1"') == 2
        assert 'glyd = { version = "9.9.9-rc.1"' in read(root, "glyd-store/Cargo.toml")
        assert 'version = "9.9.9rc1"' in read(root, "bindings/python/pyproject.toml") and '__version__ = "9.9.9rc1"' in read(root, "bindings/python/glyd/__init__.py")
        assert 'gpu = ["glyd-gpu[gpu]==9.9.9rc1"]' in read(root, "bindings/python/pyproject.toml") and 'vllm = ["glyd-gpu[vllm]==9.9.9rc1"]' in read(root, "bindings/python/pyproject.toml")
        assert 'GLYD_VERSION="${GLYD_VERSION:-9.9.9rc1}"' in read(root, "scripts/install.sh")
        assert all(read(root, p) == t for p, t in docs.items()) and "## v9.9.9 (Unreleased)" in read(root, CHANGELOG)

        bump(root, "v9.9.9-rc.2", day)  # rc to rc
        assert not problems(root) and "9.9.9rc2" in read(root, "bindings/python/pyproject.toml")

        changed = bump(root, "9.9.9", day)  # the release: the docs and the heading too (v0.25.1's ten files, and the installer's pin)
        assert not problems(root) and len(changed) == 10, changed
        assert f"## v9.9.9 {DASH} 2031-02-03\n" in read(root, CHANGELOG) and "(Unreleased)" not in read(root, CHANGELOG)
        assert "(v9.9.9, 2031-02-03)" in read(root, "ROADMAP.md") and "(v9.9.9, February 2031)" in read(root, "docs/adoption.md")
        assert "CLI; v9.9.9, February 2031." in read(root, "docs/index.html")
        assert bump(root, "9.9.9", day) == []  # a second run changes nothing

        for path, old, new in [  # each slip is found
            ("bindings/python/pyproject.toml", 'version = "9.9.9"', 'version = "9.9.9-rc.1"'),
            ("bindings/python/pyproject.toml", 'gpu = ["glyd-gpu[gpu]==9.9.9"]', 'gpu = ["glyd-gpu[gpu]==9.9.8"]'),
            ("bindings/python/pyproject.toml", 'vllm = ["glyd-gpu[vllm]==9.9.9"]', 'vllm = ["glyd-gpu[vllm]==9.9.8"]'),
            ("bindings/python/glyd/__init__.py", '"9.9.9"', '"9.9.8"'),
            ("scripts/install.sh", "GLYD_VERSION:-9.9.9}", "GLYD_VERSION:-9.9.8}"),
            ("glyd-store/Cargo.toml", 'glyd = { version = "9.9.9"', 'glyd = { version = "9.9.8"'),
            ("Cargo.lock", 'name = "glyd-store"\nversion = "9.9.9"', 'name = "glyd-store"\nversion = "9.9.8"'),
            ("ROADMAP.md", "(v9.9.9,", "(v9.9.8,"),
            ("docs/index.html", "CLI; v9.9.9,", "CLI; v9.9.8,"),
            (CHANGELOG, f"## v9.9.9 {DASH} 2031-02-03", "## v9.9.9 (Unreleased)"),
        ]:
            text = read(root, path)
            assert old in text, (path, old)
            (root / path).write_text(text.replace(old, new, 1), encoding="utf-8")
            assert problems(root), (path, old)
            (root / path).write_text(text, encoding="utf-8")
        assert not problems(root) and problems(root, parse("9.9.10"))

        bump(root, "9.9.10a1", day)  # a pre-release after a release: no docs to keep in step
        assert not problems(root) and "9.9.10-alpha.1" in read(root, "Cargo.toml")
    print("bump_version.py: ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", help="X.Y.Z or a pre-release of it, in either spelling; a tag with its v is fine")
    ap.add_argument("--check", action="store_true", help="change nothing; fail unless every file says the same version (VERSION's, if given)")
    ap.add_argument("--date", type=datetime.date.fromisoformat, default=None, help="a release's date, YYYY-MM-DD (default: today)")
    ap.add_argument("--selftest", action="store_true", help="run this script's own checks")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.check and not args.version:
        ap.error("give a version to set, or --check")
    v = parse(args.version) if args.version else current(ROOT)
    if args.check:
        bad = problems(ROOT, v)
        if bad:
            sys.exit("version files disagree:\n  " + "\n  ".join(bad) + "\n(python3 scripts/bump_version.py VERSION sets them all)")
        print(f"version files agree: Cargo {spell(v, 'cargo')}, Python {spell(v, 'py')}" + ("; the docs name it" if len(v) == 3 else ""))
        return
    changed = bump(ROOT, args.version, args.date or datetime.date.today())
    print(f"{len(changed)} files set to {spell(v, 'cargo')} (Cargo), {spell(v, 'py')} (Python): " + ", ".join(changed))
    if len(v) == 3:
        print(f"the docs and the CHANGELOG heading name it; once the release is up: python3 scripts/bump_formula.py {spell(v, 'py')}")
    else:
        print("a pre-release: the docs stay at the last release")


if __name__ == "__main__":
    main()
