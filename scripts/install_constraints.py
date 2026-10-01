#!/usr/bin/env python3
"""The package versions scripts/install.sh installs glyd[vllm] with: those an acceptance run installed.

    python3 scripts/install_constraints.py FREEZE        rewrite the list in install.sh from FREEZE
    python3 scripts/install_constraints.py --check FREEZE   fail if install.sh's list is not what FREEZE gives
    python3 scripts/install_constraints.py --list           print the list in install.sh

FREEZE is a run's logs/freeze.txt (gpu/vllm/acceptance.sh writes it: `uv tool list --show-with`, then `uv pip freeze` of the tool's
environment). Every `name==version` line goes into the list but glyd itself, which the script's own pin names; a pre-release other
than an OpenTelemetry beta is refused (the acceptance run fails on one as well). uv takes the list as constraints: it limits which
version of a package is chosen and installs nothing that the requirements do not ask for, so ziglang, which only a machine with no C
compiler gets, can be in it.
"""
import os
import re
import sys

INSTALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install.sh")
LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.+!_-]*$")
PRE = re.compile(r"==[0-9][0-9.]*(a|b|rc|dev)[0-9]+$")
BLOCK = re.compile(r"(cat <<'CONSTRAINTS'\n)(.*?)(CONSTRAINTS\n)", re.S)


def parse(freeze):
    """The sorted `name==version` lines of a freeze, without glyd; ValueError for a pre-release (other than opentelemetry's betas)."""
    out = {}
    for line in freeze.splitlines():
        line = line.strip()
        if not LINE.match(line):
            continue  # (the uv tool list lines, a `glyd @ file://...` line, comments)
        name = line.split("==")[0].lower().replace("_", "-")
        if name == "glyd":
            continue
        if PRE.search(line) and not (name.startswith("opentelemetry-") and re.search(r"==[0-9.]+b[0-9]+$", line)):
            raise ValueError(f"a pre-release in the freeze: {line}")
        out[name] = line
    return [out[k] for k in sorted(out)]


def listed(install):
    """The lines in install.sh's list."""
    m = BLOCK.search(install)
    if not m:
        raise ValueError("scripts/install.sh has no `cat <<'CONSTRAINTS'` list")
    return [l for l in m.group(2).splitlines() if l]


def rewrite(install, lines):
    return BLOCK.sub(lambda m: m.group(1) + "".join(l + "\n" for l in lines) + m.group(3), install, count=1)


def main(argv):
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    text = open(INSTALL).read()
    if "--list" in flags:
        print("\n".join(listed(text)))
        return 0
    if len(args) != 1 or set(flags) - {"--check"}:
        print(__doc__)
        return 2
    lines = parse(open(args[0]).read())
    if not lines:
        print("no name==version lines in", args[0])
        return 1
    if "--check" in flags:
        if listed(text) != lines:
            print(f"scripts/install.sh lists {len(listed(text))} packages, {args[0]} gives {len(lines)}, and they differ")
            return 1
        print(f"scripts/install.sh's {len(lines)} packages are {args[0]}'s")
        return 0
    open(INSTALL, "w").write(rewrite(text, lines))
    print(f"scripts/install.sh: {len(lines)} packages from {args[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
