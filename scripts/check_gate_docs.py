#!/usr/bin/env python3
"""Fail if the M1 gate script has a step that the guide's gate page does not name.

    python3 scripts/check_gate_docs.py

The steps are read from benchmarks/gates/m1-gate.sh: every `step PASS|FAIL|SKIP NAME` (a name like `plan:$m` is the step `plan`) and
REQUIRED_EXT. The page is docs/guide/src/gate.md, which names each step in backticks. A step that is plumbing, not a gate step, is in
PLUMBING with the reason. Both directions are checked: a step on the page that the script no longer has is stale.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUMBING = {
    "jq": "the gate needs jq to read JSON: a missing tool, not a gate step",
    "glyd-version": "`glyd --version` must start: a precondition of every step",
    "ext": "the line for a required result file that is missing: the files are named on the page",
}


def steps():
    text = (ROOT / "benchmarks" / "gates" / "m1-gate.sh").read_text()
    found = {m.group(1) for m in re.finditer(r'step (?:PASS|FAIL|SKIP) "?([a-z0-9][a-z0-9-]*)', text)}
    m = re.search(r'^REQUIRED_EXT="([^"]+)"', text, re.M)
    if not found or not m:
        sys.exit("check_gate_docs.py: m1-gate.sh no longer has the shape this reads (step lines, REQUIRED_EXT): fix the script")
    return found - set(PLUMBING), set(m.group(1).split())


def main():
    page = (ROOT / "docs" / "guide" / "src" / "gate.md").read_text()
    named = set(re.findall(r"`([a-z0-9][a-z0-9-]*)(?::MODEL)?`", page))
    inline, ext = steps()
    bad = []
    for s in sorted(inline | ext):
        if s not in named:
            bad.append(f"the gate step `{s}` is not on docs/guide/src/gate.md")
    # a step the page names in its tables that the script does not have
    rows = set()
    for line in page.splitlines():
        if line.startswith("| `"):
            rows |= set(re.findall(r"`([a-z0-9][a-z0-9-]*)(?::MODEL)?`", line.split("|")[1]))
    for s in sorted(rows - inline - ext):
        bad.append(f"docs/guide/src/gate.md names `{s}`, which m1-gate.sh no longer has")
    if bad:
        sys.exit("check_gate_docs.py:\n  " + "\n  ".join(bad))
    print(f"gate docs ok: {len(inline)} steps and {len(ext)} required results are on the page")


if __name__ == "__main__":
    main()
