#!/usr/bin/env python3
"""Fail if a milestone gate script has a step that the guide's page for it does not name.

    python3 scripts/check_gate_docs.py

The steps are read from the script (benchmarks/gates/m1-gate.sh, m2-gate.sh): every `step PASS|FAIL|SKIP|INFO NAME` (a name like `plan:$m` is the step
`plan`) and REQUIRED_EXT. The page is docs/guide/src/gate.md (M1) or gate-m2.md (M2), which names each step in backticks. A step that is plumbing, not a
gate step, is in the gate's PLUMBING with the reason. Both directions are checked: a step on the page that the script no longer has is stale.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUMBING_EXT = "the line for a required result file that is missing: the files are named on the page"
GATES = (
    # (script, page, PLUMBING)
    (
        "m1-gate.sh",
        "gate.md",
        {
            "jq": "the gate needs jq to read JSON: a missing tool, not a gate step",
            "glyd-version": "`glyd --version` must start: a precondition of every step",
            "ext": PLUMBING_EXT,
        },
    ),
    (
        "m2-gate.sh",
        "gate-m2.md",
        {
            "jq": "the gate needs jq to read JSON: a missing tool, not a gate step",
            "curl": "the gate sends its requests with curl: a missing tool, not a gate step",
            "glyd-version": "`glyd --version` must start: a precondition of every step",
            "row": "a line of the model list that does not have its ten fields: the list is broken, not a gate step",
            "ext": PLUMBING_EXT,
        },
    ),
)
# a name in backticks, with what follows its first colon (`plan:MODEL`, `plan:MODEL:LEVEL@CONTEXT`)
NAMED = re.compile(r"`([a-z0-9][a-z0-9.-]*)(?::[^`]*)?`")


def steps(script, plumbing):
    text = (ROOT / "benchmarks" / "gates" / script).read_text()
    found = {m.group(1) for m in re.finditer(r'step (?:PASS|FAIL|SKIP|INFO) "?([a-z0-9][a-z0-9-]*)', text)}
    m = re.search(r'^REQUIRED_EXT="([^"]+)"', text, re.M)
    if not found or not m:
        sys.exit(f"check_gate_docs.py: {script} no longer has the shape this reads (step lines, REQUIRED_EXT): fix the script")
    return found - set(plumbing), set(m.group(1).split())


def check(script, page_name, plumbing):
    page = (ROOT / "docs" / "guide" / "src" / page_name).read_text()
    named = set(NAMED.findall(page))
    inline, ext = steps(script, plumbing)
    bad = []
    for s in sorted(inline | ext):
        if s not in named:
            bad.append(f"the gate step `{s}` is not on docs/guide/src/{page_name}")
    # a step the page names in its tables that the script does not have
    rows = set()
    for line in page.splitlines():
        if line.startswith("| `"):
            rows |= set(NAMED.findall(line.split("|")[1]))
    for s in sorted(rows - inline - ext):
        bad.append(f"docs/guide/src/{page_name} names `{s}`, which {script} no longer has")
    return bad, len(inline), len(ext)


def main():
    bad, notes = [], []
    for script, page, plumbing in GATES:
        b, inline, ext = check(script, page, plumbing)
        bad += b
        notes.append(f"{script}: {inline} steps and {ext} required results are on the page")
    if bad:
        sys.exit("check_gate_docs.py:\n  " + "\n  ".join(bad))
    print("gate docs ok: " + "; ".join(notes))


if __name__ == "__main__":
    main()
