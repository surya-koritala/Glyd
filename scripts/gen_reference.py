#!/usr/bin/env python3
"""Put the API reference and the service reference into the guide, from the files in docs/.

    python3 scripts/gen_reference.py [OUT_DIR]        (default: docs/guide/src)

docs/serve-api.md becomes docs/guide/src/api.md and docs/service.md becomes docs/guide/src/service.md, so the site's reference
is the repository's file and cannot drift. Links are rewritten: one to the other of the two goes to its page on the site, any
other relative link goes to the file on GitHub (main). Nothing else is changed.
"""
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GH = "https://github.com/surya-koritala/Glyd"
PAGES = {"docs/serve-api.md": "api.md", "docs/service.md": "service.md"}


def rewrite(text, src):
    base = posixpath.dirname(src)

    def fix(m):
        label, target = m.group(1), m.group(2)
        if re.match(r"([a-z]+:|#|mailto:)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        rel = posixpath.normpath(posixpath.join(base, path))
        if rel in PAGES:
            return f"[{label}]({PAGES[rel]}{'#' + frag if frag else ''})"
        kind = "tree" if (ROOT / rel).is_dir() else "blob"
        if not (ROOT / rel).exists():
            sys.exit(f"gen_reference.py: {src} links to {target}, which is not in the repository")
        return f"[{label}]({GH}/{kind}/main/{rel}{'#' + frag if frag else ''})"

    return re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", fix, text)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "guide" / "src"
    out.mkdir(parents=True, exist_ok=True)
    for src, name in PAGES.items():
        text = (ROOT / src).read_text()
        note = f"<!-- Generated from {src} by scripts/gen_reference.py: edit that file. -->\n\n"
        (out / name).write_text(note + rewrite(text, src))
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
