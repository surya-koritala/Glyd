#!/usr/bin/env python3
"""Point Formula/glyd.rb at a published release, after its workflow has uploaded the tarballs:

    python3 scripts/bump_formula.py 0.21.0

then copy the file to the tap, github.com/surya-koritala/homebrew-glyd (Formula/glyd.rb).
Every sha256 is computed from the file its url names, so a missing asset fails here, not in brew.
"""
import hashlib, pathlib, re, sys, urllib.request

new = sys.argv[1].lstrip("v")
path = pathlib.Path(__file__).resolve().parent.parent / "Formula" / "glyd.rb"
text = path.read_text()
old = re.search(r'^  version "([^"]+)"$', text, re.M).group(1)
text = text.replace(old, new)

def sha256(match):
    url = match.group(2)
    with urllib.request.urlopen(url) as r:
        digest = hashlib.sha256(r.read()).hexdigest()
    print(digest, url)
    return match.group(1) + digest

text = re.sub(r'(url "([^"]+)"\n\s+sha256 ")[0-9a-f]{64}', sha256, text)
path.write_text(text)
