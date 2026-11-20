#!/usr/bin/env python3
"""Give the files of a fresh checkout the modification times they had when their content last changed.

Cargo's freshness check compares source files' modification times with its build products', so a checkout (every file new) rebuilds
the whole workspace and reruns every build script, whatever changed. A CI runner that keeps its target directory calls this
after the checkout with a file of its own beside that directory: a file whose content (git's blob id) is the one the file
remembers gets its old time back (older than the build that used it), any other keeps its new time and is remembered. So only what
changed since the last build is built again.

    python3 scripts/ci_restore_mtimes.py /cache/mtimes-sm89.json
"""
import json
import os
import subprocess
import sys

manifest = sys.argv[1]
try:
    with open(manifest) as f:
        old = json.load(f)
except (OSError, ValueError):
    old = {}
listing = subprocess.run(["git", "-c", "safe.directory=*", "ls-files", "-s", "-z"], check=True, capture_output=True).stdout
new = {}
restored = 0
for entry in listing.split(b"\0"):
    if not entry:
        continue
    meta, _, path = entry.partition(b"\t")
    path = os.fsdecode(path)
    blob = meta.split()[1].decode()
    try:
        st = os.lstat(path)
    except OSError:
        continue
    if os.path.islink(path):
        continue
    kept = old.get(path)
    if kept and kept[0] == blob:
        os.utime(path, (kept[1], kept[1]))
        new[path] = kept
        restored += 1
    else:
        new[path] = [blob, int(st.st_mtime)]
with open(manifest + ".tmp", "w") as f:
    json.dump(new, f)
os.replace(manifest + ".tmp", manifest)
print(f"mtimes: {restored} of {len(new)} files back to their old times, {len(new) - restored} new or changed")
