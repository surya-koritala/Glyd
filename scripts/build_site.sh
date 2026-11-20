#!/bin/sh
# The user guide, built: the models page from the code, then mdBook.
#   scripts/build_site.sh OUT_DIR        (needs python3 and mdbook 0.4.x on the PATH)
# The Pages workflow (.github/workflows/pages.yml) runs this for main and for each release tag.
set -eu
[ $# -eq 1 ] || { echo "usage: scripts/build_site.sh OUT_DIR" >&2; exit 2; }
cd "$(dirname "$0")/.."
out=$(mkdir -p "$1" && cd "$1" && pwd)
python3 -I scripts/gen_models.py
mdbook build docs/guide -d "$out"
