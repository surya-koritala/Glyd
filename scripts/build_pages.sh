#!/bin/sh
# The whole Pages site, built: the root pages of docs/, the guide at main in guide/, and the guide of each release in guide/vX.Y.Z/.
#   scripts/build_pages.sh OUT_DIR        (needs git with the tags, python3 and mdbook 0.4.x)
# A release is a `v*` tag whose tree has docs/guide: it is built with that tag's own text and scripts, so a release's guide
# stays what it said then. Tags from before the guide are skipped. guide/versions.html lists what was built.
set -eu
[ $# -eq 1 ] || { echo "usage: scripts/build_pages.sh OUT_DIR" >&2; exit 2; }
cd "$(dirname "$0")/.."
out=$(mkdir -p "$1" && cd "$1" && pwd)

# the pages that were always here: the codec's page and what it uses
find docs -maxdepth 1 -name '*.html' -exec cp {} "$out"/ \;
cp -r docs/img "$out"/img
touch "$out/.nojekyll"

# the guide at main
scripts/build_site.sh "$out/guide"

# the guide at each release that has one
built=""
for tag in $(git tag -l 'v*' --sort=-v:refname); do
  git cat-file -e "$tag:docs/guide/book.toml" 2>/dev/null || continue
  work=$(mktemp -d)
  git worktree add --detach "$work/src" "$tag" > /dev/null
  (cd "$work/src" && sh scripts/build_site.sh "$out/guide/$tag")
  git worktree remove --force "$work/src"
  rm -rf "$work"
  built="$built $tag"
done

{
  echo '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Glyd guide versions</title>'
  echo '<h1>Glyd guide versions</h1><ul><li><a href="./">main</a> (the current text)</li>'
  for tag in $built; do echo "<li><a href=\"$tag/\">$tag</a></li>"; done
  echo '</ul>'
} > "$out/guide/versions.html"
echo "built: main$built"
