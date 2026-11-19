#!/usr/bin/env bash
# the worktree's gpu/ and glyd-gpu-py/ and its commit (or "+dirty") as f8f_src.tar
set -e
W=/Users/suryakoritala/glyd-fp8; S=/private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad/f8f
cd $W
echo "$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- gpu glyd-gpu-py || echo +dirty)" > $S/COMMIT
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' -cf $S/f8f_src.tar gpu glyd-gpu-py -C $S COMMIT
ls -l $S/f8f_src.tar
