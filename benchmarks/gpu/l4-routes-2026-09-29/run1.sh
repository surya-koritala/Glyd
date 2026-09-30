#!/usr/bin/env bash
# the L4 route measurements on main (before the L4 routes), under the shared lock; the busy marker held unless another job holds it
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
flock $HOME/.glyd-box.lock env FILES=$HOME/l4routes R=$HOME/l4routes/results-main bash $HOME/l4routes/l4_routes.sh
