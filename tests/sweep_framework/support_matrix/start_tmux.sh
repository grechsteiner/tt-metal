#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Start run_all.sh in a detached tmux session, so it keeps running after you log out or your laptop
# sleeps. The session stays open when the run ends so the summary can be read.
#   start_tmux.sh [session] [num_shards] [out_dir] [groups]
#   tmux attach -t <session>      # watch it; detach again with Ctrl-b d
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SESSION=${1:-support-matrix}
shift || true
: "${TT_METAL_SIMULATOR:?set TT_METAL_SIMULATOR to the path of libttsim.so}"
if tmux has-session -t "$SESSION" 2> /dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
fi
tmux new-session -d -s "$SESSION" \
    "export TT_METAL_SIMULATOR='$TT_METAL_SIMULATOR'; ulimit -t unlimited; '$HERE/run_all.sh' $*; echo; echo 'run finished; press Enter to close'; read"
echo "started tmux session '$SESSION' on $(hostname); attach with: tmux attach -t $SESSION"
