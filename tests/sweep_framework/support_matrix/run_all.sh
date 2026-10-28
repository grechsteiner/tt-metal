#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Run the add support matrix on ttsim as N parallel shards (default 32), then build the explorer page.
# Needs TT_METAL_SIMULATOR set to libttsim.so. Builds the 12x10 runtime root on first use.
# Usage: run_all.sh [num_shards] [out_dir]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
N=${1:-32}
OUT=${2:-$HERE/runs}
ROOT=${RUNTIME_ROOT:-$HERE/.runtime_root_12x10}
: "${TT_METAL_SIMULATOR:?set TT_METAL_SIMULATOR to the path of libttsim.so}"
[ -d "$ROOT" ] || "$HERE/make_runtime_root.sh" "$ROOT"
mkdir -p "$OUT"
cd "$REPO"
source python_env/bin/activate
for ((i = 0; i < N; i++)); do
    python "$HERE/run_matrix.py" --out "$OUT" --runtime-root "$ROOT" --simulator "$TT_METAL_SIMULATOR" \
        --shard "$i" --num-shards "$N" > "$OUT/driver$i.log" 2>&1 &
done
wait
cat "$OUT"/results*.jsonl > "$OUT/all_results.jsonl"
python3 "$HERE/render_page.py" "$OUT/all_results.jsonl" "$HERE/explorer_template.html" "$OUT/add_explorer.html"
