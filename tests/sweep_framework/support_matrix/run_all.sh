#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Run support-matrix groups on ttsim as N parallel shards (default 32), then build the explorer page.
# Needs TT_METAL_SIMULATOR set to libttsim.so. Builds the 12x10 runtime root on first use.
# Re-running with the same out_dir only runs vectors that have no result yet.
# Usage: run_all.sh [num_shards] [out_dir] [groups, comma-separated; default all]
set -u
# Each driver's pytest process runs ~80 threads and the simulator compiles kernels in parallel, so
# dozens of drivers exceed a typical 4096 per-user process limit (posix_spawn then fails with EAGAIN
# and every kernel build fails). Lift the soft limits up to the hard ones.
ulimit -u "$(ulimit -Hu)"
ulimit -t unlimited
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
N=${1:-32}
OUT=${2:-$HERE/runs}
OP_GROUPS=${3:-}
ROOT=${RUNTIME_ROOT:-$HERE/.runtime_root_12x10}
: "${TT_METAL_SIMULATOR:?set TT_METAL_SIMULATOR to the path of libttsim.so}"
[ -d "$ROOT" ] || "$HERE/make_runtime_root.sh" "$ROOT"
mkdir -p "$OUT"
cd "$REPO"
source python_env/bin/activate
IDS="$OUT/ids${OP_GROUPS:+.${OP_GROUPS//,/_}}.txt"
python "$HERE/run_matrix.py" --write-ids "$IDS" --groups "$OP_GROUPS" > /dev/null 2>&1 || {
    echo "could not list vectors for groups '$OP_GROUPS'" >&2
    exit 1
}
echo "$(date +%T) running $(wc -l < "$IDS") vectors with $N drivers into $OUT"
for ((i = 0; i < N; i++)); do
    python "$HERE/run_matrix.py" --out "$OUT" --ids-file "$IDS" --runtime-root "$ROOT" \
        --simulator "$TT_METAL_SIMULATOR" --shard "$i" --num-shards "$N" > "$OUT/driver$i.log" 2>&1 &
done
wait
cat "$OUT"/results*.jsonl > "$OUT/all_results.jsonl"
python3 "$HERE/render_page.py" "$OUT/all_results.jsonl" "$HERE/explorer_template.html" "$OUT/support_explorer.html"
echo "$(date +%T) done: $(wc -l < "$OUT/all_results.jsonl") results, page $OUT/support_explorer.html"
