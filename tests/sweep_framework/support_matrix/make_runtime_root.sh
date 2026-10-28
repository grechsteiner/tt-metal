#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Build a TT_METAL_RUNTIME_ROOT that gives ttsim a 12x10 compute grid on Blackhole.
#
# On silicon, slow dispatch already uses the full 12x10 logical grid of a 2-column-harvested
# Blackhole, but on the simulator tt-metal falls back to the core descriptor, which reserves a
# column for dispatch (11x10). This mirrors the repo with symlinks and replaces only
# tt_metal/core_descriptors, with the 2xharvested compute range widened from [10, 9] to [11, 9].
#
# Usage: make_runtime_root.sh <out_dir>
set -eu
OUT=$1
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
rm -rf "$OUT"
mkdir -p "$OUT/tt_metal"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
    [ "$(basename "$entry")" = tt_metal ] || ln -s "$entry" "$OUT/$(basename "$entry")"
done
for entry in "$REPO"/tt_metal/* "$REPO"/tt_metal/.[!.]*; do
    [ -e "$entry" ] || continue
    [ "$(basename "$entry")" = core_descriptors ] || ln -s "$entry" "$OUT/tt_metal/$(basename "$entry")"
done
cp -r "$REPO/tt_metal/core_descriptors" "$OUT/tt_metal/"
python3 - "$OUT/tt_metal/core_descriptors/blackhole_140_arch.yaml" <<'EOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()
head, marker, rest = text.partition("2xharvested:")
end = re.search(r"\n\S", rest)
block, tail = (rest[: end.start()], rest[end.start() :]) if end else (rest, "")
widened = block.replace("end: [10, 9]", "end: [11, 9]")
assert widened.count("end: [11, 9]") == 2, "unexpected core descriptor layout"
open(path, "w").write(head + marker + widened + tail)
EOF
echo "12x10 runtime root ready: $OUT"
