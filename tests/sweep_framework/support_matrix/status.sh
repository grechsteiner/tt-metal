#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Progress of a support-matrix run: results per group and outcome, hangs so far, and an ETA from the
# last 10 minutes. Usage: status.sh <out_dir>
set -u
OUT=${1:?usage: status.sh <out_dir>}
python3 - "$OUT" << 'EOF'
import collections, glob, json, os, sys, time

out = sys.argv[1]
ids = sorted(glob.glob(os.path.join(out, "ids*.txt")), key=os.path.getmtime)
total = sum(1 for _ in open(ids[-1])) if ids else None
records, times = {}, []
for path in glob.glob(os.path.join(out, "results*.jsonl")):
    if os.path.basename(path) == "all_results.jsonl":
        continue
    for line in open(path):
        if line.strip():
            record = json.loads(line)
            records[record["nodeid"]] = record
    times.append(os.path.getmtime(path))
by_group = collections.defaultdict(collections.Counter)
for node_id, record in records.items():
    by_group[node_id.split("[", 1)[1].split("-")[0]][record["outcome"]] += 1
print(f"{time.strftime('%H:%M:%S')}  {len(records)}/{total or '?'} vectors recorded")
for group, counts in by_group.items():
    print(f"  {group:14} {sum(counts.values()):5}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
hangs = [r for r in records.values() if r["outcome"] == "crashed"]
print(f"  hangs/crashes: {len(hangs)}")
for record in hangs[-5:]:
    print(f"    {record['message']}: {record['nodeid'].split('[', 1)[1][:-1]}")
EOF
echo "  test processes running: $(pgrep -fc "pytest @$OUT")"
