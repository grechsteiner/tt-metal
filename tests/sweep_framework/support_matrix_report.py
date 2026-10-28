# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Turn sweep results (results_export/*.json) into a support matrix.

Each vector is classified as one of:
    P  pass
    R  rejected by the op (exception raised by the op, usually a TT_FATAL from validation)
    S  setup failed (input tensor could not be built; run() tagged it [SETUP])
    W  wrong result (op ran, PCC failed; run() tagged it [WRONG_RESULT])
    C  wrong config (op ran and was correct, but ignored the requested output config; tagged [WRONG_CONFIG])
    M  out of L1 memory
    H  hang / crash
    -  not run

Example:
    python tests/sweep_framework/support_matrix_report.py \
        --module eltwise.binary.add.add_support_matrix \
        --rows input_a_memory_config,input_b_memory_config \
        --cols out_memory_config \
        --facets shape_kind,layout
"""

import argparse
import collections
import glob
import itertools
import json
import os
import re

STATUS_CODES = {
    "pass": "P",
    "success": "P",
    "fail_l1_out_of_mem": "M",
    "fail_crash_hang": "H",
    "skipped": "-",
}


def load_records(results_dir, module):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("oprun_"):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for record in data if isinstance(data, list) else []:
            if module and record.get("full_test_name") != module:
                continue
            params = {}
            for p in record.get("op_params_set") or []:
                for key in ("param_value_text", "param_value_numeric", "param_value_json"):
                    if p.get(key) is not None:
                        params[p["param_name"]] = p[key]
                        break
            records.append((params, record))
    return records


def classify(record):
    status = record.get("status")
    if status in STATUS_CODES:
        return STATUS_CODES[status]
    text = record.get("exception") or record.get("message") or ""
    if text.startswith("[SETUP]"):
        return "S"
    if text.startswith("[WRONG_RESULT]"):
        return "W"
    if text.startswith("[WRONG_CONFIG]"):
        return "C"
    return "R"


def failure_reason(record):
    """Short, groupable reason: the TT_FATAL location plus its message, or the first exception line."""
    text = record.get("exception") or record.get("message") or ""
    tag = ""
    for prefix in ("[SETUP] ", "[WRONG_RESULT] ", "[WRONG_CONFIG] "):
        if text.startswith(prefix):
            tag, text = prefix.strip() + " ", text[len(prefix) :]
    location = re.search(r"TT_(?:FATAL|THROW) @ \S*?([^/\s]+:\d+)", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    message = lines[0] if lines else "(no message)"
    if "info:" in lines:
        after = lines[lines.index("info:") + 1 :]
        message = after[0] if after else message
    reason = f"{location.group(1)}: {message}" if location else message
    return tag + reason[:200]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="tests/sweep_framework/results_export")
    parser.add_argument("--module", help="Sweep module name; default is every module in the results")
    parser.add_argument("--rows", required=True, help="Comma-separated params that form the table rows")
    parser.add_argument("--cols", required=True, help="Param that forms the table columns")
    parser.add_argument("--facets", default="", help="Comma-separated params; one table per combination")
    args = parser.parse_args()

    rows, facets = args.rows.split(","), [f for f in args.facets.split(",") if f]
    records = load_records(args.results_dir, args.module)
    if not records:
        raise SystemExit(f"No results found in {args.results_dir} for module {args.module}")

    def values(name):
        return sorted({str(p.get(name)) for p, _ in records})

    cells = {}
    reasons = collections.defaultdict(list)
    for params, record in records:
        key = tuple(str(params.get(n)) for n in facets + rows + [args.cols])
        code = classify(record)
        cells[key] = code
        if code not in ("P", "-"):
            reasons[failure_reason(record)].append(params)

    totals = collections.Counter(cells.values())
    print(f"# Support matrix: {args.module or 'all modules'}\n")
    print("Totals: " + ", ".join(f"{code}={n}" for code, n in sorted(totals.items())) + "\n")
    print(
        "Legend: P pass, R rejected by op, S setup failed, W wrong result, C wrong config, M L1 OOM, H hang, - not run\n"
    )

    cols = values(args.cols)
    for facet in itertools.product(*(values(f) for f in facets)):
        if facets:
            print("## " + ", ".join(f"{n}={v}" for n, v in zip(facets, facet)) + "\n")
        print("| " + " / ".join(rows) + " | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for row in itertools.product(*(values(r) for r in rows)):
            line = [cells.get(facet + row + (c,), " ") for c in cols]
            if any(code != " " for code in line):
                print("| " + " / ".join(row) + " | " + " | ".join(line) + " |")
        print()

    print("## Failure reasons (most common first)\n")
    for reason, hits in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        example = ", ".join(f"{k}={v}" for k, v in hits[0].items() if k in facets + rows + [args.cols])
        print(f"- **{len(hits)}x** {reason}\n  - e.g. {example}")


if __name__ == "__main__":
    main()
