#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Build the support explorer page from results.jsonl: python render_page.py results.jsonl template.html out.html"""

import json
import re
import sys

CONFIGS = {
    "dram_interleaved": {"label": "DRAM interleaved", "short": "DRAM-I"},
    "l1_interleaved": {"label": "L1 interleaved", "short": "L1-I"},
    "l1_height_sharded": {"label": "L1 height-sharded", "short": "L1-HS"},
    "l1_width_sharded": {"label": "L1 width-sharded", "short": "L1-WS"},
    "l1_block_sharded": {"label": "L1 block-sharded", "short": "L1-BS"},
    "l1_nd_sharded": {"label": "L1 ND-sharded", "short": "L1-ND"},
    "dram_width_sharded": {"label": "DRAM width-sharded", "short": "DRAM-WS"},
    "dram_nd_sharded": {"label": "DRAM ND-sharded", "short": "DRAM-ND"},
    "default": {"label": "Op default", "short": "default"},
}
L1_BANK_KB = 1426.6  # 1460864 B per core, from the allocator's Out of Memory messages


def parse_id(node_id):
    """(group, layout, {role: config}) from ...[add-tile-a=dram_interleaved-b=...-out=default]."""
    group, layout, *pairs = node_id.split("[", 1)[1].rstrip("]").split("-")
    return group, layout, dict(pair.split("=", 1) for pair in pairs)


def fatal_summary(text):
    fatal = re.search(r"TT_(?:FATAL|THROW) @ \S*?([^/\s]+:\d+): ([^\n]*)", text)
    if not fatal:
        return None
    info = re.search(r"\ninfo:\s*\n([^\n]+)", text[fatal.end() :])
    return (info.group(1).strip() if info else fatal.group(2).strip()) + f" ({fatal.group(1)})"


def describe(record, labels):
    """(status, reason, detail) for one result."""
    message, error = record["message"], record["longrepr"]
    text = message + "\n" + error
    if record["outcome"] == "crashed":
        return "H", "Hung or crashed on the device", message
    if record["outcome"] == "passed":
        ulp = re.search(r"Max ULP Delta: ([\d.]+)", message)
        exact = re.search(r"exact (\d+)/(\d+)", message)
        reference = "its DRAM-interleaved output" if "vs the DRAM-interleaved" in message else "torch"
        if exact and exact.group(1) == exact.group(2):
            return "P", "Correct", f"bit-identical to {reference}"
        detail = f"max {float(ulp.group(1)):g} ULP from {reference}" if ulp else ""
        if exact:
            detail += f", {100 * int(exact.group(1)) / int(exact.group(2)):.1f}% bit-exact"
        return "P", "Correct", detail
    if "[WRONG_RESULT]" in text:
        exact = re.search(r"exact (\d+)/(\d+)", message)
        pcc = re.search(r"PCC (-?[\d.]+)", message)
        detail = ""
        if exact:
            detail = f"only {100 * int(exact.group(1)) / int(exact.group(2)):.1f}% of elements match"
        if pcc:
            detail += f", PCC {float(pcc.group(1)):.3f}"
        return "W", "Ran without error, output is wrong", detail
    if "[WRONG_CONFIG]" in text:
        return "C", "Output not in the requested config", message.split("] ", 1)[-1]
    oom = re.search(
        r"allocate (\d+) B (\w+) buffer across (\d+) banks, where each bank needs to store (\d+) B, but bank size is (\d+) B "
        r"\(allocated: (\d+) B, free: (\d+) B",
        text,
    )
    if oom:
        need, used = int(oom.group(4)) // 1024, int(oom.group(6)) // 1024
        setup_input = re.match(r"\[SETUP\]", message)
        if setup_input:
            what = "an input"
        elif "create_output_tensors" in error:
            what = "the output"
        elif "ntilize" in error:
            what = "a temporary row-major copy of the output"
        elif "ilize" in error:
            what = "a temporary tile copy of a row-major input"
        else:
            what = "a buffer"
        status = "S" if setup_input else "M"
        detail = f"{what} needs {need} KB on each of {oom.group(3)} cores; {used} KB already in use of {int(oom.group(5)) // 1024} KB"
        return status, "Out of L1", detail
    clash = re.search(r"buffers in program \d+ clash with L1 buffers on core range \[([^\]]+)\]", text)
    if clash:
        detail = f"the op's working buffers do not fit beside the tensors already in L1 on {clash.group(1)}"
        return ("S" if message.startswith("[SETUP]") else "M"), "Out of L1", detail
    cb = re.search(r"buffers on core range \[([^\]]+)\] grow to (\d+) B which is beyond max L1 size of (\d+) B", text)
    if cb:
        detail = f"the op's working buffers need {int(cb.group(2)) // 1024} KB per core on {cb.group(1)}, more than the {int(cb.group(3)) // 1024} KB of L1"
        return ("S" if message.startswith("[SETUP]") else "M"), "Out of L1", detail
    summary = fatal_summary(text)
    if message.startswith("[SETUP]"):
        return "S", "Input could not be created", summary or message[8:200]
    return "R", "Rejected by the op", summary or (error.strip().splitlines() or ["?"])[0][:200]


def main():
    results, template, out = sys.argv[1:4]
    records = {}
    for line in open(results):
        if line.strip():
            record = json.loads(line)
            records[record["nodeid"]] = record  # a vector recorded twice keeps its latest result
    groups, cases = {}, []
    for record in records.values():
        group, layout, roles = parse_id(record["nodeid"])
        info = record.get("info") or {}
        tensors = info.get("tensors") or {}
        meta = groups.setdefault(group, {"name": group, "title": info.get("title", group), "roles": {}, "layouts": []})
        if layout not in meta["layouts"]:
            meta["layouts"].append(layout)
        for role in roles:
            tensor = tensors.get(role, {})
            role_meta = meta["roles"].setdefault(role, {"key": role, "label": tensor.get("label", role), "shape": None})
            role_meta["shape"] = role_meta["shape"] or tensor.get("shape")
            if tensor.get("label"):
                role_meta["label"] = tensor["label"]
        status, reason, detail = describe(record, CONFIGS)
        cases.append(
            {
                "g": group,
                "l": layout,
                "c": roles,
                "s": status,
                "why": reason,
                "detail": detail,
                "chosen": info.get("chosen"),
                "kb": {role: t.get("kb") for role, t in tensors.items()},
                "cores": {role: t.get("cores") for role, t in tensors.items()},
            }
        )
    for meta in groups.values():
        meta["roles"] = list(meta["roles"].values())
        meta["layouts"].sort(key=["tile", "row_major"].index)
    data = {"configs": CONFIGS, "l1BankKb": L1_BANK_KB, "groups": list(groups.values()), "cases": cases}
    html = open(template).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open(out, "w").write(html)
    print(f"wrote {out} with {len(cases)} cases in {len(groups)} groups")


if __name__ == "__main__":
    main()
