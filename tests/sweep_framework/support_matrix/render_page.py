#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Build the add support explorer page from results.jsonl: python render_page.py results.jsonl template.html out.html"""

import json
import re
import sys

ID = re.compile(
    r"shape_kind=(\w+)-layout=(\w+)-input_a_memory_config=(\w+)-input_b_memory_config=(\w+)-out_memory_config=(\w+)\]"
)

# L1 per core for each config at [16384, 1024] bf16 on the 12x10 grid; these are the per-bank sizes
# the allocator reports in its Out of Memory messages. DRAM configs use no L1.
CONFIGS = {
    "dram_interleaved": {"label": "DRAM interleaved", "short": "DRAM-I", "l1_kb": 0, "where": "DRAM, 8 banks"},
    "l1_interleaved": {
        "label": "L1 interleaved",
        "short": "L1-I",
        "l1_kb": 274,
        "where": "120 cores, pages round-robin (allocator-controlled, 136-137 tiles each)",
    },
    "l1_height_sharded": {
        "label": "L1 height-sharded",
        "short": "L1-HS",
        "l1_kb": 512,
        "where": "64 cores, one 256 x 1024 shard each",
    },
    "l1_width_sharded": {
        "label": "L1 width-sharded",
        "short": "L1-WS",
        "l1_kb": 1024,
        "where": "32 cores, one 16384 x 32 shard each",
    },
    "l1_block_sharded": {
        "label": "L1 block-sharded",
        "short": "L1-BS",
        "l1_kb": 512,
        "where": "8 x 8 cores, one 2048 x 128 shard each",
    },
    "l1_nd_sharded": {
        "label": "L1 ND-sharded",
        "short": "L1-ND",
        "l1_kb": 512,
        "where": "64 cores, four 256 x 256 shards each, round-robin",
    },
    "dram_width_sharded": {
        "label": "DRAM width-sharded",
        "short": "DRAM-WS",
        "l1_kb": 0,
        "where": "8 DRAM banks, one 16384 x 128 shard each",
    },
    "dram_nd_sharded": {
        "label": "DRAM ND-sharded",
        "short": "DRAM-ND",
        "l1_kb": 0,
        "where": "8 DRAM banks, 32 256 x 256 shards each, round-robin",
    },
    "default": {"label": "Op default", "short": "default", "l1_kb": None, "where": "memory_config=None, the op picks"},
}
L1_BANK_KB = 1426.6  # 1460864 B per core, from the allocator messages

LAYOUT_NAMES = {
    ("BufferType.DRAM", "TensorMemoryLayout.INTERLEAVED"): "dram_interleaved",
    ("BufferType.L1", "TensorMemoryLayout.INTERLEAVED"): "l1_interleaved",
    ("BufferType.L1", "TensorMemoryLayout.HEIGHT_SHARDED"): "l1_height_sharded",
    ("BufferType.L1", "TensorMemoryLayout.WIDTH_SHARDED"): "l1_width_sharded",
    ("BufferType.L1", "TensorMemoryLayout.BLOCK_SHARDED"): "l1_block_sharded",
    ("BufferType.DRAM", "TensorMemoryLayout.WIDTH_SHARDED"): "dram_width_sharded",
    ("BufferType.L1", "TensorMemoryLayout.ND_SHARDED"): "l1_nd_sharded",
    ("BufferType.DRAM", "TensorMemoryLayout.ND_SHARDED"): "dram_nd_sharded",
}


def fatal_summary(text):
    fatal = re.search(r"TT_(?:FATAL|THROW) @ \S*?([^/\s]+:\d+): ([^\n]*)", text)
    if not fatal:
        return None
    info = re.search(r"\ninfo:\s*\n([^\n]+)", text[fatal.end() :])
    return (info.group(1).strip() if info else fatal.group(2).strip()) + f" ({fatal.group(1)})"


def describe(record):
    """(status, reason, detail, chosen_output) for one result."""
    message, error = record["message"], record["longrepr"]
    text = message + "\n" + error
    if record["outcome"] == "crashed":
        return "H", "Hung or crashed on the device", message, None
    if record["outcome"] == "passed":
        ulp = re.search(r"Max ULP Delta: ([\d.]+)", message)
        exact = re.search(r"exact (\d+)/(\d+)", message)
        out = re.search(r"out=(BufferType\.\w+)/(TensorMemoryLayout\.\w+)", message)
        chosen = LAYOUT_NAMES.get((out.group(1), out.group(2))) if out else None
        detail = f"max {float(ulp.group(1)):g} ULP from torch.add" if ulp else ""
        if exact:
            detail += f", {100 * int(exact.group(1)) / int(exact.group(2)):.1f}% bit-exact"
        return "P", "Correct", detail, chosen
    if "[WRONG_RESULT]" in text:
        exact = re.search(r"exact (\d+)/(\d+)", message)
        pcc = re.search(r"PCC (-?[\d.]+)", message)
        detail = ""
        if exact:
            detail = f"only {100 * int(exact.group(1)) / int(exact.group(2)):.1f}% of elements match torch.add"
        if pcc:
            detail += f", PCC {float(pcc.group(1)):.3f}"
        return "W", "Ran without error, output is wrong", detail, None
    if "[WRONG_CONFIG]" in text:
        return "C", "Output not in the requested config", message.split("] ", 1)[-1], None
    oom = re.search(
        r"allocate (\d+) B (\w+) buffer across (\d+) banks, where each bank needs to store (\d+) B, but bank size is (\d+) B "
        r"\(allocated: (\d+) B, free: (\d+) B",
        text,
    )
    if oom:
        need, used = int(oom.group(4)) // 1024, int(oom.group(6)) // 1024
        if message.startswith("[SETUP]"):
            what = "input B"
        elif "create_output_tensors" in error:
            what = "the output"
        elif "Tilize" in error or "tilize" in error:
            what = "a temporary tile copy of a row-major input"
        elif "ntilize" in error:
            what = "a temporary row-major copy of the output"
        else:
            what = "a buffer"
        status = "S" if message.startswith("[SETUP]") else "M"
        return (
            status,
            "Out of L1",
            f"{what} needs {need} KB on each of {oom.group(3)} cores; {used} KB already in use of {int(oom.group(5)) // 1024} KB",
            None,
        )
    summary = fatal_summary(text)
    if message.startswith("[SETUP]"):
        return "S", "Input could not be created", summary or message[8:200], None
    return "R", "Rejected by the op", summary or (error.strip().splitlines() or ["?"])[0][:200], None


def main():
    results, template, out = sys.argv[1:4]
    records = {}
    for line in open(results):
        if line.strip():
            record = json.loads(line)
            records[record["nodeid"]] = record  # a vector recorded twice keeps its latest result
    cases = []
    for record in records.values():
        shape, layout, a, b, o = ID.search(record["nodeid"]).groups()
        status, reason, detail, chosen = describe(record)
        cases.append(
            {"l": layout, "a": a, "b": b, "o": o, "s": status, "why": reason, "detail": detail, "chosen": chosen}
        )
    data = {"configs": CONFIGS, "l1BankKb": L1_BANK_KB, "cases": cases}
    html = open(template).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open(out, "w").write(html)
    print(f"wrote {out} with {len(cases)} cases")


if __name__ == "__main__":
    main()
