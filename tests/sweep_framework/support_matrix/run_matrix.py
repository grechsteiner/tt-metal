#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run every vector of the ttnn.add support matrix on ttsim with a 12x10 grid, until each has a result.

One pytest process per (shape_kind, layout) chunk. Results are appended to <out>/results.jsonl as
each test finishes. If a process dies, or produces no result for --stall seconds (a hang that
pytest-timeout cannot interrupt because C++ holds the GIL), the test that was running is recorded
as "crashed" and the chunk resumes after it. Re-running the script skips everything already
recorded, so it is safe to restart.
"""

import argparse
import itertools
import json
import os
import resource
import signal
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TOOLS, "..", "..", ".."))
TEST = "tests/ttnn/nightly/unit_tests/operations/eltwise/test_add_support_matrix.py::test_add_support_matrix"

# Values copied from parameters["moonvit_add"] so the driver itself does not need to import ttnn.
SHAPES = ["full"]
LAYOUTS = ["tile", "row_major"]
CONFIGS = [
    "dram_interleaved",
    "l1_interleaved",
    "l1_height_sharded",
    "l1_width_sharded",
    "l1_block_sharded",
    "l1_nd_sharded",
    "dram_width_sharded",
    "dram_nd_sharded",
]
OUTS = ["default"] + CONFIGS


def node_id(shape, layout, a, b, out):
    return (
        f"{TEST}[shape_kind={shape}-layout={layout}-input_a_memory_config={a}"
        f"-input_b_memory_config={b}-out_memory_config={out}]"
    )


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    done[record["nodeid"]] = record
    return done


def load_all_done(out_dir):
    """Results from every shard (and from earlier unsharded runs) in out_dir."""
    done = {}
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("results") and name.endswith(".jsonl"):
            done.update(load_done(os.path.join(out_dir, name)))
    return done


def no_cpu_limit():
    resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--simulator", default=os.environ.get("TT_METAL_SIMULATOR"), help="path to libttsim.so")
    parser.add_argument("--runtime-root", required=True, help="tt-metal root whose core descriptor gives 12x10")
    parser.add_argument("--timeout", type=int, default=900, help="per-test timeout in seconds")
    parser.add_argument("--stall", type=int, default=60, help="kill pytest if no result lands for this many seconds")
    parser.add_argument(
        "--shard", type=int, default=0, help="this driver runs vectors with index %% num-shards == shard"
    )
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not args.simulator:
        parser.error("pass --simulator or set TT_METAL_SIMULATOR")

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.num_shards == 1 else f".shard{args.shard}"
    jsonl = os.path.join(args.out, f"results{tag}.jsonl")
    env = dict(
        os.environ,
        TT_METAL_SIMULATOR=args.simulator,
        TT_METAL_SLOW_DISPATCH_MODE="1",
        TT_METAL_DISABLE_SFPLOADMACRO="1",
        TT_METAL_RUNTIME_ROOT=args.runtime_root,
        RESULTS_JSONL=jsonl,
        # Many drivers share the machine; torch would otherwise start a thread per core in each one.
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONPATH=TOOLS + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )

    for shape, layout in itertools.product(SHAPES, LAYOUTS):
        chunk = f"{shape}_{layout}{tag}"
        ids = [node_id(shape, layout, a, b, o) for a, b, o in itertools.product(CONFIGS, CONFIGS, OUTS)]
        ids = ids[args.shard :: args.num_shards]
        attempt = 0
        while True:
            done = load_all_done(args.out)
            remaining = [i for i in ids if i not in done]
            if not remaining:
                break
            attempt += 1
            args_file = os.path.join(args.out, f"{chunk}.args")
            with open(args_file, "w") as f:
                f.write("\n".join(remaining) + "\n")
            log_path = os.path.join(args.out, f"{chunk}.attempt{attempt}.log")
            start = time.time()
            stalled = False
            with open(log_path, "w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        f"@{args_file}",
                        "-p",
                        "results_plugin",
                        "-p",
                        "no:cacheprovider",
                        f"--timeout={args.timeout}",
                        "--timeout-method=thread",
                        "-q",
                    ],
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=no_cpu_limit,
                    start_new_session=True,
                )
                last_count, last_progress = len(load_done(jsonl)), time.time()
                while process.poll() is None:
                    time.sleep(5)
                    count = len(load_done(jsonl))
                    if count != last_count:
                        last_count, last_progress = count, time.time()
                    elif time.time() - last_progress > args.stall:
                        stalled = True
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                rc = process.returncode
            done = load_all_done(args.out)
            left = [i for i in ids if i not in done]
            print(
                f"{chunk} attempt {attempt}: rc={rc} {len(remaining) - len(left)} recorded, {len(left)} left, "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
            if left:
                if len(left) == len(remaining) and attempt > 1 and rc in (2, 3, 4, 5):
                    sys.exit(f"{chunk}: pytest made no progress (rc={rc}); see {log_path}")
                # pytest runs the ids in the order given, so the first unrecorded one was running when it died.
                with open(log_path) as log:
                    tail = log.read()[-3000:]
                message = f"hang: no result for {args.stall}s, killed" if stalled else f"crash: rc={rc}"
                with open(jsonl, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "nodeid": left[0],
                                "outcome": "crashed",
                                "message": message,
                                "longrepr": tail,
                                "duration": None,
                            }
                        )
                        + "\n"
                    )
                print(f"{chunk}: recorded {message} for {left[0].split('[')[1]}", flush=True)

    print(
        f"shard {args.shard}/{args.num_shards} finished; {len(load_all_done(args.out))} vectors recorded in {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
