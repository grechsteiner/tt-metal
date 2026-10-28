#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run support-matrix vectors on ttsim with a 12x10 grid, until each has a result.

First write the vector ids once with --write-ids (this imports ttnn to read the group registry),
then start any number of drivers with --ids-file and --shard/--num-shards. Each driver runs one
pytest process per (group, layout) chunk and appends results to its own results.jsonl as each test
finishes. If a process dies, or produces no result for --stall seconds (a hang that pytest-timeout
cannot interrupt because C++ holds the GIL), the test that was running is recorded as "crashed" and
the chunk resumes after it. Re-running skips everything already recorded, so it is safe to restart.
"""

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TOOLS, "..", "..", ".."))
TEST = "tests/ttnn/nightly/unit_tests/operations/test_op_support_matrix.py::test_support_matrix"


def write_ids(path, group_names):
    """Write one pytest node id per vector of the named groups (all groups if none are named)."""
    sys.path.insert(0, REPO)
    from tests.sweep_framework.support_matrix.groups import GROUPS

    # The linear groups take minutes per vector on the simulator, so start them first.
    names = group_names or sorted(GROUPS, key=lambda name: not name.startswith("linear_"))
    unknown = sorted(set(names) - set(GROUPS))
    if unknown:
        sys.exit(f"unknown groups {unknown}; known: {sorted(GROUPS)}")
    with open(path, "w") as f:
        for name in names:
            for vector in GROUPS[name].vectors():
                f.write(f"{TEST}[{GROUPS[name].vector_id(vector)}]\n")


def chunk_of(node_id):
    """(group, layout) of a node id such as ...[add-tile-a=...-b=...-out=...]."""
    group, layout = node_id.split("[", 1)[1].split("-")[:2]
    return group, layout


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
    parser.add_argument("--out")
    parser.add_argument("--write-ids", help="write the vector ids of --groups to this file and exit")
    parser.add_argument("--groups", default="", help="comma-separated group names for --write-ids (default all)")
    parser.add_argument("--ids-file", help="vector ids to run, from --write-ids")
    parser.add_argument("--simulator", default=os.environ.get("TT_METAL_SIMULATOR"), help="path to libttsim.so")
    parser.add_argument("--runtime-root", help="tt-metal root whose core descriptor gives 12x10")
    # pytest-timeout cannot stop C++ code holding the GIL and kills slow-but-healthy simulator runs, so
    # it is off by default; the --stall watchdog below handles hangs instead.
    parser.add_argument("--timeout", type=int, default=0, help="pytest per-test timeout in seconds (0: off)")
    parser.add_argument("--stall", type=int, default=120, help="kill pytest if no result lands for this many seconds")
    parser.add_argument(
        "--slow-stall",
        type=int,
        default=2400,
        help="--stall for groups named linear_*, whose vectors take minutes each on the simulator",
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="this driver runs vectors with index %% num-shards == shard"
    )
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.write_ids:
        write_ids(args.write_ids, [name for name in args.groups.split(",") if name])
        return
    if not (args.out and args.ids_file and args.runtime_root):
        parser.error("--out, --ids-file and --runtime-root are required to run vectors")
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
        # Baselines (an op's DRAM-interleaved output) are computed once per run and shared by every driver.
        SUPPORT_MATRIX_CACHE=os.path.join(args.out, "cache"),
        # Many drivers share the machine; torch would otherwise start a thread per core in each one.
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONPATH=TOOLS + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )

    with open(args.ids_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    chunks = {}
    for node_id in all_ids[args.shard :: args.num_shards]:
        chunks.setdefault(chunk_of(node_id), []).append(node_id)
    for (group, layout), ids in chunks.items():
        chunk = f"{group}_{layout}{tag}"
        stall = args.slow_stall if group.startswith("linear_") else args.stall
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
                    elif time.time() - last_progress > stall:
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
                message = f"hang: no result for {stall}s, killed" if stalled else f"crash: rc={rc}"
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
