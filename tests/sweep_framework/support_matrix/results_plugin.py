# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: append one JSON line per finished test to $RESULTS_JSONL, flushed immediately,
so results survive the process being killed by a crash or hang."""

import json
import os


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        # The TT_FATAL text sits before the C++ backtrace, so keep the start of the report.
        error = ""
        if report.longrepr:
            error = getattr(getattr(report.longrepr, "reprcrash", None), "message", "") or ""
            error = error[:4000] + "\n----\n" + str(report.longrepr)[-2000:]
        record = {
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "message": dict(report.user_properties).get("message", ""),
            "longrepr": error,
            "duration": round(report.duration, 2),
        }
        with open(os.environ["RESULTS_JSONL"], "a") as f:
            f.write(json.dumps(record) + "\n")
