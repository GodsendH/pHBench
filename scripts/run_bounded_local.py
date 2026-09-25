"""Run a POSIX process group with an enforced local wall-clock budget.

The limit includes startup and child processes. This is a foreground supervisor;
use it from WSL, and keep the log/status files as experiment provenance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def run(command, seconds, log, status):
    if os.name != "posix":
        raise RuntimeError("Run this supervisor inside WSL/Linux.")
    if not command or not math.isfinite(seconds) or not 0 < seconds <= 9.5 * 3600:
        raise ValueError("Local budget must be positive and at most 9.5 hours.")
    log, status = Path(log), Path(status)
    log.parent.mkdir(parents=True, exist_ok=True)
    status.parent.mkdir(parents=True, exist_ok=True)
    if status.exists() or log.exists():
        raise FileExistsError("Use new log/status paths for every supervised attempt.")
    start = time.monotonic()
    row = {"command": command, "started_unix": time.time(), "limit_seconds": seconds,
           "supervisor_pid": os.getpid(), "host_policy": "local_max_9.5_hours"}

    def save(state, **extra):
        row.update(state=state, elapsed_seconds=time.monotonic() - start, **extra)
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps(row, indent=2) + "\n")
        temp.replace(status)

    def terminate_group(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        # Kill descendants too, even when the group leader exits first.
        time.sleep(1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    with log.open("x") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        save("running", pid=process.pid)
        previous_term = signal.getsignal(signal.SIGTERM)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"supervisor received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            code = process.wait(timeout=max(.001, seconds - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            terminate_group(process.pid)
            process.wait()
            save("time_budget_exhausted", returncode=process.returncode)
            return 124
        except BaseException:
            terminate_group(process.pid)
            process.wait()
            save("interrupted", returncode=process.returncode)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        # A successful leader must not leave orphan workers running indefinitely.
        terminate_group(process.pid)
        save("complete" if code == 0 else "failed", returncode=code)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--log", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(run(command, args.hours * 3600, args.log, args.status))


if __name__ == "__main__":
    main()
