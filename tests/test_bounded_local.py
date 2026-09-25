import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("bounded", Path(__file__).resolve().parents[1] / "scripts/run_bounded_local.py")
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


@unittest.skipUnless(os.name == "posix", "WSL/Linux process group supervisor")
class BoundedLocalTests(unittest.TestCase):
    def test_success_and_immutable_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            log, status = Path(d) / "run.log", Path(d) / "status.json"
            self.assertEqual(bounded.run([sys.executable, "-c", "print('done')"], 5, log, status), 0)
            self.assertEqual(json.loads(status.read_text())["state"], "complete")
            with self.assertRaises(FileExistsError):
                bounded.run([sys.executable, "-c", "pass"], 5, log, status)

    def test_budget_kills_term_ignoring_child(self):
        with tempfile.TemporaryDirectory() as d:
            log, status = Path(d) / "run.log", Path(d) / "status.json"
            cmd = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"]
            self.assertEqual(bounded.run(cmd, .4, log, status), 124)
            row = json.loads(status.read_text())
            self.assertEqual(row["state"], "time_budget_exhausted")
            with self.assertRaises(ProcessLookupError):
                os.kill(row["pid"], 0)

    def test_ten_hours_or_nan_cannot_be_requested(self):
        for seconds in [10 * 3600, float("nan"), -1]:
            with self.assertRaises(ValueError):
                bounded.run([sys.executable], seconds, "unused.log", "unused.json")

    def test_successful_parent_cannot_leave_running_grandchild(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            code = ("import subprocess,sys,pathlib; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                    "pathlib.Path(sys.argv[1]).write_text(str(p.pid))")
            result = bounded.run([sys.executable, "-c", code, str(root / "child.pid")],
                                 5, root / "run.log", root / "status.json")
            self.assertEqual(result, 0)
            child = int((root / "child.pid").read_text())
            stat = Path(f"/proc/{child}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().split(") ", 1)[1].split()[0], "Z")


if __name__ == "__main__":
    unittest.main()
