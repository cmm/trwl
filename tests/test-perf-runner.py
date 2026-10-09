#!/usr/bin/env python3
"""Check benchmark timing and reporting; these are not trwl behavior cases."""

import argparse
import copy
import gzip
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True

import perf
import perf_support
from e2e_support import SCRATCH
from perf_support import PerformanceCase


class CancelledForTest(BaseException):
    pass


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="trwl-perf-unit-", dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.case = PerformanceCase(Path(self.temp.name) / "case", Path(sys.executable))
        self.addCleanup(self.case.close)

    def test_measured_child_and_full_compressed_output(self):
        result, record = self.case.measured(
            "unit", "-c", "import time; time.sleep(0.03); print('complete output')"
        )
        self.assertEqual(result.stdout, "complete output\n")
        self.assertEqual(record["returncode"], 0)
        self.assertGreaterEqual(record["wall_seconds"], 0.03)
        self.assertGreater(record["cpu_user_seconds"] + record["cpu_system_seconds"], 0)
        self.assertGreater(record["max_rss_kib"], 0)
        with gzip.open(record["stdout_file"], "rt") as stream:
            self.assertEqual(stream.read(), result.stdout)

    def test_cleanup_is_outside_timing(self):
        stop = perf_support.stop_process_group

        def slow_cleanup(process):
            time.sleep(0.15)
            stop(process)

        with patch("perf_support.stop_process_group", side_effect=slow_cleanup):
            started = time.perf_counter()
            _, record = self.case.measured("unit", "-c", "print('ok')")
            overall = time.perf_counter() - started
        self.assertGreaterEqual(overall - record["wall_seconds"], 0.15)

    def test_block_io_includes_waited_descendant_writes(self):
        result, record = self.case.measured(
            "io",
            "-c",
            "import subprocess, sys; subprocess.run([sys.executable, '-c', "
            '"import os; from pathlib import Path; '
            "p = Path(os.environ['TMPDIR']) / 'write.bin'; "
            "f = p.open('wb'); f.write(b'x' * (4 * 1024 * 1024)); "
            "f.flush(); os.fsync(f.fileno()); "
            "print(next(line.split()[1] for line in "
            "Path('/proc/self/io').read_text().splitlines() "
            "if line.startswith('write_bytes:')))\"], check=True)",
        )
        written = int(result.stdout)
        if written == 0:
            self.skipTest("filesystem does not account storage writes")
        self.assertGreaterEqual(record["filesystem_output_bytes"], written)

    def test_rss_excludes_runner_allocation_and_tracks_child_allocation(self):
        def measure(mebibytes):
            result, record = self.case.measured(
                "memory",
                "-c",
                "from pathlib import Path; "
                f"memory = bytearray({mebibytes} * 1024 * 1024); "
                "print(next(line.split()[1] for line in "
                "Path('/proc/self/status').read_text().splitlines() "
                "if line.startswith('VmHWM:')))",
            )
            # Allow interpreter teardown and kernel RSS-accounting granularity.
            self.assertLess(abs(record["max_rss_kib"] - int(result.stdout)), 8192)
            return record["max_rss_kib"]

        before = measure(0)
        # The local reference keeps this allocation alive through all measurements.
        _runner_memory = bytearray(128 * 1024 * 1024)
        after = measure(0)
        self.assertLess(abs(after - before), 8192)
        child_allocation = measure(64)
        self.assertGreater(child_allocation - after, 48 * 1024)

    def test_missing_or_invalid_rss_report_is_not_a_measurement_success(self):
        for value in ("-1", "invalid", FileNotFoundError("missing RSS report")):
            with self.subTest(value=value):
                with (
                    patch.object(
                        Path,
                        "read_text",
                        side_effect=value if isinstance(value, Exception) else None,
                        return_value=value,
                    ),
                    self.assertRaises((AssertionError, ValueError, FileNotFoundError)),
                ):
                    self.case.measured("unit", "-c", "print('ok')")
                record = json.loads(self.case.commands[-1].log.read_text())
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["returncode"], 0)

    def test_timeout_reaps_child_and_records_failure(self):
        with self.assertRaisesRegex(AssertionError, "timed out"):
            self.case.measured(
                "unit", "-c", "import time; time.sleep(30)", timeout=0.05
            )
        command = self.case.commands[-1]
        record = json.loads(command.log.read_text())
        self.assertTrue(record["timed_out"])
        self.assertEqual(record["status"], "timeout")
        self.assertIsNotNone(record["returncode"])
        with self.assertRaises(ProcessLookupError):
            os.kill(command.process.pid, 0)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_fast_nonzero_exit_is_not_a_measurement_success(self):
        with self.assertRaisesRegex(AssertionError, "measured command failed"):
            self.case.measured("unit", "-c", "raise SystemExit(7)")
        record = json.loads(self.case.commands[-1].log.read_text())
        self.assertEqual(record["returncode"], 7)
        self.assertFalse(record["timed_out"])

    def test_parent_cancellation_reaps_child(self):
        def cancel(_sig, _frame):
            raise CancelledForTest()

        old = signal.signal(signal.SIGTERM, cancel)
        try:
            with self.assertRaises(CancelledForTest):
                self.case.measured(
                    "unit",
                    "-c",
                    f"import os,signal,time; os.kill({os.getpid()}, signal.SIGTERM); time.sleep(30)",
                )
        finally:
            signal.signal(signal.SIGTERM, old)
        command = self.case.commands[-1]
        self.assertEqual(command.record["status"], "interrupted")
        with self.assertRaises(ProcessLookupError):
            os.kill(command.process.pid, 0)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            binary=None,
            sizes=[100, 1000],
            ratio=3.0,
            import_slack=0.25,
            read_slack=0.05,
            max_relative_spread=0.5,
        )
        self.report = {
            "providers": ["opencode"],
            "shapes": ["long"],
            "scenarios": {},
            "workloads": list(perf.WORKLOADS),
        }
        for size in self.args.sizes:
            pairs = []
            for repetition in range(-1, 5):
                run = {
                    "result": "passed",
                    "measurements": {
                        name: {"wall_seconds": 1000.0 if repetition == -1 else 1.0}
                        for name in perf.WORKLOADS
                    },
                }
                pairs.append(
                    {
                        "warmup": repetition == -1,
                        "runs": {
                            "baseline": copy.deepcopy(run),
                            "candidate": copy.deepcopy(run),
                        },
                    }
                )
            self.report["scenarios"][f"opencode-long-{size}"] = {
                **perf.scenario_spec("import", "opencode", "long", size, 256),
                "pairs": pairs,
            }

    def test_warmup_is_not_in_comparison(self):
        perf.summarize(self.report, self.args)
        self.assertEqual(self.report["result"], "passed")
        for comparison in self.report["comparisons"].values():
            self.assertEqual(comparison["baseline_samples"], [1.0] * 5)
            self.assertEqual(comparison["candidate_samples"], [1.0] * 5)

    def test_failed_samples_keep_pair_positions_and_prevent_a_pass(self):
        pairs = self.report["scenarios"]["opencode-long-100"]["pairs"]
        pairs[1]["runs"]["baseline"]["result"] = "failed"
        pairs[2]["runs"]["candidate"]["result"] = "failed"
        perf.summarize(self.report, self.args)
        comparison = self.report["comparisons"]["opencode-long-100/initial-import"]
        self.assertEqual(comparison["baseline_samples"], [None, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(comparison["candidate_samples"], [1.0, None, 1.0, 1.0, 1.0])
        self.assertEqual(comparison["status"], "inconclusive")
        self.assertEqual(self.report["result"], "failed")

    def test_missing_candidate_is_inconclusive(self):
        pairs = self.report["scenarios"]["opencode-long-100"]["pairs"]
        del pairs[1]["runs"]["candidate"]
        perf.summarize(self.report, self.args)
        self.assertEqual(self.report["result"], "inconclusive")

    def test_slow_candidate_propagates_regression(self):
        for scenario in self.report["scenarios"].values():
            for pair in scenario["pairs"]:
                pair["runs"]["candidate"]["measurements"]["initial-import"][
                    "wall_seconds"
                ] = 4.0
        perf.summarize(self.report, self.args)
        self.assertEqual(self.report["result"], "regression")

    def test_selected_semantic_workload_is_compared(self):
        for scenario in self.report["scenarios"].values():
            scenario["workloads"] = ["semantic-index"]
            for pair in scenario["pairs"]:
                for side, run in pair["runs"].items():
                    run["measurements"]["semantic-index"] = {
                        "wall_seconds": 4.0 if side == "candidate" else 1.0
                    }
        perf.summarize(self.report, self.args)
        self.assertEqual(self.report["result"], "regression")
        self.assertEqual(
            self.report["comparisons"]["opencode-long-100/semantic-index"]["status"],
            "regression",
        )


if __name__ == "__main__":
    unittest.main()
