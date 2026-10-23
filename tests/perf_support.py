"""Foreground process measurements; fixture creation and log processing are untimed."""

import gzip
import hashlib
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from e2e_support import (
    Case,
    RunningCommand,
    require,
    sha256,
    stop_process_group,
    write_json,
)


def trial_order(sides, repetition):
    sides = tuple(sides)
    return sides if repetition % 2 == 0 else sides[::-1]


@contextmanager
def benchmark_report(root, binaries, *, model_dir=None, **metadata):
    cases = {}
    report = {"result": "failed", "binaries": {}, "workloads": {}, **metadata}
    try:
        for side, binary in binaries.items():
            binary = binary.resolve(strict=True)
            report["binaries"][side] = {"path": str(binary), "sha256": sha256(binary)}
            cases[side] = PerformanceCase(root / side, binary, model_dir)
        yield cases, report
        for side, case in cases.items():
            require(
                sha256(case.binary) == report["binaries"][side]["sha256"],
                f"{side} executable changed during run",
            )
        report["result"] = "passed"
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        errors = []
        for case in cases.values():
            try:
                case.close()
            except Exception as exc:  # noqa: BLE001 - close every case and report cleanup failures
                errors.append(str(exc))
        if errors:
            report.update(result="failed", cleanup_errors=errors)
        write_json(root / "report.json", report)
        print(root / "report.json", flush=True)
        if errors and "error" not in report:
            raise RuntimeError("; ".join(errors))


def measure_pairs(cases, repetitions, measure, record):
    """Measure alternating trials and reject changes in normalized output."""
    require(repetitions > 0, "repetitions must be positive")
    record["samples"] = samples = {side: [] for side in cases}
    expected = None
    for repetition in range(-1, repetitions):
        for side in trial_order(cases, repetition):
            output, sample = measure(cases[side], repetition)
            digest = hashlib.sha256(output.encode()).hexdigest()
            if expected is None:
                record["output_sha256"] = expected = digest
            require(digest == expected, "paired output differs")
            samples[side].append({"warmup": repetition < 0, **sample})
            print(
                side, repetition, sample["workload"], sample["wall_seconds"], flush=True
            )
    record["summary"] = {
        side: summarize_resources([sample for sample in values if not sample["warmup"]])
        for side, values in samples.items()
    }


def summarize_resources(records):
    """Summarize successful command samples without a reference comparison."""
    fields = (
        "wall_seconds",
        "cpu_user_seconds",
        "cpu_system_seconds",
        "max_rss_kib",
        "filesystem_input_bytes",
        "filesystem_output_bytes",
        "minor_page_faults",
        "major_page_faults",
        "voluntary_context_switches",
        "involuntary_context_switches",
    )
    return {
        "samples": len(records),
        **{
            field: {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            for field in fields
            if (values := [record[field] for record in records if field in record])
        },
    }


class CommandTimeout(Exception):
    pass


def inference_resources(root):
    """Read the detached inference processes in this benchmark runtime directory."""
    workers = {}
    for path in root.glob("*/inference/*/pid"):
        try:
            pid = int(path.read_text())
            process = Path("/proc") / str(pid)
            stat = (process / "stat").read_text().split(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            command = (process / "cmdline").read_bytes().split(b"\0")
            require(b"embedding-server" in command, "inference PID was reused")
            memory = {
                k: int(v.split()[0])
                for k, v in (
                    line.split(":", 1)
                    for line in (process / "status").read_text().splitlines()
                )
                if k in ("VmHWM", "VmRSS")
            }
            io = dict(
                line.split(":", 1) for line in (process / "io").read_text().splitlines()
            )
        except FileNotFoundError:
            continue
        workers[str(pid)] = {
            "start_ticks": int(stat[19]),
            "cpu_user_seconds": int(stat[11]) / os.sysconf("SC_CLK_TCK"),
            "cpu_system_seconds": int(stat[12]) / os.sysconf("SC_CLK_TCK"),
            "filesystem_input_bytes": int(io["read_bytes"]),
            "filesystem_output_bytes": int(io["write_bytes"]),
            "peak_rss_kib": memory["VmHWM"],
            "resident_rss_kib": memory["VmRSS"],
        }
    return workers


def _timeout(_signum, _frame):
    raise CommandTimeout("command deadline expired")


class PerformanceCase(Case):
    def __init__(self, root, binary, model_dir=None):
        self.time_binary = shutil.which("time")
        require(self.time_binary is not None, "GNU time must be on PATH")
        # Manual foreground commands do not need a daemon-owned executable copy.
        super().__init__(root, binary, model_dir, copy_binary=False)

    def _execute(self, args, timeout: float, workload=None):
        require(math.isfinite(timeout) and timeout > 0, "timeout must be positive")
        require(
            signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
            "performance runner requires exclusive ownership of SIGALRM",
        )
        self.sequence += 1
        log = self.logs / f"{self.sequence:03d}.json"
        out_path, err_path = log.with_suffix(".stdout"), log.with_suffix(".stderr")
        argv = [str(self.binary), *map(str, args)]
        memory_path = log.with_suffix(".rss-kib")
        # Fork after exec of a small launcher: forking trwl directly carries the
        # Python runner's resident memory into Linux's child ru_maxrss accounting.
        measurement_argv = [
            self.time_binary,
            "--quiet",
            "--format=%M",
            "--output=" + str(memory_path),
            "--",
            *argv,
        ]
        record = {
            "argv": argv,
            "measurement_argv": measurement_argv,
            "memory_file": str(memory_path),
            "cwd": str(self.work),
            "environment": dict(self.env),
            "workload": workload,
            "timeout_seconds": timeout,
            "returncode": None,
            "status": "starting",
            "timed_out": False,
        }
        workers_before = inference_resources(self.runtime) if workload else {}
        write_json(log, record)
        command = process = error = None
        started = ended = None
        usage = None
        deferred = []
        cleanup_signals = []

        def defer(signum, _frame):
            deferred.append(signum)

        def defer_cleanup(signum, _frame):
            cleanup_signals.append(signum)

        handlers = {
            sig: signal.signal(sig, defer)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        alarm_handler = signal.signal(signal.SIGALRM, _timeout)
        try:
            with out_path.open("wb") as out, err_path.open("wb") as err:
                started = time.perf_counter()
                process = subprocess.Popen(
                    measurement_argv,
                    env=self.env,
                    cwd=self.work,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                command = RunningCommand(
                    process, log, out_path, err_path, record, started
                )
                self.commands.append(command)
                record.update(pid=process.pid, status="running")
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                if deferred:
                    signal.raise_signal(deferred[0])
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise CommandTimeout("process launch exceeded the command deadline")
                signal.setitimer(signal.ITIMER_REAL, remaining)
                # Blocking wait4 avoids Popen.wait(timeout)'s polling granularity
                # and returns resource usage for this child, not earlier commands.
                _, status, usage = os.wait4(process.pid, 0)
                ended = time.perf_counter()
                process.returncode = os.waitstatus_to_exitcode(status)
                signal.setitimer(signal.ITIMER_REAL, 0)
                record["max_rss_kib"] = int(memory_path.read_text().strip())
                require(record["max_rss_kib"] >= 0, "invalid GNU time RSS measurement")
                record["status"] = "exited"
        except BaseException as exc:  # noqa: BLE001 - log cancellation before rethrowing
            ended = time.perf_counter()
            error = exc
            record.update(
                status="timeout"
                if isinstance(exc, CommandTimeout)
                else "interrupted"
                if not isinstance(exc, Exception)
                else "failed",
                timed_out=isinstance(exc, CommandTimeout),
                traceback=traceback.format_exc(),
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, alarm_handler)
            # Defer cancellation until cleanup/logs finish; do not lose a first
            # cancellation that arrives during output compression.
            for sig in handlers:
                signal.signal(sig, defer_cleanup)
            try:
                if process is not None:
                    stop_process_group(process)
                    record["returncode"] = process.returncode
                if command is not None:
                    command.finished = True
                if started is not None and ended is not None:
                    record["wall_seconds"] = ended - started
                if usage is not None:
                    record.update(
                        cpu_user_seconds=usage.ru_utime,
                        cpu_system_seconds=usage.ru_stime,
                        # Linux reports block I/O in 512-byte units, independent
                        # of the filesystem's block size. Cached reads and mmap
                        # accesses need separate syscall/page-fault profiling.
                        filesystem_input_bytes=usage.ru_inblock * 512,
                        filesystem_output_bytes=usage.ru_oublock * 512,
                        minor_page_faults=usage.ru_minflt,
                        major_page_faults=usage.ru_majflt,
                        voluntary_context_switches=usage.ru_nvcsw,
                        involuntary_context_switches=usage.ru_nivcsw,
                    )
                if workload and usage is not None and "max_rss_kib" in record:
                    workers_after = inference_resources(self.runtime)
                    require(
                        workers_before.keys() <= workers_after.keys(),
                        "inference worker exited during measurement; counters are incomplete",
                    )
                    record["worker_before"] = workers_before
                    record["worker_after"] = workers_after
                    record["client_resources"] = {
                        key: record[key]
                        for key in (
                            "cpu_user_seconds",
                            "cpu_system_seconds",
                            "max_rss_kib",
                            "filesystem_input_bytes",
                            "filesystem_output_bytes",
                        )
                    }
                    for pid, worker in workers_after.items():
                        before = workers_before.get(pid, {})
                        require(
                            not before
                            or before["start_ticks"] == worker["start_ticks"],
                            "inference PID changed during measurement",
                        )
                        for key in (
                            "cpu_user_seconds",
                            "cpu_system_seconds",
                            "filesystem_input_bytes",
                            "filesystem_output_bytes",
                        ):
                            record[key] += worker[key] - before.get(key, 0)
                        # Peaks can occur at different times. This sum bounds
                        # concurrent RSS; the worker's peak covers its lifetime.
                        record["max_rss_kib"] += worker["peak_rss_kib"]
                # Keep full output compressed, including large correctness listings.
                # Compression, decoding and metadata writes happen after timing.
                outputs = {}
                for name, path in (("stdout", out_path), ("stderr", err_path)):
                    raw = path.read_bytes() if path.exists() else b""
                    outputs[name] = raw.decode("utf-8", errors="replace")
                    archived = path.with_suffix(path.suffix + ".gz")
                    with gzip.GzipFile(
                        filename=archived, mode="wb", compresslevel=1
                    ) as stream:
                        stream.write(raw)
                    if path.exists():
                        path.unlink()
                    record[name + "_file"] = str(archived)
                write_json(log, record)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        if cleanup_signals and error is None:
            signal.raise_signal(cleanup_signals[0])
        if error is not None:
            if isinstance(error, CommandTimeout):
                raise AssertionError(f"command timed out: {log}") from error
            raise error
        return subprocess.CompletedProcess(
            argv, record["returncode"], outputs["stdout"], outputs["stderr"]
        ), record

    def command(self, *args, timeout: float = 60, expect_failure=False):
        result, _ = self._execute(args, timeout)
        if expect_failure is not None:
            require(
                result.returncode != 0 if expect_failure else result.returncode == 0,
                f"unexpected command exit {result.returncode}: {self.commands[-1].log}",
            )
        return result

    def measured(self, workload, *args, timeout: float = 120):
        result, record = self._execute(args, timeout, workload)
        require(
            result.returncode == 0,
            f"measured command failed ({result.returncode}): {self.commands[-1].log}",
        )
        return result, record
