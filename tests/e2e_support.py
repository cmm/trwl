"""Standard-library helpers for external trwl cases; never open trwl storage."""

import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / ".cache"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class MissingPrerequisite(Exception):
    """The case cannot run; this is not a passing or failing behavior check."""


def git(directory, *args):
    executable = shutil.which("git")
    if executable is None:
        raise MissingPrerequisite("workspace fixtures require Git")
    result = subprocess.run(
        [
            executable,
            "-C",
            str(directory),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *map(str, args),
        ],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(directory),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TEMPLATE_DIR": "",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    require(result.returncode == 0, f"Git fixture failed: {result.stderr}")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stop_process_group(process):
    """Kill descendants even if the group leader has already exited; reap it."""
    for sig in (signal.SIGCONT, signal.SIGTERM):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


@dataclass
class RunningCommand:
    process: subprocess.Popen
    log: Path
    stdout_path: Path
    stderr_path: Path
    record: dict
    started: float
    finished: bool = False


class Case:
    """An isolated workspace with command logging and process-group cleanup."""

    def __init__(self, root, binary, model_dir=None, *, copy_binary=True):
        self.root = root
        self.root.mkdir(mode=0o700)
        self.home = root / "home"
        self.data = root / "trwl-data"
        self.work = root / "workspace"
        self.providers = root / "providers"
        self.logs = root / "commands"
        self.model_dir = model_dir
        self.commands = []
        self.sequence = 0
        self.checks = []
        self.env = {"PATH": "", "NO_COLOR": "1", "TERM": "dumb", "TZ": "UTC"}
        # No inherited environment, credentials, active-session IDs or live roots.
        for key, relative in {
            "HOME": "home",
            "USERPROFILE": "home",
            "XDG_CONFIG_HOME": "config",
            "XDG_DATA_HOME": "data",
            "XDG_STATE_HOME": "state",
            "XDG_CACHE_HOME": "cache",
            "XDG_RUNTIME_DIR": "runtime",
            "APPDATA": "appdata",
            "LOCALAPPDATA": "localappdata",
            "TMPDIR": "tmp",
            "TMP": "tmp",
            "TEMP": "tmp",
            "TRWL_DATA_ROOT": "trwl-data",
            "CODEX_HOME": "providers/codex",
            "CLAUDE_CONFIG_DIR": "providers/claude",
        }.items():
            path = root / relative
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.env[key] = str(path)
        for path in (self.work, self.logs, self.providers):
            path.mkdir(exist_ok=True, mode=0o700)
        if model_dir is not None:
            self.env["TRWL_SEMANTIC_MODEL_DIR"] = str(model_dir)
        (self.data / "config.toml").write_text(
            '[indexing]\nmode = "manual"\n'
            "[sources]\nautomatic = false\n"
            "[search]\nsemantic = false\n"
        )
        # Keep it outside data so reopening a copied index uses the same executable.
        if copy_binary:
            (root / "bin").mkdir(mode=0o700)
            self.binary = root / "bin/trwl"
            shutil.copyfile(binary, self.binary)
            self.binary.chmod(0o700)
        else:
            self.binary = binary
        write_json(
            root / "workspace.json",
            {
                "invoked_binary": str(self.binary),
                "binary_sha256": sha256(self.binary),
                "cwd": str(self.work),
                "environment": self.env,
            },
        )

    def check(self, label):
        self.checks.append(label)

    def start(self, *args, timeout=60):
        """Launch a command in a new process group, recording it before launch."""
        self.sequence += 1
        log = self.logs / f"{self.sequence:03d}.json"
        stdout_path = log.with_suffix(".stdout")
        stderr_path = log.with_suffix(".stderr")
        record = {
            "argv": [str(self.binary), *map(str, args)],
            "cwd": str(self.work),
            "environment": dict(self.env),
            "timeout_seconds": timeout,
            "timed_out": False,
            "returncode": None,
            "status": "starting",
            "stdout": "",
            "stderr": "",
        }
        write_json(log, record)
        started = time.monotonic()
        # Defer parent handlers until the child is registered. Blocking OS signals
        # here would give the executable an inherited blocked signal mask.
        deferred = []

        def defer(signum, _frame):
            deferred.append(signum)

        handlers = {
            sig: signal.signal(sig, defer)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                process = subprocess.Popen(
                    record["argv"],
                    env=self.env,
                    cwd=self.work,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
            command = RunningCommand(
                process, log, stdout_path, stderr_path, record, started
            )
            self.commands.append(command)
            record.update(status="running", pid=process.pid)
            write_json(log, record)
        except BaseException:
            record.update(status="launch_failed", traceback=traceback.format_exc())
            write_json(log, record)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if deferred:
                signal.raise_signal(deferred[0])
        return command

    def stop(self, command):
        """Stop a command/group and finalize all output, including on exceptions."""
        if command.finished:
            return
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        )
        try:
            stop_process_group(command.process)
            command.record.update(
                returncode=command.process.returncode,
                elapsed_seconds=time.monotonic() - command.started,
                stdout=command.stdout_path.read_bytes().decode(
                    "utf-8", errors="replace"
                ),
                stderr=command.stderr_path.read_bytes().decode(
                    "utf-8", errors="replace"
                ),
            )
            if command.record["status"] == "running":
                command.record["status"] = "stopped"
            write_json(command.log, command.record)
            command.finished = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def wait(self, command, *, expect_failure=False):
        """Wait with a deadline; None leaves exit-status assertions to the case."""
        try:
            remaining = max(
                0,
                command.record["timeout_seconds"]
                - (time.monotonic() - command.started),
            )
            command.process.wait(timeout=remaining)
            command.record["status"] = "exited"
        except subprocess.TimeoutExpired:
            command.record.update(
                status="timeout", timed_out=True, traceback=traceback.format_exc()
            )
            raise AssertionError(f"command timed out: {command.log}") from None
        except BaseException:
            command.record.update(
                status="interrupted", traceback=traceback.format_exc()
            )
            raise
        finally:
            self.stop(command)
        code = command.record["returncode"]
        if expect_failure is not None:
            require(
                code != 0 if expect_failure else code == 0,
                f"unexpected command exit {code}: {command.log}",
            )
        return subprocess.CompletedProcess(
            command.record["argv"],
            code,
            command.record["stdout"],
            command.record["stderr"],
        )

    def command(self, *args, timeout=60, expect_failure=False):
        return self.wait(
            self.start(*args, timeout=timeout), expect_failure=expect_failure
        )

    def json_command(self, *args, timeout=60):
        result = self.command(*args, "--format", "json", timeout=timeout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"invalid JSON: {self.commands[-1].log}") from exc

    def search(self, query, *args, backend="lexical", refresh="off", timeout=60):
        return self.json_command(
            "search",
            query,
            "--backend",
            backend,
            "--refresh",
            refresh,
            *args,
            timeout=timeout,
        )

    def close(self):
        # Do not let a second signal interrupt cleanup of another process group.
        handlers = {
            sig: signal.signal(sig, signal.SIG_IGN)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        errors = []
        try:
            for command in self.commands:
                try:
                    self.stop(command)
                except Exception:  # noqa: BLE001 - try every group before reporting errors
                    errors.append(traceback.format_exc())
            if (self.data / "inference").exists():
                try:
                    self.command("daemon", "stop", timeout=125)
                except Exception:  # noqa: BLE001 - report cleanup failures together
                    errors.append(traceback.format_exc())
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if errors:
            write_json(self.root / "cleanup-errors.json", errors)
            raise RuntimeError(
                f"process cleanup failed: {self.root / 'cleanup-errors.json'}"
            )


class OpenCodeFixture:
    """Create OpenCode SQLite sessions, messages and parts."""

    def __init__(self, path, workspace):
        self.path = path
        self.workspace = workspace
        self.sequence = 0
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT,
                    directory TEXT, branch TEXT, agent TEXT,
                    time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL);
                CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL);
                CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, time_created INTEGER NOT NULL,
                    time_updated INTEGER NOT NULL, data TEXT NOT NULL);
                CREATE INDEX message_session_time_created_id_idx ON message(session_id, time_created, id);
                CREATE INDEX part_message_time_id_idx ON part(message_id, time_created, id);
                CREATE INDEX part_session_idx ON part(session_id);
            """)

    def stamp(self):
        self.sequence += 1
        return 1_790_000_000_000 + self.sequence * 1000

    def session(self, session_id, *, parent=None, agent="build"):
        stamp = self.stamp()
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO session VALUES (?, ?, ?, 'main', ?, ?, ?)",
                (session_id, parent, str(self.workspace), agent, stamp, stamp),
            )

    def message(self, session_id, message_id, text, *, role="user"):
        stamp = self.stamp()
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, stamp, stamp, json.dumps({"role": role})),
            )
        self.part(
            session_id, message_id, "part_" + message_id, {"type": "text", "text": text}
        )

    def part(self, session_id, message_id, part_id, content):
        stamp = self.stamp()
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                (part_id, message_id, session_id, stamp, stamp, json.dumps(content)),
            )
            db.execute(
                "UPDATE message SET time_updated = ? WHERE id = ?", (stamp, message_id)
            )
            db.execute(
                "UPDATE session SET time_updated = ? WHERE id = ?", (stamp, session_id)
            )


def codex_fixture(path, workspace, session_id, messages):
    """Write native Codex rollout JSONL; messages is a sequence of (role, text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "timestamp": "2026-09-21T12:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": "2026-09-21T12:00:00Z",
                "cwd": str(workspace),
                "originator": "codex_cli_rs",
                "cli_version": "0.1.0",
                "source": "cli",
                "model_provider": "openai",
            },
        }
    ]
    for role, text in messages:
        records.append(
            {
                "timestamp": "2026-09-21T12:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path
