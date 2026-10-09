"""Real local-model retrieval, and lexical retrieval when semantics is unavailable."""

import fcntl
import json
import os
import signal
import socket
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path

from e2e_support import (
    MissingPrerequisite,
    OpenCodeFixture,
    require,
    sha256,
    write_json,
)

TEXT = "poolbeacon The database connection pool exhausts under concurrent requests."
QUERY = "Why do parallel SQL operations stall when all available connections are busy?"
SESSION = "ses_semantic_pool"


def executor_reuse(case):
    """Inference survives process boundaries, competing clients and restart."""
    model_prerequisites(case)
    _fixture(case)
    case.command("semantic", "enable", "--executor", "builtin")
    expected = case.search(QUERY, "--events", backend="semantic", refresh="wait")

    def worker():
        files = list((case.data / "inference").glob("*/pid"))
        require(len(files) == 1, "expected one inference process")
        return files[0], int(files[0].read_text())

    def query():
        result = case.search(QUERY, "--events", backend="semantic")
        require(result == expected, "reused/restarted inference changed query results")

    pid_file, pid = worker()
    query()
    require(worker()[1] == pid, "fresh CLI process did not reuse inference")
    if case.model_dir is not None:
        # A reused server keeps its first client's cwd. Model paths belong to the
        # requesting client, including when it runs from another directory.
        original_work = case.work
        original_model = case.env["TRWL_SEMANTIC_MODEL_DIR"]
        other = case.root / "other-workspace"
        other.mkdir()
        (other / "model").symlink_to(case.model_dir, target_is_directory=True)
        try:
            case.work = other
            case.env["TRWL_SEMANTIC_MODEL_DIR"] = "model"
            result = case.search(
                QUERY, "--events", "--workspace", original_work, backend="semantic"
            )
            require(result == expected, "relative model path changed inference results")
            require(worker()[1] == pid, "relative model path restarted inference")
        finally:
            case.work = original_work
            case.env["TRWL_SEMANTIC_MODEL_DIR"] = original_model
    case.command("daemon", "stop")
    require(not pid_file.exists(), "daemon stop left inference resident")

    # All four fresh clients start without a server; startup must not create
    # competing model processes or mix replies between clients.
    running = [
        case.start(
            "search",
            QUERY,
            "--backend",
            "semantic",
            "--refresh",
            "off",
            "--events",
            "--format",
            "json",
        )
        for _ in range(4)
    ]
    for command in running:
        require(
            json.loads(case.wait(command).stdout) == expected,
            "concurrent inference changed results",
        )
    pid_file, pid = worker()
    require(
        (pid_file.parent.stat().st_mode & 0o077) == 0,
        "inference socket directory is accessible to other users",
    )

    # A corrupt length must not allocate the advertised buffer or end the server.
    descriptor = os.open(pid_file.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with socket.socket(socket.AF_UNIX) as stream:
            stream.settimeout(5)
            stream.connect(f"/proc/self/fd/{descriptor}/socket")
            stream.sendall(struct.pack("<I", 0xFFFFFFFF))
            require(stream.recv(1) == b"", "server accepted an unbounded frame")
    finally:
        os.close(descriptor)
    query()
    require(worker()[1] == pid, "malformed client killed the inference server")

    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while True:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split(")", 1)[1].split()[0] == "Z":
            break
        require(time.monotonic() < deadline, "inference process did not exit")
        time.sleep(0.02)
    query()
    pid_file, replacement = worker()
    require(replacement != pid, "stale socket did not start a replacement")

    # Idle expiry releases model memory even without an explicit stop command.
    deadline = time.monotonic() + 40
    while pid_file.exists():
        require(time.monotonic() < deadline, "idle inference process did not exit")
        time.sleep(0.1)
    query()
    pid_file, _ = worker()
    case.command("semantic", "disable")
    require(not pid_file.exists(), "semantic disable left inference resident")
    case.check(
        "inference reuse, concurrent startup, malformed client isolation, crash recovery, idle expiry and shutdown"
    )


def _fixture(case, *, refreshable=False):
    fixture = OpenCodeFixture(case.providers / "semantic.db", case.work)
    fixture.session(SESSION)
    fixture.message(SESSION, "msg_pool", TEXT)
    # Similar technical subjects compete for a result set smaller than the corpus.
    distractors = [
        "Database writes block behind a long transaction holding an exclusive row lock.",
        "Worker threads stop accepting new tasks when their execution queue is full.",
        "DNS misconfiguration prevents the application from resolving the database hostname.",
        "An expired client certificate causes the database TLS handshake to fail.",
        "The database runs a full table scan because the query has no suitable index.",
        "The connection is refused because the server is not listening on the configured port.",
        "A memory leak makes the operating system terminate the application process.",
        "CPU saturation increases latency even when the database has spare connections.",
    ]
    for i, text in enumerate(distractors):
        session = f"ses_semantic_other_{i}"
        fixture.session(session)
        fixture.message(session, f"msg_other_{i}", text)
    if refreshable:
        case.command(
            "sources",
            "add",
            "semantic",
            "--provider",
            "opencode",
            "--root",
            fixture.path,
        )
        case.json_command("import", "--all")
    else:
        case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    rows = case.json_command("list", "events", "--provider", "opencode")["events"]
    require(
        Counter(row["text"] for row in rows) == Counter([TEXT, *distractors])
        and len({row["trwl_event_id"] for row in rows}) == len(rows),
        "semantic fixture lost or duplicated target/distractor content",
    )
    case.check(
        "all nine complete fixture messages are retrievable before semantic queries"
    )
    hits = case.search("poolbeacon", "--events")["results"]
    require(len(hits) == 1, "lexical search lost or duplicated the target message")
    expected = hits[0]
    _show(case, expected)
    return fixture, expected


def _show(case, expected):
    event = case.json_command("show", "event", expected["trwl_event_id"])["event"]
    require(
        event["text"] == TEXT
        and event["provider"] == "opencode"
        and event["provider_session_id"] == SESSION
        and event["trwl_event_id"] == expected["trwl_event_id"]
        and event["trwl_session_id"] == expected["trwl_session_id"],
        "retrieval changed target content or identity",
    )


def device_policy(case):
    """Full builds use workers; queries/catch-up avoid GPU even on a new cache."""
    model_prerequisites(case)
    fixture, expected = _fixture(case, refreshable=True)
    case.command("semantic", "enable", "--executor", "builtin")
    poison = case.root / "not-a-lock-directory"
    poison.write_text("a worker would fail opening this as a directory")
    case.env["TRWL_GPU_LOCK_DIR"] = str(poison)

    def query():
        result = case.search(QUERY, "--events", backend="semantic", refresh="wait")
        require(
            any(
                hit["trwl_event_id"] == expected["trwl_event_id"]
                for hit in result["results"]
            ),
            "device policy lost paraphrase retrieval",
        )
        return result

    query()
    fixture.message(
        SESSION,
        "msg_new",
        "Connection pools limit the number of parallel database clients.",
    )
    case.command("setup", "--wait")
    query()
    case.check(
        "queries and incremental setup never open the GPU lock, including initial waited search"
    )

    # A filesystem error is not reclassified as a GPU failure and hidden by CPU.
    failed = case.command("semantic", "rebuild", expect_failure=True)
    require(
        "opening GPU lock" in failed.stderr, "worker hid an unrelated filesystem error"
    )
    del case.env["TRWL_GPU_LOCK_DIR"]
    rebuilt = case.command("semantic", "rebuild", "--format", "json", timeout=180)
    require(json.loads(rebuilt.stdout)["state"] == "ready", "rebuild did not complete")
    require(
        "full embedding build using " in rebuilt.stderr,
        "full rebuild did not start its worker",
    )
    query()
    case.check("explicit rebuild runs in a worker and CPU queries use its vectors")

    lock_path = Path(case.env["XDG_RUNTIME_DIR"]) / "trwl/gpu.lock"
    with lock_path.open("r+") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contended = case.command("semantic", "rebuild", timeout=180)
        require(
            "holds the GPU lock; using CPU" in contended.stderr,
            "contention did not select CPU",
        )
    query()
    case.check("worker releases its lock on exit; competing full builds finish on CPU")

    database = case.data / "semantic/cache.sqlite3"
    with sqlite3.connect(database) as connection:
        total = connection.execute("SELECT count(*) FROM vectors").fetchone()[0]
        require(total > 1, "resume fixture needs multiple vectors")
        connection.executescript("""
            DELETE FROM vectors WHERE key=(SELECT key FROM vectors LIMIT 1);
            UPDATE builds SET complete=0;
            CREATE TABLE test_vector_writes (key TEXT);
            CREATE TRIGGER test_writes AFTER INSERT ON vectors BEGIN
                INSERT INTO test_vector_writes VALUES (new.key);
            END;
        """)
    resumed = case.command("semantic", "rebuild", timeout=180)
    require(
        "full embedding build using " in resumed.stderr,
        "resumed build lost GPU eligibility",
    )
    with sqlite3.connect(database) as connection:
        # Resumption restores the missing vector without rewriting completed ones.
        require(
            connection.execute("SELECT count(*) FROM test_vector_writes").fetchone()[0]
            == 1,
            "resume rewrote completed vectors",
        )
        require(
            connection.execute("SELECT count(*) FROM vectors").fetchone()[0] == total,
            "resume lost completed vectors",
        )
        connection.executescript(
            "DROP TRIGGER test_writes; DROP TABLE test_vector_writes;"
        )
    case.check(
        "interrupted full build resumes only missing vectors and keeps full-build eligibility"
    )

    case.env["TRWL_GPU_LOCK_DIR"] = str(poison)
    case.env["TRWL_SEMANTIC_CPU_ONLY"] = "1"
    case.command("semantic", "rebuild")
    query()
    case.check("CPU-only override avoids the GPU worker for explicit rebuilds")
    del case.env["TRWL_GPU_LOCK_DIR"]
    del case.env["TRWL_SEMANTIC_CPU_ONLY"]
    case.command(
        "semantic", "enable", "--executor", "builtin", "--model", "changed-selection"
    )
    changed = case.command("setup", "--wait", timeout=180)
    require(
        "full embedding build using " in changed.stderr,
        "new model selection was not a full build",
    )
    query()
    _show(case, expected)
    case.check(
        "first fill of a new model selection uses a worker and preserves retrieval"
    )


def _lexical_survives(case, expected):
    hits = case.search("poolbeacon", "--events")["results"]
    require(
        len(hits) == 1 and hits[0]["trwl_event_id"] == expected["trwl_event_id"],
        "lexical retrieval lost the target while semantic retrieval was unavailable",
    )
    _show(case, expected)


def unavailable(case):
    """Disabled semantic retrieval errors explicitly; hybrid reports lexical use."""
    _, expected = _fixture(case)
    failure = case.command(
        "search",
        QUERY,
        "--backend",
        "semantic",
        "--refresh",
        "off",
        "--format",
        "json",
        expect_failure=True,
    )
    require(failure.returncode > 0, "semantic request terminated by a signal")
    require(
        "semantic" in (failure.stdout + failure.stderr).lower(),
        "unavailable semantic search did not explain the failure",
    )
    result = case.search("poolbeacon", "--events", backend="hybrid")
    retrieval = result["retrieval"]
    require(
        retrieval["effective_mode"] == "lexical"
        and retrieval["semantic_status"] != "ready",
        "hybrid incorrectly reported semantic participation",
    )
    require(
        any(
            hit["trwl_event_id"] == expected["trwl_event_id"]
            for hit in result["results"]
        ),
        "hybrid fallback lost the lexical result",
    )
    _lexical_survives(case, expected)
    case.check(
        "disabled semantic search errors; hybrid reports lexical fallback and preserves content"
    )


def model_prerequisites(case):
    """Validate supplied model files and record their hashes."""
    if case.model_dir is not None:
        names = (
            "onnx/model.onnx",
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        )
        missing = [name for name in names if not (case.model_dir / name).is_file()]
        if missing:
            raise MissingPrerequisite(
                f"model files missing in {case.model_dir}: {missing}"
            )
        write_json(
            case.root / "model.json",
            {
                "path": str(case.model_dir),
                "sha256": {name: sha256(case.model_dir / name) for name in names},
            },
        )


def passive_semantic(case):
    model_prerequisites(case)
    _, expected = _fixture(case)
    case.command("semantic", "enable", "--executor", "builtin", timeout=180)
    ready = case.search(
        QUERY, "--events", backend="semantic", refresh="wait", timeout=180
    )
    require(
        ready["retrieval"]["semantic_status"] == "ready",
        "waited query did not establish ready semantic retrieval",
    )
    passive = case.search(
        QUERY, "--events", backend="semantic", refresh="off", timeout=180
    )
    require(
        passive["retrieval"]["effective_mode"] == "semantic"
        and any(
            hit["trwl_event_id"] == expected["trwl_event_id"]
            for hit in passive["results"]
        ),
        "ready indexed history cannot be searched semantically without refreshing",
    )
    _show(case, expected)
    case.check("a ready semantic index can answer a query with refresh disabled")


def background_semantic(case):
    model_prerequisites(case)
    fixture, expected = _fixture(case, refreshable=True)
    config = case.data / "config.toml"
    config.write_text(config.read_text() + "[semantic]\nbuiltin_throttling = false\n")
    case.command("semantic", "enable", "--executor", "builtin", timeout=180)
    ready = case.search(
        QUERY, "--events", backend="semantic", refresh="wait", timeout=180
    )
    require(
        ready["retrieval"]["semantic_status"] == "ready",
        "initial semantic indexing did not finish",
    )
    config.write_text(config.read_text().replace('mode = "manual"', 'mode = "auto"'))
    daemon = None
    try:
        daemon = case.start("daemon", "run", "--loop-interval-seconds", "1")
        for batch in range(2):
            marker = f"backgroundsemantic{batch}"
            for number in range(20):
                fixture.message(
                    SESSION,
                    f"msg_background_{batch}_{number}",
                    f"{marker} Build output {number}: compiling workspace dependencies.",
                )
            require(
                daemon.process.poll() is None,
                f"background updater exited: {daemon.log}",
            )
            result = case.search(
                QUERY,
                "--events",
                "--limit",
                "3",
                backend="semantic",
                refresh="off",
                timeout=180,
            )
            require(
                result["retrieval"]["effective_mode"] == "semantic"
                and any(
                    hit["trwl_event_id"] == expected["trwl_event_id"]
                    for hit in result["results"]
                ),
                "automatic updates made existing semantic results unavailable",
            )
            _show(case, expected)
            deadline = time.monotonic() + 45
            while True:
                require(
                    daemon.process.poll() is None,
                    f"background updater exited: {daemon.log}",
                )
                remaining = deadline - time.monotonic()
                require(
                    remaining > 0,
                    "background updater did not discover the new messages",
                )
                if case.search(marker, "--events", timeout=min(5, remaining))[
                    "results"
                ]:
                    break
                time.sleep(0.1)
    finally:
        try:
            case.command("index", "mode", "manual", timeout=20)
        finally:
            if daemon is not None:
                case.stop(daemon)
    case.check(
        "semantic queries keep finding existing history as automatic updates ingest new messages"
    )


def semantic(case):
    model_prerequisites(case)
    fixture, expected = _fixture(case)
    config = case.data / "config.toml"
    config.write_text(config.read_text() + "[semantic]\nbuiltin_throttling = false\n")
    case.command("semantic", "enable", "--executor", "builtin", timeout=180)
    for backend in ("semantic", "hybrid"):
        result = case.search(
            QUERY,
            "--events",
            "--limit",
            "3",
            backend=backend,
            refresh="wait",
            timeout=180,
        )
        require(
            result["retrieval"]["effective_mode"] == backend
            and result["retrieval"]["semantic_status"] == "ready",
            f"{backend} did not report semantic participation",
        )
        hits = result["results"]
        require(
            1 <= len(hits) <= 3
            and any(hit["trwl_event_id"] == expected["trwl_event_id"] for hit in hits),
            f"{backend} paraphrase missed target among at most three results",
        )
        _show(case, expected)
        case.check(
            f"{backend} paraphrase retrieves the target among three results using the real model"
        )

    filtered = case.search(
        QUERY,
        "--events",
        "--exclude-session",
        expected["trwl_session_id"],
        backend="semantic",
        refresh="wait",
        timeout=180,
    )
    require(
        filtered["retrieval"]["effective_mode"] == "semantic"
        and all(
            hit["trwl_session_id"] != expected["trwl_session_id"]
            for hit in filtered["results"]
        ),
        "semantic search ignored the excluded session",
    )
    case.check("session exclusion also applies to real semantic retrieval")

    changed_session = "ses_semantic_changes"
    fixture.session(changed_session)
    for number, body in enumerate(
        (
            "Radio interference from nearby electronics can drown out a satellite signal.",
            "A damaged antenna cable prevents the radio receiver from decoding transmissions.",
            "Orbital debris can damage a spacecraft even when its communications equipment works.",
        )
    ):
        fixture.message(changed_session, f"msg_radio_distractor_{number}", body)
    versions = [
        (
            "registrybeacon The release pipeline stops when the container registry rejects expired credentials.",
            "Why did shipping our image fail after the registry login token timed out?",
        ),
        (
            "registrybeacon\n"
            + "Build diagnostic: compiling the workspace and checking dependencies.\n"
            * 128
            + "A satellite dish must point toward the spacecraft to receive its radio transmissions.",
            "How should we aim the antenna to pick up a signal from orbit?",
        ),
    ]
    changed_ids = set()
    for index, (body, query) in enumerate(versions):
        if index == 0:
            fixture.message(changed_session, "msg_semantic_changes", body)
        else:
            stamp = fixture.stamp()
            with sqlite3.connect(fixture.path) as db:
                db.execute(
                    "UPDATE part SET data = ?, time_updated = ? WHERE message_id = ?",
                    (
                        json.dumps({"type": "text", "text": body}),
                        stamp,
                        "msg_semantic_changes",
                    ),
                )
                db.execute(
                    "UPDATE message SET time_updated = ? WHERE id = ?",
                    (stamp, "msg_semantic_changes"),
                )
                db.execute(
                    "UPDATE session SET time_updated = ? WHERE id = ?",
                    (stamp, changed_session),
                )
        case.command("import", "--provider", "opencode", "--path", fixture.path)
        lexical = case.search("registrybeacon", "--events")["results"]
        require(
            len(lexical) == 1,
            "changed message missing or duplicated before semantic refresh",
        )
        event_id = lexical[0]["trwl_event_id"]
        changed_ids.add(event_id)
        for backend in ("semantic", "hybrid"):
            result = case.search(
                query,
                "--events",
                "--limit",
                "3",
                backend=backend,
                refresh="wait",
                timeout=180,
            )
            require(
                result["retrieval"]["effective_mode"] == backend
                and result["retrieval"]["semantic_status"] == "ready",
                "changed-history search did not use the requested backend",
            )
            require(
                any(hit["trwl_event_id"] == event_id for hit in result["results"]),
                f"{backend} missed the {'appended' if index == 0 else 'edited'} message by meaning",
            )
            shown = case.json_command("show", "event", event_id)["event"]
            require(
                shown["text"] == body
                and shown["provider_session_id"] == changed_session,
                "semantic search retrieved an outdated message body",
            )
        _lexical_survives(case, expected)
    case.check(
        "semantic and hybrid search find appended and edited messages by their new meaning"
    )

    stamp = fixture.stamp()
    with sqlite3.connect(fixture.path) as db:
        db.execute("DELETE FROM part WHERE message_id = ?", ("msg_semantic_changes",))
        db.execute("DELETE FROM message WHERE id = ?", ("msg_semantic_changes",))
        db.execute(
            "UPDATE session SET time_updated = ? WHERE id = ?", (stamp, changed_session)
        )
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    require(
        not case.search("registrybeacon", "--events")["results"],
        "deleted message remains in lexical search",
    )
    for backend in ("semantic", "hybrid"):
        result = case.search(
            versions[-1][1], "--events", backend=backend, refresh="wait", timeout=180
        )
        require(
            result["retrieval"]["effective_mode"] == backend,
            "deleted-history check did not use the requested backend",
        )
        require(
            not changed_ids.intersection(
                hit["trwl_event_id"] for hit in result["results"]
            ),
            "deleted message remains in semantic results",
        )
    case.check(
        "deleted messages disappear from semantic and hybrid results after refresh"
    )

    # Make only the model unavailable, through its documented override. Do not
    # inspect or repair semantic storage to arrange the failure.
    missing_model = case.root / "missing-model"
    missing_model.mkdir()
    case.env["TRWL_SEMANTIC_MODEL_DIR"] = str(missing_model)
    failure = case.command(
        "search",
        QUERY,
        "--backend",
        "semantic",
        "--refresh",
        "wait",
        "--format",
        "json",
        timeout=180,
        expect_failure=True,
    )
    require(failure.returncode > 0, "missing-model request terminated by a signal")
    require(
        str(missing_model) in failure.stdout + failure.stderr,
        "missing-model error did not identify the unavailable model directory",
    )
    _lexical_survives(case, expected)
    case.check(
        "missing model reports its location; lexical content and IDs remain usable"
    )
