"""Publication and lexical discovery while provider and embedding work continues."""

import json
import resource
import sqlite3
import threading
import time
from contextlib import closing
from http.server import ThreadingHTTPServer
from typing import NotRequired, TypedDict

from e2e_http import MODEL, EmbeddingServer
from e2e_semantic import QUERY, SESSION, _fixture, _show, model_prerequisites
from e2e_support import OpenCodeFixture, require, write_json


class SourceWrites(TypedDict):
    writes: int
    reader_overlaps: int
    error: NotRequired[str]


def many_inputs(case):
    root = case.providers / "many-inputs"
    root.mkdir()
    for n in range(80):
        header = {
            "type": "session_meta",
            "payload": {"id": f"session-{n}", "cwd": str(case.work)},
        }
        row = {
            "type": "response_item",
            "payload": {
                "id": f"event-{n}",
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"manysourcebeacon message {n}"}
                ],
            },
        }
        (root / f"{n:03}.jsonl").write_text(
            f"{json.dumps(header)}\n{json.dumps(row)}\n"
        )
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, limits[0]), limits[1]))
        result = case.json_command("import", "--provider", "codex", "--path", root)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    require(
        result["outcome"] == "completed",
        "import exhausted descriptors while retaining parsed sources",
    )
    hits = case.search("manysourcebeacon", "--events", "--limit", "100")["results"]
    require(
        len(hits) == 80 and len({h["trwl_event_id"] for h in hits}) == 80,
        "many-input import lost or duplicated records",
    )
    case.check(
        "more changed inputs than the descriptor limit import without losing records"
    )


def busy_source(case):
    fixture = OpenCodeFixture(case.providers / "busy.db", case.work)
    session = "ses_busy_source"
    fixture.session(session)
    fixture.message(session, "msg_live", "beforelivebeacon")
    fixture.message(session, "msg_control", "stablebusybeacon")
    stamp = fixture.stamp()
    with closing(sqlite3.connect(fixture.path)) as db, db:
        for n in range(4000):
            mid = f"msg_bulk_{n}"
            db.execute(
                "INSERT INTO message VALUES (?,?,?,?,?)",
                (mid, session, stamp + n, stamp + n, '{"role":"assistant"}'),
            )
            db.execute(
                "INSERT INTO part VALUES (?,?,?,?,?,?)",
                (
                    f"part_{mid}",
                    mid,
                    session,
                    stamp + n,
                    stamp + n,
                    json.dumps({"type": "text", "text": "background filler " * 256}),
                ),
            )
    case.command("import", "--provider", "opencode", "--path", fixture.path, timeout=20)
    previous = case.search("beforelivebeacon", "--events")["results"]
    require(len(previous) == 1, "initial live message missing")
    ready, stop = threading.Event(), threading.Event()
    state: SourceWrites = {"writes": 0, "reader_overlaps": 0}

    def write():
        try:
            with closing(sqlite3.connect(fixture.path)) as db:
                db.execute("PRAGMA journal_mode=WAL")
                while not stop.is_set():
                    state["writes"] += 1
                    body = f"streamingbeacon revision {state['writes']}"
                    db.execute(
                        "UPDATE part SET data=?,time_updated=? WHERE message_id='msg_live'",
                        (
                            json.dumps({"type": "text", "text": body}),
                            stamp + 5000 + state["writes"],
                        ),
                    )
                    db.commit()
                    _, frames, checkpointed = db.execute(
                        "PRAGMA wal_checkpoint(PASSIVE)"
                    ).fetchone()
                    if checkpointed < frames:
                        state["reader_overlaps"] += 1
                    ready.set()
                    stop.wait(0.001)
        except Exception as exc:  # noqa: BLE001 - Report worker failures to the test thread.
            state["error"] = str(exc)
            ready.set()

    writer = threading.Thread(target=write)
    writer.start()
    try:
        require(ready.wait(5), "source writer did not start")
        result = case.json_command(
            "import", "--provider", "opencode", "--path", fixture.path, timeout=20
        )
        require(
            result["outcome"] in ("completed", "completed_with_rejections"),
            "busy source did not publish its read snapshot",
        )
        require(
            state["reader_overlaps"] > 0 and "error" not in state,
            "no source commit overlapped the import's SQLite read transaction",
        )
        hits = case.search("streamingbeacon", "--events", timeout=5)["results"]
        require(
            len(hits) == 1 and hits[0]["trwl_event_id"] == previous[0]["trwl_event_id"],
            "snapshot lost the live record or its identity",
        )
        require(
            len(case.search("stablebusybeacon", "--events", timeout=5)["results"]) == 1,
            "busy import lost unchanged history",
        )
    finally:
        stop.set()
        writer.join(timeout=5)
        write_json(case.root / "source-writes.json", state)
    require(not writer.is_alive() and "error" not in state, "source writer failed")
    case.command("import", "--provider", "opencode", "--path", fixture.path, timeout=20)
    event = case.json_command("show", "event", previous[0]["trwl_event_id"])["event"]
    require(
        event["text"] == f"streamingbeacon revision {state['writes']}",
        "pre-read checkpoint skipped later committed changes",
    )
    case.check(
        "busy SQLite snapshots publish without losing subsequent commits or existing references"
    )


def semantic_live(case):
    model_prerequisites(case)
    api = EmbeddingServer(case.model_dir, server_class=ThreadingHTTPServer)
    reached, release = threading.Event(), threading.Event()

    def controlled(_path, _model, texts):
        if any("heldembeddingdocument" in text for text in texts):
            reached.set()
            require(release.wait(20), "held document request was not released")

    api.request_hook = controlled
    daemon = None
    try:
        with api:
            try:
                fixture, expected = _fixture(case, refreshable=True)
                api.enable(case)
                first = case.search(
                    QUERY, "--events", backend="semantic", refresh="wait", timeout=20
                )
                require(
                    first["retrieval"]["semantic_status"] == "ready",
                    "initial embeddings not ready",
                )
                config = case.data / "config.toml"
                config.write_text(
                    config.read_text().replace('mode = "manual"', 'mode = "auto"')
                )
                daemon = case.start(
                    "daemon", "run", "--loop-interval-seconds", "1", timeout=45
                )
                fixture.message(
                    SESSION,
                    "msg_held",
                    "heldembeddingdocument Registry credentials expired.",
                )
                require(
                    reached.wait(10), "updater did not start the held embedding request"
                )
                body = "laterarrivalbeacon The release pipeline stops when the container registry rejects expired credentials."
                fixture.message(SESSION, "msg_later", body)
                deadline = time.monotonic() + 5
                while True:
                    hits = case.search("laterarrivalbeacon", "--events", timeout=2)[
                        "results"
                    ]
                    if hits:
                        break
                    require(
                        time.monotonic() < deadline,
                        "lexical discovery waited behind document embeddings",
                    )
                    time.sleep(0.05)
                require(
                    len(hits) == 1
                    and not release.is_set()
                    and daemon.process.poll() is None,
                    "lexical arrival was not observed during the held request",
                )
                added = hits[0]["trwl_event_id"]
                require(
                    case.json_command("show", "event", added)["event"]["text"] == body,
                    "concurrent lexical publication changed complete text",
                )
                for backend in ("semantic", "hybrid"):
                    found = case.search(
                        QUERY, "--events", "--limit", "3", backend=backend, timeout=5
                    )
                    require(
                        found["retrieval"]["effective_mode"] == backend
                        and found["retrieval"]["semantic_status"] == "partial"
                        and any(
                            h["trwl_event_id"] == expected["trwl_event_id"]
                            for h in found["results"]
                        ),
                        "published semantic history unavailable during catch-up",
                    )
                _show(case, expected)
                case.check(
                    "later lexical arrivals and existing semantic/hybrid reads remain available during an embedding request"
                )
                release.set()
                deadline = time.monotonic() + 10
                while True:
                    found = case.search(
                        "Why did shipping our image fail after the registry login token timed out?",
                        "--events",
                        "--limit",
                        "3",
                        backend="semantic",
                        timeout=5,
                    )
                    if found["retrieval"]["semantic_status"] == "ready":
                        require(
                            any(h["trwl_event_id"] == added for h in found["results"]),
                            "later lexical arrival never became searchable by meaning",
                        )
                        break
                    require(
                        time.monotonic() < deadline,
                        "embedding worker did not catch up with the newer commit",
                    )
                    time.sleep(0.05)
                reached.clear()
                release.clear()
                fixture.message(
                    SESSION,
                    "msg_stop_held",
                    "heldembeddingdocumentstop A pending shutdown request.",
                )
                require(reached.wait(10), "second held request did not start")
                case.command("index", "mode", "manual", timeout=5)
                case.wait(daemon)
                require(
                    not release.is_set(),
                    "shutdown check released the request prematurely",
                )
                case.check(
                    "embedding catch-up follows newer lexical commits; manual mode stops a daemon with a pending request"
                )
            finally:
                release.set()
                if daemon is not None:
                    case.stop(daemon)
    finally:
        write_json(case.root / "embedding-requests.json", api.requests)


def semantic_recovery(case):
    model_prerequisites(case)
    alternate_model = MODEL + "-alternate"
    api = EmbeddingServer(
        case.model_dir,
        server_class=ThreadingHTTPServer,
        models=(MODEL, alternate_model),
    )
    gates = []
    attempts = {}
    failed = threading.Event()

    def gate(marker, model, path):
        entry = (marker, model, path, threading.Event(), threading.Event())
        gates.append(entry)
        return entry[3:]

    def control(path, model, texts):
        if any("retrydocumentbeacon" in text for text in texts):
            key = (model, path)
            attempts[key] = attempts.get(key, 0) + 1
            if key == (MODEL, "/v1/embeddings") and attempts[key] == 1:
                failed.set()
                raise RuntimeError("injected transient request failure")
        for marker, expected_model, expected_path, reached, release in list(gates):  # noqa: PERF101 - Gates can change while a request is blocked.
            if (
                model == expected_model
                and path == expected_path
                and any(marker in text for text in texts)
            ):
                reached.set()
                if not release.wait(20):
                    raise RuntimeError("held request exceeded test deadline")

    api.request_hook = control
    daemon = None

    def await_ready(expected_id):
        deadline = time.monotonic() + 12
        while True:
            result = case.search(
                QUERY, "--events", "--limit", "100", backend="hybrid", timeout=5
            )
            if result["retrieval"]["semantic_status"] == "ready":
                require(
                    any(r["trwl_event_id"] == expected_id for r in result["results"]),
                    "semantic catch-up omitted an event",
                )
                return
            require(time.monotonic() < deadline, "semantic retry did not finish")
            time.sleep(0.05)

    def missing_selection():
        case.command(
            "search",
            QUERY,
            "--events",
            "--backend",
            "semantic",
            "--refresh",
            "off",
            "--format",
            "json",
            expect_failure=True,
            timeout=5,
        )

    def lexical_id(marker):
        hits = case.search(marker, "--events")["results"]
        require(len(hits) == 1, "pending document is not available lexically")
        return hits[0]["trwl_event_id"]

    def document_inputs(model, path):
        return {
            text
            for request in api.requests
            if request["model"] == model
            and request["path"] == path
            and request.get("status") == 200
            for text in request["texts"]
            if text != QUERY
        }

    try:
        with api:
            try:
                fixture, expected = _fixture(case, refreshable=True)
                api.enable(case)
                case.search(
                    QUERY, "--events", backend="semantic", refresh="wait", timeout=20
                )
                config = case.data / "config.toml"
                config.write_text(
                    config.read_text().replace('mode = "manual"', 'mode = "auto"')
                )
                daemon = case.start(
                    "daemon", "run", "--loop-interval-seconds", "1", timeout=90
                )

                reached, release = gate("retrydocumentbeacon", MODEL, "/v1/embeddings")
                fixture.message(
                    SESSION,
                    "msg_retry",
                    "retrydocumentbeacon An expired registry token blocks shipping images.",
                )
                require(
                    failed.wait(10) and reached.wait(10),
                    "daemon did not retry the failed request without an input change",
                )
                retry_id = lexical_id("retrydocumentbeacon")
                require(
                    case.search(QUERY, "--events", backend="semantic")["retrieval"][
                        "semantic_status"
                    ]
                    == "partial",
                    "failed fill damaged existing embeddings",
                )
                release.set()
                await_ready(retry_id)
                case.check(
                    "background embedding failures retry without another source change"
                )

                reached, release = gate("modelchangedocument", MODEL, "/v1/embeddings")
                new_reached, new_release = gate(
                    "modelchangedocument", alternate_model, "/v1/embeddings"
                )
                fixture.message(
                    SESSION,
                    "msg_model_change",
                    "modelchangedocument A document arriving during a model switch.",
                )
                require(reached.wait(10), "old model fill did not start")
                case.command(
                    "semantic",
                    "enable",
                    "--executor",
                    api.url,
                    "--model",
                    alternate_model,
                )
                missing_selection()
                release.set()
                require(
                    new_reached.wait(10),
                    "model switch did not re-embed existing content",
                )
                missing_selection()
                new_release.set()
                await_ready(lexical_id("modelchangedocument"))
                require(
                    document_inputs(alternate_model, "/v1/embeddings")
                    == document_inputs(MODEL, "/v1/embeddings"),
                    "new model did not re-embed all document inputs",
                )
                case.check(
                    "model changes during a fill do not reuse the old model's cache"
                )

                reached, release = gate(
                    "executorchangedocument", alternate_model, "/v1/embeddings"
                )
                new_reached, new_release = gate(
                    "executorchangedocument",
                    alternate_model,
                    "/alternate/v1/embeddings",
                )
                fixture.message(
                    SESSION,
                    "msg_executor_change",
                    "executorchangedocument A document arriving during an executor switch.",
                )
                require(reached.wait(10), "old executor fill did not start")
                alternate_url = api.url.replace("/v1", "/alternate/v1")
                case.command(
                    "semantic",
                    "enable",
                    "--executor",
                    alternate_url,
                    "--model",
                    alternate_model,
                )
                missing_selection()
                release.set()
                require(
                    new_reached.wait(10),
                    "executor switch did not re-embed existing content",
                )
                missing_selection()
                new_release.set()
                await_ready(lexical_id("executorchangedocument"))
                require(
                    document_inputs(alternate_model, "/alternate/v1/embeddings")
                    == document_inputs(alternate_model, "/v1/embeddings"),
                    "new executor did not re-embed all document inputs",
                )
                case.check("executor changes during a fill use a separate cache")

                reached, release = gate(
                    "restartdocumentbeacon", alternate_model, "/alternate/v1/embeddings"
                )
                fixture.message(
                    SESSION,
                    "msg_restart",
                    "restartdocumentbeacon History awaiting embedding publication.",
                )
                require(reached.wait(10), "interrupted fill did not start")
                restart_id = lexical_id("restartdocumentbeacon")
                case.stop(daemon)
                result = case.search(
                    QUERY, "--events", "--limit", "100", backend="semantic"
                )
                require(
                    result["retrieval"]["semantic_status"] == "partial"
                    and any(
                        r["trwl_event_id"] == expected["trwl_event_id"]
                        for r in result["results"]
                    )
                    and all(
                        r["trwl_event_id"] != restart_id for r in result["results"]
                    ),
                    "interrupted fill replaced published embeddings",
                )
                release.set()
                daemon = case.start(
                    "daemon", "run", "--loop-interval-seconds", "1", timeout=20
                )
                await_ready(restart_id)
                _show(case, expected)
                case.check(
                    "restart completes interrupted embedding work while preserving the previous cache"
                )
            finally:
                for _, _, _, _, release in gates:
                    release.set()
                if daemon is not None:
                    case.stop(daemon)
    finally:
        write_json(case.root / "embedding-requests.json", api.requests)
