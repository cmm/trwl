"""HTTP embedding workflows and grouped retrieval with controlled rankings."""

import importlib
import json
import sqlite3
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer

from e2e_semantic import (
    QUERY,
    SESSION,
    _fixture,
    _lexical_survives,
    _show,
    model_prerequisites,
)
from e2e_support import MissingPrerequisite, OpenCodeFixture, git, require, write_json

MODEL = "e5-fixture"
TOKEN = "fixture-only-token"


class E5Embeddings:
    def __init__(self, model_dir):
        if model_dir is None:
            raise MissingPrerequisite("semantic-http requires --model-dir")
        try:
            np = importlib.import_module("numpy")
            ort = importlib.import_module("onnxruntime")
            Tokenizer = importlib.import_module("tokenizers").Tokenizer
        except ImportError as exc:
            raise MissingPrerequisite(
                "semantic-http requires Python packages onnxruntime and tokenizers"
            ) from exc

        self.np = np
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.model = ort.InferenceSession(
            str(model_dir / "onnx/model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, texts):
        # E5's symmetric similarity mode uses the query prefix for both texts.
        encoded = self.tokenizer.encode_batch(["query: " + text for text in texts])
        arrays = {
            "input_ids": self.np.array([row.ids for row in encoded], dtype="int64"),
            "attention_mask": self.np.array(
                [row.attention_mask for row in encoded], dtype="int64"
            ),
            "token_type_ids": self.np.array(
                [row.type_ids for row in encoded], dtype="int64"
            ),
        }
        hidden = self.model.run(
            None, {item.name: arrays[item.name] for item in self.model.get_inputs()}
        )[0]
        mask = arrays["attention_mask"][..., None].astype("float32")
        vectors = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        vectors /= self.np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.tolist()


class EmbeddingServer:
    def __init__(
        self, model_dir=None, *, server_class=HTTPServer, models=(MODEL,), embed=None
    ):
        self.embed = E5Embeddings(model_dir) if embed is None else embed
        self.requests = []
        # Tokenizer state is shared. Request gates run outside this lock so a
        # held document request does not prevent query inference.
        self.inference_lock = threading.Lock()
        self.request_hook: Callable[[str, str, list[str]], None] | None = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def reply(self, status, value):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    # An updater can stop while a document request is pending.
                    pass

            def do_POST(self):
                if self.path not in ("/v1/embeddings", "/alternate/v1/embeddings"):
                    self.reply(404, {"error": {"message": "unknown endpoint"}})
                    return
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self.reply(401, {"error": {"message": "invalid fixture token"}})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    self.reply(413, {"error": {"message": "invalid request size"}})
                    return
                value = json.loads(self.rfile.read(length))
                inputs = value.get("input")
                if isinstance(inputs, str):
                    inputs = [inputs]
                if (
                    value.get("model") not in models
                    or value.get("encoding_format", "float") != "float"
                    or not isinstance(inputs, list)
                    or not inputs
                    or not all(isinstance(text, str) and text for text in inputs)
                ):
                    self.reply(400, {"error": {"message": "invalid embedding request"}})
                    return
                observed = {
                    "path": self.path,
                    "model": value["model"],
                    "inputs": len(inputs),
                    "texts": inputs,
                }
                owner.requests.append(observed)
                try:
                    if owner.request_hook is not None:
                        owner.request_hook(self.path, value["model"], inputs)
                    with owner.inference_lock:
                        vectors = owner.embed(inputs)
                except RuntimeError as exc:
                    observed["status"] = 503
                    self.reply(503, {"error": {"message": str(exc)}})
                    return
                observed["status"] = 200
                # Indices carry input association; response-array order need not.
                self.reply(
                    200,
                    {
                        "object": "list",
                        "model": value["model"],
                        "data": list(
                            reversed(
                                [
                                    {
                                        "object": "embedding",
                                        "index": index,
                                        "embedding": vector,
                                    }
                                    for index, vector in enumerate(vectors)
                                ]
                            )
                        ),
                    },
                )

        self.server = server_class(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )

    def enable(self, case):
        case.env["TRWL_SEMANTIC_MODEL_DIR"] = str(case.root / "no-local-model")
        case.env["TRWL_SEMANTIC_EMBEDDING_TOKEN"] = TOKEN
        case.command(
            "semantic", "enable", "--executor", self.url, "--model", MODEL, timeout=10
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def hybrid_grouping(case):
    fixture = OpenCodeFixture(case.providers / "grouping.db", case.work)
    expected = {"crowded", *(f"other-{i}" for i in range(10))}
    for session in sorted(expected):
        fixture.session(session)
        for n in range(120 if session == "crowded" else 1):
            text = (
                f"needle dominant {n}"
                if session == "crowded"
                else "needle " + "filler " * 30
            )
            fixture.message(session, f"{session}-{n}", text)
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)

    def vectors(texts):
        # Both retrievers rank the crowded session's events ahead of the others.
        return [
            [0.0, 0.0, 1.0]
            if text == "fusionneedle" or "semanticfavor" in text
            else [0.0, 0.0, -1.0]
            if "lexicalleader" in text
            else [1.0, 0.0, 0.0]
            if text == "needle" or "dominant" in text
            else [0.0, 1.0, 0.0]
            for text in texts
        ]

    with EmbeddingServer(embed=vectors) as api:
        api.enable(case)
        for backend in ("lexical", "semantic", "hybrid"):
            result = case.search(
                "needle", "--limit", "20", backend=backend, refresh="wait"
            )
            require(
                result["retrieval"]["effective_mode"] == backend, "retriever fell back"
            )
            sessions = [hit["provider_session_id"] for hit in result["results"]]
            require(
                len(sessions) == len(expected) and set(sessions) == expected,
                f"{backend} lost sessions behind a crowded candidate list",
            )
        events = case.search("needle", "--events", "--limit", "20", backend="hybrid")
        require(
            len(events["results"]) == 20, "event limit changed during grouped retrieval"
        )
        fixture.session("agreement")
        fixture.session("lexical-leader")
        fixture.message(
            "agreement", "agreement-lexical", "fusionneedle " + "padding " * 30
        )
        fixture.message("agreement", "agreement-semantic", "semanticfavor")
        fixture.message("lexical-leader", "leader", "fusionneedle lexicalleader")
        case.json_command("import", "--provider", "opencode", "--path", fixture.path)
        lexical = case.search("fusionneedle", backend="lexical")["results"]
        semantic = case.search("fusionneedle", backend="semantic", refresh="wait")[
            "results"
        ]
        require(
            [hit["provider_session_id"] for hit in lexical]
            == ["lexical-leader", "agreement"],
            "lexical fixture ranking changed",
        )
        require(
            semantic[0]["provider_session_id"] == "agreement",
            "semantic fixture ranking changed",
        )
        require(
            semantic[0]["trwl_event_id"] != lexical[1]["trwl_event_id"],
            "fixture needs different event anchors",
        )
        grouped = case.search("fusionneedle", backend="hybrid")["results"]
        require(
            grouped[0]["provider_session_id"] == "agreement",
            "different anchors did not reinforce their session",
        )
        require(
            abs(grouped[0]["score"] - (1 / 60 + 1 / 61)) < 1e-7,
            "session RRF used event ranks",
        )
        require(
            grouped[0]["trwl_event_id"] == semantic[0]["trwl_event_id"],
            "snippet did not use the stronger rank",
        )
        events = case.search(
            "fusionneedle", "--events", "--limit", "300", backend="hybrid"
        )["results"]
        by_id = {hit["trwl_event_id"]: hit for hit in events}
        require(
            lexical[1]["trwl_event_id"] in by_id, "event fusion lost lexical anchor"
        )
        require(
            abs(by_id[semantic[0]["trwl_event_id"]]["score"] - 1 / 60) < 1e-7,
            "event fusion combined distinct anchors",
        )
    case.check("grouped retrieval keeps other sessions behind 120 high-ranking events")
    case.check(
        "session fusion combines different anchors; event fusion keeps them separate"
    )


def semantic_cache(case):
    git(case.work, "init", "--quiet")
    fixture = OpenCodeFixture(case.providers / "cache.db", case.work)
    fixture.session("cache")
    fixture.message("cache", "first", "cachebeacon first")
    fixture.message("cache", "second", "cachebeacon second")
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    database = case.cache / "semantic" / "cache.sqlite3"
    with EmbeddingServer(embed=lambda texts: [[1.0, 0.0] for _ in texts]) as api:
        api.enable(case)

        probed_documents = False

        def no_transaction_during_inference(_path, _model, texts):
            nonlocal probed_documents
            if database.exists():
                with sqlite3.connect(database, timeout=0) as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("UPDATE selections SET touched=touched+1")
                    db.commit()
                    busy, _, _ = db.execute(
                        "PRAGMA wal_checkpoint(TRUNCATE)"
                    ).fetchone()
                    require(busy == 0, "inference holds a cache read transaction")
                probed_documents |= any(text.startswith("user:") for text in texts)

        api.request_hook = no_transaction_during_inference
        case.search("cachebeacon", backend="semantic", refresh="wait")
        require(
            probed_documents,
            "document inference did not exercise the transaction probe",
        )
        before = len(api.requests)
        cached = case.search("cachebeacon", backend="semantic", refresh="wait")
        require(
            cached["retrieval"]["semantic_status"] == "ready",
            "cached query lost coverage",
        )
        require(
            [req["texts"] for req in api.requests[before:]] == [["cachebeacon"]],
            "unchanged fill re-embedded history",
        )
        with sqlite3.connect(database) as db:
            db.execute(
                "DELETE FROM vectors WHERE key=(SELECT key FROM vectors ORDER BY key LIMIT 1)"
            )
        partial = case.search(
            "cachebeacon", "--events", backend="semantic", refresh="off"
        )
        require(
            partial["retrieval"]["semantic_status"] == "partial"
            and len(partial["results"]) == 1,
            "missing vector coverage was hidden",
        )
        with sqlite3.connect(database) as db:
            db.execute("UPDATE selections SET touched=0")
        fallback = case.search(
            "cachebeacon", "--events", backend="hybrid", refresh="off"
        )
        require(
            fallback["retrieval"]["effective_mode"] == "lexical"
            and fallback["retrieval"]["reason"],
            "expired selection did not report fallback",
        )
        require(len(fallback["results"]) == 2, "cache eviction removed history")
        case.command(
            "search",
            "cachebeacon",
            "--backend",
            "semantic",
            "--refresh",
            "off",
            expect_failure=True,
        )
        status = case.json_command("semantic", "status")
        require(status["state"] == "pending", "expired cache reported ready")
        ready = case.search(
            "cachebeacon", "--events", backend="semantic", refresh="wait"
        )
        require(
            ready["retrieval"]["semantic_status"] == "ready"
            and len(ready["results"]) == 2,
            "refill did not recover coverage",
        )

        def refresh():
            case.json_command(
                "import", "--provider", "opencode", "--path", fixture.path
            )
            case.command("setup", "--wait")

        def plans():
            with sqlite3.connect(database) as db:
                return dict(db.execute("SELECT event,input FROM prepared"))

        original = plans()
        require(len(original) == 2, "prepared plans were not persisted")
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE test_plan_writes (kind TEXT)")
            for table in ("documents", "chunks", "prepared"):
                for operation in ("INSERT", "UPDATE", "DELETE"):
                    db.execute(f"""CREATE TRIGGER test_{table}_{operation}
                        AFTER {operation} ON {table} BEGIN
                        INSERT INTO test_plan_writes VALUES ('{table}_{operation}'); END""")
        before = len(api.requests)
        fixture.part(
            "cache",
            "first",
            "tool",
            {
                "type": "tool",
                "tool": "shell",
                "callID": "call",
                "state": {
                    "status": "completed",
                    "input": {"command": "true"},
                    "output": "done",
                },
            },
        )
        refresh()
        require(
            len(api.requests) == before, "tool-only update embedded conversation text"
        )
        require(plans() == original, "tool-only update changed composed input")
        with sqlite3.connect(database) as db:
            require(
                db.execute("SELECT count(*) FROM test_plan_writes").fetchone()[0] == 0,
                "unchanged plan rows were rewritten",
            )

        fixture.message("cache", "reply", "use a bounded pool", role="assistant")
        refresh()
        paired = plans()
        require(
            set(paired) == set(original),
            "reply changed the paired user's result identity",
        )
        require(
            sum(paired[key] != original[key] for key in original) == 1,
            "reply did not invalidate exactly its paired input",
        )
        require(
            [req["texts"] for req in api.requests[before:]]
            == [["user:\ncachebeacon second\n\nassistant:\nuse a bounded pool"]],
            "reply reused a stale user-only vector",
        )

        moved = case.root / "moved-workspace"
        moved.mkdir()
        git(moved, "init", "--quiet")
        before = len(api.requests)
        with sqlite3.connect(fixture.path) as db:
            db.execute(
                "UPDATE session SET directory=?, time_updated=?",
                (str(moved), fixture.stamp()),
            )
        refresh()
        require(
            plans() == paired and len(api.requests) == before,
            "metadata edit re-embedded unchanged input",
        )
        moved_hits = case.search(
            "cachebeacon",
            "--events",
            "--workspace",
            moved,
            backend="semantic",
            refresh="off",
        )["results"]
        require(
            len(moved_hits) == 2
            and all(hit["workspace_root"] == str(moved) for hit in moved_hits),
            "reused plans kept stale workspace metadata",
        )
        require(
            not case.search(
                "cachebeacon",
                "--events",
                "--workspace",
                case.work,
                backend="semantic",
                refresh="off",
            )["results"],
            "old workspace retained semantic results",
        )

        with sqlite3.connect(fixture.path) as db:
            db.execute("DELETE FROM part WHERE message_id='first'")
            db.execute("DELETE FROM message WHERE id='first'")
        refresh()
        require(len(plans()) == 1, "deleted message retained its prepared input")
        remaining = case.search(
            "cachebeacon",
            "--events",
            "--workspace",
            moved,
            backend="semantic",
            refresh="off",
        )["results"]
        require(len(remaining) == 1, "deleted message retained its semantic result")
        with sqlite3.connect(database) as db:
            require(
                db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 1,
                "deleted message retained unreferenced vectors",
            )
            db.execute("DELETE FROM test_plan_writes")
            db.execute("UPDATE selections SET touched=0")
        case.json_command("semantic", "status")
        with sqlite3.connect(database) as db:
            require(
                db.execute(
                    "SELECT count(*) FROM test_plan_writes WHERE kind='prepared_DELETE'"
                ).fetchone()[0]
                == 1,
                "selection eviction retained prepared input",
            )
        case.check(
            "prepared plans reuse unchanged inputs while replies, metadata, deletions and eviction stay fresh"
        )
    case.check(
        "cache eviction reports partial/unavailable coverage and preserves history; waited fill recovers"
    )
    case.check("cached fills reuse vectors and inference holds no SQLite transaction")


def semantic_http(case):
    model_prerequisites(case)
    api = EmbeddingServer(case.model_dir)
    try:
        with api:
            fixture, expected = _fixture(case, refreshable=True)
            api.enable(case)
            for backend in ("semantic", "hybrid"):
                result = case.search(
                    QUERY,
                    "--events",
                    "--limit",
                    "3",
                    backend=backend,
                    refresh="wait",
                    timeout=30,
                )
                require(
                    result["retrieval"]["effective_mode"] == backend
                    and result["retrieval"]["semantic_status"] == "ready"
                    and any(
                        hit["trwl_event_id"] == expected["trwl_event_id"]
                        for hit in result["results"]
                    ),
                    f"HTTP {backend} missed the paraphrase target",
                )
                _show(case, expected)
            case.check(
                "real HTTP embeddings find the paraphrase target without a local model"
            )
            body = "httpappendbeacon The release pipeline stops when the container registry rejects expired credentials."
            fixture.message(SESSION, "msg_http_append", body)
            result = case.search(
                "Why did shipping our image fail after the registry login token timed out?",
                "--events",
                "--limit",
                "3",
                backend="semantic",
                refresh="wait",
                timeout=30,
            )
            added = case.search("httpappendbeacon", "--events")["results"]
            require(
                len(added) == 1,
                "HTTP refresh missed or duplicated the appended message",
            )
            require(
                result["retrieval"]["effective_mode"] == "semantic"
                and any(
                    hit["trwl_event_id"] == added[0]["trwl_event_id"]
                    for hit in result["results"]
                ),
                "HTTP embeddings did not incorporate new history",
            )
            shown = case.json_command("show", "event", added[0]["trwl_event_id"])[
                "event"
            ]
            require(shown["text"] == body, "HTTP result lost complete appended content")
            require(api.requests, "semantic queries never reached the HTTP service")
            case.check("HTTP semantic refresh incorporates new history")
    finally:
        write_json(case.root / "embedding-requests.json", api.requests)

    case.command(
        "search",
        QUERY,
        "--backend",
        "semantic",
        "--refresh",
        "wait",
        "--format",
        "json",
        expect_failure=True,
        timeout=15,
    )
    result = case.search(
        "poolbeacon", "--events", backend="hybrid", refresh="wait", timeout=15
    )
    require(
        result["retrieval"]["effective_mode"] == "lexical"
        and result["retrieval"]["semantic_status"] != "ready"
        and any(
            hit["trwl_event_id"] == expected["trwl_event_id"]
            for hit in result["results"]
        ),
        "unavailable HTTP service did not produce a reported lexical fallback",
    )
    _lexical_survives(case, expected)
    case.check(
        "unavailable HTTP embeddings preserve history and report lexical fallback"
    )
