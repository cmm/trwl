"""Deterministic provider inputs and untimed public-CLI performance checks."""

import hashlib
import json
import sqlite3
import time
import uuid
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from e2e_support import require

_EPOCH = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
_EPOCH_MS = int(_EPOCH.timestamp()) * 1000
_SCHEMA = """
    BEGIN;
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
"""
_WORDS = (
    "cache",
    "parser",
    "routing",
    "buffer",
    "storage",
    "timeout",
    "catalog",
    "query",
    "archive",
    "release",
    "worker",
    "request",
    "snapshot",
    "client",
    "queue",
    "record",
)


class Corpus:
    """Own native inputs under root; workspace is the only embedded directory."""

    def __init__(
        self,
        root: Path,
        workspace: Path,
        provider: str,
        shape: str,
        messages: int,
        append_count: int = 3,
        *,
        message_bytes: int = 256,
        long_message_bytes: int | None = None,
    ):
        if provider not in ("opencode", "codex") or shape not in ("many", "long"):
            raise ValueError("unsupported provider or corpus shape")
        if messages < 1 or append_count < 1:
            raise ValueError("messages and append_count must be positive")
        if message_bytes < 128:
            raise ValueError("message_bytes must be at least 128")
        if long_message_bytes is not None and long_message_bytes <= message_bytes:
            raise ValueError("long_message_bytes must exceed message_bytes")
        self.root = Path(root)
        self.workspace = Path(workspace)
        self.provider = provider
        self.shape = shape
        self.append_count = append_count
        self.message_bytes = message_bytes
        self.long_message_bytes = long_message_bytes
        self._messages = messages
        self.expected = {}
        self.root.mkdir(parents=True, exist_ok=True)
        count = (messages + 19) // 20 if shape == "many" else 1
        self._sessions = [self._id("session", i) for i in range(count)]
        records = [
            (i, self._sessions[i // 20 if shape == "many" else 0], self._text(i))
            for i in range(messages)
        ]
        if provider == "opencode":
            self.path = self.root / "opencode.db"
            if self.path.exists():
                raise FileExistsError(self.path)
            self._files = [self.path]
            self._write_opencode(records, initial=True)
        else:
            self.path = self.root / "sessions"
            self.path.mkdir()
            self._files = [
                self.path / f"rollout-2026-09-21T12-00-00-{session}.jsonl"
                for session in self._sessions
            ]
            by_session = {session: [] for session in self._sessions}
            for record in records:
                by_session[record[1]].append(record)
            for session, path in zip(self._sessions, self._files, strict=True):
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(self._meta(session)) + "\n")
                    self._write_codex(stream, by_session[session])
        self.expected.update((text, session) for _, session, text in records)
        self.target_text = records[0][2]
        self.query = self.target_text.split()[0]

    def _id(self, kind, ordinal):
        value = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"trwl-perf/{self.provider}/{self.shape}/{kind}/{ordinal}",
        )
        if self.provider == "opencode":
            return {"session": "ses_", "message": "msg_", "part": "prt_"}[
                kind
            ] + value.hex
        return str(value)

    def _text(self, ordinal):
        # Alternate short and long user/reply pairs, not lengths within a pair.
        message_bytes = (
            self.long_message_bytes
            if self.long_message_bytes is not None and (ordinal // 2) % 2
            else self.message_bytes
        )
        marker = f"trwlperf{self.provider}{self.shape}event{ordinal:08d}"
        digest = hashlib.sha256(marker.encode()).digest()
        words = " ".join(_WORDS[byte % len(_WORDS)] for byte in digest)
        text = (
            f"{marker} Review {ordinal}: investigate the {_WORDS[ordinal % len(_WORDS)]} "
            f"behavior before changing request handling.\n"
            f"Preserve complete transcript content and compare these details: {words}. "
        )
        text += (words + "\n") * (message_bytes // len(words) + 1)
        return text[: message_bytes - 1] + "."

    def _write_opencode(self, records, *, initial=False):
        # One transaction includes schema and all initial rows; append also batches.
        with closing(sqlite3.connect(self.path)) as db, db:
            if initial:
                db.executescript(_SCHEMA)
                first = {}
                for ordinal, session, _ in records:
                    first.setdefault(session, _EPOCH_MS + (ordinal + 1) * 1000)
                db.executemany(
                    "INSERT INTO session VALUES (?, NULL, ?, 'main', 'build', ?, ?)",
                    [
                        (session, str(self.workspace), stamp, stamp)
                        for session, stamp in first.items()
                    ],
                )
            messages = []
            parts = []
            updated = {}
            for ordinal, session, text in records:
                stamp = _EPOCH_MS + (ordinal + 1) * 1000
                message = self._id("message", ordinal)
                role = "assistant" if ordinal % 2 else "user"
                messages.append(
                    (message, session, stamp, stamp + 1, json.dumps({"role": role}))
                )
                parts.append(
                    (
                        self._id("part", ordinal),
                        message,
                        session,
                        stamp + 1,
                        stamp + 1,
                        json.dumps({"type": "text", "text": text}),
                    )
                )
                updated[session] = stamp + 1
            db.executemany("INSERT INTO message VALUES (?, ?, ?, ?, ?)", messages)
            db.executemany("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)", parts)
            db.executemany(
                "UPDATE session SET time_updated = ? WHERE id = ?",
                [(stamp, session) for session, stamp in updated.items()],
            )

    def _meta(self, session):
        timestamp = _EPOCH.isoformat().replace("+00:00", "Z")
        return {
            "timestamp": timestamp,
            "type": "session_meta",
            "payload": {
                "id": session,
                "timestamp": timestamp,
                "cwd": str(self.workspace),
                "originator": "codex_cli_rs",
                "cli_version": "0.1.0",
                "source": "cli",
                "model_provider": "openai",
            },
        }

    def _write_codex(self, stream, records):
        for ordinal, _, text in records:
            role = "assistant" if ordinal % 2 else "user"
            record = {
                "timestamp": (_EPOCH + timedelta(seconds=ordinal + 1))
                .isoformat()
                .replace("+00:00", "Z"),
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
            stream.write(json.dumps(record) + "\n")

    def append(self):
        """Add one batch of new messages to the first existing conversation."""
        records = [
            (i, self._sessions[0], self._text(i))
            for i in range(self._messages, self._messages + self.append_count)
        ]
        if self.provider == "opencode":
            self._write_opencode(records)
        else:
            with self._files[0].open("a", encoding="utf-8", newline="\n") as stream:
                self._write_codex(stream, records)
        self.expected.update((text, session) for _, session, text in records)
        self._messages += self.append_count

    def manifest(self):
        """Hash only known provider files, framing relative names and file bytes."""
        digest = hashlib.sha256()
        size = messages = conversations = 0
        for path in sorted(self._files):
            name = path.relative_to(self.root).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            size += len(content)
            if self.provider == "codex":
                for line in content.splitlines():
                    record = json.loads(line)
                    conversations += record["type"] == "session_meta"
                    messages += (
                        record["type"] == "response_item"
                        and record["payload"]["type"] == "message"
                    )
        if self.provider == "opencode":
            with closing(
                sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            ) as db:
                messages = db.execute("SELECT count(*) FROM message").fetchone()[0]
                conversations = db.execute("SELECT count(*) FROM session").fetchone()[0]
        return {
            "format": "opencode_sqlite"
            if self.provider == "opencode"
            else "codex_rollout_jsonl",
            "provider": self.provider,
            "shape": self.shape,
            "messages": messages,
            "conversations": conversations,
            "input_files": len(self._files),
            "input_bytes": size,
            "message_bytes": self.message_bytes,
            "long_message_bytes": self.long_message_bytes,
            "input_sha256": digest.hexdigest(),
        }


def _identity(row):
    identity = (row["trwl_event_id"], row["trwl_session_id"])
    require(
        all(isinstance(value, str) and value for value in identity), "missing trwl IDs"
    )
    return identity


def verify_corpus(case, corpus, previous=None) -> dict[str, tuple[str, str]]:
    """Enumerate all stored events, requiring full text and completed public pages."""
    rows = []
    cursors = set()
    cursor_args = ()
    deadline = time.monotonic() + 120
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "corpus enumeration timed out")
        result = case.command(
            "list",
            "events",
            "--workspace",
            str(corpus.workspace),
            "--limit",
            str(len(corpus.expected) + 1),
            "--format",
            "jsonl",
            *cursor_args,
            timeout=min(30, remaining),
        )
        require(result.returncode == 0, "corpus enumeration command failed")
        records = [json.loads(line) for line in result.stdout.splitlines()]
        require(
            records
            and records[-1]["record_type"] == "event_range_completion"
            and all(row["record_type"] == "event_range_event" for row in records[:-1]),
            "list stream omitted its final completion or emitted unexpected records",
        )
        rows.extend(row["event"] for row in records[:-1])
        completion = records[-1]
        cursor = completion.get("next_cursor")
        if completion["terminal"] is True:
            require(
                completion["truncated"] is False and not cursor,
                "incomplete terminal list",
            )
            break
        require(
            completion["terminal"] is False
            and completion["truncated"] is True
            and isinstance(cursor, str)
            and cursor
            and cursor not in cursors
            and len(records) > 1,
            "list did not complete or advance its continuation",
        )
        cursors.add(cursor)
        cursor_args = ("--cursor", cursor)
    require(
        Counter(row["text"] for row in rows) == Counter(corpus.expected.keys()),
        "corpus has missing, extra, changed or duplicate full message bodies",
    )
    identities = {}
    sessions = {}
    for row in rows:
        text = row["text"]
        require(
            row["provider"] == corpus.provider
            and row["provider_session_id"] == corpus.expected[text]
            and row["event_type"] == "message",
            "listed message has wrong provider, native session or event type",
        )
        identity = _identity(row)
        native = corpus.expected[text]
        require(
            sessions.setdefault(native, identity[1]) == identity[1],
            "split native session",
        )
        identities[text] = identity
    require(len(set(sessions.values())) == len(sessions), "native sessions collapsed")
    require(
        len({identity[0] for identity in identities.values()}) == len(rows),
        "corpus has duplicate event IDs",
    )
    if previous is not None:
        require(
            all(
                identities.get(text) == identity for text, identity in previous.items()
            ),
            "unchanged messages lost their event or session IDs",
        )
    return identities


def verify_search(case, corpus, result, identities):
    """Check parsed timed search output without launching another command."""
    require(result["query"] == corpus.query, "search used the wrong marker")
    hits = result["results"]
    require(len(hits) == 1, "search did not return exactly the target message")
    require(
        _identity(hits[0]) == identities[corpus.target_text], "search target IDs differ"
    )


def verify_show(case, corpus, result, identities):
    """Check parsed timed show-event output, including the complete target body."""
    event = result["event"]
    require(
        _identity(event) == identities[corpus.target_text]
        and event["text"] == corpus.target_text
        and event["provider"] == corpus.provider
        and event["provider_session_id"] == corpus.expected[corpus.target_text]
        and event["event_type"] == "message",
        "show changed target text, IDs, provider or native session",
    )
