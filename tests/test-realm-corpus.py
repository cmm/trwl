#!/usr/bin/env python3
"""Check native corpus preservation, task-tree sampling and source isolation."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from realm_corpus import generate, select_sessions

SCRATCH = Path(__file__).resolve().parents[1] / ".cache"
HISTORY_TABLES = ("session", "message", "part")


class CorpusTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix="trwl-realm-generator-test-", dir=SCRATCH
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.db"
        with closing(sqlite3.connect(self.source)) as db, db:
            db.executescript("""
                CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, directory TEXT, title TEXT);
                CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
                CREATE TABLE part (id TEXT PRIMARY KEY, session_id TEXT, message_id TEXT, time_created INTEGER, data TEXT);
                CREATE INDEX part_session_idx ON part(session_id);
                CREATE TABLE credential (data TEXT);
                INSERT INTO credential VALUES ('fixture credential must not be copied');
            """)
            for n, parent in (
                (0, None),
                (1, "s0"),
                (2, "s1"),
                (3, None),
                (4, "s3"),
                (5, "s4"),
            ):
                sid = f"s{n}"
                db.execute(
                    "INSERT INTO session VALUES (?,?,?,?)",
                    (sid, parent, "/workspace", f"Title {n}"),
                )
                db.execute(
                    "INSERT INTO message VALUES (?,?,?,?)",
                    (f"m{n}", sid, n, '{"role":"assistant"}'),
                )
                data = json.dumps(
                    {
                        "type": "tool",
                        "tool": "read",
                        "state": {
                            "input": {"filePath": "résumé.txt"},
                            "output": "日本語 λ\n" * (n + 1),
                            "error": None,
                        },
                    }
                )
                db.execute(
                    "INSERT INTO part VALUES (?,?,?,?,?)",
                    (f"p{n}", sid, f"m{n}", n, data),
                )

    def test_full_preserves_all_history_rows_and_indexes_without_credentials(self):
        before = self.source.read_bytes()
        output = self.root / "full"
        manifest = generate(self.source, output)
        self.assertEqual(self.source.read_bytes(), before)
        with (
            closing(sqlite3.connect(self.source)) as src,
            closing(sqlite3.connect(output / "opencode.db")) as dst,
        ):
            self.assertEqual(
                {
                    r[0]
                    for r in dst.execute(
                        "SELECT name FROM sqlite_schema WHERE type='table'"
                    )
                },
                set(HISTORY_TABLES),
            )
            for table in HISTORY_TABLES:
                a = src.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                b = dst.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                self.assertEqual(a, b)
                self.assertEqual(manifest["tables"][table]["rows"], len(b))
            self.assertIsNotNone(
                dst.execute(
                    "SELECT name FROM sqlite_schema WHERE type='index' AND name='part_session_idx'"
                ).fetchone()
            )
        self.assertEqual((output / "opencode.db").stat().st_mode & 0o777, 0o600)

    def test_sample_preserves_whole_task_trees_and_is_deterministic(self):
        a = generate(self.source, self.root / "a", 1)
        b = generate(self.source, self.root / "b", 1)
        self.assertEqual(a["tables"], b["tables"])
        with (
            closing(sqlite3.connect(self.root / "a/opencode.db")) as db,
            closing(sqlite3.connect(self.source)) as src,
        ):
            parents = dict(db.execute("SELECT id,parent_id FROM session"))
            self.assertIn(set(parents), ({"s0", "s1", "s2"}, {"s3", "s4", "s5"}))
            for table in ("message", "part"):
                expected = src.execute(
                    f"SELECT * FROM {table} WHERE session_id IN (?,?,?) ORDER BY id",
                    tuple(parents),
                ).fetchall()
                self.assertEqual(
                    db.execute(f"SELECT * FROM {table} ORDER BY id").fetchall(),
                    expected,
                )

    def test_refuses_overwrite_and_nonpositive_limit(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "keep").write_text("do not change\n")
        with self.assertRaises(FileExistsError):
            generate(self.source, output)
        self.assertEqual((output / "keep").read_text(), "do not change\n")
        with self.assertRaises(ValueError):
            generate(self.source, self.root / "invalid", 0)

    def test_cycle_is_rejected_and_input_order_does_not_matter(self):
        with self.assertRaises(ValueError):
            select_sessions({"a": "b", "b": "a"}, 1)
        parents = {"a": None, "b": "a", "c": None}
        self.assertEqual(
            select_sessions(parents, 1),
            select_sessions(dict(reversed(list(parents.items()))), 1),
        )


if __name__ == "__main__":
    unittest.main()
