#!/usr/bin/env python3
"""Generate full or sampled OpenCode corpora from a consistent read transaction."""

import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

TABLES = ("session", "message", "part")


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def encoded(row):
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: {"sqlite_blob_hex": value.hex()},
    ).encode("utf-8")


def select_sessions(parents, root_limit):
    """Sample whole task trees; input ordering does not influence selection."""
    roots = {}
    for session in parents:
        seen = set()
        current = session
        while parents.get(current) in parents:
            if current in seen:
                raise ValueError("cycle in session parent relationships")
            seen.add(current)
            current = parents[current]
        roots[session] = current
    ordered = sorted(
        set(roots.values()),
        key=lambda value: (hashlib.sha256(value.encode()).digest(), value),
    )
    selected = set(ordered if root_limit is None else ordered[:root_limit])
    return {session for session, root in roots.items() if root in selected}, len(
        selected
    )


def generate(source, output, root_limit=None):
    if root_limit is not None and root_limit < 1:
        raise ValueError("root limit must be positive")
    source = Path(source).resolve(strict=True)
    output = Path(output)
    # Refuse existing destinations, including one that contains a previous run.
    output.mkdir(mode=0o700)
    target = output / "opencode.db"
    manifest = {
        "source": str(source),
        "format": "opencode_sqlite",
        "root_limit": root_limit,
        "tables": {},
    }
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
        src.execute("PRAGMA query_only=ON")
        src.execute("BEGIN")
        parents = dict(src.execute("SELECT id,parent_id FROM session"))
        selected, root_count = select_sessions(parents, root_limit)
        manifest.update(
            source_sessions=len(parents),
            selected_sessions=len(selected),
            selected_roots=root_count,
        )
        with closing(sqlite3.connect(target)) as dst, dst:
            target.chmod(0o600)
            # Only the provider's three history tables are copied. Account,
            # credential, event-log and unrelated application tables stay out.
            for table in TABLES:
                ddl = src.execute(
                    "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if not ddl or not ddl[0]:
                    raise ValueError(f"missing {table} table schema")
                dst.execute(ddl[0])
                columns = [
                    row[1] for row in src.execute(f"PRAGMA table_info({quote(table)})")
                ]
                key = columns.index("id" if table == "session" else "session_id")
                insert = f"INSERT INTO {quote(table)} VALUES ({','.join('?' for _ in columns)})"
                digest = hashlib.sha256()
                count = size = 0
                for row in src.execute(f"SELECT * FROM {quote(table)} ORDER BY id"):
                    if root_limit is not None and row[key] not in selected:
                        continue
                    dst.execute(insert, row)
                    data = encoded(row)
                    digest.update(len(data).to_bytes(8, "big"))
                    digest.update(data)
                    count += 1
                    size += len(data)
                manifest["tables"][table] = {
                    "columns": columns,
                    "rows": count,
                    "encoded_row_bytes": size,
                    "rows_sha256": digest.hexdigest(),
                }
                for (sql,) in src.execute(
                    "SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name=? AND sql IS NOT NULL ORDER BY name",
                    (table,),
                ):
                    dst.execute(sql)
            if dst.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("generated corpus failed SQLite quick_check")
    manifest["database_bytes"] = target.stat().st_size
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "manifest.json").chmod(0o600)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root-limit",
        type=int,
        help="Keep this many deterministically selected root tasks and all their children",
    )
    args = parser.parse_args()
    os.umask(0o077)
    manifest = generate(args.source, args.output, args.root_limit)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sessions": manifest["selected_sessions"],
                "rows": {
                    name: table["rows"] for name, table in manifest["tables"].items()
                },
                "database_bytes": manifest["database_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
