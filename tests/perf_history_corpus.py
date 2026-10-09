#!/usr/bin/env python3
"""Generate deterministic, tool-heavy OpenCode history."""

import argparse
import hashlib
import json
import os
import platform
import random
import sqlite3
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from itertools import pairwise
from pathlib import Path

from perf_fixtures import _EPOCH_MS, _SCHEMA
from realm_corpus import encoded

TOPICS = (
    "cache connection pool timeout",
    "parser syntax tree validation",
    "routing request response headers",
    "storage transaction checkpoint recovery",
    "worker queue cancellation shutdown",
    "catalog search ranking pagination",
    "archive compression payload decoding",
    "release build dependency configuration",
)
LENGTHS = (8, 24, 48, 64, 80, 96, 128, 256)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total": sum(values),
        "min": values[0],
        **{f"p{q}": values[(len(values) - 1) * q // 100] for q in (50, 90, 99)},
        "max": values[-1],
    }


def size(rng, anchors):
    quantile = rng.random()
    for (lo, a), (hi, b) in pairwise(anchors):
        if quantile <= hi:
            return round(a + (b - a) * (quantile - lo) / (hi - lo))
    return anchors[-1][1]


def body(rng, topic, length):
    # Vary lines and identifiers rather than padding with one repeated sentence.
    # Include Unicode to exercise byte sizes and full-content retrieval.
    chunks = [f"Investigate {topic}. Preserve café and 日本語 in the result.\n"]
    used = len(chunks[0].encode())
    while used < length:
        line = (
            f"{rng.getrandbits(64):016x} {rng.choice(topic.split())}: "
            f"{rng.choice(('inspect', 'compare', 'update', 'verify'))} item "
            f"{rng.randrange(100000)}; expected={rng.randrange(1000)} "
            f"observed={rng.randrange(1000)}\n"
        )
        chunks.append(line)
        used += len(line)
    return "".join(chunks).encode()[:length].decode("utf-8", errors="ignore")


def generate(output, *, sessions=32, length_multiplier=1, workspaces=4, seed=0):
    if sessions < 1 or length_multiplier < 1 or not 1 <= workspaces <= sessions:
        raise ValueError(
            "require positive sessions/length multiplier and 1..sessions workspaces"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "opencode.db"
    workspace_root = output.resolve() / "workspaces"
    for index in range(workspaces):
        subprocess.run(
            [
                "git",
                "-c",
                "init.defaultBranch=main",
                "init",
                "--quiet",
                "--template=",
                str(workspace_root / f"project-{index:02d}"),
            ],
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
            check=True,
        )
    rng = random.Random(seed)
    sizes = defaultdict(list)
    counts = Counter()
    roles = Counter()
    states = Counter()
    row_hashes = {table: hashlib.sha256() for table in ("session", "message", "part")}
    stamp = _EPOCH_MS
    part_number = 0

    def identity(kind, ordinal):
        return (
            f"{kind}_"
            + hashlib.sha256(
                f"history-v1/{seed}/{kind}/{ordinal}".encode()
            ).hexdigest()[:32]
        )

    def insert(db, table, row):
        data = encoded(
            [
                value.replace(str(workspace_root), "$WORKSPACE_ROOT")
                if isinstance(value, str)
                else value
                for value in row
            ]
        )
        row_hashes[table].update(len(data).to_bytes(8, "big"))
        row_hashes[table].update(data)
        db.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", row)
        counts[table] += 1

    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(_SCHEMA)
        for s in range(sessions):
            session = identity("ses", s)
            # Each workspace has its own interleaved task trees. Two of each five
            # sessions continue the preceding task, including depth-two children.
            ordinal = s // workspaces
            parent = identity("ses", s - workspaces) if ordinal % 5 in (3, 4) else None
            workspace = str(workspace_root / f"project-{s % workspaces:02d}")
            messages = LENGTHS[s % len(LENGTHS)] * length_multiplier
            sizes["messages_per_session"].append(messages)
            insert(
                db,
                "session",
                (
                    session,
                    parent,
                    workspace,
                    "main",
                    "explore" if parent else "build",
                    stamp,
                    stamp + messages * 1000,
                ),
            )
            for m in range(messages):
                stamp += 1000
                message = identity("msg", f"{s}/{m}")
                role = "user" if m % 7 == 0 else "assistant"
                roles[role] += 1
                insert(
                    db,
                    "message",
                    (message, session, stamp, stamp, json.dumps({"role": role})),
                )
                topic = TOPICS[(s + m // 7) % len(TOPICS)]
                parts = []
                if role == "assistant":
                    parts.append({"type": "step-start", "snapshot": f"{s:08x}{m:08x}"})
                    for _ in range(rng.choice((0, 1, 1, 1, 2, 2, 2, 3))):
                        length = size(
                            rng,
                            ((0, 0), (0.65, 0), (0.9, 1024), (0.99, 24576), (1, 65536)),
                        )
                        text = body(rng, topic, length)
                        sizes["reasoning_bytes"].append(len(text.encode()))
                        opaque = rng.randbytes(rng.randrange(128, 2048)).hex()
                        parts.append(
                            {
                                "type": "reasoning",
                                "text": text,
                                "metadata": {"synthetic": {"opaque": opaque}},
                            }
                        )
                if role == "user" or m % 7 == 6:
                    length = size(
                        rng, ((0, 8), (0.5, 200), (0.9, 2048), (0.99, 8192), (1, 12288))
                    )
                    text = body(rng, topic, length)
                    sizes["text_bytes"].append(len(text.encode()))
                    parts.append({"type": "text", "text": text})
                else:
                    for t in range(rng.choice((1, 1, 2, 2, 3, 4))):
                        tool = rng.choice(
                            ("read", "read", "shell", "apply_patch", "search")
                        )
                        filename = f"src/module-{m % 17}.rs"
                        length = size(
                            rng,
                            ((0, 2), (0.5, 200), (0.9, 1100), (0.99, 6000), (1, 16000)),
                        )
                        payload = body(rng, topic, length)
                        args = {
                            "read": {"filePath": filename, "offset": m, "limit": 200},
                            "shell": {"command": f"printf '%s' '{payload}'"},
                            "search": {"query": topic, "context": payload},
                            "apply_patch": {
                                "patchText": f"*** Begin Patch\n*** Update File: {filename}\n@@\n+{payload}\n*** End Patch"
                            },
                        }[tool]
                        chance = rng.random()
                        status = (
                            "running"
                            if chance < 0.003
                            else "error"
                            if chance < 0.05
                            else "completed"
                        )
                        states[status] += 1
                        state = {
                            "status": status,
                            "input": args,
                            "time": {"start": stamp},
                        }
                        if status != "running":
                            length = size(
                                rng,
                                (
                                    (0, 0),
                                    (0.1, 128),
                                    (0.5, 1200),
                                    (0.9, 12000),
                                    (0.99, 32000),
                                    (1, 65536),
                                ),
                            )
                            text = body(rng, topic, length)
                            state["error" if status == "error" else "output"] = text
                            state["time"]["end"] = stamp + 1
                            sizes["tool_result_bytes"].append(len(text.encode()))
                        sizes["tool_input_bytes"].append(len(json.dumps(args).encode()))
                        parts.append(
                            {
                                "type": "tool",
                                "callID": f"call-{s}-{m}-{t}",
                                "tool": tool,
                                "state": state,
                            }
                        )
                        if tool == "apply_patch":
                            parts.append(
                                {
                                    "type": "patch",
                                    "hash": f"{s:08x}{m:08x}{t:08x}",
                                    "files": [filename],
                                }
                            )
                if role == "assistant":
                    parts.append(
                        {
                            "type": "step-finish",
                            "reason": "stop" if m % 7 == 6 else "tool-calls",
                            "tokens": {"input": 1024 + m, "output": 64 + m},
                        }
                    )
                sizes["parts_per_message"].append(len(parts))
                for data in parts:
                    counts["part:" + data["type"]] += 1
                    part_number += 1
                    insert(
                        db,
                        "part",
                        (
                            identity("prt", part_number),
                            message,
                            session,
                            stamp,
                            stamp,
                            json.dumps(data, ensure_ascii=False),
                        ),
                    )
        children = db.execute(
            "SELECT count(*) FROM session WHERE parent_id IS NOT NULL"
        ).fetchone()[0]
    with path.open("rb") as stream:
        database_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "corpus": "synthetic-history-v1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "parameters": {
            "sessions": sessions,
            "length_multiplier": length_multiplier,
            "workspaces": workspaces,
            "seed": seed,
        },
        "counts": dict(counts),
        "roles": dict(roles),
        "tool_states": dict(states),
        "child_sessions": children,
        "distributions": {
            key: distribution(values) for key, values in sorted(sizes.items())
        },
        "rows_sha256": {
            table: value.hexdigest() for table, value in row_hashes.items()
        },
        "database_bytes": path.stat().st_size,
        "database_sha256": database_sha256,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--length-multiplier", type=int, default=1)
    parser.add_argument("--workspaces", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        args.sessions < 1
        or args.length_multiplier < 1
        or not 1 <= args.workspaces <= args.sessions
    ):
        parser.error(
            "require positive sessions/length multiplier and 1..sessions workspaces"
        )
    print(json.dumps(generate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
