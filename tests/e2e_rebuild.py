"""Opt-in incompatible-index recovery through importing CLI workflows."""

import json
import time
from collections import Counter
from pathlib import Path

from e2e_http import EmbeddingServer
from e2e_support import OpenCodeFixture, codex_fixture, require, sha256, write_json


def _rebuild_setting(case, value):
    config = case.config / "config.toml"
    text = "".join(
        line
        for line in config.read_text().splitlines(keepends=True)
        if not line.startswith("rebuild_incompatible =")
    )
    require("[indexing]\n" in text, "fixture lost its indexing configuration")
    if value is not None:
        text = text.replace(
            "[indexing]\n", f"[indexing]\nrebuild_incompatible = {value}\n", 1
        )
    config.write_text(text)


def _obsolete(case, kind="record"):
    # Mutate CLI-generated index files; the indexed histories are fixtures.
    path = case.data / "index/meta.json"
    metadata = json.loads(path.read_bytes())
    if kind == "format":
        segment = metadata["segments"][0]["segment_id"].replace("-", "")
        store = path.parent / f"{segment}.store"
        content = store.read_bytes()
        footer_length = int.from_bytes(content[-8:-4], "little")
        require(
            0 < footer_length <= len(content) - 8, "invalid generated segment footer"
        )
        body = content[: -8 - footer_length]
        footer = json.loads(content[-8 - footer_length : -8])
        require(
            footer["version"]["index_format_version"] > 0,
            "segment format is already obsolete",
        )
        footer["version"]["index_format_version"] = 0
        encoded = json.dumps(footer, separators=(",", ":")).encode()
        # The checksum covers only the unchanged body; retain its value and magic.
        store.write_bytes(
            body + encoded + len(encoded).to_bytes(4, "little") + content[-4:]
        )
        return
    if kind == "record":
        manifest = json.loads(metadata["payload"])
        require(manifest["record_version"] > 0, "fixture has no prior record version")
        manifest["record_version"] -= 1
        metadata["payload"] = json.dumps(manifest)
    else:
        require(kind == "schema", f"unknown incompatibility fixture: {kind}")
        metadata["schema"][0]["name"] = "obsolete_fixture_field"
    write_json(path, metadata)


def _snapshot(case):
    index = case.data / "index"
    return {
        str(path.relative_to(index)): sha256(path) if path.is_file() else None
        for path in index.rglob("*")
    }


def _data_directories(case):
    return {path.name for path in case.data.iterdir() if path.is_dir()}


def _history(case, expected):
    page = case.json_command(
        "list",
        "events",
        "--workspace",
        "**",
        "--include-current-session",
        "--limit",
        "100",
    )
    rows = page["events"]
    require(page["next_cursor"] is None, "rebuild fixture history was truncated")
    require(
        Counter(
            (row["text"], row["provider"], row["provider_session_id"]) for row in rows
        )
        == Counter(expected),
        "rebuild lost, duplicated or changed history, or retained an unselected source",
    )
    require(
        len({row["trwl_event_id"] for row in rows}) == len(rows),
        "rebuilt history contains duplicate event identities",
    )
    return {row["text"]: (row["trwl_event_id"], row["trwl_session_id"]) for row in rows}


def _report(report, paths):
    require(report["outcome"] == "completed", "rebuild did not complete")
    require(
        Counter(source["path"] for source in report["sources"])
        == Counter(str(path) for path in paths),
        "rebuild missed a source or imported a configured alias more than once",
    )


class RebuildFixture:
    def __init__(self, case):
        self.configured = OpenCodeFixture(case.providers / "configured.db", case.work)
        self.configured.session("ses_rebuild_configured")
        configured_text = "rebuildbeacon Configured OpenCode history.\nKeep this line."
        self.configured.message(
            "ses_rebuild_configured", "msg_configured", configured_text
        )
        self.discovered = codex_fixture(
            Path(case.env["CODEX_HOME"]) / "sessions/rollout-rebuild.jsonl",
            case.work,
            "550e8400-e29b-41d4-a716-446655440301",
            [("user", "rebuildbeacon Discovered Codex history.\nUnicode: café.")],
        )
        self.expected = [
            (configured_text, "opencode", "ses_rebuild_configured"),
            (
                "rebuildbeacon Discovered Codex history.\nUnicode: café.",
                "codex",
                "550e8400-e29b-41d4-a716-446655440301",
            ),
        ]
        self.explicit = OpenCodeFixture(case.providers / "explicit.db", case.work)
        self.explicit.session("ses_rebuild_explicit")
        explicit_text = "rebuildbeacon Explicit one-time OpenCode import."
        self.explicit.message("ses_rebuild_explicit", "msg_explicit", explicit_text)
        self.extra = [(explicit_text, "opencode", "ses_rebuild_explicit")]
        self.paths = [self.configured.path, self.discovered]
        for name in ("configured", "configured-alias"):
            case.command(
                "sources",
                "add",
                name,
                "--provider",
                "opencode",
                "--root",
                self.configured.path,
            )
        config = case.config / "config.toml"
        config.write_text(
            config.read_text().replace("automatic = false", "automatic = true")
        )

        seed = OpenCodeFixture(case.providers / "seed.db", case.work)
        seed.session("ses_rebuild_seed")
        seed.message(
            "ses_rebuild_seed", "msg_seed", "oldseedbeacon Unconfigured history."
        )
        case.command("import", "--provider", "opencode", "--path", seed.path)


def index_rebuild(case):
    fixture = RebuildFixture(case)
    _obsolete(case)
    before = _snapshot(case)
    for value in (None, "false"):
        _rebuild_setting(case, value)
        case.command("setup", "--wait", expect_failure=True)
        require(
            _snapshot(case) == before, "disabled recovery changed the obsolete index"
        )
    case.check("absent and false rebuild settings refuse recovery through the CLI")

    _rebuild_setting(case, "true")
    require(
        case.json_command("status")["lexical"]["state"] == "unavailable"
        and case.json_command("index", "status")["state"] == "unavailable",
        "read-only status hid the unusable index",
    )
    case.command(
        "search",
        "rebuildbeacon",
        "--backend",
        "lexical",
        "--refresh",
        "off",
        expect_failure=True,
    )
    require(
        _snapshot(case) == before, "read-only commands rebuilt or changed the index"
    )
    case.check(
        "opt-in status and refresh-off search leave an incompatible index untouched"
    )

    try:
        for value in ('"true"', "1", "[]"):
            _rebuild_setting(case, value)
            case.command("status", expect_failure=True)
            require(_snapshot(case) == before, "invalid config changed the index")
    finally:
        _rebuild_setting(case, "true")
    case.check("non-boolean rebuild settings fail config loading")

    explicit = ("--provider", "opencode", "--path", fixture.explicit.path)
    commands = [
        (("setup", "--wait"), False),
        (
            (
                "search",
                "rebuildbeacon",
                "--events",
                "--backend",
                "lexical",
                "--refresh",
                "wait",
            ),
            False,
        ),
        (("import", "--all"), False),
        (("import", *explicit), True),
        (("import", "--all", "--full"), False),
        (("import", *explicit, "--full"), True),
    ]
    for attempt, (args, include_explicit) in enumerate(commands):
        if attempt:
            _obsolete(case, "schema" if attempt % 2 else "record")
        directories = _data_directories(case)
        result = case.json_command(*args)
        expected = fixture.expected + (fixture.extra if include_explicit else [])
        if args[0] == "search":
            require(
                len(result["results"]) == len(expected),
                "waited search missed rebuilt history",
            )
        else:
            _report(
                result,
                fixture.paths + ([fixture.explicit.path] if include_explicit else []),
            )
        identities = _history(case, expected)
        require(
            _data_directories(case) == directories,
            "successful recovery left a staging or old-index directory behind",
        )
        if args[0] == "import":
            case.command(*args)
            require(
                _history(case, expected) == identities,
                "repeated import changed rebuilt identities",
            )
        case.check(
            f"{' '.join(map(str, args))}: complete, deduplicated source union after recovery"
        )

    _obsolete(case, "format")
    _rebuild_setting(case, "false")
    before = _snapshot(case)
    case.command("setup", "--wait", expect_failure=True)
    require(
        _snapshot(case) == before,
        "disabled recovery changed an obsolete Tantivy segment",
    )
    _rebuild_setting(case, "true")
    _report(
        case.json_command("import", *explicit), [*fixture.paths, fixture.explicit.path]
    )
    require(
        _history(case, expected) == identities,
        "segment-format recovery changed original history",
    )
    require(
        len(case.search("rebuildbeacon", "--events")["results"]) == len(expected),
        "segment-format recovery did not restore searchable history",
    )
    require(
        _data_directories(case) == directories,
        "segment-format recovery left a staging or old index",
    )
    case.check(
        "obsolete Tantivy segment footer: disabled refusal preserves bytes; enabled import restores searchable history"
    )


def rebuild_failure(case):
    fixture = RebuildFixture(case)
    _rebuild_setting(case, "true")
    _obsolete(case)
    before = _snapshot(case)
    directories = _data_directories(case)
    archived = fixture.configured.path.with_suffix(".saved")
    fixture.configured.path.rename(archived)
    args = ("import", "--provider", "opencode", "--path", fixture.explicit.path)
    try:
        failed = case.command(*args, "--format", "json", expect_failure=None)
        require(failed.returncode >= 0, "failed rebuild terminated by a signal")
        if failed.returncode == 0:
            require(
                json.loads(failed.stdout)["outcome"]
                == "completed_with_source_failures",
                "failed rebuild reported success",
            )
        require(
            _snapshot(case) == before, "failed rebuild changed the original index bytes"
        )
        require(
            _data_directories(case) == directories,
            "failed rebuild left a staging directory",
        )
    finally:
        archived.rename(fixture.configured.path)
    case.check(
        "a missing configured source prevents explicit-import recovery and preserves every original file"
    )

    _report(case.json_command(*args), [*fixture.paths, fixture.explicit.path])
    expected = fixture.expected + fixture.extra
    _history(case, expected)
    case.check(
        "retry after source restoration recovers configured, discovered and explicit histories"
    )

    index = case.data / "index"
    inode = index.stat().st_ino
    missing = case.providers / "never-created.db"
    failed = case.command(
        "import",
        "--provider",
        "opencode",
        "--path",
        missing,
        "--format",
        "json",
        expect_failure=None,
    )
    require(failed.returncode >= 0, "missing-source import terminated by a signal")
    if failed.returncode == 0:
        require(
            json.loads(failed.stdout)["outcome"] == "completed_with_source_failures",
            "missing source import reported success",
        )
    require(
        index.stat().st_ino == inode,
        "ordinary missing-source error replaced a compatible index",
    )
    _history(case, expected)
    case.check(
        "a compatible index is not replaced for an ordinary missing-source error"
    )

    with fixture.discovered.open("a") as stream:
        stream.write("not valid JSON\n")
    _obsolete(case, "schema")
    report = case.json_command("import", "--all")
    _report(report, fixture.paths)
    require(
        sum(source.get("rejected_records", 0) for source in report["sources"]) == 1,
        "rebuild did not report its rejected input record",
    )
    _history(case, fixture.expected)
    case.check("ordinary rejected-record reporting still permits a completed rebuild")

    metadata = index / "meta.json"
    compatible = metadata.read_bytes()
    try:
        for kind in ("manifest", "metadata"):
            if kind == "manifest":
                value = json.loads(compatible)
                value["payload"] = "{invalid manifest"
                write_json(metadata, value)
            else:
                metadata.write_text("{invalid metadata\n")
            before = _snapshot(case)
            case.command("import", "--all", "--full", expect_failure=True)
            require(
                _snapshot(case) == before,
                f"corrupt {kind} triggered recovery or changed files",
            )
            require(
                _data_directories(case) == directories,
                f"corrupt {kind} left a staging directory",
            )
            metadata.write_bytes(compatible)
    finally:
        metadata.write_bytes(compatible)
    _history(case, fixture.expected)
    case.check(
        "corrupt manifests and metadata fail without replacing the index, including full import"
    )


def _await_rebuild(case, expected, daemon=None):
    deadline = time.monotonic() + 30
    while True:
        if daemon is not None:
            require(daemon.process.poll() is None, f"updater exited: {daemon.log}")
        remaining = deadline - time.monotonic()
        require(remaining > 0, "background recovery did not publish usable history")
        result = case.command(
            "search",
            "rebuildbeacon",
            "--events",
            "--backend",
            "lexical",
            "--refresh",
            "off",
            "--format",
            "json",
            expect_failure=None,
            timeout=min(5, remaining),
        )
        require(
            result.returncode >= 0, "background recovery search terminated by a signal"
        )
        if result.returncode == 0:
            require(
                len(json.loads(result.stdout)["results"]) == len(expected),
                "background recovery published incomplete history",
            )
            _history(case, expected)
            return
        time.sleep(0.05)


def rebuild_background(case):
    fixture = RebuildFixture(case)
    _rebuild_setting(case, "true")
    config = case.config / "config.toml"
    for trigger, kind in (("daemon", "record"), ("refresh", "schema")):
        _obsolete(case, kind)
        directories = _data_directories(case)
        config.write_text(
            config.read_text().replace('mode = "manual"', 'mode = "auto"', 1)
        )
        daemon = None
        try:
            if trigger == "daemon":
                daemon = case.start("daemon", "run", "--loop-interval-seconds", "1")
            else:
                # The initial read can race publication; only eventual results matter.
                result = case.command(
                    "search",
                    "rebuildbeacon",
                    "--backend",
                    "lexical",
                    "--refresh",
                    "background",
                    expect_failure=None,
                    timeout=10,
                )
                require(
                    result.returncode >= 0, "background refresh terminated by a signal"
                )
            _await_rebuild(case, fixture.expected, daemon)
        finally:
            try:
                case.command("index", "mode", "manual", timeout=20)
            finally:
                if daemon is not None:
                    case.stop(daemon)
        require(
            not case.json_command("daemon", "status")["running"],
            "rebuild updater did not stop",
        )
        require(
            _data_directories(case) == directories,
            "background recovery left an old or staging index",
        )
        case.check(
            f"{trigger}: automatic updater recovers all configured and discovered history"
        )


def rebuild_takeover(case):
    fixture = RebuildFixture(case)
    _rebuild_setting(case, "true")
    _obsolete(case)
    config = case.config / "config.toml"
    config.write_text(config.read_text().replace('mode = "manual"', 'mode = "auto"', 1))
    # A long interval keeps this owned worker between imports during the checks.
    daemon = case.start("daemon", "run", "--loop-interval-seconds", "60")
    try:
        _await_rebuild(case, fixture.expected, daemon)
        marker = case.data / "daemon.lock"
        current = marker.read_bytes()
        require(current.strip(), "updater did not write its version marker")

        _obsolete(case, "schema")
        result = case.command(
            "search",
            "rebuildbeacon",
            "--backend",
            "lexical",
            "--refresh",
            "background",
            expect_failure=None,
        )
        require(result.returncode >= 0, "current-worker refresh terminated by a signal")
        require(
            daemon.process.poll() is None and marker.read_bytes() == current,
            "background refresh replaced a current-version updater during recovery",
        )
        _report(case.json_command("setup", "--wait"), fixture.paths)
        _history(case, fixture.expected)
        require(
            daemon.process.poll() is None,
            "foreground recovery stopped the current updater",
        )
        case.check(
            "an incompatible index does not cause takeover of a current-version updater"
        )

        # CI has one executable. Keep a real lock-holding daemon, but simulate
        # the version marker an older executable would leave in that same file.
        obsolete = b"fixture-obsolete-updater\n"
        marker.write_bytes(obsolete)
        case.search("rebuildbeacon", refresh="background")
        require(
            daemon.process.poll() is None and marker.read_bytes() == obsolete,
            "a compatible index caused takeover solely because the updater version differed",
        )
        case.check("a differing updater version alone does not trigger takeover")

        _obsolete(case)
        result = case.command(
            "search",
            "rebuildbeacon",
            "--backend",
            "lexical",
            "--refresh",
            "background",
            expect_failure=None,
        )
        require(result.returncode >= 0, "takeover refresh terminated by a signal")
        case.wait(daemon)
        _await_rebuild(case, fixture.expected)
        require(
            marker.read_bytes() == current,
            "replacement updater did not publish its current version",
        )
        case.check(
            "background refresh replaces an obsolete-marked updater and recovers the index"
        )
    finally:
        try:
            case.command("index", "mode", "manual", timeout=20)
        finally:
            case.stop(daemon)
    require(
        not case.json_command("daemon", "status")["running"],
        "takeover updater did not stop",
    )


def rebuild_semantic(case):
    session = "550e8400-e29b-41d4-a716-446655440302"
    old = "oldsemrebuildbeacon Original cached conversation."
    new = "newsemrebuildbeacon Rebuilt current conversation."
    path = codex_fixture(
        case.providers / "semantic.jsonl", case.work, session, [("user", old)]
    )
    case.command(
        "sources", "add", "semantic-rebuild", "--provider", "codex", "--root", path
    )
    case.command("import", "--all")
    _rebuild_setting(case, "true")
    api = EmbeddingServer(embed=lambda texts: [[1.0, 0.0] for _ in texts])
    try:
        with api:
            api.enable(case)
            initial = case.search(
                "oldsemrebuildbeacon", "--events", backend="semantic", refresh="wait"
            )
            require(
                initial["retrieval"]["effective_mode"] == "semantic"
                and initial["retrieval"]["semantic_status"] == "ready"
                and len(initial["results"]) == 1,
                "initial semantic selection was not ready",
            )
            codex_fixture(path, case.work, session, [("user", new)])
            _obsolete(case)
            case.command("import", "--all")
            _history(case, [(new, "codex", session)])
            passive = case.search("newsemrebuildbeacon", "--events", backend="hybrid")
            require(
                passive["retrieval"]["effective_mode"] == "lexical"
                and passive["retrieval"]["reason"]
                and len(passive["results"]) == 1,
                "rebuilt index reused the stale semantic selection",
            )
            case.command(
                "search",
                "newsemrebuildbeacon",
                "--backend",
                "semantic",
                "--refresh",
                "off",
                expect_failure=True,
            )
            rebuilt = case.search(
                "newsemrebuildbeacon", "--events", backend="semantic", refresh="wait"
            )
            require(
                rebuilt["retrieval"]["effective_mode"] == "semantic"
                and rebuilt["retrieval"]["semantic_status"] == "ready"
                and len(rebuilt["results"]) == 1
                and "newsemrebuildbeacon" in rebuilt["results"][0]["snippet"]
                and "oldsemrebuildbeacon" not in rebuilt["results"][0]["snippet"],
                "semantic refill did not use the rebuilt conversation",
            )
            event = case.json_command(
                "show", "event", rebuilt["results"][0]["trwl_event_id"]
            )["event"]
            require(
                event["text"] == new,
                "semantic result did not retrieve the rebuilt full message",
            )
    finally:
        write_json(case.root / "embedding-requests.json", api.requests)
    case.check(
        "recovery invalidates the old semantic selection; waited refill retrieves the rebuilt conversation"
    )
