"""Index cleanup preserves published records and unrelated files."""

import json

from e2e_support import Case, OpenCodeFixture, require, write_json


def index_storage(case: Case):
    fixture = OpenCodeFixture(case.providers / "storage.db", case.work)
    session = "ses_storage"
    fixture.session(session)
    first = "storageorchard First message.\nUnicode: café; punctuation: [] {}."
    second = "storageorchard Complete assistant reply.\nKeep the second line."
    fixture.message(session, "msg_first", first)
    fixture.message(session, "msg_second", second, role="assistant")
    import_args = ("import", "--provider", "opencode", "--path", fixture.path)
    case.json_command(*import_args)

    index = case.data / "index"
    metadata = index / "meta.json"
    managed = index / ".managed.json"

    def segment_ids():
        return tuple(
            segment["segment_id"].replace("-", "")
            for segment in json.loads(metadata.read_bytes())["segments"]
        )

    def segment_files(ids):
        return {
            path.name: path.read_bytes()
            for path in index.iterdir()
            if path.is_file() and path.name.startswith(ids)
        }

    def events():
        page = case.json_command(
            "list",
            "events",
            "--workspace",
            "**",
            "--include-current-session",
            "--limit",
            "100",
        )
        require(page["next_cursor"] is None, "storage fixture records were truncated")
        return sorted(page["events"], key=lambda row: row["trwl_event_id"])

    original_ids = segment_ids()
    obsolete = segment_files(original_ids)
    require(original_ids and obsolete, "initial import produced no segment files")
    case.json_command(*import_args, "--full")
    published_ids = segment_ids()
    require(published_ids, "full import published no segments")
    require(
        set(original_ids).isdisjoint(published_ids),
        "full import did not replace the initial segments",
    )
    for name, content in obsolete.items():
        (index / name).write_bytes(content)

    published = segment_files(published_ids)
    stores = sorted(name for name in published if name.endswith(".store"))
    require(stores, "published segments have no store files")
    registry = json.loads(managed.read_bytes())
    require(stores[0] in registry, "published store was not in the managed registry")
    forgotten = set(obsolete) | {stores[0]}
    write_json(managed, [name for name in registry if name not in forgotten])
    sentinels = {
        "notes.store": b"Unrelated notes must survive cleanup.\n",
        "0123456789abcdef0123456789abcdef.unknown": b"Unknown extension.\n",
        "01234567-89ab-cdef-0123-456789abcdef.store": b"Noncanonical segment ID.\n",
        "0123456789abcdef0123456789abcdef.+1.del": b"Noncanonical generation.\n",
    }
    for name, content in sentinels.items():
        (index / name).write_bytes(content)

    before = events()
    require(
        len(before) == 2 and {row["text"] for row in before} == {first, second},
        "full import did not preserve exactly the fixture messages",
    )
    metadata_before = metadata.read_bytes()
    case.check(
        "real obsolete segments restored outside the registry; published store also unregistered"
    )

    result = case.json_command(*import_args)
    require(
        len(result["sources"]) == 1 and result["sources"][0].get("unchanged") is True,
        "cleanup import did not use the unchanged input checkpoint",
    )
    remaining = sorted(name for name in obsolete if (index / name).exists())
    require(
        not remaining,
        f"unchanged import left forgotten obsolete segment files: {remaining}",
    )
    require(
        metadata.read_bytes() == metadata_before,
        "unchanged import modified published metadata",
    )
    for name, content in published.items():
        path = index / name
        require(
            path.is_file() and path.read_bytes() == content,
            f"unchanged import removed or modified published segment file: {name}",
        )
    require(events() == before, "unchanged import changed visible records")
    for name, content in sentinels.items():
        path = index / name
        require(
            path.is_file() and path.read_bytes() == content,
            f"cleanup removed or modified unrelated file: {name}",
        )
    case.check(
        "unchanged import removes forgotten segments while preserving metadata, live files and records"
    )

    appended = "storageorchard Appended after cleanup.\nCheckpoint remains usable."
    fixture.message(session, "msg_appended", appended, role="assistant")
    case.json_command(*import_args)
    after = events()
    original_event_ids = {row["trwl_event_id"] for row in before}
    require(
        [row for row in after if row["trwl_event_id"] in original_event_ids] == before,
        "import after cleanup lost, duplicated or modified existing records",
    )
    added = [row for row in after if row["trwl_event_id"] not in original_event_ids]
    require(
        len(after) == 3
        and len(added) == 1
        and added[0]["text"] == appended
        and added[0]["provider_session_id"] == session
        and added[0]["provider_message_id"] == "msg_appended",
        "import after cleanup did not return the appended message exactly once",
    )
    case.check(
        "changed input imports once after cleanup; original records remain intact"
    )
