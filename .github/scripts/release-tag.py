"""Tag an unreleased Cargo version after regression succeeds on main."""

import os
import re
import subprocess
import tomllib


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def package_version(revision):
    return tomllib.loads(git("show", f"{revision}:Cargo.toml"))["package"]["version"]


def version_key(version):
    """SemVer precedence: numeric prereleases sort before textual ones."""
    match = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
        r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?",
        version,
    )
    if match is None:
        raise ValueError(f"Invalid semantic version: {version}")
    major, minor, patch, prerelease = match.groups()
    identifiers = []
    for part in prerelease.split(".") if prerelease is not None else []:
        numeric = part.isascii() and part.isdigit()
        if numeric and len(part) > 1 and part.startswith("0"):
            raise ValueError(f"Invalid numeric prerelease identifier: {version}")
        identifiers.append((0, int(part)) if numeric else (1, part))
    return int(major), int(minor), int(patch), prerelease is None, tuple(identifiers)


def main():
    before = os.environ["RELEASE_BEFORE"]
    commit = os.environ["RELEASE_SHA"]
    version = package_version(commit)
    previous = package_version(before)
    tag = f"v{version}"
    tags = git("tag", "--list").splitlines()
    if version == previous and tag in tags:
        print(f"Cargo version unchanged ({version}); no tag to create.")
        return

    key = version_key(version)
    if version != previous and key <= version_key(previous):
        raise ValueError(f"Version must increase: {previous} -> {version}")

    if tag in tags:
        if git("rev-parse", f"refs/tags/{tag}^{{commit}}") != commit:
            raise ValueError(f"{tag} already points to another commit")
        print(f"{tag} already points to {commit}.")
        return

    for existing in tags:
        if not existing.startswith("v"):
            continue
        try:
            existing_key = version_key(existing[1:])
        except ValueError:
            continue
        if key <= existing_key:
            raise ValueError(f"{tag} must be newer than existing release {existing}")

    # A failed version-bump build may be followed by a fix with the same version.
    git(
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "tag",
        "--annotate",
        tag,
        commit,
        "--message",
        tag,
    )
    git("push", "origin", f"refs/tags/{tag}")
    print(f"Published {tag} at {commit}.")


if __name__ == "__main__":
    main()
