"""Follow the newest tagged NixOS release and update only its nixpkgs lock."""

import re
import subprocess
from pathlib import Path


def main():
    repository = "https://github.com/NixOS/nixpkgs.git"
    tags = subprocess.check_output(
        ["git", "ls-remote", "--tags", repository, "[0-9][0-9].[0-9][0-9]"],
        text=True,
    )
    releases = [
        match.group(1)
        for line in tags.splitlines()
        if (match := re.fullmatch(r"[0-9a-f]+\s+refs/tags/([0-9]{2}\.[0-9]{2})", line))
    ]
    if not releases:
        raise RuntimeError("No NixOS release tags found")
    release = max(releases, key=lambda value: tuple(map(int, value.split("."))))
    branch = f"nixos-{release}"
    subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            repository,
            f"refs/heads/{branch}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    flake = Path("flake.nix")
    content, count = re.subn(
        r'(nixpkgs\.url = "github:NixOS/nixpkgs/)nixos-[^"]+(";)',
        rf"\g<1>{branch}\g<2>",
        flake.read_text(),
    )
    if count != 1:
        raise RuntimeError("Expected exactly one nixpkgs release branch in flake.nix")
    flake.write_text(content)
    subprocess.run(["nix", "flake", "update", "nixpkgs"], check=True)
    print(f"Updated nixpkgs from {branch}")


if __name__ == "__main__":
    main()
