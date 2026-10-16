# Regression and releases

GitHub Actions runs `nix build .#checks.x86_64-linux.regression` on pull
requests to `main`, pushes to `main`, and manual CI runs. This builds a CPU-only
variant of the Nix package with its Rust and install checks, then runs every
CLI end-to-end case, including real-model CPU and HTTP inference. Timing-based performance
comparisons are not part of this gate.

Regression uses `ubuntu-24.04`; tagging and auto-merge use `ubuntu-slim`.
The default Nix package includes GPU runtimes; the regression variant omits
them and the Intel OpenCL wrapper.

## GitHub settings

Configure the following repository settings before enabling automatic merges:

- Allow auto-merge and squash merging.
- Protect `main`, require the `Regression` status check from GitHub Actions,
  and require branches to be up to date before merging.
- Under Settings → Actions → General → Workflow permissions, enable
  **Allow GitHub Actions to create and approve pull requests** so the nixpkgs
  updater can open its PRs.

These settings are stored on GitHub, not in workflow files.

## Dependencies

Dependabot checks Cargo dependencies monthly and groups updates into one PR.
Its `lockfile-only` strategy keeps the requirements in `Cargo.toml` unchanged;
upgrades outside those requirements need a separate change.

The `Update nixpkgs` workflow runs on the first of each month, or manually.
It selects the newest `YY.MM` release tag from `NixOS/nixpkgs`, follows the
corresponding `nixos-YY.MM` branch, and updates only the `nixpkgs` flake input.
It opens a PR on `automation/nixpkgs` and dispatches CI on that branch.
Dispatch is explicit because PRs created with `GITHUB_TOKEN` do not trigger
pull-request workflows.

After CI succeeds, a separate workflow enables squash auto-merge for:

- Dependabot PRs that change only `Cargo.lock`.
- `github-actions[bot]` PRs from `automation/nixpkgs` that change `flake.lock`
  and optionally `flake.nix`.

It checks that the PR still has the tested head commit. GitHub's required
checks control when the merge can complete if the base branch has moved.
The auto-merge workflow does not check out PR code.

## Releases

Change the package version in `Cargo.toml` and its entry in `Cargo.lock` in
the same change. Nix reads its version from `Cargo.toml`.

After regression passes on a push to `main`, CI creates an annotated
`v<version>` tag at the tested commit if that version has not been released.
The version must be newer than existing release tags. If the version-bump
build fails, a later successful push can release the same version.

Pushes that keep an already released version do not create tags. A rerun of
a version-changing push accepts an existing tag at that commit; a tag pointing
elsewhere fails and is never moved. Version changes must increase SemVer
precedence.

Tags `v0.1.1` and `v0.1.2` contain Cargo version `0.1.0`. Those tags are left
unchanged; version-driven tagging starts with `0.1.3`.
