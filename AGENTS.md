# What

trwl, a coding agent session history search tool.

# General

- The product is an executable, not an externally-consumed
  library. The only externally-consumed interface is the CLI, so any
  backward-compatibility considerations are relevant for the CLI
  contract alone. Do not create shims, bridges, wrapper options, or
  legacy forwarding layers for anything else.
- Keep the code base well-typed (one common anti-pattern is using raw
  string values in place of enums).

# Development checks

- For Rust changes, use Cargo to compile and test during development, reusing
  the local target directory. Use `direnv exec . <command>` when a local `.envrc`
  supplies the devshell but the command does not inherit it.
- Run Nix package and regression builds after Cargo checks pass and the changes
  are ready for integration. Do not use Nix builds as the edit/compile loop:
  changed sources cause a fresh package build, including its dependencies.

# Tests

- Do not add tests reflexively, every test must have a clearly-defined
  purpose.
- Only add tests for behavior that can conceivably regress (i.e. get
  accidentally broken by changes made in places other than its textual
  source(s) of truth).
- Prefer e2e behavioral checks to unit tests -- unit tests are mostly
  worthless in a well-typed Rust code base.
- Check behavioral contracts, not code.
