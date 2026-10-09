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

# Tests

- Do not add tests reflexively, every test must have a clearly-defined
  purpose.
- Only add tests for behavior that can conceivably regress (i.e. get
  accidentally broken by changes made in places other than its textual
  source(s) of truth).
- Prefer e2e behavioral checks to unit tests -- unit tests are mostly
  worthless in a well-typed Rust code base.
- Check behavioral contracts, not code.
