use crate::record::{ContentScope, EventType, IndexMode, Provider};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "trwl", version, about = "Search coding agent history")]
pub struct Cli {
    #[arg(long, env = "TRWL_DATA_ROOT", global = true)]
    pub data_root: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "text", global = true)]
    pub format: Format,
    #[arg(long, global = true)]
    pub out: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
pub enum Format {
    /// Human-readable labels and transcripts
    Text,
    /// One JSON document containing all rows and metadata
    Json,
    /// One JSON record per row, followed by collection metadata
    Jsonl,
    /// Human-readable output with Markdown headings
    Markdown,
}

#[derive(Subcommand)]
pub enum Command {
    Search(Search),
    Import {
        #[arg(long, value_enum, required_unless_present = "all")]
        provider: Option<Provider>,
        #[arg(long, required_unless_present = "all")]
        path: Option<PathBuf>,
        #[arg(long)]
        all: bool,
        /// Reparse every input instead of resuming from stored parser state
        #[arg(long)]
        full: bool,
        #[arg(long, value_enum, default_value = "none")]
        progress: Progress,
    },
    Show {
        #[arg(value_enum)]
        kind: Entity,
        id: String,
        #[arg(long, default_value_t = 0)]
        window: usize,
    },
    Locate {
        #[arg(value_enum)]
        kind: Entity,
        id: String,
    },
    List {
        #[arg(value_enum)]
        kind: ListKind,
        #[command(flatten)]
        filters: Filters,
        #[arg(long, default_value_t = 100)]
        limit: usize,
        #[arg(long)]
        cursor: Option<String>,
    },
    Sources {
        #[command(subcommand)]
        command: Sources,
    },
    /// Configure semantic search and inspect embedding coverage
    Semantic {
        #[command(subcommand)]
        command: Semantic,
    },
    /// Inspect lexical readiness and set indexing mode
    Index {
        #[command(subcommand)]
        command: Index,
    },
    Daemon {
        #[command(subcommand)]
        command: Daemon,
    },
    Setup {
        #[arg(long)]
        wait: bool,
    },
    /// Inspect lexical and semantic readiness without starting indexing
    Status,
    #[command(hide = true)]
    EmbeddingWorker,
    #[command(hide = true)]
    EmbeddingServer,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Entity {
    Event,
    Session,
}
#[derive(Clone, Copy, ValueEnum)]
pub enum ListKind {
    Events,
    Sessions,
}
#[derive(Clone, Copy, ValueEnum, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Lexical,
    Semantic,
    Hybrid,
}
#[derive(Clone, Copy, ValueEnum, PartialEq)]
pub enum Refresh {
    Off,
    Wait,
    Background,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
pub enum Progress {
    None,
    Json,
}

#[derive(Args)]
pub struct Search {
    pub query: String,
    #[arg(long)]
    pub term: Vec<String>,
    #[arg(long, value_enum, default_value = "lexical")]
    pub backend: Backend,
    #[arg(long, value_enum, default_value = "background")]
    pub refresh: Refresh,
    #[arg(long)]
    pub events: bool,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    #[command(flatten)]
    pub filters: Filters,
}

#[derive(Args, Default)]
pub struct Filters {
    #[arg(long, value_enum)]
    pub provider: Vec<Provider>,
    #[arg(long)]
    pub session: Option<String>,
    #[arg(long)]
    pub exclude_session: Vec<String>,
    /// Select canonical workspace roots with case-insensitive pathname globs;
    /// repeat to combine with OR. Defaults to the current workspace. Quote '**'
    /// to search all history, including unresolved roots. '*' stays within one
    /// path component; '**' spans directories. Escape glob metacharacters with
    /// '\' to name a directory containing them. Literal directories resolve to
    /// their workspace roots; other globs match full canonical paths.
    #[arg(long, value_name = "PATTERN")]
    pub workspace: Vec<crate::workspace::Pattern>,
    #[arg(long)]
    pub since: Option<String>,
    #[arg(long)]
    pub until: Option<String>,
    #[arg(long, value_enum)]
    pub event_type: Option<EventType>,
    #[arg(long, value_enum, default_value = "all")]
    pub content_scope: ContentScope,
    #[arg(long)]
    pub file: Option<String>,
    #[arg(long)]
    pub primary_only: bool,
    #[arg(long)]
    pub include_current_session: bool,
}

#[derive(Subcommand)]
pub enum Sources {
    Add {
        name: String,
        #[arg(long, value_enum)]
        provider: Provider,
        #[arg(long)]
        root: PathBuf,
    },
    List,
    Remove {
        name: String,
    },
}
#[derive(Subcommand)]
pub enum Semantic {
    Enable {
        #[arg(long, default_value = "builtin")]
        executor: String,
        #[arg(long)]
        model: Option<String>,
    },
    Disable,
    /// Rebuild all embeddings, or resume an interrupted full build
    Rebuild,
    /// Inspect embedding coverage of indexed history without running inference
    Status,
}
#[derive(Subcommand)]
pub enum Index {
    Mode {
        #[arg(value_enum)]
        mode: Option<IndexMode>,
    },
    /// Compare configured/discovered inputs with the published lexical index
    Status,
}
#[derive(Subcommand)]
pub enum Daemon {
    Run {
        #[arg(long, default_value_t = 2)]
        loop_interval_seconds: u64,
    },
    Stop,
    Status,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dead_options_and_invalid_selectors_are_rejected() {
        for args in [
            vec!["trwl", "--quiet", "search", "needle"],
            vec!["trwl", "--color", "never", "search", "needle"],
            vec!["trwl", "show", "event", "id", "--mode", "log"],
            vec!["trwl", "list", "events", "--content", "full"],
            vec![
                "trwl",
                "import",
                "--provider",
                "codex",
                "--path",
                "input",
                "--progress",
                "wrong",
            ],
            vec!["trwl", "import", "--provider", "wrong", "--path", "input"],
            vec!["trwl", "index", "mode", "wrong"],
            vec!["trwl", "search", "needle", "--content-scope", "wrong"],
            vec!["trwl", "search", "needle", "--event-type", "wrong"],
        ] {
            assert!(Cli::try_parse_from(&args).is_err(), "{args:?}");
        }
        for provider in ["claude-code", "copilot-cli", "copilot_cli"] {
            assert!(
                Cli::try_parse_from([
                    "trwl",
                    "import",
                    "--provider",
                    provider,
                    "--path",
                    "input",
                    "--progress",
                    "json"
                ])
                .is_ok()
            );
        }
    }
}
