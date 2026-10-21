//! Command dispatch and the background updater.
//!
//! The updater polls sources and imports changes. One embedding worker runs
//! alongside it, so inference does not block lexical updates; changes to the
//! index or configuration schedule another fill after the worker finishes, and
//! failed fills back off. `daemon stop` writes a request file the updater
//! polls for; configuration is not modified. Lock status is probed without
//! acquiring the lock.

mod cli;
mod config;
mod output;
mod providers;
mod query;
mod record;
mod report;
mod semantic;
mod store;
#[cfg(test)]
mod test_support;
mod workspace;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use cli::{Cli, Command, Daemon, Format, Index, Progress, Refresh, Semantic, Sources};
use config::{Config, Source};
use report::{OverallStatus, Report, Running, State, Status};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command as Process, Stdio},
    thread,
    time::{Duration, Instant},
};
use toml_edit::{ArrayOfTables, Table, value};

fn root(cli: &Cli) -> Result<PathBuf> {
    let root = if let Some(root) = &cli.data_root {
        root.clone()
    } else {
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
            .context("HOME and XDG_DATA_HOME are unset")?;
        data.join("trwl")
    };
    Ok(if root.is_absolute() {
        root
    } else {
        std::env::current_dir()?.join(root)
    })
}

fn daemon_running(root: &Path) -> Result<bool> {
    config::is_locked(root, DAEMON_LOCK)
}

const DAEMON_LOCK: &str = "daemon.lock";
// A stop request is a file rather than a signal, so it also reaches an updater
// in another PID namespace that shares the data directory.
const DAEMON_STOP: &str = "daemon.stop";
// Failed fills start with a two-second retry delay and double up to this cap.
// A configuration edit retries at once.
const EMBEDDING_RETRY_CAP: Duration = Duration::from_secs(300);

fn start_daemon(config: &Config) -> Result<()> {
    if !config.automatic() || daemon_running(&config.root)? {
        return Ok(());
    }
    fs::create_dir_all(&config.root)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.root.join("daemon.log"))?;
    let mut process = Process::new(std::env::current_exe()?);
    process
        .args(["--data-root"])
        .arg(&config.root)
        .args(["daemon", "run"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log);
    // The detached process owns no terminal or inherited pipe. Its file lock
    // prevents duplicate workers, including simultaneous background requests.
    unsafe {
        process.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = process.spawn()?;
    thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

fn stop_daemon(root: &Path) -> Result<()> {
    // The running daemon polls for this file and removes it before exiting.
    // Rewrite it while waiting, in case an updater started meanwhile cleared it.
    let deadline = Instant::now() + Duration::from_secs(15);
    while daemon_running(root)? {
        fs::write(root.join(DAEMON_STOP), b"")?;
        ensure!(
            Instant::now() < deadline,
            "background updater is still finishing its current update"
        );
        thread::sleep(Duration::from_millis(50));
    }
    semantic::server::stop(root)
}

fn run_daemon(root: &Path, interval: u64) -> Result<()> {
    let Some(_guard) = config::try_lock(root, DAEMON_LOCK)? else {
        return Ok(());
    };
    let stop = root.join(DAEMON_STOP);
    // A request left by an updater that exited on its own does not apply to
    // this one. `stop_daemon` rewrites the file while it waits.
    remove_if_present(&stop)?;
    let mut embedding = EmbeddingUpdater::default();
    // Exiting ends the embedding thread with the process; an interrupted fill
    // keeps completed batches and the next updater resumes it.
    while !daemon_stopping(root, &stop)? {
        let config = Config::load(root)?;
        embedding.collect();
        if let Err(error) = update_iteration(&config, &mut embedding) {
            eprintln!("background update: {error:#}");
        }
        let deadline = Instant::now() + Duration::from_secs(interval.max(1));
        while Instant::now() < deadline {
            if daemon_stopping(root, &stop)? {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
    Ok(())
}

fn daemon_stopping(root: &Path, stop: &Path) -> Result<bool> {
    if stop.try_exists()? {
        remove_if_present(stop)?;
        return Ok(true);
    }
    Ok(!Config::load(root)?.automatic())
}

fn update_iteration(config: &Config, embedding: &mut EmbeddingUpdater) -> Result<()> {
    let report = store::update(config, &config.sources()?, false)?;
    if !report.completed() {
        eprintln!("{}", serde_json::to_string(&report)?);
    }
    if config.semantic()
        && let Some(version) = store::content_version(&config.root)?
    {
        embedding.schedule(config, &version);
    }
    Ok(())
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
        _ => Ok(()),
    }
}

/// One background embedding fill at a time. A fill runs when the index commit
/// or configuration differs from the last successful fill. After a failure,
/// unchanged configuration waits for a backoff delay before the next attempt.
#[derive(Default)]
struct EmbeddingUpdater {
    job: Option<EmbeddingJob>,
    planner: semantic::Planner,
    completed: Option<String>,
    failure: Option<Failure>,
}

struct EmbeddingJob {
    state: String,
    configuration: String,
    thread: thread::JoinHandle<(semantic::Planner, Result<()>)>,
}

struct Failure {
    configuration: String,
    count: u32,
    retry_at: Instant,
}

impl EmbeddingUpdater {
    fn collect(&mut self) {
        if !self
            .job
            .as_ref()
            .is_some_and(|job| job.thread.is_finished())
        {
            return;
        }
        let EmbeddingJob {
            state,
            configuration,
            thread,
        } = self.job.take().unwrap();
        let result = thread.join().map(|(planner, result)| {
            self.planner = planner;
            result
        });
        let error = match result {
            Ok(Ok(())) => {
                self.completed = Some(state);
                self.failure = None;
                return;
            }
            Ok(Err(error)) => format!("{error:#}"),
            Err(_) => "worker panicked".into(),
        };
        self.record_failure(configuration, &error);
    }

    fn record_failure(&mut self, configuration: String, error: &str) {
        let count = match &self.failure {
            Some(failure) if failure.configuration == configuration => failure.count + 1,
            _ => 1,
        };
        let delay = EMBEDDING_RETRY_CAP.min(Duration::from_secs(1 << count.min(16)));
        eprintln!("background embedding: {error}; retrying in {delay:?}");
        self.failure = Some(Failure {
            configuration,
            count,
            retry_at: Instant::now() + delay,
        });
    }

    fn schedule(&mut self, config: &Config, content_version: &str) {
        if self.job.is_some() {
            return;
        }
        let configuration = record::digest(&[&config.document.to_string()]);
        if self.failure.as_ref().is_some_and(|failure| {
            failure.configuration == configuration && Instant::now() < failure.retry_at
        }) {
            return;
        }
        let selection = match semantic::Planner::selection_key(config) {
            Ok(selection) => selection,
            Err(error) => {
                self.record_failure(configuration, &format!("{error:#}"));
                return;
            }
        };
        let state = record::digest(&[content_version, &configuration, &selection]);
        if self.completed.as_ref() == Some(&state) {
            return;
        }
        let worker_config = config.clone();
        // Remember the state this worker started with. A later index commit or
        // configuration edit schedules another fill after it finishes.
        let mut planner = std::mem::take(&mut self.planner);
        self.job = Some(EmbeddingJob {
            state,
            configuration,
            thread: thread::spawn(move || {
                let result = planner.refresh(&worker_config);
                (planner, result)
            }),
        });
    }
}

fn dispatch(cli: &Cli, root: &Path) -> Result<Report> {
    let config = Config::load(root)?;
    match &cli.command {
        Command::EmbeddingWorker | Command::EmbeddingServer => {
            unreachable!("handled before command dispatch")
        }
        Command::Import {
            provider,
            path,
            all,
            full,
            progress,
        } => {
            let sources = if *all {
                config.sources()?
            } else {
                vec![Source {
                    name: "explicit".into(),
                    provider: provider.context("missing --provider")?,
                    root: path.clone().context("missing --path")?,
                }]
            };
            let import = if *full {
                store::update_full
            } else {
                store::update
            };
            Ok(Report::Import(import(
                &config,
                &sources,
                *progress == Progress::Json,
            )?))
        }
        Command::Search(args) => {
            if args.refresh == Refresh::Wait {
                let report = store::update(&config, &config.sources()?, false)?;
                if !report.completed() {
                    eprintln!("{}", serde_json::to_string(&report)?);
                }
            } else if args.refresh == Refresh::Background {
                start_daemon(&config)?;
            }
            Ok(Report::Search(query::search(&config, args)?))
        }
        Command::Show { kind, id, window } => query::show(root, *kind, id, *window),
        Command::Locate { kind, id } => Ok(Report::Location(query::locate(&query::resolve(
            root, *kind, id,
        )?))),
        Command::List {
            kind,
            filters,
            limit,
            cursor,
            ..
        } => query::list(root, *kind, filters, *limit, cursor.as_deref()),
        Command::Sources { command } => match command {
            Sources::List => Ok(Report::Sources {
                sources: config.sources()?,
            }),
            Sources::Add {
                name,
                provider,
                root: source_root,
            } => {
                let provider = *provider;
                let source_root = source_root
                    .canonicalize()
                    .with_context(|| format!("opening {}", source_root.display()))?;
                Config::edit(root, |doc| {
                    if doc.get("sources").and_then(|s| s.get("roots")).is_none() {
                        doc["sources"]["roots"] =
                            toml_edit::Item::ArrayOfTables(ArrayOfTables::new());
                    }
                    let roots = doc["sources"]["roots"]
                        .as_array_of_tables_mut()
                        .context("sources.roots must be an array of tables")?;
                    roots.retain(|r| r.get("name").and_then(|n| n.as_str()) != Some(name));
                    let mut table = Table::new();
                    table["name"] = value(name);
                    table["provider"] = value(provider.as_str());
                    table["root"] = value(source_root.to_string_lossy().as_ref());
                    roots.push(table);
                    Ok(())
                })?;
                Ok(Report::SourceAdded(Source {
                    name: name.clone(),
                    provider,
                    root: source_root,
                }))
            }
            Sources::Remove { name } => {
                Config::edit(root, |doc| {
                    if let Some(roots) = doc
                        .get_mut("sources")
                        .and_then(|s| s.get_mut("roots"))
                        .and_then(|r| r.as_array_of_tables_mut())
                    {
                        roots.retain(|r| r.get("name").and_then(|n| n.as_str()) != Some(name));
                    }
                    Ok(())
                })?;
                Ok(Report::SourceRemoved {
                    removed: name.clone(),
                })
            }
        },
        Command::Semantic { command } => match command {
            Semantic::Enable { executor, model } => {
                let model = semantic::validate_selection(executor, model.as_deref())?;
                Config::edit(root, |doc| {
                    doc["search"]["semantic"] = value(true);
                    doc["semantic"]["executor"] = value(executor);
                    doc["semantic"]["model"] = value(model);
                    Ok(())
                })?;
                Ok(Report::SemanticEnabled {
                    semantic: true,
                    executor: executor.clone(),
                    model: model.into(),
                })
            }
            Semantic::Disable => {
                Config::edit(root, |doc| {
                    doc["search"]["semantic"] = value(false);
                    Ok(())
                })?;
                semantic::server::stop(root)?;
                Ok(Report::SemanticDisabled { semantic: false })
            }
            Semantic::Status => Ok(Report::Semantic(semantic::status(&config).into())),
            Semantic::Rebuild => {
                semantic::rebuild(&config)?;
                Ok(Report::Semantic(semantic::status(&config).into()))
            }
        },
        Command::Index {
            command: Index::Status,
        } => Ok(Report::Lexical(store::status(&config).into())),
        Command::Index {
            command: Index::Mode { mode },
        } => {
            if let Some(mode) = mode {
                let config = Config::set_mode(root, *mode)?;
                if config.automatic() {
                    start_daemon(&config)?;
                } else {
                    stop_daemon(root)?;
                }
            }
            Ok(Report::Mode {
                mode: Config::load(root)?.mode(),
            })
        }
        Command::Daemon { command } => match command {
            Daemon::Run {
                loop_interval_seconds,
            } => {
                run_daemon(root, *loop_interval_seconds)?;
                Ok(Report::Stopped { stopped: true })
            }
            Daemon::Stop => {
                stop_daemon(root)?;
                Ok(Report::Stopped { stopped: true })
            }
            Daemon::Status => Ok(Report::Daemon(Running {
                running: daemon_running(root)?,
            })),
        },
        Command::Setup { wait } => {
            let report = if *wait {
                let report = store::update(&config, &config.sources()?, false)?;
                semantic::refresh(&config)?;
                Report::Import(report)
            } else {
                Report::Setup { setup: true }
            };
            start_daemon(&config)?;
            Ok(report)
        }
        Command::Status => {
            let lexical: Status<_> = store::status(&config).into();
            let semantic: Status<_> = semantic::status(&config).into();
            let ready = matches!(&lexical, Status::Available(status) if status.state == State::Ready)
                && matches!(&semantic, Status::Available(status) if matches!(status.state, State::Ready | State::Disabled));
            Ok(Report::Status(OverallStatus {
                data_root: root.into(),
                indexing_mode: config.mode(),
                daemon: Running {
                    running: daemon_running(root)?,
                },
                ready,
                lexical,
                semantic,
            }))
        }
    }
}

fn main() {
    // Transcripts may contain credentials. Restrict indexes, configuration,
    // logs and exports to the current user.
    unsafe {
        libc::umask(0o077);
    }
    let cli = Cli::parse();
    if matches!(cli.command, Command::EmbeddingServer) {
        if root(&cli)
            .and_then(|root| semantic::server::run(&root))
            .is_err()
        {
            std::process::exit(1);
        }
        return;
    }
    if matches!(cli.command, Command::EmbeddingWorker) {
        if let Err(error) = semantic::worker::run() {
            eprintln!("trwl embedding worker: {error:#}");
            std::process::exit(1);
        }
        return;
    }
    let result = root(&cli)
        .and_then(|root| dispatch(&cli, &root))
        .and_then(|value| {
            let output = output::render(&value, cli.format)?;
            if let Some(path) = &cli.out {
                fs::write(path, output)?;
            } else {
                std::io::stdout().lock().write_all(output.as_bytes())?;
            }
            Ok(())
        });
    if let Err(error) = result {
        if matches!(cli.format, Format::Json | Format::Jsonl) {
            eprintln!("{}", json!({"error": format!("{error:#}")}));
        } else {
            eprintln!("trwl: {error:#}");
        }
        std::process::exit(1);
    }
}
