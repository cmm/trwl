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
mod redaction;
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
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::process::CommandExt,
    path::Path,
    process::{Command as Process, Stdio},
    thread,
    time::{Duration, Instant},
};
use toml_edit::{ArrayOfTables, Table, value};

mod directories;
use directories::Directories;

fn daemon_running(root: &Path) -> Result<bool> {
    config::is_locked(root, DAEMON_LOCK)
}

const DAEMON_LOCK: &str = "daemon.lock";
// Resource locks stay with the index so different runtime directories and PID
// namespaces still coordinate. Holding the stop lock requests shutdown without
// leaving a stale request after the requesting process exits.
const DAEMON_STOP: &str = "daemon-stop.lock";
// Serialize takeover decisions with daemon ownership and version publication.
const DAEMON_START: &str = "daemon-start.lock";
// Failed fills start with a two-second retry delay and double up to this cap.
// A configuration edit retries at once.
const EMBEDDING_RETRY_CAP: Duration = Duration::from_secs(300);

fn start_daemon(config: &Config) -> Result<()> {
    if !config.automatic() {
        return Ok(());
    }
    let startup = DaemonStartup::acquire(&config.directories)?;
    match startup.decision(config)? {
        DaemonStartDecision::KeepRunning => return Ok(()),
        DaemonStartDecision::Takeover => startup.takeover()?,
        DaemonStartDecision::Spawn => {}
    }
    startup.spawn()
}

enum DaemonStartDecision {
    KeepRunning,
    Takeover,
    Spawn,
}

struct DaemonStartup<'a> {
    directories: &'a Directories,
    _lock: File,
}

impl<'a> DaemonStartup<'a> {
    fn acquire(directories: &'a Directories) -> Result<Self> {
        Ok(Self {
            directories,
            _lock: config::lock(&directories.data, DAEMON_START)?,
        })
    }

    fn decision(&self, config: &Config) -> Result<DaemonStartDecision> {
        if !daemon_running(&self.directories.data)? {
            return Ok(DaemonStartDecision::Spawn);
        }
        // A daemon from an older executable may keep accepting the old format.
        // Do not interrupt a current daemon while it is rebuilding that index.
        let current = fs::read_to_string(self.directories.data.join(DAEMON_LOCK))
            .is_ok_and(|version| version == env!("CARGO_PKG_VERSION"));
        if !config.rebuild_incompatible()
            || current
            || !store::ReadIndex::open(&self.directories.data)
                .err()
                .is_some_and(|error| store::incompatible(&error))
        {
            return Ok(DaemonStartDecision::KeepRunning);
        }
        Ok(DaemonStartDecision::Takeover)
    }

    fn takeover(&self) -> Result<()> {
        // The stop request ends before spawn, but startup remains serialized.
        stop_daemon(self.directories)
    }

    fn spawn(self) -> Result<()> {
        fs::create_dir_all(&self.directories.state)?;
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directories.state.join("daemon.log"))?;
        let mut process = Process::new(std::env::current_exe()?);
        self.directories.child_environment(&mut process);
        process
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
        // Spawning does not claim daemon ownership. The child acquires startup
        // anew after this guard is released; the parent never waits for it.
        Ok(())
    }

    fn claim(self) -> Result<Option<RunningDaemon>> {
        let Some(mut lock) = config::try_lock(&self.directories.data, DAEMON_LOCK)? else {
            return Ok(None);
        };
        // On publication failure this local drops before the startup guard.
        lock.set_len(0)?;
        lock.write_all(env!("CARGO_PKG_VERSION").as_bytes())?;
        drop(self);
        Ok(Some(RunningDaemon { _lock: lock }))
    }
}

struct DaemonStopRequest<'a> {
    directories: &'a Directories,
    _lock: File,
}

impl<'a> DaemonStopRequest<'a> {
    fn acquire(directories: &'a Directories) -> Result<Self> {
        Ok(Self {
            directories,
            _lock: config::lock(&directories.data, DAEMON_STOP)?,
        })
    }

    fn wait(self) -> Result<()> {
        // Waiting for the request lock is outside this deadline.
        let deadline = Instant::now() + Duration::from_secs(15);
        while daemon_running(&self.directories.data)? {
            ensure!(
                Instant::now() < deadline,
                "background updater is still finishing its current update"
            );
            thread::sleep(Duration::from_millis(50));
        }
        semantic::server::stop(self.directories)
    }
}

fn stop_daemon(directories: &Directories) -> Result<()> {
    DaemonStopRequest::acquire(directories)?.wait()
}

struct RunningDaemon {
    _lock: File,
}

impl RunningDaemon {
    fn exit(self, embedding: EmbeddingUpdater) {
        // Dropping a job detaches it. Lock release must not wait for a fill;
        // process exit ends outstanding work, including an in-flight batch.
        drop(embedding);
        drop(self);
    }
}

fn run_daemon(directories: &Directories, interval: u64) -> Result<()> {
    let Some(daemon) = DaemonStartup::acquire(directories)?.claim()? else {
        return Ok(());
    };
    let mut embedding = EmbeddingUpdater::default();
    // Exiting ends the embedding thread with the process; an interrupted fill
    // keeps completed batches and the next updater resumes it.
    'running: while !daemon_stopping(directories)? {
        let config = Config::load(directories)?;
        embedding = embedding.collect();
        match update_iteration(&config) {
            Ok(Some(version)) => embedding = embedding.schedule(&config, &version),
            Ok(None) => {}
            Err(error) => eprintln!("background update: {error:#}"),
        }
        let deadline = Instant::now() + Duration::from_secs(interval.max(1));
        while Instant::now() < deadline {
            if daemon_stopping(directories)? {
                break 'running;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
    daemon.exit(embedding);
    Ok(())
}

fn daemon_stopping(directories: &Directories) -> Result<bool> {
    Ok(config::is_locked(&directories.data, DAEMON_STOP)?
        || !Config::load(directories)?.automatic())
}

fn update_iteration(config: &Config) -> Result<Option<store::ConversationRevision>> {
    let report = store::update(config, &config.sources()?, false)?;
    if !report.completed() {
        eprintln!("{}", serde_json::to_string(&report)?);
    }
    if config.semantic() {
        return store::semantic_version(&config.directories.data);
    }
    Ok(None)
}

/// One background embedding fill at a time. A fill runs when conversation inputs
/// or configuration differs from the last successful fill, or coverage is lost.
/// After a failure, unchanged configuration waits for a backoff delay before
/// the next attempt.
#[derive(Default)]
struct EmbeddingUpdater {
    work: EmbeddingWork,
    completed: Option<String>,
    failure: Option<Failure>,
}

enum EmbeddingWork {
    Idle(semantic::Planner),
    Running(EmbeddingJob),
}

impl Default for EmbeddingWork {
    fn default() -> Self {
        Self::Idle(semantic::Planner::default())
    }
}

struct Launch {
    state: String,
    configuration: String,
}

struct EmbeddingJob {
    launch: Launch,
    thread: thread::JoinHandle<(semantic::Planner, Result<()>)>,
}

enum ScheduleDecision {
    AlreadyRunning,
    WaitingForRetry,
    AlreadyCurrent,
    SelectionFailed,
    Start(Launch),
}

struct Failure {
    configuration: String,
    count: u32,
    retry_at: Instant,
}

impl EmbeddingUpdater {
    fn collect(mut self) -> Self {
        let job = match self.work {
            EmbeddingWork::Running(job) if job.thread.is_finished() => job,
            _ => return self,
        };
        let EmbeddingJob { launch, thread } = job;
        let (planner, result) = match thread.join() {
            Ok(returned) => returned,
            // A panic loses the planner owned by the thread. Normal errors keep it.
            Err(_) => (
                semantic::Planner::default(),
                Err(anyhow::anyhow!("worker panicked")),
            ),
        };
        self.work = EmbeddingWork::Idle(planner);
        let error = match result {
            Ok(()) => {
                self.completed = Some(launch.state);
                self.failure = None;
                return self;
            }
            Err(error) => format!("{error:#}"),
        };
        self.record_failure(launch.configuration, &error);
        self
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

    fn scheduling_decision(
        &mut self,
        config: &Config,
        semantic_version: &store::ConversationRevision,
    ) -> ScheduleDecision {
        if matches!(self.work, EmbeddingWork::Running(_)) {
            return ScheduleDecision::AlreadyRunning;
        }
        let configuration = record::digest(&[&config.document.to_string()]);
        if self.failure.as_ref().is_some_and(|failure| {
            failure.configuration == configuration && Instant::now() < failure.retry_at
        }) {
            return ScheduleDecision::WaitingForRetry;
        }
        let selection = match semantic::Planner::selection_key(config) {
            Ok(selection) => selection,
            Err(error) => {
                self.record_failure(configuration, &format!("{error:#}"));
                return ScheduleDecision::SelectionFailed;
            }
        };
        let state = record::digest(&[
            semantic_version.as_str(),
            &configuration,
            selection.as_str(),
        ]);
        if self.completed.as_ref() == Some(&state) {
            match semantic::Planner::covered(config, &selection, semantic_version) {
                Ok(true) => return ScheduleDecision::AlreadyCurrent,
                Ok(false) => {}
                Err(error) => {
                    self.record_failure(configuration, &format!("{error:#}"));
                    return ScheduleDecision::SelectionFailed;
                }
            }
        }
        ScheduleDecision::Start(Launch {
            state,
            configuration,
        })
    }

    fn schedule(mut self, config: &Config, semantic_version: &store::ConversationRevision) -> Self {
        let launch = match self.scheduling_decision(config, semantic_version) {
            ScheduleDecision::Start(launch) => launch,
            ScheduleDecision::AlreadyRunning
            | ScheduleDecision::WaitingForRetry
            | ScheduleDecision::AlreadyCurrent
            | ScheduleDecision::SelectionFailed => return self,
        };
        let EmbeddingWork::Idle(mut planner) = self.work else {
            unreachable!("a running job cannot schedule another fill");
        };
        let worker_config = config.clone();
        // Remember the state this worker started with. A conversation change or
        // configuration edit schedules another fill after it finishes.
        // Refresh resolves its own reader and selection later, under semantic.lock.
        self.work = EmbeddingWork::Running(EmbeddingJob {
            launch,
            thread: thread::spawn(move || {
                let result = planner.refresh(&worker_config);
                (planner, result)
            }),
        });
        self
    }
}

fn dispatch(cli: &Cli, directories: &Directories) -> Result<Report> {
    let root = &directories.data;
    let config = Config::load(directories)?;
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
                Config::edit(directories, |doc| {
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
                Config::edit(directories, |doc| {
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
                Config::edit(directories, |doc| {
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
                Config::edit(directories, |doc| {
                    doc["search"]["semantic"] = value(false);
                    Ok(())
                })?;
                semantic::server::stop(directories)?;
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
                let config = Config::set_mode(directories, *mode)?;
                if config.automatic() {
                    start_daemon(&config)?;
                } else {
                    stop_daemon(directories)?;
                }
            }
            Ok(Report::Mode {
                mode: Config::load(directories)?.mode(),
            })
        }
        Command::Daemon { command } => match command {
            Daemon::Run {
                loop_interval_seconds,
            } => {
                run_daemon(directories, *loop_interval_seconds)?;
                Ok(Report::Stopped { stopped: true })
            }
            Daemon::Stop => {
                stop_daemon(directories)?;
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
                directories: directories.clone(),
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
        if Directories::resolve()
            .and_then(|directories| semantic::server::run(&directories))
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
    let result = Directories::resolve()
        .and_then(|directories| dispatch(&cli, &directories))
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

#[cfg(test)]
mod updater_tests {
    use super::*;

    fn configuration() -> (tempfile::TempDir, Config) {
        let directory = tempfile::tempdir().unwrap();
        let mut config = Config::load(&Directories::isolated(directory.path())).unwrap();
        config.document =
            "[search]\nsemantic = true\n[semantic]\nexecutor = 'http://127.0.0.1:1'\nmodel = 'test'\n"
                .parse()
                .unwrap();
        (directory, config)
    }

    fn wait_for_job(updater: &EmbeddingUpdater) {
        let EmbeddingWork::Running(job) = &updater.work else {
            panic!("expected a running job");
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !job.thread.is_finished() {
            assert!(Instant::now() < deadline, "embedding job did not finish");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn selection_failure_backoff_tracks_the_complete_configuration() {
        let (_directory, mut config) = configuration();
        config.document["semantic"]["executor"] = value("invalid");
        let mut updater = EmbeddingUpdater::default().schedule(
            &config,
            &crate::store::ConversationRevision::from_storage("input-1".into()),
        );
        assert!(matches!(updater.work, EmbeddingWork::Idle(_)));
        let failure = updater.failure.as_ref().unwrap();
        assert_eq!(failure.count, 1);
        let retry_at = failure.retry_at;
        let failed_configuration = failure.configuration.clone();

        updater = updater.schedule(
            &config,
            &crate::store::ConversationRevision::from_storage("input-2".into()),
        );
        assert_eq!(updater.failure.as_ref().unwrap().retry_at, retry_at);
        assert!(matches!(
            updater.scheduling_decision(
                &config,
                &crate::store::ConversationRevision::from_storage("input-2".into())
            ),
            ScheduleDecision::WaitingForRetry
        ));
        updater.failure.as_mut().unwrap().retry_at = Instant::now();
        updater = updater.schedule(
            &config,
            &crate::store::ConversationRevision::from_storage("input-2".into()),
        );
        assert_eq!(updater.failure.as_ref().unwrap().count, 2);

        let mut valid = config.clone();
        valid.document["semantic"]["executor"] = value("http://127.0.0.1:1");
        assert!(matches!(
            updater.scheduling_decision(
                &valid,
                &crate::store::ConversationRevision::from_storage("input-2".into())
            ),
            ScheduleDecision::Start(_)
        ));
        assert_eq!(updater.failure.as_ref().unwrap().count, 2);
        assert!(matches!(
            updater.scheduling_decision(
                &config,
                &crate::store::ConversationRevision::from_storage("input-2".into())
            ),
            ScheduleDecision::WaitingForRetry
        ));

        config.document = format!("{}\n# configuration edit\n", config.document)
            .parse()
            .unwrap();
        updater = updater.schedule(
            &config,
            &crate::store::ConversationRevision::from_storage("input-2".into()),
        );
        let failure = updater.failure.as_ref().unwrap();
        assert_ne!(failure.configuration, failed_configuration);
        assert_eq!(failure.count, 1);
        assert!(matches!(updater.work, EmbeddingWork::Idle(_)));
    }

    #[test]
    fn completion_uses_launch_identity_after_configuration_changes() {
        let (_directory, config) = configuration();
        let lock = config::lock(&config.directories.cache, "semantic.lock").unwrap();
        let mut updater = EmbeddingUpdater::default();
        updater.record_failure("previous configuration".into(), "previous failure");
        updater = updater.schedule(
            &config,
            &crate::store::ConversationRevision::from_storage("launch-input".into()),
        );
        let EmbeddingWork::Running(job) = &updater.work else {
            panic!("expected a running job");
        };
        let launch_state = job.launch.state.clone();
        let mut changed = config.clone();
        changed.document["semantic"]["executor"] = value("invalid");

        updater = updater.collect();
        assert!(matches!(
            updater.scheduling_decision(
                &changed,
                &crate::store::ConversationRevision::from_storage("new-input".into())
            ),
            ScheduleDecision::AlreadyRunning
        ));
        assert_eq!(updater.failure.as_ref().unwrap().count, 1);
        drop(lock);
        wait_for_job(&updater);
        assert!(matches!(
            updater.scheduling_decision(
                &changed,
                &crate::store::ConversationRevision::from_storage("new-input".into())
            ),
            ScheduleDecision::AlreadyRunning
        ));
        updater = updater.collect();
        assert_eq!(updater.completed.as_ref(), Some(&launch_state));
        assert!(updater.failure.is_none());
        assert!(matches!(
            updater.scheduling_decision(
                &config,
                &crate::store::ConversationRevision::from_storage("launch-input".into())
            ),
            ScheduleDecision::Start(_)
        ));
        assert!(matches!(
            updater.scheduling_decision(
                &config,
                &crate::store::ConversationRevision::from_storage("new-input".into())
            ),
            ScheduleDecision::Start(_)
        ));
        assert!(matches!(
            updater.scheduling_decision(
                &changed,
                &crate::store::ConversationRevision::from_storage("new-input".into())
            ),
            ScheduleDecision::SelectionFailed
        ));
        assert!(matches!(
            updater.scheduling_decision(
                &config,
                &crate::store::ConversationRevision::from_storage("launch-input".into())
            ),
            ScheduleDecision::Start(_)
        ));
        assert!(updater.failure.is_some());
    }

    #[test]
    fn failed_and_panicked_jobs_retry_the_saved_configuration() {
        let (_directory, config) = configuration();
        let configuration = record::digest(&[&config.document.to_string()]);
        for panics in [false, true] {
            let mut updater = EmbeddingUpdater {
                work: EmbeddingWork::Running(EmbeddingJob {
                    launch: Launch {
                        state: "failed launch".into(),
                        configuration: configuration.clone(),
                    },
                    thread: thread::spawn(move || {
                        assert!(!panics, "injected embedding panic");
                        (
                            semantic::Planner::default(),
                            Err(anyhow::anyhow!("injected failure")),
                        )
                    }),
                }),
                completed: Some("prior success".into()),
                failure: None,
            };
            wait_for_job(&updater);
            let collected_at = Instant::now();
            updater = updater.collect();
            assert_eq!(updater.completed.as_deref(), Some("prior success"));
            let failure = updater.failure.as_ref().unwrap();
            assert_eq!(failure.configuration, configuration);
            assert_eq!(failure.count, 1);
            assert!(failure.retry_at >= collected_at + Duration::from_secs(2));
            assert!(matches!(
                updater.scheduling_decision(
                    &config,
                    &crate::store::ConversationRevision::from_storage("retry-input".into())
                ),
                ScheduleDecision::WaitingForRetry
            ));
            let mut changed = config.clone();
            changed.document["semantic"]["model"] = value("other-model");
            updater = updater.schedule(
                &changed,
                &crate::store::ConversationRevision::from_storage("retry-input".into()),
            );
            assert_eq!(updater.completed.as_deref(), Some("prior success"));
            assert_eq!(
                updater.failure.as_ref().unwrap().configuration,
                configuration
            );
            wait_for_job(&updater);
            updater = updater.collect();
            assert!(updater.failure.is_none());
            assert!(matches!(
                updater.scheduling_decision(
                    &changed,
                    &crate::store::ConversationRevision::from_storage("retry-input".into())
                ),
                ScheduleDecision::Start(_)
            ));
        }
    }
}
