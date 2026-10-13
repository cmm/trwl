//! A full build owns one inference process. The worker owns the shared GPU
//! lock; the parent owns index/cache writes and persists every completed batch.
//! Socket deadlines bound initialization (120 s) and each inference batch
//! (60 s) separately. A worker that fails, crashes or times out is killed once
//! and the parent finishes the fill on CPU; a worker taken down by SIGINT,
//! SIGTERM or SIGHUP cancels the fill instead.
use super::{Accelerator, DOCUMENT_PREFIX, Executor, Selection};
use crate::config::try_lock;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    io::{BufRead, BufReader, Write},
    os::{
        fd::{BorrowedFd, IntoRawFd, OwnedFd},
        unix::{
            net::UnixStream,
            process::{CommandExt, ExitStatusExt},
        },
    },
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::Duration,
};

const INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(120);
const BATCH_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Serialize, Deserialize)]
enum Initialization {
    Ready(Accelerator),
    Busy,
    Cpu,
    Failed(String),
}

#[derive(Serialize, Deserialize)]
enum Reply {
    Vectors(Vec<Vec<f32>>),
    Failed(String),
}

fn send(stream: &mut impl Write, value: &impl Serialize) -> Result<()> {
    // Writing JSON tokens directly issues a socket syscall for every number
    // and delimiter in the embedding batch.
    let mut frame = serde_json::to_vec(value)?;
    frame.push(b'\n');
    stream.write_all(&frame)?;
    Ok(())
}

fn receive<T: DeserializeOwned>(stream: &mut BufReader<UnixStream>) -> Result<T> {
    let mut line = String::new();
    ensure!(
        stream.read_line(&mut line)? > 0,
        "embedding worker closed its socket"
    );
    serde_json::from_str(&line).context("invalid embedding worker response")
}

pub(super) struct Client {
    child: Child,
    stream: BufReader<UnixStream>,
    pending: Option<Result<()>>,
}

impl Drop for Client {
    fn drop(&mut self) {
        // Killing and reaping precedes CPU recovery and releasing the fill lock.
        // The child owns the GPU lock until it actually exits.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Client {
    pub(super) fn start(selection: &Selection) -> Result<Option<Self>> {
        let mut process = Command::new(std::env::current_exe()?);
        process.arg("embedding-worker");
        Self::start_command(process, selection, INITIALIZATION_TIMEOUT, BATCH_TIMEOUT)
    }

    fn start_command(
        mut process: Command,
        selection: &Selection,
        timeout: Duration,
        batch_timeout: Duration,
    ) -> Result<Option<Self>> {
        let (stream, child_stream) = UnixStream::pair()?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        process
            .stdin(Stdio::from(OwnedFd::from(child_stream)))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let parent = std::process::id();
        unsafe {
            process.pre_exec(move || {
                // Linux ties this to the spawning thread. Its exit also ends
                // an abandoned background fill, even if the daemon lives on.
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() as u32 != parent {
                    return Err(std::io::Error::other("embedding parent exited"));
                }
                Ok(())
            });
        }
        let child = process.spawn().context("starting embedding worker")?;
        // Command retains its configured stdin after spawning. Close our copy
        // of the child's socket so an initialization crash delivers EOF.
        drop(process);
        let mut client = Self {
            child,
            stream: BufReader::new(stream),
            pending: None,
        };
        let reply =
            send(client.stream.get_mut(), selection).and_then(|()| receive(&mut client.stream));
        match reply {
            Ok(Initialization::Ready(device)) => {
                eprintln!("trwl: full embedding build using {}", device.as_str());
                client
                    .stream
                    .get_ref()
                    .set_read_timeout(Some(batch_timeout))?;
                client
                    .stream
                    .get_ref()
                    .set_write_timeout(Some(batch_timeout))?;
                Ok(Some(client))
            }
            Ok(Initialization::Busy) => {
                eprintln!("trwl: another embedding build holds the GPU lock; using CPU");
                Ok(None)
            }
            Ok(Initialization::Cpu) => {
                eprintln!("trwl: full embedding build using cpu");
                Ok(None)
            }
            Ok(Initialization::Failed(message)) => bail!("embedding initialization: {message}"),
            Err(error) => {
                client.recover(error)?;
                Ok(None)
            }
        }
    }

    fn recover(&mut self, error: anyhow::Error) -> Result<()> {
        let _ = self.child.kill();
        let status = self.child.wait()?;
        ensure!(
            !matches!(
                status.signal(),
                Some(libc::SIGINT | libc::SIGTERM | libc::SIGHUP)
            ),
            "embedding worker cancelled: {status}"
        );
        eprintln!("trwl: GPU embedding worker failed ({status}): {error:#}; finishing on CPU");
        Ok(())
    }

    /// None means the worker was reaped; the caller finishes this fill on CPU.
    pub(super) fn embed(&mut self, texts: &[String]) -> Result<Option<Vec<Vec<f32>>>> {
        let sent = self
            .pending
            .take()
            .unwrap_or_else(|| send(self.stream.get_mut(), &texts));
        let reply = sent.and_then(|()| receive(&mut self.stream));
        match reply {
            Ok(Reply::Vectors(vectors)) => {
                ensure!(
                    vectors.len() == texts.len(),
                    "embedding worker returned the wrong batch size"
                );
                Ok(Some(vectors))
            }
            Ok(Reply::Failed(message)) => {
                self.recover(anyhow::anyhow!(message))?;
                Ok(None)
            }
            Err(error) => {
                self.recover(error)?;
                Ok(None)
            }
        }
    }

    pub(super) fn prefetch(&mut self, texts: &[String]) {
        assert!(self.pending.is_none(), "only one batch may be in flight");
        // Defer a submission error until the next embed call, so the caller
        // still commits the successful batch it has already received.
        self.pending = Some(send(self.stream.get_mut(), &texts));
    }
}

fn lock_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("TRWL_GPU_LOCK_DIR") {
        return path.into();
    }
    // The runtime directory is per-user and cleared at logout, matching a
    // short-lived build lock; use a per-user temporary directory when unset.
    if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime).join("trwl");
    }
    let temporary = std::env::var_os("TEMPDIR")
        .or_else(|| std::env::var_os("TMPDIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let uid = unsafe { libc::geteuid() };
    temporary.join(uid.to_string()).join("trwl")
}

fn worker_stream() -> Result<UnixStream> {
    // The parent supplies a bidirectional socket as stdin, leaving stdout free
    // for native libraries without corrupting inference responses.
    let fd = unsafe { BorrowedFd::borrow_raw(libc::STDIN_FILENO) }.try_clone_to_owned()?;
    Ok(UnixStream::from(fd))
}

pub(crate) fn run() -> Result<()> {
    serve(BufReader::new(worker_stream()?), |selection| {
        Ok(
            Executor::accelerated(selection)?.map(|(mut executor, device)| {
                (
                    move |texts: &[String]| executor.embed(texts, DOCUMENT_PREFIX),
                    device,
                )
            }),
        )
    })
}

fn reserve_gpu() -> Result<bool> {
    let directory = lock_directory();
    let Some(guard) = try_lock(&directory, "gpu.lock")
        .with_context(|| format!("opening GPU lock in {}", directory.display()))?
    else {
        return Ok(false);
    };
    // Keep the descriptor open until process exit, including native runtime
    // destruction after initialization errors. The kernel releases the lock.
    let _lock_fd = guard.into_raw_fd();
    Ok(true)
}

fn serve<F>(
    mut stream: BufReader<UnixStream>,
    initialize: impl FnOnce(&Selection) -> Result<Option<(F, Accelerator)>>,
) -> Result<()>
where
    F: FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
{
    let selection: Selection = receive(&mut stream)?;
    match reserve_gpu() {
        Ok(true) => {}
        Ok(false) => return send(stream.get_mut(), &Initialization::Busy),
        Err(error) => {
            return send(
                stream.get_mut(),
                &Initialization::Failed(format!("{error:#}")),
            );
        }
    }
    let (mut embed, device) = match initialize(&selection) {
        Ok(Some(result)) => result,
        Ok(None) => {
            // End any contexts left by failed GPU initialization before the parent
            // continues locally. A CPU fill must not reserve the GPU lock.
            return send(stream.get_mut(), &Initialization::Cpu);
        }
        Err(error) => {
            return send(
                stream.get_mut(),
                &Initialization::Failed(format!("{error:#}")),
            );
        }
    };
    send(stream.get_mut(), &Initialization::Ready(device))?;
    loop {
        let texts: Vec<String> = receive(&mut stream)?;
        match embed(&texts) {
            Ok(vectors) => send(stream.get_mut(), &Reply::Vectors(vectors))?,
            Err(error) => {
                return send(stream.get_mut(), &Reply::Failed(format!("{error:#}")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::is_locked;
    use std::{fs, path::Path, thread, time::Instant};

    fn start(mode: &str, directory: &Path) -> Result<Option<Client>> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args(["--exact", "semantic::worker::tests::child", "--nocapture"])
            .env("TRWL_TEST_WORKER", mode)
            .env("TRWL_GPU_LOCK_DIR", directory);
        let selection = Selection {
            key: "test".into(),
            executor: "builtin".into(),
            model: "test".into(),
            directory: Some(directory.into()),
        };
        Client::start_command(
            command,
            &selection,
            Duration::from_secs(2),
            Duration::from_millis(500),
        )
    }

    // The server runs in a subprocess with only model initialization and
    // inference replaced, so lock ownership and teardown follow production.
    #[test]
    fn child() {
        let Ok(mode) = std::env::var("TRWL_TEST_WORKER") else {
            return;
        };
        serve(BufReader::new(worker_stream().unwrap()), |_| {
            let directory = PathBuf::from(std::env::var_os("TRWL_GPU_LOCK_DIR").unwrap());
            assert!(
                is_locked(&directory, "gpu.lock").unwrap(),
                "initialization needs the GPU lock"
            );
            match mode.as_str() {
                "init-error" => bail!("model file missing"),
                "cpu" => return Ok(None),
                "init-crash" => unsafe {
                    libc::raise(libc::SIGKILL);
                },
                "init-hang" => thread::sleep(Duration::from_secs(30)),
                _ => {}
            }
            Ok(Some((
                |texts: &[String]| {
                    match mode.as_str() {
                        "hang" => thread::sleep(Duration::from_secs(30)),
                        "crash" => unsafe {
                            libc::raise(libc::SIGKILL);
                        },
                        "gpu-error" => bail!("inference failed"),
                        _ => {}
                    }
                    Ok(texts
                        .iter()
                        .map(|text| vec![text.len() as f32, 1.0])
                        .collect())
                },
                Accelerator::Cuda,
            )))
        })
        .unwrap();
        if mode == "gpu-error" || mode == "cpu" {
            // Recovery must reap the process rather than assuming an
            // error response means its allocations/lock are released.
            thread::sleep(Duration::from_secs(30));
        }
    }

    #[test]
    fn shared_lock_and_batches_follow_worker_lifetime() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = start("success", dir.path()).unwrap().unwrap();
        assert!(is_locked(dir.path(), "gpu.lock").unwrap());
        // A contending worker must skip initialization, even if it would fail.
        assert!(start("init-error", dir.path()).unwrap().is_none());
        for _ in 0..2 {
            assert_eq!(
                first.embed(&["text".into()]).unwrap().unwrap(),
                [vec![4.0, 1.0]]
            );
        }
        let next = ["one".into(), "two".into()];
        first.prefetch(&next);
        assert_eq!(
            first.embed(&next).unwrap().unwrap(),
            [vec![3.0, 1.0], vec![3.0, 1.0]]
        );
        drop(first);
        assert!(!is_locked(dir.path(), "gpu.lock").unwrap());
        assert!(start("success", dir.path()).unwrap().is_some());
    }

    #[test]
    fn gpu_failure_crash_and_timeout_release_lock_before_cpu_recovery() {
        let dir = tempfile::tempdir().unwrap();
        for mode in ["gpu-error", "crash", "hang"] {
            let mut client = start(mode, dir.path()).unwrap().unwrap();
            assert!(client.embed(&["text".into()]).unwrap().is_none(), "{mode}");
            assert!(!is_locked(dir.path(), "gpu.lock").unwrap(), "{mode}");
        }
    }

    #[test]
    fn initialization_crash_and_timeout_fall_back_but_errors_propagate() {
        let dir = tempfile::tempdir().unwrap();
        for mode in ["cpu", "init-crash", "init-hang"] {
            let started = Instant::now();
            assert!(start(mode, dir.path()).unwrap().is_none(), "{mode}");
            if mode == "init-crash" {
                assert!(
                    started.elapsed() < Duration::from_secs(1),
                    "a dead worker must not wait for the two-second initialization timeout"
                );
            }
            assert!(!is_locked(dir.path(), "gpu.lock").unwrap());
        }
        assert!(start("init-error", dir.path()).is_err());
        assert!(!is_locked(dir.path(), "gpu.lock").unwrap());
    }

    #[test]
    fn cancellation_does_not_request_cpu_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut client = start("success", dir.path()).unwrap().unwrap();
        unsafe {
            libc::kill(client.child.id() as i32, libc::SIGTERM);
        }
        client.child.wait().unwrap();
        assert!(
            client
                .embed(&["text".into()])
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
    }

    #[test]
    fn prefetch_to_an_exited_worker_requests_cpu_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut client = start("success", dir.path()).unwrap().unwrap();
        let texts = ["text".into()];
        assert!(client.embed(&texts).unwrap().is_some());
        client.child.kill().unwrap();
        client.child.wait().unwrap();
        client.prefetch(&texts);
        assert!(client.embed(&texts).unwrap().is_none());
        assert!(!is_locked(dir.path(), "gpu.lock").unwrap());
    }

    #[test]
    fn parent_child() {
        let Some(directory) = std::env::var_os("TRWL_TEST_PARENT") else {
            return;
        };
        let directory = PathBuf::from(directory);
        let _worker = start("success", &directory).unwrap().unwrap();
        fs::write(directory.join("ready"), b"").unwrap();
        thread::sleep(Duration::from_secs(30));
    }

    #[test]
    fn parent_exit_ends_worker_and_releases_lock() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "semantic::worker::tests::parent_child",
                "--nocapture",
            ])
            .env("TRWL_TEST_PARENT", dir.path())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !dir.path().join("ready").exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        let ready = dir.path().join("ready").exists();
        parent.kill().unwrap();
        parent.wait().unwrap();
        assert!(ready, "parent did not start its worker");
        while is_locked(dir.path(), "gpu.lock").unwrap() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!is_locked(dir.path(), "gpu.lock").unwrap());
    }
}
