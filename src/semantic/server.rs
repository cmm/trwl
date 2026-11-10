//! Reusable CPU inference, separate from index updates and GPU workers.
//! One process per runtime/data/cache directory combination and executable serves
//! clients serially. It keeps one model selection, exits after 30 idle seconds,
//! and stores no input text.
use super::{
    BATCH_SIZE, ChunkInput, DOCUMENT_PREFIX, DocumentTokenizer, Environment, Executor,
    QUERY_PREFIX, Selection, prepare_chunks,
};
use crate::{config, record::digest};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{
            ffi::OsStrExt,
            fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
            net::{UnixListener, UnixStream},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

const INITIALIZE: Duration = Duration::from_secs(120);
const REQUEST: Duration = Duration::from_secs(60);
const IDLE: Duration = Duration::from_secs(30);
const FRAME_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Serialize, Deserialize)]
enum InputKind {
    Query,
    Document,
}

#[derive(Serialize, Deserialize)]
enum Request<S = Selection, T = Vec<String>> {
    Select(S),
    Embed { texts: T, kind: InputKind },
    Prepare { texts: T },
    Stop,
}

#[derive(Serialize, Deserialize)]
enum Reply {
    Ready,
    Vectors(Vec<Vec<f32>>),
    Chunks(Vec<Vec<ChunkInput>>),
    Failed(String),
}

fn send(stream: &mut UnixStream, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(
        bytes.len() <= FRAME_LIMIT,
        "inference frame exceeds size limit"
    );
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn receive<T: DeserializeOwned>(stream: &mut UnixStream) -> Result<T> {
    let mut size = [0; 4];
    stream.read_exact(&mut size)?;
    let size = u32::from_le_bytes(size) as usize;
    ensure!(size <= FRAME_LIMIT, "inference frame exceeds size limit");
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes)?;
    serde_json::from_slice(&bytes).context("invalid inference frame")
}

struct Directory(File);
impl Directory {
    fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(path)?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o077 == 0,
            "inference directory must be private to the current user: {}",
            path.display()
        );
        Ok(Self(file))
    }

    fn path(&self) -> PathBuf {
        // A directory descriptor keeps socket paths below sun_path's length limit.
        PathBuf::from(format!("/proc/self/fd/{}", self.0.as_raw_fd()))
    }
}

fn directory(directories: &crate::directories::Directories) -> Result<Directory> {
    let executable = fs::metadata(std::env::current_exe()?)?;
    fs::create_dir_all(&directories.cache)?;
    let cache = fs::metadata(&directories.cache)?;
    // One executable can load different runtimes through package wrappers.
    // A resident server must keep the requesting client's runtime selection.
    let identity = digest(&[
        &format!(
            "{}:{}:{}:{}:{}:{}:{}",
            executable.dev(),
            executable.ino(),
            executable.len(),
            executable.ctime(),
            executable.ctime_nsec(),
            cache.dev(),
            cache.ino()
        ),
        &format!("{:?}", std::env::var_os("ORT_DYLIB_PATH")),
        &format!("{:?}", std::env::var_os("OCL_ICD_VENDORS")),
    ]);
    let parent = directories.index_runtime()?.join("inference");
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&parent)?;
    let parent = Directory::open(&parent)?;
    let path = parent.path().join(identity);
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&path)?;
    Directory::open(&path)
}

fn connect_with_retry(directory: &Directory, timeout: Duration) -> Result<UnixStream> {
    let path = directory.path().join("socket");
    let bytes = path.as_os_str().as_bytes();
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    ensure!(
        bytes.len() < address.sun_path.len(),
        "inference socket path is too long"
    );
    address.sun_family = libc::AF_UNIX as _;
    for (slot, byte) in address.sun_path.iter_mut().zip(bytes) {
        *slot = *byte as _;
    }
    let descriptor = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let stream = UnixStream::from(unsafe { OwnedFd::from_raw_fd(descriptor) });
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let connected = unsafe {
            libc::connect(
                stream.as_raw_fd(),
                &address as *const _ as *const libc::sockaddr,
                (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1) as _,
            )
        };
        if connected == 0 {
            break;
        }
        let error = std::io::Error::last_os_error();
        // Linux returns EAGAIN without starting an AF_UNIX connection when the
        // accept queue is full. Do not let a saturated server block connect forever.
        if error.raw_os_error() != Some(libc::EAGAIN) {
            return Err(error.into());
        }
        ensure!(Instant::now() < deadline, "inference connection timed out");
        thread::sleep(Duration::from_millis(10));
    }
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(stream)
}

struct Startup {
    _guard: File,
    directory: Directory,
    deadline: Instant,
}
impl Startup {
    fn acquire(directory: Directory) -> Result<Self> {
        // Lock acquisition and child startup share the same budget.
        let deadline = Instant::now() + Duration::from_secs(5);
        let guard = loop {
            if let Some(guard) = config::try_lock(&directory.path(), "start.lock")? {
                break guard;
            }
            ensure!(
                Instant::now() < deadline,
                "inference startup lock timed out"
            );
            thread::sleep(Duration::from_millis(10));
        };
        Ok(Self {
            directory,
            _guard: guard,
            deadline,
        })
    }

    fn connect_or_spawn(self, directories: &crate::directories::Directories) -> Result<UnixStream> {
        // Another starter may have published an endpoint while we waited for the lock.
        if let Ok(stream) = connect_with_retry(&self.directory, INITIALIZE) {
            return Ok(stream);
        }
        let mut command = Command::new(std::env::current_exe()?);
        directories.inference_environment(&mut command)?;
        command
            .arg("embedding-server")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().context("starting inference server")?;
        self.wait_for_startup(child)
    }

    fn wait_for_startup(&self, mut child: Child) -> Result<UnixStream> {
        loop {
            if let Ok(stream) = connect_with_retry(&self.directory, INITIALIZE) {
                reap_after_exit(child);
                return Ok(stream);
            }
            if let Some(status) = child.try_wait()? {
                bail!("inference server exited: {status}");
            }
            if Instant::now() >= self.deadline {
                kill_and_reap(child);
                bail!("inference server startup timed out");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

fn reap_after_exit(mut child: Child) {
    // The updater can outlive many idle-expired servers.
    // Dropping Child would leave their exit statuses unreaped.
    thread::spawn(move || {
        let _ = child.wait();
    });
}

fn kill_and_reap(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

pub(super) struct Client {
    directories: crate::directories::Directories,
    selection: Selection,
}
impl Client {
    pub(super) fn new(
        directories: &crate::directories::Directories,
        selection: &Selection,
    ) -> Self {
        Self {
            directories: directories.clone(),
            selection: selection.clone(),
        }
    }

    fn connection(
        directories: &crate::directories::Directories,
        selection: &Selection,
    ) -> Result<SelectedConnection> {
        for attempt in 0..2 {
            let mut stream = Self::connect_or_start(directories)?;
            let reply = send(&mut stream, &Request::<_, &[&str]>::Select(selection))
                .and_then(|()| receive(&mut stream));
            match reply {
                Ok(Reply::Ready) => {
                    stream.set_read_timeout(Some(REQUEST))?;
                    stream.set_write_timeout(Some(REQUEST))?;
                    return Ok(SelectedConnection(stream));
                }
                Ok(Reply::Failed(message)) => bail!("inference initialization: {message}"),
                Ok(_) => bail!("unexpected inference initialization response"),
                // An idle server can exit between connect and selection. Retry
                // transport failure once; a model error is returned unchanged.
                Err(error)
                    if attempt == 0
                        && error.downcast_ref::<std::io::Error>().is_some_and(|e| {
                            matches!(
                                e.kind(),
                                std::io::ErrorKind::UnexpectedEof
                                    | std::io::ErrorKind::BrokenPipe
                                    | std::io::ErrorKind::ConnectionReset
                            )
                        }) => {}
                Err(error) => return Err(error),
            }
        }
        unreachable!()
    }

    fn connect_or_start(directories: &crate::directories::Directories) -> Result<UnixStream> {
        let directory = directory(directories)?;
        match connect_with_retry(&directory, INITIALIZE) {
            Ok(stream) => Ok(stream),
            Err(_) => Startup::acquire(directory)?.connect_or_spawn(directories),
        }
    }

    fn request(&self, request: &Request<&Selection, &[&str]>) -> Result<Reply> {
        // Do not hold a server connection while acquiring an index/cache lock.
        // Updaters can hold that lock while waiting for their own inference.
        Self::connection(&self.directories, &self.selection)?.request(request)
    }

    pub(super) fn prepare(&mut self, texts: &[&str]) -> Result<Vec<Vec<ChunkInput>>> {
        match self.request(&Request::Prepare { texts })? {
            Reply::Chunks(chunks) => Ok(chunks),
            _ => bail!("unexpected planning response"),
        }
    }

    pub(super) fn embed(&mut self, texts: &[&str], prefix: &str) -> Result<Vec<Vec<f32>>> {
        let kind = match prefix {
            QUERY_PREFIX => InputKind::Query,
            DOCUMENT_PREFIX => InputKind::Document,
            _ => bail!("unknown embedding prefix"),
        };
        match self.request(&Request::Embed { texts, kind })? {
            Reply::Vectors(vectors) => {
                ensure!(vectors.len() == texts.len(), "inference batch size differs");
                Ok(vectors)
            }
            _ => bail!("unexpected inference response"),
        }
    }
}

struct SelectedConnection(UnixStream);
impl SelectedConnection {
    fn request(mut self, request: &impl Serialize) -> Result<Reply> {
        send(&mut self.0, request)?;
        match receive(&mut self.0)? {
            Reply::Failed(message) => bail!("inference failed: {message}"),
            reply => Ok(reply),
        }
    }
}

pub(crate) fn stop(directories: &crate::directories::Directories) -> Result<()> {
    let path = directories.index_runtime()?.join("inference");
    if !path.try_exists()? {
        return Ok(());
    }
    let parent = Directory::open(&path)?;
    for entry in fs::read_dir(parent.path())? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let directory = Directory::open(&entry.path())?;
        if let Ok(stream) = connect_with_retry(&directory, INITIALIZE) {
            StopAcknowledged::request(&directory, stream)?.wait_for_lock_release()?;
        }
    }
    Ok(())
}

struct StopAcknowledged<'a> {
    directory: &'a Directory,
    _stream: UnixStream,
}
impl<'a> StopAcknowledged<'a> {
    fn request(directory: &'a Directory, mut stream: UnixStream) -> Result<Self> {
        send(&mut stream, &Request::<Selection, Vec<String>>::Stop)?;
        let _: Reply = receive(&mut stream)?;
        Ok(Self {
            directory,
            _stream: stream,
        })
    }

    fn wait_for_lock_release(self) -> Result<()> {
        // A reply precedes endpoint teardown. Lock release does not reap the child;
        // that belongs to the process which started it.
        let deadline = Instant::now() + INITIALIZE;
        while config::is_locked(&self.directory.path(), "server.lock")? {
            ensure!(
                Instant::now() < deadline,
                "inference server is still stopping"
            );
            thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }
}

struct Endpoint {
    directory: Directory,
    _guard: File,
}
impl Endpoint {
    fn acquire(directory: Directory) -> Result<Option<Self>> {
        Ok(
            config::try_lock(&directory.path(), "server.lock")?.map(|guard| Self {
                directory,
                _guard: guard,
            }),
        )
    }

    fn listen(&self) -> Result<UnixListener> {
        let path = self.directory.path();
        let _ = fs::remove_file(path.join("socket"));
        let listener = UnixListener::bind(path.join("socket"))?;
        listener.set_nonblocking(true)?;
        fs::write(path.join("pid"), std::process::id().to_string())?;
        Ok(listener)
    }

    fn remove_files(&mut self) {
        let _ = fs::remove_file(self.directory.path().join("socket"));
        let _ = fs::remove_file(self.directory.path().join("pid"));
    }
}
impl Drop for Endpoint {
    fn drop(&mut self) {
        // Drop runs before the fields, so file removal remains under server.lock.
        self.remove_files();
    }
}

struct Watchdog(mpsc::Sender<Option<Duration>>);
struct Deadline<'a>(&'a mut Watchdog);
impl Watchdog {
    fn new() -> Self {
        let (send, receive) = mpsc::channel::<Option<Duration>>();
        thread::spawn(move || {
            while let Ok(duration) = receive.recv() {
                if let Some(duration) = duration
                    && matches!(
                        receive.recv_timeout(duration),
                        Err(mpsc::RecvTimeoutError::Timeout)
                    )
                {
                    // A blocked native initializer or inference call cannot
                    // run destructors. The kernel releases memory and locks;
                    // the next client replaces the stale endpoint.
                    unsafe {
                        libc::_exit(1);
                    }
                }
            }
        });
        Self(send)
    }

    fn arm(&mut self, duration: Duration) -> Deadline<'_> {
        let _ = self.0.send(Some(duration));
        Deadline(self)
    }
}
impl Drop for Deadline<'_> {
    fn drop(&mut self) {
        let _ = self.0.0.send(None);
    }
}

struct IdleLease {
    last_completed: Instant,
}

impl IdleLease {
    fn new() -> Self {
        Self {
            last_completed: Instant::now(),
        }
    }

    fn remaining(&self) -> Duration {
        IDLE.saturating_sub(self.last_completed.elapsed())
    }

    fn connection_completed(&mut self) {
        self.last_completed = Instant::now();
    }
}

#[derive(Default)]
struct Resident {
    runtime: Option<InitializedRuntime>,
}

struct InitializedRuntime {
    // Drop the model before releasing the runtime it uses.
    model: Option<LoadedModel>,
    environment: Environment,
}

struct LoadedModel {
    selection: Selection,
    executor: Executor,
}

impl Resident {
    fn select(
        &mut self,
        selection: Selection,
        watchdog: &mut Watchdog,
    ) -> Result<&mut LoadedModel> {
        let directory = selection
            .directory
            .as_ref()
            .context("inference server requires a built-in model")?;
        if self
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.model.as_ref())
            .is_none_or(|model| model.selection != selection)
        {
            let _deadline = watchdog.arm(INITIALIZE);
            let runtime = match &mut self.runtime {
                Some(runtime) => runtime,
                None => self.runtime.insert(InitializedRuntime {
                    model: None,
                    environment: super::load_runtime()?,
                }),
            };
            runtime.model = None;
            let executor = Executor::builtin(directory, &runtime.environment)?;
            runtime.model = Some(LoadedModel {
                selection,
                executor,
            });
        }
        self.runtime
            .as_mut()
            .and_then(|runtime| runtime.model.as_mut())
            .context("inference executor missing")
    }
}

enum ConnectionSelection<'a> {
    Unselected,
    Selected(&'a mut LoadedModel),
}
impl ConnectionSelection<'_> {
    fn model(&mut self) -> Result<&mut LoadedModel> {
        match self {
            Self::Unselected => bail!("inference selection missing"),
            Self::Selected(model) => Ok(model),
        }
    }
}

impl LoadedModel {
    fn prepare(&mut self, texts: &[String]) -> Result<Vec<Vec<ChunkInput>>> {
        let texts: Vec<_> = texts.iter().map(String::as_str).collect();
        ensure!(texts.len() <= BATCH_SIZE, "planning batch is too large");
        let Executor::Builtin { tokenizer, .. } = &mut self.executor else {
            bail!("planning requires a built-in model")
        };
        DocumentTokenizer::prepare(tokenizer, |tokenizer| {
            prepare_chunks(&texts, Some(tokenizer))
        })
    }

    fn embed(&mut self, texts: &[String], kind: InputKind) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<_> = texts.iter().map(String::as_str).collect();
        self.executor.embed(
            &texts,
            match kind {
                InputKind::Query => QUERY_PREFIX,
                InputKind::Document => DOCUMENT_PREFIX,
            },
        )
    }
}

pub(crate) fn run(directories: &crate::directories::Directories) -> Result<()> {
    let Some(endpoint) = Endpoint::acquire(directory(directories)?)? else {
        return Ok(());
    };
    let listener = endpoint.listen()?;
    // ONNX Runtime permits one environment per process, shared across model changes.
    let mut resident = Resident::default();
    let mut watchdog = Watchdog::new();
    let mut idle = IdleLease::new();
    loop {
        let (mut stream, _) = match listener.accept() {
            Ok(client) => client,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if idle.remaining().is_zero() {
                    return Ok(());
                }
                let mut descriptor = libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let remaining = idle.remaining().as_millis() as i32;
                // poll sleeps until a client arrives or the idle deadline ends.
                let result = unsafe { libc::poll(&mut descriptor, 1, remaining) };
                if result < 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
                {
                    return Err(std::io::Error::last_os_error().into());
                }
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        stream.set_read_timeout(Some(REQUEST))?;
        stream.set_write_timeout(Some(REQUEST))?;
        let mut selected = ConnectionSelection::Unselected;
        while let Ok(request) = receive::<Request>(&mut stream) {
            let reply = match request {
                Request::Stop => {
                    send(&mut stream, &Reply::Ready)?;
                    return Ok(());
                }
                Request::Select(selection) => {
                    // Release the model borrow before replacement, including failed selection.
                    selected = ConnectionSelection::Unselected;
                    match resident.select(selection, &mut watchdog) {
                        Ok(model) => {
                            selected = ConnectionSelection::Selected(model);
                            Reply::Ready
                        }
                        Err(error) => Reply::Failed(format!("{error:#}")),
                    }
                }
                Request::Prepare { texts } => {
                    let _deadline = watchdog.arm(REQUEST);
                    let result = selected.model().and_then(|model| model.prepare(&texts));
                    match result {
                        Ok(chunks) => Reply::Chunks(chunks),
                        Err(error) => Reply::Failed(format!("{error:#}")),
                    }
                }
                Request::Embed { texts, kind } => {
                    let _deadline = watchdog.arm(REQUEST);
                    let result = selected.model().and_then(|model| model.embed(&texts, kind));
                    match result {
                        Ok(vectors) => Reply::Vectors(vectors),
                        Err(error) => Reply::Failed(format!("{error:#}")),
                    }
                }
            };
            if send(&mut stream, &reply).is_err() {
                break;
            }
        }
        idle.connection_completed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_child() {
        let Ok(mode) = std::env::var("TRWL_WATCHDOG_TEST") else {
            return;
        };
        let mut watchdog = Watchdog::new();
        if mode == "complete" {
            for _ in 0..3 {
                let _deadline = watchdog.arm(Duration::from_millis(100));
            }
            thread::sleep(Duration::from_millis(250));
        } else {
            let _deadline = watchdog.arm(Duration::from_millis(100));
            thread::sleep(Duration::from_secs(10));
        }
    }

    #[test]
    fn watchdog_exits_on_expiry_and_disarms_completed_deadlines() {
        for mode in ["blocked", "complete"] {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "semantic::server::tests::watchdog_child"])
                .env("TRWL_WATCHDOG_TEST", mode)
                .stdout(Stdio::null())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(3);
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                if Instant::now() >= deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("watchdog did not end blocked inference");
                }
                thread::sleep(Duration::from_millis(10));
            };
            assert_eq!(status.success(), mode == "complete");
        }
    }
}
