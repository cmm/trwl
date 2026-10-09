use crate::record::{IndexMode, Provider};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};
use toml_edit::{DocumentMut, value};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub name: String,
    pub provider: Provider,
    pub root: PathBuf,
}

#[derive(Clone)]
pub struct Config {
    pub root: PathBuf,
    pub document: DocumentMut,
}

impl Config {
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join("config.toml");
        let document: DocumentMut = match fs::read_to_string(&path) {
            Ok(text) => text
                .parse()
                .with_context(|| format!("parsing {}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => DocumentMut::new(),
            Err(error) => return Err(error.into()),
        };
        if let Some(mode) = document.get("indexing").and_then(|table| table.get("mode")) {
            mode.as_str()
                .context("indexing.mode must be a string")?
                .parse::<IndexMode>()?;
        }
        Ok(Self {
            root: root.into(),
            document,
        })
    }

    pub fn string(&self, table: &str, key: &str) -> Option<&str> {
        self.document.get(table)?.get(key)?.as_str()
    }

    pub fn flag(&self, table: &str, key: &str, default: bool) -> bool {
        self.document
            .get(table)
            .and_then(|t| t.get(key))
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    pub fn automatic(&self) -> bool {
        self.mode() == IndexMode::Auto
    }

    pub fn mode(&self) -> IndexMode {
        self.string("indexing", "mode")
            .unwrap_or("auto")
            .parse()
            .expect("validated index mode")
    }

    pub fn semantic(&self) -> bool {
        self.flag("search", "semantic", false)
    }

    pub fn edit(root: &Path, edit: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<Self> {
        let _lock = lock(root, "config.lock")?;
        let mut config = Self::load(root)?;
        edit(&mut config.document)?;
        atomic_write(
            &root.join("config.toml"),
            config.document.to_string().as_bytes(),
        )?;
        Ok(config)
    }

    pub fn set_mode(root: &Path, mode: IndexMode) -> Result<Self> {
        Self::edit(root, |doc| {
            doc["indexing"]["mode"] = value(mode.as_str());
            Ok(())
        })
    }

    pub fn sources(&self) -> Result<Vec<Source>> {
        self.sources_with(
            || {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .context("HOME is unset")
            },
            |name, fallback| {
                std::env::var_os(name)
                    .map(PathBuf::from)
                    .unwrap_or(fallback)
            },
        )
    }

    fn sources_with(
        &self,
        home: impl FnOnce() -> Result<PathBuf>,
        env_path: impl Fn(&str, PathBuf) -> PathBuf,
    ) -> Result<Vec<Source>> {
        #[derive(Deserialize, Default)]
        struct Sources {
            #[serde(default)]
            roots: Vec<Source>,
        }
        #[derive(Deserialize)]
        struct Entries {
            #[serde(default)]
            sources: Sources,
        }
        let mut sources = toml_edit::de::from_str::<Entries>(&self.document.to_string())?
            .sources
            .roots;
        if self.flag("sources", "automatic", true) {
            let home = home()?;
            for (provider, root) in [
                (
                    Provider::Opencode,
                    env_path("XDG_DATA_HOME", home.join(".local/share"))
                        .join("opencode/opencode.db"),
                ),
                (
                    Provider::Codex,
                    env_path("CODEX_HOME", home.join(".codex")).join("sessions"),
                ),
                (Provider::Pi, home.join(".pi/agent/sessions")),
                (
                    Provider::Claude,
                    env_path("CLAUDE_CONFIG_DIR", home.join(".claude")).join("projects"),
                ),
                (Provider::CopilotCli, home.join(".copilot/session-state")),
            ] {
                if root.exists() {
                    sources.push(Source {
                        name: format!("discovered-{provider}"),
                        provider,
                        root,
                    });
                }
            }
        }
        Ok(sources)
    }
}

fn open_lock(root: &Path, name: &str) -> Result<File> {
    fs::create_dir_all(root)?;
    Ok(OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(name))?)
}

/// Apply an open-file-description lock command to the whole file. OFD locks
/// belong to the open file, like `flock`, but `F_OFD_GETLK` can test for a
/// holder without acquiring the lock.
fn whole_file_lock(file: &File, command: libc::c_int) -> std::io::Result<libc::flock> {
    // SAFETY: flock is plain data; zero start and length cover the whole file.
    let mut lock: libc::flock = unsafe { std::mem::zeroed() };
    lock.l_type = libc::F_WRLCK as _;
    lock.l_whence = libc::SEEK_SET as _;
    loop {
        // SAFETY: the descriptor is open for the duration of the call.
        if unsafe { libc::fcntl(file.as_raw_fd(), command, &mut lock) } == 0 {
            return Ok(lock);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// Wait for an exclusive lock on `root/name`; dropping the file releases it.
pub fn lock(root: &Path, name: &str) -> Result<File> {
    let file = open_lock(root, name)?;
    whole_file_lock(&file, libc::F_OFD_SETLKW)?;
    Ok(file)
}

/// Take an exclusive lock on `root/name` unless another open file holds it.
pub fn try_lock(root: &Path, name: &str) -> Result<Option<File>> {
    let file = open_lock(root, name)?;
    match whole_file_lock(&file, libc::F_OFD_SETLK) {
        Ok(_) => Ok(Some(file)),
        Err(error) if matches!(error.raw_os_error(), Some(libc::EAGAIN | libc::EACCES)) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Report whether a lock on `root/name` is held, without taking it.
pub fn is_locked(root: &Path, name: &str) -> Result<bool> {
    let file = match File::open(root.join(name)) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let lock = whole_file_lock(&file, libc::F_OFD_GETLK)?;
    Ok(i32::from(lock.l_type) != libc::F_UNLCK)
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("output has no parent directory")?;
    fs::create_dir_all(parent)?;
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temp, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_discovery_uses_overrides_defaults_and_existing_paths() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let xdg = dir.path().join("xdg");
        let codex = dir.path().join("codex");
        let claude = dir.path().join("claude");
        let expected = [
            (Provider::Opencode, xdg.join("opencode/opencode.db")),
            (Provider::Codex, codex.join("sessions")),
            (Provider::Pi, home.join(".pi/agent/sessions")),
            (Provider::Claude, claude.join("projects")),
            (Provider::CopilotCli, home.join(".copilot/session-state")),
        ];
        for (provider, path) in &expected {
            if *provider == Provider::Opencode {
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, "").unwrap();
            } else {
                fs::create_dir_all(path).unwrap();
            }
        }
        let config = Config::load(&dir.path().join("data")).unwrap();
        let env = |name: &str, fallback: PathBuf| match name {
            "XDG_DATA_HOME" => xdg.clone(),
            "CODEX_HOME" => codex.clone(),
            "CLAUDE_CONFIG_DIR" => claude.clone(),
            _ => fallback,
        };
        let sources = config.sources_with(|| Ok(home.clone()), env).unwrap();
        assert_eq!(
            sources
                .iter()
                .map(|s| (s.provider, s.root.clone()))
                .collect::<Vec<_>>(),
            expected
        );
        let defaults = config
            .sources_with(|| Ok(home.clone()), |_, fallback| fallback)
            .unwrap();
        assert_eq!(
            defaults.iter().map(|s| s.provider).collect::<Vec<_>>(),
            [Provider::Pi, Provider::CopilotCli]
        );
        let configured = Config::edit(&config.root, |doc| {
            *doc = "[sources]
automatic = false
[[sources.roots]]
name = 'explicit'
provider = 'claude-code'
root = 'missing'
"
            .parse()
            .unwrap();
            Ok(())
        })
        .unwrap();
        let sources = configured
            .sources_with(
                || anyhow::bail!("HOME must not be needed"),
                |_, _| panic!("discovery disabled"),
            )
            .unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].provider, Provider::Claude);
        assert_eq!(sources[0].root, PathBuf::from("missing"));
    }

    #[test]
    fn lock_probe_does_not_take_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        assert!(!is_locked(root, "test.lock").unwrap());
        let held = try_lock(root, "test.lock").unwrap().expect("free lock");
        // Locks belong to open files, so a second open in this process conflicts.
        assert!(is_locked(root, "test.lock").unwrap());
        assert!(try_lock(root, "test.lock").unwrap().is_none());
        drop(held);
        assert!(!is_locked(root, "test.lock").unwrap());
        // Probing a free lock leaves it free for the next taker.
        let again = try_lock(root, "test.lock").unwrap();
        assert!(again.is_some());
    }
}
