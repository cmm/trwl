use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Debug, Serialize)]
pub struct Directories {
    pub config: PathBuf,
    pub data: PathBuf,
    pub cache: PathBuf,
    pub state: PathBuf,
    pub runtime: PathBuf,
}

fn absolute_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
}

fn home_directory(variable: &str, default: &str) -> Result<PathBuf> {
    let base = match absolute_env(variable) {
        Some(path) => path,
        None => absolute_env("HOME")
            .context("HOME must be absolute when an XDG home directory is unavailable")?
            .join(default),
    };
    Ok(base.join("trwl"))
}

pub fn private_directory(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("opening private runtime directory {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o777 == 0o700,
        "runtime directory must be owned by the current user with mode 0700: {}",
        path.display()
    );
    Ok(file)
}

fn create_private(path: &Path) -> Result<()> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    private_directory(path)?;
    Ok(())
}

pub fn runtime_directory() -> Result<PathBuf> {
    let runtime = match absolute_env("XDG_RUNTIME_DIR") {
        Some(path)
            if match path.symlink_metadata() {
                Ok(_) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => return Err(error.into()),
            } =>
        {
            private_directory(&path)?;
            path
        }
        _ => {
            let temporary = std::env::temp_dir();
            ensure!(
                temporary.is_absolute(),
                "temporary directory must be absolute"
            );
            let path = temporary.join(format!("trwl-runtime-{}", unsafe { libc::geteuid() }));
            create_private(&path)?;
            eprintln!(
                "trwl: XDG_RUNTIME_DIR is unavailable; using {}",
                path.display()
            );
            path
        }
    };
    let path = runtime.join("trwl");
    create_private(&path)?;
    Ok(path)
}

impl Directories {
    pub fn resolve() -> Result<Self> {
        Ok(Self {
            config: home_directory("XDG_CONFIG_HOME", ".config")?,
            data: home_directory("XDG_DATA_HOME", ".local/share")?,
            cache: home_directory("XDG_CACHE_HOME", ".cache")?,
            state: home_directory("XDG_STATE_HOME", ".local/state")?,
            runtime: runtime_directory()?,
        })
    }

    pub fn index_runtime(&self) -> Result<PathBuf> {
        fs::create_dir_all(&self.data)?;
        let metadata = fs::metadata(&self.data)?;
        // Inode identity also groups aliases of the same persistent directory.
        let identity = crate::record::digest(&[&format!("{}:{}", metadata.dev(), metadata.ino())]);
        let path = self.runtime.join(identity);
        create_private(&path)?;
        Ok(path)
    }

    pub fn child_environment(&self, command: &mut Command) {
        for (name, path) in [
            ("XDG_CONFIG_HOME", &self.config),
            ("XDG_DATA_HOME", &self.data),
            ("XDG_CACHE_HOME", &self.cache),
            ("XDG_STATE_HOME", &self.state),
            ("XDG_RUNTIME_DIR", &self.runtime),
        ] {
            command.env(
                name,
                path.parent().expect("application directory has a parent"),
            );
        }
    }

    pub fn inference_environment(&self, command: &mut Command) -> Result<()> {
        self.child_environment(command);
        for (name, directory) in [
            ("CUDA_CACHE_PATH", "cuda"),
            ("NEO_CACHE_DIR", "neo"),
            ("__GL_SHADER_DISK_CACHE_PATH", "gl"),
            ("POCL_CACHE_DIR", "pocl"),
        ] {
            let path = self.cache.join("inference").join(directory);
            fs::create_dir_all(&path)?;
            command.env(name, path);
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn isolated(root: &Path) -> Self {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(root.join("runtime/trwl"))
            .unwrap();
        Self {
            config: root.join("config/trwl"),
            data: root.join("data/trwl"),
            cache: root.join("cache/trwl"),
            state: root.join("state/trwl"),
            runtime: root.join("runtime/trwl"),
        }
    }
}
