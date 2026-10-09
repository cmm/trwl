//! Workspace identity comes from Git worktree discovery, or the provider's
//! directory outside Git. Roots are resolved during import and kept in the
//! manifest, keyed per session and directory, so later filesystem changes do
//! not reassign existing history; each newly imported session resolves its
//! directory once.

use anyhow::{Context, Result, ensure};
use globset::{Glob, GlobBuilder, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
    str::FromStr,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Root(PathBuf);

impl Root {
    /// Builds a root from a UTF-8 path string, such as an index term.
    pub(crate) fn from_text(root: &str) -> Self {
        Self(PathBuf::from(root))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn text(&self) -> &str {
        // Discovery rejects non-UTF-8 roots, matching provider path strings and
        // the index's string terms.
        self.0.to_str().expect("workspace root is UTF-8")
    }
}

enum Discovery {
    Git(Root),
    Directory(Root),
}

fn containing(root: Root, known: &HashSet<Root>) -> Root {
    known
        .iter()
        .filter(|candidate| root.path().starts_with(candidate.path()))
        .max_by_key(|candidate| candidate.path().components().count())
        .cloned()
        .unwrap_or(root)
}

fn discover(path: &Path) -> Result<Discovery> {
    ensure!(path.is_absolute(), "workspace directory must be absolute");
    let directory = path
        .canonicalize()
        .with_context(|| format!("resolving workspace directory {}", path.display()))?;
    ensure!(directory.is_dir(), "workspace path is not a directory");
    let (root, git) = match gix_discover::upwards(&directory) {
        Ok((location, _trust)) => {
            let (_, worktree) = location.into_repository_and_work_tree_directories();
            let worktree = worktree.context("Git repository has no worktree")?;
            (worktree.canonicalize()?, true)
        }
        Err(error)
            if error
                .downcast_any_ref::<gix_discover::upwards::Error>()
                .is_some_and(|error| {
                    matches!(
                        error,
                        gix_discover::upwards::Error::NoGitRepository { .. }
                            | gix_discover::upwards::Error::NoGitRepositoryWithinCeiling { .. }
                            | gix_discover::upwards::Error::NoGitRepositoryWithinFs { .. }
                    )
                }) =>
        {
            (directory, false)
        }
        Err(error) => return Err(error.into()),
    };
    ensure!(root.to_str().is_some(), "workspace root is not UTF-8");
    Ok(if git {
        Discovery::Git(Root(root))
    } else {
        Discovery::Directory(Root(root))
    })
}

#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Roots {
    identities: BTreeMap<String, Option<Root>>,
    #[serde(skip)]
    discovered: HashMap<String, Option<Root>>,
}

impl Roots {
    pub fn resolve(&mut self, session: &str, directory: &str) -> Option<Root> {
        let discovered = &mut self.discovered;
        self.identities
            .entry(crate::record::digest(&[session, directory]))
            .or_insert_with(|| {
                discovered
                    .entry(directory.into())
                    .or_insert_with(|| match discover(Path::new(directory)).ok()? {
                        Discovery::Git(root) | Discovery::Directory(root) => Some(root),
                    })
                    .clone()
            })
            .clone()
    }
}

#[derive(Clone, Debug)]
pub enum Pattern {
    All,
    Literal(PathBuf),
    Glob(Glob),
}

impl FromStr for Pattern {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        ensure!(!value.is_empty(), "workspace pattern is empty");
        if value == "**" {
            return Ok(Self::All);
        }
        let glob = GlobBuilder::new(value)
            .literal_separator(true)
            .case_insensitive(true)
            .backslash_escape(true)
            .build()
            .with_context(|| format!("invalid workspace pattern {value:?}"))?;
        Ok(if value.contains(['*', '?', '[', ']', '{', '}', '\\']) {
            Self::Glob(glob)
        } else {
            Self::Literal(value.into())
        })
    }
}

/// A command's workspace restriction. `Patterns` holds the indexed roots
/// matching its literals and globs; a pattern set that matches nothing
/// restricts results to nothing.
pub enum Selection {
    All,
    Current(Root),
    Patterns(HashSet<Root>),
}

impl Selection {
    pub fn resolve(
        patterns: &[Pattern],
        known: impl FnOnce() -> Result<HashSet<Root>>,
    ) -> Result<Self> {
        if patterns.iter().any(|p| matches!(p, Pattern::All)) {
            return Ok(Self::All);
        }
        if patterns.is_empty() {
            let cwd = std::env::current_dir().context("detecting current workspace")?;
            let root = match discover(&cwd)
                .context("detecting current workspace; use --workspace '**' for all history")?
            {
                Discovery::Git(root) => root,
                Discovery::Directory(root) => containing(root, &known()?),
            };
            return Ok(Self::Current(root));
        }
        let roots = known()?;
        let mut selected = HashSet::new();
        let mut globs = GlobSetBuilder::new();
        for pattern in patterns {
            match pattern {
                Pattern::All => unreachable!(),
                Pattern::Glob(glob) => {
                    globs.add(glob.clone());
                }
                Pattern::Literal(path) => {
                    let path = if path.is_absolute() {
                        path.clone()
                    } else {
                        std::env::current_dir()?.join(path)
                    };
                    // An absent literal path can still select a canonical root
                    // stored before its checkout disappeared; comparison is
                    // case-insensitive, like glob matching.
                    selected.insert(match discover(&path) {
                        Ok(Discovery::Git(root)) => root,
                        Ok(Discovery::Directory(root)) => containing(root, &roots),
                        Err(_) if !path.exists() => {
                            let text = path.to_str().context("workspace pattern is not UTF-8")?;
                            let lower = text.to_lowercase();
                            roots
                                .iter()
                                .find(|root| root.text().to_lowercase() == lower)
                                .cloned()
                                .unwrap_or_else(|| Root::from_text(text))
                        }
                        Err(error) => return Err(error),
                    });
                }
            }
        }
        let globs = globs.build()?;
        selected.extend(
            roots
                .iter()
                .filter(|root| globs.is_match(root.path()))
                .cloned(),
        );
        Ok(Self::Patterns(selected))
    }
}
