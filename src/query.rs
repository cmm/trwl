//! Search, `show`, `locate` and `list` over a pinned index commit.
//!
//! Search applies filters, then groups results by session unless event results
//! are requested; grouping lists the best-ranked session of each session tree
//! before the other sessions of that tree. This can demote a child below a
//! lower-scoring root from another tree. Score ties use newest timestamp,
//! then event ID; this also orders each list before hybrid rank fusion.
//! Grouped hybrid retrieval combines each retriever's best event per session.
//! Session exclusion compares session identities and does not follow parent
//! relationships.

use crate::{
    cli::{Backend, Entity, Filters, ListKind, Search},
    config::Config,
    record::{Record, timestamp},
    report::{
        EventsPage, Location, Report, Retrieval, SearchHit, SearchReport, SelectedEvents,
        SessionEvents, SessionsPage, State,
    },
    semantic,
    store::{self, ReadIndex},
    workspace,
};
use anyhow::{Context, Result, bail, ensure};
#[cfg(test)]
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

#[derive(Clone, Copy)]
enum SnippetSource {
    Lexical,
    Semantic,
}
type RankedRecord = (f32, Record, SnippetSource);

/// Normalize inclusive `--since`/`--until` arguments to RFC 3339 bounds.
/// The returned upper bound is exclusive; a date-only upper bound is midnight
/// after the named UTC day.
pub(crate) fn bound(value: &str, until: bool) -> Result<String> {
    match chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        Ok(date) => {
            let date = if until {
                date.succ_opt().context("date outside supported range")?
            } else {
                date
            };
            timestamp(&format!("{date}T00:00:00Z"))
        }
        // An explicit timestamp is inclusive: compare against the next millisecond.
        Err(_) if until => {
            let time = chrono::DateTime::parse_from_rfc3339(value)
                .with_context(|| format!("invalid timestamp {value:?}"))?;
            timestamp(&(time + chrono::Duration::milliseconds(1)).to_rfc3339())
        }
        Err(_) => timestamp(value),
    }
}

pub fn search(config: &Config, args: &Search) -> Result<SearchReport> {
    let index = ReadIndex::open(&config.root)?;
    search_index(config, args, index.as_ref())
}

pub(crate) fn search_index(
    config: &Config,
    args: &Search,
    index: Option<&ReadIndex>,
) -> Result<SearchReport> {
    let Some(index) = index else {
        if args.backend == Backend::Semantic {
            bail!("semantic index is unavailable: no indexed history");
        }
        return Ok(SearchReport {
            query: args.query.clone(),
            results: Vec::new(),
            retrieval: Retrieval {
                effective_mode: Backend::Lexical,
                semantic_status: State::Unavailable,
                reason: None,
            },
        });
    };
    let terms: Vec<_> = std::iter::once(args.query.clone())
        .chain(args.term.clone())
        .collect();
    let workspace =
        workspace::Selection::resolve(&args.filters.workspace, || index.workspace_roots())?;
    let selection = index.selection(&args.filters, true, &workspace)?;
    let mut mode = Backend::Lexical;
    let mut status = State::Disabled;
    let mut reason = None;
    let mut results = Vec::new();
    if args.backend != Backend::Lexical {
        match semantic::search(
            config,
            index,
            &terms,
            args.refresh == crate::cli::Refresh::Wait,
        ) {
            Ok(semantic) => {
                status = if semantic.complete {
                    State::Ready
                } else {
                    State::Partial
                };
                mode = if args.backend == Backend::Semantic {
                    Backend::Semantic
                } else {
                    Backend::Hybrid
                };
                let selected = index.selected_ids(&*selection)?;
                results = semantic
                    .hits
                    .into_iter()
                    .filter(|(_, r)| selected.contains(&r.event_id))
                    .map(|(score, record)| (score, record, SnippetSource::Semantic))
                    .collect();
                if args.backend == Backend::Hybrid && args.events {
                    results.sort_by(rank_order);
                    results.truncate(args.limit.saturating_mul(4).max(100));
                }
            }
            Err(error) if args.backend == Backend::Hybrid => {
                status = State::Unavailable;
                reason = Some(format!("{error:#}"));
            }
            Err(error) => return Err(error.context("semantic search unavailable")),
        }
    }
    let mut lexical = None;
    if args.backend != Backend::Semantic {
        let mut found = index.lexical(
            &terms,
            if mode == Backend::Hybrid && args.events {
                store::LexicalMode::Events(args.limit.saturating_mul(4).max(100))
            } else if args.events {
                store::LexicalMode::Events(args.limit)
            } else {
                store::LexicalMode::Sessions
            },
            &*selection,
        )?;
        let rows = std::mem::take(&mut found.rows)
            .into_iter()
            .map(|(score, record)| (score, record, SnippetSource::Lexical))
            .collect();
        results = if mode == Backend::Hybrid {
            fuse(rows, results, args.events)
        } else {
            rows
        };
        lexical = Some(found);
    }
    results.sort_by(rank_order);
    let mut seen = HashSet::new();
    results.retain(|(_, r, _)| seen.insert(r.event_id.clone()));
    if !args.events {
        let mut sessions = HashSet::new();
        results.retain(|(_, r, _)| sessions.insert(r.session_id.clone()));
        let parents: HashMap<_, _> = results
            .iter()
            .filter_map(|(_, r, _)| {
                r.parent_session_id
                    .as_ref()
                    .map(|p| (r.session_id.clone(), p.clone()))
            })
            .collect();
        let mut roots = HashSet::new();
        let (mut first, mut rest) = (Vec::new(), Vec::new());
        for row in results {
            let mut root = row.1.session_id.clone();
            let mut visited = HashSet::new();
            while visited.insert(root.clone()) {
                if let Some(parent) = parents.get(&root) {
                    root = parent.clone();
                } else {
                    break;
                }
            }
            if roots.insert(root) {
                first.push(row);
            } else {
                rest.push(row);
            }
        }
        first.extend(rest);
        results = first;
    }
    let results = results
        .into_iter()
        .take(args.limit)
        .enumerate()
        .map(|(rank, (score, r, source))| {
            let snippet = match source {
                SnippetSource::Lexical => lexical
                    .as_ref()
                    .context("lexical snippets missing")?
                    .snippet(index, &r.event_id)?,
                SnippetSource::Semantic => r
                    .text
                    .chars()
                    .take(crate::record::SEARCH_SNIPPET_CHARS)
                    .collect(),
            };
            Ok(SearchHit {
                trwl_event_id: r.event_id,
                trwl_session_id: r.session_id,
                provider: r.provider,
                provider_session_id: r.provider_session_id,
                timestamp: r.occurred_at,
                workspace_root: r.workspace_root,
                event_type: r.event_type,
                snippet,
                rank: rank + 1,
                score,
            })
        })
        .collect::<Result<_>>()?;
    Ok(SearchReport {
        query: args.query.clone(),
        results,
        retrieval: Retrieval {
            effective_mode: mode,
            semantic_status: status,
            reason,
        },
    })
}

fn rank_order((sa, a, _): &RankedRecord, (sb, b, _): &RankedRecord) -> std::cmp::Ordering {
    crate::record::relevance_order(
        (*sa, &a.occurred_at, &a.event_id),
        (*sb, &b.occurred_at, &b.event_id),
    )
}

fn fuse(mut a: Vec<RankedRecord>, mut b: Vec<RankedRecord>, events: bool) -> Vec<RankedRecord> {
    // Smooth reciprocal ranks so the first few results do not dominate fusion.
    // Ranks here are zero-based: the first result contributes 1 / 60.
    const RANK_OFFSET: f32 = 60.0;
    let mut scores = HashMap::<String, (f32, usize, Record, SnippetSource)>::new();
    for list in [&mut a, &mut b] {
        list.sort_by(rank_order);
        let mut seen = HashSet::new();
        list.retain(|(_, record, _)| {
            seen.insert(if events {
                record.event_id.clone()
            } else {
                record.session_id.clone()
            })
        });
        for (rank, (_, record, source)) in list.drain(..).enumerate() {
            let key = if events {
                &record.event_id
            } else {
                &record.session_id
            };
            let contribution = 1.0 / (RANK_OFFSET + rank as f32);
            match scores.entry(key.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((contribution, rank, record, source));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let entry = entry.get_mut();
                    entry.0 += contribution;
                    // Snippets use the stronger individual rank; lexical wins ties.
                    if rank < entry.1 {
                        entry.1 = rank;
                        entry.2 = record;
                        entry.3 = source;
                    }
                }
            }
        }
    }
    scores
        .into_values()
        .map(|(score, _, record, source)| (score, record, source))
        .collect()
}

fn chronological(records: &mut [Record]) {
    records.sort_by(|a, b| {
        crate::record::chronological_order(
            (&a.occurred_at, a.sequence, &a.event_id),
            (&b.occurred_at, b.sequence, &b.event_id),
        )
    });
}

pub fn resolve(root: &Path, kind: Entity, prefix: &str) -> Result<Record> {
    let index = ReadIndex::open(root)?.context("no history index")?;
    resolve_in(&index, kind, prefix)
}

fn resolve_in(index: &ReadIndex, kind: Entity, prefix: &str) -> Result<Record> {
    ensure!(!prefix.is_empty(), "empty record reference");
    let mut matching = match kind {
        Entity::Event => index.event_reference(prefix)?,
        Entity::Session => index.session_reference(prefix)?,
    };
    chronological(&mut matching);
    let first = matching.first().context("record reference not found")?;
    ensure!(
        matching.iter().all(|r| match kind {
            Entity::Event => r.event_id == first.event_id,
            Entity::Session => r.session_id == first.session_id,
        }),
        "ambiguous record reference {prefix:?}"
    );
    Ok(first.clone())
}

pub fn show(root: &Path, kind: Entity, id: &str, window: usize) -> Result<Report> {
    let index = ReadIndex::open(root)?.context("no history index")?;
    let record = resolve_in(&index, kind, id)?;
    if matches!(kind, Entity::Event) && window == 0 {
        return Ok(Report::ShowEvent(SelectedEvents {
            events: vec![record.clone()],
            event: record,
        }));
    }
    if matches!(kind, Entity::Event) {
        let events = index.window(&record, window)?;
        Ok(Report::ShowEvent(SelectedEvents {
            event: record,
            events,
        }))
    } else {
        let mut events = index.by_session(&record.session_id)?;
        chronological(&mut events);
        Ok(Report::ShowSession(SessionEvents {
            session: locate(&record),
            events,
        }))
    }
}

pub fn locate(record: &Record) -> Location {
    Location {
        trwl_event_id: record.event_id.clone(),
        trwl_session_id: record.session_id.clone(),
        provider: record.provider,
        provider_session_id: record.provider_session_id.clone(),
        provider_event_id: record.provider_event_id.clone(),
        parent_trwl_session_id: record.parent_session_id.clone(),
        source: record.source.clone(),
        workspace: record.workspace.clone(),
        workspace_root: record.workspace_root.clone(),
    }
}

pub fn list(
    root: &Path,
    kind: ListKind,
    filters: &Filters,
    limit: usize,
    cursor: Option<&str>,
) -> Result<Report> {
    ensure!(limit > 0, "list limit must be positive");
    let (records, next) = match ReadIndex::open(root)? {
        Some(index) => {
            let workspace =
                workspace::Selection::resolve(&filters.workspace, || index.workspace_roots())?;
            index.page(
                filters,
                matches!(kind, ListKind::Sessions),
                limit,
                cursor,
                &workspace,
            )?
        }
        None => {
            ensure!(cursor.is_none(), "cursor no longer matches this selection");
            (Vec::new(), None)
        }
    };
    match kind {
        ListKind::Events => Ok(Report::Events(EventsPage {
            events: records,
            next_cursor: next,
        })),
        ListKind::Sessions => Ok(Report::Sessions(SessionsPage {
            sessions: records.iter().map(locate).collect(),
            next_cursor: next,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        record::{EventType, Provider},
        test_support::{codex_message, import},
    };

    #[test]
    fn fusion_ties_do_not_depend_on_input_order() {
        let mut old = Record::new(
            Provider::Codex,
            "session",
            "old",
            EventType::Message,
            Path::new("input"),
        );
        old.occurred_at = "2026-01-01T00:00:00.000Z".into();
        let mut recent = Record::new(
            Provider::Codex,
            "session",
            "recent",
            EventType::Message,
            Path::new("input"),
        );
        recent.occurred_at = "2026-01-02T00:00:00.000Z".into();
        let scores = |rows: Vec<(f32, Record)>| {
            let rows = rows
                .into_iter()
                .map(|(score, record)| (score, record, SnippetSource::Lexical))
                .collect();
            fuse(rows, Vec::new(), true)
                .into_iter()
                .map(|(score, record, _)| (record.event_id, score))
                .collect::<HashMap<_, _>>()
        };
        let forward = scores(vec![(1.0, old.clone()), (1.0, recent.clone())]);
        let reverse = scores(vec![(1.0, recent.clone()), (1.0, old.clone())]);
        assert_eq!(forward, reverse);
        assert!(forward[&recent.event_id] > forward[&old.event_id]);
    }

    #[test]
    fn fusion_combines_different_events_in_one_session() {
        let record = |session, event| {
            Record::new(
                Provider::Codex,
                session,
                event,
                EventType::Message,
                Path::new("input"),
            )
        };
        let lexical = record("shared", "lexical");
        let semantic = record("shared", "semantic");
        let other = record("other", "other");
        let mut fused = fuse(
            vec![
                (2.0, other, SnippetSource::Lexical),
                (1.0, lexical, SnippetSource::Lexical),
            ],
            vec![(1.0, semantic.clone(), SnippetSource::Semantic)],
            false,
        );
        fused.sort_by(rank_order);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].1.event_id, semantic.event_id);
        assert!((fused[0].0 - (1.0 / 60.0 + 1.0 / 61.0)).abs() < 1e-7);
    }

    #[test]
    fn fusion_event_mode_separates_anchors_and_session_ties_prefer_lexical() {
        let lexical = Record::new(
            Provider::Codex,
            "shared",
            "lexical",
            EventType::Message,
            Path::new("input"),
        );
        let semantic = Record::new(
            Provider::Codex,
            "shared",
            "semantic",
            EventType::Message,
            Path::new("input"),
        );
        let a = vec![(1.0, lexical.clone(), SnippetSource::Lexical)];
        let b = vec![(1.0, semantic, SnippetSource::Semantic)];
        let events = fuse(a.clone(), b.clone(), true);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|(score, _, _)| *score == 1.0 / 60.0));
        let sessions = fuse(a, b, false);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].1.event_id, lexical.event_id);
        assert_eq!(sessions[0].0, 2.0 / 60.0);
    }

    #[test]
    fn date_bounds_include_the_named_day() {
        let filters = |since: &str, until: &str| Filters {
            since: Some(since.into()),
            until: Some(until.into()),
            content_scope: crate::record::ContentScope::All,
            ..Default::default()
        };
        let day = filters("2026-01-01", "2026-01-01");
        let (_dir, index) = fixture_index(&[
            "2025-12-31T23:59:59.999Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T23:59:59.999Z",
            "2026-01-02T00:00:00Z",
            "2026-01-01T12:00:00Z",
            "2026-01-01T12:00:00.001Z",
            "2026-01-01T11:59:59.999Z",
            "",
        ]);
        let selected = index
            .selected_ids(
                &*index
                    .selection(&day, false, &workspace::Selection::All)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(selected.len(), 5);
        // Explicit timestamps are inclusive on both ends.
        let exact = filters("2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z");
        let selected = index
            .selected_ids(
                &*index
                    .selection(&exact, false, &workspace::Selection::All)
                    .unwrap(),
            )
            .unwrap();
        let all = index.all().unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(
            all.iter()
                .find(|r| selected.contains(&r.event_id))
                .unwrap()
                .occurred_at,
            "2026-01-01T12:00:00.000Z"
        );
    }

    #[test]
    fn session_filters_match_whole_identities() {
        let (_dir, index) = fixture_index(&["2026-01-01T00:00:00Z"]);
        let record = index.all().unwrap().remove(0);
        let check = |session: Option<&str>, exclude: &[&str]| {
            let filters = Filters {
                session: session.map(Into::into),
                exclude_session: exclude.iter().map(|s| s.to_string()).collect(),
                content_scope: crate::record::ContentScope::All,
                ..Default::default()
            };
            !index
                .selected_ids(
                    &*index
                        .selection(&filters, false, &workspace::Selection::All)
                        .unwrap(),
                )
                .unwrap()
                .is_empty()
        };
        let prefix = &record.session_id[..8];
        assert!(check(Some(&record.session_id), &[]));
        assert!(check(Some("native"), &[]));
        assert!(!check(Some(prefix), &[]));
        assert!(!check(Some("nat"), &[]));
        assert!(check(None, &[prefix, "nat"]));
        assert!(!check(None, &[&record.session_id]));
        assert!(!check(None, &["native"]));
    }

    fn fixture_index(times: &[&str]) -> (tempfile::TempDir, ReadIndex) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.jsonl");
        let mut rows = format!(
            "{}\n",
            json!({"type":"session_meta", "payload":{"id":"native"}})
        );
        for (n, time) in times.iter().enumerate() {
            let mut row = codex_message("user", "needle");
            row["payload"]["id"] = json!(n.to_string());
            if !time.is_empty() {
                row["timestamp"] = json!(time);
            }
            rows.push_str(&format!("{row}\n"));
        }
        std::fs::write(&path, rows).unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
        let index = import(&config, Provider::Codex, &path);
        (dir, index)
    }

    #[test]
    fn grouped_sessions_demote_the_second_family_match_even_with_a_higher_rank() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.jsonl");
        let mut rows = String::new();
        for (session, parent, time) in [
            ("root", None, "2026-01-03T00:00:00Z"),
            ("child", Some("root"), "2026-01-02T00:00:00Z"),
            ("other", None, "2026-01-01T00:00:00Z"),
        ] {
            let mut header = json!({"type":"session_meta","payload":{"id":session}});
            if let Some(parent) = parent {
                header["payload"]["source"] =
                    json!({"subagent":{"thread_spawn":{"parent_thread_id":parent}}});
            }
            let mut message = codex_message("user", "needle");
            message["timestamp"] = json!(time);
            message["payload"]["id"] = json!(session);
            rows.push_str(&format!("{header}\n{message}\n"));
        }
        std::fs::write(&path, rows).unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
        import(&config, Provider::Codex, &path);
        let mut args = Search {
            query: "needle".into(),
            term: vec![],
            backend: Backend::Lexical,
            refresh: crate::cli::Refresh::Off,
            events: false,
            limit: 20,
            filters: Filters {
                include_current_session: true,
                workspace: vec![workspace::Pattern::All],
                ..Default::default()
            },
        };
        let grouped = search(&config, &args).unwrap();
        assert_eq!(
            grouped
                .results
                .iter()
                .map(|r| r.provider_session_id.as_str())
                .collect::<Vec<_>>(),
            ["root", "other", "child"]
        );
        args.events = true;
        let events = search(&config, &args).unwrap();
        assert_eq!(
            events
                .results
                .iter()
                .map(|r| r.provider_session_id.as_str())
                .collect::<Vec<_>>(),
            ["root", "child", "other"]
        );
    }

    #[test]
    fn cursor_pages_keep_ties_and_decode_only_the_page() {
        let (_dir, index) = fixture_index(&["2026-01-01T00:00:00Z"; 5]);
        let filters = Filters::default();
        let mut cursor = None;
        let mut records = Vec::new();
        loop {
            index.record_decodes.set(0);
            let (page, next) = index
                .page(
                    &filters,
                    false,
                    2,
                    cursor.as_deref(),
                    &workspace::Selection::All,
                )
                .unwrap();
            assert_eq!(index.record_decodes.get(), page.len());
            assert!(page.len() <= 2);
            records.extend(page);
            cursor = next;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(
            records
                .iter()
                .map(|r| r.provider_event_id.as_str())
                .collect::<Vec<_>>(),
            ["0", "1", "2", "3", "4"]
        );
        assert_eq!(
            records
                .iter()
                .map(|r| &r.event_id)
                .collect::<HashSet<_>>()
                .len(),
            5
        );
        let (sessions, next) = index
            .page(&filters, true, 1, None, &workspace::Selection::All)
            .unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].event_id, records[0].event_id);
        assert!(next.is_none());
        let changed = Filters {
            provider: vec![crate::record::Provider::Pi],
            ..Default::default()
        };
        assert!(
            index
                .page(
                    &changed,
                    false,
                    2,
                    Some(&records[0].event_id),
                    &workspace::Selection::All
                )
                .is_err()
        );
        assert!(
            index
                .page(
                    &filters,
                    false,
                    2,
                    Some("missing"),
                    &workspace::Selection::All
                )
                .is_err()
        );
    }

    #[test]
    fn references_reject_event_prefix_and_native_session_ambiguity() {
        let (dir, index) = fixture_index(&["2026-01-01T00:00:00Z"; 100]);
        let records = index.all().unwrap();
        let mut prefixes: HashMap<&str, usize> = HashMap::new();
        for r in &records {
            *prefixes.entry(&r.event_id[..1]).or_default() += 1;
        }
        let prefix = prefixes.iter().find(|(_, count)| **count > 1).unwrap().0;
        assert!(
            resolve_in(&index, Entity::Event, prefix)
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        assert_eq!(
            resolve_in(&index, Entity::Event, &records[0].event_id)
                .unwrap()
                .event_id,
            records[0].event_id
        );
        let path = dir.path().join("pi.jsonl");
        let header = json!({"type": "session", "version": 3, "id": "native"});
        let message = json!({
            "type": "message",
            "id": "pi-event",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "other provider"}],
            },
        });
        std::fs::write(&path, format!("{header}\n{message}\n")).unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
        let next = import(&config, Provider::Pi, &path);
        assert!(
            resolve_in(&next, Entity::Session, "native")
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        assert_eq!(
            resolve_in(&next, Entity::Session, &records[0].session_id)
                .unwrap()
                .provider,
            crate::record::Provider::Codex
        );
    }
}
