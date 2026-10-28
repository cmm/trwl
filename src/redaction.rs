use std::{collections::HashSet, ops::Range};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::record::{NativeEventKind, Provider, Record};

/// Byte ranges are recorded by the compositor, including empty text parts.
/// Native node paths belong only to supported content containers, never tool data.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SourceSpans {
    pub ordinary: bool,
    spans: Vec<SourceSpan>,
    selections: Vec<Selection>,
    collections: Vec<BlockCollection>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub target: Target,
    pub opaque: Vec<Opaque>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Target {
    Node(String),
    Fields {
        object: String,
        members: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OpaqueKind {
    Data,
    Signature,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Opaque {
    Value { pointer: String, kind: OpaqueKind },
    NativeMap(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockCollection {
    pub pointer: String,
    pub arrays: Vec<BlockArray>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockArray {
    Blocks,
    OrderedBlocks,
}

impl BlockArray {
    pub fn name(self) -> &'static str {
        match self {
            Self::Blocks => "blocks",
            Self::OrderedBlocks => "orderedBlocks",
        }
    }
}

pub struct NativeProjection {
    pub value: Value,
    pub details: Vec<Value>,
}

fn member_path(object: &str, key: &str) -> String {
    format!("{object}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn covers(parent: &str, child: &str) -> bool {
    parent == child
        || child
            .strip_prefix(parent)
            .is_some_and(|tail| tail.starts_with('/'))
}

fn disjoint(paths: &[String]) -> Result<()> {
    let mut seen = HashSet::new();
    for path in paths {
        ensure!(seen.insert(path.as_str()), "overlapping content selections");
    }
    for path in paths {
        let mut parent = path.as_str();
        while let Some((ancestor, _)) = parent.rsplit_once('/') {
            ensure!(!seen.contains(ancestor), "overlapping content selections");
            parent = ancestor;
        }
    }
    Ok(())
}

impl Target {
    fn paths(&self) -> Vec<String> {
        match self {
            Self::Node(pointer) => vec![pointer.clone()],
            Self::Fields { object, members } => {
                members.iter().map(|key| member_path(object, key)).collect()
            }
        }
    }
}

impl Opaque {
    fn pointer(&self) -> &str {
        match self {
            Self::Value { pointer, .. } | Self::NativeMap(pointer) => pointer,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SourceSpan {
    range: Range<usize>,
    node: Option<usize>,
}

#[derive(Default)]
pub struct Body {
    text: String,
    source: SourceSpans,
}

impl Body {
    pub fn ordinary(&mut self) {
        self.source.ordinary = true;
    }

    pub fn select(&mut self, selection: Selection) -> usize {
        self.source.selections.push(selection);
        self.source.selections.len() - 1
    }

    pub fn collection(&mut self, collection: BlockCollection) {
        self.source.collections.push(collection);
    }

    pub fn push(&mut self, text: &str, node: Option<usize>) {
        if !self.source.spans.is_empty() {
            self.text.push('\n');
        }
        let start = self.text.len();
        self.text.push_str(text);
        self.source.spans.push(SourceSpan {
            range: start..self.text.len(),
            node,
        });
    }

    pub fn finish(self, record: &mut Record) -> Result<()> {
        record.text = self.text;
        self.source.validate(record)?;
        record.text = self.source.project_text(&record.text)?;
        self.source
            .remove_selected(&mut record.structured_content)?;
        // Keep event visibility and collection provenance, but no selected text
        // or native payload may reach the import spool or persistent index.
        record.source_spans = Some(SourceSpans {
            ordinary: self.source.ordinary,
            spans: vec![SourceSpan {
                range: 0..record.text.len(),
                node: None,
            }],
            selections: Vec::new(),
            collections: self.source.collections,
        });
        Ok(())
    }
}

impl SourceSpans {
    pub fn project_text(&self, text: &str) -> Result<String> {
        let mut projected = String::new();
        let mut count = 0;
        for span in &self.spans {
            if span.node.is_some() {
                continue;
            }
            let text = text
                .get(span.range.clone())
                .context("invalid content range")?;
            if count != 0 {
                projected.push('\n');
            }
            projected.push_str(text);
            count += 1;
        }
        Ok(projected)
    }

    fn validate(&self, record: &Record) -> Result<()> {
        let original = &record.structured_content;
        let mut paths = Vec::new();
        for selection in &self.selections {
            let owned = selection.target.paths();
            ensure!(!owned.is_empty(), "empty content selection");
            for pointer in &owned {
                original.pointer(pointer).context("content node missing")?;
            }
            let mut opaque_paths = Vec::new();
            for opaque in &selection.opaque {
                let pointer = opaque.pointer();
                original.pointer(pointer).context("content value missing")?;
                ensure!(
                    owned.iter().any(|path| covers(path, pointer)),
                    "content value outside selection"
                );
                if let Opaque::NativeMap(_) = opaque {
                    let Target::Node(owner) = &selection.target else {
                        anyhow::bail!("invalid content value owner");
                    };
                    ensure!(
                        record.provider == Provider::Opencode
                            && original.pointer(owner).context("content owner missing")?["type"]
                                == "reasoning"
                            && (pointer == member_path(owner, "metadata")
                                || pointer == member_path(owner, "native")),
                        "invalid content value owner"
                    );
                }
                opaque_paths.push(pointer.to_owned());
            }
            disjoint(&opaque_paths)?;
            paths.extend(owned);
        }
        disjoint(&paths)?;
        let mut collection_paths = Vec::new();
        for collection in &self.collections {
            ensure!(
                record.native_kind == Some(NativeEventKind::CopilotAssistantMessage)
                    && collection.pointer == "/reasoningBlocks",
                "invalid content container owner"
            );
            let envelope = original
                .pointer(&collection.pointer)
                .and_then(Value::as_object)
                .context("content container missing")?;
            let mut arrays = HashSet::new();
            for array in &collection.arrays {
                ensure!(arrays.insert(*array), "overlapping content containers");
                envelope
                    .get(array.name())
                    .and_then(Value::as_array)
                    .context("content array missing")?;
            }
            ensure!(
                !paths.iter().any(|path| covers(path, &collection.pointer)),
                "overlapping content containers"
            );
            collection_paths.push(collection.pointer.clone());
        }
        disjoint(&collection_paths)?;
        let mut end = 0;
        for span in &self.spans {
            ensure!(
                span.range.start >= end
                    && span.node.is_none_or(|node| node < self.selections.len()),
                "invalid content span"
            );
            record
                .text
                .get(span.range.clone())
                .context("invalid content range")?;
            end = span.range.end;
        }
        Ok(())
    }

    pub fn project_native(&self, record: &Record) -> Result<NativeProjection> {
        self.validate(record)?;
        let mut value = record.structured_content.clone();
        self.remove_selected(&mut value)?;
        let mut details = Vec::new();
        // Child nodes are removed before their native envelope is relocated.
        for collection in &self.collections {
            let mut envelope = value
                .pointer(&collection.pointer)
                .and_then(Value::as_object)
                .context("content container missing")?
                .clone();
            let non_selected = envelope.iter().any(|(key, value)| {
                if collection.arrays.iter().any(|array| array.name() == key) {
                    value.as_array().is_some_and(|array| !array.is_empty())
                } else {
                    key != "provider"
                }
            });
            remove_node(&mut value, &collection.pointer)?;
            if non_selected {
                envelope.insert("type".into(), "archive".into());
                details.push(Value::Object(envelope));
            }
        }
        Ok(NativeProjection { value, details })
    }

    fn remove_selected(&self, value: &mut Value) -> Result<()> {
        // Object groups cannot shift array positions; project them first.
        for selection in &self.selections {
            if let Target::Fields { object, members } = &selection.target {
                let object = value
                    .pointer_mut(object)
                    .and_then(Value::as_object_mut)
                    .context("content object missing")?;
                for key in members {
                    object.remove(key).context("content member missing")?;
                }
            }
        }
        // Numeric path components are ordered numerically within each array.
        // Object members are removed by exact key after decoding the pointer.
        let mut nodes: Vec<_> = self
            .selections
            .iter()
            .filter_map(|selection| match &selection.target {
                Target::Node(pointer) => Some(pointer),
                Target::Fields { .. } => None,
            })
            .collect();
        nodes.sort_by(|a, b| {
            let (ap, ak) = a.rsplit_once('/').unwrap_or(("", ""));
            let (bp, bk) = b.rsplit_once('/').unwrap_or(("", ""));
            ap.cmp(bp)
                .then_with(|| match (ak.parse::<usize>(), bk.parse::<usize>()) {
                    (Ok(a), Ok(b)) => a.cmp(&b),
                    _ => ak.cmp(bk),
                })
        });
        for pointer in nodes.into_iter().rev() {
            remove_node(value, pointer)?;
        }
        Ok(())
    }
}

fn remove_node(value: &mut Value, pointer: &str) -> Result<()> {
    if pointer.is_empty() {
        *value = Value::Null;
        return Ok(());
    }
    let (parent, member) = pointer.rsplit_once('/').context("invalid content path")?;
    let parent = value
        .pointer_mut(parent)
        .context("content container missing")?;
    match parent {
        Value::Array(array) => {
            let position: usize = member.parse().context("invalid content position")?;
            ensure!(position < array.len(), "content position missing");
            array.remove(position);
        }
        Value::Object(object) => {
            let key = member.replace("~1", "/").replace("~0", "~");
            object.remove(&key).context("content member missing")?;
        }
        _ => anyhow::bail!("invalid content container"),
    }
    Ok(())
}

pub fn neutral_metadata(provider: Provider, kind: Option<NativeEventKind>, value: &mut Value) {
    if provider == Provider::Pi
        && let Some(envelope) = value.as_object_mut()
    {
        envelope.remove("thinkingLevel");
        envelope.remove("providerThinkingLevel");
    }
    if kind == Some(NativeEventKind::CopilotAssistantMessage)
        && let Some(envelope) = value.as_object_mut()
    {
        envelope.remove("reasoningWireField");
        if envelope
            .get("phase")
            .is_some_and(|phase| phase == "thinking")
        {
            envelope.insert("phase".into(), "detail".into());
        }
    }
    // These are provider-owned token counters, not keys in content or tool input.
    let paths: &[(&str, &str)] = match provider {
        Provider::Opencode
            if matches!(
                value["type"].as_str(),
                Some("assistant" | "step-finish" | "compaction")
            ) =>
        {
            &[("/tokens", "reasoning")]
        }
        Provider::Codex => &[("/usage/output_tokens_details", "reasoning_tokens")],
        Provider::Pi => &[("/usage", "reasoning")],
        _ => &[],
    };
    for (path, key) in paths {
        if let Some(map) = value.pointer_mut(path).and_then(Value::as_object_mut)
            && let Some(tokens) = map.remove(*key)
        {
            map.insert("detail_tokens".into(), tokens);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::EventType;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn member_and_group_projection_preserves_siblings_and_rejects_bad_descriptors() {
        let mut record = Record::new(
            Provider::Codex,
            "session",
            "event",
            EventType::Message,
            Path::new("input"),
        );
        record.structured_content = json!({
            "content":"apples", "details":{"native":"original"},
            "member/key":{"text":"apples", "data":[false,0,null]},
            "items":[{"text":"apples"}, {"sidecar":"apples", "opaque":{"a":null}, "details":"original sibling"}]
        });
        let mut body = Body::default();
        body.ordinary();
        body.push("apples", None);
        let member = body.select(Selection {
            target: Target::Node("/member~1key".into()),
            opaque: vec![Opaque::Value {
                pointer: "/member~1key/data".into(),
                kind: OpaqueKind::Data,
            }],
        });
        body.push("apples", Some(member));
        let removed = body.select(Selection {
            target: Target::Node("/items/0".into()),
            opaque: Vec::new(),
        });
        body.push("apples", Some(removed));
        let group = body.select(Selection {
            target: Target::Fields {
                object: "/items/1".into(),
                members: vec!["sidecar".into(), "opaque".into()],
            },
            opaque: vec![Opaque::Value {
                pointer: "/items/1/opaque".into(),
                kind: OpaqueKind::Data,
            }],
        });
        body.push("apples", Some(group));
        let mut original = record.clone();
        original.text = body.text.clone();
        original.source_spans = Some(body.source.clone());
        body.finish(&mut record).unwrap();
        // The persisted record and projections of pre-redaction records must
        // preserve identical siblings, including escaped keys and array order.
        for projected in [
            record.clone().project().unwrap(),
            original.clone().project().unwrap(),
        ] {
            assert_eq!(projected.structured_content["content"], "apples");
            assert_eq!(
                projected.structured_content["details"],
                json!({"native":"original"})
            );
            assert_eq!(
                projected.structured_content["items"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap(),
                &json!({"details":"original sibling"})
            );
            assert_eq!(projected.text, "apples");
            assert!(projected.details.is_empty());
            assert!(projected.structured_content.get("member/key").is_none());
        }
        assert_eq!(record.text, "apples");
        assert!(record.structured_content.get("member/key").is_none());
        let mut missing = original.clone();
        missing.source_spans.as_mut().unwrap().selections[0].target =
            Target::Node("/missing".into());
        assert!(missing.project().is_err());
        let spans = original.source_spans.as_mut().unwrap();
        spans.selections.push(spans.selections[0].clone());
        assert!(original.project().is_err());
    }
}
