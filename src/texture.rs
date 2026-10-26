use std::{collections::HashSet, ops::Range};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::record::{NativeEventKind, Provider, Record};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Default,
    Expanded,
    Source,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Availability {
    Ordinary,
    Expanded,
    Source,
}

impl Availability {
    pub fn indexed(self) -> u64 {
        match self {
            Self::Ordinary => 0,
            Self::Expanded => 1,
            Self::Source => 2,
        }
    }
}

impl Mode {
    pub fn from_flag(enabled: bool) -> Self {
        if !enabled {
            Self::Default
        } else if std::env::var_os("TRWL_TEXTURE").is_some_and(|v| v == "source") {
            Self::Source
        } else {
            Self::Expanded
        }
    }

    pub fn availability(self) -> Availability {
        match self {
            Self::Default => Availability::Ordinary,
            Self::Expanded => Availability::Expanded,
            Self::Source => Availability::Source,
        }
    }
}

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
        record.source_spans = Some(self.source);
        Ok(())
    }
}

pub struct TextView {
    pub text: String,
    edits: Vec<(usize, char)>,
}

impl TextView {
    pub fn render(&self, range: Range<usize>) -> Result<String> {
        let text = self
            .text
            .get(range.clone())
            .context("invalid excerpt range")?;
        let mut edits = self.edits.iter().peekable();
        while edits.peek().is_some_and(|(at, _)| *at < range.start) {
            edits.next();
        }
        Ok(text
            .char_indices()
            .map(|(offset, ch)| {
                if edits
                    .peek()
                    .is_some_and(|(at, _)| *at == range.start + offset)
                {
                    edits.next().unwrap().1
                } else {
                    ch
                }
            })
            .collect())
    }
}

pub fn view(record: &Record, mode: Mode) -> Result<TextView> {
    text_view(&record.text, record.source_spans.as_ref(), mode)
}

pub fn text_view(text: &str, source: Option<&SourceSpans>, mode: Mode) -> Result<TextView> {
    match source {
        Some(source) => source.view(text, mode),
        None => Ok(TextView {
            text: text.to_owned(),
            edits: Vec::new(),
        }),
    }
}

impl SourceSpans {
    pub fn availability(&self, text: &str) -> Result<Availability> {
        if self.ordinary {
            return Ok(Availability::Ordinary);
        }
        for span in &self.spans {
            if span.node.is_some()
                && !edits(
                    text.get(span.range.clone())
                        .context("invalid content range")?,
                )
                .is_empty()
            {
                return Ok(Availability::Expanded);
            }
        }
        Ok(Availability::Source)
    }

    fn view(&self, text: &str, mode: Mode) -> Result<TextView> {
        let mut view = TextView {
            text: String::new(),
            edits: Vec::new(),
        };
        let mut count = 0;
        for span in &self.spans {
            let text = text
                .get(span.range.clone())
                .context("invalid content range")?;
            let changes = if span.node.is_some() {
                match mode {
                    Mode::Default => continue,
                    Mode::Expanded => {
                        let changes = edits(text);
                        if changes.is_empty() {
                            continue;
                        }
                        changes
                    }
                    Mode::Source => Vec::new(),
                }
            } else {
                Vec::new()
            };
            if count != 0 {
                view.text.push('\n');
            }
            let start = view.text.len();
            view.edits
                .extend(changes.into_iter().map(|(at, ch)| (start + at, ch)));
            view.text.push_str(text);
            count += 1;
        }
        Ok(view)
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

    pub fn project_native(&self, record: &Record, mode: Mode) -> Result<NativeProjection> {
        self.validate(record)?;
        let mut replacements = Vec::new();
        // All values come from the immutable record, before removing any nodes.
        for (node, selection) in self.selections.iter().enumerate() {
            let mut text = Vec::new();
            for span in self.spans.iter().filter(|span| span.node == Some(node)) {
                let part = record
                    .text
                    .get(span.range.clone())
                    .context("invalid content range")?;
                match mode {
                    Mode::Default => {}
                    Mode::Expanded => {
                        let changes = edits(part);
                        if !changes.is_empty() {
                            let view = TextView {
                                text: part.into(),
                                edits: changes,
                            };
                            text.push(view.render(0..part.len())?);
                        }
                    }
                    Mode::Source => text.push(part.to_owned()),
                }
            }
            let included = mode == Mode::Source || (mode == Mode::Expanded && !text.is_empty());
            let replacement = if included {
                let mut detail = json!({"type": "detail", "text": text.join("\n")});
                if mode == Mode::Source {
                    let mut data = Vec::new();
                    let mut signatures = Vec::new();
                    for opaque in &selection.opaque {
                        let value = record
                            .structured_content
                            .pointer(opaque.pointer())
                            .context("content value missing")?;
                        match opaque {
                            Opaque::Value { kind, .. } => match kind {
                                OpaqueKind::Data => data.push(value.clone()),
                                OpaqueKind::Signature => signatures.push(value.clone()),
                            },
                            Opaque::NativeMap(_) => {
                                native_values(value, OpaqueKind::Data, &mut data, &mut signatures)
                            }
                        }
                    }
                    for (name, mut values) in [("data", data), ("signature", signatures)] {
                        if values.len() == 1 {
                            detail[name] = values.remove(0);
                        } else if !values.is_empty() {
                            detail[name] = values.into();
                        }
                    }
                }
                Some(detail)
            } else {
                None
            };
            replacements.push(replacement);
        }
        let mut value = record.structured_content.clone();
        let mut details = Vec::new();
        // Object groups cannot shift array positions; project them first.
        for (index, selection) in self.selections.iter().enumerate() {
            if let Target::Fields { object, members } = &selection.target {
                let object = value
                    .pointer_mut(object)
                    .and_then(Value::as_object_mut)
                    .context("content object missing")?;
                for key in members {
                    object.remove(key).context("content member missing")?;
                }
                if let Some(detail) = replacements[index].take() {
                    details.push(detail);
                }
            }
        }
        // Numeric path components are ordered numerically within each array.
        // Object members are removed by exact key after decoding the pointer.
        let mut nodes: Vec<_> = self
            .selections
            .iter()
            .enumerate()
            .filter_map(|(index, selection)| match &selection.target {
                Target::Node(pointer) => Some((pointer, index)),
                Target::Fields { .. } => None,
            })
            .collect();
        nodes.sort_by(|(a, _), (b, _)| {
            let (ap, ak) = a.rsplit_once('/').unwrap_or(("", ""));
            let (bp, bk) = b.rsplit_once('/').unwrap_or(("", ""));
            ap.cmp(bp)
                .then_with(|| match (ak.parse::<usize>(), bk.parse::<usize>()) {
                    (Ok(a), Ok(b)) => a.cmp(&b),
                    _ => ak.cmp(bk),
                })
        });
        for (pointer, index) in nodes.into_iter().rev() {
            replace_node(&mut value, pointer, replacements[index].take())?;
        }
        // Child nodes are projected before their native envelope is relocated.
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
            replace_node(&mut value, &collection.pointer, None)?;
            if mode == Mode::Source || non_selected {
                envelope.insert("type".into(), "archive".into());
                details.push(Value::Object(envelope));
            }
        }
        Ok(NativeProjection { value, details })
    }
}

fn replace_node(value: &mut Value, pointer: &str, replacement: Option<Value>) -> Result<()> {
    if pointer.is_empty() {
        *value = replacement.unwrap_or(Value::Null);
        return Ok(());
    }
    let (parent, member) = pointer.rsplit_once('/').context("invalid content path")?;
    let parent = value
        .pointer_mut(parent)
        .context("content container missing")?;
    match parent {
        Value::Array(array) => {
            let position: usize = member.parse().context("invalid content position")?;
            let node = array
                .get_mut(position)
                .context("content position missing")?;
            if let Some(replacement) = replacement {
                *node = replacement;
            } else {
                array.remove(position);
            }
        }
        Value::Object(object) => {
            let key = member.replace("~1", "/").replace("~0", "~");
            let node = object.get_mut(&key).context("content member missing")?;
            if let Some(replacement) = replacement {
                *node = replacement;
            } else {
                object.remove(&key);
            }
        }
        _ => anyhow::bail!("invalid content container"),
    }
    Ok(())
}

fn native_values(
    value: &Value,
    kind: OpaqueKind,
    data: &mut Vec<Value>,
    signatures: &mut Vec<Value>,
) {
    match value {
        Value::Object(object) if !object.is_empty() => {
            let mut members: Vec<_> = object.iter().collect();
            members.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (key, value) in members {
                let kind = if matches!(key.as_str(), "signature" | "thinkingSignature") {
                    OpaqueKind::Signature
                } else {
                    OpaqueKind::Data
                };
                native_values(value, kind, data, signatures);
            }
        }
        Value::Array(array) if !array.is_empty() => {
            for value in array {
                native_values(value, kind, data, signatures);
            }
        }
        _ => match kind {
            OpaqueKind::Data => data.push(value.clone()),
            OpaqueKind::Signature => signatures.push(value.clone()),
        },
    }
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
        body.finish(&mut record).unwrap();
        for mode in [Mode::Default, Mode::Expanded, Mode::Source] {
            let projected = record.clone().project(mode).unwrap();
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
            if mode == Mode::Default {
                assert_eq!(projected.text, "apples");
                assert!(projected.details.is_empty());
                assert!(projected.structured_content.get("member/key").is_none());
            } else {
                assert_eq!(projected.details.len(), 1);
                let member = &projected.structured_content["member/key"];
                assert_eq!(member["type"], "detail");
                if mode == Mode::Source {
                    assert_eq!(
                        member,
                        &json!({"type":"detail", "text":"apples", "data":[false,0,null]})
                    );
                    assert_eq!(
                        projected.details[0],
                        json!({"type":"detail", "text":"apples", "data":{"a":null}})
                    );
                } else {
                    assert_ne!(member["text"], "apples");
                    assert!(member.get("data").is_none());
                }
            }
        }
        let mut missing = record.clone();
        missing.source_spans.as_mut().unwrap().selections[0].target =
            Target::Node("/missing".into());
        assert!(missing.project(Mode::Source).is_err());
        let spans = record.source_spans.as_mut().unwrap();
        spans.selections.push(spans.selections[0].clone());
        assert!(record.project(Mode::Default).is_err());
    }
}

/// Edits use original Unicode scalar positions; protected spans still advance
/// the spacing counter. Backtick delimiters must have matching run lengths.
fn edits(text: &str) -> Vec<(usize, char)> {
    let mut result = Vec::new();
    let mut at = 0;
    let mut scalar = 0;
    let mut next = 0;
    while at < text.len() {
        let tail = &text[at..];
        let protected = if tail.starts_with('`') {
            let run = tail.bytes().take_while(|b| *b == b'`').count();
            let mut end = at + run;
            while end < text.len() {
                if text.as_bytes()[end] == b'`' {
                    let closing = text[end..].bytes().take_while(|b| *b == b'`').count();
                    end += closing;
                    if closing == run {
                        break;
                    }
                } else {
                    end += text[end..].chars().next().unwrap().len_utf8();
                }
            }
            Some(end)
        } else if tail
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
            || tail
                .get(..8)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
        {
            let delimiter = text[..at].chars().next_back();
            let mut end = at;
            let mut parentheses = 0;
            for ch in tail.chars() {
                if ch.is_whitespace() || (delimiter == Some('<') && ch == '>') {
                    break;
                }
                if delimiter == Some('(') {
                    if ch == '(' {
                        parentheses += 1;
                    } else if ch == ')' {
                        if parentheses == 0 {
                            break;
                        }
                        parentheses -= 1;
                    }
                }
                end += ch.len_utf8();
            }
            Some(end)
        } else {
            None
        };
        if let Some(end) = protected {
            scalar += text[at..end].chars().count();
            at = end;
            continue;
        }
        let ch = tail.chars().next().unwrap();
        let replacement = match ch {
            'a' => Some('а'),
            'c' => Some('с'),
            'e' => Some('е'),
            'o' => Some('о'),
            'p' => Some('р'),
            'x' => Some('х'),
            'A' => Some('А'),
            'C' => Some('С'),
            'E' => Some('Е'),
            'O' => Some('О'),
            'P' => Some('Р'),
            'X' => Some('Х'),
            _ => None,
        };
        if scalar >= next
            && let Some(replacement) = replacement
        {
            result.push((at, replacement));
            next = scalar + 32;
        }
        at += ch.len_utf8();
        scalar += 1;
    }
    result
}
