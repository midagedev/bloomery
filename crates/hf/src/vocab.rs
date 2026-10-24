//! A Hugging Face repo's byte-level BPE tokenizer (`tokenizer.json`, `tokenizer_config.json`,
//! `config.json`, `generation_config.json`, `chat_template.jinja`) as GGUF's `tokenizer.*`
//! keys with HF's token types, special ids and generation stop set:
//!
//! - every id below the vocabulary size names its token; an id the tokenizer has no token for is
//!   `[PAD<id>]`, type unused (5);
//! - an added token is control (3) only when HF marks it special, else user-defined (4);
//!   every other token is normal (1);
//! - an added token keeps its spelling, which serving matches and decodes as written (llama.cpp's
//!   converter writes a user-defined token's U+2581 as a space);
//! - the merges are `tokenizer.json`'s, each pair joined by one space, a pair listed twice at its
//!   last place as HF ranks it;
//! - a special id is `tokenizer_config.json`'s `<kind>_token` looked up among the added tokens,
//!   else `config.json`'s `<kind>_token_id`, its text config's, then `generation_config.json`'s;
//! - the chat template is `chat_template.jinja`, else `tokenizer_config.json`'s.
//!
//! Every key of `tokenizer.json`, its `model` and added tokens, `tokenizer_config.json` and
//! `generation_config.json` is named in a table below with the values serving computes the same
//! way HF does ([`schema`]); any other key or value is refused by name. Added tokens may not
//! overlap, so serving's longest-first split equals HF's leftmost one, and the ids HF's generate
//! stops at are written as `tokenizer.bloomery.eog_token_ids`. This nonempty id array opts
//! serving into explicit token types and ids, without name-based overrides or model defaults.
//! Serving checks every imported spelling against HF ByteLevel's decoded bytes, including
//! added tokens. The caller checks its split, special ids, automatic BOS/EOS and stop sets.
//!
//! The one pre-tokenizer the table allows is the one serving runs as `qwen35`, the name
//! (`tokenizer.ggml.pre`) llama.cpp gives Qwen's by the hash of its output. The stop id array
//! also makes serving encode as `tokenizer.json` does: the regex folds a contraction's case by
//! Unicode, and BPE skips a byte the vocabulary lacks.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use gguf::Value;
use serde_json::Value as Json;

use crate::schema::{self, Fields, Rule::*, Type};

/// Why the tokenizer was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("{0}: not JSON: {1}")]
    Json(String, #[source] serde_json::Error),
    #[error("tokenizer: {0}")]
    Vocab(String),
}

// The qwen35 pipeline's parts as tokenizer.json spells them, one place each.
macro_rules! byte_level {
    () => {
        r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#
    };
}
macro_rules! qwen35_regex {
    () => {
        r#""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""#
    };
}

/// `tokenizer.json`.
const TOKENIZER: &Fields = &[
    ("version", Only(r#"["1.0"]"#)),
    // HF truncates or pads an encoding only when these are set.
    ("truncation", Only("[null]")),
    ("padding", Only("[null]")),
    ("added_tokens", Read),
    // NFC is written as `tokenizer.ggml.normalizer`, which serving applies.
    ("normalizer", Only(r#"[null, {"type":"NFC"}]"#)),
    (
        "pre_tokenizer",
        Need(concat!(
            r#"[{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"#,
            qwen35_regex!(),
            r#"},"behavior":"Isolated","invert":false},"#,
            byte_level!(),
            "]}]"
        )),
    ),
    // ByteLevel without trim_offsets moves no id, only offsets.
    (
        "post_processor",
        Only(concat!("[null, ", byte_level!(), "]")),
    ),
    ("decoder", Need(concat!("[", byte_level!(), "]"))),
    ("model", Read),
];

/// `tokenizer.json`'s `model`: the BPE serving runs, ids and merges only.
const MODEL: &Fields = &[
    ("type", Need(r#"["BPE"]"#)),
    ("vocab", Read),
    ("merges", Read),
    ("dropout", Only("[null, 0, 0.0]")),
    ("unk_token", Only("[null]")),
    ("continuing_subword_prefix", Only(r#"[null, ""]"#)),
    ("end_of_word_suffix", Only(r#"[null, ""]"#)),
    ("fuse_unk", Only("[false]")),
    ("byte_fallback", Only("[false]")),
    ("ignore_merges", Only("[false]")),
];

/// An added token, in `added_tokens` (with [`ID`]) or as `tokenizer_config.json` gives one.
/// Serving matches its text as written wherever it occurs.
const TOKEN: &Fields = &[
    ("content", Read),
    ("special", Read),
    // false; absent only on a special token, which HF then reads as not normalized.
    ("normalized", Read),
    ("single_word", Only("[false]")),
    ("lstrip", Only("[false]")),
    ("rstrip", Only("[false]")),
];

const ID: &Fields = &[("id", Read)];

/// `tokenizer_config.json`.
const TOKENIZER_CONFIG: &Fields = &[
    ("added_tokens_decoder", Read),
    ("additional_special_tokens", Read),
    ("extra_special_tokens", Read),
    ("bos_token", Read),
    ("eos_token", Read),
    ("pad_token", Read),
    ("sep_token", Read),
    ("unk_token", Only("[null]")),
    ("add_bos_token", Read),
    ("add_eos_token", Read),
    ("chat_template", Read),
    ("add_prefix_space", Only("[false]")),
    ("do_lower_case", Only("[false]")),
    ("split_special_tokens", Only("[false]")),
    ("clean_up_tokenization_spaces", Only("[false]")),
    // Serving decodes invalid UTF-8 as U+FFFD.
    ("errors", Only(r#"["replace"]"#)),
    (
        "pretokenize_regex",
        Only(concat!("[", qwen35_regex!(), "]")),
    ),
    // Each builds the tokenizer from tokenizer.json; `TokenizersBackend` is
    // transformers' class for a tokenizer.json with no class of its own.
    (
        "tokenizer_class",
        Only(r#"["Qwen2Tokenizer", "Qwen2TokenizerFast", "TokenizersBackend"]"#),
    ),
    (
        "model_max_length",
        Ignore(
            Type::Number,
            "HF warns past it and truncates only when asked; the context length is config.json's",
        ),
    ),
    // Both build the fast tokenizer from tokenizer.json.
    ("backend", Only(r#"["tokenizers"]"#)),
    (
        "is_local",
        Ignore(
            Type::Bool,
            "where HF loaded the folder from; no id depends on it",
        ),
    ),
    (
        "processor_class",
        Ignore(
            Type::String,
            "the multimodal processor's class; the text tokenizer is tokenizer_class",
        ),
    ),
    ("audio_bos_token", Ignore(Type::String, MEDIA)),
    ("audio_eos_token", Ignore(Type::String, MEDIA)),
    ("audio_token", Ignore(Type::String, MEDIA)),
    ("image_token", Ignore(Type::String, MEDIA)),
    ("video_token", Ignore(Type::String, MEDIA)),
    ("vision_bos_token", Ignore(Type::String, MEDIA)),
    ("vision_eos_token", Ignore(Type::String, MEDIA)),
    ("model_specific_special_tokens", Ignore(Type::Object, MEDIA)),
];

const MEDIA: &str = "the processor's media markers by name; each is an added token of \
    tokenizer.json, which serving matches as written";

const SAMPLING: &str = "a default of HF's generate; serving samples by each request, and no \
    prompt's ids or logits depend on it";

/// `generation_config.json`.
const GENERATION: &Fields = &[
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("do_sample", Ignore(Type::Bool, SAMPLING)),
    ("temperature", Ignore(Type::Number, SAMPLING)),
    ("top_k", Ignore(Type::Number, SAMPLING)),
    ("top_p", Ignore(Type::Number, SAMPLING)),
];

/// An added token's object at `path`, `id` with its id: the keys [`TOKEN`] names and HF's
/// unnormalized matching.
fn token(token: &Json, path: &str, id: bool) -> Result<(), Error> {
    let tables: &[&Fields] = if id { &[ID, TOKEN] } else { &[TOKEN] };
    schema::check(token, path, tables).map_err(Error::Vocab)?;
    if token.get("special").is_some_and(|v| !v.is_boolean()) {
        return Err(Error::Vocab(format!("{path}.special is not boolean")));
    }
    match token.get("normalized") {
        Some(Json::Bool(false)) => Ok(()),
        None if token["special"] == true => Ok(()),
        _ => Err(Error::Vocab(format!("{path}.normalized must be false"))),
    }
}

/// The first pair of added tokens HF and serving would split differently: serving cuts the
/// longest token out first, HF the leftmost. They agree when no token's proper suffix starts
/// another token and is shorter than it (a token inside another is cut with it).
fn overlap(added: &HashMap<u32, (&str, bool)>) -> Option<(u32, u32)> {
    let mut ids: Vec<u32> = added.keys().copied().collect();
    ids.sort_unstable();
    for &a in &ids {
        let left = added[&a].0.as_bytes();
        for &b in &ids {
            let right = added[&b].0.as_bytes();
            if a != b
                && (1..left.len())
                    .any(|at| left.len() - at < right.len() && right.starts_with(&left[at..]))
            {
                return Some((a, b));
            }
        }
    }
    None
}

const SPECIAL: [(&str, &str); 5] = [
    ("bos", "bos_token_id"),
    ("eos", "eos_token_id"),
    ("unk", "unknown_token_id"),
    ("sep", "seperator_token_id"),
    ("pad", "padding_token_id"),
];

/// The vocabulary domain and special ids selected from `config.json`.
#[derive(Debug)]
pub struct Config {
    size: u32,
    ids: [Option<Vec<u32>>; 5],
}

impl Config {
    /// Read only the vocabulary fields, refusing malformed counts and ids by name.
    pub fn parse(config: &Json) -> Result<Self, Error> {
        let bad = |s: &str| Error::Vocab(s.into());
        if !config.is_object() {
            return Err(bad("config.json is not an object"));
        }
        let text = config.get("text_config").unwrap_or(config);
        if !text.is_object() {
            return Err(bad("text_config is not an object"));
        }
        let size = text
            .get("vocab_size")
            .or_else(|| config.get("vocab_size"))
            .and_then(Json::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| bad("vocab_size must be a positive u32"))?;
        let mut out = Self {
            size,
            ids: std::array::from_fn(|_| None),
        };
        for (i, (kind, _)) in SPECIAL.iter().enumerate() {
            let key = format!("{kind}_token_id");
            out.ids[i] = config
                .get(&key)
                .filter(|v| !v.is_null())
                .or_else(|| text.get(&key).filter(|v| !v.is_null()))
                .map(|v| out.token_ids(&key, v))
                .transpose()?;
        }
        Ok(out)
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// The last EOS id resets the PLE hash; GGUF's tokenizer uses the first.
    pub fn ple_eos(&self) -> Option<u32> {
        self.ids[1].as_ref()?.last().copied()
    }

    fn token_id(&self, key: &str, v: &Json) -> Result<u32, Error> {
        let id = v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Error::Vocab(format!("{key} {v} is not a token id in u32")))?;
        if id >= self.size {
            return Err(Error::Vocab(format!(
                "{key} {id} is outside the vocabulary of {}",
                self.size
            )));
        }
        Ok(id)
    }

    fn token_ids(&self, key: &str, v: &Json) -> Result<Vec<u32>, Error> {
        match v {
            Json::Array(a) if !a.is_empty() => a.iter().map(|v| self.token_id(key, v)).collect(),
            Json::Array(_) => Err(Error::Vocab(format!("{key} is an empty token id list"))),
            _ => self.token_id(key, v).map(|id| vec![id]),
        }
    }
}

/// A repo's tokenizer as GGUF's `tokenizer.*` keys, with the ids serving must treat as HF does.
#[derive(Debug)]
pub struct Keys {
    pub keys: Vec<(String, Value)>,
    /// The added tokens' ids: what HF splits a text on before BPE.
    pub added: Vec<u32>,
    /// The ids HF's generate stops at: `generation_config.json`'s `eos_token_id`, else
    /// `config.json`'s, else the EOS id.
    pub stop: Vec<u32>,
}

/// A tokenizer's vocabulary as the GGUF keys spell it: every id's token and type, the merges in
/// HF's rank order, and the added tokens by id (content, special).
struct Vocabulary<'a> {
    tokens: Vec<Value>,
    types: Vec<Value>,
    merges: Vec<Value>,
    added: HashMap<u32, (&'a str, bool)>,
}

/// The vocabulary of `tokenizer.json` (`model` its `model`), its added tokens checked against
/// `tokenizer_config.json` (`tc`), and its merges.
fn vocabulary<'a>(
    model: &'a Json,
    tok: &'a Json,
    tc: &Json,
    config: &Config,
) -> Result<Vocabulary<'a>, Error> {
    let bad = |what: String| Error::Vocab(what);
    let vocab = model["vocab"]
        .as_object()
        .ok_or_else(|| bad("tokenizer.json has no model.vocab".into()))?;
    let mut by_id: HashMap<u32, &str> = HashMap::new();
    let mut by_name: HashMap<&str, u32> = HashMap::new();
    for (t, id) in vocab {
        let id = config.token_id(&format!("model.vocab[{t:?}] id"), id)?;
        if by_id.contains_key(&id) {
            return Err(bad(format!("duplicate vocabulary id {id}")));
        }
        by_id.insert(id, t);
        by_name.insert(t, id);
    }
    let mut added: HashMap<u32, (&str, bool)> = HashMap::new();
    for (i, a) in tok
        .get("added_tokens")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_array()
                .ok_or_else(|| bad("added_tokens is not an array".into()))
        })
        .transpose()?
        .into_iter()
        .flatten()
        .enumerate()
    {
        let id = config.token_id("added_tokens.id", &a["id"])?;
        let t = a["content"]
            .as_str()
            .filter(|t| !t.is_empty())
            .ok_or_else(|| bad(format!("added token {a} has no content")))?;
        if added.contains_key(&id)
            || by_id.get(&id).is_some_and(|&old| old != t)
            || by_name.get(t).is_some_and(|&old| old != id)
        {
            return Err(bad(format!(
                "conflicting added token id {id} or content {t:?}"
            )));
        }
        token(a, &format!("added_tokens[{i}]"), true)?;
        by_name.insert(t, id);
        added.insert(id, (t, a["special"] == true));
        by_id.insert(id, t);
    }
    if let Some((a, b)) = overlap(&added) {
        return Err(bad(format!(
            "added tokens {a} {:?} and {b} {:?} overlap, which serving splits unlike HF",
            added[&a].0, added[&b].0
        )));
    }
    if let Some(decoder) = tc.get("added_tokens_decoder") {
        let decoder = decoder
            .as_object()
            .ok_or_else(|| bad("added_tokens_decoder is not an object".into()))?;
        for (id, t) in decoder {
            let path = format!("added_tokens_decoder[{id}]");
            token(t, &path, false)?;
            let entry = id.parse::<u32>().ok().and_then(|id| added.get(&id));
            if !entry.is_some_and(|&(content, special)| {
                t["content"].as_str() == Some(content)
                    && t.get("special")
                        .is_none_or(|v| v.as_bool() == Some(special))
            }) {
                return Err(bad(format!(
                    "{path} content or special disagrees with tokenizer.json added_tokens"
                )));
            }
        }
    }
    let mut declared = Vec::new();
    if let Some(tokens) = tc.get("additional_special_tokens") {
        let tokens = tokens
            .as_array()
            .ok_or_else(|| bad("additional_special_tokens is not an array".into()))?;
        declared.extend(tokens.iter().map(|v| ("additional_special_tokens", v)));
    }
    if let Some(tokens) = tc.get("extra_special_tokens") {
        let tokens = tokens
            .as_object()
            .ok_or_else(|| bad("extra_special_tokens is not an object".into()))?;
        declared.extend(tokens.values().map(|v| ("extra_special_tokens", v)));
    }
    for (field, t) in declared {
        if !t.is_string() {
            token(t, field, false)?;
        }
        let content = t.as_str().or_else(|| t["content"].as_str());
        if !content.is_some_and(|c| added.values().any(|&(t, special)| t == c && special)) {
            return Err(bad(format!("{field} {t} is no special added token")));
        }
    }
    let (mut tokens, mut types) = (Vec::new(), Vec::new());
    for id in 0..config.size {
        let (t, ty) = match (by_id.get(&id), added.get(&id)) {
            (None, _) => {
                let pad = format!("[PAD{id}]");
                if by_name.contains_key(pad.as_str()) {
                    return Err(bad(format!(
                        "token {pad:?} spells the filler of unused id {id}"
                    )));
                }
                (pad, 5)
            }
            (Some(t), Some(&(_, true))) => (t.to_string(), 3),
            (Some(t), Some(_)) => (t.to_string(), 4),
            (Some(t), None) => (t.to_string(), 1),
        };
        tokens.push(Value::String(t));
        types.push(Value::I32(ty));
    }
    let pairs = model["merges"]
        .as_array()
        .ok_or_else(|| bad("tokenizer.json has no model.merges".into()))?
        .iter()
        .map(|m| {
            let pair = match m {
                Json::String(s) => s.split_once(' '),
                Json::Array(p) if p.len() == 2 => p[0].as_str().zip(p[1].as_str()),
                _ => None,
            };
            let (a, b) = pair
                .filter(|(a, b)| {
                    !a.is_empty() && !b.is_empty() && !a.contains(' ') && !b.contains(' ')
                })
                .ok_or_else(|| bad(format!("model.merges: {m} is not a pair of two tokens")))?;
            if !vocab.contains_key(a)
                || !vocab.contains_key(b)
                || !vocab.contains_key(&format!("{a}{b}"))
            {
                return Err(bad(format!(
                    "model.merges: {m} names a token outside model.vocab"
                )));
            }
            Ok((a, b))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    // HF ranks a pair listed twice by its last place, serving by its first: keep the last.
    let last: HashMap<_, _> = pairs.iter().enumerate().map(|(i, p)| (p, i)).collect();
    let merges = pairs
        .iter()
        .enumerate()
        .filter(|(i, p)| last[p] == *i)
        .map(|(_, (a, b))| Value::String(format!("{a} {b}")))
        .collect();
    Ok(Vocabulary {
        tokens,
        types,
        merges,
        added,
    })
}

/// [`special_ids`]: the `tokenizer.ggml.<kind>_token_id` keys and the generation stop ids.
type SpecialIds = (Vec<(String, Value)>, Option<Vec<u32>>);

/// The special ids the keys export, as `tokenizer.ggml.<kind>_token_id` keys, and the
/// generation stop ids: each kind's token named in `tokenizer_config.json` (`tc`), else
/// `config.json`'s id, else `generation_config.json`'s.
fn special_ids(
    tc: &Json,
    generation: &Json,
    config: &Config,
    added: &HashMap<u32, (&str, bool)>,
) -> Result<SpecialIds, Error> {
    let bad = |what: String| Error::Vocab(what);
    let mut ids = Vec::new();
    let mut stop = None;
    for (i, (kind, key)) in SPECIAL.iter().enumerate() {
        let named = match &tc[format!("{kind}_token")] {
            Json::Null => None,
            Json::String(s) => Some(s.as_str()),
            v => {
                token(v, &format!("{kind}_token"), false)?;
                Some(v.get("content").and_then(Json::as_str).ok_or_else(|| {
                    bad(format!("{kind}_token {v} is not a token or an added token"))
                })?)
            }
        };
        let field = format!("{kind}_token_id");
        let generated = generation
            .get(&field)
            .filter(|v| !v.is_null())
            .map(|v| config.token_ids(&field, v))
            .transpose()?;
        let id = match named {
            Some(t) => Some(
                added
                    .iter()
                    .find(|(_, (c, _))| *c == t)
                    .map(|(&id, _)| id)
                    .ok_or_else(|| bad(format!("{kind}_token {t:?} is no added token")))?,
            ),
            None => config.ids[i]
                .as_ref()
                .or(generated.as_ref())
                .and_then(|ids| ids.first().copied()),
        };
        if *kind == "eos" {
            // HF's generate stops at any of these; serving at its end-of-generation ids.
            stop = generated
                .or_else(|| config.ids[i].clone())
                .or(id.map(|id| vec![id]));
        } else if let Some(other) = [&config.ids[i], &generated]
            .into_iter()
            .flatten()
            .flatten()
            .find(|&&other| Some(other) != id)
        {
            return Err(bad(format!(
                "{field} {other} disagrees with the exported {kind} id {id:?}"
            )));
        }
        if let Some(id) = id {
            ids.push((format!("tokenizer.ggml.{key}"), Value::U32(id)));
        }
    }
    Ok((ids, stop))
}

/// The `tokenizer.*` keys of the repo at `dir`.
///
/// # Errors
///
/// A file that cannot be read or parsed; in `tokenizer.json`, its `model`, an added token,
/// `tokenizer_config.json` or `generation_config.json`, a key the tables of this module do not
/// name or a value they do not allow; a `tokenizer.json` without its vocabulary or merges, or with
/// a merge that is not two tokens without a space whose operands and joined token are all in its
/// vocabulary; an added token whose id or content conflicts, or whose proper suffix
/// starts another added token; a token spelled as an unused id's `[PAD<id>]`; a vocabulary size
/// `config.json` does not give; a special token in `tokenizer_config.json` that is neither a
/// string nor an added token's object, or that no added token spells; a special token id that is
/// not an id or a list of them, lies outside the vocabulary, or is a BOS, pad, unknown or
/// separator id other than the one exported; an `add_bos_token` or `add_eos_token` that is not
/// boolean, and a `chat_template` that is not a string, each by name.
pub fn keys(dir: &Path, config: &Config) -> Result<Keys, Error> {
    let bad = |what: String| Error::Vocab(what);
    let read = |file: &str| crate::read_json(dir, file, Error::Io, Error::Json);
    let tok = read("tokenizer.json")?.ok_or_else(|| bad("no tokenizer.json".into()))?;
    let tc = read("tokenizer_config.json")?.unwrap_or_else(|| serde_json::json!({}));
    let generation = read("generation_config.json")?.unwrap_or_else(|| serde_json::json!({}));
    for (file, value, table) in [
        ("tokenizer.json", &tok, TOKENIZER),
        ("tokenizer_config.json", &tc, TOKENIZER_CONFIG),
        ("generation_config.json", &generation, GENERATION),
    ] {
        schema::check(value, file, &[table]).map_err(bad)?;
    }
    let model = tok
        .get("model")
        .ok_or_else(|| bad("tokenizer.json has no model".into()))?;
    schema::check(model, "model", &[MODEL]).map_err(bad)?;
    let nfc = tok["normalizer"] == serde_json::json!({"type":"NFC"});
    let Vocabulary {
        tokens,
        types,
        merges,
        added,
    } = vocabulary(model, &tok, &tc, config)?;
    let mut out = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::String("qwen35".into()),
        ),
        ("tokenizer.ggml.tokens".to_string(), Value::Array(tokens)),
        ("tokenizer.ggml.token_type".to_string(), Value::Array(types)),
        ("tokenizer.ggml.merges".to_string(), Value::Array(merges)),
    ];
    if nfc {
        out.push((
            "tokenizer.ggml.normalizer".into(),
            Value::String("NFC".into()),
        ));
    }
    let (ids, stop) = special_ids(&tc, &generation, config, &added)?;
    out.extend(ids);
    for kind in ["bos", "eos"] {
        let key = format!("add_{kind}_token");
        let b = match tc.get(&key).filter(|v| !v.is_null()) {
            Some(v) => v
                .as_bool()
                .ok_or_else(|| bad(format!("{key} {v} is not boolean")))?,
            None => false,
        };
        out.push((format!("tokenizer.ggml.{key}"), Value::Bool(b)));
    }
    let template = match fs::read_to_string(dir.join("chat_template.jinja")) {
        Ok(t) => Some(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            match tc.get("chat_template").filter(|v| !v.is_null()) {
                Some(v) => Some(
                    v.as_str()
                        .ok_or_else(|| bad(format!("chat_template {v} is not a string")))?
                        .to_string(),
                ),
                None => None,
            }
        }
        Err(e) => {
            return Err(Error::Io(
                dir.join("chat_template.jinja").display().to_string(),
                e,
            ));
        }
    };
    if let Some(t) = template {
        out.push(("tokenizer.chat_template".to_string(), Value::String(t)));
    }
    let mut stop = stop
        .filter(|ids| !ids.is_empty())
        .ok_or_else(|| bad("eos_token_id defines no generation stop tokens".into()))?;
    stop.sort_unstable();
    stop.dedup();
    out.push((
        "tokenizer.bloomery.eog_token_ids".into(),
        Value::Array(stop.iter().copied().map(Value::U32).collect()),
    ));
    let mut added: Vec<u32> = added.into_keys().collect();
    added.sort_unstable();
    Ok(Keys {
        keys: out,
        added,
        stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(dir: &Path) -> Result<Vec<(String, Value)>, Error> {
        let config =
            crate::read_json(dir, "config.json", Error::Io, Error::Json)?.unwrap_or(Json::Null);
        super::keys(dir, &Config::parse(&config)?).map(|k| k.keys)
    }

    fn repo(files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hf-vocab-{}-{}", std::process::id(), files.len()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in files {
            std::fs::write(dir.join(name), text).unwrap();
        }
        dir
    }

    fn get<'a>(keys: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
        keys.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A Qwen-shaped tokenizer: four tokens and three added ones (one special, one only looking
    /// special, one plain), a gap to the vocabulary size, a merge listed twice (HF ranks it at its
    /// last place), the special ids by name and by config.
    #[test]
    fn a_bpe_tokenizer_becomes_ggml_s_keys() {
        let dir = repo(&[
            (
                "tokenizer.json",
                r#"{"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
                "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
                "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"abb":6},"merges":["a b",["ab","b"],"a b"]},
                "added_tokens":[{"id":3,"content":"<|im_end|>","special":true},
                {"id":4,"content":"<|fim_pad|>","special":false,"normalized":false},
                {"id":5,"content":"<think>","special":false,"normalized":false}]}"#,
            ),
            (
                "tokenizer_config.json",
                r#"{"eos_token":"<|im_end|>","bos_token":null,"add_bos_token":false,"chat_template":"cfg"}"#,
            ),
            ("config.json", r#"{"text_config":{"vocab_size":8}}"#),
            (
                "generation_config.json",
                r#"{"bos_token_id":3,"pad_token_id":4}"#,
            ),
            ("chat_template.jinja", "{{ messages }}"),
        ]);
        let keys = self::keys(&dir).unwrap();
        let strings = |v: &Value| match v {
            Value::Array(a) => a
                .iter()
                .map(|x| match x {
                    Value::String(s) => s.clone(),
                    Value::I32(i) => i.to_string(),
                    _ => panic!("{x:?}"),
                })
                .collect::<Vec<_>>(),
            _ => panic!("{v:?}"),
        };
        let toks = strings(get(&keys, "tokenizer.ggml.tokens").unwrap());
        assert_eq!(
            toks,
            [
                "a",
                "b",
                "ab",
                "<|im_end|>",
                "<|fim_pad|>",
                "<think>",
                "abb",
                "[PAD7]"
            ]
        );
        let types = strings(get(&keys, "tokenizer.ggml.token_type").unwrap());
        assert_eq!(types, ["1", "1", "1", "3", "4", "4", "1", "5"]);
        assert_eq!(
            strings(get(&keys, "tokenizer.ggml.merges").unwrap()),
            ["ab b", "a b"]
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.eos_token_id"),
            Some(&Value::U32(3))
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.bos_token_id"),
            Some(&Value::U32(3))
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.padding_token_id"),
            Some(&Value::U32(4))
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.add_bos_token"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.add_eos_token"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            get(&keys, "tokenizer.bloomery.eog_token_ids"),
            Some(&Value::Array(vec![Value::U32(3)]))
        );
        assert_eq!(
            get(&keys, "tokenizer.chat_template"),
            Some(&Value::String("{{ messages }}".into()))
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.pre"),
            Some(&Value::String("qwen35".into()))
        );
        assert_eq!(
            get(&keys, "tokenizer.ggml.normalizer"),
            Some(&Value::String("NFC".into()))
        );
        let mut tokenizer: Json =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap())
                .unwrap();
        for normalizer in [Some(Json::Null), None] {
            if let Some(value) = normalizer {
                tokenizer["normalizer"] = value;
            } else {
                tokenizer.as_object_mut().unwrap().remove("normalizer");
            }
            std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
            let keys = self::keys(&dir).unwrap();
            assert_eq!(get(&keys, "tokenizer.ggml.normalizer"), None);
        }
        std::fs::write(dir.join("generation_config.json"), "{}").unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"text_config":{"vocab_size":8,"eos_token_id":[3,4]}}"#,
        )
        .unwrap();
        let keys = self::keys(&dir).unwrap();
        assert_eq!(
            get(&keys, "tokenizer.ggml.eos_token_id"),
            Some(&Value::U32(3))
        );
        assert_eq!(
            get(&keys, "tokenizer.bloomery.eog_token_ids"),
            Some(&Value::Array(vec![Value::U32(3), Value::U32(4)]))
        );
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id":[4,3,4]}"#,
        )
        .unwrap();
        let keys = self::keys(&dir).unwrap();
        assert_eq!(
            get(&keys, "tokenizer.bloomery.eog_token_ids"),
            Some(&Value::Array(vec![Value::U32(3), Value::U32(4)]))
        );
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":4}"#).unwrap();
        let keys = self::keys(&dir).unwrap();
        assert_eq!(
            get(&keys, "tokenizer.bloomery.eog_token_ids"),
            Some(&Value::Array(vec![Value::U32(4)]))
        );
        std::fs::write(dir.join("generation_config.json"), "{}").unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"text_config":{"vocab_size":8}}"#,
        )
        .unwrap();
        assert!(
            self::keys(&dir)
                .unwrap_err()
                .to_string()
                .contains("eos_token_id")
        );
        for f in [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "generation_config.json",
            "chat_template.jinja",
        ] {
            std::fs::remove_file(dir.join(f)).unwrap();
        }
        std::fs::remove_dir(&dir).unwrap();
    }

    /// The schema tables row by row: for every key of every table, one value accepted and one
    /// refused by the key's name, a key no table names refused in each object, and added tokens
    /// that overlap, which serving would split unlike HF.
    #[test]
    fn every_tokenizer_key_accepts_its_values_and_refuses_others() {
        use serde_json::json;
        let byte_level = json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,
            "use_regex":false});
        let regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let pre = json!({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":regex},
            "behavior":"Isolated","invert":false}, byte_level.clone()]});
        let end = json!({"id":3,"content":"<|im_end|>","special":true,"normalized":false,
            "single_word":false,"lstrip":false,"rstrip":false});
        let mut decoded = end.clone();
        decoded.as_object_mut().unwrap().remove("id");
        let base: std::collections::BTreeMap<&str, Json> = [
            (
                "tokenizer.json",
                json!({"version":"1.0","truncation":null,"padding":null,"normalizer":{"type":"NFC"},
                "pre_tokenizer":pre,"post_processor":byte_level,"decoder":byte_level,
                "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},
                "added_tokens":[end, {"id":4,"content":"<think>","special":false,"normalized":false}]}),
            ),
            (
                "tokenizer_config.json",
                json!({"added_tokens_decoder":{"3":decoded},"pretokenize_regex":regex}),
            ),
            ("generation_config.json", json!({})),
            ("config.json", json!({"text_config":{"vocab_size":8,"eos_token_id":3}})),
        ]
        .into_iter()
        .collect();
        let dir = repo(&[]);
        let run = |files: &std::collections::BTreeMap<&str, Json>| {
            for (name, value) in files {
                std::fs::write(dir.join(name), value.to_string()).unwrap();
            }
            let config = Config::parse(&files["config.json"])?;
            super::keys(&dir, &config)
        };
        let all = |t: &Fields| t.iter().map(|&(k, _)| k).collect::<Vec<_>>();
        let objects: [(&str, &str, Vec<&str>); 6] = [
            ("tokenizer.json", "", all(TOKENIZER)),
            ("tokenizer.json", "/model", all(MODEL)),
            (
                "tokenizer.json",
                "/added_tokens/0",
                [all(ID), all(TOKEN)].concat(),
            ),
            (
                "tokenizer_config.json",
                "/added_tokens_decoder/3",
                all(TOKEN),
            ),
            ("tokenizer_config.json", "", all(TOKENIZER_CONFIG)),
            ("generation_config.json", "", all(GENERATION)),
        ];
        // file, object, key, a value accepted ("$" keeps the base's), a value refused.
        let rows: Vec<(&str, &str, &str, Json, Json)> = serde_json::from_str(r###"[
            ["tokenizer.json", "", "version", "1.0", "2.0"],
            ["tokenizer.json", "", "truncation", null, {"max_length":1}],
            ["tokenizer.json", "", "padding", null, {"length":8}],
            ["tokenizer.json", "", "added_tokens", "$", {"id":3}],
            ["tokenizer.json", "", "normalizer", null, {"type":"NFD"}],
            ["tokenizer.json", "", "pre_tokenizer", "$", {"type":"Whitespace"}],
            ["tokenizer.json", "", "post_processor", null, {"type":"TemplateProcessing"}],
            ["tokenizer.json", "", "decoder", "$", {"type":"WordPiece"}],
            ["tokenizer.json", "", "model", "$", {"type":"Unigram"}],
            ["tokenizer.json", "/model", "type", "BPE", "Unigram"],
            ["tokenizer.json", "/model", "vocab", {"a":0,"b":1,"ab":2,"c":5}, {"a":-1}],
            ["tokenizer.json", "/model", "merges", ["a b"], [["a","b","c"]]],
            ["tokenizer.json", "/model", "dropout", 0.0, 0.1],
            ["tokenizer.json", "/model", "unk_token", null, "a"],
            ["tokenizer.json", "/model", "continuing_subword_prefix", "", "##"],
            ["tokenizer.json", "/model", "end_of_word_suffix", "", "</w>"],
            ["tokenizer.json", "/model", "fuse_unk", false, true],
            ["tokenizer.json", "/model", "byte_fallback", false, true],
            ["tokenizer.json", "/model", "ignore_merges", false, true],
            ["tokenizer.json", "/added_tokens/0", "id", 3, 8],
            ["tokenizer.json", "/added_tokens/0", "content", "$", ""],
            ["tokenizer.json", "/added_tokens/0", "special", true, "yes"],
            ["tokenizer.json", "/added_tokens/0", "normalized", false, true],
            ["tokenizer.json", "/added_tokens/0", "single_word", false, true],
            ["tokenizer.json", "/added_tokens/0", "lstrip", false, true],
            ["tokenizer.json", "/added_tokens/0", "rstrip", false, true],
            ["tokenizer_config.json", "/added_tokens_decoder/3", "content", "$", "<think>"],
            ["tokenizer_config.json", "/added_tokens_decoder/3", "special", true, false],
            ["tokenizer_config.json", "/added_tokens_decoder/3", "normalized", false, true],
            ["tokenizer_config.json", "/added_tokens_decoder/3", "single_word", false, true],
            ["tokenizer_config.json", "/added_tokens_decoder/3", "lstrip", false, true],
            ["tokenizer_config.json", "/added_tokens_decoder/3", "rstrip", false, true],
            ["tokenizer_config.json", "", "added_tokens_decoder", {}, []],
            ["tokenizer_config.json", "", "additional_special_tokens", ["<|im_end|>"], ["<think>"]],
            ["tokenizer_config.json", "", "extra_special_tokens", {"end":"<|im_end|>"}, []],
            ["tokenizer_config.json", "", "bos_token", "<|im_end|>", "missing"],
            ["tokenizer_config.json", "", "eos_token", {"content":"<|im_end|>","special":true}, 7],
            ["tokenizer_config.json", "", "pad_token", "<|im_end|>", {"content":"<|im_end|>","lstrip":true}],
            ["tokenizer_config.json", "", "sep_token", null, "missing"],
            ["tokenizer_config.json", "", "unk_token", null, "<|im_end|>"],
            ["tokenizer_config.json", "", "add_bos_token", false, "yes"],
            ["tokenizer_config.json", "", "add_eos_token", false, 1],
            ["tokenizer_config.json", "", "chat_template", "t", [{"template":"t"}]],
            ["tokenizer_config.json", "", "add_prefix_space", false, true],
            ["tokenizer_config.json", "", "do_lower_case", false, true],
            ["tokenizer_config.json", "", "split_special_tokens", false, true],
            ["tokenizer_config.json", "", "clean_up_tokenization_spaces", false, true],
            ["tokenizer_config.json", "", "errors", "replace", "ignore"],
            ["tokenizer_config.json", "", "pretokenize_regex", "$", "different"],
            ["tokenizer_config.json", "", "tokenizer_class", "Qwen2Tokenizer", "LlamaTokenizer"],
            ["tokenizer_config.json", "", "model_max_length", 262144, "long"],
            ["tokenizer_config.json", "", "backend", "tokenizers", "sentencepiece"],
            ["tokenizer_config.json", "", "is_local", true, "yes"],
            ["tokenizer_config.json", "", "processor_class", "Qwen3VLProcessor", 1],
            ["tokenizer_config.json", "", "audio_bos_token", "<|audio_start|>", 1],
            ["tokenizer_config.json", "", "audio_eos_token", "<|audio_end|>", 1],
            ["tokenizer_config.json", "", "audio_token", "<|audio_pad|>", 1],
            ["tokenizer_config.json", "", "image_token", "<|image_pad|>", 1],
            ["tokenizer_config.json", "", "video_token", "<|video_pad|>", 1],
            ["tokenizer_config.json", "", "vision_bos_token", "<|vision_start|>", 1],
            ["tokenizer_config.json", "", "vision_eos_token", "<|vision_end|>", 1],
            ["tokenizer_config.json", "", "model_specific_special_tokens", {"image_token":"<|image_pad|>"}, "x"],
            ["generation_config.json", "", "bos_token_id", 3, [3, 4]],
            ["generation_config.json", "", "eos_token_id", [3, 4], [3, "4"]],
            ["generation_config.json", "", "pad_token_id", 4, [4, 3]],
            ["generation_config.json", "", "do_sample", true, "yes"],
            ["generation_config.json", "", "temperature", 1.0, "hot"],
            ["generation_config.json", "", "top_k", 20, "20"],
            ["generation_config.json", "", "top_p", 0.95, [0.95]]
        ]"###)
        .unwrap();
        let mut missed = vec![];
        for (file, at, keys) in &objects {
            for key in keys {
                if !rows.iter().any(|r| (r.0, r.1, r.2) == (*file, *at, *key)) {
                    missed.push(format!("{file}{at} {key}: no row"));
                }
            }
        }
        let with = |file: &str, at: &str, key: &str, value: &Json| {
            let mut files = base.clone();
            let object = files.get_mut(file).unwrap().pointer_mut(at).unwrap();
            if value != "$" {
                object[key] = value.clone();
            }
            files
        };
        assert!(run(&base).is_ok(), "{:?}", run(&base));
        let unknown = objects
            .iter()
            .map(|(f, at, _)| (*f, *at, "zzz", json!("$"), json!(1)));
        for (file, at, key, allowed, refused) in rows.into_iter().chain(unknown) {
            if let Err(e) = run(&with(file, at, key, &allowed)) {
                missed.push(format!("{file}{at} {key}={allowed}: refused: {e}"));
            }
            match run(&with(file, at, key, &refused)) {
                Err(e) if e.to_string().contains(key) => {}
                other => missed.push(format!("{file}{at} {key}={refused}: {other:?}")),
            }
        }
        let mut stops = with("generation_config.json", "", "eos_token_id", &json!([4, 3]));
        stops.get_mut("tokenizer_config.json").unwrap()["eos_token"] = json!("<|im_end|>");
        let keys = run(&stops).unwrap();
        assert_eq!((keys.added, keys.stop), (vec![3, 4], vec![3, 4]));
        // A merge whose operand ("z") or joined token ("ba") is outside the vocabulary, which HF
        // refuses and serving would leave unmerged, or that is not two tokens without a space.
        for merge in [json!("a z"), json!("b a"), json!(["a b", "b"]), json!("ab")] {
            let mut files = base.clone();
            files.get_mut("tokenizer.json").unwrap()["model"]["merges"] = json!([merge]);
            match run(&files) {
                Err(e) if e.to_string().contains("model.merges") => {}
                other => missed.push(format!("merge {merge}: {other:?}")),
            }
        }
        // A token inside another and a token overlapping itself split alike; a suffix of one
        // starting another does not ("xy" then "yzw" in "xyzw": HF takes "xy", serving "yzw").
        for (texts, refused) in [
            (["xyz", "y"], false),
            (["xyx", "q"], false),
            (["xy", "yzw"], true),
            (["yzw", "xy"], true),
            (["xxy", "xyz"], true),
            (["xxxy", "xyz"], true),
        ] {
            let mut files = base.clone();
            let tokens = texts
                .iter()
                .zip(5..)
                .map(|(t, id)| json!({"id":id,"content":t,"special":false,"normalized":false}));
            files.get_mut("tokenizer.json").unwrap()["added_tokens"]
                .as_array_mut()
                .unwrap()
                .extend(tokens);
            match (run(&files), refused) {
                (Err(e), true) if e.to_string().contains("overlap") => {}
                (Ok(_), false) => {}
                (other, _) => missed.push(format!("added tokens {texts:?}: {other:?}")),
            }
        }
        for name in base.keys() {
            std::fs::remove_file(dir.join(name)).unwrap();
        }
        std::fs::remove_dir(&dir).unwrap();
        assert!(missed.is_empty(), "{}", missed.join("\n"));
    }
}
