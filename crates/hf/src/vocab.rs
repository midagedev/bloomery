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
//! way HF does ([`schema`]); any other key or value is refused by name. The ids HF's generate
//! stops at are written as `tokenizer.bloomery.eog_token_ids`. This nonempty id array opts
//! serving into explicit token types and ids, without name-based overrides or model defaults,
//! and into HF's split on the added tokens (leftmost, then longest).
//! Serving checks every imported spelling against HF ByteLevel's decoded bytes, including
//! added tokens. The caller checks its split, special ids, automatic BOS/EOS and stop sets.
//!
//! The pre-tokenizers the table allows are the rows of [`PRE`]: each is the pipeline
//! `tokenizer.json` spells for one name of `tokenizer.ggml.pre` (the name llama.cpp gives the
//! vocabulary by the hash of its output), and the row `tokenizer.json` carries is the one
//! written, whatever `pretokenize_regex` of `tokenizer_config.json` says (a regex no row spells is
//! refused) and whatever `ByteLevel` trims of an encoding's offsets. A pipeline no row spells is
//! refused by name, and a [`VocabSpec`] may name the row its model has. A row also says whether
//! its BPE takes a word that is itself a token whole, and `model.ignore_merges` must say the
//! same. For every row the stop id array also makes serving encode as `tokenizer.json` does: the
//! regex folds a contraction's case by Unicode, and BPE skips a byte the vocabulary lacks.

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

// The pipelines' parts as tokenizer.json spells them, one place each.
macro_rules! byte_level {
    ($space:literal, $trim:literal, $regex:literal) => {
        concat!(
            r#"{"type":"ByteLevel","add_prefix_space":"#,
            $space,
            r#","trim_offsets":"#,
            $trim,
            r#","use_regex":"#,
            $regex,
            "}"
        )
    };
}
macro_rules! qwen35_regex {
    () => {
        r#""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""#
    };
}
macro_rules! qwen2_regex {
    () => {
        r#""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""#
    };
}
macro_rules! llama3_regex {
    () => {
        r#""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""#
    };
}
macro_rules! minimax_m2_regex {
    () => {
        r#""[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""#
    };
}
macro_rules! tekken_regex {
    () => {
        r#""[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""#
    };
}
// One regex split, then ByteLevel without its own regex.
macro_rules! pre_tokenizer {
    ($regex:ident, $behavior:literal, $invert:literal, $trim:literal) => {
        concat!(
            r#"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"#,
            $regex!(),
            r#"},"behavior":""#,
            $behavior,
            r#"","invert":"#,
            $invert,
            "},",
            byte_level!(false, $trim, false),
            "]}"
        )
    };
}

/// The pre-tokenizers, each as `tokenizer.json` spells it, under its `tokenizer.ggml.pre` name,
/// and whether its BPE takes a word that is itself a token whole (`model.ignore_merges`, which
/// `tokenizer.json` must repeat: [`ignores_merges`]). `Removed` with `invert` keeps the regex's
/// matches and drops what lies between them, which `Isolated` keeps as pieces of their own; both
/// regexes match every codepoint, so no gap exists and the two split alike.
pub const PRE: [(&str, &str, bool); 5] = [
    (
        "qwen35",
        pre_tokenizer!(qwen35_regex, "Isolated", false, false),
        false,
    ),
    (
        "minimax-m2",
        pre_tokenizer!(minimax_m2_regex, "Removed", true, true),
        false,
    ),
    (
        "qwen2",
        pre_tokenizer!(qwen2_regex, "Isolated", false, false),
        false,
    ),
    (
        "lfm2",
        pre_tokenizer!(llama3_regex, "Isolated", false, true),
        true,
    ),
    // Nemotron-3.5's pipeline, under the name its GGUF twin states; the tokenizer crate runs the
    // name, and `tekken`, as the Tekken regex.
    (
        "pixtral",
        pre_tokenizer!(tekken_regex, "Isolated", false, true),
        true,
    ),
];

/// The names of [`PRE`], as the refusal of an unknown one lists them.
fn pre_names() -> String {
    PRE.map(|(name, _, _)| name).join(", ")
}

/// Whether the row `name` of [`PRE`] has its BPE take a word that is itself a token whole.
fn ignores_merges(name: &str) -> bool {
    PRE.iter()
        .find(|(row, _, _)| *row == name)
        .is_some_and(|&(_, _, ignore)| ignore)
}

/// `pre_tokenizer` with its `ByteLevel` parts' `trim_offsets` cleared. The flag trims the offsets an
/// encoding reports and moves no id; the keys written carry no offsets, so nothing downstream reads
/// it, and a pipeline that differs from a row only by it is that row.
fn ids_only(pre: &Json) -> Json {
    fn clear(part: &mut Json) {
        if part["type"] == "ByteLevel" && part.get("trim_offsets").is_some() {
            part["trim_offsets"] = Json::Bool(false);
        }
    }
    let mut pre = pre.clone();
    if pre["pretokenizers"].is_array() {
        pre["pretokenizers"]
            .as_array_mut()
            .into_iter()
            .flatten()
            .for_each(clear);
    } else {
        clear(&mut pre);
    }
    pre
}

/// Each row of [`PRE`] with its name, its pipeline as [`ids_only`] reads it and the regex it
/// splits by.
fn rows() -> impl Iterator<Item = (&'static str, Json, String)> {
    PRE.into_iter().map(|(name, spelled, _)| {
        let row: Json = serde_json::from_str(spelled).expect("a pre-tokenizer row is JSON");
        let regex = row["pretokenizers"][0]["pattern"]["Regex"]
            .as_str()
            .expect("a pre-tokenizer row splits by a regex")
            .to_string();
        (name, ids_only(&row), regex)
    })
}

/// The row of [`PRE`] a `tokenizer.json` pre-tokenizer is: its name and the regex it splits by.
/// A pipeline is a row when it spells the row's pretokenizers and options but for `ByteLevel`'s
/// `trim_offsets` ([`ids_only`]).
fn pre_row(pre: Option<&Json>) -> Result<(&'static str, String), Error> {
    let bad = |what: String| Error::Vocab(what);
    let pre = pre.ok_or_else(|| bad("tokenizer.json.pre_tokenizer is missing".into()))?;
    let own = ids_only(pre);
    rows()
        .find(|(_, row, _)| *row == own)
        .map(|(name, _, regex)| (name, regex))
        .ok_or_else(|| {
            bad(format!(
                "tokenizer.json.pre_tokenizer {pre} is unsupported; supported rows: {}",
                pre_names()
            ))
        })
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
    // The row it is, in `PRE`; the keys of `keys` take it from there.
    ("pre_tokenizer", Read),
    // ByteLevel without trim_offsets moves no id, only offsets, and as a post-processor its
    // `add_prefix_space` and `use_regex` act on no text (LFM3 spells it true, false, true); as a
    // decoder its options move nothing.
    (
        "post_processor",
        Only(concat!(
            "[null, ",
            byte_level!(false, false, false),
            ", ",
            byte_level!(true, false, true),
            "]"
        )),
    ),
    (
        "decoder",
        Need(concat!(
            "[",
            byte_level!(false, false, false),
            ",",
            byte_level!(true, true, true),
            "]"
        )),
    ),
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
    // The pre-tokenizer row's own value ([`PRE`]); an absent key is HF's default, false.
    ("ignore_merges", Only("[false, true]")),
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
    ("unk_token", Read),
    ("add_bos_token", Read),
    ("add_eos_token", Read),
    ("chat_template", Read),
    ("add_prefix_space", Only("[false]")),
    ("do_lower_case", Only("[false]")),
    ("split_special_tokens", Only("[false]")),
    ("clean_up_tokenization_spaces", Only("[false]")),
    // Serving decodes invalid UTF-8 as U+FFFD.
    ("errors", Only(r#"["replace"]"#)),
    // The regex of the pre-tokenizer row, when present.
    ("pretokenize_regex", Read),
    // Each builds the tokenizer from tokenizer.json; `TokenizersBackend` is
    // transformers' class for a tokenizer.json with no class of its own.
    (
        "tokenizer_class",
        Only(
            r#"["Qwen2Tokenizer", "Qwen2TokenizerFast", "PreTrainedTokenizerFast", "TokenizersBackend"]"#,
        ),
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
    // The tensors HF's model call takes; the ids are tokenizer.json's either way.
    (
        "model_input_names",
        Only(r#"[["input_ids", "attention_mask"]]"#),
    ),
    (
        "is_local",
        Ignore(
            Type::Bool,
            "where HF loaded the folder from; no id depends on it",
        ),
    ),
    (
        "local_files_only",
        Ignore(
            Type::Bool,
            "whether HF may reach the hub for the folder; no id depends on it",
        ),
    ),
    (
        "processor_class",
        Ignore(
            Type::String,
            "the multimodal processor's class; the text tokenizer is tokenizer_class",
        ),
    ),
    (
        "tool_parser_type",
        Ignore(
            Type::String,
            "the serving framework's tool-call parser; serving parses tool calls by the model's family",
        ),
    ),
    ("audio_bos_token", Ignore(Type::String, MEDIA)),
    ("audio_eos_token", Ignore(Type::String, MEDIA)),
    ("audio_token", Ignore(Type::String, MEDIA)),
    ("audio_out_token", Ignore(Type::String, MEDIA)),
    ("image_token", Ignore(Type::String, MEDIA)),
    ("video_token", Ignore(Type::String, MEDIA)),
    ("vision_bos_token", Ignore(Type::String, MEDIA)),
    ("vision_eos_token", Ignore(Type::String, MEDIA)),
    ("vision_start_token", Ignore(Type::String, MEDIA)),
    ("vision_end_token", Ignore(Type::String, MEDIA)),
    ("model_specific_special_tokens", Ignore(Type::Object, MEDIA)),
    (
        "response_template",
        Ignore(
            Type::Object,
            "transformers' patterns that split generated text into content, reasoning and tool \
             calls; the prompt's ids and rendering are the chat template's",
        ),
    ),
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
    ("repetition_penalty", Ignore(Type::Number, SAMPLING)),
    (
        "transformers_version",
        Ignore(
            Type::String,
            "the writer's library version; no id depends on it",
        ),
    ),
    (
        "_from_model_config",
        Ignore(
            Type::Bool,
            "whether HF made the file from the model's config; no id depends on it",
        ),
    ),
    (
        "output_attentions",
        Ignore(
            Type::Bool,
            "HF's forward returns the attentions beside an unchanged output",
        ),
    ),
    (
        "output_hidden_states",
        Ignore(
            Type::Bool,
            "HF's forward returns the hidden states beside an unchanged output",
        ),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
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

const SPECIAL: [(&str, &str); 5] = [
    ("bos", "bos_token_id"),
    ("eos", "eos_token_id"),
    ("unk", "unknown_token_id"),
    ("sep", "seperator_token_id"),
    ("pad", "padding_token_id"),
];

/// What a model's map tells the tokenizer reader: the vocabulary size and the special ids its
/// `config.json` names, and optionally the [`PRE`] row its model has.
#[derive(Debug)]
pub struct VocabSpec {
    size: u32,
    ids: [Option<Vec<u32>>; 5],
    pre: Option<&'static str>,
}

/// The name of [`VocabSpec`] before a folder's map held it.
pub type Config = VocabSpec;

impl VocabSpec {
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
            pre: None,
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

    /// The model's tokenizer has the [`PRE`] row `name`: [`keys`] refuses a `tokenizer.json`
    /// whose pre-tokenizer is another row. A name no row has is refused here.
    pub fn with_pre(mut self, name: &'static str) -> Result<Self, Error> {
        if !PRE.iter().any(|&(row, _, _)| row == name) {
            return Err(Error::Vocab(format!(
                "pre-tokenizer row {name:?} is not one of {}",
                pre_names()
            )));
        }
        self.pre = Some(name);
        Ok(self)
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
    spec: &VocabSpec,
) -> Result<Vocabulary<'a>, Error> {
    let bad = |what: String| Error::Vocab(what);
    let vocab = model["vocab"]
        .as_object()
        .ok_or_else(|| bad("tokenizer.json has no model.vocab".into()))?;
    let mut by_id: HashMap<u32, &str> = HashMap::new();
    let mut by_name: HashMap<&str, u32> = HashMap::new();
    for (t, id) in vocab {
        let id = spec.token_id(&format!("model.vocab[{t:?}] id"), id)?;
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
        let id = spec.token_id("added_tokens.id", &a["id"])?;
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
    // transformers writes the extra special tokens as an object of names to tokens, or as a list.
    match tc.get("extra_special_tokens") {
        None => {}
        Some(Json::Object(tokens)) => {
            declared.extend(tokens.values().map(|v| ("extra_special_tokens", v)));
        }
        Some(Json::Array(tokens)) => {
            declared.extend(tokens.iter().map(|v| ("extra_special_tokens", v)));
        }
        Some(_) => {
            return Err(bad(
                "extra_special_tokens is not an object or an array".into()
            ));
        }
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
    for id in 0..spec.size {
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
    spec: &VocabSpec,
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
            .map(|v| spec.token_ids(&field, v))
            .transpose()?;
        let id = match named {
            Some(t) => Some(
                added
                    .iter()
                    .find(|(_, (c, _))| *c == t)
                    .map(|(&id, _)| id)
                    .ok_or_else(|| bad(format!("{kind}_token {t:?} is no added token")))?,
            ),
            None => spec.ids[i]
                .as_ref()
                .or(generated.as_ref())
                .and_then(|ids| ids.first().copied()),
        };
        if *kind == "eos" {
            // HF's generate stops at any of these; serving at its end-of-generation ids.
            stop = generated
                .or_else(|| spec.ids[i].clone())
                .or(id.map(|id| vec![id]));
        } else if *kind == "pad" && named.is_some() {
            // gguf-py's SpecialVocab takes the named token and reads a config's id only for a
            // kind it has not set; no id and no serving path reads the pad id.
        } else if let Some(other) = [&spec.ids[i], &generated]
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
/// name or a value they do not allow; a pre-tokenizer no row of [`PRE`] spells or other than the
/// row `spec` names, a `pretokenize_regex` no row splits by, and an `ignore_merges` other than
/// the row's; a `tokenizer.json` without its vocabulary or merges, or with a merge that is not two tokens without a space whose
/// operands and joined token are all in its vocabulary; an added token whose id or content
/// conflicts; a token spelled as an unused id's `[PAD<id>]`; a vocabulary size
/// `config.json` does not give; a special token in `tokenizer_config.json` that is neither a
/// string nor an added token's object, or that no added token spells; a special token id that is
/// not an id or a list of them, lies outside the vocabulary, or is a BOS, pad, unknown or
/// separator id other than the one exported; an `add_bos_token` or `add_eos_token` that is not
/// boolean, and a `chat_template` that is not a string, each by name.
pub fn keys(dir: &Path, spec: &VocabSpec) -> Result<Keys, Error> {
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
    let (pre, regex) = pre_row(tok.get("pre_tokenizer"))?;
    if let Some(expected) = spec.pre.filter(|&expected| expected != pre) {
        return Err(bad(format!(
            "tokenizer.json.pre_tokenizer is the {pre:?} row, where this model's is {expected:?}"
        )));
    }
    // A construction argument of transformers' Qwen2Tokenizer: the pipeline a loaded folder runs is
    // `tokenizer.json`'s, so a regex of another row is not the folder's pipeline, and one no row
    // spells is a pipeline this converter cannot name.
    if let Some(v) = tc
        .get("pretokenize_regex")
        .filter(|&v| *v != regex && !rows().any(|(_, _, r)| *v == r))
    {
        return Err(bad(format!(
            "tokenizer_config.json.pretokenize_regex {v} is unsupported; the rows split by: {}",
            rows()
                .map(|(name, _, r)| format!("{name} {r:?}"))
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    let model = tok
        .get("model")
        .ok_or_else(|| bad("tokenizer.json has no model".into()))?;
    schema::check(model, "model", &[MODEL]).map_err(bad)?;
    let ignored = model["ignore_merges"].as_bool().unwrap_or(false);
    if ignored != ignores_merges(pre) {
        return Err(bad(format!(
            "tokenizer.json.model.ignore_merges is {ignored}, where the {pre:?} row's is {}",
            ignores_merges(pre)
        )));
    }
    let nfc = tok["normalizer"] == serde_json::json!({"type":"NFC"});
    let Vocabulary {
        tokens,
        types,
        merges,
        added,
    } = vocabulary(model, &tok, &tc, spec)?;
    let mut out = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String("gpt2".into()),
        ),
        ("tokenizer.ggml.pre".to_string(), Value::String(pre.into())),
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
    let (ids, stop) = special_ids(&tc, &generation, spec, &added)?;
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

    /// A folder of `files` of its own: named by the process and the calling test, which runs on a
    /// thread of that name.
    fn repo(files: &[(&str, &str)]) -> std::path::PathBuf {
        let test = std::thread::current()
            .name()
            .unwrap_or("main")
            .replace("::", "-");
        let dir = std::env::temp_dir().join(format!("hf-vocab-{}-{test}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in files {
            std::fs::write(dir.join(name), text).unwrap();
        }
        dir
    }

    fn get<'a>(keys: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
        keys.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A pad token the tokenizer names wins over a config's other pad id, as gguf-py's
    /// SpecialVocab takes it; a bos id that disagrees stays refused.
    #[test]
    fn a_named_pad_token_wins_over_a_config_s_pad_id() {
        let files = |tc: &'static str| {
            [
                (
                    "tokenizer.json",
                    r#"{"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
                "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
                "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},
                "added_tokens":[{"id":3,"content":"<|im_end|>","special":true}]}"#,
                ),
                ("tokenizer_config.json", tc),
                ("config.json", r#"{"text_config":{"vocab_size":4}}"#),
                (
                    "generation_config.json",
                    r#"{"bos_token_id":0,"pad_token_id":0}"#,
                ),
                ("chat_template.jinja", "{{ messages }}"),
            ]
        };
        let keys = self::keys(&repo(&files(
            r#"{"eos_token":"<|im_end|>","pad_token":"<|im_end|>","bos_token":null,"add_bos_token":false}"#,
        )))
        .unwrap();
        assert_eq!(
            get(&keys, "tokenizer.ggml.padding_token_id"),
            Some(&Value::U32(3))
        );
        let err = self::keys(&repo(&files(
            r#"{"eos_token":"<|im_end|>","bos_token":"<|im_end|>","add_bos_token":false}"#,
        )))
        .unwrap_err()
        .to_string();
        assert!(err.contains("bos_token_id 0 disagrees"), "{err}");
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
    /// that overlap accepted.
    #[test]
    fn every_tokenizer_key_accepts_its_values_and_refuses_others() {
        use serde_json::json;
        let byte_level = json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,
            "use_regex":false});
        let regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let pre = json!({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":regex},
            "behavior":"Isolated","invert":false}, byte_level]});
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
        // PIN(2026-10-28): `model.ignore_merges` true, an `extra_special_tokens` array and a
        // ByteLevel post-processor (true, false, true) were refused; the `lfm2` row has the first,
        // LFM3's `tokenizer_config.json` the second, and its `tokenizer.json` the third (the
        // post-processor's `add_prefix_space` and `use_regex` act on no text).
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
            ["tokenizer.json", "/model", "ignore_merges", false, "yes"],
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
            ["tokenizer_config.json", "", "extra_special_tokens", {"end":"<|im_end|>"}, "<|im_end|>"],
            ["tokenizer_config.json", "", "bos_token", "<|im_end|>", "missing"],
            ["tokenizer_config.json", "", "eos_token", {"content":"<|im_end|>","special":true}, 7],
            ["tokenizer_config.json", "", "pad_token", "<|im_end|>", {"content":"<|im_end|>","lstrip":true}],
            ["tokenizer_config.json", "", "sep_token", null, "missing"],
            ["tokenizer_config.json", "", "unk_token", "<|im_end|>", "missing"],
            ["tokenizer_config.json", "", "add_bos_token", false, "yes"],
            ["tokenizer_config.json", "", "add_eos_token", false, 1],
            ["tokenizer_config.json", "", "chat_template", "t", [{"template":"t"}]],
            ["tokenizer_config.json", "", "add_prefix_space", false, true],
            ["tokenizer_config.json", "", "do_lower_case", false, true],
            ["tokenizer_config.json", "", "split_special_tokens", false, true],
            ["tokenizer_config.json", "", "clean_up_tokenization_spaces", false, true],
            ["tokenizer_config.json", "", "errors", "replace", "ignore"],
            ["tokenizer_config.json", "", "pretokenize_regex", "$", "different"],
            ["tokenizer_config.json", "", "tokenizer_class", "PreTrainedTokenizerFast", "LlamaTokenizer"],
            ["tokenizer_config.json", "", "model_max_length", 262144, "long"],
            ["tokenizer_config.json", "", "backend", "tokenizers", "sentencepiece"],
            ["tokenizer_config.json", "", "model_input_names", ["input_ids", "attention_mask"], ["input_ids"]],
            ["tokenizer_config.json", "", "is_local", true, "yes"],
            ["tokenizer_config.json", "", "local_files_only", false, "yes"],
            ["tokenizer_config.json", "", "processor_class", "Qwen3VLProcessor", 1],
            ["tokenizer_config.json", "", "tool_parser_type", "qwen3_coder", 1],
            ["tokenizer_config.json", "", "audio_bos_token", "<|audio_start|>", 1],
            ["tokenizer_config.json", "", "audio_eos_token", "<|audio_end|>", 1],
            ["tokenizer_config.json", "", "audio_token", "<|audio_pad|>", 1],
            ["tokenizer_config.json", "", "audio_out_token", "<|audio_out|>", 1],
            ["tokenizer_config.json", "", "image_token", "<|image_pad|>", 1],
            ["tokenizer_config.json", "", "video_token", "<|video_pad|>", 1],
            ["tokenizer_config.json", "", "vision_bos_token", "<|vision_start|>", 1],
            ["tokenizer_config.json", "", "vision_eos_token", "<|vision_end|>", 1],
            ["tokenizer_config.json", "", "vision_start_token", "<|vision_start|>", 1],
            ["tokenizer_config.json", "", "vision_end_token", "<|vision_end|>", 1],
            ["tokenizer_config.json", "", "response_template", {"defaults":{"role":"assistant"}}, "x"],
            ["tokenizer_config.json", "", "model_specific_special_tokens", {"image_token":"<|image_pad|>"}, "x"],
            ["generation_config.json", "", "bos_token_id", 3, [3, 4]],
            ["generation_config.json", "", "eos_token_id", [3, 4], [3, "4"]],
            ["generation_config.json", "", "pad_token_id", 4, [4, 3]],
            ["generation_config.json", "", "do_sample", true, "yes"],
            ["generation_config.json", "", "temperature", 1.0, "hot"],
            ["generation_config.json", "", "top_k", 20, "20"],
            ["generation_config.json", "", "top_p", 0.95, [0.95]],
            ["generation_config.json", "", "repetition_penalty", 1.05, "high"],
            ["generation_config.json", "", "transformers_version", "4.46.1", 4],
            ["generation_config.json", "", "_from_model_config", true, "yes"],
            ["generation_config.json", "", "output_attentions", false, "no"],
            ["generation_config.json", "", "output_hidden_states", false, "no"],
            ["generation_config.json", "", "use_cache", true, 1]
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
        // Added tokens may overlap: serving splits on them as HF does (crates/tokenizer).
        for texts in [
            ["xyz", "y"],
            ["xyx", "q"],
            ["xy", "yzw"],
            ["yzw", "xy"],
            ["xxy", "xyz"],
            ["xxxy", "xyz"],
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
            if let Err(e) = run(&files) {
                missed.push(format!("added tokens {texts:?}: {e}"));
            }
        }
        for name in base.keys() {
            std::fs::remove_file(dir.join(name)).unwrap();
        }
        std::fs::remove_dir(&dir).unwrap();
        assert!(missed.is_empty(), "{}", missed.join("\n"));
    }

    /// MiniMax's tokenizer files as shipped (its pre-tokenizer, decoder, `tokenizer_class`,
    /// `unk_token` and `transformers_version`) over a vocabulary cut to a few tokens, among them
    /// special tokens that overlap.
    fn minimax_files() -> Vec<(&'static str, String)> {
        use serde_json::json;
        let regex = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let token = |id: u32, content: &str, special: bool| {
            json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,
                "normalized":false,"special":special})
        };
        let decoded = |content: &str, special: bool| {
            json!({"content":content,"lstrip":false,"normalized":false,"rstrip":false,
                "single_word":false,"special":special})
        };
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[token(5, "]!p~[", true), token(6, "]~b]", true), token(7, "[e~[", true),
                token(8, "<think>", false)],
            "normalizer":{"type":"NFC"},
            "pre_tokenizer":{"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":regex},"behavior":"Removed","invert":true},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
            "post_processor":null,
            "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"",
                "end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,
                "vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}});
        let config = json!({"add_prefix_space":false,
            "added_tokens_decoder":{"5":decoded("]!p~[", true),"6":decoded("]~b]", true),
                "7":decoded("[e~[", true),"8":decoded("<think>", false)},
            "bos_token":"]~b]","clean_up_tokenization_spaces":false,"eos_token":"[e~[",
            "pad_token":"]!p~[","model_max_length":40960000,"tokenizer_class":"PreTrainedTokenizerFast",
            "unk_token":"[e~["});
        let generation = json!({"bos_token_id":6,"do_sample":true,"eos_token_id":7,"temperature":1.0,
            "top_p":0.95,"transformers_version":"4.46.1"});
        vec![
            ("tokenizer.json", tokenizer.to_string()),
            ("tokenizer_config.json", config.to_string()),
            ("generation_config.json", generation.to_string()),
            (
                "config.json",
                json!({"model_type":"minimax_m3_vl","text_config":{"vocab_size":12}}).to_string(),
            ),
            ("chat_template.jinja", "{{ messages }}".to_string()),
        ]
    }

    fn minimax_repo(edit: impl FnOnce(&mut Json, &mut Json)) -> std::path::PathBuf {
        let mut files = minimax_files();
        let (mut tokenizer, mut config) = (
            serde_json::from_str(&files[0].1).unwrap(),
            serde_json::from_str(&files[1].1).unwrap(),
        );
        edit(&mut tokenizer, &mut config);
        files[0].1 = tokenizer.to_string();
        files[1].1 = config.to_string();
        repo(
            &files
                .iter()
                .map(|(n, t)| (*n, t.as_str()))
                .collect::<Vec<_>>(),
        )
    }

    fn spec_of(dir: &Path) -> VocabSpec {
        let config = crate::read_json(dir, "config.json", Error::Io, Error::Json).unwrap();
        VocabSpec::parse(&config.unwrap()).unwrap()
    }

    fn remove(dir: &Path) {
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A tokenizer whose pre-tokenizer is MiniMax's is written as the `minimax-m2` row, with the
    /// special ids by name (its `unk_token` is its eos), the stop id from `generation_config.json`
    /// and the chat template as the file has it. The row is the pre-tokenizer's, not a constant:
    /// the same files with Qwen's pre-tokenizer are `qwen35`.
    #[test]
    fn a_minimax_tokenizer_is_the_minimax_m2_row() {
        let dir = minimax_repo(|_, _| {});
        let keys = super::keys(&dir, &spec_of(&dir)).unwrap();
        let value = |key: &str| get(&keys.keys, key).cloned();
        assert_eq!(
            value("tokenizer.ggml.pre"),
            Some(Value::String("minimax-m2".into()))
        );
        for (key, id) in [
            ("bos_token_id", 6),
            ("eos_token_id", 7),
            ("unknown_token_id", 7),
        ] {
            assert_eq!(
                value(&format!("tokenizer.ggml.{key}")),
                Some(Value::U32(id)),
                "{key}"
            );
        }
        assert_eq!(
            value("tokenizer.ggml.padding_token_id"),
            Some(Value::U32(5))
        );
        assert_eq!(keys.stop, [7]);
        assert_eq!(keys.added, [5, 6, 7, 8]);
        assert_eq!(
            value("tokenizer.ggml.normalizer"),
            Some(Value::String("NFC".into()))
        );
        assert_eq!(
            value("tokenizer.chat_template"),
            Some(Value::String("{{ messages }}".into()))
        );
        let types = [1, 1, 1, 5, 5, 3, 3, 3, 4, 5, 5, 5].map(Value::I32);
        assert_eq!(
            value("tokenizer.ggml.token_type"),
            Some(Value::Array(types.to_vec()))
        );
        remove(&dir);

        let qwen = PRE[0].1;
        let dir = minimax_repo(|tokenizer, config| {
            tokenizer["pre_tokenizer"] = serde_json::from_str(qwen).unwrap();
            config.as_object_mut().unwrap().remove("unk_token");
        });
        let keys = super::keys(&dir, &spec_of(&dir)).unwrap();
        assert_eq!(
            get(&keys.keys, "tokenizer.ggml.pre"),
            Some(&Value::String("qwen35".into()))
        );
        remove(&dir);
    }

    /// A tokenizer whose pre-tokenizer is Qwen2's, which Qwen3-MoE keeps (a word takes letters
    /// alone, no mark), is the `qwen2` row; a spec that names the `qwen35` row refuses it by both
    /// names.
    #[test]
    fn a_qwen2_tokenizer_is_the_qwen2_row() {
        let qwen2 = PRE.iter().find(|r| r.0 == "qwen2").unwrap().1;
        let dir = minimax_repo(|tokenizer, config| {
            tokenizer["pre_tokenizer"] = serde_json::from_str(qwen2).unwrap();
            config.as_object_mut().unwrap().remove("unk_token");
        });
        let keys = super::keys(&dir, &spec_of(&dir)).unwrap();
        assert_eq!(
            get(&keys.keys, "tokenizer.ggml.pre"),
            Some(&Value::String("qwen2".into()))
        );
        super::keys(&dir, &spec_of(&dir).with_pre("qwen2").unwrap()).unwrap();
        let e = super::keys(&dir, &spec_of(&dir).with_pre("qwen35").unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("qwen2") && e.contains("qwen35") && e.contains("pre_tokenizer"),
            "{e}"
        );
        remove(&dir);
    }

    /// A tokenizer whose pre-tokenizer is Nemotron-3.5's (a `Split` on the Tekken regex, `Isolated`,
    /// then `ByteLevel` with `trim_offsets`) and whose `model.ignore_merges` is true is the
    /// `pixtral` row, the name llama.cpp gives that vocabulary; a spec that names another row
    /// refuses it by both names, and the row holds `ignore_merges` to true.
    ///
    /// FAIL-first: the row out of [`PRE`] (`tokenizer.json.pre_tokenizer ... is unsupported`), its
    /// flag false (`ignore_merges is true, where the "pixtral" row's is false`) or one character of
    /// its regex changed fails it.
    #[test]
    fn a_tekken_pipeline_is_the_pixtral_row() {
        let tekken = PRE.iter().find(|r| r.0 == "pixtral").unwrap().1;
        let edit = |tokenizer: &mut Json, config: &mut Json| {
            tokenizer["pre_tokenizer"] = serde_json::from_str(tekken).unwrap();
            tokenizer["model"]["ignore_merges"] = true.into();
            config.as_object_mut().unwrap().remove("unk_token");
        };
        let dir = minimax_repo(edit);
        let keys = super::keys(&dir, &spec_of(&dir)).unwrap();
        assert_eq!(
            get(&keys.keys, "tokenizer.ggml.pre"),
            Some(&Value::String("pixtral".into()))
        );
        super::keys(&dir, &spec_of(&dir).with_pre("pixtral").unwrap()).unwrap();
        let e = super::keys(&dir, &spec_of(&dir).with_pre("qwen35").unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("pixtral") && e.contains("qwen35") && e.contains("pre_tokenizer"),
            "{e}"
        );
        remove(&dir);

        let dir = minimax_repo(|tokenizer, config| {
            edit(tokenizer, config);
            tokenizer["model"]["ignore_merges"] = false.into();
        });
        let e = super::keys(&dir, &spec_of(&dir)).unwrap_err().to_string();
        assert!(
            e.contains("ignore_merges is false") && e.contains("\"pixtral\"") && e.contains("true"),
            "{e}"
        );
        remove(&dir);
    }

    /// The pre-tokenizer of mlx-community's Qwen3-30B-A3B-Instruct-2507-4bit at revision
    /// `e9675aa3ca5f900ccef55267914466d55ab325fa` (its `tokenizer.json`'s normalizer and pipeline,
    /// committed alone) is the `qwen2` row as the file spells it.
    #[test]
    fn the_real_qwen3_30b_pre_tokenizer_is_the_qwen2_row() {
        let path = format!(
            "{}/tests/fixtures/qwen3-30b-a3b-tokenizer-pipeline.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let pipeline: Json = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(pipeline["normalizer"]["type"], "NFC");
        assert_eq!(
            pre_row(Some(&pipeline["pre_tokenizer"])).unwrap().0,
            "qwen2"
        );
    }

    /// A pre-tokenizer no row spells is refused by name, one character of the regex off, a
    /// `Split` that keeps the gaps beside MiniMax's regex, and a lone `ByteLevel`; so is a
    /// `pretokenize_regex` no row splits by.
    #[test]
    fn a_pre_tokenizer_no_row_spells_is_refused() {
        let regex = |tokenizer: &mut Json| {
            tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].take()
        };
        for (what, edit) in [
            (
                "one regex character",
                Box::new(|t: &mut Json, _: &mut Json| {
                    let r = regex(t).as_str().unwrap().replace("{1,3}", "{1,4}");
                    t["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r.into();
                }) as Box<dyn Fn(&mut Json, &mut Json)>,
            ),
            (
                "the gaps kept",
                Box::new(|t: &mut Json, _: &mut Json| {
                    t["pre_tokenizer"]["pretokenizers"][0]["behavior"] = "Isolated".into();
                    t["pre_tokenizer"]["pretokenizers"][0]["invert"] = false.into();
                }),
            ),
            (
                "a lone ByteLevel",
                Box::new(|t: &mut Json, _: &mut Json| {
                    t["pre_tokenizer"] = t["pre_tokenizer"]["pretokenizers"][1].clone();
                }),
            ),
            (
                "a pretokenize_regex no row spells",
                Box::new(|_: &mut Json, c: &mut Json| {
                    let qwen: Json = serde_json::from_str(PRE[0].1).unwrap();
                    let r = qwen["pretokenizers"][0]["pattern"]["Regex"]
                        .as_str()
                        .unwrap();
                    c["pretokenize_regex"] = r.replace("\\p{N}|", "\\p{N}+|").into();
                }),
            ),
        ] {
            let dir = minimax_repo(|t, c| edit(t, c));
            let e = super::keys(&dir, &spec_of(&dir)).unwrap_err().to_string();
            assert!(
                e.contains("pre_tokenizer") || e.contains("pretokenize_regex"),
                "{what}: {e}"
            );
            assert!(e.contains("unsupported"), "{what}: {e}");
            remove(&dir);
        }
        let dir = minimax_repo(|t, c| {
            let own = t["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].clone();
            c["pretokenize_regex"] = own;
        });
        super::keys(&dir, &spec_of(&dir)).unwrap();
        remove(&dir);
    }

    /// A `pretokenize_regex` of another row leaves the row `tokenizer.json` spells: the pipeline a
    /// loaded folder runs is the file's, and transformers' loader and mlx-lm tokenize
    /// mlx-community's Qwen3.8-27B-4bit with the file's pipeline beside a `pretokenize_regex` of
    /// another row.
    #[test]
    fn a_pretokenize_regex_of_another_row_leaves_the_files_row() {
        // PIN(2026-10-28): was refused; the loaders' pipeline is `tokenizer.json`'s, not this key's.
        let qwen: Json = serde_json::from_str(PRE[0].1).unwrap();
        let dir = minimax_repo(|_, c| {
            c["pretokenize_regex"] = qwen["pretokenizers"][0]["pattern"]["Regex"].clone();
        });
        let keys = super::keys(&dir, &spec_of(&dir)).unwrap();
        assert_eq!(
            get(&keys.keys, "tokenizer.ggml.pre"),
            Some(&Value::String("minimax-m2".into()))
        );
        remove(&dir);
    }

    /// `ByteLevel`'s `trim_offsets` moves no id, so a pipeline that differs from a row by it is
    /// that row; its other options do move ids, and a pipeline that differs by one is refused.
    #[test]
    fn trim_offsets_moves_no_row_and_the_other_byte_level_options_do() {
        for (name, spelled, _) in PRE {
            let mut row: Json = serde_json::from_str(spelled).unwrap();
            let trim = row["pretokenizers"][1]["trim_offsets"].as_bool().unwrap();
            row["pretokenizers"][1]["trim_offsets"] = (!trim).into();
            assert_eq!(pre_row(Some(&row)).unwrap().0, name);
            for option in ["add_prefix_space", "use_regex"] {
                let mut moved: Json = serde_json::from_str(spelled).unwrap();
                moved["pretokenizers"][1][option] = true.into();
                assert!(pre_row(Some(&moved)).is_err(), "{name} {option}");
            }
        }
    }

    /// The pipelines of Qwen3.8-27B's `tokenizer.json` as two repos ship it, each with the
    /// `pretokenize_regex` of its `tokenizer_config.json` (the files' normalizer, pre-tokenizer,
    /// post-processor and decoder, committed alone), open as the row each file spells: mlx-community's
    /// re-save at `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` is the `qwen2` row (letters alone,
    /// no mark; `trim_offsets` set) beside the `qwen35` regex, the official repo's at
    /// `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` the `qwen35` row.
    #[test]
    fn the_real_qwen3_8_27b_folders_open_as_the_row_their_tokenizer_json_spells() {
        let path = format!(
            "{}/tests/fixtures/qwen3.8-27b-tokenizer-pipelines.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let pipelines: Json = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for (folder, row) in [
            (
                "mlx-community/Qwen3.8-27B-4bit@10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
                "qwen2",
            ),
            (
                "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "qwen35",
            ),
        ] {
            let p = &pipelines[folder];
            let dir = minimax_repo(|t, c| {
                for part in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
                    t[part] = p[part].clone();
                }
                c["pretokenize_regex"] = p["pretokenize_regex"].clone();
            });
            let keys =
                super::keys(&dir, &spec_of(&dir)).unwrap_or_else(|e| panic!("{folder}: {e}"));
            assert_eq!(
                get(&keys.keys, "tokenizer.ggml.pre"),
                Some(&Value::String(row.into())),
                "{folder}"
            );
            remove(&dir);
        }
    }

    /// A spec that names a row accepts that row and refuses another by both names; a name no row
    /// has is refused when the spec is made.
    #[test]
    fn a_spec_names_its_pre_row() {
        let dir = minimax_repo(|_, _| {});
        super::keys(&dir, &spec_of(&dir).with_pre("minimax-m2").unwrap()).unwrap();
        let e = super::keys(&dir, &spec_of(&dir).with_pre("qwen35").unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("minimax-m2") && e.contains("qwen35") && e.contains("pre_tokenizer"),
            "{e}"
        );
        let e = spec_of(&dir).with_pre("qwen36").unwrap_err().to_string();
        assert!(
            e.contains("qwen36") && e.contains("minimax-m2") && e.contains("qwen35"),
            "{e}"
        );
        remove(&dir);
    }

    /// The rows are distinct pipelines under distinct names, each one `tokenizer.json`'s
    /// `pre_tokenizer` accepts, and every one of the schema's allowed pre-tokenizer values is a row.
    #[test]
    fn the_pre_rows_are_distinct_and_accepted() {
        let names: std::collections::BTreeSet<_> = PRE.iter().map(|r| r.0).collect();
        let rows: Vec<Json> = PRE
            .iter()
            .map(|r| serde_json::from_str(r.1).unwrap())
            .collect();
        assert_eq!(names.len(), PRE.len());
        for (i, a) in rows.iter().enumerate() {
            assert!(
                rows[..i].iter().all(|b| ids_only(a) != ids_only(b)),
                "{}",
                PRE[i].0
            );
            assert_eq!(pre_row(Some(a)).unwrap().0, PRE[i].0);
        }
    }

    /// LFM3-1.7B-MLX-4bit's tokenizer files as the folder ships them
    /// (`lfm3-1.7b-tokenizer-files.json`: `tokenizer.json` but for the vocabulary and merges,
    /// which are cut to three tokens, then `tokenizer_config.json`, `generation_config.json` and
    /// `config.json` whole), after `edit`.
    fn lfm3_repo(edit: impl FnOnce(&mut Json)) -> std::path::PathBuf {
        let path = format!(
            "{}/tests/fixtures/lfm3-1.7b-tokenizer-files.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut files: Json = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        files["tokenizer.json"]["model"]["vocab"] = serde_json::json!({"a":0,"b":1,"ab":2});
        files["tokenizer.json"]["model"]["merges"] = serde_json::json!([["a", "b"]]);
        edit(&mut files);
        let texts: Vec<(&str, String)> = [
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "config.json",
        ]
        .into_iter()
        .map(|name| (name, files[name].to_string()))
        .collect();
        repo(
            &texts
                .iter()
                .map(|(n, t)| (*n, t.as_str()))
                .collect::<Vec<_>>(),
        )
    }

    /// LFM3's real files are the `lfm2` row, a llama3 split with `ignore_merges` on: the special
    /// ids by name (`<|startoftext|>`, `<|im_end|>`, `<|pad|>`), the stop id from the config's
    /// list, 140 added tokens of which the 16 spelled not special (`<think>`, the tool-call
    /// markers) are user-defined, and the ids past the added tokens unused. Mutants (each red
    /// at this line): the `lfm2` row out of [`PRE`] (`tokenizer.json.pre_tokenizer ... is
    /// unsupported`), `ignore_merges` back to `Only("[false]")`
    /// (`tokenizer.json.model.ignore_merges true is unsupported`), the post-processor list
    /// without `byte_level!(true, false, true)` (`tokenizer.json.post_processor ... is
    /// unsupported`), `extra_special_tokens` read as an object alone (`extra_special_tokens is
    /// not an object`), any of the six new keys' rows out of its table (`<file>.<key> is not a
    /// key this converter reads`).
    #[test]
    fn the_real_lfm3_tokenizer_files_open_as_the_lfm2_row() {
        let dir = lfm3_repo(|_| {});
        let keys = super::keys(&dir, &spec_of(&dir).with_pre("lfm2").unwrap()).unwrap();
        let value = |key: &str| get(&keys.keys, &format!("tokenizer.ggml.{key}")).cloned();
        assert_eq!(value("pre"), Some(Value::String("lfm2".into())));
        for (key, id) in [
            ("bos_token_id", 124_894),
            ("eos_token_id", 124_900),
            ("padding_token_id", 124_893),
        ] {
            assert_eq!(value(key), Some(Value::U32(id)), "{key}");
        }
        for key in ["unknown_token_id", "seperator_token_id", "normalizer"] {
            assert_eq!(value(key), None, "{key}");
        }
        assert_eq!(value("add_bos_token"), Some(Value::Bool(false)));
        assert_eq!(value("add_eos_token"), Some(Value::Bool(false)));
        assert_eq!(keys.stop, [124_900]);
        assert_eq!(keys.added, (124_893..=125_032).collect::<Vec<u32>>());
        let Some(Value::Array(types)) = value("token_type") else {
            panic!("no token_type");
        };
        assert_eq!(types.len(), 128_000);
        for (id, ty) in [
            (0, 1),
            (124_893, 3),
            (124_894, 3),
            (124_900, 3),
            (124_901, 4),
            (124_905, 4),
            (125_032, 3),
            (125_033, 5),
            (127_999, 5),
        ] {
            assert_eq!(types[id], Value::I32(ty), "id {id}");
        }
        let user_defined = types.iter().filter(|t| **t == Value::I32(4)).count();
        assert_eq!(user_defined, 16);
        remove(&dir);
    }

    /// The `lfm2` row holds `ignore_merges` to its own value, both ways: LFM3's pipeline with the
    /// flag off, a Qwen pipeline with it on, and a flag that is no boolean are each refused by
    /// the key's name. Mutant (red at the first line): the check dropped, so a pipeline whose
    /// merges serving would not skip (or would skip) reads as the row.
    #[test]
    fn ignore_merges_is_held_to_the_pre_tokenizer_row() {
        let off = lfm3_repo(|f| f["tokenizer.json"]["model"]["ignore_merges"] = false.into());
        let e = super::keys(&off, &spec_of(&off)).unwrap_err().to_string();
        assert!(
            e.contains("ignore_merges is false") && e.contains("\"lfm2\"") && e.contains("true"),
            "{e}"
        );
        remove(&off);
        let absent = lfm3_repo(|f| {
            f["tokenizer.json"]["model"]
                .as_object_mut()
                .unwrap()
                .remove("ignore_merges");
        });
        let e = super::keys(&absent, &spec_of(&absent))
            .unwrap_err()
            .to_string();
        assert!(e.contains("ignore_merges is false"), "{e}");
        remove(&absent);
        let qwen = PRE[0].1;
        let on = minimax_repo(|tokenizer, config| {
            tokenizer["pre_tokenizer"] = serde_json::from_str(qwen).unwrap();
            tokenizer["model"]["ignore_merges"] = true.into();
            config.as_object_mut().unwrap().remove("unk_token");
        });
        let e = super::keys(&on, &spec_of(&on)).unwrap_err().to_string();
        assert!(
            e.contains("ignore_merges is true") && e.contains("\"qwen35\"") && e.contains("false"),
            "{e}"
        );
        remove(&on);
        let word = lfm3_repo(|f| f["tokenizer.json"]["model"]["ignore_merges"] = "yes".into());
        let e = super::keys(&word, &spec_of(&word)).unwrap_err().to_string();
        assert!(e.contains("ignore_merges"), "{e}");
        remove(&word);
        for (name, _, ignore) in PRE {
            assert_eq!(ignores_merges(name), ignore, "{name}");
        }
        assert!(!ignores_merges("no such row"));
    }

    /// A `ByteLevel` post-processor that moves no id is read: absent, `false false false`, and
    /// LFM3's `true false true`; one with `trim_offsets` set, or another processor, is refused by
    /// the key's name. Mutant (red at the first line): the list without the third spelling.
    #[test]
    fn the_byte_level_post_processors_that_move_no_id_are_read() {
        let post = |add: bool, trim: bool, regex: bool| {
            serde_json::json!({"type":"ByteLevel","add_prefix_space":add,
                "trim_offsets":trim,"use_regex":regex})
        };
        for (what, value) in [
            ("true false true", post(true, false, true)),
            ("false false false", post(false, false, false)),
            ("absent", Json::Null),
        ] {
            let dir = lfm3_repo(|f| f["tokenizer.json"]["post_processor"] = value);
            super::keys(&dir, &spec_of(&dir)).unwrap_or_else(|e| panic!("{what}: {e}"));
            remove(&dir);
        }
        for (what, value) in [
            ("true true true", post(true, true, true)),
            ("false false true", post(false, false, true)),
            (
                "a template",
                serde_json::json!({"type":"TemplateProcessing"}),
            ),
        ] {
            let dir = lfm3_repo(|f| f["tokenizer.json"]["post_processor"] = value);
            let e = super::keys(&dir, &spec_of(&dir)).unwrap_err().to_string();
            assert!(
                e.contains("post_processor") && e.contains("unsupported"),
                "{what}: {e}"
            );
            remove(&dir);
        }
    }

    /// `extra_special_tokens` as a list reads like `additional_special_tokens`: each token must be
    /// a special added token, so a list of one that is not (`<think>`, added but not special) or
    /// that no added token spells is refused by name, and the empty list LFM3 ships is read.
    /// Mutant (red at the first line): the key read as an object alone.
    #[test]
    fn extra_special_tokens_may_be_a_list() {
        let set =
            |list: Json| lfm3_repo(|f| f["tokenizer_config.json"]["extra_special_tokens"] = list);
        for (what, list) in [
            ("empty", serde_json::json!([])),
            ("one", serde_json::json!(["<|im_end|>"])),
        ] {
            let dir = set(list);
            super::keys(&dir, &spec_of(&dir)).unwrap_or_else(|e| panic!("{what}: {e}"));
            remove(&dir);
        }
        for (what, list) in [
            ("not special", serde_json::json!(["<think>"])),
            ("not added", serde_json::json!(["<|nowhere|>"])),
            ("not a token", serde_json::json!([3])),
        ] {
            let dir = set(list);
            let e = super::keys(&dir, &spec_of(&dir)).unwrap_err().to_string();
            assert!(e.contains("extra_special_tokens"), "{what}: {e}");
            remove(&dir);
        }
    }

    /// Feed a key's value to a digest: a tag, then the value's bytes, a string and an array
    /// behind their length (u32 little endian), a number as its little endian bytes.
    fn feed(h: &mut sha2::Sha256, value: &Value) {
        use sha2::Digest;
        match value {
            Value::String(s) => {
                h.update(b"s");
                h.update(u32::try_from(s.len()).unwrap().to_le_bytes());
                h.update(s.as_bytes());
            }
            Value::U32(n) => {
                h.update(b"u");
                h.update(n.to_le_bytes());
            }
            Value::I32(n) => {
                h.update(b"i");
                h.update(n.to_le_bytes());
            }
            Value::Bool(b) => h.update([b'b', u8::from(*b)]),
            Value::Array(items) => {
                h.update(b"[");
                h.update(u32::try_from(items.len()).unwrap().to_le_bytes());
                for item in items {
                    feed(h, item);
                }
            }
            other => panic!("{other:?}"),
        }
    }

    /// The digest of a value: of one array key, or of every key in order with its name.
    fn digest(keys: &[(String, Value)], only: Option<&str>) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for (key, value) in keys.iter().filter(|(k, _)| only.is_none_or(|o| o == k)) {
            if only.is_none() {
                h.update(u32::try_from(key.len()).unwrap().to_le_bytes());
                h.update(key.as_bytes());
            }
            feed(&mut h, value);
        }
        format!("{:x}", h.finalize())
    }

    fn folder(var: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(
            std::env::var_os(var).unwrap_or_else(|| panic!("${var} names the folder")),
        )
    }

    fn derive(dir: &Path) -> Keys {
        super::keys(dir, &spec_of(dir)).unwrap()
    }

    /// The digest of every key `keys` derives from nvidia/Qwen3.8-Flash-Next-NVFP4's folder,
    /// from the reader as it was before the rows of [`PRE`].
    const QWEN38_KEYS: &str = "09dc38b98d7a8cb075c361de2671df46e4b8631b33394987b237d71921834f2b";

    /// Qwen3.8-Flash-Next's vocabulary is the one `keys` derived before the pre-tokenizer rows.
    #[test]
    #[ignore = "needs the nvidia/Qwen3.8-Flash-Next-NVFP4 tokenizer files and config.json at $BLOOMERY_QWEN38_HF"]
    fn hw_the_qwen38_vocabulary_is_unchanged() {
        let keys = derive(&folder("BLOOMERY_QWEN38_HF"));
        assert_eq!(
            get(&keys.keys, "tokenizer.ggml.pre"),
            Some(&Value::String("qwen35".into()))
        );
        assert_eq!(digest(&keys.keys, None), QWEN38_KEYS);
    }

    /// MiniMax-M3.1-Flash's folder derives MiniMax-M3's GGUF vocabulary (unsloth's UD-IQ4_XS first
    /// shard, whose header the digests were taken from): the `minimax-m2` row, 200,064 tokens with
    /// their types, 199,744 merges, and bos, eos and pad ids, but for id 16474. The GGUF spells it
    /// U+FFFD as a control token and makes it the unknown token; the folder's is HF's, `ï¿½`
    /// as a normal token, and its `unk_token` is the eos.
    #[test]
    #[ignore = "needs MiniMaxAI/MiniMax-M3.1-Flash's tokenizer files and config.json at $BLOOMERY_MM31_HF"]
    fn hw_the_minimax_folder_derives_the_m3_gguf_vocabulary() {
        let dir = folder("BLOOMERY_MM31_HF");
        let keys = derive(&dir);
        let value = |key: &str| get(&keys.keys, &format!("tokenizer.ggml.{key}"));
        assert_eq!(value("pre"), Some(&Value::String("minimax-m2".into())));
        assert_eq!(value("model"), Some(&Value::String("gpt2".into())));
        for (key, id) in [
            ("bos_token_id", 200_019),
            ("eos_token_id", 200_020),
            ("padding_token_id", 200_000),
            ("unknown_token_id", 200_020),
        ] {
            assert_eq!(value(key), Some(&Value::U32(id)), "{key}");
        }
        assert_eq!(keys.stop, [200_020]);
        for (key, n, pin) in [
            (
                "tokens",
                200_064,
                "f02f13cde2895d89e272a17827acd63bc96e6838832099e0192638d71c0fc613",
            ),
            (
                "token_type",
                200_064,
                "8b65e4928e9e3c81a52ff904e38bd650eac3aa0db04f7b03c26fd3b66ae8a8d0",
            ),
            (
                "merges",
                199_744,
                "551be7213251dff64497a95d99d1f82680e630e16ac78ebf0c80a4d78543072d",
            ),
        ] {
            let Some(Value::Array(a)) = value(key) else {
                panic!("{key} is no array");
            };
            assert_eq!(a.len(), n, "{key}");
            assert_eq!(
                digest(&keys.keys, Some(&format!("tokenizer.ggml.{key}"))),
                pin,
                "{key}"
            );
        }
        let template = std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap();
        assert_eq!(template.len(), 11_633);
        assert_eq!(
            get(&keys.keys, "tokenizer.chat_template"),
            Some(&Value::String(template))
        );
    }
}
