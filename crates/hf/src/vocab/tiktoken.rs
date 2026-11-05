//! A Hugging Face repo's tiktoken vocabulary (`tiktoken.model`, `tokenization_<module>.py`,
//! `tokenizer_config.json`, `config.json`, `generation_config.json`) as GGUF's `tokenizer.*` keys:
//! the keys llama.cpp's converter writes for the same folder, so a vocabulary read here equals the
//! header of a GGUF it converted, id for id.
//!
//! - `tiktoken.model` is one line per mergeable rank, `<base64 of the token's bytes> <rank>`. The
//!   ranks are `0..n` with no gap and no repeat, the tokens are distinct byte strings, and every
//!   byte has a token of its own.
//! - Ids `0..n` are those tokens, each spelled in GPT-2's byte-level alphabet (normal, 1).
//! - Ids `n..n+256` are the special tokens, all of them control (3): tiktoken splits a text on
//!   every one, so `tokenizer_config.json`'s `special` flag moves nothing. The name is the one
//!   `added_tokens_decoder` gives the id, else `<|reserved_token_{id}|>`.
//! - An id past those, up to `vocab_size`, is `[PAD<id>]`, unused (5).
//! - The merges are one per token of two bytes or more, in rank order: the two tokens its bytes
//!   fall into when BPE merges only pairs ranked below the token (the leftmost of the lowest rank
//!   first), each spelled in the byte-level alphabet and joined by one space. A token that does
//!   not fall into two is refused, where the converter drops its merge.
//! - The special ids are `config.json`'s `<kind>_token_id` (its text config's), the way gguf-py
//!   reads a folder with no `tokenizer.json`: a token `tokenizer_config.json` names is checked
//!   against them, not exported (Kimi's `[EOS]` is the document end, its config's EOS the turn's
//!   `<|end_of_msg|>`, and no unknown id is exported).
//! - The generation stop set is `generation_config.json`'s `eos_token_id`, else `config.json`'s,
//!   written as `tokenizer.ggml.eog_token_ids`, so serving stops where HF's generate does. The
//!   private `tokenizer.bloomery.eog_token_ids` is not written: it selects an HF reading of the
//!   pre-tokenizer, which the tiktoken rows do not have. The engine's tokenizer also marks any
//!   token spelled as one of its stop or end-of-turn names (`[EOT]`, `[EOS]`, `</s>`) control on
//!   load, so a base token with such a spelling would differ from this reader's types and be
//!   refused by the served-as-HF check; Kimi's base tokens have none.
//!
//! The pre-tokenizer is a row of [`PRE`]: the name `tokenizer.ggml.pre` gives it, and the lines
//! of `tokenization_<module>.py` that spell its regex, its special-token count and its special-token
//! names. The folder must carry those lines, so a regex no row spells is refused by name.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use gguf::Value;
use serde_json::Value as Json;

use super::{Error, GENERATION, Keys, SPECIAL, VocabSpec, chat_template, special_ids, token};
use crate::schema::{self, Fields, Rule::*, Type};

/// The special tokens after the mergeable ranks: `num_reserved_special_tokens` of the tokenizer.
const RESERVED: usize = 256;

/// One tiktoken pre-tokenizer: the name `tokenizer.ggml.pre` gives it, and where its source lives.
pub(super) struct Row {
    pub(super) name: &'static str,
    /// The folder's tokenizer module, `<module>.py` and `<module>.TikTokenTokenizer` in
    /// `tokenizer_config.json`'s `auto_map`.
    module: &'static str,
    /// Lines of that file the row pins, each block whole and in order (a line compared trimmed).
    pinned: [&'static str; 3],
}

/// Moonshot's `tokenization_kimi.py` (Kimi K2, K3 and K3.1 alike), which llama.cpp names `kimi-k2`.
pub(super) const PRE: [Row; 1] = [Row {
    name: "kimi-k2",
    module: "tokenization_kimi",
    pinned: [
        "num_reserved_special_tokens = 256",
        r##"pat_str = "|".join([
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ])"##,
        r##"self.special_tokens = {
            special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
            for i in range(num_base_tokens, num_base_tokens +
                           self.num_reserved_special_tokens)
        }"##,
    ],
}];

/// Whether `name` is a row of [`PRE`].
pub(super) fn is_pre(name: &str) -> bool {
    PRE.iter().any(|row| row.name == name)
}

/// The names of [`PRE`], as the refusal of an unknown one lists them.
pub(super) fn pre_names() -> Vec<&'static str> {
    PRE.iter().map(|row| row.name).collect()
}

/// `tokenizer_config.json` of a tiktoken folder. The special tokens are named by
/// `added_tokens_decoder`, as the tokenizer's own `__init__` maps them.
const TOKENIZER_CONFIG: &Fields = &[
    ("added_tokens_decoder", Read),
    ("additional_special_tokens", Read),
    // transformers writes an empty one; every special token is an `added_tokens_decoder` entry.
    ("extra_special_tokens", Only("[{}, []]")),
    ("bos_token", Read),
    ("eos_token", Read),
    ("pad_token", Read),
    ("sep_token", Read),
    ("unk_token", Read),
    // The tokenizer's `encode` adds no BOS or EOS whatever these say, so a true one is refused.
    ("add_bos_token", Only("[false]")),
    ("add_eos_token", Only("[false]")),
    ("chat_template", Read),
    // tiktoken's decode is the bytes of the ids; the key cleans nothing.
    ("clean_up_tokenization_spaces", Only("[false, true]")),
    (
        "model_max_length",
        Ignore(
            Type::Number,
            "HF warns past it and truncates only when asked; the context length is config.json's",
        ),
    ),
    ("tokenizer_class", Need(r#"["TikTokenTokenizer"]"#)),
    ("auto_map", Read),
];

/// The `tokenizer.*` keys of the tiktoken folder `dir`, as [`super::keys`] reads a folder: the
/// pre-tokenizer is the row `spec` names, or the row whose source the folder carries.
///
/// # Errors
///
/// A `tiktoken.model` that is missing, not UTF-8, or has a line that is not `<base64> <rank>`, a
/// rank that repeats or leaves a gap, a token that repeats, a byte with no token, or a token that
/// does not split into two lower-ranked tokens; a `vocab_size` below the ranks plus the special
/// tokens; an `added_tokens_decoder` entry outside the special ids, repeating a name or spelled
/// as a base token; a special token `tokenizer_config.json` names that no special id spells or
/// whose id `config.json` gives another; a tokenizer module that does not carry its row's pinned
/// lines; and every refusal of the shared tables, each by name.
pub(super) fn keys(dir: &Path, spec: &VocabSpec) -> Result<Keys, Error> {
    let bad = |what: String| Error::Vocab(what);
    let row = row_of(dir, spec)?;
    let read = |file: &str| crate::read_json(dir, file, Error::Io, Error::Json);
    let tc = read("tokenizer_config.json")?.unwrap_or_else(|| serde_json::json!({}));
    let generation = read("generation_config.json")?.unwrap_or_else(|| serde_json::json!({}));
    schema::check(&tc, "tokenizer_config.json", &[TOKENIZER_CONFIG]).map_err(bad)?;
    schema::check(&generation, "generation_config.json", &[GENERATION]).map_err(bad)?;
    let class = format!("{}.TikTokenTokenizer", row.module);
    if tc
        .get("auto_map")
        .is_some_and(|m| *m != serde_json::json!({"AutoTokenizer": [class, null]}))
    {
        return Err(bad(format!(
            "tokenizer_config.json.auto_map is not the {} tokenizer's {{\"AutoTokenizer\": [{class:?}, null]}}",
            row.name
        )));
    }

    let path = dir.join("tiktoken.model");
    let bytes = fs::read(&path).map_err(|e| Error::Io(path.display().to_string(), e))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| bad(format!("tiktoken.model: {e}")))?;
    let ranked = ranked(text)?;
    let base = ranked.len();
    let size = spec.size as usize;
    if size < base + RESERVED {
        return Err(bad(format!(
            "vocab_size {size} is below tiktoken.model's {base} ranks plus {RESERVED} special tokens"
        )));
    }
    let mut rank_of: HashMap<&[u8], usize> = HashMap::with_capacity(base);
    for (rank, token) in ranked.iter().enumerate() {
        if let Some(old) = rank_of.insert(token, rank) {
            return Err(bad(format!(
                "tiktoken.model: ranks {old} and {rank} are the same token bytes"
            )));
        }
    }
    if let Some(byte) = (0..=u8::MAX).find(|&b| !rank_of.contains_key([b].as_slice())) {
        return Err(bad(format!(
            "tiktoken.model has no token for byte 0x{byte:02x}"
        )));
    }

    let names = special_names(&tc, base)?;
    let spelling = Spelling::new();
    let taken = |name: &str| {
        spelling
            .unspell(name)
            .is_some_and(|b| rank_of.contains_key(b.as_slice()))
    };
    let ids = base..base + RESERVED;
    let named: HashMap<&str, usize> = names.iter().map(String::as_str).zip(ids.clone()).collect();
    if let Some(name) = names.iter().find(|n| taken(n)) {
        return Err(bad(format!(
            "special token {name:?} is also a token of tiktoken.model"
        )));
    }
    check_named(&tc, spec, &named)?;

    let mut tokens: Vec<Value> = Vec::with_capacity(size);
    let mut types: Vec<Value> = Vec::with_capacity(size);
    for token in &ranked {
        tokens.push(Value::String(spelling.spell(token)));
        types.push(Value::I32(1));
    }
    for name in &names {
        tokens.push(Value::String(name.clone()));
        types.push(Value::I32(3));
    }
    for id in base + RESERVED..size {
        let pad = format!("[PAD{id}]");
        if named.contains_key(pad.as_str()) || taken(&pad) {
            return Err(bad(format!(
                "token {pad:?} spells the filler of unused id {id}"
            )));
        }
        tokens.push(Value::String(pad));
        types.push(Value::I32(5));
    }
    let merges = merges(&ranked, &rank_of, &spelling)?;

    let mut out = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::String(row.name.into()),
        ),
        ("tokenizer.ggml.tokens".to_string(), Value::Array(tokens)),
        ("tokenizer.ggml.token_type".to_string(), Value::Array(types)),
        ("tokenizer.ggml.merges".to_string(), Value::Array(merges)),
    ];
    // The ids come from the configs alone; no token of `tokenizer_config.json` picks one.
    let (special, stop) = special_ids(
        &serde_json::json!({}),
        &generation,
        spec,
        &HashMap::new(),
        false,
    )?;
    out.extend(special);
    if let Some(template) = chat_template(dir, &tc)? {
        out.push((
            "tokenizer.chat_template".to_string(),
            Value::String(template),
        ));
    }
    let mut stop = stop
        .filter(|ids| !ids.is_empty())
        .ok_or_else(|| bad("eos_token_id defines no generation stop tokens".into()))?;
    stop.sort_unstable();
    stop.dedup();
    out.push((
        "tokenizer.ggml.eog_token_ids".into(),
        Value::Array(stop.iter().copied().map(Value::U32).collect()),
    ));
    let added = ids
        .map(|id| u32::try_from(id).map_err(|_| bad(format!("special id {id} is past u32"))))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Keys {
        keys: out,
        added,
        stop,
    })
}

/// The row of [`PRE`] the folder's tokenizer module carries: the one `spec` names, else the first
/// whose pinned lines the folder's module holds.
fn row_of(dir: &Path, spec: &VocabSpec) -> Result<&'static Row, Error> {
    let mut rows = PRE
        .iter()
        .filter(|row| spec.pre.is_none_or(|name| name == row.name))
        .peekable();
    if rows.peek().is_none() {
        return Err(Error::Vocab(format!(
            "the folder has tiktoken.model and no tokenizer.json, where this model's pre-tokenizer row {:?} is not one of the tiktoken rows: {}",
            spec.pre,
            pre_names().join(", ")
        )));
    }
    let mut first = None;
    for row in rows {
        match carries(dir, row) {
            Ok(()) => return Ok(row),
            Err(e) => first = first.or(Some(e)),
        }
    }
    Err(first.expect("a nonempty row list leaves one refusal"))
}

/// Whether the folder's `<module>.py` holds every pinned block of `row`.
fn carries(dir: &Path, row: &Row) -> Result<(), Error> {
    let file = format!("{}.py", row.module);
    let path = dir.join(&file);
    let source = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::Vocab(format!(
                "no {file}: it spells the {} pre-tokenizer's regex and special tokens",
                row.name
            )));
        }
        Err(e) => return Err(Error::Io(path.display().to_string(), e)),
    };
    let trimmed = |text: &str| -> Vec<String> {
        text.lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    };
    let have = trimmed(&source);
    for block in row.pinned {
        let want = trimmed(block);
        if !have.windows(want.len()).any(|at| at == want) {
            return Err(Error::Vocab(format!(
                "{file} does not carry the {} pre-tokenizer's source, which the folder's pipeline must spell as this: {block}",
                row.name
            )));
        }
    }
    Ok(())
}

/// `tiktoken.model`'s tokens by rank. A line is `<base64> <rank>`; the file's lines are its ranks,
/// so a rank past the last line leaves a gap and a rank twice repeats.
fn ranked(text: &str) -> Result<Vec<Vec<u8>>, Error> {
    let bad = |what: String| Error::Vocab(format!("tiktoken.model {what}"));
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    if lines.is_empty() {
        return Err(bad("has no lines".into()));
    }
    let mut by_rank: Vec<Option<(usize, Vec<u8>)>> = vec![None; lines.len()];
    for (i, line) in lines.iter().enumerate() {
        let at = i + 1;
        let (token, rank) = line
            .split_once(' ')
            .ok_or_else(|| bad(format!("line {at} is not `<base64 token> <rank>`")))?;
        let token = base64_decode(token)
            .ok_or_else(|| bad(format!("line {at}: {token:?} is not the base64 of a token")))?;
        let rank = rank
            .bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| rank.parse::<usize>().ok())
            .flatten()
            .filter(|_| rank == "0" || !rank.starts_with('0'))
            .ok_or_else(|| bad(format!("line {at}: {rank:?} is not a rank")))?;
        let slot = by_rank.get_mut(rank).ok_or_else(|| {
            bad(format!(
                "line {at}: rank {rank} leaves a gap in 0..{} (one rank per line)",
                lines.len()
            ))
        })?;
        if let Some((old, _)) = slot {
            return Err(bad(format!("lines {old} and {at} both have rank {rank}")));
        }
        *slot = Some((at, token));
    }
    by_rank
        .into_iter()
        .enumerate()
        .map(|(rank, slot)| {
            slot.map(|(_, token)| token)
                .ok_or_else(|| bad(format!("has no rank {rank}")))
        })
        .collect()
}

/// RFC 4648's base64 with padding, and nothing it does not spell: no whitespace, no characters
/// outside the alphabet, padding only at the end, and zero bits where the padding drops some.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let sextet = |c: u8| match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    };
    let (quads, _) = bytes.as_chunks::<4>();
    let last = quads.len() - 1;
    let mut out = Vec::with_capacity(quads.len() * 3);
    for (i, quad) in quads.iter().enumerate() {
        let pad = if i == last {
            quad.iter().rev().take_while(|&&c| c == b'=').count()
        } else {
            0
        };
        if pad > 2 {
            return None;
        }
        let mut n = 0u32;
        for &c in &quad[..4 - pad] {
            n = n << 6 | sextet(c)?;
        }
        n <<= 6 * pad;
        if n & ((1 << (8 * pad)) - 1) != 0 {
            return None;
        }
        out.extend_from_slice(&n.to_be_bytes()[1..4 - pad]);
    }
    Some(out)
}

/// GPT-2's byte-level alphabet, the tokenizer crate's, as the strings of a vocabulary spell it.
struct Spelling {
    glyph: [char; 256],
    byte: [Option<u8>; 324],
}

impl Spelling {
    fn new() -> Self {
        let table = tokenizer::byte_to_cpt_table();
        Self {
            glyph: table
                .map(|cpt| char::from_u32(cpt).expect("the byte alphabet holds scalar values")),
            byte: tokenizer::byte_decoder(),
        }
    }

    fn spell(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.glyph[usize::from(b)]).collect()
    }

    /// The bytes a spelling stands for, or `None` when a character is outside the alphabet.
    fn unspell(&self, text: &str) -> Option<Vec<u8>> {
        text.chars()
            .map(|c| self.byte.get(c as usize).copied().flatten())
            .collect()
    }
}

/// The merges of the tokens `ranked` (rank order), as the converter's `bpe` finds them: the two
/// parts a token's bytes make when only pairs ranked below the token are merged, the leftmost of the
/// lowest rank first. A part is a slice of the token, so a pair is looked up as one slice.
fn merges(
    ranked: &[Vec<u8>],
    rank_of: &HashMap<&[u8], usize>,
    spelling: &Spelling,
) -> Result<Vec<Value>, Error> {
    let mut out = Vec::with_capacity(ranked.len());
    for (rank, token) in ranked.iter().enumerate().filter(|(_, t)| t.len() > 1) {
        // Part `i` is `token[cuts[i]..cuts[i + 1]]`.
        let mut cuts: Vec<usize> = (0..=token.len()).collect();
        loop {
            let mut lowest: Option<(usize, usize)> = None;
            for i in 0..cuts.len() - 2 {
                if let Some(&r) = rank_of.get(&token[cuts[i]..cuts[i + 2]])
                    && lowest.is_none_or(|(l, _)| r < l)
                {
                    lowest = Some((r, i));
                }
            }
            match lowest {
                Some((r, i)) if r < rank => {
                    cuts.remove(i + 1);
                }
                _ => break,
            }
        }
        if cuts.len() != 3 {
            return Err(Error::Vocab(format!(
                "tiktoken.model: token {:?} of rank {rank} falls into {} tokens of lower rank, not two",
                spelling.spell(token),
                cuts.len() - 1
            )));
        }
        out.push(Value::String(format!(
            "{} {}",
            spelling.spell(&token[..cuts[1]]),
            spelling.spell(&token[cuts[1]..])
        )));
    }
    Ok(out)
}

/// The names of the special ids `base..base + RESERVED`: `added_tokens_decoder`'s content, else
/// `<|reserved_token_{id}|>`. An entry outside the ids, empty, or spelled as another's is refused.
fn special_names(tc: &Json, base: usize) -> Result<Vec<String>, Error> {
    let bad = |what: String| Error::Vocab(what);
    let mut names: Vec<String> = (base..base + RESERVED)
        .map(|id| format!("<|reserved_token_{id}|>"))
        .collect();
    let mut named = HashSet::new();
    if let Some(decoder) = tc.get("added_tokens_decoder") {
        let decoder = decoder
            .as_object()
            .ok_or_else(|| bad("added_tokens_decoder is not an object".into()))?;
        for (id, t) in decoder {
            let path = format!("added_tokens_decoder[{id}]");
            token(t, &path, false, false)?;
            let slot = id
                .parse::<usize>()
                .ok()
                .and_then(|id| id.checked_sub(base))
                .and_then(|i| names.get_mut(i))
                .ok_or_else(|| {
                    bad(format!(
                        "{path} is outside the special ids {base}..{}",
                        base + RESERVED
                    ))
                })?;
            let content = t["content"]
                .as_str()
                .filter(|c| !c.is_empty())
                .ok_or_else(|| bad(format!("{path} has no content")))?;
            if !named.insert(content) {
                return Err(bad(format!(
                    "{path}: {content:?} names another special id too"
                )));
            }
            *slot = content.to_string();
        }
    }
    let all: HashSet<&String> = names.iter().collect();
    if all.len() != names.len() {
        return Err(bad(
            "an added_tokens_decoder name is another special id's reserved name".into(),
        ));
    }
    Ok(names)
}

/// The special tokens `tokenizer_config.json` names (`<kind>_token`, `additional_special_tokens`)
/// must each be a special id's name; a kind but EOS must also be the id `config.json` gives it.
fn check_named(tc: &Json, spec: &VocabSpec, named: &HashMap<&str, usize>) -> Result<(), Error> {
    let bad = |what: String| Error::Vocab(what);
    let content = |v: &Json, path: &str| -> Result<Option<String>, Error> {
        match v {
            Json::Null => Ok(None),
            Json::String(s) => Ok(Some(s.clone())),
            Json::Object(_) => {
                token(v, path, false, false)?;
                v["content"]
                    .as_str()
                    .map(|c| Some(c.to_string()))
                    .ok_or_else(|| bad(format!("{path} {v} is not a token or an added token")))
            }
            _ => Err(bad(format!("{path} {v} is not a token or an added token"))),
        }
    };
    for (i, (kind, _)) in SPECIAL.iter().enumerate() {
        let path = format!("{kind}_token");
        let Some(name) = content(&tc[&path], &path)? else {
            continue;
        };
        let id = *named
            .get(name.as_str())
            .ok_or_else(|| bad(format!("{path} {name:?} is no special token")))?;
        if let Some(&other) = spec.ids[i].as_ref().and_then(|ids| ids.first())
            && *kind != "eos"
            && usize::try_from(other).ok() != Some(id)
        {
            return Err(bad(format!(
                "{path} {name:?} is id {id}, where config.json's {kind}_token_id is {other}"
            )));
        }
    }
    if let Some(extra) = tc.get("additional_special_tokens") {
        let extra = extra
            .as_array()
            .ok_or_else(|| bad("additional_special_tokens is not an array".into()))?;
        for t in extra {
            let name = content(t, "additional_special_tokens")?;
            if !name.is_some_and(|n| named.contains_key(n.as_str())) {
                return Err(bad(format!(
                    "additional_special_tokens {t} is no special token"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
