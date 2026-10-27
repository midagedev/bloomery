//! Ordered text and prepared media spans in one embedding input.

use std::ops::Range;

use crate::request::{Error, wrong_type};
use serde_json::Value;

/// One ordered embedding input part, after its media has been prepared and sized.
pub enum InputPart<'a> {
    Text(&'a str),
    /// Caller-selected delimiters around prepared media rows.
    Media {
        begin: &'a str,
        end: &'a str,
        positions: usize,
        kind: &'a str,
    },
}

/// Token ids and the image spans that replace their filler rows, in part order.
pub struct Assembled {
    pub ids: Vec<u32>,
    pub spans: Vec<Range<usize>>,
}

/// Tokenize each part separately, with BOS and EOS around the whole input. Media spans are
/// bounded by `ctx` before allocation; the caller checks the total length, including text and EOS.
/// A media span is filled with `filler`; a span with none to fill it is refused by name.
pub fn assemble<'a>(
    parts: impl Iterator<Item = Result<InputPart<'a>, String>>,
    encode: impl Fn(&str) -> Vec<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    filler: Option<u32>,
    ctx: usize,
) -> Result<Assembled, String> {
    let mut parts = parts.peekable();
    if parts.peek().is_none() {
        return Err("Input content cannot be empty".into());
    }
    let mut ids = Vec::new();
    let mut spans = Vec::new();
    if let Some(bos) = bos {
        ids.push(bos);
    }
    for part in parts {
        let (begin, end, positions, kind) = match part? {
            InputPart::Text(text) => {
                ids.extend(encode(text));
                continue;
            }
            InputPart::Media {
                begin,
                end,
                positions,
                kind,
            } => (begin, end, positions, kind),
        };
        if positions == 0 {
            return Err(format!("an embedding {kind} has an empty span"));
        }
        ids.extend(encode(begin));
        let at = ids.len();
        let last = at
            .checked_add(positions)
            .filter(|&n| n <= ctx)
            .ok_or_else(|| format!("{kind} input exceeds the embedding context"))?;
        let filler = filler
            .ok_or_else(|| format!("an embedding {kind} span has no filler token id to fill it"))?;
        ids.resize(last, filler);
        spans.push(at..last);
        ids.extend(encode(end));
    }
    if let Some(eos) = eos {
        ids.push(eos);
    }
    if ids.is_empty() {
        return Err("Input content cannot be empty".into());
    }
    Ok(Assembled { ids, spans })
}

/// A JSON token array, each id in the vocabulary's range.
pub fn token_ids(n_vocab: usize, v: &Value) -> Result<Vec<u32>, Error> {
    let Value::Array(a) = v else {
        return Err(Error::Invalid(
            "tokens must be an array of token ids".into(),
        ));
    };
    a.iter()
        .map(|x| {
            x.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < n_vocab)
                .ok_or_else(|| Error::Invalid(format!("invalid token id {x}")))
        })
        .collect()
}

/// llama-server's mixed text and token-id input; only its first text element gets BOS and EOS.
pub fn prompt_ids(
    k: &str,
    v: &Value,
    encode: impl Fn(&str) -> Vec<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    n_vocab: usize,
) -> Result<Vec<u32>, Error> {
    let mut ids = Vec::new();
    let text = |ids: &mut Vec<u32>, s: &str, special: bool| {
        if special && let Some(bos) = bos {
            ids.push(bos);
        }
        ids.extend(encode(s));
        if special && let Some(eos) = eos {
            ids.push(eos);
        }
    };
    match v {
        Value::String(s) => text(&mut ids, s, true),
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                match x {
                    Value::String(s) => text(&mut ids, s, i == 0),
                    o if carries_media(o) => return Err(no_media()),
                    other => ids.extend(token_ids(n_vocab, &Value::Array(vec![other.clone()]))?),
                }
            }
        }
        other => {
            return Err(wrong_type(
                k,
                "a string or an array of strings and token ids",
                other,
            ));
        }
    }
    Ok(ids)
}

/// Whether a prompt object carries nonempty legacy multimodal data.
pub fn carries_media(v: &Value) -> bool {
    match v.get("multimodal_data") {
        None | Some(Value::Null) => false,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(_) => true,
    }
}

/// The named refusal for media outside an ordered prepared input.
pub fn no_media() -> Error {
    Error::Invalid(
        "Multimodal data provided, but model does not support multimodal requests.".into(),
    )
}
