//! The vocabulary as the reference loads it (`llama_vocab::impl::load`):
//! token texts and attributes, the special ids, the attribute overrides it
//! applies by token text, and the special-token list the partition walks.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use gguf::Value;

use crate::Error;
use crate::pretok::Pre;

/// Token attributes, the reference's `llama_token_attr` bits.
pub mod attr {
    pub const UNKNOWN: u32 = 1 << 0;
    pub const UNUSED: u32 = 1 << 1;
    pub const NORMAL: u32 = 1 << 2;
    pub const CONTROL: u32 = 1 << 3;
    pub const USER_DEFINED: u32 = 1 << 4;
    pub const BYTE: u32 = 1 << 5;
}

/// `tokenizer.ggml.token_type` -> attribute.
fn attr_of_type(t: i64) -> u32 {
    match t {
        1 => attr::NORMAL,
        2 => attr::UNKNOWN,
        3 => attr::CONTROL,
        4 => attr::USER_DEFINED,
        5 => attr::UNUSED,
        6 => attr::BYTE,
        _ => 0,
    }
}

/// Token texts the reference recognises by name, per role, in its order.
const EOT_NAMES: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "<｜end▁of▁sentence｜>",
    "<end_of_utterance>",
];
const EOM_NAMES: &[&str] = &["<|eom_id|>"];
const FIM_PRE_NAMES: &[&str] = &[
    "<|fim_prefix|>",
    "<fim-prefix>",
    "<fim_prefix>",
    "<｜fim▁begin｜>",
    "<PRE>",
    "▁<PRE>",
    "<|code_prefix|>",
    "<|prefix|>",
];
const FIM_SUF_NAMES: &[&str] = &[
    "<|fim_suffix|>",
    "<fim-suffix>",
    "<fim_suffix>",
    "<｜fim▁hole｜>",
    "<SUF>",
    "▁<SUF>",
    "<|code_suffix|>",
    "<|suffix|>",
];
const FIM_MID_NAMES: &[&str] = &[
    "<|fim_middle|>",
    "<fim-middle>",
    "<fim_middle>",
    "<｜fim▁end｜>",
    "<MID>",
    "▁<MID>",
    "<|code_middle|>",
    "<|middle|>",
];
const FIM_PAD_NAMES: &[&str] = &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"];
const FIM_REP_NAMES: &[&str] = &[
    "<|fim_repo|>",
    "<|repo_name|>",
    "<fim-repo>",
    "<REPO>",
    "<reponame>",
];
const FIM_SEP_NAMES: &[&str] = &["<|file_sep|>"];
const EOG_NAMES: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
];
const ALWAYS_RENDERED: &[&str] = &["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"];

/// Special token ids; `None` is the reference's `LLAMA_TOKEN_NULL`.
#[derive(Clone, Debug, Default)]
pub struct Specials {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub eom: Option<u32>,
    pub unk: Option<u32>,
    pub sep: Option<u32>,
    pub pad: Option<u32>,
    pub mask: Option<u32>,
    pub fim_pre: Option<u32>,
    pub fim_suf: Option<u32>,
    pub fim_mid: Option<u32>,
    pub fim_pad: Option<u32>,
    pub fim_rep: Option<u32>,
    pub fim_sep: Option<u32>,
}

pub(crate) struct Vocab {
    pub(crate) texts: Vec<String>,
    pub(crate) attrs: Vec<u32>,
    pub(crate) token_to_id: HashMap<String, u32>,
    pub(crate) merges: Vec<String>,
    pub(crate) pre: String,
    /// The pre-tokenizer `pre` names.
    pub(crate) pretok: Pre,
    pub(crate) specials: Specials,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    pub(crate) eog: Vec<u32>,
    /// Roles the reference fills from the first matching token in hash-map
    /// order, where more than one candidate text is in this vocabulary — the
    /// reference's pick is then unspecified.
    pub(crate) ambiguous_roles: Vec<&'static str>,
    /// The metadata this load read, `(key, shown value)`, in reading order.
    pub(crate) keys_read: Vec<(String, String)>,
}

struct Meta<'a> {
    inv: &'a gguf::Inventory,
    read: Vec<(String, String)>,
}

impl<'a> Meta<'a> {
    fn get(&mut self, key: &str) -> Option<&'a Value> {
        let v = self.inv.value(key);
        let shown = match v {
            None => "(absent)".to_string(),
            Some(Value::Array(a)) => format!("array of {}", a.len()),
            Some(Value::String(s)) if s.len() > 60 => format!("string of {} bytes", s.len()),
            Some(Value::String(s)) => format!("{s:?}"),
            Some(other) => format!("{other:?}"),
        };
        self.read.push((key.to_string(), shown));
        v
    }

    fn str(&mut self, key: &'static str) -> Result<Option<&'a str>, Error> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_str().map(Some).ok_or(Error::WrongType(key)),
        }
    }

    fn strings(&mut self, key: &'static str) -> Result<Option<Vec<String>>, Error> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_str().map(str::to_string).ok_or(Error::WrongType(key)))
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
            Some(_) => Err(Error::WrongType(key)),
        }
    }

    fn ints(&mut self, key: &'static str) -> Result<Option<Vec<i64>>, Error> {
        let int = |v: &Value| match *v {
            Value::I8(x) => Some(i64::from(x)),
            Value::I16(x) => Some(i64::from(x)),
            Value::I32(x) => Some(i64::from(x)),
            Value::I64(x) => Some(x),
            _ => v.as_u64().and_then(|x| i64::try_from(x).ok()),
        };
        match self.get(key) {
            None => Ok(None),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| int(v).ok_or(Error::WrongType(key)))
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
            Some(_) => Err(Error::WrongType(key)),
        }
    }

    fn id(&mut self, key: &'static str) -> Result<Option<u64>, Error> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_unsigned().map(Some).ok_or(Error::WrongType(key)),
        }
    }

    fn flag(&mut self, key: &'static str) -> Result<Option<bool>, Error> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_bool().map(Some).ok_or(Error::WrongType(key)),
        }
    }
}

impl Vocab {
    pub(crate) fn load(path: &Path) -> Result<Vocab, Error> {
        let inv =
            gguf::inventory_of(path).map_err(|e| Error::Gguf(path.display().to_string(), e))?;
        let mut m = Meta {
            inv: &inv,
            read: Vec::new(),
        };

        let model = m
            .str("tokenizer.ggml.model")?
            .ok_or(Error::Missing("tokenizer.ggml.model"))?;
        if model != "gpt2" {
            return Err(Error::UnsupportedModel(model.to_string()));
        }
        // The names `Pre` runs share the rest of the reference's per-name
        // flags: byte-level encoding, merges always applied, spaces not
        // cleaned, no BOS unless the file asks for it.
        let pre = m.str("tokenizer.ggml.pre")?.unwrap_or("");
        let Some(pretok) = Pre::of(pre) else {
            return Err(Error::UnsupportedPre(pre.to_string()));
        };
        let merges = m
            .strings("tokenizer.ggml.merges")?
            .ok_or(Error::Missing("tokenizer.ggml.merges"))?;
        let mut texts = m
            .strings("tokenizer.ggml.tokens")?
            .ok_or(Error::Missing("tokenizer.ggml.tokens"))?;
        let types = m.ints("tokenizer.ggml.token_type")?;
        if let Some(t) = &types
            && t.len() < texts.len()
        {
            return Err(Error::TokenTypesShort {
                types: t.len(),
                tokens: texts.len(),
            });
        }

        let n = texts.len();
        let mut token_to_id = HashMap::with_capacity(n);
        let mut attrs = Vec::with_capacity(n);
        for (i, text) in texts.iter_mut().enumerate() {
            if text.is_empty() {
                *text = format!("[EMPTY_{i}]");
            }
            let id = u32::try_from(i).map_err(|_| Error::TooManyTokens(n))?;
            if token_to_id.insert(text.clone(), id).is_some() {
                return Err(Error::DuplicateToken(text.clone()));
            }
            attrs.push(types.as_ref().map_or(attr::NORMAL, |t| attr_of_type(t[i])));
        }

        // Defaults for a gpt2 vocabulary, then the file's ids where in range.
        let mut sp = Specials {
            bos: Some(11),
            eos: Some(11),
            ..Specials::default()
        };
        {
            type Slot = fn(&mut Specials) -> &mut Option<u32>;
            let slots: [(&'static str, Slot); 17] = [
                ("tokenizer.ggml.bos_token_id", |s| &mut s.bos),
                ("tokenizer.ggml.eos_token_id", |s| &mut s.eos),
                ("tokenizer.ggml.eot_token_id", |s| &mut s.eot),
                ("tokenizer.ggml.eom_token_id", |s| &mut s.eom),
                ("tokenizer.ggml.unknown_token_id", |s| &mut s.unk),
                ("tokenizer.ggml.seperator_token_id", |s| &mut s.sep),
                ("tokenizer.ggml.padding_token_id", |s| &mut s.pad),
                ("tokenizer.ggml.mask_token_id", |s| &mut s.mask),
                ("tokenizer.ggml.fim_pre_token_id", |s| &mut s.fim_pre),
                ("tokenizer.ggml.fim_suf_token_id", |s| &mut s.fim_suf),
                ("tokenizer.ggml.fim_mid_token_id", |s| &mut s.fim_mid),
                ("tokenizer.ggml.fim_pad_token_id", |s| &mut s.fim_pad),
                ("tokenizer.ggml.fim_rep_token_id", |s| &mut s.fim_rep),
                ("tokenizer.ggml.fim_sep_token_id", |s| &mut s.fim_sep),
                ("tokenizer.ggml.prefix_token_id", |s| &mut s.fim_pre),
                ("tokenizer.ggml.suffix_token_id", |s| &mut s.fim_suf),
                ("tokenizer.ggml.middle_token_id", |s| &mut s.fim_mid),
            ];
            for (key, slot) in slots {
                if let Some(id) = m.id(key)?
                    && let Ok(id) = u32::try_from(id)
                    && (id as usize) < n
                {
                    *slot(&mut sp) = Some(id);
                }
            }
        }
        let add_bos = m.flag("tokenizer.ggml.add_bos_token")?.unwrap_or(false);
        let add_eos = m.flag("tokenizer.ggml.add_eos_token")?.unwrap_or(false);

        let mut v = Vocab {
            texts,
            attrs,
            token_to_id,
            merges,
            pre: pre.to_string(),
            pretok,
            specials: sp,
            add_bos,
            add_eos,
            eog: Vec::new(),
            ambiguous_roles: Vec::new(),
            keys_read: std::mem::take(&mut m.read),
        };
        v.apply_overrides();
        Ok(v)
    }

    fn id_of(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    /// Fill an unset role from the token texts the reference recognises for
    /// it, marking the chosen token CONTROL.
    fn detect(
        &mut self,
        role: &'static str,
        names: &[&str],
        slot: fn(&mut Specials) -> &mut Option<u32>,
    ) {
        if slot(&mut self.specials).is_some() {
            return;
        }
        let found: Vec<u32> = names.iter().filter_map(|t| self.id_of(t)).collect();
        if found.len() > 1 {
            self.ambiguous_roles.push(role);
        }
        if let Some(&id) = found.iter().min() {
            *slot(&mut self.specials) = Some(id);
            self.attrs[id as usize] |= attr::CONTROL;
        }
    }

    /// The reference's text-based attribute rules, in its order.
    fn apply_overrides(&mut self) {
        self.detect("eot", EOT_NAMES, |s| &mut s.eot);
        self.detect("eom", EOM_NAMES, |s| &mut s.eom);
        self.detect("fim_pre", FIM_PRE_NAMES, |s| &mut s.fim_pre);
        self.detect("fim_suf", FIM_SUF_NAMES, |s| &mut s.fim_suf);
        self.detect("fim_mid", FIM_MID_NAMES, |s| &mut s.fim_mid);
        self.detect("fim_pad", FIM_PAD_NAMES, |s| &mut s.fim_pad);
        self.detect("fim_rep", FIM_REP_NAMES, |s| &mut s.fim_rep);
        self.detect("fim_sep", FIM_SEP_NAMES, |s| &mut s.fim_sep);

        for (text, a) in self.texts.iter().zip(self.attrs.iter_mut()) {
            if *a & attr::CONTROL != 0 && text.contains("unused") {
                *a |= attr::UNUSED;
            }
        }

        let sp = &self.specials;
        let mut eog: HashSet<u32> = [sp.fim_pad, sp.fim_rep, sp.fim_sep]
            .into_iter()
            .flatten()
            .collect();
        for name in EOG_NAMES {
            if let Some(id) = self.id_of(name) {
                eog.insert(id);
                self.attrs[id as usize] |= attr::CONTROL;
            }
        }
        for name in ALWAYS_RENDERED {
            if let Some(id) = self.id_of(name) {
                self.attrs[id as usize] = attr::USER_DEFINED;
            }
        }
        let sp = &self.specials;
        eog.extend([sp.eos, sp.eot, sp.eom].into_iter().flatten());

        let texts = &self.texts;
        let has = |eog: &HashSet<u32>, t: &str| eog.iter().any(|&id| texts[id as usize] == t);
        let call = has(&eog, "<|call|>") || has(&eog, "<|calls|>");
        let end = eog
            .iter()
            .copied()
            .find(|&id| texts[id as usize] == "<|end|>");
        if let Some(end) = end
            && ((has(&eog, "<|return|>") && call) || (call && has(&eog, "<|flush|>")))
        {
            eog.remove(&end);
            self.attrs[end as usize] = attr::USER_DEFINED;
        }
        let texts = &self.texts;
        let s = eog.iter().copied().find(|&id| texts[id as usize] == "</s>");
        if let Some(s) = s
            && eog
                .iter()
                .any(|&id| texts[id as usize] == "<|tool_response>")
        {
            eog.remove(&s);
            self.attrs[s as usize] = attr::NORMAL;
        }

        let mut eog: Vec<u32> = eog.into_iter().collect();
        eog.sort_unstable();
        self.eog = eog;
    }

    /// The tokens the partition matches as whole pieces, longest text first
    /// (ties by id — see [`crate::Tokenizer::special_overlap`]).
    pub(crate) fn special_order(&self) -> Vec<u32> {
        let mask = attr::CONTROL | attr::USER_DEFINED | attr::UNKNOWN;
        let mut ids: Vec<u32> = (0..self.texts.len() as u32)
            .filter(|&id| self.attrs[id as usize] & mask != 0)
            .collect();
        ids.sort_by_key(|&id| std::cmp::Reverse(self.texts[id as usize].len()));
        ids
    }
}
