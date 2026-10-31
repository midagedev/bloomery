//! The vocabulary as the reference loads it (`llama_vocab::impl::load`):
//! token texts and attributes, the special ids, the attribute overrides it
//! applies by token text, and the special-token list the partition walks.
//!
//! `tokenizer.ggml.eog_token_ids` is an authoritative, nonempty stop set:
//! integer ids are bounds-checked, sorted and deduplicated. It replaces guessed
//! stops without disabling GGUF's token-attribute/role overrides or changing BPE.
//! The private `tokenizer.bloomery.eog_token_ids` additionally selects the HF
//! raw-text/BPE contract. When both keys occur, their canonical sets must agree;
//! the private key still selects HF semantics. Neither list changes BOS/EOS insertion.

use std::collections::{HashMap, HashSet};

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
    "<|eot|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<|im:end|>",
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
    "[e~[",
];
const ALWAYS_RENDERED: &[&str] = &["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"];

/// The `tokenizer.ggml.pre` names whose decoded text is cleaned as the reference's
/// `clean_spaces` does: its BPE default, which the llama3 type keeps (llama-vocab.cpp sets it
/// true for every BPE and the type's branch does not clear it) and most other types clear. This
/// list holds the names whose cleaning is on; a name absent here decodes as this crate always
/// has.
const CLEAN_SPACES: &[&str] = &["llama-bpe"];

/// A `tokenizer.ggml.model` this crate runs.
#[derive(Clone, Copy)]
pub(crate) enum Model {
    /// `gpt2`: byte-level BPE, the pre-tokenizer named by the file.
    Gpt2,
    /// `gemma4`: BPE over metaspace text, always the `gemma4` pre-tokenizer.
    Gemma4,
    /// `llama`: SentencePiece merges ordered by token score.
    Spm,
}

impl Model {
    /// The names this crate runs, as `tokenizer.ggml.model` spells them.
    pub(crate) const NAMES: [(&'static str, Model); 3] = [
        ("gpt2", Model::Gpt2),
        ("gemma4", Model::Gemma4),
        ("llama", Model::Spm),
    ];
}

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

/// The model's encoding tables, separate from the shared vocabulary.
pub(crate) enum Encoding {
    Bpe {
        pre: Pre,
        merges: Vec<String>,
        nfc: bool,
        /// A Hugging Face tokenizer, as the explicit stop list says: its BPE and its
        /// added-token split are `tokenizers`'.
        hf: bool,
    },
    Spm {
        scores: Vec<f32>,
        add_space_prefix: bool,
    },
}

#[derive(Clone)]
pub(crate) struct Vocab {
    pub(crate) texts: Vec<String>,
    pub(crate) attrs: Vec<u32>,
    pub(crate) token_to_id: HashMap<String, u32>,
    pub(crate) pre: String,
    pub(crate) specials: Specials,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    /// Whether decoded text has its spaces cleaned, the reference's `clean_spaces` for the
    /// pre-tokenizer's name ([`CLEAN_SPACES`]).
    pub(crate) clean_spaces: bool,
    pub(crate) eog: Vec<u32>,
    /// Roles the reference fills from the first matching token in hash-map
    /// order, where more than one candidate text is in this vocabulary — the
    /// reference's pick is then unspecified.
    pub(crate) ambiguous_roles: Vec<&'static str>,
    /// The metadata this load read, `(key, shown value)`, in reading order.
    pub(crate) keys_read: Vec<(String, String)>,
}

struct Meta<'a> {
    keys: &'a [(String, Value)],
    read: Vec<(String, String)>,
}

impl<'a> Meta<'a> {
    fn get(&mut self, key: &str) -> Option<&'a Value> {
        let v = self.keys.iter().find(|(k, _)| k == key).map(|(_, v)| v);
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

    fn stops(&mut self, key: &'static str, n: usize) -> Result<Option<Vec<u32>>, Error> {
        self.ints(key)?
            .map(|ids| {
                if ids.is_empty() {
                    return Err(Error::InvalidStops(format!("{key} is empty")));
                }
                let mut stops = ids
                    .into_iter()
                    .map(|id| {
                        u32::try_from(id)
                            .ok()
                            .filter(|&id| (id as usize) < n)
                            .ok_or_else(|| {
                                Error::InvalidStops(format!(
                                    "{key} contains id {id} outside the vocabulary of {n}"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                stops.sort_unstable();
                stops.dedup();
                Ok(stops)
            })
            .transpose()
    }

    fn scores(&mut self, n_tokens: usize) -> Result<Vec<f32>, Error> {
        const KEY: &str = "tokenizer.ggml.scores";
        let mut scores = match self.get(KEY) {
            None => return Ok(vec![0.0; n_tokens]),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| match *v {
                    Value::F32(x) => Ok(x),
                    Value::I32(x) => Ok(x as f32),
                    _ => Err(Error::WrongType(KEY)),
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(Error::WrongType(KEY)),
        };
        if scores.len() < n_tokens {
            return Err(Error::TokenScoresShort {
                scores: scores.len(),
                tokens: n_tokens,
            });
        }
        scores.truncate(n_tokens);
        for (id, score) in scores.iter_mut().enumerate() {
            if !score.is_finite() {
                return Err(Error::InvalidTokenScore { id, score: *score });
            }
            // The reference compares signed zeros as equal priorities.
            if *score == 0.0 {
                *score = 0.0;
            }
        }
        Ok(scores)
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
    pub(crate) fn load(
        keys: &[(String, Value)],
    ) -> Result<(Vocab, Encoding, Option<Vec<String>>), Error> {
        let mut m = Meta {
            keys,
            read: Vec::new(),
        };

        let model = m
            .str("tokenizer.ggml.model")?
            .ok_or(Error::Missing("tokenizer.ggml.model"))?;
        let model = crate::named(&Model::NAMES, model)
            .ok_or_else(|| Error::UnsupportedModel(model.to_string()))?;
        let nfc = match m.str("tokenizer.ggml.normalizer")? {
            None => false,
            Some("NFC") if !matches!(model, Model::Spm) => true,
            Some(name) => return Err(Error::UnsupportedNormalizer(name.to_string())),
        };
        // The converter's boolean spelling of the same normalizer.
        let nfc = match m.flag("tokenizer.ggml.normalizer.nfc")? {
            Some(true) if matches!(model, Model::Spm) => {
                return Err(Error::UnsupportedNormalizer("nfc".to_string()));
            }
            flag => nfc || flag == Some(true),
        };
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
        let explicit_stops = m.stops("tokenizer.bloomery.eog_token_ids", n)?;
        let ggml_stops = m.stops("tokenizer.ggml.eog_token_ids", n)?;
        if let (Some(hf), Some(ggml)) = (&explicit_stops, &ggml_stops)
            && hf != ggml
        {
            return Err(Error::InvalidStops(
                "tokenizer.bloomery.eog_token_ids conflicts with tokenizer.ggml.eog_token_ids"
                    .into(),
            ));
        }
        let hf_texts = explicit_stops.as_ref().map(|_| texts.clone());
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

        let mut clean_spaces = false;
        let (pre, encoding) = match model {
            Model::Gpt2 | Model::Gemma4 => {
                let name = match model {
                    Model::Gemma4 => "gemma4",
                    _ => m.str("tokenizer.ggml.pre")?.unwrap_or(""),
                };
                let pre = crate::named(&Pre::NAMES, name)
                    .ok_or_else(|| Error::UnsupportedPre(name.to_string()))?;
                clean_spaces = CLEAN_SPACES.contains(&name);
                // Only the private stop key marks a Hugging Face tokenizer, whose pre-tokenizer
                // runs as its tokenizer.json does.
                let hf = explicit_stops.is_some();
                let pre = match (pre, hf) {
                    (pre, false) => pre,
                    (pre, true) => pre.hf_reading().ok_or_else(|| {
                        Error::InvalidStops(format!(
                            "tokenizer.bloomery.eog_token_ids marks a Hugging Face tokenizer, and \
                             pre-tokenizer {name:?} has no reading that runs as tokenizers does ({})",
                            Pre::hf_names()
                        ))
                    })?,
                };
                let merges = m
                    .strings("tokenizer.ggml.merges")?
                    .ok_or(Error::Missing("tokenizer.ggml.merges"))?;
                // The reference NFC-normalizes the text of a pre-tokenizer that names it
                // ([`Pre::nfc`]) whatever the header says.
                let nfc = nfc || pre.nfc();
                (
                    name.to_string(),
                    Encoding::Bpe {
                        pre,
                        merges,
                        nfc,
                        hf,
                    },
                )
            }
            Model::Spm if explicit_stops.is_some() => {
                return Err(Error::InvalidStops(
                    "tokenizer.bloomery.eog_token_ids marks a Hugging Face byte-level BPE tokenizer, not SentencePiece".into(),
                ));
            }
            Model::Spm => (
                "default".to_string(),
                Encoding::Spm {
                    scores: m.scores(n)?,
                    add_space_prefix: m.flag("tokenizer.ggml.add_space_prefix")?.unwrap_or(true),
                },
            ),
        };

        // The model's defaults, with the reference's per-name override (`glm4`
        // names no BOS), then the file's ids where in range.
        let mut sp = if explicit_stops.is_some() {
            Specials::default()
        } else {
            match model {
                Model::Gpt2 => Specials {
                    bos: (n > 11 && !matches!(encoding, Encoding::Bpe { pre: Pre::Glm4, .. }))
                        .then_some(11),
                    eos: (n > 11).then_some(11),
                    ..Specials::default()
                },
                Model::Gemma4 => Specials::default(),
                Model::Spm => Specials {
                    bos: (n > 1).then_some(1),
                    eos: (n > 2).then_some(2),
                    unk: (n > 0).then_some(0),
                    ..Specials::default()
                },
            }
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
            // An id outside the vocabulary keeps the default, as llama-vocab does.
            for (key, slot) in slots {
                if let Some(id) = m.id(key)?
                    && let Ok(id) = u32::try_from(id)
                    && (id as usize) < n
                {
                    *slot(&mut sp) = Some(id);
                }
            }
        }
        // SentencePiece and the reference's llama3 pre type default to adding a BOS
        // (llama-vocab.cpp), the file's key overriding either.
        let add_bos = m
            .flag("tokenizer.ggml.add_bos_token")?
            .unwrap_or(matches!(model, Model::Spm) || pre == "llama-bpe")
            || matches!(
                encoding,
                Encoding::Bpe {
                    pre: Pre::Gemma4,
                    ..
                }
            );
        let add_eos = m.flag("tokenizer.ggml.add_eos_token")?.unwrap_or(false);
        // The reference asserts a BOS id wherever it adds a BOS, and an EOS
        // id wherever it adds an EOS.
        if add_bos && sp.bos.is_none() {
            return Err(Error::Missing("tokenizer.ggml.bos_token_id"));
        }
        if add_eos && sp.eos.is_none() {
            return Err(Error::Missing("tokenizer.ggml.eos_token_id"));
        }

        let mut v = Vocab {
            texts,
            attrs,
            token_to_id,
            pre,
            specials: sp,
            add_bos,
            add_eos,
            clean_spaces,
            eog: Vec::new(),
            ambiguous_roles: Vec::new(),
            keys_read: std::mem::take(&mut m.read),
        };
        if let Some(stops) = explicit_stops {
            // An explicit HF contract keeps its ids and token types, without name-based guesses.
            v.eog = stops;
        } else {
            v.apply_overrides();
            if let Some(stops) = ggml_stops {
                v.eog = stops;
            }
        }
        Ok((v, encoding, hf_texts))
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

    /// Apply reference attribute rules in order, including pre-tokenizer-specific stop names.
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
        for name in EOG_NAMES.iter().chain(if self.pre == "flint" {
            ["<|user|>", "<|observation|>"].as_slice()
        } else {
            &[]
        }) {
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
