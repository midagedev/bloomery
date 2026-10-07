//! The label-readout family of decision models (lev; nimble, pplx-decider and OpenJev are rows of the
//! same): the model answers each question in one prompt pass, and its score for option `i` is the
//! language-model head's logit at the `i`-th label token, read at the last position of the prompt. The
//! prompt is the model's own `systemone` template (a GGUF metadata string) rendered over the question;
//! nothing is generated and no head file exists.
//!
//! This module is llama.cpp's `server-decision.cpp` at `a4cb4c61` for these models, the part that is
//! the same for every one of them, parameterised by a [`Policy`]:
//!
//! - the labels, the codes `A`..`Z` then `AA`..`ZZ` that the vocabulary holds as one token, at most 255
//!   ([`Labels`]), each shown to the template as `label` and read as a logit;
//! - the temperatures, `<arch>.decision.temperature.<type>[.<bucket>]` in the file, as llama.cpp reads them
//!   (an f32 key goes through `std::to_string` and `strtof`: [`llama_float`]) and its bucket rule
//!   ([`Temperatures`]);
//! - the prompts of a request, one per question and variant ([`LabelModel::prompts`]): the template's
//!   input `{id, type, instructions, state, options, images}` with the options in the variant's order
//!   (the second variant of a lev choice shows them reversed), every object's keys sorted when the model
//!   was trained that way, the template rendered the way llama.cpp's own jinja runtime renders it
//!   ([`jinja::Floats::Cpp`]);
//! - the label logit ([`LabelModel::logits`]) and the answer ([`LabelModel::answers`]): the variants'
//!   softmaxes averaged, a noul read on lev's 0..8 rating scale as its expected value, a choice's first
//!   highest option, a score's expected level and legend, and the confidences the SystemOne API defines;
//!   every number unrounded f64, in llama.cpp's key order.
//!
//! The backbone's pass, the tokenizer and the card are the seat's.

use std::collections::BTreeMap;

use gguf::Split;
use jinja::{ChatTemplate, Floats};
use serde_json::{Map, Number, Value};

use crate::Error;
use crate::answer::{
    confidence_choice, confidence_score, expected_level, expected_rating, softmax_averaged,
};
use crate::json::Json;
use crate::request::{Kind, Opt, Request, Rules};
use crate::rows::output_rows;

/// The most labels a model has: llama.cpp keeps the first 255 single-token codes.
pub const MAX_LABELS: usize = 255;

/// The GGUF key of the `systemone` prompt template.
pub const TEMPLATE_KEY: &str = "tokenizer.chat_template.systemone";

/// What a label-readout model does that another does not.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// How the request's options are ordered, and which checks the request meets.
    pub rules: Rules,
    /// A noul is read on this many ratings (the model's first labels, 0 certainly no up), and its
    /// answer is the expected rating over the scale; `None`: it is read on its two options.
    pub noul_ratings: Option<usize>,
    /// A choice of two options or more is asked this many times, the options shown in the reverse
    /// order from the second time on, and the answers averaged.
    pub choice_variants: usize,
    /// The template's input has every object's keys sorted: the model was trained on sorted JSON.
    pub sort_keys: bool,
    /// The temperature bucket of a question of this many options (the key's last part).
    pub bucket: fn(usize) -> &'static str,
}

/// One question of a request, planned: its options in the order the model's outputs follow, and how
/// many prompts and outputs it takes.
#[derive(Clone, Debug)]
pub struct Asked {
    /// The question's place in the request.
    pub question: usize,
    pub options: Vec<Opt>,
    /// The prompts: the variants.
    pub variants: usize,
    /// The labels read at the end of each prompt.
    pub outputs: usize,
}

/// One prompt to run: a question's variant, rendered, and the labels to read after it.
#[derive(Clone, Debug)]
pub struct Prompt {
    pub question: usize,
    pub variant: usize,
    pub text: String,
    pub outputs: usize,
}

/// The label tokens of a vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Labels {
    ids: Vec<u32>,
    texts: Vec<String>,
}

impl Labels {
    /// The codes the labels are drawn from, in order: `A`..`Z`, then `AA`..`ZZ`.
    pub fn codes() -> impl Iterator<Item = String> {
        let single = ('A'..='Z').map(String::from);
        let double = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
        single.chain(double)
    }

    /// The labels of a vocabulary whose tokenizer is `encode(text)` (no special tokens added or
    /// parsed): the codes that are one token, the first [`MAX_LABELS`] of them.
    pub fn from_vocab(mut encode: impl FnMut(&str) -> Vec<u32>) -> Labels {
        let mut labels = Labels {
            ids: Vec::new(),
            texts: Vec::new(),
        };
        for code in Labels::codes() {
            let tokens = encode(&code);
            if let [id] = tokens[..]
                && labels.ids.len() < MAX_LABELS
            {
                labels.ids.push(id);
                labels.texts.push(code);
            }
        }
        labels
    }

    /// The label tokens, in order.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The code the `i`-th label is shown as.
    #[must_use]
    pub fn text(&self, i: usize) -> &str {
        &self.texts[i]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// A float as llama.cpp's server holds a metadata float it reads back: the key's text is
/// `std::to_string`'s (`%f`, six decimals) and the value `strtof` of that text, which is not the stored
/// f32 but the nearest f32 to its six-decimal rounding.
#[must_use]
pub fn llama_float(stored: f32) -> f32 {
    format!("{:.6}", f64::from(stored))
        .parse()
        .expect("a formatted float reads back")
}

/// The temperatures of a model file, by the key's part after `<arch>.decision.temperature.`
/// (`choice.small`, `score`, ...).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Temperatures(BTreeMap<String, f32>);

impl Temperatures {
    /// The temperatures a file holds: every `<arch>.decision.temperature.*` key, an f32 or f64 that
    /// llama.cpp reads as [`llama_float`] says and is above 0. A key that is not a float, or not
    /// above 0, is refused by name (llama.cpp's server refuses to start on the latter).
    pub fn from_split(split: &Split) -> Result<Temperatures, Error> {
        let arch = split
            .architecture()
            .ok_or_else(|| Error::Label("the model file names no architecture".into()))?;
        let prefix = format!("{arch}.decision.temperature.");
        let mut found = Vec::new();
        for (key, value) in split.iter_kv() {
            let Some(name) = key.strip_prefix(&prefix) else {
                continue;
            };
            let stored = value.as_f32().ok_or_else(|| {
                Error::Label(format!("{key} is {value:?}, not a float temperature"))
            })?;
            found.push((name.to_owned(), stored));
        }
        Temperatures::from_stored(found)
    }

    /// The temperatures of `(name, stored f32)` pairs, as [`Temperatures::from_split`] reads them.
    pub fn from_stored(
        pairs: impl IntoIterator<Item = (String, f32)>,
    ) -> Result<Temperatures, Error> {
        let mut map = BTreeMap::new();
        for (name, stored) in pairs {
            let t = llama_float(stored);
            if !t.is_finite() || t <= 0.0 {
                return Err(Error::Label(format!(
                    "invalid decision temperature {name} = {stored} (read as {t}): it must be above 0"
                )));
            }
            map.insert(name, t);
        }
        Ok(Temperatures(map))
    }

    /// A question's temperature: `<type>.<bucket>` when the file has it, else `<type>`, else 1.
    #[must_use]
    pub fn of(&self, policy: &Policy, kind: Kind, n_options: usize) -> f32 {
        let word = kind.word();
        let bucketed = format!("{word}.{}", (policy.bucket)(n_options));
        [bucketed.as_str(), word]
            .into_iter()
            .find_map(|name| self.0.get(name).copied())
            .unwrap_or(1.0)
    }
}

/// A label-readout model, opened: its policy, its template, its labels with their output rows and its
/// temperatures.
pub struct LabelModel {
    policy: Policy,
    template: ChatTemplate,
    labels: Labels,
    /// The output embedding rows of the labels, `[labels, width]` row-major.
    rows: Vec<f32>,
    width: usize,
    temperatures: Temperatures,
}

impl LabelModel {
    /// A model from its parts: `rows` are the labels' output embedding rows, `width` floats each.
    pub fn new(
        policy: Policy,
        template: ChatTemplate,
        labels: Labels,
        rows: Vec<f32>,
        width: usize,
        temperatures: Temperatures,
    ) -> Result<LabelModel, Error> {
        if width == 0 || rows.len() != labels.len() * width {
            return Err(Error::Label(format!(
                "{} label rows of {width} values hold {} values, not {}",
                labels.len(),
                rows.len(),
                labels.len() * width
            )));
        }
        let need = policy.noul_ratings.unwrap_or(2).max(2);
        if labels.len() < need {
            return Err(Error::Label(format!(
                "the vocabulary has {} single-token labels, the model reads at least {need}",
                labels.len()
            )));
        }
        Ok(LabelModel {
            policy,
            template,
            labels,
            rows,
            width,
            temperatures,
        })
    }

    /// The model a GGUF file is: its `systemone` template, its temperatures, its labels from the
    /// vocabulary (`encode`: the tokenizer without special tokens) and their rows of the output head.
    pub fn open(
        policy: Policy,
        split: &Split,
        encode: impl FnMut(&str) -> Vec<u32>,
    ) -> Result<LabelModel, Error> {
        let source = split
            .value(TEMPLATE_KEY)
            .and_then(gguf::Value::as_str)
            .ok_or_else(|| Error::Label(format!("the model file has no {TEMPLATE_KEY}")))?;
        let template = ChatTemplate::parse(source)
            .map_err(|e| Error::Label(format!("{TEMPLATE_KEY} does not parse: {e}")))?;
        let width = split
            .arch_get_u64("embedding_length")
            .and_then(|w| usize::try_from(w).ok())
            .ok_or_else(|| Error::Label("the model file has no embedding_length".into()))?;
        let labels = Labels::from_vocab(encode);
        let rows = output_rows(split, labels.ids())?;
        LabelModel::new(
            policy,
            template,
            labels,
            rows,
            width,
            Temperatures::from_split(split)?,
        )
    }

    #[must_use]
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    #[must_use]
    pub fn labels(&self) -> &Labels {
        &self.labels
    }

    /// The hidden width the label rows read.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The questions of `req`, planned. A question with more options than the model has labels is
    /// refused by name.
    pub fn plan(&self, req: &Request) -> Result<Vec<Asked>, Error> {
        let p = &self.policy;
        let mut out = Vec::with_capacity(req.questions.len());
        for (i, q) in req.questions.iter().enumerate() {
            let options = q.options_with(&p.rules);
            let n = options.len();
            if n > self.labels.len() {
                return Err(Error::Question {
                    id: q.id.clone(),
                    what: format!(
                        "too many options ({n}), this model supports at most {}",
                        self.labels.len()
                    ),
                });
            }
            let variants = if q.kind == Kind::Choice && n > 1 {
                p.choice_variants
            } else {
                1
            };
            let outputs = match (q.kind, p.noul_ratings) {
                (Kind::Noul, Some(ratings)) => ratings,
                _ => n,
            };
            out.push(Asked {
                question: i,
                options,
                variants,
                outputs,
            });
        }
        Ok(out)
    }

    /// Every prompt of a planned request, in the order the answers read them back: each question's
    /// variants in turn, the questions in the request's order.
    pub fn prompts(&self, req: &Request, asked: &[Asked]) -> Result<Vec<Prompt>, Error> {
        let mut out = Vec::new();
        for a in asked {
            let q = &req.questions[a.question];
            for variant in 0..a.variants {
                let input = self.input(req, a, variant)?;
                let text = self
                    .template
                    .render_with(&input, Floats::Cpp)
                    .map_err(|e| Error::Label(format!("{}: {e}", q.id)))?;
                out.push(Prompt {
                    question: a.question,
                    variant,
                    text,
                    outputs: a.outputs,
                });
            }
        }
        Ok(out)
    }

    /// The template's input of one prompt (`render` of llama.cpp's `server_decision_context`):
    /// `id`, `type`, `instructions`, `state`, `options` (`key`, `description`, `label`, in the
    /// variant's order, the labels by display position), no images; every object's keys sorted when
    /// the policy says so.
    fn input(&self, req: &Request, a: &Asked, variant: usize) -> Result<Map<String, Value>, Error> {
        let q = &req.questions[a.question];
        let sort = self.policy.sort_keys;
        let n = a.options.len();
        let options = (0..n)
            .map(|i| {
                let opt = &a.options[if variant == 0 { i } else { n - 1 - i }];
                let mut o = Map::new();
                o.insert("key".into(), Value::String(opt.id.clone()));
                o.insert(
                    "description".into(),
                    opt.description
                        .as_ref()
                        .map_or(Ok(Value::Null), |d| to_value(d, sort))?,
                );
                o.insert(
                    "label".into(),
                    Value::String(self.labels.text(i).to_owned()),
                );
                Ok(Value::Object(sorted(o, sort)))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut input = Map::new();
        input.insert("id".into(), Value::String(q.id.clone()));
        input.insert("type".into(), Value::String(q.kind.word().to_owned()));
        input.insert(
            "instructions".into(),
            q.instructions
                .as_ref()
                .map_or(Ok(Value::Null), |v| to_value(v, sort))?,
        );
        input.insert("state".into(), to_value(&req.state, sort)?);
        input.insert("options".into(), Value::Array(options));
        let mut input = sorted(input, sort);
        input.insert("images".into(), Value::Array(Vec::new()));
        Ok(input)
    }

    /// The scores of one prompt: the logit of each of its first `outputs` labels at its last position,
    /// `hidden` (the final-norm state, `width` values) dotted with the label's output row, summed in
    /// f64.
    pub fn logits(&self, hidden: &[f32], outputs: usize) -> Result<Vec<f32>, Error> {
        if hidden.len() != self.width {
            return Err(Error::HiddenShape {
                values: hidden.len(),
                ids: 1,
                width: self.width,
            });
        }
        if outputs > self.labels.len() {
            return Err(Error::Label(format!(
                "{outputs} outputs for {} labels",
                self.labels.len()
            )));
        }
        let mut scores = Vec::with_capacity(outputs);
        for row in self.rows.chunks_exact(self.width).take(outputs) {
            let dot: f64 = row
                .iter()
                .zip(hidden)
                .map(|(r, h)| f64::from(*r) * f64::from(*h))
                .sum();
            scores.push(dot as f32);
        }
        Ok(scores)
    }

    /// The `answers` object: each question's answer from its prompts' scores (`scores`: one vector a
    /// prompt, in [`LabelModel::prompts`]' order).
    pub fn answers(
        &self,
        req: &Request,
        asked: &[Asked],
        scores: &[Vec<f32>],
    ) -> Result<Json, Error> {
        let mut at = 0;
        let mut out = Vec::with_capacity(asked.len());
        for a in asked {
            let q = &req.questions[a.question];
            let mine = scores.get(at..at + a.variants).ok_or_else(|| {
                Error::Logits(format!(
                    "{}: {} scores for {} prompts",
                    q.id,
                    scores.len(),
                    at + a.variants
                ))
            })?;
            at += a.variants;
            if let Some(bad) = mine.iter().find(|s| s.len() != a.outputs) {
                return Err(Error::Logits(format!(
                    "{}: {} scores for {} outputs",
                    q.id,
                    bad.len(),
                    a.outputs
                )));
            }
            let t = self.temperatures.of(&self.policy, q.kind, a.options.len());
            out.push((
                q.id.clone(),
                self.answer(q.kind, a, &softmax_averaged(mine, t)?),
            ));
        }
        if at != scores.len() {
            return Err(Error::Logits(format!(
                "{} scores for {at} prompts",
                scores.len()
            )));
        }
        Ok(Json::Object(out))
    }

    /// One answer of its probabilities (`format_answer`), the keys in llama.cpp's order.
    fn answer(&self, kind: Kind, a: &Asked, probs: &[f64]) -> Json {
        let num = Json::Float;
        let word = |s: &str| ("type".to_owned(), Json::Str(s.to_owned()));
        let table = || {
            Json::Object(
                a.options
                    .iter()
                    .zip(probs)
                    .map(|(o, p)| (o.id.clone(), num(*p)))
                    .collect(),
            )
        };
        let body = match kind {
            Kind::Noul => {
                let noul = if self.policy.noul_ratings.is_some() {
                    expected_rating(probs)
                } else {
                    a.options
                        .iter()
                        .zip(probs)
                        .find(|(o, _)| o.id == "true")
                        .map_or(0.0, |(_, p)| *p)
                };
                vec![word("noul"), ("noul".to_owned(), num(noul))]
            }
            Kind::Choice => {
                let mut best = 0;
                for (i, p) in probs.iter().enumerate() {
                    if *p > probs[best] {
                        best = i;
                    }
                }
                vec![
                    word("choice"),
                    ("choice".to_owned(), Json::Str(a.options[best].id.clone())),
                    ("probabilities".to_owned(), table()),
                    ("confidence".to_owned(), num(confidence_choice(probs))),
                ]
            }
            Kind::Score => vec![
                word("score"),
                ("score".to_owned(), num(expected_level(probs))),
                (
                    "legend".to_owned(),
                    Json::Object(
                        a.options
                            .iter()
                            .map(|o| (o.id.clone(), o.description.clone().unwrap_or(Json::Null)))
                            .collect(),
                    ),
                ),
                ("probabilities".to_owned(), table()),
                ("confidence".to_owned(), num(confidence_score(probs))),
            ],
        };
        Json::Object(body)
    }

    /// The response body: the server's name for the model, the answers, and the prompts' token count.
    #[must_use]
    pub fn body(&self, model: &str, answers: Json, input_tokens: usize) -> Json {
        Json::Object(vec![
            ("model".to_owned(), Json::Str(model.to_owned())),
            ("answers".to_owned(), answers),
            (
                "usage".to_owned(),
                Json::Object(vec![
                    (
                        "input_tokens".to_owned(),
                        Json::Int(input_tokens.to_string()),
                    ),
                    ("output_tokens".to_owned(), Json::Int("0".to_owned())),
                ]),
            ),
        ])
    }
}

/// `map` with its keys in byte order when `sort`, as llama.cpp's `std::map` gives them.
fn sorted(map: Map<String, Value>, sort: bool) -> Map<String, Value> {
    if !sort {
        return map;
    }
    let mut pairs: Vec<(String, Value)> = map.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs.into_iter().collect()
}

/// A request value as the template's input holds it (nlohmann's reading of the same JSON): an integer
/// that fits 64 bits is one, any other number a float, every object's keys sorted when `sort`. An
/// integer past the f64 range is refused by name.
fn to_value(j: &Json, sort: bool) -> Result<Value, Error> {
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Int(digits) => match digits.parse::<i64>() {
            Ok(i) => Value::from(i),
            Err(_) => match digits.parse::<u64>() {
                Ok(u) => Value::from(u),
                Err(_) => digits
                    .parse::<f64>()
                    .ok()
                    .and_then(Number::from_f64)
                    .map(Value::Number)
                    .ok_or_else(|| {
                        Error::Request(format!("the integer {digits} is past the f64 range"))
                    })?,
            },
        },
        Json::Float(f) => Number::from_f64(*f)
            .map(Value::Number)
            .ok_or_else(|| Error::Request(format!("the number {f} is not finite")))?,
        Json::Str(s) => Value::String(s.clone()),
        Json::Array(items) => Value::Array(
            items
                .iter()
                .map(|v| to_value(v, sort))
                .collect::<Result<_, _>>()?,
        ),
        Json::Object(pairs) => {
            let mut map = Map::new();
            for (k, v) in pairs {
                map.insert(k.clone(), to_value(v, sort)?);
            }
            Value::Object(sorted(map, sort))
        }
    })
}
