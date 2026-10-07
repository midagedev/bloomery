//! lev's own part of the decide seat (`serve_seats::decide`, its row in `ROWS`): a SystemOne request
//! read and checked as llama.cpp's server checks it, each question's prompts (one a variant: a choice
//! of two options or more is asked with its options in two orders) rendered from the model file's own
//! `systemone` template and tokenized whole, the backbone's last hidden state of each through the
//! model's own language-model head at the label tokens, the answer body of the decision crate's label
//! readout (`decision::label`), which names the model as the server does. A request the decision
//! crate refuses is a 400 with its message, an image or a video a 501, a prompt past the seat's
//! context a 400 by name (before any prompt runs), a backbone or label failure a 500. Each request's
//! stages go to stderr on one line.

use std::time::Instant;

use bloomery_gpu_gates::GateError;
use decision::Error as DError;
use decision::label::LabelModel;
use decision::release::lev;
use decision::render::dumps;
use decision::request::Request;
use gguf::Split;
use serve::decide::{DecideError, Decided, HeadSource};
use tokenizer::Tokenizer;

use crate::serve_seats::decide::{Backbone, Decision};

/// The model `from` names: the model file's `systemone` template, labels, label rows and
/// temperatures. A head file beside it is not lev's; the seat never passes one. The labels are
/// the file's own tokenizer's, read here (the backbone's, opened after the head, is the same
/// file's).
pub fn open(from: HeadSource<'_>) -> Result<Box<dyn Decision>, GateError> {
    let HeadSource::InFile(model) = from else {
        return Err("lev's head is its model file's own: no head file is read".into());
    };
    let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
    let tokenizer = Tokenizer::from_gguf(model)?;
    let model = LabelModel::open(lev::POLICY, &split, |code| {
        tokenizer.encode(code, false, false)
    })?;
    Ok(Box::new(Lev { model }))
}

/// Whether a decision crate error is the request's (a 400) or the engine's (a 500).
fn classify(e: DError) -> DecideError {
    match e {
        DError::Json { .. }
        | DError::JsonTooDeep(_)
        | DError::JsonNonFinite(_)
        | DError::JsonFloatRange(_)
        | DError::JsonLoneSurrogate(_)
        | DError::Request(_)
        | DError::Question { .. } => DecideError::Refused(e.to_string()),
        DError::NotSupported(m) => DecideError::NotSupported(m),
        _ => DecideError::Engine(e.to_string()),
    }
}

struct Lev {
    model: LabelModel,
}

impl Decision for Lev {
    fn hidden_size(&self) -> usize {
        self.model.width()
    }

    fn decide(&mut self, body: &str, on: &mut Backbone) -> Result<Decided, DecideError> {
        let t0 = Instant::now();
        let json = decision::json::parse(body).map_err(classify)?;
        let req = Request::from_json_with(&json, &self.model.policy().rules).map_err(classify)?;
        let asked = self.model.plan(&req).map_err(classify)?;
        let prompts = self.model.prompts(&req, &asked).map_err(classify)?;
        // Every prompt is tokenized and sized before the first one runs.
        let ids: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| on.tokenizer.encode(&p.text, false, true))
            .collect();
        if let Some((p, n)) = prompts
            .iter()
            .zip(&ids)
            .map(|(p, i)| (p, i.len()))
            .find(|(_, n)| *n > on.ctx)
        {
            return Err(DecideError::Refused(format!(
                "{}: the prompt of variant {} is {n} tokens, past this server's context of {}",
                req.questions[p.question].id, p.variant, on.ctx
            )));
        }
        let width = self.model.width();
        let (mut prompt_ms, mut scores) = (0.0, Vec::with_capacity(prompts.len()));
        for (p, ids) in prompts.iter().zip(&ids) {
            let (hidden, ms) = on
                .hidden(ids)
                .map_err(|e| DecideError::Engine(format!("backbone: {e}")))?;
            prompt_ms += ms;
            let last = hidden
                .len()
                .checked_sub(width)
                .and_then(|from| hidden.get(from..))
                .ok_or_else(|| {
                    DecideError::Engine(format!(
                        "backbone: {} hidden values for {} ids of width {width}",
                        hidden.len(),
                        ids.len()
                    ))
                })?;
            scores.push(self.model.logits(last, p.outputs).map_err(classify)?);
        }
        let n_tokens: usize = ids.iter().map(Vec::len).sum();
        let answers = self
            .model
            .answers(&req, &asked, &scores)
            .map_err(classify)?;
        let body = dumps(&self.model.body(&on.name, answers, n_tokens), false);
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "bloomery-serve: lev n={n_tokens} prompts={} prompt={prompt_ms:.1} answer+={:.1}",
            prompts.len(),
            total_ms - prompt_ms
        );
        Ok(Decided {
            body,
            prompt_n: n_tokens,
            prompt_ms,
            head_ms: total_ms - prompt_ms,
        })
    }
}
