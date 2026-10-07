//! Clef's own part of the decide seat (`serve_seats::decide`, its row in
//! `ROWS`): a SystemOne request parsed, validated and encoded at the
//! release's `max_length` by the decision crate, the backbone's hidden states
//! through the joint schema head with the file's output rows, the answer
//! body, which names the model as the server does. A request the decision
//! crate refuses is a 400 with its message, an image or a video a 501, an
//! encoding past the seat's context a 400 by name, a backbone or head
//! failure a 500. Each request's stages go to stderr on one line.

use std::time::Instant;

use bloomery_gpu_gates::GateError;
use decision::Error as DError;
use decision::answer::answer;
use decision::encode::{MAX_LENGTH, encode};
use decision::head::ClefHead;
use decision::render::dumps;
use decision::request::Request;
use decision::rows::output_rows;
use gguf::Split;
use serve::decide::{DecideError, Decided, HeadSource};

use crate::serve_seats::decide::{Backbone, Decision};

/// The head `from` names: the release's head file with its config, or the
/// decision tensors of a model file in llama.cpp's Clef layout.
pub fn open(from: HeadSource<'_>) -> Result<Box<dyn Decision>, GateError> {
    let head = match from {
        HeadSource::Files { head, config } => ClefHead::open(head, Some(config))?,
        HeadSource::InFile(model) => ClefHead::from_gguf(
            &Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?,
        )?,
    };
    Ok(Box::new(Clef { head }))
}

/// Whether a decision crate error is the request's (a 400) or the engine's
/// (a 500).
fn classify(e: DError) -> DecideError {
    match e {
        DError::Json { .. }
        | DError::JsonTooDeep(_)
        | DError::JsonNonFinite(_)
        | DError::JsonFloatRange(_)
        | DError::JsonLoneSurrogate(_)
        | DError::Request(_)
        | DError::Question { .. }
        | DError::EmptySpan { .. }
        | DError::SchemaTooLong { .. } => DecideError::Refused(e.to_string()),
        DError::NotSupported(m) => DecideError::NotSupported(m),
        _ => DecideError::Engine(e.to_string()),
    }
}

struct Clef {
    head: ClefHead,
}

impl Decision for Clef {
    fn hidden_size(&self) -> usize {
        self.head.config().hidden_size
    }

    fn decide(&mut self, body: &str, on: &mut Backbone) -> Result<Decided, DecideError> {
        let req = Request::from_json(&decision::json::parse(body).map_err(classify)?)
            .map_err(classify)?;
        let enc = encode(&on.tokenizer, &req, MAX_LENGTH).map_err(classify)?;
        if enc.ids.len() > on.ctx {
            return Err(DecideError::Refused(format!(
                "the request encodes to {} tokens, past this server's context of {}",
                enc.ids.len(),
                on.ctx
            )));
        }
        let (hidden, prompt_ms) = on
            .hidden(&enc.ids)
            .map_err(|e| DecideError::Engine(format!("backbone: {e}")))?;
        let t1 = Instant::now();
        let file = &on.file;
        let mut stages = Vec::new();
        let logits = self
            .head
            .forward_staged(
                &hidden,
                &enc,
                &mut |ids| output_rows(file, ids).map_err(|e| e.to_string()),
                &mut stages,
            )
            .map_err(classify)?;
        let body = dumps(
            &answer(&req, &enc, &logits, &on.name).map_err(classify)?,
            false,
        );
        eprintln!(
            "bloomery-serve: clef n={} prompt={prompt_ms:.1} head {} answer+={:.1}",
            enc.ids.len(),
            decision::head::stage_line(&stages),
            t1.elapsed().as_secs_f64() * 1e3 - stages.iter().map(|s| s.1).sum::<f64>()
        );
        Ok(Decided {
            body,
            prompt_n: enc.ids.len(),
            prompt_ms,
            head_ms: t1.elapsed().as_secs_f64() * 1e3,
        })
    }
}
