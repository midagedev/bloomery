//! Clef's own part of the decide seat (`serve_seats::decide`, its row in
//! `ROWS`): a SystemOne request parsed, validated and encoded at the
//! release's `max_length` by the decision crate, the backbone's hidden states
//! through the joint schema head with the file's output rows, the answer
//! body, which names the model as the server does. A request with images is
//! each image decoded and prepared by the seat's image input
//! ([`serve::media`]), laid out by the decision crate with the counts the
//! prepares give ([`encode_media`]) and run through the backbone's media
//! call; one prompt a request, each image encoded once. A request the
//! decision crate refuses is a 400 with its message, a video, or an image on
//! a server with no encoder, a 501, an image the media part refuses (not a
//! data URL of a PNG or a JPEG, one the model cannot take) or a placeholder
//! typed into the text a 400 by name, an encoding past the seat's context a
//! 400 by name, a backbone or head failure a 500. Each request's stages go
//! to stderr on one line.

use std::time::Instant;

use bloomery_gpu_gates::GateError;
use decision::Error as DError;
use decision::answer::answer;
use decision::encode::{MAX_LENGTH, NO_IMAGE_INPUT, encode, encode_media};
use decision::head::ClefHead;
use decision::render::dumps;
use decision::request::Request;
use decision::rows::output_rows;
use gguf::Split;
use serve::decide::{DecideError, Decided, HeadSource};
use serve::media::{self, ExpandError, MediaError, MediaModel, Prepared};

use crate::serve_seats::decide::{Backbone, Decision, Placed};

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
        | DError::SchemaTooLong { .. }
        | DError::PlaceholderInText { .. }
        | DError::ImagesTooLong { .. } => DecideError::Refused(e.to_string()),
        DError::NotSupported(m) => DecideError::NotSupported(m),
        _ => DecideError::Engine(e.to_string()),
    }
}

/// A refusal of the media part: the request's 400, but for a prepare whose span is empty, the
/// model's 500.
fn media_error(i: usize, e: &MediaError) -> DecideError {
    let m = format!("images[{i}]: {e}");
    match e {
        MediaError::Expand(ExpandError::EmptySpan(_)) => DecideError::Engine(m),
        _ => DecideError::Refused(m),
    }
}

/// Each image URL loaded (a data URL of a PNG or a JPEG) and prepared for `media`, in the
/// request's order.
fn prepare_images(urls: &[String], media: &dyn MediaModel) -> Result<Vec<Prepared>, DecideError> {
    urls.iter()
        .enumerate()
        .map(|(i, url)| {
            let file = media::load(url).map_err(|e| media_error(i, &e))?;
            media::prepare(&file, media).map_err(|e| media_error(i, &e))
        })
        .collect()
}

struct Clef {
    head: ClefHead,
}

impl Decision for Clef {
    fn hidden_size(&self) -> usize {
        self.head.config().hidden_size
    }

    fn takes_images(&self) -> bool {
        true
    }

    fn decide(&mut self, body: &str, on: &mut Backbone) -> Result<Decided, DecideError> {
        let req = Request::from_json(&decision::json::parse(body).map_err(classify)?)
            .map_err(classify)?;
        let engine = |e: &dyn std::fmt::Display| DecideError::Engine(format!("backbone: {e}"));
        // A request with images is laid out by the counts its images' prepares give; one without
        // runs the text encode and the text call, whether or not the server has an encoder.
        let (enc, images) = if req.images.is_empty() {
            (
                encode(&on.tokenizer, &req, MAX_LENGTH).map_err(classify)?,
                Vec::new(),
            )
        } else {
            let media = on
                .media()
                .ok_or_else(|| DecideError::NotSupported(NO_IMAGE_INPUT.to_owned()))?;
            let images = prepare_images(&req.images, media)?;
            let counts: Vec<usize> = images.iter().map(|p| p.span_len).collect();
            let enc =
                encode_media(&on.tokenizer, &req, media, &counts, MAX_LENGTH).map_err(classify)?;
            (enc, images)
        };
        if enc.ids.len() > on.ctx {
            return Err(DecideError::Refused(format!(
                "the request encodes to {} tokens, past this server's context of {}",
                enc.ids.len(),
                on.ctx
            )));
        }
        let (hidden, prompt_ms, tower_ms) = if images.is_empty() {
            let (hidden, prompt_ms) = on.hidden(&enc.ids).map_err(|e| engine(&e))?;
            (hidden, prompt_ms, 0.0)
        } else {
            if enc.image_spans.len() != images.len() {
                return Err(DecideError::Engine(format!(
                    "the encoding holds {} image spans for {} images",
                    enc.image_spans.len(),
                    images.len()
                )));
            }
            let placed: Vec<Placed<'_>> = enc
                .image_spans
                .iter()
                .zip(&images)
                .map(|(span, prepared)| Placed {
                    at: span.0,
                    prepared,
                })
                .collect();
            on.hidden_media(&enc.ids, &placed).map_err(|e| engine(&e))?
        };
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
            "bloomery-serve: clef n={} prompt={prompt_ms:.1} tower={tower_ms:.1} head {} answer+={:.1}",
            enc.ids.len(),
            decision::head::stage_line(&stages),
            t1.elapsed().as_secs_f64() * 1e3 - stages.iter().map(|s| s.1).sum::<f64>()
        );
        Ok(Decided {
            body,
            prompt_n: enc.ids.len(),
            // The prompt's wall as a client sees it: the tower's encode and the backbone's call.
            prompt_ms: prompt_ms + tower_ms,
            head_ms: t1.elapsed().as_secs_f64() * 1e3,
        })
    }
}
