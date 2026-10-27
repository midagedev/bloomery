//! The embeddings endpoints over the server's side model ([`crate::embed`]): `POST /v1/embeddings`
//! answers OpenAI's shape, `POST /embeddings` and `POST /embedding` llama-server's. Each reads a
//! request as llama-server's `handle_embeddings_impl` does (`tools/server/server-context.cpp` at
//! `6868a52c`), with these differences: a field of another type than it takes is a 400 naming it
//! (optional fields treat `null` as unset); `/v1/embeddings` refuses a `model` of another type
//! than a string or a name other than the side model's, while the native routes ignore it; an object
//! input with `prompt_string` is a 400, an input past the model's
//! context is a 400 with no batch limit before it, an `embd_normalize` below `-1` or one whose
//! norm f32 cannot scale by is a 400 ([`crate::embed::normalize`]), and `dimensions` cuts the
//! pooled vector to one of the model's widths before it is normalized.
//!
//! `input` (or `content`, which also asks for llama-server's shape on any route) is a string, an
//! array of token ids, an array of strings and token ids, or an array of these with no id at its
//! top level, one vector each. Text is tokenized with the side model's vocabulary, the first text
//! element of each input with its special tokens, as llama-server's `tokenize_mixed` puts them.
//! An object with a `content` array prepares image parts through the side model's media rule
//! and decoded f32 PCM through Gemma4a's preprocessor. A clip, an `input_video` or `video_url`
//! part, is decoded by the server's ffmpeg and planned by the model's video rule
//! ([`crate::video`]); each of its groups is a media span. Text and all media spans share one
//! pass, with BOS and EOS around the whole input.
//!
//! The side model runs one request at a time. Text inputs of one request that carry no image and
//! no audio share one call, which packs them up to the model's context; an image or audio input
//! keeps its own pass. A failure in it ends the server as an engine's does:
//! the request gets a 500, `/health` and every later request a 503, and the server exits after its
//! linger. A request that waited for the model while the server failed or began its stop gets the
//! 503 and no pass. Stop is seen before the shared text call and between it and a later media
//! pass, not between the text inputs of that call.

use std::io;
use std::net::TcpStream;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use ::embed::request;
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

use super::{
    ApiError, EngineFailure, State, body, dead_engine, invalid, panic_text, prompt_ids, send_error,
    send_json, stopping_gate,
};
use crate::embed::Embed;
#[cfg(test)]
use crate::embed::{Norm, normalize};
use crate::engine::Tokenizer;
use crate::http::Request;
use crate::media;
use crate::video::{self, Clip, ReadyGroup, VideoSupport};
use vision::media::Piece;

/// What a server with no side model answers: llama-server's message, with the flags that seat one.
pub(super) const NOT_SERVED: &str =
    "This server does not support embeddings. Start it with `--embd-model` or `--embd-hf`";

/// The side model and the facts the server reads of it once, when it takes it.
pub(super) struct Side {
    embed: Mutex<Box<dyn Embed>>,
    cache: Option<crate::embed_cache::Shared>,
    tok: Arc<dyn Tokenizer>,
    pub(super) name: String,
    pub(super) path: String,
    pub(super) ctx: usize,
    pub(super) prompts: std::collections::BTreeMap<String, String>,
    dims: Vec<usize>,
    media: Option<media::SharedMediaModel>,
    audio: Option<media::SharedMediaModel>,
}

impl Side {
    /// Whether the side model takes a clip: its projector has a video rule and the delimiters its
    /// groups go between.
    pub(super) fn takes_video(&self) -> bool {
        self.media
            .as_ref()
            .is_some_and(|m| m.video_rule().is_some() && m.embedding_delimiters().is_some())
    }

    /// `embed`'s facts read once. A model that names no width is a defect of the model, refused
    /// by name.
    pub(super) fn new(embed: Box<dyn Embed>) -> Side {
        let dims = embed.dims();
        assert!(
            dims.last().is_some_and(|&w| w > 0),
            "an embedding model names its width, the last of its dims: {dims:?}"
        );
        Side {
            prompts: embed.prompts(),
            cache: embed.cache(),
            media: embed.media(),
            audio: embed.audio(),
            tok: embed.tokenizer(),
            name: embed.name(),
            path: embed.path(),
            ctx: embed.ctx(),
            dims,
            embed: Mutex::new(embed),
        }
    }

    /// The pooled vector's width, the last of the model's widths.
    pub(super) fn width(&self) -> usize {
        self.dims[self.dims.len() - 1]
    }
}

/// The answer to one embeddings request; `oai` on `/v1/embeddings`.
pub(super) fn post(state: &State, req: &Request, w: &mut TcpStream, oai: bool) -> io::Result<bool> {
    match answer(state, req, oai) {
        Ok(v) => send_json(w, req, 200, &v),
        Err(e) => send_error(w, req, &e),
    }
}

fn answer(state: &State, req: &Request, oai: bool) -> Result<Value, ApiError> {
    let Some(side) = &state.side else {
        return Err(ApiError {
            retry_after: false,
            code: 501,
            kind: "not_supported_error",
            message: NOT_SERVED.to_owned(),
        });
    };
    open(state)?;
    let b = body(req)?;
    let request =
        request::Request::parse(&b, &side.name, &side.dims, oai).map_err(ApiError::from)?;
    let inputs = prepare(side, &state.video, &request)?;
    let pooled = run(state, side, &inputs)?;
    let n_tokens = inputs.iter().map(|i| i.ids.len()).sum();
    request.answer(&pooled, n_tokens).map_err(ApiError::from)
}

/// The server still takes a pass: no failure ended it and no orderly stop began, else the 503
/// naming why.
fn open(state: &State) -> Result<(), ApiError> {
    if let Some(reason) = state.fatal() {
        return Err(dead_engine(&reason));
    }
    if let Some(cause) = state.stopping() {
        return Err(stopping_gate(&cause));
    }
    Ok(())
}

/// The request's ids and prepared media: an array with no id at its top level is one input an element,
/// anything else one input, each read as a `/completion` prompt with the special tokens on. No
/// input, an input of no id and one of more ids than the model's context are refused by name.
struct Input {
    ids: Vec<u32>,
    images: Vec<media::MediaFeed>,
    audio: Vec<crate::embed::AudioFeed>,
}

pub use ::embed::input::{Assembled, InputPart};

/// Tokenize ordered parts with the side model's special tokens.
pub fn assemble<'a>(
    parts: impl Iterator<Item = Result<InputPart<'a>, String>>,
    tok: &dyn Tokenizer,
    ctx: usize,
) -> Result<Assembled, String> {
    ::embed::input::assemble(
        parts,
        |text| tok.encode(text),
        tok.bos_to_prepend()?,
        tok.add_eos().then(|| tok.eos()),
        tok.bos(),
        ctx,
    )
}

#[cfg(test)]
fn input(side: &Side, key: &str, v: &Value) -> Result<Input, ApiError> {
    let mut images = inline_images(side, std::iter::once(v))?;
    input_with(side, &VideoSupport::default(), key, v, &mut images)
}

fn inline_images<'a>(
    side: &Side,
    inputs: impl IntoIterator<Item = &'a Value>,
) -> Result<std::vec::IntoIter<media::DecodedImage>, ApiError> {
    let urls = inputs
        .into_iter()
        .filter(|_| side.media.is_some())
        .filter_map(|v| v.get("content").and_then(Value::as_array))
        .flatten()
        .filter_map(|part| match media::parse_part(part) {
            Ok(media::Part::Image(url)) => Some(url),
            _ => None,
        });
    media::decode_images(urls)
        .map(Vec::into_iter)
        .map_err(|e| invalid(e.to_string()))
}

fn input_with(
    side: &Side,
    video: &VideoSupport,
    key: &str,
    v: &Value,
    decoded_images: &mut std::vec::IntoIter<media::DecodedImage>,
) -> Result<Input, ApiError> {
    if let Some(parts) = v
        .as_object()
        .and_then(|o| o.get("content"))
        .and_then(Value::as_array)
    {
        let mut decoded = match (side.media.as_ref(), parts.first()) {
            (Some(model), Some(first))
                if matches!(media::parse_part(first), Ok(media::Part::Text(prefix))
                    if model.embedding_video().is_some_and(|(text, _)| text == prefix)) =>
            {
                Some(
                    media::prepare_video(&parts[1..], &**model, decoded_images)
                        .map_err(|e| invalid(e.to_string()))?
                        .into_iter(),
                )
            }
            _ => None,
        };
        let mut images = Vec::new();
        let mut audio = Vec::new();
        let mut order = Vec::new();
        // A clip is planned under the context the input's other media leave, so with a clip in the
        // input they are prepared first. Without one each is prepared as the assembly reaches it,
        // and an input past the context is refused before its later media are.
        let mut fixed = std::collections::BTreeMap::new();
        let mut spent = 0;
        if parts
            .iter()
            .any(|p| matches!(media::parse_part(p), Ok(media::Part::Video(_))))
        {
            for (at, part) in parts.iter().enumerate() {
                let Some(f) =
                    prepare_fixed(side, part, &mut decoded, decoded_images).map_err(invalid)?
                else {
                    continue;
                };
                spent += video::group_positions(f.positions, f.around(&*side.tok));
                if spent > side.ctx {
                    return Err(invalid(format!(
                        "{} input exceeds the embedding context",
                        f.kind()
                    )));
                }
                fixed.insert(at, f);
            }
        }
        let clips = clips(side, video, parts, spent)?;
        let ordered = parts.iter().enumerate().flat_map(|(at, part)| {
            if let Some(clip) = clips.get(&at) {
                let (begin, end) = side
                    .media
                    .as_ref()
                    .and_then(|m| m.embedding_delimiters())
                    .expect("a clip was prepared only for a model with delimiters");
                return clip
                    .pieces
                    .iter()
                    .map(|piece| match piece {
                        Piece::Text(text) => Ok(InputPart::Text(text)),
                        Piece::Group(n) => {
                            let group = &clip.groups[*n];
                            order.push((false, images.len()));
                            images.push(media::MediaFeed {
                                at: 0,
                                key: group.key,
                                prepared: Arc::clone(&group.prepared),
                            });
                            Ok(InputPart::Media {
                                begin,
                                end,
                                positions: group.prepared.span_len,
                                kind: "image",
                            })
                        }
                    })
                    .collect::<Vec<_>>();
            }
            let f = match fixed.remove(&at) {
                Some(f) => Ok(Some(f)),
                None => prepare_fixed(side, part, &mut decoded, decoded_images),
            };
            vec![match f {
                Ok(Some(f)) => {
                    match &f.feed {
                        Feed::Image(feed) => {
                            order.push((false, images.len()));
                            images.push(feed.clone());
                        }
                        Feed::Audio(feed) => {
                            order.push((true, audio.len()));
                            audio.push(feed.clone());
                        }
                    }
                    Ok(InputPart::Media {
                        begin: f.begin,
                        end: f.end,
                        positions: f.positions,
                        kind: f.kind(),
                    })
                }
                Ok(None) => match media::parse_part(part) {
                    Ok(media::Part::Text(text)) => Ok(InputPart::Text(text)),
                    Ok(_) => Err("a clip part was not prepared".to_owned()),
                    Err(e) => Err(e.to_string()),
                },
                Err(e) => Err(e),
            }]
        });
        let Assembled { ids, spans } = assemble(ordered, &*side.tok, side.ctx).map_err(invalid)?;
        for ((is_audio, index), span) in order.into_iter().zip(spans) {
            if is_audio {
                audio[index].at = span.start;
            } else {
                images[index].at = span.start;
            }
        }
        Ok(Input { ids, images, audio })
    } else {
        Ok(Input {
            ids: prompt_ids(&*side.tok, key, v, true)?,
            images: Vec::new(),
            audio: Vec::new(),
        })
    }
}

/// A still image, a frame of a decoded video or an audio span of an input, prepared: the
/// delimiters around it, the positions it takes and its feed. [`assemble`] charges it those
/// positions and the delimiters, which a clip planned beside it must leave.
struct Fixed<'a> {
    begin: &'a str,
    end: &'a str,
    positions: usize,
    feed: Feed,
}

enum Feed {
    Image(media::MediaFeed),
    Audio(crate::embed::AudioFeed),
}

impl Fixed<'_> {
    /// The positions of the delimiters around the medium.
    fn around(&self, tok: &dyn Tokenizer) -> usize {
        tok.encode(self.begin).len() + tok.encode(self.end).len()
    }

    fn kind(&self) -> &'static str {
        match self.feed {
            Feed::Image(_) => "image",
            Feed::Audio(_) => "audio",
        }
    }
}

/// The medium a part is, prepared: an audio span, or an image (the next frame of `decoded` when the
/// input is a decoded video, else loaded and prepared). `None` for any other part.
fn prepare_fixed<'a>(
    side: &'a Side,
    part: &Value,
    decoded: &mut Option<std::vec::IntoIter<media::MediaFeed>>,
    decoded_images: &mut std::vec::IntoIter<media::DecodedImage>,
) -> Result<Option<Fixed<'a>>, String> {
    if part.get("type").and_then(Value::as_str) == Some("input_audio") {
        let a = part
            .get("input_audio")
            .ok_or("input_audio needs an input_audio object")?;
        if a.get("format").and_then(Value::as_str) != Some("f32") {
            return Err("encoded audio is not supported; send base64 little-endian f32 PCM with format=f32, sample_rate=16000 and channels=1".into());
        }
        let model = side
            .audio
            .as_ref()
            .ok_or("audio input needs a gemma4a --embd-mmproj")?;
        let (begin, end, sample_rate) = model
            .embedding_audio()
            .ok_or("the side projector has no audio delimiters")?;
        if a.get("sample_rate").and_then(Value::as_u64) != Some(sample_rate as u64)
            || a.get("channels").and_then(Value::as_u64) != Some(1)
        {
            return Err("f32 PCM needs sample_rate=16000 and channels=1".into());
        }
        let data = a
            .get("data")
            .and_then(Value::as_str)
            .ok_or("f32 PCM needs base64 data")?;
        if data.is_empty() {
            return Err("f32 PCM needs at least two samples".into());
        }
        let bytes = media::base64(data).map_err(|e| e.to_string())?;
        if bytes.len() % 4 != 0 {
            return Err("f32 PCM byte count must be a multiple of four".into());
        }
        let pcm: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let chunks = model.prepare_audio(&pcm).map_err(|e| e.to_string())?;
        return Ok(Some(Fixed {
            begin,
            end,
            positions: chunks.iter().map(|c| c.span_len).sum(),
            feed: Feed::Audio(crate::embed::AudioFeed {
                at: 0,
                chunks: Arc::new(chunks),
            }),
        }));
    }
    let Ok(media::Part::Image(_)) = media::parse_part(part) else {
        return Ok(None);
    };
    let model = side
        .media
        .as_ref()
        .ok_or("image input needs --embd-mmproj")?;
    let (begin, end) = model
        .embedding_delimiters()
        .ok_or("the side projector has no embedding delimiters")?;
    let feed = if let Some(frames) = decoded {
        frames
            .next()
            .ok_or("video frame count differs from input")?
    } else {
        let image = decoded_images
            .next()
            .ok_or("image count differs from input")?;
        media::MediaFeed {
            at: 0,
            key: image.key,
            prepared: Arc::new(model.prepare(&image.image).map_err(|e| e.to_string())?),
        }
    };
    Ok(Some(Fixed {
        begin,
        end,
        positions: feed.prepared.span_len,
        feed: Feed::Image(feed),
    }))
}

/// A clip of an input, probed, planned, decoded and prepared: the side model's text for it and
/// its groups' feeds.
struct EmbedClip {
    pieces: Vec<Piece>,
    groups: Vec<ReadyGroup>,
}

/// The input's clips by their part's place, each planned under the context the input leaves: the
/// context less the input's text and its BOS and EOS, less the `spent` positions of its images and
/// audio ([`Fixed`]), and less what the clips before it spent. A clip spends its tokens, its
/// delimiters (a pair a group) and the model's text for it ([`video::plan_clip_within`]), the
/// positions [`assemble`] charges it. Done before the input's text is tokenized, because the text
/// the model writes for a clip holds what the plan chose; then decoded and prepared.
fn clips(
    side: &Side,
    video: &VideoSupport,
    parts: &[Value],
    spent: usize,
) -> Result<std::collections::BTreeMap<usize, EmbedClip>, ApiError> {
    let mut found = std::collections::BTreeMap::new();
    let mut left: Option<usize> = None;
    for (at, part) in parts.iter().enumerate() {
        let Ok(media::Part::Video(clip)) = media::parse_part(part) else {
            continue;
        };
        let model = side
            .media
            .as_ref()
            .ok_or_else(|| invalid("video input needs --embd-mmproj"))?;
        let (Some(_), Some((begin, end))) = (model.video_rule(), model.embedding_delimiters())
        else {
            return Err(invalid(media::MediaError::NoVideoRule.to_string()));
        };
        let decoder = video.decoder().map_err(|e| invalid(e.to_string()))?;
        let room = *left.get_or_insert_with(|| {
            let text: usize = parts
                .iter()
                .filter_map(|p| match media::parse_part(p) {
                    Ok(media::Part::Text(t)) => Some(side.tok.encode(t).len()),
                    _ => None,
                })
                .sum();
            let marks = usize::from(side.tok.add_bos()) + usize::from(side.tok.add_eos());
            side.ctx.saturating_sub(text + marks + spent)
        });
        let bytes = media::base64(clip.data).map_err(|e| invalid(e.to_string()))?;
        let around = side.tok.encode(begin).len() + side.tok.encode(end).len();
        let (planned, spent) = video::plan_clip_within(
            &**model,
            decoder,
            Clip::new(bytes, clip.format),
            room,
            video.thread_count(),
            |p| p.positions(|t| side.tok.encode(t).len(), around),
        )
        .map_err(|e| invalid(e.to_string()))?;
        left = Some(room - spent);
        let groups = planned
            .prepare(&**model)
            .map_err(|e| invalid(e.to_string()))?;
        found.insert(
            at,
            EmbedClip {
                pieces: planned.pieces().to_vec(),
                groups,
            },
        );
    }
    Ok(found)
}

fn prepare(
    side: &Side,
    video: &VideoSupport,
    request: &request::Request<'_>,
) -> Result<Vec<Input>, ApiError> {
    let prefix = request
        .prompt_name
        .map(|name| {
            side.prompts.get(name).ok_or_else(|| {
                invalid(format!(
                    "unknown prompt_name {name:?}; available names: {:?}",
                    side.prompts.keys().collect::<Vec<_>>()
                ))
            })
        })
        .transpose()?;
    let mut images = inline_images(side, request.inputs.iter().copied())?;
    let inputs = request
        .inputs
        .iter()
        .map(|v| {
            if let Some(prefix) = prefix {
                let text = v.as_str().ok_or_else(|| {
                    invalid("prompt_name requires text input (a string or an array of strings)")
                })?;
                input_with(
                    side,
                    video,
                    request.key,
                    &Value::String(format!("{prefix}{text}")),
                    &mut images,
                )
            } else {
                input_with(side, video, request.key, v, &mut images)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    for input in &inputs {
        request::check_tokens(input.ids.len(), side.ctx).map_err(ApiError::from)?;
    }
    Ok(inputs)
}

#[cfg(test)]
fn inputs(side: &Side, key: &str, value: &Value) -> Result<Vec<Input>, ApiError> {
    inputs_with(side, &VideoSupport::default(), key, value)
}

#[cfg(test)]
fn inputs_with(
    side: &Side,
    video: &VideoSupport,
    key: &str,
    value: &Value,
) -> Result<Vec<Input>, ApiError> {
    let body = serde_json::Map::from_iter([(key.to_owned(), value.clone())]);
    let request =
        request::Request::parse(&body, &side.name, &side.dims, true).map_err(ApiError::from)?;
    prepare(side, video, &request)
}

/// A vector that is not the model's width or not finite is fatal ([`fail`]).
fn vector_ok(state: &State, side: &Side, v: &[f32]) -> Result<(), ApiError> {
    let width = side.width();
    if v.len() != width {
        let n = v.len();
        return Err(fail(
            state,
            side,
            format!("the embedding model gave {n} values, not its width {width}"),
        ));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(fail(
            state,
            side,
            "the embedding model gave a value that is not finite".into(),
        ));
    }
    Ok(())
}

/// Cache `v` when `key` is set. The model's lock is the caller's; this takes the cache's.
fn store(state: &State, side: &Side, key: Option<[u8; 32]>, v: &[f32]) -> Result<(), ApiError> {
    if let (Some(cache), Some(key)) = (&side.cache, key) {
        cache
            .lock()
            .map_err(|_| fail(state, side, "the embedding cache's lock is poisoned".into()))?
            .insert(key, crate::embed_cache::Value::Vector(v.to_vec()));
    }
    Ok(())
}

fn model_poisoned(state: &State, side: &Side) -> ApiError {
    fail(
        state,
        side,
        "the embedding model's lock is poisoned: a request panicked while it held it".into(),
    )
}

fn model_panicked(state: &State, side: &Side, panic: Box<dyn std::any::Any + Send>) -> ApiError {
    let what = panic_text(&*panic);
    fail(state, side, format!("the embedding model panicked: {what}"))
}

/// Every input's pooled vector, through the side model under its lock. Text inputs with no image
/// and no audio share one [`Embed::embed_batch`]; an image or audio input keeps its own pass.
/// Each call is after [`open`] holds. A model failure, a panic in it, its lock poisoned, and a
/// vector that is not its width or not finite are fatal ([`fail`]).
fn run(state: &State, side: &Side, inputs: &[Input]) -> Result<Vec<Vec<f32>>, ApiError> {
    let mut pooled = vec![None; inputs.len()];
    let mut pending: Vec<(usize, Option<[u8; 32]>)> = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        open(state)?;
        let key = side
            .cache
            .as_ref()
            .filter(|_| input.audio.is_empty())
            .and_then(|_| crate::embed_cache::vector_key(&input.ids, &input.images));
        if let (Some(cache), Some(hit)) = (&side.cache, key) {
            let mut cache = cache
                .lock()
                .map_err(|_| fail(state, side, "the embedding cache's lock is poisoned".into()))?;
            if let Some(crate::embed_cache::Value::Vector(v)) = cache.get(&hit) {
                open(state)?;
                pooled[i] = Some(v);
                continue;
            }
        }
        if input.images.is_empty() && input.audio.is_empty() {
            pending.push((i, key));
        }
    }
    if !pending.is_empty() {
        let Ok(mut model) = side.embed.lock() else {
            return Err(model_poisoned(state, side));
        };
        open(state)?;
        let batch: Vec<&[u32]> = pending
            .iter()
            .map(|&(i, _)| inputs[i].ids.as_slice())
            .collect();
        let vectors = match panic::catch_unwind(AssertUnwindSafe(|| model.embed_batch(&batch))) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(fail(state, side, e.0)),
            Err(p) => return Err(model_panicked(state, side, p)),
        };
        if vectors.len() != pending.len() {
            let n = vectors.len();
            let want = pending.len();
            return Err(fail(
                state,
                side,
                format!("the embedding model gave {n} vectors, not {want}"),
            ));
        }
        for ((i, key), v) in pending.into_iter().zip(vectors) {
            vector_ok(state, side, &v)?;
            store(state, side, key, &v)?;
            pooled[i] = Some(v);
        }
    }
    for (i, input) in inputs.iter().enumerate() {
        if pooled[i].is_some() {
            continue;
        }
        open(state)?;
        let key = side
            .cache
            .as_ref()
            .filter(|_| input.audio.is_empty())
            .and_then(|_| crate::embed_cache::vector_key(&input.ids, &input.images));
        let Ok(mut model) = side.embed.lock() else {
            return Err(model_poisoned(state, side));
        };
        open(state)?;
        let v = match panic::catch_unwind(AssertUnwindSafe(|| {
            model.embed_media(&input.ids, &input.images, &input.audio)
        })) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(fail(state, side, e.0)),
            Err(p) => return Err(model_panicked(state, side, p)),
        };
        vector_ok(state, side, &v)?;
        store(state, side, key, &v)?;
        pooled[i] = Some(v);
    }
    pooled
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.ok_or_else(|| {
                fail(
                    state,
                    side,
                    format!("the embedding model gave no vector for input {i}"),
                )
            })
        })
        .collect()
}

/// Ends the server on the side model's `error`, as an engine's failure does
/// ([`crate::worker::Shared::record_fatal`]: the first failure is the one it ends with), and
/// answers the request that met it with a 500.
fn fail(state: &State, side: &Side, error: String) -> ApiError {
    let adopted = state.shared.record_fatal(EngineFailure {
        engine: format!("the embedding model {}", side.name),
        error: error.clone(),
    });
    state.shared.end_on(adopted);
    ApiError {
        retry_after: false,
        code: 500,
        kind: "server_error",
        message: error,
    }
}

#[cfg(test)]
#[path = "arch/gemma4a.rs"]
mod audio_tests;
#[cfg(test)]
#[path = "arch/gemma4v.rs"]
mod image_tests;

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::super::testserve::{mock_config, roundtrip};
    use super::*;
    use crate::api::{End, relock};
    use crate::mock::{MockEmbed, MockFail, MockTokenizer, Passes};
    use crate::{MockEngine, Server};

    /// The mock engine with `side` beside it, or none: its address, its state, its log of passes
    /// and its end.
    fn serve(side: Option<MockEmbed>) -> (SocketAddr, Arc<State>, Passes, mpsc::Receiver<End>) {
        let mut server = Server::bind("127.0.0.1:0", Box::new(MockEngine::new(64)), mock_config())
            .expect("bind");
        let calls = side
            .as_ref()
            .map(|s| Arc::clone(&s.calls))
            .unwrap_or_default();
        if let Some(side) = side {
            server = server.with_side(Box::new(side));
        }
        let addr = server.local_addr().expect("addr");
        let (state, ended) = server.start().expect("start");
        (addr, state, calls, ended)
    }

    fn post(addr: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
        let (status, text) = roundtrip(addr, "POST", path, &[], &body.to_string());
        (status, serde_json::from_str(&text).expect("a JSON answer"))
    }

    fn id(text: &str) -> u32 {
        MockTokenizer.encode(text)[0]
    }

    /// The mock's pooled vector of `ids`, normalized as `norm` names.
    fn vector(ids: &[u32], width: usize, norm: Norm) -> Vec<f32> {
        let n = ids.len() as f32;
        normalize(&[n, 1.0, 0.0, -n][..width], norm).expect("a norm")
    }

    fn floats(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("an array")
            .iter()
            .map(|x| x.as_f64().expect("a number") as f32)
            .collect()
    }

    #[test]
    fn pooled_cache_is_opt_in_and_preserves_every_cut_and_norm() {
        let (_, state, _, _) = serve(None);
        let model = MockEmbed::new(32);
        let calls = model.calls.clone();
        let mut side = Side::new(Box::new(model));
        assert!(side.cache.is_none());
        let input = Input {
            ids: vec![1, 2, 3],
            images: vec![],
            audio: vec![],
        };
        let get = |side: &Side, input: &Input| {
            run(&state, side, std::slice::from_ref(input))
                .unwrap_or_else(|e| panic!("{}", e.message))
        };
        let first = get(&side, &input);
        assert_eq!(first, get(&side, &input));
        assert_eq!(
            calls.lock().unwrap().len(),
            2,
            "disabled cache runs every pass"
        );
        side.cache = crate::embed_cache::Cache::shared(4096);
        let miss = get(&side, &input);
        let hit = get(&side, &input);
        assert_eq!(calls.lock().unwrap().len(), 3, "the hit skips the pass");
        for width in [2, 4] {
            for norm in [-1, 0, 1, 2, 3] {
                let bits = |v: &[f32]| {
                    normalize(&v[..width], Norm::of(norm).unwrap())
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&first[0]), bits(&miss[0]));
                assert_eq!(bits(&miss[0]), bits(&hit[0]));
            }
        }
        get(
            &side,
            &Input {
                ids: vec![3, 2, 1],
                images: vec![],
                audio: vec![],
            },
        );
        assert_eq!(calls.lock().unwrap().len(), 4, "different ids miss");
        fail(&state, &side, "the model ended".into());
        assert!(
            run(&state, &side, &[input]).is_err(),
            "a cached hit cannot bypass a fatal model error"
        );
    }

    #[test]
    fn a_cached_request_waiting_for_the_cache_observes_the_server_end() {
        let mut codes = Vec::new();
        for failed in [true, false] {
            let (_, state, _, _) = serve(None);
            let mut side = Side::new(Box::new(MockEmbed::new(32)));
            side.cache = crate::embed_cache::Cache::shared(4096);
            let input = Input {
                ids: vec![1, 2, 3],
                images: vec![],
                audio: vec![],
            };
            assert!(run(&state, &side, std::slice::from_ref(&input)).is_ok());
            let held = side.cache.as_ref().unwrap().lock().unwrap();
            thread::scope(|scope| {
                let (entered, ready) = mpsc::channel();
                let (finished, done) = mpsc::channel();
                let waiting_state = &state;
                let waiting_side = &side;
                let waiting = scope.spawn(move || {
                    assert!(open(waiting_state).is_ok());
                    entered.send(()).unwrap();
                    let result = run(waiting_state, waiting_side, &[input]);
                    finished.send(()).unwrap();
                    result.map_or_else(|e| e.code, |_| 200)
                });
                ready.recv_timeout(Duration::from_secs(5)).unwrap();
                assert_eq!(
                    done.recv_timeout(Duration::from_millis(100)),
                    Err(mpsc::RecvTimeoutError::Timeout),
                    "the request must wait behind the cache lock"
                );
                if failed {
                    state
                        .shared
                        .end_on(state.shared.record_fatal(EngineFailure {
                            engine: "mock".to_owned(),
                            error: "the engine failed".to_owned(),
                        }));
                } else {
                    *relock(&state.shared.stopping) = Some("a signal".to_owned());
                }
                drop(held);
                codes.push(waiting.join().unwrap());
            });
        }
        assert_eq!(codes, [503, 503], "fatal and stopping cache hits");
    }

    #[test]
    fn no_side_model_is_a_501_on_every_route() {
        let (addr, ..) = serve(None);
        for path in ["/v1/embeddings", "/embeddings", "/embedding"] {
            let (status, v) = post(addr, path, &json!({ "input": "a" }));
            assert_eq!(status, 501, "{path}: {v}");
            assert_eq!(v["error"]["type"], "not_supported_error");
            assert_eq!(v["error"]["message"], NOT_SERVED);
        }
    }

    /// llama-server's two answers: OpenAI's list on `/v1/embeddings` (base64 as the vector's
    /// little-endian f32), its own array on the other routes and wherever `content` is the field,
    /// and the side model listed second.
    #[test]
    fn each_route_answers_its_shape() {
        let (addr, _, calls, _) = serve(Some(MockEmbed::new(32)));
        for model in [json!("mock"), json!("unknown"), json!(42)] {
            let (status, v) = post(
                addr,
                "/v1/embeddings",
                &json!({ "input": "a", "model": model }),
            );
            assert_eq!(status, 400, "{model}: {v}");
            assert_eq!(v["error"]["type"], "invalid_request_error");
        }
        assert!(calls.lock().expect("calls").is_empty());
        let ids = [
            MockTokenizer.bos().expect("a BOS"),
            id("a"),
            MockTokenizer.eos(),
        ];
        let want = vector(&ids, 4, Norm::default());
        let (status, v) = post(addr, "/v1/embeddings", &json!({ "input": "a" }));
        assert_eq!(status, 200, "{v}");
        assert_eq!(v["object"], "list");
        assert_eq!(v["model"], "mock-embd");
        assert_eq!(v["usage"], json!({ "prompt_tokens": 3, "total_tokens": 3 }));
        assert_eq!(v["data"][0]["object"], "embedding");
        assert_eq!(v["data"][0]["index"], 0);
        assert_eq!(floats(&v["data"][0]["embedding"]), want);

        let (_, v) = post(
            addr,
            "/v1/embeddings",
            &json!({ "input": ["a", "a"], "encoding_format": "base64", "model": "mock-embd" }),
        );
        assert_eq!(v["model"], "mock-embd");
        assert_eq!(v["usage"]["prompt_tokens"], 6);
        let bytes: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(v["data"][1]["index"], 1);
        assert_eq!(v["data"][1]["encoding_format"], "base64");
        assert_eq!(v["data"][1]["embedding"], media::encode_base64(&bytes));

        for (path, body) in [
            ("/embedding", json!({ "input": "a" })),
            ("/embeddings", json!({ "content": "a" })),
            ("/embedding", json!({ "input": "a", "model": "unknown" })),
            ("/embedding", json!({ "input": "a", "model": 42 })),
            ("/embeddings", json!({ "content": "a", "model": "mock" })),
            ("/embeddings", json!({ "content": "a", "model": 42 })),
            (
                "/v1/embeddings",
                json!({ "content": "a", "encoding_format": "base64" }),
            ),
        ] {
            let (status, v) = post(addr, path, &body);
            assert_eq!(status, 200, "{path} {body}: {v}");
            assert_eq!(v[0]["index"], 0, "{path} {body}");
            assert_eq!(floats(&v[0]["embedding"][0]), want, "{path} {body}");
        }
        let (status, v) = post(
            addr,
            "/v1/embeddings",
            &json!({ "input": "a", "encoding_format": "f16" }),
        );
        assert_eq!(status, 400, "{v}");
    }

    /// Discovery lists every seated model and reports the side model's file only when present.
    #[test]
    fn discovery_names_the_seated_models() {
        for side in [None, Some(MockEmbed::new(32))] {
            let has_side = side.is_some();
            let (addr, state, calls, _) = serve(side);
            let mut data = vec![json!({
                "id": "mock",
                "object": "model",
                "created": state.start_unix,
                "owned_by": "bloomery",
                "meta": { "n_vocab": state.info.n_vocab, "n_ctx_train": 64 },
                "max_model_len": 64,
            })];
            let mut names = vec![json!({ "name": "mock", "model": "mock" })];
            if has_side {
                data.push(json!({
                    "id": "mock-embd",
                    "object": "model",
                    "created": state.start_unix,
                    "owned_by": "bloomery",
                    "meta": { "n_ctx": 32, "n_embd": 4, "prompt_names": [] },
                    "max_model_len": 32,
                }));
                names.push(json!({ "name": "mock-embd", "model": "mock-embd" }));
            }
            for path in ["/v1/models", "/models"] {
                let (status, body) = roundtrip(addr, "GET", path, &[], "");
                assert_eq!(status, 200, "{path}: {body}");
                let v: Value = serde_json::from_str(&body).expect("models JSON");
                assert_eq!(
                    v,
                    json!({ "object": "list", "data": data, "models": names })
                );
            }
            let (status, body) = roundtrip(addr, "GET", "/props", &[], "");
            assert_eq!(status, 200, "{body}");
            let v: Value = serde_json::from_str(&body).expect("props JSON");
            assert_eq!(v["model_alias"], "mock");
            assert_eq!(v["model_path"], state.model_path);
            if has_side {
                assert_eq!(
                    v["embedding_model"],
                    json!({ "alias": "mock-embd", "path": "/models/mock-embd.gguf" })
                );
            } else {
                assert!(v.get("embedding_model").is_none(), "{v}");
            }
            assert!(calls.lock().expect("calls").is_empty());
        }
    }

    /// An input of a vocabulary that names no BOS starts with the text where the vocabulary adds
    /// none, is refused by name where it adds a BOS it does not name, and has no filler for a media
    /// span either way.
    #[test]
    fn a_vocabulary_with_no_bos_assembles_text_only() {
        use crate::mock::NoBosTokenizer;
        use ::embed::input::InputPart;

        let text = || [Ok(InputPart::Text("a"))].into_iter();
        let media = || {
            [Ok(InputPart::Media {
                begin: "<",
                end: ">",
                positions: 2,
                kind: "image",
            })]
            .into_iter()
        };
        let ids = assemble(text(), &NoBosTokenizer(false), 16).unwrap().ids;
        assert_eq!(ids, MockTokenizer.encode("a"));
        let e = assemble(text(), &NoBosTokenizer(true), 16)
            .err()
            .expect("refused");
        assert!(
            e.contains("adds a BOS") && e.contains("names no BOS token"),
            "{e}"
        );
        let e = assemble(media(), &NoBosTokenizer(false), 16)
            .err()
            .expect("refused");
        assert!(e.contains("image span has no filler token id"), "{e}");
    }

    /// Which inputs a request holds and the ids each passes, as llama-server splits and
    /// tokenizes them, and the refusals of an input with nothing in it, an object input, and one
    /// of more ids than the context.
    #[test]
    fn inputs_split_and_tokenize_as_llama_server() {
        let (addr, _, calls, _) = serve(Some(MockEmbed::new(4)));
        let (bos, eos, a, b) = (
            MockTokenizer.bos().expect("a BOS"),
            MockTokenizer.eos(),
            id("a"),
            id("b"),
        );
        for (input, want) in [
            (json!("ab"), vec![vec![bos, a, b, eos]]),
            (
                json!(["a", "b"]),
                vec![vec![bos, a, eos], vec![bos, b, eos]],
            ),
            (json!([7, "a"]), vec![vec![7, a]]),
            (json!(["a", 7]), vec![vec![bos, a, eos, 7]]),
            (json!([[7, 8], "a"]), vec![vec![7, 8], vec![bos, a, eos]]),
        ] {
            calls.lock().expect("calls").clear();
            let (status, v) = post(addr, "/v1/embeddings", &json!({ "input": input }));
            assert_eq!(status, 200, "{input}: {v}");
            assert_eq!(*calls.lock().expect("calls"), want, "{input}");
        }
        calls.lock().expect("calls").clear();
        for (body, message) in [
            (json!({ "input": [] }), "\"input\" must not be empty"),
            (json!({ "input": [[]] }), "Input content cannot be empty"),
            (json!({}), "\"input\" or \"content\" must be provided"),
            (
                json!({ "input": { "prompt_string": "a" } }),
                "input must be a string or an array of strings and token ids, not an object",
            ),
            (
                json!({ "input": [1, 1_000_000] }),
                "invalid token id 1000000",
            ),
            (
                json!({ "input": "abcd" }),
                "input (6 tokens) is larger than the max context size (4 tokens)",
            ),
        ] {
            let (status, v) = post(addr, "/v1/embeddings", &body);
            assert_eq!(status, 400, "{body}: {v}");
            let got = v["error"]["message"].as_str().expect("a message");
            assert!(got.starts_with(message), "{body}: {got}");
        }
        assert!(calls.lock().expect("calls").is_empty(), "no refusal passes");
    }

    /// `embd_normalize` picks the normalization, `dimensions` cuts before it, and a value of
    /// either that names nothing, or a norm f32 cannot scale by, is a 400.
    #[test]
    fn prompt_name_is_validated_on_both_routes() {
        let mut side = MockEmbed::new(128);
        side.prompts = [
            ("query".into(), "task: search result | query: ".into()),
            ("document".into(), "title: none | text: ".into()),
            ("empty".into(), String::new()),
        ]
        .into();
        let prefixes = side.prompts.clone();
        let (addr, _, calls, _) = serve(Some(side));
        for path in ["/v1/embeddings", "/embedding", "/embeddings"] {
            for (name, prefix) in &prefixes {
                for key in ["input", "content"] {
                    calls.lock().expect("calls").clear();
                    let (status, v) = post(
                        addr,
                        path,
                        &json!({
                            key: ["a", "검색"], "prompt_name": name
                        }),
                    );
                    assert_eq!(status, 200, "{path}: {v}");
                    let want: Vec<_> = ["a", "검색"]
                        .iter()
                        .map(|text| {
                            let mut ids = vec![MockTokenizer.bos().expect("a BOS")];
                            ids.extend(MockTokenizer.encode(&format!("{prefix}{text}")));
                            ids.push(MockTokenizer.eos());
                            ids
                        })
                        .collect();
                    assert_eq!(*calls.lock().expect("calls"), want);
                }
            }
            calls.lock().expect("calls").clear();
            let (status, v) = post(addr, path, &json!({"input": "a"}));
            assert_eq!(status, 200, "{v}");
            assert_eq!(
                *calls.lock().expect("calls"),
                vec![vec![
                    MockTokenizer.bos().expect("a BOS"),
                    id("a"),
                    MockTokenizer.eos()
                ]]
            );
            calls.lock().expect("calls").clear();
            for name in [json!("missing"), json!("Query"), json!(7), Value::Null] {
                let (status, v) = post(
                    addr,
                    path,
                    &json!({
                        "input": "a", "prompt_name": name
                    }),
                );
                assert_eq!(status, 400, "{path}: {v}");
                assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
                assert!(
                    v["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("prompt_name"),
                    "{v}"
                );
            }
            for input in [
                json!([1, 2]),
                json!(["a", 7]),
                json!({"content": []}),
                json!("a".repeat(100)),
            ] {
                let (status, v) =
                    post(addr, path, &json!({"input": input, "prompt_name": "query"}));
                assert_eq!(status, 400, "{path}: {v}");
            }
            assert!(calls.lock().expect("calls").is_empty());
        }
        let (_, body) = roundtrip(addr, "GET", "/v1/models", &[], "");
        let v: Value = serde_json::from_str(&body).expect("models JSON");
        assert_eq!(
            v["data"][1]["meta"]["prompt_names"],
            json!(["document", "empty", "query"])
        );
    }

    #[test]
    fn normalization_and_the_cut() {
        let (addr, ..) = serve(Some(MockEmbed::new(32)));
        let ids = [
            MockTokenizer.bos().expect("a BOS"),
            id("a"),
            MockTokenizer.eos(),
        ];
        for (body, width, norm) in [
            (json!({ "input": "a", "embd_normalize": -1 }), 4, -1),
            (json!({ "input": "a", "embd_normalize": 0 }), 4, 0),
            (json!({ "input": "a", "dimensions": 2 }), 2, 2),
            (
                json!({ "input": "a", "dimensions": 2, "embd_normalize": 1 }),
                2,
                1,
            ),
        ] {
            let (status, v) = post(addr, "/v1/embeddings", &body);
            assert_eq!(status, 200, "{body}: {v}");
            let want = vector(&ids, width, Norm::of(norm).expect("norm"));
            assert_eq!(floats(&v["data"][0]["embedding"]), want, "{body}");
        }
        for (body, message) in [
            (
                json!({ "input": "a", "dimensions": 3 }),
                "dimensions 3: mock-embd cuts its vectors to 2, 4",
            ),
            (
                json!({ "input": "a", "embd_normalize": -2 }),
                "embd_normalize -2:",
            ),
            (
                json!({ "input": "a", "embd_normalize": 1000 }),
                "embd_normalize 1000: this vector's norm comes to",
            ),
            (
                json!({ "input": "a", "embd_normalize": "2" }),
                "embd_normalize must be an integer",
            ),
        ] {
            let (status, v) = post(addr, "/v1/embeddings", &body);
            assert_eq!(status, 400, "{body}: {v}");
            let got = v["error"]["message"].as_str().expect("a message");
            assert!(got.starts_with(message), "{body}: {got}");
        }
    }

    /// A failure or a panic in the side model ends the server: the request gets a 500 naming it,
    /// `/health` a 503, and the end channel the failure.
    #[test]
    fn a_side_model_failure_ends_the_server() {
        for (fail, says) in [
            (MockFail::Error, "the mock failed"),
            (
                MockFail::Panic,
                "the embedding model panicked: the mock panicked",
            ),
            (
                MockFail::Short,
                "the embedding model gave 3 values, not its width 4",
            ),
            (
                MockFail::Nan,
                "the embedding model gave a value that is not finite",
            ),
        ] {
            let mut side = MockEmbed::new(32);
            side.fail = Some((id("!"), fail));
            let (addr, _, _, ended) = serve(Some(side));
            let (status, v) = post(addr, "/v1/embeddings", &json!({ "input": "!" }));
            assert_eq!(status, 500, "{v}");
            assert_eq!(v["error"]["message"], says);
            let (status, _) = roundtrip(addr, "GET", "/health", &[], "");
            assert_eq!(status, 503);
            let (status, _) = post(addr, "/v1/embeddings", &json!({ "input": "a" }));
            assert_eq!(status, 503);
            match ended.recv_timeout(Duration::from_secs(5)) {
                Ok(End::Engine(f)) => {
                    assert_eq!(f.engine, "the embedding model mock-embd");
                    assert_eq!(f.error, says);
                }
                _ => panic!("{fail:?}: no engine end"),
            }
        }
    }

    /// A request that waits for the model while the server fails or begins its stop gets the 503
    /// naming why, and the model runs no pass for it.
    #[test]
    fn a_request_waiting_for_the_model_runs_no_pass_once_the_server_ends() {
        for failed in [true, false] {
            let (_, state, calls, _) = serve(Some(MockEmbed::new(32)));
            let side = state.side.as_ref().expect("a side model");
            let held = side.embed.lock().expect("the model's lock");
            let (entered, ready) = mpsc::channel();
            let waiting_state = Arc::clone(&state);
            let waiting = thread::spawn(move || {
                assert!(open(&waiting_state).is_ok());
                entered.send(()).expect("entered");
                let side = waiting_state.side.as_ref().expect("a side model");
                run(
                    &waiting_state,
                    side,
                    &[Input {
                        ids: vec![id("a")],
                        images: Vec::new(),
                        audio: Vec::new(),
                    }],
                )
            });
            ready
                .recv_timeout(Duration::from_secs(5))
                .expect("before the model lock");
            if failed {
                state
                    .shared
                    .end_on(state.shared.record_fatal(EngineFailure {
                        engine: "mock".to_owned(),
                        error: "the engine failed".to_owned(),
                    }));
            } else {
                *relock(&state.shared.stopping) = Some("a signal".to_owned());
            }
            drop(held);
            let error = waiting
                .join()
                .expect("the request's thread")
                .expect_err("the server ended before the pass");
            assert_eq!(error.code, 503, "{failed}");
            assert!(
                calls.lock().expect("calls").is_empty(),
                "{failed}: a pass ran"
            );
        }
    }
}
