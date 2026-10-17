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
//! An object with a `content` array prepares image parts through the side model's media rule.
//! Delimiter text and every image span share one pass, with BOS and EOS around the whole input.
//!
//! The side model runs one request at a time. A failure in it ends the server as an engine's does:
//! the request gets a 500, `/health` and every later request a 503, and the server exits after its
//! linger. A request that waited for the model while the server failed or began its stop gets the
//! 503 and no pass, and so does the rest of an array once that happens between its passes.

use std::io;
use std::net::TcpStream;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

use super::{
    ApiError, EngineFailure, State, body, dead_engine, get_i, invalid, panic_text, prompt_ids,
    send_error, send_json, stopping_gate, wrong_type,
};
use crate::embed::{Embed, Norm, normalize};
use crate::engine::Tokenizer;
use crate::http::Request;
use crate::media;

/// What a server with no side model answers: llama-server's message, with the flags that seat one.
pub(super) const NOT_SERVED: &str =
    "This server does not support embeddings. Start it with `--embd-model` or `--embd-hf`";

/// The side model and the facts the server reads of it once, when it takes it.
pub(super) struct Side {
    embed: Mutex<Box<dyn Embed>>,
    tok: Arc<dyn Tokenizer>,
    pub(super) name: String,
    pub(super) path: String,
    pub(super) ctx: usize,
    dims: Vec<usize>,
    media: Option<media::SharedMediaModel>,
}

impl Side {
    /// `embed`'s facts read once. A model that names no width is a defect of the model, refused
    /// by name.
    pub(super) fn new(embed: Box<dyn Embed>) -> Side {
        let dims = embed.dims();
        assert!(
            dims.last().is_some_and(|&w| w > 0),
            "an embedding model names its width, the last of its dims: {dims:?}"
        );
        Side {
            media: embed.media(),
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
    if oai && let Some(model) = b.get("model").filter(|v| !v.is_null()) {
        let model = model
            .as_str()
            .ok_or_else(|| invalid("model must be a string"))?;
        if model != side.name {
            return Err(invalid(format!(
                "model {model:?}: embeddings are served by {:?}",
                side.name
            )));
        }
    }
    let (key, prompt, oai) = match (b.get("input"), b.get("content")) {
        (Some(v), _) => ("input", v, oai),
        (None, Some(v)) => ("content", v, false),
        (None, None) => return Err(invalid("\"input\" or \"content\" must be provided")),
    };
    let base64 = encoding(&b)?;
    let norm = match get_i(&b, "embd_normalize")? {
        None => Norm::default(),
        Some(p) => i32::try_from(p)
            .map_err(|e| e.to_string())
            .and_then(Norm::of)
            .map_err(|e| invalid(format!("embd_normalize {p}: {e}")))?,
    };
    let width = cut(side, &b)?;
    let inputs = inputs(side, key, prompt)?;
    let pooled = run(state, side, &inputs)?;
    let vectors = pooled
        .iter()
        .map(|v| normalize(&v[..width], norm).map_err(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    if !oai {
        return Ok(Value::Array(
            vectors
                .into_iter()
                .enumerate()
                .map(|(i, v)| json!({ "embedding": [v], "index": i }))
                .collect(),
        ));
    }
    let n_tokens: usize = inputs.iter().map(|i| i.ids.len()).sum();
    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                json!({
                    "embedding": media::encode_base64(&bytes),
                    "encoding_format": "base64",
                    "index": i,
                    "object": "embedding",
                })
            } else {
                json!({ "embedding": v, "index": i, "object": "embedding" })
            }
        })
        .collect();
    Ok(json!({
        "data": data,
        "model": side.name,
        "object": "list",
        "usage": { "prompt_tokens": n_tokens, "total_tokens": n_tokens },
    }))
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

/// `encoding_format`: `float` (or unset) or `base64`, which only the OpenAI shape applies.
fn encoding(b: &Map<String, Value>) -> Result<bool, ApiError> {
    match b.get("encoding_format") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(f)) if f == "float" => Ok(false),
        Some(Value::String(f)) if f == "base64" => Ok(true),
        Some(Value::String(_)) => Err(invalid(
            "The format to return the embeddings in. Can be either float or base64",
        )),
        Some(v) => Err(wrong_type("encoding_format", "a string", v)),
    }
}

/// The width `dimensions` cuts the pooled vector to: one of the model's widths, the full width
/// when unset.
fn cut(side: &Side, b: &Map<String, Value>) -> Result<usize, ApiError> {
    let Some(k) = get_i(b, "dimensions")? else {
        return Ok(side.width());
    };
    side.dims
        .iter()
        .copied()
        .find(|&d| i64::try_from(d).is_ok_and(|d| d == k))
        .ok_or_else(|| {
            let widths: Vec<String> = side.dims.iter().map(ToString::to_string).collect();
            invalid(format!(
                "dimensions {k}: {} cuts its vectors to {}",
                side.name,
                widths.join(", ")
            ))
        })
}

/// The request's inputs as ids and prepared images: an array with no id at its top level is one input an element,
/// anything else one input, each read as a `/completion` prompt with the special tokens on. No
/// input, an input of no id and one of more ids than the model's context are refused by name.
struct Input {
    ids: Vec<u32>,
    images: Vec<media::MediaFeed>,
}

/// One ordered embedding input part, after an image has been prepared and sized.
pub enum InputPart<'a> {
    Text(&'a str),
    /// Tokenized delimiters around `positions` image rows.
    Image {
        begin: &'a str,
        end: &'a str,
        positions: usize,
    },
}

/// Token ids and the image spans that replace their filler rows, in part order.
pub struct Assembled {
    pub ids: Vec<u32>,
    pub spans: Vec<Range<usize>>,
}

/// Tokenize each part separately, with BOS and EOS around the whole input. Image spans are
/// bounded by `ctx` before allocation; the caller checks the total length, including text and EOS.
pub fn assemble<'a>(
    parts: impl Iterator<Item = Result<InputPart<'a>, String>>,
    tok: &dyn Tokenizer,
    ctx: usize,
) -> Result<Assembled, String> {
    let mut parts = parts.peekable();
    if parts.peek().is_none() {
        return Err("Input content cannot be empty".into());
    }
    let mut ids = Vec::new();
    let mut spans = Vec::new();
    if tok.add_bos() {
        ids.push(tok.bos());
    }
    for part in parts {
        match part? {
            InputPart::Text(text) => ids.extend(tok.encode(text)),
            InputPart::Image {
                begin,
                end,
                positions,
            } => {
                if positions == 0 {
                    return Err("an embedding image has an empty span".into());
                }
                ids.extend(tok.encode(begin));
                let at = ids.len();
                let last = at
                    .checked_add(positions)
                    .filter(|&n| n <= ctx)
                    .ok_or("image input exceeds the embedding context")?;
                ids.resize(last, tok.bos());
                spans.push(at..last);
                ids.extend(tok.encode(end));
            }
        }
    }
    if tok.add_eos() {
        ids.push(tok.eos());
    }
    if ids.is_empty() {
        return Err("Input content cannot be empty".into());
    }
    Ok(Assembled { ids, spans })
}

fn input(side: &Side, key: &str, v: &Value) -> Result<Input, ApiError> {
    if let Some(parts) = v
        .as_object()
        .and_then(|o| o.get("content"))
        .and_then(Value::as_array)
    {
        let mut video = match (side.media.as_ref(), parts.first()) {
            (Some(model), Some(first))
                if matches!(media::parse_part(first), Ok(media::Part::Text(prefix))
                    if model.embedding_video().is_some_and(|(text, _)| text == prefix)) =>
            {
                Some(
                    media::prepare_video(&parts[1..], &**model)
                        .map_err(|e| invalid(e.to_string()))?
                        .into_iter(),
                )
            }
            _ => None,
        };
        let mut images = Vec::new();
        let ordered = parts.iter().map(|part| -> Result<InputPart<'_>, String> {
            match media::parse_part(part).map_err(|e| e.to_string())? {
                media::Part::Text(text) => Ok(InputPart::Text(text)),
                media::Part::Image(url) => {
                    let model = side
                        .media
                        .as_ref()
                        .ok_or("image input needs --embd-mmproj")?;
                    let (begin, end) = model
                        .embedding_delimiters()
                        .ok_or("the side projector has no embedding delimiters")?;
                    let feed = if let Some(frames) = &mut video {
                        frames
                            .next()
                            .ok_or("video frame count differs from input")?
                    } else {
                        let file = media::load(url).map_err(|e| e.to_string())?;
                        media::MediaFeed {
                            at: 0,
                            key: file.key,
                            prepared: Arc::new(
                                media::prepare(&file, &**model).map_err(|e| e.to_string())?,
                            ),
                        }
                    };
                    let positions = feed.prepared.span_len;
                    images.push(feed);
                    Ok(InputPart::Image {
                        begin,
                        end,
                        positions,
                    })
                }
            }
        });
        let Assembled { ids, spans } = assemble(ordered, &*side.tok, side.ctx).map_err(invalid)?;
        for (image, span) in images.iter_mut().zip(spans) {
            image.at = span.start;
        }
        Ok(Input { ids, images })
    } else {
        Ok(Input {
            ids: prompt_ids(&*side.tok, key, v, true)?,
            images: Vec::new(),
        })
    }
}

fn inputs(side: &Side, key: &str, v: &Value) -> Result<Vec<Input>, ApiError> {
    let is_id = |x: &Value| x.is_u64() || x.is_i64();
    let inputs = match v {
        Value::Array(a) if !a.iter().any(is_id) => a
            .iter()
            .map(|x| input(side, key, x))
            .collect::<Result<Vec<_>, _>>()?,
        v => vec![input(side, key, v)?],
    };
    if inputs.is_empty() {
        return Err(invalid("\"input\" must not be empty"));
    }
    for input in &inputs {
        let ids = &input.ids;
        if ids.is_empty() {
            return Err(invalid("Input content cannot be empty"));
        }
        if ids.len() > side.ctx {
            return Err(ApiError {
                retry_after: false,
                code: 400,
                kind: "exceed_context_size_error",
                message: format!(
                    "input ({} tokens) is larger than the max context size ({} tokens)",
                    ids.len(),
                    side.ctx
                ),
            });
        }
    }
    Ok(inputs)
}

/// Every input's pooled vector, through the side model under its lock, each pass after
/// [`open`] holds. A model failure, a panic in it, its lock poisoned, and a vector that is not its
/// width or not finite are fatal ([`fail`]).
fn run(state: &State, side: &Side, inputs: &[Input]) -> Result<Vec<Vec<f32>>, ApiError> {
    let Ok(mut model) = side.embed.lock() else {
        return Err(fail(
            state,
            side,
            "the embedding model's lock is poisoned: a request panicked while it held it".into(),
        ));
    };
    let width = side.width();
    let mut pooled = Vec::with_capacity(inputs.len());
    for input in inputs {
        open(state)?;
        let v = match panic::catch_unwind(AssertUnwindSafe(|| {
            if input.images.is_empty() {
                model.embed(&input.ids)
            } else {
                model.embed_media(&input.ids, &input.images)
            }
        })) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(fail(state, side, e.0)),
            Err(p) => {
                let what = panic_text(&*p);
                return Err(fail(
                    state,
                    side,
                    format!("the embedding model panicked: {what}"),
                ));
            }
        };
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
        pooled.push(v);
    }
    Ok(pooled)
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
        let ids = [MockTokenizer.bos(), id("a"), MockTokenizer.eos()];
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
                    "meta": { "n_ctx": 32, "n_embd": 4 },
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

    /// Which inputs a request holds and the ids each passes, as llama-server splits and
    /// tokenizes them, and the refusals of an input with nothing in it, an object input, and one
    /// of more ids than the context.
    #[test]
    fn inputs_split_and_tokenize_as_llama_server() {
        let (addr, _, calls, _) = serve(Some(MockEmbed::new(4)));
        let (bos, eos, a, b) = (MockTokenizer.bos(), MockTokenizer.eos(), id("a"), id("b"));
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
    fn normalization_and_the_cut() {
        let (addr, ..) = serve(Some(MockEmbed::new(32)));
        let ids = [MockTokenizer.bos(), id("a"), MockTokenizer.eos()];
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
