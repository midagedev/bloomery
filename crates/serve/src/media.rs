//! The server's media part, the same for every model: a message's image parts, the checks an image
//! file passes before it is decoded, its cache key, and the expansion of each image's placeholder
//! token into the image's span. What a model contributes is a [`MediaModel`] (`crates/vision`):
//! the placeholder text and its token, the separator between flattened parts, and one prepare.
//!
//! What a prompt, a slot or a cached state holds is a [`Held`]: the ids and the spans of the
//! images among them. [`common_prefix`] reads both, so two prompts that differ only in their
//! images share nothing of the image's span; neither it nor [`keep_whole_spans`] ends a kept
//! prefix inside a span, which is kept whole or fed whole. [`expand_prompt`] makes a chat's
//! [`Prompt`]: its held sequence and one [`MediaFeed`] per image for the engine.
//!
//! An image arrives only inside a `data:` URL holding a base64 PNG or JPEG, in either OpenAI form
//! of an `image_url` part. Every refusal is one [`MediaError`] variant that names what it refused;
//! nothing is skipped, repaired or fetched.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};

pub use vision::{FileKind, MediaModel, Prepared, Rgb8, VisionError};

/// A [`MediaModel`] the server holds across its threads: the request threads prepare images on
/// it while the engine runs elsewhere.
pub type SharedMediaModel = Arc<dyn MediaModel + Send + Sync>;

/// One part of a message's `content` array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part<'a> {
    Text(&'a str),
    /// An `image_url` part's URL.
    Image(&'a str),
}

/// A message's content array flattened for the chat template.
#[derive(Debug, PartialEq, Eq)]
pub struct Flat<'a> {
    /// The parts in order, joined by the model's separator, each image as the model's placeholder.
    pub text: String,
    /// The images' URLs, in the order of their placeholders in `text`.
    pub images: Vec<&'a str>,
}

/// An image's cache key: the sha256 of its file bytes. It is known before the decode, so a request
/// whose image is cached skips the decode, the resize and the encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey(pub [u8; 32]);

/// An image file a request carried: its format checked against its bytes, keyed, not decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageFile {
    pub kind: FileKind,
    pub bytes: Vec<u8>,
    pub key: ImageKey,
}

/// A prompt whose image placeholders are expanded to their images' spans.
#[derive(Debug, PartialEq, Eq)]
pub struct Expanded {
    pub ids: Vec<u32>,
    /// Where each image's span sits in `ids`, in the order of the images.
    pub spans: Vec<Range<usize>>,
}

/// One image's span among a prompt's ids: its first position, its length, and the key of the
/// image that occupies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaSpan {
    pub at: usize,
    pub len: usize,
    pub key: ImageKey,
}

impl MediaSpan {
    /// The position past its last.
    #[must_use]
    pub fn end(&self) -> usize {
        self.at + self.len
    }
}

/// What a prompt, a slot or a cached state holds: the ids, one a position, and the image spans
/// among them. Every position of a span carries the image token, so the ids alone cannot tell
/// two images of one grid apart: the key can.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Held {
    pub ids: Vec<u32>,
    /// Ascending, not overlapping, each whole inside `ids`.
    pub media: Vec<MediaSpan>,
}

impl From<Vec<u32>> for Held {
    fn from(ids: Vec<u32>) -> Held {
        Held {
            ids,
            media: Vec::new(),
        }
    }
}

impl Held {
    /// The positions it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Appends one id; no span starts at it.
    pub fn push(&mut self, id: u32) {
        self.ids.push(id);
    }

    /// Keeps the first `n` ids and the spans that lie whole inside them.
    pub fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        self.media.retain(|s| s.end() <= n);
    }

    /// Whether `self` extends `head`: the same ids, and the same image spans where they overlap.
    #[must_use]
    pub fn starts_with(&self, head: &Held) -> bool {
        self.ids.starts_with(&head.ids)
            && head.media.iter().all(|s| self.media.contains(s))
            && self
                .media
                .iter()
                .all(|s| s.at >= head.ids.len() || head.media.contains(s))
    }

    /// The span a prefix of `n` positions would end inside: one that starts before `n` and ends
    /// after it.
    #[must_use]
    pub fn span_around(&self, n: usize) -> Option<&MediaSpan> {
        self.media.iter().find(|s| s.at < n && n < s.end())
    }
}

/// How many leading positions `a` and `b` share. Their ids must be equal, and every image span
/// inside the shared part must hold the same image on both sides: the same key and the same
/// length. A span the shared part would end inside, or hold with another image, ends the sharing
/// at that span's start — the position a later request resumes the prompt at.
#[must_use]
pub fn common_prefix(a: &Held, b: &Held) -> usize {
    let mut n = a.ids.iter().zip(&b.ids).take_while(|(x, y)| x == y).count();
    for s in a.media.iter().chain(&b.media) {
        if s.at >= n {
            continue;
        }
        let shared = a.media.contains(s) && b.media.contains(s);
        if s.end() > n || !shared {
            n = s.at;
        }
    }
    n
}

/// What a cache resumes `req` at when it may keep at most `n` positions of it and keeps
/// `keepable(m)` of any `m`: an image span is kept whole or fed whole, so a keep that would end
/// inside one of `req`'s spans goes back to the span's start, and the cache is asked again from
/// there (llama.cpp's `keep_first`, mistral.rs's `clamp_normal_prefix_len`).
pub fn keep_whole_spans(req: &Held, n: usize, keepable: impl Fn(usize) -> usize) -> usize {
    let mut k = keepable(n).min(n);
    while let Some(s) = req.span_around(k) {
        k = keepable(s.at).min(s.at);
    }
    k
}

/// One image a request carries into the engine: where its span sits in the prompt, its key, and
/// its prepare — the last shared by every call that feeds the span's positions, so a prompt cut
/// into calls hands the engine the same image.
#[derive(Clone, Debug)]
pub struct MediaFeed {
    /// The span's first position in the ids it rides with: the prompt's in a [`Prompt`], the
    /// call's in [`crate::Engine::prefill_media`].
    pub at: usize,
    pub key: ImageKey,
    pub prepared: Arc<Prepared>,
}

impl MediaFeed {
    /// The span this feed occupies: `prepared.span_len` positions from `at`.
    #[must_use]
    pub fn span(&self) -> MediaSpan {
        MediaSpan {
            at: self.at,
            len: self.prepared.span_len,
            key: self.key,
        }
    }
}

/// A prompt as a request hands it to the engine thread: what its slot holds once the prompt is
/// fed, and one feed per image, in the order of the spans.
#[derive(Clone, Debug, Default)]
pub struct Prompt {
    pub held: Held,
    pub feeds: Vec<MediaFeed>,
}

impl From<Vec<u32>> for Prompt {
    fn from(ids: Vec<u32>) -> Prompt {
        Prompt {
            held: Held::from(ids),
            feeds: Vec::new(),
        }
    }
}

/// Everything the media part refuses, each naming what it refused. All but [`MediaError::EmptySpan`]
/// are the request's fault.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// A content part that is not a JSON object.
    #[error("a content part must be an object")]
    PartNotObject,
    /// A content part whose `type` is missing or not one this server takes (the JSON of the value,
    /// or `missing`).
    #[error("content part type {0}: only \"text\" and \"image_url\" parts are accepted")]
    PartType(String),
    /// A content part without the field its type needs.
    #[error("a {ty} part needs {need}")]
    PartField {
        ty: &'static str,
        need: &'static str,
    },
    /// An image part in a message whose role is not `user`.
    #[error("an image part in a {0} message: only user messages carry images")]
    ImageRole(String),
    /// Message text that carries the model's image placeholder.
    #[error(
        "the message text carries the image placeholder {0}; send an image as an image_url part"
    )]
    PlaceholderInText(String),
    /// An image URL without a scheme.
    #[error("an image URL must be a data: URL holding a base64 PNG or JPEG")]
    NotAUrl,
    /// An image URL of a scheme other than `data` (lowercased); nothing is fetched.
    #[error(
        "{0}: URLs are not fetched; send the image as a data: URL holding a base64 PNG or JPEG"
    )]
    Scheme(String),
    /// A `data:` URL without the comma before its data.
    #[error("a data: URL without the comma before its data")]
    DataUrlSyntax,
    /// A `data:` URL whose data is not marked base64 (its media type).
    #[error("a data: URL of {0:?} that is not base64")]
    NotBase64(String),
    /// A declared media type other than `image/png`, `image/jpeg` or `image/jpg`.
    #[error("media type {0:?}: only image/png and image/jpeg are accepted")]
    Mime(String),
    /// A `data:` URL with no data.
    #[error("an empty image")]
    Empty,
    /// A byte outside the standard base64 alphabet (whitespace and the URL-safe `-` `_` included).
    #[error("base64: byte {byte:#04x} at {offset} is not in the alphabet")]
    Base64Char { offset: usize, byte: u8 },
    /// A base64 text whose length is not a multiple of four: unpadded, or cut.
    #[error("base64: {0} characters, not a multiple of four")]
    Base64Length(usize),
    /// Padding where it cannot be: more than two `=`, or data after one.
    #[error("base64: padding at {0} before the end")]
    Base64Padding(usize),
    /// A last symbol with bits set past the data's end: not the encoding of any bytes.
    #[error("base64: the symbol at {0} sets bits past the data's end")]
    Base64TrailingBits(usize),
    /// Bytes of the other accepted format than the URL declares.
    #[error("declared {}, but the bytes are {}", .declared.name(), .found.name())]
    Magic { declared: FileKind, found: FileKind },
    /// Bytes of a format that is not accepted.
    #[error("a {} image: only PNG and JPEG are accepted", .0.name())]
    Format(FileKind),
    /// Bytes of no image format this server knows (their first bytes, hex).
    #[error("not a PNG or JPEG file (it starts {0})")]
    Unknown(String),
    /// The file did not decode.
    #[error("image: {0}")]
    Decode(VisionError),
    /// The model refused the decoded image.
    #[error("image: {0}")]
    Prepare(VisionError),
    /// The prompt's image placeholders and the request's images do not pair up.
    #[error("{placeholders} image placeholder(s) in the prompt for {images} image(s)")]
    PlaceholderCount { placeholders: usize, images: usize },
    /// A prepared image whose span is empty (the model's fault, not the request's).
    #[error("image {0} has an empty span")]
    EmptySpan(usize),
    /// A prompt whose last position is an image's: the reply is read after a text position.
    #[error("the prompt ends with an image; its last position must be text")]
    ImageLast,
}

impl fmt::Display for ImageKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// One content part: `{"type":"text","text":…}`, or `{"type":"image_url","image_url":{"url":…}}`
/// or `{"type":"image_url","image_url":…}`.
pub fn parse_part(part: &Value) -> Result<Part<'_>, MediaError> {
    let Value::Object(part) = part else {
        return Err(MediaError::PartNotObject);
    };
    match part.get("type") {
        Some(Value::String(t)) if t == "text" => match part.get("text") {
            Some(Value::String(text)) => Ok(Part::Text(text)),
            _ => Err(MediaError::PartField {
                ty: "text",
                need: "a string 'text'",
            }),
        },
        Some(Value::String(t)) if t == "image_url" => match part.get("image_url") {
            Some(Value::String(url)) => Ok(Part::Image(url)),
            Some(Value::Object(o)) => match o.get("url") {
                Some(Value::String(url)) => Ok(Part::Image(url)),
                _ => Err(MediaError::PartField {
                    ty: "image_url",
                    need: "a string 'image_url.url'",
                }),
            },
            _ => Err(MediaError::PartField {
                ty: "image_url",
                need: "an 'image_url' object with a string 'url', or a string",
            }),
        },
        Some(other) => Err(MediaError::PartType(short(&other.to_string()))),
        None => Err(MediaError::PartType("missing".into())),
    }
}

/// A message's content array as the chat template's text: text parts as they are, each image as
/// the model's placeholder, joined by the model's separator. Only a `user` message carries images:
/// the template may move other roles' text (V4.1's gathers system messages first), and the i-th
/// placeholder of the prompt must be the i-th image.
pub fn flatten<'a>(
    role: &str,
    parts: &'a [Value],
    model: &dyn MediaModel,
) -> Result<Flat<'a>, MediaError> {
    let mut texts = Vec::with_capacity(parts.len());
    let mut images = Vec::new();
    for part in parts {
        match parse_part(part)? {
            Part::Text(text) => {
                check_text(text, model)?;
                texts.push(text);
            }
            Part::Image(_) if role != "user" => return Err(MediaError::ImageRole(short(role))),
            Part::Image(url) => {
                texts.push(model.image_placeholder());
                images.push(url);
            }
        }
    }
    Ok(Flat {
        text: texts.join(model.part_separator()),
        images,
    })
}

/// Refuse message text that carries the model's image placeholder: only an image part makes one.
pub fn check_text(text: &str, model: &dyn MediaModel) -> Result<(), MediaError> {
    let placeholder = model.image_placeholder();
    if text.contains(placeholder) {
        return Err(MediaError::PlaceholderInText(placeholder.to_owned()));
    }
    Ok(())
}

/// An image URL's file: a `data:` URL of a PNG or a JPEG, its base64 decoded strictly, its bytes
/// the format it declares, and its key.
pub fn load(url: &str) -> Result<ImageFile, MediaError> {
    let (kind, payload) = data_url(url)?;
    let bytes = base64(payload)?;
    match FileKind::sniff(&bytes) {
        found if found == kind => {}
        found @ (FileKind::Png | FileKind::Jpeg) => {
            return Err(MediaError::Magic {
                declared: kind,
                found,
            });
        }
        FileKind::Unknown => {
            let head: String = bytes.iter().take(8).map(|b| format!("{b:02x}")).collect();
            return Err(MediaError::Unknown(head));
        }
        other => return Err(MediaError::Format(other)),
    }
    Ok(ImageFile {
        kind,
        key: ImageKey(Sha256::digest(&bytes).into()),
        bytes,
    })
}

/// Decode a file and prepare it for `model`.
pub fn prepare(file: &ImageFile, model: &dyn MediaModel) -> Result<Prepared, MediaError> {
    let image = Rgb8::from_bytes(&file.bytes).map_err(MediaError::Decode)?;
    model.prepare(&image).map_err(MediaError::Prepare)
}

/// Replace the i-th `image_token` of `ids` by `span_lens[i]` copies of it.
pub fn expand_spans(
    ids: &[u32],
    image_token: u32,
    span_lens: &[usize],
) -> Result<Expanded, MediaError> {
    let placeholders = ids.iter().filter(|&&t| t == image_token).count();
    if placeholders != span_lens.len() {
        return Err(MediaError::PlaceholderCount {
            placeholders,
            images: span_lens.len(),
        });
    }
    if let Some(i) = span_lens.iter().position(|&n| n == 0) {
        return Err(MediaError::EmptySpan(i));
    }
    let mut out = Vec::with_capacity(ids.len() - placeholders + span_lens.iter().sum::<usize>());
    let mut spans = Vec::with_capacity(placeholders);
    let mut lens = span_lens.iter();
    for &t in ids {
        if t == image_token {
            let n = *lens
                .next()
                .expect("one length per placeholder, counted above");
            spans.push(out.len()..out.len() + n);
            out.extend(std::iter::repeat_n(t, n));
        } else {
            out.push(t);
        }
    }
    Ok(Expanded { ids: out, spans })
}

/// The prompt of a rendered chat whose ids are `ids` and whose images are `urls`, in the order of
/// their placeholders: each image loaded and prepared on the calling thread, its placeholder's
/// token expanded to its span, and one feed per image. The placeholders are counted and the last
/// position checked before any image is decoded, so a placeholder typed into the text, which
/// pairs with no image, costs no decode.
pub fn expand_prompt(
    ids: &[u32],
    urls: &[&str],
    model: &dyn MediaModel,
) -> Result<Prompt, MediaError> {
    let token = model.image_token();
    let placeholders = ids.iter().filter(|&&t| t == token).count();
    if placeholders != urls.len() {
        return Err(MediaError::PlaceholderCount {
            placeholders,
            images: urls.len(),
        });
    }
    if ids.last() == Some(&token) {
        return Err(MediaError::ImageLast);
    }
    let files = urls
        .iter()
        .map(|url| load(url))
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = files
        .iter()
        .map(|f| prepare(f, model))
        .collect::<Result<Vec<_>, _>>()?;
    let lens: Vec<usize> = prepared.iter().map(|p| p.span_len).collect();
    let expanded = expand_spans(ids, token, &lens)?;
    let feeds: Vec<MediaFeed> = expanded
        .spans
        .iter()
        .zip(files.iter().zip(prepared))
        .map(|(span, (file, prepared))| MediaFeed {
            at: span.start,
            key: file.key,
            prepared: Arc::new(prepared),
        })
        .collect();
    Ok(Prompt {
        held: Held {
            ids: expanded.ids,
            media: feeds.iter().map(MediaFeed::span).collect(),
        },
        feeds,
    })
}

/// A `data:` URL's declared format and its data. Scheme and media type are matched without case
/// (RFC 3986 §3.1, RFC 2045 §5.1); a media type with parameters is not one of the accepted.
fn data_url(url: &str) -> Result<(FileKind, &str), MediaError> {
    let scheme = url.split_once(':').map(|(s, _)| s).filter(|s| {
        s.bytes().next().is_some_and(|c| c.is_ascii_alphabetic())
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.'))
    });
    let Some(scheme) = scheme else {
        return Err(MediaError::NotAUrl);
    };
    if !scheme.eq_ignore_ascii_case("data") {
        return Err(MediaError::Scheme(short(&scheme.to_ascii_lowercase())));
    }
    let rest = &url[scheme.len() + 1..];
    let Some((header, payload)) = rest.split_once(',') else {
        return Err(MediaError::DataUrlSyntax);
    };
    let marker = ";base64";
    let split = header.len().checked_sub(marker.len());
    let Some(mime) = split
        .filter(|&at| header.is_char_boundary(at) && header[at..].eq_ignore_ascii_case(marker))
        .map(|at| &header[..at])
    else {
        return Err(MediaError::NotBase64(short(header)));
    };
    let kind = if mime.eq_ignore_ascii_case("image/png") {
        FileKind::Png
    } else if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        FileKind::Jpeg
    } else {
        return Err(MediaError::Mime(short(mime)));
    };
    Ok((kind, payload))
}

/// The value of each byte in the standard base64 alphabet, `INVALID` outside it.
const SEXTET: [u8; 256] = {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut t = [INVALID; 256];
    let mut i = 0;
    while i < 64 {
        t[alphabet[i] as usize] = i as u8;
        i += 1;
    }
    t
};
const INVALID: u8 = 0xFF;

/// Strict standard base64 (RFC 4648 §4): every byte from the alphabet or a final `=`, a length
/// that is a multiple of four, at most two `=` and only at the end, and no bits set past the data
/// in the last symbol.
fn base64(text: &str) -> Result<Vec<u8>, MediaError> {
    let s = text.as_bytes();
    if s.is_empty() {
        return Err(MediaError::Empty);
    }
    if let Some(offset) = s
        .iter()
        .position(|&c| SEXTET[usize::from(c)] == INVALID && c != b'=')
    {
        return Err(MediaError::Base64Char {
            offset,
            byte: s[offset],
        });
    }
    if !s.len().is_multiple_of(4) {
        return Err(MediaError::Base64Length(s.len()));
    }
    let pad = s.iter().rev().take_while(|&&c| c == b'=').count();
    let body = &s[..s.len() - pad];
    if pad > 2 {
        return Err(MediaError::Base64Padding(body.len()));
    }
    if let Some(at) = body.iter().position(|&c| c == b'=') {
        return Err(MediaError::Base64Padding(at));
    }
    let v = |c: u8| u32::from(SEXTET[usize::from(c)]);
    let (quads, tail) = body.as_chunks::<4>();
    let mut out = Vec::with_capacity(quads.len() * 3 + 2);
    for q in quads {
        let n = v(q[0]) << 18 | v(q[1]) << 12 | v(q[2]) << 6 | v(q[3]);
        out.extend_from_slice(&n.to_be_bytes()[1..]);
    }
    let at = quads.len() * 4;
    match *tail {
        [] => {}
        [a, b] => {
            if v(b) & 0xF != 0 {
                return Err(MediaError::Base64TrailingBits(at + 1));
            }
            out.push((v(a) << 2 | v(b) >> 4) as u8);
        }
        [a, b, c] => {
            if v(c) & 0x3 != 0 {
                return Err(MediaError::Base64TrailingBits(at + 2));
            }
            let n = v(a) << 10 | v(b) << 4 | v(c) >> 2;
            out.extend_from_slice(&n.to_be_bytes()[2..]);
        }
        _ => unreachable!("a multiple of four less at most two pads leaves 0, 2 or 3"),
    }
    Ok(out)
}

/// At most the first 64 characters of a value an error names, so a refusal never echoes a payload.
fn short(s: &str) -> String {
    match s.char_indices().nth(64) {
        Some((at, _)) => format!("{}…", &s[..at]),
        None => s.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use vision::{GridPlan, Patches};

    use super::{
        Expanded, FileKind, Held, ImageKey, MediaError, MediaModel, MediaSpan, Part, Prepared,
        Rgb8, VisionError, common_prefix, expand_prompt, expand_spans, flatten, keep_whole_spans,
        load, parse_part, prepare,
    };

    /// A 2×1 RGB PNG of the colour (200, 100, 50).
    const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg==";
    /// Its sha256 (hashlib).
    const PNG_SHA256: &str = "29f48d50a7db2b049176bebb2523c066f97d61b7aac0ee54d826f916bd22ccb6";

    /// A model of data only: a placeholder, its token, a separator, and a prepare that gives one
    /// span position per pixel.
    struct Stub {
        separator: Option<&'static str>,
    }

    impl MediaModel for Stub {
        fn image_placeholder(&self) -> &str {
            "<img>"
        }

        fn image_token(&self) -> u32 {
            7
        }

        fn part_separator(&self) -> &str {
            self.separator.unwrap_or("\n")
        }

        fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
            let plan = GridPlan {
                n_llm_h: image.height,
                n_llm_w: image.width,
                best_h: image.height,
                best_w: image.width,
            };
            Ok(Prepared {
                span_len: image.width * image.height,
                patches: Patches {
                    plan,
                    n_vit_h: 0,
                    n_vit_w: 0,
                    patch_len: 0,
                    bf16: Vec::new(),
                },
            })
        }
    }

    const PLAIN: Stub = Stub { separator: None };

    fn data(mime: &str, b64: &str) -> String {
        format!("data:{mime};base64,{b64}")
    }

    /// Standard base64 with padding (RFC 4648 §4), to build payloads.
    fn encode(bytes: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in bytes.chunks(3) {
            let n = c
                .iter()
                .enumerate()
                .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
            for i in 0..4 {
                s.push(if i <= c.len() {
                    char::from(A[(n >> (18 - 6 * i) & 63) as usize])
                } else {
                    '='
                });
            }
        }
        s
    }

    fn err(r: Result<impl std::fmt::Debug, MediaError>) -> MediaError {
        match r {
            Ok(v) => panic!("accepted: {v:?}"),
            Err(e) => e,
        }
    }

    #[test]
    fn parts_in_both_image_url_forms() {
        assert_eq!(
            parse_part(&json!({"type": "text", "text": "hi"})).unwrap(),
            Part::Text("hi")
        );
        assert_eq!(
            parse_part(&json!({"type": "image_url", "image_url": {"url": "data:x"}})).unwrap(),
            Part::Image("data:x")
        );
        assert_eq!(
            parse_part(&json!({"type": "image_url", "image_url": "data:y"})).unwrap(),
            Part::Image("data:y")
        );
    }

    #[test]
    fn malformed_parts_are_named() {
        assert!(matches!(
            err(parse_part(&json!("text"))),
            MediaError::PartNotObject
        ));
        for (part, ty) in [
            (json!({"text": "hi"}), "missing"),
            (
                json!({"type": "input_audio", "input_audio": {}}),
                "\"input_audio\"",
            ),
            (json!({"type": 3}), "3"),
        ] {
            match err(parse_part(&part)) {
                MediaError::PartType(got) => assert_eq!(got, ty, "{part}"),
                e => panic!("{part}: {e}"),
            }
        }
        for part in [
            json!({"type": "text"}),
            json!({"type": "text", "text": 1}),
            json!({"type": "image_url"}),
            json!({"type": "image_url", "image_url": {"uri": "data:x"}}),
            json!({"type": "image_url", "image_url": 5}),
        ] {
            assert!(
                matches!(err(parse_part(&part)), MediaError::PartField { .. }),
                "{part}"
            );
        }
    }

    #[test]
    fn flatten_joins_by_the_models_separator() {
        let parts = vec![
            json!({"type": "text", "text": "look"}),
            json!({"type": "image_url", "image_url": {"url": "data:a"}}),
            json!({"type": "text", "text": "and"}),
            json!({"type": "image_url", "image_url": "data:b"}),
        ];
        let flat = flatten("user", &parts, &PLAIN).unwrap();
        assert_eq!(flat.text, "look\n<img>\nand\n<img>");
        assert_eq!(flat.images, ["data:a", "data:b"]);
        let wide = Stub {
            separator: Some("\n\n"),
        };
        assert_eq!(
            flatten("user", &parts, &wide).unwrap().text,
            "look\n\n<img>\n\nand\n\n<img>"
        );
        let text_only = [
            json!({"type": "text", "text": "a"}),
            json!({"type": "text", "text": "b"}),
        ];
        let flat = flatten("system", &text_only, &PLAIN).unwrap();
        assert_eq!((flat.text.as_str(), flat.images.len()), ("a\nb", 0));
    }

    #[test]
    fn flatten_refusals_are_named() {
        let image = [json!({"type": "image_url", "image_url": "data:a"})];
        for role in ["system", "assistant", "tool", "developer"] {
            match err(flatten(role, &image, &PLAIN)) {
                MediaError::ImageRole(r) => assert_eq!(r, role),
                e => panic!("{role}: {e}"),
            }
        }
        let text = [json!({"type": "text", "text": "an <img> by hand"})];
        assert!(matches!(
            err(flatten("user", &text, &PLAIN)),
            MediaError::PlaceholderInText(_)
        ));
        let bad = [
            json!({"type": "image_url", "image_url": "data:a"}),
            json!({"type": "video_url"}),
        ];
        assert!(matches!(
            err(flatten("user", &bad, &PLAIN)),
            MediaError::PartType(_)
        ));
    }

    #[test]
    fn a_png_data_url_loads_with_its_key() {
        let f = load(&data("image/png", PNG_B64)).unwrap();
        assert_eq!(f.kind, FileKind::Png);
        assert_eq!(f.bytes.len(), 70);
        assert_eq!(f.key.to_string(), PNG_SHA256);
        assert_eq!(f.key, ImageKey(sha256_of(&f.bytes)));
        // Media type and scheme names are case-blind (RFC 2397, RFC 3986).
        let g = load(&format!("DATA:Image/PNG;BASE64,{PNG_B64}")).unwrap();
        assert_eq!(g, f);
    }

    fn sha256_of(bytes: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        Sha256::digest(bytes).into()
    }

    #[test]
    fn jpeg_and_its_jpg_alias_load() {
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0, 16, b'J', b'F', b'I', b'F'];
        for mime in ["image/jpeg", "image/jpg"] {
            let f = load(&data(mime, &encode(&jpeg))).unwrap();
            assert_eq!(
                (f.kind, f.bytes.as_slice()),
                (FileKind::Jpeg, &jpeg[..]),
                "{mime}"
            );
        }
    }

    #[test]
    fn other_schemes_are_refused_by_name() {
        for (url, scheme) in [
            ("http://example.com/a.png", "http"),
            ("https://example.com/a.png", "https"),
            ("file:///etc/passwd", "file"),
            ("ftp://x/y.png", "ftp"),
            ("HTTPS://example.com/a.png", "https"),
        ] {
            match err(load(url)) {
                MediaError::Scheme(s) => assert_eq!(s, scheme, "{url}"),
                e => panic!("{url}: {e}"),
            }
        }
        for url in [PNG_B64, "", "://x", "1http://x"] {
            assert!(matches!(err(load(url)), MediaError::NotAUrl), "{url}");
        }
    }

    #[test]
    fn data_url_refusals_are_named() {
        assert!(matches!(
            err(load("data:image/png;base64")),
            MediaError::DataUrlSyntax
        ));
        assert!(matches!(
            err(load(&format!("data:image/png,{PNG_B64}"))),
            MediaError::NotBase64(_)
        ));
        for mime in [
            "image/webp",
            "image/gif",
            "text/plain",
            "",
            "image/png;charset=utf-8",
        ] {
            match err(load(&data(mime, PNG_B64))) {
                MediaError::Mime(m) => assert_eq!(m, mime),
                e => panic!("{mime}: {e}"),
            }
        }
        assert!(matches!(
            err(load(&data("image/png", ""))),
            MediaError::Empty
        ));
    }

    #[test]
    fn base64_is_strict() {
        // RFC 4648 §10.
        for (plain, b64) in [
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(super::base64(b64).unwrap(), plain.as_bytes(), "{b64}");
        }
        let all: Vec<u8> = (0..=255).collect();
        for n in [254, 255, 256] {
            assert_eq!(super::base64(&encode(&all[..n])).unwrap(), &all[..n]);
        }
        for (b64, want) in [
            ("Zm9v\nYmFy", "char 10 at 4"),
            ("Zm9v YmFy", "char 32 at 4"),
            ("Zm9-", "char 45 at 3"),
            ("Zm9_", "char 95 at 3"),
            ("Zm*v", "char 42 at 2"),
            ("Zm9vYmF", "length 7"),
            ("Zm9vYmFy=", "length 9"),
            ("Zg==Zg==", "padding at 2"),
            ("Z===", "padding at 1"),
            ("====", "padding at 0"),
            ("Zh==", "trailing at 1"),
            ("Zm9=", "trailing at 2"),
        ] {
            let got = match err(super::base64(b64)) {
                MediaError::Base64Char { offset, byte } => format!("char {byte} at {offset}"),
                MediaError::Base64Length(n) => format!("length {n}"),
                MediaError::Base64Padding(at) => format!("padding at {at}"),
                MediaError::Base64TrailingBits(at) => format!("trailing at {at}"),
                e => panic!("{b64:?}: {e}"),
            };
            assert_eq!(got, want, "{b64:?}");
        }
    }

    #[test]
    fn bytes_must_be_the_declared_format() {
        let png = super::base64(PNG_B64).unwrap();
        let jpeg = [0xFF, 0xD8, 0xFF, 0xDB];
        match err(load(&data("image/jpeg", PNG_B64))) {
            MediaError::Magic { declared, found } => {
                assert_eq!((declared, found), (FileKind::Jpeg, FileKind::Png));
            }
            e => panic!("{e}"),
        }
        match err(load(&data("image/png", &encode(&jpeg)))) {
            MediaError::Magic { declared, found } => {
                assert_eq!((declared, found), (FileKind::Png, FileKind::Jpeg));
            }
            e => panic!("{e}"),
        }
        for (bytes, kind) in [
            (&b"RIFF\x24\0\0\0WEBPVP8 "[..], FileKind::Webp),
            (b"GIF89a\x01\0\x01\0", FileKind::Gif),
        ] {
            match err(load(&data("image/png", &encode(bytes)))) {
                MediaError::Format(k) => assert_eq!(k, kind),
                e => panic!("{kind:?}: {e}"),
            }
        }
        match err(load(&data("image/png", &encode(b"BM\x3a\0\0\0")))) {
            MediaError::Unknown(head) => assert_eq!(head, "424d3a000000"),
            e => panic!("{e}"),
        }
        assert_eq!(png.len(), 70);
    }

    #[test]
    fn prepare_decodes_then_asks_the_model() {
        let f = load(&data("image/png", PNG_B64)).unwrap();
        let p = prepare(&f, &PLAIN).unwrap();
        assert_eq!(p.span_len, 2);
        assert_eq!((p.patches.plan.best_w, p.patches.plan.best_h), (2, 1));
        let jpeg = [0xFF, 0xD8, 0xFF, 0xDB, 0, 4, 0, 0];
        let broken = load(&data("image/jpeg", &encode(&jpeg))).unwrap();
        assert!(matches!(
            err(prepare(&broken, &PLAIN)),
            MediaError::Decode(VisionError::Decode(_))
        ));
    }

    #[test]
    fn spans_expand_in_order() {
        let got = expand_spans(&[1, 7, 2, 7, 3], 7, &[3, 1]).unwrap();
        assert_eq!(
            got,
            Expanded {
                ids: vec![1, 7, 7, 7, 2, 7, 3],
                spans: vec![1..4, 5..6],
            }
        );
        let none = expand_spans(&[1, 2], 7, &[]).unwrap();
        assert_eq!((none.ids, none.spans.len()), (vec![1, 2], 0));
        let first = expand_spans(&[7], 7, &[2]).unwrap();
        assert_eq!(first.ids, [7, 7]);
        assert_eq!((first.spans.len(), first.spans.first()), (1, Some(&(0..2))));
    }

    #[test]
    fn spans_that_do_not_pair_up_are_named() {
        for (ids, lens, want) in [
            (&[1, 7, 7][..], &[2][..], (2, 1)),
            (&[1, 7][..], &[2, 2][..], (1, 2)),
            (&[1, 2][..], &[2][..], (0, 1)),
        ] {
            match err(expand_spans(ids, 7, lens)) {
                MediaError::PlaceholderCount {
                    placeholders,
                    images,
                } => assert_eq!((placeholders, images), want),
                e => panic!("{ids:?} {lens:?}: {e}"),
            }
        }
        assert!(matches!(
            err(expand_spans(&[7, 7], 7, &[1, 0])),
            MediaError::EmptySpan(1)
        ));
    }

    #[test]
    fn errors_do_not_echo_a_whole_payload() {
        let long = format!("data:image/{};base64,AAAA", "x".repeat(10_000));
        let msg = err(load(&long)).to_string();
        assert!(msg.len() < 200, "{} bytes: {msg}", msg.len());
        let part: Value = json!({"type": "y".repeat(10_000)});
        assert!(err(parse_part(&part)).to_string().len() < 200);
    }

    /// One piece of a [`seq`]: `n` text positions, or an image of `len` positions and key `k`.
    enum Piece {
        Text(usize),
        Image(usize, u8),
    }
    use Piece::{Image, Text};

    /// A well-formed held sequence: each text position its own id (`100 +` its position), each
    /// image position the stub's image token 7, each image a span with the key `[k; 32]`.
    fn seq(pieces: &[Piece]) -> Held {
        let mut h = Held::default();
        for p in pieces {
            match *p {
                Text(n) => {
                    let at = h.ids.len();
                    h.ids.extend((at..at + n).map(|i| 100 + i as u32));
                }
                Image(len, k) => {
                    h.media.push(MediaSpan {
                        at: h.ids.len(),
                        len,
                        key: ImageKey([k; 32]),
                    });
                    h.ids.extend(std::iter::repeat_n(7, len));
                }
            }
        }
        h
    }

    /// Two images of one grid have the same ids (every span position carries the image token): the
    /// key stops the sharing at the span's start. The same image is shared whole; a span of
    /// another length, or one only one side marks, is another image.
    #[test]
    fn the_shared_prefix_needs_the_same_image_in_the_span() {
        let a = seq(&[Text(3), Image(3, 1), Text(2)]);
        let longer = seq(&[Text(3), Image(3, 1), Text(4)]);
        assert_eq!(common_prefix(&a, &longer), 8, "the same image: all of a");
        let other = seq(&[Text(3), Image(3, 2), Text(2)]);
        assert_eq!(a.ids, other.ids, "one grid: the ids are equal");
        assert_eq!(
            common_prefix(&a, &other),
            3,
            "another image: the span's start"
        );
        assert_eq!(common_prefix(&other, &a), 3, "either order");
        let wider = seq(&[Text(3), Image(4, 1), Text(1)]);
        assert_eq!(
            common_prefix(&a, &wider),
            3,
            "another length is another image"
        );
        let unmarked = Held::from(a.ids.clone());
        assert_eq!(
            common_prefix(&a, &unmarked),
            3,
            "a span one side does not mark"
        );
        let two = seq(&[Text(3), Image(3, 1), Text(2), Image(2, 9), Text(1)]);
        let two_b = seq(&[Text(3), Image(3, 1), Text(2), Image(2, 8), Text(1)]);
        assert_eq!(
            common_prefix(&two, &two_b),
            8,
            "the first shared, the second not"
        );
        assert_eq!(common_prefix(&two, &two), two.len());
        let text = Held::from(vec![1, 2, 3]);
        assert_eq!(common_prefix(&text, &Held::from(vec![1, 2, 4])), 2);
        assert_eq!(common_prefix(&Held::default(), &text), 0);
    }

    /// A state cut inside a span (its ids stop among the span's image tokens, its table drops the
    /// span) shares with the request that carries the span only up to the span's start.
    #[test]
    fn a_prefix_ending_inside_a_span_is_cut_to_its_start() {
        let req = seq(&[Text(3), Image(3, 1), Text(2)]);
        let mut cut = req.clone();
        cut.truncate(5);
        assert_eq!(
            (cut.len(), cut.media.len()),
            (5, 0),
            "4 and 5 of the span 3..6"
        );
        assert_eq!(common_prefix(&cut, &req), 3);
        assert_eq!(common_prefix(&req, &cut), 3);
        let mut at_end = req.clone();
        at_end.truncate(6);
        assert_eq!(at_end.media, req.media, "the span lies whole inside 6");
        assert_eq!(common_prefix(&at_end, &req), 6);
        let two = seq(&[Text(3), Image(3, 1), Text(2), Image(2, 1), Text(1)]);
        let mut in_second = two.clone();
        in_second.truncate(9);
        assert_eq!(common_prefix(&in_second, &two), 8, "9 lies inside 8..10");
    }

    /// An engine that keeps less than asked keeps no part of an image: a keep inside a span goes
    /// back to its start and asks again.
    #[test]
    fn a_keep_inside_a_span_goes_back_to_its_start() {
        let req = seq(&[Text(3), Image(3, 1), Text(2)]);
        let all = |n: usize| n;
        assert_eq!(keep_whole_spans(&req, 7, all), 7);
        assert_eq!(keep_whole_spans(&req, 6, all), 6, "the span's end");
        assert_eq!(keep_whole_spans(&req, 5, all), 3, "inside: its start");
        let even = |n: usize| n / 2 * 2;
        assert_eq!(keep_whole_spans(&req, 7, even), 6);
        assert_eq!(keep_whole_spans(&req, 5, |n: usize| n.min(4)), 3);
        let fours = |n: usize| n / 4 * 4;
        assert_eq!(
            keep_whole_spans(&req, 7, fours),
            0,
            "4 lies inside 3..6; the engine keeps 0 of 3"
        );
        let grow = |n: usize| n + 2;
        assert_eq!(keep_whole_spans(&req, 2, grow), 2, "never past n");
    }

    /// [`Held::starts_with`] needs the same ids and the same images where they overlap: a state of
    /// another image is not the sequence it covers.
    #[test]
    fn held_starts_with_needs_the_same_images() {
        let a = seq(&[Text(3), Image(3, 1), Text(2)]);
        assert!(a.starts_with(&a));
        assert!(a.starts_with(&Held::from(a.ids[..3].to_vec())));
        assert!(a.starts_with(&seq(&[Text(3), Image(3, 1)])));
        assert!(Held::default().starts_with(&Held::default()));
        assert!(
            !a.starts_with(&Held::from(a.ids.clone())),
            "the head marks no span"
        );
        assert!(
            !a.starts_with(&seq(&[Text(3), Image(3, 2), Text(2)])),
            "the head holds another image"
        );
        let mut cut = a.clone();
        cut.truncate(5);
        assert!(!a.starts_with(&cut), "the head ends inside the span");
        assert!(!cut.starts_with(&a));
    }

    /// A chat's prompt: the placeholders counted against the images before a decode, each image's
    /// span and feed in placeholder order, and a prompt that ends with an image refused.
    #[test]
    fn a_prompt_expands_with_one_feed_per_image() {
        let png = data("image/png", PNG_B64);
        let p = expand_prompt(&[1, 7, 2, 7, 3], &[&png, &png], &PLAIN).unwrap();
        assert_eq!(p.held.ids, [1, 7, 7, 2, 7, 7, 3]);
        let key = load(&png).unwrap().key;
        assert_eq!(
            p.held.media,
            [
                MediaSpan { at: 1, len: 2, key },
                MediaSpan { at: 4, len: 2, key }
            ]
        );
        let spans: Vec<MediaSpan> = p.feeds.iter().map(super::MediaFeed::span).collect();
        assert_eq!(spans, p.held.media, "one feed per span");
        let text = expand_prompt(&[1, 2], &[], &PLAIN).unwrap();
        assert_eq!((text.held, text.feeds.len()), (Held::from(vec![1, 2]), 0));
        for (ids, urls, want) in [
            (&[1, 7, 2][..], &[][..], (1, 0)),
            (&[1, 7, 7, 2][..], &[png.as_str()][..], (2, 1)),
            (&[1, 2][..], &[png.as_str()][..], (0, 1)),
        ] {
            match err(expand_prompt(ids, urls, &PLAIN)) {
                MediaError::PlaceholderCount {
                    placeholders,
                    images,
                } => assert_eq!((placeholders, images), want, "{ids:?}"),
                e => panic!("{ids:?}: {e}"),
            }
        }
        assert!(matches!(
            err(expand_prompt(&[1, 7], &[&png], &PLAIN)),
            MediaError::ImageLast
        ));
        assert!(matches!(
            err(expand_prompt(&[1, 7, 2], &["https://x/a.png"], &PLAIN)),
            MediaError::Scheme(_)
        ));
    }
}
