//! The server's media part, the same for every model: a message's image parts, the checks an image
//! file passes before it is decoded, its cache key, and the expansion of each image's placeholder
//! token into the image's span. What a model contributes is a [`MediaModel`] (`crates/vision`):
//! the placeholder text and its token, how flattened parts join ([`PartJoin`]), and one prepare.
//!
//! What a prompt, a slot or a cached state holds is a [`Held`]: the ids and the spans of the
//! images among them. [`common_prefix`] reads both, so two prompts that differ only in their
//! images share nothing of the image's span; neither it nor [`keep_whole_spans`] ends a kept
//! prefix inside a span, which is kept whole or fed whole. [`expand_prompt`] makes a chat's
//! [`Prompt`]: its held sequence and one [`MediaFeed`] per image for the engine.
//!
//! The images a server has prepared it keeps ([`PreparedImages`]): a request that resends them —
//! an agent client resends its whole history every turn — is served without decoding or preparing
//! the ones the cache still holds, and only the others share the decode budget.
//!
//! An image arrives only inside a `data:` URL holding a base64 PNG or JPEG, in either OpenAI form
//! of an `image_url` part. A clip arrives as an `input_video` part's base64 `data` and `format`, or
//! a `video_url` part's `data:video/*;base64,` URL ([`Part::Video`]); [`flatten`] probes and plans
//! it before it writes the message's text ([`crate::video`]), and [`expand_prompt`] decodes it and
//! expands each of its groups' placeholders to the group's span. Every refusal is one
//! [`MediaError`] variant that names what it refused; nothing is skipped, repaired or fetched.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use sha2::{Digest, Sha256};

pub use vision::media::ExpandError;
pub use vision::media::PartJoin;
use vision::media::{Piece, expand_marked};
use vision::{EncoderInput, Patches};
pub use vision::{FileKind, MediaModel, Prepared, Rgb8, VisionError};

use crate::video::{Clip, PlannedClip, VideoError, VideoSupport, plan_clip};

/// The decoded RGB a request's new images may retain together: the keys its server's
/// [`PreparedImages`] holds not, each distinct key charged once.
const DECODED_IMAGE_BYTES: usize = 64 * 1024 * 1024;

/// The heap bytes of prepared images one server retains ([`PreparedImages`]): what the cache
/// costs at its worst, whatever a request sends.
pub const PREPARED_IMAGE_BYTES: usize = 512 * 1024 * 1024;

/// An inline image decoded under the request's shared RGB budget, before model preparation.
pub(crate) struct DecodedImage {
    pub key: ImageKey,
    pub image: Rgb8,
}

/// Decode all inline images before preparing any, charging retained RGB once per image.
pub(crate) fn decode_images<'a>(
    urls: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<DecodedImage>, MediaError> {
    decode_images_with_limit(urls, DECODED_IMAGE_BYTES).map_err(|(_, error)| error)
}

fn decode_images_with_limit<'a>(
    urls: impl IntoIterator<Item = &'a str>,
    mut left_bytes: usize,
) -> Result<Vec<DecodedImage>, (usize, MediaError)> {
    urls.into_iter()
        .enumerate()
        .map(|(index, url)| {
            let file = load(url).map_err(|error| (index, error))?;
            let image = Rgb8::from_bytes_with_limit(&file.bytes, left_bytes)
                .map_err(|error| (index, MediaError::Decode(error)))?;
            left_bytes -= image.data.len();
            Ok(DecodedImage {
                key: file.key,
                image,
            })
        })
        .collect()
}

/// A [`MediaModel`] the server holds across its threads: the request threads prepare images on
/// it while the engine runs elsewhere.
pub type SharedMediaModel = Arc<dyn MediaModel + Send + Sync>;

/// The images a server has already prepared for its one media model, keyed by the image file's
/// bytes: a request whose image the server still holds takes its [`Prepared`] from here without a
/// decode or a prepare, and that image draws on none of the decode budget. One server holds one of
/// these beside its one model (a prepared image depends on the model); it retains at most
/// [`PREPARED_IMAGE_BYTES`] of heap in [`crate::embed_cache::Cache`]'s LRU order. An evicted image
/// is decoded and prepared again and is charged to the decode budget like a new one.
pub struct PreparedImages {
    held: Mutex<crate::embed_cache::Cache<Arc<Prepared>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// What `/metrics` reads of a [`PreparedImages`]: the lookups served and missed since the start,
/// and the bytes and entries retained now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparedStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes: usize,
    pub entries: usize,
}

impl PreparedImages {
    /// Retain prepared images within the byte limit; zero retains nothing (a request's own repeats
    /// still share one decode, which needs no retention).
    #[must_use]
    pub fn new(limit: usize) -> PreparedImages {
        PreparedImages {
            held: Mutex::new(crate::embed_cache::Cache::new(limit, |p| prepared_bytes(p))),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn get(&self, key: &ImageKey) -> Option<Arc<Prepared>> {
        let found = crate::api::relock(&self.held).get(&key.0);
        let counter = if found.is_some() {
            &self.hits
        } else {
            &self.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        found
    }

    fn insert(&self, key: ImageKey, prepared: Arc<Prepared>) {
        crate::api::relock(&self.held).insert(key.0, prepared);
    }

    #[must_use]
    pub fn stats(&self) -> PreparedStats {
        let held = crate::api::relock(&self.held);
        PreparedStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes: held.bytes(),
            entries: held.len(),
        }
    }
}

/// The heap bytes one prepared image retains: its patch values and the metadata around them.
fn prepared_bytes(prepared: &Prepared) -> usize {
    let patches = |p: &Patches| size_of::<Patches>() + p.bf16.capacity() * size_of::<u16>();
    size_of::<Prepared>()
        + match &prepared.input {
            EncoderInput::Grid(p) => patches(p),
            EncoderInput::Tiles(g) => size_of_val(g) + g.tiles.iter().map(patches).sum::<usize>(),
        }
}

/// A model that numbers its images, as MiniCPM-V's template does: the placeholder of image `i` is
/// `<image_id>i</image_id><image><unk>`, and `<image><unk>` is what every placeholder carries.
#[cfg(test)]
pub(crate) struct NumberedMedia;

#[cfg(test)]
impl MediaModel for NumberedMedia {
    fn image_placeholder(&self, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("<image_id>{index}</image_id><image><unk>"))
    }

    fn image_marker(&self) -> &str {
        "<image><unk>"
    }

    fn image_head(&self, index: usize) -> Option<Vec<u32>> {
        Some(vec![100, 200 + index as u32, 300])
    }

    fn image_token(&self) -> Option<u32> {
        Some(7)
    }

    fn prepare(&self, _: &Rgb8) -> Result<Prepared, VisionError> {
        Err(VisionError::Limits {
            detail: "the numbered test model prepares nothing".into(),
        })
    }
}

/// A model of data only that takes clips: an image token 7 that is one position, a video token 9
/// that is two positions a group, a group one frame, and a rule of one to six frames within eight
/// tokens. A group's prepared input carries its frame's first byte, so a test reads which source
/// frame each group was cut from.
#[cfg(test)]
pub(crate) struct ClipMedia;

#[cfg(test)]
pub(crate) const CLIP_RULE: vision::video::VideoRule = vision::video::VideoRule {
    fps: (1, 1),
    frames: (1, 6),
    sampler: vision::video::Sampler::Linspace,
    overflow: vision::video::Overflow::Subsample,
    temporal: 1,
    tail: vision::video::Tail::RepeatLast,
    group_tokens: 2,
    clip_tokens: 8,
    overhead: 0,
    stamp: vision::video::Stamp::None,
};

#[cfg(test)]
fn grid(first: u16, size: (usize, usize)) -> vision::Patches {
    vision::Patches {
        plan: vision::GridPlan {
            n_llm_h: 1,
            n_llm_w: 1,
            best_h: size.1,
            best_w: size.0,
        },
        n_vit_h: 0,
        n_vit_w: 0,
        patch_len: 0,
        bf16: vec![first],
    }
}

#[cfg(test)]
impl vision::video::Sizer for ClipMedia {
    fn group_size(
        &self,
        width: usize,
        height: usize,
        cap: usize,
    ) -> Result<((usize, usize), usize), VisionError> {
        if cap < 2 {
            return Err(VisionError::Video(format!(
                "a group takes 2 tokens, over the {cap} a group may"
            )));
        }
        Ok(((width, height), 2))
    }
}

#[cfg(test)]
impl MediaModel for ClipMedia {
    fn image_placeholder(&self, _index: usize) -> Cow<'_, str> {
        Cow::Borrowed("<img>")
    }

    fn image_marker(&self) -> &str {
        "<img>"
    }

    fn image_token(&self) -> Option<u32> {
        Some(7)
    }

    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        Ok(Prepared::of_grid(1, grid(0, (image.width, image.height))))
    }

    fn video_rule(&self) -> Option<&vision::video::VideoRule> {
        Some(&CLIP_RULE)
    }

    fn video_sizer(&self) -> &dyn vision::video::Sizer {
        self
    }

    fn prepare_group(
        &self,
        frames: &[Rgb8],
        size: (usize, usize),
    ) -> Result<Prepared, VisionError> {
        let [frame] = frames else {
            return Err(VisionError::Video("a group is one frame".into()));
        };
        Ok(Prepared::of_grid(2, grid(u16::from(frame.data[0]), size)))
    }

    fn video_text(&self, plan: &vision::video::ClipPlan) -> Result<Vec<Piece>, VisionError> {
        let mut out = vec![Piece::Text("[".into())];
        for g in 0..plan.groups.len() {
            if g > 0 {
                out.push(Piece::Text(",".into()));
            }
            out.push(Piece::Group(g));
        }
        out.push(Piece::Text("]".into()));
        Ok(out)
    }

    fn video_marker(&self) -> Option<&str> {
        Some("<vid>")
    }

    fn video_token(&self) -> Option<u32> {
        Some(9)
    }
}

/// One part of a message's `content` array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part<'a> {
    Text(&'a str),
    /// An `image_url` part's URL.
    Image(&'a str),
    /// An `input_video` or `video_url` part's clip.
    Video(VideoPart<'a>),
}

/// An encoded clip as a part carries it: its base64 text, and the container's name when the part
/// gives one (`input_video`'s `format`, a `video_url`'s media subtype).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoPart<'a> {
    pub data: &'a str,
    pub format: Option<&'a str>,
}

/// One medium a request carries, in the order of its placeholders in the prompt.
#[derive(Debug)]
pub enum Medium {
    /// An `image_url` part's URL.
    Image(String),
    /// A clip, probed and planned: one placeholder a group.
    Video(Box<PlannedClip>),
}

impl Medium {
    /// The media of `urls` as images.
    #[cfg(test)]
    #[must_use]
    pub fn images(urls: &[&str]) -> Vec<Medium> {
        urls.iter()
            .map(|u| Medium::Image((*u).to_owned()))
            .collect()
    }
}

/// A message's content array flattened for the chat template.
#[derive(Debug)]
pub struct Flat {
    /// The parts in order, joined by the model's [`PartJoin`], each image as the model's
    /// placeholder and each clip as its text with one placeholder a group.
    pub text: String,
    /// The media, in the order of their placeholders in `text`.
    pub media: Vec<Medium>,
}

impl Flat {
    /// The images' URLs, in the order of their placeholders.
    #[cfg(test)]
    #[must_use]
    pub fn images(&self) -> Vec<&str> {
        self.media
            .iter()
            .filter_map(|m| match m {
                Medium::Image(url) => Some(url.as_str()),
                Medium::Video(_) => None,
            })
            .collect()
    }
}

/// What a request's clips need beyond the model: the decoder and its threads, and the positions of
/// context the clips may still take, which each planned clip spends.
pub struct VideoCtx {
    support: VideoSupport,
    left: usize,
}

impl VideoCtx {
    /// A context for clips that may take `budget` positions together.
    #[must_use]
    pub fn new(support: VideoSupport, budget: usize) -> VideoCtx {
        VideoCtx {
            support,
            left: budget,
        }
    }

    /// A context for a request that carries no clip: a clip part is refused by name.
    #[cfg(test)]
    #[must_use]
    pub fn none() -> VideoCtx {
        VideoCtx::new(VideoSupport::default(), 0)
    }
}

/// An image's cache key: the sha256 of its file bytes. It is known before the decode, so a server
/// whose [`PreparedImages`] holds the key serves the image without decoding or repeating the
/// model's image preprocessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey(pub [u8; 32]);

/// An image file a request carried: its format checked against its bytes, keyed, not decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageFile {
    pub kind: FileKind,
    pub bytes: Vec<u8>,
    pub key: ImageKey,
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

/// Everything the media part refuses, each naming what it refused. All but [`ExpandError::EmptySpan`]
/// are the request's fault.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// The projector supports embedding delimiters rather than a chat placeholder.
    #[error("the projector has no chat image token; use its embedding input")]
    NoChatImageToken,
    /// A content part that is not a JSON object.
    #[error("a content part must be an object")]
    PartNotObject,
    /// A content part whose `type` is missing or not one this server takes (the JSON of the value,
    /// or `missing`).
    #[error(
        "content part type {0}: only \"text\", \"image_url\", \"input_video\" and \"video_url\" parts are accepted"
    )]
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
    /// A video part in a message whose role is not `user`.
    #[error("a video part in a {0} message: only user messages carry video")]
    VideoRole(String),
    /// Message text that carries the model's video marker.
    #[error("the message text carries the video marker {0}; send a clip as an input_video part")]
    VideoMarkerInText(String),
    /// A `video_url` of a media type other than `video/*`.
    #[error("media type {0:?}: a video_url takes a data: URL of video/*, such as video/mp4")]
    VideoMime(String),
    /// A clip for a model that declares no video rule.
    #[error("this model takes no video: its projector declares no video rule")]
    NoVideoRule,
    /// The projector has no chat video token.
    #[error("the projector has no chat video token; use its embedding input")]
    NoChatVideoToken,
    /// The server's refusal of the clip: no decoder, a file the decoder cannot read, a decode that
    /// ended early ([`VideoError`]).
    #[error(transparent)]
    Clip(#[from] VideoError),
    /// The planner's or the model's refusal of the clip: over its budget at its fewest frames, a
    /// frame count it does not take, a group it cannot prepare ([`vision::video`]).
    #[error(transparent)]
    Video(VisionError),
    /// The prompt's placeholders and the request's media do not pair up, with a clip among them.
    #[error(
        "{placeholders} media placeholder(s) in the prompt for {images} image(s) and {groups} \
         video group(s)"
    )]
    MediaCount {
        placeholders: usize,
        images: usize,
        groups: usize,
    },
    /// A placeholder of one kind where the request's media order has the other.
    #[error(
        "placeholder {index} of the prompt is {found}, where the request's media order has {want}"
    )]
    MediaOrder {
        index: usize,
        found: &'static str,
        want: &'static str,
    },
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
    /// The prompt's placeholders and the request's images do not pair up, or a prepared image's
    /// span is empty ([`ExpandError::EmptySpan`], the model's fault, not the request's).
    #[error(transparent)]
    Expand(#[from] ExpandError),
    /// A prompt whose last position is an image's: the reply is read after a text position.
    #[error("the prompt ends with an image; its last position must be text")]
    ImageLast,
    /// A numbered placeholder whose number in the prompt's ids is not its image's position: the
    /// ids before image `index`'s token are not [`MediaModel::image_head`]`(index)`.
    #[error(
        "image {index}: the ids before its token do not carry its number; \
         the template must render each image's placeholder with the image's position"
    )]
    ImageNumber { index: usize },
}

impl fmt::Display for ImageKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// One content part: `{"type":"text","text":…}`, or `{"type":"image_url","image_url":{"url":…}}`
/// or `{"type":"image_url","image_url":…}`, or a clip: `{"type":"input_video","input_video":
/// {"data":<base64>,"format":<container>}}` or `{"type":"video_url","video_url":{"url":<data: URL
/// of video/*>}}` (the URL also as the string itself).
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
        Some(Value::String(t)) if t == "input_video" => match part.get("input_video") {
            Some(Value::Object(o)) => {
                let Some(Value::String(data)) = o.get("data") else {
                    return Err(MediaError::PartField {
                        ty: "input_video",
                        need: "a string 'input_video.data' of base64",
                    });
                };
                let format = match o.get("format") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(f)) => Some(f.as_str()),
                    Some(_) => {
                        return Err(MediaError::PartField {
                            ty: "input_video",
                            need: "a string 'input_video.format', or none",
                        });
                    }
                };
                Ok(Part::Video(VideoPart { data, format }))
            }
            _ => Err(MediaError::PartField {
                ty: "input_video",
                need: "an 'input_video' object with a string 'data' and a 'format'",
            }),
        },
        Some(Value::String(t)) if t == "video_url" => {
            let url = match part.get("video_url") {
                Some(Value::String(url)) => url,
                Some(Value::Object(o)) => match o.get("url") {
                    Some(Value::String(url)) => url,
                    _ => {
                        return Err(MediaError::PartField {
                            ty: "video_url",
                            need: "a string 'video_url.url'",
                        });
                    }
                },
                _ => {
                    return Err(MediaError::PartField {
                        ty: "video_url",
                        need: "a 'video_url' object with a string 'url', or a string",
                    });
                }
            };
            let (mime, data) = split_data_url(url)?;
            let Some((_, subtype)) = mime
                .split_once('/')
                .filter(|(kind, sub)| kind.eq_ignore_ascii_case("video") && !sub.is_empty())
            else {
                return Err(MediaError::VideoMime(short(mime)));
            };
            Ok(Part::Video(VideoPart {
                data,
                format: Some(subtype),
            }))
        }
        Some(other) => Err(MediaError::PartType(short(&other.to_string()))),
        None => Err(MediaError::PartType("missing".into())),
    }
}

/// A message's content array as the chat template's text: text parts as they are, each image as
/// the model's placeholder and each clip as the model's text for it with one placeholder a group,
/// joined by the model's [`PartJoin`]: the same separator between every two parts, or
/// llama-server's rule (a `\n` before a text part unless it is the first text or follows a
/// placeholder, and none around a placeholder). Only a `user` message carries media: the template
/// may move other roles' text (V4.1's gathers system messages first), and the i-th placeholder of
/// the prompt must be the i-th medium. `first_image` is the index of the first image this message
/// places, the count of the images the messages before it placed; the model writes each image's
/// index into its placeholder ([`MediaModel::image_placeholder`]). A clip is probed and planned
/// here, before its text is written, because the text holds what the plan chose; its frames are
/// decoded later, by [`expand_prompt`].
pub fn flatten(
    role: &str,
    parts: &[Value],
    model: &dyn MediaModel,
    first_image: usize,
    video: &mut VideoCtx,
) -> Result<Flat, MediaError> {
    flatten_with(role, parts, model, first_image, video, &|t| {
        Cow::Borrowed(t)
    })
}

/// [`flatten`] with `text` applied to each text part after the marker check: the text the template
/// gets for it. The placeholders and a clip's text are the server's and are not passed through it.
pub(crate) fn flatten_with(
    role: &str,
    parts: &[Value],
    model: &dyn MediaModel,
    first_image: usize,
    video: &mut VideoCtx,
    text_of: &dyn for<'t> Fn(&'t str) -> Cow<'t, str>,
) -> Result<Flat, MediaError> {
    let join = model.part_join();
    let mut text = String::new();
    let mut media = Vec::new();
    let mut images = 0;
    let mut after_media = false;
    for (i, part) in parts.iter().enumerate() {
        let (piece, is_media) = match parse_part(part)? {
            Part::Text(t) => {
                check_text(t, model)?;
                (text_of(t), false)
            }
            Part::Image(_) if role != "user" => return Err(MediaError::ImageRole(short(role))),
            Part::Video(_) if role != "user" => return Err(MediaError::VideoRole(short(role))),
            Part::Image(url) => {
                let piece = model.image_placeholder(first_image + images);
                images += 1;
                media.push(Medium::Image(url.to_owned()));
                (piece, true)
            }
            Part::Video(clip) => {
                let (clip_text, planned) = plan_part(clip, model, video)?;
                media.push(Medium::Video(Box::new(planned)));
                (Cow::Owned(clip_text), true)
            }
        };
        match join {
            PartJoin::Uniform(sep) if i > 0 => text.push_str(sep),
            PartJoin::LlamaServer if !is_media && !after_media && !text.is_empty() => {
                text.push('\n');
            }
            _ => {}
        }
        text.push_str(&piece);
        after_media = is_media;
    }
    Ok(Flat { text, media })
}

/// One clip part, probed and planned under what the request's context has left: the clip's text
/// with the model's video marker for each group's placeholder, and the plan. The decoder's absence,
/// a model with no video rule and a base64 that is not one are refused before anything runs.
fn plan_part(
    clip: VideoPart<'_>,
    model: &dyn MediaModel,
    video: &mut VideoCtx,
) -> Result<(String, PlannedClip), MediaError> {
    model.video_rule().ok_or(MediaError::NoVideoRule)?;
    let marker = model.video_marker().ok_or(MediaError::NoChatVideoToken)?;
    model.video_token().ok_or(MediaError::NoChatVideoToken)?;
    let decoder = video.support.decoder()?;
    let bytes = base64(clip.data)?;
    let left = video.left;
    let planned = plan_clip(
        model,
        decoder,
        Clip::new(bytes, clip.format),
        left,
        video.support.thread_count(),
    )?;
    video.left = left.saturating_sub(planned.tokens());
    let mut text = String::new();
    let mut group = 0;
    for piece in planned.pieces() {
        match piece {
            Piece::Text(t) => text.push_str(t),
            Piece::Group(n) if *n == group => {
                group += 1;
                text.push_str(marker);
            }
            Piece::Group(n) => {
                return Err(MediaError::Video(VisionError::Video(format!(
                    "the model's clip text places group {n} where group {group} is next"
                ))));
            }
        }
    }
    if group != planned.groups() {
        return Err(MediaError::Video(VisionError::Video(format!(
            "the model's clip text places {group} of the plan's {} groups",
            planned.groups()
        ))));
    }
    Ok((text, planned))
}

/// Refuse message text that carries the model's image marker ([`MediaModel::image_marker`]), the
/// part of a placeholder that makes an image slot whatever its index, or its video marker: only a
/// media part makes one.
pub fn check_text(text: &str, model: &dyn MediaModel) -> Result<(), MediaError> {
    let marker = model.image_marker();
    if text.contains(marker) {
        return Err(MediaError::PlaceholderInText(marker.to_owned()));
    }
    if let Some(marker) = model.video_marker().filter(|m| text.contains(m)) {
        return Err(MediaError::VideoMarkerInText(marker.to_owned()));
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

/// Decode a single image within the serving RGB budget and prepare it for `model`.
pub fn prepare(file: &ImageFile, model: &dyn MediaModel) -> Result<Prepared, MediaError> {
    let image = Rgb8::from_bytes_with_limit(&file.bytes, DECODED_IMAGE_BYTES)
        .map_err(MediaError::Decode)?;
    model.prepare(&image).map_err(MediaError::Prepare)
}

/// Decode all image URLs under one serving RGB budget before preparing any, returning the
/// failing image's zero-based index and error.
pub fn prepare_images<'a>(
    urls: impl IntoIterator<Item = &'a str>,
    model: &dyn MediaModel,
) -> Result<Vec<Prepared>, (usize, MediaError)> {
    prepare_images_with_limit(urls, model, DECODED_IMAGE_BYTES)
}

fn prepare_images_with_limit<'a>(
    urls: impl IntoIterator<Item = &'a str>,
    model: &dyn MediaModel,
    max_bytes: usize,
) -> Result<Vec<Prepared>, (usize, MediaError)> {
    decode_images_with_limit(urls, max_bytes)?
        .into_iter()
        .enumerate()
        .map(|(index, image)| {
            let prepared = model
                .prepare(&image.image)
                .map_err(|error| (index, MediaError::Prepare(error)))?;
            if prepared.span_len == 0 {
                return Err((index, MediaError::Expand(ExpandError::EmptySpan(index))));
            }
            Ok(prepared)
        })
        .collect()
}

/// Prepare a decoded video's image parts through the model's video rule: the frames were sampled
/// by the sender, so each is a group of its own, sized by the model's sizer at its own shape as a
/// clip's frames are, and the count is held to the rule's frame bounds.
pub(crate) fn prepare_video(
    parts: &[Value],
    model: &dyn MediaModel,
    images: &mut impl Iterator<Item = DecodedImage>,
) -> Result<Vec<MediaFeed>, MediaError> {
    let rule =
        model
            .video_rule()
            .filter(|rule| rule.temporal == 1)
            .ok_or(MediaError::PartField {
                ty: "decoded video",
                need: "a projector with a video rule of one-frame groups",
            })?;
    let (least, most) = rule.frames;
    if parts.len() < least || parts.len() > most {
        return Err(MediaError::Prepare(VisionError::Limits {
            detail: format!("decoded video takes {least} to {most} sampled frames"),
        }));
    }
    parts
        .iter()
        .map(|part| {
            let Part::Image(_) = parse_part(part)? else {
                return Err(MediaError::PartField {
                    ty: "decoded video",
                    need: "only image_url frames after its prefix",
                });
            };
            let frame = images.next().ok_or(MediaError::PartField {
                ty: "decoded video",
                need: "one decoded image per frame",
            })?;
            let (size, _) = model
                .video_sizer()
                .group_size(frame.image.width, frame.image.height, rule.group_tokens)
                .map_err(MediaError::Prepare)?;
            let prepared = model
                .prepare_group(std::slice::from_ref(&frame.image), size)
                .map_err(MediaError::Prepare)?;
            Ok(MediaFeed {
                at: 0,
                key: frame.key,
                prepared: Arc::new(prepared),
            })
        })
        .collect()
}

/// What a placeholder of the prompt is, for the refusal that names it.
fn kind_name(is_image: bool) -> &'static str {
    if is_image {
        "an image placeholder"
    } else {
        "a video placeholder"
    }
}

/// The prompt of a rendered chat whose ids are `ids` and whose media are `media`, in the order of
/// their placeholders: each image loaded and prepared on the calling thread, each clip decoded and
/// prepared by group, each placeholder's token expanded to its span, and one feed per image and per
/// group. The placeholders are counted, their kinds and numbers checked and the last position
/// checked before anything is decoded, so a placeholder typed into the text, which pairs with no
/// medium, costs no decode. Every image is a new one here: a server that keeps its prepared images
/// expands through [`expand_prompt_with`].
pub fn expand_prompt(
    ids: &[u32],
    media: &[Medium],
    model: &dyn MediaModel,
) -> Result<Prompt, MediaError> {
    expand_prompt_with(ids, media, model, &PreparedImages::new(0))
}

/// [`expand_prompt`] beside the server's prepared images: an image whose key the cache holds is
/// served from it without a decode or a prepare, and only the images the cache holds not share
/// the decode budget — a key the request itself repeats counted once. The image loads still run
/// (the key is the file bytes' hash), and every image is decoded before any is prepared.
pub fn expand_prompt_with(
    ids: &[u32],
    media: &[Medium],
    model: &dyn MediaModel,
    images: &PreparedImages,
) -> Result<Prompt, MediaError> {
    let image_token = model.image_token().ok_or(MediaError::NoChatImageToken)?;
    let video_token = model.video_token();
    let is_mark = |t: u32| t == image_token || Some(t) == video_token;
    // Each placeholder the media want, in order: an image's index, or `None` for a group.
    let mut want: Vec<Option<usize>> = Vec::new();
    let mut image_count = 0;
    for m in media {
        match m {
            Medium::Image(_) => {
                want.push(Some(image_count));
                image_count += 1;
            }
            Medium::Video(clip) => {
                video_token.ok_or(MediaError::NoChatVideoToken)?;
                want.extend(std::iter::repeat_n(None, clip.groups()));
            }
        }
    }
    let placeholders = ids.iter().filter(|&&t| is_mark(t)).count();
    if placeholders != want.len() {
        let groups = want.len() - image_count;
        return Err(if groups == 0 {
            MediaError::Expand(ExpandError::PlaceholderCount {
                placeholders,
                images: image_count,
            })
        } else {
            MediaError::MediaCount {
                placeholders,
                images: image_count,
                groups,
            }
        });
    }
    if ids.last().is_some_and(|&t| is_mark(t)) {
        return Err(MediaError::ImageLast);
    }
    check_placeholders(ids, &want, image_token, model)?;
    // Load and key every image, take the keys the cache holds, and decode the rest under the
    // budget, each distinct key once; nothing is prepared until every image is decoded.
    let mut left_bytes = DECODED_IMAGE_BYTES;
    let mut served: HashMap<ImageKey, Arc<Prepared>> = HashMap::new();
    let mut fresh: Vec<(ImageKey, Rgb8)> = Vec::new();
    let mut loaded: HashSet<ImageKey> = HashSet::new();
    let mut keys = Vec::new();
    for m in media {
        let Medium::Image(url) = m else {
            continue;
        };
        let file = load(url)?;
        keys.push(file.key);
        if served.contains_key(&file.key) || !loaded.insert(file.key) {
            continue;
        }
        match images.get(&file.key) {
            Some(prepared) => {
                served.insert(file.key, prepared);
            }
            None => {
                let image = Rgb8::from_bytes_with_limit(&file.bytes, left_bytes)
                    .map_err(MediaError::Decode)?;
                left_bytes -= image.data.len();
                fresh.push((file.key, image));
            }
        }
    }
    for (key, image) in &fresh {
        let prepared = Arc::new(model.prepare(image).map_err(MediaError::Prepare)?);
        images.insert(*key, Arc::clone(&prepared));
        served.insert(*key, prepared);
    }
    let mut keys = keys.into_iter();
    let mut ready: Vec<(ImageKey, Arc<Prepared>)> = Vec::with_capacity(want.len());
    for m in media {
        match m {
            Medium::Image(_) => {
                let key = keys.next().expect("one key per image medium, loaded above");
                let prepared = served
                    .get(&key)
                    .expect("every image's key was resolved above");
                ready.push((key, Arc::clone(prepared)));
            }
            Medium::Video(clip) => {
                ready.extend(
                    clip.prepare(model)?
                        .into_iter()
                        .map(|g| (g.key, g.prepared)),
                );
            }
        }
    }
    let lens: Vec<usize> = ready.iter().map(|(_, p)| p.span_len).collect();
    let expanded = expand_marked(ids, is_mark, &lens)?;
    let feeds: Vec<MediaFeed> = expanded
        .spans
        .iter()
        .zip(ready)
        .map(|(span, (key, prepared))| MediaFeed {
            at: span.start,
            key,
            prepared,
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

/// The i-th placeholder of the prompt is the kind of the i-th medium wants, and for a model whose
/// placeholder numbers its images ([`MediaModel::image_head`]) the ids before an image's token end
/// with the head of that image, so a template that numbered the images otherwise than by position
/// is refused before anything is decoded.
fn check_placeholders(
    ids: &[u32],
    want: &[Option<usize>],
    image_token: u32,
    model: &dyn MediaModel,
) -> Result<(), MediaError> {
    let marks = ids
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == image_token || Some(t) == model.video_token());
    for (index, ((at, &token), want)) in marks.zip(want).enumerate() {
        let is_image = token == image_token;
        // One id for both kinds cannot tell them apart: the counts are the whole check.
        if model.video_token() != Some(image_token) && is_image != want.is_some() {
            return Err(MediaError::MediaOrder {
                index,
                found: kind_name(is_image),
                want: kind_name(want.is_some()),
            });
        }
        let Some(image) = *want else {
            continue;
        };
        let Some(head) = model.image_head(image) else {
            continue;
        };
        if !ids[..at].ends_with(&head) {
            return Err(MediaError::ImageNumber { index: image });
        }
    }
    Ok(())
}

/// A `data:` URL's media type and its base64 data. Scheme and `;base64` are matched without case
/// (RFC 3986 §3.1, RFC 2045 §5.1); the media type is returned as written, parameters and all.
fn split_data_url(url: &str) -> Result<(&str, &str), MediaError> {
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
    Ok((mime, payload))
}

/// A `data:` URL's declared image format and its data: the media type is one of the accepted, a
/// media type with parameters is not.
fn data_url(url: &str) -> Result<(FileKind, &str), MediaError> {
    let (mime, payload) = split_data_url(url)?;
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
pub(crate) fn base64(text: &str) -> Result<Vec<u8>, MediaError> {
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

/// `bytes` as standard base64 with padding (RFC 4648 §4), the text [`base64`] reads back.
pub use ::embed::encode_base64;

fn short(s: &str) -> String {
    match s.char_indices().nth(64) {
        Some((at, _)) => format!("{}…", &s[..at]),
        None => s.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;
    use std::sync::Arc;

    use serde_json::{Value, json};
    use vision::{GridPlan, Patches};

    use super::{
        ExpandError, FileKind, Held, ImageKey, MediaError, MediaModel, MediaSpan, Medium,
        NumberedMedia, PREPARED_IMAGE_BYTES, Part, PartJoin, Prepared, PreparedImages, Rgb8,
        VideoCtx, VisionError, common_prefix, encode_base64, expand_prompt, expand_prompt_with,
        flatten, keep_whole_spans, load, parse_part, prepare, prepared_bytes,
    };

    /// A 2×1 RGB PNG of the colour (200, 100, 50).
    const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg==";
    /// Its sha256 (hashlib).
    const PNG_SHA256: &str = "29f48d50a7db2b049176bebb2523c066f97d61b7aac0ee54d826f916bd22ccb6";

    /// A model of data only: a placeholder, its token, a join rule, and a prepare that gives one
    /// span position per pixel.
    struct Stub {
        join: PartJoin,
    }

    impl MediaModel for Stub {
        fn image_placeholder(&self, _index: usize) -> Cow<'_, str> {
            Cow::Borrowed("<img>")
        }

        fn image_marker(&self) -> &str {
            "<img>"
        }

        fn image_token(&self) -> Option<u32> {
            Some(7)
        }

        fn part_join(&self) -> PartJoin {
            self.join
        }

        fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
            let plan = GridPlan {
                n_llm_h: image.height,
                n_llm_w: image.width,
                best_h: image.height,
                best_w: image.width,
            };
            Ok(Prepared::of_grid(
                image.width * image.height,
                Patches {
                    plan,
                    n_vit_h: 0,
                    n_vit_w: 0,
                    patch_len: 0,
                    bf16: Vec::new(),
                },
            ))
        }
    }

    const PLAIN: Stub = Stub {
        join: PartJoin::Uniform("\n"),
    };
    const LLAMA: Stub = Stub {
        join: PartJoin::LlamaServer,
    };

    fn data(mime: &str, b64: &str) -> String {
        format!("data:{mime};base64,{b64}")
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
        let flat = flatten("user", &parts, &PLAIN, 0, &mut VideoCtx::none()).unwrap();
        assert_eq!(flat.text, "look\n<img>\nand\n<img>");
        assert_eq!(flat.images(), ["data:a", "data:b"]);
        let wide = Stub {
            join: PartJoin::Uniform("\n\n"),
        };
        assert_eq!(
            flatten("user", &parts, &wide, 0, &mut VideoCtx::none())
                .unwrap()
                .text,
            "look\n\n<img>\n\nand\n\n<img>"
        );
        let text_only = [
            json!({"type": "text", "text": "a"}),
            json!({"type": "text", "text": "b"}),
        ];
        let flat = flatten("system", &text_only, &PLAIN, 0, &mut VideoCtx::none()).unwrap();
        assert_eq!((flat.text.as_str(), flat.images().len()), ("a\nb", 0));
    }

    /// llama-server's join rule edge by edge (`concat_content_parts` in `common/chat.cpp`): a `\n`
    /// before a text part once any text has been written and the part before it is not a
    /// placeholder, none before or after a placeholder; an empty text part writes nothing but is
    /// still text.
    #[test]
    fn flatten_follows_llama_servers_rule() {
        let part = |p: &str| match p {
            "<img>" => json!({"type": "image_url", "image_url": "data:a"}),
            t => json!({"type": "text", "text": t}),
        };
        for (parts, want) in [
            (vec!["a", "b"], "a\nb"),
            (vec!["a", "<img>", "b"], "a<img>b"),
            (vec!["<img>", "a", "b"], "<img>a\nb"),
            (vec!["a", "b", "<img>"], "a\nb<img>"),
            (vec!["<img>", "<img>"], "<img><img>"),
            (vec!["a", "<img>", "<img>", "b"], "a<img><img>b"),
            (vec!["", "b"], "b"),
            (vec!["a", "", "b"], "a\n\nb"),
            (vec!["a", "<img>", "", "b"], "a<img>\nb"),
            (vec!["<img>", "", "b"], "<img>\nb"),
        ] {
            let json: Vec<Value> = parts.iter().map(|p| part(p)).collect();
            let flat = flatten("user", &json, &LLAMA, 0, &mut VideoCtx::none()).unwrap();
            assert_eq!(flat.text, want, "{parts:?}");
            let n = parts.iter().filter(|&&p| p == "<img>").count();
            assert_eq!(flat.images().len(), n, "{parts:?}");
        }
    }

    /// A toy template and tokenizer for the hand cases: a message is `role:content;`, the stub's
    /// placeholder is its token 7, every other character is its own code.
    fn chat(messages: &[(&str, Value)], model: &Stub) -> (String, Vec<u32>, Vec<String>) {
        let (mut text, mut urls) = (String::new(), Vec::new());
        for (role, content) in messages {
            let body = match content {
                Value::String(t) => t.clone(),
                Value::Array(parts) => {
                    let flat =
                        flatten(role, parts, model, urls.len(), &mut VideoCtx::none()).unwrap();
                    urls.extend(flat.images().into_iter().map(str::to_owned));
                    flat.text
                }
                other => panic!("{other}"),
            };
            text.push_str(&format!("{role}:{body};"));
        }
        (text.clone(), tokens(&text), urls)
    }

    fn tokens(text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for (i, chunk) in text.split("<img>").enumerate() {
            if i > 0 {
                out.push(7);
            }
            out.extend(chunk.chars().map(|c| c as u32));
        }
        out
    }

    /// J1: the prompts of llama-server's chat path, by hand. A chat is flattened by the model's
    /// join rule, rendered, tokenized and expanded: the ids are the text's ids with each placeholder
    /// replaced by its image's span (the PNG is 2 by 1, a span of 2).
    #[test]
    fn j1_flatten_then_expand_on_hand_cases() {
        let png = data("image/png", PNG_B64);
        let image = |url: &str| json!({"type": "image_url", "image_url": {"url": url}});
        let text = |t: &str| json!({"type": "text", "text": t});
        let cases: [(Vec<(&str, Value)>, &str); 3] = [
            (
                vec![("user", json!([text("look"), image(&png), text("and")]))],
                "user:look<img>and;",
            ),
            (
                vec![
                    ("system", json!([text("You are brief.")])),
                    ("user", json!([image(&png)])),
                ],
                "system:You are brief.;user:<img>;",
            ),
            (
                vec![
                    ("user", json!([image(&png), text("what is it?")])),
                    ("assistant", json!("a cat")),
                    ("user", json!([text("and this"), image(&png)])),
                ],
                "user:<img>what is it?;assistant:a cat;user:and this<img>;",
            ),
        ];
        for (messages, want) in cases {
            let (rendered, ids, urls) = chat(&messages, &LLAMA);
            assert_eq!(rendered, want);
            let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
            let p = expand_prompt(&ids, &Medium::images(&urls), &LLAMA).unwrap();
            let mut expect = Vec::new();
            for (i, chunk) in want.split("<img>").enumerate() {
                if i > 0 {
                    expect.extend([7, 7]);
                }
                expect.extend(chunk.chars().map(|c| c as u32));
            }
            assert_eq!(p.held.ids, expect, "{want}");
            assert_eq!(p.feeds.len(), want.matches("<img>").count(), "{want}");
        }
        // The same chats under a uniform join differ where a part follows a part.
        let (rendered, _, _) = chat(
            &[("user", json!([text("look"), image(&png), text("and")]))],
            &PLAIN,
        );
        assert_eq!(rendered, "user:look\n<img>\nand;");
    }

    #[test]
    fn flatten_refusals_are_named() {
        let image = [json!({"type": "image_url", "image_url": "data:a"})];
        for role in ["system", "assistant", "tool", "developer"] {
            match err(flatten(role, &image, &PLAIN, 0, &mut VideoCtx::none())) {
                MediaError::ImageRole(r) => assert_eq!(r, role),
                e => panic!("{role}: {e}"),
            }
        }
        let text = [json!({"type": "text", "text": "an <img> by hand"})];
        assert!(matches!(
            err(flatten("user", &text, &PLAIN, 0, &mut VideoCtx::none())),
            MediaError::PlaceholderInText(_)
        ));
        let bad = [
            json!({"type": "image_url", "image_url": "data:a"}),
            json!({"type": "video_url"}),
        ];
        assert!(matches!(
            err(flatten("user", &bad, &PLAIN, 0, &mut VideoCtx::none())),
            MediaError::PartField {
                ty: "video_url",
                ..
            }
        ));
    }

    /// A model that numbers its images gets each placeholder's index from the images before it:
    /// the first image of a message is `first_image`, the next one more.
    #[test]
    fn flatten_numbers_the_images_it_places() {
        let image = |u: &str| json!({"type": "image_url", "image_url": u});
        let parts = [
            json!({"type": "text", "text": "look"}),
            image("data:a"),
            image("data:b"),
        ];
        for first in [0, 5] {
            let flat =
                flatten("user", &parts, &NumberedMedia, first, &mut VideoCtx::none()).unwrap();
            let (a, b) = (first, first + 1);
            assert_eq!(
                flat.text,
                format!(
                    "look\n<image_id>{a}</image_id><image><unk>\n<image_id>{b}</image_id><image><unk>"
                )
            );
            assert_eq!(flat.images(), ["data:a", "data:b"]);
        }
    }

    /// A typed placeholder is refused by name whatever its index, for it carries the marker every
    /// placeholder carries: the one the next image would get, another, and the marker alone.
    #[test]
    fn a_typed_numbered_placeholder_is_refused_by_name() {
        for typed in [
            "an <image_id>0</image_id><image><unk> by hand",
            "an <image_id>9</image_id><image><unk> by hand",
            "an <image><unk> by hand",
        ] {
            let parts = [json!({"type": "text", "text": typed})];
            match err(flatten(
                "user",
                &parts,
                &NumberedMedia,
                0,
                &mut VideoCtx::none(),
            )) {
                MediaError::PlaceholderInText(m) => assert_eq!(m, "<image><unk>", "{typed}"),
                e => panic!("{typed}: {e}"),
            }
        }
        let tag = [json!({"type": "text", "text": "the <image_id>0</image_id> tag"})];
        assert_eq!(
            flatten("user", &tag, &NumberedMedia, 0, &mut VideoCtx::none())
                .unwrap()
                .text,
            "the <image_id>0</image_id> tag",
            "the index tag alone makes no slot"
        );
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
            let f = load(&data(mime, &encode_base64(&jpeg))).unwrap();
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
            assert_eq!(encode_base64(plain.as_bytes()), b64, "{plain}");
        }
        let all: Vec<u8> = (0..=255).collect();
        for n in [254, 255, 256] {
            assert_eq!(super::base64(&encode_base64(&all[..n])).unwrap(), &all[..n]);
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
        match err(load(&data("image/png", &encode_base64(&jpeg)))) {
            MediaError::Magic { declared, found } => {
                assert_eq!((declared, found), (FileKind::Png, FileKind::Jpeg));
            }
            e => panic!("{e}"),
        }
        for (bytes, kind) in [
            (&b"RIFF\x24\0\0\0WEBPVP8 "[..], FileKind::Webp),
            (b"GIF89a\x01\0\x01\0", FileKind::Gif),
        ] {
            match err(load(&data("image/png", &encode_base64(bytes)))) {
                MediaError::Format(k) => assert_eq!(k, kind),
                e => panic!("{kind:?}: {e}"),
            }
        }
        match err(load(&data("image/png", &encode_base64(b"BM\x3a\0\0\0")))) {
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
        let patches = p.grid().unwrap();
        assert_eq!((patches.plan.best_w, patches.plan.best_h), (2, 1));
        let jpeg = [0xFF, 0xD8, 0xFF, 0xDB, 0, 4, 0, 0];
        let broken = load(&data("image/jpeg", &encode_base64(&jpeg))).unwrap();
        assert!(matches!(
            err(prepare(&broken, &PLAIN)),
            MediaError::Decode(VisionError::Decode(_))
        ));
    }

    #[test]
    fn bulk_images_share_one_rgb_budget_and_preserve_error_indices() {
        let url = data("image/png", PNG_B64);
        let urls = [url.as_str(), url.as_str()];
        let images = super::prepare_images_with_limit(urls, &PLAIN, 12).unwrap();
        assert_eq!(
            images
                .iter()
                .map(|image| image.span_len)
                .collect::<Vec<_>>(),
            [2, 2]
        );
        assert!(matches!(
            super::prepare_images_with_limit(urls, &NumberedMedia, 11),
            Err((1, MediaError::Decode(VisionError::Limits { .. })))
        ));
        assert!(matches!(
            super::prepare_images_with_limit(urls, &NumberedMedia, 12),
            Err((0, MediaError::Prepare(VisionError::Limits { .. })))
        ));
        assert!(matches!(
            super::prepare_images([url.as_str(), "https://example.com/image.png"], &PLAIN),
            Err((1, MediaError::Scheme(_)))
        ));
        struct Empty;
        impl MediaModel for Empty {
            fn image_placeholder(&self, _index: usize) -> Cow<'_, str> {
                Cow::Borrowed("<img>")
            }

            fn image_marker(&self) -> &str {
                "<img>"
            }

            fn image_token(&self) -> Option<u32> {
                Some(7)
            }

            fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
                let mut prepared = PLAIN.prepare(image)?;
                prepared.span_len = 0;
                Ok(prepared)
            }
        }
        assert!(matches!(
            super::prepare_images([url.as_str()], &Empty),
            Err((0, MediaError::Expand(ExpandError::EmptySpan(0))))
        ));
    }

    #[test]
    fn all_inline_images_decode_before_any_model_prepare() {
        let url = data("image/png", PNG_B64);
        let broken = data("image/png", &encode_base64(b"\x89PNG\r\n\x1a\n"));
        // NumberedMedia refuses any prepare; the later decode must fail first instead.
        assert!(matches!(
            expand_prompt(
                &[100, 200, 300, 7, 100, 201, 300, 7, 1],
                &Medium::images(&[&url, &broken]),
                &NumberedMedia,
            ),
            Err(MediaError::Decode(VisionError::Decode(_)))
        ));
    }

    /// A flat `width`×`height` RGB PNG of one shade, as a data URL: another shade is another
    /// file, so each shade is its own image key. Its decode is `width * height * 3` bytes.
    fn flat_png(width: u32, height: u32, shade: u8) -> String {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&vec![shade; width as usize * height as usize * 3])
                .unwrap();
        }
        data("image/png", &encode_base64(&bytes))
    }

    /// A model whose every prepare gives the same small span and counts its calls: the
    /// prepared-image tests read how many images a request carried to a prepare.
    struct Tally {
        span_len: usize,
        prepares: Cell<usize>,
    }

    impl MediaModel for Tally {
        fn image_placeholder(&self, _index: usize) -> Cow<'_, str> {
            Cow::Borrowed("<img>")
        }

        fn image_marker(&self) -> &str {
            "<img>"
        }

        fn image_token(&self) -> Option<u32> {
            Some(7)
        }

        fn prepare(&self, _image: &Rgb8) -> Result<Prepared, VisionError> {
            self.prepares.set(self.prepares.get() + 1);
            Ok(Prepared::of_grid(
                self.span_len,
                super::grid(0, (self.span_len, 1)),
            ))
        }
    }

    /// The ids of a prompt of `n` images: text, then each placeholder followed by text.
    fn shot_ids(n: usize) -> Vec<u32> {
        let mut ids = vec![1];
        ids.extend(std::iter::repeat_n([7, 1], n).flatten());
        ids
    }

    /// An agent resends its whole conversation every turn, so a request carries every screenshot
    /// the conversation already took. The images an earlier request of the conversation served
    /// cost no decode and no prepare — only the new image draws on the decode budget — and a
    /// history whose images together pass the budget is served. Twenty-one flat 1024×1024 PNGs
    /// decode to 66,060,288 bytes together, inside the 67,108,864-byte budget; a twenty-second
    /// crosses it [derived].
    #[test]
    fn a_history_of_prepared_images_costs_only_its_new_images() {
        let history: Vec<String> = (0..21).map(|i| flat_png(1024, 1024, i)).collect();
        let model = Tally {
            span_len: 3,
            prepares: Cell::new(0),
        };
        let images = PreparedImages::new(PREPARED_IMAGE_BYTES);
        let old: Vec<&str> = history.iter().map(String::as_str).collect();
        let first =
            expand_prompt_with(&shot_ids(21), &Medium::images(&old), &model, &images).unwrap();
        assert_eq!(model.prepares.get(), 21);
        let mut resent = history;
        resent.push(flat_png(1024, 1024, 99));
        let urls: Vec<&str> = resent.iter().map(String::as_str).collect();
        let second =
            expand_prompt_with(&shot_ids(22), &Medium::images(&urls), &model, &images).unwrap();
        assert_eq!(model.prepares.get(), 22, "only the new image is prepared");
        assert_eq!(second.feeds.len(), 22);
        for (was, now) in first.feeds.iter().zip(&second.feeds) {
            assert!(Arc::ptr_eq(&was.prepared, &now.prepared));
        }
    }

    /// The budget still bounds a request whose images are all new: nothing is prepared past the
    /// decode, and the refusal names the decode budget.
    #[test]
    fn only_new_images_over_the_budget_are_refused_by_name() {
        let urls: Vec<String> = (0..22).map(|i| flat_png(1024, 1024, i)).collect();
        let model = Tally {
            span_len: 3,
            prepares: Cell::new(0),
        };
        let images = PreparedImages::new(PREPARED_IMAGE_BYTES);
        let fresh: Vec<&str> = urls.iter().map(String::as_str).collect();
        match err(expand_prompt_with(
            &shot_ids(22),
            &Medium::images(&fresh),
            &model,
            &images,
        )) {
            MediaError::Decode(VisionError::Limits { detail }) => assert!(
                detail.ends_with("decoded-image bytes remaining"),
                "{detail}"
            ),
            e => panic!("{e}"),
        }
        assert_eq!(model.prepares.get(), 0);
    }

    /// A request that sends the same image twice prepares it once and both placeholders hold the
    /// same prepared image, with no retention of anything (a zero cache).
    #[test]
    fn repeats_within_one_request_share_one_prepare() {
        let png = data("image/png", PNG_B64);
        let model = Tally {
            span_len: 2,
            prepares: Cell::new(0),
        };
        let images = PreparedImages::new(0);
        let urls = [png.as_str(), png.as_str()];
        let p =
            expand_prompt_with(&[1, 7, 2, 7, 3], &Medium::images(&urls), &model, &images).unwrap();
        assert_eq!(model.prepares.get(), 1);
        assert_eq!(p.held.ids, [1, 7, 7, 2, 7, 7, 3]);
        assert!(Arc::ptr_eq(&p.feeds[0].prepared, &p.feeds[1].prepared));
    }

    /// The retained images stay inside their bound: the least recently used is the first out, a
    /// hit moves its entry to the recent end, and an entry the whole bound cannot hold is not
    /// retained and evicts nothing.
    #[test]
    fn prepared_images_are_bounded_least_recently_first() {
        let image = |values: usize| {
            Arc::new(Prepared::of_grid(
                1,
                Patches {
                    plan: GridPlan {
                        n_llm_h: 1,
                        n_llm_w: 1,
                        best_h: 1,
                        best_w: 1,
                    },
                    n_vit_h: 1,
                    n_vit_w: 1,
                    patch_len: values,
                    bf16: vec![0; values],
                },
            ))
        };
        let one = prepared_bytes(&image(4));
        let images = PreparedImages::new(2 * one);
        let key = |k: u8| ImageKey([k; 32]);
        images.insert(key(1), Arc::clone(&image(4)));
        images.insert(key(2), Arc::clone(&image(4)));
        // A hit moves key 1 to the recent end, so key 2 is the one a third entry evicts.
        assert!(images.get(&key(1)).is_some());
        images.insert(key(3), Arc::clone(&image(4)));
        assert!(images.get(&key(1)).is_some());
        assert!(images.get(&key(3)).is_some());
        assert!(images.get(&key(2)).is_none(), "the least recently used");
        let held = crate::api::relock(&images.held);
        assert!(
            held.bytes() <= held.limit(),
            "{} <= {}",
            held.bytes(),
            held.limit()
        );
        assert_eq!(held.len(), 2);
        drop(held);
        // An entry the whole bound cannot hold is not retained and evicts nothing.
        let huge = image(4 * one);
        images.insert(key(9), Arc::clone(&huge));
        assert!(images.get(&key(1)).is_some());
        assert!(images.get(&key(9)).is_none());
        let stats = images.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (4, 2, 2));
        assert_eq!(stats.bytes, 2 * one);
    }

    /// The expansion's refusals reach the client as `ExpandError` wrote them: the text is the
    /// one message, and the variants stay apart for `api::media_error`'s split (`EmptySpan` the
    /// model's 500, `ImageNumber` and the placeholder count the request's 400).
    #[test]
    fn expansion_refusals_keep_their_text_and_variants() {
        let empty = MediaError::from(ExpandError::EmptySpan(1));
        assert!(matches!(
            empty,
            MediaError::Expand(ExpandError::EmptySpan(1))
        ));
        assert_eq!(empty.to_string(), "image 1 has an empty span");
        let count = MediaError::from(ExpandError::PlaceholderCount {
            placeholders: 2,
            images: 1,
        });
        assert!(matches!(
            count,
            MediaError::Expand(ExpandError::PlaceholderCount {
                placeholders: 2,
                images: 1
            })
        ));
        assert_eq!(
            count.to_string(),
            "2 image placeholder(s) in the prompt for 1 image(s)"
        );
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
        let p = expand_prompt(&[1, 7, 2, 7, 3], &Medium::images(&[&png, &png]), &PLAIN).unwrap();
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
            match err(expand_prompt(ids, &Medium::images(urls), &PLAIN)) {
                MediaError::Expand(ExpandError::PlaceholderCount {
                    placeholders,
                    images,
                }) => assert_eq!((placeholders, images), want, "{ids:?}"),
                e => panic!("{ids:?}: {e}"),
            }
        }
        assert!(matches!(
            err(expand_prompt(&[1, 7], &Medium::images(&[&png]), &PLAIN)),
            MediaError::ImageLast
        ));
        assert!(matches!(
            err(expand_prompt(
                &[1, 7, 2],
                &Medium::images(&["https://x/a.png"]),
                &PLAIN
            )),
            MediaError::Scheme(_)
        ));
    }

    /// A model that numbers its images finds each number in the ids before the image's token,
    /// before any image is decoded: a second image that carries the first's number, or a first
    /// with no head before it, is refused by name, and the right numbers reach the prepare, which
    /// the numbered test model then refuses.
    #[test]
    fn a_numbered_placeholder_must_carry_its_images_position() {
        let png = data("image/png", PNG_B64);
        let head = |i: u32| [100, 200 + i, 300];
        let ids = |a: u32, b: u32| [&[1][..], &head(a), &[7, 2], &head(b), &[7, 3]].concat();
        for (ids, index) in [(ids(0, 0), 1), (vec![7, 2], 0)] {
            let urls = vec![png.as_str(); if index == 1 { 2 } else { 1 }];
            match err(expand_prompt(&ids, &Medium::images(&urls), &NumberedMedia)) {
                MediaError::ImageNumber { index: got } => assert_eq!(got, index, "{ids:?}"),
                e => panic!("{ids:?}: {e}"),
            }
        }
        assert!(matches!(
            err(expand_prompt(
                &ids(0, 1),
                &Medium::images(&[&png, &png]),
                &NumberedMedia
            )),
            MediaError::Prepare(_)
        ));
    }

    // ---- clips

    use super::{ClipMedia, MediaFeed};
    use crate::video::{MockClips, VideoError};

    /// An `input_video` part whose clip is the mock's `spec` (frames, rate, size).
    fn clip_part(spec: &str) -> Value {
        json!({"type": "input_video", "input_video": {
            "data": encode_base64(spec.as_bytes()), "format": "mp4"
        }})
    }

    /// The ids of `text` for the clip model: its two markers are the ids 7 and 9, any other
    /// character its own id.
    fn clip_ids(text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(r) = rest.strip_prefix("<img>") {
                out.push(7);
                rest = r;
            } else if let Some(r) = rest.strip_prefix("<vid>") {
                out.push(9);
                rest = r;
            } else {
                let c = rest.chars().next().expect("a character");
                out.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
        }
        out
    }

    /// A clip part comes in both spellings: `input_video`'s base64 and optional container name, and
    /// a `video_url` of a `data:video/*;base64,` URL as an object or as the string. Each refusal
    /// names what was wrong.
    #[test]
    fn a_clip_part_has_two_spellings_and_named_refusals() {
        let video = |part: Value| match parse_part(&part) {
            Ok(Part::Video(v)) => (v.data.to_owned(), v.format.map(str::to_owned)),
            other => panic!("{part}: {other:?}"),
        };
        assert_eq!(
            video(json!({"type": "input_video", "input_video": {"data": "AAAA", "format": "mkv"}})),
            ("AAAA".into(), Some("mkv".into()))
        );
        assert_eq!(
            video(json!({"type": "input_video", "input_video": {"data": "AAAA"}})),
            ("AAAA".into(), None)
        );
        let url = "data:video/mp4;base64,AAAA";
        for part in [
            json!({"type": "video_url", "video_url": {"url": url}}),
            json!({"type": "video_url", "video_url": url}),
        ] {
            assert_eq!(video(part), ("AAAA".into(), Some("mp4".into())));
        }
        let refused = |part: Value| parse_part(&part).unwrap_err();
        for part in [
            json!({"type": "input_video"}),
            json!({"type": "input_video", "input_video": {"format": "mp4"}}),
            json!({"type": "input_video", "input_video": {"data": 3}}),
            json!({"type": "input_video", "input_video": {"data": "AAAA", "format": 3}}),
            json!({"type": "video_url"}),
            json!({"type": "video_url", "video_url": {"url": 3}}),
        ] {
            assert!(matches!(refused(part), MediaError::PartField { .. }));
        }
        assert!(matches!(
            refused(json!({"type": "video_url", "video_url": "https://x/v.mp4"})),
            MediaError::Scheme(s) if s == "https"
        ));
        assert!(matches!(
            refused(json!({"type": "video_url", "video_url": "data:video/mp4,AAAA"})),
            MediaError::NotBase64(_)
        ));
        assert!(matches!(
            refused(json!({"type": "video_url", "video_url": "data:image/png;base64,AAAA"})),
            MediaError::VideoMime(m) if m == "image/png"
        ));
        assert!(matches!(
            refused(json!({"type": "video_url", "video_url": "data:video/;base64,AAAA"})),
            MediaError::VideoMime(_)
        ));
        assert!(matches!(
            refused(json!({"type": "video_url", "video_url": "AAAA"})),
            MediaError::NotAUrl
        ));
        assert!(
            refused(json!({"type": "audio"}))
                .to_string()
                .contains("\"input_video\" and \"video_url\"")
        );
    }

    /// A clip is probed and planned before its text is written: the text holds one placeholder per
    /// group the plan made, in order, the model's own text around them, joined as any part is, and
    /// the clip's plan spends the request's context budget so a second clip gets what is left.
    #[test]
    fn a_clip_is_planned_before_its_text_is_written() {
        let mock = MockClips::new();
        let parts = [
            json!({"type": "text", "text": "see"}),
            clip_part("30 10/1 4x2"),
            json!({"type": "text", "text": "and"}),
            json!({"type": "image_url", "image_url": "data:a"}),
        ];
        let mut ctx = VideoCtx::new(mock.support(3), 100);
        let flat = flatten("user", &parts, &ClipMedia, 0, &mut ctx).unwrap();
        // 30 frames at 10 fps is 3 seconds: three frames, three groups of two tokens.
        assert_eq!(flat.text, "see\n[<vid>,<vid>,<vid>]\nand\n<img>");
        assert_eq!(flat.images(), ["data:a"]);
        let [Medium::Video(clip), Medium::Image(_)] = &flat.media[..] else {
            panic!("a clip then an image: {:?}", flat.media);
        };
        assert_eq!((clip.groups(), clip.tokens(), clip.frames()), (3, 6, 3));
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "nothing decodes before the expansion"
        );
        // The plan spent 6 of the budget: a second clip may take what is left (two groups' worth
        // would be 4; only 2 tokens remain, so one group).
        let mut ctx = VideoCtx::new(mock.support(3), 8);
        let two = [clip_part("30 10/1 4x2"), clip_part("30 10/1 4x2")];
        let flat = flatten("user", &two, &ClipMedia, 0, &mut ctx).unwrap();
        assert_eq!(flat.text, "[<vid>,<vid>,<vid>]\n[<vid>]");
    }

    /// Where a clip cannot be taken it is refused by name, before any text is written or frame
    /// decoded: no decoder, a model with no video rule, another role, the marker typed into the
    /// text, base64 that is not, a probe that fails, a clip over its budget at its fewest frames.
    #[test]
    fn a_clip_is_refused_by_name_where_it_cannot_be_taken() {
        let mock = MockClips::new();
        let on = |budget| VideoCtx::new(mock.support(2), budget);
        let clip = [clip_part("30 10/1 4x2")];
        let absent = err(flatten("user", &clip, &ClipMedia, 0, &mut VideoCtx::none()));
        assert!(matches!(absent, MediaError::Clip(VideoError::Absent(_))));
        assert_eq!(
            absent.to_string(),
            "video input needs ffmpeg and ffprobe on PATH (no decoder was set); install ffmpeg, \
             or send the clip as decoded one-FPS image_url frames after a Video: text part"
        );
        assert!(matches!(
            err(flatten("user", &clip, &NumberedMedia, 0, &mut on(100))),
            MediaError::NoVideoRule
        ));
        assert!(matches!(
            err(flatten(
                "user",
                &clip,
                &Stub {
                    join: PartJoin::Uniform("\n")
                },
                0,
                &mut on(100)
            )),
            MediaError::NoVideoRule
        ));
        assert!(matches!(
            err(flatten("assistant", &clip, &ClipMedia, 0, &mut on(100))),
            MediaError::VideoRole(r) if r == "assistant"
        ));
        assert!(matches!(
            err(flatten(
                "user",
                &[json!({"type": "text", "text": "a <vid> b"})],
                &ClipMedia,
                0,
                &mut on(100)
            )),
            MediaError::VideoMarkerInText(m) if m == "<vid>"
        ));
        let bad64 = [json!({"type": "input_video", "input_video": {"data": "a b"}})];
        assert!(matches!(
            err(flatten("user", &bad64, &ClipMedia, 0, &mut on(100))),
            MediaError::Base64Char { .. }
        ));
        let unreadable = [clip_part("not a clip")];
        let e = err(flatten("user", &unreadable, &ClipMedia, 0, &mut on(100)));
        assert!(matches!(e, MediaError::Clip(VideoError::Probe(_))), "{e}");
        let e = err(flatten("user", &clip, &ClipMedia, 0, &mut on(1)));
        assert!(matches!(e, MediaError::Video(_)), "{e}");
        assert!(e.to_string().starts_with("video: "), "{e}");
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    /// The prompt's placeholders are matched to the request's media by id and order: each group's
    /// token is expanded to its span and fed, the clip decoded once at the plan's indices on the
    /// seat's threads, and every pairing fault is named before anything is decoded.
    #[test]
    fn clip_placeholders_are_matched_by_id_and_decoded_in_order() {
        let mock = MockClips::new();
        let png = data("image/png", PNG_B64);
        let mut ctx = VideoCtx::new(mock.support(3), 100);
        let parts = [
            clip_part("30 10/1 4x2"),
            json!({"type": "image_url", "image_url": png}),
        ];
        let flat = flatten("user", &parts, &ClipMedia, 0, &mut ctx).unwrap();
        let ids = clip_ids(&format!("a{}z", flat.text));
        let p = expand_prompt(&ids, &flat.media, &ClipMedia).unwrap();
        // "a[", three groups of two video positions with commas, "]", the image's one, "z".
        let want = clip_ids("a[<vid><vid>,<vid><vid>,<vid><vid>]\n<img>z");
        assert_eq!(want.len(), 14);
        assert_eq!(p.held.ids, want);
        assert_eq!(p.feeds.len(), 4);
        let at: Vec<usize> = p.feeds.iter().map(|f| f.at).collect();
        assert_eq!(at, [2, 5, 8, 12]);
        // Each group carries the source frame it was cut from; the image none.
        let frames: Vec<u16> = p
            .feeds
            .iter()
            .map(|f| f.prepared.grid().unwrap().bf16[0])
            .collect();
        assert_eq!(frames, [0, 14, 29, 0]);
        let keys: std::collections::HashSet<_> = p.feeds.iter().map(|f| f.key).collect();
        assert_eq!(keys.len(), 4, "each group is its own medium");
        assert_eq!(
            p.held.media,
            p.feeds.iter().map(MediaFeed::span).collect::<Vec<_>>()
        );
        assert_eq!(*mock.calls.lock().unwrap(), [(vec![0, 14, 29], 3)]);
    }

    /// The same clip planned the same way keys the same, group for group: a prefix cache keeps its
    /// spans across requests. Other bytes planned the same way key differently, every group: the
    /// plan alone does not name a medium.
    #[test]
    fn a_clip_planned_the_same_way_keys_the_same() {
        let mock = MockClips::new();
        let mut ctx = VideoCtx::new(mock.support(3), 100);
        let mut keys = |parts: &[Value]| {
            let flat = flatten("user", parts, &ClipMedia, 0, &mut ctx).unwrap();
            let ids = clip_ids(&flat.text);
            let p = expand_prompt(&ids, &flat.media, &ClipMedia).unwrap();
            p.feeds.iter().map(|f| f.key).collect::<Vec<_>>()
        };
        let parts = [clip_part("30 10/1 4x2")];
        let (first, second) = (keys(&parts), keys(&parts));
        assert_eq!(first, second);
        let distinct: std::collections::HashSet<_> = first.iter().collect();
        assert_eq!(distinct.len(), 3);
        // The mock reads a clip's first three words, so a fourth makes other bytes of the same clip.
        let other = keys(&[clip_part("30 10/1 4x2 again")]);
        let calls = mock.calls.lock().unwrap();
        assert_eq!(calls[0], calls[2], "the same plan");
        assert_eq!(other.len(), first.len());
        assert!(
            first.iter().all(|key| !other.contains(key)),
            "other bytes keyed as the first clip"
        );
    }

    /// A placeholder pairing fault is named, and nothing is decoded for it: too few or too many
    /// placeholders for the media, an image's marker where the request has a clip's group (and
    /// the reverse), a model with no video token, a prompt that ends in a group.
    #[test]
    fn a_clip_pairing_fault_is_named_before_any_decode() {
        let mock = MockClips::new();
        let png = data("image/png", PNG_B64);
        let mut ctx = VideoCtx::new(mock.support(1), 100);
        let parts = [
            json!({"type": "image_url", "image_url": png}),
            clip_part("30 10/1 4x2"),
        ];
        let flat = flatten("user", &parts, &ClipMedia, 0, &mut ctx).unwrap();
        let run = |text: &str| err(expand_prompt(&clip_ids(text), &flat.media, &ClipMedia));
        // An image, then three groups: the right ids pass the pairing (checked by the success
        // above); each fault below is the same prompt altered.
        assert!(matches!(
            run("a<img>b<vid>,<vid>c"),
            MediaError::MediaCount {
                placeholders: 3,
                images: 1,
                groups: 3
            }
        ));
        assert!(matches!(
            run("a<img><vid><vid><vid><vid>"),
            MediaError::MediaCount {
                placeholders: 5,
                images: 1,
                groups: 3
            }
        ));
        let swapped = run("a<vid><img><vid><vid>z");
        assert!(
            matches!(&swapped, MediaError::MediaOrder { index: 0, found, want }
                if *found == "a video placeholder" && *want == "an image placeholder"),
            "{swapped}"
        );
        assert!(
            swapped
                .to_string()
                .contains("placeholder 0 of the prompt is a video placeholder")
        );
        let image_where_group = run("a<img><vid><img><vid>z");
        assert!(matches!(
            image_where_group,
            MediaError::MediaOrder { index: 2, .. }
        ));
        assert!(matches!(
            run("a<img><vid><vid><vid>"),
            MediaError::ImageLast
        ));
        assert!(matches!(
            err(expand_prompt(
                &clip_ids("<img><vid><vid><vid>z"),
                &flat.media,
                &NumberedMedia
            )),
            MediaError::NoChatVideoToken
        ));
        // A typed marker with no clip behind it is the pairing's fault like any typed placeholder.
        assert!(matches!(
            err(expand_prompt(&clip_ids("a<vid>b"), &[], &ClipMedia)),
            MediaError::Expand(ExpandError::PlaceholderCount {
                placeholders: 1,
                images: 0
            })
        ));
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "no pairing fault decoded a frame"
        );
    }
}
