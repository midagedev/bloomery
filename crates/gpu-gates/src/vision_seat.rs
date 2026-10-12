//! The vision seat every serve seat that takes images shares: the `--mmproj` flags, the projector
//! file's checks against the text model, where the image encoder sits, the plan's reserve for it,
//! the sequence table's per-position bytes, the card binding around a tower call, and the one
//! `vision` record.
//!
//! A seat owns only what its text model's architecture forces (its tensors, its recurrent state,
//! how an image's rows enter its prompt call); everything below is the same for every seat, and the
//! checks are keyed by the projector's type ([`rules`]), never by seat:
//!
//! * [`Flags`] — llama-server's spelling: `--mmproj FILE`, `--no-mmproj`, `--image-min-tokens N`,
//!   `--image-max-tokens N`. A limit not given is the projector's own default ([`Rule`]: 8 and 4096
//!   for `qwen3vl_merger`, the file's 16 and 8000 for `glm5v`); a projector that plans its own
//!   sizes takes no limit.
//! * [`Projector`] — the file opened on the host: its projector type refused by name unless the
//!   seat takes it, the limits resolved once, the tower's bytes on its card, the model's image
//!   input, and the row width; [`Projector::check_text`] holds it against the text model, every
//!   mismatch refused by name.
//! * [`open_seat`] — the flags to the opened encoder: the projector held against the text model and
//!   seated for the plan ([`Opened`]); [`Resident::open`] loads its tower on that card.
//! * [`EncoderSeat`] — where the tower sits: the smallest card outside the plan, else a reserve on
//!   the stage card ([`reserve_on`] puts the same reserve into a placed plan and a whole-fit probe).
//! * [`RESERVE`], [`reserve_on`], [`reserve_in`], [`with_reserve`], [`refusal`] and
//!   [`table_bytes_per_position`] —
//!   the plan's side of it, in [`crate::image_plan`] (a placement module of every build takes it):
//!   the tower's reserve in a placed plan and a whole-fit probe, and what a plan adds per position
//!   of a sequence for a media load.
//! * [`on_encoder_card`] — binds the tower's card for a tower call and the model's card back.
//! * [`Resident`] — a serve seat's resident encoder: the tower on its card, its stream, the row
//!   width and the model's image input; [`Resident::encode_with`] encodes a media call's images
//!   and reads them back as the rows a prompt call splices. The seat gives only its model's splice
//!   of one image's rows.
//! * [`Resident::encode_with`] and [`Encoded::prompt`] — the two steps of a seat's media call
//!   (GLM, Qwen3.6 and V4.1; Clef takes the first and makes its own hidden-state call), the one
//!   place the seats' rows pass between the readback and the prompt call: the images encoded on the model's card and read back as [`Encoded`] rows
//!   with the seat's own splice and kinds ([`Resident::encode`] is the merger towers' splice,
//!   [`merged_cells`]; a reference reads the rows with [`Encoded::rows`]), then the one prompt
//!   call over the spans the seat's builder makes of them ([`app::Session::prompt_media`]). What
//!   a seat does beside them (GLM's draft turn-off between the two, V4.1's draft refusal before
//!   them) and how its splice and spans are made stay the seat's.
//! * [`SpanRows::media_spans`] and [`SpanRows::image_spans`] — the call's spans as the plain
//!   [`MediaSpan`]s of a body that takes nothing beside a span's positions and rows, and as the
//!   [`ImageSpan`]s of a body that also takes each image's token grid, which [`span_rows_of`]
//!   records in each span ([`SpanAt::grid`]) from the image the span is of — a seat passes none.
//! * [`span_rows_of`] — the host half of a resident's encode, which takes the readback as a
//!   closure; every tower writes one row of `width` a cell of the plan's token grid.
//! * [`seat_record`] — the `vision_seat` line a seat prints with its plan, before any load (also the
//!   projector [`kept`] drops, as `seat=not-loaded`); [`record`] — the `vision` line after the
//!   tower's load.

use std::fmt::Display;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloomery_gpu::media::{ImageSpan, MediaBody, MediaSpan};
use cuda_core::CudaContext;
use gguf::Gguf;
use model::placement::workstation::{CardSpec, DeviceInfo};
use vision::arch::card::CardBytes;
use vision::arch::qwen3vl::TokenLimits;
use vision::arch::{deepseek41v, glm5v, qwen3vl};
use vision::{MediaModel, Patches, Prepared, VisionError};

use crate::GateError;
use crate::generate::Place;
pub use crate::image_plan::{
    RESERVE, refusal, reserve_in, reserve_on, table_bytes_per_position, with_reserve,
};
use crate::record::{self, Record};

/// A [`MediaModel`] the server holds across its threads (`serve::media::SharedMediaModel`).
pub type SharedMedia = Arc<dyn MediaModel + Send + Sync>;

/// The image-input flags of a serve seat, llama-server's spelling.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    mmproj: Option<PathBuf>,
    no_mmproj: bool,
    min_tokens: Option<usize>,
    max_tokens: Option<usize>,
}

/// The flags [`Flags::take`] reads, for a seat's usage line.
pub const USAGE: &str = "[--mmproj FILE | --no-mmproj] [--image-min-tokens N] \
                         [--image-max-tokens N]";

impl Flags {
    /// Reads `flag` — and its value from `rest`, which `--no-mmproj` has none of — when it is one
    /// of the four; `false` when it is another flag, which the caller reads.
    pub fn take<'a>(
        &mut self,
        flag: &str,
        rest: &mut impl Iterator<Item = &'a str>,
    ) -> Result<bool, GateError> {
        let mut value = || {
            rest.next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))
        };
        match flag {
            "--mmproj" => self.mmproj = Some(PathBuf::from(value()?)),
            "--no-mmproj" => self.no_mmproj = true,
            "--image-min-tokens" => self.min_tokens = Some(count(flag, value()?)?),
            "--image-max-tokens" => self.max_tokens = Some(count(flag, value()?)?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The encoder file `--mmproj` names; `None` serves no image. `--mmproj` and `--no-mmproj`
    /// together are refused by name.
    pub fn mmproj(&self) -> Result<Option<&Path>, GateError> {
        match (&self.mmproj, self.no_mmproj) {
            (Some(p), true) => Err(format!(
                "--mmproj {} with --no-mmproj: the two flags contradict; give one",
                p.display()
            )
            .into()),
            (Some(p), false) => Ok(Some(p)),
            (None, _) => Ok(None),
        }
    }

    /// [`Flags::mmproj`], refused by name when a token limit is given with no encoder file to
    /// bound: no `--mmproj`, and no `--hf` repo whose projector [`Flags::encoder`] could pick.
    pub fn wanted(&self) -> Result<Option<&Path>, GateError> {
        match self.mmproj()? {
            None if self.sets_limits() && !self.may_pick() => Err(
                "--image-min-tokens/--image-max-tokens without --mmproj: no image input to bound"
                    .into(),
            ),
            file => Ok(file),
        }
    }

    /// Whether [`Flags::encoder`] may take the `--hf` repo's projector: a build with the encoder,
    /// a model file from `--hf`, and neither `--mmproj` nor `--no-mmproj`.
    fn may_pick(&self) -> bool {
        cfg!(feature = "vision")
            && self.mmproj.is_none()
            && !self.no_mmproj
            && crate::model_file::hf_repo().is_some()
    }

    /// The encoder file the seat serves images with, as llama-server's `-hf` picks it:
    /// `--mmproj`'s file, else, when the seat `takes` an image in this mode, the projector the
    /// `--hf` repo holds beside the model ([`crate::model_file::hf_projector`]) unless
    /// `--no-mmproj` says none. `None` serves no image.
    pub fn encoder(&self, takes: bool) -> Result<Option<PathBuf>, GateError> {
        record::require(
            record::groups::vision_seat::RECORDS,
            "vision_seat::Flags::encoder",
        );
        record::require(
            record::groups::model_file::FETCH,
            "vision_seat::Flags::encoder",
        );
        match self.mmproj()? {
            Some(file) => Ok(Some(file.to_owned())),
            None if takes && self.may_pick() => crate::model_file::hf_projector(),
            None => Ok(None),
        }
    }

    /// Whether `--image-min-tokens` or `--image-max-tokens` was given.
    #[must_use]
    pub fn sets_limits(&self) -> bool {
        self.min_tokens.is_some() || self.max_tokens.is_some()
    }

    /// The image-token limits: the flags' values over `default`, the limits the projector starts
    /// from when no flag gives one ([`Rule`]), refused by name when `min` is 0 or above `max`.
    pub fn limits_over(&self, default: TokenLimits) -> Result<TokenLimits, GateError> {
        TokenLimits::new(
            self.min_tokens.unwrap_or(default.min()),
            self.max_tokens.unwrap_or(default.max()),
        )
        .map_err(|e| format!("--image-min-tokens/--image-max-tokens: {e}").into())
    }
}

/// A flag's count: a positive integer, or refused by name.
fn count(flag: &str, v: &str) -> Result<usize, GateError> {
    v.parse::<usize>()
        .map_err(|e| format!("{flag} {v:?}: {e}").into())
}

/// What the common checks know of a projector type: the ids its prompt carries and the text model
/// it needs. One row a type, as data.
#[derive(Debug)]
pub struct Rule {
    /// `clip.projector_type`.
    pub projector: &'static str,
    /// The ids of the image's span, each with the string the text model's vocabulary holds it as.
    pub ids: &'static [(u32, &'static str)],
    /// The IMROPE sections the text model must rotate by; `None` asks nothing of its rope.
    pub text_rope: Option<[u32; 4]>,
    /// Whether the image-token limits steer the projector's plan: V4.1 plans from its file's own
    /// grid, so a limit given for it is refused by name.
    pub takes_limits: bool,
    /// The row width the projector writes (the text model's embedding width).
    out_dim: fn(&Gguf) -> Result<usize, VisionError>,
    /// The limits the projector starts from when a flag gives none: its size rule's own (V4.1
    /// takes none, and the limits it is opened with steer nothing).
    default_limits: fn(&Gguf) -> Result<TokenLimits, VisionError>,
    /// The most image tokens one image takes, at `limits`.
    cap: fn(&Gguf, TokenLimits) -> Result<usize, VisionError>,
}

/// The projector types the seats run, by `clip.projector_type`: the one table that maps a type to
/// its host crate's module.
#[must_use]
pub fn rules() -> &'static [Rule] {
    static RULES: &[Rule] = &[
        Rule {
            projector: qwen3vl::PROJECTOR_TYPE,
            ids: &[
                (qwen3vl::media::VISION_START_ID, "<|vision_start|>"),
                (qwen3vl::media::VISION_END_ID, "<|vision_end|>"),
                (qwen3vl::media::IMAGE_TOKEN_ID, "<|image_pad|>"),
            ],
            text_rope: Some([11, 11, 10, 0]),
            takes_limits: true,
            out_dim: |f| Ok(qwen3vl::Hparams::read(f)?.out_dim),
            default_limits: |_| Ok(TokenLimits::DEFAULT),
            cap: |_, limits| Ok(limits.max()),
        },
        Rule {
            projector: deepseek41v::PROJECTOR_TYPE,
            ids: &[(
                deepseek41v::IMAGE_TOKEN_ID,
                deepseek41v::media::IMAGE_PLACEHOLDER,
            )],
            text_rope: None,
            takes_limits: false,
            out_dim: |f| Ok(deepseek41v::Hparams::read(f)?.out_dim),
            default_limits: |_| Ok(TokenLimits::DEFAULT),
            cap: |f, _| Ok(deepseek41v::Hparams::read(f)?.grid().max_tokens),
        },
        Rule {
            projector: glm5v::PROJECTOR_TYPE,
            ids: &[
                (glm5v::media::BEGIN_OF_IMAGE_ID, "<|begin_of_image|>"),
                (glm5v::media::END_OF_IMAGE_ID, "<|end_of_image|>"),
                (glm5v::media::IMAGE_TOKEN_ID, "<|image|>"),
            ],
            text_rope: None,
            takes_limits: true,
            out_dim: |f| Ok(glm5v::Hparams::read(f)?.out_dim),
            default_limits: |f| glm5v::Hparams::read(f)?.default_limits(),
            cap: |_, limits| Ok(limits.max()),
        },
    ];
    RULES
}

/// What the text model's file says that its projector must agree with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextFacts {
    /// `embedding_length`.
    pub n_embd: usize,
    /// The rope's `rope.dimension_sections` when the text rotates by IMROPE; `None` for any other
    /// rope.
    pub imrope: Option<[u32; 4]>,
    /// The text model's `ple.image_token_id`.
    pub ple: PleImage,
}

/// A text model's `ple.image_token_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PleImage {
    /// The architecture has no per-layer embedding.
    Absent,
    /// It has one and the file carries no `ple.image_token_id`.
    NoKey,
    Id(u32),
}

/// An encoder file opened on the host, held against its seat by [`Projector::check_text`]. Nothing
/// of it is on a card: the tower's load is the seat's.
pub struct Projector {
    pub path: PathBuf,
    pub rule: &'static Rule,
    /// The tower's weight and activation bytes on its card.
    pub card: CardBytes,
    /// The model's image input.
    pub media: SharedMedia,
    /// The width of a row the projector writes.
    pub out_dim: usize,
    /// The image-token limits the file was opened at, resolved once: the flags over the
    /// projector's own defaults ([`Rule`]).
    pub limits: TokenLimits,
    /// The most image tokens one image takes, which the tower's activations are sized for.
    pub cap: usize,
}

impl Projector {
    /// The encoder file at `path`, opened as the type its `clip.projector_type` names and refused
    /// by name when that type is not one of `takes` (the projector types the seat's text model
    /// can run), when the file is not one the host crate runs, or when the limits `flags` gives
    /// steer a projector that plans its own sizes. The limits are `flags`' over the projector's
    /// own defaults, once: the host crate, the cap and the tower all read [`Projector::limits`].
    pub fn open(path: &Path, flags: &Flags, takes: &[&str]) -> Result<Projector, GateError> {
        let at = |e: &dyn Display| format!("--mmproj {}: {e}", path.display());
        let file = Gguf::open(path).map_err(|e| at(&e))?;
        let kind = file
            .value("clip.projector_type")
            .and_then(gguf::Value::as_str)
            .ok_or_else(|| at(&"clip.projector_type is absent or not a string"))?;
        if !takes.contains(&kind) {
            return Err(at(&format!(
                "projector type {kind:?}; this seat's text model takes {takes:?}"
            ))
            .into());
        }
        let rule = rules()
            .iter()
            .find(|r| r.projector == kind)
            .ok_or_else(|| {
                at(&format!(
                    "projector type {kind:?} has no rule in the vision seat's table"
                ))
            })?;
        if flags.sets_limits() && !rule.takes_limits {
            return Err(at(&format!(
                "the {kind} projector plans an image from its file's own grid; \
                 --image-min-tokens and --image-max-tokens steer only the projectors whose \
                 size rule they bound"
            ))
            .into());
        }
        let default = (rule.default_limits)(&file).map_err(|e| at(&e))?;
        let limits = flags.limits_over(default)?;
        let (card, media) = vision::arch::open(&file, limits).map_err(|e| at(&e))?;
        let out_dim = (rule.out_dim)(&file).map_err(|e| at(&e))?;
        let cap = (rule.cap)(&file, limits).map_err(|e| at(&e))?;
        Ok(Projector {
            path: path.to_owned(),
            rule,
            card,
            media,
            out_dim,
            limits,
            cap,
        })
    }

    /// The projector seated on `seat`: the file, the model's image input, the cap, the card
    /// figure and the limits it was opened at ([`Projector::limits`]), which the tower is loaded
    /// at.
    fn seated(self, seat: EncoderSeat) -> Opened {
        Opened {
            path: self.path,
            seat,
            media: self.media,
            cap: self.cap,
            card: self.card,
            limits: self.limits,
        }
    }

    /// The projector against the text model: every mismatch refused by name. `token` is the
    /// text model's vocabulary, an id's string.
    pub fn check_text(
        &self,
        text: &TextFacts,
        token: &dyn Fn(u32) -> Option<String>,
    ) -> Result<(), GateError> {
        let at = |e: &dyn Display| -> GateError {
            format!("--mmproj {}: {e}", self.path.display()).into()
        };
        if self.out_dim != text.n_embd {
            return Err(at(&format!(
                "the encoder's aligner rows are {} wide, the model's embedding is {}",
                self.out_dim, text.n_embd
            )));
        }
        for &(id, want) in self.rule.ids {
            match token(id) {
                Some(got) if got == want => {}
                got => {
                    return Err(at(&format!(
                        "the {} projector's image id {id} is {want:?}, and the model's \
                         vocabulary holds {}",
                        self.rule.projector,
                        got.map_or("no such token".to_owned(), |g| format!("{g:?}"))
                    )));
                }
            }
        }
        if let Some(want) = self.rule.text_rope
            && text.imrope != Some(want)
        {
            return Err(at(&format!(
                "the {} projector places image rows by IMROPE sections {want:?}, and the text \
                 model's rope is {}",
                self.rule.projector,
                text.imrope.map_or("not IMROPE".to_owned(), |s| format!(
                    "IMROPE sections {s:?}"
                ))
            )));
        }
        let image = self.media.image_token();
        match text.ple {
            PleImage::Absent => {}
            PleImage::NoKey => {
                return Err(at(&"the text model carries per-layer embeddings and no \
                    ple.image_token_id, which must be the projector's image token"));
            }
            PleImage::Id(id) if id != image => {
                return Err(at(&format!(
                    "the text model's ple.image_token_id is {id}, the projector's image token \
                     is {image}"
                )));
            }
            PleImage::Id(_) => {}
        }
        Ok(())
    }
}

/// What `--mmproj` opened for a seat's plan: the encoder file held against the text model, where
/// its tower sits and the bytes the plan reserves for it.
#[derive(Clone)]
pub struct Opened {
    pub path: PathBuf,
    pub seat: EncoderSeat,
    /// The model's image input, which the server renders and expands image parts by.
    pub media: SharedMedia,
    /// The most image tokens one image takes ([`Projector::cap`]).
    pub cap: usize,
    /// The tower's card figure: the plan's reserve on a stage-reserve seat, and what the tower's
    /// load in [`Resident::open`] is held to.
    pub card: CardBytes,
    /// The image-token limits the tower's activations are sized for ([`Projector::limits`]).
    pub limits: TokenLimits,
}

/// The seat's encoder: `file`, the encoder file the seat resolved ([`Flags::encoder`]: `--mmproj`'s,
/// or the `--hf` repo's projector). `None`
/// serves no image, and a limit given then is refused by name (no image input to bound); else the
/// file opened ([`Projector::open`], its type one of `takes`, the limits from `flags`), held
/// against the text model (`text`, `token` — [`Projector::check_text`]) and seated for a plan on
/// `place` ([`encoder_seat`]). Every check runs before a card is touched; whether a file the
/// `--hf` repo picked stays is the plan's question ([`kept`]).
pub fn open_seat(
    flags: &Flags,
    file: Option<&Path>,
    place: Place,
    takes: &[&str],
    text: &TextFacts,
    token: &dyn Fn(u32) -> Option<String>,
) -> Result<Option<Opened>, GateError> {
    let Some(path) = file else {
        // `wanted` refuses a limit with neither `--mmproj` nor an `--hf` repo; a repo that holds
        // no projector leaves nothing to bound either.
        flags.wanted()?;
        if flags.sets_limits() {
            return Err(
                "--image-min-tokens/--image-max-tokens with no encoder file (no --mmproj, and \
                 no projector the --hf repo holds): no image input to bound"
                    .into(),
            );
        }
        return Ok(None);
    };
    let projector = Projector::open(path, flags, takes)?;
    projector.check_text(text, token)?;
    let seat = encoder_seat(place, projector.card.total())?;
    Ok(Some(projector.seated(seat)))
}

/// What a plan gives the text model, for comparing a plan with the encoder's reserve and without:
/// whether the model loads whole on a card, the experts a placed plan's cards hold, and the
/// context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub whole: bool,
    pub card_experts: u64,
    pub ctx: u64,
}

impl Cost {
    /// A placed plan's: the experts its stage cards and tier cards hold, at `ctx` positions.
    #[must_use]
    pub fn of(plan: &model::placement::Plan<'_>, ctx: u64) -> Cost {
        Cost {
            whole: false,
            card_experts: plan.n_l.iter().sum::<u64>()
                + plan.tier_n_l.iter().flatten().sum::<u64>(),
            ctx,
        }
    }

    /// A load whole on one card, at `ctx` positions.
    #[must_use]
    pub fn whole(ctx: u64) -> Cost {
        Cost {
            whole: true,
            card_experts: 0,
            ctx,
        }
    }
}

/// `vision` as the seat's plan keeps it. A projector the `--hf` repo picked ([`Flags::encoder`])
/// that sits as a stage-card reserve stays only when it is free: `cost`, the seat's plan with
/// that encoder and without one, gives the text model the same experts and context. Otherwise it
/// is dropped by a `vision_seat` record (`seat=not-loaded`) giving its bytes and what it would
/// cost, and the seat serves text.
/// An `--mmproj` file, a limit given (both ask for images) and a seat off the plan stay as they
/// are. A plan that refuses the reserve is a cost; one that refuses without it is the seat's
/// own refusal.
pub fn kept(
    vision: Option<Opened>,
    flags: &Flags,
    cost: impl Fn(Option<&Opened>) -> Result<Cost, GateError>,
) -> Result<Option<Opened>, GateError> {
    let Some(v) = vision else {
        return Ok(None);
    };
    if v.seat.reserve.is_none() || flags.mmproj()?.is_some() || flags.sets_limits() {
        return Ok(Some(v));
    }
    let without = cost(None)?;
    let Some(costs) = cost_of(without, cost(Some(&v)).map_err(|e| e.to_string())) else {
        return Ok(Some(v));
    };
    let line = not_loaded(&v, &costs).line();
    // A seat plans a placement more than once (its offer, then the load): one record per verdict.
    static SAID: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut said = SAID
        .lock()
        .map_err(|_| "the vision_seat record's lock is poisoned")?;
    if !said.contains(&line) {
        eprintln!("{line}");
        eprintln!(
            "vision: images are off: the --hf repo's projector would cost the plan what the record above \
             says; --mmproj {} loads it anyway",
            v.path.display()
        );
        said.push(line);
    }
    Ok(None)
}

/// What the encoder's reserve costs the text model.
#[derive(Debug, PartialEq, Eq)]
enum Costs {
    /// The card experts and the context positions the plan gives up.
    Plan { experts: u64, ctx: u64 },
    /// The load that fits whole on one card is placed instead.
    Whole,
    /// The plan refuses the reserve, for this reason.
    Refused(String),
}

/// What the encoder's reserve costs the text model, `with` it against `without` it; `None` when
/// it costs nothing.
fn cost_of(without: Cost, with: Result<Cost, String>) -> Option<Costs> {
    match with {
        Ok(with) if with == without => None,
        Ok(with) if with.whole != without.whole => Some(Costs::Whole),
        Ok(with) => Some(Costs::Plan {
            experts: without.card_experts.saturating_sub(with.card_experts),
            ctx: without.ctx.saturating_sub(with.ctx),
        }),
        Err(e) => Some(Costs::Refused(e)),
    }
}

/// Where the image encoder sits under `--mmproj`: on a card this process holds that the
/// placement's plan leaves off it (the 3090 under `--place a` on this workstation, at no cost to
/// the plan) or, when every held card is on the plan, on the stage card as a named reserve the
/// plan's fit counts ([`model::placement::Card::reserves`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderSeat {
    /// The card's name as the placement's plan gives it.
    pub name: String,
    /// The card's CUDA ordinal in this process.
    pub ordinal: u32,
    /// The bytes the plan reserves on the stage card — the tower's card figure
    /// ([`CardBytes::total`]), the one its load holds itself to; `None` off the plan.
    pub reserve: Option<u64>,
}

impl EncoderSeat {
    /// The `seat` word of the `vision` record.
    #[must_use]
    pub fn word(&self) -> &'static str {
        if self.reserve.is_some() {
            "stage-reserve"
        } else {
            "off-plan"
        }
    }
}

/// The tower's seat under `place` for a tower of `bytes` on its card: the census resolved against
/// the placement, then [`seat_of`].
pub fn encoder_seat(place: Place, bytes: u64) -> Result<EncoderSeat, GateError> {
    // The aliases' specs name this workstation's cards with no device; the census resolution is
    // what makes the plan's cards mean the cards this process holds.
    let specs = place.on_host()?.card_specs()?;
    let census = bloomery_gpu::census()?;
    seat_of(&census, &specs, place.name(), bytes)
}

/// The seat for a tower of `bytes` among `census`, the cards this process sees, given `plan`, the
/// cards the placement `place` uses, the stage card first: the off-plan card of the fewest bytes —
/// several tie to the lower ordinal — else the stage card with `bytes` as its reserve.
pub fn seat_of(
    census: &[DeviceInfo],
    plan: &[CardSpec],
    place: &str,
    bytes: u64,
) -> Result<EncoderSeat, GateError> {
    let on_plan = |uuid: &[u8; 16]| {
        plan.iter()
            .filter_map(|c| c.device)
            .any(|d| &d.uuid == uuid)
    };
    if let Some(d) = census
        .iter()
        .filter(|d| !on_plan(&d.uuid))
        .min_by_key(|d| (d.total_bytes, d.ordinal))
    {
        return Ok(EncoderSeat {
            name: d.name.clone(),
            ordinal: d.ordinal,
            reserve: None,
        });
    }
    let spec = plan.first().ok_or("--place names no stage card")?;
    let device = spec
        .device
        .ok_or_else(|| format!("--place {place}: the stage card was not resolved to a device"))?;
    Ok(EncoderSeat {
        name: spec.name.to_string(),
        ordinal: device.ordinal,
        reserve: Some(bytes),
    })
}

/// Runs `work` — tower calls, which leave the tower's card bound to the engine thread — then binds
/// `model` (the model's own context) back, whether or not `work` succeeded, so every body call
/// after it runs with the model's card bound.
pub fn on_encoder_card<T>(
    model: &CudaContext,
    work: impl FnOnce() -> Result<T, GateError>,
) -> Result<T, GateError> {
    let out = work();
    // An allocation lands in the thread's current context (cuda-core's DeviceBuffer), not in its
    // stream's: the body's would land on the tower's card.
    model
        .bind_to_thread()
        .map_err(|e| format!("binding the model's card back after the tower: {e}"))?;
    out
}

/// The tower of `at`'s file loaded on its seat's card, with the stream its readbacks run on
/// ([`span_rows`]): the card bound for the load and the model's context (`model`) bound back
/// ([`on_encoder_card`]), refused by name when the load's weight and activation bytes are not the
/// card figure the plan reserved.
#[cfg(feature = "vision")]
fn open_tower(
    model: &CudaContext,
    at: &Opened,
) -> Result<
    (
        Arc<cuda_core::CudaStream>,
        Box<dyn bloomery_gpu_vision::Tower>,
    ),
    GateError,
> {
    let on = |e: &dyn Display| format!("--mmproj {}: {e}", at.path.display());
    let file = Gguf::open(&at.path).map_err(|e| on(&e))?;
    let ordinal = usize::try_from(at.seat.ordinal)
        .map_err(|_| on(&format!("the card's ordinal {}", at.seat.ordinal)))?;
    on_encoder_card(model, || {
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(ordinal)
            .map_err(|e| on(&format!("the card {}: {e}", at.seat.name)))?;
        let tower =
            bloomery_gpu_vision::open(&ctx, &stream, &file, at.limits).map_err(|e| on(&e))?;
        let loaded = tower.weight_bytes() + tower.scratch_bytes();
        if loaded != at.card.total() {
            return Err(on(&format!(
                "the tower holds {} B of weights and {} B of activations, {loaded} B; the plan \
                 reserved {} B for it",
                tower.weight_bytes(),
                tower.scratch_bytes(),
                at.card.total()
            ))
            .into());
        }
        Ok((stream, tower))
    })
}

/// One image of a media call: where its span starts in the call's ids, the patches the tower
/// encodes, and the positions its span takes (what the model's prepare said).
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub at: usize,
    pub patches: &'a Patches,
    pub span_len: usize,
}

impl<'a> Image<'a> {
    /// The image whose span starts at `at` in the call's ids, as its model's prepare made it.
    #[must_use]
    pub fn of(at: usize, prepared: &'a Prepared) -> Image<'a> {
        Image {
            at,
            patches: &prepared.patches,
            span_len: prepared.span_len,
        }
    }
}

/// A media call's spans in one pair of buffers: the rows (`width` bf16 values a position), the
/// kind of each position, and each span's ranges of them.
#[derive(Debug, PartialEq, Eq)]
pub struct SpanRows<K> {
    pub rows: Vec<u16>,
    pub kinds: Vec<K>,
    pub ats: Vec<SpanAt>,
}

impl<K> SpanRows<K> {
    /// The call's spans as [`MediaSpan`]s, each borrowing its own values of [`SpanRows::rows`]: a
    /// body whose media call takes nothing beside a span's positions and rows.
    #[must_use]
    pub fn media_spans(&self) -> Vec<MediaSpan<'_>> {
        self.ats
            .iter()
            .map(|a| MediaSpan {
                at: a.positions.clone(),
                rows: &self.rows[a.rows.clone()],
            })
            .collect()
    }

    /// [`SpanRows::media_spans`] with each span's token grid, the `(nx, ny)` its own image's plan
    /// gave it ([`SpanAt::grid`]): a body that shifts its rotation table by each image's grid. A
    /// grid holds `nx · ny` positions only for a span that is its cells ([`merged_cells`]); a span
    /// with rows around them (V4.1's delimiters) fails `ImageSpan::check`.
    #[must_use]
    pub fn image_spans(&self) -> Vec<ImageSpan<'_>> {
        self.media_spans()
            .into_iter()
            .zip(&self.ats)
            .map(|(span, a)| ImageSpan { span, grid: a.grid })
            .collect()
    }
}

/// One span's ranges: its positions in the call's ids, its values of [`SpanRows::rows`] and its
/// entries of [`SpanRows::kinds`], and its image's token grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanAt {
    pub positions: Range<usize>,
    pub rows: Range<usize>,
    pub kinds: Range<usize>,
    /// `(nx, ny)` of the image's plan: tokens along a line and lines
    /// (`vision::GridPlan::span_grid` holds the order), recorded where the span is made, from the
    /// image the span is of.
    pub grid: (usize, usize),
}

/// [`SpanRows`] of `images`, encoded on `tower` in order and read back on `stream`, the stream the
/// tower was opened with ([`span_rows_of`] holds the rules).
#[cfg(feature = "vision")]
fn span_rows<'a, T: bloomery_gpu_vision::Tower + ?Sized, K>(
    tower: &mut T,
    stream: &Arc<cuda_core::CudaStream>,
    images: impl IntoIterator<Item = Image<'a>>,
    width: usize,
    splice: impl FnMut(&[u16], &Patches) -> Result<(Vec<u16>, Vec<K>), GateError>,
) -> Result<SpanRows<K>, GateError> {
    span_rows_of(
        images,
        width,
        |patches| {
            let encoded = tower.encode(patches).map_err(|e| e.to_string())?;
            encoded
                .rows
                .buf()
                .to_host_vec(stream)
                .map_err(|e| e.to_string())
        },
        splice,
    )
}

/// A serve seat's resident image encoder under `--mmproj`: the tower on the card the plan seated
/// it ([`Resident::open`]), the stream its readbacks run on, the width of a row (the text model's
/// embedding, which the tower writes) and the model's image input the server renders and expands
/// image parts by. What a seat adds is its model's splice of one image's rows.
#[cfg(feature = "vision")]
pub struct Resident {
    tower: Box<dyn bloomery_gpu_vision::Tower>,
    stream: Arc<cuda_core::CudaStream>,
    n_embd: usize,
    media: SharedMedia,
}

#[cfg(feature = "vision")]
impl Resident {
    /// The tower of `at`'s file loaded on its seat's card with the model's context (`model`) bound
    /// back, refused by name when the load's bytes are not the card figure the plan reserved; the
    /// row width is the tower's.
    pub fn open(model: &CudaContext, at: &Opened) -> Result<Resident, GateError> {
        let (stream, tower) = open_tower(model, at)?;
        Ok(Resident {
            n_embd: tower.out_dim(),
            tower,
            stream,
            media: at.media.clone(),
        })
    }

    /// The `vision` record of the load that began at `started`: the card, which seat it is, the
    /// tower's bytes, where its activations live and the most image tokens one image takes.
    pub fn announce(&self, at: &Opened, home: ScratchHome, started: std::time::Instant) {
        record(
            &at.seat,
            (self.tower.weight_bytes(), self.tower.scratch_bytes()),
            home,
            at.cap,
            started.elapsed().as_secs_f64(),
        )
        .eprint();
    }

    /// The model's image input.
    #[must_use]
    pub fn media(&self) -> SharedMedia {
        self.media.clone()
    }

    /// The model's image input, borrowed.
    #[must_use]
    pub fn media_ref(&self) -> &dyn MediaModel {
        &*self.media
    }

    /// The width of a row.
    #[must_use]
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// [`span_rows`] of `images` on this tower and stream, at the row width.
    fn span_rows<'a, K>(
        &mut self,
        images: impl IntoIterator<Item = Image<'a>>,
        splice: impl FnMut(&[u16], &Patches) -> Result<(Vec<u16>, Vec<K>), GateError>,
    ) -> Result<SpanRows<K>, GateError> {
        span_rows(&mut *self.tower, &self.stream, images, self.n_embd, splice)
    }

    /// Step (e) of a seat's media call: `images` encoded and read back as the rows the prompt call
    /// splices — [`Resident::span_rows`] with the seat's own `splice` (what its spans add around
    /// the cells' rows, and the kind of each position), the tower's card bound for the calls and
    /// `model`, the context the seat's body runs on, bound back after them
    /// ([`on_encoder_card`]). The prompt call over them is [`Encoded::prompt`].
    pub fn encode_with<'a, K>(
        &mut self,
        model: &CudaContext,
        images: impl IntoIterator<Item = Image<'a>>,
        splice: impl FnMut(&[u16], &Patches) -> Result<(Vec<u16>, Vec<K>), GateError>,
    ) -> Result<Encoded<K>, GateError> {
        let rows = on_encoder_card(model, || self.span_rows(images, splice))?;
        Ok(Encoded { rows })
    }

    /// [`Resident::encode_with`] of a merger tower's span: [`merged_cells`] (one row a cell of
    /// each grid, the ids carry the delimiters).
    pub fn encode<'a>(
        &mut self,
        model: &CudaContext,
        images: impl IntoIterator<Item = Image<'a>>,
    ) -> Result<Encoded, GateError> {
        self.encode_with(model, images, merged_cells)
    }
}

/// The rows of one media call, read back by [`Resident::encode_with`] ([`Resident::encode`] for a
/// merger tower: `K = ()`) and not yet spliced into a prompt call; `K` is the kind a seat's splice
/// gives each position. [`Encoded::rows`] lends them read-only; [`Encoded::prompt`] makes the one
/// call that splices them and consumes them, so no later read can see what it did to its own copy.
#[derive(Debug)]
pub struct Encoded<K = ()> {
    rows: SpanRows<K>,
}

impl<K> Encoded<K> {
    /// The rows of a media call read back elsewhere than [`Resident::encode_with`], wrapped to
    /// take [`Encoded::prompt`]: a gate that holds a seat's prompt step to its body without a
    /// tower (`gate_ds41_media`: the vision set's aligner rows, spliced by [`span_rows_of`])
    /// builds its [`SpanRows`] and hands them here. It only wraps: the prompt call over the rows
    /// is [`Encoded::prompt`], the one call, and a seat's rows come from its tower.
    #[must_use]
    pub fn from_rows(rows: SpanRows<K>) -> Encoded<K> {
        Encoded { rows }
    }

    /// The call's rows and each span's ranges of them.
    #[must_use]
    pub fn rows(&self) -> &SpanRows<K> {
        &self.rows
    }

    /// Step (p) of a media call, the only place a seat's rows pass on their way to the prompt
    /// call: `build` makes the body's spans of the rows (the seat's own: [`SpanRows::media_spans`]
    /// for a body that takes nothing beside positions and rows, [`SpanRows::image_spans`] for one
    /// that takes each image's grid, the rows with their kinds for one that checks a splice
    /// shape), then one [`app::Session::prompt_media`] of `ids` over them, the greedy token after
    /// the last id. It is the call's one residency pass: a body's own `prefill_media` is never
    /// called beside it.
    pub fn prompt<B, F>(
        self,
        s: &mut app::Session<B>,
        ids: &[u32],
        build: F,
    ) -> Result<u32, GateError>
    where
        B: MediaBody,
        F: for<'r> FnOnce(&'r SpanRows<K>) -> Vec<B::Span<'r>>,
    {
        self.prompt_by::<B, _, _>(build, |spans| {
            Ok(s.prompt_media(ids, spans, runtime::Want::Argmax)?.argmax())
        })
    }

    /// [`Encoded::prompt`] with the call the seat's own: `build` makes the body's spans of the
    /// rows, and `call` is the one prompt call over them, which returns the greedy token after
    /// the last id (the session's [`app::Session::prompt_media`], or a draft's media prompt call
    /// that opens the residency pass itself). The rows are consumed, as in [`Encoded::prompt`]:
    /// `call` is the only reader of them. The body `B` is named by the caller (`prompt_by::<B, _,
    /// _>`): a span type does not name its body.
    pub fn prompt_by<B, F, C>(self, build: F, call: C) -> Result<u32, GateError>
    where
        B: MediaBody,
        F: for<'r> FnOnce(&'r SpanRows<K>) -> Vec<B::Span<'r>>,
        C: for<'r> FnOnce(&[B::Span<'r>]) -> Result<u32, GateError>,
    {
        let rows = self.rows;
        let spans = build(&rows);
        call(&spans)
    }
}

/// The host half of a resident's encode ([`Resident::encode_with`]): each image encoded and read
/// back as bf16 by `encode`, the first `n_llm_h · n_llm_w · width` values kept (the tower writes
/// one row of `width` a cell of the plan's grid; fewer is refused by name) and handed to `splice`
/// with the patches, which returns the span's rows and the kind of each of its positions; their
/// count is the image's `span_len`, refused by name when it is not. The span records its image's
/// own token grid ([`SpanAt::grid`]). What a span adds around the cells' rows (V4.1's delimiters,
/// none for a merger's tokens) is `splice`; the rest is the same for every tower.
pub fn span_rows_of<'a, K, E: Display>(
    images: impl IntoIterator<Item = Image<'a>>,
    width: usize,
    mut encode: impl FnMut(&Patches) -> Result<Vec<u16>, E>,
    mut splice: impl FnMut(&[u16], &Patches) -> Result<(Vec<u16>, Vec<K>), GateError>,
) -> Result<SpanRows<K>, GateError> {
    let mut out = SpanRows {
        rows: Vec::new(),
        kinds: Vec::new(),
        ats: Vec::new(),
    };
    for Image {
        at,
        patches,
        span_len,
    } in images
    {
        let (h, w) = (patches.plan.n_llm_h, patches.plan.n_llm_w);
        let all = encode(patches).map_err(|e| format!("the image at {at}: {e}"))?;
        let want = h * w * width;
        let image = all.get(..want).ok_or_else(|| {
            format!(
                "the image at {at}: {} encoded values for a {h}x{w} grid, want {want}",
                all.len()
            )
        })?;
        let (rows, kinds) = splice(image, patches)?;
        if kinds.len() != span_len {
            return Err(format!(
                "the image at {at}: a {h}x{w} grid splices {} positions, its prepare said {span_len}",
                kinds.len()
            )
            .into());
        }
        out.ats.push(SpanAt {
            positions: at..at + kinds.len(),
            rows: out.rows.len()..out.rows.len() + rows.len(),
            kinds: out.kinds.len()..out.kinds.len() + kinds.len(),
            grid: patches.plan.span_grid(),
        });
        out.rows.extend(rows);
        out.kinds.extend(kinds);
    }
    Ok(out)
}

/// The splice of a merger tower's span ([`Resident::encode`]): the cells' rows as the tower wrote
/// them, one position a cell, nothing around them — the placeholder's delimiters are ordinary ids.
pub fn merged_cells(rows: &[u16], patches: &Patches) -> Result<(Vec<u16>, Vec<()>), GateError> {
    let cells = patches.plan.n_llm_h * patches.plan.n_llm_w;
    Ok((rows.to_vec(), vec![(); cells]))
}

/// Where a tower's activations live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScratchHome {
    /// The tower's own buffers.
    Own,
    /// The prompt's arena, lent while no prompt call runs.
    Arena,
}

impl ScratchHome {
    fn word(self) -> &'static str {
        match self {
            ScratchHome::Own => "own",
            ScratchHome::Arena => "arena",
        }
    }
}

/// The `vision` record after a tower's load: the card, which seat it is, the tower's weight and
/// activation bytes, where the activations live, the most image tokens one image takes, and the
/// load's seconds.
pub fn record(
    seat: &EncoderSeat,
    (weights, scratch): (u64, u64),
    home: ScratchHome,
    cap: usize,
    load_s: f64,
) -> Record {
    Record::new(&record::VISION)
        .w("card", record::name_token(&seat.name))
        .w("seat", seat.word())
        .u("weights", weights)
        .u("scratch", scratch)
        .w("scratch_home", home.word())
        .u("cap", cap)
        .f("load_s", load_s)
}

/// The `seat` word of a projector [`kept`] drops.
const NOT_LOADED: &str = "not-loaded";

/// The `vision_seat` record of the encoder `at` seated for the plan: the card, which seat it is
/// and the reserve it takes on the stage card (0 off the plan), the tower's card figure and the
/// most image tokens one image takes. Printed with the plan, before any load.
pub fn seat_record(at: &Opened) -> Record {
    seat_line(at, at.seat.word(), None)
}

/// The `vision_seat` record of a projector [`kept`] drops: the seat it would have taken, the
/// reserve it would have reserved and what that costs the plan.
fn not_loaded(at: &Opened, costs: &Costs) -> Record {
    seat_line(at, NOT_LOADED, Some(costs))
}

fn seat_line(at: &Opened, seat: &str, dropped: Option<&Costs>) -> Record {
    let (experts, ctx) = match dropped {
        Some(&Costs::Plan { experts, ctx }) => (experts, ctx),
        _ => (0, 0),
    };
    let line = Record::new(&record::VISION_SEAT)
        .w("card", record::name_token(&at.seat.name))
        .w("seat", seat)
        .u("reserve", at.seat.reserve.unwrap_or(0))
        .u("weights", at.card.weights)
        .u("scratch", at.card.scratch)
        .u("cap", at.cap)
        .u("cost_experts", experts)
        .u("cost_ctx", ctx);
    match dropped {
        None => line,
        Some(Costs::Plan { .. }) => line.w("why", "plan-cost"),
        Some(Costs::Whole) => line.w("why", "whole-load"),
        Some(Costs::Refused(e)) => line.w("why", "plan-refused").w("refusal", e),
    }
}

#[cfg(test)]
mod tests {
    use gguf::Value;
    use model::placement::devices::DeviceId;
    use model::placement::workstation::{A6000, RTX_3090};

    use super::*;

    fn flags(args: &[&str]) -> Result<Flags, String> {
        let mut f = Flags::default();
        let mut it = args.iter().copied();
        while let Some(flag) = it.next() {
            if !f.take(flag, &mut it).map_err(|e| e.to_string())? {
                return Err(format!("unread {flag}"));
            }
        }
        Ok(f)
    }

    /// The limits a flag set gives over mtmd's defaults for the Qwen3-VL projector (8 and 4096).
    fn over_qwens(f: &Flags) -> (usize, usize) {
        let l = f.limits_over(TokenLimits::DEFAULT).unwrap();
        (l.min(), l.max())
    }

    /// The four flags in llama-server's spelling; `--no-mmproj` has no value, a limit not given is
    /// the default the caller passes (mtmd's 8 and 4096 here), and every refusal names its flag.
    #[test]
    fn the_flags_read_llama_servers_spelling() {
        let f = flags(&["--mmproj", "/m/p.gguf", "--image-min-tokens", "16"]).unwrap();
        assert_eq!(f.mmproj().unwrap(), Some(Path::new("/m/p.gguf")));
        assert_eq!(over_qwens(&f), (16, 4096));
        assert!(f.sets_limits());
        let f = flags(&["--no-mmproj", "--image-max-tokens", "1024"]).unwrap();
        assert_eq!(f.mmproj().unwrap(), None);
        assert_eq!(over_qwens(&f), (8, 1024));
        let d = flags(&[]).unwrap();
        assert_eq!((d.mmproj().unwrap(), d.sets_limits()), (None, false));
        assert_eq!(over_qwens(&d), (8, 4096));
        // Another seat flag is the caller's, and is not consumed.
        let mut other = Flags::default();
        assert!(!other.take("--ctx", &mut ["8"].into_iter()).unwrap());
        assert_eq!(other, Flags::default());
    }

    #[test]
    fn the_flags_refuse_by_name() {
        let said = |a: &[&str]| {
            flags(a)
                .and_then(|f| {
                    f.mmproj()
                        .map(|_| ())
                        .and(f.limits_over(TokenLimits::DEFAULT).map(|_| ()))
                        .map_err(|e| e.to_string())
                })
                .unwrap_err()
        };
        assert_eq!(
            said(&["--mmproj"]),
            format!("--mmproj needs a value: {USAGE}")
        );
        assert_eq!(
            said(&["--mmproj", "/m/p.gguf", "--no-mmproj"]),
            "--mmproj /m/p.gguf with --no-mmproj: the two flags contradict; give one"
        );
        assert!(said(&["--image-min-tokens", "x"]).starts_with("--image-min-tokens \"x\": "));
        assert!(said(&["--image-max-tokens", "-1"]).starts_with("--image-max-tokens \"-1\": "));
        assert!(
            said(&["--image-min-tokens", "9", "--image-max-tokens", "8"])
                .starts_with("--image-min-tokens/--image-max-tokens: image size limits: min 9")
        );
        assert!(said(&["--image-min-tokens", "0"]).contains("need 1 <= min <= max"));
    }

    fn device(ordinal: u32, name: &str, gib: u64, uuid: u8) -> DeviceInfo {
        DeviceInfo {
            ordinal,
            name: name.to_owned(),
            total_bytes: gib << 30,
            free_bytes: gib << 30,
            uuid: [uuid; 16],
            pci_bus: format!("0000:0{ordinal}:00.0"),
            held_by: None,
            integrated: false,
        }
    }

    fn on(spec: CardSpec, ordinal: u32, uuid: u8) -> CardSpec {
        CardSpec {
            device: Some(DeviceId {
                ordinal,
                uuid: [uuid; 16],
            }),
            ..spec
        }
    }

    const BYTES: u64 = 1_476_000_000;

    /// Two cards with the plan on one: the encoder takes the card the plan leaves, no reserve.
    #[test]
    fn two_cards_with_the_plan_on_one_seat_the_encoder_off_plan() {
        let census = [
            device(0, "NVIDIA RTX A6000", 48, 1),
            device(1, "NVIDIA GeForce RTX 3090", 24, 2),
        ];
        let plan = [on(A6000, 0, 1)];
        let seat = seat_of(&census, &plan, "a", BYTES).unwrap();
        assert_eq!(
            seat,
            EncoderSeat {
                name: "NVIDIA GeForce RTX 3090".to_owned(),
                ordinal: 1,
                reserve: None
            }
        );
        assert_eq!(seat.word(), "off-plan");
        // Of several off-plan cards the smallest, a tie to the lower ordinal.
        let three = [
            device(0, "NVIDIA RTX A6000", 48, 1),
            device(1, "NVIDIA GeForce RTX 3090", 24, 2),
            device(2, "NVIDIA GeForce RTX 4090", 24, 3),
            device(3, "NVIDIA RTX 6000 Ada", 48, 4),
        ];
        let seat = seat_of(&three, &plan, "a", BYTES).unwrap();
        assert_eq!((seat.ordinal, seat.reserve), (1, None));
    }

    /// One card, or every held card on the plan: a reserve of the encoder's bytes on the stage card.
    #[test]
    fn one_card_seats_the_encoder_as_a_stage_reserve() {
        let one = [device(0, "NVIDIA RTX A6000", 48, 1)];
        let seat = seat_of(&one, &[on(A6000, 0, 1)], "a", BYTES).unwrap();
        assert_eq!(
            seat,
            EncoderSeat {
                name: A6000.name.to_owned(),
                ordinal: 0,
                reserve: Some(BYTES)
            }
        );
        assert_eq!(seat.word(), "stage-reserve");
        let both = [
            device(0, "NVIDIA RTX A6000", 48, 1),
            device(1, "NVIDIA GeForce RTX 3090", 24, 2),
        ];
        let bp = [on(A6000, 0, 1), on(RTX_3090, 1, 2)];
        let seat = seat_of(&both, &bp, "bp", BYTES).unwrap();
        assert_eq!((seat.ordinal, seat.reserve), (0, Some(BYTES)));
        let said = |e: GateError| e.to_string();
        // No visible card at all: the plan has no stage card, or its stage card was never resolved.
        assert_eq!(
            said(seat_of(&[], &[], "x", BYTES).unwrap_err()),
            "--place names no stage card"
        );
        assert_eq!(
            said(seat_of(&[], &[A6000], "a", BYTES).unwrap_err()),
            "--place a: the stage card was not resolved to a device"
        );
    }

    /// The encoder `seat` opened, with a card figure that sums to [`BYTES`].
    fn opened(seat: EncoderSeat) -> Opened {
        Opened {
            path: PathBuf::from("/m/mmproj.gguf"),
            seat,
            media: Arc::new(Stub(1)),
            cap: 4096,
            card: CardBytes {
                weights: 1_000_000_000,
                scratch: 476_000_000,
            },
            limits: TokenLimits::DEFAULT,
        }
    }

    /// The plan-time record: an off-plan seat names its card with reserve 0, a stage seat its
    /// reserve in bytes (the tower's card figure), and both the figure the seat was sized from.
    #[test]
    fn the_seat_record_names_the_seat_and_its_reserve() {
        let both = [
            device(0, "NVIDIA RTX A6000", 48, 1),
            device(1, "NVIDIA GeForce RTX 3090", 24, 2),
        ];
        let off = opened(seat_of(&both, &[on(A6000, 0, 1)], "a", BYTES).unwrap());
        assert_eq!(
            seat_record(&off).line(),
            "vision_seat card=NVIDIA_GeForce_RTX_3090 seat=off-plan reserve=0 weights=1000000000 \
             scratch=476000000 cap=4096 cost_experts=0 cost_ctx=0"
        );
        let bp = [on(A6000, 0, 1), on(RTX_3090, 1, 2)];
        let stage = opened(seat_of(&both, &bp, "bp", BYTES).unwrap());
        assert_eq!(
            seat_record(&stage).line(),
            format!(
                "vision_seat card={} seat=stage-reserve reserve={BYTES} weights=1000000000 \
                 scratch=476000000 cap=4096 cost_experts=0 cost_ctx=0",
                record::name_token(A6000.name)
            )
        );
        assert_eq!(BYTES, stage.card.total());
    }

    /// A media model of one image token.
    struct Stub(u32);

    impl MediaModel for Stub {
        fn image_placeholder(&self) -> &str {
            "<img>"
        }
        fn image_token(&self) -> u32 {
            self.0
        }
        fn prepare(&self, _: &vision::Rgb8) -> Result<vision::Prepared, VisionError> {
            Err(VisionError::Limits {
                detail: "the stub prepares nothing".into(),
            })
        }
    }

    fn projector(rule: usize, out_dim: usize, token: u32) -> Projector {
        Projector {
            path: PathBuf::from("/m/mmproj.gguf"),
            rule: &rules()[rule],
            card: CardBytes {
                weights: 1,
                scratch: 2,
            },
            media: Arc::new(Stub(token)),
            out_dim,
            limits: TokenLimits::DEFAULT,
            cap: 4096,
        }
    }

    /// The vocabulary the Qwen3-VL family's text models carry.
    fn qwen_vocab(id: u32) -> Option<String> {
        match id {
            248_053 => Some("<|vision_start|>".to_owned()),
            248_054 => Some("<|vision_end|>".to_owned()),
            248_056 => Some("<|image_pad|>".to_owned()),
            _ => None,
        }
    }

    const QWEN_ROPE: Option<[u32; 4]> = Some([11, 11, 10, 0]);

    fn facts(n_embd: usize, imrope: Option<[u32; 4]>, ple: PleImage) -> TextFacts {
        TextFacts {
            n_embd,
            imrope,
            ple,
        }
    }

    /// `open_seat` before any file or card: no `--mmproj` is no encoder, a limit without it is
    /// refused by name (also beside `--no-mmproj`), a file that is not there is refused naming it.
    #[test]
    fn open_seat_refuses_by_name_before_a_card() {
        let text = facts(2048, QWEN_ROPE, PleImage::Absent);
        let open = |a: &[&str]| -> Result<bool, GateError> {
            let f = flags(a).unwrap();
            open_seat(&f, f.mmproj()?, Place::A, &[], &text, &qwen_vocab).map(|o| o.is_some())
        };
        assert!(!open(&[]).unwrap());
        assert!(!open(&["--no-mmproj"]).unwrap());
        let limits =
            "--image-min-tokens/--image-max-tokens without --mmproj: no image input to bound";
        assert_eq!(
            open(&["--image-min-tokens", "4"]).unwrap_err().to_string(),
            limits
        );
        assert_eq!(
            open(&["--no-mmproj", "--image-max-tokens", "9"])
                .unwrap_err()
                .to_string(),
            limits
        );
        let gone = open(&["--mmproj", "/nowhere/p.gguf"])
            .unwrap_err()
            .to_string();
        assert!(gone.starts_with("--mmproj /nowhere/p.gguf: "), "{gone}");
    }

    /// `encoder` returns `--mmproj`'s file whatever the mode, and with no `--hf` model file (a
    /// test process names none) no file at all; the contradiction stays refused by name.
    #[test]
    fn the_encoder_is_the_flags_file_without_an_hf_repo() {
        let f = flags(&["--mmproj", "/m/p.gguf"]).unwrap();
        assert_eq!(f.encoder(true).unwrap(), Some(PathBuf::from("/m/p.gguf")));
        assert_eq!(f.encoder(false).unwrap(), Some(PathBuf::from("/m/p.gguf")));
        assert_eq!(Flags::default().encoder(true).unwrap(), None);
        assert_eq!(
            flags(&["--no-mmproj"]).unwrap().encoder(true).unwrap(),
            None
        );
        let both = flags(&["--mmproj", "/m/p.gguf", "--no-mmproj"]).unwrap();
        assert!(both.encoder(true).is_err());
    }

    /// A reserve is free only when the plan gives the same load, experts and context with it; each
    /// cost is named: experts and context lost, the whole load lost, a plan that refuses it. The
    /// dropped projector's record carries the seat it would have taken and the cost in its fields.
    #[test]
    fn a_reserve_is_free_only_when_the_plan_is_unchanged() {
        let placed = |card_experts, ctx| Cost {
            whole: false,
            card_experts,
            ctx,
        };
        assert_eq!(cost_of(placed(940, 4096), Ok(placed(940, 4096))), None);
        assert_eq!(cost_of(Cost::whole(4096), Ok(Cost::whole(4096))), None);
        let one = [device(0, "NVIDIA RTX A6000", 48, 1)];
        let at = opened(seat_of(&one, &[on(A6000, 0, 1)], "a", BYTES).unwrap());
        let said = |costs: Option<Costs>| not_loaded(&at, &costs.unwrap()).line();
        let head = format!(
            "vision_seat card={} seat=not-loaded reserve={BYTES} weights=1000000000 \
             scratch=476000000 cap=4096",
            record::name_token(A6000.name)
        );
        assert_eq!(
            said(cost_of(placed(940, 4096), Ok(placed(727, 4096)))),
            format!("{head} cost_experts=213 cost_ctx=0 why=plan-cost")
        );
        assert_eq!(
            said(cost_of(placed(940, 8192), Ok(placed(940, 4096)))),
            format!("{head} cost_experts=0 cost_ctx=4096 why=plan-cost")
        );
        assert_eq!(
            said(cost_of(Cost::whole(4096), Ok(placed(900, 4096)))),
            format!("{head} cost_experts=0 cost_ctx=0 why=whole-load")
        );
        assert_eq!(
            said(cost_of(placed(940, 4096), Err("no room".to_owned()))),
            format!("{head} cost_experts=0 cost_ctx=0 why=plan-refused refusal=no room")
        );
    }

    /// The projector-to-text checks of the qwen3vl type, each refused by name; the Qwen3.8 text
    /// (PLE with the media token), the Qwen3.6 and Clef texts (no PLE) pass.
    #[test]
    fn a_qwen3vl_projector_is_held_against_its_text_model() {
        let said = |p: &Projector, t: TextFacts, v: &dyn Fn(u32) -> Option<String>| {
            p.check_text(&t, v).unwrap_err().to_string()
        };
        let p = projector(0, 2560, 248_056);
        p.check_text(&facts(2560, QWEN_ROPE, PleImage::Id(248_056)), &qwen_vocab)
            .unwrap();
        p.check_text(&facts(2560, QWEN_ROPE, PleImage::Absent), &qwen_vocab)
            .unwrap();
        assert_eq!(
            said(&p, facts(2048, QWEN_ROPE, PleImage::Absent), &qwen_vocab),
            "--mmproj /m/mmproj.gguf: the encoder's aligner rows are 2560 wide, the model's \
             embedding is 2048"
        );
        assert_eq!(
            said(&p, facts(2560, QWEN_ROPE, PleImage::Absent), &|id| {
                (id != 248_054).then(|| qwen_vocab(id)).flatten()
            }),
            "--mmproj /m/mmproj.gguf: the qwen3vl_merger projector's image id 248054 is \
             \"<|vision_end|>\", and the model's vocabulary holds no such token"
        );
        assert_eq!(
            said(&p, facts(2560, QWEN_ROPE, PleImage::Absent), &|id| {
                if id == 248_056 {
                    Some("<|pad|>".to_owned())
                } else {
                    qwen_vocab(id)
                }
            }),
            "--mmproj /m/mmproj.gguf: the qwen3vl_merger projector's image id 248056 is \
             \"<|image_pad|>\", and the model's vocabulary holds \"<|pad|>\""
        );
        assert_eq!(
            said(&p, facts(2560, None, PleImage::Absent), &qwen_vocab),
            "--mmproj /m/mmproj.gguf: the qwen3vl_merger projector places image rows by IMROPE \
             sections [11, 11, 10, 0], and the text model's rope is not IMROPE"
        );
        assert_eq!(
            said(
                &p,
                facts(2560, Some([24, 20, 20, 0]), PleImage::Absent),
                &qwen_vocab
            ),
            "--mmproj /m/mmproj.gguf: the qwen3vl_merger projector places image rows by IMROPE \
             sections [11, 11, 10, 0], and the text model's rope is IMROPE sections [24, 20, 20, 0]"
        );
        assert_eq!(
            said(
                &p,
                facts(2560, QWEN_ROPE, PleImage::Id(248_044)),
                &qwen_vocab
            ),
            "--mmproj /m/mmproj.gguf: the text model's ple.image_token_id is 248044, the \
             projector's image token is 248056"
        );
        assert_eq!(
            said(&p, facts(2560, QWEN_ROPE, PleImage::NoKey), &qwen_vocab),
            "--mmproj /m/mmproj.gguf: the text model carries per-layer embeddings and no \
             ple.image_token_id, which must be the projector's image token"
        );
    }

    /// V4.1's type asks nothing of the rope, and its one id is the placeholder's string.
    #[test]
    fn a_deepseek41v_projector_is_held_by_its_one_id() {
        let p = projector(1, 4096, 129_264);
        let vocab = |id: u32| (id == 129_264).then(|| "<｜deepseek_image｜>".to_owned());
        p.check_text(&facts(4096, None, PleImage::Absent), &vocab)
            .unwrap();
        let e = p
            .check_text(&facts(4096, None, PleImage::Absent), &qwen_vocab)
            .unwrap_err();
        assert_eq!(
            e.to_string(),
            "--mmproj /m/mmproj.gguf: the deepseek41v projector's image id 129264 is \
             \"<｜deepseek_image｜>\", and the model's vocabulary holds no such token"
        );
    }

    /// The vocabulary of GLM-5.3-Flash's text model, for the ids its projector places.
    fn glm_vocab(id: u32) -> Option<String> {
        match id {
            154_830 => Some("<|begin_of_image|>".to_owned()),
            154_831 => Some("<|end_of_image|>".to_owned()),
            154_854 => Some("<|image|>".to_owned()),
            _ => None,
        }
    }

    /// GLM's type is the third row: its three ids, no rope sections asked of the text, and the
    /// limits steering its plan; the text is held to the ids, each refused by name, and to the row
    /// width.
    #[test]
    fn a_glm5v_projector_is_held_by_its_three_ids() {
        let row = &rules()[2];
        assert_eq!(row.projector, "glm5v");
        assert_eq!(
            row.ids,
            [
                (154_830, "<|begin_of_image|>"),
                (154_831, "<|end_of_image|>"),
                (154_854, "<|image|>")
            ]
        );
        assert_eq!((row.text_rope, row.takes_limits), (None, true));
        let p = projector(2, 4096, 154_854);
        p.check_text(&facts(4096, None, PleImage::Absent), &glm_vocab)
            .unwrap();
        let said = |t: TextFacts, v: &dyn Fn(u32) -> Option<String>| {
            p.check_text(&t, v).unwrap_err().to_string()
        };
        assert_eq!(
            said(facts(2048, None, PleImage::Absent), &glm_vocab),
            "--mmproj /m/mmproj.gguf: the encoder's aligner rows are 4096 wide, the model's \
             embedding is 2048"
        );
        assert_eq!(
            said(facts(4096, None, PleImage::Absent), &|id| {
                (id != 154_854).then(|| glm_vocab(id)).flatten()
            }),
            "--mmproj /m/mmproj.gguf: the glm5v projector's image id 154854 is \"<|image|>\", \
             and the model's vocabulary holds no such token"
        );
        assert_eq!(
            said(facts(4096, None, PleImage::Absent), &|id| {
                if id == 154_830 {
                    Some("<|begin_of_box|>".to_owned())
                } else {
                    glm_vocab(id)
                }
            }),
            "--mmproj /m/mmproj.gguf: the glm5v projector's image id 154830 is \
             \"<|begin_of_image|>\", and the model's vocabulary holds \"<|begin_of_box|>\""
        );
        assert_eq!(
            said(facts(4096, None, PleImage::Id(154_000)), &glm_vocab),
            "--mmproj /m/mmproj.gguf: the text model's ple.image_token_id is 154000, the \
             projector's image token is 154854"
        );
    }

    /// The keys of GLM-5.3-Flash's mmproj header that the host crate holds a file to
    /// (`glm5v::Hparams::read`), without the architecture and the projector type.
    fn glm5v_header() -> Vec<(String, Value)> {
        let clip =
            |bits: [u32; 3]| Value::Array(bits.map(|b| Value::F32(f32::from_bits(b))).to_vec());
        vec![
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_silu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(24)),
            ("clip.vision.embedding_length", Value::U32(1024)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(4096)),
            ("clip.vision.patch_size", Value::U32(14)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(4096)),
            ("clip.vision.image_size", Value::U32(448)),
            ("clip.vision.image_min_pixels", Value::U32(12_544)),
            ("clip.vision.image_max_pixels", Value::U32(6_272_000)),
            ("clip.vision.swiglu_clamp", Value::F32(10.0)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
            (
                "clip.vision.image_mean",
                clip([0x3ef6_813a, 0x3eea_685e, 0x3ed1_00ff]),
            ),
            (
                "clip.vision.image_std",
                clip([0x3e89_89d0, 0x3e85_c974, 0x3e8d_32a8]),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }

    /// A header-only GGUF with `clip.projector_type` `kind` and the keys `more`, deleted when
    /// dropped.
    struct Clip(PathBuf);

    impl Clip {
        fn new(kind: Option<&str>) -> Clip {
            Clip::with(kind, Vec::new())
        }

        fn with(kind: Option<&str>, more: Vec<(String, Value)>) -> Clip {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let mut kvs = vec![(
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String("clip".into()),
            )];
            if let Some(k) = kind {
                kvs.push(("clip.projector_type".to_owned(), Value::String(k.into())));
            }
            kvs.extend(more);
            let layout = gguf::write::Layout::new(&kvs, Vec::new()).expect("layout");
            let path = std::env::temp_dir().join(format!(
                "bloomery-vision-seat-{}-{}.gguf",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let file = std::fs::File::create(&path).expect("create");
            gguf::write::Writer::new(file, layout)
                .and_then(gguf::write::Writer::finish)
                .expect("write");
            Clip(path)
        }
    }

    impl Drop for Clip {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// The file's own refusals before the host crate reads it: a projector type the seat's text
    /// cannot run, one with no rule, an absent type, a limit on a projector that plans its own, a
    /// file the host crate refuses by key, and a file that is no GGUF.
    #[test]
    fn the_file_is_refused_by_name_before_it_is_read() {
        let open = |c: &Clip, f: &Flags, takes: &[&str]| {
            Projector::open(&c.0, f, takes)
                .err()
                .expect("refused")
                .to_string()
        };
        let none = Flags::default();
        let qwen = Clip::new(Some("qwen3vl_merger"));
        assert_eq!(
            open(&qwen, &none, &["deepseek41v"]),
            format!(
                "--mmproj {}: projector type \"qwen3vl_merger\"; this seat's text model takes \
                 [\"deepseek41v\"]",
                qwen.0.display()
            )
        );
        let stranger = Clip::new(Some("qwen2vl_merger"));
        assert_eq!(
            open(&stranger, &none, &["qwen2vl_merger"]),
            format!(
                "--mmproj {}: projector type \"qwen2vl_merger\" has no rule in the vision \
                 seat's table",
                stranger.0.display()
            )
        );
        let glm = Clip::new(Some("glm5v"));
        assert_eq!(
            open(&glm, &none, &["deepseek41v"]),
            format!(
                "--mmproj {}: projector type \"glm5v\"; this seat's text model takes \
                 [\"deepseek41v\"]",
                glm.0.display()
            )
        );
        let said = open(&glm, &none, &["glm5v"]);
        assert!(
            said.starts_with(&format!("--mmproj {}: metadata clip.", glm.0.display())),
            "{said}"
        );
        let bare = Clip::new(None);
        assert_eq!(
            open(&bare, &none, &["deepseek41v"]),
            format!(
                "--mmproj {}: clip.projector_type is absent or not a string",
                bare.0.display()
            )
        );
        let v41 = Clip::new(Some("deepseek41v"));
        let limited = flags(&["--image-max-tokens", "512"]).unwrap();
        let said = open(&v41, &limited, &["deepseek41v"]);
        assert!(
            said.contains("the deepseek41v projector plans an image from its file's own grid"),
            "{said}"
        );
        let missing = PathBuf::from("/nonexistent/mmproj.gguf");
        let e = Projector::open(&missing, &none, &["deepseek41v"])
            .err()
            .expect("refused");
        assert!(
            e.to_string()
                .starts_with("--mmproj /nonexistent/mmproj.gguf: "),
            "{e}"
        );
    }

    /// The limits are resolved once over the projector's own defaults: GLM's file gives 16 and
    /// 8000 tokens with no flag, each flag moves its own side, and the card figure and the cap
    /// follow the resolved limits; a minimum over the file's maximum is refused by name. The
    /// Qwen3-VL and V4.1 rows hand out mtmd's 8 and 4096.
    #[test]
    fn the_limits_resolve_once_over_the_projectors_own_defaults() {
        let glm = Clip::with(Some("glm5v"), glm5v_header());
        let at = |a: &[&str]| {
            let f = flags(a).unwrap();
            Projector::open(&glm.0, &f, &["glm5v"])
                .map(|p| {
                    (
                        p.limits.min(),
                        p.limits.max(),
                        p.cap,
                        p.card.total(),
                        p.out_dim,
                    )
                })
                .map_err(|e| e.to_string())
        };
        assert_eq!(at(&[]), Ok((16, 8000, 8000, 3_144_794_112, 4096)));
        assert_eq!(
            at(&["--image-max-tokens", "1024"]),
            Ok((16, 1024, 1024, 1_386_172_416, 4096))
        );
        assert_eq!(
            at(&["--image-min-tokens", "100"]),
            Ok((100, 8000, 8000, 3_144_794_112, 4096))
        );
        assert_eq!(
            at(&["--image-min-tokens", "9000"]),
            Err(
                "--image-min-tokens/--image-max-tokens: image size limits: min 9000 and max \
                 8000 tokens; need 1 <= min <= max"
                    .to_owned()
            )
        );
        let bare = |kind: &str| {
            let c = Clip::new(Some(kind));
            let file = Gguf::open(&c.0).expect("opens");
            let rule = rules()
                .iter()
                .find(|r| r.projector == kind)
                .expect("a rule");
            let l = (rule.default_limits)(&file).expect("limits");
            (l.min(), l.max())
        };
        assert_eq!(bare("qwen3vl_merger"), (8, 4096));
        assert_eq!(bare("deepseek41v"), (8, 4096));
    }

    /// The seat takes the projector's resolved limits, cap and card figure as they are, so the
    /// tower is loaded at the limits its card figure was computed for.
    #[test]
    fn a_seated_projector_keeps_the_limits_it_was_opened_at() {
        let mut p = projector(2, 4096, 154_854);
        p.limits = TokenLimits::new(16, 1024).unwrap();
        p.cap = 1024;
        let seat = EncoderSeat {
            name: "NVIDIA RTX A6000".to_owned(),
            ordinal: 0,
            reserve: Some(3),
        };
        let at = p.seated(seat.clone());
        assert_eq!((at.limits.min(), at.limits.max(), at.cap), (16, 1024, 1024));
        assert_eq!((at.card.total(), at.seat), (3, seat));
        assert_eq!(at.path, PathBuf::from("/m/mmproj.gguf"));
    }

    /// An image of a media call reads its three facts off the prepare: the position its span starts
    /// at, the patches it encodes (those, not a copy) and the positions it takes.
    #[test]
    fn an_image_is_read_off_its_prepare() {
        let prepared = Prepared {
            span_len: 7,
            patches: patches(2, 3),
        };
        let image = Image::of(5, &prepared);
        assert_eq!((image.at, image.span_len), (5, 7));
        assert!(std::ptr::eq(image.patches, &prepared.patches));
    }

    /// The `vision` record: the line the old eprintln printed, with the home and the cap added,
    /// read back whole by its kind.
    #[test]
    fn the_vision_record_renders_and_reads_back() {
        let seat = EncoderSeat {
            name: "NVIDIA GeForce RTX 3090".to_owned(),
            ordinal: 1,
            reserve: None,
        };
        let line = record(
            &seat,
            (970_924_032, 339_878_648),
            ScratchHome::Own,
            1024,
            3.04,
        )
        .line();
        assert_eq!(
            line,
            "vision card=NVIDIA_GeForce_RTX_3090 seat=off-plan weights=970924032 \
             scratch=339878648 scratch_home=own cap=1024 load_s=3.0"
        );
        let log = record::Log::of(&line, &[&record::VISION]);
        let f = log.one(&record::VISION).unwrap();
        assert_eq!(f.word("seat").unwrap(), "off-plan");
        assert_eq!(
            (f.u64("weights").unwrap(), f.u64("cap").unwrap()),
            (970_924_032, 1024)
        );
        let reserve = EncoderSeat {
            reserve: Some(1),
            ..seat
        };
        let line = record(&reserve, (1, 2), ScratchHome::Arena, 4096, 0.5).line();
        assert!(
            line.contains("seat=stage-reserve") && line.contains("scratch_home=arena"),
            "{line}"
        );
    }

    /// Values a position holds in the span tests.
    const W: usize = 3;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        Start,
        Image,
        NewLine,
        End,
    }

    fn patches(h: usize, w: usize) -> Patches {
        Patches {
            plan: vision::GridPlan {
                n_llm_h: h,
                n_llm_w: w,
                best_h: 0,
                best_w: 0,
            },
            n_vit_h: 0,
            n_vit_w: 0,
            patch_len: 0,
            bf16: Vec::new(),
        }
    }

    /// The rows a tower writes for a plan's grid, one row a cell.
    fn cells(p: &Patches) -> usize {
        p.plan.n_llm_h * p.plan.n_llm_w
    }

    /// A merger tower's span ([`merged_cells`]) is the rows the tower wrote, a position a cell of
    /// the plan's grid, with nothing around them.
    #[test]
    fn a_merged_span_is_the_cells_rows() {
        let rows: Vec<u16> = (0..6 * W as u16).collect();
        let (span, kinds) = merged_cells(&rows, &patches(2, 3)).unwrap();
        assert_eq!(span, rows);
        assert_eq!(kinds.len(), 6);
    }

    /// A merger tower's spans become the plain spans of a media call: each at its own positions
    /// over its own values of the one buffer of rows.
    #[test]
    fn merged_spans_are_plain_media_spans_over_the_one_buffer() {
        let (a, b) = (patches(1, 2), patches(2, 1));
        let images = [
            Image {
                at: 3,
                patches: &a,
                span_len: 2,
            },
            Image {
                at: 9,
                patches: &b,
                span_len: 2,
            },
        ];
        let mut n = 0u16;
        let encode = |p: &Patches| -> Result<Vec<u16>, String> {
            n += 1;
            Ok((0..(cells(p) * W) as u16).map(|i| n * 100 + i).collect())
        };
        let got = span_rows_of(images, W, encode, merged_cells).unwrap();
        let spans = got.media_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].at, 3..5);
        assert_eq!(spans[0].rows, [100, 101, 102, 103, 104, 105]);
        assert_eq!(spans[1].at, 9..11);
        assert_eq!(spans[1].rows, [200, 201, 202, 203, 204, 205]);
    }

    /// The values a tower writes for a plan's grid, 0, 1, 2, ... one row a cell.
    fn ramp(p: &Patches) -> Result<Vec<u16>, String> {
        Ok((0..(cells(p) * W) as u16).collect())
    }

    /// The two spans of a merged call as `ImageSpan`s: the same positions over the same values as
    /// the plain spans, each with the grid of its own image.
    #[test]
    fn image_spans_are_the_media_spans_with_each_spans_grid() {
        let (a, b) = (patches(1, 2), patches(2, 1));
        let images = [
            Image {
                at: 3,
                patches: &a,
                span_len: 2,
            },
            Image {
                at: 9,
                patches: &b,
                span_len: 2,
            },
        ];
        let got = span_rows_of(images, W, ramp, merged_cells).unwrap();
        let spans = got.image_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].span.at.clone(), spans[0].grid), (3..5, (2, 1)));
        assert_eq!((spans[1].span.at.clone(), spans[1].grid), (9..11, (1, 2)));
        let plain = got.media_spans();
        assert_eq!(spans[0].span.rows, plain[0].rows);
        assert_eq!(spans[1].span.rows, plain[1].rows);
    }

    /// Two images of equal area whose grids are each other's transpose: a span takes the grid of
    /// its own image, `(nx, ny)` = `(n_llm_w, n_llm_h)`, in feed order. A swapped pair, or a span
    /// given another image's grid, still holds `nx · ny` positions, so `ImageSpan::check` passes it.
    #[test]
    fn equal_area_images_keep_their_own_swapped_grids_in_feed_order() {
        let (a, b) = (patches(2, 3), patches(3, 2));
        let images = [
            Image {
                at: 3,
                patches: &a,
                span_len: 6,
            },
            Image {
                at: 12,
                patches: &b,
                span_len: 6,
            },
        ];
        let got = span_rows_of(images, W, ramp, merged_cells).unwrap();
        let spans = got.image_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].grid, a.plan.span_grid());
        assert_eq!(spans[1].grid, b.plan.span_grid());
        assert_eq!([spans[0].grid, spans[1].grid], [(3, 2), (2, 3)]);
        assert!(spans.iter().all(|s| s.grid.0 * s.grid.1 == s.span.at.len()));
        assert_eq!(
            ImageSpan::rows_of(0, &spans),
            [(3, 3, 2), (12, 2, 3)],
            "the rotation table's (first row, nx, ny)"
        );
    }

    /// A span that adds rows around the tower's: a Start row of 9s, each grid row's cells and a
    /// NewLine row of 8s, an End row of 7s.
    fn delimited(rows: &[u16], p: &Patches) -> Result<(Vec<u16>, Vec<Kind>), GateError> {
        let (mut out, mut kinds) = (vec![9; W], vec![Kind::Start]);
        for row in rows.chunks(p.plan.n_llm_w * W) {
            out.extend_from_slice(row);
            kinds.extend(vec![Kind::Image; row.len() / W]);
            out.extend([8; W]);
            kinds.push(Kind::NewLine);
        }
        out.extend([7; W]);
        kinds.push(Kind::End);
        Ok((out, kinds))
    }

    #[test]
    fn two_images_come_back_in_one_pair_of_buffers_with_each_spans_ranges() {
        let (a, b) = (patches(2, 2), patches(1, 3));
        let images = [
            Image {
                at: 5,
                patches: &a,
                span_len: 8,
            },
            Image {
                at: 20,
                patches: &b,
                span_len: 6,
            },
        ];
        let mut seen = Vec::new();
        // The tower's rows of image n: values n·100…, and the first image's window is wider than
        // its grid: only the grid's rows are the image's.
        let encode = |p: &Patches| -> Result<Vec<u16>, String> {
            seen.push((p.plan.n_llm_h, p.plan.n_llm_w));
            let n = seen.len() as u16;
            let len = cells(p) * W + if n == 1 { 2 * W } else { 0 };
            Ok((0..len as u16).map(|i| n * 100 + i).collect())
        };
        let got = span_rows_of(images, W, encode, delimited).unwrap();
        assert_eq!(seen, [(2, 2), (1, 3)]);
        assert_eq!(
            got.ats,
            [
                SpanAt {
                    positions: 5..13,
                    rows: 0..8 * W,
                    kinds: 0..8,
                    grid: (2, 2)
                },
                SpanAt {
                    positions: 20..26,
                    rows: 8 * W..14 * W,
                    kinds: 8..14,
                    grid: (3, 1)
                },
            ]
        );
        assert_eq!((got.rows.len(), got.kinds.len()), (14 * W, 14));
        let one = &got.rows[got.ats[0].rows.clone()];
        let cell = |first: u16| [first, first + 1, first + 2];
        let want_one: Vec<u16> = [
            [9; W],
            cell(100),
            cell(103),
            [8; W],
            cell(106),
            cell(109),
            [8; W],
            [7; W],
        ]
        .concat();
        assert_eq!(one, want_one);
        let two = &got.rows[got.ats[1].rows.clone()];
        let want_two: Vec<u16> = [[9; W], cell(200), cell(203), cell(206), [8; W], [7; W]].concat();
        assert_eq!(two, want_two);
        assert_eq!(
            &got.kinds[got.ats[1].kinds.clone()],
            [
                Kind::Start,
                Kind::Image,
                Kind::Image,
                Kind::Image,
                Kind::NewLine,
                Kind::End
            ]
        );
    }

    #[test]
    fn a_short_window_a_wrong_span_and_a_failed_encode_are_refused_by_name() {
        let p = patches(2, 2);
        let image = |span_len| {
            [Image {
                at: 5,
                patches: &p,
                span_len,
            }]
        };
        let rows = |len: usize| move |_: &Patches| Ok::<_, String>(vec![1; len]);
        let refused = |r: Result<SpanRows<Kind>, GateError>| r.unwrap_err().to_string();
        let short = span_rows_of(image(8), W, rows(4 * W - 1), delimited);
        assert_eq!(
            refused(short),
            "the image at 5: 11 encoded values for a 2x2 grid, want 12"
        );
        let span = span_rows_of(image(7), W, rows(4 * W), delimited);
        assert_eq!(
            refused(span),
            "the image at 5: a 2x2 grid splices 8 positions, its prepare said 7"
        );
        let failed = span_rows_of(
            image(8),
            W,
            |_: &Patches| Err::<Vec<u16>, _>("device lost"),
            delimited,
        );
        assert_eq!(refused(failed), "the image at 5: device lost");
    }
}
