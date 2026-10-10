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
//!   `--image-max-tokens N` (8 and 4096 unless the file's projector plans its own sizes).
//! * [`Projector`] — the file opened on the host: its projector type refused by name unless the
//!   seat takes it, the tower's bytes on its card, the model's image input, and the row width;
//!   [`Projector::check_text`] holds it against the text model, every mismatch refused by name.
//! * [`EncoderSeat`] — where the tower sits: the smallest card outside the plan, else a reserve on
//!   the stage card ([`reserve_on`] puts the same reserve into a placed plan and a whole-fit probe).
//! * [`table_bytes_per_position`] — what a plan adds per position of a sequence for a media load.
//! * [`on_encoder_card`] — binds the tower's card for a tower call and the model's card back.
//! * [`record`] — the `vision` line after the tower's load.

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cuda_core::CudaContext;
use gguf::Gguf;
use model::placement::Machine;
use model::placement::workstation::{CardSpec, DeviceInfo};
use vision::arch::card::CardBytes;
use vision::arch::qwen3vl::TokenLimits;
use vision::arch::{deepseek41v, qwen3vl};
use vision::{MediaModel, VisionError};

use crate::GateError;
use crate::generate::Place;
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

    /// Whether `--image-min-tokens` or `--image-max-tokens` was given.
    #[must_use]
    pub fn sets_limits(&self) -> bool {
        self.min_tokens.is_some() || self.max_tokens.is_some()
    }

    /// The image-token limits: the flags' values over mtmd's defaults, refused by name when
    /// `min` is 0 or above `max`.
    pub fn limits(&self) -> Result<TokenLimits, GateError> {
        let d = TokenLimits::DEFAULT;
        TokenLimits::new(
            self.min_tokens.unwrap_or(d.min()),
            self.max_tokens.unwrap_or(d.max()),
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
            cap: |f, _| Ok(deepseek41v::Hparams::read(f)?.grid().max_tokens),
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
    /// The most image tokens one image takes, which the tower's activations are sized for.
    pub cap: usize,
}

impl Projector {
    /// The encoder file at `path`, opened as the type its `clip.projector_type` names and refused
    /// by name when that type is not one of `takes` (the projector types the seat's text model
    /// can run), when the file is not one the host crate runs, or when the limits `flags` gives
    /// steer a projector that plans its own sizes.
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
        let limits = flags.limits()?;
        let (card, media) = vision::arch::open(&file, limits).map_err(|e| at(&e))?;
        let out_dim = (rule.out_dim)(&file).map_err(|e| at(&e))?;
        let cap = (rule.cap)(&file, limits).map_err(|e| at(&e))?;
        Ok(Projector {
            path: path.to_owned(),
            rule,
            card,
            media,
            out_dim,
            cap,
        })
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

/// What the plan's reserve for the tower is called, as the plan's terms name it (beside
/// `workstation`'s draft and tier-batch reserves).
pub const RESERVE: &str = "image encoder";

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

/// `machine` with the tower's `reserve` on its stage card, when the seat is a stage reserve. The
/// one place the reserve enters a machine: a placed plan's builder and a whole-fit probe both call
/// it, so both fit the same bytes.
pub fn reserve_on(machine: &mut Machine, reserve: Option<u64>) {
    if let Some(bytes) = reserve {
        machine.cards[0].reserves.push((RESERVE.to_owned(), bytes));
    }
}

/// A machine builder (`layers` to its [`Machine`]) with [`reserve_on`] applied to what it builds.
pub fn with_reserve(
    plain: impl Fn(usize) -> Machine,
    reserve: Option<u64>,
) -> impl Fn(usize) -> Machine {
    move |layers| {
        let mut machine = plain(layers);
        reserve_on(&mut machine, reserve);
        machine
    }
}

/// A plan's refusal, beside the reserve that was in it: with no reserve the refusal as it is.
pub fn refusal(reserve: Option<u64>, e: impl Display) -> String {
    reserve.map_or_else(
        || e.to_string(),
        |bytes| {
            format!(
                "the plan holds the image encoder's {bytes} B as the stage card's {RESERVE} \
                 reserve: {e}"
            )
        },
    )
}

/// The bytes a plan adds per position of a sequence for a media load: the sequence's own rotation
/// table (`gpu::media::SeqRope`) holds a row of `width` f32 for each of its positions, `width`
/// being the f32 a rope row turns (64 for the Qwen3-VL family's text, 256 B a position).
#[must_use]
pub const fn table_bytes_per_position(width: usize) -> u64 {
    width as u64 * size_of::<f32>() as u64
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
        .w("card", seat.name.replace(' ', "_"))
        .w("seat", seat.word())
        .u("weights", weights)
        .u("scratch", scratch)
        .w("scratch_home", home.word())
        .u("cap", cap)
        .f("load_s", load_s)
}

#[cfg(test)]
mod tests {
    use gguf::Value;
    use model::placement::devices::DeviceId;
    use model::placement::workstation::{A6000, RTX_3090, plan_a};

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

    /// The four flags in llama-server's spelling; `--no-mmproj` has no value, the limits default to
    /// mtmd's 8 and 4096, and every refusal names its flag.
    #[test]
    fn the_flags_read_llama_servers_spelling() {
        let f = flags(&["--mmproj", "/m/p.gguf", "--image-min-tokens", "16"]).unwrap();
        assert_eq!(f.mmproj().unwrap(), Some(Path::new("/m/p.gguf")));
        let l = f.limits().unwrap();
        assert_eq!((l.min(), l.max()), (16, 4096));
        assert!(f.sets_limits());
        let f = flags(&["--no-mmproj", "--image-max-tokens", "1024"]).unwrap();
        assert_eq!(f.mmproj().unwrap(), None);
        assert_eq!(
            (f.limits().unwrap().min(), f.limits().unwrap().max()),
            (8, 1024)
        );
        let d = flags(&[]).unwrap();
        assert_eq!((d.mmproj().unwrap(), d.sets_limits()), (None, false));
        assert_eq!(
            (d.limits().unwrap().min(), d.limits().unwrap().max()),
            (8, 4096)
        );
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
                        .and(f.limits().map(|_| ()))
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

    /// The whole-fit probe and the placed plan take the one reserve: the machines both build carry
    /// the same named bytes, the card sets aside exactly that much more, and none changes nothing.
    #[test]
    fn the_probe_and_the_placed_plan_see_the_same_reserve() {
        let plain = plan_a(40);
        let mut probe = plan_a(40);
        reserve_on(&mut probe, Some(BYTES));
        let placed = with_reserve(plan_a, Some(BYTES))(40);
        assert_eq!(probe, placed);
        assert_eq!(
            probe.cards[0].reserves.last(),
            Some(&(RESERVE.to_owned(), BYTES))
        );
        assert_eq!(
            probe.cards[0].reserve_bytes(),
            plain.cards[0].reserve_bytes() + BYTES
        );
        assert_eq!(
            probe.cards[0].set_aside_bytes(),
            plain.cards[0].set_aside_bytes() + BYTES
        );
        let mut none = plan_a(40);
        reserve_on(&mut none, None);
        assert_eq!(none, plain);
        assert_eq!(with_reserve(plan_a, None)(40), plain);
        assert_eq!(refusal(None, "no fit"), "no fit");
        assert_eq!(
            refusal(Some(BYTES), "no fit"),
            "the plan holds the image encoder's 1476000000 B as the stage card's image encoder \
             reserve: no fit"
        );
    }

    /// The table term: a row of the rope's width in f32 a position (256 B at 64).
    #[test]
    fn the_table_term_is_a_row_of_f32_a_position() {
        assert_eq!(table_bytes_per_position(64), 256);
        assert_eq!(table_bytes_per_position(128), 512);
        assert_eq!(table_bytes_per_position(0), 0);
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

    /// A header-only GGUF with `clip.projector_type` `kind`, deleted when dropped.
    struct Clip(PathBuf);

    impl Clip {
        fn new(kind: Option<&str>) -> Clip {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let mut kvs = vec![(
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String("clip".into()),
            )];
            if let Some(k) = kind {
                kvs.push(("clip.projector_type".to_owned(), Value::String(k.into())));
            }
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
    /// cannot run, one with no rule, an absent type, a limit on a projector that plans its own, and
    /// a file that is no GGUF.
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
        let glm = Clip::new(Some("glm5v"));
        assert_eq!(
            open(&glm, &none, &["glm5v"]),
            format!(
                "--mmproj {}: projector type \"glm5v\" has no rule in the vision seat's table",
                glm.0.display()
            )
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
}
