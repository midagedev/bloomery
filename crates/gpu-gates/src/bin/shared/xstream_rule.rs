//! The expert stream rule a load runs: `BLOOMERY_XSTREAM` set or unset by
//! one rule ([`xstream_rule`]), over any body that implements
//! [`XStreamBody`] — for `generate_qwen3moe` and the Qwen3.8, MiniMax and
//! Muse Spark serve seats, each including this file; the MiniMax gates' seat
//! asks its [`plans_ring`]. It lives beside the
//! bins, not in the lib, because it drives a load's body, whose own guards
//! the trait carries.
#![allow(
    dead_code,
    reason = "the rule serves several binaries; each takes the helpers it reads"
)]

use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::xstream::{
    XMode, XSTREAM_FORM, XSTREAM_NO_HOST, XSTREAM_ROOM, XStreamBody,
};
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;

/// Where a load stages, as the unset `BLOOMERY_XSTREAM` rule
/// ([`xstream_rule`]) reads its `--place`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Plan (a)'s stage card with no expert tier card (`--place a`).
    A,
    /// Plan (a)'s stage card beside an expert tier card (`--place bp`).
    Tiered,
    /// Any other stage (`--place gate`).
    Other,
}

impl Stage {
    /// The stage `place` is: an expert tier card beside the stage card is
    /// [`Stage::Tiered`], `--place gate` [`Stage::Other`], any other
    /// placement plan (a)'s [`Stage::A`].
    pub fn of(place: Place) -> Stage {
        if !place.tier_cards().is_empty() {
            Stage::Tiered
        } else if place == Place::Gate {
            Stage::Other
        } else {
            Stage::A
        }
    }
}

/// Whether [`xstream_rule`] starts the stream at `split` on a load at `stage`:
/// `set` is `split`, or unset at [`Stage::A`] beside a residency
/// (`residency_off` is `BLOOMERY_RESIDENCY=off` as set; an unset word the plan
/// turns `off` is read only after the plan, where the rule then ends at `off`).
/// Every load asks this before its plan: one that streams reserves the lane's
/// staging on the host (`workstation::reserve_stream_staging`), and a body whose
/// plan keeps the stream's ring out of its expert budget plans the ring.
pub fn plans_ring(set: Option<&str>, stage: Stage, residency_off: bool) -> bool {
    match set {
        Some(word) => word == "split",
        None => stage == Stage::A && !residency_off,
    }
}

/// `BLOOMERY_XSTREAM` on a load ([`XStreamBody::set_xstream`]), for
/// `generate_qwen3moe` and the Qwen3.8 and Muse Spark serve seats alike, and
/// the `xstream=` line the binary prints (the word, then why): `set` as given; unset,
/// `split` at [`Stage::A`] with a residency machine (`admit` when the card
/// has no room for the stream's ring or the ring cannot hold the source layout, `off` when the plan holds no host
/// expert to stream, the refusal named on the line),
/// `admit` at [`Stage::Tiered`] with one (its expert tier the ring does not
/// serve), `off` everywhere else. `admit` or `split` beside
/// `BLOOMERY_RESIDENCY=off` is refused by name (the body's `set_xstream`).
pub fn xstream_rule<B: XStreamBody>(
    gpu: &Gpu,
    body: &mut B,
    set: Option<&str>,
    stage: Stage,
    residency: Residency,
) -> Result<String, GateError> {
    let why = match set {
        Some(word) => {
            body.set_xstream(gpu, XMode::parse(word)?)?;
            "set".to_string()
        }
        None if plans_ring(None, stage, residency == Residency::Off) => {
            match body.set_xstream(gpu, XMode::Split) {
                Ok(()) => "unset: --place a with a residency".to_string(),
                Err(
                    e @ GpuError::Shape {
                        what: XSTREAM_ROOM, ..
                    },
                ) => {
                    body.set_xstream(gpu, XMode::Admit)?;
                    format!("unset: split has no room: {e}")
                }
                Err(
                    e @ GpuError::State {
                        what: XSTREAM_NO_HOST,
                        ..
                    },
                ) => {
                    body.set_xstream(gpu, XMode::Off)?;
                    format!("unset: the plan holds no host expert to stream: {e}")
                }
                Err(
                    e @ GpuError::Shape {
                        what: XSTREAM_FORM, ..
                    },
                ) => {
                    body.set_xstream(gpu, XMode::Admit)?;
                    format!("unset: split cannot hold the stack layout: {e}")
                }
                Err(e) => return Err(e.into()),
            }
        }
        None if residency == Residency::Off || stage == Stage::Other => {
            body.set_xstream(gpu, XMode::Off)?;
            "unset: no residency or not --place a".to_string()
        }
        None => {
            body.set_xstream(gpu, XMode::Admit)?;
            "unset: --place bp, whose expert tier the ring does not serve".to_string()
        }
    };
    Ok(format!("xstream={} ({why})", body.xstream_word()))
}
