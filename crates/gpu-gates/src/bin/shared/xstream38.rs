//! The expert stream a qwen4exp load runs: `BLOOMERY_XSTREAM` set or
//! unset by one rule ([`xstream38`]), for `generate_qwen3moe` and the
//! Qwen3.8 serve seat, each including this file. It lives beside the bins,
//! not in the lib, because it names the qwen3moe body.

use bloomery_gpu::arch::qwen3moe::Body38;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::xstream::{XMode, XSTREAM_ROOM};
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_gates::GateError;

/// Where a qwen4exp load stages, as the unset `BLOOMERY_XSTREAM` rule
/// ([`xstream38`]) reads its `--place`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage38 {
    /// Plan (a)'s stage card with no expert tier card (`--place a`).
    A,
    /// Plan (a)'s stage card beside an expert tier card (`--place bp`).
    Tiered,
    /// Any other stage (`--place gate`).
    Other,
}

/// `BLOOMERY_XSTREAM` on a qwen4exp load (`Body38::set_xstream`), for
/// `generate_qwen3moe` and the Qwen3.8 serve seat alike, and the `xstream=`
/// line the binary prints (the word, then why): `set` as given; unset,
/// `split` at [`Stage38::A`] with a residency machine (`admit` when the card
/// has no room for the stream's ring, the refusal named on the line),
/// `admit` at [`Stage38::Tiered`] with one (its expert tier the ring does
/// not serve), `off` everywhere else. `admit` or `split` beside
/// `BLOOMERY_RESIDENCY=off` is refused by name (`Body38::set_xstream`).
pub fn xstream38(
    gpu: &Gpu,
    body: &mut Body38,
    set: Option<&str>,
    stage: Stage38,
    residency: Residency,
) -> Result<String, GateError> {
    let why = match set {
        Some(word) => {
            body.set_xstream(gpu, XMode::parse(word)?)?;
            "set".to_string()
        }
        None if residency == Residency::Off || stage == Stage38::Other => {
            body.set_xstream(gpu, XMode::Off)?;
            "unset: no residency or not --place a".to_string()
        }
        None if stage == Stage38::Tiered => {
            body.set_xstream(gpu, XMode::Admit)?;
            "unset: --place bp, whose expert tier the ring does not serve".to_string()
        }
        None => match body.set_xstream(gpu, XMode::Split) {
            Ok(()) => "unset: --place a with a residency".to_string(),
            Err(
                e @ GpuError::Shape {
                    what: XSTREAM_ROOM, ..
                },
            ) => {
                body.set_xstream(gpu, XMode::Admit)?;
                format!("unset: split has no room: {e}")
            }
            Err(e) => return Err(e.into()),
        },
    };
    Ok(format!("xstream={} ({why})", body.xstream_word()))
}
