//! Qwen3.6's [`MtpBody`]: [`Body35`]'s in-file next-token layer
//! ([`Mtp35`](bloomery_gpu::arch::qwen3moe::Mtp35)) behind the shared window.
//! Each call is one call into the layer's own API. The prompt path is
//! [`path35`](super::path35) of the ids' length — the undrafted prompt's
//! schedule — not the path the window stored at open, which is one value for
//! every length.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::arch::qwen3moe::{
    Body35, MtpArena, Qwen35moeModel, mtp_chain, mtp_walk, prompt_with35,
};

use super::path35;
use crate::mtp::{DraftHead, Feed, MtpBody, UnitSink, WalkMode};

/// The path value the window stores. [`MtpBody::prompt_with`] does not read
/// it: a Qwen3.6 prompt's schedule is [`path35`] of that call's length.
#[derive(Clone, Copy, Debug)]
pub struct Prompt35;

impl MtpBody for Body35 {
    /// A proposal of one id: the next-token layer predicts one position ahead.
    const WIDTH: usize = 1;
    /// A refresh of the verify's two rows is one walk.
    const WALK_ROWS: usize = 2;
    /// The prompt's store walk takes the same two rows.
    const STORE_ROWS: usize = 2;

    type Arena = MtpArena;
    /// A plain step's post-norm row is the head's last position, copied on
    /// the card before the next step overwrites it.
    const STEP_ARENA: MtpArena = MtpArena::Step;
    /// A verify leaves its rows in the pass arena.
    const VERIFY_ARENA: MtpArena = MtpArena::Pass;

    type Path = Prompt35;

    /// The full vocabulary: the load opens no row list. Refused by name on a
    /// load without the next-token layer.
    fn head(m: &GpuModel<Body35>) -> Result<DraftHead, GpuError> {
        match m.body("qwen35moe MtpBody")?.mtp() {
            Some(_) => Ok(DraftHead::Full),
            None => Err(GpuError::Shape {
                what: "qwen35moe MtpBody",
                detail: "an MTP draft on a load without the next-token layer (open_mtp loads it)"
                    .into(),
            }),
        }
    }

    /// One row of the draft's embedding width.
    fn hidden_width(m: &Qwen35moeModel) -> Result<usize, GpuError> {
        match m.body("qwen35moe MtpBody")?.mtp() {
            Some(d) => Ok(d.hidden()),
            None => Err(GpuError::Shape {
                what: "qwen35moe MtpBody",
                detail: "the draft's hidden width on a load without the next-token layer".into(),
            }),
        }
    }

    /// Positions the draft's store holds. Refused by name on a load without
    /// the layer.
    fn held(m: &Qwen35moeModel) -> Result<usize, GpuError> {
        match m.body("qwen35moe MtpBody")?.mtp() {
            Some(d) => Ok(d.held() as usize),
            None => Err(GpuError::Shape {
                what: "qwen35moe MtpBody",
                detail: "the draft's store on a load without the next-token layer".into(),
            }),
        }
    }

    /// [`prompt_with35`] on [`path35`] of `ids.len()`. The stored `path` is
    /// not the schedule: the undrafted prompt chooses per length.
    fn prompt_with(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        _path: Prompt35,
        sink: Option<&mut UnitSink<'_, Body35>>,
    ) -> Result<u32, GpuError> {
        prompt_with35(m, ids, path35(ids.len()), sink)
    }

    fn walk(
        m: &mut Qwen35moeModel,
        feed: Feed<'_, MtpArena>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        mtp_walk(m, feed, head, mode)
    }

    /// The next-token head reads back its ids alone: `p` is refused by name,
    /// and so is an own row (the proposal holds one id).
    fn chain(
        m: &mut Qwen35moeModel,
        refresh: Feed<'_, MtpArena>,
        own: usize,
        head: DraftHead,
        mode: WalkMode,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError> {
        mtp_chain(m, refresh, own, head, mode, out, p)
    }
}

// A verify of two rows is one walk, and the body's passes take it.
const _: () = <Body35 as MtpBody>::FITS;
