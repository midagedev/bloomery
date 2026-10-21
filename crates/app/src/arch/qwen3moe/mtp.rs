//! Qwen3.8's [`MtpBody`]: [`Body38`]'s draft layer ([`Mtp38`]) behind the
//! shared window's calls, each one call into the model's own API. The MTP
//! layer walks the shared feed and walk mode; the arena, the head and the
//! prompt path are Qwen3.8's own types.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::{
    Body38, MTP_GRAPH_ROWS, MTP_ROWS, MTP_STORE_ROWS, Mtp38, MtpFeed, MtpHead, Prompt38,
    Qwen38Model, TargetRows,
};
use bloomery_gpu::model::Rows;

use super::WHAT;
use crate::mtp::{Feed, MtpBody, UnitSink, WalkMode};

impl MtpBody for Body38 {
    /// A proposal fills the widest verify the GDN lanes hold.
    const WIDTH: usize = <Body38 as Rows>::MAX_ROWS - 1;
    const WALK_ROWS: usize = MTP_ROWS;
    /// The prompt's store walks run wide, through the GEMM path.
    const STORE_ROWS: usize = MTP_STORE_ROWS;
    /// The walk's argmax writes each row's probability beside its id, and
    /// the chain reads both back at once ([`Qwen38Model::mtp_chain`]).
    const PROBS: bool = true;

    type Arena = TargetRows;
    const STEP_ARENA: TargetRows = TargetRows::Step;
    const VERIFY_ARENA: TargetRows = TargetRows::Pass;

    type Head = MtpHead;
    type Path = Prompt38;

    /// The row list when the load opened one, else the full vocabulary.
    fn head(m: &Qwen38Model) -> Result<MtpHead, GpuError> {
        let rows = m.body(WHAT)?.mtp().is_some_and(|d| d.head_map().is_some());
        Ok(if rows { MtpHead::Rows } else { MtpHead::Full })
    }

    /// Four streams of the model's width ([`Body38::mtp_tap_width`]).
    fn hidden_width(m: &Qwen38Model) -> Result<usize, GpuError> {
        Ok(m.body(WHAT)?.mtp_tap_width())
    }

    /// [`Mtp38::held`]; 0 on a load without the layer.
    fn held(m: &Qwen38Model) -> Result<usize, GpuError> {
        Ok(m.body(WHAT)?.mtp().map_or(0, Mtp38::held))
    }

    fn prompt_with(
        m: &mut Qwen38Model,
        ids: &[u32],
        path: Prompt38,
        sink: Option<&mut UnitSink<'_, Body38>>,
    ) -> Result<u32, GpuError> {
        m.prompt38_with(ids, path, sink)
    }

    fn walk(
        m: &mut Qwen38Model,
        feed: Feed<'_, TargetRows>,
        head: MtpHead,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        m.mtp_walk(MtpFeed::Rows(feed), head, mode)
    }

    fn chain(
        m: &mut Qwen38Model,
        refresh: Feed<'_, TargetRows>,
        own: usize,
        head: MtpHead,
        mode: WalkMode,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError> {
        let d = m.mtp_chain(MtpFeed::Rows(refresh), own, head, mode)?;
        let n = d.tokens.len();
        let places = out.len();
        out.get_mut(..n)
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("a proposal of {n} ids into {places} places"),
            })?
            .copy_from_slice(&d.tokens);
        if let Some(p) = p {
            let places = p.len();
            p.get_mut(..n)
                .ok_or_else(|| GpuError::Shape {
                    what: WHAT,
                    detail: format!("a proposal's {n} probabilities into {places} places"),
                })?
                .copy_from_slice(&d.p);
        }
        Ok(n)
    }
}

// The adapter's widths fit together, and a refresh of a whole verify's
// rows is a captured walk.
const _: () = <Body38 as MtpBody>::FITS;
const _: () = assert!(<Body38 as MtpBody>::VERIFY_ROWS <= MTP_GRAPH_ROWS);
