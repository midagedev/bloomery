//! Qwen3.8's [`MtpBody`]: [`Body38`]'s draft layer ([`Mtp38`]) behind the
//! shared window's calls, each one call into the model's own API. The MTP
//! layer walks the shared feed, head and walk mode; the arena and the prompt
//! path are Qwen3.8's own types.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::{Body38, Mtp38, Prompt38, Qwen38Model, TargetRows};
use bloomery_gpu::draft::{DraftFeed, WalkBounds};
use bloomery_gpu::media::ImageSpan;
use bloomery_gpu::model::Rows;

use super::WHAT;
use crate::mtp::{
    DraftHead, Feed, MtpBody, SlotChain, UnitSink, WalkMode, chains_at_once, write_proposal,
};
use runtime::prompt::CallEnd;

impl MtpBody for Body38 {
    /// A proposal fills the widest verify the GDN lanes hold.
    const WIDTH: usize = <Body38 as Rows>::MAX_ROWS - 1;
    const WALK_ROWS: usize = <Mtp38 as WalkBounds>::WALK_ROWS;
    /// The prompt's store walks run wide, through the GEMM path.
    const STORE_ROWS: usize = <Mtp38 as WalkBounds>::STORE_ROWS;
    /// The walk's argmax writes each row's probability beside its id, and
    /// the chain reads both back at once ([`Qwen38Model::mtp_chain`]).
    const PROBS: bool = true;

    type Arena = TargetRows;
    const STEP_ARENA: TargetRows = TargetRows::Step;
    const VERIFY_ARENA: TargetRows = TargetRows::Pass;

    type Path = Prompt38;

    /// The row list when the load opened one, else the full vocabulary.
    fn head(m: &Qwen38Model) -> Result<DraftHead, GpuError> {
        let rows = m.body(WHAT)?.mtp().is_some_and(|d| d.head_map().is_some());
        Ok(if rows {
            DraftHead::Rows
        } else {
            DraftHead::Full
        })
    }

    /// Four streams of the model's width ([`Body38::mtp_tap_width`]).
    fn hidden_width(m: &Qwen38Model) -> Result<usize, GpuError> {
        Ok(m.body(WHAT)?.mtp_tap_width())
    }

    /// One row of the model's width ([`Body38::embed_width`]): a hidden row
    /// is four streams of it.
    fn embed_width(m: &Qwen38Model) -> Result<usize, GpuError> {
        Ok(m.body(WHAT)?.embed_width())
    }

    /// [`Mtp38::held`]; 0 on a load without the layer.
    fn held(m: &Qwen38Model) -> Result<usize, GpuError> {
        Ok(m.body(WHAT)?.mtp().map_or(0, Mtp38::held))
    }

    fn retain_prompt_row(
        m: &mut Qwen38Model,
        walk: TargetRows,
        first: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        let (gpu, _, body) = m.body_parts(WHAT)?;
        body.retain_prompt_row(gpu, walk, first, pos)
    }

    /// [`Qwen38Model::prompt38_call`]: the body folds a closing call's last
    /// id into its last ubatch where that ubatch holds a row before it.
    fn prompt_with(
        m: &mut Qwen38Model,
        ids: &[u32],
        path: Prompt38,
        end: CallEnd,
        sink: Option<&mut UnitSink<'_, Body38>>,
    ) -> Result<Option<u32>, GpuError> {
        m.prompt38_call(ids, path, end, sink)
    }

    fn media_frame(
        m: &mut Qwen38Model,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
        run: impl FnOnce(&mut Qwen38Model) -> Result<u32, GpuError>,
    ) -> Result<u32, GpuError> {
        m.media38_frame(ids, spans, run)
    }

    /// The target's ubatch call inside `media_frame`: its rotations and span rows are
    /// already installed before the draft's join and remain live through every unit.
    fn prompt_media_with(
        m: &mut Qwen38Model,
        ids: &[u32],
        _spans: &[ImageSpan<'_>],
        sink: Option<&mut UnitSink<'_, Body38>>,
    ) -> Result<u32, GpuError> {
        m.prompt38_with(ids, Prompt38::Gemm, sink)
    }

    fn walk(
        m: &mut Qwen38Model,
        feed: Feed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        m.mtp_walk(DraftFeed::Rows(feed), head, mode)
    }

    /// One store walk of every slot's rows at once
    /// ([`Qwen38Model::draft_store_slots`]).
    fn store_slots(
        m: &mut Qwen38Model,
        walks: &[(usize, Feed<'_, TargetRows>)],
        head: DraftHead,
    ) -> Result<(), GpuError> {
        m.draft_store_slots(walks, head)
    }

    fn chain(
        m: &mut Qwen38Model,
        refresh: Feed<'_, TargetRows>,
        own: usize,
        head: DraftHead,
        mode: WalkMode,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError> {
        let d = m.mtp_chain(DraftFeed::Rows(refresh), own, head, mode)?;
        write_proposal(&d, out, p)
    }

    /// Every slot's chains at once ([`chains_at_once`]: one walk a depth
    /// over every slot's rows).
    fn chain_slots(
        m: &mut Qwen38Model,
        chains: &mut [SlotChain<'_, TargetRows>],
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Vec<usize>, GpuError> {
        chains_at_once(m, chains, head, mode)
    }
}

// The adapter's widths fit together, and a refresh of a whole verify's
// rows is a captured walk.
const _: () = <Body38 as MtpBody>::FITS;
const _: () = assert!(<Body38 as MtpBody>::VERIFY_ROWS <= <Mtp38 as WalkBounds>::GRAPH_ROWS);
