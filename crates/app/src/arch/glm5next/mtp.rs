//! GLM-5.3-Flash's [`MtpBody`]: [`Body`]'s NextN layer
//! ([`bloomery_gpu_glm5next::Nextn`]) behind the shared window's calls, each
//! one call into the body's own API. The NextN program walks the shared feed,
//! head and walk mode; the arena and the prompt path are GLM's own types.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::draft::{DraftFeed, WalkBounds};
use bloomery_gpu::media::MediaSpan;
use bloomery_gpu::model::Rows;
use bloomery_gpu_glm5next::{Body, GlmArena, Nextn, PrefillMode};

use super::WHAT;
use crate::mtp::{DraftHead, Feed, MtpBody, UnitSink, WalkMode};
use runtime::prompt::CallEnd;

impl MtpBody for Body {
    /// A proposal of one id: the NextN layer predicts one position ahead.
    const WIDTH: usize = 1;
    const WALK_ROWS: usize = <Nextn as WalkBounds>::WALK_ROWS;

    type Arena = GlmArena;
    const STEP_ARENA: GlmArena = GlmArena::Step;
    const VERIFY_ARENA: GlmArena = GlmArena::Pair;

    type Path = PrefillMode;

    /// The full vocabulary: the load opens no row list. Refused by name on a
    /// load without the NextN layer, so the window never opens over it.
    fn head(m: &GpuModel<Body>) -> Result<DraftHead, GpuError> {
        match m.body(WHAT)?.nextn() {
            Some(_) => Ok(DraftHead::Full),
            None => Err(GpuError::Shape {
                what: WHAT,
                detail: "an MTP draft on a load without the NextN layer (open_nextn loads it)"
                    .to_string(),
            }),
        }
    }

    /// One row of the model's width ([`Body::nextn_hidden_width`]), normed
    /// as the target's head norms it.
    fn hidden_width(m: &GpuModel<Body>) -> Result<usize, GpuError> {
        Ok(m.body(WHAT)?.nextn_hidden_width())
    }

    /// [`Nextn::held`]; refused by name on a load without the layer.
    fn held(m: &GpuModel<Body>) -> Result<usize, GpuError> {
        m.body(WHAT)?
            .nextn()
            .map(Nextn::held)
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: "the draft's store on a load without the NextN layer".to_string(),
            })
    }

    fn retain_prompt_row(
        m: &mut GpuModel<Body>,
        walk: GlmArena,
        first: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        bloomery_gpu_glm5next::retain_prompt_row(m, walk, first, pos)
    }

    fn prompt_with(
        m: &mut GpuModel<Body>,
        ids: &[u32],
        path: PrefillMode,
        end: CallEnd,
        sink: Option<&mut UnitSink<'_, Body>>,
    ) -> Result<Option<u32>, GpuError> {
        bloomery_gpu_glm5next::prompt_with(m, ids, path, end, sink)
    }

    /// [`bloomery_gpu_glm5next::prompt_media_with`]: the body's media call with the sink, its
    /// residency pass its own. The draft's walks over its units read the span rows at a span's
    /// positions ([`Feed::media`]).
    fn prompt_media_with(
        m: &mut GpuModel<Body>,
        ids: &[u32],
        spans: &[MediaSpan<'_>],
        sink: Option<&mut UnitSink<'_, Body>>,
    ) -> Result<u32, GpuError> {
        bloomery_gpu_glm5next::prompt_media_with(m, ids, spans, sink)
    }

    fn walk(
        m: &mut GpuModel<Body>,
        feed: Feed<'_, GlmArena>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        bloomery_gpu_glm5next::nextn_walk(m, DraftFeed::Rows(feed), head, mode)
    }

    /// The NextN head reads back its ids alone: `p` is refused by name.
    fn chain(
        m: &mut GpuModel<Body>,
        refresh: Feed<'_, GlmArena>,
        own: usize,
        head: DraftHead,
        mode: WalkMode,
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, GpuError> {
        if p.is_some() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: "a chain's probabilities: the NextN head reads back its ids alone".into(),
            });
        }
        bloomery_gpu_glm5next::nextn_chain(m, DraftFeed::Rows(refresh), own, head, mode, out)
    }
}

// The adapter's widths fit together: a verify of two rows, the pass's
// widest, is one walk.
const _: () = <Body as MtpBody>::FITS;
const _: () = assert!(<Body as MtpBody>::VERIFY_ROWS == <Body as Rows>::MAX_ROWS);
// A proposal of one id has no second draft iteration; a wider one reads the
// file's `index_share_mtp` (`Hparams::index_share_mtp`) first.
const _: () = assert!(
    <Body as MtpBody>::WIDTH == 1,
    "a GLM proposal wider than one id needs the index_share_mtp rule (Hparams::index_share_mtp)"
);
