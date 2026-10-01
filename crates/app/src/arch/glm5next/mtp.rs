//! GLM-5.3-Flash's [`MtpBody`]: [`Body`]'s NextN layer
//! ([`bloomery_gpu_glm5next::Nextn`]) behind the shared window's calls, each
//! one call into the body's own API. The shared feed and walk mode map onto
//! the NextN program's one to one; the arena, the head and the prompt path
//! are GLM's own types.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::model::Rows;
use bloomery_gpu_glm5next::{
    Body, GlmArena, Nextn, NextnFeed, NextnHead, NextnHidden, NextnMode, PrefillMode, WALK_ROWS,
};

use super::WHAT;
use crate::mtp::{Feed, Hidden, MtpBody, UnitSink, WalkMode};

impl MtpBody for Body {
    /// A proposal of one id: the NextN layer predicts one position ahead.
    const WIDTH: usize = 1;
    const WALK_ROWS: usize = WALK_ROWS;

    type Arena = GlmArena;
    const STEP_ARENA: GlmArena = GlmArena::Step;
    const VERIFY_ARENA: GlmArena = GlmArena::Pair;

    type Head = NextnHead;
    type Path = PrefillMode;

    /// The full vocabulary: the load opens no row list. Refused by name on a
    /// load without the NextN layer, so the window never opens over it.
    fn head(m: &GpuModel<Body>) -> Result<NextnHead, GpuError> {
        match m.body(WHAT)?.nextn() {
            Some(_) => Ok(NextnHead::Full),
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

    fn prompt_with(
        m: &mut GpuModel<Body>,
        ids: &[u32],
        path: PrefillMode,
        sink: Option<&mut UnitSink<'_, Body>>,
    ) -> Result<u32, GpuError> {
        bloomery_gpu_glm5next::prompt_with(m, ids, path, sink)
    }

    fn walk(
        m: &mut GpuModel<Body>,
        feed: Feed<'_, GlmArena>,
        head: NextnHead,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        bloomery_gpu_glm5next::nextn_walk(m, nextn_feed(feed), head, nextn_mode(mode))
    }

    /// The NextN head reads back its ids alone: `p` is refused by name.
    fn chain(
        m: &mut GpuModel<Body>,
        refresh: Feed<'_, GlmArena>,
        own: usize,
        head: NextnHead,
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
        bloomery_gpu_glm5next::nextn_chain(m, nextn_feed(refresh), own, head, nextn_mode(mode), out)
    }
}

// The adapter's widths fit together: a verify of two rows, the pass's
// widest, is one walk.
const _: () = <Body as MtpBody>::FITS;
const _: () = assert!(<Body as MtpBody>::VERIFY_ROWS == <Body as Rows>::MAX_ROWS);

/// The shared feed as the NextN program's, field for field.
fn nextn_feed(f: Feed<'_, GlmArena>) -> NextnFeed<'_> {
    NextnFeed {
        tokens: f.tokens,
        pos0: f.pos0,
        hidden: match f.hidden {
            Hidden::Host(v) => NextnHidden::Host(v),
            Hidden::Target { walk, first } => NextnHidden::Target { walk, first },
        },
    }
}

/// The walk mode as the NextN program's: each shared mode is one of its
/// own; the program refuses a captured walk by name.
fn nextn_mode(m: WalkMode) -> NextnMode {
    match m {
        WalkMode::Eager => NextnMode::Eager,
        WalkMode::Graph => NextnMode::Graph,
        WalkMode::Store => NextnMode::Store,
    }
}
