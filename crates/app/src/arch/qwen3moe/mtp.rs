//! Qwen3.8's [`MtpBody`]: [`Body38`]'s draft layer ([`Mtp38`]) behind the
//! shared window's calls, each one call into the model's own API. The shared
//! feed and walk mode map onto Qwen3.8's one to one; the arena, the head and
//! the prompt path are Qwen3.8's own types.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::{
    Body38, MTP_GRAPH_ROWS, MTP_ROWS, Mtp38, MtpFeed, MtpHead, MtpHidden, MtpMode, Prompt38,
    Qwen38Model, TargetRows,
};
use bloomery_gpu::model::Rows;

use super::WHAT;
use crate::mtp::{Feed, Hidden, MtpBody, UnitSink, WalkMode};

impl MtpBody for Body38 {
    /// A proposal fills the widest verify the GDN lanes hold.
    const WIDTH: usize = <Body38 as Rows>::MAX_ROWS - 1;
    const WALK_ROWS: usize = MTP_ROWS;

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
        m.mtp_walk(mtp_feed(feed), head, mtp_mode(mode))
    }

    fn chain(
        m: &mut Qwen38Model,
        refresh: Feed<'_, TargetRows>,
        own: usize,
        head: MtpHead,
        mode: WalkMode,
        out: &mut [u32],
    ) -> Result<usize, GpuError> {
        let d = m.mtp_chain(mtp_feed(refresh), own, head, mtp_mode(mode))?;
        let n = d.tokens.len();
        let places = out.len();
        out.get_mut(..n)
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("a proposal of {n} ids into {places} places"),
            })?
            .copy_from_slice(&d.tokens);
        Ok(n)
    }
}

// The adapter's widths fit together, and a refresh of a whole verify's
// rows is a captured walk.
const _: () = <Body38 as MtpBody>::FITS;
const _: () = assert!(<Body38 as MtpBody>::VERIFY_ROWS <= MTP_GRAPH_ROWS);

/// The shared feed as Qwen3.8's: always [`MtpFeed::Rows`], which the window
/// builds; [`MtpFeed::Own`] is the chain's own.
fn mtp_feed(f: Feed<'_, TargetRows>) -> MtpFeed<'_> {
    MtpFeed::Rows {
        tokens: f.tokens,
        pos0: f.pos0,
        hidden: match f.hidden {
            Hidden::Host(v) => MtpHidden::Host(v),
            Hidden::Target { walk, first } => MtpHidden::Target { walk, first },
        },
    }
}

/// The walk mode as Qwen3.8's: each shared mode is one of the draft's own,
/// the store-only walk included.
fn mtp_mode(m: WalkMode) -> MtpMode {
    match m {
        WalkMode::Eager => MtpMode::Eager,
        WalkMode::Graph => MtpMode::Graph,
        WalkMode::Store => MtpMode::Store,
    }
}

#[cfg(test)]
mod tests {
    use super::{mtp_feed, mtp_mode};
    use crate::mtp::{Feed, Hidden, WalkMode};
    use bloomery_gpu::arch::qwen3moe::{MtpFeed, MtpHidden, MtpMode, TargetRows};

    /// The shared walk mode and feed map onto Qwen3.8's one to one: each
    /// value comes back from its image as itself, and a variant added to
    /// either walk mode or to the shared hidden source fails to compile.
    #[test]
    fn feed_and_mode_map_one_to_one() {
        fn mode_back(m: MtpMode) -> WalkMode {
            match m {
                MtpMode::Eager => WalkMode::Eager,
                MtpMode::Graph => WalkMode::Graph,
                MtpMode::Store => WalkMode::Store,
            }
        }
        for m in [WalkMode::Eager, WalkMode::Graph, WalkMode::Store] {
            assert_eq!(mode_back(mtp_mode(m)), m);
        }

        let zeros = [0.0f32; 2];
        let tokens = [7u32, 8];
        let sources = [
            Hidden::Host(&zeros[..]),
            Hidden::Target {
                walk: TargetRows::Step,
                first: 0,
            },
            Hidden::Target {
                walk: TargetRows::Pass,
                first: 1,
            },
            Hidden::Target {
                walk: TargetRows::Ubatch,
                first: 3,
            },
        ];
        for h in sources {
            let f = mtp_feed(Feed {
                tokens: &tokens,
                pos0: 5,
                hidden: h,
            });
            let MtpFeed::Rows {
                tokens: t,
                pos0,
                hidden,
            } = f
            else {
                panic!("the shared feed of {h:?} mapped to {f:?}");
            };
            assert!(std::ptr::eq(t, &tokens[..]) && pos0 == 5, "{f:?}");
            match (h, hidden) {
                (Hidden::Host(a), MtpHidden::Host(b)) => assert!(std::ptr::eq(a, b)),
                (Hidden::Target { walk: a, first: i }, MtpHidden::Target { walk: b, first: j }) => {
                    assert_eq!((a, i), (b, j));
                }
                (h, hidden) => panic!("{h:?} mapped to {hidden:?}"),
            }
        }
    }
}
