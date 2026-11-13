//! The head128 program's [`MtpBody`]: the body's MTP layer ([`bloomery_gpu_flint::draft`]) behind
//! the shared window's calls, each one call into the body's own API. The program walks the shared
//! feed, head and walk mode; the arena is the body's own type ([`FlintArena`]) and the prompt's path
//! is the body's one batch path, which takes no choice.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::draft::{DraftFeed, WalkBounds};
use bloomery_gpu::model::Rows;
use bloomery_gpu_flint::draft::{self, Draft, FlintArena};
use runtime::prompt::CallEnd;

use crate::mtp::{DraftHead, Feed, MtpBody, UnitSink, WalkMode};

macro_rules! flint_mtp {
    ($ty:ty) => {
        impl MtpBody for $ty {
            /// A proposal of one id: the MTP layer predicts one position ahead.
            const WIDTH: usize = 1;
            const WALK_ROWS: usize = <Draft as WalkBounds>::WALK_ROWS;

            type Arena = FlintArena;
            const STEP_ARENA: FlintArena = FlintArena::Step;
            const VERIFY_ARENA: FlintArena = FlintArena::Pair;

            /// The body's one batch path.
            type Path = ();

            /// The full vocabulary; refused by name on a load without the layer
            /// (`open_placed_slots_drafted` loads it), so the window never opens over it.
            fn head(m: &GpuModel<$ty>) -> Result<DraftHead, GpuError> {
                draft::draft_head(m)
            }

            fn hidden_width(m: &GpuModel<$ty>) -> Result<usize, GpuError> {
                draft::hidden_width(m)
            }

            fn held(m: &GpuModel<$ty>) -> Result<usize, GpuError> {
                draft::held(m)
            }

            fn retain_prompt_row(
                m: &mut GpuModel<$ty>,
                walk: FlintArena,
                first: usize,
                pos: u32,
            ) -> Result<(), GpuError> {
                draft::retain_prompt_row(m, walk, first, pos)
            }

            fn prompt_with(
                m: &mut GpuModel<$ty>,
                ids: &[u32],
                (): (),
                end: CallEnd,
                sink: Option<&mut UnitSink<'_, $ty>>,
            ) -> Result<Option<u32>, GpuError> {
                draft::prompt_with(m, ids, end, sink)
            }

            fn walk(
                m: &mut GpuModel<$ty>,
                feed: Feed<'_, FlintArena>,
                head: DraftHead,
                mode: WalkMode,
            ) -> Result<(), GpuError> {
                draft::walk(m, DraftFeed::Rows(feed), head, mode)
            }

            /// The head reads back its ids alone: `p` is refused by name.
            fn chain(
                m: &mut GpuModel<$ty>,
                refresh: Feed<'_, FlintArena>,
                own: usize,
                head: DraftHead,
                mode: WalkMode,
                out: &mut [u32],
                p: Option<&mut [f32]>,
            ) -> Result<usize, GpuError> {
                if p.is_some() {
                    return Err(GpuError::Shape {
                        what: "flint MTP",
                        detail: "a chain's probabilities: the MTP head reads back its ids alone"
                            .into(),
                    });
                }
                draft::chain(m, DraftFeed::Rows(refresh), own, head, mode, out)
            }
        }

        // The adapter's widths fit together: a verify of two rows, the pass's widest, is one walk.
        const _: () = <$ty as MtpBody>::FITS;
        const _: () = assert!(<$ty as MtpBody>::VERIFY_ROWS == <$ty as Rows>::MAX_ROWS);
    };
}

flint_mtp!(bloomery_gpu_flint::Body);
flint_mtp!(bloomery_gpu_flint::ExaoneBody);
