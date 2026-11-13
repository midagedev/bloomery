//! The verify of two rows behind [`Rows`]: row 0 the current token at `pos`, row 1 the drafted one at
//! `pos + 1`, each through the one-token step's launches in the step's order (`crate::step::walk`), the
//! two rows one layer apart on the one stream ([`Chain::Pair`]) so the host serves one row's layer while
//! the card runs the other's front and shadow.
//!
//! Row 1 reads what row 0 wrote at the same layer just before it (the key row its attention appends at
//! `pos`), and row 0 reads nothing row 1 writes, so each row's logits and every store row are what two
//! steps in turn write, bit for bit. A rejected row 1 is taken back by position as a cut is
//! ([`Rollback`](bloomery_gpu::model::Rollback)): the next step at `pos + 1` writes its rows before any
//! launch reads them, and a ring holds the window plus one prompt batch less one, so a cut of one
//! position leaves what the pass at `pos` reads.
//!
//! A pass of more rows has no chain: the step port's points are one and two rows a column, and the drafts
//! verify every width from 2 up to their rows. The pass refuses the taps by name (they copy the step's
//! one row).

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::model::{ChainBody, Rows};
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;

use crate::body::{FlintBody, PASS_ROWS};

impl<R: RouterEntry> Rows for FlintBody<R>
where
    FlintBody<R>: ChainBody,
{
    const MAX_ROWS: usize = PASS_ROWS;
    const CHAIN: Chain = Chain::Pair;

    /// Each row's step input written into its own buffers ([`FlintBody::plan_pass`]).
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.plan_pass(stream, tokens, pos)
    }

    /// The walk of the planned rows, row `r` into `heads[r]`.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        crate::step::walk(gpu, w, self, heads)
    }
}
