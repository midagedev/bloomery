//! A MiniMax-M3 sequence on the host ([`MiniMaxSeq`], over the shared
//! `bloomery_gpu::checkpoint::planes`): what a server's prompt cache holds of
//! a sequence, and puts back on the slot that resumes it.
//!
//! Every store is a plane of positions: each layer's K and V rows, and a
//! sparse layer's index key a position. The sequence is those rows of the held
//! positions and the written end, and any prefix of it is a valid cut. The
//! residency is the model's, not a sequence's: a state carries no slot map.
//! Several sequences park the planes ([`bloomery_gpu::Slots`]); a save and a
//! resume are of the selected slot, whose planes the select made live.

use bloomery_gpu::GpuError;
use bloomery_gpu::checkpoint::planes::{PlaneBody, PlaneSeq, head_major};
use bloomery_gpu::checkpoint::saved::{Identity, Spans};
use bloomery_gpu::flash_gqa::HEAD;

use crate::attn::DIM;
use crate::body::Body;

/// A MiniMax-M3 sequence state on the host
/// ([`bloomery_gpu::checkpoint::planes::seq_save`]).
pub type MiniMaxSeq = PlaneSeq<Body>;

impl PlaneBody for Body {
    const WHAT: &'static str = "minimaxm3 sequence";

    fn who(&self) -> &Identity {
        &self.who
    }

    /// Each layer's K, V and (on a sparse layer) index-key rows of the first
    /// `n` positions, in layer order.
    fn lend(&mut self, n: usize) -> Result<Vec<Spans<'_>>, GpuError> {
        let heads = (self.dims.kv_heads, self.dims.ctx, HEAD);
        let mut spans = Vec::with_capacity(3 * self.stores.len());
        for store in &mut self.stores {
            spans.push(head_major(&mut store.gqa.kc, heads, n)?);
            spans.push(head_major(&mut store.gqa.vc, heads, n)?);
            if let Some(index) = &mut store.index {
                spans.push(Spans::by_position(&mut index.keys, 1, 0, n, DIM)?);
            }
        }
        Ok(spans)
    }

    /// The positions the live sequence's stores hold.
    fn written(&self) -> Option<u32> {
        Some(self.at)
    }

    fn stand(&mut self, n: u32) {
        self.at = n;
    }
}
