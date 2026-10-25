//! A Muse Spark sequence on the host (the body's
//! [`PlaneSeq`](crate::checkpoint::planes::PlaneSeq), over the shared
//! `checkpoint::planes`): what a server's prompt cache holds of a sequence.
//!
//! Every store is a plane of positions: each layer's K and V, head-major, a
//! sliding layer's too (its planes span the whole context; the window is the
//! flash's, not the store's), so the sequence is those rows of the held
//! positions and nothing more, and any prefix of it is a valid cut. The
//! residency is the model's, not a sequence's: a state carries no slot map.

use super::Body;
use crate::GpuError;
use crate::checkpoint::planes::{PlaneBody, head_major};
use crate::checkpoint::saved::{Identity, Spans};

impl PlaneBody for Body {
    const WHAT: &'static str = "musespark sequence";

    fn who(&self) -> &Identity {
        &self.who
    }

    /// Each layer's K then V rows of the first `n` positions, in layer order.
    fn lend(&mut self, n: usize) -> Result<Vec<Spans<'_>>, GpuError> {
        let heads = (self.sh.n_kv, self.sh.ctx, self.sh.head);
        let mut spans = Vec::with_capacity(2 * self.layers.len());
        for layer in &mut self.layers {
            spans.push(head_major(&mut layer.kc, heads, n)?);
            spans.push(head_major(&mut layer.vc, heads, n)?);
        }
        Ok(spans)
    }
}
