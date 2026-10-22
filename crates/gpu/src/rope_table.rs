//! The rope table's card buffers, and the unfused rotation core of the NORM
//! mode kernels. The table itself — ggml's recipe as ik's CPU build compiles
//! it, computed once per position — is chain's ([`RopeTable`], re-exported
//! here under the path this crate's users read it at); a kernel reads
//! `[cos_0, sin_0, cos_1, …]` per token and turns pair `i` with table pair
//! `i`. Which values form pair `i` and how they turn is the kernel's:
//! adjacent `(2i, 2i + 1)` through [`rope_pair_rn`] in NORM mode, `(i, i +
//! n_dims/2)` through `rope_neox::neox_pair` in NEOX mode — the table is the
//! same. [`RopeRows`] is a table's rows for every cache position, on the
//! card.

use crate::GpuError;
pub use chain::rope::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
use cuda_core::{CudaStream, DeviceBuffer};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use std::time::{Duration, Instant};

/// Every cache position's rope row on the card, the table every path's rope
/// launch reads by position: row `p`, `width` f32 at `p · width` (the values
/// a head turns), holds [`RopeTable::push`]'s bits for position `p`, for each
/// `p` below the cache's rows — so the one check a position gets, below the
/// cache's rows, keeps the read inside the table. Built once at load.
pub struct RopeRows {
    /// The rows, `ctx` of `width` f32.
    pub table: DeviceBuffer<f32>,
    /// f32 a row: the turned values of a head.
    pub width: usize,
    /// The host time the rows took at load.
    pub build: Duration,
}

impl RopeRows {
    /// Rows `0..ctx` of `rope` (a `width`-wide spec, else refused), one
    /// `push` per position in order, as the host's f32 values: what [`RopeRows::new`]
    /// copies to the card and a sequence's rotation table starts from.
    pub fn host_rows(rope: &RopeTable, width: usize, ctx: usize) -> Result<Vec<f32>, GpuError> {
        const WHAT: &str = "RopeRows::host_rows";
        let positions = u32::try_from(ctx).map_err(|_| {
            GpuError::shape(
                WHAT,
                format!("a cache of {ctx} rows: positions and live key counts are u32"),
            )
        })?;
        if rope.n_dims() != width || width == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a rope table of {} values a position; a row is {width}",
                    rope.n_dims()
                ),
            ));
        }
        let mut host = Vec::with_capacity(ctx * width);
        for pos in 0..positions {
            rope.push(pos, Direction::Forward, &mut host);
        }
        Ok(host)
    }

    /// Rows `0..ctx` of `rope` ([`RopeRows::host_rows`]) copied to the card.
    /// Load-time only.
    pub fn new(
        stream: &CudaStream,
        rope: &RopeTable,
        width: usize,
        ctx: usize,
    ) -> Result<RopeRows, GpuError> {
        let t0 = Instant::now();
        let host = RopeRows::host_rows(rope, width, ctx)?;
        let build = t0.elapsed();
        Ok(RopeRows {
            table: DeviceBuffer::from_host(stream, &host)?,
            width,
            build,
        })
    }

    /// The positions the table holds: the cache's rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.table.len() / self.width
    }
}

// ------------------------------------------------------------------ cores

/// One pair turned by `(c, s)`:
/// `y0 = x0·c − x1·s`, `y1 = x0·s + x1·c`, each product and each sum rounded
/// on its own. ik's CPU rope rotates unfused; the compiler contracts a plain
/// `a*b − c*d` into an FMA, so every op here is an explicit round-to-nearest
/// intrinsic, which it never contracts. The inverse turn is this core on a
/// table with `s` negated.
#[inline(always)]
pub fn rope_pair_rn(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    let y0 = add_rn_f32(mul_rn_f32(x0, c), -mul_rn_f32(x1, s));
    let y1 = add_rn_f32(mul_rn_f32(x0, s), mul_rn_f32(x1, c));
    (y0, y1)
}
