//! The verify of two rows behind [`Rows`]: row 0 the current token at `pos`,
//! row 1 the drafted one at `pos + 1`, each through the one-token step's
//! launches in the step's order (`crate::step`'s walk), the two rows one layer
//! apart on the one stream (`runtime::sched`'s point `(2, 1, Step)`,
//! [`Chain::Pair`]) so the host serves one row's layer while the card runs
//! the other's front and shadow.
//!
//! Row 1's layers read what row 0 wrote at the same layer just before them:
//! the latent row an owner layer appends at `pos`, the index key a full
//! indexer layer stores at `pos`. Row 0 reads nothing row 1 writes: its live
//! rows end at `pos`, so neither its index scores nor its selected rows reach
//! the row row 1 appends at `pos + 1`. So each row's logits, every latent row
//! and index key, are bit for bit two steps in turn, and a rejected row 1 is
//! taken back by position as a cut is
//! ([`Rollback`](bloomery_gpu::model::Rollback): the next step at `pos + 1`
//! writes its rows before any launch reads them).
//!
//! What a row writes in one part and reads in a later one is its own: the
//! running row, the words the step's launches read (the position, the live
//! and selected counts, the walk's visible pair, the rope row and its bits),
//! each full indexer layer's list (a shared-indexer layer reads its source's
//! list after the other row's front has run), and the routed block's step
//! buffers, card sums and the host boundary's row (the common block's
//! `front_of`, `shadow_of` and `back_of`, armed by `Routed::arm_rows`). The
//! attention's other buffers are written and read inside one front, so the
//! rows share them. Row 0's own buffers are the step's; row 1's
//! ([`RowSet`]) are bound in their place for row 1's parts by pointer
//! exchanges alone, and the walk binds row 0's again before it returns,
//! whatever it returns.
//!
//! A pass of more rows has no chain: the step port's points are one, two and
//! four rows a column, and the drafts verify every width from 2 up to their
//! rows, so a load that held four would be asked for three. A pass refuses
//! the taps and a forced selection list by name (a gate's instruments of the
//! step).

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::model::{ChainBody, Rows};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};

use super::{Body, LayerStore, Scratch, Shape, words};

/// What the pass's refusals name.
const WHAT: &str = "mistrallarge4::Body (verify rows)";

/// The rows a load holds in flight at once: the step's and a verify's
/// drafted one. The host boundary, the expert tier's page and the routed
/// block's step buffers are made for this many rows.
pub const PASS_ROWS: usize = 2;

/// One row's own buffers past row 0's (the body's): the running row, the
/// step's words, and each full indexer layer's list in layer order.
struct RowBufs {
    x: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    ints: DeviceBuffer<u32>,
    vis: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    flash_n: DeviceBuffer<u32>,
    cs: DeviceBuffer<f32>,
    table_words: DeviceBuffer<u32>,
    lists: Vec<DeviceBuffer<u32>>,
}

impl RowBufs {
    /// The buffers of `sh`'s widths, a list for each of `full` layers: the
    /// list's width word written once, as the load writes row 0's.
    fn new(stream: &CudaStream, sh: &Shape, full: usize) -> Result<RowBufs, GpuError> {
        let mut ints = [0u32; 2];
        ints[words::TOP_K] = launch_u32(sh.top_k)?;
        Ok(RowBufs {
            x: DeviceBuffer::zeroed(stream, sh.hidden)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            ints: DeviceBuffer::from_host(stream, &ints)?,
            vis: DeviceBuffer::zeroed(stream, 2)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
            flash_n: DeviceBuffer::zeroed(stream, 1)?,
            cs: DeviceBuffer::zeroed(stream, sh.rope)?,
            table_words: DeviceBuffer::zeroed(stream, sh.rope)?,
            lists: (0..full)
                .map(|_| DeviceBuffer::zeroed(stream, sh.top_k))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn bytes(&self) -> usize {
        self.x.num_bytes()
            + self.pos.num_bytes()
            + self.ints.num_bytes()
            + self.vis.num_bytes()
            + self.n_keys.num_bytes()
            + self.flash_n.num_bytes()
            + self.cs.num_bytes()
            + self.table_words.num_bytes()
            + self
                .lists
                .iter()
                .map(DeviceBuffer::<u32>::num_bytes)
                .sum::<usize>()
    }

    /// Exchange these buffers with the ones the body's fields hold: pointer
    /// moves, no device work; a second exchange undoes the first.
    fn exchange(&mut self, s: &mut Scratch, x: &mut DeviceBuffer<f32>, stores: &mut [LayerStore]) {
        std::mem::swap(&mut self.x, x);
        std::mem::swap(&mut self.pos, &mut s.pos);
        std::mem::swap(&mut self.ints, &mut s.ints);
        std::mem::swap(&mut self.vis, &mut s.vis);
        std::mem::swap(&mut self.n_keys, &mut s.n_keys);
        std::mem::swap(&mut self.flash_n, &mut s.flash_n);
        std::mem::swap(&mut self.cs, &mut s.cs);
        std::mem::swap(&mut self.table_words, &mut s.table_words);
        let lists = stores.iter_mut().filter_map(|st| st.idx.as_mut());
        for (own, idx) in self.lists.iter_mut().zip(lists) {
            std::mem::swap(own, &mut idx.list);
        }
    }
}

/// The rows past row 0 a pass binds in row 0's place, and the row whose
/// buffers the body's fields hold now: 0 outside a pass.
pub(crate) struct RowSet {
    rows: Vec<RowBufs>,
    bound: usize,
}

impl RowSet {
    /// The buffers of rows 1 to [`PASS_ROWS`] − 1 at `sh`'s widths over
    /// `stores`' full indexer layers. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        sh: &Shape,
        stores: &[LayerStore],
    ) -> Result<RowSet, GpuError> {
        let full = stores.iter().filter(|st| st.idx.is_some()).count();
        Ok(RowSet {
            rows: (1..PASS_ROWS)
                .map(|_| RowBufs::new(stream, sh, full))
                .collect::<Result<Vec<_>, _>>()?,
            bound: 0,
        })
    }

    /// Device bytes of the rows' buffers: a term of the load's scratch.
    pub(super) fn bytes(&self) -> usize {
        self.rows.iter().map(RowBufs::bytes).sum()
    }

    /// Bind row `row`'s buffers in the body's fields (`s`, `x`, `stores`'
    /// lists): the row bound now exchanged back first, then `row`'s in; row
    /// 0's are the body's own. Pointer moves only. A row past the load's is
    /// refused by name, nothing moved.
    pub(crate) fn bind(
        &mut self,
        row: usize,
        s: &mut Scratch,
        x: &mut DeviceBuffer<f32>,
        stores: &mut [LayerStore],
    ) -> Result<(), GpuError> {
        if row == self.bound {
            return Ok(());
        }
        if row > self.rows.len() {
            return Err(GpuError::shape(
                WHAT,
                format!("row {row} of a load of {PASS_ROWS} rows in flight"),
            ));
        }
        if let Some(b) = self.bound.checked_sub(1) {
            self.rows[b].exchange(s, x, stores);
        }
        if let Some(r) = row.checked_sub(1) {
            self.rows[r].exchange(s, x, stores);
        }
        self.bound = row;
        Ok(())
    }
}

/// `n` as the u32 a device word holds, refused by name past it.
fn launch_u32(n: usize) -> Result<u32, GpuError> {
    u32::try_from(n).map_err(|_| GpuError::shape(WHAT, format!("{n} past a u32 word")))
}

impl Body {
    /// Bind row `row`'s buffers in the body's fields ([`RowSet::bind`]).
    fn bind_row(&mut self, row: usize) -> Result<(), GpuError> {
        self.pass_rows
            .bind(row, &mut self.s, &mut self.x, &mut self.stores)
    }

    /// The pass refuses the gate instruments by name: the taps copy the step's one row, and a
    /// forced selection list stands for the step's one pick.
    fn refuse_instruments(&self) -> Result<(), GpuError> {
        if self.taps.is_some() {
            return Err(GpuError::shape(
                WHAT,
                "a verify pass with the taps armed: they copy the step's one row (arm_taps off)",
            ));
        }
        if self.forced.is_some() {
            return Err(GpuError::shape(
                WHAT,
                "a verify pass with a forced selection list: it stands for the step's one pick \
                 (clear_forced_selection)",
            ));
        }
        Ok(())
    }
}

impl Rows for Body {
    const MAX_ROWS: usize = PASS_ROWS;
    const CHAIN: Chain = Chain::Pair;

    /// Each row's step input written into its own buffers: row `r`'s
    /// embedding of `tokens[r]` and its words at `pos + r`, the refresh the
    /// step runs, with row `r`'s buffers bound for it and row 0's bound again
    /// after. Refused by name: a pass of other than [`PASS_ROWS`] rows, the
    /// taps armed, a forced selection list standing, a row past the cache.
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.refuse_instruments()?;
        if tokens.len() != PASS_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!("{} rows; the pass runs {PASS_ROWS}", tokens.len()),
            ));
        }
        for (r, &token) in tokens.iter().enumerate() {
            let at = pos
                .checked_add(launch_u32(r)?)
                .ok_or_else(|| GpuError::shape(WHAT, format!("row {r} past position {pos}")))?;
            self.bind_row(r)?;
            let wrote = self
                .decode_input(token, at)
                .and_then(|input| self.refresh(stream, &input));
            let back = self.bind_row(0);
            wrote.and(back)?;
        }
        // The pass lays every row in the feature tap; each refresh planned one.
        if let Some(f) = self.feat.as_mut() {
            f.step.plan(PASS_ROWS);
        }
        Ok(())
    }

    /// The walk of the planned rows (`crate::step`'s, at two units), row `r`
    /// into `heads[r]`. Refused by name: other than [`PASS_ROWS`] heads, the
    /// taps armed, a forced selection list standing.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        self.refuse_instruments()?;
        if heads.len() != PASS_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!("{} heads; the pass runs {PASS_ROWS}", heads.len()),
            ));
        }
        crate::step::walk(gpu, w, self, heads)
    }
}
