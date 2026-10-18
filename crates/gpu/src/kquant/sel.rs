//! The family's expert-select entries: the down `_sel` (slot `s` dots the
//! rows of expert `sel[s]` with activation column `s`) and the gate·up with
//! its activation rule as a launch argument (slot `s` against column `s /
//! slots_per_col`), each one launch for all slots, the ids read from a
//! device buffer so a captured graph's replay follows whatever the router
//! last wrote. One warp a row, eight rows a 256-thread block; thread row
//! `n = slot · rows_per_expert + r` reads weight row `sel[slot] ·
//! rows_per_expert + r` and runs Walk A ([`super::walk::row_dot_1col`]) with
//! its format's decoder, reduced by the fixed warp tree.
//!
//! The bodies are generic over the decoder ([`gemv_sel_body`],
//! [`gate_up_act_body`], and the `_at` forms); [`crate::quant_sel`] writes
//! the entries for Q5_K and Q8_0 and the gate·up for Q4_K (whose down `_sel`
//! is `q4k_sel`'s `q4k_gemv_sel`), and this module holds their launchers on
//! `QuantSelKernels`. A caller outside the crate may instantiate the bodies for
//! another format in its own module (its entries then land in its own binary
//! alone).
//!
//! A non-routed FFN (a dense layer, a shared expert) is the one-expert
//! case: a stack of one expert, every id 0, one slot per token.
//!
//! Ids. [`HOST`] is a slot the host tier serves: both entries skip it,
//! write nothing and raise nothing. Any other id at or past the stack's
//! expert count raises [`FaultSite::ExpertId`] before the slot's first
//! weight load (warp-uniform); the down leaves the slot's rows of `y` as
//! they were, the gate·up writes NaN into its rows of `h`, so no stale row of
//! an earlier launch passes for its output.
//!
//! Activations. A q8_1 block the quantizer refused (a NaN or infinite value)
//! holds a NaN scale; the walk multiplies every term by its block scale, so
//! every row that reads the block is NaN.

use super::act::{self, Act};
use super::walk::{SbDecode, row_dot_1col};
use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, warp};

pub use crate::quant_sel::{ROWS_PER_BLOCK, THREADS};

/// The planes Walk A reads from a q8_1 activation: the q4 plane (`256 ·
/// ceil(n_sb/4)` words a column), the 32-value sums (`8 · n_sb` a column)
/// and the block scales (`2 · n_sb` a column).
#[must_use]
pub fn walk_a_planes(act: &Q8Act) -> (&DeviceBuffer<u32>, &DeviceBuffer<i32>, &DeviceBuffer<f32>) {
    (&act.q4, &act.s8, &act.d8)
}

/// The body of a down `_sel` entry over format `D` (module doc): thread row
/// `n = slot · rows_per_expert + r` stores `y[n]`, the dot of weight row
/// `sel[slot] · rows_per_expert + r` with activation column `slot`.
///
/// # Safety
///
/// The entry's launch contract over its `n_slots` slots: `w.len() >=
/// n_experts · rows_per_expert · D::WORDS · n_sb`, `q.len() >= n_slots · 256
/// · iters`, `s8.len() >= n_slots · 8 · n_sb`, `d8.len() >= n_slots · 2 ·
/// n_sb`, `sel.len() >= n_slots`, `y.len() >= n_slots · rows_per_expert`;
/// `iters = ceil(n_sb / 4)`; `row < n_slots · rows_per_expert` is the
/// calling warp's row (one warp a row), all 32 lanes of the warp here.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gemv_sel_body<D: SbDecode>(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    sel: &[u32],
    n_experts: u32,
    rows_per_expert: u32,
    n_sb: u32,
    iters: u32,
    fault: FaultSink,
    row: usize,
    y: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by this fn's contract; the load is
    // warp-uniform (the warp's 32 lanes share `row`, hence `slot`), so the
    // returns below never diverge a warp.
    let id = unsafe { *sel.get_unchecked(slot) };
    let lane = warp::lane_id() as usize;
    if id >= n_experts {
        if id != HOST && lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        return;
    }
    let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
    // SAFETY: row_abs < n_experts · rows_per_expert rows of `w`, column slot
    // < n_slots of the planes, iters = ceil(n_sb/4), and all 32 lanes of the
    // warp are here (both returns above are warp-uniform).
    let f0 = unsafe { row_dot_1col::<D>(w, q, s8, d8, n_sb as usize, iters, row_abs, slot, lane) };
    let s0 = warp::reduce_sum_f32(f0);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= y.len() by this fn's
        // contract; only lane 0 of the warp writes y[row].
        unsafe {
            *y.get_unchecked_mut(row) = s0;
        }
    }
}

/// The body of a gate·up entry over format `D` (module doc): thread row
/// `n = slot · rows_per_expert + r` dots weight row `sel[slot] ·
/// rows_per_expert + r` of `wg` and of `wu` with activation column `slot /
/// slots_per_col`, reduces each with the warp tree and stores `h[n] =
/// act::apply(act, limit, g, u)` from lane 0.
///
/// # Safety
///
/// The entry's launch contract over its `n_slots` slots: both stacks hold
/// `n_experts · rows_per_expert` rows of `D::WORDS · n_sb` words,
/// `slots_per_col >= 1`,
/// `n_slots <= m_cols · slots_per_col`, the planes hold `m_cols` columns,
/// `sel.len() >= n_slots`, `h.len() >= n_slots · rows_per_expert`; `iters =
/// ceil(n_sb / 4)`; `row < n_slots · rows_per_expert` is thread row `row`'s
/// (one warp a row), all 32 lanes of the warp here.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gate_up_act_body<D: SbDecode>(
    wg: &[u32],
    wu: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    sel: &[u32],
    n_experts: u32,
    rows_per_expert: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    row: usize,
    h: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by this fn's contract; the load is
    // warp-uniform, so the returns below never diverge a warp.
    let id = unsafe { *sel.get_unchecked(slot) };
    let lane = warp::lane_id() as usize;
    if id >= n_experts {
        if id != HOST && lane == 0 {
            fault.raise(FaultSite::ExpertId);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
            // contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = f32::NAN };
        }
        return;
    }
    let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
    let col = slot / slots_per_col as usize;
    // SAFETY: row_abs < n_experts · rows_per_expert rows of both stacks,
    // column col < m_cols because slot < n_slots <= m_cols · slots_per_col,
    // iters = ceil(n_sb/4), and all 32 lanes of the warp are here.
    let (fg, fu) = unsafe {
        (
            row_dot_1col::<D>(wg, q, s8, d8, n_sb as usize, iters, row_abs, col, lane),
            row_dot_1col::<D>(wu, q, s8, d8, n_sb as usize, iters, row_abs, col, lane),
        )
    };
    let g = warp::reduce_sum_f32(fg);
    let u = warp::reduce_sum_f32(fu);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
        // contract; only lane 0 of the warp writes h[row].
        unsafe { *h.get_unchecked_mut(row) = act::apply(act, limit, g, u) };
    }
}

/// An arena entry's per-slot table ([`GateUpAt`], [`SelDownAt`]): slot `s`'s
/// part `p` (gate 0, up 1, down 2) lies at word `offs[s · ARENA_PARTS + p]`
/// of the arena, the expert's rows of that stack as its file holds them.
pub const ARENA_PARTS: usize = 3;
/// The arena table's gate, up and down part of a slot.
pub const PART_GATE: usize = 0;
pub const PART_UP: usize = 1;
pub const PART_DOWN: usize = 2;
/// An arena table offset marking a slot the arena entries skip: they write
/// nothing and raise nothing for it.
pub const ARENA_SKIP: u64 = u64::MAX;

/// The `words` words of `arena` from word `off`, or `None` when they do not
/// all lie inside it.
#[inline(always)]
pub fn arena_part(arena: &[u32], off: u64, words: usize) -> Option<&[u32]> {
    let end = off.checked_add(words as u64)?;
    if end > arena.len() as u64 {
        return None;
    }
    // SAFETY: off <= end <= arena.len(), checked above.
    Some(unsafe { arena.get_unchecked(off as usize..end as usize) })
}

/// The body of a down arena entry over format `D`: [`gemv_sel_body`] with
/// slot `slot`'s expert read from its down part in `arena` (the slot's
/// [`PART_DOWN`] offset) instead of from a stack by id. Row `r` of the part
/// is the expert's row `r`, so the same bytes give the same bits. A slot
/// marked [`ARENA_SKIP`] is left alone; a part that does not lie inside the
/// arena raises [`FaultSite::ExpertId`] before any weight load and leaves
/// the slot's rows of `y` as they were.
///
/// # Safety
///
/// As [`gemv_sel_body`] with `offs.len() >= n_slots · ARENA_PARTS` in place
/// of the stack and id bounds.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gemv_at_body<D: SbDecode>(
    arena: &[u32],
    offs: &[u64],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    rows_per_expert: u32,
    n_sb: u32,
    iters: u32,
    fault: FaultSink,
    row: usize,
    y: &mut DisjointSlice<f32>,
) {
    let rpe = rows_per_expert as usize;
    let slot = row / rpe;
    // SAFETY: slot < n_slots, so the index is inside offs by this fn's
    // contract; the load is warp-uniform, so the returns below never
    // diverge a warp.
    let off = unsafe { *offs.get_unchecked(slot * ARENA_PARTS + PART_DOWN) };
    let lane = warp::lane_id() as usize;
    if off == ARENA_SKIP {
        return;
    }
    let Some(w) = arena_part(arena, off, rpe * D::WORDS * n_sb as usize) else {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        return;
    };
    // SAFETY: `w` is the part's rpe rows of D::WORDS · n_sb words and row %
    // rpe one of them; column slot < n_slots of the planes, iters =
    // ceil(n_sb/4), and all 32 lanes of the warp are here.
    let f0 =
        unsafe { row_dot_1col::<D>(w, q, s8, d8, n_sb as usize, iters, row % rpe, slot, lane) };
    let s0 = warp::reduce_sum_f32(f0);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= y.len() by this fn's
        // contract; only lane 0 of the warp writes y[row].
        unsafe {
            *y.get_unchecked_mut(row) = s0;
        }
    }
}

/// The body of a gate·up arena entry over format `D`: [`gate_up_act_body`]
/// with slot `slot`'s expert read from its gate and up parts in `arena`.
/// A slot whose gate offset is [`ARENA_SKIP`] is left alone; a gate or up
/// part that does not lie inside the arena raises [`FaultSite::ExpertId`]
/// and writes NaN into the slot's rows of `h`.
///
/// # Safety
///
/// As [`gate_up_act_body`] with `offs.len() >= n_slots · ARENA_PARTS` in
/// place of the stacks' and ids' bounds.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gate_up_act_at_body<D: SbDecode>(
    arena: &[u32],
    offs: &[u64],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    rows_per_expert: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    row: usize,
    h: &mut DisjointSlice<f32>,
) {
    let rpe = rows_per_expert as usize;
    let slot = row / rpe;
    // SAFETY: slot < n_slots, so both indices are inside offs by this fn's
    // contract; the loads are warp-uniform.
    let (og, ou) = unsafe {
        (
            *offs.get_unchecked(slot * ARENA_PARTS + PART_GATE),
            *offs.get_unchecked(slot * ARENA_PARTS + PART_UP),
        )
    };
    let lane = warp::lane_id() as usize;
    if og == ARENA_SKIP {
        return;
    }
    let words = rpe * D::WORDS * n_sb as usize;
    let (Some(wg), Some(wu)) = (arena_part(arena, og, words), arena_part(arena, ou, words)) else {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by this
            // fn's contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = f32::NAN };
        }
        return;
    };
    let (r, col) = (row % rpe, slot / slots_per_col as usize);
    // SAFETY: both parts hold rpe rows of D::WORDS · n_sb words and r is one
    // of them; column col < m_cols because slot < n_slots <= m_cols ·
    // slots_per_col; iters = ceil(n_sb/4); all 32 lanes of the warp are here.
    let (fg, fu) = unsafe {
        (
            row_dot_1col::<D>(wg, q, s8, d8, n_sb as usize, iters, r, col, lane),
            row_dot_1col::<D>(wu, q, s8, d8, n_sb as usize, iters, r, col, lane),
        )
    };
    let g = warp::reduce_sum_f32(fg);
    let u = warp::reduce_sum_f32(fu);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
        // contract; only lane 0 of the warp writes h[row].
        unsafe { *h.get_unchecked_mut(row) = act::apply(act, limit, g, u) };
    }
}

/// A down `_sel` launch (`QuantSelKernels::enqueue_gemv_q5k_sel`,
/// `QuantSelKernels::enqueue_gemv_q8_0_sel`).
pub struct SelDown<'a> {
    /// The resident stack: `n_experts · rows_per_expert` rows of `W · n_sb`
    /// words, `W` the entry's format's super-block words (Q5_K 44, Q8_0 68;
    /// `w.rows()` a positive multiple of `rows_per_expert`).
    pub w: &'a DeviceTensor<u32>,
    /// At least `n_slots` q8_1 columns: column `slot` is read for `slot <
    /// n_slots` and the rest are not touched, so a cap-sized activation serves
    /// a smaller batch.
    pub act: &'a Q8Act,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// A gate·up launch (`QuantSelKernels::enqueue_gate_up_q4k`,
/// `QuantSelKernels::enqueue_gate_up_q5k`,
/// `QuantSelKernels::enqueue_gate_up_q8_0`).
pub struct GateUpAct<'a> {
    /// The gate stack, as [`SelDown::w`] (Q4_K 36 words a super-block).
    pub wg: &'a DeviceTensor<u32>,
    /// The up stack, the same shape as `wg`.
    pub wu: &'a DeviceTensor<u32>,
    /// `act.m()` q8_1 columns; slot `s` reads column `s / slots_per_col`.
    pub act: &'a Q8Act,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    /// Slots a column: a token's run of slots.
    pub slots_per_col: usize,
    pub rule: Act,
}

/// A down arena launch (`QuantSelKernels::enqueue_gemv_q5k_sel_at`,
/// `Q6kSelKernels::enqueue_gemv_q6k_sel_at`): [`SelDown`] with each slot's
/// expert read from its down part in `arena` at the slot's [`PART_DOWN`]
/// offset of `offs` instead of from a stack by id.
pub struct SelDownAt<'a> {
    /// The arena as the card addresses it, in words.
    pub arena: &'a DeviceBuffer<u32>,
    /// [`ARENA_PARTS`] word offsets a slot, at least `n_slots` slots' worth,
    /// read on the device at each launch; [`ARENA_SKIP`] skips the slot.
    pub offs: &'a DeviceBuffer<u64>,
    /// As [`SelDown::act`].
    pub act: &'a Q8Act,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// A gate·up arena launch (`QuantSelKernels::enqueue_gate_up_q4k_at`,
/// `QuantSelKernels::enqueue_gate_up_q5k_at`): [`GateUpAct`] with each
/// slot's expert read from its gate and up parts in `arena`.
pub struct GateUpAt<'a> {
    /// The arena as the card addresses it, in words.
    pub arena: &'a DeviceBuffer<u32>,
    /// As [`SelDownAt::offs`]; a slot's [`PART_GATE`] offset marks the skip.
    pub offs: &'a DeviceBuffer<u64>,
    /// As [`GateUpAct::act`].
    pub act: &'a Q8Act,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    pub slots_per_col: usize,
    pub rule: Act,
}

/// A format's name and super-block words, as its launchers check a stack.
#[derive(Clone, Copy)]
pub(crate) struct Fmt {
    pub(crate) name: &'static str,
    pub(crate) words: usize,
}

/// The experts of a stack of `fmt.words · n_sb`-word rows, or the shape
/// error naming `what`.
fn stack_experts(
    what: &'static str,
    fmt: Fmt,
    w: &DeviceTensor<u32>,
    n_sb: usize,
    rows_per_expert: usize,
) -> Result<usize, GpuError> {
    if w.cols() != fmt.words * n_sb {
        return Err(GpuError::shape(
            what,
            format!(
                "{} rows are {}*{n_sb} = {} words, got {}",
                fmt.name,
                fmt.words,
                fmt.words * n_sb,
                w.cols()
            ),
        ));
    }
    if rows_per_expert == 0 || w.rows() == 0 || !w.rows().is_multiple_of(rows_per_expert) {
        return Err(GpuError::shape(
            what,
            format!(
                "w.rows() {} must be a positive multiple of rows_per_expert {rows_per_expert}",
                w.rows()
            ),
        ));
    }
    Ok(w.rows() / rows_per_expert)
}

/// A checked down `_sel` launch's grid and scalars, the entry's argument
/// order.
pub(crate) struct DownDims {
    pub(crate) grid: u32,
    pub(crate) n_experts: u32,
    pub(crate) rows_per_expert: u32,
    pub(crate) n_slots: u32,
    pub(crate) n_sb: u32,
    pub(crate) iters: u32,
}

/// [`SelDown`] checked against `fmt` and an output of `y_len` values.
pub(crate) fn down_dims(
    what: &'static str,
    fmt: Fmt,
    a: &SelDown<'_>,
    y_len: usize,
) -> Result<DownDims, GpuError> {
    let n_sb = a.act.n_sb();
    let n_experts = stack_experts(what, fmt, a.w, n_sb, a.rows_per_expert)?;
    let (n_slots, rpe) = (a.n_slots, a.rows_per_expert);
    if n_slots == 0 || a.act.m() < n_slots || a.sel.len() < n_slots || y_len < n_slots * rpe {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots (at least one): an activation column a slot (act.m() {} >= \
                 n_slots), an id a slot (sel.len() {}), y.len() {y_len} >= n_slots*rows_per_expert = {}",
                a.act.m(),
                a.sel.len(),
                n_slots * rpe
            ),
        ));
    }
    Ok(DownDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        n_experts: launch_u32(what, "n_experts", n_experts)?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
    })
}

/// A checked gate·up launch's grid and scalars, the entry's argument order.
pub(crate) struct GateUpDims {
    pub(crate) grid: u32,
    pub(crate) n_experts: u32,
    pub(crate) rows_per_expert: u32,
    pub(crate) n_slots: u32,
    pub(crate) m_cols: u32,
    pub(crate) slots_per_col: u32,
    pub(crate) n_sb: u32,
    pub(crate) iters: u32,
    pub(crate) act: u32,
    pub(crate) limit: f32,
}

/// [`GateUpAct`] checked against `fmt` and an output of `h_len` values.
pub(crate) fn gate_up_dims(
    what: &'static str,
    fmt: Fmt,
    a: &GateUpAct<'_>,
    h_len: usize,
) -> Result<GateUpDims, GpuError> {
    let n_sb = a.act.n_sb();
    let n_experts = stack_experts(what, fmt, a.wg, n_sb, a.rows_per_expert)?;
    let (n_slots, rpe, spc, m) = (a.n_slots, a.rows_per_expert, a.slots_per_col, a.act.m());
    if a.wu.rows() != a.wg.rows() || a.wu.cols() != a.wg.cols() {
        return Err(GpuError::shape(
            what,
            format!(
                "the up stack {} x {} must be the gate stack's {} x {}",
                a.wu.rows(),
                a.wu.cols(),
                a.wg.rows(),
                a.wg.cols()
            ),
        ));
    }
    if n_slots == 0
        || spc == 0
        || n_slots > m * spc
        || a.sel.len() < n_slots
        || h_len < n_slots * rpe
    {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots (at least one) at {spc} a column (at least one) over {m} \
                 columns, sel.len() {}, h.len() {h_len} >= n_slots*rows_per_expert = {}",
                a.sel.len(),
                n_slots * rpe
            ),
        ));
    }
    let (act, limit) = a.rule.code();
    Ok(GateUpDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        n_experts: launch_u32(what, "n_experts", n_experts)?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        m_cols: launch_u32(what, "m_cols", m)?,
        slots_per_col: launch_u32(what, "slots_per_col", spc)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
        act,
        limit,
    })
}

/// A checked down arena launch's grid and scalars, the entry's argument
/// order.
pub(crate) struct DownAtDims {
    pub(crate) grid: u32,
    pub(crate) rows_per_expert: u32,
    pub(crate) n_slots: u32,
    pub(crate) n_sb: u32,
    pub(crate) iters: u32,
}

/// A checked gate·up arena launch's grid and scalars, the entry's argument
/// order.
pub(crate) struct GateUpAtDims {
    pub(crate) grid: u32,
    pub(crate) rows_per_expert: u32,
    pub(crate) n_slots: u32,
    pub(crate) m_cols: u32,
    pub(crate) slots_per_col: u32,
    pub(crate) n_sb: u32,
    pub(crate) iters: u32,
    pub(crate) act: u32,
    pub(crate) limit: f32,
}

/// The slot checks the arena launches share: at least one slot, an
/// offset table of `n_slots` slots, a nonempty arena and positive rows a
/// part.
pub(crate) fn arena_slots(
    what: &'static str,
    arena: &DeviceBuffer<u32>,
    offs: &DeviceBuffer<u64>,
    n_slots: usize,
    rows_per_expert: usize,
) -> Result<(), GpuError> {
    if n_slots == 0
        || offs.len() < n_slots * ARENA_PARTS
        || arena.is_empty()
        || rows_per_expert == 0
    {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots (at least one): offs.len() {} >= n_slots*{ARENA_PARTS}, an \
                 arena of {} words (at least one), rows_per_expert {rows_per_expert} (at least \
                 one)",
                offs.len(),
                arena.len()
            ),
        ));
    }
    Ok(())
}

/// [`SelDownAt`] checked as [`down_dims`] checks a stack's launch, for an
/// output of `y_len` values.
pub(crate) fn down_at_dims(
    what: &'static str,
    a: &SelDownAt<'_>,
    y_len: usize,
) -> Result<DownAtDims, GpuError> {
    let (n_slots, rpe, n_sb) = (a.n_slots, a.rows_per_expert, a.act.n_sb());
    arena_slots(what, a.arena, a.offs, n_slots, rpe)?;
    if a.act.m() < n_slots || y_len < n_slots * rpe {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots: an activation column a slot (act.m() {}), y.len() {y_len} >= \
                 n_slots*rows_per_expert = {}",
                a.act.m(),
                n_slots * rpe
            ),
        ));
    }
    Ok(DownAtDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
    })
}

/// [`GateUpAt`] checked as [`gate_up_dims`] checks a stack's launch, for an
/// output of `h_len` values.
pub(crate) fn gate_up_at_dims(
    what: &'static str,
    a: &GateUpAt<'_>,
    h_len: usize,
) -> Result<GateUpAtDims, GpuError> {
    let (n_slots, rpe, spc, m, n_sb) = (
        a.n_slots,
        a.rows_per_expert,
        a.slots_per_col,
        a.act.m(),
        a.act.n_sb(),
    );
    arena_slots(what, a.arena, a.offs, n_slots, rpe)?;
    if spc == 0 || n_slots > m * spc || h_len < n_slots * rpe {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots at {spc} a column (at least one) over {m} columns, h.len() \
                 {h_len} >= n_slots*rows_per_expert = {}",
                n_slots * rpe
            ),
        ));
    }
    let (act, limit) = a.rule.code();
    Ok(GateUpAtDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        m_cols: launch_u32(what, "m_cols", m)?,
        slots_per_col: launch_u32(what, "slots_per_col", spc)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
        act,
        limit,
    })
}
