//! qwen3moe's pass of several resident slots ([`SlotRows`], `GpuModel::step_slots`): every busy
//! slot's rows in one walk over the prompt arena. Each layer's row-wise launches — the norms, the
//! q·k·v and output projections, the router, the experts and the combine — run once over all `R`
//! rows; the launches bound to one sequence's stores run once per busy slot, over that slot's row
//! window of the arena and that slot's own K/V planes: the embedding from the slot's own input
//! record (its first position and its ids), and per layer the head norm and rope with the cache
//! append, and the flash's segment pass. Each of those is the launch the slot's rows would run
//! alone, and a row's arithmetic in every launch of the walk does not depend on the rows beside it
//! (`flash_gqa`'s row contract; the m-row launches against the steps, `gate_qwen3moe_e2e` (p)), so
//! every row is its slot's own step bit for bit. Each row then ends in its own head
//! ([`Tail::Rows`]).
//!
//! [`SlotRows`]: crate::model::SlotRows

use super::body::ATTN_SCALE;
use super::dispatch::{self, Ctx, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::plan::{GqaKind, GqaPlan, MixerPlan};
use super::program::{self, Tail};
use super::scratch::{
    Append128, Arena, FlashPass, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, put_input,
};
use super::wide::GEMV_COLS;
use crate::GpuError;
use crate::flash_gqa::{partials_ms_len, partials_v_len};
use crate::head::Head;
use crate::model::lookup::f32_gain;
use crate::model::{MAX_PASS_ROWS, SlotRange};
use crate::tensor::{Window, WindowMut};
use cuda_core::{CudaStream, DeviceBuffer};
use std::ops::Range;

const WHAT: &str = "qwen3moe::slot_pass";

/// Words of the pass's input records: a pass of `R` rows over `n` slots
/// fills `R + n`, at most two a row.
const SLOT_IN_WORDS: usize = 2 * MAX_PASS_ROWS;

/// The input records of a pass of several slots, packed in pass order: the
/// `j`-th busy slot's, whose rows start at pass row `r`, at word `r + j` —
/// its first position, then its ids ([`put_input`]'s record) — so the pass
/// fills `R + n` words, moved by one asynchronous copy. Allocated at load.
pub(super) struct SlotIn(Inbox);

impl SlotIn {
    /// Zeroed records. Load-time only.
    pub(super) fn new(stream: &CudaStream) -> Result<SlotIn, GpuError> {
        Ok(SlotIn(Inbox::new(stream, SLOT_IN_WORDS)?))
    }

    /// Write each busy slot's record — `rows[j].pos0`, then `ids[rows[j].rows]`
    /// — and enqueue their copy. Asynchronous: the pass behind it reads it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        let host = self.0.host_mut()?;
        let mut words = 0;
        for (j, r) in rows.iter().enumerate() {
            let at = record_at(j, r);
            let slot_ids = ids.get(r.rows.clone()).ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("slot {}'s rows {:?} of {} ids", r.slot, r.rows, ids.len()),
                )
            })?;
            let rec = host.get_mut(at..).ok_or_else(|| {
                GpuError::shape(WHAT, format!("a record at word {at} of {SLOT_IN_WORDS}"))
            })?;
            put_input(rec, slot_ids, r.pos0)?;
            words = at + IN_IDS + slot_ids.len();
        }
        self.0.upload(stream, words)
    }

    /// The `j`-th busy slot's record as its embedding reads it: its ids and
    /// its position word.
    pub(super) fn io(&self, j: usize, r: &SlotRange) -> Result<Record<'_>, GpuError> {
        let at = record_at(j, r);
        let word = size_of::<u32>();
        Ok(Record {
            ids: Window::of(self.0.dev(), (at + IN_IDS) * word, r.rows.len())?,
            pos0: Window::of(self.0.dev(), (at + IN_POS0) * word, 1)?,
        })
    }

    /// Device bytes of the records.
    pub(super) fn bytes(&self) -> usize {
        self.0.bytes()
    }
}

/// Where the `j`-th busy slot's record, of rows `r`, starts.
fn record_at(j: usize, r: &SlotRange) -> usize {
    r.rows.start + j
}

/// One busy slot's windows onto the records.
pub(super) struct Record<'a> {
    ids: Window<'a, u32>,
    pos0: Window<'a, u32>,
}

impl Record<'_> {
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            ids: &self.ids,
            pos0: &self.pos0,
            first: 0,
            lane: None,
        }
    }
}

/// Rows `r` of `parent`, `width` values a row, shared.
fn rows<'a, T>(
    parent: &'a DeviceBuffer<T>,
    r: &Range<usize>,
    width: usize,
) -> Result<Window<'a, T>, GpuError> {
    Window::of(parent, r.start * width * size_of::<T>(), r.len() * width)
}

/// Rows `r` of `parent`, `width` values a row, to write through.
fn rows_mut<'a, T>(
    parent: &'a mut DeviceBuffer<T>,
    r: &Range<usize>,
    width: usize,
) -> Result<WindowMut<'a, T>, GpuError> {
    WindowMut::of_mut(parent, r.start * width * size_of::<T>(), r.len() * width)
}

/// One pass of several slots over the prompt arena `s`: `rows` the busy
/// slots' ranges in pass order, `live` the live sequence's planes (slot 0's,
/// the homes canonical), `parked` every other busy slot's, in `rows` order;
/// `ins` their records, `heads` one a row.
pub(super) struct SlotPass<'a, 'p> {
    pub(super) c: &'a PassCtx<'a>,
    pub(super) live: &'a mut [KvPlanes],
    pub(super) parked: &'a mut [&'p mut [KvPlanes]],
    pub(super) rows: &'a [SlotRange],
    pub(super) ins: &'a SlotIn,
    pub(super) s: &'a mut Arena,
    pub(super) heads: &'a mut [Head],
    pub(super) state: &'a mut HeadArgmaxState,
}

impl SlotPass<'_, '_> {
    /// Enqueue the pass (module doc): each slot's embedding, every layer,
    /// each row's head. Refused by name before any launch: ranges that do
    /// not tile `0..R` in order, `R` past the arena's gemv rows, a head
    /// count or a parked count other than the ranges', a layer other than
    /// head-128 attention.
    pub(super) fn enqueue(self) -> Result<(), GpuError> {
        let SlotPass {
            c,
            live,
            parked,
            rows,
            ins,
            s,
            heads,
            state,
        } = self;
        let total = check(rows, s.rows.min(GEMV_COLS), heads.len(), parked.len())?;
        let hidden = s.dims.hidden;
        for (j, r) in rows.iter().enumerate() {
            let rec = ins.io(j, r)?;
            let Arena { x, pos, n_keys, .. } = &mut *s;
            let mut x = rows_mut(x, &r.rows, hidden)?;
            let mut pos = rows_mut(pos, &r.rows, 1)?;
            let mut n_keys = rows_mut(n_keys, &r.rows, 1)?;
            dispatch::embed_into(c.gpu, c.w, c.k, &rec.io(), (&mut x, &mut pos, &mut n_keys))?;
        }
        for (l, p) in c.plans.iter().enumerate() {
            let lc = Ctx::new(c.gpu, c.w, (p, l), c.k, c.mma, c.eps, c.table)?;
            let g = match &p.mixer {
                MixerPlan::Gqa(g) if g.kind == GqaKind::Neox128 => g,
                _ => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {l} is not head-128 attention over K/V planes, the one mixer \
                             a pass of slots binds per slot"
                        ),
                    ));
                }
            };
            dispatch::attn_in(&lc, g, s, total)?;
            let mut others = parked.iter_mut();
            for r in rows {
                let planes = if r.slot == 0 {
                    live.get_mut(l)
                } else {
                    others.next().and_then(|p| p.get_mut(l))
                };
                let kv = planes.ok_or_else(|| {
                    GpuError::shape(WHAT, format!("slot {}'s planes of layer {l}", r.slot))
                })?;
                slot_attention(&lc, g, kv, s, &r.rows)?;
            }
            let i = s.col(total)?;
            c.gpu
                .enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], l)?;
            dispatch::attn_out(&lc, g, s, total)?;
            dispatch::ffn(&lc, &p.ffn, s, total, None)?;
        }
        program::enqueue_tail(c, s, total, &mut Tail::Rows { heads, state })
    }
}

/// The pass's rows `R`, once `rows` tile `0..R` in order with no empty
/// range, `R` is at most `most`, and the heads and the parked sequences are
/// as many as the rows and the ranges past slot 0; else refused by name.
fn check(rows: &[SlotRange], most: usize, heads: usize, parked: usize) -> Result<usize, GpuError> {
    let mut at = 0;
    for r in rows {
        if r.rows.start != at || r.rows.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "slot {}'s rows {:?}; the ranges tile the pass in order from row 0",
                    r.slot, r.rows
                ),
            ));
        }
        at = r.rows.end;
    }
    let others = rows.iter().filter(|r| r.slot != 0).count();
    if at == 0 || at > most || heads != at || parked != others {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a pass of {at} rows (1..={most}) with {heads} heads and {parked} parked \
                 sequences for {others} slots past slot 0"
            ),
        ));
    }
    Ok(at)
}

/// One busy slot's sequence-bound attention over its planes `kv`, rows `r`
/// of the arena: the head norm and rope with the cache append, then the
/// flash — each over the slot's row window alone, so each is the launch of
/// those rows alone. The flash's partials are row-indexed (row `t`'s at
/// `t · n_head · SEGMENTS` of `part_v`/`part_ms`, `flash_gqa`), so the
/// slot's window of them sits where a pass of all the rows would put them.
fn slot_attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    r: &Range<usize>,
) -> Result<(), GpuError> {
    let d = s.dims;
    let m = r.len();
    let stream = c.gpu.stream();
    let Arena {
        q,
        k,
        v,
        pos,
        n_keys,
        part_v,
        part_ms,
        attn,
        ..
    } = s;
    let mut q = rows_mut(q, r, d.q_rows)?;
    let mut k = rows_mut(k, r, d.kv_len())?;
    let v = rows(v, r, d.kv_len())?;
    let pos = rows(pos, r, 1)?;
    kv.append_128(
        &c.k.neox,
        stream,
        Append128 {
            q: &mut q,
            k: &mut k,
            v: &v,
            gq: f32_gain(c.w, &n.attn_q_norm)?,
            gk: f32_gain(c.w, &n.attn_k_norm)?,
            table: c.table,
            pos: &pos,
            eps: c.eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            fault: c.sink,
        },
    )?;
    let n_keys = rows(n_keys, r, 1)?;
    let mut part_v = rows_mut(part_v, r, partials_v_len(1, d.n_head))?;
    let mut part_ms = rows_mut(part_ms, r, partials_ms_len(1, d.n_head))?;
    let mut y = rows_mut(attn, r, d.attn_len())?;
    kv.flash_128(
        &c.k.flash,
        stream,
        FlashPass {
            q: &q,
            n_keys: &n_keys,
            scale: ATTN_SCALE,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            part_v: &mut part_v,
            part_ms: &mut part_ms,
            fault: c.sink,
            y: &mut y,
        },
        c.mma,
    )
}
