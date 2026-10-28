//! qwen3moe's pass of several resident slots ([`SlotRows`], `GpuModel::step_slots`): every busy
//! slot's rows in one walk over the prompt arena. Each layer's row-wise launches — the norms, the
//! q·k·v and output projections, the router, the experts and the combine — run once over all `R`
//! rows. The embedding runs once per busy slot, over that slot's row window, from the slot's own
//! input record (its first position and its ids). Each layer's launches bound to the sequences'
//! K/V planes — the head norm and rope with the cache append, the flash's segment pass and its
//! merge — run once over all `R` rows when several slots are busy and the flash is the tensor-core
//! pass over an f16 cache: row `t` reads and writes the planes of its own slot through the per-row
//! table ([`RowTable`]), built from the slots' planes in pass order. A q8_0 cache, the scalar pass
//! (the eager measurement tool, `set_flash_mma`) and a pass of one slot run them once per busy slot
//! instead, over that slot's row window and its own planes: no per-row entry serves the first two.
//! Either way each is the launch the row would run with its slot alone, and a row's arithmetic in
//! every launch of the walk does not depend on the rows beside it (`flash_gqa`'s row contract; the
//! m-row launches against the steps, `gate_qwen3moe_e2e` (p)), so every row is its slot's own step
//! bit for bit. The pass then ends in one head of every row ([`Tail::One`]): the rows read the
//! lm_head once, each row's logits and token bit for bit its one-row head's (`Head`'s rows are
//! independent).
//!
//! [`SlotRows`]: crate::model::SlotRows

use super::body::ATTN_SCALE;
use super::dispatch::{self, Ctx, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::plan::{DeltaPlan, FfnPlan, GqaKind, GqaPlan};
use super::program::{self, Tail, layer_ctx};
use super::scratch::{
    Append128, Arena, FlashPass, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, put_input,
};
use super::wide::GEMV_COLS;
use crate::GpuError;
use crate::flash_gqa::{ROW_PLANES, RowTable, partials_ms_len, partials_v_len};
use crate::head::Head;
use crate::model::lookup::f32_gain;
use crate::model::{MAX_PASS_ROWS, SlotRange, SlotRowOrder};
use crate::tensor::{Window, WindowMut};
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::walk::Walk;
use cuda_core::{CudaStream, DeviceBuffer};
use std::ops::Range;

const WHAT: &str = "qwen3moe::slot_pass";

/// Words of the pass's input records: a pass of `R` rows over `n` slots
/// fills `R + n`, at most two a row.
const SLOT_IN_WORDS: usize = 2 * MAX_PASS_ROWS;

// The per-row table holds every row of a pass.
const _: () = assert!(MAX_PASS_ROWS <= ROW_PLANES);

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
            let slot_ids = r.ids(WHAT, ids)?;
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
/// `ins` their records, `heads` one head of every row.
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
    /// the one head of every row — the chain's walk over the pass's ops
    /// ([`SlotOps`]). Refused by name before any launch: ranges that do not
    /// tile `0..R` in order, `R` past the arena's gemv rows, a head count
    /// other than one or a parked count other than the ranges'; a layer
    /// other than head-128 attention is refused by name at its mixer.
    pub(super) fn enqueue(self) -> Result<(), GpuError> {
        let total = check(
            self.rows,
            self.s.rows.min(GEMV_COLS),
            self.heads.len(),
            self.parked.len(),
        )?;
        // The per-row launches when several slots are busy and they serve the
        // planes (the module doc): the load's one cache format, read from the
        // live planes.
        let per_row =
            self.rows.len() > 1 && self.live.first().is_some_and(|p| p.rows_serve(self.c.mma));
        let (plans, arena) = (self.c.plans, self.s.rows);
        let mut ops = SlotOps {
            pass: self,
            per_row,
        };
        Walk::new(&mut ops, plans, total).run(arena)
    }
}

/// A [`SlotPass`] as the ops the chain's walk orders: the row-wise launches
/// over all the pass's rows, the launches bound to the sequences' planes
/// per row through the slots' table when `per_row`, else per busy slot.
struct SlotOps<'a, 'p> {
    pass: SlotPass<'a, 'p>,
    per_row: bool,
}

/// The refusal of layer `l`, whose mixer is not the one a pass of slots
/// binds per slot.
fn not_neox(l: usize) -> GpuError {
    GpuError::shape(
        WHAT,
        format!(
            "layer {l} is not head-128 attention over K/V planes, the one mixer a pass of \
             slots binds per slot"
        ),
    )
}

impl ChainOps for SlotOps<'_, '_> {
    type Error = GpuError;

    /// Each busy slot's embedding over its row window, from the slot's own
    /// input record.
    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        let SlotPass {
            c, rows, ins, s, ..
        } = &mut self.pass;
        let hidden = s.dims.hidden;
        for (j, r) in rows.iter().enumerate() {
            let rec = ins.io(j, r)?;
            let Arena { x, pos, n_keys, .. } = &mut **s;
            let mut x = rows_mut(x, &r.rows, hidden)?;
            let mut pos = rows_mut(pos, &r.rows, 1)?;
            let mut n_keys = rows_mut(n_keys, &r.rows, 1)?;
            dispatch::embed_into(c.gpu, c.w, c.k, &rec.io(), (&mut x, &mut pos, &mut n_keys))?;
        }
        Ok(())
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        if g.kind != GqaKind::Neox128 {
            return Err(not_neox(l.layer));
        }
        let p = &mut self.pass;
        dispatch::attn_in(&layer_ctx(p.c, l)?, g, p.s, l.m)
    }

    /// The launches bound to the sequences' planes ([`attention`]), then the
    /// attention rows' q8_1 over all the rows.
    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        let SlotPass {
            c,
            live,
            parked,
            rows,
            s,
            ..
        } = &mut self.pass;
        let lc = layer_ctx(c, l)?;
        let slots = layer_planes(live, parked, rows, l.layer);
        if self.per_row {
            let mut table = RowTable::default();
            for slot in slots {
                let (r, kv) = slot?;
                kv.put_rows(&mut table, r.rows.clone())?;
            }
            attention(&lc, g, Planes::Rows(&table), s, &(0..l.m))?;
        } else {
            for slot in slots {
                let (r, kv) = slot?;
                attention(&lc, g, Planes::Slot(kv), s, &r.rows)?;
            }
        }
        let i = s.col(l.m)?;
        c.gpu
            .enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], l.layer)
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::SlotOps::attn_out")?;
        let p = &mut self.pass;
        dispatch::attn_out(&layer_ctx(p.c, l)?, g, p.s, l.m)
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), GpuError> {
        Err(not_neox(l.layer))
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), GpuError> {
        Err(not_neox(l.layer))
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), GpuError> {
        Err(not_neox(l.layer))
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), GpuError> {
        let p = &mut self.pass;
        dispatch::route(&layer_ctx(p.c, l)?, f, p.s, l.m)
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::SlotOps::experts")?;
        let p = &mut self.pass;
        dispatch::experts(&layer_ctx(p.c, l)?, f, p.s, l.m, None)
    }

    /// The one head of every row ([`Tail::One`]).
    fn head(&mut self, m: usize) -> Result<(), GpuError> {
        let SlotPass {
            c, s, heads, state, ..
        } = &mut self.pass;
        let [head] = &mut **heads else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} heads; a pass of several slots runs into one head of every row",
                    heads.len()
                ),
            ));
        };
        program::enqueue_tail(c, s, m, &mut Tail::One { head, state })
    }
}

/// The pass's rows `R`, once `rows` tile `0..R` in order with no empty
/// range, `R` is at most `most`, and the head is one with the parked
/// sequences as many as the ranges past slot 0; else refused by name.
fn check(rows: &[SlotRange], most: usize, heads: usize, parked: usize) -> Result<usize, GpuError> {
    let at = SlotRowOrder::Any.tiled_rows(WHAT, rows)?;
    let others = rows.iter().filter(|r| r.slot != 0).count();
    if at == 0 || at > most || heads != 1 || parked != others {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a pass of {at} rows (1..={most}) with {heads} heads and {parked} parked \
                 sequences for {others} slots past slot 0; the pass runs into one head of \
                 every row"
            ),
        ));
    }
    Ok(at)
}

/// Each busy slot of `rows` with its planes of layer `l`, in pass order: slot
/// 0's are the live ones, every other slot's the next parked sequence's; a
/// slot whose planes are missing is refused by name.
fn layer_planes<'a>(
    live: &'a mut [KvPlanes],
    parked: &'a mut [&mut [KvPlanes]],
    rows: &'a [SlotRange],
    l: usize,
) -> impl Iterator<Item = Result<(&'a SlotRange, &'a mut KvPlanes), GpuError>> {
    let mut live = Some(live);
    let mut others = parked.iter_mut();
    rows.iter().map(move |r| {
        let planes = if r.slot == 0 {
            live.take().and_then(|p| p.get_mut(l))
        } else {
            others.next().and_then(|p| p.get_mut(l))
        };
        planes
            .map(|kv| (r, kv))
            .ok_or_else(|| GpuError::shape(WHAT, format!("slot {}'s planes of layer {l}", r.slot)))
    })
}

/// The K/V planes a layer's sequence-bound launches read and write: one busy
/// slot's, or every busy slot's through the per-row table.
enum Planes<'k, 't> {
    Slot(&'k mut KvPlanes),
    Rows(&'k RowTable<'t>),
}

/// The sequence-bound attention of rows `r` of the arena over `planes`: the
/// head norm and rope with the cache append, then the flash — one slot's
/// launches over its row window alone, or the per-row launches over all the
/// pass's rows; either way each row's is the launch of its slot's rows
/// alone. The flash's partials are row-indexed (row `t`'s at `t · n_head ·
/// SEGMENTS` of `part_v`/`part_ms`, `flash_gqa`), so a slot's window of them
/// sits where a launch over all the rows puts them.
fn attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    mut planes: Planes<'_, '_>,
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
    let append = Append128 {
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
    };
    match &mut planes {
        // SAFETY: the sequence uploads consecutive positions for its rows; different sequences use disjoint planes.
        Planes::Slot(kv) => unsafe { kv.append_128(&c.k.neox, stream, append) }?,
        Planes::Rows(table) => {
            // SAFETY: the sequence uploads consecutive positions for its rows; different sequences use disjoint
            // planes.
            unsafe { KvPlanes::append_128_rows(&c.k.neox, stream, table, append) }?
        }
    }
    let n_keys = rows(n_keys, r, 1)?;
    let mut part_v = rows_mut(part_v, r, partials_v_len(1, d.n_head))?;
    let mut part_ms = rows_mut(part_ms, r, partials_ms_len(1, d.n_head))?;
    let mut y = rows_mut(attn, r, d.attn_len())?;
    let flash = FlashPass {
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
    };
    match planes {
        Planes::Slot(kv) => kv.flash_128(&c.k.flash, stream, flash, c.mma),
        Planes::Rows(table) => KvPlanes::flash_128_rows(&c.k.flash, stream, table, flash, c.mma),
    }
}

#[cfg(test)]
mod tests {
    use super::{SlotRange, WHAT, check};

    #[test]
    fn slot_pass_checks_arena_heads_and_parked_sequences_after_tiling() {
        let rows = [
            SlotRange {
                slot: 4,
                rows: 0..2,
                pos0: 8,
            },
            SlotRange {
                slot: 0,
                rows: 2..3,
                pos0: 5,
            },
        ];
        assert_eq!(check(&rows, 3, 1, 1).unwrap(), 3);
        for (most, heads, parked) in [(2, 1, 1), (3, 0, 1), (3, 2, 1), (3, 1, 0), (3, 1, 2)] {
            let error = check(&rows, most, heads, parked).unwrap_err();
            let crate::GpuError::Shape { what, detail } = error else {
                panic!("expected a slot pass shape refusal");
            };
            assert_eq!(what, WHAT);
            assert_eq!(
                detail,
                format!(
                    "a pass of 3 rows (1..={most}) with {heads} heads and {parked} parked \
                     sequences for 1 slots past slot 0; the pass runs into one head of every row"
                )
            );
        }
        assert!(
            check(&[], 3, 1, 0)
                .unwrap_err()
                .to_string()
                .contains("a pass of 0 rows")
        );
        let malformed = [SlotRange {
            slot: 4,
            rows: 1..2,
            pos0: 8,
        }];
        assert!(
            check(&malformed, 0, 0, 0)
                .unwrap_err()
                .to_string()
                .contains("the ranges tile")
        );
        assert!(
            check(
                &[SlotRange {
                    slot: 0,
                    rows: 0..usize::MAX,
                    pos0: 0
                }],
                3,
                1,
                0
            )
            .unwrap_err()
            .to_string()
            .contains(&format!("a pass of {} rows", usize::MAX))
        );
    }
}
