//! Qwen3.6's pass of several resident slots ([`SlotRows`],
//! `GpuModel::step_slots`): every busy slot's rows in one walk over the
//! pass arena. Each layer's row-wise launches — the norms, the projections,
//! a delta layer's gated norm, an attention layer's gated quantizer, the
//! router, the experts and the combine — run once over all `R` rows. The
//! launches bound to one sequence run once per busy slot, over that slot's
//! row window, its input record and its own stores: the embedding (its
//! first position and its ids), a delta layer's conv over its conv ring and
//! delta step over its state, and an attention layer's head-256 rope with
//! the cache append and the flash's segment pass and merge over its K/V
//! planes. Every slot's rows read one lane word, `LANE`: each slot's
//! recurrent stores are its own and hold one lane, so two slots never share
//! a lane, and two ranges of one slot are refused before any body code runs
//! (`GpuModel::step_slots`). Each launch is the one that slot's rows would
//! run alone, and a row's arithmetic in every launch does not depend on the
//! rows beside it (a pass of `m` rows leaves its `m` steps' bits,
//! `gate_qwen35moe_e2e` (p)), so every row is its slot's own step bit for
//! bit. The pass ends in one head of every row ([`Tail::One`]): the rows
//! read the lm_head once, each row's logits and token its one-row head's
//! (`Head`'s rows are independent).
//!
//! A pass takes no checkpoint: those are a prompt call's. A cut waiting on
//! a slot is carried out on that slot's stores before the pass's launches,
//! as its step's refresh carries it out ([`Body35::stand_held`]), so no
//! restore lands over rows a pass wrote. A pass writes its rows' states in
//! place into each slot's one lane, so it keeps its rows whole
//! ([`SlotRows::SETTLES_PARTIAL_KEEP`] stays false): keeping a part of them
//! would need the state after a kept row, which no lane holds.

use super::body::ATTN_SCALE_256;
use super::body35::{Body35, Slot35};
use super::delta;
use super::dispatch::{self, Ctx, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::plan::LayerPlan;
use super::plan::{DeltaPlan, GqaKind, GqaPlan, MixerPlan};
use super::program::{self, Tail};
use super::scratch::{Append256, Arena, FlashPass, Io, KvPlanes, LayerStore, RecStore};
use crate::flash_gqa::{partials_ms_len, partials_v_len_256};
use crate::gated_quant::GateLayout;
use crate::head::Head;
use crate::hybrid::Chain;
use crate::linear::{self, conv::ConvArgs, delta::DeltaArgs, norm_gate::NormGateArgs};
use crate::model::lookup::f32_gain;
use crate::model::{MAX_PASS_ROWS, RowHeads, SlotRange, SlotRows};
use crate::q38::OutGateArgs;
use crate::tensor::{Window, WindowMut};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use std::ops::Range;

/// What a pass of several slots' refusals name.
const WHAT: &str = "qwen35moe slots";

/// A delta layer's launches bound to one sequence: the conv over its ring
/// and the delta step over its state.
const DELTA_SEQ_LAUNCHES: usize = 2;
/// An attention layer's launches bound to one sequence: the rope with the
/// cache append, and the flash's segment pass and merge.
const GQA_SEQ_LAUNCHES: usize = 3;

impl SlotRows for Body35 {
    /// The pass arena's rows.
    const MAX_ROWS: usize = MAX_PASS_ROWS;

    /// One head of every row: the pass's rows read the lm_head once.
    const HEADS: RowHeads = RowHeads::One;

    /// Never read: a whole-card load serves no host work, and a placed load
    /// refuses the pass in [`SlotRows::plan_slots`].
    fn chain_of(_rows: &[SlotRange]) -> Chain {
        Chain::Step
    }

    /// Plan the pass of `rows` (the busy slots' ranges in pass order, any
    /// slot's first) over `ids` (every row's token), `parked`
    /// every busy slot's sequence but slot 0's — the live one — in `rows`
    /// order. Refused by name before anything moves: a placed load, the
    /// layer taps armed, ranges that do not tile the pass from row 0, a
    /// parked count other than the ranges past slot
    /// 0, an id count other than the rows, and for each slot on its own
    /// sequence stores that do not stand at its first position (a call
    /// there failed past its launch, [`Body35::stores_at`]) or an id past
    /// the vocabulary. Then each slot on its own sequence: its waiting cut
    /// carried out and its stores counted past its rows
    /// ([`Body35::stand_held`]); and the slots' input records written.
    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut Slot35],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        self.slots_planned = None;
        self.whole_card()?;
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT,
                "layer taps off (a pass of several slots writes none)",
            ));
        }
        let total = slot_rows(rows, parked.len())?;
        if ids.len() != total {
            return Err(GpuError::shape(
                WHAT,
                format!("{} ids for a pass of {total} rows", ids.len()),
            ));
        }
        for_slots(self, parked, rows, |b, r| {
            b.check_slot(r, slot_ids(ids, r)?)
        })?;
        for_slots(self, parked, rows, |b, r| {
            b.stand_held(stream, r.pos0, r.rows.len())
        })?;
        self.slot_in.write(stream, rows, ids)?;
        self.slots_planned = Some(rows.to_vec());
        Ok(())
    }

    /// Enqueue the pass of `rows` into `heads[0]`, one head of every row,
    /// `parked` as [`SlotRows::plan_slots`] takes it: slot 0's rows over the
    /// live stores, every other slot's over its parked sequence's. An eager
    /// pass needs the plan of these very ranges; a capture records the
    /// launches only. Refused by name: a placed load, ranges as the plan
    /// refuses them, more rows than the pass arena holds, any other heads.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Slot35],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        self.whole_card()?;
        let total = slot_rows(rows, parked.len())?;
        if total > self.a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {total} rows on a {}-row pass arena", self.a.rows),
            ));
        }
        if !crate::capturing(gpu.stream())? && self.slots_planned.as_deref() != Some(rows) {
            return Err(GpuError::state(
                WHAT,
                "each slot's rows planned (plan_slots before the pass)",
            ));
        }
        let [head] = heads else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} heads; a pass of several slots runs into one head of every row",
                    heads.len()
                ),
            ));
        };
        let Body35 {
            eps,
            plans,
            stores,
            rope,
            a,
            k,
            head_state,
            mma,
            slot_in,
            slot_lane,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: *eps,
            table: &rope.table,
        };
        let recs = rows
            .iter()
            .enumerate()
            .map(|(j, r)| slot_in.io(j, r))
            .collect::<Result<Vec<_>, GpuError>>()?;
        let mut live = Some(stores.as_mut_slice());
        let mut others = parked.iter_mut();
        let mut seqs = Vec::with_capacity(rows.len());
        for (r, rec) in rows.iter().zip(&recs) {
            let stores = if r.slot == 0 {
                live.take().ok_or_else(|| slot_twice(r))?
            } else {
                others
                    .next()
                    .ok_or_else(|| no_parked(r))?
                    .stores
                    .as_mut_slice()
            };
            seqs.push(SlotSeq35 {
                stores,
                io: Io {
                    lane: Some(&*slot_lane),
                    ..rec.io()
                },
                rows: r.rows.clone(),
            });
        }
        walk(&c, &mut seqs, a, total, (head, head_state))
    }
}

impl Body35 {
    /// The launches of the captured pass of several slots of `key` (each
    /// `(slot, rows)`, in pass order) without its head: the row-wise
    /// launches at every row of the pass with one embedding
    /// ([`Body35::pass_launches`]), and each busy slot past the first's
    /// sequence-bound launches ([`seq_launches`]). The head of the pass's
    /// rows adds the copy of the rows into its input and its own launches.
    #[must_use]
    pub fn slots_launches(&self, key: &[(usize, usize)]) -> usize {
        let rows = key.iter().map(|&(_, m)| m).sum();
        self.pass_launches(rows) + key.len().saturating_sub(1) * seq_launches(&self.plans)
    }

    /// Refused by name on a placed load: its prompt runs through the host
    /// tier's batch port and its step through the placed chain, one
    /// sequence each, and neither runs a pass of several slots.
    fn whole_card(&self) -> Result<(), GpuError> {
        if self.placed.is_some() {
            return Err(GpuError::shape(
                WHAT,
                "a pass of several slots on a placed load: its host tier's batch port and its \
                 placed chain run one sequence",
            ));
        }
        Ok(())
    }

    /// Refused by name unless the live sequence takes `tokens` from `r.pos0`
    /// as slot `r.slot`'s rows of a pass: its stores stand there, and every
    /// id is below the vocabulary.
    fn check_slot(&self, r: &SlotRange, tokens: &[u32]) -> Result<(), GpuError> {
        let named = |e: GpuError| GpuError::shape(WHAT, format!("slot {}: {e}", r.slot));
        self.stores_at(r.pos0).map_err(named)?;
        super::refuse_past_vocab(WHAT, tokens, self.vocab).map_err(named)
    }
}

/// The launches one more busy slot adds to a pass of `plans`: its
/// embedding, each delta layer's [`DELTA_SEQ_LAUNCHES`] and each attention
/// layer's [`GQA_SEQ_LAUNCHES`] — every launch that binds a sequence's
/// stores or its input record.
fn seq_launches(plans: &[LayerPlan]) -> usize {
    1 + plans
        .iter()
        .map(|p| match p.mixer {
            MixerPlan::Delta(_) => DELTA_SEQ_LAUNCHES,
            MixerPlan::Gqa(_) => GQA_SEQ_LAUNCHES,
        })
        .sum::<usize>()
}

/// One busy slot of a pass: its layers' stores, its input record (its
/// first position, its ids and the lane word) and the rows of the pass it
/// holds.
struct SlotSeq35<'a> {
    stores: &'a mut [LayerStore],
    io: Io<'a>,
    rows: Range<usize>,
}

/// Enqueue the pass (module doc): each slot's embedding into its rows, every
/// layer at the pass's `m` rows, then the one head of every row.
fn walk(
    c: &PassCtx<'_>,
    seqs: &mut [SlotSeq35<'_>],
    s: &mut Arena,
    m: usize,
    (head, state): (&mut Head, &mut HeadArgmaxState),
) -> Result<(), GpuError> {
    let hidden = s.dims.hidden;
    for q in seqs.iter() {
        let Arena { x, pos, n_keys, .. } = &mut *s;
        let mut x = rows_mut(x, &q.rows, hidden)?;
        let mut pos = rows_mut(pos, &q.rows, 1)?;
        let mut n_keys = rows_mut(n_keys, &q.rows, 1)?;
        dispatch::embed_into(c.gpu, c.w, c.k, &q.io, (&mut x, &mut pos, &mut n_keys))?;
    }
    for (l, p) in c.plans.iter().enumerate() {
        let lc = Ctx::new(c.gpu, c.w, (p, l), c.k, c.mma, c.eps, c.table)?;
        match &p.mixer {
            MixerPlan::Delta(d) => delta_layer(&lc, d, seqs, s, m)?,
            MixerPlan::Gqa(g) => gqa_layer(&lc, g, seqs, s, m)?,
        }
        dispatch::ffn(&lc, &p.ffn, s, m, None)?;
    }
    program::enqueue_tail(c, s, m, &mut Tail::One { head, state })
}

/// A delta layer's mixer at the pass's `m` rows (`delta`'s launches in its
/// order): the input projections over every row, each slot's conv and
/// delta step over its rows, its conv ring and its state at the slot's
/// lane word, then the gated norm and the output projection over every row.
fn delta_layer(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    seqs: &mut [SlotSeq35<'_>],
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, w) = (c.gpu, c.w);
    let lin = &c.k.q35(WHAT)?.linear;
    let stream = gpu.stream();
    match s.gdn.as_ref() {
        Some(g) if g.shape == d.shape => {}
        Some(g) => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {}: the arena is cut for {:?}, the plan runs {:?}",
                    c.layer, g.shape, d.shape
                ),
            ));
        }
        None => return Err(GpuError::state(WHAT, "the arena's delta intermediates")),
    }
    delta::project(c, d, s, m)?;
    let (ch, nv) = (d.shape.channels(), d.shape.n_v);
    let (wc, dt, sa) = (
        f32_gain(w, &d.conv)?,
        f32_gain(w, &d.dt_bias)?,
        f32_gain(w, &d.ssm_a)?,
    );
    let Arena { pos, attn, gdn, .. } = &mut *s;
    let g = gdn
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the arena's delta intermediates"))?;
    for q in seqs.iter_mut() {
        let r = &q.rows;
        let lane =
            q.io.lane
                .ok_or(GpuError::state(WHAT, "a slot's lane word"))?;
        let mut conv = rows_mut(&mut g.conv, r, ch)?;
        let mut beta = rows_mut(&mut g.beta, r, nv)?;
        let mut decay = rows_mut(&mut g.decay, r, nv)?;
        // The conv over the slot's ring, then the delta step over its
        // state: each launch binds the slot's own store.
        lin.conv.enqueue_conv_prep(
            stream,
            ConvArgs {
                x: &*rows(&g.x, r, ch)?,
                b_raw: &*rows(&g.b, r, nv)?,
                a_raw: &*rows(&g.a, r, nv)?,
                w: wc,
                dt_bias: dt,
                ssm_a: sa,
                pos: &*rows(pos, r, 1)?,
                shape: d.shape,
                eps: c.eps,
                m: r.len(),
                fault: c.sink,
                y: &mut conv,
                beta: &mut beta,
                decay: &mut decay,
                ring: &mut rec_of(&mut *q.stores, c.layer)?.ring,
            },
        )?;
        let st = rec_of(&mut *q.stores, c.layer)?;
        lin.delta.enqueue_delta(
            stream,
            DeltaArgs {
                qkv: &conv,
                beta: &beta,
                decay: &decay,
                lane,
                lane_at: 0,
                lanes: st.lanes,
                shape: d.shape,
                m: r.len(),
                fault: c.sink,
                o: &mut *rows_mut(&mut g.o, r, nv * linear::HEAD)?,
                state: &mut st.state,
            },
        )?;
    }
    lin.norm_gate.enqueue_norm_gate(
        stream,
        NormGateArgs {
            o: &g.o,
            z: &g.z,
            w: f32_gain(w, &d.ssm_norm)?,
            eps: c.eps,
            n_v: nv,
            m,
            fault: c.sink,
            y: attn,
        },
    )?;
    delta::output(c, d, s, m)
}

/// A gated head-256 attention layer's mixer at the pass's `m` rows (the
/// step's launches in its order): q, k and v over every row, each slot's
/// q/k norm, rope and cache append and its flash over its rows and its K/V
/// planes, then the output gate over every row — in the output
/// projection's quantizer for a K-quant, as f32 rows otherwise — and the
/// output projection. Any other attention is refused by name.
fn gqa_layer(
    c: &Ctx<'_>,
    n: &GqaPlan,
    seqs: &mut [SlotSeq35<'_>],
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    if n.kind != GqaKind::Gated256 {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {} is not gated attention at head 256, the one attention a pass of slots \
                 binds per slot",
                c.layer
            ),
        ));
    }
    dispatch::attn_in(c, n, s, m)?;
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let q35 = k.q35(WHAT)?;
    let stream = gpu.stream();
    let d = s.dims;
    let (gq, gk) = (f32_gain(w, &n.attn_q_norm)?, f32_gain(w, &n.attn_k_norm)?);
    {
        let Arena {
            q,
            k: kr,
            v,
            pos,
            n_keys,
            q_out,
            part_v,
            part_ms,
            attn,
            ..
        } = &mut *s;
        let q_out = q_out
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the arena's buffer of gated queries"))?;
        for qs in seqs.iter_mut() {
            let r = &qs.rows;
            let kv = kv_of(&mut *qs.stores, c.layer)?;
            let pos = rows(pos, r, 1)?;
            let mut qo = rows_mut(q_out, r, d.attn_len())?;
            kv.append_256(
                &k.neox,
                stream,
                Append256 {
                    qg: &*rows(q, r, d.q_rows)?,
                    q: &mut qo,
                    k: &mut *rows_mut(kr, r, d.kv_len())?,
                    v: &*rows(v, r, d.kv_len())?,
                    gq,
                    gk,
                    table: c.table,
                    pos: &pos,
                    eps: c.eps,
                    n_head: d.n_head,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m: r.len(),
                    fault: c.sink,
                },
            )?;
            kv.flash_256(
                &k.flash,
                stream,
                FlashPass {
                    q: &qo,
                    n_keys: &*rows(n_keys, r, 1)?,
                    scale: ATTN_SCALE_256,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m: r.len(),
                    part_v: &mut *rows_mut(part_v, r, partials_v_len_256(1, d.n_head))?,
                    part_ms: &mut *rows_mut(part_ms, r, partials_ms_len(1, d.n_head))?,
                    fault: c.sink,
                    y: &mut *rows_mut(attn, r, d.attn_len())?,
                },
                d.n_head,
                n.flash,
                c.mma,
            )?;
        }
    }
    let i = s.col(m)?;
    if n.o_ty.kquant() {
        q35.gated.enqueue_q8act(
            stream,
            (&s.attn, &s.q),
            GateLayout {
                head: d.head,
                head_stride: 2 * d.head,
                offset: d.head,
                col_stride: d.q_rows,
            },
            &mut s.act_attn[i],
            m,
            c.sink,
        )?;
    } else {
        let q_out = s
            .q_out
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the arena's buffer of gated queries"))?;
        q35.q38.enqueue_out_gate(
            stream,
            OutGateArgs {
                attn: &s.attn,
                qg: &s.q,
                n_head: d.n_head,
                m,
                fault: c.sink,
                y: q_out,
            },
        )?;
    }
    dispatch::attn_out(c, n, s, m)
}

/// Layer `l`'s recurrent store of a slot's `stores`, else refused by name.
fn rec_of(stores: &mut [LayerStore], l: usize) -> Result<&mut RecStore, GpuError> {
    match stores.get_mut(l) {
        Some(LayerStore::Rec(r)) => Ok(r),
        _ => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a delta layer's store of a slot"),
        )),
    }
}

/// Layer `l`'s K/V planes of a slot's `stores`, else refused by name.
fn kv_of(stores: &mut [LayerStore], l: usize) -> Result<&mut KvPlanes, GpuError> {
    match stores.get_mut(l) {
        Some(LayerStore::Kv(kv)) => Ok(kv),
        _ => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: an attention layer's planes of a slot"),
        )),
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

/// The pass's rows `R`, once `rows` tile `0..R` in order from row 0 with
/// no empty range, the slots in any order, and `parked` counts the ranges
/// of slots past slot 0; else refused by name.
fn slot_rows(rows: &[SlotRange], parked: usize) -> Result<usize, GpuError> {
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
    if at == 0 || parked != others {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a pass of {at} rows with {parked} parked sequences for {others} slots past slot 0"
            ),
        ));
    }
    Ok(at)
}

/// Slot `r.slot`'s ids of the pass's `ids`.
fn slot_ids<'i>(ids: &'i [u32], r: &SlotRange) -> Result<&'i [u32], GpuError> {
    ids.get(r.rows.clone()).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!("slot {}'s rows {:?} of {} ids", r.slot, r.rows, ids.len()),
        )
    })
}

/// Slot 0 named twice in one pass, refused.
fn slot_twice(r: &SlotRange) -> GpuError {
    GpuError::shape(WHAT, format!("slot {} twice in one pass", r.slot))
}

/// A busy slot past slot 0 with no parked sequence handed for it, refused.
fn no_parked(r: &SlotRange) -> GpuError {
    GpuError::shape(WHAT, format!("slot {}'s parked sequence", r.slot))
}

/// `f` on `b` for each range of `rows` in order, with that slot's sequence
/// live: slot 0's the live one, every other slot's the next of `parked`,
/// exchanged in ([`Body35::exchange`]) and back out whatever `f` returns.
fn for_slots(
    b: &mut Body35,
    parked: &mut [&mut Slot35],
    rows: &[SlotRange],
    mut f: impl FnMut(&mut Body35, &SlotRange) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let mut others = parked.iter_mut();
    for r in rows {
        if r.slot == 0 {
            f(b, r)?;
            continue;
        }
        let seq = &mut **others.next().ok_or_else(|| no_parked(r))?;
        b.exchange(seq);
        let out = f(b, r);
        b.exchange(seq);
        out?;
    }
    Ok(())
}
