//! Qwen3.6's pass of several resident slots ([`SlotRows`],
//! `GpuModel::step_slots`): every busy slot's rows in one walk over the
//! pass arena. Each layer's row-wise launches — the norms, the projections,
//! a delta layer's gated norm, an attention layer's gated quantizer, the
//! router, the experts and the combine — run once over all `R` rows. A
//! delta layer's conv over each slot's conv ring and delta step over its
//! state run once over every slot's rows too, each slot's rows over its own
//! store (`linear::seqs`). The other launches bound to one sequence run once
//! per busy slot, over that slot's row window, its input record and its own
//! stores: the embedding (its first position and its ids) and an attention
//! layer's rope with the cache append and the flash's segment pass and merge
//! over its K/V planes, the rope reading the slot's own rotation table (a
//! media load's images shift one sequence's rows and no other's); a
//! head-256 layer's rope and append on f16 planes run once over every
//! slot's rows, each over its own planes and table. Every slot's rows read
//! one lane word, `LANE`: each slot's recurrent stores are its own and hold
//! one lane, so two slots never share
//! a lane, and two ranges of one slot are refused before any body code runs
//! (`GpuModel::step_slots`). Each launch is the one that slot's rows would
//! run alone, and a row's arithmetic in every launch does not depend on the
//! rows beside it (a pass of `m` rows leaves its `m` steps' bits,
//! `gate_qwen35moe_e2e` (p)), so every row is its slot's own step bit for
//! bit. A placed load runs that FFN as one pass of the rows through the
//! host tier's step port ([`ColsWalk`]): each column's handoff, one go,
//! the card leg over the rows. The pass ends in one head of every row
//! ([`Tail::One`]): the rows
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

use super::body::{ATTN_SCALE, ATTN_SCALE_256, KvQ8};
use super::body35::{Body35, Slot35};
use super::delta;
use super::dispatch::{self, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::image::MediaCarrier;
use super::placed::{ColsMix, ColsWalk, Placed, slot_chain};
use super::plan::LayerPlan;
use super::plan::{DeltaPlan, FfnPlan, GqaKind, GqaPlan, MixerPlan, Pairs};
use super::program::{self, Tail, layer_ctx};
use super::scratch::{Append256, AppendNorm, Arena, FlashPass, Io, KvPlanes, LayerStore, RecStore};
use crate::flash_gqa::{partials_ms_len, partials_v_len, partials_v_len_256};
use crate::gated_quant::GateLayout;
use crate::head::Head;
use crate::host::StepLeg;
use crate::hybrid::Chain;
use crate::linear::seqs::{GdnSeq, GdnSeqsArgs};
use crate::linear::{self, norm_gate::NormGateArgs};
use crate::media::rope_table;
use crate::model::lookup::f32_gain;
use crate::model::slots::slot_twice;
use crate::model::{MAX_PASS_ROWS, RowHeads, SlotRange, SlotRowOrder, SlotRows};
use crate::q38::OutGateArgs;
use crate::rope_neox::PartialNeoxSeqsArgs;
use crate::tensor::{Window, WindowMut};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::walk::{self, Walk};
use cuda_core::{CudaStream, DeviceBuffer};
use std::ops::Range;

/// What a pass of several slots' refusals name.
const WHAT: &str = "qwen35moe slots";

/// An attention layer's launches bound to one sequence: the rope with the
/// cache append, and the flash's segment pass and merge. A head-256 layer's
/// append on f16 planes runs once over every sequence's rows instead
/// ([`seq_launches`]).
const GQA_SEQ_LAUNCHES: usize = 3;

impl SlotRows for Body35 {
    /// The pass arena's rows.
    const MAX_ROWS: usize = MAX_PASS_ROWS;

    /// One head of every row: the pass's rows read the lm_head once.
    const HEADS: RowHeads = RowHeads::One;

    /// The chain the placed replay serves ([`slot_chain`]). A whole-card
    /// load has no host service of the pass.
    fn chain_of(rows: &[SlotRange]) -> Chain {
        slot_chain(rows)
    }

    /// Plan the pass of `rows` (the busy slots' ranges in pass order, any
    /// slot's first) over `ids` (every row's token), `parked`
    /// every busy slot's sequence but slot 0's — the live one — in `rows`
    /// order. Refused by name before anything moves: a load that verifies
    /// drafted rows, the
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
        self.refuse_verify()?;
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT,
                "layer taps off (a pass of several slots writes none)",
            ));
        }
        if self.draft.is_some() {
            return Err(GpuError::state(
                WHAT,
                "draft taps off (a pass of several slots lays none in)",
            ));
        }
        let total = SlotRowOrder::Any.admit(WHAT, rows, parked.len())?;
        if ids.len() != total {
            return Err(GpuError::shape(
                WHAT,
                format!("{} ids for a pass of {total} rows", ids.len()),
            ));
        }
        for_slots(self, parked, rows, |b, r| {
            b.check_slot(r, r.ids(WHAT, ids)?)
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
    /// launches only. A placed load runs the FFN as one pass of the rows
    /// ([`walk_placed`]). Refused by name: a load that verifies drafted rows,
    /// ranges as the plan refuses them, more rows than the pass arena
    /// holds, any other heads.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Slot35],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        self.refuse_verify()?;
        let total = SlotRowOrder::Any.admit(WHAT, rows, parked.len())?;
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
            seq_rope,
            a,
            k,
            head_state,
            mma,
            slot_in,
            slot_lane,
            placed,
            ..
        } = self;
        let live = rope_table(rope, seq_rope.as_ref());
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: *eps,
            table: live,
        };
        let recs = rows
            .iter()
            .enumerate()
            .map(|(j, r)| slot_in.io(j, r))
            .collect::<Result<Vec<_>, GpuError>>()?;
        let mut live_stores = Some(stores.as_mut_slice());
        let mut others = parked.iter_mut();
        let mut seqs = Vec::with_capacity(rows.len());
        for (r, rec) in rows.iter().zip(&recs) {
            let (stores, table) = if r.slot == 0 {
                (
                    live_stores.take().ok_or_else(|| slot_twice(WHAT, r.slot))?,
                    live,
                )
            } else {
                let p = others.next().ok_or_else(|| r.no_parked(WHAT))?;
                (p.stores.as_mut_slice(), rope_table(rope, p.rope.as_ref()))
            };
            seqs.push(SlotSeq35 {
                stores,
                table,
                io: Io {
                    lane: Some(&*slot_lane),
                    ..rec.io()
                },
                rows: r.rows.clone(),
            });
        }
        let Some(placed) = placed.as_mut() else {
            let rows = a.rows;
            let mut ops = Slots35 {
                c: &c,
                seqs: &mut seqs,
                s: a,
                head: Some((head, head_state)),
            };
            return Walk::new(&mut ops, c.plans, total).run(rows);
        };
        walk_placed(&c, &mut seqs, a, total, (head, head_state), placed)
    }
}

impl Body35 {
    /// The launches of the captured whole-card pass of several slots of
    /// `key` (each `(slot, rows)`, in pass order) without its head: the
    /// row-wise launches at every row of the pass with one embedding
    /// ([`Body35::pass_launches`]), and each busy slot past the first's
    /// sequence-bound launches ([`seq_launches`]). The head of the pass's
    /// rows adds the copy of the rows into its input and its own launches.
    /// A placed pass's handoff, go and card leg are not in this count.
    #[must_use]
    pub fn slots_launches(&self, key: &[(usize, usize)]) -> usize {
        let rows = key.iter().map(|&(_, m)| m).sum();
        self.pass_launches(rows)
            + key.len().saturating_sub(1) * seq_launches(&self.plans, self.kv_format())
    }

    /// The format the attention layers' planes hold: the load's one choice,
    /// read back from the live planes (f16 on a load of none).
    fn kv_format(&self) -> KvQ8 {
        self.stores
            .iter()
            .find_map(|s| match s {
                LayerStore::Kv(kv) => Some(kv.format()),
                LayerStore::Rec(_) => None,
            })
            .unwrap_or_default()
    }

    /// Refused by name on a load that verifies drafted rows, whose delta
    /// stores keep their lanes for one sequence's verify.
    fn refuse_verify(&self) -> Result<(), GpuError> {
        if self.lanes() > 1 {
            return Err(GpuError::shape(
                WHAT,
                "a pass of several slots on a load that verifies drafted rows: its delta stores \
                 keep their lanes for one sequence's verify",
            ));
        }
        Ok(())
    }

    /// Refused by name unless the live sequence takes `tokens` from `r.pos0`
    /// as slot `r.slot`'s rows of a pass: its stores stand there, the
    /// position is not inside an image its table holds
    /// ([`MediaCarrier::refuse_inside_image`]), and every id is below the
    /// vocabulary.
    fn check_slot(&self, r: &SlotRange, tokens: &[u32]) -> Result<(), GpuError> {
        let named = |e: GpuError| GpuError::shape(WHAT, format!("slot {}: {e}", r.slot));
        self.stores_at(r.pos0).map_err(named)?;
        self.refuse_inside_image(WHAT, r.pos0).map_err(named)?;
        super::refuse_past_vocab(WHAT, tokens, self.vocab).map_err(named)
    }
}

/// The launches one more busy slot adds to a pass of `plans` over planes of
/// format `kv`: its embedding and each attention layer's
/// [`GQA_SEQ_LAUNCHES`], less a head-256 layer's append on f16 planes —
/// every launch that binds a sequence's stores or its input record and runs
/// once a sequence. A delta layer's conv and step and a head-256 layer's
/// append on f16 planes run once over every sequence's rows
/// (`linear::seqs`, `KvPlanes::kv_seq`).
fn seq_launches(plans: &[LayerPlan], kv: KvQ8) -> usize {
    1 + plans
        .iter()
        .map(|p| match &p.mixer {
            MixerPlan::Delta(_) => 0,
            MixerPlan::Gqa(n) if n.kind == GqaKind::Gated256 && kv == KvQ8::F16 => {
                GQA_SEQ_LAUNCHES - 1
            }
            MixerPlan::Gqa(_) => GQA_SEQ_LAUNCHES,
            MixerPlan::ShortConv(_) => {
                panic!("qwen35moe::seq_launches: a short-conv layer, which `kinds35` refuses")
            }
        })
        .sum::<usize>()
}

/// One busy slot of a pass: its layers' stores, the rotation table its rope
/// launches read, its input record (its first position, its ids and the
/// lane word) and the rows of the pass it holds.
struct SlotSeq35<'a> {
    stores: &'a mut [LayerStore],
    table: &'a DeviceBuffer<f32>,
    io: Io<'a>,
    rows: Range<usize>,
}

/// The pass (module doc) as the ops the chain's walk orders: the row-wise
/// launches over all `R` rows of the arena `s`, the launches bound to a
/// sequence per busy slot of `seqs`, the one head of every row last
/// (`head`; a placed pass's head is its walk's, [`walk_placed`]).
struct Slots35<'a, 'q> {
    c: &'a PassCtx<'a>,
    seqs: &'a mut [SlotSeq35<'q>],
    s: &'a mut Arena,
    head: Option<(&'a mut Head, &'a mut HeadArgmaxState)>,
}

/// A placed load's pass (module doc): each layer's front through the
/// slots' ops ([`Mix35`]: the slots' embedding ahead of layer 0, then the
/// sequence-bound launches) and then that layer's FFN as [`ColsWalk`],
/// whose tail is the one head of every row.
fn walk_placed(
    c: &PassCtx<'_>,
    seqs: &mut [SlotSeq35<'_>],
    s: &mut Arena,
    m: usize,
    (head, state): (&mut Head, &mut HeadArgmaxState),
    Placed {
        hybrid, side, pass, ..
    }: &mut Placed,
) -> Result<(), GpuError> {
    let mut leg = StepLeg::new(c.gpu.stream(), hybrid);
    ColsWalk {
        c,
        s,
        m,
        side,
        rows: pass,
        mix: Mix35 { seqs, m },
        tail: Tail::One { head, state },
    }
    .walk(&mut leg)
}

/// A placed pass's mixer: layer `layer`'s front over the slots' ops
/// ([`walk::front`]), leaving the FFN input in the arena for [`ColsWalk`].
struct Mix35<'a, 's> {
    seqs: &'a mut [SlotSeq35<'s>],
    m: usize,
}

impl ColsMix for Mix35<'_, '_> {
    fn mix_layer(&mut self, layer: usize, c: &PassCtx<'_>, s: &mut Arena) -> Result<(), GpuError> {
        let cx = LayerCtx::of(c.plans, layer, self.m)?;
        let mut ops = Slots35 {
            c,
            seqs: &mut *self.seqs,
            s,
            head: None,
        };
        walk::front(&mut ops, cx)
    }
}

/// The head-128 attention with a NORM rope and no head norm, the one head-128 kind a pass of slots
/// binds per slot.
const NORM_128: GqaKind = GqaKind::Head128 {
    pairs: Pairs::Norm,
    head_norm: false,
};

/// A [`NORM_128`] layer's middle at the pass's `m` rows (the step's launches in its order): each
/// slot's rope turn and cache append and its flash over its rows and its K/V planes, then a K-quant
/// output projection's q8_1 rows of the flash output.
fn norm_core(
    c: &dispatch::Ctx<'_>,
    n: &GqaPlan,
    seqs: &mut [SlotSeq35<'_>],
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, k) = (c.gpu, c.k);
    let norm = &k.q35(WHAT)?.norm;
    let stream = gpu.stream();
    let d = s.dims;
    {
        let Arena {
            q,
            k: kr,
            v,
            pos,
            n_keys,
            part_v,
            part_ms,
            attn,
            ..
        } = &mut *s;
        for qs in seqs.iter_mut() {
            let r = &qs.rows;
            let kv = kv_of(&mut *qs.stores, c.layer)?;
            let pos = rows(pos, r, 1)?;
            let mut qr = rows_mut(q, r, d.attn_len())?;
            kv.append_norm(
                norm,
                stream,
                AppendNorm {
                    q: &mut qr,
                    k: &mut *rows_mut(kr, r, d.kv_len())?,
                    v: &*rows(v, r, d.kv_len())?,
                    table: qs.table,
                    pos: &pos,
                    n_head: d.n_head,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m: r.len(),
                    fault: c.sink,
                },
            )?;
            kv.flash_head128(
                &k.flash,
                stream,
                FlashPass {
                    q: &qr,
                    n_keys: &*rows(n_keys, r, 1)?,
                    scale: ATTN_SCALE,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m: r.len(),
                    part_v: &mut *rows_mut(part_v, r, partials_v_len(1, d.n_head))?,
                    part_ms: &mut *rows_mut(part_ms, r, partials_ms_len(1, d.n_head))?,
                    fault: c.sink,
                    y: &mut *rows_mut(attn, r, d.attn_len())?,
                },
                n.flash,
                c.mma,
            )?;
        }
    }
    if n.o_ty.kquant() {
        let i = s.col(m)?;
        gpu.enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], c.layer)?;
    }
    Ok(())
}

impl ChainOps for Slots35<'_, '_> {
    type Error = GpuError;

    /// Each slot's embedding into its rows, from its own input record.
    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        let (c, s) = (self.c, &mut *self.s);
        let hidden = s.dims.hidden;
        for q in self.seqs.iter() {
            let Arena { x, pos, n_keys, .. } = &mut *s;
            let mut x = rows_mut(x, &q.rows, hidden)?;
            let mut pos = rows_mut(pos, &q.rows, 1)?;
            let mut n_keys = rows_mut(n_keys, &q.rows, 1)?;
            dispatch::embed_into(c.gpu, c.w, c.k, &q.io, (&mut x, &mut pos, &mut n_keys))?;
        }
        Ok(())
    }

    /// q, k and v over every row; any attention other than gated head 256
    /// or head 128 turned by NORM pairs with no head norm is refused by name.
    fn attn_in(&mut self, l: LayerCtx<'_>, n: &GqaPlan) -> Result<(), GpuError> {
        let c = &layer_ctx(self.c, l)?;
        if n.kind != GqaKind::Gated256 && n.kind != NORM_128 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {} is {:?} attention, and a pass of slots binds only gated attention at \
                     head 256 or head 128 turned by NORM pairs with no head norm per slot",
                    c.layer, n.kind
                ),
            ));
        }
        dispatch::attn_in(c, n, self.s, l.m)
    }

    /// Each slot's q/k norm, rope and cache append over its rows and its K/V
    /// planes (one launch over every slot on f16 planes), its flash over
    /// them, then the output gate over every row — in
    /// the output projection's quantizer for a K-quant, as f32 rows
    /// otherwise. A head-128 layer turned by NORM pairs runs [`norm_core`].
    fn attn_core(&mut self, l: LayerCtx<'_>, n: &GqaPlan) -> Result<(), GpuError> {
        let (c, seqs, s, m) = (&layer_ctx(self.c, l)?, &mut *self.seqs, &mut *self.s, l.m);
        if n.kind == NORM_128 {
            return norm_core(c, n, seqs, s, m);
        }
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
            let mut planes = Vec::with_capacity(seqs.len());
            for qs in seqs.iter_mut() {
                planes.push((qs.rows.clone(), kv_of(&mut *qs.stores, c.layer)?, qs.table));
            }
            if planes.iter().all(|(_, kv, _)| kv.seqs_serve()) {
                let kv = planes
                    .iter_mut()
                    .map(|(r, kv, table)| kv.kv_seq(r.clone(), table))
                    .collect::<Result<Vec<_>, GpuError>>()?;
                // SAFETY: the layer program embeds consecutive positions
                // within each slot's rows; each sequence appends into its
                // own planes, turned by its own table.
                unsafe {
                    k.neox.enqueue_head_norm_neox_append_256_seqs(
                        stream,
                        &kv,
                        PartialNeoxSeqsArgs {
                            qg: q,
                            q: q_out,
                            k: kr,
                            v,
                            gq,
                            gk,
                            pos,
                            eps: c.eps,
                            n_head: d.n_head,
                            n_kv: d.n_kv,
                            ctx: d.ctx,
                            fault: c.sink,
                        },
                    )
                }?;
            } else {
                for (r, kv, table) in &mut planes {
                    let pos = rows(pos, r, 1)?;
                    // SAFETY: the layer program embeds consecutive positions
                    // before attention; saturation is outside the cache. Each
                    // call uses one sequence's row window and its own cache
                    // planes, not the combined positions of different slots.
                    unsafe {
                        kv.append_256(
                            &k.neox,
                            stream,
                            Append256 {
                                qg: &*rows(q, r, d.q_rows)?,
                                q: &mut *rows_mut(q_out, r, d.attn_len())?,
                                k: &mut *rows_mut(kr, r, d.kv_len())?,
                                v: &*rows(v, r, d.kv_len())?,
                                gq,
                                gk,
                                table,
                                pos: &pos,
                                eps: c.eps,
                                n_head: d.n_head,
                                n_kv: d.n_kv,
                                ctx: d.ctx,
                                m: r.len(),
                                fault: c.sink,
                            },
                        )
                    }?;
                }
            }
            for (r, kv, _) in &mut planes {
                let qo = rows(q_out, r, d.attn_len())?;
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
            let gated = s
                .gated_rows
                .as_mut()
                .ok_or(GpuError::state(WHAT, "the arena's gated rows"))?;
            q35.q38.enqueue_out_gate(
                stream,
                OutGateArgs {
                    attn: &s.attn,
                    qg: &s.q,
                    n_head: d.n_head,
                    m,
                    fault: c.sink,
                    y: gated,
                },
            )?;
        }
        Ok(())
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, n: &GqaPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::Slots35::attn_out")?;
        dispatch::attn_out(&layer_ctx(self.c, l)?, n, self.s, l.m)
    }

    /// The arena checked against the plan's shape, then the input
    /// projections over every row: `delta`'s own input half (a pass's rows
    /// never go past `GEMV_COLS`, so it takes the gemv arm).
    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), GpuError> {
        delta::delta_in(&layer_ctx(self.c, l)?, d, self.s, l.m)
    }

    /// Every slot's conv and delta step over its rows, its conv ring and its
    /// state at the slot's lane word, one launch each over every slot
    /// (`linear::seqs`), then the gated norm over every row.
    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), GpuError> {
        let (c, seqs, s, m) = (&layer_ctx(self.c, l)?, &mut *self.seqs, &mut *self.s, l.m);
        let (gpu, w) = (c.gpu, c.w);
        let lin = &c.k.q35(WHAT)?.linear;
        let stream = gpu.stream();
        let nv = d.shape.n_v;
        let Arena { pos, attn, gdn, .. } = &mut *s;
        let g = gdn
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the arena's delta intermediates"))?;
        let mut recs = Vec::with_capacity(seqs.len());
        for q in seqs.iter_mut() {
            recs.push(GdnSeq {
                rows: q.rows.clone(),
                lane: q
                    .io
                    .lane
                    .ok_or(GpuError::state(WHAT, "a slot's lane word"))?,
                rec: rec_of(&mut *q.stores, c.layer)?,
                verify: None,
            });
        }
        linear::seqs::enqueue_gdn_seqs(
            lin,
            stream,
            GdnSeqsArgs {
                x: &g.x,
                b_raw: &g.b,
                a_raw: &g.a,
                ba_pitch: nv,
                w: f32_gain(w, &d.conv)?,
                dt_bias: f32_gain(w, &d.dt_bias)?,
                ssm_a: f32_gain(w, &d.ssm_a)?,
                pos,
                shape: d.shape,
                eps: c.eps,
                fault: c.sink,
                conv: &mut g.conv,
                beta: &mut g.beta,
                decay: &mut g.decay,
                o: &mut g.o,
            },
            &mut recs,
        )?;
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
        Ok(())
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::Slots35::delta_out")?;
        delta::output(&layer_ctx(self.c, l)?, d, self.s, l.m)
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), GpuError> {
        dispatch::route(&layer_ctx(self.c, l)?, f, self.s, l.m)
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::Slots35::experts")?;
        dispatch::experts(&layer_ctx(self.c, l)?, f, self.s, l.m, None)
    }

    /// The one head of every row ([`Tail::One`]); a placed pass's mixer
    /// ops hold none, its walk running the head.
    fn head(&mut self, m: usize) -> Result<(), GpuError> {
        let (head, state) = self.head.as_mut().ok_or(GpuError::state(
            WHAT,
            "the pass's head (a placed pass's walk runs it)",
        ))?;
        let tail = &mut Tail::One { head, state };
        program::enqueue_tail(self.c, self.s, m, tail)
    }
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
        let seq = &mut **others.next().ok_or_else(|| r.no_parked(WHAT))?;
        b.exchange(seq);
        let out = f(b, r);
        b.exchange(seq);
        out?;
    }
    Ok(())
}
