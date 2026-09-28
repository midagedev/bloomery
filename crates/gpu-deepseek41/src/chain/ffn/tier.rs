//! The MoE sub-layer's tier layer ([`bloomery_gpu::host::tier`]): a layer
//! whose slot-map row puts experts on the tier card. The stage card runs the
//! layer as [`super`]'s module comment says, with two entries in place of the
//! handoff and the join and the tier's go and wait in the same batches:
//!
//! - `ds41_ffn_handoff_tier` writes the handoff as `ds41_ffn_handoff` does and
//!   the row's tier image: each slot's tier place from the map's tier view
//!   ([`TierPiece`]'s card copy), the norm's q8_1 codes and scales of the
//!   activation; the tier places also go to a card buffer the join reads.
//! - `ds41_ffn_post_tier` (`_streams_tier` without the fold) is the join with
//!   the card sum over the slots on the stage card or on the tier, in the
//!   router's pick order, each slot's down output read from the stage card's
//!   rows or, for a tier slot, from the tier's rows through the mapping:
//!   [`card_sum_elem`]'s rule over that mask, then [`join_elem`]. With the
//!   union of both cards' experts on one card the combine reads the same
//!   values in the same order, so the two agree bit for bit.
//!
//! The tier's own layer ([`Ds41Tier`], the architecture's
//! [`TierExperts`]) is the stage card's routed launches over the tier's
//! stacks: the `_sel` gate·up on the staged activation, the q8_1 of the tier
//! slots' columns of `h`, and the `_sel` down into the tier's rows; over a
//! prompt batch's block, the stage card's tile path over its own scratch
//! ([`super::batch::enqueue_tiled_experts`]).

use bloomery_gpu::FaultSite;
use bloomery_gpu::host::tier::{TierBlock, TierExperts, TierIo, TierSet, TierTarget};

use super::batch::{FfnBatchKernels, TileScratch, TiledBlock, enqueue_tiled_experts};

use super::*;

/// What the tier entries' errors name.
const TIER_WHAT: &str = "FfnPiece::enqueue (tier layer)";

#[cuda_module]
mod tier_kernels {
    use super::*;

    /// `ds41_ffn_handoff` and the row's tier image, one thread per
    /// activation value `d < n`: thread `d` copies `x[d]` to the image's word
    /// `x_at + d`; threads `s < 6` copy slot `s`'s id and weight to words
    /// `ids_at + s` and `wts_at + s`, write `sel[s] = map[row_off + id]` and
    /// `tsel[s] = tmap[row_off + id]` and the tier place to the tier image's
    /// word `tsel_at + s` (an id not below `n_expert` raises
    /// [`FaultSite::ExpertId`] and both places are [`HOST`]); threads `d <
    /// nq` copy the q8_1 code `q3[d]` to the tier image's words `q3_at + 2d`
    /// (low half) and `q3_at + 2d + 1`; threads `d < nd` copy `d8[d]` to word
    /// `d8_at + d`; thread 0 copies the sequence word. The go that follows
    /// orders every write here before its generation and the tier's go.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            ids_in.len() >= 6,
            w_in.len() >= 6,
            map.len() >= row_off + n_expert,
            tmap.len() >= row_off + n_expert,
            x.len() >= n,
            seq.len() >= 1,
            q3.len() >= nq,
            d8.len() >= nd,
            n >= 6,
            n >= nq,
            n >= nd,
            ids_at >= seq_at + 1,
            wts_at >= ids_at + 6,
            x_at >= wts_at + 6,
            image.len() >= x_at + n,
            q3_at >= tsel_at + 6,
            d8_at >= q3_at + 2 * nq,
            timage.len() >= d8_at + nd,
            sel.len() >= 6,
            tsel.len() >= 6
        )
    )]
    pub fn ds41_ffn_handoff_tier(
        ids_in: &[u32],
        w_in: &[f32],
        map: &[u32],
        tmap: &[u32],
        row_off: u32,
        n_expert: u32,
        x: &[f32],
        seq: &[u32],
        q3: &[u64],
        d8: &[f32],
        n: u32,
        nq: u32,
        nd: u32,
        seq_at: u32,
        ids_at: u32,
        wts_at: u32,
        x_at: u32,
        tsel_at: u32,
        q3_at: u32,
        d8_at: u32,
        fault: FaultSink,
        mut image: DisjointSlice<u32>,
        mut timage: DisjointSlice<u32>,
        mut sel: DisjointSlice<u32>,
        mut tsel: DisjointSlice<u32>,
    ) {
        let d = thread::index_1d().get();
        if d >= n as usize {
            return;
        }
        // SAFETY: d < n <= x.len(), and x_at + d < x_at + n <= image.len(), by
        // the launch contract; the image's words past x_at are the activation's
        // alone, and thread d is word x_at + d's only writer.
        unsafe {
            *image.get_unchecked_mut(x_at as usize + d) = (*x.get_unchecked(d)).to_bits();
        }
        if d < N_USED {
            // SAFETY: d < 6 <= ids_in.len() and w_in.len() by the launch
            // contract.
            let (id, w) = unsafe { (*ids_in.get_unchecked(d), *w_in.get_unchecked(d)) };
            let (place, tplace) = if id < n_expert {
                let at = row_off as usize + id as usize;
                // SAFETY: id < n_expert, so row_off + id < map.len() and
                // tmap.len() by the launch contract.
                unsafe { (*map.get_unchecked(at), *tmap.get_unchecked(at)) }
            } else {
                fault.raise(FaultSite::ExpertId);
                (HOST, HOST)
            };
            // SAFETY: d < 6 <= sel.len() and tsel.len(); ids_at + d and wts_at
            // + d lie in the routing's two spans of the image, apart from each
            // other, from seq_at and from the activation; tsel_at + d lies in
            // the tier image's places, below q3_at; thread d is each of those
            // words' only writer.
            unsafe {
                *image.get_unchecked_mut(ids_at as usize + d) = id;
                *image.get_unchecked_mut(wts_at as usize + d) = w.to_bits();
                *sel.get_unchecked_mut(d) = place;
                *tsel.get_unchecked_mut(d) = tplace;
                *timage.get_unchecked_mut(tsel_at as usize + d) = tplace;
            }
        }
        if d < nq as usize {
            // SAFETY: d < nq <= q3.len(); q3_at + 2d + 1 < q3_at + 2·nq <= d8_at
            // < timage.len(), the codes' span, by the launch contract; thread d
            // is both words' only writer.
            unsafe {
                let q = *q3.get_unchecked(d);
                *timage.get_unchecked_mut(q3_at as usize + 2 * d) = q as u32;
                *timage.get_unchecked_mut(q3_at as usize + 2 * d + 1) = (q >> 32) as u32;
            }
        }
        if d < nd as usize {
            // SAFETY: d < nd <= d8.len(); d8_at + d < d8_at + nd <=
            // timage.len(), the scales' span, by the launch contract; thread d
            // is the word's only writer.
            unsafe {
                *timage.get_unchecked_mut(d8_at as usize + d) = (*d8.get_unchecked(d)).to_bits();
            }
        }
        if d == 0 {
            // SAFETY: seq.len() >= 1, and seq_at < ids_at is inside the image
            // and no other span's word, by the launch contract; thread 0 alone
            // writes it.
            unsafe { *image.get_unchecked_mut(seq_at as usize) = *seq.get_unchecked(0) };
        }
    }

    /// The combine and HC_POST with the next fold of a tier layer, one
    /// thread per value `d` ([`combine_post_tier_at`]): `ds41_ffn_post`'s
    /// outputs, the card sum over the slots on the stage card or the tier.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 6 * rows,
            trows.len() >= 6 * rows,
            w.len() >= 6,
            sel.len() >= 6,
            tsel.len() >= 6,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows,
            fold.len() >= rows
        )
    )]
    pub fn ds41_ffn_post_tier(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        n_tier: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows, and the launch contract gives every length the
        // helper's contract asks for.
        let (yv, o, pre) =
            unsafe { combine_post_tier_at(&a, trows, tsel, rows, [n_card, n_tier], d) };
        // SAFETY: d < rows <= y.len() and fold.len(), and d + 3·rows < 4·rows
        // <= out.len(), by the launch contract; thread d is the only writer of
        // y[d], fold[d] and the four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
            *fold.get_unchecked_mut(d) = hc_fold_elem(o, pre);
        }
    }

    /// [`ds41_ffn_post_tier`] without the fold: into an engram layer and
    /// after the last layer.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 6 * rows,
            trows.len() >= 6 * rows,
            w.len() >= 6,
            sel.len() >= 6,
            tsel.len() >= 6,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows
        )
    )]
    pub fn ds41_ffn_post_streams_tier(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        n_tier: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows (checked above), and the launch contract gives
        // every length the helper's contract asks for.
        let (yv, o, _) =
            unsafe { combine_post_tier_at(&a, trows, tsel, rows, [n_card, n_tier], d) };
        // SAFETY: d < rows <= y.len() and d + 3·rows < 4·rows <= out.len(),
        // by the launch contract; thread d is the only writer of y[d] and the
        // four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
        }
    }
}

/// Value `d`'s combine of a tier layer, then HC_POST of it: slot `j` is on
/// the stage card when its place `sel[j]` is below `n[0]` — its down output
/// is `a.down`'s — and on the tier when its tier place `tsel[j]` is below
/// `n[1]` — its down output is `trows`' (`rows` each, slot-major) — and
/// [`combine_elem`] runs over that mask; every other slot's rows are never
/// read. The combine's `y`, the four new stream values and `pre`.
///
/// SAFETY: `d < rows`, `down.len()` and `trows.len() >= 6 * rows`,
/// `w.len()`, `sel.len()` and `tsel.len() >= 6`, `hsum.len()` and
/// `shexp.len() >= rows`, `res.len() >= 4 * rows` and `hc.len() >= 24`.
#[inline(always)]
unsafe fn combine_post_tier_at(
    a: &PostIn<'_>,
    trows: &[f32],
    tsel: &[u32],
    rows: usize,
    n: [u32; 2],
    d: usize,
) -> (f32, [f32; 4], [f32; 4]) {
    let mut dv = [0.0f32; N_USED];
    let mut wv = [0.0f32; N_USED];
    let mut on = [false; N_USED];
    let mut j = 0usize;
    while j < N_USED {
        // SAFETY: j < 6 <= sel.len(), tsel.len() and w.len() by this fn's
        // contract.
        let (place, tplace, wj) = unsafe {
            (
                *a.sel.get_unchecked(j),
                *tsel.get_unchecked(j),
                *a.w.get_unchecked(j),
            )
        };
        if place < n[0] {
            on[j] = true;
            wv[j] = wj;
            // SAFETY: j < 6 and d < rows, so j·rows + d < 6·rows <=
            // down.len() by this fn's contract.
            dv[j] = unsafe { *a.down.get_unchecked(j * rows + d) };
        } else if tplace < n[1] {
            on[j] = true;
            wv[j] = wj;
            // SAFETY: as above, for trows.
            dv[j] = unsafe { *trows.get_unchecked(j * rows + d) };
        }
        j += 1;
    }
    // SAFETY: d < rows <= hsum.len() and shexp.len(); d + 3·rows < 4·rows <=
    // res.len(); 23 < 24 <= hc.len() — all by this fn's contract.
    let (hs, sh, r, pre, post, comb) = unsafe {
        (
            *a.hsum.get_unchecked(d),
            *a.shexp.get_unchecked(d),
            [
                *a.res.get_unchecked(d),
                *a.res.get_unchecked(d + rows),
                *a.res.get_unchecked(d + 2 * rows),
                *a.res.get_unchecked(d + 3 * rows),
            ],
            [
                *a.hc.get_unchecked(0),
                *a.hc.get_unchecked(1),
                *a.hc.get_unchecked(2),
                *a.hc.get_unchecked(3),
            ],
            [
                *a.hc.get_unchecked(4),
                *a.hc.get_unchecked(5),
                *a.hc.get_unchecked(6),
                *a.hc.get_unchecked(7),
            ],
            [
                *a.hc.get_unchecked(8),
                *a.hc.get_unchecked(9),
                *a.hc.get_unchecked(10),
                *a.hc.get_unchecked(11),
                *a.hc.get_unchecked(12),
                *a.hc.get_unchecked(13),
                *a.hc.get_unchecked(14),
                *a.hc.get_unchecked(15),
                *a.hc.get_unchecked(16),
                *a.hc.get_unchecked(17),
                *a.hc.get_unchecked(18),
                *a.hc.get_unchecked(19),
                *a.hc.get_unchecked(20),
                *a.hc.get_unchecked(21),
                *a.hc.get_unchecked(22),
                *a.hc.get_unchecked(23),
            ],
        )
    };
    let y = combine_elem(dv, wv, on, hs, sh);
    (y, hc_post_elem(y, r, post, &comb), pre)
}

/// The stage card's side of the tier layers: their two entries, each row's
/// tier places (what the join reads), and the card copy of the map's tier
/// view (what the handoff reads the tier places from). Built once at load
/// by a body with a tier card.
pub struct TierPiece {
    module: tier_kernels::LoadedModule,
    tsel: Vec<DeviceBuffer<u32>>,
    places: DeviceTensor<u32>,
}

impl TierPiece {
    /// The stage card's tier side for `map`'s layers, `rows` rows. Load-time
    /// only.
    pub fn new(gpu: &Gpu, map: &SlotMap, rows: usize) -> Result<TierPiece, GpuError> {
        let ctx = gpu.context();
        let stream = gpu.stream();
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { tier_kernels::load(ctx)? };
        let tsel = (0..rows)
            .map(|_| DeviceBuffer::zeroed(stream, N_USED))
            .collect::<Result<Vec<_>, _>>()?;
        let places =
            DeviceTensor::upload(stream, &map.tier_view(), map.layers().len(), map.n_expert())?;
        Ok(TierPiece {
            module,
            tsel,
            places,
        })
    }

    /// Device bytes the piece holds: the rows' places and the tier view.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.tsel.iter().map(DeviceBuffer::num_bytes).sum::<usize>() + self.places.buf().num_bytes()
    }

    /// The card copy of the map's tier view, a row of places per layer at
    /// the card copy's row offsets: what a prompt batch's route reads its
    /// slots' tier places from.
    #[must_use]
    pub fn places(&self) -> &DeviceTensor<u32> {
        &self.places
    }

    fn tsel(&self, row: usize) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.tsel.get(row).ok_or_else(|| GpuError::Shape {
            what: TIER_WHAT,
            detail: format!("row {row} of a tier piece of {} rows", self.tsel.len()),
        })
    }
}

impl FfnPiece {
    /// [`FfnPiece::enqueue_go_front`] of a tier layer — one whose slot-map
    /// row puts experts on the tier card: the norm and the router, then
    /// `ds41_ffn_handoff_tier` into row `row`'s image and tier image, then
    /// the go with the tier's ([`Hybrid::enqueue_tier_go`]). The shadow that
    /// follows is [`FfnPiece::enqueue_go_shadow`], unchanged: the stage
    /// card's slots only.
    #[allow(
        clippy::too_many_arguments,
        reason = "enqueue's arguments, the row and the tier side (rust-quality R8)"
    )]
    pub fn enqueue_go_front_tier<'w, H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        w: &'w Weights,
        card: Option<CardStacks<'w>>,
        io: &FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        row: usize,
        tier: &mut TierPiece,
    ) -> Result<GoFront<'w>, GpuError> {
        let i = self.check_io(layer, hybrid.slots(), io, row)?;
        if hybrid.on_tier(layer)? == 0 {
            return Err(GpuError::Shape {
                what: TIER_WHAT,
                detail: format!("layer {layer} holds no tier expert: its handoff is the plain one"),
            });
        }
        let card = self.check_card(layer, i, card)?;
        let lw = LayerWeights::resolve(
            &self.cfg[i],
            w,
            [self.n_embd, self.ff],
            self.hc_eps,
            self.hc_iters,
        )?;
        self.enqueue_route(gpu, row, &lw, io, hybrid, layer)?;
        let stream = gpu.stream();
        let fault = gpu.layer_sink(layer)?;
        let (n, n_expert) = (self.n_embd, self.n_expert);
        let c = &self.cfg[i];
        let r = &mut self.rows[row];
        let tsel = tier.tsel.get_mut(row).ok_or_else(|| GpuError::Shape {
            what: TIER_WHAT,
            detail: format!("row {row} of the tier piece"),
        })?;
        let (target, ttarget) = hybrid.tier_handoff(row)?;
        let h = Handoff {
            ids: &r.rout.ids,
            weights: &r.rout.weights,
            map: io.slots.buf(),
            row_off: c.row_off,
            n_expert,
        };
        enqueue_handoff_tier(
            &tier.module,
            stream,
            &h,
            tier.places.buf(),
            &r.act_x,
            (target, ttarget),
            n,
            fault,
            (&mut r.sel, tsel),
        )?;
        hybrid.enqueue_tier_go(stream, layer, row)?;
        Ok(GoFront {
            layer,
            row,
            i,
            lw,
            card,
        })
    }

    /// [`FfnPiece::enqueue_join_half`] of a tier layer: the wait for the
    /// host's counter and the tier's ([`Hybrid::enqueue_tier_back`]), then
    /// `ds41_ffn_post_tier` (or `_streams_tier`), then the host tier told the
    /// row's layer is enqueued.
    pub fn enqueue_join_half_tier<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        io: FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        row: usize,
        tier: &TierPiece,
    ) -> Result<(), GpuError> {
        let i = self.check_io(layer, hybrid.slots(), &io, row)?;
        let n_tier = hybrid.on_tier(layer)?;
        let stream = gpu.stream();
        hybrid.enqueue_tier_back(stream, row)?;
        {
            let hsum = hybrid.boundary().hsum_of(row)?;
            let trows = hybrid.tier_rows(row)?;
            let r = &mut self.rows[row];
            let what = if io.fold_out.is_some() {
                "ds41_ffn_post_tier"
            } else {
                "ds41_ffn_post_streams_tier"
            };
            let rows = self.n_embd;
            let grid = launch_u32(what, "grid", rows.div_ceil(POST_THREADS as usize))?;
            let cfg = LaunchConfig1D::new(grid, POST_THREADS, 0);
            let (rows32, n_card, n_tier) = (
                launch_u32(what, "rows", rows)?,
                launch_u32(what, "n_card", self.cfg[i].n_card)?,
                launch_u32(what, "n_tier", n_tier)?,
            );
            let tsel = tier.tsel(row)?;
            let m = &tier.module;
            match io.fold_out {
                Some(fold) => {
                    let prep = m.prepare_ds41_ffn_post_tier(cfg)?;
                    m.ds41_ffn_post_tier(
                        stream,
                        &prep,
                        &r.down,
                        trows,
                        &r.rout.weights,
                        &r.sel,
                        tsel,
                        hsum,
                        &r.sh_y,
                        io.streams,
                        &r.hc_out,
                        rows32,
                        n_card,
                        n_tier,
                        &mut r.y,
                        io.streams_out,
                        fold,
                    )?;
                }
                None => {
                    let prep = m.prepare_ds41_ffn_post_streams_tier(cfg)?;
                    m.ds41_ffn_post_streams_tier(
                        stream,
                        &prep,
                        &r.down,
                        trows,
                        &r.rout.weights,
                        &r.sel,
                        tsel,
                        hsum,
                        &r.sh_y,
                        io.streams,
                        &r.hc_out,
                        rows32,
                        n_card,
                        n_tier,
                        &mut r.y,
                        io.streams_out,
                    )?;
                }
            }
        }
        hybrid.row_enqueued(layer, row)
    }
}

/// Enqueue `ds41_ffn_handoff_tier`: `h`'s routing and the target's
/// activation into the target's image, the stage card's places into
/// `sel.0`, the tier's (from `tmap`, the map's tier view) into `sel.1` and
/// the tier image, and `act`'s q8_1 codes and scales into the tier image.
/// One launch.
#[allow(
    clippy::too_many_arguments,
    reason = "one launch's routing, maps, activation, targets and places (rust-quality R8)"
)]
fn enqueue_handoff_tier(
    module: &tier_kernels::LoadedModule,
    stream: &CudaStream,
    h: &Handoff<'_>,
    tmap: &DeviceBuffer<u32>,
    act: &Q8Act,
    (target, ttarget): (HandoffTarget<'_>, TierTarget<'_>),
    n: usize,
    fault: FaultSink,
    (sel, tsel): (&mut DeviceBuffer<u32>, &mut DeviceBuffer<u32>),
) -> Result<(), GpuError> {
    const WHAT: &str = "ds41_ffn_handoff_tier";
    let lay = target.layout;
    let tl = ttarget.layout;
    let (nq, nd) = (act.q3().len(), act.d8().len());
    if lay.n_used != N_USED
        || lay.hidden != n
        || tl.n_used != N_USED
        || tl.q3_words != 2 * nq
        || tl.d8_len != nd
    {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the boundary carries {} slots of {} values and the tier image {} slots, {} code \
                 words, {} scales; the piece hands over {N_USED} slots of {n}, {} codes, {nd} scales",
                lay.n_used,
                lay.hidden,
                tl.n_used,
                tl.q3_words,
                tl.d8_len,
                2 * nq
            ),
        });
    }
    let grid = launch_u32(WHAT, "grid", n.div_ceil(256))?;
    let cfg = LaunchConfig1D::new(grid, 256, 0);
    let prep = module.prepare_ds41_ffn_handoff_tier(cfg)?;
    module.ds41_ffn_handoff_tier(
        stream,
        &prep,
        h.ids,
        h.weights,
        h.map,
        tmap,
        launch_u32(WHAT, "row_off", h.row_off)?,
        launch_u32(WHAT, "n_expert", h.n_expert)?,
        target.x,
        target.seq,
        act.q3(),
        act.d8(),
        launch_u32(WHAT, "n", n)?,
        launch_u32(WHAT, "nq", nq)?,
        launch_u32(WHAT, "nd", nd)?,
        launch_u32(WHAT, "seq_at", lay.seq)?,
        launch_u32(WHAT, "ids_at", lay.ids)?,
        launch_u32(WHAT, "wts_at", lay.weights)?,
        launch_u32(WHAT, "x_at", lay.x)?,
        launch_u32(WHAT, "tsel_at", tl.sel)?,
        launch_u32(WHAT, "q3_at", tl.q3)?,
        launch_u32(WHAT, "d8_at", tl.d8)?,
        fault,
        target.image,
        ttarget.image,
        sel,
        tsel,
    )?;
    Ok(())
}

/// The V4.1 tier card's computation ([`TierExperts`]): per tier layer the
/// stage card's routed launches over the tier's stacks, into the tier's
/// rows; over a prompt batch's block, the tile path. Its scratch — the
/// slots' SwiGLU outputs and their q8_1, and the tile path's for a block of
/// the host union's columns — is made at load.
pub struct Ds41Tier {
    experts: ExpertKernels,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    batch: FfnBatchKernels,
    tile: TileScratch,
    layers: Range<usize>,
    /// Per layer of `layers`, the experts the tier holds and the layer's
    /// `swiglu_clamp_exp`.
    cfg: Vec<(usize, f32)>,
    n_embd: usize,
    ff: usize,
}

impl Ds41Tier {
    /// The tier's computation on `gpu` over `set`, whose stacks `w` holds
    /// (each tier layer's routed gate, up and down with its experts in tier
    /// slot order), for the model `hp`. A tier layer whose stacks are absent
    /// or of another shape is refused by name. Load-time only.
    pub fn new(gpu: &Gpu, hp: &Hparams, set: &TierSet, w: &Weights) -> Result<Ds41Tier, GpuError> {
        const WHAT: &str = "Ds41Tier::new";
        let (n, ff) = (hp.n_embd, hp.experts.ff);
        if hp.experts.n_used != N_USED || set.n_expert() != hp.experts.n_expert {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} slots a token and a set of {} experts: the tier is built for {N_USED} of {}",
                    hp.experts.n_used,
                    set.n_expert(),
                    hp.experts.n_expert
                ),
            });
        }
        let layers = set.layers();
        let mut cfg = Vec::with_capacity(layers.len());
        for l in layers.clone() {
            let k = set.on_tier(l)?;
            let kind = hp.layers.get(l).ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("layer {l} of a model of {} layers", hp.n_layer),
            })?;
            let stacks = CardStacks::of(w, l)?;
            match (k, stacks) {
                (0, None) => {}
                (k, Some(s))
                    if k > 0
                        && s.gate.rows() == k * ff
                        && s.up.rows() == k * ff
                        && s.down.rows() == k * n => {}
                (k, s) => {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "layer {l}: the tier holds {k} experts and its stacks are {}",
                            s.map_or_else(
                                || "absent".to_string(),
                                |s| format!(
                                    "gate {} up {} down {} rows",
                                    s.gate.rows(),
                                    s.up.rows(),
                                    s.down.rows()
                                )
                            )
                        ),
                    });
                }
            }
            cfg.push((k, kind.swiglu_limit));
        }
        let stream = gpu.stream();
        let tile = TileScratch::new(stream, UNION_MAX_COLS * N_USED, n, ff)?;
        Ok(Ds41Tier {
            experts: ExpertKernels::load(gpu.context())?,
            h: DeviceBuffer::zeroed(stream, N_USED * ff)?,
            act_h: Q8Act::with_k(stream, N_USED, ff)?,
            batch: FfnBatchKernels::load(gpu.context())?,
            tile,
            layers,
            cfg,
            n_embd: n,
            ff,
        })
    }
}

impl Ds41Tier {
    /// Layer `layer`'s tier experts and SwiGLU clamp; a layer the tier
    /// holds none of is refused by name.
    fn tier_layer(&self, layer: usize, what: &'static str) -> Result<(usize, f32), GpuError> {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.cfg.get(i))
            .copied()
            .filter(|&(k, _)| k > 0)
            .ok_or_else(|| GpuError::Shape {
                what,
                detail: format!("layer {layer} holds no tier expert"),
            })
    }
}

impl TierExperts for Ds41Tier {
    /// The tile path's scratch, for blocks of up to [`UNION_MAX_COLS`]
    /// tokens.
    fn block_bytes(&self) -> usize {
        self.tile.device_bytes()
    }

    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Ds41Tier::enqueue_block";
        let (_, limit) = self.tier_layer(layer, WHAT)?;
        let stacks = CardStacks::of(weights, layer)?.ok_or_else(|| GpuError::Tensor {
            what: WHAT,
            name: names::ffn_down_exps(layer),
            need: "the tier's routed stacks of a tier layer",
        })?;
        let t = TiledBlock {
            stacks,
            q3: io.act.q3(),
            d8: io.act.d8(),
            col0: 0,
            cols: io.cols,
            sel: io.sel,
            layer,
            limit,
        };
        enqueue_tiled_experts(gpu, &self.batch, &t, &mut self.tile, io.down)
    }

    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Ds41Tier::enqueue_layer";
        let (k, limit) = self.tier_layer(layer, WHAT)?;
        let s = CardStacks::of(weights, layer)?.ok_or_else(|| GpuError::Tensor {
            what: WHAT,
            name: names::ffn_down_exps(layer),
            need: "the tier's routed stacks of a tier layer",
        })?;
        let stream = gpu.stream();
        let args = ExpertGateUp {
            wg: s.gate,
            wu: s.up,
            act: io.act,
            sel: io.sel,
            n_slots: N_USED,
            rows_per_expert: self.ff,
            limit,
        };
        self.experts
            .enqueue_expert_gate_up(stream, &args, &mut self.h)?;
        let q = QuantSel {
            x: &self.h,
            cols: 0..N_USED,
            sel: io.sel,
            n_card: k,
        };
        gpu.q4k_sel()
            .enqueue_quantize_sel(stream, &q, gpu.layer_sink(layer)?, &mut self.act_h)?;
        gpu.q4k_sel().enqueue_gemv_q4k_sel(
            stream,
            s.down,
            &self.act_h,
            io.sel,
            N_USED,
            self.n_embd,
            io.rows,
        )
    }
}
