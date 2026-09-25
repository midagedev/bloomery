//! Q6_K gemv over expert slots selected on the device: the MoE down
//! projection of a Q6_K expert stack in the decode shape, all `n_slots`
//! routed experts in one launch — the per-slot-column shape of
//! `q4k_gemv_sel`: slot `s` dots the rows of expert `sel[s]` with
//! activation column `s`.
//!
//! The per-row body is `q6k_gemv`'s column-0 path verbatim — the same
//! byte-window loads with the 16-bit funnel for a super-block at 2 mod 4,
//! the same dp4a chain, the same `f0 += a · (d8 · d · sc)` accumulation and
//! the same warp reduction — with weight row `r` of expert `sel[slot]` and
//! the activation column `slot`. So slot `s` is bit-identical to `q6k_gemv`
//! run over expert `sel[s]`'s rows against a one-column activation
//! quantized from input column `s`; `gate_qwen3moe_down` pins that. Rows
//! are addressed by byte offset in the stack's word stream, so any
//! super-block count serves, odd ones included (a 768-value row is 630
//! bytes and every other row starts at 2 mod 4).

use crate::cores::{funnel16, half_to_f32, q6k_chain, q6k_dequant, q6k_sub_scale};
use crate::fault::{FaultSink, FaultSite, LAYER_NONE};
use crate::hybrid::HOST;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod q6k_sel_kernels {
    use super::*;

    /// Q6_K gemv over expert slots selected on the device: one warp per
    /// output row, eight rows per 256-thread block; thread row `n = slot ·
    /// rows_per_expert + r` stores `y[n]` from weight row `sel[slot] ·
    /// rows_per_expert + r` (`210 · n_sb` bytes at byte `row · 210 · n_sb` of
    /// the stack) against q8_1 column `slot` in the q6 permutation. An id
    /// `>= n_experts` returns the slot's warps before their first weight
    /// load (warp-uniform), leaving that slot of `y` untouched; one that is
    /// not [`HOST`] (a slot the host tier serves, skipped by contract)
    /// raises [`FaultSite::ExpertId`] on `fault` first.
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
            4 * w.len() >= n_experts * rows_per_expert * 210 * n_sb,
            q.len() >= n_slots * 128 * iters,
            d8.len() >= n_slots * 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q6k_gemv_sel(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes share `row`, hence `slot`).
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
        let n_sb = n_sb as usize;
        let row_bytes = 210 * n_sb;
        let q_col = 128 * iters as usize;
        let d8_col = 2 * n_sb;
        let q0c = q_col * slot;
        let d0c = d8_col * slot;

        let w16 = lane & 15;
        let half = lane >> 4;
        let qlok = 16 * (w16 & 3);
        let qhok = 16 * (w16 & 1);
        let nib_sh = ((w16 >> 2) & 1) as u32 * 4;
        let hib_sh = 2 * ((w16 >> 1) as u32 & 3);

        let mut f0 = 0.0f32;
        let mut it: u32 = 0;
        while it < iters {
            let sbp = ((it << 1) | half as u32) as usize;
            if sbp < n_sb {
                let base = row_abs * row_bytes + sbp * 210;
                let par = (base >> 1) & 1;

                let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
                // SAFETY: row_abs < n_experts·rows_per_expert, so the window
                // ends inside the stack's words (launch contract: 4·w.len()
                // covers every row's 210·n_sb bytes; the floored word index
                // of a window stays inside the row's words, as in q6k_gemv).
                let (l0, l1, l2, l3, l4) = unsafe {
                    (
                        *w.get_unchecked(lk),
                        *w.get_unchecked(lk + 1),
                        *w.get_unchecked(lk + 2),
                        *w.get_unchecked(lk + 3),
                        *w.get_unchecked(lk + 4),
                    )
                };
                let ql0 = if par == 0 { l0 } else { funnel16(l0, l1) };
                let ql1 = if par == 0 { l1 } else { funnel16(l1, l2) };
                let ql2 = if par == 0 { l2 } else { funnel16(l2, l3) };
                let ql3 = if par == 0 { l3 } else { funnel16(l3, l4) };

                let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
                // SAFETY: the same row bounds as the ql window.
                let (h0, h1, h2, h3, h4) = unsafe {
                    (
                        *w.get_unchecked(hk),
                        *w.get_unchecked(hk + 1),
                        *w.get_unchecked(hk + 2),
                        *w.get_unchecked(hk + 3),
                        *w.get_unchecked(hk + 4),
                    )
                };
                let qh0 = if par == 0 { h0 } else { funnel16(h0, h1) };
                let qh1 = if par == 0 { h1 } else { funnel16(h1, h2) };
                let qh2 = if par == 0 { h2 } else { funnel16(h2, h3) };
                let qh3 = if par == 0 { h3 } else { funnel16(h3, h4) };

                let ak = (base + 192) >> 2;
                // SAFETY: ak + 4 is at most the stack's final (possibly
                // zero-padded) word for the last super-block of the last
                // row; the launch contract bounds 4·w.len().
                let (a0, a1, a2, a3, a4) = unsafe {
                    (
                        *w.get_unchecked(ak),
                        *w.get_unchecked(ak + 1),
                        *w.get_unchecked(ak + 2),
                        *w.get_unchecked(ak + 3),
                        *w.get_unchecked(ak + 4),
                    )
                };
                let (sw0, sw1, sw2, sw3, d_bits) = if par == 0 {
                    (a0, a1, a2, a3, (a4 & 0xffff) as u16)
                } else {
                    (
                        funnel16(a0, a1),
                        funnel16(a1, a2),
                        funnel16(a2, a3),
                        funnel16(a3, a4),
                        (a4 >> 16) as u16,
                    )
                };
                let sc = q6k_sub_scale(&[sw0, sw1, sw2, sw3], w16);
                let drow = half_to_f32(d_bits);
                let vi = [
                    q6k_dequant(ql0, qh0, nib_sh, hib_sh),
                    q6k_dequant(ql1, qh1, nib_sh, hib_sh),
                    q6k_dequant(ql2, qh2, nib_sh, hib_sh),
                    q6k_dequant(ql3, qh3, nib_sh, hib_sh),
                ];
                let qb = q0c + 128 * it as usize + lane;
                let d8b = d0c + 2 * sbp + (w16 >> 3);
                // SAFETY: qb + 99 < q0c + q_col — this lane's four q8 words
                // are inside column `slot`'s q_col words, slot < n_slots; d8b
                // < d0c + d8_col by the sbp guard.
                let (q0, q1, q2, q3, e0) = unsafe {
                    (
                        *q.get_unchecked(qb),
                        *q.get_unchecked(qb + 32),
                        *q.get_unchecked(qb + 64),
                        *q.get_unchecked(qb + 96),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                f0 += (a as f32) * (e0 * drow * sc as f32);
            }
            it += 1;
        }
        let s0 = warp::reduce_sum_f32(f0);
        if lane == 0 {
            // SAFETY: row < n_slots·rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }
}

/// The loaded Q6_K expert-select module. Owns no stream: each enqueue takes
/// the engine stream.
pub struct Q6kSelKernels {
    module: q6k_sel_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context.
    fault: Arc<DeviceBuffer<u32>>,
}

impl Q6kSelKernels {
    /// Load this file's device bundle into `ctx`, raising into `word`, the
    /// fault word of the `Gpu` that owns `ctx` ([`crate::Gpu::fault_word`]);
    /// a word of another context is refused. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Q6kSelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Q6kSelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { q6k_sel_kernels::load(ctx)? };
        Ok(Q6kSelKernels { module, fault })
    }

    /// Enqueue the device-indirect MoE down projection for a Q6_K expert
    /// stack: slot `s` writes `y[s·rows_per_expert ..][..rows_per_expert]`
    /// as expert `sel[s]`'s rows against `act` column `s`. `w` is the whole
    /// stack as its byte stream in u32 words (`w.rows()` rows, a positive
    /// multiple of `rows_per_expert`, row `r` at byte `r · 210 · n_sb`, the
    /// tail zero-padded); `act` holds exactly `n_slots` columns. An id
    /// `>= n_experts` leaves its slot of `y` untouched, and one that is not
    /// [`HOST`] raises [`FaultSite::ExpertId`] on the owning `Gpu`'s fault
    /// word as an unlabelled launch ([`LAYER_NONE`]: the launcher knows no
    /// layer). Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher, the shape of enqueue_gemv_q4k_sel"
    )]
    pub fn enqueue_gemv_q6k_sel(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q6k_sel";
        let n_sb = act.n_sb();
        if act.m() != n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "one activation column per slot: act.m() = {} must equal n_slots = {n_slots}",
                    act.m()
                ),
            ));
        }
        if rows_per_expert == 0 || !w.rows().is_multiple_of(rows_per_expert) {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.rows() {} must be a positive multiple of rows_per_expert {rows_per_expert}",
                    w.rows()
                ),
            ));
        }
        // The widest reach is the scales window of the last row's last
        // super-block: its fifth word holds the super-block's `d`, the
        // stream's last bytes, so the stream's own words suffice.
        let need = (w.rows() * 210 * n_sb).div_ceil(4);
        if w.buf().len() < need {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} rows of 210*{n_sb} bytes need {need} words, got {}",
                    w.rows(),
                    w.buf().len()
                ),
            ));
        }
        if n_slots == 0 || sel.len() < n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_slots >= 1 and sel.len() >= n_slots, got n_slots {n_slots} \
                     sel.len() {}",
                    sel.len()
                ),
            ));
        }
        if y.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                what,
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {}",
                    y.len(),
                    n_slots * rows_per_expert
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rows_per_expert).div_ceil(8))?;
        let n_experts = launch_u32(what, "n_experts", w.rows() / rows_per_expert)?;
        let rows_per_expert = launch_u32(what, "rows_per_expert", rows_per_expert)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q6k_gemv_sel(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.q6k_gemv_sel(
            stream,
            &prep,
            w.buf(),
            &act.q6,
            &act.d8,
            sel,
            n_experts,
            rows_per_expert,
            n_slots,
            n_sb,
            n_sb.div_ceil(2),
            crate::sink_over(&self.fault, LAYER_NONE),
            y,
        )?;
        Ok(())
    }
}
