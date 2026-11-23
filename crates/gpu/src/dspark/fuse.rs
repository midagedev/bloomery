//! The fused context of a draft that mixes the target's tapped rows itself instead of running fc
//! over them ([`runtime::dspark::Context::Fuse`], Inkling 2's draft): one mix for each draft layer
//! and one for the head, mix `j` of a row being `Σ_k w[j][k] · tap_k` (the weights a softmax of the
//! file's row `j`, taken on the host at load), RMS-normed by its own gain.
//!
//! [`FuseKernels::enqueue_fuse`] is one launch over every mix of every row of an append: block `b =
//! j · n + t`, the row's sum in tap order with each product and sum rounded on its own (the
//! reference's multiply, then its row sum), the squares summed as the engine's RMS norm sums them
//! (`elem::rms_warp_tree`) and the scale `elem::rms_scale_checked`'s, so a sum of squares with no norm
//! behind it raises [`crate::FaultSite::F32Product`] and writes NaN. Mix `j` of row `t` lands at row
//! `j · group + t` of `x`; the last mix of the last row also lands in `last`, the head's context
//! row.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;

use crate::elem::{RMS_WARPS, rms_scale_checked, rms_warp_tree};
use crate::{FaultSink, GpuError, launch_u32};

const THREADS: usize = 256;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS / 32 == RMS_WARPS);

/// Mix `j`'s value `d` of row `t`: the taps' products summed in tap order.
///
/// # Safety
///
/// `t · taps + k` rows of `hidden` and `j · taps + k` weights inside `feats` and `mix` for every
/// `k < taps`, `d < hidden`.
#[inline(always)]
unsafe fn mixed(
    feats: &[f32],
    mix: &[f32],
    taps: usize,
    hidden: usize,
    j: usize,
    t: usize,
    d: usize,
) -> f32 {
    let mut v = 0.0f32;
    let mut k = 0usize;
    while k < taps {
        // SAFETY: the caller's bounds.
        let (w, f) = unsafe {
            (
                *mix.get_unchecked(j * taps + k),
                *feats.get_unchecked((t * taps + k) * hidden + d),
            )
        };
        v = add_rn_f32(v, mul_rn_f32(w, f));
        k += 1;
    }
    v
}

#[cuda_module]
mod fuse_kernels {
    use super::*;

    /// Every mix of every row (module doc): block `b = j · n + t`.
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
            n >= 1,
            n <= group,
            taps >= 1,
            mixes >= 1,
            hidden >= 1,
            feats.len() >= n * taps * hidden,
            mix.len() >= mixes * taps,
            gains.len() >= mixes * hidden,
            x.len() >= mixes * group * hidden,
            last.len() >= hidden
        )
    )]
    pub fn dspark_fuse(
        feats: &[f32],
        mix: &[f32],
        gains: &[f32],
        taps: u32,
        hidden: u32,
        n: u32,
        mixes: u32,
        group: u32,
        eps: f32,
        fault: FaultSink,
        mut x: DisjointSlice<f32>,
        mut last: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let rows = n as usize;
        if b >= mixes as usize * rows {
            return; // block-uniform
        }
        let j = b / rows;
        let t = b - j * rows;
        let (taps, k) = (taps as usize, hidden as usize);
        let tid = thread::threadIdx_x() as usize;
        let mut sq = 0.0f32;
        let mut d = tid;
        while d < k {
            // SAFETY: t < n, j < mixes, d < hidden: inside feats and mix by the launch contract.
            let v = unsafe { mixed(feats, mix, taps, k, j, t, d) };
            sq += v * v;
            d += THREADS;
        }
        let part = warp::reduce_sum_f32(sq);
        // SAFETY: WSUM is this block's own shared allocation; the raw form reaches it without a
        // reference. Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < RMS_WARPS; one lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = part };
        }
        thread::sync_threads();
        // SAFETY: every slot was written above and is visible past the barrier.
        let sums = unsafe {
            [
                *ws.add(0),
                *ws.add(1),
                *ws.add(2),
                *ws.add(3),
                *ws.add(4),
                *ws.add(5),
                *ws.add(6),
                *ws.add(7),
            ]
        };
        let scale = rms_scale_checked(rms_warp_tree(sums), hidden, eps, fault, tid == 0);
        let head_row = j + 1 == mixes as usize && t + 1 == rows;
        let out_row = (j * group as usize + t) * k;
        let mut d = tid;
        while d < k {
            // SAFETY: as above; j·hidden + d < mixes·hidden <= gains.len(); out_row + d <
            // mixes·group·hidden <= x.len() and d < hidden <= last.len(), each value this thread's
            // alone.
            unsafe {
                let v = mixed(feats, mix, taps, k, j, t, d);
                let y = (scale * *gains.get_unchecked(j * k + d)) * v;
                *x.get_unchecked_mut(out_row + d) = y;
                if head_row {
                    *last.get_unchecked_mut(d) = y;
                }
            }
            d += THREADS;
        }
    }
}

/// [`FuseKernels::enqueue_fuse`]'s arguments.
pub struct FuseArgs<'a> {
    /// An append's tapped rows: `n` rows of `taps · hidden` f32, each row's taps in order.
    pub feats: &'a DeviceBuffer<f32>,
    /// The mix weights, `mixes · taps` f32.
    pub mix: &'a DeviceBuffer<f32>,
    /// The mixes' norm gains, `mixes · hidden` f32.
    pub gains: &'a DeviceBuffer<f32>,
    pub taps: usize,
    pub hidden: usize,
    pub n: usize,
    pub mixes: usize,
    /// Rows a mix's block of `x` holds: `n <= group`.
    pub group: usize,
    pub eps: f32,
    pub fault: FaultSink,
    /// `mixes · group · hidden` f32.
    pub x: &'a mut DeviceBuffer<f32>,
    /// The head's context row, `hidden` f32.
    pub last: &'a mut DeviceBuffer<f32>,
}

/// The loaded fuse module. Owns no stream.
pub struct FuseKernels {
    module: fuse_kernels::LoadedModule,
}

impl FuseKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FuseKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; each
        // launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(fuse_kernels, ctx)? };
        Ok(FuseKernels { module })
    }

    /// Enqueue every mix of `n` rows (module doc). Asynchronous, allocation-free, capturable.
    pub fn enqueue_fuse(&self, stream: &CudaStream, a: FuseArgs<'_>) -> Result<(), GpuError> {
        let what = "FuseKernels::enqueue_fuse";
        let FuseArgs {
            feats,
            mix,
            gains,
            taps,
            hidden,
            n,
            mixes,
            group,
            eps,
            fault,
            x,
            last,
        } = a;
        if taps == 0
            || hidden == 0
            || mixes == 0
            || n == 0
            || n > group
            || feats.len() < n * taps * hidden
            || mix.len() < mixes * taps
            || gains.len() < mixes * hidden
            || x.len() < mixes * group * hidden
            || last.len() < hidden
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n} rows (1..={group}) of {taps} taps of {hidden} into {mixes} mixes: feats \
                     {}, weights {}, gains {}, x {}, last {}",
                    feats.len(),
                    mix.len(),
                    gains.len(),
                    x.len(),
                    last.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", mixes * n)?;
        let prep = self
            .module
            .prepare_dspark_fuse(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.dspark_fuse(
            stream,
            &prep,
            feats,
            mix,
            gains,
            launch_u32(what, "taps", taps)?,
            launch_u32(what, "hidden", hidden)?,
            launch_u32(what, "n", n)?,
            launch_u32(what, "mixes", mixes)?,
            launch_u32(what, "group", group)?,
            eps,
            fault,
            x,
            last,
        )?;
        Ok(())
    }
}

/// The mix weights of a file's `fuse_weight` rows (`mixes` rows of `taps` logits): each row's
/// softmax, `exp(l − max) / Σ exp(l − max)` in f32 as the reference's softmax takes it. A logit that
/// is not finite is refused by name.
pub fn mix_weights(logits: &[f32], taps: usize) -> Result<Vec<f32>, GpuError> {
    if taps == 0 || logits.is_empty() || !logits.len().is_multiple_of(taps) {
        return Err(GpuError::shape(
            "dspark::fuse",
            format!("{} fuse logits as rows of {taps}", logits.len()),
        ));
    }
    if let Some(bad) = logits.iter().find(|l| !l.is_finite()) {
        return Err(GpuError::shape(
            "dspark::fuse",
            format!("a fuse logit of {bad}: every one finite"),
        ));
    }
    let mut out = Vec::with_capacity(logits.len());
    for row in logits.chunks_exact(taps) {
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|&l| (l - mx).exp()).collect();
        let sum: f32 = e.iter().sum();
        out.extend(e.iter().map(|&v| v * (1.0 / sum)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;

    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 4_194_304.0 - 2.0
            })
            .collect()
    }

    /// The card's mixes are the host rule (the f64 mix, norm and gain) within the norm's f32
    /// rounding, at Inkling 2's four taps and four mixes of 2304 over 1 and 3 rows of a group of 8:
    /// each mix at row `j · 8 + t`, the head's mix of the last row in `last`, rows past `n` and the
    /// tail past the mixes untouched.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_fuse_is_the_host_rule() {
        const GUARD: f32 = 12_345.0;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let k = FuseKernels::load(gpu.context()).expect("the fuse module");
        let (taps, hidden, mixes, group, eps) = (4, 2304, 4, 8, 1e-6f32);
        let w = mix_weights(&seeded(mixes * taps, 3), taps).expect("weights");
        let gains = seeded(mixes * hidden, 4);
        let mut worst = 0.0f64;
        for n in [1usize, 3] {
            let feats = seeded(n * taps * hidden, 10 + n as u32);
            let fd = DeviceBuffer::from_host(stream, &feats).expect("feats");
            let wd = DeviceBuffer::from_host(stream, &w).expect("w");
            let gd = DeviceBuffer::from_host(stream, &gains).expect("gains");
            let mut x = DeviceBuffer::from_host(stream, &vec![GUARD; mixes * group * hidden + 5])
                .expect("x");
            let mut last = DeviceBuffer::zeroed(stream, hidden).expect("last");
            k.enqueue_fuse(
                stream,
                FuseArgs {
                    feats: &fd,
                    mix: &wd,
                    gains: &gd,
                    taps,
                    hidden,
                    n,
                    mixes,
                    group,
                    eps,
                    fault: gpu.unlabelled_sink(),
                    x: &mut x,
                    last: &mut last,
                },
            )
            .expect("fuse");
            let got = x.to_host_vec(stream).expect("x back");
            let head = last.to_host_vec(stream).expect("last back");
            for j in 0..mixes {
                for t in 0..group {
                    let row = &got[(j * group + t) * hidden..][..hidden];
                    if t >= n {
                        assert!(
                            row.iter().all(|&v| v == GUARD),
                            "mix {j} row {t} past n {n}"
                        );
                        continue;
                    }
                    let v: Vec<f64> = (0..hidden)
                        .map(|d| {
                            (0..taps)
                                .map(|kk| {
                                    f64::from(w[j * taps + kk])
                                        * f64::from(feats[(t * taps + kk) * hidden + d])
                                })
                                .sum()
                        })
                        .collect();
                    let ms = v.iter().map(|a| a * a).sum::<f64>() / hidden as f64;
                    let s = 1.0 / (ms + f64::from(eps)).sqrt();
                    for d in 0..hidden {
                        let want = v[d] * s * f64::from(gains[j * hidden + d]);
                        worst = worst.max((f64::from(row[d]) - want).abs());
                    }
                    if j + 1 == mixes && t + 1 == n {
                        assert_eq!(row, &head[..], "the head's mix of the last row in last");
                    }
                }
            }
            assert!(
                got[mixes * group * hidden..].iter().all(|&v| v == GUARD),
                "the tail"
            );
        }
        println!("fuse: max |card - host| {worst:.3e}");
        assert!(worst <= 1e-5, "fuse against the host rule: max {worst:.3e}");
    }
}
