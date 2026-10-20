//! Solar's KDA projections over typed sites: the one part of the KDA mixer
//! the file's types force (Q4_K q, k, `f_a`, `g_a`, β and out, v Q4_K or
//! Q6_K by layer; Q5_0 `f_b`, `g_b`, K = 128). One q8_1 of `xn` serves the
//! six input sites that read it; one Q8Blocks32 form each of `fa` and `ga`
//! the two Q5_0 halves.
//!
//! The step's projections are gemvs over one row ([`Sites`] at `m = 1`); a
//! prompt batch's are GEMMs over its rows and are the batch's own impl of
//! [`KdaProj`].

use bloomery_gpu::arch::hybrid::{Acts, GqaCx};
use bloomery_gpu::linear::mixer_kda::{KdaIn, KdaProj};
use bloomery_gpu::linear::{HEAD, LinearShape};
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::span::span_mut;
use cuda_core::DeviceBuffer;

use crate::body::KdaSites;

/// What the refusals of this file name.
const WHAT: &str = "solar2 kda::Sites";

/// A layer's sites, the activation forms they read and the launch context.
pub struct Sites<'a> {
    pub cx: &'a GqaCx<'a>,
    pub sites: &'a KdaSites,
    /// The mixer's head counts: the widths of q, k and v in the joined rows.
    pub shape: LinearShape,
    pub acts: &'a mut Acts,
}

/// The step is one row: a site's gemv runs one column.
fn one_row(l: usize, m: usize) -> Result<(), GpuError> {
    if m == 1 {
        return Ok(());
    }
    Err(GpuError::Shape {
        what: WHAT,
        detail: format!("layer {l}: {m} rows through the step's gemvs; a batch runs the GEMM impl"),
    })
}

impl KdaProj for Sites<'_> {
    /// q, k and v into the joined rows, `f_a`, `g_a` and β's raw rows, all
    /// from the one q8_1 form of `xn`; then `f = f_b · fa` and `z = g_b · ga`,
    /// each half quantized into the Q8Blocks32 form its Q5_0 site reads.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        one_row(l, m)?;
        let (cx, s) = (self.cx, self.sites);
        {
            let act = self.acts.q8_1_at(WHAT, cx.hidden, 1)?;
            gpu.enqueue_quantize_q8_1_layer(xn, act, l)?;
            let (qk, v) = (self.shape.n_k * HEAD, self.shape.n_v * HEAD);
            let mut at = 0;
            for (site, rows) in s.qkv.iter().zip([qk, qk, v]) {
                let mut part = span_mut(WHAT, &mut *y.qkv, at, rows)?;
                site.kgemv(cx, act, l, &mut part)?;
                at += rows;
            }
            s.f_a.kgemv(cx, act, l, y.fa)?;
            s.g_a.kgemv(cx, act, l, y.ga)?;
            s.beta.kgemv(cx, act, l, y.beta_raw)?;
        }
        s.f_b.project(cx, self.acts, l, y.fa, y.f)?;
        s.g_b.project(cx, self.acts, l, y.ga, y.z)
    }

    /// `out = o_proj · gated`: the site quantizes `gated` into its own form.
    fn output(
        &mut self,
        _gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        one_row(l, m)?;
        self.sites.out.project(self.cx, self.acts, l, gated, out)
    }
}
