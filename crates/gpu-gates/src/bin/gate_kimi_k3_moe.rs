//! GPU gate for Kimi K3's routed block (`arch::kimi_k3::moe`: the common routed block of
//! `routed_ffn` with the latent pair, the routed-sum norm and the gated shared stack), against an
//! f64 reference written line by line from llama.cpp `src/models/kimi-k3.cpp` at
//! cc78da97c0c7596dc821595f21b5644bbdc77df3 (`build_latent_moe` with `build_moe_ffn` at
//! `LLM_FFN_SITU`, `kimi_k3_situ`):
//!
//! ```text
//! xn   = rms_norm(x) · ffn_norm                       p     = σ(ffn_gate_inp · xn)
//! ids  = top-k of p + exp_probs_b                     w     = p[ids] / Σ p[ids] · scale
//! lat  = ffn_routed_down · xn                         o_e   = down_e(situ(gate_e · lat, up_e · lat))
//! s    = Σ_e w_e · o_e                                r     = ffn_routed_up · (rms_norm(s) · ffn_routed_norm)
//! sh   = ffn_down_shexp · situ(ffn_gate_shexp · xn, ffn_up_shexp · xn)
//! y    = x + r + sh
//! ```
//!
//! Each target (the fixture file's routed layers f1, f2 and f3, and one layer of random tensors at
//! the real widths: hidden 7168, latent 3584, experts 3072 wide, 896 of them with the top 16, a
//! shared stack of 6144) holds these clauses:
//!
//! - `decode`: the block's one-row step against the reference, the largest relative L2 of
//!   `y − x` over the checked tokens within [`BAND`], and each token's picks the reference's (the
//!   same expert set, each expert's weight within [`WEIGHT_TOL`]).
//! - `prefill`: a pass of [`TOKENS`] rows equals [`TOKENS`] steps bit for bit, the picks included
//!   (the owner's pass is each row's one-row bits).
//! - `norm`: the reference without the routed norm lies past [`SEP`] bands of the card's output.
//! - `gate`: the reference with the shared stack's gate factor `β tanh(g/β) σ(g)` replaced by 1 lies
//!   past [`SEP`] bands of the card's output.
//!
//! and the clause `refused` (no weights needed): a latent or expert width that is not a whole
//! number of 256-value super-blocks, an expert count the router instances do not take, an
//! unnormalized router and a swapped `situ` pair are each refused at load by their name.
//!
//! The fixture's weights are its own; the real-width layer's are seeded: Q6_K blocks of random
//! codes under bounded scales, IQ1_S blocks of random codes (`iq_rows::seeded_blocks`) under one
//! `d`, every `d` chosen so the stages' values are of order one, written as a one-layer file that
//! loads as a model file does and is removed after.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_moe", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::kimi_k3::moe::{Moe, MoeDims};
    use bloomery_gpu::iq::IqFormat;
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu_gates::iq_rows::{ref_dequant, seeded_blocks, set_d};
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{
        GateError, GateRng, bits_equal, checks_failed, par_map, ref_model_path, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::dequant_row;
    use gguf::write::{Layout, TensorDecl, Writer};
    use gguf::{GgmlType, Split, Value};
    use model::arch::kimi_k3::hparams::Hparams;
    use std::path::PathBuf;

    /// The host reference's routing: the kept experts' scores, their ids, the weights and the biased gap at the cut.
    type Routing = (Vec<f64>, Vec<usize>, Vec<f64>, f64);
    /// What the card's block returns: the output rows, the picks' ids and their weights.
    type Out = (Vec<f32>, Vec<u32>, Vec<f32>);

    /// Rows of the prefill clause: a pass is at most this many.
    const TOKENS: usize = 8;
    /// The card against the f64 reference, relative L2 of the block's output (`y − x`). Derived: every
    /// K-quant, IQ1_S and IQ2_XXS site reads its input as q8_1 blocks of 128 values, whose rounding
    /// (a step of the block's largest magnitude over 127, uniform within it) is 0.6 % of the
    /// input's RMS for Gaussian blocks. The routed path crosses four such sites (the latent down,
    /// the gate and up as one, the down, the latent up) and the shared path two, each adding its
    /// noise to the carried error and the situ's slope (at most 1.5) scaling it, so a branch's
    /// relative error is about 2.2 · 0.6 % = 1.3 %, the two branches' sum about 1.5 % at worst. The
    /// band is three times that, rounded up. The weights are exact in both.
    const BAND: f64 = 5e-2;
    /// How many bands a reference missing a clause must lie from the card's output: the routed norm
    /// and the shared stack's gate each move their branch by far more than the whole of it (the
    /// norm by the sum's RMS over the gain, the gate by the ratio of `β tanh(g/β) σ(g)` to 1).
    const SEP: f64 = 4.0;
    /// The picks' weights, card against reference, relative: the card's f32 scores carry the
    /// sigmoid's `expf` (2 ulps) and the router's f32 dot (the logit within 1e-6), ten times
    /// together.
    const WEIGHT_TOL: f64 = 1e-5;
    /// The gap in the biased score between the last kept expert and the first left out that a
    /// token's input must have, so f32 and f64 select the same set: twenty times the card's score
    /// error (1e-5).
    const TIE: f64 = 2e-4;

    /// The real-width layer's widths and the layer the file holds it at.
    const REAL: MoeDims = MoeDims {
        hidden: 7168,
        latent: 3584,
        expert_ff: 3072,
        shared_ff: 6144,
        n_expert: 896,
        n_used: 16,
        scale: 1.0,
        weights_norm: true,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: 25.0,
    };
    const REAL_LAYER: usize = 1;

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        println!(
            "gate_kimi_k3_moe: device {} — band {BAND}, separation {SEP} bands, tie margin {TIE}",
            gpu.device_name()?
        );
        let mut results = Vec::new();
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = Hparams::read(&split).map_err(|e| format!("{}: {e}", path.display()))?;
        let dims = MoeDims::of(&hp);
        println!(
            "  file {}: hidden {}, latent {}, experts {} of {} wide (top {}), shared {}",
            path.display(),
            dims.hidden,
            dims.latent,
            dims.n_expert,
            dims.expert_ff,
            dims.n_used,
            dims.shared_ff
        );
        for l in 1..=3 {
            if hp.dense(l) {
                return Err(format!("layer {l} of {} is dense, not routed", path.display()).into());
            }
            let w = Weights::load(gpu.stream(), &split, l..l + 1, false)?;
            let name = format!("f{l}");
            let t = Target {
                name: &name,
                split: &split,
                w: &w,
                l,
                dims,
                checked: TOKENS,
            };
            results.extend(target_clauses(&gpu, &t)?);
            if l == 1 {
                results.extend(refusal_clauses(&gpu, &t)?);
            }
        }
        drop(split);
        let file = RealFile::write(&REAL)?;
        let split = Split::open(&file.0).map_err(|e| format!("open {}: {e}", file.0.display()))?;
        let w = Weights::load(gpu.stream(), &split, REAL_LAYER..REAL_LAYER + 1, false)?;
        let t = Target {
            name: "real",
            split: &split,
            w: &w,
            l: REAL_LAYER,
            dims: REAL,
            checked: 1,
        };
        results.extend(target_clauses(&gpu, &t)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!("{}", tier::tally_line());
        println!(
            "gate_kimi_k3_moe: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// One layer of a model file: the tensors the block reads, its widths and how many of the
    /// tokens the reference checks.
    #[derive(Clone, Copy)]
    struct Target<'a> {
        name: &'a str,
        split: &'a Split,
        w: &'a Weights,
        l: usize,
        dims: MoeDims,
        checked: usize,
    }

    // ------------------------------------------------------------------ host

    /// `‖a − b‖₂ / ‖b‖₂`, infinite for a length that differs or a value that is not finite.
    fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (x - y).powi(2);
            den += y * y;
        }
        if a.len() != b.len() || !num.is_finite() {
            return f64::INFINITY;
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `kimi_k3_situ`: `β tanh(g/β) σ(g) · β₂ tanh(u/β₂)`.
    fn situ(g: f64, u: f64, (beta, linear): (f64, f64)) -> f64 {
        beta * (g / beta).tanh() * sigmoid(g) * (linear * (u / linear).tanh())
    }

    /// `ggml_rms_norm` then the gain: `x / √(mean(x²) + eps) · gain`.
    fn rms_norm(x: &[f64], gain: &[f32], eps: f64) -> Vec<f64> {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter()
            .zip(gain)
            .map(|(v, g)| v * inv * f64::from(*g))
            .collect()
    }

    /// The root mean square of `x`.
    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// The row `bytes` of type `ty` as f32: each i-quant by its rule (`iq_rows::ref_dequant`), the
    /// rest by `gguf::dequant_row`.
    fn dequant(ty: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<(), GateError> {
        match ty {
            GgmlType::IQ1_S => ref_dequant(IqFormat::Iq1S, bytes, out),
            GgmlType::IQ2_XXS => ref_dequant(IqFormat::Iq2Xxs, bytes, out),
            other => dequant_row(other, bytes, out),
        }
        .map_err(|e| format!("dequant {ty:?}: {e}").into())
    }

    /// Layer `l`'s tensors of a model file, read as the reference reads them: whole, in f64.
    struct Host<'a> {
        split: &'a Split,
        l: usize,
    }

    impl Host<'_> {
        fn name(&self, stem: &str) -> String {
            format!("blk.{}.{stem}", self.l)
        }

        /// The f32 tensor `stem`, `len` values.
        fn f32s(&self, stem: &str, len: usize) -> Result<Vec<f32>, GateError> {
            let name = self.name(stem);
            let (_, info) = self.split.find(&name).ok_or_else(|| format!("no {name}"))?;
            if info.ty != GgmlType::F32 {
                return Err(format!("{name} is {:?}, not F32", info.ty).into());
            }
            let bytes = self.split.data(&name).map_err(|e| format!("{name}: {e}"))?;
            if bytes.len() != 4 * len {
                return Err(format!("{name}: {} bytes, want {} values", bytes.len(), len).into());
            }
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect())
        }

        /// `W · x` for `rows` rows of `k` values of tensor `stem` (expert `e` of a stack, 0 of a
        /// matrix), in f64 over the row's f32 dequantization.
        fn matvec(
            &self,
            stem: &str,
            e: usize,
            (rows, k): (usize, usize),
            x: &[f64],
        ) -> Result<Vec<f64>, GateError> {
            let name = self.name(stem);
            let (_, info) = self.split.find(&name).ok_or_else(|| format!("no {name}"))?;
            let ty = info.ty;
            let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
            if dims.len() < 2 || dims[0] != k || dims[1] != rows || x.len() != k {
                return Err(format!(
                    "{name}: dims {dims:?} for {rows} rows of {k} over an input of {}",
                    x.len()
                )
                .into());
            }
            let (blck, size) = (
                ty.blck_size().ok_or("a sized type")? as usize,
                ty.type_size().ok_or("a sized type")? as usize,
            );
            let row_bytes = k / blck * size;
            let all = self.split.data(&name).map_err(|e| format!("{name}: {e}"))?;
            let base = e * rows * row_bytes;
            let out = par_map(rows, |r| -> Result<f64, String> {
                let mut w = vec![0f32; k];
                let at = base + r * row_bytes;
                dequant(ty, &all[at..at + row_bytes], &mut w).map_err(|e| e.to_string())?;
                Ok(w.iter().zip(x).map(|(a, b)| f64::from(*a) * b).sum())
            });
            out.into_iter()
                .map(|v| v.map_err(GateError::from))
                .collect()
        }
    }

    /// What the reference computes before the routed norm: the normed row, the picks and the
    /// routed experts' weighted sum in the latent.
    struct Front {
        xn: Vec<f64>,
        ids: Vec<usize>,
        w: Vec<f64>,
        /// The biased score's gap between the last kept expert and the first left out.
        gap: f64,
        lat: Vec<f64>,
        sum: Vec<f64>,
    }

    /// The router's choice for `x`: the normed row, the kept experts (the top `n_used` of the
    /// biased scores, a lower index first among equals), their unbiased scores renormalized
    /// and scaled, and the biased gap at the cut.
    fn route(h: &Host, d: &MoeDims, x: &[f32]) -> Result<Routing, GateError> {
        let gain = h.f32s("ffn_norm.weight", d.hidden)?;
        let xf: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let xn = rms_norm(&xf, &gain, f64::from(d.eps));
        let router = h.f32s("ffn_gate_inp.weight", d.n_expert * d.hidden)?;
        let bias = h.f32s("exp_probs_b.bias", d.n_expert)?;
        let p: Vec<f64> = (0..d.n_expert)
            .map(|e| {
                let row = &router[e * d.hidden..(e + 1) * d.hidden];
                sigmoid(row.iter().zip(&xn).map(|(a, b)| f64::from(*a) * b).sum())
            })
            .collect();
        let sel: Vec<f64> = p
            .iter()
            .zip(&bias)
            .map(|(a, b)| a + f64::from(*b))
            .collect();
        let mut order: Vec<usize> = (0..d.n_expert).collect();
        order.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then(a.cmp(&b)));
        let gap = sel[order[d.n_used - 1]] - sel[order[d.n_used]];
        let ids = order[..d.n_used].to_vec();
        let total: f64 = ids.iter().map(|&e| p[e]).sum();
        let w = ids
            .iter()
            .map(|&e| p[e] / total * f64::from(d.scale))
            .collect();
        Ok((xn, ids, w, gap))
    }

    /// The reference up to the routed sum.
    fn front(h: &Host, d: &MoeDims, x: &[f32]) -> Result<Front, GateError> {
        let (xn, ids, w, gap) = route(h, d, x)?;
        let lat = h.matvec("ffn_routed_down.weight", 0, (d.latent, d.hidden), &xn)?;
        let beta = (f64::from(d.situ_beta), f64::from(d.situ_linear_beta));
        let mut sum = vec![0.0f64; d.latent];
        for (&e, &we) in ids.iter().zip(&w) {
            let g = h.matvec("ffn_gate_exps.weight", e, (d.expert_ff, d.latent), &lat)?;
            let u = h.matvec("ffn_up_exps.weight", e, (d.expert_ff, d.latent), &lat)?;
            let act: Vec<f64> = g.iter().zip(&u).map(|(&g, &u)| situ(g, u, beta)).collect();
            let o = h.matvec("ffn_down_exps.weight", e, (d.latent, d.expert_ff), &act)?;
            for (s, o) in sum.iter_mut().zip(&o) {
                *s += we * o;
            }
        }
        Ok(Front {
            xn,
            ids,
            w,
            gap,
            lat,
            sum,
        })
    }

    /// Which clause of the graph the reference leaves out.
    #[derive(Clone, Copy, PartialEq)]
    enum Skip {
        None,
        /// `ffn_routed_norm` (the sum goes to the latent up as it is).
        Norm,
        /// The shared stack's gate factor (replaced by 1).
        Gate,
    }

    /// The block's output `r + sh` over the routed sum, the latent up and the shared stack, with
    /// `skip`'s clause left out.
    fn back(h: &Host, d: &MoeDims, f: &Front, skip: Skip) -> Result<Vec<f64>, GateError> {
        let sn = if skip == Skip::Norm {
            f.sum.clone()
        } else {
            let gain = h.f32s("ffn_routed_norm.weight", d.latent)?;
            rms_norm(&f.sum, &gain, f64::from(d.eps))
        };
        let up = h.matvec("ffn_routed_up.weight", 0, (d.hidden, d.latent), &sn)?;
        let gs = h.matvec("ffn_gate_shexp.weight", 0, (d.shared_ff, d.hidden), &f.xn)?;
        let us = h.matvec("ffn_up_shexp.weight", 0, (d.shared_ff, d.hidden), &f.xn)?;
        let (beta, linear) = (f64::from(d.situ_beta), f64::from(d.situ_linear_beta));
        let act: Vec<f64> = gs
            .iter()
            .zip(&us)
            .map(|(&g, &u)| {
                if skip == Skip::Gate {
                    linear * (u / linear).tanh()
                } else {
                    situ(g, u, (beta, linear))
                }
            })
            .collect();
        let sh = h.matvec("ffn_down_shexp.weight", 0, (d.hidden, d.shared_ff), &act)?;
        Ok(up.iter().zip(&sh).map(|(a, b)| a + b).collect())
    }

    /// `n` inputs of `hidden` values, each one whose routing has a biased gap at the cut of at
    /// least [`TIE`], drawn from `seed` upward.
    fn tokens(h: &Host, d: &MoeDims, n: usize, seed: u64) -> Result<Vec<Vec<f32>>, GateError> {
        let mut out = Vec::with_capacity(n);
        let mut rng = GateRng(seed | 1);
        let mut tried = 0usize;
        while out.len() < n {
            let x: Vec<f32> = (0..d.hidden).map(|_| rng.uni(-1.5, 1.5)).collect();
            let (_, _, _, gap) = route(h, d, &x)?;
            tried += 1;
            if gap >= TIE {
                out.push(x);
            } else if tried > 64 * n {
                return Err(
                    format!("no input with a routing gap of {TIE} in {tried} draws").into(),
                );
            }
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- device

    /// One block step over `x`: the output row and the picks (`ids`, weights).
    fn step(gpu: &Gpu, moe: &mut Moe, w: &Weights, x: &[f32]) -> Result<Out, GateError> {
        let s = gpu.stream();
        let xd = DeviceBuffer::from_host(s, x)?;
        let mut yd = DeviceBuffer::<f32>::zeroed(s, x.len())?;
        moe.step(gpu, w, &xd, &mut yd)?;
        s.synchronize()?;
        let (ids, wts) = moe.picks(gpu, false)?;
        Ok((yd.to_host_vec(s)?, ids, wts))
    }

    /// A pass over `m` rows of `x`: the output rows and the picks of every row.
    fn rows(
        gpu: &Gpu,
        moe: &mut Moe,
        w: &Weights,
        (m, x): (usize, &[f32]),
    ) -> Result<Out, GateError> {
        let s = gpu.stream();
        let xd = DeviceBuffer::from_host(s, x)?;
        let mut yd = DeviceBuffer::<f32>::zeroed(s, x.len())?;
        moe.rows(gpu, w, (m, &xd), &mut yd)?;
        s.synchronize()?;
        let (ids, wts) = moe.picks(gpu, true)?;
        Ok((yd.to_host_vec(s)?, ids, wts))
    }

    /// Whether a card pick equals the reference's: the same expert set, each expert's weight within
    /// [`WEIGHT_TOL`] relative. Returns the pass and the largest weight error.
    fn picks_match(f: &Front, ids: &[u32], wts: &[f32]) -> (bool, f64) {
        let mut worst = 0.0f64;
        let mut same_set = ids.len() == f.ids.len();
        for (&e, &we) in f.ids.iter().zip(&f.w) {
            match ids.iter().position(|&i| i as usize == e) {
                Some(s) => worst = worst.max((f64::from(wts[s]) - we).abs() / we),
                None => same_set = false,
            }
        }
        (same_set && worst <= WEIGHT_TOL, worst)
    }

    /// The block's output of a step against `x`: `y − x` in f64.
    fn delta(y: &[f32], x: &[f32]) -> Vec<f64> {
        y.iter()
            .zip(x)
            .map(|(&a, &b)| f64::from(a) - f64::from(b))
            .collect()
    }

    fn target_clauses(gpu: &Gpu, t: &Target) -> Result<Vec<bool>, GateError> {
        let Target {
            name,
            split,
            w,
            l,
            dims,
            checked,
        } = *t;
        let mut out = Vec::new();
        let mut moe = Moe::load(gpu, split, w, l, dims)?;
        let h = Host { split, l };
        let xs = tokens(&h, &dims, TOKENS, 0x6b33_0000 + l as u64)?;
        let mut ys = Vec::new();
        let mut picks = Vec::new();
        for x in &xs {
            let (y, ids, wts) = step(gpu, &mut moe, w, x)?;
            ys.push(y);
            picks.push((ids, wts));
        }
        let fault = gpu.fault()?;
        // decode
        let cname = format!("{name} decode");
        if tier::run_clause(&cname, Tag::SelfConsistency)? {
            let (mut worst, mut picks_ok, mut wmax) = (0.0f64, true, 0.0f64);
            let mut shown = None;
            for i in 0..checked {
                let f = front(&h, &dims, &xs[i])?;
                let r = back(&h, &dims, &f, Skip::None)?;
                let rel = rel_l2(&delta(&ys[i], &xs[i]), &r);
                let (ok, werr) = picks_match(&f, &picks[i].0, &picks[i].1);
                worst = worst.max(rel);
                picks_ok &= ok;
                wmax = wmax.max(werr);
                if i == 0 {
                    shown = Some((rms(&f.lat), rms(&f.sum), rms(&r), f.gap));
                }
            }
            let pass = worst <= BAND && picks_ok && fault.is_none();
            println!(
                "  {cname}: {checked} token(s), relative L2 of y−x {worst:.3e} (band {BAND:.0e}), picks \
                 the reference's {picks_ok} (weights within {wmax:.2e}), fault {fault:?}, token 0: \
                 rms lat/sum/out {shown:?}: {}",
                verdict(pass)
            );
            out.push(pass);
        }
        // prefill
        let cname = format!("{name} prefill");
        if tier::run_clause(&cname, Tag::SelfConsistency)? {
            let flat: Vec<f32> = xs.iter().flatten().copied().collect();
            let (yr, ids, wts) = rows(gpu, &mut moe, w, (TOKENS, &flat))?;
            let steps: Vec<f32> = ys.iter().flatten().copied().collect();
            let step_ids: Vec<u32> = picks.iter().flat_map(|p| p.0.clone()).collect();
            let step_wts: Vec<f32> = picks.iter().flat_map(|p| p.1.clone()).collect();
            let same_out = bits_equal(&yr, &steps);
            let same_pick = ids == step_ids && bits_equal(&wts, &step_wts);
            let rel = rel_l2(
                &yr.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                &steps.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            );
            let pass = same_out && same_pick && gpu.fault()?.is_none();
            println!(
                "  {cname}: {TOKENS} rows = {TOKENS} steps, output bits {same_out} (relative L2 \
                 {rel:.3e}), picks bits {same_pick}: {}",
                verdict(pass)
            );
            out.push(pass);
        }
        // the separations, on token 0
        let f = front(&h, &dims, &xs[0])?;
        let dev = delta(&ys[0], &xs[0]);
        for (clause, skip) in [("norm", Skip::Norm), ("gate", Skip::Gate)] {
            let cname = format!("{name} {clause}");
            if tier::run_clause(&cname, Tag::SelfConsistency)? {
                let r = back(&h, &dims, &f, skip)?;
                let rel = rel_l2(&dev, &r);
                let pass = rel > SEP * BAND;
                println!(
                    "  {cname}: the reference without it lies {:.1} bands from the card (past {SEP}): {}",
                    rel / BAND,
                    verdict(pass)
                );
                out.push(pass);
            }
        }
        Ok(out)
    }

    /// Doctored widths of the target `t` that `Moe::load` refuses by name: each message holds the
    /// text.
    fn refusal_clauses(gpu: &Gpu, t: &Target) -> Result<Vec<bool>, GateError> {
        let d = t.dims;
        let cases = [
            (
                "a latent that is not whole super-blocks",
                MoeDims {
                    latent: d.latent - 128,
                    ..d
                },
                "super-blocks of 256",
            ),
            (
                "an expert width that is not whole super-blocks",
                MoeDims {
                    expert_ff: d.expert_ff + 32,
                    ..d
                },
                "super-blocks of 256",
            ),
            (
                "an expert count no router instance has",
                MoeDims { n_expert: 7, ..d },
                "instances route over",
            ),
            (
                "a kept count no router instance has",
                MoeDims { n_used: 3, ..d },
                "instances route over",
            ),
            (
                "an unnormalized router",
                MoeDims {
                    weights_norm: false,
                    ..d
                },
                "renormalizes",
            ),
            (
                "a situ pair swapped",
                MoeDims {
                    situ_beta: d.situ_linear_beta,
                    situ_linear_beta: d.situ_beta,
                    ..d
                },
                "situ runs at linear beta",
            ),
        ];
        let mut out = Vec::new();
        let cname = format!("{} refused", t.name);
        if tier::run_clause(&cname, Tag::SelfConsistency)? {
            for (what, dims, text) in cases {
                let got = match Moe::load(gpu, t.split, t.w, t.l, dims) {
                    Ok(_) => None,
                    Err(e) => Some(e.to_string()),
                };
                let pass = got.as_deref().is_some_and(|m| m.contains(text));
                println!(
                    "  {cname}: {what}: refused with {text:?}: {} ({})",
                    verdict(pass),
                    got.as_deref().unwrap_or("loaded")
                );
                out.push(pass);
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------- the file

    /// Bits of the positive normal f16 nearest below `v`.
    fn f16_bits(v: f32) -> u16 {
        let b = v.to_bits();
        let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
        assert!((1..31).contains(&e), "{v} is not a normal f16");
        ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16
    }

    /// `n` Q6_K blocks of random codes: scales in −6..=6 (RMS 3.7), the 6-bit codes uniform (RMS
    /// 18.5 about their zero point), `d` as given.
    fn q6k_blocks(rng: &mut GateRng, n: usize, d: f32) -> Vec<u8> {
        let mut v = vec![0u8; n * 210];
        for blk in v.as_chunks_mut::<210>().0.iter_mut() {
            for b in &mut blk[..192] {
                *b = (rng.unit() * 256.0) as u8;
            }
            for s in &mut blk[192..208] {
                *s = (rng.below(13) as i32 - 6) as i8 as u8;
            }
            blk[208..].copy_from_slice(&f16_bits(d).to_le_bytes());
        }
        v
    }

    /// The `d` that gives a Q6_K site's outputs a standard deviation near `target` over `k` inputs
    /// of RMS 1: the weights' RMS is `d · 3.7 · 18.5`.
    fn q6k_d(k: usize, target: f32) -> f32 {
        target / (68.5 * (k as f32).sqrt())
    }

    /// The `d` for IQ1_S blocks (weights' RMS about `7.4 d`) over `k` inputs of RMS `rms_in`.
    fn iq1s_d(k: usize, rms_in: f32, target: f32) -> f32 {
        target / (7.4 * (k as f32).sqrt() * rms_in)
    }

    /// A seeded F32 tensor of `n` values in `lo..hi`.
    fn f32_bytes(rng: &mut GateRng, n: usize, (lo, hi): (f32, f32)) -> Vec<u8> {
        (0..n).flat_map(|_| rng.uni(lo, hi).to_le_bytes()).collect()
    }

    /// The temporary one-layer file at the real widths, removed when dropped.
    struct RealFile(PathBuf);

    impl Drop for RealFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// One tensor of the real-width file: its stem, ggml dims and type.
    type Spec = (&'static str, Vec<usize>, GgmlType);

    impl RealFile {
        fn write(d: &MoeDims) -> Result<RealFile, GateError> {
            use GgmlType::{F32, IQ1_S, Q6_K};
            let l = REAL_LAYER;
            let (n, lat, ff, sh, e) = (d.hidden, d.latent, d.expert_ff, d.shared_ff, d.n_expert);
            let specs: Vec<Spec> = vec![
                ("ffn_norm.weight", vec![n], F32),
                ("ffn_gate_inp.weight", vec![n, e], F32),
                ("exp_probs_b.bias", vec![e], F32),
                ("ffn_gate_exps.weight", vec![lat, ff, e], IQ1_S),
                ("ffn_up_exps.weight", vec![lat, ff, e], IQ1_S),
                ("ffn_down_exps.weight", vec![ff, lat, e], IQ1_S),
                ("ffn_routed_down.weight", vec![n, lat], Q6_K),
                ("ffn_routed_up.weight", vec![lat, n], Q6_K),
                ("ffn_routed_norm.weight", vec![lat], F32),
                ("ffn_gate_shexp.weight", vec![n, sh], Q6_K),
                ("ffn_up_shexp.weight", vec![n, sh], Q6_K),
                ("ffn_down_shexp.weight", vec![sh, n], Q6_K),
            ];
            let nbytes = |ty: GgmlType, dims: &[usize]| -> Result<u64, GateError> {
                let row = ty.row().ok_or("a sized type")?;
                let dims: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
                gguf::tensor_bytes(row.blck, row.bytes, &dims).ok_or_else(|| "a byte count".into())
            };
            let decls = specs
                .iter()
                .map(|(stem, dims, ty)| {
                    Ok(TensorDecl {
                        name: format!("blk.{l}.{stem}"),
                        dims: dims.iter().map(|&d| d as u64).collect(),
                        type_id: ty.as_u32(),
                        nbytes: nbytes(*ty, dims)?,
                    })
                })
                .collect::<Result<Vec<_>, GateError>>()?;
            let keys = vec![
                (
                    gguf::GENERAL_ARCHITECTURE.to_owned(),
                    Value::String("kimi-k3".to_owned()),
                ),
                ("kimi-k3.block_count".to_owned(), Value::U32(l as u32 + 1)),
            ];
            let layout = Layout::new(&keys, decls).map_err(|e| format!("layout: {e}"))?;
            let path =
                std::env::temp_dir().join(format!("kimi-k3-moe-real-{}.gguf", std::process::id()));
            let file = RealFile(path);
            let out = std::io::BufWriter::new(
                std::fs::File::create(&file.0)
                    .map_err(|e| format!("create {}: {e}", file.0.display()))?,
            );
            let mut wr = Writer::new(out, layout).map_err(|e| format!("writer: {e}"))?;
            let mut rng = GateRng(0x6b33_7265_616c_0001);
            let logit_half = 1.0f32 * 3f32.sqrt() / (n as f32).sqrt();
            for (stem, dims, _) in &specs {
                let name = format!("blk.{l}.{stem}");
                let len: usize = dims.iter().product();
                let written = match *stem {
                    "ffn_norm.weight" | "ffn_routed_norm.weight" => {
                        wr.tensor(&name, &f32_bytes(&mut rng, len, (0.5, 1.5)))
                    }
                    "ffn_gate_inp.weight" => {
                        wr.tensor(&name, &f32_bytes(&mut rng, len, (-logit_half, logit_half)))
                    }
                    "exp_probs_b.bias" => {
                        wr.tensor(&name, &f32_bytes(&mut rng, len, (-0.05, 0.05)))
                    }
                    "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
                        // Gate and up read the latent row (RMS 1); the down reads the situ's
                        // output (RMS about 1.3 at a gate and up of standard deviation 1.5).
                        let (k, rms_in, target) = if *stem == "ffn_down_exps.weight" {
                            (ff, 1.3, 0.4)
                        } else {
                            (lat, 1.0, 1.5)
                        };
                        let d6 = iq1s_d(k, rms_in, target);
                        let per = dims[1] * dims[0] / 256;
                        let seed = rng.below(1 << 30) as u64;
                        let bits = f16_bits(d6);
                        let experts = dims[2];
                        let parts = (0..experts).step_by(32).map(|e0| {
                            par_map(32.min(experts - e0), |i| {
                                let mut v = seeded_blocks(
                                    IqFormat::Iq1S,
                                    per,
                                    seed ^ ((e0 + i) as u64 + 1)
                                        .wrapping_mul(0x9e37_79b9_7f4a_7c15),
                                );
                                for blk in v.chunks_exact_mut(IqFormat::Iq1S.block_bytes()) {
                                    set_d(IqFormat::Iq1S, blk, bits);
                                }
                                v
                            })
                            .concat()
                        });
                        wr.tensor_parts(&name, parts)
                    }
                    "ffn_routed_down.weight" => {
                        wr.tensor(&name, &q6k_blocks(&mut rng, len / 256, q6k_d(n, 1.0)))
                    }
                    "ffn_routed_up.weight" => {
                        wr.tensor(&name, &q6k_blocks(&mut rng, len / 256, q6k_d(lat, 1.0)))
                    }
                    "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" => {
                        wr.tensor(&name, &q6k_blocks(&mut rng, len / 256, q6k_d(n, 1.5)))
                    }
                    "ffn_down_shexp.weight" => {
                        wr.tensor(&name, &q6k_blocks(&mut rng, len / 256, q6k_d(sh, 1.0)))
                    }
                    other => return Err(format!("no generator for {other}").into()),
                };
                written.map_err(|e| format!("write {name}: {e}"))?;
            }
            wr.finish().map_err(|e| format!("finish: {e}"))?;
            println!(
                "  real-width file {}: {} MiB",
                file.0.display(),
                std::fs::metadata(&file.0).map_or(0, |m| m.len()) >> 20
            );
            Ok(file)
        }
    }
}
