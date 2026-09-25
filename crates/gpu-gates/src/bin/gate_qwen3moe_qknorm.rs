//! GPU gate for qwen3moe's per-head RMS norm of the query and key heads
//! (`attn_q_norm` / `attn_k_norm`), the first half of
//! `rope_neox::head_norm_neox_append`, against ik's CPU dumps. The kernel
//! runs with the identity table (every cos 1, every sin 0), under which the
//! turn returns each normalized value unchanged, so its query and key heads
//! are the norm alone.
//!
//! Three layers per (set, layer):
//! 1. the kernel against this binary's transcription of our rule
//!    (`qwen3moe::head_norm`) — bit-identical — and a rerun, bit-identical;
//! 2. ik's rule (`ik_norm::fused`, the f64 serial sum) on the dumped input
//!    against `Qcur_normed-L`/`Kcur_normed-L` — bit-identical: the semantics
//!    (per-head rows, the gain per head, eps) proven on ik's own values;
//! 3. the kernel against the dump within [`NORM_BAND`] per value.
//!
//! Sets: every qwen3moe set (the 5-token prefill and the decode steps).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_qknorm: built without the `gpu` feature; see `just gate-gpu-qwen3moe-qknorm`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_qknorm", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::rope_neox::RopeNeoxKernels;
    use bloomery_gpu_gates::qwen3moe::dev::run as run_neox;
    use bloomery_gpu_gates::qwen3moe::{AttnRows, HEAD, head_norm, sets};
    use bloomery_gpu_gates::rounding::U_F32;
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, ik_norm, open_split, same_bits, split_f32, verdict,
    };
    use model::arch::Arch;
    use model::arch::qwen3moe::hparams::Hparams;

    /// Band for a normalized value against ik's, relative to ik's value. The
    /// rules differ in the order of the f64 sum of the same 128 f32 squares:
    /// each side's 127 additions err by at most `2^-53` of the sum, so the
    /// two sums are within `254·2^-53` of each other and the means, each
    /// rounded once to f32, are equal or one f32 ulp apart — at most `2u`
    /// relative. Each later op rounds both sides once more on inputs that
    /// already differ: `+ eps` (a positive constant, which only shrinks a
    /// relative difference) `2u + 2u`; the square root halves that and adds
    /// `2u` (`4u`); the reciprocal `6u`; `· gain` `8u`; `· x` `10u`.
    const NORM_BAND: f32 = 10.0 * U_F32;

    pub fn run() -> Result<(), GateError> {
        let split = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-qknorm")?;
        let hp = Hparams::read(&split)?;
        let gpu = Gpu::new()?;
        let k = RopeNeoxKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        println!(
            "gate_qwen3moe_qknorm: device {} — {} layers, {} q / {} kv heads of {}, eps {:e}",
            gpu.device_name()?,
            hp.n_layer,
            hp.n_head,
            hp.n_head_kv,
            hp.head_dim,
            hp.rms_eps
        );
        let (mut sites, mut failed) = (0u32, 0u32);
        for (label, man) in sets()? {
            let mut worst = 0.0f32;
            let (mut same_q, mut same_k, mut n_q, mut n_k) = (0usize, 0usize, 0usize, 0usize);
            for layer in 0..hp.n_layer {
                let rows = AttnRows::read(&man, layer)?
                    .ok_or_else(|| format!("{label}: no Qcur_normed-{layer}"))?;
                let gq = split_f32(&split, &rows.gq_name, HEAD)?;
                let gk = split_f32(&split, &rows.gk_name, HEAD)?;
                let identity: Vec<f32> = (0..rows.m * HEAD / 2).flat_map(|_| [1.0, 0.0]).collect();
                let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);

                // Layer 1: the kernel against our rule, and a rerun.
                let a = run_neox(&k, stream, unl, &rows, &gq, &gk, &identity, hp.rms_eps, ctx)?;
                let b = run_neox(&k, stream, unl, &rows, &gq, &gk, &identity, hp.rms_eps, ctx)?;
                let host = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| head_norm(h, g, hp.rms_eps))
                        .collect()
                };
                let (hq, hk) = (host(&rows.q, &gq), host(&rows.k, &gk));
                let exact = bits_equal(&a.q, &hq) && bits_equal(&a.k, &hk);
                let rerun = bits_equal(&a.q, &b.q) && bits_equal(&a.k, &b.k);

                // Layer 2: ik's rule against the dump.
                let sim = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| ik_norm::fused(h, g, hp.rms_eps))
                        .collect()
                };
                let sim_q = same_bits(&sim(&rows.q, &gq), &rows.q_normed);
                let sim_k = same_bits(&sim(&rows.k, &gk), &rows.k_normed);
                let sim_ok = sim_q == rows.q_normed.len() && sim_k == rows.k_normed.len();

                // Layer 3: the kernel against the dump, per value.
                let rel = |y: &[f32], want: &[f32]| -> f32 {
                    y.iter().zip(want).fold(0.0f32, |acc, (&a, &w)| {
                        let d = (a - w).abs();
                        let r = if w == 0.0 {
                            if d == 0.0 { 0.0 } else { f32::INFINITY }
                        } else {
                            d / w.abs()
                        };
                        acc.max(r)
                    })
                };
                let rq = rel(&a.q, &rows.q_normed);
                let rk = rel(&a.k, &rows.k_normed);
                let sq = same_bits(&a.q, &rows.q_normed);
                let sk = same_bits(&a.k, &rows.k_normed);
                let pass = exact && rerun && sim_ok && rq <= NORM_BAND && rk <= NORM_BAND;
                worst = worst.max(rq).max(rk);
                same_q += sq;
                same_k += sk;
                n_q += rows.q_normed.len();
                n_k += rows.k_normed.len();
                sites += 1;
                failed += u32::from(!pass);
                if !pass || layer == 0 || layer + 1 == hp.n_layer {
                    println!(
                        "qknorm set={label} layer={layer} m={} pos={:?} bit_exact_host={exact} \
                         bit_identical_rerun={rerun} ik_sim_same q={sim_q}/{} k={sim_k}/{} \
                         same q={sq}/{} k={sk}/{} max_rel q={rq:.3e} k={rk:.3e} (band {NORM_BAND:.3e}) {}",
                        rows.m,
                        rows.pos,
                        rows.q_normed.len(),
                        rows.k_normed.len(),
                        rows.q_normed.len(),
                        rows.k_normed.len(),
                        verdict(pass)
                    );
                }
            }
            println!(
                "set {label}: {} (build {}) — {} layers, kernel = ik bit for bit on q {same_q}/{n_q} \
                 k {same_k}/{n_k}, worst rel {worst:.3e}",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-"),
                hp.n_layer
            );
        }
        let pass = failed == 0;
        println!(
            "gate_qwen3moe_qknorm: {sites} (set, layer) sites, {failed} failed — {}",
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }
}
