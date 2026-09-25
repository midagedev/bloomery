//! GPU gate for qwen3moe's NEOX rope over the whole 128-value head and the
//! K/V append — `rope_neox::head_norm_neox_append` with the model's table —
//! against ik's CPU dumps (`Qcur_roped-L`, `Kcur_roped-L`, and the two cache
//! writes `cache_k_lL (view) (copy of Kcur_roped-L)` and `v_cache_view-L
//! (copy of Vcur-L)`).
//!
//! Three layers per (set, layer):
//! 1. the kernel against this binary's transcription of our rule — the norm
//!    (`qwen3moe::head_norm`), the turn (`qwen3moe::neox_rotate`) on the
//!    engine's table, each plane row the f16 of its value, every other plane
//!    slot untouched — bit-identical, and a rerun bit-identical;
//! 2. ik's rule against the dump: the table from `ggml_rope_cache` (ggml's
//!    recipe at `ne0` = 128), the NEOX turn on ik's own normed rows in the
//!    fused form its compiled loop takes (`fma(x0, c, −(x1·s))`,
//!    `fma(x0, s, x1·c)`; the unfused turn is not ik's on small values),
//!    the append the f16 of ik's roped K and of its V — within one
//!    ulp for the turn (its libm, as for V4.1's rope), bit for bit for the
//!    append;
//! 3. the kernel against the dump: the turned heads within [`ROPE_BAND`] of
//!    the largest value, the V rows bit for bit, and each K row equal to
//!    ik's wherever the two f32 values it rounds agree.
//!
//! Sets: every qwen3moe set; positions 0–4 (prefill), 4, 1,024, 4,096.
//! Positions past those are the table's host unit test
//! (`bloomery_gpu::rope_table`, `just gate-gpu-lib`).
//!
//! And once, a position at or past the cache: the prefill set's layer 0 with
//! its last token's position moved to `ctx`, launched with layer 13's sink.
//! That token is appended nowhere (its rows keep the sentinel) and the launch
//! raises `FaultSite::CachePos` with that layer; the turned heads and every
//! other plane slot are the clean run's bit for bit; the word is clean before
//! and after a clean run.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_rope: built without the `gpu` feature; see `just gate-gpu-qwen3moe-rope`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_rope", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::rope_neox::RopeNeoxKernels;
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use bloomery_gpu::{Fault, FaultSite, Gpu};
    use bloomery_gpu_gates::qwen3moe::dev::{SENTINEL, run as run_neox, run_graph};
    use bloomery_gpu_gates::qwen3moe::{AttnRows, HEAD, head_norm, neox_rotate, sets};
    use bloomery_gpu_gates::rounding::U_F32;
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, max_ulps, open_split, same_bits, split_f32, verdict,
    };
    use gguf::Split;
    use gguf::quant::f32_to_f16_bits;
    use model::arch::Arch;
    use model::arch::qwen3moe::hparams::Hparams;

    /// Band for a turned head against ik's, as `max|Δ| / M` with `M` the
    /// largest `|value|` ik wrote. The normalized values differ by at most
    /// `10u` of themselves (`gate_qwen3moe_qknorm`'s band). A pair then
    /// turns in the same fused form on both sides: each term carries that
    /// `10u`, the inner product rounds on both sides (`12u` of `|x0·c|`,
    /// `|x1·s|` at most), the fused add rounds once on both (`2u` of the
    /// result), and `|x0·c| + |x1·s| <= |(x0, x1)| <= √2·M` because
    /// `c² + s² = 1`: `(12·√2 + 2)·u < 20u` of `M`. The same inputs give the
    /// same bits (layer 2 proves the table and the turn are ik's), so the
    /// band only ever spends what the norm moved.
    const ROPE_BAND: f32 = 20.0 * U_F32;

    /// Band for ik's turn simulated here against ik's own dump, in ulps: our
    /// table is ggml's recipe on the same libm and the turn is its compiled
    /// form, so the derived distance is 0; the one ulp is what a trig or
    /// `powf` from another libm would move.
    const ROPE_ULP_BAND: u32 = 1;

    pub fn run() -> Result<(), GateError> {
        let split = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-rope")?;
        let hp = Hparams::read(&split)?;
        let spec = RopeSpec::window(hp.rope.base, hp.rope.dims);
        if hp.rope.dims != HEAD || hp.head_dim != HEAD {
            return Err(format!(
                "the kernel turns whole {HEAD}-value heads; the file ropes {} of {}",
                hp.rope.dims, hp.head_dim
            )
            .into());
        }
        let table = RopeTable::new(&spec)?;
        let gpu = Gpu::new()?;
        let k = RopeNeoxKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        println!(
            "gate_qwen3moe_rope: device {} — rope {spec:?}, {} layers",
            gpu.device_name()?,
            hp.n_layer
        );
        let (mut sites, mut failed) = (0u32, 0u32);
        for (label, man) in sets()? {
            let mut worst = 0.0f32;
            let mut pos_seen = Vec::new();
            for layer in 0..hp.n_layer {
                let rows = AttnRows::read(&man, layer)?
                    .ok_or_else(|| format!("{label}: no Qcur_normed-{layer}"))?;
                pos_seen.clone_from(&rows.pos);
                let gq = split_f32(&split, &rows.gq_name, HEAD)?;
                let gk = split_f32(&split, &rows.gk_name, HEAD)?;
                let mut cs = Vec::with_capacity(rows.m * HEAD);
                for &p in &rows.pos {
                    table.push(p, Direction::Forward, &mut cs);
                }
                let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
                let (n_kv, m) = (rows.n_kv, rows.m);

                // Layer 1: the kernel against our rule, and a rerun.
                let a = run_neox(&k, stream, unl, &rows, &gq, &gk, &cs, hp.rms_eps, ctx)?;
                let b = run_neox(&k, stream, unl, &rows, &gq, &gk, &cs, hp.rms_eps, ctx)?;
                let norm = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| head_norm(h, g, hp.rms_eps))
                        .collect()
                };
                let hq = neox_rotate(&norm(&rows.q, &gq), &cs, rows.n_head);
                let hk = neox_rotate(&norm(&rows.k, &gk), &cs, n_kv);
                let exact = bits_equal(&a.q, &hq) && bits_equal(&a.k, &hk);
                let mut want_k = vec![SENTINEL; n_kv * ctx * HEAD];
                let mut want_v = want_k.clone();
                for (t, &p) in rows.pos.iter().enumerate() {
                    for h in 0..n_kv {
                        let src = (t * n_kv + h) * HEAD;
                        let dst = (h * ctx + p as usize) * HEAD;
                        for d in 0..HEAD {
                            want_k[dst + d] = f32_to_f16_bits(hk[src + d]);
                            want_v[dst + d] = f32_to_f16_bits(rows.v[src + d]);
                        }
                    }
                }
                let planes_exact = a.cache_k == want_k && a.cache_v == want_v;
                let rerun = bits_equal(&a.q, &b.q)
                    && bits_equal(&a.k, &b.k)
                    && a.cache_k == b.cache_k
                    && a.cache_v == b.cache_v;

                // Layer 2: ik's rule against the dump.
                let mut ik_cs = Vec::with_capacity(m * HEAD);
                for &p in &rows.pos {
                    ik_cs.extend(ggml_rope_cache(&spec, p, HEAD, Direction::Forward));
                }
                let table_is_recipe = bits_equal(&cs, &ik_cs);
                let sim_q = neox_rotate(&rows.q_normed, &ik_cs, rows.n_head);
                let sim_k = neox_rotate(&rows.k_normed, &ik_cs, n_kv);
                let sim_ulps = max_ulps(&sim_q, &rows.q_roped).max(max_ulps(&sim_k, &rows.k_roped));
                let sim_same = same_bits(&sim_q, &rows.q_roped) + same_bits(&sim_k, &rows.k_roped);
                let append_k: Vec<u16> = rows.k_roped.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let append_v: Vec<u16> = rows.v.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let append_same = append_k == rows.k_rows && append_v == rows.v_rows;

                // Layer 3: the kernel against the dump.
                let band = |y: &[f32], want: &[f32]| -> f32 {
                    let mx = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    let d = y
                        .iter()
                        .zip(want)
                        .fold(0.0f32, |a, (&p, &w)| a.max((p - w).abs()));
                    if mx > 0.0 { d / mx } else { d }
                };
                let rel = band(&a.q, &rows.q_roped).max(band(&a.k, &rows.k_roped));
                let same = same_bits(&a.q, &rows.q_roped) + same_bits(&a.k, &rows.k_roped);
                let n = rows.q_roped.len() + rows.k_roped.len();
                let (mut our_k, mut our_v) = (Vec::new(), Vec::new());
                for &p in &rows.pos {
                    for h in 0..n_kv {
                        let dst = (h * ctx + p as usize) * HEAD;
                        our_k.extend_from_slice(&a.cache_k[dst..dst + HEAD]);
                        our_v.extend_from_slice(&a.cache_v[dst..dst + HEAD]);
                    }
                }
                let v_same = our_v == rows.v_rows;
                let k_on_equal =
                    a.k.iter()
                        .zip(&rows.k_roped)
                        .zip(our_k.iter().zip(&rows.k_rows))
                        .all(|((x, y), (c, d))| x.to_bits() != y.to_bits() || c == d);
                let k_same = our_k
                    .iter()
                    .zip(&rows.k_rows)
                    .filter(|(a, b)| a == b)
                    .count();

                let pass = exact
                    && planes_exact
                    && rerun
                    && table_is_recipe
                    && sim_ulps <= ROPE_ULP_BAND
                    && append_same
                    && rel <= ROPE_BAND
                    && v_same
                    && k_on_equal;
                worst = worst.max(rel);
                sites += 1;
                failed += u32::from(!pass);
                if !pass || layer == 0 || layer + 1 == hp.n_layer {
                    println!(
                        "rope set={label} layer={layer} m={m} pos={:?} table_is_recipe={table_is_recipe} \
                         bit_exact_host={exact} planes_exact={planes_exact} bit_identical_rerun={rerun} \
                         ik_sim_same={sim_same}/{n} ik_sim_max_ulp={sim_ulps} ik_append_same={append_same} \
                         same={same}/{n} rel={rel:.3e} (band {ROPE_BAND:.3e}) v_rows_same={v_same} \
                         k_rows_same={k_same}/{} k_rows_on_equal_inputs={k_on_equal} {}",
                        rows.pos,
                        rows.k_rows.len(),
                        verdict(pass)
                    );
                }
            }
            println!(
                "set {label}: {} (build {}) — {} layers at positions {pos_seen:?}, worst rel {worst:.3e}",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-"),
                hp.n_layer
            );
        }
        // The launch as a captured graph: one node, the replay equal to the
        // eager launch on step 4's layer 0.
        let sets = sets()?;
        let (label, man) = &sets[1];
        let rows = AttnRows::read(man, 0)?.ok_or_else(|| format!("{label}: no layer 0"))?;
        let gq = split_f32(&split, &rows.gq_name, HEAD)?;
        let gk = split_f32(&split, &rows.gk_name, HEAD)?;
        let mut cs = Vec::new();
        for &p in &rows.pos {
            table.push(p, Direction::Forward, &mut cs);
        }
        let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
        let eager = run_neox(&k, stream, unl, &rows, &gq, &gk, &cs, hp.rms_eps, ctx)?;
        let (replay, nodes) = run_graph(&gpu, &k, unl, &rows, &gq, &gk, &cs, hp.rms_eps, ctx)?;
        let same = bits_equal(&eager.q, &replay.q)
            && bits_equal(&eager.k, &replay.k)
            && eager.cache_k == replay.cache_k
            && eager.cache_v == replay.cache_v;
        let graph_ok = same && nodes == 1;
        println!(
            "graph op=head_norm_neox_append set={label} layer=0 eager_vs_graph_bit_identical={same} \
             graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        failed += u32::from(!graph_ok);
        failed += u32::from(!cache_pos(&gpu, &k, &split, &table, hp.rms_eps)?);

        let pass = failed == 0;
        println!(
            "gate_qwen3moe_rope: {sites} (set, layer) sites and the graph, {failed} failed — {}",
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// The position clause (module doc): whether it held.
    fn cache_pos(
        gpu: &Gpu,
        k: &RopeNeoxKernels,
        split: &Split,
        table: &RopeTable,
        eps: f32,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sets = sets()?;
        let (label, man) = &sets[0];
        let mut rows = AttnRows::read(man, 0)?.ok_or_else(|| format!("{label}: no layer 0"))?;
        if rows.m < 2 {
            return Err(format!(
                "{label}: the position clause wants several tokens, got {}",
                rows.m
            )
            .into());
        }
        let gq = split_f32(split, &rows.gq_name, HEAD)?;
        let gk = split_f32(split, &rows.gk_name, HEAD)?;
        let mut cs = Vec::new();
        for &p in &rows.pos {
            table.push(p, Direction::Forward, &mut cs);
        }
        let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
        let layer = 13usize;
        let sink = gpu.layer_sink(layer)?;
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::CachePos));
        let before = gpu.fault()?;
        let clean = run_neox(k, stream, sink, &rows, &gq, &gk, &cs, eps, ctx)?;
        let after_clean = gpu.fault()?;
        let bad_t = rows.m - 1;
        let bad_p = rows.pos[bad_t] as usize;
        rows.pos[bad_t] = u32::try_from(ctx)?;
        let bad = run_neox(k, stream, sink, &rows, &gq, &gk, &cs, eps, ctx)?;
        let word = gpu.take_fault()?;
        rows.pos[bad_t] = u32::try_from(bad_p)?;
        let (mut want_k, mut want_v) = (clean.cache_k.clone(), clean.cache_v.clone());
        let mut clean_wrote = true;
        for h in 0..rows.n_kv {
            let row = (h * ctx + bad_p) * HEAD..(h * ctx + bad_p + 1) * HEAD;
            clean_wrote &= clean.cache_k[row.clone()].iter().all(|&b| b != SENTINEL);
            want_k[row.clone()].fill(SENTINEL);
            want_v[row].fill(SENTINEL);
        }
        let planes = bad.cache_k == want_k && bad.cache_v == want_v;
        let heads = bits_equal(&bad.q, &clean.q) && bits_equal(&bad.k, &clean.k);
        let again = run_neox(k, stream, sink, &rows, &gq, &gk, &cs, eps, ctx)?;
        let clean_again = bits_equal(&again.q, &clean.q)
            && again.cache_k == clean.cache_k
            && again.cache_v == clean.cache_v
            && gpu.fault()?.is_none();
        let pass = before.is_none()
            && after_clean.is_none()
            && word == want
            && clean_wrote
            && planes
            && heads
            && clean_again;
        println!(
            "cache_pos set={label} layer=0 token {bad_t} at position {ctx} of a {ctx}-row cache: word \"{}\" \
             (want \"{}\", clean before {} and after the clean run {}) its rows appended nowhere and \
             every other plane slot the clean run's {planes} (the clean run wrote them {clean_wrote}), \
             turned heads bit-identical {heads}, clean rerun bits and word clean {clean_again} {}",
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            want.map_or_else(String::new, |f| f.to_string()),
            before.is_none(),
            after_clean.is_none(),
            verdict(pass)
        );
        Ok(pass)
    }
}
