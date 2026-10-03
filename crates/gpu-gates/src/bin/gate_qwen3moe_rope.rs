//! GPU gate for qwen3moe's attention head kernel,
//! `rope_neox::head_norm_neox_append` — the per-head RMS norm of the query
//! and key heads, the NEOX turn over the whole 128-value head and the K/V
//! append in one launch — against ik's CPU dumps, in two clauses.
//!
//! The norm (`attn_q_norm` / `attn_k_norm`, against `Qcur_normed-L` and
//! `Kcur_normed-L`): the kernel runs with the identity table (every cos 1,
//! every sin 0, a row for each cache position), under which the turn returns
//! each normalized value unchanged, so its query and key heads are the norm
//! alone. Three layers per (set, layer):
//! 1. the kernel against this binary's transcription of our rule
//!    (`qwen3moe::head_norm`) — bit-identical — and a rerun, bit-identical;
//! 2. ik's rule (`ik_norm::fused`, the f64 serial sum) on the dumped input
//!    against `Qcur_normed-L`/`Kcur_normed-L` — bit-identical: the semantics
//!    (per-head rows, the gain per head, eps) proven on ik's own values;
//! 3. the kernel against the dump within [`NORM_BAND`] per value.
//!
//! The turn and the append (against `Qcur_roped-L`, `Kcur_roped-L`, and the
//! two cache writes `cache_k_lL (view) (copy of Kcur_roped-L)` and
//! `v_cache_view-L (copy of Vcur-L)`): the kernel reads its rope row from a
//! table of every cache position, at the token's position; the gate hands it
//! the engine's table (`RopeTable::push` for positions `0..ctx`).
//!
//! Three layers per (set, layer):
//! 1. the kernel against this binary's transcription of our rule — the norm
//!    (`qwen3moe::head_norm`), the turn (`qwen3moe::neox_rotate`) on the
//!    table's row at each token's position, each plane row the f16 of its
//!    value, every other plane slot untouched — bit-identical, and a rerun
//!    bit-identical;
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
//! That token has no table row: its query and key heads are NaN, it is
//! appended nowhere (its rows keep the sentinel), and the launch raises
//! `FaultSite::CachePos` with that layer; every other token's heads and every
//! other plane slot are the clean run's bit for bit; the word is clean before
//! and after a clean run.
//!
//! And the q8_0 append (`head_norm_neox_append_q8`), the same norm and turn
//! with each K/V row quantized, per (set, layer) in one clause: the query and
//! key heads bit for bit the f16 append's (the prefix the two entries share),
//! every appended row the two planes of `quantize_q8_0` (the engine's one Q8_0
//! quantizer, `model::arch::deepseek2::attn::quantize_q8_0`) over the same
//! f32 values the f16 append rounds, packed by `weights::q8_0_planes` — byte
//! for byte, both sides — every other plane slot the sentinel, and a rerun
//! bit-identical. The launch captured as a graph: one node, the replay the
//! eager launch's bits. And a value row holding a NaN: the block that holds
//! it raises `FaultSite::KvQuant` and is stored with a NaN scale and zero
//! codes — every value it dequantizes to is NaN, never a plausible number —
//! with every other block and both whole K planes the clean run's bits, and
//! the word clean after a clean rerun.

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
    use bloomery_gpu::rope_neox::{RopeNeoxKernels, q8_plane_lens};
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{Fault, FaultSite, Gpu};
    use bloomery_gpu_gates::qwen3moe::dev::{
        SENTINEL, SENTINEL_Q8_CODE, SENTINEL_Q8_SCALE, run as run_neox, run_graph,
        run_q8 as run_neox_q8, run_q8_graph,
    };
    use bloomery_gpu_gates::qwen3moe::{AttnRows, HEAD, head_norm, neox_rotate, sets};
    use bloomery_gpu_gates::rounding::U_F32;
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, ik_norm, max_ulps, open_split,
        same_bits, split_f32, verdict,
    };
    use gguf::Split;
    use gguf::quant::{Q8Block, f32_to_f16_bits};
    use model::arch::Arch;
    use model::arch::deepseek2::attn::quantize_q8_0;
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

    /// Band for a turned head against ik's, as `max|Δ| / M` with `M` the
    /// largest `|value|` ik wrote. The normalized values differ by at most
    /// `10u` of themselves ([`NORM_BAND`]). A pair then
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
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        bloomery_levers::at_main(&[])?;
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
        let mut grown = Vec::new();
        let gpu = Gpu::new()?;
        let k = RopeNeoxKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        println!(
            "gate_qwen3moe_rope: device {} — rope {spec:?}, {} layers",
            gpu.device_name()?,
            hp.n_layer
        );
        let sets = read_sets(&split, &hp)?;
        let norm_ok = norm_clause(&gpu, &k, &sets, &hp)?;
        let q8_ok = q8_clause(&gpu, &k, &sets, &hp, &table)?;

        // The turn and the append.
        let (mut sites, mut failed) = (0u32, 0u32);
        for SetLayers { label, man, layers } in &sets {
            let mut worst = 0.0f32;
            let mut pos_seen = Vec::new();
            for (layer, Layer { rows, gq, gk }) in layers.iter().enumerate() {
                pos_seen.clone_from(&rows.pos);
                let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
                let tab = table_rows(&table, &mut grown, ctx)?;
                let cs = rows_at(tab, &rows.pos);
                let (n_kv, m) = (rows.n_kv, rows.m);

                // Layer 1: the kernel against our rule, and a rerun.
                let a = run_neox(&k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
                let b = run_neox(&k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
                let norm = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| head_norm(h, g, hp.rms_eps))
                        .collect()
                };
                let hq = neox_rotate(&norm(&rows.q, gq), &cs, rows.n_head);
                let hk = neox_rotate(&norm(&rows.k, gk), &cs, n_kv);
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
        let SetLayers { label, layers, .. } = &sets[1];
        let Layer { rows, gq, gk } = &layers[0];
        let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
        let tab = table_rows(&table, &mut grown, ctx)?;
        let eager = run_neox(&k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
        let (replay, nodes) = run_graph(&gpu, &k, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
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
        failed += u32::from(!cache_pos(
            &gpu, &k, &split, &table, &mut grown, hp.rms_eps,
        )?);

        let pass = failed == 0;
        println!(
            "rope: {sites} (set, layer) sites and the graph, {failed} failed — {}",
            verdict(pass)
        );
        let ok = norm_ok && q8_ok && pass;
        println!("gate_qwen3moe_rope: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// One layer of a set, read once for both clauses: ik's rows and the
    /// layer's q and k norm gains.
    struct Layer {
        rows: AttnRows,
        gq: Vec<f32>,
        gk: Vec<f32>,
    }

    /// One qwen3moe set with each of its layers read.
    struct SetLayers {
        label: &'static str,
        man: RefManifest,
        layers: Vec<Layer>,
    }

    /// Every qwen3moe set ([`sets`], in its order) with each of its layers
    /// read.
    fn read_sets(split: &Split, hp: &Hparams) -> Result<Vec<SetLayers>, GateError> {
        let mut out = Vec::new();
        for (label, man) in sets()? {
            let mut layers = Vec::with_capacity(hp.n_layer);
            for layer in 0..hp.n_layer {
                let rows = AttnRows::read(&man, layer)?
                    .ok_or_else(|| format!("{label}: no Qcur_normed-{layer}"))?;
                let gq = split_f32(split, &rows.gq_name, HEAD)?;
                let gk = split_f32(split, &rows.gk_name, HEAD)?;
                layers.push(Layer { rows, gq, gk });
            }
            out.push(SetLayers { label, man, layers });
        }
        Ok(out)
    }

    /// The norm clause (module doc) over the sets' layers: whether it held.
    fn norm_clause(
        gpu: &Gpu,
        k: &RopeNeoxKernels,
        sets: &[SetLayers],
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        println!(
            "qknorm: device {} — {} layers, {} q / {} kv heads of {}, eps {:e}",
            gpu.device_name()?,
            hp.n_layer,
            hp.n_head,
            hp.n_head_kv,
            hp.head_dim,
            hp.rms_eps
        );
        let (mut sites, mut failed) = (0u32, 0u32);
        for SetLayers { label, man, layers } in sets {
            let mut worst = 0.0f32;
            let (mut same_q, mut same_k, mut n_q, mut n_k) = (0usize, 0usize, 0usize, 0usize);
            for (layer, Layer { rows, gq, gk }) in layers.iter().enumerate() {
                let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
                let identity: Vec<f32> = (0..ctx * HEAD / 2).flat_map(|_| [1.0, 0.0]).collect();

                // Layer 1: the kernel against our rule, and a rerun.
                let a = run_neox(k, stream, unl, rows, gq, gk, &identity, hp.rms_eps, ctx)?;
                let b = run_neox(k, stream, unl, rows, gq, gk, &identity, hp.rms_eps, ctx)?;
                let host = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| head_norm(h, g, hp.rms_eps))
                        .collect()
                };
                let (hq, hk) = (host(&rows.q, gq), host(&rows.k, gk));
                let exact = bits_equal(&a.q, &hq) && bits_equal(&a.k, &hk);
                let rerun = bits_equal(&a.q, &b.q) && bits_equal(&a.k, &b.k);

                // Layer 2: ik's rule against the dump.
                let sim = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| ik_norm::fused(h, g, hp.rms_eps))
                        .collect()
                };
                let sim_q = same_bits(&sim(&rows.q, gq), &rows.q_normed);
                let sim_k = same_bits(&sim(&rows.k, gk), &rows.k_normed);
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
            "qknorm: {sites} (set, layer) sites, {failed} failed — {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The q8_0 append's expected planes: row `(h·ctx + p)` of each side the
    /// blocks `quantize_q8_0` makes of the row's `HEAD` values, packed by
    /// `q8_0_planes` (the engine's one Q8_0 format), every other slot the
    /// sentinel.
    fn q8_planes_want(
        rows: &AttnRows,
        hk: &[f32],
        ctx: usize,
    ) -> (Vec<u32>, Vec<u16>, Vec<u32>, Vec<u16>) {
        let (words, scales) = q8_plane_lens(HEAD, rows.n_kv, ctx);
        let mut kq = vec![SENTINEL_Q8_CODE; words];
        let mut kd = vec![SENTINEL_Q8_SCALE; scales];
        let mut vq = kq.clone();
        let mut vd = kd.clone();
        let pack = |vals: &[f32], q: &mut [u32], d: &mut [u16], row: usize| {
            let blocks: Vec<Q8Block> = vals.chunks(32).map(quantize_q8_0).collect();
            let (qs, ds) = q8_0_planes(&blocks);
            let (wq, wd) = (row * (HEAD / 4), row * (HEAD / 32));
            q[wq..wq + qs.len()].copy_from_slice(&qs);
            d[wd..wd + ds.len()].copy_from_slice(&ds);
        };
        for (t, &p) in rows.pos.iter().enumerate() {
            for h in 0..rows.n_kv {
                let src = (t * rows.n_kv + h) * HEAD;
                let row = h * ctx + p as usize;
                pack(&hk[src..src + HEAD], &mut kq, &mut kd, row);
                pack(&rows.v[src..src + HEAD], &mut vq, &mut vd, row);
            }
        }
        (kq, kd, vq, vd)
    }

    /// The q8_0 append clause (module doc): whether it held.
    fn q8_clause(
        gpu: &Gpu,
        k: &RopeNeoxKernels,
        sets: &[SetLayers],
        hp: &Hparams,
        table: &RopeTable,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let mut grown = Vec::new();
        println!(
            "q8 append: device {} — {} layers, {} q / {} kv heads of {}",
            gpu.device_name()?,
            hp.n_layer,
            hp.n_head,
            hp.n_head_kv,
            hp.head_dim
        );
        let (mut sites, mut failed) = (0u32, 0u32);
        for SetLayers { label, layers, .. } in sets {
            for (layer, Layer { rows, gq, gk }) in layers.iter().enumerate() {
                let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
                let tab = table_rows(table, &mut grown, ctx)?;
                let cs = rows_at(tab, &rows.pos);
                let norm = |x: &[f32], g: &[f32]| -> Vec<f32> {
                    x.chunks(HEAD)
                        .flat_map(|h| head_norm(h, g, hp.rms_eps))
                        .collect()
                };
                let hk = neox_rotate(&norm(&rows.k, gk), &cs, rows.n_kv);

                let f16 = run_neox(k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
                let a = run_neox_q8(k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
                let b = run_neox_q8(k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
                // The prefix the two entries share: the heads bit for bit.
                let heads = bits_equal(&a.q, &f16.q) && bits_equal(&a.k, &f16.k);
                let (wkq, wkd, wvq, wvd) = q8_planes_want(rows, &hk, ctx);
                let planes = a.kq == wkq && a.kd == wkd && a.vq == wvq && a.vd == wvd;
                let rerun = bits_equal(&a.q, &b.q)
                    && bits_equal(&a.k, &b.k)
                    && a.kq == b.kq
                    && a.kd == b.kd
                    && a.vq == b.vq
                    && a.vd == b.vd;
                let pass = heads && planes && rerun;
                sites += 1;
                failed += u32::from(!pass);
                if !pass || layer == 0 || layer + 1 == hp.n_layer {
                    println!(
                        "q8 append set={label} layer={layer} m={} pos={:?} heads_f16_bits={heads} \
                         planes_exact={planes} rerun={rerun} {}",
                        rows.m,
                        rows.pos,
                        verdict(pass)
                    );
                }
            }
        }
        // The captured graph: one node, the replay the eager launch's bits.
        let SetLayers { label, layers, .. } = &sets[1];
        let Layer { rows, gq, gk } = &layers[0];
        let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
        let tab = table_rows(table, &mut grown, ctx)?;
        let eager = run_neox_q8(k, stream, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
        let (replay, nodes) = run_q8_graph(gpu, k, unl, rows, gq, gk, tab, hp.rms_eps, ctx)?;
        let same = bits_equal(&eager.q, &replay.q)
            && bits_equal(&eager.k, &replay.k)
            && eager.kq == replay.kq
            && eager.kd == replay.kd
            && eager.vq == replay.vq
            && eager.vd == replay.vd;
        let graph_ok = same && nodes == 1;
        println!(
            "graph op=head_norm_neox_append_q8 set={label} layer=0 \
             eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        failed += u32::from(!graph_ok);
        failed += u32::from(!kv_quant_clause(gpu, k, sets, hp, table)?);

        let pass = failed == 0;
        println!(
            "q8 append: {sites} (set, layer) sites and the graph, {failed} failed — {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// A value row holding a NaN (module doc): whether the refusal held.
    fn kv_quant_clause(
        gpu: &Gpu,
        k: &RopeNeoxKernels,
        sets: &[SetLayers],
        hp: &Hparams,
        table: &RopeTable,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let SetLayers { label, layers, .. } = &sets[0];
        let (gq, gk) = (&layers[0].gq, &layers[0].gk);
        let mut rows = layers[0].rows.clone();
        if rows.m < 2 || rows.n_kv == 0 {
            return Err(format!(
                "{label}: the kv_quant clause wants two tokens and a key head, got {} and {}",
                rows.m, rows.n_kv
            )
            .into());
        }
        let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
        let mut grown = Vec::new();
        let tab = table_rows(table, &mut grown, ctx)?;
        let unl = gpu.unlabelled_sink();
        let layer = 13usize;
        let sink = gpu.layer_sink(layer)?;
        let before = gpu.fault()?;
        let clean = run_neox_q8(k, stream, unl, &rows, gq, gk, tab, hp.rms_eps, ctx)?;
        let after_clean = gpu.fault()?;
        // Token 1, key head 0, value 100: block 3 of its V row.
        let (bad_t, bad_h, bad_val) = (1usize, 0usize, 100usize);
        let src = (bad_t * rows.n_kv + bad_h) * HEAD + bad_val;
        let bad_block = bad_val / 32;
        rows.v[src] = f32::NAN;
        let bad = run_neox_q8(k, stream, sink, &rows, gq, gk, tab, hp.rms_eps, ctx)?;
        let word = gpu.take_fault()?;
        rows.v[src] = layers[0].rows.v[src];

        let row = bad_h * ctx + rows.pos[bad_t] as usize;
        let (mut wvq, mut wvd) = (clean.vq.clone(), clean.vd.clone());
        wvq[row * (HEAD / 4) + 8 * bad_block..row * (HEAD / 4) + 8 * (bad_block + 1)].fill(0);
        wvd[row * (HEAD / 32) + bad_block] = f32_to_f16_bits(f32::NAN);
        let planes = bad.kq == clean.kq
            && bad.kd == clean.kd
            && bad.vq == wvq
            && bad.vd == wvd
            && bits_equal(&bad.q, &clean.q)
            && bits_equal(&bad.k, &clean.k);
        let again = run_neox_q8(k, stream, unl, &rows, gq, gk, tab, hp.rms_eps, ctx)?;
        let clean_again = bits_equal(&again.q, &clean.q)
            && again.kq == clean.kq
            && again.vd == clean.vd
            && gpu.fault()?.is_none();
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::KvQuant));
        let pass =
            before.is_none() && after_clean.is_none() && word == want && planes && clean_again;
        println!(
            "kv_quant set={label} layer=0 token {bad_t} head {bad_h} value {bad_val} NaN: word \"{}\" \
             (want \"{}\", clean before {} and after the clean run {}) the block refused with a NaN \
             scale and zero codes, every other bit the clean run's {planes}, clean rerun and word \
             clean {clean_again} {}",
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            want.map_or(String::new(), |f| f.to_string()),
            before.is_none(),
            after_clean.is_none(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Rows `0..ctx` of `table` (`push` per position, in order), grown in
    /// `grown` as far as a call asks and kept for the next.
    fn table_rows<'a>(
        table: &RopeTable,
        grown: &'a mut Vec<f32>,
        ctx: usize,
    ) -> Result<&'a [f32], GateError> {
        for p in grown.len() / HEAD..ctx {
            table.push(u32::try_from(p)?, Direction::Forward, grown);
        }
        Ok(&grown[..ctx * HEAD])
    }

    /// The rows of `tab` at `pos`, token after token: what the kernel reads
    /// for each token.
    fn rows_at(tab: &[f32], pos: &[u32]) -> Vec<f32> {
        pos.iter()
            .flat_map(|&p| &tab[p as usize * HEAD..(p as usize + 1) * HEAD])
            .copied()
            .collect()
    }

    /// The position clause (module doc): whether it held.
    fn cache_pos(
        gpu: &Gpu,
        k: &RopeNeoxKernels,
        split: &Split,
        table: &RopeTable,
        grown: &mut Vec<f32>,
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
        let ctx = rows.pos.iter().max().map_or(1, |&p| p as usize + 1);
        let tab = table_rows(table, grown, ctx)?;
        let layer = 13usize;
        let sink = gpu.layer_sink(layer)?;
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::CachePos));
        let before = gpu.fault()?;
        let clean = run_neox(k, stream, sink, &rows, &gq, &gk, tab, eps, ctx)?;
        let after_clean = gpu.fault()?;
        let bad_t = rows.m - 1;
        let bad_p = rows.pos[bad_t] as usize;
        rows.pos[bad_t] = u32::try_from(ctx)?;
        let bad = run_neox(k, stream, sink, &rows, &gq, &gk, tab, eps, ctx)?;
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
        // Token by token: the bad token's heads all NaN, every other token's
        // the clean run's bits.
        let heads_of = |got: &[f32], want: &[f32], width: usize| {
            got.chunks(width)
                .zip(want.chunks(width))
                .enumerate()
                .all(|(t, (g, w))| {
                    if t == bad_t {
                        g.iter().all(|v| v.is_nan())
                    } else {
                        bits_equal(g, w)
                    }
                })
        };
        let heads = heads_of(&bad.q, &clean.q, rows.n_head * HEAD)
            && heads_of(&bad.k, &clean.k, rows.n_kv * HEAD);
        let again = run_neox(k, stream, sink, &rows, &gq, &gk, tab, eps, ctx)?;
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
             its heads NaN and every other token's the clean run's bits {heads}, clean rerun bits and \
             word clean {clean_again} {}",
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            want.map_or_else(String::new, |f| f.to_string()),
            before.is_none(),
            after_clean.is_none(),
            verdict(pass)
        );
        Ok(pass)
    }
}
