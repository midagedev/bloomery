//! The Clef backbone gate: a Qwen3.5 dense file (`qwen35`, 64 layers — 48
//! gated-delta, 16 gated GQA at 24/4 heads of 256 — each with a dense SwiGLU
//! FFN of 17,408) whole on one card, its prompt call's final-norm hidden
//! states (`Qwen35moeModel::prefill_hidden`, `Tail::Hidden`) against llama.cpp
//! mainline's `result_norm` on the same file (`refset::arch::qwen35`: the
//! first 64, 600 and 4,096 ids of the prose corpus, `just dump-hidden-qwen35`).
//!
//! What is asserted:
//! - (r) the tensor-core decode flash is refused by name at group 6 (its
//!   pass needs four rows a pack; the pairs run the scalar pass).
//! - (h) each set's prompt from a `reset` on the auto path (one GEMM ubatch:
//!   the wide arm of every op, the dense FFN as a one-slot route table):
//!   `positions × 5120` values, and each position's relative L2 distance
//!   from mainline's row (`‖ours − lcpp‖ / ‖lcpp‖`). Over the positions:
//!   the median at most [`MEDIAN_BAND`], the median of each quarter of the
//!   prompt at most [`QUARTER_BAND`] (an error that grows with position),
//!   the 90th percentile at most [`P90_BAND`], and the share of positions
//!   past [`TAIL_CUT`] at most [`TAIL_SHARE`]; the 99th percentile, the
//!   worst position and the largest max abs error printed. The 64-id set
//!   also runs on the pass path (eight passes of eight: the gemv arm, the
//!   dense FFN through `_sel` with route 0), held to the same bands.
//!
//! - (q) the same on Clef-Flash as bartowski publishes it at Q8_0 (32
//!   layers of width 4,096 — 24 gated-delta, 8 gated GQA at 16/4 heads —
//!   every projection Q8_0, β and α F32; `refset::arch::qwen35`'s
//!   [`HIDDEN_FLASH_Q8`] sets): every site through the launch its type
//!   picks (`bloomery_gpu::site`), held to the Q8_0 bands: on the auto path
//!   mainline's own CPU-vs-CUDA spread, on the pass path (ours reads the f32
//!   rows, mainline q8 blocks of 32) its 8-bit error composed; and the
//!   body's launch count of a plan whose sites launch alone
//!   (`Body35::pass_launches`) is the captured graphs' node count: the
//!   decode step, its count plus the Q8_0 head's three; a pass of 5 and of 8
//!   rows, its count plus each row's copy into its head and the head.
//!
//! No bound is put on the worst position: at some positions of a prompt
//! mainline's own two backends lie order one apart (the bands' notes).
//!
//! The decode flash is the scalar segment pass (`mma` false); the prompt's
//! flash is the pairs' prefill body.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_clef_hidden: built without the `gpu` feature; see `just gate-gpu-clef-hidden`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_clef_hidden", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen35_open.rs"]
mod qwen35_open;

#[cfg(feature = "gpu")]
mod gate {
    use super::qwen35_open;
    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::qwen3moe::{Open35, PrefillPath, Qwen35moeModel};
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, load_ref_in, ref_model_path, verdict,
    };
    use gguf::Split;
    use refset::arch::qwen35::{
        FLASH_Q8_P64, FLASH_Q8_P600, FLASH_Q8_P4096, HIDDEN, HIDDEN_FLASH_Q8, MODEL_FLASH_Q8, P64,
        P600, P4096,
    };
    use refset::family::Family;
    use std::path::Path;

    /// The 27B's width and Clef-Flash's, as the headers state them.
    const WIDTH: usize = 5120;
    const FLASH_WIDTH: usize = 4096;

    /// Cache rows and the ubatch: the longest set's prompt as one ubatch.
    const CTX: usize = 4096;
    const U_GATE: usize = 4096;

    /// PIN(2026-10-02): the bound on the median position's relative L2
    /// distance from mainline's final-norm row. Both sides read the same
    /// Q4_K/Q6_K codes; they differ in the 8-bit activations each projection
    /// reads (ours q8_1 per 128 values, mainline's MMQ q8_1 per 32), about
    /// 1e-2 to 2e-2 apart (the qwen35moe gate's `quant_gap`), and in f32 sum
    /// order (some 1e-6). The qwen3moe family's free-running composition, a
    /// layer's error independent of the others', in quadrature over 64
    /// layers at a typical 1.5e-2 a layer: √64 · 1.5e-2 = 0.12. The final norm
    /// keeps the relative distance to first order. Mainline's CPU twin
    /// (`just dump-hidden-qwen35 --cpu-twin`: its CPU kernels, q8_K per 256,
    /// another 8-bit realization of the same rule) reads medians of 0.064,
    /// 0.098 and 0.095 from mainline's CUDA rows on the three sets
    /// (`tools/ref/hidden-diff.py <set>.cpu <set>`), inside it.
    const MEDIAN_BAND: f64 = 0.12;

    /// PIN(2026-10-02): the bound on each quarter's median. The same
    /// composition holds at every position (the decayed state and the
    /// softmax weigh earlier positions' errors with weights summing to at
    /// most one), so a quarter reads the whole prompt's median within its
    /// sampling spread; the twin's quarters read 0.043 to 0.113 (the highest
    /// a quarter of 16 positions of the 64-id set), and 0.15 is 1.3 times
    /// that. An error that grows with position (a rope or cache row wrong past
    /// some position) reads order one there.
    const QUARTER_BAND: f64 = 0.15;

    /// PIN(2026-10-02): the bound on the 90th percentile. The composition
    /// above gives no tail: at some positions of the prompt the model
    /// amplifies a small difference to order one, the same positions for
    /// any two realizations — the twin's p90 reads 0.115, 0.134 and 0.155 on
    /// the three sets, its 99th 0.14, 0.30 and 0.52, its worst 0.14, 0.92 and
    /// 1.00. 0.20 is 1.3 times the twin's largest p90.
    const P90_BAND: f64 = 0.20;

    /// PIN(2026-10-02): the tail's cut and the share of positions past it.
    /// The twin puts 0, 6 and 94 positions past 0.32 (0 %, 1.0 % and 2.3 %);
    /// of the 4,096-id set's positions past it on this engine (91), 67 are
    /// past it on the twin too, against about 2 if the two tails were
    /// independent — the tail is the prompt's and the rule's, not either
    /// engine's. 5 % is twice the twin's largest share. A wiring fault (the
    /// final norm skipped, a layer's FFN update dropped) moves every
    /// position, past the median band and this share at once.
    const TAIL_CUT: f64 = 0.32;
    const TAIL_SHARE: f64 = 0.05;

    /// A set's bands: the median, each quarter's median, the 90th
    /// percentile, and the share of positions past the tail's cut.
    struct Bands {
        median: f64,
        quarter: f64,
        p90: f64,
        cut: f64,
        share: f64,
    }

    /// The 27B Q4_K_M sets' bands (above).
    const Q4KM: Bands = Bands {
        median: MEDIAN_BAND,
        quarter: QUARTER_BAND,
        p90: P90_BAND,
        cut: TAIL_CUT,
        share: TAIL_SHARE,
    };

    /// PIN(2026-10-02): Clef-Flash Q8_0 on the auto path (one GEMM ubatch).
    /// Both sides read the same Q8_0 codes and quantize each projection's
    /// input by the same rule (q8 per 32 values, `d = amax/127`: our
    /// `quantize_gemm32`, mainline MMQ's `quantize_q8_1`), so the error a
    /// layer was taken as the rest — the flash's f16 tiles, the delta
    /// recurrence, the norms, f32 order — at most 1e-3, and scaled from the
    /// 27B sets to a median of 3.3e-3: a band of 0.01 read red at medians
    /// 0.023, 0.030 and 0.033. The wrong term is the scaling: mainline's own
    /// CPU twin, the same 8-bit rule on its CPU kernels, reads medians of
    /// 0.022, 0.029 and 0.032 from its CUDA rows (`tools/ref/hidden-diff.py
    /// <set>.cpu <set>`), and this engine sits as far from the twin as from
    /// CUDA (0.029 and 0.031 on the 600- and 4,096-id sets): any two
    /// realizations of the rule lie that far apart at this width, the floor of
    /// the oracle. The bands are the twin's: the median 1.23 times its largest
    /// (0.0325), each quarter's median 1.3 times its largest (0.035), the 90th
    /// percentile 1.3 times its largest (0.047), and past 0.12 at most twice
    /// its largest share (23 of 4,096 positions, 0.56 %).
    const FLASH_Q8_WIDE: Bands = Bands {
        median: 0.04,
        quarter: 0.046,
        p90: 0.062,
        cut: 0.12,
        share: 0.012,
    };

    /// PIN(2026-10-02): Clef-Flash Q8_0 on the pass path (the gemv arm): ours
    /// reads each projection's input as f32 rows, mainline's MMQ as q8 per
    /// 32 values, so a layer carries mainline's own 8-bit error,
    /// `(1/127)/√12 · amax/rms` ≈ 5e-3–7e-3, on top of the realizations'
    /// floor above (the twin's median 0.022 on these 64 ids). The bands are
    /// half the 27B Q4_K_M sets' (the error a layer a third of theirs,
    /// √(32/64) = 0.71: 0.24 of them, with margin), above that floor.
    const FLASH_Q8_PASS: Bands = Bands {
        median: 0.06,
        quarter: 0.075,
        p90: 0.10,
        cut: 0.16,
        share: 0.05,
    };

    /// `‖a − b‖ / ‖b‖` in f64, a NaN reading infinite; and the max abs error.
    fn distance(a: &[f32], b: &[f32]) -> (f64, f64) {
        let (mut num, mut den, mut max) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let e = f64::from(x) - f64::from(y);
            num += e * e;
            den += f64::from(y).powi(2);
            max = if e.is_nan() {
                f64::INFINITY
            } else {
                max.max(e.abs())
            };
        }
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        (if rel.is_nan() { f64::INFINITY } else { rel }, max)
    }

    /// The value at quantile `q` of `v` (the `⌊q · (n − 1)⌉`-th smallest), a
    /// NaN-free slice; `median` is `quantile(v, 0.5)`.
    fn quantile(v: &[f64], q: f64) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let at = (q * (s.len() - 1) as f64).round() as usize;
        s[at]
    }

    fn median(v: &[f64]) -> f64 {
        quantile(v, 0.5)
    }

    /// The sha256 of `path`, by `sha256sum`.
    fn sha256(path: &Path) -> Result<String, GateError> {
        let out = std::process::Command::new("sha256sum").arg(path).output()?;
        let text = String::from_utf8_lossy(&out.stdout);
        match text.split_whitespace().next() {
            Some(d) if out.status.success() && d.len() == 64 => Ok(d.to_string()),
            _ => Err(format!("sha256sum {}: {:?}", path.display(), text.trim()).into()),
        }
    }

    /// One set: its ids, read from the tokens file its header names and held
    /// to the header's digest and count, and mainline's rows.
    fn open_set(
        family: &Family,
        (name, n, width): (&str, usize, usize),
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        let man = RefManifest::open(&data_dir().join(name), family)?;
        let h = &man.header;
        let file = h
            .tokens_file
            .as_deref()
            .ok_or_else(|| format!("{name}: no `# tokens_file` line"))?;
        let want = h
            .tokens_file_sha256
            .as_deref()
            .ok_or_else(|| format!("{name}: no `# tokens_file_sha256` line"))?;
        let got = sha256(Path::new(file))?;
        if got != want || h.tokens_count != Some(u64::try_from(n)?) {
            return Err(format!(
                "{name}: the set was dumped from {want} ({:?} ids); {file} is {got}, {n} wanted",
                h.tokens_count
            )
            .into());
        }
        let ids = qwen35_open::read_ids(Path::new(file), Some(n))?;
        let (row, lcpp) = load_ref_in(&man, "result_norm", 0)?;
        row.expect(name, "f32", [width as u64, n as u64, 1, 1], "RMS_NORM")?;
        Ok((ids, lcpp))
    }

    /// One prompt from a `reset` by `path`, rows of `width`, against
    /// mainline's rows: the per-position distances, the bands `b`, one line.
    fn check(
        engine: &mut Qwen35moeModel,
        (name, path, width): (&str, PrefillPath, usize),
        b: &Bands,
        ids: &[u32],
        lcpp: &[f32],
    ) -> Result<bool, GateError> {
        engine.reset()?;
        let ours = engine.prefill_hidden(ids, path)?;
        if ours.len() != ids.len() * width || lcpp.len() != ours.len() {
            println!(
                "  {name} {path:?}: {} values and mainline's {}, {} due  FAIL",
                ours.len(),
                lcpp.len(),
                ids.len() * width
            );
            return Ok(false);
        }
        let per: Vec<(f64, f64)> = ours
            .chunks_exact(width)
            .zip(lcpp.chunks_exact(width))
            .map(|(o, l)| distance(o, l))
            .collect();
        let rel: Vec<f64> = per.iter().map(|p| p.0).collect();
        let n = rel.len();
        let quarters: Vec<f64> = (0..4)
            .map(|q| median(&rel[q * n / 4..(q + 1) * n / 4]))
            .collect();
        let (med, p90, p99) = (median(&rel), quantile(&rel, 0.9), quantile(&rel, 0.99));
        let tail = rel.iter().filter(|&&r| r > b.cut).count();
        let share = tail as f64 / n as f64;
        let (worst_at, worst) = rel
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("no positions")?;
        let (abs_at, abs) = per
            .iter()
            .map(|p| p.1)
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("no positions")?;
        let ok = med <= b.median
            && quarters.iter().all(|&q| q <= b.quarter)
            && p90 <= b.p90
            && share <= b.share;
        let quarters: Vec<String> = quarters.iter().map(|q| format!("{q:.4e}")).collect();
        println!(
            "  {name} {path:?}: {n} positions, rel L2 median {med:.4e} (band {}), quarters {} \
             (band {}), p90 {p90:.4e} (band {}), {tail} past {} = {:.2} % (band {:.1} %); p99 \
             {p99:.4e}, worst {worst:.4e} at {worst_at}, max abs {abs:.4e} at {abs_at}  {}",
            b.median,
            quarters.join(" "),
            b.quarter,
            b.p90,
            b.cut,
            share * 100.0,
            b.share * 100.0,
            verdict(ok)
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let path = ref_model_path()?;
        println!("gate_clef_hidden: {}", path.display());
        let mut ok = true;

        println!("(r) the tensor-core decode flash at group 6");
        let refused = match Qwen35moeModel::open(
            Gpu::new()?,
            Split::open(&path)?,
            Open35 {
                ctx: CTX,
                mma: true,
                ubatch: U_GATE,
            },
        ) {
            Ok(_) => "opened".to_string(),
            Err(e) => e.to_string(),
        };
        let r = refused.contains("the pairs' pass is scalar only");
        println!("  mma true: {refused}  {}", verdict(r));
        ok &= r;

        let (mut engine, _) = qwen35_open::open(&path, CTX, U_GATE)?;
        println!("(h) final-norm hidden states against mainline's result_norm");
        for (name, n) in [(P64, 64), (P600, 600), (P4096, 4096)] {
            let (ids, lcpp) = open_set(&HIDDEN, (name, n, WIDTH))?;
            ok &= check(
                &mut engine,
                (name, PrefillPath::Auto, WIDTH),
                &Q4KM,
                &ids,
                &lcpp,
            )?;
            if n == 64 {
                ok &= check(
                    &mut engine,
                    (name, PrefillPath::Pass, WIDTH),
                    &Q4KM,
                    &ids,
                    &lcpp,
                )?;
            }
        }
        drop(engine);

        let flash = Path::new(MODEL_FLASH_Q8);
        let (mut engine, _) = qwen35_open::open(flash, CTX, U_GATE)?;
        println!(
            "(q) Clef-Flash Q8_0 ({}): final-norm hidden states against mainline's result_norm",
            flash.display()
        );
        for (name, n) in [
            (FLASH_Q8_P64, 64),
            (FLASH_Q8_P600, 600),
            (FLASH_Q8_P4096, 4096),
        ] {
            let (ids, lcpp) = open_set(&HIDDEN_FLASH_Q8, (name, n, FLASH_WIDTH))?;
            let at = |path| (name, path, FLASH_WIDTH);
            ok &= check(
                &mut engine,
                at(PrefillPath::Auto),
                &FLASH_Q8_WIDE,
                &ids,
                &lcpp,
            )?;
            if n == 64 {
                ok &= check(
                    &mut engine,
                    at(PrefillPath::Pass),
                    &FLASH_Q8_PASS,
                    &ids,
                    &lcpp,
                )?;
            }
        }
        ok &= launches(&mut engine)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// The body's launch counts of `engine`'s plans against the node counts
    /// of its captured graphs (module doc, (q)).
    fn launches(engine: &mut Qwen35moeModel) -> Result<bool, GateError> {
        engine.reset()?;
        let body = engine.body("gate_clef_hidden")?;
        let (one, many) = (body.pass_launches(1), body.pass_launches(2));
        let step = engine.capture_step()?;
        let mut ok = step == one + 3;
        println!(
            "  launches: decode graph_nodes={step}, the body counts {one} + 3  {}",
            verdict(ok)
        );
        for (rows, nodes) in [
            (5usize, engine.capture_rows::<5>()?),
            (8, engine.capture_rows::<8>()?),
        ] {
            let pass = nodes == many + 4 * rows;
            println!(
                "  launches: pass m={rows} graph_nodes={nodes}, the body counts {many} + 4·{rows}  {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }
}
