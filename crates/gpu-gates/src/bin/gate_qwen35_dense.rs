//! The dense Qwen3.8-27B generation gate: the `qwen35` file (64 layers, 48 gated-delta and 16 gated GQA at 24/4 heads of
//! 256, each with a dense SwiGLU FFN of 17,408; the 15 `blk.64.*` next-token tensors unused) whole on one card, the
//! Body35 chain a request runs: the prompt call, the decode step, the pass, the checkpoints and the resident slots,
//! against llama.cpp mainline's forced-logits rows (`refset::arch::qwen35::dense`, `just dump-logits-qwen38d`). The
//! file fits either card, so the gate holds the same clauses in both tiers. `gate_clef_hidden` holds the trunk's
//! hidden-state band on the same geometry; this gate does not repeat it.
//!
//! What is asserted:
//! - (a) load: the whole-fit verdict's weights term, the buffers the load uploaded and the header derivation
//!   (`weights::resident_size` of every tensor but `blk.64.*`) are the same bytes; no `blk.64` tensor is resident;
//!   the verdict's cache term and the body's store bytes are their derivation from the header's geometry.
//! - (b) oracle: for each set (the 64- and 4,096-id prompts, 32 teacher-forced rows each) the prompt call and the
//!   forced decode steps' logits rows against mainline's CUDA rows: each row's relative L2 within [`REL_BAND`],
//!   the median within [`MEDIAN_BAND`], the 90th percentile within [`P90_BAND`]; and the argmax equal on every row
//!   whose reference top-2 margin exceeds its stability bound ([`STABLE_K`] times the oracle's own spread on that row,
//!   mainline's CPU twin against its CUDA rows); the rows at or under it are counted and pinned ([`OTHER_ROWS`]).
//! - (c) graph = eager = pass: five tokens as graph decode steps, one pass of five rows, eager steps and graph steps
//!   after a `reset`, each leaving the same logits, tokens and every layer's store bit for bit.
//! - (d) reset: a 600-id sequence with 8 greedy steps, a `reset`, a shorter sequence, a `reset`, the first again: the
//!   two runs equal bit for bit (logits, tokens, every store).
//! - (e) checkpoints: a prompt of [`KP`] ids takes its points at the ubatch's multiples and its end; one checkpoint's
//!   bytes and slots are their derivation; a cut to a non-checkpoint is refused by name; a cut to the last inner mark
//!   and the re-fed tail equal the uncut call bit for bit; a `reset` drops every point.
//! - (f) slots: the slots harness (`slots_gate`, H1-H7) over two streams on one load, a pass of one row a slot equal to
//!   each alone, the pass refusals and each slot's checkpoints its own.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35_dense", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen35_open.rs"]
#[allow(
    dead_code,
    reason = "the loader serves the hidden, media and dense gates; this one reads the load and the whole-fit verdict"
)]
mod qwen35_open;

#[cfg(feature = "gpu")]
#[path = "shared/q35_clauses.rs"]
mod q35_clauses;

#[cfg(feature = "gpu")]
mod gate {
    use super::q35_clauses::{
        self as q35, Geometry35, PathRun, Slots35, fresh, lcg_ids, same_run, stores, uncut_call,
    };
    use super::qwen35_open;
    use bloomery_gpu::arch::qwen3moe::{Body35, KvQ8, Open35, Qwen35moeModel};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::weights::resident_size;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{
        GateError, checks_failed, data_dir, gpu_census, kld, quantile, ref_model_path, verdict,
    };
    use gguf::{GgmlType, Split};
    use refset::arch::qwen35::dense::{ForcedLogits, MODEL, P64, P4096};
    use std::path::Path;

    /// The cache every clause loads: the oracle sets' context (a 4,096-id prompt and 31 forced steps fit), with the room
    /// the checkpoint clause's marks need.
    const CTX: usize = 4352;
    /// The load's ubatch: four units of the 4,096-id prompt, marks at its multiples.
    const U_GATE: usize = 1024;

    /// The real file's literals, beside the geometry its header gives (the move proof, `Geometry35::of`).
    const REAL: Geometry35 = Geometry35 {
        n_attn: 16,
        n_delta: 48,
        n_kv: 4,
        head: 256,
        n_v: 48,
        head_v: 128,
        conv_c: 10240,
    };
    /// The `blk.<n>.` prefix of the next-token block the Body35 reader leaves unused (`qwen35.nextn_predict_layers`
    /// is 1, after the 64 trunk layers).
    const UNUSED_BLOCK: &str = "blk.64.";

    /// PIN(2026-10-19): the bound on one row's relative L2 distance from mainline's CUDA row. Derivation: this engine
    /// and mainline read the same Q4_K/Q6_K codes and differ in the 8-bit activations (ours q8_1 per 128 values,
    /// mainline's MMQ q8_1 per 32) and the f32 sum order, as mainline's CPU twin does (q8_K per 256); the twin's rows
    /// against its CUDA rows are the oracle's own floor (`dense-logits.py floor`): per-row relative L2 at most
    /// 0.1492 (64-id set) and 0.1008 (4,096-id set). 0.20 is 1.34 times the largest. A wiring fault (a layer, a
    /// position or a forced id off by one) moves a row by order one.
    const REL_BAND: f64 = 0.20;
    /// PIN(2026-10-19): the bound on a set's median row. The floor's medians are 0.0445 and 0.0525; 0.066 is 1.25
    /// times the largest, the factor `gate_clef_hidden`'s median band takes over its twin.
    const MEDIAN_BAND: f64 = 0.066;
    /// PIN(2026-10-19): the bound on a set's 90th percentile. The floor's are 0.0643 and 0.0685; 0.09 is 1.3 times the
    /// largest, as `gate_clef_hidden`'s p90 band.
    const P90_BAND: f64 = 0.09;
    /// PIN(2026-10-19): the factor on a row's stability bound. Two logits swap only if their margin is within the
    /// difference of their two moves, at most `min(2 · max_abs, √2 · l2)` of the move's vector
    /// (`dense-logits.py stable_margin`); the move taken is the oracle's own spread on that row (twin against CUDA,
    /// the same positions amplify for any two realizations) times 1.3, the factor the p90 band takes.
    const STABLE_K: f64 = 1.3;
    /// PIN(2026-10-19): the rows at or under their stability bound (the only rows that could flip), per set: 20 of 32
    /// and 17 of 32 at [`STABLE_K`] (28 of 64 at a factor of 1, the floor's report). A count that moves means the
    /// sets or the twin moved, not the engine.
    const OTHER_ROWS: [usize; 2] = [20, 17];

    /// Greedy steps after each prompt of (d).
    const D_STEPS: usize = 8;
    /// (d)'s two prompts: the longer first, the shorter after a `reset`, so no row of the first lies past the second's.
    const D_LONG: usize = 600;
    const D_SHORT: usize = 64;

    /// (e)'s prompt: past two marks (1,024 and 2,048) with a run of 200 ids left after them, short of the third.
    const KP: usize = 2 * U_GATE + 200;

    /// (f)'s streams, A and B: each past the load's first mark by a run the mark keeps.
    const SLOT_A: usize = U_GATE + 16;
    const SLOT_B: usize = U_GATE + 32;

    /// The oracle's two sets, with the row counts they hold.
    const SETS: [(&str, usize); 2] = [(P64, 64), (P4096, 4096)];

    fn load(path: &Path) -> Result<(Qwen35moeModel, Split), GateError> {
        qwen35_open::open(path, CTX, U_GATE)
    }

    /// `(‖a − b‖₂, max |a − b|, ‖b‖₂)` in f64; a NaN reads infinite.
    fn norms(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
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
        let l2 = if num.is_nan() {
            f64::INFINITY
        } else {
            num.sqrt()
        };
        (l2, max, den.sqrt())
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let path = ref_model_path()?;
        println!("gate_qwen35_dense: {}", path.display());
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let geom = Geometry35::of(&split)?;
        let witness = geom == REAL;
        println!(
            "geometry from the header {geom:?} (the real file's {REAL:?}{}) {}",
            if path == Path::new(MODEL) {
                ""
            } else {
                ", another file"
            },
            verdict(witness || path != Path::new(MODEL))
        );
        let mut ok = witness || path != Path::new(MODEL);
        drop(split);

        // `--only <a-f>` runs one clause (the FAIL-first mutants' loop); no flag runs them all.
        let only = std::env::args().skip_while(|a| a != "--only").nth(1);
        let want = |c: &str| only.as_deref().is_none_or(|o| o == c);
        if let Some(o) = &only
            && !["a", "b", "c", "d", "e", "f"].contains(&o.as_str())
        {
            return Err(format!("--only {o}: a clause is one of a, b, c, d, e, f").into());
        }
        let vocab = if ["a", "b", "c", "d", "e"].iter().any(|c| want(c)) {
            let o = Open35 {
                ctx: CTX,
                mma: false,
                ubatch: U_GATE,
                kv: KvQ8::F16,
                media: false,
            };
            if want("a") {
                ok &= load_clause(&path, o, geom)?;
            }
            let (mut m, split) = load(&path)?;
            if want("a") {
                ok &= resident_clause(&m, &split, geom)?;
            }
            drop(split);
            m.set_mode(StepMode::Graph);
            let mut sets = Vec::new();
            for (name, _) in SETS {
                sets.push(ForcedLogits::open(&data_dir().join(name))?);
            }
            if want("b") {
                println!("(b) forced logits rows against mainline's CUDA rows");
                for (i, (name, prompt)) in SETS.iter().enumerate() {
                    ok &= oracle(&mut m, name, *prompt, &sets[i], OTHER_ROWS[i])?;
                }
            }
            if want("c") {
                println!("(c) graph = eager = pass");
                let toks: Vec<u32> = sets[0].prompt.iter().take(5).copied().collect();
                ok &= q35::paths(&mut m, &toks)?.0;
            }
            if want("d") {
                ok &= reset_clause(&mut m)?;
            }
            if want("e") {
                ok &= checkpoint_clause(&mut m, geom)?;
            }
            m.body("gate_qwen35_dense")?.vocab()
        } else {
            let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            split
                .find("token_embd.weight")
                .map(|(_, t)| t.dims[1] as usize)
                .ok_or("the file holds no token_embd.weight")?
        };

        if want("f") {
            println!("(f) resident slots: the harness's contracts over two lcg streams");
            expect_q6k_head(&path)?;
            let slots = Slots35::new(
                lcg_ids(SLOT_A + SLOT_B, vocab),
                SLOT_A,
                (U_GATE, CTX),
                (geom, 1 + 2 * geom.n_delta + 3 * geom.n_attn),
                Box::new(move || load(&path).map(|(m, _)| m)),
            )?;
            ok &= q35::slots_two(&slots)?;
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// The head is Q6_K, else refused by name: the slots adapter's pass-node count is the Q6_K head's.
    fn expect_q6k_head(path: &Path) -> Result<(), GateError> {
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let ty = split.find("output.weight").map(|(_, t)| t.ty);
        match ty {
            Some(GgmlType::Q6_K) => Ok(()),
            other => Err(format!(
                "output.weight is {other:?}; the slots derivation is written for Q6_K"
            )
            .into()),
        }
    }

    /// Tensors of the header but `blk.64.*`: `(name, resident bytes)` by `weights::resident_size`.
    fn derived_weights(split: &Split) -> Result<Vec<(String, usize)>, GateError> {
        split
            .iter_tensors()
            .map(|(_, t)| t)
            .filter(|t| !t.name.starts_with(UNUSED_BLOCK))
            .map(|t| {
                let rows = t.dims[1..].iter().product::<u64>() as usize;
                resident_size(t.ty, t.dims[0] as usize, rows)
                    .map(|b| (t.name.clone(), b))
                    .ok_or_else(|| format!("{}: no device format for {:?}", t.name, t.ty).into())
            })
            .collect()
    }

    /// (a), the verdict half, before the load: the whole-fit line, its weights and cache terms against the derivation.
    fn load_clause(path: &Path, o: Open35, geom: Geometry35) -> Result<bool, GateError> {
        println!(
            "(a) load: the whole-fit verdict and the header derivation without {UNUSED_BLOCK}*"
        );
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let census = gpu_census::census()?;
        let card = Place::A.on(&census)?.card_specs()?[0];
        let need = qwen35_open::whole_verdict(&split, o, card, &|_| {})?;
        println!("  whole fit at ctx {CTX}: {need}");
        let derived: usize = derived_weights(&split)?.iter().map(|d| d.1).sum();
        let all: usize = split
            .iter_tensors()
            .map(|(_, t)| {
                let rows = t.dims[1..].iter().product::<u64>() as usize;
                resident_size(t.ty, t.dims[0] as usize, rows)
                    .unwrap_or_else(|| panic!("{}: no device format for {:?}", t.name, t.ty))
            })
            .sum();
        let kv = geom.store_bytes(CTX) as u64;
        let ok = need.weights_unrounded == derived as u64 && need.kv == kv && all > derived;
        println!(
            "  verdict weights {} B of buffers, derived {derived} B (with {UNUSED_BLOCK}* {all} B, {} B more); cache {} \
             B, derived {kv} B ({} attention layers' K/V planes of 2 x {} heads x {CTX} x {} f16 + {} layers' state \
             and conv ring) {}",
            need.weights_unrounded,
            all - derived,
            need.kv,
            geom.n_attn,
            geom.n_kv,
            geom.head,
            geom.n_delta,
            verdict(ok)
        );
        Ok(ok)
    }

    /// (a), the load half: the weights the load holds against the derivation, name for name; none is `blk.64.*`; the
    /// body's stores are the geometry's.
    fn resident_clause(
        m: &Qwen35moeModel,
        split: &Split,
        geom: Geometry35,
    ) -> Result<bool, GateError> {
        let want = derived_weights(split)?;
        let w = m.weights();
        let names: Vec<&str> = w.names().filter(|n| !n.starts_with("derived.")).collect();
        let held: usize = names
            .iter()
            .filter_map(|n| w.get(n))
            .map(|d| d.resident_bytes())
            .sum();
        let want_names: Vec<&str> = want.iter().map(|d| d.0.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        let mut want_sorted = want_names.clone();
        want_sorted.sort_unstable();
        let unused = names.iter().filter(|n| n.starts_with(UNUSED_BLOCK)).count();
        let derived: usize = want.iter().map(|d| d.1).sum();
        let stores = m.body("gate_qwen35_dense")?.store_bytes();
        let ok = sorted == want_sorted
            && held == derived
            && unused == 0
            && stores == geom.store_bytes(CTX);
        println!(
            "  resident: {} file tensors {held} B (derived {derived} B over {} tensors), {unused} of {UNUSED_BLOCK}* \
             resident; stores {stores} B (derived {} B); resident_bytes {} B {}",
            names.len(),
            want.len(),
            geom.store_bytes(CTX),
            m.resident_bytes(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The prompt call and the forced steps of `set` on `m` from a `reset`: the logits row after the prompt, then after
    /// each forced id fed.
    fn rows_of(m: &mut Qwen35moeModel, set: &ForcedLogits) -> Result<Vec<Vec<f32>>, GateError> {
        m.reset()?;
        <Body35 as app::Prompt>::prompt(m, &set.prompt)?;
        let mut rows = vec![m.logits()?];
        for &id in &set.forced[..set.rows() - 1] {
            m.step(&[id])?;
            rows.push(m.logits()?);
        }
        Ok(rows)
    }

    /// (b) on one set: the clause's bands, the stability rule and the pinned count of rows under it.
    fn oracle(
        m: &mut Qwen35moeModel,
        name: &str,
        prompt: usize,
        set: &ForcedLogits,
        other: usize,
    ) -> Result<bool, GateError> {
        let twin_dir = data_dir().join(format!("{name}.cpu"));
        let twin = ForcedLogits::open_cpu(&twin_dir)?;
        if twin.prompt != set.prompt || twin.forced != set.forced || set.prompt.len() != prompt {
            return Err(format!(
                "{name}: its twin is not the CPU run of the same prompt and forced ids"
            )
            .into());
        }
        let ours = rows_of(m, set)?;
        let (mut rel, mut flips_stable, mut stable, mut under, mut flips_under) =
            (Vec::new(), 0usize, 0usize, 0usize, 0usize);
        let mut worst = (0.0f64, 0usize);
        for (i, row) in ours.iter().enumerate() {
            let reference = set.row(i);
            if row.len() != reference.len() {
                return Err(format!(
                    "{name} row {i}: {} logits, the set's {}",
                    row.len(),
                    reference.len()
                )
                .into());
            }
            let (l2, _, norm) = norms(row, reference);
            rel.push(l2 / norm);
            if l2 / norm > worst.0 || l2.is_infinite() {
                worst = (l2 / norm, i);
            }
            let (twin_l2, twin_max, _) = norms(twin.row(i), reference);
            let bound = STABLE_K * f64::min(2.0 * twin_max, std::f64::consts::SQRT_2 * twin_l2);
            let (ours_top, _, _, _) = kld::top2(row);
            let (ref_top, first, _, second) = kld::top2(reference);
            let margin = f64::from(first) - f64::from(second);
            if margin > bound {
                stable += 1;
                flips_stable += usize::from(ours_top != ref_top);
            } else {
                under += 1;
                flips_under += usize::from(ours_top != ref_top);
            }
        }
        let (median, p90) = (quantile(&rel, 0.5), quantile(&rel, 0.9));
        let ok = rel.iter().all(|&r| r <= REL_BAND)
            && median <= MEDIAN_BAND
            && p90 <= P90_BAND
            && flips_stable == 0
            && under == other;
        println!(
            "  {name}: {} rows after {prompt} ids; rel L2 median {median:.4} (band {MEDIAN_BAND}), p90 {p90:.4} (band \
             {P90_BAND}), worst {:.4} at row {} (band {REL_BAND}); argmax equal on {}/{stable} stable rows; {under} \
             rows at or under their stability bound (pinned {other}), {flips_under} of them flipped {}",
            ours.len(),
            worst.0,
            worst.1,
            stable - flips_stable,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `ids` as one prompt call and `steps` greedy steps from the model's state, every position's logits and token and
    /// every store read back.
    fn sequence(m: &mut Qwen35moeModel, ids: &[u32], steps: usize) -> Result<PathRun, GateError> {
        let mut tok = <Body35 as app::Prompt>::prompt(m, ids)?;
        let (mut logits, mut tokens) = (vec![m.logits()?], vec![tok]);
        for _ in 0..steps {
            tok = m.step(&[tok])?;
            logits.push(m.logits()?);
            tokens.push(tok);
        }
        Ok(PathRun {
            logits,
            tokens,
            stores: stores(m)?,
        })
    }

    /// (d): the first sequence again after a `reset`, a shorter sequence between, equal bit for bit.
    fn reset_clause(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        println!("(d) reset: the same sequence after a reset and another sequence");
        let vocab = m.body("gate_qwen35_dense")?.vocab();
        let (long, short) = (lcg_ids(D_LONG, vocab), lcg_ids(D_SHORT, vocab));
        fresh(m)?;
        let first = sequence(m, &long, D_STEPS)?;
        m.reset()?;
        sequence(m, &short, D_STEPS)?;
        m.reset()?;
        let again = sequence(m, &long, D_STEPS)?;
        Ok(same_run(
            "the first sequence after a reset and a shorter one",
            &again,
            &first,
        ))
    }

    /// (e): the checkpoints of a marked prompt call and a cut with its re-fed tail (module doc).
    fn checkpoint_clause(m: &mut Qwen35moeModel, geom: Geometry35) -> Result<bool, GateError> {
        println!("(e) checkpoints: the points, the cut, the re-feed and the reset");
        m.body_parts("gate_qwen35_dense")?.2.set_checkpoints(true);
        let ids = lcg_ids(KP, m.body("gate_qwen35_dense")?.vocab());
        let ub = u32::try_from(m.ubatch()?)?;
        let c = m.body("gate_qwen35_dense")?.checkpoints();
        let want_bytes = geom.checkpoint_bytes() as u64;
        let fit = runtime::seqstate::capacity(runtime::seqstate::HOST_BUDGET, want_bytes)
            .map_err(|e| e.to_string())?;
        let want_slots = (CTX.div_ceil(ub as usize) + 1).min(fit);
        let bytes_ok = c.bytes() == want_bytes && c.capacity() == want_slots;
        println!(
            "  one copy {} B (derived {want_bytes} B), {} slots (want {want_slots}) {}",
            c.bytes(),
            c.capacity(),
            verdict(bytes_ok)
        );
        let uncut = uncut_call(m, &ids)?;
        let want_pts: Vec<u32> = (1..)
            .map(|k| ub * k)
            .take_while(|&p| p < KP as u32)
            .chain([KP as u32])
            .collect();
        let body = m.body("gate_qwen35_dense")?;
        let (pts, walks) = (body.checkpoints().positions(), body.prompt_walks());
        let walks_want = (KP as u32).div_ceil(ub) as usize;
        let points_ok = pts == want_pts && walks == walks_want;
        println!(
            "  a prompt of {KP} ids took its points at {pts:?} (want {want_pts:?}), {walks} walks (want {walks_want}) {}",
            verdict(points_ok)
        );
        let ask = (KP - 50) as u32;
        let refused = m
            .rollback(ask)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let refused_ok = refused.contains("no checkpoint restores it")
            && m.body("gate_qwen35_dense")?.checkpoints().positions() == want_pts;
        println!(
            "  a cut to {ask}: {refused}; the points stand {}",
            verdict(refused_ok)
        );
        let cut_at = ub * (ask / ub);
        let refed_ok = q35::refed_after_cut(m, &ids, cut_at, &uncut, &want_pts, ub)?;
        fresh(m)?;
        let dropped = m
            .body("gate_qwen35_dense")?
            .checkpoints()
            .positions()
            .is_empty();
        println!("  a reset drops every point {}", verdict(dropped));
        m.set_mode(StepMode::Graph);
        Ok(bytes_ok && points_ok && refused_ok && refed_ok && dropped)
    }
}
