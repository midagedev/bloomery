//! The raon1nano end-to-end gate: the whole decode chain of the Nano dialect
//! — the embedding and the per-layer input into Raon_mHC's four streams, the
//! fused-stem RDA layers and the gated latent layers, the dense SwiGLU blocks
//! with the folded chart, the streams' mean, the tied head and the argmax —
//! on one card, over the Raon-1-nano Q8_0 file, one prompt id a step, against
//! the vendor branch's CPU sets (refset family `lcpp-raon1nano`, `just
//! dump-ref-raon1nano`). Mini's e2e method (`gate_raon1_e2e`), cut to what
//! the dialect runs: no router tap (the file routes no experts), no wide arm
//! (its widths and its per-layer input keep every prompt on the rows pass)
//! and no draft (a nano file drafts nothing).
//!
//! For each prompt set of [`SETS`] (English prose, Korean prose, Rust code):
//!
//! - `top1_is_vendors`: the argmax after the prompt is the set's top-k
//!   table's first id. The top-[`TOP_K`] overlap of the two last-position
//!   rows and `d`, the largest distance of our logit from the vendor's over
//!   the table's ids, are printed beside it.
//! - `logits_are_vendors`: the whole last-position row's relative distance
//!   from the vendor's `result_output` is at most [`LOGITS_REL_BAND`]. It
//!   bounds `d`, so a chain that moves every logit cannot widen the named-tie
//!   band below with its own error.
//! - `greedy_is_vendors`: the greedy continuation is the set's `greedy.tsv`,
//!   id for id, over its whole length, or up to a named tie: a first
//!   divergence at a step whose vendor margin is at most `2d`, the most two
//!   logits each `d` off can close. A divergence past `2d` is red. Either is
//!   printed with the vendor's margin and our own gap between the two ids;
//!   the steps after a tie continue another text and are not compared.
//! - `route_is_vendors` (printed): the last layer's feed-forward site's route
//!   — the folded chart's `pre`, `post` and `comb` — ours beside the
//!   vendor's, `comb` from its transpose too (the two orientations of
//!   `comb[src · 4 + dst]`).
//! - `prompt_rows_is_steps`: the first 8 ids of the prompt as the rows pass
//!   (`GpuModel::prompt_rows`) and one id a step: the same argmax, the same
//!   logits row by bits, the same steps after it.
//!
//! On the plain load, once:
//!
//! - `wide_is_off_for_the_file`: the load names what keeps the wide arm off
//!   this file's prompts (`Body::wide_off`: the two attention widths, the
//!   dense blocks, the per-layer input), so every prompt runs the rows pass.
//!   FAIL-first: a load that raises the wide arm for the file answers
//!   `wide_ok` and the clause is red; mini's own clause (`wide_runs_the_file`)
//!   is the mirror and goes red here the same way.
//! - `prompt_rows_is_vendors`, on the 1,536-id wikitext set: the whole prompt
//!   as the rows pass reaches the vendor's argmax and its last logits row is
//!   within [`LOGITS_REL_BAND`] of the vendor's `result_output`; the row the
//!   same prompt one id a step reaches is printed beside it (same bits or
//!   distance). FAIL-first: a pass that mislays a position or a carried state
//!   answers another argmax or a row past the band.
//! - `prompt_rows_are_partitions`, on the same set: the prompt as one call
//!   and as two cut at [`SPLIT_AT`] gives the last logits row by bits. A
//!   rows-pass row depends on its own id, position and the carried state
//!   alone, so a pass that mislays the conv ring, the delta state or a cache
//!   row moves it.
//!
//! Then, over the same file in the verifying load (`GpuModel::open_verifying`),
//! as a session in graph mode — no draft, a nano file drafts nothing:
//!
//! - `verify_is_steps`, in eager and in graph mode: after the prompt call, a
//!   verify of the plain run's first 8 ids (and of its first 2) gives the
//!   plain run's next ids, each row's logits the plain step's by bits.
//! - `commit_keeps_steps`, in eager and in graph mode: a verify of 8 rows
//!   committed at 1, 3 and 8 rows, then steps: the plain run's ids.
//!
//! A step that raises the card's fault word ends the gate by name. A set
//! absent from the box, dumped from another file or build, or unfinished ends
//! the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raonnano_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::{rel, top_ids};
    use crate::greedy::greedy;
    use app::Session;
    use bloomery_gpu::arch::raon1::{Body, HC_OUT, STREAMS};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, data_dir, open_split,
        ref_tensor_logical_in, verdict,
    };
    use model::arch::Arch;
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::arch::raon1nano::{A, B, C, D, LCPP};
    use runtime::{RowLogits, Target, Verify, Want};

    /// The sets the step path is held to, in the order they run.
    const SETS: &[&str] = &[A, B, C];

    /// Rows of each latent layer's cache: the longest prompt and its
    /// continuation fit (the 1,536-id set included).
    const CTX: usize = 2048;

    /// The rows of a pass: a call of up to this many ids runs as the steps do.
    const PASS: usize = 8;
    const _: () = assert!(PASS == bloomery_gpu::linear::PASS_ROWS);

    /// The file's layers: the last one's route is printed.
    const LAYERS: usize = 24;

    /// The last-position ids whose overlap is printed.
    const TOP_K: usize = 5;

    /// The last-position logits row's relative distance from the vendor's.
    /// PIN(2026-11-13): mini's measured band (3.33e-2, 3.87e-2 and 3.16e-2 on
    /// its three sets, pinned at twice the worst); the lead measures nano's
    /// own and re-pins. Derived to sit inside it: the chain is one layer
    /// shorter, its dots are 1,536-wide over the same Q8_0 weights the vendor
    /// reads (no dequant class between the two engines), and the dense blocks
    /// have no router tie to flip.
    const LOGITS_REL_BAND: f64 = 5.0e-2;

    /// Where `prompt_rows_are_partitions` cuts the long prompt: a multiple of
    /// a pass's rows, and of neither a unit's rows nor a latent attention
    /// chunk's, so the second call's launches start at other positions than
    /// the first call's: a row's bits do not depend on where its launch
    /// begins.
    const SPLIT_AT: usize = 600;
    const _: () = assert!(SPLIT_AT.is_multiple_of(PASS));

    /// The steps `prompt_rows_is_steps` and `commit_keeps_steps` run after the call they check.
    const AFTER: usize = 4;

    /// A verify of a proposal's rows.
    const VERIFY_ROWS: usize = 8;

    /// The rows `commit_keeps_steps` keeps of a verify of [`VERIFY_ROWS`].
    const KEEPS: [usize; 3] = [1, 3, VERIFY_ROWS];

    type Seat = Session<Body>;

    /// One set's prompt and its two tables.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// The vendor's whole logits row at the prompt's last position
        /// (`result_output`).
        logits: Vec<f32>,
        /// The vendor's route of the last layer's feed-forward site at the
        /// prompt's last position: `pre`, `post`, `comb` in [`HC_OUT`]'s order;
        /// `None` for a set dumped without those taps.
        route: Option<Vec<f32>>,
    }

    /// The last `n` values of tap `name`'s occurrence `occurrence` in `man`.
    fn last_of(
        man: &RefManifest,
        name: &str,
        occurrence: u32,
        n: usize,
    ) -> Result<Vec<f32>, GateError> {
        let row = man.tensor(name, occurrence)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let at = all
            .len()
            .checked_sub(n)
            .ok_or_else(|| format!("{name}: {} values, want {n}", all.len()))?;
        Ok(all[at..].to_vec())
    }

    fn open_set(name: &'static str) -> Result<Set, GateError> {
        let man = RefManifest::open(&LCPP.path_in(&data_dir(), name), &LCPP)?;
        let row = man.tensor("result_output", 0)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let vocab = usize::try_from(row.ne[0])?;
        if vocab == 0 || all.len() % vocab != 0 {
            return Err(format!("{name}: result_output is {:?}", row.ne).into());
        }
        let logits = all[all.len() - vocab..].to_vec();
        let last = LAYERS - 1;
        let route = match man.tensor(&format!("hc_comb-{last}"), 1) {
            Err(_) => None,
            Ok(_) => {
                let mut r = last_of(&man, &format!("hc_pre-{last}"), 1, STREAMS)?;
                r.extend(last_of(&man, &format!("hc_post-{last}"), 1, STREAMS)?);
                r.extend(last_of(
                    &man,
                    &format!("hc_comb-{last}"),
                    1,
                    STREAMS * STREAMS,
                )?);
                Some(r)
            }
        };
        Ok(Set {
            name,
            prompt: prompt_tables::tokens(&man)?.to_vec(),
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            logits,
            route,
        })
    }

    /// Our route of the last layer's feed-forward site at the prompt's last
    /// position beside the vendor's: `pre`'s and `post`'s relative distances,
    /// and `comb`'s from the vendor's and from its transpose (the two
    /// orientations of `comb[src · 4 + dst]`).
    fn print_route(m: &GpuModel<Body>, s: &Set) -> Result<(), GateError> {
        let Some(theirs) = &s.route else {
            println!(
                "route set={} has no layer-{} route taps",
                s.name,
                LAYERS - 1
            );
            return Ok(());
        };
        let ours = m.body("gate_raonnano_e2e")?.last_route(m.gpu().stream())?;
        if ours.len() != HC_OUT || theirs.len() != HC_OUT {
            return Err(format!("routes of {} and {} values", ours.len(), theirs.len()).into());
        }
        let k = 2 * STREAMS;
        let transposed: Vec<f32> = (0..STREAMS * STREAMS)
            .map(|i| theirs[k + (i % STREAMS) * STREAMS + i / STREAMS])
            .collect();
        println!(
            "route set={} layer={} pre_rel={:.3e} post_rel={:.3e} comb_rel={:.3e} \
             comb_rel_transposed={:.3e}",
            s.name,
            LAYERS - 1,
            rel(&ours[..STREAMS], &theirs[..STREAMS]),
            rel(&ours[STREAMS..k], &theirs[STREAMS..k]),
            rel(&ours[k..], &theirs[k..]),
            rel(&ours[k..], &transposed),
        );
        Ok(())
    }

    /// The first step whose id is not the vendor's, and our logit gap
    /// between our id and the vendor's there.
    type Divergence = Option<(usize, f32)>;

    /// The greedy ids from `first` (the argmax of `row`, the prompt's last
    /// logits), each fed back, until the set's length or the first id that
    /// is not the vendor's: that step and our logit gap between our id and
    /// the vendor's there.
    fn continue_greedy(
        m: &mut GpuModel<Body>,
        s: &Set,
        first: u32,
        mut row: Vec<f32>,
    ) -> Result<(Vec<u32>, Divergence), GateError> {
        let mut ids = vec![first];
        loop {
            let i = ids.len() - 1;
            if ids[i] != s.greedy.ids[i] {
                let gap = row[ids[i] as usize] - row[s.greedy.ids[i] as usize];
                return Ok((ids, Some((i, gap))));
            }
            if ids.len() == s.greedy.ids.len() {
                return Ok((ids, None));
            }
            ids.push(m.step(&[ids[i]])?);
            row = m.logits()?;
        }
    }

    /// Clause `prompt_rows_is_steps` on set `s`: the prompt's first [`PASS`] ids one id a step and
    /// as the rows pass (`GpuModel::prompt_rows`), each from a reset model: the same argmax, the
    /// same logits row by bits, and the same [`AFTER`] steps after it.
    fn prompt_rows_is_steps(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let head = &s.prompt[..PASS.min(s.prompt.len())];
        let mut run = |rows: bool| -> Result<(Vec<u32>, Vec<f32>), GateError> {
            m.reset()?;
            let first = if rows {
                m.prompt_rows(head)?
            } else {
                m.step(head)?
            };
            let row = m.logits()?;
            let mut ids = vec![first];
            while ids.len() <= AFTER {
                let last = ids[ids.len() - 1];
                ids.push(m.step(&[last])?);
            }
            Ok((ids, row))
        };
        let (steps, steps_row) = run(false)?;
        let (rows, rows_row) = run(true)?;
        let same_row = bits_equal(&steps_row, &rows_row);
        let pass = steps == rows && same_row;
        println!(
            "prompt_rows_is_steps set={} steps={steps:?} rows={rows:?} logits_same_bits={same_row} {}",
            s.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `prompt_rows_is_vendors` on the 1,536-id set `s`: the whole prompt as the rows pass
    /// from a reset model reaches the vendor's top-1 id, and its last logits row is within
    /// [`LOGITS_REL_BAND`] of the vendor's `result_output`. The row the same prompt one id a step
    /// reaches is printed beside it, with whether it is the rows pass's own bits.
    fn prompt_rows_is_vendors(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let theirs = s
            .top
            .first()
            .ok_or_else(|| format!("{}: an empty top-k table", s.name))?;
        m.reset()?;
        let first = m.prompt_rows(&s.prompt)?;
        let row = m.logits()?;
        let logits_rel = rel(&row, &s.logits);
        m.reset()?;
        m.step(&s.prompt)?;
        let steps_row = m.logits()?;
        let steps_rel = rel(&row, &steps_row);
        let steps_bits = bits_equal(&row, &steps_row);
        let pass = first == theirs.id && logits_rel <= LOGITS_REL_BAND;
        println!(
            "prompt_rows_is_vendors set={} prompt_ids={} ours={first} vendor={} \
             logits_rel={logits_rel:.4e} band={LOGITS_REL_BAND:.1e} steps_rel={steps_rel:.4e} \
             steps_same_bits={steps_bits} {}",
            s.name,
            s.prompt.len(),
            theirs.id,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `prompt_rows_are_partitions` on set `s`: the prompt as one call and as two cut at
    /// [`SPLIT_AT`] reach the same last logits row, by bits.
    fn prompt_rows_are_partitions(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if s.prompt.len() < SPLIT_AT + PASS + 1 {
            return Err(format!(
                "{}: a prompt of {} ids; the split clause runs from {}",
                s.name,
                s.prompt.len(),
                SPLIT_AT + PASS + 1
            )
            .into());
        }
        m.reset()?;
        let whole = m.prompt_rows(&s.prompt)?;
        let whole_row = m.logits()?;
        m.reset()?;
        m.prompt_rows(&s.prompt[..SPLIT_AT])?;
        let cut = m.prompt_rows(&s.prompt[SPLIT_AT..])?;
        let same_row = bits_equal(&whole_row, &m.logits()?);
        let pass = whole == cut && same_row;
        println!(
            "prompt_rows_are_partitions set={} prompt_ids={} split_at={SPLIT_AT} ids_equal={} \
             logits_same_bits={same_row} {}",
            s.name,
            s.prompt.len(),
            whole == cut,
            verdict(pass)
        );
        Ok(pass)
    }

    /// A verify of `M` rows of the plain run `plain` from the prompt's end, through the session
    /// from a reset: whether the prompt call's id is `plain[0]` and the verify's ids are
    /// `plain[1..=M]`, and whether each row's logits are, by bits, `rows` one id later (`rows[r]`
    /// is the row whose argmax is `plain[r]`). The verify is left waiting for its commit.
    fn verified<const M: usize>(
        seat: &mut Seat,
        s: &Set,
        (plain, rows): (&[u32], &[Vec<f32>]),
    ) -> Result<(bool, bool), GateError> {
        Target::reset(seat)?;
        let first = Target::prompt(seat, &s.prompt, Want::Argmax)?.argmax();
        let out = Verify::verify::<M>(seat, <[u32; M]>::try_from(&plain[..M])?)?;
        let ids = first == plain[0] && out[..] == plain[1..=M];
        let mut logits = true;
        for (r, want) in rows[1..=M].iter().enumerate() {
            logits &= bits_equal(RowLogits::row_logits(seat, r)?, want);
        }
        Ok((ids, logits))
    }

    /// Clauses `verify_is_steps` (verifies of [`VERIFY_ROWS`] and 2 rows) and
    /// `commit_keeps_steps` (a verify of [`VERIFY_ROWS`] rows committed at each of [`KEEPS`], then
    /// [`AFTER`] steps, the plain run's ids) on set `s` in the seat's mode `mode`.
    fn verify_is_steps(seat: &mut Seat, s: &Set, mode: &str) -> Result<bool, GateError> {
        let (plain, rows) = greedy(seat, &s.prompt, VERIFY_ROWS + 1 + AFTER, true)?;
        let run = (&plain[..], &rows[..]);
        let mut ok = true;
        for (m, (ids, logits)) in [
            (VERIFY_ROWS, verified::<VERIFY_ROWS>(seat, s, run)?),
            (2, verified::<2>(seat, s, run)?),
        ] {
            let pass = ids && logits;
            println!(
                "verify_is_steps set={} mode={mode} rows={m} ids={ids} logits_same_bits={logits} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        for k in KEEPS {
            let (ids, logits) = verified::<VERIFY_ROWS>(seat, s, run)?;
            Verify::commit(seat, k)?;
            let mut after = vec![plain[k]];
            while after.len() <= AFTER {
                let last = after[after.len() - 1];
                after.push(Target::step(seat, last, Want::Argmax)?.argmax());
            }
            let kept = after[..] == plain[k..=k + AFTER];
            let pass = ids && logits && kept;
            println!(
                "commit_keeps_steps set={} mode={mode} kept={k} after={:?} plain={:?} {}",
                s.name,
                &after[1..],
                &plain[k + 1..=k + AFTER],
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set(n))
            .collect::<Result<_, _>>()?;
        for s in &sets {
            if s.greedy.ids.is_empty() || s.greedy.margins.len() != s.greedy.ids.len() {
                return Err(format!(
                    "{}: a greedy table of {} ids and {} margins",
                    s.name,
                    s.greedy.ids.len(),
                    s.greedy.margins.len()
                )
                .into());
            }
            if s.prompt.len() + s.greedy.ids.len() > CTX {
                return Err(format!(
                    "{}: {} positions, the gate's caches hold {CTX}",
                    s.name,
                    s.prompt.len() + s.greedy.ids.len()
                )
                .into());
            }
        }
        let long = open_set(D)?;
        if long.prompt.len() + 1 > CTX {
            return Err(format!(
                "{}: {} positions, the gate's caches hold {CTX}",
                long.name,
                long.prompt.len() + 1
            )
            .into());
        }
        let file = open_split(Arch::Raon1Nano, "gate-gpu-raonnano-e2e")?;
        let t = std::time::Instant::now();
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, CTX)?;
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        if m.layers().len() != LAYERS {
            return Err(format!(
                "a file of {} layers; the sets' is {LAYERS}",
                m.layers().len()
            )
            .into());
        }
        let mut ok = true;
        {
            // The wide arm's refusal is the dialect's own fact: the file's two attention widths,
            // its dense blocks and its per-layer input keep every prompt on the rows pass.
            let body = m.body("gate_raonnano_e2e")?;
            let (wide_ok, why) = (body.wide_ok(), body.wide_off().map(str::to_owned));
            println!(
                "wide_is_off_for_the_file wide_ok={wide_ok} refusal={why:?} {}",
                verdict(!wide_ok && why.is_some())
            );
            ok &= !wide_ok && why.is_some();
        }
        for s in &sets {
            m.reset()?;
            let first = m.step(&s.prompt)?;
            let row = m.logits()?;
            let theirs = s
                .top
                .first()
                .ok_or_else(|| format!("{}: an empty top-k table", s.name))?;
            let ours5 = top_ids(&row, TOP_K);
            let d = s
                .top
                .iter()
                .map(|t| (row[t.id as usize] - t.logit).abs())
                .fold(0.0f32, f32::max);
            let logits_rel = rel(&row, &s.logits);
            let logits_ok = logits_rel <= LOGITS_REL_BAND;
            print_route(&m, s)?;
            let overlap = s
                .top
                .iter()
                .take(TOP_K)
                .filter(|t| ours5.contains(&(t.id as usize)))
                .count();
            let top1 = first == theirs.id;
            println!(
                "top1_is_vendors set={} prompt_ids={} ours={first} vendor={} top{TOP_K}_overlap={overlap}/{TOP_K} \
                 ours_top{TOP_K}={ours5:?} d={d} over {} ids {}",
                s.name,
                s.prompt.len(),
                theirs.id,
                s.top.len(),
                verdict(top1)
            );
            let (ids, diverge) = continue_greedy(&mut m, s, first, row)?;
            let (pass, at) = match diverge {
                None => (true, "none".to_owned()),
                Some((i, our_gap)) => {
                    let margin = s.greedy.margins[i];
                    let tie = margin <= 2.0 * d;
                    let word = if tie { "a named tie" } else { "past 2d" };
                    let at = format!(
                        "step {i} ours={} vendor={} vendor_margin={margin} our_gap={our_gap} \
                         2d={} ({word})",
                        ids[i],
                        s.greedy.ids[i],
                        2.0 * d
                    );
                    (tie, at)
                }
            };
            println!(
                "greedy_is_vendors set={} steps={} first_divergence={at} {}",
                s.name,
                ids.len(),
                verdict(pass)
            );
            println!("greedy set={} ours={ids:?}", s.name);
            println!(
                "logits_are_vendors set={} logits_rel={logits_rel:.4e} band={LOGITS_REL_BAND:.1e} {}",
                s.name,
                verdict(logits_ok)
            );
            ok &= top1 && pass && logits_ok;
            ok &= prompt_rows_is_steps(&mut m, s)?;
        }
        ok &= prompt_rows_is_vendors(&mut m, &long)?;
        ok &= prompt_rows_are_partitions(&mut m, &long)?;
        // The plain load's card bytes go back before the verifying load takes its own.
        drop(m);
        {
            let file = open_split(Arch::Raon1Nano, "gate-gpu-raonnano-e2e")?;
            let mut m = GpuModel::<Body>::open_verifying(Gpu::new()?, file, CTX)?;
            m.set_mode(StepMode::Graph);
            m.capture_step()?;
            let mut seat = Seat::from_model(m, u32::try_from(CTX)?);
            for s in &sets {
                for (mode, word) in [(StepMode::Eager, "eager"), (StepMode::Graph, "graph")] {
                    seat.model_mut().set_mode(mode);
                    ok &= verify_is_steps(&mut seat, s, word)?;
                }
                Target::reset(&mut seat)?;
            }
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_raonnano_e2e on every set of {SETS:?} the argmax after the prompt is the \
             vendor's, the greedy continuation is the vendor's id for id, or up to a named tie, and \
             the 1,536-id prompt as the rows pass reaches the vendor's row within the band"
        );
        Ok(())
    }
}
