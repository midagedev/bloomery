//! The nemotron_h_moe verify gate: a load opened to verify drafted rows
//! (`GpuModel::<Body>::open_verifying`) keeps any `accepted` in `1..=m` of
//! a verify of `m` rows — the Mamba-2 layers' scan state restored by one
//! device copy of row `accepted − 1`'s lane — and a plain load is
//! untouched. The fixture tier (`BLOOMERY_TIER=fixture`, the Bolt fixture,
//! any card) is the recipe's default; the real tier (`BLOOMERY_TIER=real`)
//! runs the same clauses against the real Bolt file and adds (b)'s
//! partial-commit pin. Every clause but that pin is self-consistency
//! against the same load's own plain run, so no reference set is read.
//!
//! What is asserted, greedy:
//! - (a) every accepted: for `m` in {2, 4} and each `accepted` in `1..=m`,
//!   a 40-id prompt (a fixed lcg walk), one verify of `m` forced rows (the
//!   plain run's next ids), `commit(accepted)`, then 16 plain steps. The
//!   verify's row argmaxes, the 16 ids and the last step's logits equal a
//!   plain load's run from the same prompt, bit for bit. Where engine's
//!   m-row pass is not bit-equal to m single steps (the multi-column
//!   launches' own contract, `prefill`'s module doc), the logits are held
//!   to the band `gate_nemotron_h_moe_e2e` uses instead — cosine at least
//!   0.999, an id exact wherever the plain run's own top-1 margin exceeds
//!   twice the row's max |delta| — and the line says which held. A chain of
//!   commits (2/1, 4/3, 4/4, 2/2, 4/1) then 16 steps, the same way, walks
//!   the lanes round. Graph and eager.
//! - (b) a lookup draft (`runtime::lookup::Lookup`) driven over
//!   `Session<Body>` (`Session::from_model`, `Session::with_draft`, a
//!   verify of two rows) for 64 tokens on three prompts of one short id
//!   walk repeated (8 ids × 5, three walks — the draft's n-grams come from
//!   the prompt itself) emits the plain greedy ids. A pass proposes only
//!   where the context's tail token occurred earlier, and a fixture's
//!   greedy stream need repeat no token, so the proposals, the accepted
//!   drafts and the partial commits are printed as information in the
//!   fixture tier; the real tier pins the accepted drafts above 0 and
//!   below the proposals on at least one run (partial commits ran), where
//!   a real file's stream repeats.
//! - (c) the plain load is untouched (a plain open holds `None` lanes: no
//!   bytes, `Body::verify` never made), and the verifying load holds
//!   exactly `n_mamba · (rows − 1) · d_inner · 128 · 4` bytes more — the
//!   fixture's own Mamba-2 layer count and `d_inner` read from the file.
//! - (e) refusals by name: a step, a pass and a prompt call while a verify
//!   waits for its commit; a rollback behind the pending rows; a plain
//!   load's verify; `open_verifying` with rows 1 and past
//!   [`MAX_PASS_ROWS`]. Each refused call leaves the verify committable.
//!
//! The FAIL-first mutants (`tools/mutant-run.sh`):
//! - (a) the commit copies `lanes[k]` instead of `lanes[k − 1]`;
//! - (b) the commit always keeps `m` (no lane copy, the pending dropped as
//!   if every row was kept) — red in (a), whose chain commits `k < m`: (b)
//!   itself proposes nothing in the fixture tier;
//! - (c) the lanes allocated on a plain load too;
//! - (e) the step's waiting refusal removed.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_h_moe_verify", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "gpu")]
mod gate {
    use crate::greedy::greedy;
    use app::{RowsLog, Session, SessionError};
    use bloomery_gpu::arch::nemotron_h_moe::Body;
    use bloomery_gpu::model::{MAX_PASS_ROWS, StepMode};
    use bloomery_gpu::{Gpu, GpuError, NemotronHMoeModel};
    use bloomery_gpu_gates::tier::Tier;
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use model::arch::Arch;
    use model::arch::nemotron_h_moe::hparams::{Hparams, Kind};
    use runtime::{Advance, Lookup, Speculative, Target, Verify, Want};

    type Seat = Session<Body>;

    const NAME: &str = "gate_nemotron_h_moe_verify";

    /// Cache rows of every load: the longest run below ((a)'s 40-id prompt,
    /// its verifies and the steps after them, (b)'s prompts and 64 tokens)
    /// with room.
    const CTX: usize = 256;

    /// (a)'s prompt, and the plain steps after each commit.
    const PROMPT_A: usize = 40;
    const STEPS_AFTER: usize = 16;

    /// Ids of the plain reference run: the longest verify chain, a verify's
    /// rows and the steps after it read at most this many.
    const REF_IDS: usize = 72;

    /// (b)'s generated tokens, and the lookup's verify width (a proposal of
    /// one id).
    const GEN: usize = 64;
    const PAIR: usize = 2;

    /// The cosine the last step's logits keep against the plain run's when
    /// the bits differ ([`gate_nemotron_h_moe_e2e`]'s floor).
    const COSINE_FLOOR: f64 = 0.999;

    /// (a)'s chain of commits: `(m, accepted)` stages.
    const CHAIN: [(usize, usize); 5] = [(2, 1), (4, 3), (4, 4), (2, 2), (4, 1)];

    /// The rows a verifying load keeps lanes for: the most a pass takes.
    const VERIFY_ROWS: usize = MAX_PASS_ROWS;

    struct Log;

    impl RowsLog for Log {
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError> {
            println!("lookup verify pass of {rows} rows captured: {nodes} nodes");
            Ok(())
        }
    }

    /// The fixture's file, read once for the header facts (c) derives its
    /// bytes from.
    fn header() -> Result<(Hparams, usize, usize), GateError> {
        let file = open_split(Arch::NemotronHMoe, "gate-gpu-nemotron-h-moe-verify")?;
        let hp = Hparams::read(&file).map_err(|e| format!("nemotron_h_moe header: {e}"))?;
        let n_mamba = hp.kinds.iter().filter(|&&k| k == Kind::Mamba).count();
        let d_inner = hp.ssm.d_inner;
        Ok((hp, n_mamba, d_inner))
    }

    fn open_seat(verifying: bool, hp: &Hparams, tier: Tier) -> Result<Seat, GateError> {
        let file = open_split(Arch::NemotronHMoe, NAME)?;
        let gpu = Gpu::new()?;
        let m = if verifying {
            NemotronHMoeModel::open_verifying(gpu, file, CTX, VERIFY_ROWS)?
        } else {
            NemotronHMoeModel::open(gpu, file, CTX)?
        };
        println!(
            "{NAME}: load verifying={verifying} resident_bytes={} ctx={CTX} vocab={} ({}: {} \
             layers, {MAX_PASS_ROWS} lanes a Mamba-2 layer)",
            m.resident_bytes(),
            hp.n_vocab,
            tier.name(),
            hp.n_layer
        );
        Ok(Session::from_model(m, CTX as u32))
    }

    /// `n` ids of an lcg over the vocabulary from `seed`: every id a row of
    /// the embedding, the routing spread over the experts.
    fn lcg(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((x >> 33) % vocab as u64) as u32
            })
            .collect()
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The cosine and the max |Δ| of `got` against `want`.
    fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
        let max_delta = got
            .iter()
            .zip(want)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0, f64::max);
        let dot: f64 = got
            .iter()
            .zip(want)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        (dot / (norm(got) * norm(want)), max_delta)
    }

    /// The ids of the `k` largest values, largest first, lower id first on a
    /// tie.
    fn top_ids(v: &[f32], k: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..v.len()).collect();
        ids.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
        ids.truncate(k);
        ids
    }

    /// One id against the plain run's, with the row that produced the plain
    /// run's: equal, or — where the pass's rows are not the steps' bits —
    /// exact wherever the plain run's own top-1 margin exceeds twice the
    /// row's max |Δ| (by the triangle inequality no element's error can
    /// then swap the two). `why` names the first band judgement made.
    fn id_holds(got: u32, want: u32, row: &[f32], band: &mut bool, why: &mut String) -> bool {
        if got == want {
            return true;
        }
        *band = true;
        if why.is_empty() {
            let top = top_ids(row, 2);
            let margin = f64::from(row[top[0]]) - f64::from(row[top[1]]);
            *why =
                format!("id {got} against the plain run's {want} (margin {margin:.4} of its row)");
        }
        false
    }

    /// One stage of a chain: verify `m` forced rows of the reference ids
    /// from `idx`, then commit `accepted` of them. The verify's row argmaxes.
    fn stage_m<const M: usize>(
        s: &mut Seat,
        r: &[u32],
        idx: usize,
        accepted: usize,
    ) -> Result<Vec<u32>, SessionError> {
        let rows: [u32; M] = std::array::from_fn(|j| r[idx + j]);
        let out = s.verify::<M>(rows)?;
        s.commit(accepted)?;
        Ok(out.to_vec())
    }

    fn stage(
        s: &mut Seat,
        r: &[u32],
        idx: usize,
        (m, accepted): (usize, usize),
    ) -> Result<Vec<u32>, SessionError> {
        match m {
            2 => stage_m::<2>(s, r, idx, accepted),
            4 => stage_m::<4>(s, r, idx, accepted),
            _ => Err(SessionError::Refused(format!("a verify of {m} rows"))),
        }
    }

    /// A script of stages from a reset and the prompt, then
    /// [`STEPS_AFTER`] plain steps: whether every verify's rows, every
    /// step's id and the last step's logits equal the plain run's — bit for
    /// bit, or by the e2e gate's band where the pass's rows are not the
    /// steps' bits (which held, and the first difference, is said).
    fn script(
        s: &mut Seat,
        prompt: &[u32],
        reference: &(Vec<u32>, Vec<Vec<f32>>),
        stages: &[(usize, usize)],
    ) -> Result<(bool, String), GateError> {
        let (r, rows) = reference;
        s.reset()?;
        let first = s.prompt(prompt, Want::Argmax)?.argmax();
        if first != r[0] {
            return Ok((
                false,
                format!("the prompt's argmax {first}, the plain run's {}", r[0]),
            ));
        }
        let (mut idx, mut band, mut why) = (0, false, String::new());
        for &(m, accepted) in stages {
            let out = match stage(s, r, idx, (m, accepted)) {
                Ok(out) => out,
                Err(e) => return Ok((false, format!("verify {m}/{accepted} refused: {e}"))),
            };
            for (j, &got) in out.iter().enumerate() {
                if !id_holds(got, r[idx + 1 + j], &rows[idx + 1 + j], &mut band, &mut why) {
                    return Ok((
                        false,
                        format!("verify of {m} rows at {idx}: row {} {}", j + 1, why),
                    ));
                }
            }
            idx += accepted;
        }
        let mut last_row = Vec::new();
        for i in 0..STEPS_AFTER {
            let out = match s.step(r[idx + i], Want::Logits) {
                Ok(out) => out,
                Err(e) => return Ok((false, format!("step {i} after the commits: {e}"))),
            };
            if !id_holds(
                out.argmax(),
                r[idx + i + 1],
                &rows[idx + i + 1],
                &mut band,
                &mut why,
            ) {
                return Ok((false, format!("step {i}: {why}")));
            }
            if let runtime::Out::Logits { row, .. } = out {
                last_row = row.to_vec();
            }
        }
        let bits = same_bits(&last_row, &rows[idx + STEPS_AFTER]);
        if !bits {
            let (cosine, max_delta) = compare(&last_row, &rows[idx + STEPS_AFTER]);
            if cosine < COSINE_FLOOR {
                return Ok((
                    false,
                    format!(
                        "the last step's logits: cosine {cosine:.6} under the floor \
                         {COSINE_FLOOR} (max |delta| {max_delta:.4})"
                    ),
                ));
            }
            band = true;
            if why.is_empty() {
                why = format!("the logits by the band (cosine {cosine:.6})");
            }
        }
        let held = if band {
            format!("the band: {why}")
        } else {
            "bit for bit".to_string()
        };
        Ok((
            true,
            format!("{} commits, {STEPS_AFTER} steps, {held}", stages.len()),
        ))
    }

    /// (a) (module doc).
    fn every_accepted(
        s: &mut Seat,
        refs: &(Vec<u32>, Vec<Vec<f32>>),
        prompt: &[u32],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        for mode in [StepMode::Graph, StepMode::Eager] {
            s.model_mut().set_mode(mode);
            let (mut red, mut first_red, mut cases) = (0, String::new(), 0);
            for m in [2usize, 4] {
                for accepted in 1..=m {
                    let (pass, note) = script(s, prompt, refs, &[(m, accepted)])?;
                    cases += 1;
                    if !pass {
                        red += 1;
                        println!("(a) {mode:?} m={m} accepted={accepted}: {note} FAIL");
                        if first_red.is_empty() {
                            first_red = note;
                        }
                    }
                }
            }
            println!(
                "(a) {mode:?}: {cases} cases (m in 2, 4; accepted 1..=m), {red} red {}",
                verdict(red == 0)
            );
            ok &= red == 0;
            let (pass, note) = script(s, prompt, refs, &CHAIN)?;
            println!("(a) {mode:?} chain {CHAIN:?}: {note} {}", verdict(pass));
            ok &= pass;
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// One lookup-drafted generation of [`GEN`] tokens from `prompt`: the
    /// tokens, the proposals, the drafts accepted and the partial commits.
    fn lookup_run(
        s: &mut Seat,
        prompt: &[u32],
    ) -> Result<(Vec<u32>, usize, usize, usize), GateError> {
        s.reset()?;
        let mut spec: Speculative<Lookup, PAIR> =
            s.with_draft::<Lookup, PAIR>(Lookup::new(), &mut Log)?;
        let first = Advance::prompt(&mut spec, s, prompt)?;
        Advance::begin(&mut spec, s, prompt, first)?;
        let mut tokens = vec![first];
        let (mut proposals, mut accepted, mut partial) = (0usize, 0usize, 0usize);
        while tokens.len() < GEN {
            let last = tokens[tokens.len() - 1];
            let c = Advance::pass(&mut spec, s, last, &mut tokens)?;
            if c.proposed {
                proposals += 1;
                accepted += c.kept - 1;
                partial += usize::from(c.kept < c.rows);
            }
        }
        Ok((tokens, proposals, accepted, partial))
    }

    /// (b) (module doc): the equality in every tier, the partial-commit pin
    /// in the real tier only.
    fn lookup(s: &mut Seat, refs: &[(Vec<u32>, Vec<u32>)], tier: Tier) -> Result<bool, GateError> {
        let mut ok = true;
        let mut partial_on_some = false;
        for mode in [StepMode::Graph, StepMode::Eager] {
            s.model_mut().set_mode(mode);
            for (i, (prompt, want)) in refs.iter().enumerate() {
                match lookup_run(s, prompt) {
                    Ok((tokens, proposals, accepted, partial)) => {
                        let same = tokens[..GEN] == want[..GEN];
                        partial_on_some |= accepted > 0 && accepted < proposals;
                        println!(
                            "(b) {mode:?} prompt {i} ({} ids): {GEN} tokens equal the plain \
                             run's {same}; proposals {proposals}, accepted drafts {accepted}, \
                             partial commits {partial} {}",
                            prompt.len(),
                            verdict(same)
                        );
                        ok &= same;
                    }
                    Err(e) => {
                        println!(
                            "(b) {mode:?} prompt {i}: ended in an error: {e} {}",
                            verdict(false)
                        );
                        ok = false;
                    }
                }
            }
        }
        let pin = tier == Tier::Real;
        if pin {
            println!(
                "(b) accepted drafts above 0 and below the proposals on some run (partial \
                 commits ran) {}",
                verdict(partial_on_some)
            );
        } else {
            println!(
                "(b) the partial-commit pin is the real tier's; the fixture tier pins the \
                 equality only (the counts above are information) {}",
                verdict(true)
            );
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok && (!pin || partial_on_some))
    }

    /// (e) (module doc): the refusals by name, each leaving the verify
    /// committable.
    fn refusals(
        s: &mut Seat,
        refs: &(Vec<u32>, Vec<Vec<f32>>),
        prompt: &[u32],
    ) -> Result<bool, GateError> {
        let r = &refs.0;
        s.reset()?;
        s.prompt(prompt, Want::Argmax)?;
        let pos0 = s.model().pos();
        s.verify::<4>(std::array::from_fn(|j| r[j]))?;
        let said = |e: Result<(), GpuError>| e.err().map(|e| e.to_string()).unwrap_or_default();
        let step = said(s.model_mut().step(&[r[1]]).map(|_| ()));
        let pass = said(s.model_mut().step_rows::<2>([r[1], r[2]]).map(|_| ()));
        let prompt_call = said(s.model_mut().prefill(&prompt[..8]).map(|_| ()));
        let behind = said(s.model_mut().rollback(pos0));
        let waits = |text: &str| text.contains("waits for its commit");
        let behind_named = behind.contains("keeps 1..=");
        let named = waits(&step) && waits(&pass) && waits(&prompt_call) && behind_named;
        println!(
            "(e) step while the verify waits: {step} {}",
            verdict(waits(&step))
        );
        println!(
            "(e) pass while the verify waits: {pass} {}",
            verdict(waits(&pass))
        );
        println!(
            "(e) prompt while the verify waits: {prompt_call} {}",
            verdict(waits(&prompt_call))
        );
        println!(
            "(e) a rollback behind the pending rows: {behind} {}",
            verdict(behind_named)
        );
        s.commit(2)?;
        let after = s.step(r[2], Want::Argmax)?.argmax();
        let committable = after == r[3];
        println!(
            "(e) the refused calls moved nothing: commit(2) then the step gives {after} (the \
             plain run's {}) {}",
            r[3],
            verdict(committable)
        );
        Ok(named && committable)
    }

    /// (e)'s plain-load and open refusals: a plain load's verify, and
    /// `open_verifying` with rows 1 and past [`MAX_PASS_ROWS`], each refused
    /// before anything uploads.
    fn open_refusals(s: &mut Seat, prompt: &[u32]) -> Result<bool, GateError> {
        // The plain run's next ids, stepped here: the plain seat stands
        // after its prompt.
        let mut r = Vec::new();
        let mut last = s.prompt(prompt, Want::Argmax)?.argmax();
        for _ in 0..4 {
            r.push(last);
            last = s.step(last, Want::Argmax)?.argmax();
        }
        let refused = s
            .verify::<2>([r[0], r[1]])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let plain = refused.contains("cannot take rows back");
        println!("(e) a plain load's verify: {refused} {}", verdict(plain));
        let said = |rows: usize| -> String {
            open_split(Arch::NemotronHMoe, NAME)
                .and_then(|f| Ok((Gpu::new()?, f)))
                .and_then(|(g, f)| Ok(NemotronHMoeModel::open_verifying(g, f, CTX, rows)?))
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        };
        let one = said(1);
        let past = said(MAX_PASS_ROWS + 1);
        let rows_named = one.contains("2..=") && past.contains("2..=");
        println!(
            "(e) open_verifying of 1 row: {one}; of {} rows: {past} {}",
            MAX_PASS_ROWS + 1,
            verdict(rows_named)
        );
        Ok(plain && rows_named)
    }

    /// A clause's verdict; an error it ended in is a red line of its own, and
    /// the seat is reset so the next clause starts from a model that can run.
    fn clause(name: &str, s: &mut Seat, r: Result<bool, GateError>) -> bool {
        match r {
            Ok(ok) => ok,
            Err(e) => {
                println!("{name}: ended in an error: {e} {}", verdict(false));
                if let Err(e) = s.reset() {
                    println!("{name}: the reset after it: {e}");
                }
                false
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let tier = Tier::from_env().map_err(|e| e.to_string())?;
        println!("{NAME}: the {} tier", tier.name());
        let (hp, n_mamba, d_inner) = header()?;
        let prompt_a = lcg(PROMPT_A, hp.n_vocab, 0x9e37_79b9_7f4a_7c15);
        // The plain phase: the reference run and (c)'s plain bytes.
        let mut plain = open_seat(false, &hp, tier)?;
        let plain_bytes = plain.model().resident_bytes();
        let refs = greedy(&mut plain, &prompt_a, REF_IDS, true)?;
        let lookups = {
            // Three prompts of one short walk repeated (8 ids × 5): the
            // lookup draft's n-grams come from the prompt itself. The walks
            // sit on tiny alphabets (4 ids, an alternating pair, one
            // constant) — a real file's stream echoes such a period, where
            // a spread over the vocabulary never repeats an n-gram at all;
            // the fixture's stream repeats no token even here, which the
            // diagnostic below says.
            let t = lcg(1, hp.n_vocab, 7)[0];
            let four = lcg(8, 4, 1).iter().map(|&x| t + x).collect::<Vec<_>>();
            let pair = [t, t.wrapping_add(1)];
            let prompts: Vec<Vec<u32>> =
                vec![four.repeat(5), pair.repeat(20), vec![t; 8].repeat(5)];
            let mut out = Vec::new();
            for p in prompts {
                let (ids, _) = greedy(&mut plain, &p, GEN + 8, false)?;
                // The diagnostic behind the clause: how many of the run's
                // tokens an earlier one repeats — the n-grams a lookup could
                // propose from at all.
                let mut seen = std::collections::BTreeSet::new();
                seen.extend(p.iter().copied());
                let repeats = ids.iter().filter(|x| !seen.insert(**x)).count();
                println!(
                    "(b) the plain run of a repeated walk repeats a token {repeats} times \
                     (information)"
                );
                out.push((p, ids));
            }
            out
        };
        let r = open_refusals(&mut plain, &prompt_a);
        let mut ok = clause("(e)", &mut plain, r);
        drop(plain);
        // The verifying phase: (c)'s delta, then the clauses on one seat.
        let mut s = open_seat(true, &hp, tier)?;
        let lanes = n_mamba * (VERIFY_ROWS - 1) * d_inner * 128 * 4;
        let got = s.model().resident_bytes();
        let bytes_ok = got == plain_bytes + lanes;
        println!(
            "(c) the plain load holds {plain_bytes} bytes; the verifying load {} (want the \
             plain bytes and {lanes} = {n_mamba} Mamba-2 layers x {VERIFY_ROWS} − 1 lane rows x \
             {d_inner} x 128 x 4) {}",
            got,
            verdict(bytes_ok)
        );
        ok &= bytes_ok;
        let r = every_accepted(&mut s, &refs, &prompt_a);
        ok &= clause("(a)", &mut s, r);
        let r = lookup(&mut s, &lookups, tier);
        ok &= clause("(b)", &mut s, r);
        let r = refusals(&mut s, &refs, &prompt_a);
        ok &= clause("(e)", &mut s, r);
        println!("{NAME}: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
