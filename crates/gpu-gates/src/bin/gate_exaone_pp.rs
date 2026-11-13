//! The K-EXAONE-2.0 prompt-batch gate: the fixture GGUF's prompt sets fed through the composed
//! program's prompt batch ([`prefill`] through [`GpuModel::run_rows`], a batched walk of up to
//! `PROMPT_ROWS` rows a unit) against the same ids run one decode step at a time, on the gate
//! card. The Kimi K3 pp gate's method on this body's own pass.
//!
//! What the bits hold. A pass of at most `GATED_FROM − 1` rows (8) runs the step's own launches
//! per row everywhere but the dense lead block. The attention's projections are the linear
//! owner's m-column launches, whose per-cell order the 32-value GEMM's numeric contract pins as
//! a function of K alone, never of the grid, a slot's tile or the slot count
//! (`gemm/gemm32.rs`); its core attends each row over its own keys; the router's batch entries
//! are bit for bit what the one-token launch writes (the router instance's own doc); the shared
//! expert below `GATED_FROM` runs the step's per-column gemvs and the card experts' slots are
//! what the step's launches write, whatever the row count (`card_experts`' doc). The dense
//! block's batch (`shared_routed::Routed::dense_rows`) runs the wide gated GEMMs at every row
//! count, as the shared expert's does from `GATED_FROM` rows (`shared_routed`'s doc) — another
//! order of the same exact integer block sums, not the step's gemv bits at any width, one row
//! included. Every pass row past the first dense block sits at that distance from the step's,
//! and so do the stores its rows write, so no clause of this gate holds a pass's row or the
//! steps after one to the step's bits: both are held the way every batched call of this engine
//! is held against another order of its own sums, [`logits_band`]'s pins (its module doc: the
//! gates that compare a batched call with another order of the same call defer to it) — the
//! pass rows through its band, the continuations through its walk.
//!
//! For each prompt set of [`SETS`] (an English paragraph of 24 ids, a Korean one of 18, a Rust
//! function head of 30 and a five-id question, refset family `lcpp-exaone-moe`), against the same
//! ids run as steps (the reference: every position's logits row and argmax, then [`CONT`] greedy
//! steps on):
//!
//! - `rows_near_steps`: the prompt as passes of [`PASS_ROWS`] rows and again as passes of
//!   [`ODD_ROWS`] — the two runs' passes end on different ids (8, 8, 8 and 5, 5, 5, 5, 5 over 24
//!   ids), so a boundary at each residue runs — each pass's last logits row folded into
//!   [`logits_band::LogitsBand`] against the step's row at that position: no clear top-1
//!   disagreement and the max and RMS |Δlogit| within the band's pins.
//! - `state_advances_near_steps`: after those passes, [`CONT`] decode steps on the reference's
//!   own ids walked against the reference's by [`logits_band::walk`]: the stores the passes
//!   wrote carry the wide block's distance, so the continuation runs on other values and is
//!   held to the walk's band, not to bits.
//! - `prompt_call_then_steps_near_steps`: the whole prompt as one batched call from a reset
//!   (whose shared expert runs the wide GEMMs too, past its own bound), then [`CONT`] steps on
//!   the reference's ids walked the same way; the call's own last row is printed beside the
//!   step's, not held.
//! - `last_row_is_vendors`: that call's last logits row within [`LOGITS_REL_BAND`] relative L2 of
//!   the oracle's `result_output` at the prompt's last position, and the steps' row's distance
//!   printed beside it.
//! - `pass_refused`: a pass of no rows, a token past the vocabulary and a call past the context
//!   are each refused by name — the first and the third by the pass's own runner
//!   (`GpuModel::run_rows`: no rows, then the resident cache's bound; the walk's `call_end` bound
//!   never reaches its own refusal, the runner's check stands before the walk), the second by
//!   the walk's vocabulary check — all three before any launch, so the model stands where it
//!   did: the next step is the reference's first.

#[cfg(feature = "flint")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_exaone_pp", gate::run())
}

#[cfg(feature = "flint")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/exaone_seat.rs"]
mod seat;

#[cfg(feature = "flint")]
mod gate {
    use crate::e2e::{argmax, rel};
    use crate::lcpp_e2e::Seat as _;
    use crate::seat::{LOGITS_REL_BAND, Loaded};
    use bloomery_gpu_gates::logits_band::{
        self, LOGITS_CLEAR_PIN, LOGITS_MAX_PIN, LOGITS_RMS_PIN, LogitsBand,
    };
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, ref_tensor_logical_in, verdict,
    };
    use refset::arch::exaone_moe::{self, A, B, C, D};
    use refset::arch::prompt_tables;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_exaone_pp";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-exaone-pp";

    /// The sets the pass is held to, in the order they run: prompts of 24, 18, 30 and 5 ids.
    const SETS: &[&str] = &[A, B, C, D];

    /// Rows of a pass that keeps every row's reductions the step's own: one under the shared
    /// routed block's `GATED_FROM` (below it the gated sites run the step's per-column gemvs).
    const PASS_ROWS: usize = bloomery_gpu_flint::GATED_FROM - 1;

    /// The odd pass size, so the two runs' passes end on different ids.
    const ODD_ROWS: usize = 5;

    /// Decode steps run after a prompt, on the reference's own ids.
    const CONT: usize = 4;

    /// The context the gate's caches hold: the longest prompt and its continuation fit, with the
    /// room the refusal clause's past-the-context call needs.
    const CTX: usize = 64;

    /// A token past the file's vocabulary.
    const PAST_VOCAB: u32 = 1_000_000;

    /// One set's prompt and the oracle's whole logits row at its last position.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        logits: Vec<f32>,
    }

    fn open_set(clause: &str, name: &'static str) -> Result<Set, GateError> {
        let path = exaone_moe::LCPP.path(name);
        let man =
            RefManifest::open(&path, &exaone_moe::LCPP).map_err(|e| format!("{clause}: {e}"))?;
        let row = man.tensor("result_output", 0)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let vocab = usize::try_from(row.ne[0])?;
        if vocab == 0 || all.len() % vocab != 0 {
            return Err(format!("{name}: result_output is {:?}", row.ne).into());
        }
        let prompt = prompt_tables::tokens(&man)?.to_vec();
        if prompt.len() + CONT > CTX {
            return Err(format!(
                "{name}: {} positions and {CONT} more, the gate's caches hold {CTX}",
                prompt.len()
            )
            .into());
        }
        Ok(Set {
            name,
            prompt,
            logits: all[all.len() - vocab..].to_vec(),
        })
    }

    /// What a run leaves: every position's logits row and the argmax after it.
    struct Trace {
        logits: Vec<Vec<f32>>,
        ids: Vec<u32>,
    }

    /// The reference: set `s`'s prompt one decode step at a time from a reset, then [`CONT`]
    /// greedy steps.
    fn steps(m: &mut Loaded, s: &Set) -> Result<Trace, GateError> {
        m.reset()?;
        let mut t = Trace {
            logits: Vec::new(),
            ids: Vec::new(),
        };
        for i in 0..s.prompt.len() + CONT {
            let input = if i < s.prompt.len() {
                s.prompt[i]
            } else {
                t.ids[i - 1]
            };
            let row = m.step(input)?;
            t.ids.push(argmax(&row));
            t.logits.push(row);
        }
        Ok(t)
    }

    /// The continuation of [`CONT`] steps on the reference's own ids from position `from`: the
    /// ids the run took and the rows it took them from, as [`logits_band::walk`] takes a run.
    fn continuation(
        m: &mut Loaded,
        reference: &Trace,
        from: usize,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
        let mut ids = Vec::with_capacity(CONT);
        let mut rows = Vec::with_capacity(CONT);
        for j in 0..CONT {
            let row = m.step(reference.ids[from + j - 1])?;
            ids.push(argmax(&row));
            rows.push(row);
        }
        Ok((ids, rows))
    }

    /// The reference's positions `from..` as [`logits_band::walk`]'s reference.
    fn tail(reference: &Trace, from: usize) -> logits_band::Positions<'_> {
        logits_band::Positions {
            tokens: &reference.ids[from..],
            rows: &reference.logits[from..],
        }
    }

    /// A row as the bits [`LogitsBand::add`] takes.
    fn bits(row: &[f32]) -> Vec<u32> {
        row.iter().map(|v| v.to_bits()).collect()
    }

    /// Clauses `rows_near_steps` and `state_advances_near_steps` of set `s` at passes of `p`
    /// rows: the prompt as batched calls of `p` rows from a reset, each call's last row folded
    /// into the band against the step's at that position, then the continuation walked.
    fn pass_clauses(
        m: &mut Loaded,
        s: &Set,
        reference: &Trace,
        p: usize,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let mut band = LogitsBand::new();
        let mut same_bits = 0usize;
        let mut sizes = Vec::new();
        for (c, chunk) in s.prompt.chunks(p).enumerate() {
            let (row, _) = m
                .pass(chunk)?
                .ok_or("a pass runs the head on its last row")?;
            let at = c * p + chunk.len() - 1;
            if argmax(&row) == reference.ids[at] && bits_equal(&row, &reference.logits[at]) {
                same_bits += 1;
            }
            band.add(&bits(&row), &bits(&reference.logits[at]));
            sizes.push(chunk.len());
        }
        let mut ok = true;
        let name = format!("rows_near_steps set={} p={p}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let pass = band.ok();
            println!(
                "{name} rows={} chunks={sizes:?} pass_ends={} of them the step's bits (printed, \
                 not held); band {} clear-top-1 of {} clear (pin {LOGITS_CLEAR_PIN}), max \
                 |Δlogit| {:.3e} (pin {LOGITS_MAX_PIN}), rms {:.3e} (pin {LOGITS_RMS_PIN}) {}",
                s.prompt.len(),
                same_bits,
                band.disagree,
                band.clear,
                band.max,
                band.rms(),
                verdict(pass)
            );
            ok &= pass;
        }
        let name = format!("state_advances_near_steps set={} p={p}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let (ids, rows) = continuation(m, reference, s.prompt.len())?;
            let walked = logits_band::walk(
                &format!("{name} continuation={CONT}"),
                tail(reference, s.prompt.len()),
                logits_band::Positions {
                    tokens: &ids,
                    rows: &rows,
                },
            );
            ok &= walked.ok;
        }
        Ok(ok)
    }

    /// Clauses `prompt_call_then_steps_near_steps` and `last_row_is_vendors` of set `s`: the whole
    /// prompt as one batched call from a reset, its continuation walked against the reference's,
    /// its last row against the oracle's.
    fn prompt_clauses(m: &mut Loaded, s: &Set, reference: &Trace) -> Result<bool, GateError> {
        m.reset()?;
        let (row, _) = m
            .pass(&s.prompt)?
            .ok_or("the prompt call runs the head on its last row")?;
        let at = s.prompt.len() - 1;
        let same_row = argmax(&row) == reference.ids[at] && bits_equal(&row, &reference.logits[at]);
        let mut ok = true;
        let name = format!("prompt_call_then_steps_near_steps set={}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            println!(
                "{name} ids={} last_row_same_bits={same_row} (the wide GEMMs' own sum order, \
                 printed, not held)",
                s.prompt.len(),
            );
            let (ids, rows) = continuation(m, reference, s.prompt.len())?;
            let walked = logits_band::walk(
                &format!("{name} continuation={CONT}"),
                tail(reference, s.prompt.len()),
                logits_band::Positions {
                    tokens: &ids,
                    rows: &rows,
                },
            );
            ok &= walked.ok;
        }
        let name = format!("last_row_is_vendors set={}", s.name);
        if tier::run_clause(&name, Tag::Oracle)? {
            let call_rel = rel(&row, &s.logits);
            let steps_rel = rel(&reference.logits[at], &s.logits);
            let pass = call_rel <= LOGITS_REL_BAND;
            println!(
                "{name} call_rel={call_rel:.4e} steps_rel={steps_rel:.4e} \
                 band={LOGITS_REL_BAND:.1e} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause `pass_refused`: the calls the body refuses by name, each before any launch, then the
    /// model where it stood: its next step is the reference's first.
    fn pass_refused(m: &mut Loaded, s: &Set, reference: &Trace) -> Result<bool, GateError> {
        let name = "pass_refused";
        if !tier::run_clause(name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        m.reset()?;
        let mut ok = true;
        let mut refused = |what: &str, run: Result<(), GateError>, want: &str| {
            let text = run.err().map(|e| e.to_string());
            let pass = text.as_deref().is_some_and(|t| t.contains(want));
            println!(
                "  refused {what}: {text:?} (wants {want:?}) {}",
                verdict(pass)
            );
            ok &= pass;
        };
        refused(
            "a pass of no rows",
            m.pass(&[]).map(|_| ()),
            "a pass needs positions",
        );
        refused(
            "a token past the vocabulary",
            m.pass(&[PAST_VOCAB]).map(|_| ()),
            "past vocabulary",
        );
        let past: Vec<u32> = s.prompt.iter().copied().chain([s.prompt[0]; CTX]).collect();
        refused(
            "a call past the context",
            m.pass(&past).map(|_| ()),
            "exceeds the resident cache",
        );
        let pos_kept = m.pos() == 0;
        let row = m.step(s.prompt[0])?;
        let id = argmax(&row);
        let again = id == reference.ids[0] && bits_equal(&row, &reference.logits[0]);
        let pass = ok && pos_kept && again;
        println!(
            "{name} position_kept={pos_kept} next_step_is_reference={again} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? == Tier::Fixture {
            return Err(format!(
                "{NAME}: the exaone-moe family's file is the fixture GGUF itself and its sets are \
                 the real tier's, so the gate has no fixture tier: just {RECIPE}"
            )
            .into());
        }
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set("the pass clauses", n))
            .collect::<Result<_, _>>()?;
        let path = crate::lcpp_e2e::model_path(Tier::Real, RECIPE, exaone_moe::ARCH)?;
        let mut m = crate::seat::open_loaded(&path, CTX, crate::lcpp_e2e::Arm::A)?;
        let dims = m.dims();
        println!(
            "load arm=a layers={} embd={} vocab={} ctx={CTX}",
            dims.layers, dims.embd, dims.vocab
        );
        let mut ok = true;
        for s in &sets {
            let reference = steps(&mut m, s)?;
            for p in [PASS_ROWS, ODD_ROWS] {
                ok &= pass_clauses(&mut m, s, &reference, p)?;
            }
            ok &= prompt_clauses(&mut m, s, &reference)?;
            if s.name == SETS[0] {
                ok &= pass_refused(&mut m, s, &reference)?;
            }
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} on every set of {SETS:?} the prompt as passes of {} and {ODD_ROWS} \
             rows keeps each pass's last logits row within logits_band's pins of the decode \
             steps' at that position (the dense lead block's batch runs the wide GEMMs at every \
             width), the steps after the passes and after the whole-prompt call walk within the \
             band against the reference's, that call's last row is within {LOGITS_REL_BAND:.1e} \
             of the oracle's, and the calls the body does not take are refused by name",
            PASS_ROWS
        );
        Ok(())
    }
}
