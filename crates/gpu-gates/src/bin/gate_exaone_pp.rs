//! The K-EXAONE-2.0 prompt-batch gate: the fixture GGUF's prompt sets fed through the composed
//! program's prompt batch ([`prefill`] through [`GpuModel::run_rows`], a batched walk of up to
//! `PROMPT_ROWS` rows a unit) against the same ids run one decode step at a time, on the gate
//! card. The Kimi K3 pp gate's method on this body's own pass.
//!
//! What the bits hold. A pass of at most `GATED_FROM − 1` rows (8, the shared routed block's own
//! bound: below it every gated site runs the step's per-column gemvs, `bloomery_gpu_flint`'s
//! `GATED_FROM`) leaves each row's reductions the step's own: the shared expert's gate·up and down
//! by the step's launches, the card experts' rows the step's under the same slot map
//! (`shared_routed::Routed::card_rows`), the attention's projections by the linear owner's
//! m-column launches (a column a row) and its flash core over each row's own keys. A wider unit (a
//! whole prompt in one call, units of `PROMPT_ROWS` rows) sums a row's products in the wide GEMMs'
//! tile order and is not the steps' bits: that call is held to the oracle's own row within the e2e
//! gate's logits band, and its state by the steps after it.
//!
//! For each prompt set of [`SETS`] (an English paragraph of 24 ids, a Korean one of 18, a Rust
//! function head of 30 and a five-id question, refset family `lcpp-exaone-moe`), against the same
//! ids run as steps (the reference: every position's logits row and argmax, then [`CONT`] greedy
//! steps on):
//!
//! - `rows_are_steps`: the prompt as passes of [`PASS_ROWS`] rows and again as passes of
//!   [`ODD_ROWS`] — the two runs' passes end on different ids (8, 8, 8 and 5, 5, 5, 5, 5 over 24
//!   ids), so a boundary at each residue runs — each pass's last logits row and argmax bit for bit
//!   the step's at that position.
//! - `state_advances_as_steps`: after those passes, [`CONT`] decode steps on the reference's own
//!   ids give the reference's logits and argmax bit for bit: the stores and the position book stand
//!   where as many steps leave them.
//! - `prompt_call_is_steps`: the whole prompt as one batched call from a reset, then [`CONT`]
//!   steps on the reference's ids: each is the reference's bit for bit (the call's state is the
//!   steps'); the call's own last row is printed beside the step's, not held — its sum order is the
//!   wide GEMMs'.
//! - `last_row_is_vendors`: that call's last logits row within [`LOGITS_REL_BAND`] relative L2 of
//!   the oracle's `result_output` at the prompt's last position, and the steps' row's distance
//!   printed beside it.
//! - `pass_refused`: a pass of no rows, a token past the vocabulary and a call past the context
//!   are each refused by name — all three before any launch, so the model stands where it did: the
//!   next step is the reference's first.

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

    /// The continuation of [`CONT`] steps on the reference's own ids, each against the
    /// reference's position `from + j`: how many differ.
    fn continuation(m: &mut Loaded, reference: &Trace, from: usize) -> Result<usize, GateError> {
        let mut differing = 0usize;
        for j in 0..CONT {
            let row = m.step(reference.ids[from + j - 1])?;
            let id = argmax(&row);
            if id != reference.ids[from + j] || !bits_equal(&row, &reference.logits[from + j]) {
                differing += 1;
            }
        }
        Ok(differing)
    }

    /// Clauses `rows_are_steps` and `state_advances_as_steps` of set `s` at passes of `p` rows:
    /// the prompt as batched calls of `p` rows from a reset, each call's last row the step's bits
    /// at that position, then the continuation.
    fn pass_clauses(
        m: &mut Loaded,
        s: &Set,
        reference: &Trace,
        p: usize,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let mut differing = 0usize;
        let mut sizes = Vec::new();
        for (c, chunk) in s.prompt.chunks(p).enumerate() {
            let (row, _) = m
                .pass(chunk)?
                .ok_or("a pass runs the head on its last row")?;
            let id = argmax(&row);
            let at = c * p + chunk.len() - 1;
            if id != reference.ids[at] || !bits_equal(&row, &reference.logits[at]) {
                differing += 1;
            }
            sizes.push(chunk.len());
        }
        let mut ok = true;
        let name = format!("rows_are_steps set={} p={p}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let pass = differing == 0;
            println!(
                "{name} rows={} chunks={sizes:?} differing_pass_ends={} {}",
                s.prompt.len(),
                differing,
                verdict(pass)
            );
            ok &= pass;
        }
        let name = format!("state_advances_as_steps set={} p={p}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let d = continuation(m, reference, s.prompt.len())?;
            let pass = d == 0;
            println!(
                "{name} continuation={CONT} differing_steps={d} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clauses `prompt_call_is_steps` and `last_row_is_vendors` of set `s`: the whole prompt as one
    /// batched call from a reset, its continuation against the reference's, its last row against
    /// the oracle's.
    fn prompt_clauses(m: &mut Loaded, s: &Set, reference: &Trace) -> Result<bool, GateError> {
        m.reset()?;
        let (row, _) = m
            .pass(&s.prompt)?
            .ok_or("the prompt call runs the head on its last row")?;
        let id = argmax(&row);
        let at = s.prompt.len() - 1;
        let same_row = id == reference.ids[at] && bits_equal(&row, &reference.logits[at]);
        let mut ok = true;
        let name = format!("prompt_call_is_steps set={}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let d = continuation(m, reference, s.prompt.len())?;
            let pass = d == 0;
            println!(
                "{name} ids={} last_row_same_bits={same_row} (the wide GEMMs' own sum order, \
                 printed, not held) continuation_differing_steps={d} {}",
                s.prompt.len(),
                verdict(pass)
            );
            ok &= pass;
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
            "positions the stores hold",
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
             rows gives each pass's last logits row and argmax bit for bit the decode steps' at \
             that position, the steps after the passes and after the whole-prompt call are the \
             reference's bit for bit, that call's last row is within {LOGITS_REL_BAND:.1e} of the \
             oracle's, and the calls the body does not take are refused by name",
            PASS_ROWS
        );
        Ok(())
    }
}
