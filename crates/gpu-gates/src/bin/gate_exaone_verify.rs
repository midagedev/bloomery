//! The K-EXAONE-2.0 verify gate: the body's pass of two rows (`impl Rows for FlintBody`,
//! `bloomery_gpu_flint::rows`: row 0 the token at the position, row 1 a drafted one, each through the
//! step's own launches, one layer apart, [`Chain::Pair`](bloomery_gpu::hybrid::Chain::Pair) on the host
//! tier) driven through `Session<ExaoneBody>`'s `Verify`, on the gate plan of the fixture GGUF (the pp
//! gate's file, card and plan: `Arm::A`).
//!
//! Every clause is a self-consistency clause that runs in the captured graph and eager:
//!
//! - `rows_match_steps`: at P in {1, 17} prompt ids (the prompt call's batch), a verify of `[t0, d]`
//!   (`t0` the prompt's argmax, `d` the plain run's next id, an accepted draft, or another id, a
//!   rejected one) gives each row's id and logits equal to two plain steps of `t0` then `d` from the
//!   same prompt, bit for bit. Unlike the prompt batch (whose dense lead block runs the wide GEMMs at
//!   every row count, `gate_exaone_pp`'s doc), the pass runs the step's own launches per row.
//! - `rollback_restores`: a 17-id prompt, a verify whose row 1 is rejected (commit 1) and one whose
//!   rows are both kept (commit 2), then plain steps to [`AFTER`] positions past the prompt: the kept
//!   and stepped ids and the last logits equal the plain run's, bit for bit (the body holds its stores
//!   private, so the stores' rows are read through the steps that follow).
//! - `lookup_is_plain`: a lookup draft (`runtime::Lookup`) driven over `Session<ExaoneBody>`
//!   (`Session::with_draft`, a verify of two rows) for [`GEN`] tokens on three prompts of a repeated
//!   pattern emits the plain run's ids on every prompt, and some prompt proposes. The draft's
//!   acceptance is printed per prompt. The fixture's weights do not continue the pattern, so a draft
//!   kept is a reading there, and `rows_match_steps` is the clause that keeps one.
//!
//! The MTP draft (`bloomery_gpu_flint::draft`, the file's `nextn` layer on the card beside the target, opened
//! as the serve seat opens it: `seat::open_model_drafted`) runs on its own load of the same file after those
//! clauses, one session, [`MTP_PROMPTS`] prompts of [`MTP_PROMPT`] ids, [`MTP_GEN`] generated tokens:
//!
//! - `mtp_is_plain`: the window loop (`Speculative<MtpDraft<ExaoneBody>, 2>`, a proposal of one id, a
//!   verify of two rows) emits the plain run's ids, graph and eager, and the draft proposes on some
//!   window: a draft that wrote into the target's buffers or moved its position moves a token. The
//!   fixture's trunk is a cut of the real one, so the acceptance printed per prompt is a reading there,
//!   and a real file's is the speed-up's `a`.
//! - `mtp_paths_agree`: the draft's proposal at a target position is the same id whichever way the
//!   target reached it: the plain step the draft follows (the step arena, the width chooser's shadow
//!   protocol) and the verify pass of two rows (the pair arena) whose second row is the plain run's next
//!   id on every other window (kept, commit 2) and another id on the rest (taken back, commit 1). The two
//!   runs' proposals are compared at every position both made, and both runs' tokens are the plain
//!   run's. A row copied from the wrong arena, a held count that a commit or a cut moved wrong, or a
//!   window that reads the step's row after the next one overwrote it gives another id there.
//!
//! The host tier: the gate plan holds every routed expert of the fixture on the card, so the pair chain
//! (each row of a verify pass through its own one-column call over the experts the host serves) would
//! never run above. The plain and the drafted load open again on a plan that holds about a quarter of the
//! routed experts on the card and the rest on the host (`seat::open_model_hosted`: the card budget is the
//! load's own floor and `seat::HOST_EXTRA`, the floor read off the refusal of a budget below it; the
//! residency is off, so the experts stay where the plan put them), and a load that holds none on the host
//! or none on the card is refused by name:
//!
//! - `host_rows_match_steps` and `host_rollback_restores`: the plain load's two clauses above, unchanged,
//!   on the host-tier plan: row 0 of a verify of two rows is the 1-row step's id and logits bit for bit
//!   (the pair chain sums a row as the step's one-column call does), and a commit of one or both rows
//!   restores the plain run's ids and last logits.
//! - `host_mtp_is_plain` and `host_mtp_paths_agree`: the drafted load's two clauses above on that plan.
//!
//! The file: the recipe runs the gate on the Q4_K fixture and again on the IQ2_XXS fixture (routed gate and up
//! IQ2_XXS, downs Q4_K, the rest Q8_0: the real second quant's mix, `tools/ref/exaone-iq2-fixture.sh`). A row of the
//! pass runs the step's own one-column launches, so the card's IQ2_XXS routed path and the host tier's
//! `iq2_xxs k=6144 path=fused` serve a verify row as they serve a step, and every clause holds on both files
//! unchanged. What a clause compares is two loads of one placement: a load whose draft takes card bytes plans fewer
//! card experts than a plain load at the same budget (`BLOOMERY_DRAFT`'s registry row), and an expert that moves
//! between the card and the host runs in the other tier's kernel, so the hosted clauses give each load its own
//! floor plus the same extra (the same experts on the card).
//!
//! `--time` (the runner `tools/ref/time-gate.sh`, `just time-gpu-exaone-verify`) runs no clause: it
//! prints the wall of one plain step and of one verify pass of two rows whose second id is the plain
//! run's own next one (an accepted draft), the medians over [`TIMED`] each, and their ratio, the
//! pass's cost in steps (`c_v`) the draft's speedup is divided by. A real file's host tier reads each
//! routed expert once for the rows that pick it, so the ratio carries the two rows' expert overlap.

#[cfg(feature = "flint")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_exaone_verify", gate::run())
}

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
    reason = "the card the gate plans on; the seat reads it through lcpp_e2e"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the seat serves three gates; each takes the calls it reads"
)]
#[path = "shared/exaone_seat.rs"]
mod seat;

#[cfg(feature = "flint")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "flint")]
mod gate {
    use crate::e2e::{same_bits, walk};
    use crate::greedy::greedy;
    use app::mtp::{MtpDraft, open_draft};
    use app::{RowsLog, Session, SessionError};
    use bloomery_gpu::model::{Rows, StepMode};
    use bloomery_gpu_flint::ExaoneBody;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use refset::arch::exaone_moe;
    use runtime::{Advance, Draft, Lookup, Out, RowLogits, Speculative, Target, Verify, Want};

    type Seat = Session<ExaoneBody>;

    const NAME: &str = "gate_exaone_verify";
    const RECIPE: &str = "gate-gpu-exaone-verify";

    /// The rows of every verify here: the body's pass.
    const ROWS: usize = 2;

    /// `rows_match_steps`' prompt lengths.
    const PROMPTS: [usize; 2] = [1, 17];

    /// `rollback_restores`' prompt, and the positions past it the drafted and the plain runs both
    /// reach.
    const PROMPT_B: usize = 17;
    const AFTER: usize = 8;

    /// `lookup_is_plain`'s generated tokens, and its prompts: a pattern of [`PATTERN`] ids
    /// [`REPEATS`] times.
    const GEN: usize = 32;
    const PATTERN: usize = 12;
    const REPEATS: usize = 4;
    const LOOKUPS: usize = 3;

    /// The MTP clauses' prompts (a walk of ids each [`MTP_PROMPT`] long) and generated tokens.
    const MTP_PROMPTS: usize = 2;
    const MTP_PROMPT: usize = 17;
    const MTP_GEN: usize = 24;

    /// `--time`'s plain steps, and the pass count that replays them as accepted pairs.
    const TIMED: usize = 64;
    /// The calls at the start of each timed run left out: the first of each captures its graph.
    const WARM: usize = 4;
    /// `--time`'s prompt.
    const TIME_PROMPT: usize = 17;

    /// A drafted id no clause's plain run follows with: the walk's first id, moved by one where the
    /// plain run's next id is it.
    const REJECT: u32 = 1_000;

    /// Rows of the caches: the longest run of the clauses (a lookup pass may run one row past
    /// [`GEN`]), with room.
    const CTX: usize = PATTERN * REPEATS + GEN + ROWS + 16;

    const MODES: [StepMode; 2] = [StepMode::Graph, StepMode::Eager];

    type Clause = fn(&mut Seat, usize) -> Result<bool, GateError>;

    const CLAUSES: &[(&str, Clause)] = &[
        ("rows_match_steps", rows_match_steps),
        ("rollback_restores", rollback_restores),
        ("lookup_is_plain", lookup_is_plain),
    ];

    /// The clauses that run on the drafted load, after [`CLAUSES`] on the plain one.
    const DRAFTED: &[(&str, Clause)] = &[
        ("mtp_is_plain", mtp_is_plain),
        ("mtp_paths_agree", mtp_paths_agree),
    ];

    /// The clauses that run again on the plain load with part of the routed experts on the host.
    const HOSTED: &[(&str, Clause)] = &[
        ("host_rows_match_steps", rows_match_steps),
        ("host_rollback_restores", rollback_restores),
    ];

    /// The clauses that run again on the drafted load with part of the routed experts on the host.
    const HOSTED_DRAFTED: &[(&str, Clause)] = &[
        ("host_mtp_is_plain", mtp_is_plain),
        ("host_mtp_paths_agree", mtp_paths_agree),
    ];

    fn rejected(next: u32) -> u32 {
        if next == REJECT { REJECT + 1 } else { REJECT }
    }

    /// A call's argmax and logits row; a call that read back its argmax only is refused by name.
    fn logits_of(out: Out<'_>) -> Result<(u32, Vec<f32>), GateError> {
        match out {
            Out::Logits { argmax, row } => Ok((argmax, row.to_vec())),
            Out::Argmax(_) => {
                Err("a call that asked for its logits read back its argmax only".into())
            }
        }
    }

    /// From a reset, `prompt` and its argmax `t0`.
    fn prompted(s: &mut Seat, prompt: &[u32]) -> Result<u32, GateError> {
        s.reset()?;
        Ok(s.prompt(prompt, Want::Argmax)?.argmax())
    }

    /// `n` greedy steps from `first` at the target's position: each step's argmax, and the last
    /// one's logits.
    fn steps_from(s: &mut Seat, first: u32, n: usize) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        let (mut ids, mut last) = (Vec::with_capacity(n), Vec::new());
        let mut id = first;
        for i in 0..n {
            let want = if i + 1 == n {
                Want::Logits
            } else {
                Want::Argmax
            };
            let out = s.step(id, want)?;
            id = out.argmax();
            if let Out::Logits { row, .. } = out {
                last = row.to_vec();
            }
            ids.push(id);
        }
        Ok((ids, last))
    }

    /// `rows_match_steps` (module doc).
    fn rows_match_steps(s: &mut Seat, n_vocab: usize) -> Result<bool, GateError> {
        const CLAUSE: &str = "rows_match_steps";
        let mut ok = true;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for p in PROMPTS {
                let prompt = walk(p, n_vocab);
                for accept in [true, false] {
                    let t0 = prompted(s, &prompt)?;
                    let step0 = logits_of(s.step(t0, Want::Logits)?)?;
                    let d = if accept { step0.0 } else { rejected(step0.0) };
                    let step1 = logits_of(s.step(d, Want::Logits)?)?;
                    let t0_pass = prompted(s, &prompt)?;
                    let ids = s.verify([t0_pass, d])?;
                    let row0 = s.row_logits(0)?.to_vec();
                    let row1 = s.row_logits(1)?.to_vec();
                    s.commit(ROWS)?;
                    let same_t0 = t0_pass == t0;
                    let same_ids = ids == [step0.0, step1.0];
                    let bits0 = same_bits(&row0, &step0.1);
                    let bits1 = same_bits(&row1, &step1.1);
                    let pass = same_t0 && same_ids && bits0 && bits1;
                    println!(
                        "{CLAUSE} {mode:?} P={p} {} draft: the prompt's argmax the same {same_t0}, \
                         row ids {ids:?} against the steps' [{}, {}] {same_ids}, row 0 logits bit \
                         for bit {bits0}, row 1 {bits1} {}",
                        if accept { "accepted" } else { "rejected" },
                        step0.0,
                        step1.0,
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// `rollback_restores` (module doc).
    fn rollback_restores(s: &mut Seat, n_vocab: usize) -> Result<bool, GateError> {
        const CLAUSE: &str = "rollback_restores";
        let prompt = walk(PROMPT_B, n_vocab);
        let mut ok = true;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            let t0 = prompted(s, &prompt)?;
            let (plain, plain_last) = steps_from(s, t0, AFTER)?;
            for accepted in [1, ROWS] {
                let t0_pass = prompted(s, &prompt)?;
                let d = if accepted == ROWS {
                    plain[0]
                } else {
                    rejected(plain[0])
                };
                let rows = s.verify([t0_pass, d])?;
                s.commit(accepted)?;
                let mut ids = rows[..accepted].to_vec();
                let (after, last) = steps_from(s, ids[accepted - 1], AFTER - accepted)?;
                ids.extend(after);
                let same_ids = t0_pass == t0 && ids == plain;
                let bits = same_bits(&last, &plain_last);
                let pass = same_ids && bits;
                println!(
                    "{CLAUSE} {mode:?} commit {accepted} of [{t0_pass}, {d}]: {} ids against the \
                     plain run's {same_ids}, the last logits bit for bit {bits}, at position {} {}",
                    ids.len(),
                    s.pos(),
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    struct Log;

    impl RowsLog for Log {
        fn capture_rows(
            &mut self,
            rows: usize,
            nodes: usize,
            _: app::PhaseBytes,
        ) -> Result<(), SessionError> {
            println!("verify pass of {rows} rows captured: {nodes} nodes");
            Ok(())
        }
    }

    /// One lookup-drafted generation of [`GEN`] tokens from `prompt`: the tokens, the proposals and
    /// the drafts kept.
    fn lookup_run(s: &mut Seat, prompt: &[u32]) -> Result<(Vec<u32>, usize, usize), GateError> {
        s.reset()?;
        let mut spec: Speculative<Lookup, ROWS> =
            s.with_draft::<Lookup, ROWS>(Lookup::new(), &mut Log)?;
        let first = Advance::prompt(&mut spec, s, prompt)?;
        Advance::begin(&mut spec, s, prompt, first)?;
        let mut tokens = vec![first];
        let (mut proposals, mut accepted) = (0usize, 0usize);
        while tokens.len() < GEN {
            let last = tokens[tokens.len() - 1];
            let c = Advance::pass(&mut spec, s, last, &mut tokens)?;
            if c.proposed {
                proposals += 1;
                accepted += c.kept - 1;
            }
        }
        Ok((tokens, proposals, accepted))
    }

    /// `lookup_is_plain` (module doc).
    fn lookup_is_plain(s: &mut Seat, n_vocab: usize) -> Result<bool, GateError> {
        const CLAUSE: &str = "lookup_is_plain";
        let base = walk(PATTERN * LOOKUPS, n_vocab);
        let prompts: Vec<Vec<u32>> = base.chunks(PATTERN).map(|p| p.repeat(REPEATS)).collect();
        let mut ok = true;
        let mut proposed_some = false;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for (i, prompt) in prompts.iter().enumerate() {
                let (want, _) = greedy(s, prompt, GEN, false)?;
                let (tokens, proposals, accepted) = lookup_run(s, prompt)?;
                let same = tokens.get(..GEN) == Some(&want[..]);
                proposed_some |= proposals > 0;
                println!(
                    "{CLAUSE} {mode:?} prompt {i} ({} ids): {GEN} tokens equal the plain run's \
                     {same}; proposals {proposals}, drafts kept {accepted} {}",
                    prompt.len(),
                    verdict(same)
                );
                ok &= same;
            }
        }
        println!(
            "{CLAUSE}: a draft proposed on some prompt {}",
            verdict(proposed_some)
        );
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok && proposed_some)
    }

    /// The MTP clauses' prompts.
    fn mtp_prompts(n_vocab: usize) -> Vec<Vec<u32>> {
        walk(MTP_PROMPT * MTP_PROMPTS, n_vocab)
            .chunks(MTP_PROMPT)
            .map(<[u32]>::to_vec)
            .collect()
    }

    /// One MTP-drafted generation of [`MTP_GEN`] tokens from `prompt` through the window loop: the
    /// tokens, the proposals and the drafts kept.
    fn mtp_run(s: &mut Seat, prompt: &[u32]) -> Result<(Vec<u32>, usize, usize), GateError> {
        s.reset()?;
        let draft = open_draft(s.model(), (), StepMode::Eager)?;
        let mut spec: Speculative<MtpDraft<ExaoneBody>, ROWS> =
            s.with_draft::<MtpDraft<ExaoneBody>, ROWS>(draft, &mut Log)?;
        let first = Advance::prompt(&mut spec, s, prompt)?;
        Advance::begin(&mut spec, s, prompt, first)?;
        let mut tokens = vec![first];
        let (mut proposals, mut accepted) = (0usize, 0usize);
        while tokens.len() < MTP_GEN {
            let last = tokens[tokens.len() - 1];
            let c = Advance::pass(&mut spec, s, last, &mut tokens)?;
            if c.proposed {
                proposals += 1;
                accepted += c.kept - 1;
            }
        }
        Ok((tokens, proposals, accepted))
    }

    /// `mtp_is_plain` (module doc).
    fn mtp_is_plain(s: &mut Seat, n_vocab: usize) -> Result<bool, GateError> {
        const CLAUSE: &str = "mtp_is_plain";
        let mut ok = true;
        let mut proposed_some = false;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for (i, prompt) in mtp_prompts(n_vocab).iter().enumerate() {
                let (want, _) = greedy(s, prompt, MTP_GEN, false)?;
                let (tokens, proposals, accepted) = mtp_run(s, prompt)?;
                let same = tokens.get(..MTP_GEN) == Some(&want[..]);
                proposed_some |= proposals > 0;
                println!(
                    "{CLAUSE} {mode:?} prompt {i} ({} ids): {MTP_GEN} tokens equal the plain run's \
                     {same}; windows {proposals}, drafts kept {accepted} {}",
                    prompt.len(),
                    verdict(same)
                );
                ok &= same;
            }
        }
        println!(
            "{CLAUSE}: the draft proposed on some window {}",
            verdict(proposed_some)
        );
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok && proposed_some)
    }

    /// The draft's proposal at each target position of one run over `want` (the plain run's ids): the
    /// plain steps, each followed by the draft as the width chooser's shadow window follows one
    /// (propose, then `unproposed`, then the step and `held`) when `stepped`, else the verify passes of
    /// two rows. Each is the run's tokens too, which are checked against `want`.
    fn proposals(
        s: &mut Seat,
        prompt: &[u32],
        want: &[u32],
        stepped: bool,
    ) -> Result<(Vec<(u32, u32)>, bool), GateError> {
        s.reset()?;
        let mut d = open_draft(s.model(), (), StepMode::Eager)?;
        let first = d.prompt(s, prompt)?;
        d.begin(s, prompt, first)?;
        let mut tokens = vec![first];
        let mut found = Vec::new();
        let mut window = 0usize;
        while tokens.len() < want.len() {
            let last = tokens[tokens.len() - 1];
            let (pos, mut one) = (s.pos(), [0u32; 1]);
            let n = d.propose(s, last, &mut one)?;
            if n != 1 {
                return Err(format!(
                    "mtp_paths_agree: the draft proposed {n} ids at position {pos}, not its one"
                )
                .into());
            }
            found.push((pos, one[0]));
            let next = want[tokens.len()];
            if stepped {
                d.unproposed();
                let id = s.step(last, Want::Argmax)?.argmax();
                d.held(s, last, id)?;
                tokens.push(id);
            } else {
                let second = if window.is_multiple_of(2) {
                    next
                } else {
                    rejected(next)
                };
                window += 1;
                let rows = [last, second];
                let argmax = s.verify(rows)?;
                let kept = if second == argmax[0] { ROWS } else { 1 };
                d.accept(s, &rows, &argmax, kept)?;
                s.commit(kept)?;
                tokens.extend_from_slice(&argmax[..kept]);
            }
        }
        let plain = first == want[0] && tokens.get(..want.len()) == Some(want);
        Ok((found, plain))
    }

    /// `mtp_paths_agree` (module doc).
    fn mtp_paths_agree(s: &mut Seat, n_vocab: usize) -> Result<bool, GateError> {
        const CLAUSE: &str = "mtp_paths_agree";
        let mut ok = true;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for (i, prompt) in mtp_prompts(n_vocab).iter().enumerate() {
                let (want, _) = greedy(s, prompt, MTP_GEN, false)?;
                let (by_step, plain_step) = proposals(s, prompt, &want, true)?;
                let (by_pass, plain_pass) = proposals(s, prompt, &want, false)?;
                let both: Vec<(u32, u32, u32)> = by_pass
                    .iter()
                    .filter_map(|&(pos, id)| {
                        by_step
                            .iter()
                            .find(|&&(p, _)| p == pos)
                            .map(|&(_, step_id)| (pos, step_id, id))
                    })
                    .collect();
                let differ: Vec<_> = both.iter().filter(|&&(_, a, b)| a != b).collect();
                let hits = by_step
                    .iter()
                    .filter(|&&(pos, id)| want.get(pos as usize - prompt.len() + 1) == Some(&id))
                    .count();
                let pass = plain_step && plain_pass && !both.is_empty() && differ.is_empty();
                println!(
                    "{CLAUSE} {mode:?} prompt {i} ({} ids): tokens equal the plain run's by steps \
                     {plain_step}, by passes {plain_pass}; {} positions proposed both ways, ids \
                     differing {differ:?}; the stepped run's {} proposals match the plain next id \
                     {hits} times {}",
                    prompt.len(),
                    both.len(),
                    by_step.len(),
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// The median of `v`, which holds at least one value.
    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// `--time` (module doc): [`TIMED`] plain steps from a prompt, then the same ids replayed as
    /// [`TIMED`] / 2 accepted pairs through the verify pass, each call's wall read back to the host.
    fn time_rows(s: &mut Seat, n_vocab: usize) -> Result<(), GateError> {
        use std::time::Instant;
        let prompt = walk(TIME_PROMPT, n_vocab);
        s.model_mut().set_mode(StepMode::Graph);
        let mut ids = vec![prompted(s, &prompt)?];
        let mut step_ms = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let started = Instant::now();
            let next = s.step(ids[ids.len() - 1], Want::Argmax)?.argmax();
            step_ms.push(started.elapsed().as_secs_f64() * 1e3);
            ids.push(next);
        }
        prompted(s, &prompt)?;
        let mut pass_ms = Vec::with_capacity(TIMED / 2);
        let mut kept = 0usize;
        for pair in ids.as_chunks::<ROWS>().0.iter().take(TIMED / ROWS) {
            let started = Instant::now();
            let got = s.verify([pair[0], pair[1]])?;
            s.commit(ROWS)?;
            pass_ms.push(started.elapsed().as_secs_f64() * 1e3);
            kept += usize::from(got[0] == pair[1]);
        }
        let (step, pass) = (median(&mut step_ms[WARM..]), median(&mut pass_ms[WARM..]));
        println!(
            "time prompt={TIME_PROMPT} steps={TIMED} passes={} accepted_pairs={kept}: step median \
             {step:.3} ms, pass of {ROWS} rows median {pass:.3} ms, c_v = pass / step = {:.3}",
            TIMED / ROWS,
            pass / step
        );
        Ok(())
    }

    /// Each clause of `list` the tier runs, on `s`; whether every one passed.
    fn clauses(s: &mut Seat, list: &[(&str, Clause)], n_vocab: usize) -> Result<bool, GateError> {
        let mut ok = true;
        for (name, clause) in list {
            if !tier::run_clause(name, Tag::SelfConsistency)? {
                continue;
            }
            eprintln!("clause {name}");
            let pass = clause(s, n_vocab)?;
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if !pass {
                println!("clause {name} FAIL");
            }
            ok &= pass;
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let Some(other) = args.iter().find(|a| a.as_str() != "--time") {
            return Err(
                format!("{NAME}: unknown argument {other:?}: the gate takes --time").into(),
            );
        }
        if Tier::from_env()? == Tier::Fixture {
            return Err(format!(
                "{NAME}: the exaone-moe family's file is the fixture GGUF itself, so the gate has \
                 no fixture tier: just {RECIPE}"
            )
            .into());
        }
        let path = crate::lcpp_e2e::model_path(Tier::Real, RECIPE, exaone_moe::ARCH)?;
        let (m, dims) = crate::seat::open_model(&path, CTX, crate::lcpp_e2e::Arm::A)?;
        println!(
            "load arm=a layers={} embd={} vocab={} ctx={CTX}",
            dims.layers, dims.embd, dims.vocab
        );
        let mut s = Seat::from_model(m, u32::try_from(CTX)?);
        println!(
            "{NAME}: a verify pass runs {} rows (Rows::MAX_ROWS); a pass of more does not compile",
            <ExaoneBody as Rows>::MAX_ROWS
        );
        if !args.is_empty() {
            time_rows(&mut s, dims.vocab)?;
            return Ok(());
        }
        let mut ok = clauses(&mut s, CLAUSES, dims.vocab)?;
        drop(s);
        let (m, dims) = crate::seat::open_model_drafted(&path, CTX, crate::lcpp_e2e::Arm::A)?;
        println!(
            "load arm=a draft=mtp layers={} embd={} vocab={} ctx={CTX}",
            dims.layers, dims.embd, dims.vocab
        );
        let mut s = Seat::from_model(m, u32::try_from(CTX)?);
        ok &= clauses(&mut s, DRAFTED, dims.vocab)?;
        drop(s);
        for (draft, list) in [(false, HOSTED), (true, HOSTED_DRAFTED)] {
            let (m, dims, experts) =
                crate::seat::open_model_hosted(&path, CTX, crate::lcpp_e2e::Arm::A, draft)?;
            println!(
                "load arm=a{} host layers={} embd={} vocab={} ctx={CTX}: routed experts {} on the \
                 card, {} on the host",
                if draft { " draft=mtp" } else { "" },
                dims.layers,
                dims.embd,
                dims.vocab,
                experts.card,
                experts.host
            );
            if experts.card == 0 || experts.host == 0 {
                return Err(format!(
                    "{NAME}: the host-tier load holds {} routed experts on the card and {} on the \
                     host: the clauses would not run the host tier",
                    experts.card, experts.host
                )
                .into());
            }
            let mut s = Seat::from_model(m, u32::try_from(CTX)?);
            ok &= clauses(&mut s, list, dims.vocab)?;
        }
        println!("{NAME}: {}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} each row of a verify of two rows gives the plain steps' id and logits \
             bit for bit at prompts of 1 and 17 ids, accepted and rejected drafts, in the captured \
             graph and eager; a commit of one or both rows and the steps after it leave the plain \
             run's ids and last logits; a lookup draft over the pass emits the plain run's tokens; the \
             MTP draft over the pass emits the plain run's tokens and proposes the same id at a \
             position whether the target stepped or verified to reach it"
        );
        Ok(())
    }
}
