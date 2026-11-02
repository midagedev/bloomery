//! The Mistral Large 4 verify gate: the body's pass of two rows
//! (`impl Rows for Body`, `bloomery_gpu_mistrallarge4::body::rows`: row 0 the
//! token at the position, row 1 a drafted one, one layer apart on the step's
//! launches, [`Chain::Pair`](bloomery_gpu::hybrid::Chain::Pair) on the host
//! tier) driven through `Session<Body>`'s `Verify`, on the placed load of the
//! gate plan (`Body::open_placed`: the gate card's bytes, the routed experts
//! the card does not hold on the host tier; in the fixture tier a budget lever
//! keeps half of the card's experts, so both legs run, as the e2e gate's).
//!
//! Every clause is a self-consistency clause, so both tiers run them all, each
//! in the captured graph and eager:
//!
//! - `rows_match_steps`: at P ∈ {1, 17, 513} prompt ids (the shared walk; 513
//!   crosses the prompt call's 512-id unit), a verify of `[t0, d]` — `t0` the
//!   prompt's argmax, `d` the plain run's next id (an accepted draft) or
//!   another id (a rejected one) — gives each row's id and logits equal to two
//!   plain steps of `t0` then `d` from the same prompt, bit for bit. A pass of
//!   four rows does not compile: `Rows::MAX_ROWS` is 2 (`rows_fit`'s const
//!   assertion in `GpuModel::step_rows`).
//! - `rollback_restores`: a 17-id prompt, a verify whose row 1 is rejected
//!   (commit 1) and one whose rows are both kept (commit 2), then plain steps
//!   to [`AFTER_B`] positions past the prompt: the kept and stepped ids, the
//!   last logits and every store's rows below the position (`Body::store_rows`:
//!   each owner's latent rows, each full layer's index keys) equal the plain
//!   run's, byte for byte.
//! - `pair_rows_served`: a verify serves every host layer once a row: its
//!   services and host calls are twice a plain step's (which serves some),
//!   its row 1 lists slots and a step lists none as row 1. The union of the two
//!   rows' host lists (`host_slots − overlap_slots`) is printed, a reading.
//! - `features_match_steps`: with the feature tap attached for layers 1, half the layer count
//!   and the layer count (the last layer's output), at P ∈ {1, 17} prompt ids a verify of
//!   `[t0, d]` gives each row's tapped rows (`Tapped::taps`: the rows entering those layers) equal
//!   to those two plain steps of `t0` then `d` leave, bit for bit, accepted and rejected drafts,
//!   in the captured graph and eager; the rows are not all zero and the two rows differ.
//! - `prompt_taps_are_plain`: a tapped prompt call (`TappedPrompt::prompt_tapped`) of 17 and of 513
//!   ids (which crosses the prompt call's 512-id unit) with a window of [`WINDOW`] positions
//!   leaves the plain call's argmax, last logits and every store's rows, byte for byte, and hands
//!   its sink exactly the window's positions once each, in order, one tap row a position, finite
//!   and not all zero.
//! - `lookup_is_plain`: a lookup draft (`runtime::Lookup`) driven over
//!   `Session<Body>` (`Session::with_draft`, a verify of two rows) for [`GEN`]
//!   tokens on three prompts of a repeated pattern emits the plain run's ids on
//!   every prompt, and some prompt proposes. On the real file some prompt also
//!   keeps a draft, so row 1 was kept; the fixture's weights do not continue
//!   the pattern, and there `rows_match_steps` is the clause that keeps one.

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_verify", gate::run())
}

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the fixture arm reads its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "mistrallarge4")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "mistrallarge4")]
mod gate {
    use crate::e2e::{same_bits, walk};
    use crate::gate_card;
    use crate::greedy::greedy;
    use app::{RowsLog, Session, SessionError};
    use bloomery_gpu::model::{Rows, StepMode};
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use bloomery_gpu_mistrallarge4::body::Body;
    use bloomery_gpu_mistrallarge4::features::attach_features;
    use model::arch::Arch;
    use model::arch::mistrallarge4::hparams::Hparams;
    use model::arch::mistrallarge4::place::PlanInputs;
    use model::placement::{Machine, PlanLevers, workstation};
    use runtime::{
        Advance, Lookup, Out, RowLogits, Speculative, Tapped, TappedPrompt, Target, Verify, Want,
    };

    type Seat = Session<Body>;

    const NAME: &str = "gate_mistrallarge4_verify";
    const RECIPE: &str = "gate-gpu-mistrallarge4-verify";

    /// The rows of every verify here: the body's pass.
    const ROWS: usize = 2;

    /// `rows_match_steps`' prompt lengths; the last crosses the prompt call's
    /// 512-id unit.
    const LAST_P: usize = 513;
    const PROMPTS: [usize; 3] = [1, 17, LAST_P];

    /// `features_match_steps`' prompt lengths.
    const FEATURE_PROMPTS: [usize; 2] = [1, 17];

    /// `prompt_taps_are_plain`' prompt lengths (the last crosses the prompt call's 512-id unit)
    /// and the trailing positions whose tap rows the call hands over.
    const TAPPED_PROMPTS: [usize; 2] = [17, LAST_P];
    const WINDOW: usize = 8;

    /// `rollback_restores`' and `pair_rows_served`' prompt, and the positions
    /// past it the drafted and the plain runs both reach.
    const PROMPT_B: usize = 17;
    const AFTER_B: usize = 8;

    /// `lookup_is_plain`'s generated tokens, and its prompts: a pattern of
    /// [`PATTERN`] ids [`REPEATS`] times.
    const GEN: usize = 64;
    const PATTERN: usize = 12;
    const REPEATS: usize = 4;
    const LOOKUPS: usize = 3;

    /// A drafted id no clause's plain run follows with: the walk's first id,
    /// moved by one where the plain run's next id is it.
    const REJECT: u32 = 1_000;

    /// Rows of the caches beyond the longest run.
    const CTX_SPARE: usize = 8;

    const fn max(a: usize, b: usize) -> usize {
        if a > b { a } else { b }
    }

    /// The cache rows of the load: the longest run of the clauses (a lookup
    /// pass may run one row past [`GEN`]), with room.
    const CTX: usize = max(
        max(LAST_P + ROWS, PROMPT_B + AFTER_B + ROWS),
        PATTERN * REPEATS + GEN + ROWS,
    ) + CTX_SPARE;

    /// The host tier's configuration the load opens with: no lever is read.
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    struct Case {
        n_vocab: usize,
        /// The model's width, and the layers the feature tap reads
        /// ([`tap_layers`]).
        n_embd: usize,
        taps: Vec<usize>,
        /// The real file: a model that continues a repeated pattern.
        real: bool,
    }

    /// The layers the gate's feature tap reads: the row entering layer 1, the one entering the
    /// middle layer and the last layer's output (the layer count), strictly rising whatever the
    /// layer count.
    fn tap_layers(n_layer: usize) -> Vec<usize> {
        let mut layers = vec![1, n_layer / 2, n_layer];
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    type Clause = fn(&mut Seat, &Case) -> Result<bool, GateError>;

    const CLAUSES: &[(&str, Clause)] = &[
        ("rows_match_steps", rows_match_steps),
        ("rollback_restores", rollback_restores),
        ("pair_rows_served", pair_rows_served),
        ("features_match_steps", features_match_steps),
        ("prompt_taps_are_plain", prompt_taps_are_plain),
        ("lookup_is_plain", lookup_is_plain),
    ];

    const MODES: [StepMode; 2] = [StepMode::Graph, StepMode::Eager];

    fn rejected(next: u32) -> u32 {
        if next == REJECT { REJECT + 1 } else { REJECT }
    }

    /// A call's argmax and logits row; a call that read back its argmax only
    /// is refused by name.
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

    /// `n` greedy steps from `first` at the target's position: each step's
    /// argmax, and the last one's logits.
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

    /// Every store's rows below the target's position.
    fn stores(s: &mut Seat) -> Result<Vec<(String, Vec<u8>)>, GateError> {
        let pos = usize::try_from(s.pos())?;
        let (gpu, _w, body) = s.model_mut().body_parts(NAME)?;
        Ok(body.store_rows(gpu, pos)?)
    }

    /// The first store of `got` whose rows differ from `want`'s, or the count
    /// where the lists differ in length.
    fn first_moved(got: &[(String, Vec<u8>)], want: &[(String, Vec<u8>)]) -> Option<String> {
        if got.len() != want.len() {
            return Some(format!("{} stores against {}", got.len(), want.len()));
        }
        got.iter()
            .zip(want)
            .find(|(g, w)| g != w)
            .map(|(g, _)| g.0.clone())
    }

    /// `rows_match_steps` (module doc).
    fn rows_match_steps(s: &mut Seat, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "rows_match_steps";
        let mut ok = true;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for p in PROMPTS {
                let prompt = walk(p, case.n_vocab);
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
    fn rollback_restores(s: &mut Seat, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "rollback_restores";
        let prompt = walk(PROMPT_B, case.n_vocab);
        let mut ok = true;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            let t0 = prompted(s, &prompt)?;
            let (plain, plain_last) = steps_from(s, t0, AFTER_B)?;
            let plain_stores = stores(s)?;
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
                let (after, last) = steps_from(s, ids[accepted - 1], AFTER_B - accepted)?;
                ids.extend(after);
                let got_stores = stores(s)?;
                let same_ids = t0_pass == t0 && ids == plain;
                let bits = same_bits(&last, &plain_last);
                let moved = first_moved(&got_stores, &plain_stores);
                let pass = same_ids && bits && moved.is_none();
                println!(
                    "{CLAUSE} {mode:?} commit {accepted} of [{t0_pass}, {d}]: {} ids against the \
                     plain run's {same_ids}, the last logits bit for bit {bits}, every store's \
                     rows below {} {} {}",
                    ids.len(),
                    s.pos(),
                    moved.map_or_else(|| "equal".to_string(), |m| format!("differ at {m}")),
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// The host tier's step counters a clause reads.
    #[derive(Clone, Copy, Debug)]
    struct Served {
        served: u64,
        host_calls: u64,
        host_slots: u64,
        row1_slots: u64,
        overlap_slots: u64,
    }

    impl Served {
        fn read(s: &mut Seat) -> Result<Served, GateError> {
            let (_gpu, _w, body) = s.model_mut().body_parts(NAME)?;
            let st = body.hybrid().stats();
            Ok(Served {
                served: st.served,
                host_calls: st.host_calls,
                host_slots: st.host_slots,
                row1_slots: st.pair_row1_slots,
                overlap_slots: st.overlap_slots,
            })
        }

        /// The counts between `before` and `self`, refused by name where one
        /// went back.
        fn since(self, before: Served) -> Result<Served, GateError> {
            let d = |a: u64, b: u64| {
                a.checked_sub(b)
                    .ok_or_else(|| format!("a host counter went back: {b} then {a}"))
            };
            Ok(Served {
                served: d(self.served, before.served)?,
                host_calls: d(self.host_calls, before.host_calls)?,
                host_slots: d(self.host_slots, before.host_slots)?,
                row1_slots: d(self.row1_slots, before.row1_slots)?,
                overlap_slots: d(self.overlap_slots, before.overlap_slots)?,
            })
        }
    }

    /// `pair_rows_served` (module doc).
    fn pair_rows_served(s: &mut Seat, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "pair_rows_served";
        let prompt = walk(PROMPT_B, case.n_vocab);
        let mut ok = true;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            let t0 = prompted(s, &prompt)?;
            let a = Served::read(s)?;
            let g0 = s.step(t0, Want::Argmax)?.argmax();
            let b = Served::read(s)?;
            s.verify([g0, REJECT])?;
            let c = Served::read(s)?;
            s.commit(1)?;
            let step = b.since(a)?;
            let verify = c.since(b)?;
            let serves = step.served > 0 && verify.served == 2 * step.served;
            let calls = verify.host_calls == 2 * step.host_calls;
            let row1 = step.row1_slots == 0 && step.overlap_slots == 0 && verify.row1_slots > 0;
            let pass = serves && calls && row1;
            let row0 = verify.host_slots.saturating_sub(verify.row1_slots);
            println!(
                "{CLAUSE} {mode:?}: a step's services {} and host calls {}, the verify's {} and {} \
                 (twice the step's: {serves}, {calls}); row 1's slots {} (a step's {}: {row1}); \
                 reading: row 0 {row0} host slots, row 1 {}, {} in common, their union {} {}",
                step.served,
                step.host_calls,
                verify.served,
                verify.host_calls,
                verify.row1_slots,
                step.row1_slots,
                verify.row1_slots,
                verify.overlap_slots,
                verify.host_slots.saturating_sub(verify.overlap_slots),
                verdict(pass)
            );
            ok &= pass;
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// A step's tapped rows of the position it just ran.
    fn step_taps(s: &mut Seat, id: u32) -> Result<(u32, Vec<f32>), GateError> {
        let next = s.step(id, Want::Argmax)?.argmax();
        Ok((next, s.taps(1)?.to_vec()))
    }

    /// `features_match_steps` (module doc).
    fn features_match_steps(s: &mut Seat, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "features_match_steps";
        let width = case.taps.len() * case.n_embd;
        let held = s.tap_width();
        let held_layers = s
            .model()
            .body(NAME)?
            .feature_layers()
            .map(<[usize]>::to_vec);
        let wide = held == width && held_layers.as_deref() == Some(&case.taps[..]);
        println!(
            "{CLAUSE}: the tap reads layers {:?} ({} values a row), the body holds {held_layers:?}              ({held}) {}",
            case.taps,
            width,
            verdict(wide)
        );
        let mut ok = wide;
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for p in FEATURE_PROMPTS {
                let prompt = walk(p, case.n_vocab);
                for accept in [true, false] {
                    let t0 = prompted(s, &prompt)?;
                    let (next0, f0) = step_taps(s, t0)?;
                    let d = if accept { next0 } else { rejected(next0) };
                    let (_, f1) = step_taps(s, d)?;
                    let t0_pass = prompted(s, &prompt)?;
                    s.verify([t0_pass, d])?;
                    let rows = s.taps(ROWS)?.to_vec();
                    s.commit(ROWS)?;
                    let (r0, r1) = rows.split_at(rows.len().min(width));
                    let bits0 = same_bits(r0, &f0);
                    let bits1 = same_bits(r1, &f1);
                    let lit = f0.iter().chain(&f1).all(|v| v.is_finite())
                        && f0.iter().any(|v| *v != 0.0)
                        && f1.iter().any(|v| *v != 0.0);
                    let apart = !same_bits(&f0, &f1);
                    let pass = t0_pass == t0
                        && f0.len() == width
                        && rows.len() == ROWS * width
                        && bits0
                        && bits1
                        && lit
                        && apart;
                    println!(
                        "{CLAUSE} {mode:?} P={p} {} draft: row 0 features bit for bit {bits0},                          row 1 {bits1}; the steps' rows finite and not all zero {lit}, the two                          positions' rows differ {apart} {}",
                        if accept { "accepted" } else { "rejected" },
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// `prompt_taps_are_plain` (module doc).
    fn prompt_taps_are_plain(s: &mut Seat, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "prompt_taps_are_plain";
        let width = case.taps.len() * case.n_embd;
        let mut ok = true;
        for p in TAPPED_PROMPTS {
            let prompt = walk(p, case.n_vocab);
            s.reset()?;
            let (plain_next, plain_row) = logits_of(s.prompt(&prompt, Want::Logits)?)?;
            let plain_stores = stores(s)?;
            s.reset()?;
            let mut handed: Vec<(u32, usize)> = Vec::new();
            let mut tapped = Vec::new();
            let next = s.prompt_tapped(&prompt, WINDOW, &mut |first, rows| {
                handed.push((first, rows.len() / width));
                tapped.extend_from_slice(rows);
                Ok(())
            })?;
            let mut row = vec![0.0f32; case.n_vocab];
            s.model().logits_into(&mut row)?;
            let moved = first_moved(&stores(s)?, &plain_stores);
            // The window's positions once each, in order.
            let first = u32::try_from(p - WINDOW)?;
            let mut at = first;
            let in_order = handed.iter().all(|&(f, n)| {
                let here = f == at;
                at += u32::try_from(n).unwrap_or(u32::MAX);
                here
            });
            let covered = in_order && at == first + u32::try_from(WINDOW)?;
            let lit = tapped.len() == WINDOW * width
                && tapped.iter().all(|v| v.is_finite())
                && tapped.iter().any(|v| *v != 0.0);
            let bits = same_bits(&row, &plain_row);
            let pass = next == plain_next && bits && moved.is_none() && covered && lit;
            println!(
                "{CLAUSE} P={p}: the argmax the plain call's {} and its last logits bit for bit \
                 {bits}, every store's rows {}; the sink was handed {handed:?} (the {WINDOW} \
                 positions from {first} once each, in order: {covered}), {} values, finite and \
                 not all zero {lit} {}",
                next == plain_next,
                moved.map_or_else(|| "equal".to_string(), |m| format!("differ at {m}")),
                tapped.len(),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    struct Log;

    impl RowsLog for Log {
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError> {
            println!("lookup verify pass of {rows} rows captured: {nodes} nodes");
            Ok(())
        }
    }

    /// One lookup-drafted generation of [`GEN`] tokens from `prompt`: the
    /// tokens, the proposals and the drafts kept.
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
    fn lookup_is_plain(s: &mut Seat, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "lookup_is_plain";
        let base = walk(PATTERN * LOOKUPS, case.n_vocab);
        let prompts: Vec<Vec<u32>> = base.chunks(PATTERN).map(|p| p.repeat(REPEATS)).collect();
        let mut ok = true;
        let (mut proposed_some, mut kept_some) = (false, false);
        for mode in MODES {
            s.model_mut().set_mode(mode);
            for (i, prompt) in prompts.iter().enumerate() {
                let (want, _) = greedy(s, prompt, GEN, false)?;
                let (tokens, proposals, accepted) = lookup_run(s, prompt)?;
                let same = tokens.get(..GEN) == Some(&want[..]);
                kept_some |= accepted > 0;
                proposed_some |= proposals > 0;
                let pass = same;
                println!(
                    "{CLAUSE} {mode:?} prompt {i} ({} ids): {GEN} tokens equal the plain run's \
                     {same}; proposals {proposals} (above 0: {}), drafts kept {accepted} {}",
                    prompt.len(),
                    proposals > 0,
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        println!(
            "{CLAUSE}: a draft proposed on some prompt {}",
            verdict(proposed_some)
        );
        if case.real {
            println!(
                "{CLAUSE}: a draft kept on some prompt (row 1's id kept as the next token) {}",
                verdict(kept_some)
            );
        } else {
            println!(
                "{CLAUSE}: drafts kept on some prompt {kept_some} (not judged: the fixture's \
                 weights do not continue the pattern; rows_match_steps keeps a draft here)"
            );
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok && proposed_some && (kept_some || !case.real))
    }

    /// The plan's levers: in the fixture tier the card keeps half of the
    /// experts the gate card would hold (so the host leg serves the rest and
    /// both legs run), in the real tier the card's own bytes, which already
    /// leave most of the experts to the host (the e2e gate's).
    fn levers(
        inputs: &PlanInputs,
        machine: &Machine,
        (ctx, tier_now): (u64, tier::Tier),
    ) -> Result<PlanLevers, GateError> {
        if tier_now == tier::Tier::Real {
            return Ok(PlanLevers::default());
        }
        let all = inputs.plan(machine, ctx, &PlanLevers::default())?;
        let c = &all.cards[0];
        let card = &machine.cards[0];
        let kept = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + inputs.prompt_reserve
            + c.expert_bytes / 2;
        Ok(PlanLevers {
            card_budget_bytes: Some(kept).into(),
        })
    }

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::MistralLarge4, RECIPE)?;
        let hp = Hparams::read(&file).map_err(|e| format!("mistral-large4 header: {e}"))?;
        let tier_now = tier::Tier::from_env()?;
        let inputs = PlanInputs::describe(&file)?;
        let spec = gate_card::card()?;
        let machine = workstation::placed_on(spec, hp.n_layer, workstation::SCRATCH);
        let ctx64 = u64::try_from(CTX)?;
        let plan = inputs.plan(
            &machine,
            ctx64,
            &levers(&inputs, &machine, (ctx64, tier_now))?,
        )?;
        println!(
            "plan: ctx={CTX} card experts {} ({} B) host experts {} ({} B) (plan values)",
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes
        );
        let mut m = Body::open_placed(file, &plan, &inputs, 0, HOST_CFG)?;
        // The feature tap is attached before the first capture, so every clause below runs the
        // chain with its copies: the other clauses' bits hold with the tap on.
        let taps = tap_layers(hp.n_layer);
        attach_features(&mut m, &taps)?;
        let mut s = Seat::from_model(m, u32::try_from(CTX)?);
        println!(
            "{NAME}: a verify pass runs {} rows (Rows::MAX_ROWS); a pass of more does not compile",
            <Body as Rows>::MAX_ROWS
        );
        let case = Case {
            n_vocab: hp.n_vocab,
            n_embd: hp.n_embd,
            taps,
            real: tier_now == tier::Tier::Real,
        };
        let mut ok = true;
        for (name, clause) in CLAUSES {
            if !tier::run_clause(name, Tag::SelfConsistency)? {
                continue;
            }
            eprintln!("clause {name}");
            let pass = clause(&mut s, &case)?;
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if !pass {
                println!("clause {name} FAIL");
            }
            ok &= pass;
        }
        println!("{NAME}: {}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} each row of a verify of two rows gives the plain steps' id and logits \
             bit for bit at prompts of 1, 17 and 513 ids, accepted and rejected drafts; a commit of \
             one or both rows and the steps after it leave the plain run's ids, logits and store \
             rows; a verify serves every host layer once a row and taps the plain steps' feature \
             rows; a tapped prompt call is the plain call with the window's rows handed over; a \
             lookup draft emits the plain run's tokens"
        );
        Ok(())
    }
}
