//! The Kimi K3 rows-pass gate: a pass of up to eight consecutive positions through the whole chain
//! (the embedding, the delta and gated latent layers, the dense and routed `situ` blocks, the AttnRes
//! mixes over the banked block inputs of every row, the output head) as one walk, against the same
//! ids run one decode step at a time, over the fixture file (4 layers: a dense delta layer, a routed
//! delta layer, a routed latent layer, a routed delta layer; hidden 512, an attn_res block of 2).
//! The model opens through `GpuModel<Body>::open` and the pass is `Body::rows` over a `Pass` made
//! beside it, as `gate_kimi_k3_e2e` opens the step.
//!
//! The fixture tier only: the real file (594 GB) does not fit one card and its placement is not
//! wired, so the real tier ends by name before anything loads.
//!
//! For each prompt set of [`SETS`] (an English paragraph of 35 ids, a Korean one of 46 and a Rust
//! function head of 48, refset family `fx-lcpp-kimi-k3`), against the same ids run as steps (the
//! reference: every position's whole logits row and argmax, then [`CONT`] greedy steps on):
//!
//! - `rows_are_steps`: the prompt as passes of [`PASS_ROWS`] rows and again as passes of 5, every
//!   row's whole logits row and argmax bit for bit the step's at that position. The chunks end on
//!   different ids in the two runs (3, 6 and 8 rows last at 8; 5, 1 and 3 rows last at 5), so a
//!   pass of 1 (the step's own launches in the pass's buffers) and the odd row counts run too.
//! - `state_advances_as_steps`: after those passes, [`CONT`] decode steps from the pass's last
//!   argmax give the reference's logits and ids bit for bit: the delta layers' stores, the latent
//!   caches and the position book stand where as many steps leave them.
//!   One more run of the clause stops after a single pass of [`PASS_ROWS`] rows and steps the
//!   prompt's next [`CONT`] ids (`one_pass`), so a pass that leaves the position book or a store
//!   unadvanced is named by that clause alone, with no second pass to refuse.
//! - `prompt_rows_are_steps`: `GpuModel::prompt_rows` over the whole prompt (passes of eight, the
//!   head on the last row alone) gives the last position's logits row and argmax bit for bit, and
//!   its continuation of [`CONT`] steps the reference's.
//! - `last_row_is_vendors` (fixture oracle): that last logits row within [`LOGITS_REL_BAND`]
//!   relative L2 of the vendor's `result_output` at the prompt's last position.
//! - `pass_refused`: a pass of more than [`PASS_ROWS`] rows, of none, at a position the stores do
//!   not stand at, past the caches, a [`Tail::Head`] of another row count and a [`Tail::Last`] of
//!   more than one row are each refused by name, and a refused call leaves the model where it
//!   was: the next step is the reference's first.
//!
//! The order is the gate. A pass launches the step's own kernels over `m` rows, and no launch sums
//! across rows or changes the order within a row: the embedding, norms, adds and the AttnRes mix
//! are a block a row (the mix reads bank `b`'s row `t` at `b · stride + t · n`, so a row's mix
//! sees only that row's banked inputs); the delta layer's recurrence runs the rows in position
//! order on the one-lane store in place; the latent layer appends its rows to the cache and
//! attends each over the keys up to its own position; every projection is each row's one-row
//! launch (a K-quant m-column gemv, which is the one-column launch per column; the f32 and q8_0
//! m-column launches; the one-row launch on each row for a Q6_K site of an odd super-block count,
//! which has no m-column launch); the router, the card leg's slots and the shared block run a row
//! each. So each row's reductions are the step's, in the step's order, and the logits are equal
//! to the bit. A wide unit (a GEMM over more rows than a pass) sums a row's products in the GEMM's
//! tile order and is not this gate's.
//!
//! Before the card loads, a set absent from the box, dumped from another file or build, or
//! unfinished ends the gate by name. A pass that raises the fault word ends the gate by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_pp", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::rel;
    use bloomery_gpu::arch::kimi_k3::{Body, Pass, Tail};
    use bloomery_gpu::head::Head;
    use bloomery_gpu::linear::PASS_ROWS;
    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, open_split, ref_tensor_logical_in,
        verdict,
    };
    use model::arch::Arch;
    use refset::arch::kimi_k3::fixture::{self, A, B, C};
    use refset::arch::prompt_tables;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_kimi_k3_pp";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-kimik3-pp";

    /// The sets the pass is held to, in the order they run: prompts of 35, 46 and 48 ids.
    const SETS: &[&str] = &[A, B, C];

    /// Rows of each latent layer's cache: the longest prompt and its continuation fit.
    const CTX: usize = 64;

    /// Decode steps run after a prompt, from its last argmax.
    const CONT: usize = 4;

    /// The chunk sizes the prompt runs as: the whole pass and an odd one, so the two runs'
    /// chunks end on different ids.
    const CHUNKS: [usize; 2] = [PASS_ROWS, 5];

    /// The last-position logits row's relative distance from the vendor's.
    /// PIN(2026-11-06): the e2e gate's `LOGITS_REL_BAND`, the same band: the row is closer to the
    /// vendor's than the zero row is.
    const LOGITS_REL_BAND: f64 = 1.0;

    /// The vendor clauses a run declares: one a set.
    const FX_ORACLE: usize = SETS.len();

    /// One set's prompt and the vendor's whole logits row at its last position.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        logits: Vec<f32>,
    }

    fn open_set(clause: &str, name: &'static str) -> Result<Set, GateError> {
        let path = tier::fixture_set(clause, &fixture::LCPP, name)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
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

    /// What a run leaves: every position's whole logits row and the argmax after it.
    struct Trace {
        logits: Vec<Vec<f32>>,
        ids: Vec<u32>,
    }

    /// The reference: set `s`'s prompt one decode step at a time from a reset, then [`CONT`]
    /// greedy steps.
    fn steps(m: &mut GpuModel<Body>, s: &Set) -> Result<Trace, GateError> {
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
            let id = m.step(&[input])?;
            t.ids.push(id);
            t.logits.push(m.logits()?);
        }
        Ok(t)
    }

    /// The heads of the passes' row counts, made once each.
    struct Heads(Vec<Option<Head>>);

    impl Heads {
        fn new() -> Heads {
            Heads((0..=PASS_ROWS).map(|_| None).collect())
        }

        /// The head of `rows` rows, made on first use.
        fn of(&mut self, m: &GpuModel<Body>, rows: usize) -> Result<&mut Head, GateError> {
            let eps = m.body(NAME)?.head_eps();
            let slot = self
                .0
                .get_mut(rows)
                .ok_or_else(|| format!("a head of {rows} rows"))?;
            if slot.is_none() {
                *slot = Some(Head::with_m(m.gpu(), m.weights(), eps, rows)?);
            }
            slot.as_mut()
                .ok_or_else(|| format!("a head of {rows} rows").into())
        }
    }

    /// One pass of `ids` from the model's position into `head` (of as many rows): each row's whole
    /// logits row and argmax. The head's token read raises the fault word.
    fn run_pass(
        m: &mut GpuModel<Body>,
        pass: &mut Pass,
        head: &mut Head,
        ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<u32>), GpuError> {
        let rows = ids.len();
        m.run_rows(rows, NAME, |gpu, w, body, _model_head, pos| {
            body.rows(gpu, w, pass, (ids, pos), Tail::Head(&mut *head))?;
            Ok(false)
        })?;
        let tokens = head.tokens(m.gpu())?;
        let all = head.logits_to_host(m.gpu())?;
        let vocab = head.n_vocab();
        let cols = (0..rows)
            .map(|c| (0..vocab).map(|v| all[v * rows + c]).collect())
            .collect();
        Ok((cols, tokens))
    }

    /// The continuation: [`CONT`] steps from `last`, each against the reference's position
    /// `from + j`. The rows that differ, or the refusal the model gave.
    fn continuation(
        m: &mut GpuModel<Body>,
        reference: &Trace,
        (from, mut last): (usize, u32),
    ) -> Result<usize, String> {
        let mut differing = 0;
        for j in 0..CONT {
            let id = m.step(&[last]).map_err(|e| e.to_string())?;
            let row = m.logits().map_err(|e| e.to_string())?;
            if id != reference.ids[from + j] || !bits_equal(&row, &reference.logits[from + j]) {
                differing += 1;
            }
            last = id;
        }
        Ok(differing)
    }

    /// A refusal of a call, which a clause reports as its failure: a shape or a state error. Any
    /// other error (the fault word, a driver error) ends the gate.
    fn refusal(e: GpuError) -> Result<GpuError, GateError> {
        match e {
            GpuError::Shape { .. } | GpuError::State { .. } => Ok(e),
            other => Err(other.into()),
        }
    }

    /// What the prompt run as passes left: the rows that differ from the reference's, the chunk
    /// sizes and the last argmax; or the refusal the model gave.
    struct Walked {
        differing: usize,
        sizes: Vec<usize>,
        last: u32,
    }

    /// Set `s`'s prompt as passes of `p` rows from a reset against the reference.
    fn walk_prompt(
        (m, pass, heads): (&mut GpuModel<Body>, &mut Pass, &mut Heads),
        s: &Set,
        reference: &Trace,
        p: usize,
    ) -> Result<Result<Walked, GpuError>, GateError> {
        m.reset()?;
        let mut w = Walked {
            differing: 0,
            sizes: Vec::new(),
            last: 0,
        };
        for (c, chunk) in s.prompt.chunks(p).enumerate() {
            let head = heads.of(m, chunk.len())?;
            let (rows, ids) = match run_pass(m, pass, head, chunk) {
                Ok(r) => r,
                Err(e) => return Ok(Err(refusal(e)?)),
            };
            for (r, (row, id)) in rows.iter().zip(&ids).enumerate() {
                let at = c * p + r;
                if *id != reference.ids[at] || !bits_equal(row, &reference.logits[at]) {
                    w.differing += 1;
                }
            }
            w.sizes.push(chunk.len());
            w.last = ids[chunk.len() - 1];
        }
        Ok(Ok(w))
    }

    /// Clauses `rows_are_steps` and `state_advances_as_steps` of set `s` at chunks of `p`: the
    /// prompt as passes of `p` rows from a reset against the reference, then the continuation.
    fn pass_clauses(
        at: (&mut GpuModel<Body>, &mut Pass, &mut Heads),
        s: &Set,
        reference: &Trace,
        p: usize,
    ) -> Result<bool, GateError> {
        let m = &mut *at.0;
        let walked = walk_prompt((&mut *m, at.1, at.2), s, reference, p)?;
        let mut ok = true;
        let name = format!("rows_are_steps set={} p={p}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let (pass_ok, text) = match &walked {
                Ok(w) => (
                    w.differing == 0,
                    format!(
                        "rows={} chunks={:?} differing_rows={}",
                        s.prompt.len(),
                        w.sizes,
                        w.differing
                    ),
                ),
                Err(e) => (false, format!("pass refused: {e}")),
            };
            println!("{name} {text} {}", verdict(pass_ok));
            ok &= pass_ok;
        }
        let name = format!("state_advances_as_steps set={} p={p}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let (pass_ok, text) = match &walked {
                Ok(w) => match continuation(m, reference, (s.prompt.len(), w.last)) {
                    Ok(d) => (d == 0, format!("continuation={CONT} differing_steps={d}")),
                    Err(e) => (false, format!("continuation refused: {e}")),
                },
                Err(_) => (false, "no continuation: the passes were refused".to_owned()),
            };
            println!("{name} {text} {}", verdict(pass_ok));
            ok &= pass_ok;
        }
        Ok(ok)
    }

    /// Clause `state_advances_as_steps` after a single pass: the prompt's first [`PASS_ROWS`] ids
    /// as one pass from a reset, then its next [`CONT`] ids as steps, each step's whole logits row
    /// and argmax the reference's at that position.
    fn one_pass_clause(
        (m, pass, heads): (&mut GpuModel<Body>, &mut Pass, &mut Heads),
        s: &Set,
        reference: &Trace,
    ) -> Result<bool, GateError> {
        let name = format!(
            "state_advances_as_steps set={} one_pass={PASS_ROWS}",
            s.name
        );
        if !tier::run_clause(&name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        m.reset()?;
        let head = heads.of(m, PASS_ROWS)?;
        let (pass_ok, text) = match run_pass(m, pass, head, &s.prompt[..PASS_ROWS]) {
            Err(e) => (false, format!("pass refused: {}", refusal(e)?)),
            Ok(_) => {
                let mut differing = 0;
                let mut refused = None;
                for at in PASS_ROWS..PASS_ROWS + CONT {
                    match m.step(&[s.prompt[at]]) {
                        Ok(id) => {
                            let row = m.logits()?;
                            if id != reference.ids[at] || !bits_equal(&row, &reference.logits[at]) {
                                differing += 1;
                            }
                        }
                        Err(e) => {
                            refused = Some(refusal(e)?);
                            break;
                        }
                    }
                }
                match refused {
                    Some(e) => (false, format!("step refused: {e}")),
                    None => (
                        differing == 0,
                        format!("steps={CONT} differing_steps={differing}"),
                    ),
                }
            }
        };
        println!("{name} {text} {}", verdict(pass_ok));
        Ok(pass_ok)
    }

    /// Clauses `prompt_rows_are_steps` and `last_row_is_vendors` of set `s`: the model's prompt
    /// call from a reset against the reference's last position, its continuation, and its last
    /// logits row against the vendor's.
    fn prompt_clauses(
        m: &mut GpuModel<Body>,
        pass: &mut Pass,
        s: &Set,
        reference: &Trace,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        m.reset()?;
        let at = s.prompt.len() - 1;
        let called = match m.prompt_rows(pass, &s.prompt) {
            Ok(id) => Ok((id, m.logits()?)),
            Err(e) => Err(refusal(e)?),
        };
        let name = format!("prompt_rows_are_steps set={}", s.name);
        if tier::run_clause(&name, Tag::SelfConsistency)? {
            let (pass_ok, text) = match &called {
                Ok((id, row)) => {
                    let same = *id == reference.ids[at] && bits_equal(row, &reference.logits[at]);
                    match continuation(m, reference, (s.prompt.len(), *id)) {
                        Ok(d) => (
                            same && d == 0,
                            format!(
                                "ids={} last_row_same_bits={same} continuation_differing_steps={d}",
                                s.prompt.len()
                            ),
                        ),
                        Err(e) => (
                            false,
                            format!("last_row_same_bits={same} continuation refused: {e}"),
                        ),
                    }
                }
                Err(e) => (false, format!("prompt call refused: {e}")),
            };
            println!("{name} {text} {}", verdict(pass_ok));
            ok &= pass_ok;
        }
        let name = format!("last_row_is_vendors set={}", s.name);
        if tier::run_clause(&name, Tag::FixtureOracle)? {
            let (pass_ok, text) = match &called {
                Ok((_, row)) => {
                    let logits_rel = rel(row, &s.logits);
                    (
                        logits_rel <= LOGITS_REL_BAND,
                        format!("logits_rel={logits_rel:.4e} band={LOGITS_REL_BAND:.1e}"),
                    )
                }
                Err(e) => (false, format!("prompt call refused: {e}")),
            };
            println!("{name} {text} {}", verdict(pass_ok));
            ok &= pass_ok;
        }
        Ok(ok)
    }

    /// Whether `r` is a refusal naming `want`, printed with what it said.
    fn refused(what: &str, r: Result<(), GpuError>, want: &str) -> bool {
        match r {
            Err(e) if e.to_string().contains(want) => {
                println!("  refused {what}: {e}");
                true
            }
            Err(e) => {
                println!("  {what} refused without naming {want:?}: {e}");
                false
            }
            Ok(()) => {
                println!("  {what} was not refused");
                false
            }
        }
    }

    /// Clause `pass_refused`: the shapes a pass refuses by name, each with nothing enqueued, then
    /// the model where it stood: its next step is the reference's first.
    fn pass_refused(
        (m, pass, heads): (&mut GpuModel<Body>, &mut Pass, &mut Heads),
        s: &Set,
        reference: &Trace,
    ) -> Result<bool, GateError> {
        let name = "pass_refused";
        if !tier::run_clause(name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        m.reset()?;
        let too_many: Vec<u32> = s.prompt.iter().copied().take(PASS_ROWS + 1).collect();
        let some: Vec<u32> = s.prompt.iter().copied().take(3).collect();
        let mut ok = true;
        let head5 = heads.of(m, 5)?;
        let (gpu, w, body) = m.body_parts(NAME)?;
        ok &= refused(
            "a pass above the pass",
            body.rows(gpu, w, pass, (&too_many, 0), Tail::Open),
            &format!("a pass runs 1 to {PASS_ROWS}"),
        );
        ok &= refused(
            "a pass of no rows",
            body.rows(gpu, w, pass, (&[], 0), Tail::Open),
            "a pass of 0 rows",
        );
        ok &= refused(
            "a pass off the stores' position",
            body.rows(gpu, w, pass, (&some, 1), Tail::Open),
            "the stores stand at 0",
        );
        let end = u32::try_from(CTX - 1)?;
        ok &= refused(
            "a pass past the caches",
            body.rows(gpu, w, pass, (&some, end), Tail::Open),
            "in caches of",
        );
        ok &= refused(
            "a head of other rows than the pass's",
            body.rows(gpu, w, pass, (&some, 0), Tail::Head(&mut *head5)),
            "under a pass of 3",
        );
        ok &= refused(
            "a head of several rows on the last row",
            body.rows(gpu, w, pass, (&some, 0), Tail::Last(&mut *head5)),
            "on the last row alone",
        );
        let pos_kept = m.pos() == 0;
        let id = m.step(&s.prompt[..1])?;
        let row = m.logits()?;
        let again = id == reference.ids[0] && bits_equal(&row, &reference.logits[0]);
        let pass_ok = ok && pos_kept && again;
        println!(
            "{name} position_kept={pos_kept} next_step_is_reference={again} {}",
            verdict(pass_ok)
        );
        Ok(pass_ok)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the real file's placement is not wired (the Program wiring is round \
                 kimik3place's), so the gate runs the fixture tier: BLOOMERY_TIER=fixture just \
                 {RECIPE}"
            )
            .into());
        }
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set("the last row is the vendor's", n))
            .collect::<Result<_, _>>()?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let t = std::time::Instant::now();
        let opened = GpuModel::<Body>::open(Gpu::new()?, file, CTX);
        let mut m = match opened {
            Ok(m) => m,
            Err(e) => {
                println!("opens_the_fixture {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        let made = {
            let (gpu, _, body) = m.body_parts(NAME)?;
            Pass::new(gpu, body)
        };
        let mut pass = match made {
            Ok(p) => p,
            Err(e) => {
                println!("opens_the_fixture the pass: {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        println!(
            "opens_the_fixture resident_bytes={} pass_bytes={} ctx={CTX} layers={} in {:.1} s {}",
            m.resident_bytes(),
            pass.bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64(),
            verdict(true)
        );
        let mut heads = Heads::new();
        let mut ok = true;
        for s in &sets {
            let reference = steps(&mut m, s)?;
            for p in CHUNKS {
                ok &= pass_clauses((&mut m, &mut pass, &mut heads), s, &reference, p)?;
            }
            ok &= one_pass_clause((&mut m, &mut pass, &mut heads), s, &reference)?;
            ok &= prompt_clauses(&mut m, &mut pass, s, &reference)?;
            if s.name == SETS[0] {
                ok &= pass_refused((&mut m, &mut pass, &mut heads), s, &reference)?;
            }
        }
        tier::expect_fixture_oracle(NAME, FX_ORACLE)?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} on every set of {SETS:?} the prompt as passes of {CHUNKS:?} rows is the \
             steps' logits and argmax at every position bit for bit, the {CONT} steps after it \
             are the reference's, the model's prompt call is the last position bit for bit and \
             within {LOGITS_REL_BAND:.1e} of the vendor's row, and the shapes a pass does not \
             take are refused by name"
        );
        Ok(())
    }
}
