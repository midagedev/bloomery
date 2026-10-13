//! The Qwen3.6 verify gate: a load opened to verify drafted rows
//! (`GpuModel::<Body35>::open_verifying`) commits any `accepted` in `1..=m`
//! of a verify of `m` rows with no re-feed and no copy, and a plain load is
//! untouched.
//!
//! What is asserted, on the real Qwen3.6-35B-A3B file (`refset`'s `MODEL`,
//! the file the e2e gate's real tier opens), greedy, at the model's default
//! step mode (graph) and eager:
//! - (a) every accepted: for `m` in {4, 8} and each `accepted` in `1..=m`, a
//!   40-id prompt, one verify of `m` forced rows (the plain run's next ids,
//!   the token at the position first), `commit(accepted)`, then 16 plain
//!   steps. The verify's row argmaxes, the 16 steps' ids and the last step's
//!   logits equal a plain load's run from the same prompt, bit for bit. A
//!   chain of commits (`m, accepted` = 4/3, 8/5, 8/8, 2/1, 8/2, 3/3, 8/1)
//!   walks the lane word round and past the lane count, then 16 steps do the
//!   same. On the base (one lane) every `accepted < m` is refused at commit;
//!   the plain load here prints that refusal (a1) and holds it.
//! - (b) a lookup draft (`runtime::lookup::Lookup`) driven over
//!   `Session<Body35>` (`Session::with_draft`, a verify of two rows) for 64
//!   tokens on three prompts with repeats emits the plain run's ids; the
//!   accepted drafts are above 0 and below the proposals on at least one
//!   prompt, so partial commits ran.
//! - (c) the plain load is untouched: its store bytes and the plan's layout
//!   bytes equal the derivation of the file's header ([`BASE_STORE`]), its
//!   captured decode step holds [`NODES_DECODE`] nodes and a pass of `m`
//!   rows [`NODES_PASS`] + 4m (the e2e gate's pins), and the verifying load
//!   holds the same node counts and exactly 30 · `delta_lane_bytes(32, 128,
//!   8)` = [`LANE_EXTRA`] more store bytes [derived: 420 MiB of lanes and 960
//!   B of stamps], which the plan's layout (`PlanInputs::with_lanes`) prices
//!   to the byte.
//! - (d) a cut behind a commit: a marked prompt of 210 ids, a verify and a
//!   commit of three rows that move the committed lane to lane 2, a cut back
//!   to the mark at 192 (the checkpoint's copy lands in lane 2, which is
//!   restamped), and the replay equal the plain load's cut and replay, bit
//!   for bit.
//! - (e) refusals by name: a step or a pass while a verify waits for its
//!   commit, and a second resident sequence on the verifying load; the
//!   refused call leaves the verify committable.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen35moe_verify: built without the `gpu` feature; see \
         `just gate-gpu-qwen35moe-verify`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_verify", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use app::{RowsLog, Session, SessionError};
    use bloomery_gpu::arch::qwen3moe::{Body35, KvQ8, Open35, Qwen35moeModel};
    use bloomery_gpu::model::{MAX_PASS_ROWS, StepMode};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, verdict};
    use gguf::Split;
    use model::arch::qwen35moe::place::PlanInputs;
    use model::placement::KvBytes;
    use refset::arch::qwen35moe::{D1K, IK, MODEL};
    use runtime::stores::delta_lane_bytes;
    use runtime::{Advance, Lookup, Speculative, Target, Verify, Want};

    type Seat = Session<Body35>;

    /// Cache rows of every load: the longest run below (a 210-id prompt, a
    /// verify and the replay) with room.
    const CTX: usize = 512;

    /// The prompt call's ubatch, which is also the checkpoints' spacing: the
    /// marks of (d)'s 210-id prompt fall at 64, 128 and 192.
    const UBATCH: usize = 64;

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

    /// (d)'s prompt, the mark it cuts back to, and the rows it verifies.
    const PROMPT_D: usize = 210;
    const MARK_D: u32 = 192;

    /// The file's delta shape, from the e2e gate's derivations: 30 delta
    /// layers of 32 value heads of 128, 10 attention layers of 2 KV heads of
    /// 256.
    const N_DELTA: usize = 30;
    const N_ATTN: usize = 10;
    const V_HEADS: usize = 32;
    const STATE: usize = 128;

    /// One delta layer's state (`32·128·128` f32) and conv ring (`11` rows of
    /// `8192` channels, f32), and one attention layer's K and V a position
    /// (`2 · 2 · 256` f16).
    const REC_BYTES: usize = V_HEADS * STATE * STATE * 4 + 11 * 8192 * 4;
    const KV_ROW_BYTES: usize = 2 * 2 * 256 * 2;

    /// [derived] The store bytes of a plain load at [`CTX`]: `30 · 2,457,600 +
    /// 10 · 512 · 2,048`. The plan's layout sums to it, and so does the
    /// load's own `store_bytes`.
    const BASE_STORE: usize = N_DELTA * REC_BYTES + N_ATTN * CTX * KV_ROW_BYTES;

    /// [derived] What the lanes add on the verifying load: 30 layers of the
    /// seven more lanes (2,097,152 B each) and a u32 stamp for each of the
    /// eight, 440,402,880 B = 420 MiB + 960 B.
    const LANE_EXTRA: usize = N_DELTA * (7 * V_HEADS * STATE * STATE * 4 + 8 * 4);

    /// The e2e gate's pins of the plain load's captured graphs (decode step
    /// 520 nodes, a pass of `m` rows 537 + 4m), which the verifying load
    /// holds too: `gdn_delta_lanes` is one launch where `gdn_delta` is.
    const NODES_DECODE: usize = 520;
    const NODES_PASS: usize = 537;

    const WHAT: &str = "gate_qwen35moe_verify";

    /// The chain of (a)'s second case: `(m, accepted)` stages.
    const CHAIN: [(usize, usize); 7] = [(4, 3), (8, 5), (8, 8), (2, 1), (8, 2), (3, 3), (8, 1)];

    struct Log;

    impl RowsLog for Log {
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError> {
            println!("lookup verify pass of {rows} rows captured: {nodes} nodes");
            Ok(())
        }
    }

    fn open_at(verifying: bool) -> Result<Seat, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        if file.architecture() != Some("qwen35moe") {
            return Err(format!("{MODEL} is {:?}, not qwen35moe", file.architecture()).into());
        }
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: UBATCH,
            kv: KvQ8::F16,
            media: false,
        };
        let gpu = Gpu::new()?;
        let m = if verifying {
            Qwen35moeModel::open_verifying(gpu, file, o)?
        } else {
            Qwen35moeModel::open(gpu, file, o)?
        };
        let lanes = m.body(WHAT)?.lanes();
        println!(
            "load verifying={verifying} lanes={lanes} resident_bytes={} store_bytes={} ctx={CTX} \
             ubatch={UBATCH}",
            m.resident_bytes(),
            m.body(WHAT)?.store_bytes()
        );
        let mut s = Seat::from_model(m, u32::try_from(CTX)?);
        s.model_mut().body_parts(WHAT)?.2.set_checkpoints(true);
        Ok(s)
    }

    /// The prompt ids: the first 1,024-token set's prose, `n` of them from
    /// `from`.
    fn prose(n: usize, from: usize) -> Result<Vec<u32>, GateError> {
        let man = RefManifest::open(&data_dir().join(D1K), &IK)?;
        let (_, _, before) = man.step()?;
        before
            .get(from..from + n)
            .map(<[u32]>::to_vec)
            .ok_or_else(|| {
                GateError::from(format!(
                    "the prose set holds {} ids, {} asked for",
                    before.len(),
                    from + n
                ))
            })
    }

    /// `n` greedy ids from `prompt`, each with the logits that produced it
    /// when `logits`: id `i + 1` is the argmax after feeding id `i`. Starts
    /// from a reset.
    fn greedy(
        s: &mut Seat,
        prompt: &[u32],
        n: usize,
        logits: bool,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
        let want = if logits { Want::Logits } else { Want::Argmax };
        s.reset()?;
        let (mut ids, mut rows) = (Vec::with_capacity(n), Vec::new());
        let push = |out: runtime::Out<'_>, ids: &mut Vec<u32>, rows: &mut Vec<Vec<f32>>| {
            ids.push(out.argmax());
            if let runtime::Out::Logits { row, .. } = out {
                rows.push(row.to_vec());
            }
        };
        let out = s.prompt(prompt, want)?;
        push(out, &mut ids, &mut rows);
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            let out = s.step(last, want)?;
            push(out, &mut ids, &mut rows);
        }
        Ok((ids, rows))
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// One stage of a chain: verify `m` forced rows of the reference ids from
    /// `idx`, then commit `accepted` of them. The verify's row argmaxes.
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
            3 => stage_m::<3>(s, r, idx, accepted),
            4 => stage_m::<4>(s, r, idx, accepted),
            5 => stage_m::<5>(s, r, idx, accepted),
            6 => stage_m::<6>(s, r, idx, accepted),
            7 => stage_m::<7>(s, r, idx, accepted),
            8 => stage_m::<8>(s, r, idx, accepted),
            _ => Err(SessionError::Refused(format!("a verify of {m} rows"))),
        }
    }

    /// A script of stages from a reset and the prompt, then
    /// [`STEPS_AFTER`] plain steps: whether every verify's rows, every step's
    /// id and the last step's logits equal the plain run's, and the first
    /// refusal if one came.
    fn script(
        s: &mut Seat,
        prompt: &[u32],
        reference: &(Vec<u32>, Vec<Vec<f32>>),
        stages: &[(usize, usize)],
    ) -> Result<(bool, String), GateError> {
        let (r, logits) = reference;
        s.reset()?;
        let first = s.prompt(prompt, Want::Argmax)?.argmax();
        if first != r[0] {
            return Ok((
                false,
                format!("the prompt's argmax {first}, the plain run's {}", r[0]),
            ));
        }
        let (mut idx, mut lane) = (0, 0u32);
        for &(m, accepted) in stages {
            match stage(s, r, idx, (m, accepted)) {
                Ok(out) => {
                    if out[..] != r[idx + 1..idx + 1 + m] {
                        return Ok((
                            false,
                            format!(
                                "verify of {m} rows at {idx}: row argmaxes {out:?}, the plain run's {:?}",
                                &r[idx + 1..idx + 1 + m]
                            ),
                        ));
                    }
                }
                Err(e) => return Ok((false, format!("verify {m}/{accepted} refused: {e}"))),
            }
            idx += accepted;
            // The lane word after a commit: the lane row `accepted − 1` wrote,
            // `(c + accepted − 1) mod lanes`, written here from the rule's
            // statement and not from the code that applies it.
            lane = (lane + u32::try_from(accepted - 1)?) % u32::try_from(MAX_PASS_ROWS)?;
            let got = s.model().body(WHAT)?.committed_lane();
            if got != Some(lane) {
                return Ok((
                    false,
                    format!("after verify {m}/{accepted} the lane word is {got:?}, want {lane}"),
                ));
            }
        }
        let mut last_row = Vec::new();
        for i in 0..STEPS_AFTER {
            let out = match s.step(r[idx + i], Want::Logits) {
                Ok(out) => out,
                Err(e) => return Ok((false, format!("step {i} after the commits: {e}"))),
            };
            if out.argmax() != r[idx + i + 1] {
                return Ok((
                    false,
                    format!(
                        "step {i}: id {}, the plain run's {}",
                        out.argmax(),
                        r[idx + i + 1]
                    ),
                ));
            }
            if let runtime::Out::Logits { row, .. } = out {
                last_row = row.to_vec();
            }
        }
        if !same_bits(&last_row, &logits[idx + STEPS_AFTER]) {
            return Ok((
                false,
                "the last step's logits differ from the plain run's".into(),
            ));
        }
        Ok((
            true,
            format!("{} commits, {STEPS_AFTER} steps", stages.len()),
        ))
    }

    /// The captured graphs and the bytes of a load: decode nodes, the pass's
    /// at 5 and 8 rows, the stores' bytes.
    struct Facts {
        decode: usize,
        pass5: usize,
        pass8: usize,
        stores: usize,
        resident: usize,
    }

    fn facts(s: &mut Seat) -> Result<Facts, GateError> {
        let m = s.model_mut();
        let decode = m.capture_step()?;
        let pass5 = m.capture_rows::<5>()?;
        let pass8 = m.capture_rows::<8>()?;
        Ok(Facts {
            decode,
            pass5,
            pass8,
            stores: m.body(WHAT)?.store_bytes(),
            resident: m.resident_bytes(),
        })
    }

    /// The plan's layout bytes of the file at [`CTX`]: every layer's
    /// recurrent and positional bytes, with `lanes` lanes when asked.
    fn plan_bytes(lanes: Option<usize>) -> Result<usize, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = PlanInputs::describe(&file)?;
        let inputs = match lanes {
            Some(l) => inputs.with_lanes(l),
            None => inputs,
        };
        let sum: u64 = (0..inputs.hp.n_trunk)
            .map(|l| inputs.kv.layer_bytes(l, CTX as u64))
            .sum();
        Ok(usize::try_from(sum)?)
    }

    /// (c) (module doc).
    fn untouched(plain: &Facts, verifying: &Facts) -> Result<bool, GateError> {
        let plan_plain = plan_bytes(None)?;
        let plan_ver = plan_bytes(Some(MAX_PASS_ROWS))?;
        let nodes = |f: &Facts| {
            f.decode == NODES_DECODE
                && f.pass5 == NODES_PASS + 4 * 5
                && f.pass8 == NODES_PASS + 4 * 8
        };
        let plain_ok = nodes(plain) && plain.stores == BASE_STORE && plan_plain == BASE_STORE;
        println!(
            "plain load: graph nodes decode {} pass5 {} pass8 {} (want {NODES_DECODE}, {}, {}); \
             store bytes {} and plan layout bytes {plan_plain} (want {BASE_STORE}, derived); \
             resident {} {}",
            plain.decode,
            plain.pass5,
            plain.pass8,
            NODES_PASS + 20,
            NODES_PASS + 32,
            plain.stores,
            plain.resident,
            verdict(plain_ok)
        );
        // 4 B more: the lane word.
        let ver_ok = nodes(verifying)
            && verifying.stores == BASE_STORE + LANE_EXTRA
            && plan_ver == BASE_STORE + LANE_EXTRA
            && verifying.resident == plain.resident + LANE_EXTRA + 4;
        println!(
            "verifying load: graph nodes decode {} pass5 {} pass8 {}; store bytes {} and plan \
             layout bytes {plan_ver} (want {}: the plain bytes and {LANE_EXTRA} B of lanes and \
             stamps, 30 · delta_lane_bytes(32, 128, 8) = {}); resident {} (want {}) {}",
            verifying.decode,
            verifying.pass5,
            verifying.pass8,
            verifying.stores,
            BASE_STORE + LANE_EXTRA,
            30 * delta_lane_bytes(V_HEADS, STATE, MAX_PASS_ROWS),
            verifying.resident,
            plain.resident + LANE_EXTRA + 4,
            verdict(ver_ok)
        );
        Ok(plain_ok && ver_ok)
    }

    /// What the plain phase leaves for the verifying phase to equal.
    struct Refs {
        prompt_a: Vec<u32>,
        a: (Vec<u32>, Vec<Vec<f32>>),
        lookups: Vec<(Vec<u32>, Vec<u32>)>,
        prompt_d: Vec<u32>,
        d: (Vec<u32>, Vec<f32>),
    }

    /// (d)'s run: a marked prompt, the verify's detour when `detour` (a
    /// verify of four rows of (d)'s own ids and a commit of three), a cut to
    /// the mark and the replay — the ids from the mark on, the steps, and
    /// the last logits.
    fn cut_run(
        s: &mut Seat,
        prompt: &[u32],
        detour: bool,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        s.reset()?;
        let first = s.prompt(prompt, Want::Argmax)?.argmax();
        if detour {
            let rows = [first, prompt[0], prompt[1], prompt[2]];
            s.verify::<4>(rows)?;
            s.commit(3)?;
        }
        s.cut(MARK_D)?;
        let mut ids = Vec::new();
        let mut last = Vec::new();
        for &t in &prompt[MARK_D as usize..] {
            ids.push(s.step(t, Want::Argmax)?.argmax());
        }
        for _ in 0..STEPS_AFTER {
            let t = ids[ids.len() - 1];
            let out = s.step(t, Want::Logits)?;
            ids.push(out.argmax());
            if let runtime::Out::Logits { row, .. } = out {
                last = row.to_vec();
            }
        }
        Ok((ids, last))
    }

    /// The repeating prompts of (b): a passage twice, a passage and the
    /// start of its repeat, and a short passage four times.
    fn lookup_prompts() -> Result<Vec<Vec<u32>>, GateError> {
        let seg = prose(48, 64)?;
        let twice = [seg.clone(), seg.clone()].concat();
        let start = [seg.clone(), seg[..24].to_vec()].concat();
        let four = seg[..16].repeat(4);
        Ok(vec![twice, start, four])
    }

    /// The plain load's phase: the references, the structure facts, and the
    /// base refusal of a partial commit (a1).
    fn plain_phase() -> Result<(Refs, Facts, bool), GateError> {
        let mut s = open_at(false)?;
        let mut ok = true;
        let f = facts(&mut s)?;
        let prompt_a = prose(PROMPT_A, 0)?;
        let a = greedy(&mut s, &prompt_a, REF_IDS, true)?;
        let lookups = lookup_prompts()?
            .into_iter()
            .map(|p| greedy(&mut s, &p, GEN + 8, false).map(|(ids, _)| (p, ids)))
            .collect::<Result<Vec<_>, _>>()?;
        let prompt_d = lcg(PROMPT_D, 248_000);
        let d = cut_run(&mut s, &prompt_d, false)?;
        // (a1) the base: one lane keeps no state behind the verify's rows, so a
        // commit of fewer than the verify's rows is refused by name and the
        // whole verify commits.
        s.reset()?;
        s.prompt(&prompt_a, Want::Argmax)?;
        s.verify::<4>(std::array::from_fn(|j| a.0[j]))?;
        let refused = match s.commit(2) {
            Err(e) => e.to_string(),
            Ok(()) => String::new(),
        };
        let a1 = refused.contains("no checkpoint") && !refused.is_empty();
        println!(
            "(a1) plain load, verify of 4 rows, commit(2): {refused} {}",
            verdict(a1)
        );
        ok &= a1;
        s.reset()?;
        s.prompt(&prompt_a, Want::Argmax)?;
        s.verify::<4>(std::array::from_fn(|j| a.0[j]))?;
        let whole = s.commit(4);
        println!(
            "(a1) plain load, the whole verify commits: {whole:?} {}",
            verdict(whole.is_ok())
        );
        ok &= whole.is_ok();
        Ok((
            Refs {
                prompt_a,
                a,
                lookups,
                prompt_d,
                d,
            },
            f,
            ok,
        ))
    }

    /// `n` ids of an lcg over the vocabulary: every id a row of the
    /// embedding, the routing spread over the experts.
    fn lcg(n: usize, vocab: usize) -> Vec<u32> {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((x >> 33) % vocab as u64) as u32
            })
            .collect()
    }

    /// (a) (module doc).
    fn every_accepted(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let mut ok = true;
        for mode in [StepMode::Graph, StepMode::Eager] {
            s.model_mut().set_mode(mode);
            let mut red = 0;
            let mut first_red = String::new();
            let mut cases = 0;
            for m in [4usize, 8] {
                for accepted in 1..=m {
                    let (pass, note) = script(s, &refs.prompt_a, &refs.a, &[(m, accepted)])?;
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
                "(a) {mode:?}: {cases} cases (m in 4, 8; accepted 1..=m), {red} red {}",
                verdict(red == 0)
            );
            ok &= red == 0;
            let (pass, note) = script(s, &refs.prompt_a, &refs.a, &CHAIN)?;
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

    /// (b) (module doc).
    fn lookup(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let mut ok = true;
        let mut partial_on_some = false;
        for mode in [StepMode::Graph, StepMode::Eager] {
            s.model_mut().set_mode(mode);
            for (i, (prompt, want)) in refs.lookups.iter().enumerate() {
                match lookup_run(s, prompt) {
                    Ok((tokens, proposals, accepted, partial)) => {
                        let same = tokens[..GEN] == want[..GEN];
                        partial_on_some |= accepted > 0 && accepted < proposals;
                        println!(
                            "(b) {mode:?} prompt {i} ({} ids): {GEN} tokens equal the plain run's \
                             {same}; proposals {proposals}, accepted drafts {accepted}, partial \
                             commits {partial} {}",
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
        println!(
            "(b) accepted drafts above 0 and below the proposals on some prompt (partial commits \
             ran) {}",
            verdict(partial_on_some)
        );
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok && partial_on_some)
    }

    /// (d) (module doc).
    fn behind_a_commit(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let got = cut_run(s, &refs.prompt_d, true)?;
        let ids = got.0 == refs.d.0;
        let bits = same_bits(&got.1, &refs.d.1);
        println!(
            "(d) cut to {MARK_D} after a commit of 3/4 (lane 2), replay: {} ids equal the plain \
             cut's {ids}, last logits bit for bit {bits} {}",
            got.0.len(),
            verdict(ids && bits)
        );
        Ok(ids && bits)
    }

    /// (e) (module doc).
    fn refusals(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let r = &refs.a.0;
        s.reset()?;
        s.prompt(&refs.prompt_a, Want::Argmax)?;
        s.verify::<4>(std::array::from_fn(|j| r[j]))?;
        let step = match s.model_mut().step(&[r[1]]) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        let pass = match s.model_mut().step_rows::<2>([r[1], r[2]]) {
            Err(e) => e.to_string(),
            Ok(_) => String::new(),
        };
        let waits = step.contains("waits for its commit") && pass.contains("waits for its commit");
        println!(
            "(e) step while a verify waits: {step}; pass: {pass} {}",
            verdict(waits)
        );
        s.commit(2)?;
        let after = s.step(r[2], Want::Argmax)?.argmax();
        let committable = after == r[3];
        println!(
            "(e) the refused calls moved nothing: commit(2) then the step gives {after} (the plain \
             run's {}) {}",
            r[3],
            verdict(committable)
        );
        let slots = match s.model_mut().add_slots(2) {
            Err(GpuError::Shape { .. }) => true,
            other => {
                println!("(e) a second resident sequence: {other:?}");
                false
            }
        };
        println!(
            "(e) a second resident sequence is refused by name {}",
            verdict(slots)
        );
        Ok(waits && committable && slots)
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
        let (refs, plain_facts, mut ok) = plain_phase()?;
        let mut s = open_at(true)?;
        let ver_facts = facts(&mut s)?;
        ok &= untouched(&plain_facts, &ver_facts)?;
        let r = every_accepted(&mut s, &refs);
        ok &= clause("(a)", &mut s, r);
        let r = lookup(&mut s, &refs);
        ok &= clause("(b)", &mut s, r);
        let r = behind_a_commit(&mut s, &refs);
        ok &= clause("(d)", &mut s, r);
        let r = refusals(&mut s, &refs);
        ok &= clause("(e)", &mut s, r);
        println!("{WHAT}: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
