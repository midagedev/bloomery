//! The Qwen3.6 verify gate: a load opened to verify drafted rows
//! (`GpuModel::<Body35>::open_verifying`) commits any `accepted` in `1..=m`
//! of a verify of `m` rows with no re-feed and no copy, and a plain load is
//! untouched.
//!
//! What is asserted, on the real Qwen3.6-35B-A3B file (`BLOOMERY_REF_MODEL`,
//! the file the e2e gate's real tier opens), greedy, at the model's default
//! step mode (graph) and eager, on prompts of the shared prose corpus
//! (`prose_ids`, Qwen3.8's ids: the same vocabulary):
//! - (a) every accepted: for `m` in {4, 8} and each `accepted` in `1..=m`, a
//!   40-id prompt, one verify of `m` forced rows (the plain run's next ids,
//!   the token at the position first), `commit(accepted)`, then 16 plain
//!   steps. The verify's row argmaxes, the 16 steps' ids and the last step's
//!   logits equal a plain load's run from the same prompt, bit for bit. A
//!   chain of commits (`m, accepted` = 4/3, 8/5, 8/8, 2/1, 8/2, 3/3, 8/1)
//!   replays every kept count from the records, then 16 steps do the same.
//!   On a plain load every `accepted < m` is refused at commit; the plain
//!   load here prints that refusal (a1) and holds it.
//! - (b) a lookup draft (`runtime::lookup::Lookup`) driven over
//!   `Session<Body35>` (`Session::with_draft`, a verify of two rows) for 64
//!   tokens on three prompts with repeats emits the plain run's ids; the
//!   accepted drafts are above 0 and below the proposals on at least one
//!   prompt, so partial commits ran.
//! - (c) the plain load is untouched: its store bytes and the plan's layout
//!   bytes equal the derivation of the file's header ([`BASE_STORE`]), its
//!   captured decode step holds [`NODES_DECODE`] nodes and a pass of `m`
//!   rows [`NODES_PASS`] + 4m (the e2e gate's pins), and the verifying load
//!   holds the same node counts and exactly 30 · `delta_record_bytes(32,
//!   128, 8)` = [`RECORD_EXTRA`] more store bytes [derived: 11,796,480 B of
//!   verify records], which the plan's layout (`PlanInputs::verifying`)
//!   prices to the byte.
//! - (d) a cut behind a commit: a marked prompt of 210 ids, a verify and a
//!   commit of three rows, a cut back to the mark at 192 (the checkpoint's
//!   copy lands in the lane, which is restamped), and the replay equal the
//!   plain load's cut and replay, bit for bit.
//! - (e) refusals by name: a step or a pass while a verify waits for its
//!   commit; the refused call leaves the verify committable.
//! - (f) an unread prompt call (`CallEnd::Unread`, no head) over a fault
//!   planted in the card's fault word returns that fault by name and leaves
//!   the model poisoned with it, and a reset clears it (the prompt after it
//!   gives the plain run's argmax). FAIL-first: without the fault read after
//!   a headless last unit (`Body35::feed35`) the call returns `Ok`.
//! - (g) a second resident sequence opens on the verifying load (each slot
//!   keeps its own records), run last so the slot stays resident. Then, taps
//!   off, both slots at the end of the (a) prompt and slot 1's verify of two
//!   rows waiting: a pass of slot 0 then slot 1 is refused by name for slot
//!   1 before any slot moves, so slot 0 steps on to the plain run's id, and
//!   slot 1's commit of both rows and its step give the plain run's ids too.
//!   FAIL-first: without the pass's check of a waiting verify
//!   (`check_slot`'s `refuse_pending`) slot 0 holds a verify of its own after
//!   the refusal and its step is refused.
//! - (h) a closing call (`CallEnd::Close`) by the body's prompt schedule
//!   (`app::Prompt`, `prefill_gate::closed_run`), marks on: at 150 and 160
//!   ids from a reset the wide path, its last id folded into the last ubatch
//!   (rows 128..n, a chunk left open at 150 and completed at 160, which the
//!   call keeps out of the lanes), at 5 ids passes, the last id a run of its
//!   own (`closes_by_ubatch` says which). The call's argmax
//!   and logits equal the one call's of the same ids, and so do three greedy
//!   steps' ids and logits where no chunk was kept; its end mark stands at
//!   `n − 1` and not at `n`; the last id stepped from that mark (a rollback
//!   there) gives the call less that id and its step bit for bit. FAIL-first:
//!   a point whose carries keep the row past the mark (`recur::close_point`
//!   not run) steps from position `n`.
//! - the draft taps (`Tapped + TappedPrompt for Session<Body35>`: the rows
//!   entering the layers [`TAP_LAYERS`], 0 and the last among them, which a
//!   block draft reads), set on the verifying load:
//!   - (td) verify equals steps: the taps of each row of an 8-row verify of
//!     forced rows equal 8 plain steps' taps at the same positions, bit for
//!     bit, graph and eager; rows past the last walk are refused by name.
//!   - (te) taps perturb nothing: `prompt_tapped`'s argmax and last logits
//!     equal `Target::prompt`'s on the same ids, and so do the next 16
//!     steps' (ids and logits), bit for bit, on an 8-id prompt (the pass
//!     path) and a 210-id one (four ubatches, the wide path).
//!   - (tf) the window: the sink is called exactly for the last `window`
//!     positions, in order, with no gap or repeat (windows 0, 1, 5 and
//!     more than the call on the pass prompt; 100 on the wide one, where the
//!     units are derived; 24 over five pass units of a model-level call).
//!     The pass path's rows equal a step run's taps bit for bit; the wide
//!     path's rows equal its embedding slot (no bit-equal reference past
//!     it, the distance is printed); a sink that refuses ends the call with
//!     its error.
//!   - (tg) the layer anchor: the row entering layer 0 is the token's
//!     embedding row (the file's `token_embd.weight` dequantized on the
//!     host), on steps and on the wide path; at a step, the row entering
//!     layer `L > 0` equals the gate instrument's output tap of layer
//!     `L − 1` (`set_layer_taps`), bit for bit, and the instrument still
//!     refuses a prompt unit by name.
//!   - (th) the captured step and passes hold one copy node per tapped layer
//!     more than (c)'s counts, and the counts return with the taps off.
//!   - (ti) taps past the layer count or out of order are refused by name,
//!     and a refused call leaves taps off.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_verify", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "gpu")]
mod gate {
    use app::{RowsLog, Session, SessionError};
    use bloomery_gpu::arch::qwen3moe::{Body35, Open35, PrefillPath, Qwen35moeModel};
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu::model::{MAX_PASS_ROWS, StepMode};
    use bloomery_gpu::{Fault, FaultSite, Gpu, GpuError, LAYER_HEAD};
    use bloomery_gpu_gates::prefill_gate::closed_run;
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, prose_ids, verdict};
    use gguf::quant::dequant_row;
    use model::arch::Arch;
    use model::arch::qwen35moe::place::PlanInputs;
    use model::placement::KvBytes;
    use runtime::prompt::CallEnd;
    use runtime::stores::{DELTA_REPLAY_BYTES, delta_carry_bytes, delta_record_bytes};
    use runtime::{
        Advance, Lookup, RowLogits, Speculative, Tapped, TappedPrompt, Target, Verify, Want,
    };

    use crate::greedy::greedy;

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

    /// One delta layer's state (`32·128·128` f32), conv ring (`11` rows of
    /// `8192` channels, f32), the lane's u32 stamp and the chunked prompt
    /// rule's carry (4 words and 16 rows of `(16 + 32)·128 + 2·32` f32,
    /// 397,328 B), and one attention layer's K and V a position
    /// (`2 · 2 · 256` f16).
    const CARRY_BYTES: usize = (4 + 16 * ((16 + V_HEADS) * STATE + 2 * V_HEADS)) * 4;
    // PIN(2026-11-02): one stamped lane, its carry and a verify record replace the
    // verifying load's lanes, so the store bytes count the stamp and the carry.
    const REC_BYTES: usize = V_HEADS * STATE * STATE * 4 + 11 * 8192 * 4 + 4 + CARRY_BYTES;
    const KV_ROW_BYTES: usize = 2 * 2 * 256 * 2;

    /// [derived] The store bytes of a plain load at [`CTX`]: `30 · 2,854,932 +
    /// 10 · 512 · 2,048`. The plan's layout sums to it, and so does the
    /// load's own `store_bytes`.
    const BASE_STORE: usize = N_DELTA * REC_BYTES + N_ATTN * CTX * KV_ROW_BYTES;

    /// [derived] What the records add on the verifying load: 30 layers of a
    /// record of 8 rows of `32 · 3 · 128` f32 (393,216 B), 11,796,480 B.
    const RECORD_EXTRA: usize = N_DELTA * MAX_PASS_ROWS * V_HEADS * 3 * STATE * 4;

    /// [derived] The commit's replay table on the verifying load: a row of
    /// five u64 words a delta layer, 1,200 B.
    const TABLE_BYTES: usize = N_DELTA * 5 * 8;
    const _: () = assert!(
        CARRY_BYTES as u64 == delta_carry_bytes(V_HEADS, 16, STATE)
            && TABLE_BYTES as u64 == N_DELTA as u64 * DELTA_REPLAY_BYTES
    );

    /// The e2e gate's pins of the plain load's captured graphs (decode step
    /// 520 nodes, a pass of `m` rows 537 + 4m), which the verifying load
    /// holds too: `gdn_delta_lanes` is one launch where `gdn_delta` is.
    const NODES_DECODE: usize = 520;
    const NODES_PASS: usize = 537;

    const WHAT: &str = "gate_qwen35moe_verify";
    const RECIPE: &str = "gate-gpu-qwen35moe-verify";

    /// The chain of (a)'s second case: `(m, accepted)` stages.
    const CHAIN: [(usize, usize); 7] = [(4, 3), (8, 5), (8, 8), (2, 1), (8, 2), (3, 3), (8, 1)];

    struct Log;

    impl RowsLog for Log {
        fn capture_rows(
            &mut self,
            rows: usize,
            nodes: usize,
            _: app::PhaseBytes,
        ) -> Result<(), SessionError> {
            println!("lookup verify pass of {rows} rows captured: {nodes} nodes");
            Ok(())
        }
    }

    fn open_at(verifying: bool) -> Result<Seat, GateError> {
        let file = open_split(Arch::Qwen35moe, RECIPE)?;
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
        let verifies = m.body(WHAT)?.verifies();
        println!(
            "load verifying={verifying} verifies={verifies} resident_bytes={} store_bytes={} \
             ctx={CTX} ubatch={UBATCH}",
            m.resident_bytes(),
            m.body(WHAT)?.store_bytes()
        );
        let mut s = Seat::from_model(m, u32::try_from(CTX)?);
        s.model_mut().body_parts(WHAT)?.2.set_checkpoints(true);
        Ok(s)
    }

    /// The prompt ids: `n` ids of the shared prose corpus from `from`.
    fn prose(n: usize, from: usize) -> Result<Vec<u32>, GateError> {
        Ok(prose_ids("qwen4exp", from + n)?.split_off(from))
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
        let mut idx = 0;
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
    /// recurrent and positional bytes, of a load that verifies or not.
    fn plan_bytes(verifies: bool) -> Result<usize, GateError> {
        let file = open_split(Arch::Qwen35moe, RECIPE)?;
        let inputs = PlanInputs::describe(&file)?.verifying(verifies)?;
        let sum: u64 = (0..inputs.spec.layers.len())
            .map(|l| inputs.kv.layer_bytes(l, CTX as u64))
            .sum();
        Ok(usize::try_from(sum)?)
    }

    /// (c) (module doc).
    fn untouched(plain: &Facts, verifying: &Facts) -> Result<bool, GateError> {
        let plan_plain = plan_bytes(false)?;
        let plan_ver = plan_bytes(true)?;
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
        // The replay table more: one row a delta layer.
        let ver_ok = nodes(verifying)
            && verifying.stores == BASE_STORE + RECORD_EXTRA
            && plan_ver == BASE_STORE + RECORD_EXTRA
            && verifying.resident == plain.resident + RECORD_EXTRA + TABLE_BYTES;
        println!(
            "verifying load: graph nodes decode {} pass5 {} pass8 {}; store bytes {} and plan \
             layout bytes {plan_ver} (want {}: the plain bytes and {RECORD_EXTRA} B of records, \
             30 · delta_record_bytes(32, 128, 8) = {}); resident {} (want {}) {}",
            verifying.decode,
            verifying.pass5,
            verifying.pass8,
            verifying.stores,
            BASE_STORE + RECORD_EXTRA,
            30 * delta_record_bytes(V_HEADS, STATE, MAX_PASS_ROWS),
            verifying.resident,
            plain.resident + RECORD_EXTRA + TABLE_BYTES,
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
        // (a1) the base: a plain load's verify runs in place and records no
        // rows, so a commit of fewer than the verify's rows is refused by name
        // and the whole verify commits.
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
            "(d) cut to {MARK_D} after a commit of 3/4, replay: {} ids equal the plain cut's \
             {ids}, last logits bit for bit {bits} {}",
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
        Ok(waits && committable)
    }

    /// (f) (module doc).
    fn unread_fault(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        s.reset()?;
        let fault = Fault::at(LAYER_HEAD, FaultSite::QuantColumn);
        s.model().plant_fault(fault)?;
        let call = s
            .model_mut()
            .prefill_call(&refs.prompt_a, PrefillPath::Wide, CallEnd::Unread);
        let poisoned = s.model().poisoned();
        let named = matches!(&call, Err(GpuError::Fault { fault: f, .. }) if *f == fault);
        s.reset()?;
        let clean = s.model().poisoned().is_none()
            && s.prompt(&refs.prompt_a, Want::Argmax)?.argmax() == refs.a.0[0];
        let pass = named && poisoned == Some(fault) && clean;
        println!(
            "(f) an unread prompt call over a planted fault: {}, poisoned {poisoned:?}, after reset \
             a clean prompt {clean} {}",
            match &call {
                Ok(_) => "returned".to_owned(),
                Err(e) => format!("refused \"{e}\""),
            },
            verdict(pass)
        );
        Ok(pass)
    }

    /// (h)'s calls: ids, whether the last id folds into the last ubatch.
    // PIN(2026-11-02): Wide borrows its short tail, so 136 ids close; the exact Gemm cut is checked below.
    const CLOSE_CASES: [(usize, bool); 4] = [(136, true), (150, true), (160, true), (5, false)];

    /// (h) (module doc).
    fn close_fold(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let ids = &refs.prompt_d;
        let mut ok = true;
        // Gemm keeps its eight-row ubatch; Wide borrows a short tail up to
        // nine rows. The strict cut applies before a closing call chooses
        // the chunked rule. Changing > to >= makes the eight-row case red.
        for (n, want) in [(8, false), (9, true)] {
            s.reset()?;
            let got = s.model().closes_by_ubatch(n, PrefillPath::Gemm)?;
            let pass = got == want;
            println!(
                "(h) Gemm {n} ids: folds {got} (want {want}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        for (n, folds) in CLOSE_CASES {
            let (ids, end) = (&ids[..n], u32::try_from(n - 1)?);
            let (head, last) = (&ids[..n - 1], &ids[n - 1..]);
            s.reset()?;
            let rides = s.model().closes_by_ubatch(n, PrefillPath::Wide)?;
            let want = closed_run(s.model_mut(), ids, CallEnd::Argmax)?;
            s.reset()?;
            s.model_mut()
                .prefill_call(head, PrefillPath::Wide, CallEnd::Unread)?;
            let cut = closed_run(s.model_mut(), last, CallEnd::Argmax)?;
            s.reset()?;
            let whole = closed_run(s.model_mut(), ids, CallEnd::Close)?;
            let points = s.model().body(WHAT)?.checkpoints().positions();
            s.model_mut().rollback(end)?;
            let stepped = closed_run(s.model_mut(), last, CallEnd::Argmax)?;
            let marked = points.contains(&end) && !points.contains(&(end + 1));
            let kept = n.is_multiple_of(bloomery_gpu::linear::CHUNK_ROWS);
            let first = (whole.0, &whole.1) == (want.0, &want.1);
            let like = first && (kept || whole == want);
            let pass = like && stepped == cut && marked && rides == folds;
            println!(
                "(h) {n} ids: the closing call = the one call's argmax and logits {first}, its \
                 steps' bits {} (held {}); its last id stepped from the end mark = the call \
                 less it and its step {}; points {points:?}; folds {rides} (want {folds}) {}",
                whole.2 == want.2,
                !kept,
                stepped == cut,
                verdict(pass)
            );
            ok &= pass;
        }
        s.reset()?;
        Ok(ok)
    }

    /// (g) (module doc).
    // PIN(2026-11-02): each slot holds its own records, so a second resident
    // sequence opens on the verifying load where it was refused.
    fn second_slot(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let slots = match s.model_mut().add_slots(2) {
            Ok(()) => true,
            Err(e) => {
                println!("(g) a second resident sequence: {e}");
                false
            }
        };
        println!(
            "(g) a second resident sequence opens on the verifying load {}",
            verdict(slots)
        );
        Ok(slots && pass_beside_a_verify(s, refs)?)
    }

    /// (g)'s pass beside a waiting verify (module doc).
    fn pass_beside_a_verify(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let r = &refs.a.0;
        let n = u32::try_from(refs.prompt_a.len())?;
        s.set_taps(&[])?;
        for slot in [0, 1] {
            s.model_mut().select_slot(slot)?;
            s.reset()?;
            s.prompt(&refs.prompt_a, Want::Argmax)?;
        }
        s.model_mut().step_rows::<2>([r[0], r[1]])?;
        s.model_mut().select_slot(0)?;
        let refused = match s.model_mut().step_slots(&[(0, &[r[0]]), (1, &[r[2]])]) {
            Err(e) => e.to_string(),
            Ok(out) => format!("ran: {:?}", out.ids),
        };
        let named = refused.contains("slot 1") && refused.contains("waits for its commit");
        s.model_mut().select_slot(0)?;
        let zero = s.model_mut().step(&[r[0]]).map_err(|e| e.to_string());
        s.model_mut().select_slot(1)?;
        s.model_mut().rollback(n + 2)?;
        let one = s.model_mut().step(&[r[2]]).map_err(|e| e.to_string());
        s.model_mut().select_slot(0)?;
        let pass = named && zero == Ok(r[1]) && one == Ok(r[3]);
        println!(
            "(g) a pass while slot 1's verify waits: {refused}; then slot 0 steps to {zero:?} \
             (the plain run's {}), slot 1 commits both rows and steps to {one:?} (the plain \
             run's {}) {}",
            r[1],
            r[3],
            verdict(pass)
        );
        Ok(pass)
    }

    /// The tapped layers of (td)–(ti): a spread of the 40 layers with the
    /// first and the last [derived: [`N_DELTA`] + [`N_ATTN`] layers].
    const TAP_LAYERS: [usize; 5] = [0, 9, 19, 29, 39];
    const N_LAYER: usize = N_DELTA + N_ATTN;

    /// (tf)'s wide window, and the units it expects [derived]: the 210-id
    /// prompt cut at its marks (64, 128, 192) runs ubatches of 64, 64, 64
    /// and 18 rows (a wide unit takes 9 rows and more), so the window of the
    /// last 100 positions, from 110, is read after three of them: 18 rows of
    /// the second, the whole third, the whole fourth.
    const WINDOW_WIDE: usize = 100;
    const CALLS_WIDE: [(u32, usize); 3] = [(110, 18), (128, 64), (192, 18)];

    /// (tf)'s windows on the 8-id prompt (one pass): none, the last position,
    /// five, and more than the call holds.
    const WINDOWS_SHORT: [usize; 4] = [0, 1, 5, 1000];

    /// (tf)'s multi-unit pass-path call: [`PROMPT_A`] ids by passes of
    /// `GEMV_COLS` = 8 rows with a window of 24 positions, from 16
    /// [derived]: three units.
    const WINDOW_PASSES: usize = 24;
    const CALLS_PASSES: [(u32, usize); 3] = [(16, 8), (24, 8), (32, 8)];

    /// Whether every `hidden`-value segment of `rows` is finite and holds a
    /// nonzero value: rows the walk wrote, not a buffer's zeros.
    fn live(rows: &[f32], hidden: usize) -> bool {
        !rows.is_empty()
            && rows
                .chunks(hidden)
                .all(|c| c.iter().all(|v| v.is_finite()) && c.iter().any(|&v| v != 0.0))
    }

    /// The first `(row, slot)` of `a` and `b` (rows of `width` values, `slot`s
    /// of `hidden`) that differ in bits, with the largest absolute
    /// difference; `None` when equal.
    fn first_diff(a: &[f32], b: &[f32], (width, hidden): (usize, usize)) -> Option<String> {
        if a.len() != b.len() {
            return Some(format!("{} values against {}", a.len(), b.len()));
        }
        let at = a
            .iter()
            .zip(b)
            .position(|(x, y)| x.to_bits() != y.to_bits())?;
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        Some(format!(
            "row {} slot {} differ first (largest |difference| {worst:e})",
            at / width,
            at % width / hidden
        ))
    }

    /// The tapped rows of positions `from..` of `ids` fed one step each from
    /// a reset, and the argmax after the last.
    fn step_taps(s: &mut Seat, ids: &[u32], from: usize) -> Result<(Vec<f32>, u32), GateError> {
        s.reset()?;
        let (mut rows, mut last) = (Vec::new(), 0);
        for (p, &t) in ids.iter().enumerate() {
            last = s.step(t, Want::Argmax)?.argmax();
            if p >= from {
                rows.extend_from_slice(s.taps(1)?);
            }
        }
        Ok((rows, last))
    }

    /// A greedy run's ids and the logits that produced them ([`greedy`]).
    type Plain = (Vec<u32>, Vec<Vec<f32>>);

    /// A tapped prompt's sink calls: the first position, the rows.
    type Calls = Vec<(u32, Vec<f32>)>;

    /// [`greedy`] with the prompt through `prompt_tapped`: the same ids and
    /// logits (the prompt's from row 0 of the call), and the sink's calls,
    /// `(first position, rows)`.
    fn greedy_tapped(
        s: &mut Seat,
        prompt: &[u32],
        n: usize,
        window: usize,
    ) -> Result<(Plain, Calls), GateError> {
        s.reset()?;
        let mut calls = Vec::new();
        let first = s.prompt_tapped(prompt, window, &mut |at, rows| {
            calls.push((at, rows.to_vec()));
            Ok(())
        })?;
        let (mut ids, mut rows) = (vec![first], vec![s.row_logits(0)?.to_vec()]);
        while ids.len() < n {
            let out = s.step(ids[ids.len() - 1], Want::Logits)?;
            ids.push(out.argmax());
            if let runtime::Out::Logits { row, .. } = out {
                rows.push(row.to_vec());
            }
        }
        Ok(((ids, rows), calls))
    }

    /// The embedding rows of `ids` as the file's `token_embd.weight`
    /// dequantizes (`gguf::quant::dequant_row`, the scalar reference), one
    /// `hidden` row each.
    fn embedding_rows(ids: &[u32]) -> Result<Vec<f32>, GateError> {
        let file = open_split(Arch::Qwen35moe, RECIPE)?;
        let (shard, t) = file
            .find("token_embd.weight")
            .ok_or_else(|| GateError::from("no token_embd.weight".to_string()))?;
        let data = file
            .shard(shard)
            .ok_or_else(|| GateError::from("no shard of token_embd.weight".to_string()))?
            .data(t)
            .map_err(|e| format!("token_embd.weight: {e}"))?;
        let k = usize::try_from(t.dims[0])?;
        let blck =
            t.ty.blck_size()
                .ok_or_else(|| GateError::from("token_embd.weight: no block size".to_string()))?
                as usize;
        let tsz =
            t.ty.type_size()
                .ok_or_else(|| GateError::from("token_embd.weight: no type size".to_string()))?
                as usize;
        let row_bytes = k / blck * tsz;
        let mut out = vec![0f32; ids.len() * k];
        for (&id, row) in ids.iter().zip(out.chunks_mut(k)) {
            let b = id as usize * row_bytes;
            dequant_row(t.ty, &data[b..b + row_bytes], row)
                .map_err(|e| format!("token_embd.weight row {id}: {e:?}"))?;
        }
        Ok(out)
    }

    /// Whether `rows`, the `ids`' tapped rows (`width` values, tap slot 0
    /// first), hold the embedding of each id at slot 0 bit for bit.
    fn slot0_is_embedding(
        rows: &[f32],
        ids: &[u32],
        (width, hidden): (usize, usize),
    ) -> Result<bool, GateError> {
        let emb = embedding_rows(ids)?;
        let slot0: Vec<f32> = rows
            .chunks(width)
            .flat_map(|r| r[..hidden].to_vec())
            .collect();
        Ok(same_bits(&slot0, &emb))
    }

    /// (td) (module doc).
    fn taps_verify(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let (width, hidden) = (s.tap_width(), s.model().body(WHAT)?.hidden());
        let r = &refs.a.0;
        let mut ok = width == TAP_LAYERS.len() * hidden;
        for mode in [StepMode::Graph, StepMode::Eager] {
            s.model_mut().set_mode(mode);
            s.reset()?;
            s.prompt(&refs.prompt_a, Want::Argmax)?;
            let mut steps = Vec::with_capacity(8 * width);
            for &t in &r[..8] {
                s.step(t, Want::Argmax)?;
                steps.extend_from_slice(s.taps(1)?);
            }
            let past_step = s.taps(2).err().map(|e| e.to_string()).unwrap_or_default();
            s.reset()?;
            s.prompt(&refs.prompt_a, Want::Argmax)?;
            let out = s.verify::<8>(std::array::from_fn(|j| r[j]))?;
            let rows = s.taps(8)?.to_vec();
            let past_verify = s.taps(9).err().map(|e| e.to_string()).unwrap_or_default();
            s.commit(8)?;
            let ids = out[..] == r[1..9];
            let diff = first_diff(&rows, &steps, (width, hidden));
            let refused = past_step.contains("last walk") && past_verify.contains("last walk");
            println!(
                "(td) {mode:?}: the taps of an 8-row verify of forced rows ({} floats, layers \
                 {TAP_LAYERS:?}) equal 8 steps' bit for bit {} [{}]; the rows are live {}; rows \
                 past the last walk are refused by name ({past_step} / {past_verify}) {}",
                rows.len(),
                diff.is_none() && ids,
                diff.as_deref().unwrap_or("equal"),
                live(&rows, hidden),
                verdict(diff.is_none() && ids && live(&rows, hidden) && refused)
            );
            ok &= diff.is_none() && ids && live(&rows, hidden) && refused;
        }
        s.model_mut().set_mode(StepMode::Graph);
        Ok(ok)
    }

    /// One tapped prompt call against the plain call: the argmax, the last
    /// logits and the next 16 steps' ids and logits bit for bit (te), and
    /// the sink's calls (tf): `calls` ends at the call's last position, in
    /// order, with no gap or repeat, from the window's first position, and
    /// `expect` when the units are derived. `steps` are the step run's
    /// rows of positions `steps_from..`, which the sink's rows equal at
    /// every tap slot (`all_slots`) or at slot 0 alone.
    fn tapped_case(
        s: &mut Seat,
        name: &str,
        (prompt, window): (&[u32], usize),
        plain: &Plain,
        steps: (&[f32], usize, bool),
        expect: Option<&[(u32, usize)]>,
        (width, hidden): (usize, usize),
    ) -> Result<bool, GateError> {
        let ((ids, logits), calls) = greedy_tapped(s, prompt, STEPS_AFTER + 1, window)?;
        let ids_ok = ids == plain.0;
        let bits_ok = ids_ok
            && logits.len() == plain.1.len()
            && logits.iter().zip(&plain.1).all(|(a, b)| same_bits(a, b));
        // (tf): the window.
        let n = prompt.len() as u32;
        let held = n - window.min(prompt.len()) as u32;
        let shape: Vec<(u32, usize)> = calls.iter().map(|(at, r)| (*at, r.len() / width)).collect();
        let mut next = held;
        let mut tiled = calls.iter().all(|(_, r)| r.len() % width == 0);
        for &(at, rows) in &shape {
            tiled &= at == next;
            next = at + rows as u32;
        }
        tiled &= if window == 0 {
            calls.is_empty()
        } else {
            next == n
        };
        let derived = expect.is_none_or(|e| e == shape);
        let (rows0, from, all_slots) = steps;
        let mut equal = true;
        let mut note = String::from("equal");
        for (at, rows) in &calls {
            let first = *at as usize - from;
            let want = &rows0[first * width..first * width + rows.len()];
            if all_slots {
                if let Some(d) = first_diff(rows, want, (width, hidden)) {
                    equal = false;
                    note = format!("positions from {at}: {d}");
                    break;
                }
            } else {
                let a: Vec<f32> = rows
                    .chunks(width)
                    .flat_map(|r| r[..hidden].to_vec())
                    .collect();
                let b: Vec<f32> = want
                    .chunks(width)
                    .flat_map(|r| r[..hidden].to_vec())
                    .collect();
                if let Some(d) = first_diff(&a, &b, (hidden, hidden)) {
                    equal = false;
                    note = format!("slot 0, positions from {at}: {d}");
                    break;
                }
            }
        }
        let alive = calls.iter().all(|(_, r)| live(r, hidden));
        println!(
            "(te) {name}: argmax, last logits and {STEPS_AFTER} steps equal the plain call's: ids \
             {ids_ok}, logits bit for bit {bits_ok} {}",
            verdict(ids_ok && bits_ok)
        );
        println!(
            "(tf) {name}: window {window}: sink calls {shape:?} tile {held}..{n} in order {tiled}, \
             the derived units {derived}; the rows equal the step run's at {} [{note}], live \
             {alive} {}",
            if all_slots { "every slot" } else { "slot 0" },
            verdict(tiled && derived && equal && alive)
        );
        Ok(ids_ok && bits_ok && tiled && derived && equal && alive)
    }

    /// (te) and (tf) (module doc).
    fn taps_prompt(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let hidden = s.model().body(WHAT)?.hidden();
        let width = TAP_LAYERS.len() * hidden;
        let short = &refs.prompt_a[..MAX_PASS_ROWS];
        // The references with the taps off: the plain calls.
        s.set_taps(&[])?;
        let plain_short = greedy(s, short, STEPS_AFTER + 1, true)?;
        let plain_wide = greedy(s, &refs.prompt_d, STEPS_AFTER + 1, true)?;
        s.set_taps(&TAP_LAYERS)?;
        let mut ok = s.tap_width() == width;
        // The step runs the sink's rows equal.
        let (steps_short, _) = step_taps(s, short, 0)?;
        let (steps_wide, _) = step_taps(s, &refs.prompt_d, 210 - WINDOW_WIDE)?;
        let (steps_a, last_a) = step_taps(s, &refs.prompt_a, PROMPT_A - WINDOW_PASSES)?;
        for window in WINDOWS_SHORT {
            ok &= tapped_case(
                s,
                &format!("the {MAX_PASS_ROWS}-id prompt (a pass)"),
                (short, window),
                &plain_short,
                (&steps_short, 0, true),
                None,
                (width, hidden),
            )?;
        }
        ok &= tapped_case(
            s,
            &format!("the {}-id prompt (wide units)", refs.prompt_d.len()),
            (&refs.prompt_d, WINDOW_WIDE),
            &plain_wide,
            (&steps_wide, 210 - WINDOW_WIDE, false),
            Some(&CALLS_WIDE),
            (width, hidden),
        )?;
        // The wide path's rows have no bit-equal reference past slot 0 (the
        // embedding is the same bits on every path): their distance from the
        // step run's is printed.
        let ((_, _), wide_calls) = greedy_tapped(s, &refs.prompt_d, 1, WINDOW_WIDE)?;
        let wide_rows: Vec<f32> = wide_calls.iter().flat_map(|(_, r)| r.clone()).collect();
        let (mut worst, mut scale) = (0f32, 0f32);
        for (a, b) in wide_rows.iter().zip(&steps_wide) {
            worst = worst.max((a - b).abs());
            scale = scale.max(b.abs());
        }
        println!(
            "(tf) the wide path's rows against the step run's, every slot (no bit-equal reference, \
             information): largest |difference| {worst:e} of the largest |value| {scale:e}"
        );
        // Several pass units: a model-level call by the pass path.
        let mut calls = Vec::new();
        s.reset()?;
        let argmax = s.model_mut().prefill_tapped(
            &refs.prompt_a,
            PrefillPath::Pass,
            WINDOW_PASSES,
            &mut |at, rows| {
                calls.push((at, rows.to_vec()));
                Ok(())
            },
        )?;
        let shape: Vec<(u32, usize)> = calls.iter().map(|(at, r)| (*at, r.len() / width)).collect();
        let rows: Vec<f32> = calls.iter().flat_map(|(_, r)| r.clone()).collect();
        let diff = first_diff(&rows, &steps_a, (width, hidden));
        let passes = shape == CALLS_PASSES && argmax == last_a && diff.is_none();
        println!(
            "(tf) {PROMPT_A} ids by passes, window {WINDOW_PASSES}: sink calls {shape:?} (derived \
             {CALLS_PASSES:?}), the rows equal the {PROMPT_A} steps' at every slot bit for bit \
             [{}], the argmax {argmax} the step run's {last_a} {}",
            diff.as_deref().unwrap_or("equal"),
            verdict(passes)
        );
        ok &= passes;
        // A sink that refuses ends the call with its own error.
        s.reset()?;
        let refused = s
            .prompt_tapped(short, 1, &mut |_, _| {
                Err(SessionError::Refused("the sink says no".into()))
            })
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let sink = refused.contains("the sink says no");
        println!("(tf) a sink that refuses: {refused} {}", verdict(sink));
        s.reset()?;
        Ok(ok && sink)
    }

    /// (tg) (module doc).
    fn taps_anchor(s: &mut Seat, refs: &Refs) -> Result<bool, GateError> {
        let hidden = s.model().body(WHAT)?.hidden();
        let width = TAP_LAYERS.len() * hidden;
        let mut ok = true;
        // Layer 0: the token's embedding row, on the step path and on the
        // wide path.
        let short = &refs.prompt_a[..MAX_PASS_ROWS];
        let (steps, _) = step_taps(s, short, 0)?;
        let step_emb = slot0_is_embedding(&steps, short, (width, hidden))?;
        let (_, calls) = greedy_tapped(s, &refs.prompt_d, 1, WINDOW_WIDE)?;
        let wide_rows: Vec<f32> = calls.iter().flat_map(|(_, r)| r.clone()).collect();
        let wide_ids = &refs.prompt_d[210 - WINDOW_WIDE..];
        let wide_emb = slot0_is_embedding(&wide_rows, wide_ids, (width, hidden))?;
        println!(
            "(tg) the row entering layer 0 is the token's embedding row (token_embd.weight, \
             dequantized on the host), bit for bit: {} step rows {step_emb}, {} wide-path rows \
             {wide_emb} {}",
            short.len(),
            wide_ids.len(),
            verdict(step_emb && wide_emb)
        );
        ok &= step_emb && wide_emb;
        // Layer L > 0: the gate instrument's output of layer L − 1.
        s.model_mut().set_layer_taps(true)?;
        s.reset()?;
        let (mut compared, mut diff) = (0, None);
        for &t in &refs.prompt_a[..4] {
            s.step(t, Want::Argmax)?;
            let row = s.taps(1)?.to_vec();
            let outs = s.model_mut().layer_taps()?;
            for (k, &l) in TAP_LAYERS.iter().enumerate().filter(|&(_, &l)| l > 0) {
                compared += 1;
                if diff.is_none() {
                    diff = first_diff(
                        &row[k * hidden..(k + 1) * hidden],
                        &outs[l - 1],
                        (hidden, hidden),
                    )
                    .map(|d| format!("layer {l}: {d}"));
                }
            }
        }
        // The instrument still refuses a prompt unit by name, and so does the
        // tapped prompt.
        let plain = s
            .prompt(&refs.prompt_a, Want::Argmax)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let tapped = s
            .prompt_tapped(&refs.prompt_a, 4, &mut |_, _| Ok(()))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let refusal = plain.contains("layer taps off") && tapped.contains("layer taps off");
        println!(
            "(tg) the row entering layer L > 0 equals the instrument's output tap of layer L − 1 \
             ({compared} row-layers of 4 steps, layers {TAP_LAYERS:?}) bit for bit [{}] {}; the \
             instrument refuses a prompt unit and a tapped prompt by name ({plain} / {tapped}) {}",
            diff.as_deref().unwrap_or("equal"),
            verdict(diff.is_none() && compared > 0),
            verdict(refusal)
        );
        s.model_mut().set_layer_taps(false)?;
        s.model_mut().set_mode(StepMode::Graph);
        s.reset()?;
        Ok(ok && diff.is_none() && compared > 0 && refusal)
    }

    /// (th) (module doc): the captured graphs of the tapped load hold one
    /// copy node per tapped layer more than the untapped counts, and the
    /// untapped counts come back.
    fn taps_nodes(s: &mut Seat) -> Result<bool, GateError> {
        let n = TAP_LAYERS.len();
        let on = facts(s)?;
        s.set_taps(&[])?;
        let off = facts(s)?;
        s.set_taps(&TAP_LAYERS)?;
        let on_ok = on.decode == NODES_DECODE + n
            && on.pass5 == NODES_PASS + 4 * 5 + n
            && on.pass8 == NODES_PASS + 4 * 8 + n;
        let off_ok = off.decode == NODES_DECODE
            && off.pass5 == NODES_PASS + 4 * 5
            && off.pass8 == NODES_PASS + 4 * 8;
        println!(
            "(th) graph nodes with {n} taps: decode {} pass5 {} pass8 {} (want {}, {}, {}: the \
             untapped counts and one copy a tapped layer) {}; taps off again: {} {} {} (want \
             {NODES_DECODE}, {}, {}) {}",
            on.decode,
            on.pass5,
            on.pass8,
            NODES_DECODE + n,
            NODES_PASS + 20 + n,
            NODES_PASS + 32 + n,
            verdict(on_ok),
            off.decode,
            off.pass5,
            off.pass8,
            NODES_PASS + 20,
            NODES_PASS + 32,
            verdict(off_ok)
        );
        Ok(on_ok && off_ok)
    }

    /// (ti) (module doc): taps refused by name, and a refused call leaves
    /// them off.
    fn taps_refusals(s: &mut Seat) -> Result<bool, GateError> {
        let mut said = |layers: &[usize]| -> String {
            s.set_taps(layers)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        };
        let past = said(&[0, N_LAYER]);
        let order = said(&[3, 3]);
        let down = said(&[5, 2]);
        let named = past.contains("layer") && order.contains("rising") && down.contains("rising");
        let off = s.tap_width() == 0;
        println!(
            "(ti) taps past the {N_LAYER} layers ({past}), repeated ({order}) and falling \
             ({down}) are refused by name, and the refused call leaves taps off {} {}",
            verdict(named),
            verdict(off)
        );
        let wrong_rows = s.taps(1).err().map(|e| e.to_string()).unwrap_or_default();
        let no_taps = wrong_rows.contains("taps set");
        println!(
            "(ti) a read with no taps set: {wrong_rows} {}",
            verdict(no_taps)
        );
        s.set_taps(&TAP_LAYERS)?;
        Ok(named && off && no_taps)
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
        let r = unread_fault(&mut s, &refs);
        ok &= clause("(f)", &mut s, r);
        let r = close_fold(&mut s, &refs);
        ok &= clause("(h)", &mut s, r);
        // The taps clauses: Qwen3.6 as a block draft's target.
        s.set_taps(&TAP_LAYERS)?;
        let r = taps_nodes(&mut s);
        ok &= clause("(th)", &mut s, r);
        let r = taps_verify(&mut s, &refs);
        ok &= clause("(td)", &mut s, r);
        let r = taps_prompt(&mut s, &refs);
        ok &= clause("(te)/(tf)", &mut s, r);
        let r = taps_anchor(&mut s, &refs);
        ok &= clause("(tg)", &mut s, r);
        let r = taps_refusals(&mut s);
        ok &= clause("(ti)", &mut s, r);
        // Last: the second slot stays resident.
        let r = second_slot(&mut s, &refs);
        ok &= clause("(g)", &mut s, r);
        println!("{WHAT}: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
