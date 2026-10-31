//! The clauses every Body35 gate reads off the model itself, with no oracle set: the stores of a run read back and
//! compared bit for bit (`PathRun`, `paths`), the checkpoint cut and its re-fed tail, the file's geometry from its
//! header, and the resident-slot adapter with its pass refusals. `gate_qwen35moe_e2e` (Qwen3.6, MoE) and
//! `gate_qwen35_dense` (the dense 27B) share them; a model's own constants enter through [`Geometry35`] and the load
//! closure of [`Slots35`], nothing else.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::{Body35, PrefillPath, Qwen35moeModel, StoreHost};
use bloomery_gpu::linear::RING_ROWS;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu_gates::slots_gate::{self, Derived, Launches, PassAdapter, SlotsAdapter};
use bloomery_gpu_gates::{Fnv1a64, GateError, bits_equal, verdict};
use gguf::Split;
use model::arch::models::Mixer;
use model::arch::models::shape::AttnShape;

/// The shape facts a Body35 file's stores and checkpoints are derived from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry35 {
    pub n_attn: usize,
    pub n_delta: usize,
    pub n_kv: usize,
    pub head: usize,
    pub n_v: usize,
    pub head_v: usize,
    /// The conv ring's row width: `(2 · k_heads + v_heads) · head_v`.
    pub conv_c: usize,
}

impl Geometry35 {
    /// The geometry `split`'s header gives: every attention layer's heads and every delta layer's state shape must
    /// agree, else the file is refused by name here, before a derivation is written against a mixed one. A trunk of
    /// attention layers alone is legal: its description holds no delta layer, and the delta fields are zero.
    #[allow(
        dead_code,
        reason = "the dense gate reads its geometry off the header; the MoE gate states its constants"
    )]
    pub fn of(split: &Split) -> Result<Geometry35, GateError> {
        let spec = model::arch::spec(split)
            .map_err(|e| format!("the header: {e}"))?
            .spec;
        let (mut attn, mut delta): (Option<[usize; 3]>, Option<[usize; 3]>) = (None, None);
        let (mut n_attn, mut n_delta) = (0, 0);
        for (l, layer) in spec.layers.iter().enumerate() {
            let (slot, got, n) = match &layer.mixer {
                Mixer::Gqa(g) => {
                    let a = AttnShape::of(g);
                    (
                        &mut attn,
                        [a.n_kv as usize, a.head as usize, 0],
                        &mut n_attn,
                    )
                }
                Mixer::DeltaRule(d) => (
                    &mut delta,
                    [
                        d.v_heads as usize,
                        d.d as usize,
                        ((2 * d.k_heads + d.v_heads) * d.d) as usize,
                    ],
                    &mut n_delta,
                ),
                _ => return Err(format!("layer {l}: neither a GQA nor a delta rule").into()),
            };
            if slot.is_some_and(|seen| seen != got) {
                return Err(
                    format!("layer {l}: a shape {got:?} unlike the layers before it").into(),
                );
            }
            *slot = Some(got);
            *n += 1;
        }
        let Some([n_kv, head, _]) = attn else {
            return Err("the file holds no attention layer".into());
        };
        let [n_v, head_v, conv_c] = delta.unwrap_or([0; 3]);
        Ok(Geometry35 {
            n_attn,
            n_delta,
            n_kv,
            head,
            n_v,
            head_v,
            conv_c,
        })
    }

    /// One delta layer's state and conv ring, f32 values.
    fn delta_values(&self) -> usize {
        self.n_v * self.head_v * self.head_v + RING_ROWS * self.conv_c
    }

    /// One checkpoint's bytes: every delta layer's state and ring.
    pub fn checkpoint_bytes(&self) -> usize {
        self.n_delta * self.delta_values() * 4
    }

    /// The layer stores' bytes at `ctx` rows: each attention layer's K and V planes (`2 · n_kv · ctx · head` f16) and
    /// each delta layer's state and ring.
    pub fn store_bytes(&self, ctx: usize) -> usize {
        self.n_attn * 2 * self.n_kv * ctx * self.head * 2 + self.checkpoint_bytes()
    }
}

/// `n` ids of an lcg over the vocabulary: every id a row of the embedding.
pub fn lcg_ids(n: usize, vocab: usize) -> Vec<u32> {
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

/// Every layer's store, read back.
pub fn stores(m: &mut Qwen35moeModel) -> Result<Vec<StoreHost>, GateError> {
    let n = m.layers().len();
    let (gpu, _, body) = m.body_parts("stores")?;
    Ok((0..n)
        .map(|l| body.store(gpu, l))
        .collect::<Result<_, GpuError>>()?)
}

pub fn same_store(a: &StoreHost, b: &StoreHost) -> bool {
    match (a, b) {
        (StoreHost::Kv { k: ka, v: va }, StoreHost::Kv { k: kb, v: vb }) => ka == kb && va == vb,
        (
            StoreHost::Rec {
                state: sa,
                ring: ra,
            },
            StoreHost::Rec {
                state: sb,
                ring: rb,
            },
        ) => bits_equal(sa, sb) && bits_equal(ra, rb),
        _ => false,
    }
}

/// What a run of tokens left: each position's logits and token, and every layer's store.
pub struct PathRun {
    pub logits: Vec<Vec<f32>>,
    pub tokens: Vec<u32>,
    pub stores: Vec<StoreHost>,
}

/// `reset`, then every layer's store written to zeros through `set_store`: a start from zero state that does not rest
/// on `reset`'s own clearing, so only the clause that tests `reset` reads it.
pub fn fresh(m: &mut Qwen35moeModel) -> Result<(), GateError> {
    m.reset()?;
    let n = m.layers().len();
    let (gpu, _, body) = m.body_parts("fresh")?;
    for l in 0..n {
        let zero = match body.store(gpu, l)? {
            StoreHost::Kv { k, v } => StoreHost::Kv {
                k: vec![0; k.len()],
                v: vec![0; v.len()],
            },
            StoreHost::Rec { state, ring } => StoreHost::Rec {
                state: vec![0.0; state.len()],
                ring: vec![0.0; ring.len()],
            },
        };
        body.set_store(gpu, l, &zero)?;
    }
    Ok(())
}

/// `toks` as decode steps from position 0: from [`fresh`]'s zero stores, or (`by_reset`) from `reset` alone.
pub fn decode_run(
    m: &mut Qwen35moeModel,
    toks: &[u32],
    by_reset: bool,
) -> Result<PathRun, GateError> {
    if by_reset {
        m.reset()?;
    } else {
        fresh(m)?;
    }
    let (mut logits, mut tokens) = (Vec::new(), Vec::new());
    for &t in toks {
        tokens.push(m.step(&[t])?);
        logits.push(m.logits()?);
    }
    Ok(PathRun {
        logits,
        tokens,
        stores: stores(m)?,
    })
}

pub fn same_run(label: &str, got: &PathRun, want: &PathRun) -> bool {
    let logits = got.logits.len() == want.logits.len()
        && got
            .logits
            .iter()
            .zip(&want.logits)
            .all(|(a, b)| bits_equal(a, b));
    let stores_same = got.stores.len() == want.stores.len()
        && got
            .stores
            .iter()
            .zip(&want.stores)
            .all(|(a, b)| same_store(a, b));
    let differ: Vec<usize> = (0..got.stores.len().min(want.stores.len()))
        .filter(|&l| !same_store(&got.stores[l], &want.stores[l]))
        .collect();
    let ok = logits && stores_same && got.tokens == want.tokens;
    println!(
        "paths {label}: tokens {:?} vs {:?}, every position's logits bit-identical={logits}, \
         every store bit-identical={stores_same} (layers differing {differ:?}) {}",
        got.tokens,
        want.tokens,
        verdict(ok)
    );
    ok
}

/// Five tokens: graph decode steps, one pass of five rows, eager steps, and the same five steps from `reset` alone
/// after the eager run's state, each from [`fresh`]'s zero stores, leave the same per-position logits and tokens and
/// every layer's store bit for bit. The decode run is handed back.
pub fn paths(m: &mut Qwen35moeModel, toks: &[u32]) -> Result<(bool, PathRun), GateError> {
    let toks5: [u32; 5] = toks
        .try_into()
        .map_err(|_| format!("the batch holds {} tokens, want 5", toks.len()))?;
    m.set_mode(StepMode::Graph);
    let decode = decode_run(m, toks, false)?;
    // One pass of five rows.
    fresh(m)?;
    let tokens = m.step_rows::<5>(toks5)?.to_vec();
    let pass = PathRun {
        logits: m.rows_logits::<5>()?.to_vec(),
        tokens,
        stores: stores(m)?,
    };
    let mut ok = same_run("pass m=5 vs five decode steps (graph)", &pass, &decode);
    m.set_mode(StepMode::Eager);
    let eager = decode_run(m, toks, false)?;
    ok &= same_run("five eager steps vs five graph replays", &eager, &decode);
    m.set_mode(StepMode::Graph);
    // The only run that starts from `reset` alone, after the eager run left its state: `reset` must clear it.
    let again = decode_run(m, toks, true)?;
    ok &= same_run("after reset alone, the same five steps", &again, &decode);
    Ok((ok, decode))
}

/// What an uncut prompt call left, the reference of a cut and re-fed tail.
pub struct Uncut {
    pub tok: u32,
    pub logits: Vec<f32>,
    pub stores: Vec<StoreHost>,
}

/// `ids` as one prompt call on the seat's schedule from [`fresh`]'s zero stores.
pub fn uncut_call(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<Uncut, GateError> {
    fresh(m)?;
    let tok = m.prefill_with(ids, PrefillPath::Wide)?;
    Ok(Uncut {
        tok,
        logits: m.logits()?,
        stores: stores(m)?,
    })
}

/// A cut to the checkpoint at `cut_at` and the rest of `ids` re-fed leave the token, the last logits and every store
/// bit for bit the uncut call's, the points back at `want_pts`, and one walk of the tail's own plan.
pub fn refed_after_cut(
    m: &mut Qwen35moeModel,
    ids: &[u32],
    cut_at: u32,
    uncut: &Uncut,
    want_pts: &[u32],
    ub: u32,
) -> Result<bool, GateError> {
    m.rollback(cut_at)?;
    let tail = &ids[cut_at as usize..];
    let tok = m.prefill_with(tail, PrefillPath::Wide)?;
    let (logits, st) = (m.logits()?, stores(m)?);
    let differ: Vec<usize> = (0..st.len())
        .filter(|&l| !uncut.stores.get(l).is_some_and(|w| same_store(&st[l], w)))
        .collect();
    let same_logits = bits_equal(&logits, &uncut.logits);
    let body = m.body("refed_after_cut")?;
    let refed_walks = body.prompt_walks();
    let refed_ok = tok == uncut.tok
        && same_logits
        && differ.is_empty()
        && st.len() == uncut.stores.len()
        && body.checkpoints().positions() == want_pts
        // A cut lands on a mark, so the re-fed call's only run is the tail itself: one walk of its own plan.
        && refed_walks == (tail.len() as u32).div_ceil(ub) as usize;
    println!(
        "checkpoints: cut to {cut_at} and {} ids re-fed: token {tok} vs {}, last logits \
         bit-identical={same_logits}, every store bit-identical (layers differing {differ:?}), \
         the points back at {want_pts:?}, {refed_walks} walk {}",
        tail.len(),
        uncut.tok,
        verdict(refed_ok)
    );
    Ok(refed_ok)
}

/// The nodes of a pass's one head of `rows` rows over the file's Q6_K
/// lm_head: the copy of the rows into its input, then the fused head's
/// three at one row, or the norm, the quantizer, the gemv and the rows'
/// argmax past it.
fn pass_head_nodes(rows: usize) -> usize {
    1 + if rows == 1 { 3 } else { 4 }
}

/// A call that must be refused by `what` with a detail holding `phrase`.
pub fn refused<T>(r: Result<T, GpuError>, what: &str, phrase: &str) -> bool {
    match r {
        Err(GpuError::Shape { what: w, detail }) => w == what && detail.contains(phrase),
        Err(GpuError::State { what: w, missing }) => w == what && missing.contains(phrase),
        _ => false,
    }
}

/// Greedy steps a slot stream takes past its prompt in the interleave, and of one continuation check: each solo run
/// steps `SLOT_STEPS + 3 · SLOT_TAIL` (`slots_gate`'s `solo`).
// PIN(2026-10-05): the tail 8 -> 5, as the harness's solo run grew to STEPS + 3·TAIL: B's 1056 + 16 + 3·8 stepped past
// the 1088-row cache of the MoE gate, and 1056 + 16 + 3·5 = 1087 fits.
pub const SLOT_STEPS: usize = 16;
pub const SLOT_TAIL: usize = 5;

/// The load a slots clause opens: the whole-card model at its context with the checkpoints armed as the seat arms them.
pub type Load = Box<dyn Fn() -> Result<Qwen35moeModel, GateError>>;

/// Body35's resident-slot adapter: two streams of ids over one load, A then B.
pub struct Slots35 {
    /// Stream 0's ids (A), then stream 1's (B).
    pub ids: Vec<u32>,
    /// A's length.
    pub a_len: usize,
    /// The load's first mark (its ubatch): each stream is past it by a run the mark keeps.
    pub mark: usize,
    /// The context the load was opened at.
    pub ctx: usize,
    pub geom: Geometry35,
    /// The sequence-bound launches one more busy slot adds to a captured pass (a delta layer's conv and delta, an
    /// attention layer's rope-append, flash segment pass and merge, and the embedding).
    pub nodes_slot: usize,
    pub load: Load,
}

impl Slots35 {
    /// The streams must fit the cache with the harness's steps, and sit past the first mark by a run it keeps (a run
    /// of fewer than `WIDE_FROM` = 9 rows beside a mark drops the mark).
    pub fn new(
        ids: Vec<u32>,
        a_len: usize,
        (mark, ctx): (usize, usize),
        shape: (Geometry35, usize),
        load: Load,
    ) -> Result<Slots35, GateError> {
        let b_len = ids.len().saturating_sub(a_len);
        let reach = SLOT_STEPS + 3 * SLOT_TAIL;
        if a_len < mark + 9 || b_len < mark + 9 || a_len + reach > ctx || b_len + reach > ctx {
            return Err(format!(
                "slot streams of {a_len} and {b_len} ids at mark {mark}, cache {ctx}: each must pass the mark by 9 \
                 rows and fit {reach} steps"
            )
            .into());
        }
        Ok(Slots35 {
            ids,
            a_len,
            mark,
            ctx,
            geom: shape.0,
            nodes_slot: shape.1,
            load,
        })
    }

    /// Stream `stream`'s ids, else refused by name.
    pub fn ids(&self, stream: usize) -> Result<&[u32], GateError> {
        let (a, b) = self.ids.split_at(self.a_len);
        match stream {
            0 => Ok(a),
            1 => Ok(b),
            s => Err(format!("stream {s} of the adapter's two").into()),
        }
    }
}

impl SlotsAdapter for Slots35 {
    type Body = Body35;

    const STEPS: usize = SLOT_STEPS;
    const TAIL: usize = SLOT_TAIL;

    /// The load, the checkpoints armed. It is the same load at every slot count: its fit check counts no slot, and each
    /// sequence's stores are allocated by `add_slots`.
    fn open(&self, _slots: usize) -> Result<Qwen35moeModel, GateError> {
        let mut m = (self.load)()?;
        m.body_parts("slots")?.2.set_checkpoints(true);
        Ok(m)
    }

    /// `reset`: every row `state_hash` reads is written by a call from it.
    fn rewind(&self, m: &mut Qwen35moeModel) -> Result<(), GateError> {
        Ok(m.reset()?)
    }

    /// The session's schedule (`app::Prompt for Body35`): the wide walk from `GEMM_FROM` ids on, passes below.
    fn prompt(&self, m: &mut Qwen35moeModel, stream: usize) -> Result<u32, GateError> {
        Ok(<Body35 as app::Prompt>::prompt(m, self.ids(stream)?)?)
    }

    /// Every attention layer's K and V rows below the model's position — rows past it are no state: the flash never
    /// reads them, and a slot rewound after a longer sequence still holds that sequence's there — and every delta
    /// layer's whole state and conv ring.
    fn state_hash(&self, m: &mut Qwen35moeModel) -> Result<u64, GateError> {
        let body = m.body("slots")?;
        let rows = usize::try_from(m.pos())? * self.geom.head;
        let plane = body.ctx_rows() * self.geom.head;
        let mut h = Fnv1a64::default();
        for l in m.layers() {
            h = match body.store(m.gpu(), l)? {
                StoreHost::Kv { k, v } => {
                    k.chunks(plane)
                        .chain(v.chunks(plane))
                        .try_fold(h, |h, head| {
                            let below = head.get(..rows).ok_or("a position past the cache rows")?;
                            Ok::<_, GateError>(
                                below.iter().fold(h, |h, w| h.bytes(&w.to_le_bytes())),
                            )
                        })?
                }
                StoreHost::Rec { state, ring } => h.f32s(&state).f32s(&ring),
            };
        }
        Ok(h.value())
    }

    /// The stores' derivation from the header ([`Geometry35::store_bytes`]).
    fn seq_bytes_derived(&self, _m: &Qwen35moeModel) -> Result<Derived, GateError> {
        let g = &self.geom;
        Ok(Derived {
            bytes: g.store_bytes(self.ctx),
            terms: format!(
                "{} x K/V planes of 2 x {} x {} x {} f16 + {} x (state {} x {} x {} + ring {RING_ROWS} x {}) f32",
                g.n_attn, g.n_kv, self.ctx, g.head, g.n_delta, g.n_v, g.head_v, g.head_v, g.conv_c
            ),
        })
    }

    /// H5's planter: the tier through the body's placed side, the host service of this body; `false` on a load with
    /// no placed side, which serves no host work.
    fn plant_refusal(&self, m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let Some(placed) = m.body_parts("q35_clauses slots")?.2.host() else {
            return Ok(false);
        };
        placed
            .hybrid_mut()
            .plant_refusal("a planted refusal (the slots harness's seam)");
        Ok(true)
    }

    /// H5's window: the tier's own refusal, read through the body's placed side.
    fn tier_poisoned(&self, m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let placed = m
            .body_parts("q35_clauses slots")?
            .2
            .host()
            .ok_or("the placed side (the host tier)")?;
        Ok(placed.hybrid().refuse_if_poisoned("slots H5").is_err())
    }
}

impl PassAdapter for Slots35 {
    fn added_slot_launches(&self, _m: &Qwen35moeModel) -> Result<Launches, GateError> {
        Ok(Launches {
            n: self.nodes_slot,
            terms: format!(
                "the embedding 1 + {} attention layers' flash segment pass and merge 2",
                self.geom.n_attn
            ),
        })
    }

    /// The body's count without the head (`Body35::slots_launches`), plus the pass's one head.
    fn slots_launches(
        &self,
        m: &Qwen35moeModel,
        key: &[(usize, usize)],
    ) -> Result<usize, GateError> {
        let rows = key.iter().map(|&(_, r)| r).sum();
        Ok(m.body("slots")?.slots_launches(key) + pass_head_nodes(rows))
    }
}

/// The pass refusals: each pass refused by name before anything runs, every slot's position kept and no slot
/// poisoned. `last` is each slot's last id, its next step's input.
fn pass_refusals(m: &mut Qwen35moeModel, last: [u32; 2]) -> Result<bool, GateError> {
    const COMMON: &str = "GpuModel::step_slots";
    const BODY: &str = "qwen35moe slots";
    let positions = |m: &mut Qwen35moeModel| -> Result<Vec<u32>, GateError> {
        (0..2)
            .map(|s| {
                m.select_slot(s)?;
                Ok(m.pos())
            })
            .collect()
    };
    let before = positions(m)?;
    let vocab = u32::try_from(m.body("slots")?.vocab())?;
    let five = [last[0]; 5];
    let four = [last[1]; 4];
    let arms = [
        (
            "nine rows",
            refused(
                m.step_slots(&[(0, &five[..]), (1, &four[..])]),
                COMMON,
                "9 rows in one pass",
            ),
        ),
        (
            "a slot twice",
            refused(
                m.step_slots(&[(0, &[last[0]]), (0, &[last[0]])]),
                COMMON,
                "slot 0 twice",
            ),
        ),
        (
            "a slot out of range",
            refused(
                m.step_slots(&[(2, &[last[0]])]),
                COMMON,
                "slot 2 of a model that serves 0..2",
            ),
        ),
        (
            "a slot of no token",
            refused(m.step_slots(&[(1, &[])]), COMMON, "slot 1 with no token"),
        ),
        (
            "an id past the vocabulary",
            refused(
                m.step_slots(&[(0, &[last[0]]), (1, &[vocab])]),
                BODY,
                "slot 1:",
            ),
        ),
    ];
    let taps = {
        m.set_layer_taps(true)?;
        let named = refused(
            m.step_slots(&[(0, &[last[0]]), (1, &[last[1]])]),
            BODY,
            "layer taps off",
        );
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        named
    };
    let missed: Vec<&str> = arms
        .iter()
        .filter(|(_, named)| !named)
        .map(|(arm, _)| *arm)
        .chain((!taps).then_some("the layer taps armed"))
        .collect();
    let after = positions(m)?;
    let kept = after == before;
    let pass = missed.is_empty() && kept;
    println!(
        "slots pass refusals: nine rows, a slot twice, a slot out of range, a slot of no token, an id past the \
         vocabulary and the layer taps armed each refused by name{}; positions {after:?} kept {kept} {}",
        if missed.is_empty() {
            String::new()
        } else {
            format!(" — not named: {}", missed.join(", "))
        },
        verdict(pass)
    );
    Ok(pass)
}

/// The selected slot's checkpoints' positions.
fn points(m: &Qwen35moeModel) -> Result<Vec<u32>, GateError> {
    Ok(m.body("slots")?.checkpoints().positions())
}

/// The harness's contracts over `a`'s two streams, and between its halves each slot's checkpoints its own under a cut
/// and a reset of the other slot. The clause's model is dropped with its second sequence.
pub fn slots_two(a: &Slots35) -> Result<bool, GateError> {
    let mut s = slots_gate::interleave(a)?;
    s.one_pass();
    let mark = u32::try_from(a.mark)?;
    let ends = [
        u32::try_from(a.a_len)?,
        u32::try_from(a.ids.len() - a.a_len)?,
    ];
    let b_last = s.last(1)?;
    let last = [s.last(0)?, b_last];
    let refusals_ok = pass_refusals(s.model(), last)?;
    let m = s.model();
    // Each slot's points as its prompt in the interleave took them, and where each stands.
    m.select_slot(0)?;
    let a_held = (a.state_hash(m)?, points(m)?);
    m.select_slot(1)?;
    let b_held = (a.state_hash(m)?, points(m)?, m.pos());
    // A trunk with a delta layer takes the mark and each prompt's end; one without holds no store to copy, so no point.
    let (want_a, want_b, why) = if a.geom.n_delta > 0 {
        (
            vec![mark, ends[0]],
            vec![mark, ends[1]],
            "the mark and each prompt's end",
        )
    } else {
        (
            Vec::new(),
            Vec::new(),
            "none: no layer holds a store to copy",
        )
    };
    let taken = a_held.1 == want_a && b_held.1 == want_b;
    println!(
        "slots points: slot 0 {:?}, slot 1 {:?} (want {want_a:?} and {want_b:?}: {why}) {}",
        a_held.1,
        b_held.1,
        verdict(taken)
    );
    // Slot 1 cut back to the mark; slot 0 runs before slot 1 does again.
    m.rollback(mark)?;
    m.select_slot(0)?;
    let a_now = (a.state_hash(m)?, points(m)?);
    let cont = s.continues(0)?;
    let held_ok = a_now == a_held && cont;
    println!(
        "slots cut: slot 1 cut back to {mark}; slot 0's points {:?} and state digest {} as it stood, its next \
         {SLOT_TAIL} ids {} its solo run's continuation {}",
        a_now.1,
        if a_now.0 == a_held.0 {
            "equal"
        } else {
            "differ from"
        },
        if cont { "equal" } else { "differ from" },
        verdict(held_ok)
    );
    // Slot 1: the rest of B from the mark, then the interleave's steps.
    let m = s.model();
    m.select_slot(1)?;
    let rest = a
        .ids(1)?
        .get(a.mark..)
        .ok_or("stream B shorter than the first mark")?;
    let mut ids = vec![<Body35 as app::Prompt>::prompt(m, rest)?];
    slots_gate::greedy(m, &mut ids, SLOT_STEPS + SLOT_TAIL)?;
    let b_now = (a.state_hash(m)?, points(m)?, m.pos());
    let last = ids.last().copied();
    let refed_ok = last == Some(b_last) && b_now == b_held;
    println!(
        "slots cut: slot 1 re-fed {} ids from {mark} and stepped {} (the interleave's steps and H2's passes): last \
         id {last:?} vs {b_last}, state digest equal={}, points {:?} vs {:?}, position {} vs {} {}",
        rest.len(),
        SLOT_STEPS + SLOT_TAIL,
        b_now.0 == b_held.0,
        b_now.1,
        b_held.1,
        b_now.2,
        b_held.2,
        verdict(refed_ok)
    );
    // A reset of slot 1 leaves slot 0's points.
    a.rewind(m)?;
    m.select_slot(0)?;
    let kept = points(m)?;
    let kept_ok = kept == a_held.1;
    println!(
        "slots cut: after slot 1's reset slot 0's points {kept:?} (want {:?}) {}",
        a_held.1,
        verdict(kept_ok)
    );
    let harness_ok = s.finish()?;
    Ok(taken && held_ok && refed_ok && kept_ok && refusals_ok && harness_ok)
}
