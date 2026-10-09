//! The GLM-5.3-Flash NextN draft's gate: the target opened by its NextN plan
//! on the gate card with the next-token layer beside it
//! (`Body::open_placed_nextn`, `place::PlanInputs::plan_nextn`), the NextN
//! program's walks (`bloomery_gpu_glm5next::nextn_walk`, `nextn_chain`)
//! held against ik's MTP draft set (refset family `mtp-glm5next`) and the
//! drafted windows (`app::mtp::MtpDraft<Body>`) against the plain run.
//!
//! What is asserted:
//! - (m) the set's MTP input: its warmup graph's `enorm` reads `inp_embd`
//!   itself, with no position-mask node between, and no row of `inp_embd`
//!   is zeros, position 0 included — the rule the program's embedding rows
//!   follow.
//! - (o) teacher-forced, every graph of the set replayed in ik's order — the
//!   warmup, then each block's graphs — each row's token, position and
//!   target hidden row (ik's `inp_tokens`, `inp_pos`, `inp_mtp_states`) fed
//!   from the host in runs of up to [`WALK_ROWS`] rows, the runs before the
//!   last store-only walks and the last one a chain: the chain's id is the
//!   argmax of our full head's logits, first index on a tie; on the graph's
//!   last row that argmax is ik's `result_output` argmax wherever ik's
//!   top-2 margin clears [`flip::margin_cap`] at [`logits_band`], and ik's
//!   runner-up otherwise. Each block's proposal — the argmax of the graph
//!   before its update graph — is the set's draft token for the block, or
//!   that graph's tie. Every graph with ik's head on its last row holds our
//!   logits within [`logits_band`] of ik's (`‖ours − ik‖ / ‖ik‖`), the worst
//!   printed over the band. At least one graph is held off a tie.
//! - (p) the pairing: the hidden rows a walk fed the target's arenas reads
//!   (`bloomery_gpu_glm5next::nextn_hidden`, the walk's own gather): the
//!   set's first `CHUNK` prompt ids in a batch and by steps leave the same
//!   rows in the prompt-batch and step arenas, bit for bit (a batch of at
//!   most a chunk runs the step's gemvs; the whole prompt's distance between
//!   the two printed, a longer batch running the GEMM), and a prompt of
//!   [`GROUPED`] of its ids cycled (two batches) at groups of two leaves the
//!   rows it leaves at groups of one, bit for bit, every unit's rows read
//!   after the group; each value of every row of the whole prompt's
//!   within its derived bound of the f64 replica of the gather — the four
//!   streams' mean, then `output_norm`'s RMS at the file's eps — on the
//!   target streams the row was made from (`nextn_target_streams`; the
//!   bound is `HiddenRef`'s); the row at position `q` closer to ik's warmup
//!   row at `q + 1` than to its rows `q` and `q + 2`, each row's three
//!   distances printed, not held (no band in the tree derives a row past a
//!   flip, and the set carries no routing to tell the rows off every flip's
//!   path); a verify of two
//!   rows leaves in the pair arena the rows two plain steps leave, bit for
//!   bit, and keeps them past its commit and the next step; a verify whose
//!   commit keeps row 0 alone leaves that row readable and row 1 refused by
//!   name (the arena holds the kept position alone).
//! - (t) the target is the NextN load's own: the plain run of the set's
//!   prompt and [`N`] greedy ids on a load of the NextN plan's target plan
//!   without the layer (two KDA lanes, as the plan counts them), and the
//!   same on the NextN load, give the same ids and every step's logits bit
//!   for bit. ik's committed stream beside them is printed, not held.
//! - (w) the windows end to end, the prompt fed in batches and by steps:
//!   under `Speculative<MtpDraft<Body>, 2>` the drafted greedy ids are the
//!   plain run's on the same path for [`N`] tokens, with a rejected row and an accepted
//!   proposal among the windows — else either path never ran and the clause
//!   is red.
//! - (f) the walk's refusals by name: a walk on a load without the layer,
//!   and the MTP window opened over it (`MtpBody::head`), a captured walk, no
//!   rows, [`WALK_ROWS`] + 1 rows, a walk from past the positions the store
//!   holds, a token past the vocabulary, a host hidden slice of the wrong
//!   length, target rows past the arena's, and a chain with own walks, in
//!   the store walk's mode or into no place; each leaves the store's
//!   positions where they were. Then the target's rows by position
//!   (`nextn_hidden`, the walk's own gather): the pair arena before any
//!   verify, the step arena for a walk at position 0, and after a prompt of
//!   [`FED`] ids in one batch the prompt-batch arena's row past the batch's
//!   last and its rows read as other positions, each refused by name.
//! - (h) a fault in a chain poisons the model at once: the layer's
//!   `shared_head_norm` gain's first value set to NaN, a chain from a reset
//!   is a fault by name and leaves the model poisoned by that fault before
//!   any target call, and a walk after it is refused as poisoned; the gain
//!   put back and a reset, the same chain gives the clean chain's id.
//! - (s) a sequence state (`bloomery_gpu_glm5next::seq_save`) of the load
//!   without the layer, taken after its plain run, put back on the NextN
//!   load (`seq_resume`): the store layouts differ (the NextN layer's store
//!   rows), so it is refused by name before any copy, and the model stays at
//!   position 0, unpoisoned. The same file, card and context: the refusal is
//!   the layout's, not the identity's. Both loads' states pin
//!   `bloomery_gpu_glm5next::seq_bytes` — the plan-time host bytes of a
//!   state, the elastic `--parallel` default's unit — equal to the state's
//!   `GlmSeq::bytes` at its positions and its draft setting.
//! - (z) two windows' drafted rounds as one pass of both slots' verify rows
//!   a round (`app::mtp::pass_slots`), on a load of its own whose plan
//!   counts two sequences (the e2e gate's slots load), the prose set's
//!   prompts the e2e gate's (sd) feeds: each window's draft its own and its
//!   proposal capped at the seat's depth (`app::mtp::SLOT_DEPTH`: two rows a
//!   slot, a pass of four), every window's ids, kept rows a round, draft
//!   store and position its run alone on the load's one sequence before any
//!   slot exists, through the session's verify of one sequence — with a
//!   round keeping fewer rows than it ran and a round keeping different
//!   counts on the two slots, else the clause is red (the per-slot keep
//!   never ran apart).
//!
//! Tiers (`BLOOMERY_TIER`): the clauses that read ik's MTP set (the set's input, the replayed graphs, the walk's
//! warm-up rows) are `Tag::Oracle`, and those that need the file's trained draft (a window that accepts, two slots that
//! keep different counts) are `Tag::FileBound`; the fixture tier defers them to the real tier by name and runs the rest
//! on the fixture, whose random draft rejects every proposal (the state-bytes clause counts the pair arena at the rows
//! its last window kept). The prompt is the first 64 ids of the prose corpus in both tiers; the real tier witnesses that
//! they are the set's.
//!
//! The fixture tier reads ik's MTP set dumped on the fixture file too, through `tier::fixture_set`
//! (`Tag::FixtureOracle`, one declared for each clause below, `tier::expect_fixture_oracle`; the real tier leaves each
//! to the fixture tier by name): (pfx) the pairing's rule, (mfx) the set's MTP input, (ofx) the replay with the logits
//! held to a band counted over the layer's own projections ([`fixture_band`]) and the argmax as a tie band
//! (`flip::head_tie`) in place of ik's runner-up, which a near-flat random head's rounding leaves too often, and
//! (tfx-p) ik's committed stream, printed. (ofx) also reads the NextN router the replay left (`nextn_router`) against
//! ik's `ffn_moe_logits`, `ffn_moe_probs_biased` and `ffn_moe_topk` of the layer: on every graph our router's logits
//! within the router's band of ik's ([`FxRule`]); a graph whose picks differ from ik's is a flip, allowed as
//! `gate_glm5next_e2e`'s flips are (each exchanged pair's gap within our error, the error within `flip::flip_cap` at
//! the router's band), its logits printed, not held, its argmax still a tie band; a flip not allowed fails.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_mtp: built without the `glm5next` feature; see `just gate-gpu-glm5next-mtp`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_mtp", gate::run())
}

#[cfg(feature = "glm5next")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_tier.rs"]
mod glm5next_tier;

#[cfg(feature = "glm5next")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::Session;
    use app::mtp::{MtpDraft, SLOT_DEPTH, SlotWindow, pass_slots};
    use bloomery_gpu::GpuError;
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::act_rule::Arm;
    use bloomery_gpu_gates::flip;
    use bloomery_gpu_gates::residency38::{glm_seqs, reserve_checkpoints};
    use bloomery_gpu_gates::rounding::q8_32_rel;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{Fnv1a64, GateError, checks_failed, patch_bytes, verdict};
    use bloomery_gpu_glm5next::{
        Body, CHUNK, GEMM_FROM, Glm5nextModel, GlmArena, GlmPromptSink, GlmSeq, NextnFeed,
        NextnHead, NextnHidden, NextnMode, PrefillMode, WALK_ROWS, feed, nextn_chain, nextn_hidden,
        nextn_logits, nextn_router, nextn_store, nextn_target_streams, nextn_walk, prompt_with,
        seq_bytes, seq_resume, seq_save, set_prefill, set_prefill_group,
    };
    use gguf::Split;
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{KdaLanes, NextnInputs, PlanInputs};
    use refset::arch::glm5next::D1K;
    use refset::ik::Layout;
    use refset::mtpref::{Graph, MtpSet};
    use runtime::layer::{FfnKind, MixerKind};
    use runtime::swaprule::KeptRows;
    use runtime::{Advance as _, Committed, Draft, PassSink, Target, Verify, Want, accepted_rows};

    use crate::glm5next_tier;
    use crate::quiet::Quiet;

    /// Cache rows: the e2e gate's main load.
    const CTX: usize = 3136;
    /// (p)'s grouped prompt: two batches, the second short.
    const GROUPED: usize = 600;
    /// (f)'s prompt in one batch: its rows are the prompt-batch arena's.
    const FED: usize = 12;
    /// Generated ids a plain or drafted run holds: the set's.
    const N: usize = 64;
    /// The verify's rows: a proposal of one id and the token before it.
    const M: usize = 2;

    /// The fixture-oracle clauses this gate declares (`tier::expect_fixture_oracle`): the pairing,
    /// the set's MTP input, the replayed graphs and ik's committed stream, each on the fixture's
    /// MTP set.
    const FIXTURE_CLAUSES: &[&str] = &[PFX, MFX, OFX, TFX_P];
    const FIXTURE_ORACLE: usize = FIXTURE_CLAUSES.len();
    const PFX: &str = "(pfx) our row at q against the fixture's ik warmup row q + 1";
    const MFX: &str = "(mfx) the fixture set's MTP input: no position mask, no zero row";
    const OFX: &str =
        "(ofx) teacher-forced: every graph of the fixture's set replayed against ik's";
    const TFX_P: &str =
        "(tfx-p) ik's committed stream on the fixture beside the plain run (printed, not held)";

    /// PIN(2026-10-01): the band of the NextN head's logits against ik's that
    /// (o) holds and its tie cap reads: on ik's own inputs (the token and the hidden row
    /// fed from the set), the projections in series where ik's CPU side reads
    /// 32-value q8 activations and ours f32 — `eh_proj`, the joined latent
    /// projection, the query's up projection, the key and value absorbs, the
    /// attention's output projection, the routed experts' gate·up and down,
    /// the shared expert's gate·up and down, and the head's projection:
    /// eleven terms of [`q8_32_rel`], √11 · 1.2858e-2 = 4.26e-2.
    fn logits_band() -> f64 {
        11f64.sqrt() * q8_32_rel()
    }

    /// The fixture's replay numbers ([`Rule::Fixture`]): the band of the NextN head's logits, the
    /// band of the layer's router logits, and the layer's index in the file, which names ik's
    /// routing nodes.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct FxRule {
        band: f64,
        router: f64,
        layer: usize,
    }

    /// The bands of the NextN head's logits against ik's on the fixture, and of the layer's router
    /// logits, on ik's own inputs, counted as `gate_glm5next_e2e`'s fixture bands count: the same
    /// projections as [`logits_band`]'s eleven — `eh_proj`, the layer's mixer sites, its block's
    /// (the routed experts' gate·up and down, the shared expert's) and the head's own input — each
    /// by how both engines round its activation (`act_rule::site_rel` at the worst crest of each
    /// side's block), as variances; the router reads what `eh_proj` and the mixer leave, so its
    /// band counts those alone. The m-column walk reads f32 as the gemv does.
    fn fixture_band(split: &Split, index: usize) -> Result<FxRule, GateError> {
        let eh = glm5next_tier::site(split, &names::nextn_eh_proj(index), Arm::Gemv, 1.0)?;
        let mixer = glm5next_tier::mixer_terms(split, index, MixerKind::Latent, Arm::Gemv)?;
        let block = glm5next_tier::block_terms(split, index, FfnKind::Moe, Arm::Gemv)?;
        let head = glm5next_tier::site(split, &names::output(), Arm::Gemv, 1.0)?;
        let router = eh.var() + glm5next_tier::var(&mixer);
        let var = router + block.var() + head.var();
        println!(
            "bands: the fixture's logits' (held, and the argmax cap's) {:.4e} = sqrt(var {var:.4e}), the router's (held, and the flip cap's) {:.4e} = sqrt(var {router:.4e}):              {}; mixer [{}]; block [{} | shared {}]; {}",
            var.sqrt(),
            router.sqrt(),
            eh.text(),
            glm5next_tier::terms_text(&mixer),
            glm5next_tier::terms_text(&block.main),
            glm5next_tier::terms_text(&block.shared),
            head.text()
        );
        Ok(FxRule {
            band: var.sqrt(),
            router: router.sqrt(),
            layer: index,
        })
    }

    /// How a replayed graph's last row is judged: the real file's rule, or the fixture's.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Rule {
        /// Ours is ik's argmax, or ik's runner-up where ik's margin is within the cap, at
        /// [`logits_band`].
        Real,
        /// Ours is ik's argmax, or ik's margin is within the cap and ik's logit at ours within
        /// the tie band ([`flip::head_tie`]) of its top, at the band given: a near-flat head's
        /// argmax is a tie band, which asks for no runner-up. On every graph our router's logits
        /// are within the router's band of ik's; where our picks differ from ik's the graph is a
        /// flip, allowed as the end-to-end gates allow one, its logits printed, not held.
        Fixture(FxRule),
    }

    impl Rule {
        /// The band the logits are held to and the caps read.
        fn band(self) -> f64 {
            match self {
                Rule::Real => logits_band(),
                Rule::Fixture(f) => f.band,
            }
        }

        /// Whether `ours`, which is not ik's argmax, is excused as a tie of ik's row `ik`, whose
        /// runner-up is `ik_2`.
        fn excuses(self, ik: &[f32], ours: usize, ik_2: usize) -> bool {
            match self {
                Rule::Real => ours == ik_2,
                Rule::Fixture(f) => {
                    u32::try_from(ours).is_ok_and(|o| flip::head_tie(ik, o, f.band))
                }
            }
        }

        /// What a graph's pass line says of a tie.
        fn tie_text(self) -> &'static str {
            match self {
                Rule::Real => "PASS (a tie: ik's runner-up)",
                Rule::Fixture(_) => "PASS (a tie band)",
            }
        }
    }

    /// A fixture-oracle clause's verdict: an error it ended in, a set the tier refused among them,
    /// is its named FAIL, never the gate's end.
    fn fixture_verdict(what: &str, r: Result<bool, GateError>) -> bool {
        r.unwrap_or_else(|e| {
            println!("{what}: ended in error \"{e}\" {}", verdict(false));
            false
        })
    }

    /// ik's NextN router on a graph's last row: its logits, the scores it ranks by (the sigmoid of
    /// the logits plus the selection bias) and the experts it picked.
    struct IkRouter {
        logits: Vec<f32>,
        biased: Vec<f32>,
        picks: Vec<i32>,
    }

    impl IkRouter {
        fn read(
            set: &MtpSet,
            block: i32,
            graph: Graph,
            layer: usize,
        ) -> Result<IkRouter, GateError> {
            let find = |stem: &str| set.find(block, graph, &format!("{stem}-{layer}"), 0);
            let r = IkRouter {
                logits: set.logical_f32s(find("ffn_moe_logits")?)?,
                biased: set.logical_f32s(find("ffn_moe_probs_biased")?)?,
                picks: set.i32s(find("ffn_moe_topk")?, Layout::Logical)?,
            };
            if r.logits.is_empty()
                || r.logits.len() != r.biased.len()
                || r.picks
                    .iter()
                    .any(|&e| !usize::try_from(e).is_ok_and(|e| e < r.logits.len()))
            {
                return Err(format!(
                    "block {block} {}: ik's router holds {} logits, {} scores and picks {:?}",
                    graph.as_str(),
                    r.logits.len(),
                    r.biased.len(),
                    r.picks
                )
                .into());
            }
            Ok(r)
        }
    }

    /// One graph's router against ik's.
    #[derive(Clone, Copy, Debug)]
    struct Route {
        /// Our logits are within the router's band, and any flip is allowed; true where nothing
        /// was asked.
        ok: bool,
        /// Our picks differ from ik's.
        flipped: bool,
        /// Our router logits' distance from ik's.
        rel: f64,
    }

    impl Route {
        /// No router read: the real file's rule.
        const NONE: Route = Route {
            ok: true,
            flipped: false,
            rel: 0.0,
        };
    }

    /// The router just walked against ik's: our logits within the router's band of ik's, and our
    /// picks ik's or a flip allowed. Ours are ranked by ik's scores moved by the distance of our
    /// logits' sigmoids from ik's, the selection bias being the file's, so each exchanged pair's
    /// error is our logits' own ([`flip::Flip`]); the cap is `fx_route_cap`'s form at the router's
    /// band on ik's logits.
    fn route_of(
        m: &mut Glm5nextModel,
        g: &IkGraph,
        ik: &IkRouter,
        fx: FxRule,
    ) -> Result<Route, GateError> {
        let (z, ids) = nextn_router(m)?;
        if z.len() != ik.logits.len() || ids.len() != ik.picks.len() {
            return Err(format!(
                "{}: our router holds {} logits and {} picks, ik's {} and {}",
                g.label(),
                z.len(),
                ids.len(),
                ik.logits.len(),
                ik.picks.len()
            )
            .into());
        }
        let r = rel(&z, &ik.logits);
        let sigmoid = |x: f32| 1.0 / (1.0 + (-f64::from(x)).exp());
        let ours: Vec<f32> = ik
            .biased
            .iter()
            .zip(z.iter().zip(&ik.logits))
            .map(|(&b, (&a, &i))| (f64::from(b) + sigmoid(a) - sigmoid(i)) as f32)
            .collect();
        let our_ids: Vec<i32> = ids
            .iter()
            .map(|&e| i32::try_from(e))
            .collect::<Result<_, _>>()?;
        let ik_margin = flip::margin(&ik.biased, &ik.picks);
        let cap = flip::flip_cap(fx.router / 4.0, &ik.logits);
        let swap = flip::Flip::between(
            (fx.layer, 0),
            (&ids, &ours),
            (&ik.picks, &ik.biased),
            ik_margin,
        );
        let picks = match &swap {
            None => format!("picks ik's (ik margin {ik_margin:.3e})"),
            Some(f) => format!(
                "{} (our margin {:.3e})",
                f.line("flip", cap),
                flip::margin(&ours, &our_ids)
            ),
        };
        let ok = r <= fx.router && swap.as_ref().is_none_or(|f| f.allowed(cap));
        println!(
            "(o) {}: router logits rel {r:.3e} (held <= {:.3e}); {picks} {}",
            g.label(),
            fx.router,
            verdict(ok)
        );
        Ok(Route {
            ok,
            flipped: swap.is_some(),
            rel: r,
        })
    }

    /// `‖a − b‖ / ‖b‖` in f64; infinite on a NaN or a length mismatch.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The first index of the largest value, and the largest of the rest:
    /// the head's tie rule.
    fn top2(v: &[f32]) -> (usize, usize) {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        let mut second = usize::from(best == 0);
        for (i, &x) in v.iter().enumerate() {
            if i != best && x > v[second] {
                second = i;
            }
        }
        (best, second)
    }

    /// One graph of the set: its block, label and rows, ik's side.
    struct IkGraph {
        block: i32,
        graph: Graph,
        tokens: Vec<u32>,
        pos: Vec<u32>,
        states: Vec<f32>,
        /// ik's head rows: `inp_out_ids`, or the one row of a one-row graph.
        out_ids: Vec<usize>,
        logits: Vec<f32>,
    }

    impl IkGraph {
        fn read(
            set: &MtpSet,
            block: i32,
            graph: Graph,
            hidden: usize,
        ) -> Result<IkGraph, GateError> {
            let find = |name: &str| set.find(block, graph, name, 0);
            let unsigned = |name: &str| -> Result<Vec<u32>, GateError> {
                set.i32s(find(name)?, Layout::Flat)?
                    .into_iter()
                    .map(|v| u32::try_from(v).map_err(|_| format!("{name} holds {v}").into()))
                    .collect()
            };
            let tokens = unsigned("inp_tokens")?;
            let n = tokens.len();
            // GLM's MTP layer has no rope, so ik's graph has no `inp_pos`: a
            // row's position is its mask row's count of visible keys, less one.
            // The mask is `ne[0]` keys by its rows padded past `n`.
            let mask_row = find("KQ_mask")?;
            let kv = usize::try_from(mask_row.row.ne[0])?;
            let mask = set.logical_f32s(mask_row)?;
            if n == 0 || kv == 0 || mask.len() < n * kv {
                return Err(
                    format!("KQ_mask holds {} values, {n} rows of {kv} keys", mask.len()).into(),
                );
            }
            let pos: Vec<u32> = (0..n)
                .map(|r| {
                    let seen = mask[r * kv..(r + 1) * kv]
                        .iter()
                        .filter(|x| x.is_finite())
                        .count();
                    u32::try_from(seen)
                        .ok()
                        .and_then(|c| c.checked_sub(1))
                        .ok_or_else(|| GateError::from(format!("KQ_mask row {r} shows no key")))
                })
                .collect::<Result<_, _>>()?;
            // ik builds `inp_out_ids` only for a graph of more than one row.
            let out_ids = if n > 1 {
                unsigned("inp_out_ids")?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect()
            } else {
                vec![0]
            };
            let g = IkGraph {
                block,
                graph,
                states: set.logical_f32s(find("inp_mtp_states")?)?,
                logits: set.logical_f32s(find("result_output")?)?,
                tokens,
                pos,
                out_ids,
            };
            let label = g.label();
            if n == 0 || g.pos.len() != n || g.states.len() != n * hidden {
                return Err(format!(
                    "{label}: {n} tokens, {} positions, {} hidden values; {n} rows of {hidden} take \
                     {}",
                    g.pos.len(),
                    g.states.len(),
                    n * hidden
                )
                .into());
            }
            if g.pos.windows(2).any(|w| w[1] != w[0] + 1) || g.out_ids.iter().any(|&o| o >= n) {
                return Err(format!(
                    "{label}: positions not consecutive, or an output row past {n}"
                )
                .into());
            }
            Ok(g)
        }

        fn label(&self) -> String {
            format!("block {} {}", self.block, self.graph.as_str())
        }
    }

    /// The set's graphs in the order ik computed them: the warmup, then each
    /// verified block's graphs in file order.
    fn graphs_of(set: &MtpSet) -> Vec<(i32, Graph)> {
        let mut order: Vec<(i32, Graph)> = Vec::new();
        for r in &set.rows {
            if !order.contains(&(r.block, r.graph)) {
                order.push((r.block, r.graph));
            }
        }
        order
    }

    /// How a graph's last row's argmax stood against ik's.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Head {
        Same,
        /// ik's runner-up, ik's margin under the cap.
        Tie,
        Bad,
        /// ik computed no head on the last row: not held.
        Unheld,
    }

    /// One graph's replay: our id, ik's, how they stood, and our logits'
    /// distance from ik's where ik computed the head.
    struct Replayed {
        ours: u32,
        ik: Option<u32>,
        head: Head,
        rel: Option<f64>,
        route: Route,
    }

    /// Replay `g` from the host: the runs before the last store-only walks,
    /// the last a chain; our logits against ik's on the last row.
    fn replay(
        m: &mut Glm5nextModel,
        (g, ik_router): (&IkGraph, Option<&IkRouter>),
        hidden: usize,
        rule: Rule,
    ) -> Result<Replayed, GateError> {
        let n = g.tokens.len();
        let mut c0 = 0;
        let mut ours = None;
        while c0 < n {
            let rows = WALK_ROWS.min(n - c0);
            let f = NextnFeed {
                tokens: &g.tokens[c0..c0 + rows],
                pos0: g.pos[c0],
                hidden: NextnHidden::Host(&g.states[c0 * hidden..(c0 + rows) * hidden]),
            };
            if c0 + rows < n {
                nextn_walk(m, f, NextnHead::Full, NextnMode::Store)?;
            } else {
                let mut out = [0u32; 1];
                let k = nextn_chain(m, f, 0, NextnHead::Full, NextnMode::Eager, &mut out)?;
                if k != 1 {
                    return Err(format!("{}: a chain of {k} ids, want 1", g.label()).into());
                }
                ours = Some(out[0]);
            }
            c0 += rows;
        }
        let ours = ours.ok_or_else(|| format!("{}: no chain ran", g.label()))?;
        let logits = nextn_logits(m)?;
        let route = match (rule, ik_router) {
            (Rule::Fixture(fx), Some(ik)) => route_of(m, g, ik, fx)?,
            _ => Route::NONE,
        };
        let vocab = logits.len();
        let (top, _) = top2(&logits);
        if top != ours as usize {
            println!(
                "(o) {}: the chain names {ours}, our logits' argmax {top} FAIL",
                g.label()
            );
            return Ok(Replayed {
                ours,
                ik: None,
                head: Head::Bad,
                rel: None,
                route,
            });
        }
        if g.logits.len() != g.out_ids.len() * vocab {
            return Err(format!(
                "{}: result_output holds {} values, {} output rows of {vocab}",
                g.label(),
                g.logits.len(),
                g.out_ids.len()
            )
            .into());
        }
        let Some(k) = g.out_ids.iter().position(|&o| o == n - 1) else {
            println!(
                "(o) {}: ik computed no head on row {}, not held",
                g.label(),
                n - 1
            );
            return Ok(Replayed {
                ours,
                ik: None,
                head: Head::Unheld,
                rel: None,
                route,
            });
        };
        let ik = &g.logits[k * vocab..(k + 1) * vocab];
        let (ik_top, ik_2) = top2(ik);
        let margin = f64::from(ik[ik_top]) - f64::from(ik[ik_2]);
        let band = rule.band();
        let cap = flip::margin_cap(band, ik);
        let head = if top == ik_top {
            Head::Same
        } else if margin <= cap && rule.excuses(ik, top, ik_2) {
            Head::Tie
        } else {
            Head::Bad
        };
        let r = rel(&logits, ik);
        let in_band = route.flipped || r <= band;
        let held = if route.flipped {
            "printed, a flip".to_string()
        } else {
            format!("held <= {band:.3e}")
        };
        println!(
            "(o) {}: {n} rows from {}, argmax {top} ik {ik_top} (runner-up {ik_2}, margin \
             {margin:.3e}, cap {cap:.3e}), logits rel {r:.3e} ({held}) {}",
            g.label(),
            g.pos[0],
            match (head, in_band) {
                (Head::Same, true) => "PASS",
                (Head::Tie, true) => rule.tie_text(),
                (Head::Same | Head::Tie, false) => "FAIL (past the band)",
                _ => "FAIL",
            }
        );
        Ok(Replayed {
            ours,
            ik: Some(ik_top as u32),
            head,
            rel: Some(r),
            route,
        })
    }

    /// (m): the warmup graph's embedding rows, unmasked at every position.
    fn pos_mask(set: &MtpSet, hidden: usize) -> Result<bool, GateError> {
        let masked = set
            .find(-1, Graph::Warmup, "mtp_token_embd_pos_masked", 0)
            .is_ok();
        let e = set.logical_f32s(set.find(-1, Graph::Warmup, "inp_embd", 0)?)?;
        if e.is_empty() || e.len() % hidden != 0 {
            return Err(format!("inp_embd holds {} values, rows of {hidden}", e.len()).into());
        }
        let rows = e.len() / hidden;
        // The warmup is the prompt from position 0, so its row 0 is position 0.
        let ok =
            !masked && (0..rows).all(|r| e[r * hidden..(r + 1) * hidden].iter().any(|&x| x != 0.0));
        println!(
            "(m) the warmup's {rows} embedding rows (the prompt from position 0): no mask node, none zeros {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// (o): every graph replayed, each block's proposal against the set's.
    fn oracle(
        m: &mut Glm5nextModel,
        set: &MtpSet,
        hidden: usize,
        rule: Rule,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let mut ok = true;
        let mut last: Option<Replayed> = None;
        let (mut same, mut ties, mut bad, mut unheld) = (0usize, 0usize, 0usize, 0usize);
        let (mut blocks_ok, mut blocks_tie, mut blocks_bad) = (0usize, 0usize, 0usize);
        let band = rule.band();
        let (mut worst, mut past) = (0.0f64, 0usize);
        let (mut route_bad, mut flips, mut router_worst) = (0usize, Vec::new(), 0.0f64);
        for (b, g) in graphs_of(set) {
            if g == Graph::Update {
                let draft = set.drafts_of(b);
                let (&[want], Some(prev)) = (draft.as_slice(), last.as_ref()) else {
                    println!(
                        "(o) block {b}: {} draft rows and {} graph before its update FAIL",
                        draft.len(),
                        if last.is_some() { "a" } else { "no" }
                    );
                    blocks_bad += 1;
                    continue;
                };
                if prev.ik.is_some_and(|ik| ik != want) {
                    println!(
                        "(o) block {b}: the set's draft {want} is not the argmax {:?} of the graph \
                         before its update FAIL",
                        prev.ik
                    );
                    blocks_bad += 1;
                } else if prev.ours == want {
                    blocks_ok += 1;
                } else if prev.head == Head::Tie {
                    blocks_tie += 1;
                    println!(
                        "(o) block {b}: our proposal {} against the set's {want}, a tie",
                        prev.ours
                    );
                } else {
                    blocks_bad += 1;
                    println!(
                        "(o) block {b}: our proposal {} against the set's {want} FAIL",
                        prev.ours
                    );
                }
            }
            let ig = IkGraph::read(set, b, g, hidden)?;
            let ik_router = match rule {
                Rule::Fixture(fx) => Some(IkRouter::read(set, b, g, fx.layer)?),
                Rule::Real => None,
            };
            let r = replay(m, (&ig, ik_router.as_ref()), hidden, rule)?;
            route_bad += usize::from(!r.route.ok);
            router_worst = router_worst.max(r.route.rel);
            if r.route.flipped {
                flips.push(ig.label());
            }
            match r.head {
                Head::Same => same += 1,
                Head::Tie => ties += 1,
                Head::Bad => bad += 1,
                Head::Unheld => unheld += 1,
            }
            if let Some(d) = r.rel.filter(|_| !r.route.flipped) {
                worst = worst.max(d);
                // `rel` reads a NaN as infinite: past the band.
                if d > band {
                    past += 1;
                }
            }
            last = Some(r);
        }
        let blocks = set.blocks().len();
        let graphs_ok = bad == 0 && same > 0 && past == 0 && route_bad == 0;
        let props_ok = blocks_bad == 0 && blocks_ok + blocks_tie == blocks && blocks_ok > 0;
        ok &= graphs_ok && props_ok;
        println!(
            "(o) graphs: {same} ik's argmax, {ties} ik's runner-up at a tie, {bad} other, {unheld} \
             without ik's head; logits rel worst {worst:.3e}, {:.3} of the band {band:.3e}, {past} \
             past it {}",
            worst / band,
            verdict(graphs_ok)
        );
        println!(
            "(o) blocks: {blocks_ok} of {blocks} proposals the set's draft, {blocks_tie} a tie, \
             {blocks_bad} other {}",
            verdict(props_ok)
        );
        if let Rule::Fixture(fx) = rule {
            println!(
                "(o) router: {} graphs where our picks differ from ik's [{}]; router logits rel worst \
                 {router_worst:.3e}, {:.3} of the band {:.3e}; {route_bad} graphs past it or with a flip \
                 not allowed {}",
                flips.len(),
                flips.join(", "),
                router_worst / fx.router,
                fx.router,
                verdict(route_bad == 0)
            );
        }
        Ok(ok)
    }

    /// Every hidden row a prompt call's units leave, in position order, read
    /// by [`nextn_hidden`] from the unit's arena in runs of [`WALK_ROWS`],
    /// and the target streams each run was made from
    /// ([`nextn_target_streams`]).
    fn unit_rows(
        m: &mut Glm5nextModel,
        prompt: &[u32],
        path: PrefillMode,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        m.reset()?;
        set_prefill(m, path)?;
        let mut got: Vec<f32> = Vec::new();
        let mut streams: Vec<f32> = Vec::new();
        let mut units = |m: &mut Glm5nextModel, arena: GlmArena, first: u32, rows: usize| {
            let mut c0 = 0;
            while c0 < rows {
                let r = WALK_ROWS.min(rows - c0);
                // The walk from the position after the unit's row c0 reads it.
                let pos0 = first + c0 as u32 + 1;
                let target = NextnHidden::Target {
                    walk: arena,
                    first: c0,
                };
                got.extend(nextn_hidden(m, pos0, target, r)?);
                streams.extend(nextn_target_streams(m, pos0, target, r)?);
                c0 += r;
            }
            Ok(())
        };
        let sink: &mut GlmPromptSink<'_> = &mut units;
        prompt_with(m, prompt, path, Some(sink))?;
        Ok((got, streams))
    }

    /// The unit roundoff of f32.
    const U: f64 = 1.0 / (1u64 << 24) as f64;

    /// PIN(2026-10-01): the bound on one hidden value the walk's gather
    /// writes — `ds41_hc_mean` over the four streams, then `rms_norm` over
    /// `n` = 4,096 values with `output_norm`'s gain — against its f64
    /// replica on the same f32 streams, first order in `u` = 2^-24, derived
    /// from the kernels' own order:
    /// - the mean `(((a + b) + c) + d) · 0.25`: three additions, the product
    ///   exact, so `|δm_i| <= 3u · S_i`, `S_i` the mean of the four `|s|`
    ///   (absolute: the streams may cancel);
    /// - the sum of squares of the device's own means: each of 256 threads
    ///   sums its 16 squares in order, then a five-level warp butterfly and
    ///   the three-level warp tree — every term through at most 23 additions
    ///   and its square's rounding, all terms positive: relative `24u`; the
    ///   means' error moves it by at most `6u · ρ` relative, `ρ = Σ|m_i|S_i /
    ///   Σm_i²`;
    /// - `/ n` exact (a power of two), `+ eps` `u`, the square root halves its
    ///   argument's error and adds its own, and the reciprocal its own, each
    ///   2u (the approximate forms' bound): `ε_scale = (25u + 6u·ρ) / 2 +
    ///   4u`;
    /// - `(scale · g_i) · m_i`: two roundings, `2u`.
    ///
    /// So `|y_i − ŷ_i| <= b_i`, with `b_i = |g_i| · scale · (3u · S_i +
    /// |m_i| · (ε_scale + 2u))`. The clause holds every value within `2 ·
    /// b_i` (the second order and the replica's own f64 error are below
    /// 10^-6 of `b_i`) plus `2^-126 · (1 + |g_i| · scale)`, a flushed
    /// subnormal. A value is bounded, not a row's norm: the bound is
    /// componentwise.
    struct HiddenRef {
        /// The replica's values and each value's bound, `2 · b_i` plus the
        /// subnormal floor.
        y: Vec<f64>,
        bound: Vec<f64>,
        /// The row's mean square before the norm.
        ms: f64,
    }

    /// Row `streams` (`4 · n`, stream-major) through the walk's gather in
    /// f64, with the bound on each value ([`HiddenRef`]).
    fn hidden_ref(streams: &[f32], gain: &[f32], eps: f32) -> HiddenRef {
        let n = gain.len();
        let at = |j: usize, i: usize| f64::from(streams[j * n + i]);
        let m: Vec<f64> = (0..n)
            .map(|i| (at(0, i) + at(1, i) + at(2, i) + at(3, i)) * 0.25)
            .collect();
        let s: Vec<f64> = (0..n)
            .map(|i| (0..4).map(|j| at(j, i).abs()).sum::<f64>() * 0.25)
            .collect();
        let sq: f64 = m.iter().map(|v| v * v).sum();
        let ms = sq / n as f64;
        let scale = 1.0 / (ms + f64::from(eps)).sqrt();
        let rho = if sq > 0.0 {
            m.iter().zip(&s).map(|(a, b)| a.abs() * b).sum::<f64>() / sq
        } else {
            0.0
        };
        let e_scale = (25.0 * U + 6.0 * U * rho) / 2.0 + 4.0 * U;
        let floor = f64::from(f32::MIN_POSITIVE);
        let mut y = Vec::with_capacity(n);
        let mut bound = Vec::with_capacity(n);
        for i in 0..n {
            let g = f64::from(gain[i]).abs() * scale;
            y.push(f64::from(gain[i]) * scale * m[i]);
            bound.push(
                2.0 * g * (3.0 * U * s[i] + m[i].abs() * (e_scale + 2.0 * U)) + floor * (1.0 + g),
            );
        }
        HiddenRef { y, bound, ms }
    }

    /// (p) the value held: every hidden row a prompt call's units leave, as
    /// the walk's gather wrote it, within its derived bound of the f64
    /// replica on the target streams it read ([`HiddenRef`]); the worst
    /// value's distance over its bound, the rows' rel-L2, their mean squares
    /// and the bit-equal count printed.
    fn hidden_held(rows: &[f32], streams: &[f32], gain: &[f32], eps: f32, path: &str) -> bool {
        let n = gain.len();
        let count = rows.len() / n.max(1);
        let shape_ok = n > 0 && rows.len() == count * n && streams.len() == count * 4 * n;
        let (mut worst, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        let (mut out, mut bits, mut ms_lo, mut ms_hi) = (0usize, 0usize, f64::MAX, 0.0f64);
        if shape_ok {
            for r in 0..count {
                let h = hidden_ref(&streams[r * 4 * n..(r + 1) * 4 * n], gain, eps);
                ms_lo = ms_lo.min(h.ms);
                ms_hi = ms_hi.max(h.ms);
                for i in 0..n {
                    let got = f64::from(rows[r * n + i]);
                    let d = (got - h.y[i]).abs();
                    worst = worst.max(d / h.bound[i]);
                    if d.is_nan() || d > h.bound[i] {
                        out += 1;
                    }
                    if got.to_bits() == (h.y[i] as f32 as f64).to_bits() {
                        bits += 1;
                    }
                    num += d * d;
                    den += h.y[i] * h.y[i];
                }
            }
        }
        let ok = shape_ok && count > 0 && out == 0;
        println!(
            "(p) the gather's value ({path}): {count} rows of {n} against the f64 replica of \
             mean-then-norm on their target streams: {out} values past their bound, the worst at \
             {worst:.3e} of it (held <= 1), rel-L2 {:.3e}, mean square {ms_lo:.3e}..{ms_hi:.3e}, \
             {bits} values the replica's f32 bits (printed) {}",
            (num / den.max(f64::MIN_POSITIVE)).sqrt(),
            verdict(ok)
        );
        ok
    }

    /// (p)'s row at `q` against ik's warmup row `q + 1` (`set`'s, `tag` the clause's name in the
    /// lines): ik's MTP row `p` reads the target's hidden of `p − 1`, so each of the first `n − 1`
    /// rows of `batch` (the prompt's hidden rows) is closer to ik's row `q + 1` than to its rows
    /// `q` and `q + 2`, the distances printed. `None` when the warmup's rows are not the prompt's
    /// at positions `0..n`, which ends the pairing.
    fn pair_shift(
        set: &MtpSet,
        tag: &str,
        (prompt, batch): (&[u32], &[f32]),
        hidden: usize,
    ) -> Result<Option<bool>, GateError> {
        let n = prompt.len();
        let warm = IkGraph::read(set, -1, Graph::Warmup, hidden)?;
        let pos_ok = warm.tokens == prompt && warm.pos.iter().copied().eq(0..n as u32);
        if !pos_ok {
            println!("{tag} the warmup's rows are not the prompt's at positions 0.. FAIL");
            return Ok(None);
        }
        let ik_row = |p: usize| &warm.states[p * hidden..(p + 1) * hidden];
        let (mut worst, mut bad_shift) = (0.0f64, 0usize);
        for q in 0..n - 1 {
            let ours = &batch[q * hidden..(q + 1) * hidden];
            let at = rel(ours, ik_row(q + 1));
            worst = worst.max(at);
            let before = rel(ours, ik_row(q));
            let after = (q + 2 < n).then(|| rel(ours, ik_row(q + 2)));
            let shifted = before <= at || after.is_some_and(|a| a <= at);
            if shifted {
                bad_shift += 1;
            }
            println!(
                "{tag} row {q}: against ik's row {} {at:.3e}, row {q} {before:.3e}, row {} {}{}",
                q + 1,
                q + 2,
                after.map_or_else(|| "-".to_string(), |a| format!("{a:.3e}")),
                if shifted { " closer off its pair" } else { "" }
            );
        }
        let ik_ok = bad_shift == 0;
        println!(
            "{tag} our row at q against ik's warmup row q + 1, {} rows: worst {worst:.3e} (printed), \
             {bad_shift} closer at q or q + 2 {}",
            n - 1,
            verdict(ik_ok)
        );
        Ok(Some(ik_ok))
    }

    /// (p) the pairing: the hidden rows a walk fed the target's arenas reads.
    /// A prompt call's first chunk of units in a batch (the prompt-batch
    /// arena) and by steps (the step arena) leave the same rows bit for bit,
    /// the whole prompt's two printed, each value of either within
    /// its bound of the gather's f64 replica on the rows' target streams
    /// ([`hidden_held`], at the file's `eps`); the row at position
    /// `q` lies closer to ik's warmup row at `q + 1` — ik's MTP row `p` reads
    /// the target's hidden of `p − 1` — than to its rows `q` and `q + 2`, the
    /// distances printed. Then a verify of two rows after the prompt leaves in the
    /// pair arena the rows two plain steps leave in the step arena, bit for
    /// bit, and after its commit and one more step still does, the step arena
    /// holding the plain run's next row: the verify's row 0 kept past the
    /// step that writes its buffers.
    fn pairing(
        m: &mut Glm5nextModel,
        prompt: &[u32],
        hidden: usize,
        eps: f32,
    ) -> Result<bool, GateError> {
        let (batch, batch_streams) = unit_rows(m, prompt, PrefillMode::Batch)?;
        let (steps, steps_streams) = unit_rows(m, prompt, PrefillMode::Steps)?;
        let gain = {
            let (gpu, w, _) = m.body_parts("pairing")?;
            let name = names::output_norm();
            let Some(DevWeight::F32 { w: g, .. }) = w.get(&name) else {
                return Err(format!("{name} is not resident as F32").into());
            };
            g.buf().to_host_vec(gpu.stream())?
        };
        let held_ok = hidden_held(&batch, &batch_streams, &gain, eps, "batches")
            & hidden_held(&steps, &steps_streams, &gain, eps, "steps");
        let n = prompt.len();
        let bits = |a: &[f32], b: &[f32]| {
            a.len() == b.len()
                && a.iter()
                    .map(|x| x.to_bits())
                    .eq(b.iter().map(|x| x.to_bits()))
        };
        // A batch of at most a chunk runs the step's gemvs, so its rows are
        // the steps' bits; a longer one runs the GEMM, whose rows the gather's
        // bound holds on their own target streams above.
        let short = &prompt[..n.min(CHUNK)];
        let (batch_short, _) = unit_rows(m, short, PrefillMode::Batch)?;
        let (steps_short, _) = unit_rows(m, short, PrefillMode::Steps)?;
        let same = batch_short.len() == short.len() * hidden && bits(&batch_short, &steps_short);
        let mut ok = same && held_ok && batch.len() == n * hidden;
        println!(
            "(p) the prompt's first {} units' hidden rows, in a batch = by steps, bit for bit {}; \
             all {n} units' in batches against by steps {:.3e} (printed: from {GEMM_FROM} \
             positions a batch runs the GEMM)",
            short.len(),
            verdict(same),
            rel(&batch, &steps)
        );
        // Two batches in one group: the sink reads each unit's own final
        // streams after the group; the same batches at groups of one.
        let two: Vec<u32> = prompt.iter().copied().cycle().take(GROUPED).collect();
        set_prefill_group(m, 2)?;
        let grouped = unit_rows(m, &two, PrefillMode::Batch);
        set_prefill_group(m, 1)?;
        let (grouped, _) = grouped?;
        let (ones, _) = unit_rows(m, &two, PrefillMode::Batch)?;
        let same2 = grouped.len() == GROUPED * hidden && bits(&grouped, &ones);
        ok &= same2;
        println!(
            "(p) {GROUPED} ids (two batches) at groups of two: every unit's hidden rows = at \
             groups of one, bit for bit {}",
            verdict(same2)
        );
        // Oracle: ik's warmup rows are of the real file.
        if tier::run_clause(
            "(p) our row at q against ik's warmup row q + 1",
            Tag::Oracle,
        )? {
            let Some(ik_ok) =
                pair_shift(glm5next_tier::mtp_set()?, "(p)", (prompt, &batch), hidden)?
            else {
                return Ok(false);
            };
            ok &= ik_ok;
        }
        // The same rule on the fixture's rows: random rows are near orthogonal, so the pair decides
        // more sharply than on the real file's.
        if tier::run_clause(PFX, Tag::FixtureOracle)? {
            ok &= fixture_verdict(
                "(pfx)",
                glm5next_tier::fx_mtp_set(PFX).and_then(|set| {
                    Ok(pair_shift(&set, "(pfx)", (prompt, &batch), hidden)?.unwrap_or(false))
                }),
            );
        }

        m.reset()?;
        set_prefill(m, PrefillMode::Batch)?;
        let step_row = |m: &mut Glm5nextModel| -> Result<Vec<f32>, GpuError> {
            let pos0 = m.pos();
            nextn_hidden(
                m,
                pos0,
                NextnHidden::Target {
                    walk: GlmArena::Step,
                    first: 0,
                },
                1,
            )
        };
        let t0 = feed(m, prompt)?;
        let p = m.pos();
        let t1 = m.step(&[t0])?;
        let s0 = step_row(m)?;
        let t2 = m.step(&[t1])?;
        let s1 = step_row(m)?;
        m.step(&[t2])?;
        let s2 = step_row(m)?;
        let plain: Vec<u32> = s0.iter().chain(&s1).map(|x| x.to_bits()).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        m.reset()?;
        feed(m, prompt)?;
        let rows = m.step_rows::<2>([t0, t1])?;
        let pair_row = |m: &mut Glm5nextModel| {
            nextn_hidden(
                m,
                p + 1,
                NextnHidden::Target {
                    walk: GlmArena::Pair,
                    first: 0,
                },
                2,
            )
        };
        let v = pair_row(m)?;
        let verify_ok = rows == [t1, t2] && bits(&v) == plain;
        m.rollback(p + 2)?;
        m.keep_rows(KeptRows::prefix(2), PassKind::Pair)?;
        m.step(&[t2])?;
        let kept_ok = bits(&pair_row(m)?) == plain && bits(&step_row(m)?) == bits(&s2);
        ok &= verify_ok && kept_ok;
        println!(
            "(p) a verify of two rows at {p}: tokens {rows:?} (want {:?}), the pair arena's rows = \
             two plain steps' {}; after its commit and a step, the pair arena's rows unchanged and \
             the step arena's the plain run's next {}",
            [t1, t2],
            verdict(verify_ok),
            verdict(kept_ok)
        );
        // A verify whose row 1 its commit takes back: the pair arena then
        // holds the kept row's position alone.
        let q = m.pos();
        m.step_rows::<2>([t2, t2])?;
        m.rollback(q + 1)?;
        m.keep_rows(KeptRows::prefix(1), PassKind::Pair)?;
        let pair = |first: usize| NextnHidden::Target {
            walk: GlmArena::Pair,
            first,
        };
        let one = nextn_hidden(m, q + 1, pair(0), 1);
        let both = nextn_hidden(m, q + 1, pair(0), 2);
        let held = format!("the arena holds positions {q}..{}", q + 1);
        let cut_ok = one.is_ok() && matches!(&both, Err(e) if e.to_string().contains(&held));
        ok &= cut_ok;
        println!(
            "(p) a verify at {q} with row 0 kept alone: the kept row {}, both rows {} (want \
             refused: {held}) {}",
            match &one {
                Ok(v) => format!("{} values", v.len()),
                Err(e) => format!("error \"{e}\""),
            },
            match &both {
                Ok(_) => "accepted".to_string(),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(cut_ok)
        );
        m.reset()?;
        Ok(ok)
    }

    /// A plain run's ids and each step's logits digest.
    struct Plain {
        ids: Vec<u32>,
        steps: Vec<u64>,
    }

    /// The set's prompt fed by `path` from a reset, then [`N`] − 1 steps.
    fn plain(m: &mut Glm5nextModel, prompt: &[u32], path: PrefillMode) -> Result<Plain, GateError> {
        m.reset()?;
        set_prefill(m, path)?;
        let mut ids = vec![feed(m, prompt)?];
        let mut steps = Vec::with_capacity(N);
        for _ in 1..N {
            let t = m.step(&[ids[ids.len() - 1]])?;
            ids.push(t);
            steps.push(Fnv1a64::default().f32s(&m.logits()?).value());
        }
        Ok(Plain { ids, steps })
    }

    /// ik's committed stream after the prompt: each round's target tokens,
    /// then its plain steps' tokens.
    fn ik_stream(set: &MtpSet) -> Vec<u32> {
        let mut s: Vec<u32> = set
            .verify
            .iter()
            .flat_map(|v| v.target.iter().copied())
            .collect();
        let mut plain = set.plain.clone();
        plain.sort_by_key(|p| p.pos);
        s.extend(plain.iter().map(|p| p.token));
        s
    }

    /// ik's committed stream beside the plain run's ids: how many of them agree, printed, not
    /// held.
    fn print_stream(set: &MtpSet, ids: &[u32]) {
        let ik = ik_stream(set);
        let agree = ik.iter().zip(ids).take_while(|(a, b)| a == b).count();
        println!(
            "(t) ik's committed stream: {} ids, the plain run's first {agree} of them \
             (printed, not held)",
            ik.len()
        );
    }

    /// The generation's passes, kept.
    #[derive(Default)]
    struct Kept {
        rows: Vec<usize>,
        proposed: Vec<bool>,
    }

    impl PassSink<Session<Body>> for Kept {
        type Error = GateError;

        fn begin(&mut self, _: &Session<Body>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &Session<Body>,
            c: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            self.rows.push(c.kept);
            self.proposed.push(c.proposed);
            Ok(())
        }
    }

    /// (w) one drafted run: the set's prompt fed by `path` from a reset, the
    /// windows for [`N`] ids, against `plain`.
    fn drafted(
        m: Glm5nextModel,
        prompt: &[u32],
        path: PrefillMode,
        plain: &Plain,
        set: Option<&MtpSet>,
    ) -> Result<(Glm5nextModel, bool, Option<usize>), GateError> {
        let mut m = m;
        m.reset()?;
        set_prefill(&mut m, path)?;
        let ctx = u32::try_from(CTX)?;
        let mut s = Session::from_model(m, ctx);
        let draft = MtpDraft::open(s.model(), path, StepMode::Eager)?;
        let mut spec = s.with_draft::<MtpDraft<Body>, M>(draft, &mut Quiet)?;
        let first = spec.prompt(&mut s, prompt)?;
        let mut k = Kept::default();
        let out = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(N, ctx)?,
            &mut k,
        )?;
        let m = s.into_model();
        let ids_ok = out.tokens.len() >= N && out.tokens[..N] == plain.ids[..N];
        let windows = k.proposed.iter().filter(|&&p| p).count();
        let accepted = k
            .rows
            .iter()
            .zip(&k.proposed)
            .filter(|&(&r, &p)| p && r == M)
            .count();
        let rejected = windows - accepted;
        // A proposal the target accepts is the file's draft: a fixture's NextN layer is random,
        // whose proposals the target refuses, so only the real tier requires one.
        let accepts = tier::premise(
            "(w) a window accepts a proposal",
            Tag::FileBound,
            accepted > 0,
        )?;
        let rows_ok = rejected > 0 && accepts;
        let ik = set.map_or_else(
            || "no set of ik's".to_string(),
            |s| {
                format!(
                    "{} blocks, {} accepted",
                    s.verify.len(),
                    s.verify.iter().map(|v| v.accepted).sum::<usize>()
                )
            },
        );
        println!(
            "(w) {path:?}: the drafted run's {N} ids = the plain run's {}",
            verdict(ids_ok)
        );
        println!(
            "(w) {path:?}: {windows} windows, {accepted} accepted, {rejected} rejected a row (ik: \
             {ik}) {}",
            verdict(rows_ok)
        );
        let last_window = k
            .rows
            .iter()
            .zip(&k.proposed)
            .rev()
            .find_map(|(&r, &p)| p.then_some(r));
        Ok((m, ids_ok && rows_ok, last_window))
    }

    /// Whether `r` is refused by name, `want` in its message.
    fn refused<T>(what: &str, r: Result<T, GpuError>, want: &str) -> bool {
        let ok = matches!(&r, Err(e) if e.to_string().contains(want));
        let got = match &r {
            Ok(_) => "accepted".to_string(),
            Err(e) => e.to_string(),
        };
        println!("(f) {what} -> {got} {}", verdict(ok));
        ok
    }

    /// (s) `without`, a state of the load without the layer, put back on the
    /// empty NextN load `m`: refused by name, and `m` still at position 0,
    /// unpoisoned. Mutant: the body's NextN-side check taken out (the shared
    /// layout check refuses it under its own name).
    fn layout_refused(m: &mut Glm5nextModel, without: &GlmSeq) -> bool {
        let r = seq_resume(m, without);
        let want = "a state without the NextN rows put back on a load with the layer";
        let named = matches!(&r, Err(e) if e.to_string().contains(want));
        let empty = m.pos() == 0 && m.poisoned().is_none();
        println!(
            "(s) a state of the load without the layer put back on the NextN load -> {} {}; the \
             model after it at position {}, poisoned {:?} {}",
            match &r {
                Ok(()) => "accepted".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(named),
            m.pos(),
            m.poisoned(),
            verdict(empty)
        );
        named && empty
    }

    /// A walk's feed.
    fn f<'a>(tokens: &'a [u32], pos0: u32, hidden: NextnHidden<'a>) -> NextnFeed<'a> {
        NextnFeed {
            tokens,
            pos0,
            hidden,
        }
    }

    /// (f) on the NextN load, from a reset: every refusal leaves the store's
    /// positions at 0.
    fn refusals(m: &mut Glm5nextModel, hidden: usize) -> Result<bool, GateError> {
        m.reset()?;
        let zeros = vec![0.0f32; (WALK_ROWS + 1) * hidden];
        let host = |rows: usize| NextnHidden::Host(&zeros[..rows * hidden]);
        let toks = vec![1u32; WALK_ROWS + 1];
        let mut ok = true;
        ok &= refused(
            "a captured walk",
            nextn_walk(
                m,
                f(&toks[..1], 0, host(1)),
                NextnHead::Full,
                NextnMode::Graph,
            ),
            "a captured NextN walk",
        );
        ok &= refused(
            "no rows",
            nextn_walk(
                m,
                f(&toks[..0], 0, host(0)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "a walk of 0 rows",
        );
        ok &= refused(
            "one row past the walk's width",
            nextn_walk(
                m,
                f(&toks, 0, host(WALK_ROWS + 1)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            &format!("a walk of {} rows", WALK_ROWS + 1),
        );
        ok &= refused(
            "a walk from past the store's positions",
            nextn_walk(
                m,
                f(&toks[..1], 1, host(1)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "a walk from position 1",
        );
        ok &= refused(
            "a token past the vocabulary",
            nextn_walk(
                m,
                f(&[u32::MAX], 0, host(1)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "past the",
        );
        ok &= refused(
            "a host hidden slice of the wrong length",
            nextn_walk(
                m,
                f(&toks[..1], 0, NextnHidden::Host(&zeros[..hidden - 1])),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "hidden values for 1 rows",
        );
        ok &= refused(
            "target rows past the step arena's one",
            nextn_walk(
                m,
                f(
                    &toks[..1],
                    0,
                    NextnHidden::Target {
                        walk: GlmArena::Step,
                        first: 1,
                    },
                ),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "of the Step arena's 1",
        );
        let mut out = [0u32; 1];
        ok &= refused(
            "a chain with an own walk",
            nextn_chain(
                m,
                f(&toks[..1], 0, host(1)),
                1,
                NextnHead::Full,
                NextnMode::Eager,
                &mut out,
            ),
            "a chain of 1 own walks",
        );
        ok &= refused(
            "a chain in the store walk's mode",
            nextn_chain(
                m,
                f(&toks[..1], 0, host(1)),
                0,
                NextnHead::Full,
                NextnMode::Store,
                &mut out,
            ),
            "a chain in the store walk's mode",
        );
        ok &= refused(
            "a chain into no place",
            nextn_chain(
                m,
                f(&toks[..1], 0, host(1)),
                0,
                NextnHead::Full,
                NextnMode::Eager,
                &mut [],
            ),
            "into 0 places",
        );
        let held = m
            .body("refusals")?
            .nextn()
            .map_or(usize::MAX, bloomery_gpu_glm5next::Nextn::held);
        let held_ok = held == 0;
        ok &= held_ok;
        println!(
            "(f) the store's positions after the refusals: {held} {}",
            verdict(held_ok)
        );
        ok &= by_position(m, &toks)?;
        Ok(ok)
    }

    /// (f) the target's rows by position, through the walk's own gather:
    /// the pair arena from a reset (no verify has written it), the step arena
    /// for a walk at position 0, then, after [`FED`] ids in one batch, the
    /// prompt-batch arena's row [`FED`] (past the batch's rows, inside its
    /// buffers) for the walk that follows, and its rows 0 and 1 for a walk
    /// from 5, which reads positions 4 and 5: each refused by name. The model
    /// left reset.
    fn by_position(m: &mut Glm5nextModel, toks: &[u32]) -> Result<bool, GateError> {
        let target = |walk: GlmArena, first: usize| NextnHidden::Target { walk, first };
        m.reset()?;
        let mut ok = refused(
            "the pair arena before any verify",
            nextn_hidden(m, 1, target(GlmArena::Pair, 0), 1),
            "of the Pair arena as positions 0..1 for a walk from 1: the arena holds no position",
        );
        ok &= refused(
            "the step arena for a walk at position 0",
            nextn_hidden(m, 0, target(GlmArena::Step, 0), 1),
            "for a walk at position 0",
        );
        set_prefill(m, PrefillMode::Batch)?;
        let ids: Vec<u32> = toks.iter().copied().cycle().take(FED).collect();
        feed(m, &ids)?;
        let fed = FED as u32;
        ok &= refused(
            "a prompt-batch row past the batch's last",
            nextn_hidden(m, fed + 1, target(GlmArena::Prefill, FED), 1),
            &format!(
                "for a walk from {}: the arena holds positions 0..{FED}",
                FED + 1
            ),
        );
        ok &= refused(
            "prompt-batch rows read as other positions",
            nextn_hidden(m, 5, target(GlmArena::Prefill, 0), 2),
            "as positions 4..6",
        );
        let good = nextn_hidden(m, fed, target(GlmArena::Prefill, FED - 1), 1);
        let good_ok = good.is_ok();
        ok &= good_ok;
        println!(
            "(f) the batch's last row for the walk after it: {} {}",
            match &good {
                Ok(v) => format!("{} values", v.len()),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(good_ok)
        );
        m.reset()?;
        Ok(ok)
    }

    /// Write `bytes` over the first value of the NextN layer's
    /// `shared_head_norm` gain and return the bytes it replaced.
    fn patch_head_norm(
        m: &mut Glm5nextModel,
        index: usize,
        bytes: [u8; 4],
    ) -> Result<[u8; 4], GateError> {
        let name = names::nextn_shared_head_norm(index);
        let nx = m
            .body("patch_head_norm")?
            .nextn()
            .ok_or("the NextN load holds no NextN layer")?;
        let Some(DevWeight::F32 { w: gain, .. }) = nx.weights().get(&name) else {
            return Err(format!("{name} is not resident as F32").into());
        };
        patch_bytes(m.gpu().stream(), gain.buf(), 0, bytes)
    }

    /// (h): a NaN gain in the head's norm makes the chain's logits NaN; the
    /// chain is a fault by name and the model poisoned by it at once, a walk
    /// after it refused as poisoned; the gain put back and a reset, the
    /// chain's id is the clean one.
    fn chain_fault(m: &mut Glm5nextModel, index: usize, hidden: usize) -> Result<bool, GateError> {
        let zeros = vec![0.0f32; hidden];
        let tok = [1u32];
        let feed = || f(&tok, 0, NextnHidden::Host(&zeros));
        let mut out = [0u32; 1];
        m.reset()?;
        nextn_chain(m, feed(), 0, NextnHead::Full, NextnMode::Eager, &mut out)?;
        let clean = out[0];
        m.reset()?;
        let old = patch_head_norm(m, index, f32::NAN.to_le_bytes())?;
        let first =
            nextn_chain(m, feed(), 0, NextnHead::Full, NextnMode::Eager, &mut out).map(|_| out[0]);
        let poisoned = m.poisoned();
        let walk = nextn_walk(m, feed(), NextnHead::Full, NextnMode::Store);
        patch_head_norm(m, index, old)?;
        m.reset()?;
        let again =
            nextn_chain(m, feed(), 0, NextnHead::Full, NextnMode::Eager, &mut out).map(|_| out[0]);
        m.reset()?;
        let named = match &first {
            Err(GpuError::Fault { fault, .. }) => poisoned == Some(*fault),
            _ => false,
        };
        let walk_refused = matches!(walk, Err(GpuError::Poisoned { .. }));
        let ok = named && walk_refused && matches!(again, Ok(t) if t == clean);
        let shown = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("id {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        println!(
            "(h) NaN in the NextN head norm's gain, a chain: {} (want a fault), the model poisoned \
             by {} (want the chain's fault), a walk after it: {}; the gain put back and a reset: \
             {} (want the clean {clean}) {}",
            shown(&first),
            poisoned.map_or_else(|| "none".to_string(), |f| f.to_string()),
            match &walk {
                Ok(()) => "accepted".to_string(),
                Err(e) => format!("error \"{e}\""),
            },
            shown(&again),
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------- (z) two slots in one pass

    /// (z)'s drafted rounds a window: the e2e gate's (sd) passes a slot.
    const Z_ROUNDS: usize = 8;

    /// (z)'s load: a slot's context and the sequences its plan counts, the
    /// e2e gate's slots load's.
    const Z_CTX: usize = 256;
    const Z_SLOTS: usize = 2;

    /// The rows of a window's verify at the seat's depth: the token at the
    /// slot's position, then the proposal ([`SLOT_DEPTH`] ids).
    const Z_ROWS: usize = 1 + SLOT_DEPTH;

    /// One drafted window's run: its ids (the first step's, then each
    /// round's kept ids), every round's [`Committed`], its draft store and
    /// its position at its end.
    struct ZRun {
        ids: Vec<u32>,
        rounds: Vec<Committed>,
        store: (Vec<u16>, Vec<u16>),
        pos: u32,
    }

    /// The selected slot's draft store, read back at the count its own walks
    /// hold: the rows the sequence's walks wrote.
    fn z_store(s: &mut Session<Body>) -> Result<(Vec<u16>, Vec<u16>), GateError> {
        let n = s
            .model()
            .body("(z)")?
            .nextn()
            .ok_or("(z)'s load holds no NextN layer")?
            .held();
        Ok(nextn_store(s.model_mut(), n)?)
    }

    /// The selected slot's window from its reset, as the server feeds it and
    /// the e2e gate's (sd) starts: every id of `ids` but the last through
    /// the draft's prompt call, then the last one drafted step; its id.
    fn z_start(
        s: &mut Session<Body>,
        d: &mut MtpDraft<Body>,
        ids: &[u32],
    ) -> Result<Vec<u32>, GateError> {
        s.reset()?;
        d.restart();
        let (&last, head) = ids.split_last().ok_or("an empty prompt")?;
        Draft::prompt(d, s, head)?;
        d.before_step(s, last)?;
        let next = Target::step(s, last, Want::Argmax)?.argmax();
        Draft::stepped(d, s, last, next)?;
        Ok(vec![next])
    }

    /// Window `ids` alone on the selected slot, [`Z_ROUNDS`] drafted rounds
    /// through the session's verify of one sequence, each proposal capped at
    /// the seat's depth: the proposal, the verify of its rows, the draft told
    /// the rule's kept rows, the commit.
    fn z_alone(s: &mut Session<Body>, ids: &[u32]) -> Result<ZRun, GateError> {
        let mut d = MtpDraft::open(s.model(), PrefillMode::Batch, StepMode::Eager)?;
        let mut out = z_start(s, &mut d, ids)?;
        let mut rounds = Vec::with_capacity(Z_ROUNDS);
        for _ in 0..Z_ROUNDS {
            let last = *out.last().ok_or("a window with no id")?;
            let pos = s.model().pos();
            let mut rows = [last; Z_ROWS];
            let n = Draft::propose(&mut d, s, last, &mut rows[1..])?;
            if n != SLOT_DEPTH {
                return Err(
                    format!("the draft proposed {n} ids; a round reads {SLOT_DEPTH}").into(),
                );
            }
            let got = Verify::verify(s, rows)?;
            let kept = accepted_rows(&rows, &got);
            d.record(pos, &rows, &got, kept)?;
            Verify::commit(s, kept)?;
            out.extend_from_slice(&got[..kept]);
            rounds.push(Committed {
                pos,
                kept,
                rows: Z_ROWS,
                proposed: true,
            });
        }
        Ok(ZRun {
            ids: out,
            rounds,
            store: z_store(s)?,
            pos: s.model().pos(),
        })
    }

    /// (z) (module doc): the two windows alone on the load's one sequence,
    /// then on their own slots, every round one pass of both
    /// ([`pass_slots`]).
    fn slots_pass(
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        levers: &bloomery_levers::Levers,
        windows: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let file = glm5next_tier::open()?;
        let mut machine = crate::gate_card::plan_gate(inputs.model.layers);
        reserve_checkpoints(&mut machine, glm_seqs(Z_SLOTS));
        let place = glm5next_tier::plan_levers(levers, 0)?;
        let np =
            inputs.plan_nextn_slots(&machine, u64::try_from(Z_CTX)?, &place, nextn, Z_SLOTS)?;
        let t = Instant::now();
        let mut m = Body::open_placed_nextn_slots(
            file,
            &np,
            inputs,
            nextn,
            0,
            levers.host(),
            Residency::Off,
            Z_SLOTS,
        )?;
        set_prefill(&mut m, PrefillMode::Batch)?;
        m.set_mode(StepMode::Graph);
        println!(
            "(z) the slots load: ctx {Z_CTX}, {Z_SLOTS} sequences, {} resident bytes in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        let mut s = Session::from_model(m, u32::try_from(Z_CTX)?);
        // The references, each alone on the load's one sequence before any
        // slot exists: what the seat's one-request runs produce.
        let alone = [z_alone(&mut s, windows[0])?, z_alone(&mut s, windows[1])?];
        s.add_slots(Z_SLOTS)?;
        let mut drafts = [
            MtpDraft::open(s.model(), PrefillMode::Batch, StepMode::Eager)?,
            MtpDraft::open(s.model(), PrefillMode::Batch, StepMode::Eager)?,
        ];
        let mut outs = [Vec::new(), Vec::new()];
        for (slot, ((d, out), ids)) in drafts.iter_mut().zip(&mut outs).zip(windows).enumerate() {
            s.select_slot(slot)?;
            *out = z_start(&mut s, d, ids)?;
        }
        let mut rounds = [Vec::new(), Vec::new()];
        for _ in 0..Z_ROUNDS {
            let [d0, d1] = &mut drafts;
            let [o0, o1] = &mut outs;
            let (l0, l1) = (
                *o0.last().ok_or("slot 0 gave no id")?,
                *o1.last().ok_or("slot 1 gave no id")?,
            );
            let mut w = [
                SlotWindow {
                    slot: 0,
                    draft: d0,
                    last: l0,
                    depth: SLOT_DEPTH,
                    out: o0,
                },
                SlotWindow {
                    slot: 1,
                    draft: d1,
                    last: l1,
                    depth: SLOT_DEPTH,
                    out: o1,
                },
            ];
            let done = pass_slots(&mut s, &mut w)?;
            let [c0, c1] = <[Committed; 2]>::try_from(done)
                .map_err(|d| format!("a pass of 2 slots committed {} slots", d.len()))?;
            rounds[0].push(c0);
            rounds[1].push(c1);
        }
        let mut off = Vec::new();
        for (slot, ((out, rs), r)) in outs.iter().zip(&rounds).zip(&alone).enumerate() {
            s.select_slot(slot)?;
            let parts = [
                ("ids", *out == r.ids),
                ("kept rounds", *rs == r.rounds),
                ("draft store", z_store(&mut s)? == r.store),
                ("position", s.model().pos() == r.pos),
            ];
            off.extend(
                parts
                    .into_iter()
                    .filter(|&(_, same)| !same)
                    .map(|(part, _)| format!("slot {slot} {part}")),
            );
        }
        let rejected = rounds.iter().flatten().filter(|c| c.kept < c.rows).count();
        let apart = rounds[0]
            .iter()
            .zip(&rounds[1])
            .filter(|(x, y)| x.kept != y.kept)
            .count();
        let kept = |rs: &[Committed]| rs.iter().map(|c| c.kept).collect::<Vec<_>>();
        let same = off.is_empty();
        // Two slots keeping different counts in a round needs a draft that is right on one and
        // wrong on the other, which the file's draft does and a fixture's random layer does not.
        let uneven = tier::premise(
            "(z) a round keeps different counts on the two slots",
            Tag::FileBound,
            apart > 0,
        )?;
        let ok = same && rejected > 0 && uneven;
        println!(
            "(z) two slots in one pass: {Z_ROUNDS} rounds, each window's draft its own and its \
             proposal capped at the seat's depth {SLOT_DEPTH} (a pass of {} rows), kept slot 0 \
             {:?}, slot 1 {:?}: {}; {rejected} slot rounds kept fewer rows than they ran, \
             {apart} rounds kept different counts on the two slots {}",
            2 * Z_ROWS,
            kept(&rounds[0]),
            kept(&rounds[1]),
            if same {
                "each window's ids, kept rounds, draft store and position its alone run's"
                    .to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(ok)
        );
        Ok(ok)
    }

    pub(super) fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&tier::acts_on(&[])?)?;
        crate::gate_card::init()?;
        glm5next_tier::init_file()?;
        // The prompt is the prose corpus's first ids in both tiers (the set's own, held equal to
        // them, in the real tier); the set itself is read by the Oracle clauses alone.
        let prompt = glm5next_tier::mtp_prompt()?;
        let set = if Tier::from_env()? == Tier::Real {
            Some(glm5next_tier::mtp_set()?)
        } else {
            None
        };
        if let Some(set) = set {
            let accepted: usize = set.verify.iter().map(|v| v.accepted).sum();
            println!(
                "{}: {} graphs over {} blocks ({accepted} accepted), prompt {} ids, family {}",
                set.dir.display(),
                graphs_of(set).len(),
                set.blocks().len(),
                prompt.len(),
                refset::arch::glm5next::MTP.name
            );
        }
        println!(
            "bands: the logits' (held, and the argmax cap's) {:.4e}",
            logits_band()
        );
        let open = glm5next_tier::open;
        let file = open()?;
        let inputs = PlanInputs::read(&file)?;
        let nextn = NextnInputs::read(&inputs)?;
        // The fixture's band is counted over the layer's own tensors, so it is read from the file
        // the loads open.
        let fx_band = if Tier::from_env()? == Tier::Fixture {
            Some(fixture_band(&file, nextn.index)?)
        } else {
            None
        };
        let mut machine = crate::gate_card::plan_gate(inputs.model.layers);
        reserve_checkpoints(&mut machine, glm_seqs(1));
        let place = glm5next_tier::plan_levers(&levers, 0)?;
        let ctx = u64::try_from(CTX)?;
        let base = inputs.plan(&machine, ctx, &place)?;
        let np = inputs.plan_nextn(&machine, ctx, &place, &nextn)?;
        let card = |n_l: &[u64]| n_l.iter().sum::<u64>();
        println!(
            "plan: layer {} beside the target, its card bytes {} and arena {}; card experts {} \
             without it, {} with it",
            nextn.index,
            np.nextn_card_bytes(),
            np.arena_bytes,
            card(&base.n_l),
            card(&np.plan.n_l)
        );
        drop(file);

        // (t) the target's plan loaded without the layer: the reference run,
        // at the two KDA lanes the plan counts.
        let t = Instant::now();
        let mut m = Body::open_placed_lanes(
            open()?,
            &np.plan,
            &inputs,
            0,
            levers.host(),
            Residency::Off,
            KdaLanes::Two,
        )?;
        m.set_mode(StepMode::Graph);
        println!(
            "load without the layer: {:.1} s, {} resident bytes",
            t.elapsed().as_secs_f64(),
            m.resident_bytes()
        );
        let hidden = m.body("run")?.nextn_hidden_width();
        tier::sc("(f) refusals on a load without the layer")?;
        let mut ok = refused(
            "a walk on a load without the layer",
            nextn_walk(
                &mut m,
                NextnFeed {
                    tokens: &[1],
                    pos0: 0,
                    hidden: NextnHidden::Host(&vec![0.0; hidden]),
                },
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "without the NextN layer",
        );
        ok &= refused(
            "the MTP window over a load without the layer",
            MtpDraft::<Body>::open(&m, PrefillMode::Batch, StepMode::Eager).map_err(|e| {
                GpuError::Shape {
                    what: "MtpDraft::open",
                    detail: e.to_string(),
                }
            }),
            "an MTP draft on a load without the NextN layer",
        );
        tier::sc("(t) the NextN load's plain run is the load without the layer's")?;
        let reference = plain(&mut m, &prompt, PrefillMode::Batch)?;
        // (s) this load's sequence state, for the NextN load to refuse.
        tier::sc("(s) a sequence state across the two layouts, and seq_bytes")?;
        let without = seq_save(&mut m)?;
        // The state a load really holds pins the plan-time formula of its
        // bytes (`seq_bytes`, the elastic `--parallel` default's unit) at the
        // state's own positions and its draft setting — here the load
        // without the layer, whose state carries no draft side.
        let without_at = usize::try_from(without.positions())?;
        let without_bytes = seq_bytes(&inputs, without_at, false);
        ok &= without_bytes == without.bytes() as u64;
        println!(
            "(s) the state's {} B = seq_bytes at {without_at} positions without the draft: {}",
            without.bytes(),
            verdict(without_bytes == without.bytes() as u64)
        );
        drop(m);

        // The NextN load.
        let t = Instant::now();
        let mut m = Body::open_placed_nextn(open()?, &np, &inputs, &nextn, 0, levers.host())?;
        m.set_mode(StepMode::Graph);
        let (res, arena) = m
            .body("run")?
            .nextn()
            .map(|n| (n.resident_bytes(), n.arena_bytes()))
            .ok_or("the NextN load holds no NextN layer")?;
        println!(
            "load with layer {}: {:.1} s, {} resident bytes, the layer's {res} and its arena \
             {arena}",
            nextn.index,
            t.elapsed().as_secs_f64(),
            m.resident_bytes()
        );

        ok &= layout_refused(&mut m, &without);
        drop(without);
        if tier::run_clause(
            "(m) the set's MTP input: no position mask, no zero row",
            Tag::Oracle,
        )? {
            ok &= pos_mask(glm5next_tier::mtp_set()?, hidden)?;
        }
        if tier::run_clause(MFX, Tag::FixtureOracle)? {
            ok &= fixture_verdict(
                "(mfx)",
                glm5next_tier::fx_mtp_set(MFX).and_then(|set| pos_mask(&set, hidden)),
            );
        }
        tier::sc("(f) the walk's refusals on the NextN load")?;
        ok &= refusals(&mut m, hidden)?;
        tier::sc("(h) a fault in a chain poisons the model")?;
        ok &= chain_fault(&mut m, nextn.index, hidden)?;
        tier::sc("(p) the pairing: the hidden rows a walk reads")?;
        ok &= pairing(&mut m, &prompt, hidden, inputs.hp.rms_eps)?;
        if tier::run_clause(
            "(o) teacher-forced: every graph of the set replayed against ik's",
            Tag::Oracle,
        )? {
            ok &= oracle(&mut m, glm5next_tier::mtp_set()?, hidden, Rule::Real)?;
        }
        if tier::run_clause(OFX, Tag::FixtureOracle)? {
            ok &= fixture_verdict(
                "(ofx)",
                glm5next_tier::fx_mtp_set(OFX).and_then(|set| {
                    let band = fx_band.ok_or("the fixture band is read in the fixture tier")?;
                    oracle(&mut m, &set, hidden, Rule::Fixture(band))
                }),
            );
        }

        let with = plain(&mut m, &prompt, PrefillMode::Batch)?;
        let ids_same = with.ids == reference.ids;
        let steps_same = with.steps == reference.steps;
        ok &= ids_same && steps_same;
        println!(
            "(t) the NextN load's plain run = the load without the layer: {N} ids {}, {} steps' \
             logits bit for bit {}",
            verdict(ids_same),
            with.steps.len(),
            verdict(steps_same)
        );
        if tier::run_clause(
            "(t) ik's committed stream beside the plain run (printed, not held)",
            Tag::Oracle,
        )? {
            print_stream(glm5next_tier::mtp_set()?, &with.ids);
        }
        if tier::run_clause(TFX_P, Tag::FixtureOracle)? {
            ok &= fixture_verdict(
                "(tfx-p)",
                glm5next_tier::fx_mtp_set(TFX_P).map(|set| {
                    print_stream(&set, &with.ids);
                    true
                }),
            );
        }

        // Each path against its own plain run: a prompt batch past a chunk
        // runs the GEMM, so the two paths' plain runs are not one run's bits.
        let by_steps = plain(&mut m, &prompt, PrefillMode::Steps)?;
        let mut kept_last = None;
        for (path, plain) in [(PrefillMode::Batch, &with), (PrefillMode::Steps, &by_steps)] {
            tier::sc(&format!(
                "(w) the drafted windows on the {path:?} feed are the plain run's"
            ))?;
            let (model, w_ok, kept) = drafted(m, &prompt, path, plain, set)?;
            m = model;
            ok &= w_ok;
            kept_last = kept;
        }
        // The drafted load's own state pins the formula's draft side: the
        // last window's verify behind a step leaves both arenas at the rows
        // the formula's draft term counts. A window that keeps one row leaves the pair arena
        // that row short, which the fixture tier's random draft does (it rejects every proposal);
        // the real tier holds the formula as it stands.
        tier::sc("(s) the drafted load's state = seq_bytes with the draft")?;
        let with_state = seq_save(&mut m)?;
        let with_at = usize::try_from(with_state.positions())?;
        let arena_row = (inputs.hp.hc.streams * inputs.hp.n_embd * size_of::<f32>()) as u64;
        let short = match Tier::from_env()? {
            Tier::Real => 0,
            Tier::Fixture => M - kept_last.unwrap_or(M).min(M),
        };
        let with_bytes = seq_bytes(&inputs, with_at, true) - short as u64 * arena_row;
        let state_ok = with_bytes == with_state.bytes() as u64;
        ok &= state_ok;
        println!(
            "(s) the drafted load's state {} B = seq_bytes at {with_at} positions with the \
             draft, {short} pair row(s) short: {}",
            with_state.bytes(),
            verdict(state_ok)
        );
        m.reset()?;
        // (z) on a load of its own: this one's plan counts one sequence, and
        // the card holds one load.
        drop(m);
        let prefill = glm5next_tier::d1k_prefill()?;
        if prefill.len() < 340 {
            return Err(format!("{D1K}: {} prefill ids, (z) reads 340", prefill.len()).into());
        }
        let windows = [&prefill[..33], &prefill[300..340]];
        tier::sc("(z) two windows' drafted rounds as one pass of both slots")?;
        ok &= slots_pass(&inputs, &nextn, &levers, windows).unwrap_or_else(|e| {
            println!("(z) a call failed: {e} {}", verdict(false));
            false
        });
        println!("gate_glm5next_mtp: {}", tier::tally_line());
        tier::expect_fixture_oracle("gate_glm5next_mtp", FIXTURE_ORACLE)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
