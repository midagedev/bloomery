//! The MiniMax-M3.1-Flash prefill gate: the prompt batch (`bloomery_gpu_minimaxm3::prefill`: up to `T_MAX` rows a unit, every
//! layer over the whole unit, the index keys of the unit's rows before any row is scored, the block-max selection over
//! each row's own live count, the routed block over the unit's slots) is the one-token step run per id to within the
//! family's own band, in the logits row and in every layer's output, and the stores it leaves are the ones the next step
//! reads; the prompt group changes no bit.
//!
//! The batch keeps its own arithmetic (the wide GEMM's sum order, where the step's sites read one column a gemv), so it
//! is the steps' within a band, not bit for bit. The band is the family's own: the forced arm's per-layer error over the
//! layers its prefix covers ([`seat::prefill_ceil`], the error model of the end-to-end gate's forced arm), and no ceiling
//! over the layers from that prefix on: a top-4 near tie can flip between the batch's GEMM sum order and the step's gemv,
//! and a flip is a discrete exchange of experts, not a rounding of the activations.
//!
//! The prompt lengths straddle the unit and the shared expert's gated threshold: `P = 1` (a unit of one row, the step's
//! shape), `7` (rows short of the gated GEMMs), `16` and `17` (rows past it, short of a unit), `T_MAX` (one whole unit),
//! `T_MAX + 1` (a whole unit and a unit of one row: the K and V planes and the index keys cross the cut between two
//! units) and `2 · T_MAX + 1` (two whole units, which the prompt group runs together, and a unit of one row; the flash
//! of the dense and the sparse layers runs in more than one launch in each unit). The ids are the fixed walk of
//! `e2e::walk`, none of the oracle's: the clauses compare the engine's two paths to each other and need no reference.
//!
//! The clauses (all self-consistency) are the common ones ([`prefill_clauses`]): `prefill_near_steps`,
//! `prefill_then_step_near_steps`, `prefill_group_is_group_one` and `fault_word_clean`, and the family's own,
//! `prefill_batch_forms`, which runs first and on no file: the batch's wide pools quantize the input of every projection
//! type the loader accepts (each type alone at both widths, then a mix), so a model whose projections are K-quants
//! prefills as the checkpoint folder's MXFP8 ones do; and `prefill_units_full`, after the common ones on a single card:
//! a prompt of `2 · T_MAX` ids at the default prompt group is one walk of two whole units, and each routed layer's
//! host leg serves each unit once over all its `T_MAX` rows (the walk's stats), so the host reads a layer's experts
//! once a unit of `T_MAX` rows and not once a smaller cut.
//!
//! The real tier's model is the checkpoint folder the end-to-end gate reads; the fixture tier's is its fixture cut.
//!
//! The stream arm (`BLOOMERY_RESIDENCY=mid-p<P>-s<S> BLOOMERY_XSTREAM=split`, a plan that leaves host experts): the seat
//! sets the body's host streaming at the load and the gate drives the stream rule to stream ([`SPLIT_COSTS`]), so the
//! common clauses' prompt calls run streamed experts on the card over the ring's slots while the steps run them on the
//! host — `prefill_near_steps` holds the two to the family's band. Then `prefill_stream_streamed`: a prompt of
//! [`STREAM_P`] ids streamed an expert on some layer (its stream records), or the arm proved nothing and fails by name.
//!
//! The two-card arm (`--arm bp`, `just gate-gpu-minimaxm3-twocard` under `BLOOMERY_CARD=both`): plan (b′) (the A6000 the
//! stage card, the 3090 an expert tier holding each layer's next ids after the stage's prefix, the host the rest), planned
//! under one card budget no environment moves (`seat::TWO_BUDGET`), in the real tier only. Before the common clauses it holds, against the reference — the plan with every tier segment on the stage card
//! (`model::placement::union_plan`), the A6000 alone, opened first and released:
//! - `two_card_load`: the plan puts experts on the tier, and the load's tier opened on the 3090 beside the stage card with
//!   the plan's experts on each layer;
//! - `two_card_first_tokens_are_union`: [`TWO_WALK`] decode steps of the walk read the reference's ids and logits rows bit
//!   for bit — the routed block joins the tier's slots in the slot order the union's card sum runs (the same join the
//!   Solar-Open2 two-card gate pins) — and the walk sent every tier layer a routed slot (or the run proves nothing there and
//!   fails by name);
//! - `two_card_eager_is_replay`: the walk eager and as the captured graph's replays reads the same ids and logits bits.
//!
//! The common clauses then run on the two-card load, and the tier must have served their batches (`two_card_batch_sent`).

#[cfg(feature = "minimaxm3")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minimaxm3_prefill", gate::run())
}

#[cfg(feature = "minimaxm3")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "minimaxm3")]
#[path = "shared/xstream_rule.rs"]
mod xstream_rule;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the prefill clauses serve several gates; each takes the helpers they read"
)]
#[path = "shared/prefill_clauses.rs"]
mod prefill_clauses;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/minimaxm3_seat.rs"]
mod seat;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the two-card gates share the module; each takes the helpers it reads"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;

#[cfg(feature = "minimaxm3")]
mod gate {
    use crate::lcpp_e2e::{Arm, Dims as SeatDims, Seat, arm_arg, model_path};
    use crate::prefill_clauses::Case;
    use crate::seat;
    use crate::tier_twocard::{
        Leg, Sent, same, sent_between, sent_every, tier_counts, tier_on_3090,
    };
    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::hybrid::Site;
    use bloomery_gpu::flash_gqa::HEAD;
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::host::xstream::{Costs, XMode};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::site::{SiteTy, WideActs, WideKernels};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_minimaxm3::MiniMaxM3Model;
    use bloomery_gpu_minimaxm3::attn::MsaShape;
    use bloomery_gpu_minimaxm3::body::{Dims, INDEX, PROJ};
    use bloomery_gpu_minimaxm3::prefill::{
        Batch, T_MAX, set_prefill_group, set_prompt_stats, take_prompt_stats,
    };
    use bloomery_gpu_minimaxm3::stream::XSTREAM_COSTS;
    use cuda_core::DeviceBuffer;
    use refset::arch::minimaxm3::ARCH;

    const NAME: &str = "gate_minimaxm3_prefill";

    /// The recipe the file is the one of.
    const RECIPE: &str = "gate-gpu-minimaxm3-prefill";

    /// The prompt lengths: one row, rows short of the gated GEMMs, rows past them and short of a unit, one whole
    /// unit, a unit and a row, two units and a row.
    const LENGTHS: [usize; 7] = [1, 7, 16, 17, T_MAX, T_MAX + 1, LONGEST];

    /// The longest prompt: two whole units and a unit of one row.
    const LONGEST: usize = 2 * T_MAX + 1;

    /// Positions of the cache beyond the longest prompt and the step after it.
    const CTX_SPARE: usize = 8;

    /// Decode steps of the two-card arm's walk: each layer's tier holds the ids after the stage's prefix, so the walk's
    /// routing must reach every tier layer (`two_card_first_tokens_are_union` fails by name where it does not). The real
    /// file's tier holds few experts on some layers, so the walk is long enough to route a slot to each.
    const TWO_WALK: usize = 96;

    /// The synthetic batch's widths: the model width, and the output projection's (the query heads of [`HEAD`] values).
    /// Both are whole blocks of the q8_1 form's 128 values and of the q8 form's 64-value step.
    const EMBD: usize = 6144;
    const QUERY_HEADS: usize = 64;

    /// The clause `prefill_batch_forms`: whether a batch cut for the sites it is told of quantizes the input of each of
    /// them. Each type the loader accepts is run alone at both widths (the projections' input at the model width, the
    /// output projection's at the query width), then a mix of the Q8_0 and K-quant types at both. A pool that holds no
    /// form a site's type reads refuses the quantize by name; each refusal is a line and fails the clause.
    fn batch_forms() -> Result<bool, GateError> {
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let d = Dims {
            embd: EMBD,
            vocab: 1,
            heads: QUERY_HEADS,
            kv_heads: 4,
            rms_eps: 1.0e-6,
            rope_base: 1.0e7,
            ctx: 256,
        };
        let q_width = QUERY_HEADS * HEAD;
        let shape = MsaShape::new(16, 1)?;
        let g32 = Gemm32Kernels::load(gpu.context())?;
        let x = DeviceBuffer::<f32>::zeroed(stream, T_MAX * q_width)?;
        let accepted = INDEX
            .iter()
            .chain(PROJ.iter().filter(|t| !INDEX.contains(t)))
            .map(|&t| (format!("{t:?}"), vec![t], vec![t]));
        let mixed = (
            "mixed".to_string(),
            vec![SiteTy::Q8_0, SiteTy::Q4K, SiteTy::F32],
            vec![SiteTy::Q6K, SiteTy::Q8_0, SiteTy::Q5K],
        );
        let mut ok = true;
        for (label, at_embd, at_query) in accepted.chain([mixed]) {
            let sites: Vec<Site> = at_embd
                .iter()
                .map(|&t| Site::new(format!("{label}.embd"), t, EMBD))
                .chain(
                    at_query
                        .iter()
                        .map(|&t| Site::new(format!("{label}.query"), t, q_width)),
                )
                .collect();
            let refs: Vec<&Site> = sites.iter().collect();
            let mut batch = Batch::new(&gpu, (&d, &shape), T_MAX, &refs)?;
            let wk = WideKernels {
                gemm: &batch.gemm,
                g32: Some(&g32),
            };
            let mut quantize = |pool: &mut WideActs, tys: &[SiteTy]| -> Result<String, GateError> {
                Ok(
                    match pool.quantize(&gpu, &wk, gpu.layer_sink(0)?, &x, T_MAX, tys) {
                        Ok(()) => "ok".to_string(),
                        Err(e) => {
                            ok = false;
                            e.to_string()
                        }
                    },
                )
            };
            let model = quantize(&mut batch.wide, &at_embd)?;
            let output = quantize(&mut batch.arm, &at_query)?;
            println!(
                "prefill_batch_forms: {label}: the model width's pool {model}; the output width's pool {output}"
            );
        }
        stream.synchronize()?;
        println!(
            "prefill_batch_forms: every type the loader accepts, alone and mixed {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// What the tier was sent since `m` loaded; `None` on a load with no tier.
    fn counts(m: &MiniMaxM3Model) -> Result<Option<Sent>, GateError> {
        Ok(tier_counts(m.body(NAME)?.hybrid()))
    }

    /// `ids` one decode step each from a reset, every step's id and logits row.
    fn feed(m: &mut MiniMaxM3Model, ids: &[u32]) -> Result<Leg, GateError> {
        m.reset()?;
        let mut leg = Leg::default();
        for &id in ids {
            leg.tokens.push(m.step(&[id])?);
            leg.logits.push(m.logits()?);
        }
        Ok(leg)
    }

    /// The two-card arm's own clauses (module doc) on `two`, which [`seat::two_card`] loaded; the load is left as
    /// [`seat::open`] leaves one, the step captured. `Ok((pass, tier layers))`.
    fn two_card_clauses(two: &mut seat::TwoCard) -> Result<(bool, Vec<usize>), GateError> {
        let tier_layers: Vec<usize> = (0..two.planned.len())
            .filter(|&l| two.planned[l] > 0)
            .collect();
        let m = two.loaded.model_mut();
        let mut pass = true;

        tier::sc("two_card_load")?;
        tier_on_3090(
            "arm bp load",
            two.opened,
            (&m.gpu().device_name()?, m.resident_bytes()),
            m.body(NAME)?.hybrid(),
        )?;
        let loaded: Vec<u64> = {
            let hybrid = m.body(NAME)?.hybrid();
            let set = hybrid.tiers().first().map(|t| t.set());
            (0..two.planned.len())
                .map(|l| set.and_then(|s| s.ids(l)).map_or(0, |ids| ids.len() as u64))
                .collect()
        };
        let load_ok = !tier_layers.is_empty() && loaded == two.planned;
        println!(
            "two_card_load: the tier holds experts on {} layers; the load's tier experts by layer {} the plan's {}",
            tier_layers.len(),
            if loaded == two.planned {
                "are"
            } else {
                "are not"
            },
            verdict(load_ok)
        );
        if loaded != two.planned {
            println!("  loaded {loaded:?}\n  planned {:?}", two.planned);
        }
        pass &= load_ok;

        tier::sc("two_card_first_tokens_are_union")?;
        m.set_mode(StepMode::Graph);
        let before = counts(m)?;
        let replay = feed(m, &two.walk)?;
        let sent = sent_between(before, counts(m)?);
        let want = Leg {
            tokens: two.reference.0.clone(),
            logits: two.reference.1.clone(),
            kept: Vec::new(),
        };
        pass &= same("two_card_first_tokens_are_union", &want, &replay);
        pass &= sent_every(
            "two_card_first_tokens_are_union walk",
            sent.as_ref(),
            &tier_layers,
            false,
        );

        tier::sc("two_card_eager_is_replay")?;
        m.set_mode(StepMode::Eager);
        let eager = feed(m, &two.walk)?;
        pass &= same("two_card_eager_is_replay", &eager, &replay);
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        m.reset()?;
        Ok((pass, tier_layers))
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        // Before the file loads: a pool the batch was cut wrong for fails in seconds, not after the load.
        if tier::run_clause("prefill_batch_forms", Tag::SelfConsistency)? && !batch_forms()? {
            println!("clause prefill_batch_forms FAIL");
            println!("{}", tier::tally_line());
            return Err(checks_failed());
        }
        let path = model_path(tier_now, RECIPE, ARCH)?;
        let ctx = LONGEST + CTX_SPARE;
        let t = std::time::Instant::now();
        let arm = arm_arg()?;
        if arm == Arm::Bp {
            return two_card_run(&path, (ctx, tier_now), t);
        }
        let mut seat = seat::open_loaded(&path, ctx, arm)?;
        let dims = seat.dims();
        let split = drive_split(&mut seat)?;
        println!(
            "load arm={} layers={} embd={} vocab={} ctx={ctx} t_max={T_MAX} xstream={} in {:.1} s (runtime value)",
            arm.word(),
            dims.layers,
            dims.embd,
            dims.vocab,
            seat.model().body(NAME)?.xstream_word(),
            t.elapsed().as_secs_f64()
        );
        let common = crate::prefill_clauses::run(NAME, &case(dims, tier_now), &mut seat);
        if split
            && tier::run_clause("prefill_stream_streamed", Tag::SelfConsistency)?
            && !stream_streamed(&mut seat, dims.vocab)?
        {
            println!("clause prefill_stream_streamed FAIL");
            println!("{}", tier::tally_line());
            return Err(checks_failed());
        }
        if tier::run_clause("prefill_units_full", Tag::SelfConsistency)?
            && !units_full(&mut seat, dims.vocab)?
        {
            println!("clause prefill_units_full FAIL");
            println!("{}", tier::tally_line());
            return Err(checks_failed());
        }
        common
    }

    /// The prompt length of `prefill_units_full`: two whole units, one walk of the default prompt group.
    const UNITS_P: usize = 2 * T_MAX;

    /// The prompt group `prefill_units_full` runs at: the lever's default, which a serving load runs.
    const UNITS_GROUP: usize = bloomery_levers::PREFILL_GROUP_DEFAULT as usize;

    /// The clause `prefill_units_full` (module doc): a prompt of [`UNITS_P`] ids of the fixed walk from a reset, at
    /// [`UNITS_GROUP`] units a walk, with the walk's timing armed. It holds the walk's stats to one walk of two units, and
    /// every routed layer to one serve a unit, each over `T_MAX` columns. A call that walked no routed layer proves
    /// nothing and fails by name. The seat leaves at group 1 with the timing disarmed, as it came.
    fn units_full(seat: &mut seat::Loaded, vocab: usize) -> Result<bool, GateError> {
        let ids = crate::e2e::walk(UNITS_P, vocab);
        seat.reset()?;
        set_prefill_group(seat.model_mut(), UNITS_GROUP)?;
        set_prompt_stats(seat.model_mut(), true)?;
        let fed = seat.prompt(&ids);
        let stats = take_prompt_stats(seat.model_mut());
        set_prompt_stats(seat.model_mut(), false)?;
        set_prefill_group(seat.model_mut(), 1)?;
        seat.reset()?;
        fed?;
        let Some(st) = stats? else {
            println!("prefill_units_full: a prompt of {UNITS_P} ids left no walk stats FAIL");
            return Ok(false);
        };
        let mut layers: Vec<usize> = st.rows.iter().map(|r| r.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        let serves = |l: usize, u: usize| {
            st.rows
                .iter()
                .filter(|r| r.layer == l && r.unit == u)
                .count()
        };
        let short: Vec<(usize, usize, usize)> = st
            .rows
            .iter()
            .filter(|r| r.cols != T_MAX)
            .map(|r| (r.layer, r.unit, r.cols))
            .collect();
        let once = layers
            .iter()
            .all(|&l| serves(l, 0) == 1 && serves(l, 1) == 1);
        let experts =
            st.rows.iter().map(|r| r.experts).sum::<usize>() as f64 / st.rows.len().max(1) as f64;
        let ok = st.walks == 1
            && st.ubatches == 2
            && !layers.is_empty()
            && st.rows.len() == 2 * layers.len()
            && once
            && short.is_empty();
        println!(
            "prefill_units_full: a prompt of {UNITS_P} ids at group {UNITS_GROUP}: {} walks, {} units, {} routed layers, \
             {} serves (one a unit a layer {}), units short of {T_MAX} columns {short:?}, {experts:.1} host experts a \
             serve {}",
            st.walks,
            st.ubatches,
            layers.len(),
            st.rows.len(),
            verdict(once),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The stream arm's prompt length: a group of two whole units, past the walk's gate at [`SPLIT_COSTS`].
    const STREAM_P: usize = 2 * T_MAX;

    /// The stream arm's rule costs: the family's, with a host column of a tenth of a second, so the stream's floor sits
    /// under one column and every host expert a unit routes streams until the ring's half or the balance cuts it.
    const SPLIT_COSTS: Costs = Costs {
        host_us_per_col: 1.0e5,
        ..XSTREAM_COSTS
    };

    /// Under `BLOOMERY_XSTREAM=split` (the seat set it at the load), the rule's costs the stream arm runs at
    /// ([`SPLIT_COSTS`]); whether the load streams. Nothing otherwise.
    fn drive_split(seat: &mut seat::Loaded) -> Result<bool, GateError> {
        let (_, _, body) = seat.model_mut().body_parts(NAME)?;
        if body.xstream_mode() != XMode::Split {
            return Ok(false);
        }
        body.set_xstream_costs(SPLIT_COSTS)?;
        Ok(true)
    }

    /// The clause `prefill_stream_streamed` (module doc): a prompt of [`STREAM_P`] ids of the fixed walk from a reset, and
    /// its stream records. A load whose plan leaves no host expert, or a ring the card holds too few slots for, streams
    /// nothing and fails here by name.
    fn stream_streamed(seat: &mut seat::Loaded, vocab: usize) -> Result<bool, GateError> {
        let ids = crate::e2e::walk(STREAM_P, vocab);
        seat.reset()?;
        seat.prompt(&ids)?;
        let (_, _, body) = seat.model_mut().body_parts(NAME)?;
        let (layers, call) = body.take_xstream_records();
        let streamed: usize = layers.iter().map(|(_, x)| x.streamed).sum();
        let mut on: Vec<usize> = layers
            .iter()
            .filter(|(_, x)| x.streamed > 0)
            .map(|(_, x)| x.layer)
            .collect();
        on.sort_unstable();
        on.dedup();
        let ok = streamed > 0;
        println!(
            "prefill_stream_streamed: a prompt of {STREAM_P} ids streamed {streamed} experts on layers {on:?} \
             (call {call:?}) {}",
            verdict(ok)
        );
        seat.reset()?;
        Ok(ok)
    }

    /// The clauses' case on a load of `dims` in tier `tier_now`.
    fn case(dims: SeatDims, tier_now: Tier) -> Case {
        Case {
            lengths: LENGTHS.to_vec(),
            tap_ceil: seat::prefill_ceil,
            tap_ceils: seat::prefill_ceils,
            group: 2,
            // A top-4 near tie can flip between the batch's GEMM sum order and the step's gemv, and
            // a flip is a discrete exchange of experts: the comparison is deferred, as the
            // end-to-end gate defers its top-logprobs clause.
            logit_bound: None,
            sigma_logit: seat::SIGMA_LOGIT,
            vocab: dims.vocab,
            embd: dims.embd,
            tier: tier_now,
        }
    }

    /// The two-card arm (module doc): its loads and own clauses, then the common clauses on the two-card load, which must
    /// have sent the tier their batches.
    fn two_card_run(
        path: &std::path::Path,
        (ctx, tier_now): (usize, Tier),
        t: std::time::Instant,
    ) -> Result<(), GateError> {
        // The walk's positions too: the reference and the two-card load hold the same cache rows.
        let ctx = ctx.max(TWO_WALK + 1);
        let mut two = seat::two_card(path, (ctx, tier_now), (TWO_WALK, crate::e2e::walk))?;
        let (pass, layers) = two_card_clauses(&mut two)?;
        let mut loaded = two.loaded;
        let dims = loaded.dims();
        println!(
            "load arm=bp layers={} embd={} vocab={} ctx={ctx} t_max={T_MAX} in {:.1} s (runtime value, both loads)",
            dims.layers,
            dims.embd,
            dims.vocab,
            t.elapsed().as_secs_f64()
        );
        let before = counts(loaded.model())?;
        let common = crate::prefill_clauses::run(NAME, &case(dims, tier_now), &mut loaded);
        tier::sc("two_card_batch_sent")?;
        let sent = sent_between(before, counts(loaded.model())?);
        let batch = sent_every("two_card_batch_sent", sent.as_ref(), &layers, true);
        common?;
        if !(pass && batch) {
            println!("arm bp FAIL");
            println!("{}", tier::tally_line());
            return Err(checks_failed());
        }
        Ok(())
    }
}
