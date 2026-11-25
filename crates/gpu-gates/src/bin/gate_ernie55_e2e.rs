//! The ERNIE 5.5 Flash Preview end-to-end gate: the whole program on one card (the embedding into
//! the four streams, the mHC pre and post around every sub-layer, the HCA and the MLA-DSA
//! attention, the dense SiTU lead and the latent routed blocks, the head's fold, the BF16 head and
//! its MultiMax epilogue) over the fixture cut (`fixture cut`: the released folder's layers 0, 1
//! and 7, an HCA layer with the dense lead, an HCA routed layer and an MLA-DSA routed layer),
//! against the numpy f32 forward's sets of that cut (`tools/ref/ernie55_ref.py`, refset family
//! `fx-np-ernie55`: `short`, 34 ids, and `long`, 2,328 ids, the one prompt past the indexer's
//! top-k). The model opens through `Body::open_placed` of `bloomery-gpu-ernie55` in arms of the
//! one placement rule:
//!
//! - `card`: the plan of the card in view, which holds every expert of the cut: the prompt runs as
//!   the pass's units (`bloomery_gpu_ernie55::prompt`), the steps as the captured chain;
//! - `hosted`: the same card under a card-experts cap of half the routed experts, so every routed
//!   layer serves the rest of its picks from the common host tier (the card leg's front, shadow
//!   and back around the tier's go and wait in a step, around the tier's batch port in the pass:
//!   a prompt unit's rows at once, each distinct host expert read once). It opens the cut's GGUF,
//!   the file `bloomery-quantize` writes of the folder, named by [`HOSTED_GGUF`]: the folder's
//!   MXFP4 expert stacks are made at load, so no load from the folder splits a stack between the
//!   card and the host. Without the argument the arm is refused by one printed line
//!   ([`HOSTED_REFUSED`]), and the gate's verdict is the card arm's;
//! - `hosted-mid` (the hosted arm's file, opened again once the `hosted` arm is closed): the
//!   residency a server runs by default ([`RESIDENT`], `mid-p0-s1`) over the same hosted load.
//!   `short` is fed once, which gives the machine a usage; every routed layer's card set is then
//!   reseeded (`SwapMachine::reseed`) to the experts that prompt picked first and the highest ids
//!   after, so the card takes experts the plan left on the host into slots below its card count,
//!   never the slot of their own id; `short`'s reference clauses run again over that layout, and
//!   then `xstream_admit_matches_off`.
//!
//! For each set and arm, against the writer's rows at the prompt's last positions:
//!
//! - `layer_outputs_are_reference`: each layer's four streams after it (`l_out-L`, the last four
//!   rows) within [`L_OUT_BAND`] by relative L2.
//! - `stepped_layer_outputs_are_reference` (every arm's prompt runs as the pass): the prompt fed
//!   again one id a step from a reset, the last step's four streams after each layer within
//!   [`L_OUT_BAND`] of the writer's last row, so each arm's step chain, the whole-card one and the
//!   hosted ones, is held layer by layer and not by the teacher-forced argmax alone.
//! - `picks_are_reference`: each routed layer's picks at those rows are the writer's ids
//!   (`ffn_moe_topk-L`) as a set, exactly, and their weights by id within [`WEIGHTS_BAND`] of
//!   `ffn_moe_weights-L` (renormalized, times the learned scale).
//! - `selection_is_reference` (`long` alone): the MLA-DSA layer's selected positions at the last
//!   row against the writer's `indexer_top_k-L`: at most [`SEL_SWAPS`] positions of either side
//!   outside the other.
//! - `head_is_reference`: the logits before MultiMax (`result_output_raw`) and after it
//!   (`result_output`) within [`LOGITS_BAND`], and both argmaxes the writer's.
//! - `forced_top1_is_reference`: the writer's greedy ids fed back one a step (teacher-forced),
//!   our argmax at each of the greedy table's positions against its id: the disagreements at a
//!   writer margin of [`MARGIN_FLOOR`] or more at most the pin the Gaussian error model gives at
//!   the σ̂ of the arm's own last prompt row ([`forced_pin`]).
//!
//! and the clauses that need no reference:
//!
//! - `prompt_runs_as_a_pass` (`hosted`, `hosted-mid`): after a prompt, the host tier's batch port
//!   served one layer-batch a hosted layer a unit ([`PASS_ROWS`] rows a unit), over the prompt's
//!   rows, listing host slots, and the step port served nothing of it: the prompt ran as the pass
//!   over the batch port, and did not step.
//! - `residency_reads_promoted_experts` (`hosted-mid`): the reseed promoted at least one expert in
//!   every routed layer (on the host before and on the card after, the card's copy of the slot map
//!   read back each time), and the writer's picks at `short`'s last row include at least one
//!   expert the card reads from a slot other than its id's, so the reference clauses that follow
//!   are not a pass over the plan's own layout; what the MXFP4 card slot path does with such an
//!   expert is what they hold to [`L_OUT_BAND`] and the rest.
//! - `residency_unset_serves` (`card`, `hosted-mid`): the unset residency word as the seats resolve
//!   it (`bloomery_levers::residency_unset` over `residency::at_plan`, the body's `UNHOSTED_POOL`)
//!   on each arm's plan. The `hosted-mid` plan hosts experts, so its serving word is
//!   [`RESIDENCY_SERVING`] with why `place`, the word [`RESIDENT`] runs; beside a fixed placement
//!   and beside the step feed it is `off`. The `card` plan holds every expert, which leaves the
//!   machine none to move onto the card: `off`, why `no_host_experts`.
//! - `residency_started` (`hosted`, `hosted-mid`): the `hosted-mid` load runs a residency machine
//!   and holds a source with parts for every routed layer, each layer's seed the plan's card
//!   experts less the spare slots, in id order; the `hosted` load (opened `Residency::Off`) runs
//!   no machine.
//! - `residency_passes`, `residency_c1`, `residency_c7` (`hosted-mid`, run first on the plan's own
//!   seed before the reseed moves it; `short`'s prompt then [`HIST_STEPS`] greedy steps, each
//!   history begun by [`Session::clear`], the seat's own call):
//!   the history's boundaries end, in order, no pass, the prompt call (one pass, 0 rows kept) and
//!   each step (1 row kept; the last step's own boundary is made ahead of its readback), so the
//!   count is `HIST_STEPS + 2` [derived]; the history twice gives the same ids, last logits row
//!   (bit for bit), boundaries and landed flips, and at least one flip lands (the measured term:
//!   the rule plans every 4 kept rows, makes at most one flip a layer in flight, so at most one
//!   a routed layer a plan, 2 x HIST_STEPS / 4 = 96 for the cut's two routed layers); the clear
//!   after it hands back a reset report of diff 0 and no dropped host byte, every seed layer's
//!   live set its seed. The clear is the session's, so a seat that did not clear the machine
//!   fails here, which a direct `residency_reset` would not show.
//! - `xstream_admit_matches_off` (`hosted-mid`): one id fed [`SKEW_IDS`] times (so every row of a
//!   unit routes the same experts and the pick's counts pass its floor on any router) as a prompt
//!   and [`GRAPH_STEPS`] steps with the prompt stream `off`, then again with it `admit` and the
//!   `off` run's ids fed back: the admit call picked and admitted at least one expert (or the
//!   clause proves nothing), the `off` call picked none, every logits row of the `admit` run is
//!   within [`LOGITS_BAND`] of the `off` run's and its argmax is the `off` run's wherever the
//!   `off` row's margin is [`MARGIN_FLOOR`] or more. The experts the pick moved are computed on
//!   the card where the host computed them, so the runs differ by the two tiers' rounding of the
//!   same experts, each of which the reference clauses hold to the writer within the bands above;
//!   [`LOGITS_BAND`] is used here as an unmeasured upper bound, and the clause's line prints the
//!   worst row's value.
//! - `reset_is_fresh` (the card arm and `hosted`, before the arm's first reset): `long` as the
//!   open's first prompt and again after a reset give the same token, taps, logits and selection
//!   bit for bit, the bits a fresh process writes, which the mixers' reset and the body's reset
//!   promise.
//! - `reset_clears_the_caches` (the card arm): the HCA caches and the MLA-DSA stores, read back and
//!   hashed, are the same after `long` and a reset as after `short` and a reset, and each moved
//!   from what its prompt left. No prompt's output shows a cache the reset skipped, since every
//!   cell a prompt reads it wrote first; the reset is the tripwire that makes a stray read loud
//!   (the NaN of the MLA-DSA stores), and this holds the tripwire armed.
//! - `opens_the_fixture`: each arm opens and carries the cut's three layers of their kinds; the
//!   card arm hosts no layer, the hosted arm every routed one.
//! - `graph_is_eager`: `short`'s prompt and [`GRAPH_STEPS`] greedy steps on the card arm in graph
//!   mode (the chain captured once from the reset model, before any token, as a session's open
//!   captures it, and replayed at every position) give the eager chain's ids and last logits row
//!   bit for bit.
//! - `cut_restores_the_checkpoint`: on the card arm, `short`'s prompt as a call, [`CUT_STEPS`]
//!   greedy steps, a rollback to the prompt's end and the prompt's ids again give the argmax and
//!   last logits row bit for bit of the same two calls from a reset: the HCA window and compressor
//!   rings come back from the checkpoint the first call ended with.
//!
//! Before the card loads, a set absent from the box, dumped from another cut or writer, or
//! unfinished ends the gate by name. A step that raises the fault word ends the gate by name.
//!
//! What the bands hold [derived]. The writer runs every formula in f32 over the dequantized
//! weights; the card stores the attention rows in f16 (a value's step 2⁻¹¹ ≈ 4.9e-4 relative)
//! and its quantized projections read their input in 8-bit blocks (a block's step `amax / 127`,
//! about 0.6 % rms of a row). A sub-layer chains up to five such sites, 1.3 % in quadrature, and
//! a layer's streams carry the residual beside it, so the third layer's streams sit within
//! 1.3 % and [`L_OUT_BAND`] is 1.5 times that; the head reads that row normed through one more
//! site ([`LOGITS_BAND`]). The router reads the same row in f32 ([`WEIGHTS_BAND`]). Two discrete
//! choices turn a rounding gap into another computation once a margin is narrower than it: a
//! routed row's tenth and eleventh scores, and the indexer's cut. The fixture's sets are fixed,
//! so a pick at such a tie is red on every run, never by chance; its line prints both sides.
//!
//! The real tier ends by name before anything loads: the released folder's sets
//! (`ref_ernie55_{short,long}`) have no family yet.

#[cfg(feature = "ernie55")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_e2e", gate::run())
}

#[cfg(feature = "ernie55")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "ernie55")]
#[path = "shared/serve_seats/folder.rs"]
#[allow(
    dead_code,
    reason = "the shared folder open serves several seats; this gate takes open and model"
)]
mod folder;

#[cfg(feature = "ernie55")]
#[allow(
    dead_code,
    reason = "the residency clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/residency_clauses.rs"]
mod residency_clauses;

#[cfg(feature = "ernie55")]
mod gate {
    use crate::residency_clauses::{c1_clause_with, c7_clause, passes_clause};
    use crate::{folder, gate_card};
    use app::Session;
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::folder::Model;
    use bloomery_gpu::host::placed::PlacedHost;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::host::slots::HOST;
    use bloomery_gpu::host::swap::{CardTable, PassReport, Residency, SwapSource};
    use bloomery_gpu::host::xstream::{XMode, XStreamBody};
    use bloomery_gpu::host::{HybridStats, PassKind};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_ernie55::{Body, PASS_ROWS, TapRows};
    use bloomery_gpu_gates::generate::Residence;
    use bloomery_gpu_gates::prompts::{
        GreedyRow, MARGIN_FLOOR, compare_forced, read_greedy, std_normal_cdf,
    };
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::residency;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed, ref_ints,
        ref_tensor_logical_in, verdict,
    };
    use bloomery_levers::{
        HostCfg, RESIDENCY_SERVING, ResidencyAt, ResidencyPick, ResidencyWhy, residency_unset,
    };
    use gguf::Split;
    use model::arch::ernie55::place::PlanInputs;
    use model::placement::PlanLevers;
    use model::placement::workstation;
    use refset::arch::ernie55::fixture::{LONG, NP, SHORT};
    use std::path::{Path, PathBuf};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_ernie55_e2e";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-ernie55-e2e";
    /// The fixture table's and the sets' name of the model.
    const ARCH: &str = "ernie5_5";
    /// The sets of the writer's family of the cut ([`NP`]), in the order they run; the second is
    /// the one past the indexer's top-k.
    const SETS: [&str; 2] = [SHORT, LONG];

    /// The cut's layers and their kinds, (latent MLA-DSA, dense lead), in layer order: the
    /// released folder's layers 0, 1 and 7.
    const KINDS: [(bool, bool); 3] = [(false, true), (false, false), (true, false)];

    /// Positions of every store: `long`'s 2,328 ids and its greedy ids fit.
    const CTX: usize = 4096;
    /// The rows the writer keeps of each layer: the prompt's last four.
    const TAP_ROWS: usize = 4;

    /// A layer's streams, relative L2 from the writer's `l_out-L` [derived: the module doc].
    const L_OUT_BAND: f64 = 2.0e-2;
    /// The logits before and after MultiMax, relative L2 [derived: the module doc].
    const LOGITS_BAND: f64 = 3.0e-2;
    /// A routed row's pick weights by id, relative L2 [derived: the module doc].
    const WEIGHTS_BAND: f64 = 2.0e-2;
    /// The MLA-DSA layer's selected positions either side holds that the other does not, of the
    /// writer's 2,048 [derived: positions whose score lies within the row's error of the cut's;
    /// the first run reads the count].
    const SEL_SWAPS: usize = 16;
    /// The floor of one logit's deviation in the forced pin, in logit units: an arm whose last
    /// prompt row matches the writer's more closely is pinned at this σ.
    const SIGMA_FLOOR: f64 = 1.0e-3;

    /// The greedy steps `graph_is_eager` takes after the prompt.
    const GRAPH_STEPS: usize = 8;
    /// The steps `cut_restores_the_checkpoint` takes past its checkpoint: more than a window ring
    /// (128) and a compressor block (128) hold.
    const CUT_STEPS: usize = 144;

    /// The ids `xstream_admit_matches_off` feeds: one id repeated for two units of [`PASS_ROWS`]
    /// rows, so every row of a unit routes the same experts and the unit's count of each is the
    /// unit's rows, past the admit floor (`ADMIT_FLOOR`, 32) on any router.
    const SKEW_IDS: usize = 2 * PASS_ROWS;

    /// The reference clauses of an arm: four a set, and `long`'s selection.
    const FX_ARM: usize = 4 * SETS.len() + 1;

    /// The reference clauses of the card arm and of the hosted arm, each: [`FX_ARM`] and the
    /// stepped layer outputs a set, every arm's prompt running as the pass.
    const FX_ARM_STEPPED: usize = FX_ARM + SETS.len();

    /// The reference clauses of the `hosted-mid` arm: `short`'s four (it dumps no selection) and
    /// its stepped layer outputs.
    const FX_RESIDENT: usize = 4 + 1;

    /// The residency a server runs by default: no pinned expert, one spare slot a routed layer.
    const RESIDENT: Residency = Residency::Mid {
        pinned: 0,
        spares: 1,
    };

    /// The greedy steps of a residency history: 48 planning intervals of the rule, which plans
    /// every 4 kept rows (the length of the V4.1 and Mistral Large 4 histories).
    const HIST_STEPS: usize = 192;

    /// The argument that names the hosted arm's file, the cut's GGUF.
    const HOSTED_GGUF: &str = "--hosted-gguf";

    /// The line a run without [`HOSTED_GGUF`] prints in place of the hosted arm.
    const HOSTED_REFUSED: &str = "opens_the_fixture arm=hosted REFUSED: the folder's MXFP4 expert \
        stacks are made at load, so no load from the folder splits them between the card and the \
        host; a hosted load needs the cut's GGUF bloomery-quantize writes (--hosted-gguf <file>)";

    /// The host tier's configuration the load opens with: no lever is read. A load under a
    /// residency reads its host set in ([`open_arm`]).
    const HOST_CFG: HostCfg = HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// One set's prompt, greedy table and the writer's rows at the prompt's last positions.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        greedy: GreedyRow,
        /// Per layer, the last rows' four streams, row-major `[rows, 4, hidden]`.
        l_out: Vec<Vec<f32>>,
        /// Per layer, the last rows' picks and weights, row-major `[rows, n_used]`; empty on the
        /// dense lead.
        picks: Vec<(Vec<u32>, Vec<f32>)>,
        /// Per layer, the selected positions at the last row, ascending; `None` where not dumped.
        sel: Vec<Option<Vec<u32>>>,
        raw: Vec<f32>,
        logits: Vec<f32>,
    }

    /// Integer row `name` of `man`, each value a `u32`.
    fn ids(man: &RefManifest, name: &str) -> Result<Vec<u32>, GateError> {
        ref_ints(man, name, 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|e| GateError::from(format!("{name}: {v}: {e}"))))
            .collect()
    }

    fn open_set(clause: &str, name: &'static str) -> Result<Set, GateError> {
        let path = tier::fixture_set(clause, &NP, name)?;
        let man = RefManifest::open(&path, &NP)?;
        let prompt = refset::arch::prompt_tables::tokens(&man)?.to_vec();
        let greedy = read_greedy(&man.dir.join("greedy.tsv"))?
            .into_iter()
            .next()
            .ok_or_else(|| format!("{name}: greedy.tsv holds no row"))?;
        if greedy.gen_ids.is_empty() || greedy.gen_margins.len() != greedy.gen_ids.len() {
            return Err(format!(
                "{name}: a greedy table of {} ids and {} margins",
                greedy.gen_ids.len(),
                greedy.gen_margins.len()
            )
            .into());
        }
        let f32s = |n: &str| -> Result<Vec<f32>, GateError> {
            Ok(ref_tensor_logical_in(&man.dir, man.tensor(n, 0)?)?)
        };
        let (mut l_out, mut picks, mut sel) = (Vec::new(), Vec::new(), Vec::new());
        for (l, &(_, dense)) in KINDS.iter().enumerate() {
            l_out.push(f32s(&format!("l_out-{l}"))?);
            picks.push(if dense {
                (Vec::new(), Vec::new())
            } else {
                (
                    ids(&man, &format!("ffn_moe_topk-{l}"))?,
                    f32s(&format!("ffn_moe_weights-{l}"))?,
                )
            });
            let top = format!("indexer_top_k-{l}");
            sel.push(match man.find(RowKind::Tensor, &top, 0) {
                Some(_) => Some(ids(&man, &top)?),
                None => None,
            });
        }
        let need = prompt.len() + greedy.gen_ids.len();
        if need > CTX {
            return Err(format!("{name}: {need} positions, the gate's stores hold {CTX}").into());
        }
        Ok(Set {
            name,
            prompt,
            greedy,
            l_out,
            picks,
            sel,
            raw: f32s("result_output_raw")?,
            logits: f32s("result_output")?,
        })
    }

    /// What an arm's plan says of its residency: the unset word as the seats resolve it
    /// (`bloomery_levers::residency_unset` over `residency::at_plan` with the body's
    /// `UNHOSTED_POOL`) at a serving placement, at a fixed one and beside the step feed, and the
    /// plan's card experts a layer.
    struct Planned {
        serving: ResidencyPick,
        fixed: ResidencyPick,
        steps: ResidencyPick,
        n_l: Vec<u64>,
    }

    /// An arm: its name, the model it opened and what its plan says of the residency.
    struct Arm {
        name: &'static str,
        m: GpuModel<Body>,
        planned: Planned,
    }

    /// The hosted arm's file: the value of [`HOSTED_GGUF`], the gate's one argument.
    fn hosted_gguf() -> Result<Option<PathBuf>, GateError> {
        let usage = format!("{NAME} [{HOSTED_GGUF} <the cut's GGUF>]");
        let mut path = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            if arg != HOSTED_GGUF {
                return Err(format!("unknown argument {arg:?}: {usage}").into());
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{HOSTED_GGUF} needs a value: {usage}"))?;
            path = Some(PathBuf::from(v));
        }
        Ok(path)
    }

    /// The fixture cut opened on the card in view, from `gguf` where given and else from the cut
    /// folder, under a card-experts cap of half the routed experts where `capped` and the
    /// residency `residency`: the plan, whether it hosts experts, and the model with the prompt
    /// pass (over the host tier's batch port where the plan hosts experts).
    fn open_arm(
        name: &'static str,
        capped: bool,
        gguf: Option<&Path>,
        residency: Residency,
    ) -> Result<(Arm, bool), GateError> {
        let model: Model = match gguf {
            Some(path) => {
                let split = Split::open(path)
                    .map_err(|e| format!("{NAME} {name}: open {}: {e}", path.display()))?;
                Tier::from_env()?.check_file(path, split.iter_kv().map(|(k, _)| k))?;
                split.into()
            }
            None => {
                let f = folder::open(&tier::cut(ARCH)?, ARCH)?;
                folder::model(f.header, f.source)?
            }
        };
        let inputs = PlanInputs::read(model.header())
            .map_err(|e| format!("{NAME} {name}: the cut's description: {e}"))?;
        let hp = &inputs.hp;
        let routed = (0..hp.n_layer).filter(|&l| !hp.dense(l)).count() as u64;
        let levers = PlanLevers {
            card_experts_cap: capped.then_some(routed * hp.experts.n_expert as u64 / 2),
            ..PlanLevers::default()
        };
        let machine = workstation::plan_on(gate_card::room()?, hp.n_layer);
        let plan = Body::plan(&inputs, &machine, CTX as u64, &levers)?;
        let hosts = Body::plan_hosts(&plan, &inputs);
        let unset = |serving_place: bool, prefill_steps: bool| {
            residency::at_plan(
                <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
                &plan,
                0,
                residency_unset(ResidencyAt {
                    serving_place,
                    check_finite: false,
                    route_trace: false,
                    prefill_steps,
                }),
            )
        };
        let planned = Planned {
            serving: unset(true, false)?,
            fixed: unset(false, false)?,
            steps: unset(true, true)?,
            n_l: plan.n_l.iter().copied().collect(),
        };
        let host = HostCfg {
            populate: residency != Residency::Off,
            ..HOST_CFG
        };
        let m = Body::open_placed(
            model,
            (&plan, &inputs),
            0,
            host,
            (Some(PASS_ROWS), residency),
        )?;
        Ok((Arm { name, m, planned }, hosts))
    }

    /// `ids` from the model's position: the pass where the load has one, else one id a step.
    fn feed(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GateError> {
        if m.body(NAME)?.pass_rows().is_some() {
            Ok(bloomery_gpu_ernie55::prompt(m, ids)?)
        } else {
            Ok(m.step(ids)?)
        }
    }

    /// The taps of the last walk.
    fn taps(m: &mut GpuModel<Body>) -> Result<TapRows, GateError> {
        let (gpu, _, body) = m.body_parts(NAME)?;
        Ok(body.taps(gpu.stream())?)
    }

    /// The last `n` of `all`'s rows of `width` values.
    fn last_rows(all: &[f32], width: usize, n: usize) -> &[f32] {
        &all[all.len().saturating_sub(n * width)..]
    }

    /// Clause `layer_outputs_are_reference` of the walk `t` left.
    fn layer_outputs(arm: &str, s: &Set, t: &TapRows) -> Result<bool, GateError> {
        let mut ok = true;
        for (l, theirs) in s.l_out.iter().enumerate() {
            let width = theirs.len() / TAP_ROWS;
            let r = rel(&t.l_out[l], last_rows(theirs, width, t.rows));
            let pass = r <= L_OUT_BAND;
            ok &= pass;
            println!(
                "layer_outputs_are_reference set={} arm={arm} layer={l} rows={} rel={r:.3e} \
                 band={L_OUT_BAND:.1e} {}",
                s.name,
                t.rows,
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// Clause `stepped_layer_outputs_are_reference`, on every arm (each prompt runs as the pass):
    /// the prompt fed one id a step from a reset, the last step's layer outputs against the
    /// writer's last row ([`layer_outputs`]): the arm's step chain, whole-card or hosted.
    fn stepped_layer_outputs(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        arm.m.reset()?;
        arm.m.step(&s.prompt)?;
        let t = taps(&mut arm.m)?;
        layer_outputs(&format!("{}-stepped", arm.name), s, &t)
    }

    /// Picks as `(id, weight)` sorted by id.
    fn by_id(ids: &[u32], ws: &[f32]) -> Vec<(u32, f32)> {
        let mut v: Vec<(u32, f32)> = ids.iter().copied().zip(ws.iter().copied()).collect();
        v.sort_unstable_by_key(|p| p.0);
        v
    }

    /// Clause `picks_are_reference` of the walk `t` left: per routed layer and tapped row.
    fn picks(arm: &str, s: &Set, t: &TapRows, k: usize) -> Result<bool, GateError> {
        let mut ok = true;
        for (l, (ids, ws)) in s.picks.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let (their_ids, their_ws) = (last_rows_u(ids, k, t.rows), last_rows(ws, k, t.rows));
            for r in 0..t.rows {
                let row = r * k..(r + 1) * k;
                let ours = by_id(&t.ids[l][row.clone()], &t.weights[l][row.clone()]);
                let theirs = by_id(&their_ids[row.clone()], &their_ws[row]);
                let same = ours.iter().map(|p| p.0).eq(theirs.iter().map(|p| p.0));
                let wr = if same {
                    let a: Vec<f32> = ours.iter().map(|p| p.1).collect();
                    let b: Vec<f32> = theirs.iter().map(|p| p.1).collect();
                    rel(&a, &b)
                } else {
                    f64::INFINITY
                };
                let pass = same && wr <= WEIGHTS_BAND;
                ok &= pass;
                let only = |x: &[(u32, f32)], y: &[(u32, f32)]| -> Vec<(u32, f32)> {
                    x.iter()
                        .copied()
                        .filter(|p| !y.iter().any(|q| q.0 == p.0))
                        .collect()
                };
                println!(
                    "picks_are_reference set={} arm={arm} layer={l} row={r} same_ids={same} \
                     weights_rel={wr:.3e} band={WEIGHTS_BAND:.1e} ours_only={:?} \
                     reference_only={:?} {}",
                    s.name,
                    only(&ours, &theirs),
                    only(&theirs, &ours),
                    verdict(pass)
                );
            }
        }
        Ok(ok)
    }

    /// The last `n` of `all`'s rows of `width` ids.
    fn last_rows_u(all: &[u32], width: usize, n: usize) -> &[u32] {
        &all[all.len().saturating_sub(n * width)..]
    }

    /// Clause `selection_is_reference`: each layer the set dumps a selection of, at the last row.
    fn selection(arm: &str, s: &Set, m: &mut GpuModel<Body>) -> Result<bool, GateError> {
        let mut ok = true;
        for (l, theirs) in s.sel.iter().enumerate() {
            let Some(theirs) = theirs else { continue };
            let (gpu, _, body) = m.body_parts(NAME)?;
            let ours = body.selection(gpu.stream(), l)?;
            let (swaps, n) = match &ours {
                Some(v) => {
                    let a: std::collections::BTreeSet<u32> = v.iter().copied().collect();
                    let b: std::collections::BTreeSet<u32> = theirs.iter().copied().collect();
                    (
                        a.difference(&b).count().max(b.difference(&a).count()),
                        v.len(),
                    )
                }
                None => (usize::MAX, 0),
            };
            let pass = n == theirs.len() && swaps <= SEL_SWAPS;
            ok &= pass;
            println!(
                "selection_is_reference set={} arm={arm} layer={l} ours={n} reference={} \
                 swaps={} pin={SEL_SWAPS} {}",
                s.name,
                theirs.len(),
                if ours.is_some() {
                    swaps.to_string()
                } else {
                    "none".to_string()
                },
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// The root mean square of the logits' difference: one logit's deviation, σ̂.
    fn sigma_hat(ours: &[f32], theirs: &[f32]) -> f64 {
        let s: f64 = ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        (s / ours.len().max(1) as f64).sqrt()
    }

    /// The index of the largest value, the first of equals.
    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    }

    /// Clause `head_is_reference` of the prompt's last row: σ̂ of the MultiMax logits for the
    /// forced pin, and whether the clause holds.
    fn head(arm: &str, s: &Set, t: &TapRows, logits: &[f32]) -> (f64, bool) {
        let (r_raw, r_mm) = (rel(&t.raw, &s.raw), rel(logits, &s.logits));
        let (a_raw, a_mm) = (argmax(&t.raw), argmax(logits));
        let want = s.greedy.argmax as usize;
        let pass =
            r_raw <= LOGITS_BAND && r_mm <= LOGITS_BAND && a_raw == argmax(&s.raw) && a_mm == want;
        let sigma = sigma_hat(logits, &s.logits);
        println!(
            "head_is_reference set={} arm={arm} raw_rel={r_raw:.3e} multimax_rel={r_mm:.3e} \
             band={LOGITS_BAND:.1e} argmax_raw={a_raw}/{} argmax={a_mm}/{want} sigma_hat={sigma:.3e} {}",
            s.name,
            argmax(&s.raw),
            verdict(pass)
        );
        (sigma, pass)
    }

    /// The disagreement count the Gaussian error model expects at the writer's greedy margins at
    /// one logit's deviation `sigma`, and the pin: that count plus three standard deviations of
    /// it, rounded up.
    ///
    /// [derived]: at the writer's margin `m` the argmax flips with probability Φ(−m / (√2 σ)).
    fn forced_pin(margins: &[f32], sigma: f64) -> (f64, usize) {
        let sigma_pair = sigma.max(SIGMA_FLOOR) * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// Clause `forced_top1_is_reference` from a reset model: the prompt, then the writer's greedy
    /// ids fed back one a step.
    fn forced_top1(arm: &mut Arm, s: &Set, sigma: f64) -> Result<bool, GateError> {
        let m = &mut arm.m;
        m.reset()?;
        let mut table = vec![feed(m, &s.prompt)?];
        for &id in &s.greedy.gen_ids[..s.greedy.gen_ids.len() - 1] {
            table.push(m.step(&[id])?);
        }
        let report = compare_forced(&[table], std::slice::from_ref(&s.greedy), MARGIN_FLOOR);
        let (expected, pin) = forced_pin(&s.greedy.gen_margins, sigma);
        let pass = report.disagree_ge_floor <= pin;
        println!(
            "forced_top1_is_reference set={} arm={} positions={} disagree={} buckets={:?} \
             ge_floor={} expected {expected:.2} pin {pin} {}",
            s.name,
            arm.name,
            report.positions,
            report.disagree.len(),
            report.buckets,
            report.disagree_ge_floor,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Whether `tier` runs reference clause `clause` of set `s` and arm `arm` (counted either way).
    fn runs(clause: &str, s: &Set, arm: &str) -> Result<bool, GateError> {
        tier::run_clause(
            &format!("{clause} set={} arm={arm}", s.name),
            Tag::FixtureOracle,
        )
    }

    /// The host tier's service counters since load ([`HybridStats`]) of an arm whose plan hosts
    /// experts, read back before a prompt; `None` on an arm with no host tier.
    fn host_stats(m: &mut GpuModel<Body>) -> Result<Option<HybridStats>, GateError> {
        let body = m.body(NAME)?;
        Ok((!body.hosted().is_empty()).then(|| body.hybrid().stats()))
    }

    /// Clause `prompt_runs_as_a_pass` of an arm whose plan hosts `hosted` routed layers, between
    /// the counters `before` the prompt and `after` it: the batch port served one layer-batch a
    /// hosted layer a unit of [`PASS_ROWS`] rows, over the prompt's `rows`, listing host slots,
    /// and the step port served nothing of it. The pass ran over the batch port, and did not step.
    fn prompt_runs_as_a_pass(
        arm: &str,
        s: &Set,
        before: &HybridStats,
        after: &HybridStats,
        hosted: usize,
    ) -> bool {
        let rows = s.prompt.len();
        let units = rows.div_ceil(PASS_ROWS);
        let served = after.batch_served - before.batch_served;
        let cols = after.batch_cols - before.batch_cols;
        let slots = after.batch_host_slots - before.batch_host_slots;
        let steps = after.served - before.served;
        let pass = hosted > 0
            && served == (units * hosted) as u64
            && cols == (rows * hosted) as u64
            && slots > 0
            && steps == 0;
        println!(
            "prompt_runs_as_a_pass set={} arm={arm} rows={rows} units={units} hosted_layers={hosted} \
             batch_served={served} batch_cols={cols} batch_host_slots={slots} step_served={steps} {}",
            s.name,
            verdict(pass)
        );
        pass
    }

    /// [`prompt_runs_as_a_pass`] on `arm` where its plan hosts experts and `tier` runs the
    /// clause; `true` elsewhere.
    fn pass_served(arm: &mut Arm, s: &Set, before: Option<HybridStats>) -> Result<bool, GateError> {
        let hosted = arm.m.body(NAME)?.hosted().len();
        let clause = format!("prompt_runs_as_a_pass set={} arm={}", s.name, arm.name);
        let Some(before) = before else {
            return Ok(true);
        };
        if !tier::run_clause(&clause, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let after = host_stats(&mut arm.m)?.ok_or("the arm's host tier is gone")?;
        Ok(prompt_runs_as_a_pass(arm.name, s, &before, &after, hosted))
    }

    /// The reference clauses of set `s` on `arm`, in the order their state needs: the prompt from
    /// a reset and its taps (and, where the plan hosts experts, the pass it ran as), the stepped
    /// prompt from a reset, then the forced argmax from a reset.
    fn reference_clauses(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        let a = arm.name;
        arm.m.reset()?;
        let before = host_stats(&mut arm.m)?;
        feed(&mut arm.m, &s.prompt)?;
        let mut ok = pass_served(arm, s, before)?;
        let t = taps(&mut arm.m)?;
        let logits = arm.m.logits()?;
        let k = arm.m.body(NAME)?.shape().n_used;
        if runs("layer_outputs_are_reference", s, a)? {
            ok &= layer_outputs(a, s, &t)?;
        }
        if runs("picks_are_reference", s, a)? {
            ok &= picks(a, s, &t, k)?;
        }
        if s.sel.iter().any(Option::is_some) && runs("selection_is_reference", s, a)? {
            ok &= selection(a, s, &mut arm.m)?;
        }
        let mut sigma = SIGMA_FLOOR;
        if runs("head_is_reference", s, a)? {
            let (sg, pass) = head(a, s, &t, &logits);
            sigma = sg;
            ok &= pass;
        }
        if arm.m.body(NAME)?.pass_rows().is_some()
            && runs("stepped_layer_outputs_are_reference", s, a)?
        {
            ok &= stepped_layer_outputs(arm, s)?;
        }
        if runs("forced_top1_is_reference", s, a)? {
            ok &= forced_top1(arm, s, sigma)?;
        }
        Ok(ok)
    }

    /// A prompt's outputs: the answer, the taps, the logits and each layer's selection.
    struct Walk {
        token: u32,
        taps: TapRows,
        logits: Vec<f32>,
        sel: Vec<Option<Vec<u32>>>,
    }

    /// `ids` from the model's position as one prompt, and what it left.
    fn walk(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<Walk, GateError> {
        let token = feed(m, ids)?;
        let t = taps(m)?;
        let logits = m.logits()?;
        let (gpu, _, body) = m.body_parts(NAME)?;
        let sel = (0..KINDS.len())
            .map(|l| body.selection(gpu.stream(), l))
            .collect::<Result<_, _>>()?;
        Ok(Walk {
            token,
            taps: t,
            logits,
            sel,
        })
    }

    /// The parts of `a` that are not `b`'s, bit for bit, by name.
    fn differing(a: &Walk, b: &Walk) -> Vec<String> {
        let mut d = Vec::new();
        if a.token != b.token {
            d.push("token".to_string());
        }
        if a.taps.rows != b.taps.rows {
            d.push("rows".to_string());
        }
        for l in 0..a.taps.l_out.len() {
            if !bits_equal(&a.taps.l_out[l], &b.taps.l_out[l]) {
                d.push(format!("l_out-{l}"));
            }
            if a.taps.ids[l] != b.taps.ids[l] {
                d.push(format!("ids-{l}"));
            }
            if !bits_equal(&a.taps.weights[l], &b.taps.weights[l]) {
                d.push(format!("weights-{l}"));
            }
        }
        if !bits_equal(&a.taps.raw, &b.taps.raw) {
            d.push("raw".to_string());
        }
        if !bits_equal(&a.logits, &b.logits) {
            d.push("logits".to_string());
        }
        if a.sel != b.sel {
            d.push("selection".to_string());
        }
        d
    }

    /// Clause `reset_is_fresh` of one arm, whose model has run nothing: `s`'s prompt as the open's
    /// first, then again after a reset, bit for bit.
    fn reset_is_fresh(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        let clause = format!("reset_is_fresh set={} arm={}", s.name, arm.name);
        if !tier::run_clause(&clause, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let fresh = walk(&mut arm.m, &s.prompt)?;
        arm.m.reset()?;
        let again = walk(&mut arm.m, &s.prompt)?;
        let differs = differing(&fresh, &again);
        let pass = differs.is_empty();
        println!(
            "reset_is_fresh set={} arm={} token={}/{} differs={differs:?} {}",
            s.name,
            arm.name,
            fresh.token,
            again.token,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The digests of the HCA caches and the MLA-DSA stores of the arm's model.
    fn digests(arm: &mut Arm) -> Result<[u64; 2], GateError> {
        let (gpu, _, body) = arm.m.body_parts(NAME)?;
        Ok(body.cache_digests(gpu.stream())?)
    }

    /// Clause `reset_clears_the_caches` of one arm: after `long` and a reset the caches' digests
    /// are those after `short` and a reset, and each moved off what its prompt left.
    fn reset_clears_the_caches(
        arm: &mut Arm,
        (long, short): (&Set, &Set),
    ) -> Result<bool, GateError> {
        let clause = format!("reset_clears_the_caches arm={}", arm.name);
        if !tier::run_clause(&clause, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let mut states = [([0u64; 2], [0u64; 2]); 2];
        for (s, state) in [long, short].into_iter().zip(&mut states) {
            arm.m.reset()?;
            feed(&mut arm.m, &s.prompt)?;
            let left = digests(arm)?;
            arm.m.reset()?;
            *state = (left, digests(arm)?);
        }
        let [(left, after_long), (_, after_short)] = states;
        let pass = after_long == after_short && (0..2).all(|i| left[i] != after_long[i]);
        println!(
            "reset_clears_the_caches arm={} hca after_long={:016x} after_short={:016x} \
             left_by_long={:016x} mla after_long={:016x} after_short={:016x} left_by_long={:016x} {}",
            arm.name,
            after_long[0],
            after_short[0],
            left[0],
            after_long[1],
            after_short[1],
            left[1],
            verdict(pass)
        );
        Ok(pass)
    }

    /// The card's copy of the slot map, read back (synchronizes the stream).
    fn card_table(m: &mut GpuModel<Body>) -> Result<CardTable, GateError> {
        let (gpu, _, body) = m.body_parts(NAME)?;
        let parts = body
            .residency_parts()
            .ok_or("the load runs no residency machine")?;
        Residence::of(gpu, parts.slots, parts.hybrid).table()
    }

    /// The reseed order of one routed layer of `n_expert` ids: the ids `picked` first, then every
    /// other id from the highest, so the card's set after the reseed holds experts the plan left
    /// on the host.
    fn promote_order(picked: &[u32], n_expert: usize) -> Vec<u32> {
        let mut order: Vec<u32> = Vec::with_capacity(n_expert);
        for &id in picked {
            if !order.contains(&id) {
                order.push(id);
            }
        }
        let high = u32::try_from(n_expert).expect("an expert count fits u32");
        order.extend((0..high).rev().filter(|id| !picked.contains(id)));
        order
    }

    /// The `hosted-mid` arm's clauses on `s`: its prompt fed once, every routed layer reseeded to
    /// the experts that prompt picked, clause `residency_reads_promoted_experts`, then `s`'s
    /// reference clauses over the layout the reseed made.
    fn resident_clauses(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        let a = arm.name;
        arm.m.reset()?;
        let before = host_stats(&mut arm.m)?;
        feed(&mut arm.m, &s.prompt)?;
        let first = taps(&mut arm.m)?;
        let passed = pass_served(arm, s, before)?;
        let k = arm.m.body(NAME)?.shape().n_used;
        let (layers, n_expert) = {
            let (_, _, body) = arm.m.body_parts(NAME)?;
            let parts = body
                .residency_parts()
                .ok_or("the load runs no residency machine")?;
            let map = parts.hybrid.slots();
            (map.layers(), map.n_expert())
        };
        let before = card_table(&mut arm.m)?;
        let ranked: Vec<Vec<u32>> = layers
            .clone()
            .map(|l| promote_order(&first.ids[l], n_expert))
            .collect();
        let order: Vec<&[u32]> = ranked.iter().map(Vec::as_slice).collect();
        arm.m
            .residency_reseed(&order)?
            .ok_or("the load reseeded no residency machine")?;
        let after = card_table(&mut arm.m)?;
        let mut ok = passed;
        let clause = format!("residency_reads_promoted_experts set={} arm={a}", s.name);
        if tier::run_clause(&clause, Tag::SelfConsistency)? {
            for (i, l) in layers.enumerate() {
                let at = i * n_expert..(i + 1) * n_expert;
                let (was, now) = (&before.entries()[at.clone()], &after.entries()[at]);
                let off_id = |e: usize| now[e] != HOST && now[e] as usize != e;
                let promoted = (0..n_expert)
                    .filter(|&e| was[e] == HOST && now[e] != HOST)
                    .count();
                let moved = (0..n_expert).filter(|&e| off_id(e)).count();
                let used = last_rows_u(&s.picks[l].0, k, 1)
                    .iter()
                    .filter(|&&id| off_id(id as usize))
                    .count();
                let pass = promoted >= 1 && used >= 1;
                ok &= pass;
                println!(
                    "residency_reads_promoted_experts set={} arm={a} layer={l} promoted={promoted} \
                     slot_not_id={moved} picks_on_them={used}/{k} {}",
                    s.name,
                    verdict(pass)
                );
            }
        }
        ok &= reference_clauses(arm, s)?;
        Ok(ok)
    }

    /// The body's host tier on `m`: the placed host a load with experts on the host holds.
    fn host_of(m: &mut GpuModel<Body>) -> Result<&mut PlacedHost, GateError> {
        let (_, _, body) = m.body_parts(NAME)?;
        <Body as bloomery_gpu::model::ChainBody>::host(body)
            .ok_or_else(|| GateError::from(format!("{NAME}: the load has no host tier")))
    }

    /// Clause `residency_unset_serves`: the unset word on the `hosted-mid` arm's plan, which hosts
    /// experts, is the serving word with why `place` and the word [`RESIDENT`] runs, and beside a
    /// fixed placement and the step feed it is `off`; on the `card` arm's plan, which holds every
    /// expert and leaves the machine none to move onto the card, it is `off` with why
    /// `no_host_experts`.
    fn unset_serves(card: &Planned, hosted: &Planned) -> Result<bool, GateError> {
        const CLAUSE: &str = "residency_unset_serves";
        if !tier::run_clause(CLAUSE, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let want = |word: &'static str, why: ResidencyWhy| ResidencyPick { word, why };
        let serving = hosted.serving == want(RESIDENCY_SERVING, ResidencyWhy::Place)
            && Residency::parse(hosted.serving.word)? == RESIDENT;
        let off = hosted.fixed == want("off", ResidencyWhy::FixedPlace)
            && hosted.steps == want("off", ResidencyWhy::PrefillSteps);
        let whole = card.serving == want("off", ResidencyWhy::NoHostExperts);
        let pass = serving && off && whole;
        println!(
            "{CLAUSE}: the hosted plan's serving word {} ({}) is the arm's {serving}, beside a fixed \
             placement {} ({}) and the step feed {} ({}) {off}; the whole-card plan's {} ({}) {whole} {}",
            hosted.serving.word,
            hosted.serving.why.name(),
            hosted.fixed.word,
            hosted.fixed.why.name(),
            hosted.steps.word,
            hosted.steps.why.name(),
            card.serving.word,
            card.serving.why.name(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `residency_started` on the `hosted` arm, opened `Residency::Off`: no machine.
    fn started_off(arm: &Arm) -> Result<bool, GateError> {
        let clause = format!("residency_started arm={}", arm.name);
        if !tier::run_clause(&clause, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let machine = arm.m.body(NAME)?.hybrid().swap().is_some();
        println!(
            "{clause}: the off load runs a machine {machine} (want none) {}",
            verdict(!machine)
        );
        Ok(!machine)
    }

    /// Clause `residency_started` on the `hosted-mid` arm: the load runs a machine over its routed
    /// layers and holds a source with parts for every layer the plan gives a card expert, and each
    /// layer's seed is the plan's card experts less the word's spare slots, in id order.
    fn started_mid(arm: &mut Arm) -> Result<bool, GateError> {
        let clause = format!("residency_started arm={}", arm.name);
        if !tier::run_clause(&clause, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let Residency::Mid { spares, .. } = RESIDENT else {
            return Err(format!("{NAME}: the arm's word {RESIDENT:?} is off").into());
        };
        let layers = arm.m.body(NAME)?.hybrid().slots().layers();
        let (sourced, with_parts, bare) = {
            let host = host_of(&mut arm.m)?;
            match host.shell.residency_source() {
                Some(source) => {
                    let mut bare = Vec::new();
                    let mut with_parts = 0usize;
                    for l in layers.clone() {
                        if source.part_bytes(l).is_empty() {
                            if arm.planned.n_l.get(l).copied().unwrap_or(0) > 0 {
                                bare.push(l);
                            }
                        } else {
                            with_parts += 1;
                        }
                    }
                    (true, with_parts, bare)
                }
                None => (false, 0, Vec::new()),
            }
        };
        let machine = arm.m.body(NAME)?.hybrid().swap();
        let mut seeds_ok = machine.is_some();
        if let Some(machine) = machine {
            for l in layers.clone() {
                let held = arm.planned.n_l.get(l).copied().unwrap_or(0);
                let want: Vec<u32> =
                    (0..u32::try_from(held.saturating_sub(spares as u64))?).collect();
                seeds_ok &= machine.seed(l)? == want;
            }
        }
        let pass = sourced && with_parts > 0 && bare.is_empty() && seeds_ok;
        println!(
            "{clause}: the mid load runs a machine and holds a source {sourced}, with parts for \
             {with_parts} of its {} routed layers (card experts and no part: {bare:?}), each seed \
             the plan's card experts less {spares} spare(s) in id order {seeds_ok} {}",
            layers.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// One history of the `hosted-mid` session: the ids it decoded, its last logits row, the
    /// residency boundaries it ended (each pass's kind and kept rows) and the flips that landed
    /// at them.
    struct Hist {
        tokens: Vec<u32>,
        logits: Vec<f32>,
        passes: Vec<(PassKind, usize)>,
        landed: usize,
    }

    /// The residency boundaries' reports of `m` since the last take, each with the kind of the
    /// pass it ended.
    fn take_passes(m: &mut GpuModel<Body>) -> Result<Vec<(PassKind, PassReport)>, GateError> {
        Ok(host_of(m)?.shell.take_residency_passes())
    }

    /// The history of `ids` from a clear (the model's reset and the residency's seed,
    /// [`Session::clear`]): the prompt as the pass, then [`HIST_STEPS`] greedy steps.
    fn history(s: &mut Session<Body>, ids: &[u32]) -> Result<Hist, GateError> {
        s.clear()?;
        let _ = take_passes(s.model_mut())?;
        let mut next = feed(s.model_mut(), ids)?;
        let mut tokens = vec![next];
        for _ in 0..HIST_STEPS {
            next = s.model_mut().step(&[next])?;
            tokens.push(next);
        }
        let logits = s.model().logits()?;
        let ended = take_passes(s.model_mut())?;
        Ok(Hist {
            tokens,
            logits,
            passes: ended.iter().map(|&(k, r)| (k, r.kept)).collect(),
            landed: ended.iter().map(|(_, r)| r.landed).sum(),
        })
    }

    /// Clauses `residency_passes`, `residency_c1` and `residency_c7` on the `hosted-mid` model
    /// `m` as the open left it (the plan's own seed), driven through a [`Session`] over it: `s`'s
    /// prompt and [`HIST_STEPS`] greedy steps twice, each from [`Session::clear`], then one clear
    /// more whose reset report is the clause `residency_c7`'s. The clear leaves the machine at its
    /// seed, and the model is handed back for the clauses that reseed it.
    fn history_clauses(m: GpuModel<Body>, s: &Set) -> Result<(bool, GpuModel<Body>), GateError> {
        let mut session = Session::from_model(m, u32::try_from(CTX)?);
        session.model_mut().set_mode(StepMode::Eager);
        host_of(session.model_mut())?
            .shell
            .log_residency(HIST_STEPS + 2);
        let first = history(&mut session, &s.prompt)?;
        let second = history(&mut session, &s.prompt)?;
        let mut ok = true;
        if tier::run_clause("residency_passes", Tag::SelfConsistency)? {
            ok &= passes_clause(&first.passes, HIST_STEPS);
        }
        if tier::run_clause("residency_c1", Tag::SelfConsistency)? {
            let same = first.tokens == second.tokens
                && first.passes == second.passes
                && first.landed == second.landed
                && bits_equal(&first.logits, &second.logits);
            ok &= c1_clause_with(first.tokens.len(), first.landed, same, true);
        }
        session.clear()?;
        let report = session
            .take_cleared()
            .ok_or("the session's clear made no residency reset")?;
        if tier::run_clause("residency_c7", Tag::SelfConsistency)? {
            let body = session.model().body(NAME)?;
            let machine = body
                .hybrid()
                .swap()
                .ok_or("the load runs no residency machine")?;
            let seeds = crate::residency_clauses::seeds(machine, body.hybrid().slots().layers())?;
            ok &= c7_clause(&report, machine, &seeds);
        }
        Ok((ok, session.into_model()))
    }

    /// What `xstream_admit_matches_off` reads of one run: the argmax and the logits row after the
    /// prompt and after each step, and the picks the prompt call made with the experts they
    /// admitted.
    struct StreamRun {
        tokens: Vec<u32>,
        rows: Vec<Vec<f32>>,
        picks: usize,
        admitted: usize,
    }

    /// `ids` as a prompt from a reset with the prompt stream at `mode`, then [`GRAPH_STEPS`] steps
    /// fed the ids `forced` names, or each step's own greedy id where `forced` is `None`.
    fn stream_run(
        arm: &mut Arm,
        mode: XMode,
        ids: &[u32],
        forced: Option<&[u32]>,
    ) -> Result<StreamRun, GateError> {
        {
            let (gpu, _, body) = arm.m.body_parts(NAME)?;
            body.set_xstream(gpu, mode)?;
        }
        arm.m.reset()?;
        let mut tokens = vec![feed(&mut arm.m, ids)?];
        let mut rows = vec![arm.m.logits()?];
        let (records, _) = arm.m.body_parts(NAME)?.2.take_stream_records();
        let admitted: usize = records.iter().map(|(_, pick)| pick.admitted).sum();
        while tokens.len() <= GRAPH_STEPS {
            let at = tokens.len() - 1;
            let id = forced.map_or(tokens[at], |f| f[at]);
            tokens.push(arm.m.step(&[id])?);
            rows.push(arm.m.logits()?);
        }
        Ok(StreamRun {
            tokens,
            rows,
            picks: records.len(),
            admitted,
        })
    }

    /// A logits row's largest value less its second largest.
    fn margin(row: &[f32]) -> f32 {
        let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in row {
            if v > first {
                second = first;
                first = v;
            } else if v > second {
                second = v;
            }
        }
        first - second
    }

    /// Clause `xstream_admit_matches_off` of the arm that runs a residency over a plan that hosts
    /// experts: a prompt of [`SKEW_IDS`] copies of `s`'s last id and [`GRAPH_STEPS`] steps with the
    /// prompt stream `off` and then `admit`, the `admit` run fed the `off` run's ids. The premise
    /// that keeps it from proving nothing: the `admit` call picked and admitted an expert. The
    /// `off` call picked none; the logits rows are within [`LOGITS_BAND`] of the `off` run's
    /// (the moved experts are computed on the card in place of the host: the tiers' rounding of
    /// the same experts, each held to the writer within the bands above, so the band is an
    /// unmeasured upper bound; the line prints the worst row's value) and the argmaxes are the
    /// `off` run's wherever its margin is [`MARGIN_FLOOR`] or more. The arm's stream is `off`
    /// again after.
    fn xstream_admit_matches_off(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        let clause = format!("xstream_admit_matches_off set={} arm={}", s.name, arm.name);
        if !tier::run_clause(&clause, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let id = *s.prompt.last().ok_or("the set holds no prompt")?;
        let ids = vec![id; SKEW_IDS];
        let off = stream_run(arm, XMode::Off, &ids, None)?;
        let on = stream_run(arm, XMode::Admit, &ids, Some(&off.tokens[..GRAPH_STEPS]));
        {
            let (gpu, _, body) = arm.m.body_parts(NAME)?;
            body.set_xstream(gpu, XMode::Off)?;
        }
        let on = on?;
        let worst = on
            .rows
            .iter()
            .zip(&off.rows)
            .map(|(a, b)| rel(a, b))
            .fold(0.0f64, f64::max);
        let flips: Vec<usize> = on
            .tokens
            .iter()
            .zip(&off.tokens)
            .zip(&off.rows)
            .enumerate()
            .filter(|(_, ((a, b), row))| a != b && margin(row) >= MARGIN_FLOOR)
            .map(|(i, _)| i)
            .collect();
        let admitted = tier::premise(
            "xstream_admit_matches_off: the admit call admitted an expert (the skewed prompt's counts)",
            Tag::SelfConsistency,
            on.picks > 0 && on.admitted > 0,
        )?;
        let pass = admitted
            && off.picks == 0
            && off.admitted == 0
            && worst <= LOGITS_BAND
            && flips.is_empty();
        println!(
            "xstream_admit_matches_off set={} arm={} ids={SKEW_IDS} steps={GRAPH_STEPS} \
             admit_picks={} admitted={} off_picks={} worst_rel={worst:.3e} band={LOGITS_BAND:.1e} \
             argmax_flips_at_margin={flips:?} {}",
            s.name,
            arm.name,
            on.picks,
            on.admitted,
            off.picks,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The `hosted-mid` arm: the hosted file opened under [`RESIDENT`] (`opened`: whether clause
    /// `opens_the_fixture` runs; `card` the card arm, whose plan the unset word is also read
    /// on), its residency clauses and its clauses on `s`, closed.
    fn resident_arm(gguf: &Path, card: &Arm, s: &Set, opened: bool) -> Result<bool, GateError> {
        let (mut arm, hosts) = open_arm("hosted-mid", true, Some(gguf), RESIDENT)?;
        let mut ok = if opened {
            opens(&arm, hosts, true)?
        } else {
            true
        };
        ok &= unset_serves(&card.planned, &arm.planned)?;
        ok &= started_mid(&mut arm)?;
        // The histories run on the plan's own seed, before the reseed below moves it.
        let (held, m) = history_clauses(arm.m, s)?;
        arm.m = m;
        ok &= held;
        let (gpu, _, body) = arm.m.body_parts(NAME)?;
        body.arm_taps(gpu, TAP_ROWS)?;
        ok &= resident_clauses(&mut arm, s)?;
        ok &= xstream_admit_matches_off(&mut arm, s)?;
        Ok(ok)
    }

    /// Clause `graph_is_eager`: `s`'s prompt and [`GRAPH_STEPS`] greedy steps in graph mode give
    /// the eager chain's ids and last logits row bit for bit, the chain captured from the reset
    /// model before any token, as a session's open captures it.
    fn graph_is_eager(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause("graph_is_eager", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let run =
            |m: &mut GpuModel<Body>, mode: StepMode| -> Result<(Vec<u32>, Vec<f32>), GateError> {
                m.set_mode(mode);
                m.reset()?;
                if mode == StepMode::Graph {
                    m.capture_step()?;
                }
                let mut ids = vec![feed(m, &s.prompt)?];
                while ids.len() <= GRAPH_STEPS {
                    let last = ids[ids.len() - 1];
                    ids.push(m.step(&[last])?);
                }
                Ok((ids, m.logits()?))
            };
        let eager = run(m, StepMode::Eager)?;
        let graph = run(m, StepMode::Graph)?;
        m.set_mode(StepMode::Eager);
        let same_bits = bits_equal(&eager.1, &graph.1);
        let pass = eager.0 == graph.0 && same_bits;
        println!(
            "graph_is_eager set={} eager={:?} graph={:?} logits_same_bits={same_bits} {}",
            s.name,
            eager.0,
            graph.0,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `cut_restores_the_checkpoint`: `s`'s prompt as a call (its end a checkpoint),
    /// [`CUT_STEPS`] greedy steps (past a whole window ring and a compressor block, so every
    /// slot of both rings is overwritten), a rollback to the prompt's end, then the prompt's
    /// ids again as a second call, give the argmax and the last logits row bit for bit of the
    /// same two calls from a reset with no steps between them. The two runs make the same calls
    /// at the same positions, so only the restored rings can differ.
    fn cut_restores(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        const CLAUSE: &str = "cut_restores_the_checkpoint";
        if !tier::run_clause(CLAUSE, Tag::SelfConsistency)? {
            return Ok(true);
        }
        if m.body(NAME)?.pass_rows().is_none() {
            return Err(format!("{NAME}: {CLAUSE} needs a load with a prompt pass").into());
        }
        m.set_mode(StepMode::Eager);
        let at = u32::try_from(s.prompt.len())?;
        m.reset()?;
        let mut next = feed(m, &s.prompt)?;
        for _ in 0..CUT_STEPS {
            next = m.step(&[next])?;
        }
        m.rollback(at)?;
        let cut_id = feed(m, &s.prompt)?;
        let cut_row = m.logits()?;
        m.reset()?;
        let _ = feed(m, &s.prompt)?;
        let fresh_id = feed(m, &s.prompt)?;
        let fresh_row = m.logits()?;
        let same_bits = bits_equal(&cut_row, &fresh_row);
        let pass = cut_id == fresh_id && same_bits;
        println!(
            "{CLAUSE} set={} at={at} steps={CUT_STEPS} ids cut={cut_id} fresh={fresh_id} \
             logits_same_bits={same_bits} {}",
            s.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `opens_the_fixture` of one arm: its kinds, and its hosted layers those `want_hosts`
    /// asks (none, or every routed layer).
    fn opens(arm: &Arm, hosts: bool, want_hosts: bool) -> Result<bool, GateError> {
        let body = arm.m.body(NAME)?;
        let kinds = body.kinds();
        let hosted = body.hosted();
        let routed: Vec<usize> = (0..KINDS.len()).filter(|&l| !KINDS[l].1).collect();
        let want: Vec<usize> = if want_hosts { routed } else { Vec::new() };
        let pass = kinds == KINDS && hosts == want_hosts && hosted == want;
        println!(
            "opens_the_fixture arm={} kinds={kinds:?} plan_hosts={hosts} hosted={hosted:?} \
             pass_rows={:?} resident_bytes={} {}",
            arm.name,
            body.pass_rows(),
            arm.m.resident_bytes(),
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let gguf = hosted_gguf()?;
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the released folder's sets (ref_ernie55_short, ref_ernie55_long) have no \
                 refset family yet, so the gate runs the fixture tier: BLOOMERY_TIER=fixture just \
                 {RECIPE}"
            )
            .into());
        }
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set("the reference clauses", n))
            .collect::<Result<_, _>>()?;
        let opened = tier::run_clause("opens_the_fixture", Tag::SelfConsistency)?;
        let (mut card, card_hosts) = open_arm("card", false, None, Residency::Off)?;
        let mut hosted = match &gguf {
            Some(path) => Some(open_arm("hosted", true, Some(path), Residency::Off)?),
            None => {
                println!("{HOSTED_REFUSED}");
                None
            }
        };
        let mut ok = true;
        if opened {
            ok &= opens(&card, card_hosts, false)?;
            if let Some((arm, hosts)) = &hosted {
                ok &= opens(arm, *hosts, true)?;
            }
            if !ok {
                return Err(checks_failed());
            }
        }
        if let Some((arm, _)) = &hosted {
            ok &= started_off(arm)?;
        }
        let named = |name: &str| {
            sets.iter()
                .find(|s| s.name == name)
                .ok_or_else(|| format!("{NAME}: the sets hold no {name}"))
        };
        let (short, long) = (named(SHORT)?, named(LONG)?);
        let mut arms = vec![&mut card];
        arms.extend(hosted.as_mut().map(|(arm, _)| arm));
        for arm in arms {
            let (gpu, _, body) = arm.m.body_parts(NAME)?;
            body.arm_taps(gpu, TAP_ROWS)?;
            ok &= reset_is_fresh(arm, long)?;
            for s in &sets {
                ok &= reference_clauses(arm, s)?;
            }
        }
        ok &= reset_clears_the_caches(&mut card, (long, short))?;
        let arms = if hosted.is_some() {
            "card, hosted and hosted-mid"
        } else {
            "card, hosted refused"
        };
        // The hosted arm closes before its file opens again under the residency.
        drop(hosted);
        if let Some(path) = &gguf {
            ok &= resident_arm(path, &card, short, opened)?;
        }
        tier::expect_fixture_oracle(
            NAME,
            if gguf.is_some() {
                2 * FX_ARM_STEPPED + FX_RESIDENT
            } else {
                FX_ARM_STEPPED
            },
        )?;
        ok &= graph_is_eager(&mut card.m, &sets[0])?;
        ok &= cut_restores(&mut card.m, &sets[0])?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} on {SETS:?}, arms {arms}: every layer within \
             {L_OUT_BAND:.1e}, the writer's picks with weights within {WEIGHTS_BAND:.1e}, the \
             selection within {SEL_SWAPS} swaps, the logits before and after MultiMax within \
             {LOGITS_BAND:.1e}, the teacher-forced argmax within the pin; a reset model writes \
             a fresh open's bits and holds its caches cleared; experts promoted into other slots are read within the same \
             bands; the graph is the eager chain; the unset residency word is the serving word on \
             the hosted plan (off on the whole-card plan, beside a fixed placement and the step \
             feed), the mid load runs a machine seeded by its plan, its history's boundaries end \
             none, the prompt call and each step, the history repeats bit for bit with flips \
             landing, and a session's clear restores the seed"
        );
        Ok(())
    }
}
