//! The end-to-end gate of a model whose oracle is a llama.cpp tree's node dump (`tools/ref/cb_dump.cpp`, the shared dumper: a
//! prompt set a prompt, each holding the last position's whole logits row `result_output`, optionally each layer's output
//! `l_out-L`, the 40 largest logprobs `topk.tsv` and the greedy continuation `greedy.tsv`, read through
//! `refset::arch::prompt_tables`). A gate includes it by `#[path]` beside `shared/e2e.rs` and `shared/gate_card.rs` (as modules
//! `lcpp_e2e`, `e2e` and `gate_card` of its crate) and holds only what the model forces: a [`Gate`] descriptor and its seat
//! (the [`Seat`] the body is driven through, its tap ceilings, its load).
//!
//! Each prompt of the family's sets is fed one id a step from a reset, and the oracle's reads are the set's own. The clauses are
//! a list ([`Clause`]); each prints lines ending in PASS or FAIL and carries its tier tag (`tier::run_clause`):
//! - `finite_and_fault_clean` (self-consistency): every prompt's logits row and layer outputs are finite and the fault word is
//!   clean.
//! - `reset_rewinds` (self-consistency): after a reset prompt 0 gives the same logits and layer outputs bit for bit.
//! - `layer_outputs_near_<slug>` (oracle) and its fixture twin `layer_outputs_near_<slug>_fx` (fixture-oracle, declared only by a
//!   gate with a fixture family): each layer's output at the last position of the sets that hold `l_out-L`, by the relative
//!   distance `‖ours − oracle‖ / ‖oracle‖`, every layer within [`Gate::tap_ceil`]. The first layer past it is named.
//! - `top_logprobs_in_band` (oracle): on each prompt, over the ids both our top 40 and the oracle's top 40 hold, the log-softmax
//!   of the logits row within the error model's band of the oracle's logprob ([`sigma_lp`] an id, the worst of the ids within
//!   [`k_of`]), and our top 10 sharing at least [`overlap_floor`] ids with the oracle's. Both are two implementations' difference
//!   of one quantized model ([`SIGMA_LOGIT`]); the argmax is printed, not a clause.
//! - `forced_greedy_count_pin` (oracle): the oracle's greedy tokens fed as input (teacher-forced), our argmax at each position;
//!   the disagreements at an oracle margin of at least `MARGIN_FLOOR` are held to the pin derived from the margins themselves
//!   ([`forced_pin`]). A set with no greedy continuation is left out.
//! - `prefill_near_<slug>` (oracle, declared only by a gate whose seat has a prompt batch, [`Gate::has_prefill`]): the same
//!   prompts fed by the prompt batch ([`Seat::prompt`]) from a reset, their layer outputs and logprobs held to the bands the
//!   steps are.
//!
//! The real tier runs every clause but the fixture's; the fixture tier runs the self-consistency clauses and the fixture-oracle
//! one and leaves the others to the real tier by name. A gate with no fixture family refuses the fixture tier by name before
//! anything loads. A set absent from the box, dumped from another file or build, or unfinished ends the gate by name before the
//! card loads.
//!
//! `--arm a|bp` picks the placement arm ([`arm_arg`]); a seat refuses by name an arm it does not serve.

use crate::e2e::{argmax, ik_last, rel, tap, top_ids};
use bloomery_gpu_gates::prompts::{
    GreedyRow, MARGIN_FLOOR, SIGMA_LOGIT, compare_forced, k_of, log_softmax, sigma_lp,
    std_normal_cdf,
};
use bloomery_gpu_gates::tier::{self, Tag, Tier};
use bloomery_gpu_gates::{
    GateError, RefManifest, bits_equal, checks_failed, ref_model_path, verdict,
};
use gguf::Split;
use refset::arch::prompt_tables::{self as read, Greedy, Top};
use refset::family::Family;
use std::path::{Path, PathBuf};

// ------------------------------------------------------------ the seat

/// What the loaded file's shape is: the layers a tap table has, the width of a layer's output and of a logits row.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub layers: usize,
    pub embd: usize,
    pub vocab: usize,
}

/// One loaded body on the gate card, as a gate drives it. The contract of each call is the body's, fixed here so the gates and
/// the body are written against one text:
/// - [`Seat::prompt`] runs `ids` from the seat's position by the prompt batch and answers the last id's logits row;
///   [`Seat::step`] runs one id by the captured step;
/// - [`Seat::taps`] is `layers * embd` values, layer `l`'s output at the last position of the last call at `[l * embd..]`,
///   whichever path ran it;
/// - [`Seat::stores`] names every store the written positions fill with a 64-bit digest of its bytes ([`fnv1a`]), in layer order;
/// - [`Seat::fault`] is `Ok` while the fault word is clean and the model not poisoned.
///
/// A body without a prompt batch refuses [`Seat::prompt`], [`Seat::set_group`] and [`Seat::stores`] by name and its gate's
/// descriptor says [`Gate::has_prefill`] is false.
pub trait Seat {
    fn dims(&self) -> Dims;

    /// Empty every store; the position is 0.
    fn reset(&mut self) -> Result<(), GateError>;

    /// The position the next id lands at.
    fn pos(&self) -> u32;

    /// Run `id` by the captured step; the logits row after it.
    fn step(&mut self, id: u32) -> Result<Vec<f32>, GateError>;

    /// Run `ids` by the prompt batch; the logits row after the last.
    fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError>;

    /// Set the prompt group to `group` units a walk (`BLOOMERY_PREFILL_GROUP`'s meaning).
    fn set_group(&mut self, group: usize) -> Result<(), GateError>;

    /// Every layer's output at the last position of the last call.
    fn taps(&mut self) -> Result<Vec<f32>, GateError>;

    /// Every store's name and digest over the positions written so far.
    fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError>;

    /// `Ok` while the card's fault word is clean and the model is not poisoned.
    fn fault(&mut self) -> Result<(), GateError>;
}

/// The placement arm a gate runs: plan (a), the largest card the stage with the experts that fit on it and the rest on the host,
/// or plan (b′), the next-largest card an expert tier under the host tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    A,
    Bp,
}

impl Arm {
    /// The arm a `--arm` word names; any other word is refused by name.
    pub fn parse(word: &str) -> Result<Arm, GateError> {
        match word {
            "a" => Ok(Arm::A),
            "bp" => Ok(Arm::Bp),
            _ => Err(format!("--arm is a or bp, got {word:?}").into()),
        }
    }

    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Arm::A => "a",
            Arm::Bp => "bp",
        }
    }
}

/// The arm the command line names: `--arm <a|bp>`, `a` without it. An argument it does not know is refused by name.
pub fn arm_arg() -> Result<Arm, GateError> {
    let mut args = std::env::args().skip(1);
    let mut arm = Arm::A;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arm" => arm = Arm::parse(&args.next().ok_or("--arm wants a or bp")?)?,
            other => {
                return Err(
                    format!("unknown argument {other:?}: the gate takes --arm a|bp").into(),
                );
            }
        }
    }
    Ok(arm)
}

/// FNV-1a over bytes: the digest a store is held to.
#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The model file the profile names, checked before anything loads: an `arch` file, and under the fixture tier a whole fixture
/// (`tier::Tier::check_file`). `recipe` is the `just` recipe that picks the profile.
pub fn model_path(tier_now: Tier, recipe: &str, arch: &str) -> Result<PathBuf, GateError> {
    let path = ref_model_path()?;
    let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if file.architecture() != Some(arch) {
        return Err(format!(
            "the model file is {:?}, want {arch} — run through `just {recipe}`, which picks the {arch} profile",
            file.architecture()
        )
        .into());
    }
    let keys = [tier::VERSION_KEY, tier::SUBSET_KEY]
        .into_iter()
        .filter(|k| file.value(k).is_some());
    tier_now.check_file(&path, keys)?;
    Ok(path)
}

// ------------------------------------------------------------ the descriptor

/// A seat's load: the file on the gate card at `ctx` positions of `arm`, the taps armed.
pub type OpenSeat = fn(file: &Path, ctx: usize, arm: Arm) -> Result<Box<dyn Seat>, GateError>;

/// What a model gives the engine: the rest is the engine's.
pub struct Gate {
    /// The gate bin's name: the prefix of its refusals and of its PASSED line.
    pub name: &'static str,
    /// The `just` recipe that runs the gate under the model's profile.
    pub recipe: &'static str,
    /// The GGUF `general.architecture` the model file must name.
    pub arch: &'static str,
    /// The oracle as a clause name spells it (`layer_outputs_near_<slug>`) and as a line prints its answers.
    pub slug: &'static str,
    /// The oracle as prose (`<who>'s l_out-L`).
    pub who: &'static str,
    /// The oracle's family in the real tier: its sets, in the order the gate reads them.
    pub real: &'static Family,
    /// The oracle's family on the fixture file, `None` for a model with no fixture file (the fixture tier is then refused by
    /// name, and the fixture-oracle clause is not declared).
    pub fixture: Option<&'static Family>,
    /// The ceiling of layer `layer`'s output distance in `tier`'s file.
    pub tap_ceil: fn(tier: Tier, layer: usize) -> f64,
    /// The ceilings of [`Gate::tap_ceil`] as a line prints them.
    pub tap_ceils: fn(tier: Tier) -> String,
    /// The seat's load.
    pub open: OpenSeat,
    /// Whether the seat has a prompt batch: the `prefill_near_<slug>` clause is declared only then.
    pub has_prefill: bool,
}

/// The clauses, in the order they run.
#[derive(Clone, Copy)]
enum Clause {
    Finite,
    Reset,
    Layers,
    LayersFx,
    Top,
    Forced,
    Prefill,
}

impl Clause {
    const ALL: [Clause; 7] = [
        Clause::Finite,
        Clause::Reset,
        Clause::Layers,
        Clause::LayersFx,
        Clause::Top,
        Clause::Forced,
        Clause::Prefill,
    ];

    fn name(self, slug: &str) -> String {
        match self {
            Clause::Finite => "finite_and_fault_clean".to_string(),
            Clause::Reset => "reset_rewinds".to_string(),
            Clause::Layers => format!("layer_outputs_near_{slug}"),
            Clause::LayersFx => format!("layer_outputs_near_{slug}_fx"),
            Clause::Top => "top_logprobs_in_band".to_string(),
            Clause::Forced => "forced_greedy_count_pin".to_string(),
            Clause::Prefill => format!("prefill_near_{slug}"),
        }
    }

    fn tag(self) -> Tag {
        match self {
            Clause::Finite | Clause::Reset => Tag::SelfConsistency,
            Clause::LayersFx => Tag::FixtureOracle,
            Clause::Layers | Clause::Top | Clause::Forced | Clause::Prefill => Tag::Oracle,
        }
    }

    /// Whether `g` declares the clause.
    fn declared(self, g: &Gate) -> bool {
        match self {
            Clause::LayersFx => g.fixture.is_some(),
            Clause::Prefill => g.has_prefill,
            _ => true,
        }
    }

    fn run(self, seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        match self {
            Clause::Finite => finite_and_fault(seat, case),
            Clause::Reset => reset_rewinds(seat, case),
            Clause::Layers | Clause::LayersFx => layer_outputs(seat, case),
            Clause::Top => top_logprobs(seat, case),
            Clause::Forced => forced_greedy(seat, case),
            Clause::Prefill => prefill_near(seat, case),
        }
    }
}

/// Positions of the cache beyond the longest prompt and its continuation.
const CTX_SPARE: usize = 16;

/// The standard deviation of the top1-top2 margin of ours against the oracle's: two implementations ([`SIGMA_LOGIT`]; the
/// oracle's CPU dump is the second), each σ from the exact one.
const SIGMA_PAIR: f64 = SIGMA_LOGIT * std::f64::consts::SQRT_2;

/// The draws [`overlap_floor`] takes.
const DRAWS: usize = 20_000;

/// The top-10 overlap floor against the row `lp` (log-probabilities) over its ranked ids `top`, and the mean overlap the
/// model expects.
///
/// [derived]: two implementations' logits differ by an independent N(0, σ²) an id ([`SIGMA_LOGIT`]). The ten of `top`
/// so moved are drawn [`DRAWS`] times (a fixed seed, so the floor is the row's alone; the ids past `top` trail its last by
/// more), and the floor is the least overlap the draws fall below no more often than one Gaussian past three standard
/// deviations, Φ(−3).
fn overlap_floor(lp: &[f64], top: &[usize]) -> (f64, usize) {
    let ten = top.len().min(10);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    // splitmix64, and its top 53 bits as a uniform in (0, 1).
    let mut unit = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 + 0.5 / (1u64 << 53) as f64
    };
    let mut hist = [0usize; 11];
    let mut moved: Vec<(f64, usize)> = Vec::with_capacity(top.len());
    for _ in 0..DRAWS {
        moved.clear();
        for (rank, &id) in top.iter().enumerate() {
            // Box-Muller: one standard normal from two uniforms.
            let g = (-2.0 * unit().ln()).sqrt() * (std::f64::consts::TAU * unit()).cos();
            moved.push((lp[id] + SIGMA_LOGIT * g, rank));
        }
        moved.sort_by(|x, y| y.0.total_cmp(&x.0));
        hist[moved[..ten].iter().filter(|m| m.1 < ten).count()] += 1;
    }
    let mean = hist
        .iter()
        .enumerate()
        .map(|(c, &n)| (c * n) as f64)
        .sum::<f64>()
        / DRAWS as f64;
    let rate = std_normal_cdf(-3.0) * DRAWS as f64;
    let (mut below, mut floor) = (0usize, 0usize);
    for (c, &n) in hist.iter().enumerate() {
        if (below + n) as f64 > rate {
            floor = c;
            break;
        }
        below += n;
    }
    (mean, floor)
}

/// One prompt's oracle: its set, ids, the oracle's last logits row, top table and greedy continuation.
struct Oracle {
    name: &'static str,
    man: RefManifest,
    ids: Vec<u32>,
    logits: Vec<f32>,
    top: Vec<Top>,
    greedy: Greedy,
    /// Whether the set holds every layer's output.
    layer_outputs: bool,
}

/// What the body did on a prompt, fed one id a step from a reset.
struct Run {
    logits: Vec<f32>,
    taps: Vec<f32>,
}

struct Case<'g> {
    gate: &'g Gate,
    tier: Tier,
    dims: Dims,
    oracles: Vec<Oracle>,
    runs: Vec<Run>,
}

// ------------------------------------------------------------ the oracle

/// The set `name` of `family` opened through its check and read: the tier decides the path (the real set under
/// `Family::path`, the fixture's through `tier::fixture_set`).
fn open_oracle(
    g: &Gate,
    tier_now: Tier,
    name: &'static str,
    family: &'static Family,
) -> Result<Oracle, GateError> {
    let path = match tier_now {
        Tier::Real => family.path(name),
        Tier::Fixture => tier::fixture_set(&format!("the {} clauses", g.who), family, name)?,
    };
    let man = RefManifest::open(&path, family)?;
    let ids = read::tokens(&man)?.to_vec();
    let vocab = usize::try_from(man.tensor("result_output", 0)?.ne[0])?;
    let logits = ik_last(&man, vocab)?;
    let top = read::topk(&man)?;
    let greedy = read::greedy(&man)?;
    // The set's own tables are the logits row's: a dump whose top-k table disagrees with its logits is no oracle.
    let lp = log_softmax(&logits);
    let consistent = top.first().map(|t| t.id) == Some(argmax(&logits))
        && top
            .iter()
            .all(|t| (lp[t.id as usize] - t.logprob).abs() < 1e-4);
    if !consistent {
        return Err(format!(
            "{}: topk.tsv is not the log-softmax of its result_output row",
            path.display()
        )
        .into());
    }
    let layer_outputs = man.tensor("l_out-0", 0).is_ok();
    Ok(Oracle {
        name,
        man,
        ids,
        logits,
        top,
        greedy,
        layer_outputs,
    })
}

/// Every set of the tier's family, opened. A fixture tier with no fixture family is refused by name.
fn open_oracles(g: &Gate, tier_now: Tier) -> Result<Vec<Oracle>, GateError> {
    let family = match tier_now {
        Tier::Real => g.real,
        Tier::Fixture => g.fixture.ok_or_else(|| {
            format!(
                "{}: the {} family has no fixture file, so the gate has no fixture tier: run it in the real tier (`just {}`)",
                g.name, g.real.name, g.recipe
            )
        })?,
    };
    family
        .sets
        .iter()
        .map(|&n| open_oracle(g, tier_now, n, family))
        .collect()
}

// ------------------------------------------------------------ the body

/// A reset, then `ids` one a step; the last row.
fn feed(seat: &mut dyn Seat, ids: &[u32]) -> Result<Vec<f32>, GateError> {
    seat.reset()?;
    let mut row = Vec::new();
    for &id in ids {
        row = seat.step(id)?;
    }
    Ok(row)
}

fn run_of(seat: &mut dyn Seat, ids: &[u32]) -> Result<Run, GateError> {
    let logits = feed(seat, ids)?;
    Ok(Run {
        logits,
        taps: seat.taps()?,
    })
}

/// A reset, then `ids` as one prompt call.
fn run_by_prompt(seat: &mut dyn Seat, ids: &[u32]) -> Result<Run, GateError> {
    seat.reset()?;
    let logits = seat.prompt(ids)?;
    Ok(Run {
        logits,
        taps: seat.taps()?,
    })
}

// ------------------------------------------------------------ the clauses

fn finite_and_fault(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let mut ok = true;
    for (o, r) in case.oracles.iter().zip(&case.runs) {
        let finite = r.logits.iter().chain(&r.taps).all(|v| v.is_finite())
            && r.logits.len() == case.dims.vocab
            && r.taps.len() == case.dims.layers * case.dims.embd;
        println!(
            "finite_and_fault_clean: prompt {} {} {}",
            o.name,
            o.ids.len(),
            verdict(finite)
        );
        ok &= finite;
    }
    let fault = seat.fault();
    println!(
        "finite_and_fault_clean: fault word {fault:?} {}",
        verdict(fault.is_ok())
    );
    Ok(ok && fault.is_ok())
}

fn reset_rewinds(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let (o, first) = (&case.oracles[0], &case.runs[0]);
    let again = run_of(seat, &o.ids)?;
    let pass = bits_equal(&again.logits, &first.logits) && bits_equal(&again.taps, &first.taps);
    println!(
        "reset_rewinds: prompt {} again from a reset, logits and layer outputs bit for bit {}",
        o.name,
        verdict(pass)
    );
    Ok(pass)
}

/// Each layer's output at the last position against the oracle's `l_out-L`: the last `embd` values of the node (the last
/// layer's holds the last position alone, the others every position).
fn layer_outputs(_seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    layer_outputs_of("layer_outputs", case, &case.runs)
}

/// [`layer_outputs`] over `runs`, one a prompt of the case, its lines headed `label`.
fn layer_outputs_of(label: &str, case: &Case, runs: &[Run]) -> Result<bool, GateError> {
    let Dims { layers, embd, .. } = case.dims;
    let mut ok = true;
    for (o, r) in case
        .oracles
        .iter()
        .zip(runs)
        .filter(|(o, _)| o.layer_outputs)
    {
        let mut table = Vec::with_capacity(layers);
        for l in 0..layers {
            let theirs = tap(&o.man, &format!("l_out-{l}"))?;
            let row = theirs
                .len()
                .checked_sub(embd)
                .map(|at| &theirs[at..])
                .ok_or_else(|| {
                    format!(
                        "{}: l_out-{l} holds {} values, a layer output is {embd}",
                        o.name,
                        theirs.len()
                    )
                })?;
            table.push(rel(&r.taps[l * embd..(l + 1) * embd], row));
        }
        let (worst, at) =
            table.iter().enumerate().fold(
                (0.0f64, 0usize),
                |w, (l, &d)| if d > w.0 { (d, l) } else { w },
            );
        let first = table
            .iter()
            .enumerate()
            .position(|(l, &d)| d > (case.gate.tap_ceil)(case.tier, l));
        let pass = first.is_none();
        println!(
            "{label}: prompt {} ({} ids): worst {worst:.4} at layer {at}, {}, first past it {first:?} {}",
            o.name,
            o.ids.len(),
            (case.gate.tap_ceils)(case.tier),
            verdict(pass)
        );
        println!(
            "  by layer: {}",
            table
                .iter()
                .map(|d| format!("{d:.4}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        ok &= pass;
    }
    Ok(ok)
}

fn top_logprobs(_seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    top_logprobs_of("top_logprobs", case, &case.runs)
}

/// [`top_logprobs`] over `runs`, one a prompt of the case, its lines headed `label`.
// PIN(2026-10-20): the band was PAI's own cut-to-cut spread (a 0.424, b 0.438, c 0.683, d 0.853), one implementation's
// instability, not ours against it; now two implementations' ([`sigma_lp`], [`k_of`], [`overlap_floor`]). FAIL-first:
// s2step's M3 (worst 2.08 to 8.61) and M5 (d 2.28) are past the widest band an id can have, 3.99 · √2 · 0.35 = 1.97;
// M1, M2 and M4 share at most one top-10 id, under every floor.
fn top_logprobs_of(label: &str, case: &Case, runs: &[Run]) -> Result<bool, GateError> {
    let slug = case.gate.slug;
    let mut ok = true;
    for (o, r) in case.oracles.iter().zip(runs) {
        // The oracle's side from its own logits row, which `open_oracle` held its top table to.
        let (lp, theirs) = (log_softmax(&r.logits), log_softmax(&o.logits));
        let sq: f64 = theirs.iter().map(|&v| (2.0 * v).exp()).sum();
        let ours = top_ids(&r.logits, o.top.len());
        let their_top: Vec<usize> = o.top.iter().map(|t| t.id as usize).collect();
        let shared: Vec<usize> = their_top
            .iter()
            .copied()
            .filter(|id| ours.contains(id))
            .collect();
        let k = k_of(shared.len());
        // A NaN distance is out of band and wins the fold.
        let (z, at) = shared.iter().fold((0.0f64, 0usize), |w, &id| {
            let z =
                (lp[id] - theirs[id]).abs() / sigma_lp(theirs[id].exp(), sq).max(f64::MIN_POSITIVE);
            if z.is_nan() || z > w.0 { (z, id) } else { w }
        });
        let (e, floor) = overlap_floor(&theirs, &their_top);
        let overlap = their_top
            .iter()
            .take(10)
            .filter(|id| ours[..10].contains(id))
            .count();
        let pass = z <= k && overlap >= floor;
        println!(
            "{label}: prompt {} ({} ids): worst z {z:.2} at id {at} (|dlogprob| {:.4}, sigma {:.4}) over {} shared ids of {}, \
             bound {k:.2}; top-10 overlap {overlap}, floor {floor} (mean {e:.2}); argmax ours {} {slug} {} {}",
            o.name,
            o.ids.len(),
            (lp[at] - theirs[at]).abs(),
            sigma_lp(theirs[at].exp(), sq),
            shared.len(),
            o.top.len(),
            ours[0],
            o.top[0].id,
            verdict(pass)
        );
        ok &= pass;
    }
    Ok(ok)
}

/// The prompts fed by the prompt batch instead of steps: finite, then the layer outputs and the logprobs held to the steps'
/// own bands.
fn prefill_near(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let slug = case.gate.slug;
    let mut runs = Vec::with_capacity(case.oracles.len());
    for o in &case.oracles {
        runs.push(run_by_prompt(seat, &o.ids)?);
    }
    let finite = runs
        .iter()
        .all(|r| r.logits.iter().chain(&r.taps).all(|v| v.is_finite()));
    println!(
        "prefill_near_{slug}: every prompt's logits row and layer outputs finite {}",
        verdict(finite)
    );
    let layers = layer_outputs_of(&format!("prefill_near_{slug} layer_outputs"), case, &runs)?;
    let logprobs = top_logprobs_of(&format!("prefill_near_{slug} top_logprobs"), case, &runs)?;
    let fault = seat.fault();
    println!(
        "prefill_near_{slug}: fault word {fault:?} {}",
        verdict(fault.is_ok())
    );
    Ok(finite && layers && logprobs && fault.is_ok())
}

/// The disagreements the Gaussian error of two implementations predicts at the clear positions, and the pin that follows:
/// the expected count plus three standard deviations of it (Poisson), rounded up.
///
/// [derived]: at an oracle margin `m` the argmax flips with probability Φ(−m/σ_pair) ([`SIGMA_PAIR`]); the sum over the
/// positions at `m >= MARGIN_FLOOR` is the expected count of disagreements the floor classifies as clear.
// PIN(2026-10-20): provisional with `SIGMA_LOGIT`; the pin is computed from the set's margins, not a literal.
fn forced_pin(rows: &[GreedyRow]) -> (f64, usize) {
    let expected: f64 = rows
        .iter()
        .flat_map(|r| &r.gen_margins)
        .filter(|&&m| m >= MARGIN_FLOOR)
        .map(|&m| std_normal_cdf(-f64::from(m) / SIGMA_PAIR))
        .sum();
    let pin = (expected + 3.0 * expected.sqrt()).ceil();
    (expected, pin as usize)
}

fn forced_greedy(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let slug = case.gate.slug;
    let (mut rows, mut ours) = (Vec::new(), Vec::new());
    for o in case.oracles.iter().filter(|o| !o.greedy.ids.is_empty()) {
        let g = &o.greedy;
        let mut row = feed(seat, &o.ids)?;
        let mut table = vec![argmax(&row)];
        for &id in &g.ids[..g.ids.len() - 1] {
            row = seat.step(id)?;
            table.push(argmax(&row));
        }
        ours.push(table);
        rows.push(GreedyRow {
            id: rows.len(),
            n_tokens: o.ids.len(),
            argmax: o.top[0].id,
            top5: o.top.iter().take(5).map(|t| t.id).collect(),
            gen_ids: g.ids.clone(),
            gen_margins: g.margins.clone(),
        });
    }
    let report = compare_forced(&ours, &rows, MARGIN_FLOOR);
    let (expected, pin) = forced_pin(&rows);
    let pass = report.disagree_ge_floor <= pin;
    println!(
        "forced_greedy: positions={} disagree={} buckets={:?} ge_floor={} expected {expected:.2}, pin {pin}, max margin {:?} {}",
        report.positions,
        report.disagree.len(),
        report.buckets,
        report.disagree_ge_floor,
        report.max_margin,
        verdict(pass)
    );
    for p in &report.disagree {
        println!(
            "  prompt {} step {}: ours {} {slug} {} ({slug} margin {:.4})",
            rows[p.id].n_tokens, p.step, p.ours, p.theirs, p.ref_margin
        );
    }
    Ok(pass)
}

// ------------------------------------------------------------ run

/// The gate: the tier's sets opened, the body loaded, every declared clause run, the verdict.
pub fn run(g: &Gate) -> Result<(), GateError> {
    let tier_now = Tier::from_env()?;
    // The sets open before the card loads: one the tier runs, missing or stale, ends the gate here.
    let oracles = open_oracles(g, tier_now)?;
    let path = model_path(tier_now, g.recipe, g.arch)?;
    let longest = oracles
        .iter()
        .map(|o| o.ids.len() + o.greedy.ids.len())
        .max()
        .unwrap_or(0);
    let t = std::time::Instant::now();
    let arm = arm_arg()?;
    let mut seat = (g.open)(&path, longest + CTX_SPARE, arm)?;
    let dims = seat.dims();
    println!(
        "load arm={} layers={} embd={} vocab={} ctx={} in {:.1} s (runtime value)",
        arm.word(),
        dims.layers,
        dims.embd,
        dims.vocab,
        longest + CTX_SPARE,
        t.elapsed().as_secs_f64()
    );
    let mut runs = Vec::with_capacity(oracles.len());
    for o in &oracles {
        if o.logits.len() != dims.vocab {
            return Err(format!(
                "{}: {}'s logits row is {} wide, the body's vocabulary {}",
                o.name,
                g.who,
                o.logits.len(),
                dims.vocab
            )
            .into());
        }
        runs.push(run_of(seat.as_mut(), &o.ids)?);
    }
    let case = Case {
        gate: g,
        tier: tier_now,
        dims,
        oracles,
        runs,
    };
    let clauses: Vec<Clause> = Clause::ALL.into_iter().filter(|c| c.declared(g)).collect();
    let mut ok = true;
    for clause in &clauses {
        let name = clause.name(g.slug);
        if !tier::run_clause(&name, clause.tag())? {
            continue;
        }
        let pass = clause.run(seat.as_mut(), &case)?;
        if !pass {
            println!("clause {name} FAIL");
        }
        ok &= pass;
    }
    println!("{}", tier::tally_line());
    let fixture_oracle = clauses
        .iter()
        .filter(|c| c.tag() == Tag::FixtureOracle)
        .count();
    tier::expect_fixture_oracle(g.name, fixture_oracle)?;
    if !ok {
        return Err(checks_failed());
    }
    println!(
        "PASSED: {} the logits and layer outputs are finite with the fault word clean; a reset rewinds them bit for bit; \
         every layer's output stays within its tier's ({}) distance of {}'s; in the real tier the shared top logprobs of each prompt stay within \
         two implementations' band of {}'s with the top-10 overlap held, and the teacher-forced greedy disagreements at a clear margin stay within the pin",
        g.name,
        (g.tap_ceils)(tier_now),
        g.who,
        g.who
    );
    Ok(())
}
