//! The history the Qwen3.8 tier gates hold a load to (`gate_qwen38_twocard`,
//! `gate_qwen38_nodes`): from a clear, the first [`PROMPT`] ids of the
//! qwen4exp prose corpus one decode step each, [`STEPS`] greedy steps, the
//! verifies of [`WINDOWS`] each committed at a fixed count, then [`STEPS`]
//! greedy steps more; every token and logits row it read, in order, and its
//! stage graph nodes ([`Run`]), and a run held to another bit for bit
//! ([`same`]). Also the tier's structure after the history
//! ([`tier_graphs_and_chains`]) over the plan's tier layers ([`tier_layers`]).

use app::Session;
use bloomery_gpu::arch::qwen3moe::Body38;
use bloomery_gpu::host::tier::{TierAct, TierCard};
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu_gates::{GateError, verdict};
use model::placement::Plan;
use runtime::{Out, Target, Verify, Want};

/// Prompt positions, fed one decode step each.
pub const PROMPT: usize = 32;
/// Greedy steps before the verifies and after them.
pub const STEPS: usize = 24;
/// The verify windows: their rows and the rows each keeps.
pub const WINDOWS: [(usize, usize); 6] = [(4, 2), (3, 3), (2, 1), (4, 4), (2, 2), (3, 1)];
/// Nodes of a tier graph a tier layer, nine: the go wait, the
/// activation's and the places' copies in, the card leg's four launches
/// into the tier card's rows, the rows' copy out to the page, the signal.
pub const TIER_LAYER_NODES: usize = TierCard::nodes_per_layer(TierAct::F32, 4);
// PIN(2026-10-06): 8 -> 9, the rows' copy out; launches storing into the page held 385 against 433.
const _: () = assert!(TIER_LAYER_NODES == 9);

/// Every token and logits row a history read, in order, its stage graph
/// nodes (the step's, then each verify width's), and the flips that landed
/// over it where a gate counts them.
#[derive(Default)]
pub struct Run {
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub stage_nodes: Vec<usize>,
    #[allow(
        dead_code,
        reason = "gate_qwen38_twocard counts the residency's flips; gate_qwen38_nodes runs no residency"
    )]
    pub landed: usize,
}

/// One greedy step of `next`, its token and row into `run`.
pub fn step(s: &mut Session<Body38>, run: &mut Run, next: u32) -> Result<u32, GateError> {
    let Out::Logits { argmax, row } = s.step(next, Want::Logits)? else {
        return Err("a step asked for its logits gave the argmax alone".into());
    };
    run.tokens.push(argmax);
    run.logits.push(row.to_vec());
    Ok(argmax)
}

/// A verify of `M` rows (`next`, then `fill`'s ids) committed at `keep`:
/// every row's token and logits into `run`; the next token, the last kept
/// row's argmax.
fn window<const M: usize>(
    s: &mut Session<Body38>,
    run: &mut Run,
    next: u32,
    fill: &[u32],
    keep: usize,
) -> Result<u32, GateError> {
    let mut rows = [next; M];
    rows[1..].copy_from_slice(&fill[..M - 1]);
    let out = s.verify::<M>(rows)?;
    run.tokens.extend_from_slice(&out);
    run.logits.extend(s.model().rows_logits::<M>()?);
    s.commit(keep)?;
    Ok(out[keep - 1])
}

/// The history (module doc) from a clear over `ids`; `cleared` runs right
/// after the clear, before the first step.
pub fn history(
    s: &mut Session<Body38>,
    ids: &[u32],
    cleared: impl FnOnce(&mut Session<Body38>) -> Result<(), GateError>,
) -> Result<Run, GateError> {
    s.clear()?;
    cleared(s)?;
    let mut run = Run::default();
    let mut next = 0;
    for &id in &ids[..PROMPT] {
        next = step(s, &mut run, id)?;
    }
    for _ in 0..STEPS {
        next = step(s, &mut run, next)?;
    }
    let mut at = PROMPT;
    for (m, keep) in WINDOWS {
        let fill = &ids[at..];
        next = match m {
            2 => window::<2>(s, &mut run, next, fill, keep)?,
            3 => window::<3>(s, &mut run, next, fill, keep)?,
            _ => window::<4>(s, &mut run, next, fill, keep)?,
        };
        at += m;
    }
    for _ in 0..STEPS {
        next = step(s, &mut run, next)?;
    }
    let m = s.model();
    run.stage_nodes = vec![
        m.step_graph_nodes()?.len(),
        m.rows_graph_nodes::<2>()?.len(),
        m.rows_graph_nodes::<3>()?.len(),
        m.rows_graph_nodes::<4>()?.len(),
    ];
    Ok(run)
}

/// Whether `got` is `want` bit for bit; prints the first difference.
pub fn same(what: &str, want: &Run, got: &Run) -> bool {
    if want.tokens != got.tokens {
        let at = want
            .tokens
            .iter()
            .zip(&got.tokens)
            .position(|(a, b)| a != b);
        println!(
            "FAIL {what}: tokens apart at {at:?} ({} and {} tokens)",
            got.tokens.len(),
            want.tokens.len()
        );
        return false;
    }
    for (i, (w, g)) in want.logits.iter().zip(&got.logits).enumerate() {
        let first = w
            .iter()
            .zip(g)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        if w.len() != g.len() || first.is_some() {
            println!("FAIL {what}: logits row {i}: first logit apart {first:?}");
            return false;
        }
    }
    let ok = want.logits.len() == got.logits.len();
    println!(
        "{} {what}: {} tokens, {} logits rows bit for bit the reference's",
        if ok { "ok" } else { "FAIL" },
        got.tokens.len(),
        got.logits.len()
    );
    ok
}

/// The tier layers: the plan's layers with a tier expert.
pub fn tier_layers(plan: &Plan<'_>) -> Vec<usize> {
    plan.tier_n_l
        .first()
        .map(|t| (0..t.len()).filter(|&l| t[l] > 0).collect())
        .unwrap_or_default()
}

/// `tier`'s structure after the history over `layers` tier layers, one
/// line a check: its graphs of the step and of each verify width hold
/// [`TIER_LAYER_NODES`] a tier layer and the fault copy, and each of the
/// history's chains was fed and captured once and replayed by its other
/// passes.
pub fn tier_graphs_and_chains(tier: &TierCard, layers: usize) -> bool {
    let want = Some(TIER_LAYER_NODES * layers + 1);
    let graphs: Vec<Option<usize>> = [Chain::Step, Chain::Cols(2), Chain::Cols(3), Chain::Cols(4)]
        .iter()
        .map(|&c| tier.graph_nodes(c))
        .collect();
    let graphs_ok = graphs.iter().all(|&g| g == want);
    println!(
        "structure: the tier's graphs of the step and the 2, 3, 4-column verifies hold {graphs:?} \
         nodes, {TIER_LAYER_NODES} a tier layer and the fault copy ({want:?}): {}",
        verdict(graphs_ok)
    );
    let st = tier.stats();
    // The history's passes: the steps, then one verify a window; its
    // chains: the step's and one a distinct window width. Each chain's
    // first pass is fed and its graph captured after it.
    let mut widths: Vec<usize> = WINDOWS.iter().map(|w| w.0).collect();
    widths.sort_unstable();
    widths.dedup();
    let chains = 1 + widths.len() as u64;
    let passes = (PROMPT + 2 * STEPS + WINDOWS.len()) as u64;
    let replays = (st.feeds, st.captures, st.replays) == (chains, chains, passes - chains);
    println!(
        "structure: the tier's chains fed {} and captured {} times, replayed {} times; each of \
         the history's {chains} chains fed and captured once, its {passes} passes' others \
         replayed ({chains}, {chains}, {}): {}",
        st.feeds,
        st.captures,
        st.replays,
        passes - chains,
        verdict(replays)
    );
    graphs_ok && replays
}
