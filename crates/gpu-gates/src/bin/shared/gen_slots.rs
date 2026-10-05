//! `BLOOMERY_GEN_SLOTS`, the generate binaries' arm of several streams:
//! one arm's fed ids cut into N windows, window j prefilled into slot j,
//! then rounds of one pass of a row a slot (`GpuModel::step_slots`), each
//! slot fed its own argmax. What a model forces stays in its binary: its
//! family's refusals, its prompt call and the lines it prints. The cut, the
//! refusals every body makes, the pass's capture, the rounds and their
//! counted rate are here, for `generate_qwen3moe` and `generate_ds41` alike.

use app::Session;
use bloomery_gpu::model::SlotRows;
use bloomery_gpu_gates::GateError;
use std::time::Instant;

/// `BLOOMERY_GEN_SLOTS=<n>`, as every refusal names the lever.
fn lever(n: usize) -> String {
    format!("{}={n}", bloomery_levers::GEN_SLOTS)
}

/// `n` streams on body `B` (`arch` names it) refused by name: a pass of
/// more rows than `B`'s pass holds, one row a slot (`SlotRows::MAX_ROWS`),
/// and beside each flag of `beside` that is set, which the arm does not run.
pub fn refused<B: SlotRows>(
    n: usize,
    arch: &str,
    beside: &[(&str, bool)],
) -> Result<(), GateError> {
    let lever = lever(n);
    if n > B::MAX_ROWS {
        return Err(format!(
            "{lever} decodes {n} streams in one pass, a row a slot; the {arch} body's pass holds \
             at most {} rows (SlotRows::MAX_ROWS)",
            B::MAX_ROWS
        )
        .into());
    }
    if let Some((flag, _)) = beside.iter().find(|(_, set)| *set) {
        return Err(format!(
            "{lever} decodes several streams in one pass, each slot a window of the fed ids; \
             {flag} does not run with it"
        )
        .into());
    }
    Ok(())
}

/// `ids` (`what` names them) cut into `n` windows of one length, window j
/// slot j's prompt. Fewer ids than windows, or a count `n` does not divide,
/// is refused by name.
pub fn windows<'a>(ids: &'a [u32], n: usize, what: &str) -> Result<Vec<&'a [u32]>, GateError> {
    if n == 0 || ids.len() < n || !ids.len().is_multiple_of(n) {
        return Err(format!(
            "{} cuts {what} into {n} windows of one length, one a slot, and its {} ids do not",
            lever(n),
            ids.len()
        )
        .into());
    }
    Ok(ids.chunks_exact(ids.len() / n).collect())
}

/// The pass of a row a slot over slots `0..n`, captured into the model-wide
/// cache before the rounds so no round pays for it (`GpuModel::capture_slots`);
/// its node count. Graph mode only: an eager pass has nothing to capture.
pub fn capture<B: SlotRows>(s: &mut Session<B>, n: usize) -> Result<usize, GateError>
where
    B::Seq: 'static,
{
    let key: Vec<(usize, usize)> = (0..n).map(|j| (j, 1)).collect();
    Ok(s.model_mut().capture_slots(&key)?)
}

/// What the rounds ran: each slot's ids, token 0 first, and each round's
/// wall in ms.
pub struct Rounds {
    pub ids: Vec<Vec<u32>>,
    pub walls: Vec<f64>,
}

/// `n_gen` − 1 rounds of one pass of a row a slot, slot j fed its token 0
/// `first[j]` and then its own argmax. Each round is timed around the pass,
/// the pass's ids read back inside it; nothing is written between rounds.
/// A pass that gives other than a token a slot is refused by name.
pub fn rounds<B: SlotRows>(
    s: &mut Session<B>,
    first: Vec<u32>,
    n_gen: usize,
) -> Result<Rounds, GateError>
where
    B::Seq: 'static,
{
    let n = first.len();
    let mut ids: Vec<Vec<u32>> = first
        .iter()
        .map(|&t| {
            let mut v = Vec::with_capacity(n_gen);
            v.push(t);
            v
        })
        .collect();
    let mut walls: Vec<f64> = Vec::with_capacity(n_gen.saturating_sub(1));
    let mut next = first;
    for _ in 1..n_gen {
        let (out, ms) = {
            let rows: Vec<(usize, &[u32])> = next
                .iter()
                .enumerate()
                .map(|(j, t)| (j, std::slice::from_ref(t)))
                .collect();
            let t0 = Instant::now();
            let out = s.step_slots(&rows)?;
            (out, t0.elapsed().as_secs_f64() * 1e3)
        };
        if out.ids.len() != n {
            return Err(format!(
                "a pass of {n} slots at a row each gave {} ids",
                out.ids.len()
            )
            .into());
        }
        walls.push(ms);
        for (v, &tok) in ids.iter_mut().zip(&out.ids) {
            v.push(tok);
        }
        next = out.ids;
    }
    Ok(Rounds { ids, walls })
}

/// The rounds `--warm` keeps of a run of `n` slots: how many, the positions
/// they advanced (`n` a round), the p50 (the upper median) and mean ms a
/// round, and the aggregate rate, Σ positions · 1000 / Σ ms.
pub struct Counted {
    pub rounds: usize,
    pub positions: usize,
    pub p50: f64,
    pub mean: f64,
    pub aggregate: f64,
}

/// [`Counted`] over `walls` past the first `warm`; refused by name when
/// none is left.
pub fn counted(walls: &[f64], warm: usize, n: usize) -> Result<Counted, GateError> {
    let kept = walls.get(warm..).unwrap_or_default();
    if kept.is_empty() {
        return Err(format!(
            "--warm {warm} leaves no counted round of the {} this run took",
            walls.len()
        )
        .into());
    }
    let mut sorted = kept.to_vec();
    sorted.sort_by(f64::total_cmp);
    let ms: f64 = kept.iter().sum();
    let positions = kept.len() * n;
    Ok(Counted {
        rounds: kept.len(),
        positions,
        p50: sorted[sorted.len() / 2],
        mean: ms / kept.len() as f64,
        aggregate: positions as f64 * 1e3 / ms,
    })
}
