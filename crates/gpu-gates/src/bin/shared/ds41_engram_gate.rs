//! GPU gate for DeepSeek-V4.1's engram host half: the engram rows the step's
//! helper thread reads and the ones the calling thread reads itself are the
//! same bytes under the same ids ([`helper_rows`]), step by step over a
//! synthetic token stream of `HELPER_STEPS` steps — each step's rows are the
//! ones it waited for, not the other buffer of the helper's pair — and a
//! rank's columns of them are those columns of the whole. The gate launches no
//! kernel.
//!
//! No clause here reads ik's dump.
//! PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause

use bloomery_gpu_deepseek41::chain::glue::{RowsLevers, StepRows};
use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
use gguf::Split;
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::plan::{Planner, StepPlan};

pub fn run() -> Result<(), GateError> {
    let split = open_split(Arch::Deepseek41, "gate-gpu-ds41-kernels")?;
    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
    let pass = helper_rows(&split)?;
    println!(
        "gate_deepseek41_engram: host rows helper = direct {}",
        verdict(pass)
    );
    if !pass {
        return Err(checks_failed());
    }
    Ok(())
}

/// Steps of the synthetic stream [`helper_rows`] walks.
const HELPER_STEPS: usize = 256;

/// The host half's rows two ways over one synthetic token stream: a
/// [`StepRows`] whose helper thread reads the engram rows (keeping stats,
/// so the classifying path runs too) and one that reads them on this
/// thread, both filled from the same plan at every step; per step the
/// ids of every site and the engram bytes must be equal. The stream
/// repeats earlier tokens often, so n-grams recur and rows come back warm,
/// and draws the rest over the whole vocabulary.
fn helper_rows(split: &Split) -> Result<bool, GateError> {
    let hp = Hparams::read(split)?;
    let planner = Planner::from_file(split, &hp, HELPER_STEPS as u64)?;
    let all = 0..hp.engram()?.rows_per_token();
    // A rank's columns, as a rank of a tensor-parallel load reads them: the second quarter.
    let quarter = all.end / 4;
    let cols = quarter..2 * quarter;
    let open = |helper, cols| {
        StepRows::open(
            split,
            &hp,
            1,
            RowsLevers {
                helper,
                stats: helper,
            },
            cols,
            None,
        )
    };
    let mut helper = open(true, all.clone())?;
    let mut direct = open(false, all)?;
    let mut rank = open(true, cols.clone())?;
    let row_bytes = direct.row_bytes();
    let sites = hp.engram()?.layer_ids.len();
    let mut plan = StepPlan::default();
    let mut history: Vec<u32> = Vec::with_capacity(HELPER_STEPS);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let (mut bad_bytes, mut bad_ids, mut first_bad) = (0usize, 0usize, None);
    for pos in 0..HELPER_STEPS {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let draw = (state >> 33) as usize;
        let token = match history.len() {
            n if n > 0 && draw.is_multiple_of(3) => history[draw / 3 % n],
            _ => (draw % hp.n_vocab) as u32,
        };
        planner.plan_into(&[token], pos as u32, &history, &mut plan)?;
        helper.fill(split, &plan)?;
        direct.fill(split, &plan)?;
        rank.fill(split, &plan)?;
        let ids_eq = (0..sites).all(|s| {
            helper.ids(0, s) == direct.ids(0, s)
                && rank.ids(0, s) == &direct.ids(0, s)[cols.clone()]
        });
        let site_bytes = direct.engram().len() / sites;
        let rank_bytes = (0..sites).all(|s| {
            let site = &direct.engram()[s * site_bytes..][..site_bytes];
            let own = cols.start * row_bytes..cols.end * row_bytes;
            rank.engram()[s * own.len()..][..own.len()] == site[own]
        });
        let bytes_eq =
            rank_bytes && helper.engram() == direct.engram() && helper.embd() == direct.embd();
        bad_ids += usize::from(!ids_eq);
        bad_bytes += usize::from(!bytes_eq);
        if (!ids_eq || !bytes_eq) && first_bad.is_none() {
            first_bad = Some(pos);
        }
        history.push(token);
    }
    let st = helper.engram_stats();
    let pass = helper.helper_cpu().is_some() && bad_ids == 0 && bad_bytes == 0 && st.direct == 0;
    println!(
        "host rows: {HELPER_STEPS} steps x {sites} sites, helper {} and columns {cols:?} vs direct: \
             ids differ at {bad_ids} steps, bytes at {bad_bytes} (first {first_bad:?}); helper rows \
             warm {} cold {} direct {}: {}",
        match helper.helper_cpu() {
            None => "off".to_string(),
            Some(None) => "floating".to_string(),
            Some(Some(c)) => format!("on cpu{c}"),
        },
        st.warm,
        st.cold,
        st.direct,
        verdict(pass)
    );
    Ok(pass)
}
