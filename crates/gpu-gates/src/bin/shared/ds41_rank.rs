//! One rank of a V4.1 tensor-parallel group past the group's open: its load with the group's
//! sums, then the group's greedy run (`app::tp::greedy`). `generate_ds41 --place tp` and
//! `gate_deepseek41_tp`'s rank processes include it by `#[path]`.

use std::time::{Duration, Instant};

use app::tp::greedy;
use bloomery_gpu::Gpu;
use bloomery_gpu::host::rank::{RankError, RankSession, Round};
use bloomery_gpu_deepseek41::body::{self, BodyLevers, Deepseek41Model, RankFrom};
use bloomery_gpu_gates::GateError;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::tp::{TpUnits, rank_span};
use model::part::Source;
use model::placement::HEAD_TENSOR;

/// Rank `rank` of `ranks` in `session`: `from` loaded on `gpu` with the session's sums, holding
/// `ctx` positions under `levers`; then the group's greedy run of `prompt` (rank 0's; empty on
/// another rank) for `n` tokens, each round's ids stepped in order and the last one's pick and
/// logits sent. `seen` gets each round, the rank's logits and the time its steps took. The
/// group's tokens, the rank's model and how long its load took; a load that fails tells the group.
pub fn rank_greedy<S: Source + Sync>(
    gpu: Gpu,
    session: &mut RankSession,
    (rank, ranks): (usize, usize),
    from: RankFrom<'_, S>,
    (ctx, levers): (usize, BodyLevers),
    (prompt, n): (&[u32], usize),
    mut seen: impl FnMut(&Round, &[f32], Duration),
) -> Result<(Vec<u32>, Deepseek41Model, Duration), GateError> {
    let hp = Hparams::read(&from.file).map_err(|e| e.to_string())?;
    let (_, head) = from
        .file
        .find(HEAD_TENSOR)
        .ok_or("the folder holds no head")?;
    let vocab_at = rank_span(
        &head.name,
        head.ty,
        &head.dims,
        &TpUnits::of(&hp),
        rank,
        ranks,
    )?
    .map_or(0, |(_, at)| at.start);
    let sums = session.sums();
    let t = Instant::now();
    let mut model = body::open_rank(gpu, from, (rank, ranks), sums, ctx, levers)
        .map_err(|e| session.fail(&format!("its load: {e}")))?;
    let load = t.elapsed();
    let run = greedy(
        session,
        rank,
        u32::try_from(vocab_at)?,
        (prompt, n),
        |round| {
            let t = Instant::now();
            let mut pick = 0;
            for &id in &round.tokens {
                pick = model.step(&[id])?;
            }
            let logits = model.logits()?;
            seen(round, &logits, t.elapsed());
            Ok((pick, logits))
        },
    );
    match run {
        Ok(tokens) => Ok((tokens, model, load)),
        Err(e) => {
            // A release that did not settle leaves a card waiting on the page: dropping the
            // model would synchronize that wait for good.
            if matches!(e, RankError::Unsettled { .. }) {
                std::mem::forget(model);
            }
            Err(e.into())
        }
    }
}
