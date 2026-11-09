//! A tensor-parallel group's greedy run, the same for every model ([`greedy`]): rank 0 decides
//! each round (the prompt, then each step's token, then the stop) and every rank runs it on its
//! own rank of the model; each rank picks over its part of the vocabulary
//! (`bloomery_gpu::head::local_pick`) and rank 0 merges the ranks' picks into the next token
//! (`bloomery_gpu::head::merge`). The model enters as one step: a round in, the rank's logits over
//! its part of the vocabulary out.

use bloomery_gpu::GpuError;
use bloomery_gpu::head::{local_pick, merge};
use bloomery_gpu::host::rank::{RankError, RankSession, Round};

const WHAT: &str = "tp greedy";

/// Run `prompt` then `max` greedy tokens on rank `me` of `session`, whose part of the vocabulary
/// starts at id `vocab_at`; `step` runs a round's tokens at its position and gives the card's
/// argmax over the rank's part and the rank's logits over it. Rank 0 decides from `prompt` and returns the tokens; every other rank
/// passes an empty prompt and returns none. An empty prompt on rank 0, a step that fails, and a
/// pick no rank agrees on fail the group by name ([`RankSession::fail`]).
pub fn greedy(
    session: &mut RankSession,
    me: usize,
    vocab_at: u32,
    (prompt, max): (&[u32], usize),
    mut step: impl FnMut(&Round) -> Result<(u32, Vec<f32>), GpuError>,
) -> Result<Vec<u32>, RankError> {
    let Ok(prompt_len) = u32::try_from(prompt.len()) else {
        return Err(session.fail(&format!("{WHAT}: a prompt past u32 tokens")));
    };
    if me == 0 && prompt.is_empty() {
        return Err(session.fail(&format!("{WHAT}: rank 0 runs no empty prompt")));
    }
    // Every rank enters the first round loaded: a rank still loading would hold the others' first
    // sums past their step's deadline.
    session.collect(&[])?;
    let mut out = Vec::with_capacity(max);
    loop {
        let decided = (me == 0).then(|| {
            let tokens = out.last().map_or_else(|| prompt.to_vec(), |&t| vec![t]);
            Round {
                prompt: out.is_empty().then_some(0..prompt_len),
                tokens,
                stop: out.len() >= max,
            }
        });
        let round = session.round(decided.as_ref())?;
        if round.stop {
            return Ok(out);
        }
        let pick = step(&round).and_then(|(id, logits)| local_pick(&logits, id, vocab_at));
        let (value, id) = pick.map_err(|e| session.fail(&format!("{WHAT}: {e}")))?;
        let picks = session.collect(&[u64::from(value.to_bits()), u64::from(id)])?;
        if me == 0 {
            let token = picks
                .iter()
                .map(|p| match *p.as_slice() {
                    [v, id] => match (u32::try_from(v), u32::try_from(id)) {
                        (Ok(v), Ok(id)) => Ok((f32::from_bits(v), id)),
                        _ => Err(GpuError::shape(WHAT, format!("a pick of {p:?}"))),
                    },
                    _ => Err(GpuError::shape(WHAT, format!("a pick of {p:?}"))),
                })
                .collect::<Result<Vec<_>, _>>()
                .and_then(|p| merge(&p))
                .map_err(|e| session.fail(&e.to_string()))?;
            out.push(token);
        }
    }
}
