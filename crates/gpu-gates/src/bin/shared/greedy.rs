use bloomery_gpu_gates::GateError;
use runtime::{Out, Target, Want};

/// Greedy ids and optional logit rows from a reset target; n is positive.
pub(crate) fn greedy<T: Target>(
    s: &mut T,
    prompt: &[u32],
    n: usize,
    logits: bool,
) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
    let want = if logits { Want::Logits } else { Want::Argmax };
    s.reset()?;
    let (mut ids, mut rows) = (Vec::with_capacity(n), Vec::new());
    let push = |out: Out<'_>, ids: &mut Vec<u32>, rows: &mut Vec<Vec<f32>>| {
        ids.push(out.argmax());
        if let Out::Logits { row, .. } = out {
            rows.push(row.to_vec());
        }
    };
    push(s.prompt(prompt, want)?, &mut ids, &mut rows);
    while ids.len() < n {
        let last = ids[ids.len() - 1];
        push(s.step(last, want)?, &mut ids, &mut rows);
    }
    Ok((ids, rows))
}
