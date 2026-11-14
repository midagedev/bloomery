//! The server's engine over a draft (`serve::TargetEngine::drafted`, as
//! `bloomery-serve-metal --draft mtp|lookup` opens it) serves greedy requests
//! with the plain run's ids while the draft proposes and the target keeps what
//! it proposed. A prompt call that carries a mark keeps that position for a
//! cut, as the plain target's does, and the request after it drafts too.
//!
//! - `v_mtp_serve`: over the MTP draft behind the fixed-width chooser, for
//!   the prompt called in passes and by steps (the hidden row a checkpoint
//!   keeps comes from a pass's rows or from the step's): a fresh request, one
//!   that continues the cache from the mark (a cut to a checkpoint, which
//!   rejoins the draft with the checkpoint's hidden row, so the draft still
//!   proposes) and one that starts over after a reset
//!   all pick the plain run's ids. SKIPs as `v_mtp` does.
//! - `v_mtp_point`: the hidden row a checkpoint keeps beside it is the plain
//!   steps' row of the position before it: bit for bit when the prompt call ran
//!   by steps, nearer to it than to either neighbouring position's when it ran in
//!   passes, and a cut to the point writes it to the step arena for the draft's
//!   next walk. SKIPs as `v_mtp` does.
//! - `v_lookup_serve`: the same over the lookup draft, which needs no sidecar:
//!   a fresh request, a cut and a reset draft the same proposals and keeps (the
//!   lookup's context follows the target by position, which the gate reads
//!   off the window after each request, and the fixed width verifies every
//!   proposal, so the counts of one request are the other's), and
//!   a request whose first ids read the logits row (as a request that samples
//!   does) keeps drafting.

use std::path::Path;
use std::sync::Arc;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::session::{Session, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::HiddenRow;
use bloomery_metal::library::Library;
use model_hf::ModelFile;
use runtime::prompt::{CallEnd, PrefillMode};
use runtime::width::{Chosen, Mode as WidthMode, Wall};
use runtime::{Choosing, Lookup, Speculative};
use serve::vocab::Vocab;
use serve::{DraftWindow, Engine, EngineProps, TargetEngine};

use super::mtp::{CTX, GEN, ROWS, plain};
use super::{row, says, sets};

/// The mark the clauses' prompt calls carry, inside the prompt.
const MARK: usize = 24;

/// How many of a sampled request's first ids the clause reads off the logits
/// row, taking the argmax as the draw: the step the server runs for a request
/// that samples.
const DRAWN: usize = 3;

/// What a draft did in one request: the ids it proposed and the target kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    proposed: usize,
    accepted: usize,
}

/// One greedy request through `e`: the prompt's tail from the cache's end, the
/// ids picked until [`GEN`] of them are in hand, and the draft's counts. The
/// first `drawn` ids are steps that read the logits row.
fn serve_request<W: DraftWindow<Session> + Send>(
    e: &mut TargetEngine<Session, W>,
    ids: &[u32],
    marks: &[usize],
    drawn: usize,
    what: &str,
) -> Result<(Vec<u32>, Counts), String> {
    let mut logits = vec![0.0f32; e.tokenizer().n_vocab()];
    let first = e
        .prefill_next(ids, &[], marks, (drawn > 0).then_some(&mut logits[..]))
        .map_err(|err| format!("{what}: {err}"))?;
    let mut got = vec![first];
    let mut counts = Counts {
        proposed: 0,
        accepted: 0,
    };
    while got.len() < GEN {
        let last = *got.last().expect("the first id is in hand");
        if got.len() < drawn {
            let next = e
                .next(last, Some(&mut logits))
                .map_err(|err| format!("{what}: {err}"))?;
            got.push(next);
            continue;
        }
        let mut out = Vec::new();
        let d = e
            .advance(last, &mut out)
            .map_err(|err| format!("{what}: {err}"))?;
        if out.is_empty() {
            return Err(format!("{what}: a pass kept no id"));
        }
        counts.proposed += d.proposed;
        counts.accepted += d.accepted;
        got.extend(out);
    }
    got.truncate(GEN);
    Ok((got, counts))
}

/// `s` served through `window`, with the vocabulary of the file at `path`.
fn engine<W: DraftWindow<Session> + Send>(
    s: Session,
    window: W,
    path: &Path,
) -> Result<TargetEngine<Session, W>, String> {
    let head_rows = s.vocab();
    let file = ModelFile::open(path).map_err(|e| e.to_string())?;
    let vocab = Arc::new(Vocab::new(file.tokenizer().map_err(|e| e.to_string())?)?);
    TargetEngine::drafted(
        s,
        window,
        head_rows,
        vocab,
        "the e2e session".into(),
        EngineProps::default(),
    )
}

/// The request's ids are the plain run's.
fn same(ids: &[u32], plain_ids: &[u32], what: &str) -> Result<(), String> {
    if ids == plain_ids {
        return Ok(());
    }
    let at = ids.iter().zip(plain_ids).position(|(a, b)| a != b);
    Err(format!(
        "{what} through the drafted engine picked {ids:?}; the plain run picked {plain_ids:?} \
         (they part at {at:?})"
    ))
}

/// The lookup's context after a request is the target's token history: the
/// prompt, then the plain run's ids, one token a position through the
/// target's, which a cut back or a reset does not leave a tail of.
fn context_follows(
    e: &TargetEngine<Session, Speculative<Choosing<Lookup>, 2>>,
    prompt: &[u32],
    plain_ids: &[u32],
    what: &str,
) -> Result<(), String> {
    let ctx = e.window().draft().draft().context();
    let pos = e.target().pos() as usize;
    if ctx.len() != pos + 1 {
        return Err(format!(
            "{what}: the lookup's context holds {} tokens, the target stands at position {pos}: \
             a context holds one token a position and the token at the target's",
            ctx.len()
        ));
    }
    let k = (ctx.len() - prompt.len()).min(plain_ids.len());
    if ctx[..prompt.len()] != *prompt || ctx[prompt.len()..][..k] != plain_ids[..k] {
        return Err(format!(
            "{what}: the lookup's context is not the prompt and the plain run's ids"
        ));
    }
    Ok(())
}

/// The prompt's mark is a position the cache keeps, as a plain prompt call's
/// is.
fn mark_kept<W: DraftWindow<Session> + Send>(e: &TargetEngine<Session, W>) -> Result<(), String> {
    let kept = e.keepable(MARK);
    if kept == MARK {
        return Ok(());
    }
    Err(format!(
        "the prompt call's mark at {MARK}: the drafted engine keeps {kept} of those positions"
    ))
}

/// `v_mtp_serve`: the server's engine over the MTP draft picks the plain run's
/// ids, and a request cut to a checkpoint keeps drafting.
pub fn v_mtp_serve(
    device: &MetalDevice,
    library: &Library,
    path: &Path,
    sidecar: Option<&Path>,
) -> Clause {
    let name = row::current().name;
    if !ROWS.contains(&name) {
        return Ok(Verdict::Skip(format!(
            "row {name}: the sidecar is made for the rows {ROWS:?}: its hidden size, its vocabulary \
             and the layer it is shaped after are theirs"
        )));
    }
    let Some(sidecar) = sidecar else {
        return Ok(Verdict::Skip(
            "no sidecar: name the MTP sidecar's folder with --mtp SIDECAR".into(),
        ));
    };
    let prompt = sets::prompt_ids(48)?;
    let mut reference = Session::open(device, library, path, CTX).map_err(says)?;
    if let Err(e) = reference.verify_serves() {
        return Ok(Verdict::Skip(format!(
            "the verify pass does not serve this file: {e}"
        )));
    }
    // A hidden row of a checkpoint is a pass's row or the step's: both calls.
    let modes = [
        (PrefillMode::Batch, "passes"),
        (PrefillMode::Steps, "steps"),
    ];
    let mut plains = Vec::new();
    for (mode, _) in modes {
        plains.push(plain(&mut reference, &prompt, mode)?);
    }
    drop(reference);

    let mut lines = Vec::new();
    for ((mode, how), plain_ids) in modes.into_iter().zip(&plains) {
        let mut s = Session::open(device, library, path, CTX).map_err(says)?;
        s.open_mtp(sidecar).map_err(says)?;
        s.set_mtp_reading(HiddenRow::Post).map_err(says)?;
        s.set_prefill(mode);
        s.set_argmax_mode(StepMode::HostFed).map_err(says)?;
        let window: Speculative<_, 2> = Speculative::new(
            s.mtp_draft()
                .map_err(says)?
                .choosing(WidthMode::Fixed)
                .map_err(|e| e.to_string())?,
        );
        let mut e = engine(s, window, path)?;

        // A fresh request, its prompt call marked the way a chat turn's is.
        let what = format!("a fresh request, prefill by {how}");
        let (ids, fresh) = serve_request(&mut e, &prompt, &[MARK], 0, &what)?;
        same(&ids, plain_ids, &what)?;
        if fresh.proposed == 0 || fresh.accepted > fresh.proposed {
            return Err(format!(
                "{what}: the draft proposed {} ids and the target kept {}; it proposes, and keeps \
                 no more than it proposed",
                fresh.proposed, fresh.accepted
            ));
        }
        mark_kept(&e)?;
        // A request that continues from the mark: the same prompt again. The
        // draft rejoins the cut with the checkpoint's hidden row, so it
        // proposes, and what it proposes the target keeps in part.
        e.cut(MARK)
            .map_err(|err| format!("a cut to {MARK}: {err}"))?;
        let what = format!("a request after a cut to {MARK}, prefill by {how}");
        let (ids, cut) = serve_request(&mut e, &prompt[MARK..], &[], 0, &what)?;
        same(&ids, plain_ids, &what)?;
        if cut.proposed == 0 {
            return Err(format!(
                "{what}: the draft proposed nothing: a cut to a checkpoint keeps it drafting, \
                 rejoined with the checkpoint's hidden row"
            ));
        }
        if cut.accepted > cut.proposed {
            return Err(format!(
                "{what}: the draft proposed {} ids and the target kept {}; it keeps no more than \
                 it proposed",
                cut.proposed, cut.accepted
            ));
        }
        // A request that starts over: the draft joins it from its first position.
        e.reset().map_err(|err| format!("a reset: {err}"))?;
        let what = format!("a request after a reset, prefill by {how}");
        let (ids, again) = serve_request(&mut e, &prompt, &[MARK], 0, &what)?;
        same(&ids, plain_ids, &what)?;
        if again.proposed == 0 {
            return Err(format!("{what}: the draft proposed nothing"));
        }
        lines.push(format!(
            "prefill by {how}: fresh {}/{}, after the cut {}/{}, after the reset {}/{} \
             (kept/proposed)",
            fresh.accepted,
            fresh.proposed,
            cut.accepted,
            cut.proposed,
            again.accepted,
            again.proposed
        ));
    }
    Ok(Verdict::Noted(format!(
        "the drafted engine behind the fixed-width chooser picks the plain run's {GEN} ids of a \
         {}-id prompt as a fresh request, keeps the position of the prompt call's mark at {MARK}, \
         picks them again from a cut to it with the draft rejoined there (it proposes), and after a \
         reset picks them with the draft proposing: {}",
        prompt.len(),
        lines.join("; ")
    )))
}

/// The distance of `a` from `b` over `b`'s norm.
fn distance(a: &[f32], b: &[f32]) -> f64 {
    let gap: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let norm: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (gap / norm).sqrt()
}

/// `v_mtp_point`: a checkpoint's hidden row is the target's row of the
/// position before it, and a cut to the point leaves it in the step arena.
pub fn v_mtp_point(
    device: &MetalDevice,
    library: &Library,
    path: &Path,
    sidecar: Option<&Path>,
) -> Clause {
    let name = row::current().name;
    if !ROWS.contains(&name) {
        return Ok(Verdict::Skip(format!(
            "row {name}: the sidecar is made for the rows {ROWS:?}: its hidden size, its vocabulary \
             and the layer it is shaped after are theirs"
        )));
    }
    let Some(sidecar) = sidecar else {
        return Ok(Verdict::Skip(
            "no sidecar: name the MTP sidecar's folder with --mtp SIDECAR".into(),
        ));
    };
    // The server's prompt call: all the prompt's ids but the last, marked inside it.
    let prompt = sets::prompt_ids(48)?;
    let call = &prompt[..prompt.len() - 1];
    let end = u32::try_from(call.len()).map_err(|_| "a call past u32".to_string())?;
    let mark = u32::try_from(MARK).map_err(|_| "a mark past u32".to_string())?;
    let mut s = Session::open(device, library, path, CTX).map_err(says)?;
    if let Err(e) = s.verify_serves() {
        return Ok(Verdict::Skip(format!(
            "the verify pass does not serve this file: {e}"
        )));
    }
    s.open_mtp(sidecar).map_err(says)?;
    s.set_mtp_reading(HiddenRow::Post).map_err(says)?;
    s.set_argmax_mode(StepMode::HostFed).map_err(says)?;

    // The plain steps' row at each position of the call.
    s.set_prefill(PrefillMode::Steps);
    s.reset().map_err(says)?;
    let mut steps = Vec::with_capacity(call.len());
    for &id in call {
        s.step(id, StepMode::HostFed).map_err(says)?;
        steps.push(s.hidden_row().map_err(says)?);
    }

    let mut lines = Vec::new();
    for (mode, how) in [
        (PrefillMode::Steps, "steps"),
        (PrefillMode::Batch, "passes"),
    ] {
        s.set_prefill(mode);
        s.reset().map_err(says)?;
        let mut draft = s.mtp_draft().map_err(says)?;
        draft
            .prompt_call_marked(&mut s, call, &[mark], CallEnd::Argmax)
            .map_err(says)?;
        let mut kept = Vec::new();
        for at in [mark, end] {
            let held = s.point_hidden_row(at).map(<[f32]>::to_vec).ok_or_else(|| {
                format!("prefill by {how}: the point at {at} holds no hidden row")
            })?;
            let q = (at - 1) as usize;
            let own = distance(&held, &steps[q]);
            if mode == PrefillMode::Steps {
                if held != steps[q] {
                    return Err(format!(
                        "prefill by {how}: the point at {at}'s hidden row is not the plain step's \
                         row of position {q} bit for bit (distance {own:e})"
                    ));
                }
                lines.push(format!(
                    "{how}: the point at {at} equals the step's row of {q}"
                ));
            } else {
                let before = distance(&held, &steps[q - 1]);
                let after = steps.get(q + 1).map(|r| distance(&held, r));
                if own >= before || after.is_some_and(|a| own >= a) {
                    return Err(format!(
                        "prefill by {how}: the point at {at}'s hidden row is {own:e} from the step's \
                         row of position {q}, {before:e} from the row before it and {after:?} from \
                         the one after: its own position's row is the nearest"
                    ));
                }
                lines.push(format!(
                    "{how}: the point at {at} is {own:.1e} from the step's row of {q} ({before:.1e} \
                     from the one before)"
                ));
            }
            kept.push((at, held));
        }
        // A cut to the mark: the draft rejoins there, and the step arena holds the row.
        if !s.draft_rejoins(mark) {
            return Err(format!(
                "prefill by {how}: a cut to the point at {mark} does not keep the draft"
            ));
        }
        if s.draft_rejoins(mark + 1) || s.draft_rejoins(0) {
            return Err(format!(
                "prefill by {how}: a cut to a position with no point ({} or 0) keeps the draft",
                mark + 1
            ));
        }
        s.cut(mark).map_err(says)?;
        if s.hidden_row().map_err(says)? != kept[0].1 {
            return Err(format!(
                "prefill by {how}: after a cut to {mark} the step arena's row is not the point's"
            ));
        }
        // The cut dropped the point past it, and its row with it.
        if s.point_hidden_row(end).is_some() {
            return Err(format!(
                "prefill by {how}: after a cut to {mark} the point at {end} still holds a row"
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "a checkpoint keeps the target's hidden row of the position before it, for the points at \
         {mark} and {end} of a {}-id call ({}); a cut to the mark writes it to the step arena, \
         the draft rejoins there, and the point past the cut loses its row",
        call.len(),
        lines.join("; ")
    )))
}

/// `v_lookup_serve`: the server's engine over the lookup draft picks the plain
/// run's ids, and a cut, a reset and a request that reads the logits row leave
/// it drafting what the fresh request drafted.
pub fn v_lookup_serve(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    // The prompt twice over: the lookup's n-grams find the second lap in the first.
    let seg = sets::prompt_ids(48)?;
    let prompt = [seg.clone(), seg].concat();
    let mut reference = Session::open(device, library, path, CTX).map_err(says)?;
    if let Err(e) = reference.verify_serves() {
        return Ok(Verdict::Skip(format!(
            "the verify pass does not serve this file: {e}"
        )));
    }
    let plain_ids = plain(&mut reference, &prompt, PrefillMode::Batch)?;
    drop(reference);

    let mut s = Session::open(device, library, path, CTX).map_err(says)?;
    s.set_prefill(PrefillMode::Batch);
    s.set_argmax_mode(StepMode::HostFed).map_err(says)?;
    let window: Speculative<_, 2> = Speculative::new(
        <Lookup as Chosen<Session>>::choosing_by(
            Lookup::new(),
            WidthMode::Fixed,
            Lookup::RULE,
            Wall::new(),
        )
        .map_err(|e| e.to_string())?,
    );
    let mut e = engine(s, window, path)?;

    let what = "a fresh request";
    let (ids, fresh) = serve_request(&mut e, &prompt, &[MARK], 0, what)?;
    same(&ids, &plain_ids, what)?;
    context_follows(&e, &prompt, &plain_ids, what)?;
    if fresh.proposed == 0 || fresh.accepted > fresh.proposed {
        return Err(format!(
            "{what}: the lookup proposed {} ids and the target kept {}; it proposes, and keeps no \
             more than it proposed",
            fresh.proposed, fresh.accepted
        ));
    }
    mark_kept(&e)?;

    // A cut to the mark: the lookup's context is made what the cut leaves, so
    // its request drafts what the fresh one did.
    e.cut(MARK)
        .map_err(|err| format!("a cut to {MARK}: {err}"))?;
    let what = "a request after a cut";
    let (ids, cut) = serve_request(&mut e, &prompt[MARK..], &[], 0, what)?;
    same(&ids, &plain_ids, what)?;
    context_follows(&e, &prompt, &plain_ids, what)?;
    if cut != fresh {
        return Err(format!(
            "{what}: the lookup proposed {} ids and the target kept {}; the fresh request's were \
             {} and {}: its context follows the target by position, so a cut back leaves the \
             proposals the fresh request made",
            cut.proposed, cut.accepted, fresh.proposed, fresh.accepted
        ));
    }

    // A request that reads the logits row for its first ids, as one that
    // samples does: the lookup stays on for the passes after them.
    e.cut(MARK)
        .map_err(|err| format!("a second cut to {MARK}: {err}"))?;
    let what = "a request whose first ids read the logits row";
    let (ids, sampled) = serve_request(&mut e, &prompt[MARK..], &[], DRAWN, what)?;
    same(&ids, &plain_ids, what)?;
    context_follows(&e, &prompt, &plain_ids, what)?;
    if sampled.proposed == 0 {
        return Err(format!(
            "{what}: the lookup proposed nothing after the {DRAWN} ids that read the logits row"
        ));
    }

    // A request that starts over.
    e.reset().map_err(|err| format!("a reset: {err}"))?;
    let what = "a request after a reset";
    let (ids, again) = serve_request(&mut e, &prompt, &[MARK], 0, what)?;
    same(&ids, &plain_ids, what)?;
    context_follows(&e, &prompt, &plain_ids, what)?;
    if again != fresh {
        return Err(format!(
            "{what}: the lookup proposed {} ids and the target kept {}; the fresh request's were \
             {} and {}: a reset starts its context over, so the request is the first one again",
            again.proposed, again.accepted, fresh.proposed, fresh.accepted
        ));
    }
    Ok(Verdict::Noted(format!(
        "the drafted engine over the lookup behind the fixed-width chooser picks the plain run's \
         {GEN} ids of a {}-id prompt (the segment twice) as a fresh request (the lookup proposed \
         {} ids, the target kept {}), keeps the position of the prompt call's mark at {MARK}, and \
         picks them again with the same proposals and keeps from a cut to it and after a reset; a \
         request whose first {DRAWN} ids read the logits row keeps drafting ({} proposed, {} kept)",
        prompt.len(),
        fresh.proposed,
        fresh.accepted,
        sampled.proposed,
        sampled.accepted
    )))
}
