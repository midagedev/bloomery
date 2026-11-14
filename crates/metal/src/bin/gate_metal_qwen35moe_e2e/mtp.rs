//! The MTP draft clause: the session as the host of the shared MTP window
//! (`runtime::mtp::MtpDraft` over `Session`'s `Host`), drafting the plain
//! greedy run's ids.
//!
//! - `v_mtp`: over the row's model with the MTP sidecar `--mtp SIDECAR` names,
//!   for each of three prompts (one id, a tail a prompt call steps, a prompt
//!   past the pass's window), the plain greedy run's 64 ids are the ids of the
//!   loop `Speculative<Choosing<MtpDraft<Session>>, 2>` picks, under the fixed
//!   width (every proposal verified), with the draft's proposal scripted per
//!   window: its own id, the plain run's right next id in its place (a window
//!   kept whole, which walks a refresh of two rows), or none (a plain step
//!   whose row the next window's chain reads). The drafted ids are the
//!   plain run's whatever the draft proposes, so the clause holds the draft
//!   itself to a reference the host's walks are not part of:
//!   - the reference is the layer's own single-row draft step
//!     (`Session::draft_step`), run over the plain run's every position with
//!     the hidden row of the step before it (`tap_mtp_hidden`); the loop's
//!     proposal at each position, and the bits of the logits row it came
//!     from, equal the reference's at that position, so the draft's store, the
//!     hidden row each walk joined (the step's `x`, the verify's rows, a
//!     prompt unit's) and the chain's pick are the layer's own. This holds on
//!     the prefill by steps, where every hidden row is the step's bits; a
//!     prompt in passes leaves rows within the pass's band of the step's, so
//!     that run holds the ids and the host's recount alone;
//!   - the host recounts each verified window from the plain run's ids: its
//!     position, and the rows it kept, 2 when the proposal is the plain run's
//!     next id and 1 when it is not, equal the loop's own account of the pass;
//!     the passes' kept rows sum to the ids the run picked after its first;
//!   - every kind of pass is counted and each is expected: a window kept whole,
//!     a window that kept one row, and a plain step.
//!
//!   With no sidecar named, on a row whose layer the sidecar is not made for,
//!   or on a file the verify pass does not serve, the clause prints SKIP with
//!   the reason.
//!
//! The server's engine over each draft is `serve.rs`'s.

use std::path::Path;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::generate::{Generated, greedy, greedy_with};
use bloomery_metal::arch::qwen3moe::mtp_check::fnv1a;
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::HiddenRow;
use bloomery_metal::library::Library;
use runtime::mtp::MtpDraft;
use runtime::prompt::PrefillMode;
use runtime::width::{Chosen, Mode as WidthMode};
use runtime::{Draft, Speculative, TapNeed};

use super::{row, says, sets};

/// The ids a run picks: the count the clause holds equal to the plain run's.
pub(super) const GEN: usize = 64;

/// The cache rows: the longest prompt, the ids a run picks and the rows a
/// verify runs past them, as `v_lookup`'s session holds.
pub(super) const CTX: usize = 256;

/// The rows whose model the sidecar is made for: its hidden size, vocabulary
/// and the layer it is shaped after are Qwen3.8-27B's.
pub(super) const ROWS: &[&str] = &["qwen38d"];

/// What a window of the script does with the draft's own proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    /// The draft's own id is verified.
    Own,
    /// The plain run's next id is verified in its place: the window is kept
    /// whole.
    Right,
    /// No proposal: the pass is one plain step.
    Hold,
}

/// The windows' script, in turn.
const SCRIPT: &[Act] = &[
    Act::Right,
    Act::Own,
    Act::Hold,
    Act::Right,
    Act::Right,
    Act::Own,
];

/// A window the draft proposed in.
#[derive(Clone, Copy, Debug)]
struct Seen {
    /// The position of the token the window verifies behind.
    pos: u32,
    /// The draft's own id, and the bits of the logits row it is the argmax of.
    own: u32,
    logits: u64,
    /// The id verified: none when the window was held back.
    applied: Option<u32>,
}

/// The MTP draft as the loop drives it, its proposals recorded and each
/// replaced or held back as [`SCRIPT`] says.
struct Scripted {
    inner: MtpDraft<Session>,
    /// The plain run's ids, the first at position `start`.
    plain: Vec<u32>,
    start: u32,
    windows: usize,
    seen: Vec<Seen>,
}

/// FNV-1a 64 over the f32 little-endian bytes of `row`.
fn row_hash(row: &[f32]) -> u64 {
    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
    fnv1a(&bytes)
}

impl Draft<Session> for Scripted {
    const WIDTH: usize = <MtpDraft<Session> as Draft<Session>>::WIDTH;
    const TAPS: TapNeed = <MtpDraft<Session> as Draft<Session>>::TAPS;
    const CHAIN: bool = <MtpDraft<Session> as Draft<Session>>::CHAIN;

    fn prompt(&mut self, t: &mut Session, ids: &[u32]) -> Result<u32, SessionError> {
        self.inner.prompt(t, ids)
    }

    fn begin(&mut self, t: &Session, prompt: &[u32], first: u32) -> Result<(), SessionError> {
        self.inner.begin(t, prompt, first)
    }

    fn propose(
        &mut self,
        t: &mut Session,
        last: u32,
        out: &mut [u32],
    ) -> Result<usize, SessionError> {
        let n = self.inner.propose(t, last, out)?;
        if n == 0 {
            return Ok(0);
        }
        let pos = t.pos();
        let own = out[0];
        let logits = row_hash(t.logits_row()?);
        let act = SCRIPT[self.windows % SCRIPT.len()];
        self.windows += 1;
        let right = (pos as usize + 1)
            .checked_sub(self.start as usize)
            .and_then(|k| self.plain.get(k))
            .copied();
        let applied = match (act, right) {
            (Act::Hold, _) => None,
            (Act::Right, Some(id)) => {
                out[0] = id;
                Some(id)
            }
            _ => Some(own),
        };
        self.seen.push(Seen {
            pos,
            own,
            logits,
            applied,
        });
        Ok(usize::from(applied.is_some()))
    }

    fn unproposed(&mut self) {
        self.inner.unproposed();
    }

    fn accept(
        &mut self,
        t: &mut Session,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        self.inner.accept(t, rows, out, accepted)
    }

    fn stepped(&mut self, t: &mut Session, last: u32, next: u32) -> Result<(), SessionError> {
        self.inner.stepped(t, last, next)
    }

    fn before_plain(&mut self, t: &mut Session, last: u32) -> Result<(), SessionError> {
        self.inner.before_plain(t, last)
    }
}

/// The plain greedy run of `prompt` under `prefill`: its ids.
pub(super) fn plain(
    s: &mut Session,
    prompt: &[u32],
    prefill: PrefillMode,
) -> Result<Vec<u32>, String> {
    s.set_prefill(prefill);
    s.reset().map_err(says)?;
    let out = greedy(s, prompt, GEN, &[], |_| {}).map_err(|e| e.to_string())?;
    if out.ids.len() != GEN {
        return Err(format!(
            "the plain run of {} prompt ids picked {} ids, not {GEN}",
            prompt.len(),
            out.ids.len()
        ));
    }
    Ok(out.ids)
}

/// The layer's own draft step at each position of `seq`, the hidden row of the
/// target's step before it injected: the draft's pick there and the bits of
/// its logits row. The target steps `seq` host-fed, so each row is the step's.
fn reference(s: &mut Session, seq: &[u32]) -> Result<Vec<(u32, u64)>, String> {
    s.reset().map_err(says)?;
    let zeros = vec![0.0f32; s.geom().hidden];
    let mut picks = Vec::with_capacity(seq.len());
    for (q, &id) in seq.iter().enumerate() {
        let pos = u32::try_from(q).map_err(|_| "a reference position past u32".to_string())?;
        if q == 0 {
            s.set_mtp_hidden(&zeros).map_err(says)?;
        } else {
            s.tap_mtp_hidden().map_err(says)?;
        }
        let drafted = s
            .draft_step(id, pos, HiddenRow::Post)
            .map_err(|e| format!("the reference draft step at position {q}: {e}"))?;
        picks.push((drafted.id, row_hash(s.logits_row().map_err(says)?)));
        s.step(id, StepMode::HostFed).map_err(says)?;
    }
    Ok(picks)
}

/// What a drafted run did.
struct Run {
    out: Generated,
    seen: Vec<Seen>,
}

/// The loop over the scripted MTP draft behind the fixed-width chooser,
/// every pass host-fed.
fn drafted(s: &mut Session, prompt: &[u32], plain: &[u32]) -> Result<Run, String> {
    s.reset().map_err(says)?;
    let draft = Scripted {
        inner: s.mtp_draft().map_err(says)?,
        plain: plain.to_vec(),
        start: u32::try_from(prompt.len()).map_err(|_| "a prompt past u32".to_string())?,
        windows: 0,
        seen: Vec::new(),
    };
    let draft = <Scripted as Chosen<Session>>::choosing(draft, WidthMode::Fixed)
        .map_err(|e| e.to_string())?;
    let mut spec: Speculative<_, 2> = Speculative::new(draft);
    let chained = s.argmax_mode();
    s.set_argmax_mode(StepMode::HostFed).map_err(says)?;
    let out = greedy_with(s, &mut spec, prompt, GEN, &[], |_| {});
    s.set_argmax_mode(chained).map_err(says)?;
    let out = out.map_err(|e| e.to_string())?;
    let seen = std::mem::take(&mut spec.draft_mut().draft_mut().seen);
    Ok(Run { out, seen })
}

/// How many passes of each kind a run ran: a window kept whole, a window that
/// kept one row, a plain step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Kinds {
    whole: usize,
    one: usize,
    step: usize,
}

impl Kinds {
    fn add(&mut self, other: Kinds) {
        self.whole += other.whole;
        self.one += other.one;
        self.step += other.step;
    }
}

/// The run held to the plain run's ids and to the host's recount of its
/// windows, and to the reference's proposals where one is given; the kinds of
/// pass it ran.
fn check(
    what: &str,
    prompt: usize,
    plain: &[u32],
    run: &Run,
    reference: Option<&[(u32, u64)]>,
) -> Result<Kinds, String> {
    let ids = &run.out.ids;
    if ids != plain {
        let at = ids.iter().zip(plain).position(|(a, b)| a != b);
        return Err(format!(
            "{what}: the MTP-drafted ids part from the plain run's at {at:?} ({} ids against {})",
            ids.len(),
            plain.len()
        ));
    }
    // The passes: positions chain from the prompt's end, and each pass that
    // verified a proposal is the next window the draft proposed in, kept as
    // the host's recount of its proposal says.
    let mut windows = run.seen.iter().filter(|w| w.applied.is_some());
    let mut at = u32::try_from(prompt).map_err(|_| "a prompt past u32".to_string())?;
    let (mut kinds, mut kept_sum) = (Kinds::default(), 0usize);
    for (i, c) in run.out.commits.iter().enumerate() {
        if c.pos != at {
            return Err(format!(
                "{what}: pass {i} ran at position {}; the passes before it end at {at}",
                c.pos
            ));
        }
        if c.proposed {
            let w = windows.next().ok_or_else(|| {
                format!("{what}: pass {i} verified a proposal the draft made no window for")
            })?;
            if w.pos != c.pos {
                return Err(format!(
                    "{what}: pass {i} verified at position {}; the window the draft proposed in \
                     is at {}",
                    c.pos, w.pos
                ));
            }
            let next = (w.pos as usize + 1)
                .checked_sub(prompt)
                .and_then(|k| plain.get(k));
            let want = 1 + usize::from(next.is_some() && next == w.applied.as_ref());
            if c.rows != 2 || c.kept != want {
                return Err(format!(
                    "{what}: pass {i} at position {} kept {} of {} rows; the host's recount of \
                     its window (id {:?} against the plain run's {next:?}) keeps {want} of 2",
                    c.pos, c.kept, c.rows, w.applied
                ));
            }
            if want == 2 {
                kinds.whole += 1;
            } else {
                kinds.one += 1;
            }
        } else {
            if c.rows != 1 || c.kept != 1 {
                return Err(format!(
                    "{what}: the plain step at position {} ran {} rows and kept {}",
                    c.pos, c.rows, c.kept
                ));
            }
            kinds.step += 1;
        }
        kept_sum += c.kept;
        at += u32::try_from(c.kept).map_err(|_| "a pass kept past u32 rows".to_string())?;
    }
    if windows.next().is_some() {
        return Err(format!(
            "{what}: the draft proposed in windows the loop never verified"
        ));
    }
    // The passes' rows are the ids past the first, one more at the budget's edge.
    if kept_sum + 1 < GEN || kept_sum > GEN {
        return Err(format!(
            "{what}: the passes kept {kept_sum} rows in all; the {GEN} ids are the prompt call's \
             one and the rows the passes kept, one more at most"
        ));
    }
    if let Some(reference) = reference {
        for w in &run.seen {
            let (id, logits) = *reference.get(w.pos as usize).ok_or_else(|| {
                format!(
                    "{what}: the draft proposed at position {}, past the reference's {}",
                    w.pos,
                    reference.len()
                )
            })?;
            if (w.own, w.logits) != (id, logits) {
                return Err(format!(
                    "{what}: the draft's proposal at position {} (id {}, logits {:016x}) is not \
                     the reference draft step's (id {id}, logits {logits:016x})",
                    w.pos, w.own, w.logits
                ));
            }
        }
    }
    Ok(kinds)
}

/// The prompts: one id, a tail a prompt call steps, a prompt past the pass's
/// window.
fn prompts() -> Result<Vec<(&'static str, Vec<u32>)>, String> {
    let seg = sets::prompt_ids(48)?;
    Ok(vec![
        ("one", seg[..1].to_vec()),
        ("tail", seg[..5].to_vec()),
        ("seg", seg),
    ])
}

/// `v_mtp`: the MTP-drafted ids are the plain run's.
pub fn v_mtp(
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
    let mut s = Session::open(device, library, path, CTX).map_err(says)?;
    if let Err(e) = s.verify_serves() {
        return Ok(Verdict::Skip(format!(
            "the verify pass does not serve this file: {e}"
        )));
    }
    s.open_mtp(sidecar).map_err(says)?;
    s.set_mtp_reading(HiddenRow::Post).map_err(says)?;
    let mut lines = Vec::new();
    let mut total = Kinds::default();
    for (label, p) in prompts()? {
        // By steps: every hidden row is the step's bits, so the proposals are
        // held to the reference's.
        let by_steps = plain(&mut s, &p, PrefillMode::Steps)?;
        let seq = [p.clone(), by_steps.clone()].concat();
        let refs = reference(&mut s, &seq)?;
        s.set_prefill(PrefillMode::Steps);
        let run = drafted(&mut s, &p, &by_steps)?;
        let a = check(
            &format!("{label}, prefill by steps"),
            p.len(),
            &by_steps,
            &run,
            Some(&refs),
        )?;
        // In passes, where the prompt's rows are a pass's: the plain run's
        // ids and the host's recount.
        let by_pass = plain(&mut s, &p, PrefillMode::Batch)?;
        let run = drafted(&mut s, &p, &by_pass)?;
        let b = check(
            &format!("{label}, prefill in passes"),
            p.len(),
            &by_pass,
            &run,
            None,
        )?;
        for (how, k) in [("steps", a), ("passes", b)] {
            lines.push(format!(
                "{label} ({} ids), prefill by {how}: {} windows kept whole, {} kept one row, {} \
                 plain steps",
                p.len(),
                k.whole,
                k.one,
                k.step
            ));
            total.add(k);
        }
    }
    // Each kind of pass is expected: the script holds every kind on any file,
    // so a run that met none of one has a hole where a defect hides.
    if total.whole == 0 || total.one == 0 || total.step == 0 {
        return Err(format!(
            "the runs' passes: {} windows kept whole, {} kept one row, {} plain steps; each kind \
             is expected ({})",
            total.whole,
            total.one,
            total.step,
            lines.join("; ")
        ));
    }
    Ok(Verdict::Noted(format!(
        "{GEN} ids of each prompt equal the plain greedy run's through the MTP draft behind the \
         fixed-width chooser, its proposals scripted per window (its own id, the plain run's next \
         in its place, none), by steps and in passes; each proposal the loop made by steps is the \
         layer's own draft step at its position, logits bit for bit, and each window's kept rows \
         are the host's recount; {}",
        lines.join("; ")
    )))
}
