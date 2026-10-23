//! `m_mlx`: the MLX-format fixture split opens through the session, and its
//! greedy picks are the session's own host-fed steps'.
//!
//! The split (`qwen35moe-mlxsplit` under the fixture root) holds every typed
//! site at MLX_A4 and its routers at MLX_A8: the reader admits the sites
//! through Metal's site lists (`session::SITES`), the arena uploads the A4
//! blocks and dequantizes the routers exactly at open, and the session steps.
//! The reference is the session's own host-fed steps over the same ids —
//! `t_target`'s greedy pattern, on the MLX split's session. No external set
//! pins the split yet; the e2e pin against mlx-lm's own logits is another
//! round's.

use std::path::PathBuf;
use std::time::Duration;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use runtime::{Advance, Committed, PassSink, Plain, Stop, StopReason, Target};

use super::{CTX, SEED, says};

/// The greedy tokens the loop and the host-fed reference run.
const GREEDY: usize = 24;

/// A [`PassSink`] that hears nothing.
struct Quiet;

impl PassSink<Session> for Quiet {
    type Error = SessionError;

    fn begin(&mut self, _t: &Session) -> Result<(), SessionError> {
        Ok(())
    }

    fn pass(
        &mut self,
        _t: &Session,
        _c: &Committed,
        _tokens: &[u32],
        _wall: Duration,
    ) -> Result<(), SessionError> {
        Ok(())
    }
}

/// The MLX-format fixture split this clause reads: the one
/// `*-00001-of-*.gguf` under the fixture root's `qwen35moe-mlxsplit`
/// directory, resolved beside the gate's own fixture (`BLOOMERY_FIXTURE_ROOT`,
/// the reference sets' root) — the kernels gate's `--mlx-fixture`
/// resolution without its flag.
fn mlx_fixture_path() -> Result<PathBuf, String> {
    let root = refset::fixture::root().map_err(|e| format!("the fixture root: {e}"))?;
    let at = format!("{root}/qwen35moe-mlxsplit");
    let entries = std::fs::read_dir(&at).map_err(|e| format!("the MLX fixture {at}: {e}"))?;
    let mut found: Vec<String> = entries
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.ends_with(".gguf") && n.contains("-00001-of-"))
        .collect();
    found.sort();
    match found.as_slice() {
        [one] => Ok(PathBuf::from(format!("{at}/{one}"))),
        _ => Err(format!(
            "the MLX fixture {at} holds {} first shards, not one",
            found.len()
        )),
    }
}

/// `m_mlx`: the MLX split's session opens and generate over `Plain` emits
/// the session's own host-fed picks — the open is the reader's admission of
/// every MLX_A4 site through Metal's lists, the picks the session's first
/// greedy steps.
pub fn m_mlx(device: &MetalDevice, library: &Library) -> Clause {
    let path = mlx_fixture_path()?;
    let mut s = Session::open(device, library, &path, CTX).map_err(says)?;

    // The reference: the first id and each pick host-fed, one step each.
    s.reset().map_err(says)?;
    let mut ref_picks = Vec::with_capacity(GREEDY);
    let mut id = SEED;
    for _ in 0..GREEDY {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        ref_picks.push(id);
    }

    // The loop over the target: the prompt again, then the passes.
    s.reset().map_err(says)?;
    let mut plain = Plain;
    let prompt = [SEED];
    let picked = plain
        .prompt(&mut s, &prompt)
        .map_err(|e| format!("the prompt's argmax: {}", says(e)))?;
    if picked != ref_picks[0] {
        return Err(format!(
            "the prompt picked {picked} where the session's own step picks {}",
            ref_picks[0]
        ));
    }
    let stop = Stop::new(GREEDY, Target::ctx(&s)).map_err(|e| format!("the stop rule: {e}"))?;
    let outcome =
        runtime::generate(&mut s, &mut plain, &prompt, picked, &stop, &mut Quiet).map_err(says)?;
    if outcome.tokens != ref_picks {
        let at = outcome
            .tokens
            .iter()
            .zip(&ref_picks)
            .position(|(a, b)| a != b)
            .unwrap_or(outcome.tokens.len().min(ref_picks.len()));
        return Err(format!(
            "the loop's token {at} is {:?} where the session's own steps pick {:?}",
            outcome.tokens.get(at),
            ref_picks.get(at)
        ));
    }
    if outcome.stop != StopReason::Length {
        return Err(format!(
            "the loop ended by {:?}; length after {GREEDY} is expected",
            outcome.stop
        ));
    }
    Ok(Verdict::Noted(format!(
        "the MLX split {} opened ({} layers, every typed site MLX_A4) and generate over Plain \
         emitted the session's own {GREEDY} picks",
        path.display(),
        s.layer_count()
    )))
}
