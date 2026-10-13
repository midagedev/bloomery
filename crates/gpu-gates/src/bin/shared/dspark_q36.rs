//! The Qwen3.6 DSpark draft beside its verifying target: what it adds to the target's whole-card
//! load ([`Extra`]) and the open of the draft on the target's card ([`open`]). The two binaries that
//! decode a qwen35moe file under `BLOOMERY_DRAFT=dspark` (`generate_qwen3moe` and the qwen3 seat of
//! `bloomery-serve`) include this file, so both price and open the draft one way. The file's reader is
//! `dspark_hf`, which the includer declares beside this module.
//!
//! The target is `GpuModel::<Body35>::open_verifying`: its delta layers hold [`Body35::VERIFY_LANES`]
//! lanes of state, which the plan's layout prices through `PlanInputs::with_lanes` and which the
//! whole-fit verdict of the seat's and the CLI's planner does not count (it describes the file with one
//! lane). The tap buffer is allocated at the first tapped call (`GpuModel::set_draft_taps`), and the
//! draft's own bytes are `dspark_hf::card_bytes`. The three are the draft's [`Extra`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use app::Session;
use app::dspark::CardDspark;
use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::Body35;
use bloomery_gpu::dspark::DsparkBody;
use bloomery_gpu::dspark::dense::DenseLayers;
use bloomery_gpu::model::MAX_PASS_ROWS;
use bloomery_gpu_gates::GateError;
use gguf::Split;
use model::arch::qwen35moe::place::PlanInputs;
use model::placement::KvBytes;
use runtime::Draft as RtDraft;

use super::dspark_hf;

/// The word the draft's directory is named by.
const MODEL_ENV: &str = "BLOOMERY_DSPARK_MODEL";

/// The draft run on a verify of every width up to this many rows: its proposal's ids and the id it
/// follows.
pub const ROWS: usize = <CardDspark<DenseLayers> as RtDraft<Session<Body35>>>::WIDTH + 1;

/// The draft on a Qwen3.6 session, as the seat and the CLI hold it.
pub type Draft36 = CardDspark<DenseLayers>;

/// The draft's directory: `$BLOOMERY_DSPARK_MODEL`, refused by name when it is unset.
pub fn dir() -> Result<PathBuf, GateError> {
    match std::env::var_os(MODEL_ENV) {
        Some(d) if !d.is_empty() => Ok(PathBuf::from(d)),
        _ => Err(format!(
            "BLOOMERY_DRAFT=dspark: {MODEL_ENV} names the draft's directory (config.json and \
             model.safetensors) and is unset"
        )
        .into()),
    }
}

/// What a DSpark draft adds to the whole-card load of a qwen35moe file, in card bytes, term by term.
#[derive(Clone, Copy, Debug)]
pub struct Extra {
    /// The draft's own load at the context ([`dspark_hf::card_bytes`]).
    pub draft: u64,
    /// The delta layers' extra lanes and their stamps: the verifying load's layout less the plain one.
    pub lanes: u64,
    /// The tap buffer: a ubatch's rows of the five tapped layers' f32 rows.
    pub taps: u64,
}

impl Extra {
    /// The sum the plan carries as a reserve of the stage card.
    pub fn total(&self) -> u64 {
        self.draft + self.lanes + self.taps
    }

    /// The terms in one line.
    pub fn words(&self) -> String {
        format!(
            "{} B = the {} {} B + the verifying lanes {} B + the tap buffer {} B",
            self.total(),
            dspark_hf::RESERVE,
            self.draft,
            self.lanes,
            self.taps
        )
    }
}

/// The draft of `dir` priced against the load of `file` at `ctx` positions with prompt ubatches of
/// `ubatch` rows: the tap buffer holds a tap's rows for each layer the draft file names, at the
/// target's hidden width.
pub fn extra(file: &Split, dir: &Path, ctx: usize, ubatch: usize) -> Result<Extra, GateError> {
    let draft = dspark_hf::card_bytes(dir, ctx)?;
    let plain = PlanInputs::describe(file)?;
    let laned = PlanInputs::describe(file)?.with_lanes(Body35::VERIFY_LANES);
    let at = |i: &PlanInputs| -> u64 { (0..i.hp.n_trunk).map(|l| i.kv.layer_bytes(l, 0)).sum() };
    let lanes = at(&laned) - at(&plain);
    let file = dspark_hf::HfFile::open(dir)?;
    let layers = dspark_hf::shape(dir, &file)?.taps.len();
    let rows = ubatch.max(MAX_PASS_ROWS);
    let taps = u64::try_from(rows * layers * plain.hp.n_embd * size_of::<f32>())?;
    Ok(Extra { draft, lanes, taps })
}

/// The draft of `dir` opened on the card `card` the target `s` runs on (the two share its primary
/// context), for contexts of up to `ctx` positions. `resident` is the card's free bytes before and
/// after the load. Refused by name when the draft's proposal does not fit the target's verify.
pub fn open(
    s: &Session<Body35>,
    dir: &Path,
    card: &'static str,
    ctx: usize,
) -> Result<Draft36, GateError> {
    let target = s.model().gpu();
    let gpu = Gpu::open_card(card, Some(target.device_id()?))?;
    let (before, _) = gpu.mem_info()?;
    let dspark_hf::Parts {
        shape,
        layers,
        fc,
        head,
        embed,
    } = dspark_hf::open(&gpu, dir, ctx)?;
    let body = DsparkBody::new(&gpu, &shape, layers, fc, head, embed)?;
    let (after, _) = gpu.mem_info()?;
    Ok(CardDspark::new::<Body35>(
        body,
        gpu,
        Arc::clone(target.context()),
        card,
        shape,
        before.saturating_sub(after),
    )?)
}
