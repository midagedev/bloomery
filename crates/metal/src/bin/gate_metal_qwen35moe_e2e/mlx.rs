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
//!
//! `m_mlx_folder`: the checkpoint folder the split was made from opens through
//! the session as it is, with no `prepare`, and is the split's model: the
//! same plan, the same greedy ids and the same first-step logits to the bit.

use std::path::Path;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::check;
use bloomery_metal::arch::qwen3moe::plan::{self, Data, DeviceRoom, HostMemory};
use bloomery_metal::arch::qwen3moe::session::{Session, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use model_hf::ModelFile;
use runtime::{Advance, Plain, Stop, StopReason, Target};

use super::target::Quiet;
use super::{CTX, SEED, mlx_split, row, says};

/// The greedy tokens the loop and the host-fed reference run.
const GREEDY: usize = 24;

/// `m_mlx`: the MLX split's session opens and generate over `Plain` emits
/// the session's own host-fed picks — the open is the reader's admission of
/// every MLX_A4 site through Metal's lists, the picks the session's first
/// greedy steps.
pub fn m_mlx(device: &MetalDevice, library: &Library) -> Clause {
    let path = mlx_split::path("qwen35moe")?;
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

/// What one model file's session says of itself: the plan's lines, the greedy
/// ids from [`SEED`] and the logits row of the first step.
struct Told {
    plan: Vec<String>,
    ids: Vec<u32>,
    first_logits: Vec<f32>,
}

/// `file` opened through the session and run: the plan [`plan::render`] prints
/// for it, then [`GREEDY`] host-fed steps from [`SEED`], the first step's
/// logits row kept. The room lines are left out, they are the machine's.
fn tell(device: &MetalDevice, library: &Library, file: &ModelFile) -> Result<Told, String> {
    let outcome = check::check(file.header(), CTX);
    let no_device: Result<DeviceRoom, String> = Err("not compared".into());
    let no_host: Result<HostMemory, String> = Err("not compared".into());
    let plan = plan::render(
        Path::new("model"),
        Data::Complete,
        CTX,
        &outcome,
        &no_device,
        &no_host,
    )
    .lines;
    let mut s = Session::open_model(device, library, file, CTX).map_err(says)?;
    let mut ids = Vec::with_capacity(GREEDY);
    let mut first_logits = Vec::new();
    let mut id = SEED;
    for step in 0..GREEDY {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        if step == 0 {
            first_logits = s.logits_row().map_err(says)?.to_vec();
        }
        ids.push(id);
    }
    Ok(Told {
        plan,
        ids,
        first_logits,
    })
}

/// `m_mlx_folder`: the row's MLX fixture folder, opened as it is, tells what
/// the split `prepare` made of it tells — the plan's lines, the session's
/// first [`GREEDY`] host-fed ids and the first step's logits row bit for bit.
/// The ids alone could tie where weights differ a little; the logits cannot.
pub fn m_mlx_folder(device: &MetalDevice, library: &Library) -> Clause {
    let family = row::current().arch;
    let folder = mlx_split::folder(family)?;
    let split = mlx_split::path(family)?;
    let open = |path: &Path| ModelFile::open(path).map_err(|e| e.to_string());
    let (folder_file, split_file) = (open(&folder)?, open(&split)?);
    if !matches!(folder_file, ModelFile::Folder(_)) || !matches!(split_file, ModelFile::Gguf(_)) {
        return Err(format!(
            "{} opened as a split or {} as a folder",
            folder.display(),
            split.display()
        ));
    }
    let from_folder = tell(device, library, &folder_file)?;
    let from_split = tell(device, library, &split_file)?;
    if from_folder.plan != from_split.plan {
        let at = from_folder
            .plan
            .iter()
            .zip(&from_split.plan)
            .position(|(a, b)| a != b)
            .unwrap_or(from_folder.plan.len().min(from_split.plan.len()));
        return Err(format!(
            "the plan's line {at} is {:?} for the folder and {:?} for its split",
            from_folder.plan.get(at),
            from_split.plan.get(at)
        ));
    }
    if from_folder.ids != from_split.ids {
        let at = from_folder
            .ids
            .iter()
            .zip(&from_split.ids)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        return Err(format!(
            "the folder's id {at} is {} where its split's is {}",
            from_folder.ids[at], from_split.ids[at]
        ));
    }
    if from_folder.first_logits.len() != from_split.first_logits.len() {
        return Err(format!(
            "the first step's logits row holds {} values for the folder and {} for its split",
            from_folder.first_logits.len(),
            from_split.first_logits.len()
        ));
    }
    if let Some(at) = from_folder
        .first_logits
        .iter()
        .zip(&from_split.first_logits)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!(
            "the first step's logit {at} is {:#010x} for the folder and {:#010x} for its split",
            from_folder.first_logits[at].to_bits(),
            from_split.first_logits[at].to_bits()
        ));
    }
    Ok(Verdict::Noted(format!(
        "the folder {} opened with no prepare tells its split's plan ({} lines), {GREEDY} ids and \
         {} first-step logits, bit for bit",
        folder.display(),
        from_folder.plan.len(),
        from_folder.first_logits.len()
    )))
}
