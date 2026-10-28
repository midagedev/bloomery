//! `m_mlx`: the row's MLX-format fixture split opens through the session, and
//! its greedy picks are the session's own host-fed steps'.
//!
//! The split (`<family>-mlxsplit` under the fixture root) holds every typed
//! site at MLX_A4 and its routers at MLX_A8: the reader admits the sites
//! through Metal's site lists (`session::SITES`), the arena uploads the A4
//! blocks and dequantizes the routers exactly at open, and the session steps.
//! The reference is the session's own host-fed steps over the same ids.
//!
//! `m_mlx_folder`: the checkpoint folder the split was made from opens through
//! the session as it is, with no `prepare`, and is the split's model: the
//! same plan, the same greedy ids and the same first-step logits to the bit.
//!
//! `m_logits`: the session's logit rows against a committed dump of that
//! folder (`pins/<family>.txt` and `.f32` beside this module). The pin names
//! its ids; the session runs them teacher-forced, one id a step with its cache
//! kept, the way the dump does, and every step's row is held to the row's band
//! (`Row::logit_abs`). A pin that names its argmax (a line `argmax`, one id a
//! step, held to its own rows at read) also holds the session's pick of each
//! step to it. Position 0 reads one key (softmax 1), a rope angle of 0 and a
//! zero recurrent state; the steps above it read the rope, the scale, the QK
//! layout and the decay or the conv taps too. The dump is the external oracle:
//! the reference's modules run in f32 over the exact dequant, so it carries the
//! model's arithmetic without bf16 activation rounding. A missing pin, or a pin
//! whose steps, ids, argmax and bytes disagree, is a named error.

use std::path::{Path, PathBuf};
use std::time::Duration;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::check;
use bloomery_metal::arch::qwen3moe::plan::{self, Data, DeviceRoom, HostMemory};
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use model_hf::ModelFile;
use runtime::{Advance, Committed, PassSink, Plain, Stop, StopReason, Target};

use super::{CTX, mlx_split, row, says};

/// The greedy tokens the loop and the host-fed reference run.
const GREEDY: usize = 24;

/// The token `m_mlx` and `m_mlx_folder` start from: the row's seed.
fn seed() -> u32 {
    row::current().seed
}

/// FNV-1a 64-bit, the pin file's checksum of the f32 little-endian bytes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

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

/// The committed pin for `family`: the text and the f32 little-endian rows.
fn pin_of(family: &str) -> Result<(String, Vec<u8>), String> {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/gate_metal_qwen35moe_e2e/pins");
    let text_path = dir.join(format!("{family}.txt"));
    let raw_path = dir.join(format!("{family}.f32"));
    if !text_path.is_file() {
        return Err(format!(
            "the mlx-lm pin is not there: {}",
            text_path.display()
        ));
    }
    let text = std::fs::read_to_string(&text_path)
        .map_err(|e| format!("the mlx-lm pin {}: {e}", text_path.display()))?;
    if !raw_path.is_file() {
        return Err(format!(
            "the mlx-lm pin is not there: {}",
            raw_path.display()
        ));
    }
    let raw = std::fs::read(&raw_path)
        .map_err(|e| format!("the mlx-lm pin {}: {e}", raw_path.display()))?;
    Ok((text, raw))
}

struct Pin {
    ids: Vec<u32>,
    /// The argmax of each step's row the pin's own line names, when it has one.
    argmax: Option<Vec<u32>>,
    n: usize,
    rows: Vec<f32>,
}

/// The first index of the largest value of `row`: a NaN is not a candidate.
fn argmax_of(row: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in row.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.and_then(|(i, _)| u32::try_from(i).ok())
}

fn field<'a>(map: &'a [(&str, &str)], key: &str) -> Result<&'a str, String> {
    map.iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .ok_or_else(|| format!("the mlx-lm pin has no {key}"))
}

fn parse_pin(family: &str, text: &str, raw: &[u8]) -> Result<Pin, String> {
    let mut map = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let Some((key, value)) = line.split_once(' ') else {
            return Err(format!(
                "the mlx-lm pin's line {} is not a key and a value",
                i + 1
            ));
        };
        if map.iter().any(|(k, _)| *k == key) {
            return Err(format!("the mlx-lm pin repeats {key}"));
        }
        map.push((key, value));
    }
    let version = field(&map, "version")?;
    if version != "1" {
        return Err(format!(
            "the mlx-lm pin is version {version}; 1 is the one this gate reads"
        ));
    }
    let got = field(&map, "family")?;
    if got != family {
        return Err(format!("the mlx-lm pin is for {got}; this row is {family}"));
    }
    let ids: Vec<u32> = field(&map, "ids")?
        .split(',')
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| format!("the mlx-lm pin's ids hold {s:?}"))
        })
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("the mlx-lm pin's ids are empty".to_string());
    }
    let steps: usize = field(&map, "steps")?
        .parse()
        .map_err(|_| "the mlx-lm pin's steps is not a count".to_string())?;
    if steps != ids.len() {
        return Err(format!(
            "the mlx-lm pin has {steps} steps and {} ids",
            ids.len()
        ));
    }
    let n: usize = field(&map, "n")?
        .parse()
        .map_err(|_| "the mlx-lm pin's n is not a count".to_string())?;
    if n == 0 {
        return Err("the mlx-lm pin's n is 0".to_string());
    }
    let bytes = steps
        .checked_mul(n)
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| format!("the mlx-lm pin's {steps} steps of {n} values wrap"))?;
    if raw.len() != bytes {
        return Err(format!(
            "the mlx-lm pin holds {} bytes; {steps} steps of {n} f32s are {bytes} bytes",
            raw.len()
        ));
    }
    let want = format!("{:016x}", fnv1a(raw));
    let got_sum = field(&map, "fnv1a")?;
    if got_sum != want {
        return Err(format!(
            "the mlx-lm pin's checksum is {got_sum}; the f32 bytes checksum to {want}"
        ));
    }
    let mut rows = Vec::with_capacity(steps * n);
    for chunk in raw.as_chunks::<4>().0 {
        rows.push(f32::from_le_bytes(*chunk));
    }
    let argmax = match map.iter().find(|(k, _)| *k == "argmax") {
        None => None,
        Some((_, list)) => {
            let named: Vec<u32> = list
                .split(',')
                .map(|s| {
                    s.parse::<u32>()
                        .map_err(|_| format!("the mlx-lm pin's argmax holds {s:?}"))
                })
                .collect::<Result<_, _>>()?;
            let own: Vec<Option<u32>> = rows.chunks_exact(n).map(argmax_of).collect();
            if own.len() != named.len() || own.iter().zip(&named).any(|(o, w)| *o != Some(*w)) {
                return Err(format!(
                    "the mlx-lm pin names the argmax {named:?}; its rows' are {own:?}"
                ));
            }
            Some(named)
        }
    };
    Ok(Pin {
        ids,
        argmax,
        n,
        rows,
    })
}

/// One step's row against the pin's row: the largest |a − b| and where, the
/// values over the band and the rms.
struct Distance {
    max: f64,
    at: usize,
    session: f32,
    stored: f32,
    over: usize,
    rms: f64,
}

/// `got` against `want`. A NaN or an infinity is not a distance.
fn distance(got: &[f32], want: &[f32], band: f64) -> Result<Distance, String> {
    if got.len() != want.len() {
        return Err(format!(
            "the logit row holds {} values; the pin holds {}",
            got.len(),
            want.len()
        ));
    }
    let mut worst = Distance {
        max: 0.0,
        at: 0,
        session: 0.0,
        stored: 0.0,
        over: 0,
        rms: 0.0,
    };
    let mut sum_sq = 0.0f64;
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        let d = f64::from(*x) - f64::from(*y);
        if !d.is_finite() {
            return Err(format!("logit {i} is {x} against the pin's {y}"));
        }
        sum_sq += d * d;
        let d = d.abs();
        if d > band {
            worst.over += 1;
        }
        if i == 0 || d > worst.max {
            worst.max = d;
            worst.at = i;
            worst.session = *x;
            worst.stored = *y;
        }
    }
    worst.rms = (sum_sq / got.len() as f64).sqrt();
    Ok(worst)
}

/// `m_mlx`: the row's MLX model opens and generate over `Plain` emits the
/// session's own host-fed picks.
pub fn m_mlx(device: &MetalDevice, library: &Library) -> Clause {
    let path = super::model_path()?;
    let mut s = Session::open(device, library, &path, CTX).map_err(says)?;

    // The reference: the first id and each pick host-fed, one step each.
    s.reset().map_err(says)?;
    let mut ref_picks = Vec::with_capacity(GREEDY);
    let mut id = seed();
    for _ in 0..GREEDY {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        ref_picks.push(id);
    }

    // The loop over the target: the prompt again, then the passes.
    s.reset().map_err(says)?;
    let mut plain = Plain;
    let prompt = [seed()];
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
        "the MLX model {} opened ({} layers, every typed site MLX_A4) and generate over Plain \
         emitted the session's own {GREEDY} picks",
        path.display(),
        s.layer_count()
    )))
}

/// What one model file's session says of itself: the plan's lines, the greedy
/// ids from the row's seed and the logits row of the first step.
struct Told {
    plan: Vec<String>,
    ids: Vec<u32>,
    first_logits: Vec<f32>,
}

/// `file` opened through the session and run: the plan [`plan::render`] prints
/// for it, then [`GREEDY`] host-fed steps from the row's seed, the first step's
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
    let mut id = seed();
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

/// `m_logits`: the row's model, host-fed on the pin's ids one step each with
/// its cache kept, against the reference's logit rows of the folder. Each
/// step's distance is the largest absolute difference of its row, held to the
/// row's band (`Row::logit_abs`), and its pick to the pin's argmax where the
/// pin names it; the verdict names every step's distance and the worst.
///
/// Mutant: the row's band set below the largest distance a run prints
/// (`LFM3_LOGIT_ABS` in `row.rs`), or the first id of the pin's `argmax` line
/// changed (a named error at read, before any step runs).
pub fn m_logits(device: &MetalDevice, library: &Library) -> Clause {
    let spec = row::current();
    let family = spec.arch;
    let band = spec.logit_abs;
    let (text, raw) = pin_of(family)?;
    let pin = parse_pin(family, &text, &raw)?;
    let path = super::model_path()?;
    let mut s = Session::open(device, library, &path, CTX).map_err(says)?;
    let mut steps = Vec::with_capacity(pin.ids.len());
    for (k, (&id, want)) in pin.ids.iter().zip(pin.rows.chunks_exact(pin.n)).enumerate() {
        let picked = s.step(id, StepMode::HostFed).map_err(says)?.id;
        let row = s.logits_row().map_err(says)?;
        let d = distance(row, want, band).map_err(|e| format!("step {k} (id {id}): {e}"))?;
        let pinned = pin.argmax.as_ref().map(|a| a[k]);
        println!(
            "m_logits step {k} (id {id}): max |logit| difference {:.3e} at {} (session {:.6}, \
             pin {:.6}), {} values over the band, rms {:.3e}, pick {picked}{}",
            d.max,
            d.at,
            d.session,
            d.stored,
            d.over,
            d.rms,
            pinned.map_or_else(String::new, |p| format!(" (pin {p})"))
        );
        if let Some(p) = pinned
            && p != picked
        {
            return Err(format!(
                "step {k} (id {id}) picked {picked}; the pin's argmax is {p} (max |logit| \
                 difference {:.3e}, band {band})",
                d.max
            ));
        }
        steps.push(d);
    }
    let worst = (1..steps.len()).fold(0, |w, k| if steps[k].max > steps[w].max { k } else { w });
    let w = &steps[worst];
    let per_step: Vec<String> = steps.iter().map(|d| format!("{:.2e}", d.max)).collect();
    let over: usize = steps.iter().map(|d| d.over).sum();
    let line = format!(
        "pin {family}: max |logit| difference {:.3e} at step {worst} (id {}), logit {} \
         (session {:.6}, pin {:.6}); per step [{}]; band {band}, {over} values over it; \
         {} step(s) of {} values{}",
        w.max,
        pin.ids[worst],
        w.at,
        w.session,
        w.stored,
        per_step.join(", "),
        steps.len(),
        pin.n,
        pin.argmax.as_ref().map_or_else(String::new, |a| format!(
            "; every pick the pin's argmax {a:?}"
        ))
    );
    if w.max <= band {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}
