//! The official Inkling 2 Mini checkpoint folder on the card (thinkingmachines/Inkling-2-Mini:
//! ModelOpt NVFP4 experts on layers 0 to 19, MXFP8 on the coda, BF16 dense), real tier only. The
//! folder opens through `model_hf::folder::open_with` in both arms of `BLOOMERY_INK2_DENSE`:
//! `bf16`, the dense as the folder stores it, and `q8_0`, the dense written as Q8_0 at load. Each
//! arm loads as `GpuModel::<Body>::open` loads a folder: the BF16 attention projections, token
//! embedding and head in their stored words, the coda's MXFP8 experts on the card leg's MXFP8 arm.
//! `--folder <dir>` names the folder (the recipe passes `BLOOMERY_INK2_OFFICIAL`).
//!
//! - `arms_differ_in_bytes`: the dense tensors (six attention projections a layer, the token
//!   embedding and the head) are BF16 in the `bf16` arm's header and Q8_0 in the `q8_0` arm's,
//!   every other tensor is the same type in both, and the `bf16` arm's resident bytes exceed the
//!   `q8_0` arm's.
//! - `forced_top1_is_vendors_official` (each arm, each set): the Mini's greedy tables
//!   (`ref_inkling2_*`, the vendor's CPU run of the GGUF twin) fed back one id a step after the
//!   prompt; our argmax disagreements at a vendor margin of [`MARGIN_FLOOR`] or more are at most
//!   the pin the Gaussian error model gives at [`SIGMA`], and σ̂ of the prompt's last logits row
//!   against the vendor's is under [`SIGMA`]. Inkling 2 has no f64 referee: the vendor's run of
//!   the twin is the family's oracle, and both arms differ from the twin in bytes (the coda's
//!   MXFP8 against the twin's Q8_0, and in the `bf16` arm the unrounded dense).
//!
//! Each arm's first [`FIRST_IDS`] greedy ids from set `a`'s prompt are printed beside the
//! vendor's. A step that raises the fault word ends the gate by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_official", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};

    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::folder::Folder;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::prompts::{GreedyRow, MARGIN_FLOOR, compare_forced, std_normal_cdf};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, ref_tensor_logical_in, verdict,
    };
    use gguf::GgmlType;
    use refset::arch::inkling2;
    use refset::arch::prompt_tables::{self, Greedy};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_inkling2_official";

    /// Positions of every store: the longest prompt (set `e`, 1,038 ids) and its 32 greedy ids
    /// fit.
    const CTX: usize = 2048;

    /// The greedy ids each arm prints from set `a`'s prompt.
    const FIRST_IDS: usize = 32;

    /// The dense tensors' sites in a layer: the arms differ in these, the token embedding and the
    /// head.
    const DENSE_SITES: [&str; 6] = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_rel.weight",
        "attn_gate.weight",
        "attn_output.weight",
    ];

    /// One arm's logit deviation from the vendor's run of the twin, in logit units: the σ of the
    /// teacher-forced argmax pin ([`forced_pin`]) and the bound on σ̂.
    /// PIN(2026-11-23): 1.67 times the measured maximum σ̂ 0.329 (both arms' five sets, 0.121 to
    /// 0.329), beside the family's real-tier 0.5 the twin is held to.
    const SIGMA: f64 = 0.55;

    /// The two arms: the name `BLOOMERY_INK2_DENSE` gives each and whether it asks for Q8_0.
    const ARMS: [(&str, bool); 2] = [("bf16", false), ("q8_0", true)];

    /// One set's prompt, its greedy table and the vendor's logits row at the prompt's last
    /// position.
    struct Set {
        label: &'static str,
        prompt: Vec<u32>,
        greedy: Greedy,
        logits: Vec<f32>,
    }

    /// One arm's header facts the bytes clause reads: each tensor's name and type.
    struct Arm {
        word: &'static str,
        types: Vec<(String, GgmlType)>,
        resident: usize,
    }

    /// The folder `--folder` names.
    fn folder_arg() -> Result<PathBuf, GateError> {
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            if a == "--folder" {
                return args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| format!("{NAME}: --folder takes a directory").into());
            }
        }
        Err(format!("{NAME}: --folder <dir> names the official folder").into())
    }

    /// The last row of `n` values of node `name`'s last occurrence.
    fn last_row(man: &RefManifest, name: &str, n: usize) -> Result<Vec<f32>, GateError> {
        let mut occ = 0u32;
        while man.tensor(name, occ + 1).is_ok() {
            occ += 1;
        }
        let row = man.tensor(name, occ)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        if all.len() < n || !all.len().is_multiple_of(n) {
            return Err(format!("{name} occurrence {occ} is {:?}, not rows of {n}", row.ne).into());
        }
        Ok(all[all.len() - n..].to_vec())
    }

    fn open_set(name: &'static str) -> Result<Set, GateError> {
        let man = RefManifest::open(&inkling2::LCPP.path(name), &inkling2::LCPP)?;
        let vocab = man.tensor("result_output", 0)?.ne[0] as usize;
        let set = Set {
            label: name.trim_start_matches("ref_inkling2_"),
            prompt: prompt_tables::tokens(&man)?.to_vec(),
            greedy: prompt_tables::greedy(&man)?,
            logits: last_row(&man, "result_output", vocab)?,
        };
        let need = set.prompt.len() + set.greedy.ids.len();
        if set.greedy.ids.is_empty() || set.greedy.margins.len() != set.greedy.ids.len() {
            return Err(format!("{name}: a greedy table of {} ids", set.greedy.ids.len()).into());
        }
        if need > CTX {
            return Err(format!("{name}: {need} positions, the gate's stores hold {CTX}").into());
        }
        Ok(set)
    }

    /// The RMS of the two rows' difference over the vendor's columns, in logit units.
    fn sigma_hat(ours: &[f32], theirs: &[f32]) -> f64 {
        let n = theirs.len().max(1) as f64;
        (ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// The disagreements the Gaussian error of two implementations predicts at the clear
    /// positions of `margins` (those at or past [`MARGIN_FLOOR`]) and the pin that follows: the
    /// expected count plus three standard deviations of it, rounded up. [derived]: at the vendor's
    /// margin `m` the argmax flips with probability Φ(−m / (√2 σ)).
    fn forced_pin(margins: &[f32], sigma: f64) -> (f64, usize) {
        let sigma_pair = sigma * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// Whether `name` is one of the tensors the arms differ in.
    fn dense(name: &str) -> bool {
        name == "token_embd.weight"
            || name == "output.weight"
            || name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .is_some_and(|(_, site)| DENSE_SITES.contains(&site))
    }

    /// The model of the folder `dir` in the arm `(word, q8)` on a fresh card handle.
    fn open_arm(
        dir: &Path,
        (word, q8): (&'static str, bool),
    ) -> Result<(Arm, GpuModel<Body>), GateError> {
        let t = std::time::Instant::now();
        let opened = model_hf::folder::open_with(dir, q8)
            .map_err(|e| format!("{NAME}: {} in arm {word}: {e}", dir.display()))?;
        let read = t.elapsed();
        let types = opened
            .header
            .iter_tensors()
            .map(|(_, t)| (t.name.clone(), t.ty))
            .collect();
        let model = Folder::new(opened.header, opened.source);
        let mut m = GpuModel::<Body>::open(Gpu::new()?, model, CTX)?;
        let resident = m.resident_bytes();
        println!(
            "{NAME}: arm {word}: device {}, resident_bytes={resident} ctx={CTX}, folder read in \
             {:.1} s, loaded in {:.1} s",
            m.body_parts(NAME)?.0.device_name()?,
            read.as_secs_f64(),
            (t.elapsed() - read).as_secs_f64()
        );
        Ok((
            Arm {
                word,
                types,
                resident,
            },
            m,
        ))
    }

    /// Clause `forced_top1_is_vendors_official` of set `s` in arm `word`, and set `a`'s first
    /// greedy ids printed.
    fn forced(m: &mut GpuModel<Body>, word: &str, s: &Set) -> Result<bool, GateError> {
        if s.label == "a" {
            m.reset()?;
            let mut ids = vec![m.step(&s.prompt)?];
            while ids.len() < FIRST_IDS {
                let last = ids[ids.len() - 1];
                ids.push(m.step(&[last])?);
            }
            let same = ids
                .iter()
                .zip(&s.greedy.ids)
                .take_while(|(a, b)| a == b)
                .count();
            println!("first_ids arm={word} set=a ours   {ids:?}");
            println!(
                "first_ids arm={word} set=a vendor {:?} (first {same} the same)",
                &s.greedy.ids[..FIRST_IDS.min(s.greedy.ids.len())]
            );
        }
        if !tier::run_clause(
            &format!("forced_top1_is_vendors_official arm={word} set={}", s.label),
            Tag::Oracle,
        )? {
            return Ok(true);
        }
        m.reset()?;
        let mut table = vec![m.step(&s.prompt)?];
        let row = m.logits()?;
        let sigma = sigma_hat(&row[..s.logits.len()], &s.logits);
        for &id in &s.greedy.ids[..s.greedy.ids.len() - 1] {
            table.push(m.step(&[id])?);
        }
        let vendor = GreedyRow {
            id: 0,
            n_tokens: s.prompt.len(),
            argmax: s.greedy.ids[0],
            top5: vec![s.greedy.ids[0]],
            gen_ids: s.greedy.ids.clone(),
            gen_margins: s.greedy.margins.clone(),
        };
        let report = compare_forced(&[table], &[vendor], MARGIN_FLOOR);
        let (expected, pin) = forced_pin(&s.greedy.margins, SIGMA);
        let pass = report.disagree_ge_floor <= pin && sigma < SIGMA;
        println!(
            "forced_top1_is_vendors_official arm={word} set={} positions={} disagree={} \
             ge_floor={} expected {expected:.2} pin {pin} sigma_hat={sigma:.3} (under {SIGMA}) {}",
            s.label,
            report.positions,
            report.disagree.len(),
            report.disagree_ge_floor,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `arms_differ_in_bytes` over the two arms' headers and resident bytes.
    fn arms_differ_in_bytes(bf16: &Arm, q8: &Arm) -> Result<bool, GateError> {
        if !tier::run_clause("arms_differ_in_bytes", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let count =
            |a: &Arm, ty: GgmlType| a.types.iter().filter(|(n, t)| dense(n) && *t == ty).count();
        let dense_total = bf16.types.iter().filter(|(n, _)| dense(n)).count();
        let others_same = bf16.types.len() == q8.types.len()
            && bf16
                .types
                .iter()
                .zip(&q8.types)
                .all(|((na, ta), (nb, tb))| na == nb && (dense(na) || ta == tb));
        let (b, q) = (count(bf16, GgmlType::BF16), count(q8, GgmlType::Q8_0));
        let pass = dense_total > 0
            && b == dense_total
            && q == dense_total
            && others_same
            && bf16.resident > q8.resident;
        println!(
            "arms_differ_in_bytes dense={dense_total} {}_bf16={b} {}_q8_0={q} others_same={others_same} \
             resident {}={} {}={} {}",
            bf16.word,
            q8.word,
            bf16.word,
            bf16.resident,
            q8.word,
            q8.resident,
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let tier = Tier::from_env()?;
        if !matches!(tier, Tier::Real) {
            return Err(format!(
                "{NAME}: the official folder has no fixture; the gate runs in the real tier only"
            )
            .into());
        }
        let dir = folder_arg()?;
        let sets: Vec<Set> = inkling2::SETS
            .iter()
            .map(|&n| open_set(n))
            .collect::<Result<_, _>>()?;
        let mut ok = true;
        let mut arms = Vec::new();
        for arm in ARMS {
            let (facts, mut m) = open_arm(&dir, arm)?;
            m.set_mode(StepMode::Graph);
            for s in &sets {
                ok &= forced(&mut m, arm.0, s)?;
            }
            arms.push(facts);
        }
        ok &= arms_differ_in_bytes(&arms[0], &arms[1])?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME}: the official folder in both dense arms on the card, the arms' dense \
             tensors BF16 and Q8_0 and their resident bytes apart, each arm's teacher-forced \
             argmax over the Mini's greedy tables within the pin at σ {SIGMA}"
        );
        Ok(())
    }
}
