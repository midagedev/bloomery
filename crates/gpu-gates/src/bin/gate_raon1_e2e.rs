//! The raon1 end-to-end gate: the whole decode chain — the embedding into
//! Raon_mHC's four streams, the RDA and gated latent layers, the dense and
//! routed SiTU blocks, the streams' mean, the head and the argmax — on one
//! card, over the Raon-1-mini Q4_K_M file, one prompt id a step, against the
//! vendor branch's CPU sets (refset family `lcpp-raon1`, `just
//! dump-ref-raon1`).
//!
//! For each prompt set of [`SETS`] (English prose, Korean prose, Rust code):
//!
//! - `top1_is_vendors`: the argmax after the prompt is the set's top-k
//!   table's first id. The top-[`TOP_K`] overlap of the two last-position
//!   rows and `d`, the largest distance of our logit from the vendor's over
//!   the table's ids, are printed beside it.
//! - `logits_are_vendors`: the whole last-position row's relative distance
//!   from the vendor's `result_output` is at most [`LOGITS_REL_BAND`]. It
//!   bounds `d`, so a chain that moves every logit cannot widen the named-tie
//!   band below with its own error.
//! - `greedy_is_vendors`: the greedy continuation is the set's `greedy.tsv`,
//!   id for id, over its whole length, or up to a named tie: a first
//!   divergence at a step whose vendor margin is at most `2d`, the most two
//!   logits each `d` off can close. A divergence past `2d` is red. Either is
//!   printed with the vendor's margin and our own gap between the two ids;
//!   the steps after a tie continue another text and are not compared.
//!
//! Then, for each set, over the same file in the verifying load a seat with a draft opens
//! (`GpuModel::open_verifying`), as a session in graph mode unless named:
//!
//! - `prompt_rows_is_steps`: the first 8 ids of the prompt as the rows pass
//!   (`GpuModel::prompt_rows`, a call of up to a pass's rows) and one id a step: the same argmax,
//!   the same logits row by bits, the same steps after it.
//! - `prompt_wide_is_vendors`, on the sets of at least [`WIDE_P`] ids (the Rust set and the 1,536-id
//!   wikitext one): the whole prompt as the rows pass, whose units of more than 8 rows run as GEMMs
//!   (the wide arm), reaches the vendor's argmax, and its last-position logits row is within
//!   [`WIDE_REL_BAND`] of the vendor's `result_output`, and of the steps' row at the full length and
//!   at each of [`DRIFT_AT`]; a set with a greedy table also holds the named-tie continuation. The
//!   steps' own distance from the vendor's is printed beside it.
//! - `prompt_wide_is_partitions`, on the sets past [`SPLIT_AT`] ids: the same prompt as one call and
//!   as two cut at [`SPLIT_AT`] gives the last logits row by bits.
//! - `wide_runs_the_file`, on the plain load: the load found every layer of the file runnable by
//!   the wide arm (`Body::wide_ok`), so the wide clauses run the wide arm and not the passes a
//!   file it cannot run takes.
//! - `verify_is_steps`, in eager and in graph mode: after the prompt call, a verify of the plain
//!   run's first 8 ids (and of its first 2) gives the plain run's next ids, each row's logits the
//!   plain step's by bits.
//! - `commit_keeps_steps`, in eager and in graph mode: a verify of 8 rows committed at 1, 3 and 8
//!   rows, then steps: the plain run's ids.
//! - `taps_are_steps`: the rows entering the draft's tapped layers, as a step taps them, are the
//!   rows the rows pass hands its sink over the prompt (every position once, in order) and the
//!   rows a verify taps: by bits for a call of a pass's rows, for the wide units' within
//!   [`TAPS_MEDIAN_BAND`] at the median and [`TAPS_P90_BAND`] at the 90th percentile of a layer's rows. The 1,536-id set runs it with a window of [`TAPS_WINDOW`] positions, whose
//!   first row lies inside a unit, and prints the spread of its rows a layer and a block of positions.
//! - `speculative_is_plain`: Raon-1-mini's DFlash2 draft beside it (`raon1_dflash`: the serve
//!   seat's pick and open), every pass a proposal of 7 ids verified by one pass of 8 rows: the
//!   first [`DRAFTED`] greedy ids are the plain session's (its prompt call, then a step an id), id
//!   for id. The passes, and the proposed ids the target kept, are printed beside it.
//!
//! Before the card loads, the band check of `taps_are_steps` is held to what it refuses: eight
//! rows at a distance inside the bands pass it, and the same rows with one tap of the last row NaN
//! do not.
//!
//! A step that raises the card's fault word ends the gate by name. A set
//! absent from the box, dumped from another file or build, or unfinished ends
//! the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raon1_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "gpu")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the serve seat and two gates share the opener; each takes the parts it reads"
)]
#[path = "shared/raon1_dflash.rs"]
mod raon1_dflash;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::e2e::{rel, top_ids};
    use crate::greedy::greedy;
    use crate::quiet::Quiet;
    use crate::raon1_dflash::{self, Pick, RaonDraft};
    use app::Session;
    use bloomery_gpu::arch::raon1::{Body, HC_OUT, STREAMS};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, data_dir, open_split, ref_model_path,
        ref_tensor_logical_in, verdict,
    };
    use model::arch::Arch;
    use model::arch::dflash2::Hparams as DraftHparams;
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::arch::raon1::{A, B, C, D, LCPP};
    use runtime::{Advance, RowLogits, Speculative, Target, Verify, Want};

    /// The sets the step path is held to, in the order they run.
    const SETS: &[&str] = &[A, B, C];

    /// Rows of each latent layer's cache: the longest prompt and its
    /// continuation fit (the 1,536-id set included).
    const CTX: usize = 2048;

    /// The rows of a pass: a call of up to this many ids runs as the steps do.
    const PASS: usize = 8;
    const _: () = assert!(PASS == bloomery_gpu::linear::PASS_ROWS);

    /// The median, over the rows of a wide arm's tapped window (or of a verify after it), of a
    /// layer's relative distance from the row a step taps. PIN(2026-10-22): 7.0e-2, 2.2 times the
    /// worst measured (3.2e-2, layer 22 of the 1,536-id set; 6.5e-3 at layer 2): a wide unit's
    /// activation blocks (128 values) and its GEMM sums differ from the step's, and the rounding
    /// chains through the layers; a stage wrong for the rows of a unit moves the median to O(1).
    const TAPS_MEDIAN_BAND: f64 = 7.0e-2;

    /// The 90th percentile of the same distances. PIN(2026-10-22): 1.5e-1, 2.1 times the worst
    /// measured (7.0e-2, the last 256 positions of the 1,536-id set). The largest rows (4.3e-1 a
    /// layer) are a heavy tail, likely tokens whose router took another expert on a
    /// rounding-sized score gap (not measured): printed, not pinned; a stage wrong for a quarter
    /// of the rows moves the 90th percentile to O(1).
    const TAPS_P90_BAND: f64 = 1.5e-1;

    /// The tap window of the long set's `taps_are_steps`: its last rows only, from inside a unit
    /// (1,536 ids, so the window opens at position 536 of units of 256).
    const TAPS_WINDOW: usize = 1000;

    /// The positions a row of `taps_are_steps`'s profile covers.
    const TAPS_BLOCK: usize = 256;

    /// The fewest ids of a set `prompt_wide_is_vendors` runs.
    const WIDE_P: usize = 64;

    /// The wide arm's last-position logits row's relative distance from the vendor's, and from the
    /// steps' row at the same length. PIN(2026-10-22): 8.0e-2, 1.5 times the worst measured (5.3e-2
    /// from the vendor on the 1,536-id set, where the steps' own distance reads 5.0e-2; 3.9e-2 on
    /// the 72-id set beside the steps' 4.2e-2; 3.6e-2 from the steps' row at every prefix): a wide
    /// unit quantizes its activations in blocks of 128 values where the step's K-quant dots take
    /// 256 (and its per-head value projection takes f32 rows), so the arm is a rounding class away
    /// from the steps that the vendor's own quantization sits as far from; a stage that reads
    /// another row, position or expert is O(1) away.
    const WIDE_REL_BAND: f64 = 8.0e-2;

    /// The prefix lengths of a wide set at which the wide row is also held [`WIDE_REL_BAND`] from
    /// the steps' row: the drift with the positions is printed, and a jump at a unit's end names
    /// a stage that carries state across units.
    const DRIFT_AT: &[usize] = &[64, 256, 512, 1024];

    /// Where `prompt_wide_is_partitions` cuts a prompt: a multiple of a pass's rows, and of
    /// neither a unit's rows nor a latent attention chunk's, so the second call's launches start
    /// at other positions than the first call's: a row's bits do not depend on where its launch
    /// begins.
    const SPLIT_AT: usize = 600;
    const _: () = assert!(SPLIT_AT.is_multiple_of(PASS));

    /// The fewest ids of a set `prompt_wide_is_partitions` runs: both calls hold a wide unit.
    const SPLIT_FROM: usize = SPLIT_AT + PASS + 1;

    /// The sets the wide arm is held to the vendor on: past [`WIDE_P`] ids.
    const WIDE_SETS: &[&str] = &[C, D];

    /// The file's layers: the last one's route is printed.
    const LAYERS: usize = 25;

    /// The last-position ids whose overlap is printed.
    const TOP_K: usize = 5;

    /// The last-position logits row's relative distance from the vendor's.
    /// PIN(2026-10-21): measured, not derived: the chain at 3.33e-2, 3.87e-2
    /// and 3.16e-2 on sets a, b and c; SiTU's gate beta 3 for 4 at 7.90e-2,
    /// 8.88e-2 and 1.98e-1.
    const LOGITS_REL_BAND: f64 = 5.0e-2;

    /// The greedy ids `speculative_is_plain` compares a set: past the vendor's table, so a run
    /// holds several dozen passes.
    const DRAFTED: usize = 64;

    /// A verify of a proposal: the token at the target's position, then the draft's 7 ids.
    const VERIFY_ROWS: usize = 8;

    /// The steps `prompt_rows_is_steps` and `commit_keeps_steps` run after the call they check.
    const AFTER: usize = 4;

    /// The rows `commit_keeps_steps` keeps of a verify of [`VERIFY_ROWS`].
    const KEEPS: [usize; 3] = [1, 3, VERIFY_ROWS];

    type Seat = Session<Body>;
    type Spec = Speculative<RaonDraft, VERIFY_ROWS>;

    /// One set's prompt and its two tables.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// The vendor's whole logits row at the prompt's last position
        /// (`result_output`).
        logits: Vec<f32>,
        /// The vendor's route of the last layer's feed-forward site at the
        /// prompt's last position: `pre`, `post`, `comb` in [`HC_OUT`]'s order;
        /// `None` for a set dumped without those taps.
        route: Option<Vec<f32>>,
    }

    /// The last `n` values of tap `name`'s occurrence `occurrence` in `man`.
    fn last_of(
        man: &RefManifest,
        name: &str,
        occurrence: u32,
        n: usize,
    ) -> Result<Vec<f32>, GateError> {
        let row = man.tensor(name, occurrence)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let at = all
            .len()
            .checked_sub(n)
            .ok_or_else(|| format!("{name}: {} values, want {n}", all.len()))?;
        Ok(all[at..].to_vec())
    }

    fn open_set(name: &'static str) -> Result<Set, GateError> {
        let man = RefManifest::open(&LCPP.path_in(&data_dir(), name), &LCPP)?;
        let row = man.tensor("result_output", 0)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let vocab = usize::try_from(row.ne[0])?;
        if vocab == 0 || all.len() % vocab != 0 {
            return Err(format!("{name}: result_output is {:?}", row.ne).into());
        }
        let logits = all[all.len() - vocab..].to_vec();
        let last = LAYERS - 1;
        let route = match man.tensor(&format!("hc_comb-{last}"), 1) {
            Err(_) => None,
            Ok(_) => {
                let mut r = last_of(&man, &format!("hc_pre-{last}"), 1, STREAMS)?;
                r.extend(last_of(&man, &format!("hc_post-{last}"), 1, STREAMS)?);
                r.extend(last_of(
                    &man,
                    &format!("hc_comb-{last}"),
                    1,
                    STREAMS * STREAMS,
                )?);
                Some(r)
            }
        };
        Ok(Set {
            name,
            prompt: prompt_tables::tokens(&man)?.to_vec(),
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            logits,
            route,
        })
    }

    /// Our route of the last layer's feed-forward site at the prompt's last
    /// position beside the vendor's: `pre`'s and `post`'s relative distances,
    /// and `comb`'s from the vendor's and from its transpose (the two
    /// orientations of `comb[src · 4 + dst]`).
    fn print_route(m: &GpuModel<Body>, s: &Set) -> Result<(), GateError> {
        let Some(theirs) = &s.route else {
            println!(
                "route set={} has no layer-{} route taps",
                s.name,
                LAYERS - 1
            );
            return Ok(());
        };
        let ours = m.body("gate_raon1_e2e")?.last_route(m.gpu().stream())?;
        if ours.len() != HC_OUT || theirs.len() != HC_OUT {
            return Err(format!("routes of {} and {} values", ours.len(), theirs.len()).into());
        }
        let k = 2 * STREAMS;
        let transposed: Vec<f32> = (0..STREAMS * STREAMS)
            .map(|i| theirs[k + (i % STREAMS) * STREAMS + i / STREAMS])
            .collect();
        println!(
            "route set={} layer={} pre_rel={:.3e} post_rel={:.3e} comb_rel={:.3e} \
             comb_rel_transposed={:.3e}",
            s.name,
            LAYERS - 1,
            rel(&ours[..STREAMS], &theirs[..STREAMS]),
            rel(&ours[STREAMS..k], &theirs[STREAMS..k]),
            rel(&ours[k..], &theirs[k..]),
            rel(&ours[k..], &transposed),
        );
        Ok(())
    }

    /// The first step whose id is not the vendor's, and our logit gap
    /// between our id and the vendor's there.
    type Divergence = Option<(usize, f32)>;

    /// The greedy ids from `first` (the argmax of `row`, the prompt's last
    /// logits), each fed back, until the set's length or the first id that
    /// is not the vendor's: that step and our logit gap between our id and
    /// the vendor's there.
    fn continue_greedy(
        m: &mut GpuModel<Body>,
        s: &Set,
        first: u32,
        mut row: Vec<f32>,
    ) -> Result<(Vec<u32>, Divergence), GateError> {
        let mut ids = vec![first];
        loop {
            let i = ids.len() - 1;
            if ids[i] != s.greedy.ids[i] {
                let gap = row[ids[i] as usize] - row[s.greedy.ids[i] as usize];
                return Ok((ids, Some((i, gap))));
            }
            if ids.len() == s.greedy.ids.len() {
                return Ok((ids, None));
            }
            ids.push(m.step(&[ids[i]])?);
            row = m.logits()?;
        }
    }

    /// The DFlash2 draft's file beside the target, as the serve seat picks it, and the layers it
    /// taps.
    fn draft_file() -> Result<(PathBuf, Vec<usize>), GateError> {
        let target = ref_model_path()?;
        let path = match raon1_dflash::pick(None, &target)? {
            Pick::File(path, _) => path,
            Pick::Absent(dirs) => {
                return Err(format!(
                    "no DFlash2 draft beside {} (looked at {dirs:?})",
                    target.display()
                )
                .into());
            }
        };
        let split =
            gguf::Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = DraftHparams::read(&split).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((path, hp.target_layers))
    }

    /// The DFlash2 draft of the file at `path` on the seat's card, as the serve seat opens it.
    fn open_draft(seat: &Seat, path: &Path) -> Result<RaonDraft, GateError> {
        let file = Arc::new(open_split(Arch::Raon1, "gate-gpu-raon1-e2e")?);
        let draft = raon1_dflash::open(Gpu::new()?, path, (file, seat.model()), CTX, "gate")?;
        println!(
            "draft {} resident_bytes={}",
            path.display(),
            draft.resident_bytes()
        );
        Ok(draft)
    }

    /// `n` greedy ids of `prompt` from a reset seat through `spec`, every pass a proposal; the
    /// passes, and the proposed ids the target kept.
    fn drafted(
        spec: &mut Spec,
        seat: &mut Seat,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, usize, usize), GateError> {
        Target::reset(seat)?;
        let first = Advance::prompt(spec, seat, prompt)?;
        Advance::begin(spec, seat, prompt, first)?;
        let mut ids = vec![first];
        let (mut passes, mut kept) = (0, 0);
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            let before = ids.len();
            let c = Advance::pass(spec, seat, last, &mut ids)?;
            if !c.proposed || c.rows != VERIFY_ROWS || ids.len() - before != c.kept {
                return Err(format!(
                    "a pass of {} rows (proposed {}) kept {} and appended {}: every pass is a \
                     proposal of {} verified by {VERIFY_ROWS} rows",
                    c.rows,
                    c.proposed,
                    c.kept,
                    ids.len() - before,
                    VERIFY_ROWS - 1
                )
                .into());
            }
            passes += 1;
            kept += c.kept - 1;
        }
        ids.truncate(n);
        Ok((ids, passes, kept))
    }

    /// Clause `prompt_rows_is_steps` on set `s`: the prompt's first [`PASS`] ids one id a step and
    /// as the rows pass (`GpuModel::prompt_rows`), each from a reset model: the same argmax, the
    /// same logits row by bits, and the same [`AFTER`] steps after it.
    fn prompt_rows_is_steps(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let head = &s.prompt[..PASS.min(s.prompt.len())];
        let mut run = |rows: bool| -> Result<(Vec<u32>, Vec<f32>), GateError> {
            m.reset()?;
            let first = if rows {
                m.prompt_rows(head)?
            } else {
                m.step(head)?
            };
            let row = m.logits()?;
            let mut ids = vec![first];
            while ids.len() <= AFTER {
                let last = ids[ids.len() - 1];
                ids.push(m.step(&[last])?);
            }
            Ok((ids, row))
        };
        let (steps, steps_row) = run(false)?;
        let (rows, rows_row) = run(true)?;
        let same_row = bits_equal(&steps_row, &rows_row);
        let pass = steps == rows && same_row;
        println!(
            "prompt_rows_is_steps set={} steps={steps:?} rows={rows:?} logits_same_bits={same_row} {}",
            s.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `prompt_wide_is_vendors` on set `s` (at least [`WIDE_P`] ids): the prompt as the
    /// rows pass from a reset model, its units of more than a pass's rows run wide. The argmax is
    /// the vendor's top-1, the last logits row is within [`WIDE_REL_BAND`] of the vendor's
    /// `result_output` and of the steps' row at the prompt's length and at each of [`DRIFT_AT`]
    /// below it, and a set with a greedy table continues as the vendor does, up to a named tie.
    /// The steps' own distance from the vendor's is printed beside it.
    fn prompt_wide_is_vendors(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if s.prompt.len() < WIDE_P {
            return Err(format!(
                "{}: a prompt of {} ids; the wide clause runs from {WIDE_P}",
                s.name,
                s.prompt.len()
            )
            .into());
        }
        let theirs = s
            .top
            .first()
            .ok_or_else(|| format!("{}: an empty top-k table", s.name))?;
        m.reset()?;
        let first = m.prompt_rows(&s.prompt)?;
        let row = m.logits()?;
        let d = s
            .top
            .iter()
            .map(|t| (row[t.id as usize] - t.logit).abs())
            .fold(0.0f32, f32::max);
        let ours5 = top_ids(&row, TOP_K);
        let overlap = s
            .top
            .iter()
            .take(TOP_K)
            .filter(|t| ours5.contains(&(t.id as usize)))
            .count();
        let logits_rel = rel(&row, &s.logits);
        let top1 = first == theirs.id;
        let logits_ok = logits_rel <= WIDE_REL_BAND;
        let mut pass = top1 && logits_ok;
        if !s.greedy.ids.is_empty() && s.greedy.margins.len() == s.greedy.ids.len() {
            let (_, diverge) = continue_greedy(m, s, first, row.clone())?;
            let (ok, at) = match diverge {
                None => (true, "none".to_owned()),
                Some((i, our_gap)) => {
                    let margin = s.greedy.margins[i];
                    let tie = margin <= 2.0 * d;
                    (
                        tie,
                        format!(
                            "step {i} vendor_margin={margin} our_gap={our_gap} 2d={} ({})",
                            2.0 * d,
                            if tie { "a named tie" } else { "past 2d" }
                        ),
                    )
                }
            };
            println!(
                "prompt_wide_greedy set={} first_divergence={at} {}",
                s.name,
                verdict(ok)
            );
            pass &= ok;
        }
        m.reset()?;
        m.step(&s.prompt)?;
        let steps_row = m.logits()?;
        let (steps_rel, steps_vendor_rel) = (rel(&row, &steps_row), rel(&steps_row, &s.logits));
        let mut drift = vec![(s.prompt.len(), steps_rel)];
        for &n in DRIFT_AT.iter().filter(|&&n| n < s.prompt.len()) {
            m.reset()?;
            m.prompt_rows(&s.prompt[..n])?;
            let wide = m.logits()?;
            m.reset()?;
            m.step(&s.prompt[..n])?;
            drift.push((n, rel(&wide, &m.logits()?)));
        }
        drift.sort_by_key(|&(n, _)| n);
        let drift_ok = drift.iter().all(|&(_, r)| r <= WIDE_REL_BAND);
        pass &= drift_ok;
        let drift = drift
            .iter()
            .map(|(n, r)| format!("{n}:{r:.3e}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "prompt_wide_is_vendors set={} prompt_ids={} ours={first} vendor={} \
             top{TOP_K}_overlap={overlap}/{TOP_K} d={d} logits_rel={logits_rel:.4e} \
             band={WIDE_REL_BAND:.1e} steps_vendor_rel={steps_vendor_rel:.4e} \
             steps_rel_by_ids=[{drift}] {}",
            s.name,
            s.prompt.len(),
            theirs.id,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `prompt_wide_is_partitions` on set `s`: the prompt as one call and as two cut at
    /// [`SPLIT_AT`] reach the same last logits row, by bits. A wide row depends on its own id,
    /// position and the carried state alone, so a unit or a call that mislays a position, the
    /// conv ring, the delta state or the cache rows moves it.
    fn prompt_wide_is_partitions(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if s.prompt.len() < SPLIT_FROM {
            return Err(format!(
                "{}: a prompt of {} ids; the split clause runs from {}",
                s.name,
                s.prompt.len(),
                SPLIT_FROM
            )
            .into());
        }
        m.reset()?;
        let whole = m.prompt_rows(&s.prompt)?;
        let whole_row = m.logits()?;
        m.reset()?;
        m.prompt_rows(&s.prompt[..SPLIT_AT])?;
        let cut = m.prompt_rows(&s.prompt[SPLIT_AT..])?;
        let same_row = bits_equal(&whole_row, &m.logits()?);
        let pass = whole == cut && same_row;
        println!(
            "prompt_wide_is_partitions set={} prompt_ids={} split_at={SPLIT_AT} ids_equal={} \
             logits_same_bits={same_row} {}",
            s.name,
            s.prompt.len(),
            whole == cut,
            verdict(pass)
        );
        Ok(pass)
    }

    /// A verify of `M` rows of the plain run `plain` from the prompt's end, through the session
    /// from a reset: whether the prompt call's id is `plain[0]` and the verify's ids are
    /// `plain[1..=M]`, and whether each row's logits are, by bits, `rows` one id later (`rows[r]`
    /// is the row whose argmax is `plain[r]`). The verify is left waiting for its commit.
    fn verified<const M: usize>(
        seat: &mut Seat,
        s: &Set,
        (plain, rows): (&[u32], &[Vec<f32>]),
    ) -> Result<(bool, bool), GateError> {
        Target::reset(seat)?;
        let first = Target::prompt(seat, &s.prompt, Want::Argmax)?.argmax();
        let out = Verify::verify::<M>(seat, <[u32; M]>::try_from(&plain[..M])?)?;
        let ids = first == plain[0] && out[..] == plain[1..=M];
        let mut logits = true;
        for (r, want) in rows[1..=M].iter().enumerate() {
            logits &= bits_equal(RowLogits::row_logits(seat, r)?, want);
        }
        Ok((ids, logits))
    }

    /// Clauses `verify_is_steps` (verifies of [`VERIFY_ROWS`] and 2 rows) and
    /// `commit_keeps_steps` (a verify of [`VERIFY_ROWS`] rows committed at each of [`KEEPS`], then
    /// [`AFTER`] steps, the plain run's ids) on set `s` in the seat's mode `mode`.
    fn verify_is_steps(seat: &mut Seat, s: &Set, mode: &str) -> Result<bool, GateError> {
        let (plain, rows) = greedy(seat, &s.prompt, VERIFY_ROWS + 1 + AFTER, true)?;
        let run = (&plain[..], &rows[..]);
        let mut ok = true;
        for (m, (ids, logits)) in [
            (VERIFY_ROWS, verified::<VERIFY_ROWS>(seat, s, run)?),
            (2, verified::<2>(seat, s, run)?),
        ] {
            let pass = ids && logits;
            println!(
                "verify_is_steps set={} mode={mode} rows={m} ids={ids} logits_same_bits={logits} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        for k in KEEPS {
            let (ids, logits) = verified::<VERIFY_ROWS>(seat, s, run)?;
            Verify::commit(seat, k)?;
            let mut after = vec![plain[k]];
            while after.len() <= AFTER {
                let last = after[after.len() - 1];
                after.push(Target::step(seat, last, Want::Argmax)?.argmax());
            }
            let kept = after[..] == plain[k..=k + AFTER];
            let pass = ids && logits && kept;
            println!(
                "commit_keeps_steps set={} mode={mode} kept={k} after={:?} plain={:?} {}",
                s.name,
                &after[1..],
                &plain[k + 1..=k + AFTER],
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// A window's rows against the steps': `(position, distance over every value, distance over
    /// each tapped layer)`.
    type Rels = Vec<(usize, f64, Vec<f64>)>;

    /// The median, 90th percentile and largest of `v`.
    fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        (at(0.5), at(0.9), at(1.0))
    }

    /// The rows `ours` holds, from position `from`, against the steps' rows `theirs` (from
    /// position 0): per row its relative distance over all `w` values, and over each of the
    /// `layers` segments of `w / layers` values.
    fn row_rels(ours: &[f32], theirs: &[f32], w: usize, (from, layers): (usize, usize)) -> Rels {
        let seg = w / layers;
        let rows = ours.chunks_exact(w).zip(theirs[from * w..].chunks_exact(w));
        rows.enumerate()
            .map(|(i, (a, b))| {
                let per = (0..layers)
                    .map(|l| rel(&a[l * seg..(l + 1) * seg], &b[l * seg..(l + 1) * seg]))
                    .collect();
                (from + i, rel(a, b), per)
            })
            .collect()
    }

    /// Whether one layer's rows' distances `v` hold the bands: every one finite, then the median
    /// at most [`TAPS_MEDIAN_BAND`] and the 90th percentile at most [`TAPS_P90_BAND`]. A NaN tap
    /// is an infinite distance ([`rel`]), and the percentiles of a few rows skip the largest
    /// (eight rows: the median reads the fifth, the 90th percentile the seventh), so finiteness
    /// is its own condition.
    fn within_bands(v: Vec<f64>) -> bool {
        if v.iter().any(|r| !r.is_finite()) {
            return false;
        }
        let (med, p90, _) = spread(v);
        med <= TAPS_MEDIAN_BAND && p90 <= TAPS_P90_BAND
    }

    /// The band check held to what it refuses, on the host before the card loads: [`VERIFY_ROWS`]
    /// rows of two layers' taps at a distance of 1 % of their rows' size pass it in both layers,
    /// and the same rows with one tap of the last row NaN do not pass it in that tap's layer.
    fn within_bands_separate() -> Result<(), GateError> {
        const LAYERS_TAPPED: usize = 2;
        const SEG: usize = 4;
        let w = LAYERS_TAPPED * SEG;
        let theirs: Vec<f32> = (0..VERIFY_ROWS * w).map(|i| 1.0 + (i % 7) as f32).collect();
        let ours: Vec<f32> = theirs.iter().map(|v| v * 1.01).collect();
        let held = |ours: &[f32], layer: usize| {
            let rels = row_rels(ours, &theirs, w, (0, LAYERS_TAPPED));
            within_bands(rels.iter().map(|r| r.2[layer]).collect())
        };
        let clean = (0..LAYERS_TAPPED).all(|l| held(&ours, l));
        let mut nan = ours;
        nan[(VERIFY_ROWS - 1) * w + SEG] = f32::NAN;
        let refused = !held(&nan, 1) && held(&nan, 0);
        println!(
            "taps_band_separation rows={VERIFY_ROWS} clean_held={clean} nan_last_row_refused={refused} {}",
            verdict(clean && refused)
        );
        if clean && refused {
            Ok(())
        } else {
            Err(format!(
                "the taps' band check holds clean rows {clean} and refuses a NaN tap {refused}"
            )
            .into())
        }
    }

    /// Clause `taps_are_steps` on set `s`, taps on `layers`: the rows a step taps
    /// (`GpuModel::draft_taps` of 1 after each id of the prompt, then of the plain run's first
    /// [`VERIFY_ROWS`] ids) against the rows the rows pass hands its sink
    /// (`GpuModel::prompt_rows_tapped` of the last `window` positions, every one once and in
    /// order), then the rows a verify of those ids taps (`draft_taps` of [`VERIFY_ROWS`]). The
    /// first [`PASS`] positions of the prompt, a call of a pass's rows, are by bits; the window's
    /// rows, whose wide units are not the steps' bits, and the verify after it hold
    /// [`TAPS_MEDIAN_BAND`] and [`TAPS_P90_BAND`] a layer; the spread of the window's rows a layer
    /// and a [`TAPS_BLOCK`] of positions is printed.
    /// The taps are off again after it.
    fn taps_are_steps(
        m: &mut GpuModel<Body>,
        s: &Set,
        plain: &[u32],
        (layers, window): (&[usize], usize),
    ) -> Result<bool, GateError> {
        m.set_draft_taps(layers)?;
        let w = m.draft_tap_width();
        m.reset()?;
        let mut stepped = Vec::new();
        for &id in s.prompt.iter().chain(&plain[..VERIFY_ROWS]) {
            m.step(&[id])?;
            stepped.extend(m.draft_taps(1)?);
        }
        let tapped_call = |m: &mut GpuModel<Body>,
                           ids: &[u32],
                           window: usize|
         -> Result<(Vec<f32>, bool), GateError> {
            m.reset()?;
            let (p, from) = (ids.len(), ids.len() - window.min(ids.len()));
            let mut sunk = vec![f32::NAN; (p - from) * w];
            let (mut next, mut in_order) = (from, true);
            m.prompt_rows_tapped(ids, window, &mut |first: u32, rows: &[f32]| {
                let (first, n) = (first as usize, rows.len() / w);
                in_order &= first == next && first + n <= p && rows.len() == n * w;
                if in_order {
                    sunk[(first - from) * w..(first + n - from) * w].copy_from_slice(rows);
                }
                next = first + n;
                Ok::<_, GpuError>(())
            })?;
            in_order &= next == p;
            Ok((sunk, in_order))
        };
        let head = &s.prompt[..PASS.min(s.prompt.len())];
        let (sunk_head, head_in_order) = tapped_call(m, head, head.len())?;
        let head_bits = head_in_order && bits_equal(&sunk_head, &stepped[..head.len() * w]);
        let p = s.prompt.len();
        let from = p - window.min(p);
        let (sunk, in_order) = tapped_call(m, &s.prompt, window)?;
        m.step_rows::<VERIFY_ROWS>(<[u32; VERIFY_ROWS]>::try_from(&plain[..VERIFY_ROWS])?)?;
        let verified = m.draft_taps(VERIFY_ROWS)?;
        m.reset()?;
        m.set_draft_taps(&[])?;
        let rels = row_rels(&sunk, &stepped, w, (from, layers.len()));
        let prompt_ok =
            (0..layers.len()).all(|l| within_bands(rels.iter().map(|r| r.2[l]).collect()));
        let by_layer = (0..layers.len())
            .map(|l| {
                let (med, p90, max) = spread(rels.iter().map(|r| r.2[l]).collect());
                format!("L{}:{med:.1e}/{p90:.1e}/{max:.1e}", layers[l])
            })
            .collect::<Vec<_>>()
            .join(",");
        let mut by_block: Vec<(usize, Vec<f64>)> = Vec::new();
        for &(pos, r, _) in &rels {
            match by_block.last_mut() {
                Some((f, v)) if *f == pos - pos % TAPS_BLOCK => v.push(r),
                _ => by_block.push((pos - pos % TAPS_BLOCK, vec![r])),
            }
        }
        let by_block = by_block
            .into_iter()
            .map(|(f, v)| {
                let (med, p90, max) = spread(v);
                format!("{f}:{med:.1e}/{p90:.1e}/{max:.1e}")
            })
            .collect::<Vec<_>>()
            .join(",");
        let verify = row_rels(&verified, &stepped, w, (p, layers.len()));
        let verify_ok =
            (0..layers.len()).all(|l| within_bands(verify.iter().map(|r| r.2[l]).collect()));
        let (vmed, vp90, vmax) = spread(verify.iter().map(|r| r.1).collect());
        let pass = head_bits && in_order && prompt_ok && verify_ok;
        println!(
            "taps_are_steps set={} layers={layers:?} width={w} prompt_rows={p} window={} \
             in_order={in_order} head_same_bits={head_bits} \
             bands=med<={TAPS_MEDIAN_BAND:.1e},p90<={TAPS_P90_BAND:.1e} \
             by_layer(med/p90/max)=[{by_layer}] by_{TAPS_BLOCK}_positions=[{by_block}] \
             verify_rows(med/p90/max)={vmed:.1e}/{vp90:.1e}/{vmax:.1e} {}",
            s.name,
            p - from,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The clauses over the verifying load ([`GpuModel::open_verifying`], the one a seat with a
    /// draft opens), as a graph-mode session, on every set: `prompt_rows_is_steps`,
    /// `verify_is_steps` and `commit_keeps_steps` in eager and in graph mode, `taps_are_steps`,
    /// then `speculative_is_plain` with the draft beside it.
    fn verifying(sets: &[Set]) -> Result<bool, GateError> {
        let file = open_split(Arch::Raon1, "gate-gpu-raon1-e2e")?;
        let mut m = GpuModel::<Body>::open_verifying(Gpu::new()?, file, CTX)?;
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        let mut seat = Seat::from_model(m, u32::try_from(CTX)?);
        let (path, layers) = draft_file()?;
        let mut ok = true;
        for s in sets {
            ok &= prompt_rows_is_steps(seat.model_mut(), s)?;
            for (mode, word) in [(StepMode::Eager, "eager"), (StepMode::Graph, "graph")] {
                seat.model_mut().set_mode(mode);
                ok &= verify_is_steps(&mut seat, s, word)?;
            }
            let (plain, _) = greedy(&mut seat, &s.prompt, VERIFY_ROWS, false)?;
            ok &= taps_are_steps(seat.model_mut(), s, &plain, (&layers, s.prompt.len()))?;
            Target::reset(&mut seat)?;
        }
        for name in WIDE_SETS.iter().filter(|&&n| !SETS.contains(&n)) {
            let s = open_set(name)?;
            let (plain, _) = greedy(&mut seat, &s.prompt, VERIFY_ROWS, false)?;
            ok &= taps_are_steps(seat.model_mut(), &s, &plain, (&layers, TAPS_WINDOW))?;
            Target::reset(&mut seat)?;
        }
        Ok(speculative_is_plain(seat, &path, sets)? && ok)
    }

    /// Clause `speculative_is_plain` on every set, over `seat` with the draft of the file at
    /// `path` beside it.
    fn speculative_is_plain(mut seat: Seat, path: &Path, sets: &[Set]) -> Result<bool, GateError> {
        let draft = open_draft(&seat, path)?;
        let mut spec = seat.with_draft::<RaonDraft, VERIFY_ROWS>(draft, &mut Quiet)?;
        let mut ok = true;
        for s in sets {
            let (plain, _) = greedy(&mut seat, &s.prompt, DRAFTED, false)?;
            let (ids, passes, kept) = drafted(&mut spec, &mut seat, &s.prompt, DRAFTED)?;
            let differ = plain.iter().zip(&ids).position(|(a, b)| a != b);
            let pass = differ.is_none() && plain.len() == ids.len();
            let at = match differ {
                None => "none".to_owned(),
                Some(i) => format!("step {i} plain={} drafted={}", plain[i], ids[i]),
            };
            println!(
                "speculative_is_plain set={} ids={DRAFTED} passes={passes} kept_proposed={kept} \
                 first_difference={at} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        within_bands_separate()?;
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set(n))
            .collect::<Result<_, _>>()?;
        for s in &sets {
            if s.greedy.ids.is_empty() || s.greedy.margins.len() != s.greedy.ids.len() {
                return Err(format!(
                    "{}: a greedy table of {} ids and {} margins",
                    s.name,
                    s.greedy.ids.len(),
                    s.greedy.margins.len()
                )
                .into());
            }
            let need =
                (s.prompt.len() + s.greedy.ids.len()).max(s.prompt.len() + DRAFTED + VERIFY_ROWS);
            if need > CTX {
                return Err(
                    format!("{}: {need} positions, the gate's caches hold {CTX}", s.name).into(),
                );
            }
        }
        let file = open_split(Arch::Raon1, "gate-gpu-raon1-e2e")?;
        let t = std::time::Instant::now();
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, CTX)?;
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        if m.layers().len() != LAYERS {
            return Err(format!(
                "a file of {} layers; the sets' is {LAYERS}",
                m.layers().len()
            )
            .into());
        }
        let mut ok = true;
        {
            let body = m.body("gate_raon1_e2e")?;
            let wide_ok = body.wide_ok();
            println!(
                "wide_runs_the_file wide_ok={wide_ok} refusal={:?} {}",
                body.wide_off(),
                verdict(wide_ok)
            );
            ok &= wide_ok;
        }
        for s in &sets {
            m.reset()?;
            let first = m.step(&s.prompt)?;
            let row = m.logits()?;
            let theirs = s
                .top
                .first()
                .ok_or_else(|| format!("{}: an empty top-k table", s.name))?;
            let ours5 = top_ids(&row, TOP_K);
            let d = s
                .top
                .iter()
                .map(|t| (row[t.id as usize] - t.logit).abs())
                .fold(0.0f32, f32::max);
            let logits_rel = rel(&row, &s.logits);
            let logits_ok = logits_rel <= LOGITS_REL_BAND;
            print_route(&m, s)?;
            let overlap = s
                .top
                .iter()
                .take(TOP_K)
                .filter(|t| ours5.contains(&(t.id as usize)))
                .count();
            let top1 = first == theirs.id;
            println!(
                "top1_is_vendors set={} prompt_ids={} ours={first} vendor={} top{TOP_K}_overlap={overlap}/{TOP_K} \
                 ours_top{TOP_K}={ours5:?} d={d} over {} ids {}",
                s.name,
                s.prompt.len(),
                theirs.id,
                s.top.len(),
                verdict(top1)
            );
            let (ids, diverge) = continue_greedy(&mut m, s, first, row)?;
            let (pass, at) = match diverge {
                None => (true, "none".to_owned()),
                Some((i, our_gap)) => {
                    let margin = s.greedy.margins[i];
                    let tie = margin <= 2.0 * d;
                    let word = if tie { "a named tie" } else { "past 2d" };
                    let at = format!(
                        "step {i} ours={} vendor={} vendor_margin={margin} our_gap={our_gap} \
                         2d={} ({word})",
                        ids[i],
                        s.greedy.ids[i],
                        2.0 * d
                    );
                    (tie, at)
                }
            };
            println!(
                "greedy_is_vendors set={} steps={} first_divergence={at} {}",
                s.name,
                ids.len(),
                verdict(pass)
            );
            println!("greedy set={} ours={ids:?}", s.name);
            println!(
                "logits_are_vendors set={} logits_rel={logits_rel:.4e} band={LOGITS_REL_BAND:.1e} {}",
                s.name,
                verdict(logits_ok)
            );
            ok &= top1 && pass && logits_ok;
        }
        for name in WIDE_SETS {
            let s = open_set(name)?;
            ok &= prompt_wide_is_vendors(&mut m, &s)?;
            if s.prompt.len() >= SPLIT_FROM {
                ok &= prompt_wide_is_partitions(&mut m, &s)?;
            }
        }
        // The plain load's card bytes go back before the verifying load takes its own.
        drop(m);
        ok &= verifying(&sets)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_raon1_e2e on every set of {SETS:?} the argmax after the prompt is the \
             vendor's, the greedy continuation is the vendor's id for id, or up to a named tie, and \
             the drafted greedy run is the plain one id for id"
        );
        Ok(())
    }
}
