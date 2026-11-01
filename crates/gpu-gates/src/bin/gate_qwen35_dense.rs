//! The dense Qwen3.8-27B generation gate: the `qwen35` file (64 layers, 48 gated-delta and 16 gated GQA at 24/4 heads of
//! 256, each with a dense SwiGLU FFN of 17,408; the 15 `blk.64.*` next-token tensors unused) whole on one card, the
//! Body35 chain a request runs: the prompt call, the decode step, the pass, the checkpoints and the resident slots,
//! against llama.cpp mainline's forced-logits rows (`refset::arch::qwen35::dense`, `just dump-logits-qwen38d`). The
//! file fits either card. `gate_clef_hidden` holds the trunk's hidden-state band on the same geometry; this gate does
//! not repeat it.
//!
//! The clauses (a) to (f) are the shared dense-gate module's (`shared/dense_e2e.rs`, the same code for every dense
//! Body35 file); this bin is the Qwen3.8-27B's row of it: the file, its sets, its header's geometry and the bands below.
//!
//! The fixture tier's model is the NVFP4 cut of `nvidia/Qwen3.8-27B-NVFP4` (`tools/fixture-cuts/qwen38nv.json`: layers 0 to
//! 3, a folder in its stored bits) and its GGUF twin, which mainline's converter makes of the cut (`tools/ref/models/
//! qwen38nv.sh`); the twin is the file mainline reads, as it reads no folder and ik reads no NVFP4. Two fixture-oracle
//! clauses (module `nv`) hold that route: G3, ours on the twin against mainline's forced-logits rows, within bands that are
//! multiples of the oracle's own CPU floor; G4, ours on the folder equal bit for bit to ours on the twin, which is the
//! folder map's bytes held to the converter's. Clauses (a) to (f) run on the real file in the real tier only (the
//! family's rotation schedules them); the fixture tier holds no run of them on any file.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35_dense", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen35_open.rs"]
#[allow(
    dead_code,
    reason = "the loader serves the hidden, media and dense gates; this one reads the load and the whole-fit verdict"
)]
mod qwen35_open;

#[cfg(feature = "gpu")]
#[path = "shared/q35_clauses.rs"]
mod q35_clauses;

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/folder.rs"]
#[allow(
    dead_code,
    reason = "the gate opens a dense Qwen3.8 folder alone: the checked open is what it reads"
)]
mod folder;

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/qwen3_folder.rs"]
#[allow(
    dead_code,
    reason = "the gate takes the model at a path, not the seat's open"
)]
mod qwen3_folder;

#[cfg(feature = "gpu")]
#[path = "shared/dense_e2e.rs"]
mod dense_e2e;

#[cfg(feature = "gpu")]
mod gate {
    use super::dense_e2e::{self, Bands, Row};
    use super::q35_clauses::Geometry35;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::tier::Tier;
    use refset::arch::qwen35::dense::{MODEL, P64, P4096, Qwen38d};

    /// The real file's literals, beside the geometry its header gives (the move proof, `Geometry35::of`).
    const REAL: Geometry35 = Geometry35 {
        n_attn: 16,
        n_delta: 48,
        n_kv: 4,
        head: 256,
        n_v: 48,
        head_v: 128,
        conv_c: 10240,
    };
    /// The `blk.<n>.` prefix of the next-token block the Body35 reader leaves unused (`qwen35.nextn_predict_layers`
    /// is 1, after the 64 trunk layers).
    const UNUSED_BLOCK: &str = "blk.64.";

    /// PIN(2026-10-19): the bound on one row's relative L2 distance from mainline's CUDA row. Derivation: this engine
    /// and mainline read the same Q4_K/Q6_K codes and differ in the 8-bit activations (ours q8_1 per 128 values,
    /// mainline's MMQ q8_1 per 32) and the f32 sum order, as mainline's CPU twin does (q8_K per 256); the twin's rows
    /// against its CUDA rows are the oracle's own floor (`dense-logits.py floor`): per-row relative L2 at most
    /// 0.1492 (64-id set) and 0.1008 (4,096-id set). 0.20 is 1.34 times the largest. A wiring fault (a layer, a
    /// position or a forced id off by one) moves a row by order one.
    const REL_BAND: f64 = 0.20;
    /// PIN(2026-10-19): the bound on a set's median row. The floor's medians are 0.0445 and 0.0525; 0.066 is 1.25
    /// times the largest, the factor `gate_clef_hidden`'s median band takes over its twin.
    const MEDIAN_BAND: f64 = 0.066;
    /// PIN(2026-10-19): the bound on a set's 90th percentile. The floor's are 0.0643 and 0.0685; 0.09 is 1.3 times the
    /// largest, as `gate_clef_hidden`'s p90 band.
    const P90_BAND: f64 = 0.09;
    /// PIN(2026-10-19): the factor on a row's stability bound. Two logits swap only if their margin is within the
    /// difference of their two moves, at most `min(2 · max_abs, √2 · l2)` of the move's vector
    /// (`dense-logits.py stable_margin`); the move taken is the oracle's own spread on that row (twin against CUDA,
    /// the same positions amplify for any two realizations) times 1.3, the factor the p90 band takes.
    pub(super) const STABLE_K: f64 = 1.3;
    /// PIN(2026-10-19): the rows at or under their stability bound (the only rows that could flip), per set: 20 of 32
    /// and 17 of 32 at [`STABLE_K`] (28 of 64 at a factor of 1, the floor's report). A count that moves means the
    /// sets or the twin moved, not the engine.
    const OTHER_ROWS: [usize; 2] = [20, 17];

    /// The Qwen3.8-27B's row: the scalar flash (the file's group of 6 runs the pairs' pass, which has no tensor-core
    /// form).
    const ROW: Row = Row {
        gate: "gate_qwen35_dense",
        model: MODEL,
        sets: [P64, P4096],
        real: REAL,
        unused_block: Some(UNUSED_BLOCK),
        mma: false,
        bands: Bands {
            rel: REL_BAND,
            median: MEDIAN_BAND,
            p90: P90_BAND,
            stable_k: STABLE_K,
            other_rows: OTHER_ROWS,
        },
    };

    /// The real tier runs the real file's clauses and leaves the cut's to the fixture tier by name; the fixture tier runs
    /// the cut's.
    pub fn run() -> Result<(), GateError> {
        match Tier::from_env()? {
            Tier::Real => {
                dense_e2e::run::<Qwen38d>(&ROW)?;
                super::nv::declare()
            }
            Tier::Fixture => super::nv::run(),
        }
    }
}

#[cfg(feature = "gpu")]
mod nv {
    use super::dense_e2e::{self, Bands, norms, rows_of, twin_of};
    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::qwen3moe::{KvQ8, Open35, Qwen35moeModel};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, quantile, verdict};
    use bloomery_levers::{CARD_BUDGET, CardBudget};
    use refset::arch::qwen35::dense::{NV_P64, NV_P4096, NV_PROFILE, NV_TWIN, NvForcedLogits};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// The gate's name: its first line and the `what` of its refusals.
    const GATE: &str = "gate_qwen35_dense";
    /// The fixture-oracle clauses this gate declares, in order.
    const G3: &str = "g3_twin_against_mainline";
    const G4: &str = "g4_folder_is_twin";
    const DECLARED: usize = 2;

    /// The cache both clauses load with: the sets' context (a 4,096-id prompt and 31 forced steps fit) at the real
    /// file's gate ubatch, f16 K/V, the scalar flash (the model's group of 6 query heads runs the pairs' pass).
    const CTX: usize = 4352;
    const U_GATE: usize = 1024;

    /// Each band of G3 composes two terms in quadrature, the rows' error model:
    /// - the activation rule, which ours carries by design and the oracle's floor does not hold: ours' decode gemvs
    ///   quantize an NVFP4 weight's activations to q8 per 128 (f32 d) and read a Q8_0 weight's in f32, where mainline's
    ///   CUDA and CPU kernels alike quantize them to q8 per 32 (f16 d). Its size is `ACT_*` below.
    /// - the f32 sum order, as the oracle's own floor on the twin measures it: mainline's CPU kernels against its CUDA
    ///   kernels on the same file, prompt and ids (`dense-logits.py floor`, the rows `<set>.cpu` beside each set), pooled
    ///   over both sets as the real file's bands are, times the factors the real file's `PIN` bands take over its own floor
    ///   (0.20 over the largest row 0.1492, 0.066 over the larger median 0.0525, 0.09 over the larger p90 0.0685): 1.34 on
    ///   the largest relative L2, 1.25 on the median, 1.3 on the 90th percentile.
    ///
    /// A wiring fault (a dropped scale, a stale activation block, a layer or id off by one) moves a row by order one, far
    /// past both.
    const REL_FACTOR: f64 = 1.34;
    const MEDIAN_FACTOR: f64 = 1.25;
    const P90_FACTOR: f64 = 1.3;
    /// PIN(2026-10-29): the activation rule's term in a decode row's relative L2 (median, 90th percentile, largest row).
    /// Derived on mainline's own CPU activations of the p64 set's rows, each site's error vector being its weight times the
    /// difference of the two rules on the same input: the head (NVFP4, 0.0047 median alone), the FFN's gate/up/down, the
    /// mixers' output projections and their z and attention gates, carried to the logits through the final norm's Jacobian
    /// and the head, 0.0049 / 0.0053 / 0.0058 together; and the mixers' q/k/v inputs (the delta nets' qkv, the attention
    /// layer's q, k and v), taken isotropic and carried to the layer's output at gain one (the free-running composition
    /// of the house's error model), 0.0096 / 0.0111 / 0.0120. In quadrature: 0.0109 / 0.0121 / 0.0130. Row 0, the
    /// prompt call's, holds the head's term alone.
    const ACT_MEDIAN: f64 = 0.0109;
    const ACT_P90: f64 = 0.0121;
    const ACT_ROW: f64 = 0.0130;
    /// PIN(2026-10-29): the twin's rows at or under their stability bound (the real file's factor, [`super::gate::STABLE_K`]),
    /// per set: 1 and 2 of 32, read from G3's own lines `ref_qwen38nv_logits_p64: ... 1 rows at or under their stability
    /// bound` and `ref_qwen38nv_logits_p4096: ... 2 rows at or under their stability bound`. The count reads the sets and the
    /// CPU floor alone, so a count that moves means the sets or the twin moved, not the engine.
    const OTHER_ROWS: [usize; 2] = [1, 2];

    fn options() -> Open35 {
        Open35 {
            ctx: CTX,
            mma: false,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        }
    }

    /// The real tier leaves both clauses to the fixture tier, each a [`tier::FIXTURE_ONLY`] line the batch counts.
    pub fn declare() -> Result<(), GateError> {
        for clause in [G3, G4] {
            tier::run_clause(clause, Tag::FixtureOracle)?;
        }
        tier::expect_fixture_oracle(GATE, DECLARED)?;
        println!("{}", tier::tally_line());
        Ok(())
    }

    /// The twin through ours, then the folder through ours, one model on the card at a time. Each side's weights are
    /// held to the card budget the cut records (`BLOOMERY_CARD_BUDGET`, which the fixture tier exports): a dense cut
    /// plans whole on one card, so its weights fit under that budget.
    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let CardBudget::Every(budget) = levers.card_budget() else {
            return Err(format!(
                "{GATE}: {CARD_BUDGET} is {:?}; the fixture tier sets it to the one byte count the cut's header records",
                levers.card_budget()
            )
            .into());
        };
        // A profile's cut is read only by the clauses that name it (tools/fixture-dirs.tsv): the folder is the table's
        // cut of NV_PROFILE, not the model path the tier resolves by architecture (qwen35 has no fixture of its own).
        let folder = tier::cut(NV_PROFILE)?;
        println!("{GATE} (fixture tier): the cut {}", folder.display());
        let twin = Path::new(NV_TWIN);
        if !twin.is_file() {
            return Err(format!(
                "{GATE}: the cut's GGUF twin {NV_TWIN} is not there (the command is in tools/ref/models/qwen38nv.sh)"
            )
            .into());
        }
        // `--only g3|g4` runs one clause (the FAIL-first mutants' loop); no flag runs both.
        let only = std::env::args().skip_while(|a| a != "--only").nth(1);
        if let Some(o) = &only
            && !["g3", "g4"].contains(&o.as_str())
        {
            return Err(format!("--only {o}: a clause is one of g3, g4").into());
        }
        let want = |c: &str| only.as_deref().is_none_or(|o| o == c);
        let sets = [(NV_P64, 64usize), (NV_P4096, 4096usize)];
        let mut ok = true;
        let mut twin_run: Option<Side> = None;
        if want("g3") && tier::run_clause(G3, Tag::FixtureOracle)? {
            let side = run_side(twin, load_twin)?;
            within_budget("the twin", &side, budget)?;
            ok &= g3(&side, sets)?;
            twin_run = Some(side);
        }
        if want("g4") && tier::run_clause(G4, Tag::FixtureOracle)? {
            let twin_side = match twin_run {
                Some(side) => side,
                None => run_side(twin, load_twin)?,
            };
            let folder_side = run_side(&folder, load_folder)?;
            within_budget("the folder", &folder_side, budget)?;
            ok &= g4(&folder_side, &twin_side, sets)?;
        }
        if only.is_none() {
            tier::expect_fixture_oracle(GATE, DECLARED)?;
        }
        println!("{}", tier::tally_line());
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// One model's run of both sets: every row of each, and the weights the load holds.
    struct Side {
        rows: Vec<Vec<Vec<f32>>>,
        weights: BTreeMap<String, usize>,
    }

    fn load_twin(path: &Path) -> Result<Qwen35moeModel, GateError> {
        super::qwen35_open::open_options(path, options(), &|_| {}).map(|(m, _)| m)
    }

    /// The checkpoint folder at `path` on the card `open_options` would take, through the one folder open
    /// (`serve_seats/qwen3_folder.rs`). The cut fits any card the twin fits, so no whole-fit probe runs first.
    fn load_folder(path: &Path) -> Result<Qwen35moeModel, GateError> {
        let model = super::qwen3_folder::model_at(path)?;
        let census = bloomery_gpu_gates::gpu_census::census()?;
        let card = Place::A.on(&census)?.card_specs()?[0];
        let gpu = Gpu::open_card(card.name, card.device)?;
        Ok(Qwen35moeModel::open(gpu, model, options())?)
    }

    /// `path` loaded by `load` on the card and run over both sets' prompts and forced ids from a `reset`; the model is gone
    /// when this returns.
    fn run_side(
        path: &Path,
        load: fn(&Path) -> Result<Qwen35moeModel, GateError>,
    ) -> Result<Side, GateError> {
        let mut m = load(path)?;
        m.set_mode(StepMode::Graph);
        let mut rows = Vec::new();
        for name in [NV_P64, NV_P4096] {
            let set = NvForcedLogits::open(&data_dir().join(name))?;
            rows.push(rows_of(&mut m, &set)?);
        }
        let w = m.weights();
        let weights = w
            .names()
            .filter(|n| !n.starts_with("derived."))
            .filter_map(|n| w.get(n).map(|d| (n.to_string(), d.resident_bytes())))
            .collect();
        Ok(Side { rows, weights })
    }

    /// `side`'s resident weight bytes against the cut's card budget: over it is a named error, since the clauses would
    /// then compare a load the budget does not describe.
    fn within_budget(what: &str, side: &Side, budget: u64) -> Result<(), GateError> {
        let bytes: u64 = side.weights.values().map(|&b| b as u64).sum();
        println!("  {what}: {bytes} B of weights resident, card budget {budget} B");
        if bytes > budget {
            return Err(format!(
                "{GATE}: {what} holds {bytes} B of weights on the card, over the cut's card budget of {budget} B"
            )
            .into());
        }
        Ok(())
    }

    /// The median, 90th percentile and largest of `v`.
    fn spread(v: &[f64]) -> (f64, f64, f64) {
        (
            quantile(v, 0.5),
            quantile(v, 0.9),
            v.iter().copied().fold(0.0, f64::max),
        )
    }

    /// G3: ours on the twin against mainline's CUDA rows of each set, by the per-row rule the real file's clause holds
    /// (`dense_e2e::oracle`) within bands that the floor sets here: each row's relative L2 within the largest-row band, the
    /// median and the 90th percentile within theirs, the argmax equal on every row whose reference top-2 margin exceeds its
    /// stability bound, and the rows at or under it counted against [`OTHER_ROWS`].
    fn g3(side: &Side, sets: [(&str, usize); 2]) -> Result<bool, GateError> {
        println!("(G3) the twin through ours against mainline's forced logits rows");
        let mut opened = Vec::new();
        let mut floor_spread = (0.0f64, 0.0f64, 0.0f64);
        for (name, prompt) in sets {
            let set = NvForcedLogits::open(&data_dir().join(name))?;
            let cpu = twin_of(name, prompt, &set)?;
            let floor: Vec<f64> = (0..set.rows())
                .map(|i| {
                    let (l2, _, norm) = norms(cpu.row(i), set.row(i));
                    l2 / norm
                })
                .collect();
            let (median, p90, largest) = spread(&floor);
            println!(
                "  {name}: the oracle's floor (CPU against CUDA) rel L2 median {median:.4}, p90 {p90:.4}, largest {largest:.4}"
            );
            floor_spread = (
                floor_spread.0.max(median),
                floor_spread.1.max(p90),
                floor_spread.2.max(largest),
            );
            opened.push((name, prompt, set, cpu));
        }
        let bands = Bands {
            rel: ACT_ROW.hypot(REL_FACTOR * floor_spread.2),
            median: ACT_MEDIAN.hypot(MEDIAN_FACTOR * floor_spread.0),
            p90: ACT_P90.hypot(P90_FACTOR * floor_spread.1),
            stable_k: super::gate::STABLE_K,
            other_rows: OTHER_ROWS,
        };
        println!(
            "  bands (the activation rule's term in quadrature with the floor's): median {:.4} ({ACT_MEDIAN} with \
             {MEDIAN_FACTOR} x the larger median), p90 {:.4} ({ACT_P90} with {P90_FACTOR} x the larger p90), row \
             {:.4} ({ACT_ROW} with {REL_FACTOR} x the largest row)",
            bands.median, bands.p90, bands.rel
        );
        let mut ok = true;
        for (i, ((name, prompt, set, cpu), ours)) in opened.iter().zip(&side.rows).enumerate() {
            ok &= dense_e2e::oracle(ours, bands, (*name, *prompt), (set, cpu), i)?;
        }
        Ok(ok)
    }

    /// Where two runs of the same rows differ, bit for bit. A NaN in either reads as a difference unless both bit patterns
    /// are equal.
    struct Diff {
        rows: usize,
        lanes: usize,
        /// The first differing `(row, lane)`.
        first: Option<(usize, usize)>,
        /// The largest `|a - b|` and `‖a - b‖₂ / ‖b‖₂`.
        max: f64,
        rel: f64,
    }

    fn differ(a: &[Vec<f32>], b: &[Vec<f32>]) -> Diff {
        let mut d = Diff {
            rows: 0,
            lanes: 0,
            first: None,
            max: 0.0,
            rel: 0.0,
        };
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let before = d.lanes;
            for (j, (&u, &v)) in x.iter().zip(y).enumerate() {
                if u.to_bits() != v.to_bits() {
                    d.lanes += 1;
                    d.first.get_or_insert((i, j));
                    let delta = f64::from(u) - f64::from(v);
                    num += delta * delta;
                    d.max = d.max.max(delta.abs());
                }
                den += f64::from(v).powi(2);
            }
            d.rows += usize::from(d.lanes > before);
        }
        d.rel = (num / den).sqrt();
        d
    }

    /// G4: ours on the folder equal bit for bit to ours on the twin, every row of both sets. The weights each load holds are
    /// printed beside it (names and resident bytes), the first place a difference in the map's bytes would show.
    fn g4(folder: &Side, twin: &Side, sets: [(&str, usize); 2]) -> Result<bool, GateError> {
        println!("(G4) the cut folder through ours equals the twin through ours, bit for bit");
        let (a, b) = (&folder.weights, &twin.weights);
        let only_folder: Vec<&String> = a.keys().filter(|k| !b.contains_key(*k)).collect();
        let only_twin: Vec<&String> = b.keys().filter(|k| !a.contains_key(*k)).collect();
        let bytes: Vec<String> = a
            .iter()
            .filter_map(|(k, &n)| {
                b.get(k)
                    .filter(|&&m| m != n)
                    .map(|m| format!("{k} {n} vs {m}"))
            })
            .collect();
        let same_weights = only_folder.is_empty() && only_twin.is_empty() && bytes.is_empty();
        println!(
            "  weights: folder {} tensors {} B, twin {} tensors {} B; {}",
            a.len(),
            a.values().sum::<usize>(),
            b.len(),
            b.values().sum::<usize>(),
            if same_weights {
                "the same names and bytes".to_string()
            } else {
                format!(
                    "only in the folder {:?}, only in the twin {:?}, other bytes {:?}",
                    &only_folder[..only_folder.len().min(8)],
                    &only_twin[..only_twin.len().min(8)],
                    &bytes[..bytes.len().min(8)]
                )
            }
        );
        let mut ok = true;
        for (i, (name, prompt)) in sets.iter().enumerate() {
            let d = differ(&folder.rows[i], &twin.rows[i]);
            let pass = d.lanes == 0 && folder.rows[i].len() == twin.rows[i].len();
            let at = d.first.map_or(String::new(), |(r, l)| {
                let (f, t) = (folder.rows[i][r][l], twin.rows[i][r][l]);
                format!("; first at row {r} lane {l}: folder {f} vs twin {t}")
            });
            println!(
                "  {name}: {} rows after {prompt} ids; {} rows and {} lanes differ, largest |d| {:.3e}, rel L2 {:.3e}{at} {}",
                folder.rows[i].len(),
                d.rows,
                d.lanes,
                d.max,
                d.rel,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }
}
