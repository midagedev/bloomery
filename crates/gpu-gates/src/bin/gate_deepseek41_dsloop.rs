//! GPU gate for the DSpark loop's target side: the feature tap
//! (`body::attach_features`, `Body::read_features`) on the gate placement
//! (`crate::ds41_tier::plan_gate`), its layers the draft file's `target_layers`
//! (`$BLOOMERY_DSPARK_MODEL`, header only: no draft is loaded here).
//!
//! - `--structure`: the step and the pair pass captured with the tap hold
//!   exactly the nodes they hold without it plus one kernel per tapped layer
//!   and row; those counts are the step's node count predicted from the file
//!   (`shared/ds41_nodes.rs`, the step gate's own table) plus the taps — twice the plain
//!   step's for the pair — where they were two literals of the real file (1182 and 2364);
//!   in the step's capture each `ds41_hc_mean` depends on
//!   the MoE join (`ds41_ffn_post*`) of the layer before its tapped layer
//!   alone. Without the tap the counts are the plain step's (the step gate
//!   pins those).
//! - `--tap`: from a reset, prompt row 0 then greedy steps through the graph
//!   to [`RUN`] positions, each position's features read after its step.
//!   Then at each of [`CHECKS`]: the positions `p` and `p + 1` stepped
//!   eagerly with every sub-layer's streams read back (the finite probe's
//!   `observed_step`), and the mean of the four streams each tapped layer
//!   reads computed on the host, `((s0 + s1) + s2) + s3` then `× 0.25`;
//!   against it, bit for bit: the eager step's features, the graph run's,
//!   and both rows of a pair pass over the same two tokens replayed after a
//!   rollback. A read of row 1 after a one-row step is refused.
//!
//! Every clause is self-consistency (the engine against its own prediction and its own eager
//! step), so the fixture tier runs both arms; the prompt's ids and the greedy steps are inputs.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_dsloop: built without the `deepseek41` feature; see `just gate-gpu-ds41-dspark-loop`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_dsloop", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_finite.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the probe's streams and token; its non-finite report serves the other bins"
)]
mod finite;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the draft's header only; the loop half serves generate_ds41"
)]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_nodes.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the predicted node count; the step gate prints the table of kinds"
)]
mod nodes;

#[cfg(feature = "deepseek41")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the card, the path and the clause tags; the triangle's facts serve the prefill gate"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model, PAIR_ROWS, Seam};
    use bloomery_gpu_gates::nodes::{Captured, StepNode, capture_order};
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8,
    };
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::workstation;

    use crate::{dspark, finite};

    const NAME: &str = "gate_deepseek41_dsloop";
    /// The step captured with the tl37 draft's tap on the real file, as a literal: the plain
    /// step's 1179 nodes (1169 before the candidate mask's 2 + 4 × 2 = 10 launches, 1167 before the
    /// engram rows' wait and copy nodes) and one `ds41_hc_mean` per tapped layer. The gate's pin
    /// is the value derived from the file ([`structure`]); the real tier prints it beside this.
    const REAL_TAP_STEP_NODES: usize = 1182;
    /// The pair pass with the same tap on the real file, as a literal: twice the plain step's
    /// nodes (each row runs its own ten candidate launches) and one `ds41_hc_mean` per tapped
    /// layer and row.
    const REAL_TAP_PAIR_NODES: usize = 2364;
    /// Positions the greedy run stands at before the checks.
    const RUN: u32 = 144;
    /// The positions checked: even, so a rollback to them is granted
    /// (`Body::keep_point` with a ratio-2 stream); one before the window
    /// ring wraps and one after.
    const CHECKS: [u32; 2] = [140, 4];

    struct Args {
        structure: bool,
        tap: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_dsloop [--structure] [--tap]";
        let mut a = Args {
            structure: false,
            tap: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--structure" => a.structure = true,
                "--tap" => a.tap = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.structure || a.tap) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

    /// Prompt row 0's ids (`$BLOOMERY_DATA/greedy-ds41/prompt0.tsv`, written
    /// by `just ik-greedy-ds41`).
    fn prompt0() -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("greedy-ds41").join("prompt0.tsv");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e} — run just ik-greedy-ds41 0", path.display()))?;
        let row = text
            .lines()
            .find(|l| !l.starts_with('#') && !l.is_empty())
            .ok_or_else(|| format!("{}: no prompt row", path.display()))?;
        let ids = row
            .split('\t')
            .nth(2)
            .ok_or_else(|| format!("{}: the row has no ids", path.display()))?;
        Ok(ids
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let args = parse_args()?;
        crate::ds41_tier::init()?;
        let mut cfg = body::OpenCfg::from_levers(&levers)?;
        let (_, dhp) = dspark::draft_hparams()?;
        let layers = dhp.target_layers.clone();
        let path = crate::ds41_tier::model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&split)?;
        // The draft is read for its header only, so the plan reserves nothing for it.
        cfg.place = tier::plan_levers(&split, &levers, 0)?;
        premises(&hp, &layers)?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let mut m = body::open(
            file,
            crate::ds41_tier::plan_gate,
            usize::try_from(workstation::CTX_MAX)?,
            &cfg,
        )?;
        m.set_mode(StepMode::Graph);
        let ids = prompt0()?;
        println!(
            "{NAME}: tap layers {layers:?} (the draft's target_layers), {} values each; prompt 0 \
             {ids:?}",
            hp.n_embd
        );
        let mut pass = true;
        let plain = if args.structure {
            Some(plain_counts(&mut m, ids[0], &split, &hp)?)
        } else {
            None
        };
        m.set_mode(StepMode::Eager);
        body::attach_features(&mut m, &layers)?;
        m.set_mode(StepMode::Graph);
        if let Some(plain) = plain {
            crate::ds41_tier::sc(
                "--structure: the tapped step and pair are the plain nodes plus the taps",
            )?;
            pass &= structure(&mut m, &layers, ids[0], plain)?;
        }
        if args.tap {
            crate::ds41_tier::sc("--tap: the features are the host mean, eager, graph and pair")?;
            pass &= tap(&mut m, &hp, &layers, &ids)?;
        }
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// The run's own shape against the file: the two checked positions straddle the window ring's
    /// wrap (the earlier one before it wraps, both pairs inside the run) and the run stands past
    /// the later one; each is the real file's literal at its window of 128 and is printed beside
    /// the header's.
    fn premises(hp: &Hparams, layers: &[usize]) -> Result<(), GateError> {
        let [late, early] = CHECKS;
        let (late, early, run) = (late as usize, early as usize, RUN as usize);
        if !(early + 1 < hp.window && hp.window < late && late + 2 < run) {
            return Err(format!(
                "{NAME}: the checks at {early} and {late} of {run} positions do not straddle the \
                 window of {} rows",
                hp.window
            )
            .into());
        }
        if layers.iter().any(|&l| l >= hp.n_layer || l == 0) {
            return Err(format!(
                "{NAME}: the draft's target layers {layers:?} are not inside the file's {} \
                 layers (the tap reads the MoE join before each, so none is layer 0)",
                hp.n_layer
            )
            .into());
        }
        Ok(())
    }

    /// The step's and the pair's node counts without a tap, and the step's node count predicted
    /// from the file ([`crate::nodes::predicted`]) before the tap is attached.
    fn plain_counts(
        m: &mut Deepseek41Model,
        t: u32,
        split: &Split,
        hp: &Hparams,
    ) -> Result<[usize; 3], GateError> {
        let predicted = {
            let (_, _, body) = m.body_parts(NAME)?;
            crate::nodes::predicted(split, hp, body)?.nodes()
        };
        let step = m.capture_step()?;
        m.step_rows([t, t])?;
        let pair = m.rows_graph_nodes::<PAIR_ROWS>()?.len();
        m.reset()?;
        println!(
            "{NAME}: structure | no tap: step {step} nodes (predicted {predicted}), pair {pair}"
        );
        Ok([step, pair, predicted])
    }

    /// `--structure` (module doc), with the tap attached.
    fn structure(
        m: &mut Deepseek41Model,
        layers: &[usize],
        t: u32,
        [step0, pair0, predicted]: [usize; 3],
    ) -> Result<bool, GateError> {
        let n = layers.len();
        let step = m.capture_step()?;
        m.step_rows([t, t])?;
        let pair = m.rows_graph_nodes::<PAIR_ROWS>()?.len();
        m.reset()?;
        // The pins the real file held as literals: the plain step's predicted nodes and one
        // `ds41_hc_mean` a tapped layer, twice that and one a layer and row for the pair.
        let (pin_step, pin_pair) = (predicted + n, 2 * predicted + 2 * n);
        let moved = tier::witness("tap step nodes", pin_step, REAL_TAP_STEP_NODES)
            & tier::witness("tap pair nodes", pin_pair, REAL_TAP_PAIR_NODES);
        let counts = step == step0 + n
            && pair == pair0 + 2 * n
            && step == pin_step
            && pair == pin_pair
            && moved;
        println!(
            "{NAME}: structure | with the tap: step {step} nodes (plain {step0} + {n}, predicted \
             {pin_step}), pair {pair} (plain {pair0} + {}, predicted {pin_pair}): {}",
            2 * n,
            verdict(counts)
        );
        let g = {
            let (gpu, w, b) = m.body_parts(NAME)?;
            let mut head = Head::new(gpu, w, b.head_eps())?;
            capture_order(gpu.stream(), || b.enqueue_chain(gpu, w, &mut head))?
        };
        Ok(counts & placed(&g, layers))
    }

    /// Each `ds41_hc_mean` of the captured step depends on exactly one node:
    /// the MoE join of layer `layers[k] − 1`, the `layers[k]`-th join in
    /// stream order, for the `k`-th mean.
    fn placed(g: &Captured, layers: &[usize]) -> bool {
        let joins: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| g.kernel(i).is_some_and(|k| k.starts_with("ds41_ffn_post")))
            .collect();
        let means: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| g.kernel(i) == Some("ds41_hc_mean"))
            .collect();
        let mut ok = means.len() == layers.len();
        for (k, &i) in means.iter().enumerate() {
            let want = layers
                .get(k)
                .and_then(|l| l.checked_sub(1))
                .and_then(|l| joins.get(l));
            let good = want.is_some_and(|&j| g.deps[i] == [j]);
            let after = g.deps[i]
                .iter()
                .map(|&j| match &g.nodes[j] {
                    StepNode::Kernel(k) => {
                        let layer = joins.iter().position(|&x| x == j);
                        format!("{k} (layer {layer:?})")
                    }
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "{NAME}: structure | ds41_hc_mean {k} (tapped layer {:?}) depends on [{after}]: {}",
                layers.get(k),
                verdict(good)
            );
            ok &= good;
        }
        println!(
            "{NAME}: structure | {} ds41_hc_mean nodes for {} tapped layers, {} joins: {}",
            means.len(),
            layers.len(),
            joins.len(),
            verdict(ok)
        );
        ok
    }

    /// `--tap` (module doc).
    fn tap(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        layers: &[usize],
        prompt: &[u32],
    ) -> Result<bool, GateError> {
        let width = layers.len() * hp.n_embd;
        m.reset()?;
        // seq[p] is the token stepped at position p; feats[p] its features.
        let mut seq: Vec<u32> = Vec::with_capacity(RUN as usize + 1);
        let mut feats: Vec<Vec<f32>> = Vec::with_capacity(RUN as usize);
        let mut next = 0;
        for p in 0..RUN as usize {
            let t = prompt.get(p).copied().unwrap_or(next);
            seq.push(t);
            next = m.step(&[t])?;
            let (gpu, _, b) = m.body_parts(NAME)?;
            let f = b.read_features(gpu, 1)?;
            if f.pos as usize != p || f.values.len() != width {
                return Err(format!(
                    "{NAME}: after the step at {p} the features name position {} ({} values)",
                    f.pos,
                    f.values.len()
                )
                .into());
            }
            feats.push(f.values.to_vec());
        }
        seq.push(next);
        let mut head = {
            let (gpu, w, b) = m.body_parts(NAME)?;
            Head::new(gpu, w, b.head_eps())?
        };
        let mut ok = true;
        for p in CHECKS {
            ok &= check_at(m, &mut head, hp, layers, &seq, &feats, p)?;
        }
        Ok(ok)
    }

    /// The host mean of the streams entering each tapped layer, from an
    /// eager step of `token` at `pos` with every seam's streams read back.
    fn observed(
        m: &mut Deepseek41Model,
        head: &mut Head,
        hp: &Hparams,
        layers: &[usize],
        token: u32,
        pos: u32,
    ) -> Result<(Vec<f32>, u32), GateError> {
        let n = hp.n_embd;
        let mut mean = vec![f32::NAN; layers.len() * n];
        let mut seen = vec![false; layers.len()];
        let o = finite::observed_step(m, head, token, pos, &mut |_, seam, v| {
            if let Seam::Ffn { layer, .. } = seam
                && let Some(k) = layers.iter().position(|&l| l == layer + 1)
            {
                for d in 0..n {
                    mean[k * n + d] = (((v[d] + v[n + d]) + v[2 * n + d]) + v[3 * n + d]) * 0.25;
                }
                seen[k] = true;
            }
            Ok(())
        })?;
        if !seen.iter().all(|&s| s) {
            return Err(format!(
                "{NAME}: the eager step at {pos} showed no MoE seam before {layers:?}: {seen:?}"
            )
            .into());
        }
        Ok((mean, o.token()))
    }

    /// Features `got` against the host mean `want`, bit for bit: one line.
    fn same(what: &str, got: &[f32], want: &[f32]) -> bool {
        let diff = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let first = got
            .iter()
            .zip(want)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        let ok = got.len() == want.len() && diff == 0;
        println!(
            "{NAME}: tap | {what}: {} values, {diff} differ{}: {}",
            want.len(),
            first.map_or_else(String::new, |i| format!(
                " (first at {i}: {:e} vs {:e})",
                got[i], want[i]
            )),
            verdict(ok)
        );
        ok
    }

    /// The checks at position `p` (module doc).
    #[allow(
        clippy::too_many_arguments,
        reason = "the model, the probe's head and the run's record"
    )]
    fn check_at(
        m: &mut Deepseek41Model,
        head: &mut Head,
        hp: &Hparams,
        layers: &[usize],
        seq: &[u32],
        feats: &[Vec<f32>],
        p: u32,
    ) -> Result<bool, GateError> {
        let (t, t1) = (seq[p as usize], seq[p as usize + 1]);
        m.rollback(p)?;
        let mut ok = true;
        let (h_a, _) = observed(m, head, hp, layers, t, p)?;
        {
            let (gpu, _, b) = m.body_parts(NAME)?;
            let f = b.read_features(gpu, 1)?.values.to_vec();
            ok &= same(&format!("pos {p} eager step"), &f, &h_a);
            let refused = b.read_features(gpu, 2).is_err();
            println!(
                "{NAME}: tap | pos {p}: row 1 read after a one-row step refused: {}",
                verdict(refused)
            );
            ok &= refused;
        }
        let (h_b, _) = observed(m, head, hp, layers, t1, p + 1)?;
        {
            let (gpu, _, b) = m.body_parts(NAME)?;
            let f = b.read_features(gpu, 1)?.values.to_vec();
            ok &= same(&format!("pos {} eager step", p + 1), &f, &h_b);
        }
        ok &= same(&format!("pos {p} graph step"), &feats[p as usize], &h_a);
        ok &= same(
            &format!("pos {} graph step", p + 1),
            &feats[p as usize + 1],
            &h_b,
        );
        m.rollback(p)?;
        let [ta, tb] = m.step_rows([t, t1])?;
        let tokens = ta == t1 && tb == seq[p as usize + 2];
        println!(
            "{NAME}: tap | pos {p}: pair over [{t}, {t1}] gives [{ta}, {tb}], the run's [{t1}, {}]: {}",
            seq[p as usize + 2],
            verdict(tokens)
        );
        ok &= tokens;
        let (gpu, _, b) = m.body_parts(NAME)?;
        let f = b.read_features(gpu, 2)?;
        if f.pos != p {
            return Err(format!(
                "{NAME}: the pair's features name position {}, not {p}",
                f.pos
            )
            .into());
        }
        ok &= same(
            &format!("pos {p} pair row A"),
            f.row(0).ok_or("no row 0")?,
            &h_a,
        );
        ok &= same(
            &format!("pos {} pair row B", p + 1),
            f.row(1).ok_or("no row 1")?,
            &h_b,
        );
        Ok(ok)
    }
}
