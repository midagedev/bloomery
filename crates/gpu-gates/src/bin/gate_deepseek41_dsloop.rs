//! GPU gate for the DSpark loop's target side: the feature tap
//! (`body::attach_features`, `Body::read_features`) on the gate placement
//! (`workstation::plan_gate`), its layers the draft file's `target_layers`
//! (`$BLOOMERY_DSPARK_MODEL`, header only: no draft is loaded here).
//!
//! - `--structure`: the step and the pair pass captured with the tap hold
//!   exactly the nodes they hold without it plus one kernel per tapped layer
//!   and row, pinned; in the step's capture each `ds41_hc_mean` depends on
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
mod gate {
    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model, Seam};
    use bloomery_gpu_gates::nodes::{Captured, StepNode, capture_order};
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::workstation;

    use crate::{dspark, finite};

    const NAME: &str = "gate_deepseek41_dsloop";
    /// PIN(2026-09-25): the step captured with the tl37 draft's tap: the
    /// plain step's 1169 nodes (1167 before the engram rows' wait and copy
    /// nodes) and one `ds41_hc_mean` per tapped layer.
    const TAP_STEP_NODES: usize = 1172;
    /// PIN(2026-09-25): the pair pass with the same tap: twice the plain
    /// step's nodes and one `ds41_hc_mean` per tapped layer and row.
    const TAP_PAIR_NODES: usize = 2344;
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
        let args = parse_args()?;
        let (_, dhp) = dspark::draft_hparams()?;
        let layers = dhp.target_layers.clone();
        let path = workstation::model_v41();
        let hp = Hparams::read(&Split::open(&path).map_err(|e| format!("open {path}: {e}"))?)?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let mut m = body::open(file, workstation::plan_gate, workstation::CTX_MAX as usize)?;
        m.set_mode(StepMode::Graph);
        let ids = prompt0()?;
        println!(
            "{NAME}: tap layers {layers:?} (the draft's target_layers), {} values each; prompt 0 \
             {ids:?}",
            hp.n_embd
        );
        let mut pass = true;
        let plain = if args.structure {
            Some(plain_counts(&mut m, ids[0])?)
        } else {
            None
        };
        m.set_mode(StepMode::Eager);
        body::attach_features(&mut m, &layers)?;
        m.set_mode(StepMode::Graph);
        if let Some(plain) = plain {
            pass &= structure(&mut m, &layers, ids[0], plain)?;
        }
        if args.tap {
            pass &= tap(&mut m, &hp, &layers, &ids)?;
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// The step's and the pair's node counts without a tap.
    fn plain_counts(m: &mut Deepseek41Model, t: u32) -> Result<[usize; 2], GateError> {
        let step = m.capture_step()?;
        m.step_pair(t, t)?;
        let pair = m.pair_graph_nodes()?.len();
        m.reset()?;
        println!("{NAME}: structure | no tap: step {step} nodes, pair {pair}");
        Ok([step, pair])
    }

    /// `--structure` (module doc), with the tap attached.
    fn structure(
        m: &mut Deepseek41Model,
        layers: &[usize],
        t: u32,
        [step0, pair0]: [usize; 2],
    ) -> Result<bool, GateError> {
        let n = layers.len();
        let step = m.capture_step()?;
        m.step_pair(t, t)?;
        let pair = m.pair_graph_nodes()?.len();
        m.reset()?;
        let counts = step == step0 + n
            && pair == pair0 + 2 * n
            && step == TAP_STEP_NODES
            && pair == TAP_PAIR_NODES;
        println!(
            "{NAME}: structure | with the tap: step {step} nodes (plain {step0} + {n}, pinned \
             {TAP_STEP_NODES}), pair {pair} (plain {pair0} + {}, pinned {TAP_PAIR_NODES}): {}",
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
        let [ta, tb] = m.step_pair(t, t1)?;
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
