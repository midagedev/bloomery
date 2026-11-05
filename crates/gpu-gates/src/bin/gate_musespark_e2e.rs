//! The muse-spark end-to-end gate: the whole decode chain of one repeated
//! attention-over-latent-MoE layer — the per-head q/k norms, the NORM-pair
//! rope on the sliding layers and none on the full ones, the gated output,
//! the α/β residuals, the sigmoid router with its selection bias, the
//! SiLU-gated IQ2_XXS experts between the latent pair, the scale-then-norm
//! post-FFN norm and the softcapped head — against mainline llama.cpp's node
//! dumps. The fixture tier (a 4-layer mini keeping the window pattern
//! `[T,F,T,T]`: layers 0→1 and 1→2 the SWA→full and full→SWA transitions)
//! runs the whole-card load, then the placed arms (`placed_near_whole_card`).
//! The real tier runs the real file (`/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS`)
//! as the placed load of the gate plan (`gate_card`: the 3090's bytes on the
//! card in view, so nearly every routed expert is the host tier's)
//! against the real sets `ref_musespark_{a,b,c,e}`.
//!
//! The clauses are a list ([`CLAUSES`]); each prints lines ending in PASS or
//! FAIL, and a step that raises the card's fault word is a FAIL naming the
//! fault, never a pass. Each carries its tier tag (`tier::run_clause`):
//!
//! - `structure` (self-consistency): the captured step holds the entries the
//!   layer derives — one Q4_K embedding row, the weightless norms of its row
//!   and the head's; per layer seven `rms_norm` (the attention norm, the per-head q and k
//!   norms over their heads' rows, the per-head output norm, the
//!   post-attention norm, the ffn norm and the pre-expert norm), two
//!   `gain_rms_norm` (the post-expert and post-FFN
//!   scale-then-norms; the post-expert one only where the card keeps experts
//!   of the layer), two `q6k_gemv` (q and o), five `q8_0_gemv` (k, v,
//!   gate and the latent pair), `q3k_quantize_q8_1` (the q and o forms, and
//!   the experts' two where the card keeps experts), the router at its own
//!   count, `iq2_xxs_gate_up_sel` and `iq2_xxs_gemv_sel` (where the card
//!   keeps experts), one `weighted_sum` (a layer the card holds whole),
//!   `sigmoid_rows` and `mul_rows` (the gate), two `mix_rows` (the
//!   residuals), two `rope` and `2 · n_kv` appends and the head-64 segment
//!   pass and merge at the layer's window on a sliding layer, none of the
//!   rope on a full one, and the head's `scale_rows` + `softcap_rows` +
//!   `argmax_fault`; on a placed load a hosted layer adds the handoff of its
//!   `n_used` slots (`ds41_ffn_handoff_8`: the file routes 8 and has no
//!   shared expert) and, where the card keeps some of its experts, their sum
//!   (`ds41_ffn_card_acc_8`). A load with no host share has no host node.
//!   The span, the most nodes on one dependency path, is the node count less
//!   four a layer: the gate's gemv, its copy and its sigmoid run beside the
//!   attention chain, and the router beside the latent down and its norm.
//! - `eager_equals_replay`, `reset_rewinds`, `state_carries`
//!   (self-consistency): as `gate_nemotron_h_moe_e2e`'s, over a 16-token
//!   walk; the carried state is the K/V rows (the model holds no scan state).
//! - `router_lane_nan` (self-consistency): a NaN planted in one router lane
//!   of the body's own instance raises `Router` on the fault word and is
//!   picked by no slot — never a defined pick.
//! - `moe_is_its_rule` (self-consistency): at every position of the 16-token
//!   walk and every layer (the real tier: the first, middle and last layer at
//!   the walk's first [`REAL_RULE_POSITIONS`] positions), the picked experts'
//!   weights are the sigmoid scores of the body's own router logits over
//!   their sum (times `expert_weights_scale`), and `ffn_moe_out` is the
//!   host's f64 chain of the experts' rule on the body's own
//!   `ffn_latent_down`, ids and weights: both inputs of an expert rounded by
//!   the rule of the side that serves it (q8_1 for the ids below the layer's
//!   card count, the weight type's q8_K for the rest, the host tier's),
//!   SiLU-gated IQ2_XXS gate/up and down,
//!   the post-expert gain and weightless RMS per expert row, then the weighted
//!   sum — each to [`RULE_REL_BAND`], the chain's output at all but
//!   [`FLIPS_PER_LAYER`] positions a layer, which stay within
//!   [`FLIP_REL_BAND`] (a code at a rounding boundary). The experts are read
//!   in place from the file, one layer's at a time. The oracle clauses' bands
//!   are loose enough for the activation rules' difference from mainline's
//!   that a few-percent defect in the experts or the weights sits inside
//!   them; this clause catches it.
//! - `nodes_near_mainlines` (fixture oracle, set a; real tier `_real`, the
//!   real set a): every compared node at
//!   every layer (the fixture's four, layers 0→1 and 1→2 the SWA→full
//!   transitions among them), one layer at a time on mainline's own input:
//!   layer `l`'s walk replaces its input with the dump's `l_out` of the
//!   layer before it (`GpuModel::set_input_override`) and runs that layer
//!   alone (`GpuModel::set_layer_span`; the last layer with the head), so
//!   each layer's nodes
//!   carry that layer's rounding alone — the activation rules' difference
//!   from mainline's — and not what the layers before it handed on, which
//!   the fixture's attention and routers multiply 2 to 4 times a layer. The
//!   nodes are the attention block's taps (`attn_gate_proj`,
//!   `attn_gate_sig`, `attn_gated`, `attn_o_norm`, `attn_o_proj`,
//!   `attn_post_norm`), the α/β residual `ffn_inp`, the latent pair, the
//!   experts' weighted sum `ffn_moe_out`, `l_out`, the router's logits and
//!   the head's `result_output` (the last layer's walk). The last layer's
//!   nodes from `ffn_inp` on are the batch's output token (the prompt's
//!   last); every other row is one per prompt position, occurrences of one
//!   node concatenated in order. Logit-kind nodes keep the cosine floor the
//!   nemotron_h_moe gate holds its logits to; the activation nodes the routed
//!   experts feed (`ffn_moe_out`, `ffn_latent_up`, `l_out`) keep
//!   [`EXPERT_REL_BAND`] and the others [`DENSE_REL_BAND`], a projection's
//!   row (the attention's output projection and its post norm, the latent
//!   pair) that band times the row's gain scale ([`gain_scales`]); the kept ids keep
//!   mainline's set or differ only inside the
//!   crossing band (`cross_band`'s form, the nemotron_h_moe gate's, its eps
//!   pooled over the layer-positions that kept the set), and the nodes the
//!   router's choice feeds are not compared at a position whose set crossed.
//! - `window_past_mainlines` (fixture oracle, set e; real tier `_real`, the
//!   real set e on [`Sample::real`]'s layers): the prompt's 2,560
//!   ids. Every layer but the last writes `l_out` at every position (one
//!   occurrence a ubatch, concatenated in order); the last layer's `l_out`
//!   is the output token, the prompt's last. Each layer runs on mainline's
//!   input as above and each of those rows keeps [`EXPERT_REL_BAND`]; a row
//!   over it is excused only where the body's own selection order puts
//!   its last kept expert inside the crossing band of the next one (the set
//!   holds no ids). Each layer has a row past the 2,048 sliding window, so
//!   the sliding layers read a window of real cells and the full layers the
//!   whole context.
//! - `greedy_is_mainlines` (fixture oracle, sets a, b and c; real tier
//!   `_real`, the real sets): the greedy
//!   continuation, teacher-forced on the oracle's own ids — the body's
//!   argmax is the oracle's wherever the oracle's margin exceeds twice the
//!   run's max |Δ| (the nemotron_h_moe gate's top-1 rule; the Δ the body's
//!   own last-prompt logits carry against `topk.tsv`'s logit values, the one
//!   place the dump states the oracle's logits); below the band the two ids
//!   and the margin are printed, not gated. On every step the run's recorded
//!   id is the argmax of its own logits row.
//! - `fault_word_clean` (self-consistency).
//! - `placed_near_whole_card` (self-consistency, run after the others on
//!   loads of its own): see `placed_arms` in the source.
//!
//! The real tier's bands are the fixture's until its first run measures the
//! real file's (every node clause prints its worst relative distance by kind).
//! A set absent from the box, dumped from another fixture or build, or
//! unfinished ends the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_musespark_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the placed arms and the real tier's load read its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::{
        Run, Selection, argmax, cosine, cross_band, feed, finite, fresh, histogram, or_fault, same,
        selection_order, walk,
    };
    use crate::gate_card;
    use bloomery_gpu::arch::musespark::{Body, LayerNodes};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{
        GateError, checks_failed, open_split, q8_1_dequant, row_bytes, split_f32, verdict,
    };
    use cuda_core::sys;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, quantize_activations};
    use model::arch::Arch;
    use model::arch::musespark::hparams::Hparams;
    use model::arch::musespark::place::{PlaceError, PlanInputs};
    use model::placement::{PlacementError, Plan, PlanLevers, workstation};
    use refset::arch::musespark as real_sets;
    use refset::arch::musespark::fixture;
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::ik::ref_tensor_logical_in;
    use refset::ik::{RefManifest, topk_ids_logical_within};
    use std::collections::BTreeMap;

    type Model = GpuModel<Body>;

    /// A finished walk: the step's logits and each walked position's kept value.
    type Walk<T> = Result<Option<(Vec<f32>, Vec<T>)>, GateError>;

    const NAME: &str = "gate_musespark_e2e";

    /// Rows of the K/V planes a key head holds beyond the longest prompt the
    /// gate runs.
    const CTX_SPARE: usize = 16;

    /// The cosine the logits-kind nodes keep against mainline's: the floor
    /// `gate_nemotron_h_moe_e2e` holds its last-position logits to
    /// (`COSINE_FLOOR` there).
    const COSINE_FLOOR: f64 = 0.999;

    /// The relative distance a node keeps that the routed experts do not
    /// reach (the attention block's taps, `ffn_inp`, the latent
    /// down-projection): only the Q8_0 and Q6_K projections' activation
    /// rules differ from mainline's there. A projection's row keeps it at the
    /// layer's median gain ([`gain_scales`]).
    // PIN(2026-10-23): 0.03, the one band this gate held every node to (0.0447, nemotron_h_moe's) split by node kind; these nodes reach 2.5 %.
    const DENSE_REL_BAND: f64 = 0.03;

    /// The relative distance a node keeps that the routed experts feed
    /// (`ffn_moe_out`, `ffn_latent_up`, `l_out`): mainline rounds the experts'
    /// IQ2_XXS inputs as Q8_K (256-value blocks), the card as q8_1 (128), and
    /// the layer's attention rounding enters the experts' inputs as well.
    // PIN(2026-10-23): 0.055 (cosine 0.9985), was 0.0447 (floor 0.999, nemotron_h_moe's): off a router flip set e's rows run to 5.08 %, p999 4.85 %.
    const EXPERT_REL_BAND: f64 = 0.055;

    /// The least sum mainline divides a token's kept router scores by
    /// (llama-graph.cpp's build_moe_ffn clamps `ffn_moe_weights_sum` at
    /// 2^-14, the f16 range's least normal).
    const RENORM_CLAMP: f32 = 1.0 / 16_384.0;

    /// The projections whose band a row's gain scales ([`gain_scales`]): the
    /// node, the tap its gain is read from and that tap's input. The
    /// post-attention norm is the output projection's row normalized.
    // PIN(2026-10-23): flat bands held every row; on the real set a, κ = rel(W·q8_0(x), W·x) / rel(q8_0(x), x) of ffn_latent_down is 4.0 at r 0.25 (L28 p2), 4.8 at r 0.19 (L48 p1), 1.0 at r ≈ 1, and the 69 rows over the flat bands sit at most 0.81 of the scaled ones.
    const PROJECTIONS: &[(&str, &str, &str)] = &[
        ("attn_o_proj", "attn_o_proj", "attn_gated"),
        ("attn_post_norm", "attn_o_proj", "attn_gated"),
        ("ffn_latent_down", "ffn_latent_down", "ffn_norm"),
        ("ffn_latent_up", "ffn_latent_up", "ffn_moe_out"),
    ];

    /// The relative distance the host's f64 transcription of the card's own
    /// rule may sit from the card: f32 accumulation, a few units of the last
    /// place over a row.
    const RULE_REL_BAND: f64 = 1e-6;

    /// The distance a position may sit from the host's chain when one value of
    /// the experts' q8_1 inputs rounds to the neighbouring code: the host's f64
    /// accumulation and the card's f32 put a value on either side of a rounding
    /// boundary at an occasional position, and one code is `1/127` of its
    /// block's peak, so such a position stays within a percent. A defect in the
    /// experts or the weights moves most positions by more.
    const FLIP_REL_BAND: f64 = 1e-2;

    /// The positions of a layer that may sit between [`RULE_REL_BAND`] and
    /// [`FLIP_REL_BAND`] (a defect in an expert moves about half of them).
    const FLIPS_PER_LAYER: usize = 2;

    /// Tokens of the walk every own-contract clause runs.
    const PROMPT_LEN: usize = 16;

    /// The fixture-oracle clauses the gate declares.
    const FIXTURE_ORACLE: usize = 3;

    /// The clause that holds the placed arms to the load the clauses ran on.
    const PLACED_CLAUSE: &str = "placed_near_whole_card";

    /// The host tier's configuration every placed load of the gate opens with:
    /// no lever is read, so one left in the runner's environment cannot move a
    /// clause.
    const HOST: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// The activation nodes the gate compares, a layer's row each.
    const NODES: &[&str] = &[
        "attn_gate_proj",
        "attn_gate_sig",
        "attn_gated",
        "attn_o_norm",
        "attn_o_proj",
        "attn_post_norm",
        "ffn_inp",
        "ffn_latent_down",
        "ffn_moe_out",
        "ffn_latent_up",
        "l_out",
    ];

    /// The walk, the header, the file's selection biases, what the model the
    /// clauses run on holds, and the oracle sets of the tier.
    struct Case {
        hp: Hparams,
        /// The model file, for the host's reading of the experts.
        file: Split,
        walk: Vec<u32>,
        /// Each layer's `exp_probs_b`, from the file, in layer order.
        bias: Vec<Vec<f32>>,
        /// The routed experts the card holds of each layer.
        layout: Layout,
        /// The layers and positions the clauses whose cost grows with them run.
        sample: Sample,
        sets: Option<Sets>,
        /// The node clause's pooled router-logit eps and the kept sets it
        /// came from, for the window clause's crossing band.
        eps: std::cell::Cell<(f32, usize)>,
    }

    /// The oracle sets the tier runs: the fixture's mirrors in the fixture
    /// tier, the real file's in the real tier. `a` and `e` are the node and
    /// window clauses' (the greedy clause takes `a`, `b` and `c`).
    struct Sets {
        a: RefManifest,
        b: RefManifest,
        c: RefManifest,
        e: RefManifest,
    }

    /// The routed experts of each layer the card holds: all of them on a
    /// whole-card load, the plan's `n_l` on a placed one, the host tier
    /// serving the rest.
    struct Layout {
        n_card: Vec<usize>,
        n_expert: usize,
    }

    impl Layout {
        /// The whole-card load of `hp`'s file.
        fn whole(hp: &Hparams) -> Layout {
            Layout {
                n_card: vec![hp.n_expert; hp.n_layer],
                n_expert: hp.n_expert,
            }
        }

        /// `plan`'s card counts, one a layer of `hp`'s file.
        fn of_plan(plan: &Plan<'_>, hp: &Hparams) -> Result<Layout, GateError> {
            let n_card = plan
                .n_l
                .iter()
                .map(|&n| usize::try_from(n))
                .collect::<Result<Vec<_>, _>>()?;
            if n_card.len() != hp.n_layer || n_card.iter().any(|&n| n > hp.n_expert) {
                return Err(format!(
                    "the plan's card counts {n_card:?} are not one a layer (0..={}) of the file's {} layers",
                    hp.n_expert, hp.n_layer
                )
                .into());
            }
            Ok(Layout {
                n_card,
                n_expert: hp.n_expert,
            })
        }

        /// Layers with `keep` of their card count.
        fn count(&self, keep: impl Fn(usize) -> bool) -> usize {
            self.n_card.iter().filter(|&&n| keep(n)).count()
        }

        /// Layers the host serves a share of.
        fn hosted(&self) -> usize {
            self.count(|n| n < self.n_expert)
        }
    }

    /// The layers and positions the heavy clauses run over. The fixture's are
    /// every one; the real file's are a sample, for a walk of one layer costs
    /// the prompt's positions in decode steps and `moe_is_its_rule`'s f64
    /// chain dequantizes eight 4096 x 4096 experts a position.
    struct Sample {
        /// Layers the window clause forces and walks.
        window: Vec<usize>,
        /// Layers `moe_is_its_rule` checks.
        rule: Vec<usize>,
        /// Positions of the walk it checks them at.
        rule_positions: usize,
    }

    /// Positions of the walk `moe_is_its_rule` checks on the real file.
    const REAL_RULE_POSITIONS: usize = 4;

    impl Sample {
        /// Every layer and position of the fixture.
        fn all(hp: &Hparams) -> Sample {
            Sample {
                window: (0..hp.n_layer).collect(),
                rule: (0..hp.n_layer).collect(),
                rule_positions: PROMPT_LEN,
            }
        }

        /// The real file's: the window clause's first, middle and last full
        /// layers, each with the sliding layer before it (a full layer reads
        /// the whole context, a sliding one a window of real cells past 2,048,
        /// and each pair is a SWA→full transition), the rule clause's first,
        /// middle and last layers.
        fn real(hp: &Hparams) -> Result<Sample, GateError> {
            let full: Vec<usize> = (0..hp.n_layer).filter(|&l| !hp.sliding[l]).collect();
            let (Some(&first), Some(&last)) = (full.first(), full.last()) else {
                return Err("the file has no full-attention layer".into());
            };
            let mut window = Vec::new();
            for f in [first, full[full.len() / 2], last] {
                if f > 0 && hp.sliding[f - 1] {
                    window.push(f - 1);
                }
                window.push(f);
            }
            window.dedup();
            Ok(Sample {
                window,
                rule: vec![0, hp.n_layer / 2, hp.n_layer - 1],
                rule_positions: REAL_RULE_POSITIONS,
            })
        }
    }

    /// One set's node taps, read once: a layer's tensor by node name
    /// `name-layer`, with its row width (`ne[0]`, a position's values).
    /// Occurrences of one name are that node's ubatches, concatenated in
    /// occurrence order. The crossing band's scores come from the body's
    /// own logits, so the oracle's `ffn_moe_probs` is not held.
    struct Taps(BTreeMap<String, Tap>);

    /// One node's values. `at` is the prompt position of each row; `None`
    /// means row `i` is position `i`.
    struct Tap {
        width: usize,
        vals: Vec<f32>,
        at: Option<Vec<usize>>,
    }

    /// The last layer's nodes from `ffn_inp` on are the output tokens
    /// (`ggml_get_rows` before that layer's FFN). Everything else is one
    /// row a prompt position.
    fn output_rows(name: &str) -> bool {
        matches!(
            name,
            "ffn_inp"
                | "ffn_norm"
                | "ffn_latent_down"
                | "ffn_moe_logits"
                | "ffn_moe_probs"
                | "ffn_moe_out"
                | "ffn_latent_up"
                | "l_out"
                | "ffn_moe_topk"
        )
    }

    /// Occurrences of `name` in order, one logical row-major buffer, and a
    /// short geometry (`occurrences` and each occurrence's row count).
    fn stitch(man: &RefManifest, name: &str) -> Result<(usize, Vec<f32>, String), GateError> {
        let mut rows: Vec<_> = man.tensors.iter().filter(|r| r.name == name).collect();
        rows.sort_by_key(|r| r.occurrence);
        if rows.is_empty() {
            return Err(format!("{name} has no tensor row").into());
        }
        let got: Vec<u32> = rows.iter().map(|r| r.occurrence).collect();
        let want: Vec<u32> = (0..rows.len() as u32).collect();
        if got != want {
            return Err(format!("{name} occurrences {got:?} are not 0..{}", rows.len()).into());
        }
        let w = rows[0].ne[0] as usize;
        if w == 0 || rows.iter().any(|r| r.ne[0] as usize != w) {
            return Err(format!("{name} row widths are not all {w}").into());
        }
        let mut vals = Vec::new();
        let mut counts = Vec::with_capacity(rows.len());
        for r in &rows {
            let part = ref_tensor_logical_in(&man.dir, r)?;
            if !part.len().is_multiple_of(w) {
                return Err(format!(
                    "{name}/{} holds {} values, not a multiple of width {w}",
                    r.occurrence,
                    part.len()
                )
                .into());
            }
            counts.push(part.len() / w);
            vals.extend(part);
        }
        let geom = format!(
            "{} occ {}",
            rows.len(),
            counts
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("+")
        );
        Ok((w, vals, geom))
    }

    /// `None` is one row a prompt position. One row on an output-token node
    /// is the prompt's last token (the batch's only output). Any other
    /// shape is named and refused.
    fn row_at(n_pos: usize, nrows: usize, output_only: bool) -> Result<Option<Vec<usize>>, String> {
        if nrows == n_pos {
            return Ok(None);
        }
        if output_only && nrows == 1 && n_pos > 1 {
            return Ok(Some(vec![n_pos - 1]));
        }
        Err(format!("{nrows} rows against {n_pos} prompt positions"))
    }

    impl Taps {
        /// The named nodes of `layers` of a model of `n_layer`, against a
        /// prompt of `n_pos` tokens.
        fn of(
            man: &RefManifest,
            n_layer: usize,
            layers: impl IntoIterator<Item = usize>,
            names: &[&str],
            n_pos: usize,
        ) -> Result<Taps, GateError> {
            let mut map = BTreeMap::new();
            for l in layers {
                for name in names {
                    let key = format!("{name}-{l}");
                    let (w, vals, geom) = stitch(man, &key)?;
                    let nrows = vals.len() / w;
                    let output_only = l + 1 == n_layer && output_rows(name);
                    let at = row_at(n_pos, nrows, output_only)
                        .map_err(|e| format!("{key} ({geom}) {e}"))?;
                    map.insert(key, Tap { width: w, vals, at });
                }
            }
            Ok(Taps(map))
        }

        /// The tap `name-layer`'s row at position `p`, or `None` when the
        /// dump does not hold that position.
        fn row(&self, name: &str, l: usize, p: usize) -> Result<Option<&[f32]>, GateError> {
            let key = format!("{name}-{l}");
            let tap = self
                .0
                .get(&key)
                .ok_or_else(|| format!("{key} was not loaded"))?;
            let i = match &tap.at {
                None => p,
                Some(at) => match at.iter().position(|&q| q == p) {
                    Some(i) => i,
                    None => return Ok(None),
                },
            };
            tap.vals
                .get(i * tap.width..(i + 1) * tap.width)
                .map(Some)
                .ok_or_else(|| format!("{key} holds no row {i} of width {}", tap.width).into())
        }
    }

    /// One layer's expert ids, `n_used` a position, at the positions the
    /// dump holds.
    struct IdRows {
        ids: Vec<i32>,
        at: Option<Vec<usize>>,
        n_used: usize,
    }

    impl IdRows {
        fn at(&self, p: usize) -> Option<&[i32]> {
            let i = match &self.at {
                None => p,
                Some(at) => at.iter().position(|&q| q == p)?,
            };
            self.ids.get(i * self.n_used..(i + 1) * self.n_used)
        }
    }

    /// One clause: its name, its tier tag and its check.
    type Clause = fn(&mut Model, &Case) -> Result<bool, GateError>;

    /// The clauses, in the order they run.
    const CLAUSES: &[(&str, Tag, Clause)] = &[
        ("structure", Tag::SelfConsistency, structure),
        (
            "eager_equals_replay",
            Tag::SelfConsistency,
            eager_equals_replay,
        ),
        ("reset_rewinds", Tag::SelfConsistency, reset_rewinds),
        ("state_carries", Tag::SelfConsistency, state_carries),
        ("router_lane_nan", Tag::SelfConsistency, router_lane_nan),
        ("moe_is_its_rule", Tag::SelfConsistency, moe_is_its_rule),
        (
            "nodes_near_mainlines",
            Tag::FixtureOracle,
            nodes_near_mainlines,
        ),
        (
            "window_past_mainlines",
            Tag::FixtureOracle,
            window_past_mainlines,
        ),
        (
            "greedy_is_mainlines",
            Tag::FixtureOracle,
            greedy_is_mainlines,
        ),
        // The same three against the real file's sets (ref_musespark_{a,b,c,e}), in the real
        // tier, on the placed load of the gate plan; the window clause walks a sample of the
        // layers ([`Sample::real`]).
        (
            "nodes_near_mainlines_real",
            Tag::Oracle,
            nodes_near_mainlines,
        ),
        (
            "window_past_mainlines_real",
            Tag::Oracle,
            window_past_mainlines,
        ),
        ("greedy_is_mainlines_real", Tag::Oracle, greedy_is_mainlines),
        ("fault_word_clean", Tag::SelfConsistency, fault_word_clean),
    ];

    /// The captured step's kernel entries against what `layout`'s layers
    /// derive: the whole-card half runs on a layer the card holds whole
    /// (`weighted_sum`), a hosted layer's front hands off its `n_used` slots
    /// to the host (`ds41_ffn_handoff_8` at the file's 8) and a hosted layer
    /// that keeps card experts adds their gate·up, down, post-expert norm and
    /// sum (`ds41_ffn_card_acc_8`).
    /// The span (the most nodes on one dependency path) is the node count
    /// less the four nodes a layer's two forks run beside the rest.
    /// The pass, the captured node count, the host nodes, the listed
    /// got/want entries and every kernel entry with its count.
    fn graph_entries(
        m: &mut Model,
        hp: &Hparams,
        layout: &Layout,
    ) -> Result<(bool, usize, usize, String, String), GateError> {
        let l = hp.n_layer;
        let sliding = hp.sliding.iter().filter(|&&s| s).count();
        let n_kv = hp.n_head_kv;
        let hosted = layout.hosted();
        let whole = l - hosted;
        let with_card = layout.count(|n| n > 0);
        let hosted_card = layout.count(|n| n > 0 && n < layout.n_expert);
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let list = m.step_graph_nodes()?;
        let span = m.step_graph_span()?;
        let hist = histogram(&list);
        let ([host], _) = count_kinds(&list, [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST]);
        let router = format!("sigmoid_{}_{}_router", hp.n_expert, hp.n_used);
        // The handoff and the card slots' sum at the layer's slot count.
        let handoff = format!("ds41_ffn_handoff_{}", hp.n_used);
        let card_acc = format!("ds41_ffn_card_acc_{}", hp.n_used);
        let got = |name: &str| hist.get(name).copied().unwrap_or(0);
        let want: Vec<(String, usize)> = vec![
            ("embed_rows_q4k".into(), 1),
            // The embedding's weightless norm and the head's output norm, then
            // per layer the attention norm, the q and k per-head norms, the
            // per-head output norm, the post-attention norm, the ffn norm and
            // the pre-expert norm.
            ("rms_norm".into(), 2 + 7 * l),
            // The post-FFN norm of every layer and the post-expert norm of
            // every layer that keeps card experts.
            ("gain_rms_norm".into(), l + with_card),
            ("q6k_gemv".into(), 2 * l + 1),
            ("q8_0_gemv".into(), 5 * l),
            // The q and o forms, the head's, and the card experts' two.
            ("q3k_quantize_q8_1".into(), 2 * l + 1 + 2 * with_card),
            (router, l),
            ("iq2_xxs_gate_up_sel".into(), with_card),
            ("iq2_xxs_gemv_sel".into(), with_card),
            ("weighted_sum".into(), whole),
            (handoff, hosted),
            (card_acc, hosted_card),
            ("sigmoid_rows".into(), l),
            ("mul_rows".into(), l),
            ("mix_rows".into(), 2 * l),
            ("rope".into(), 2 * sliding),
            ("kv_append_pos_buf".into(), 2 * n_kv * l),
            ("gqa_flash_seg_64".into(), l),
            ("gqa_flash_merge_64".into(), l),
            ("scale_rows".into(), 1),
            ("softcap_rows".into(), 1),
            ("argmax_fault".into(), 1),
        ];
        // A load with no host share has no host node; a hosted one is not asked.
        let mut pass = hosted > 0 || host == 0;
        for (name, n) in &want {
            pass &= got(name) == *n;
        }
        // Each layer forks twice (the body's `attention` and `ffn_pre`): the
        // gate's gemv, its copy and its sigmoid beside the attention chain,
        // and the router beside the latent down and its norm. Each fork's
        // shorter side is off the one path the rest of the step runs on.
        let span_want = nodes.checked_sub(4 * l);
        pass &= span_want == Some(span);
        let listed = want
            .iter()
            .map(|(name, n)| format!("{name}={}/{n}", got(name)))
            .chain(std::iter::once(format!("span={span}/{nodes}-4*{l}")))
            .collect::<Vec<_>>()
            .join(" ");
        let kernels = hist
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(" ");
        Ok((pass, nodes, host, listed, kernels))
    }

    fn structure(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        let hp = &case.hp;
        let (pass, nodes, host, listed, kernels) = graph_entries(m, hp, &case.layout)?;
        println!(
            "structure: layers={} hosted={} sliding={} kv_heads={}, graph_nodes={nodes} \
             host={host}, entries (got/want) {listed} {}",
            hp.n_layer,
            case.layout.hosted(),
            hp.sliding.iter().filter(|&&s| s).count(),
            hp.n_head_kv,
            verdict(pass)
        );
        println!("structure: kernel entries {kernels}");
        Ok(pass)
    }

    fn eager_equals_replay(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "eager_equals_replay";
        m.set_mode(StepMode::Eager);
        let eager = fresh(CLAUSE, "eager", m, &case.walk)?;
        m.set_mode(StepMode::Graph);
        let replay = fresh(CLAUSE, "replay", m, &case.walk)?;
        let (Some(eager), Some(replay)) = (eager, replay) else {
            return Ok(false);
        };
        let identical = same(&eager, &replay);
        let pass = identical && finite(&eager);
        println!(
            "{CLAUSE}: walk{PROMPT_LEN} ids {:?}, logits bit-identical {identical}, finite {} {}",
            &eager.ids[eager.ids.len().saturating_sub(4)..],
            finite(&eager),
            verdict(pass)
        );
        Ok(pass)
    }

    fn reset_rewinds(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "reset_rewinds";
        m.set_mode(StepMode::Graph);
        let first = fresh(CLAUSE, "the first run", m, &case.walk)?;
        let second = fresh(CLAUSE, "the second run", m, &case.walk)?;
        let (Some(first), Some(second)) = (first, second) else {
            return Ok(false);
        };
        let ids = first.ids == second.ids;
        let bits = same(&first, &second);
        let pass = ids && bits;
        println!(
            "{CLAUSE}: walk{PROMPT_LEN} ids equal {ids}, logits bit-identical {bits} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    fn state_carries(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "state_carries";
        m.set_mode(StepMode::Graph);
        let Some(&last) = case.walk.last() else {
            return Ok(false);
        };
        if fresh(CLAUSE, "the first run", m, &case.walk)?.is_none() {
            return Ok(false);
        }
        let Some(after_a) = or_fault(CLAUSE, "the last token", feed(m, &[last]))? else {
            return Ok(false);
        };
        if fresh(CLAUSE, "the second run", m, &case.walk)?.is_none() {
            return Ok(false);
        }
        let Some(after_b) = or_fault(CLAUSE, "the last token", feed(m, &[last]))? else {
            return Ok(false);
        };
        let Some(alone) = fresh(CLAUSE, "the last token alone", m, &[last])? else {
            return Ok(false);
        };
        let twice = after_a.ids == after_b.ids && same(&after_a, &after_b);
        let carried = !same(&after_a, &alone);
        let pass = twice && carried;
        println!(
            "{CLAUSE}: last token {last} stepped twice from the same reset: ids {:?} vs {:?} equal \
             {twice}; the K/V rows carry (the logits after the walk differ from the token alone) \
             {carried} {}",
            after_a.ids,
            after_b.ids,
            verdict(pass)
        );
        Ok(pass)
    }

    /// A NaN planted in one router lane raises the fault word and yields no
    /// defined pick.
    fn router_lane_nan(m: &mut Model, _case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "router_lane_nan";
        let layer = 0;
        let (raised, defined) = m.router_lane_nan(layer)?;
        let pass = raised && !defined;
        println!(
            "{CLAUSE}: layer {layer}'s instance over a planted NaN lane: raised {raised}, a \
             defined pick {defined} (want none) {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// One routed expert's matrices dequantized on the host: gate and up
    /// `[ff][latent]` (`latent` contiguous), down `[latent][ff]`.
    struct Expert {
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    }

    /// Expert `e` of layer `l`, read in place from the file's IQ2_XXS stacks
    /// (`[in, out, n_expert]`, one expert's rows contiguous).
    fn expert(file: &Split, hp: &Hparams, l: usize, e: usize) -> Result<Expert, GateError> {
        let (latent, ff) = (hp.latent, hp.expert_ff[l]);
        let matrix = |kind: &str, rows: usize, cols: usize| -> Result<Vec<f32>, GateError> {
            let name = format!("blk.{l}.ffn_{kind}_exps.weight");
            let (s, t) = file
                .find(&name)
                .ok_or_else(|| format!("{name} is not in the file"))?;
            if t.ty != GgmlType::IQ2_XXS || t.dims != [cols as u64, rows as u64, hp.n_expert as u64]
            {
                return Err(format!(
                    "{name} is {:?} {:?}, want IQ2_XXS [{cols}, {rows}, {}]",
                    t.ty, t.dims, hp.n_expert
                )
                .into());
            }
            let per = rows * row_bytes(t.ty, cols)?;
            let bytes = file.shard(s).ok_or("shard index out of range")?.data(t)?;
            let src = bytes
                .get(e * per..(e + 1) * per)
                .ok_or_else(|| format!("{name} holds no expert {e}"))?;
            let mut out = vec![0.0f32; rows * cols];
            dequant_row(t.ty, src, &mut out).map_err(|err| format!("{name}: {err}"))?;
            Ok(out)
        };
        Ok(Expert {
            gate: matrix("gate", ff, latent)?,
            up: matrix("up", ff, latent)?,
            down: matrix("down", latent, ff)?,
        })
    }

    /// The routed experts' rule, on the host in f64 over the latent row `x`:
    /// both of an expert's inputs rounded by the rule of the side that serves
    /// it — q8_1 on the card, q8_K on the host tier (`on_card[i]`, the
    /// weight's own activation format) — the gate and up, `silu(g) · u`, the
    /// down, the post-expert gain and weightless RMS of that expert's row
    /// (before the router weights, mainline's order), then the weighted sum.
    fn host_experts(
        picked: &[(&Expert, bool)],
        (latent, ff, eps): (usize, usize, f32),
        post: &[f32],
        x: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>, GateError> {
        let round = |v: &[f32], on_card: bool| -> Result<Vec<f32>, GateError> {
            if on_card {
                return Ok(q8_1_dequant(v, v.len(), 1));
            }
            let mut out = vec![0.0f32; v.len()];
            quantize_activations(GgmlType::IQ2_XXS, v, &mut out)?;
            Ok(out)
        };
        let (x_card, x_host) = (round(x, true)?, round(x, false)?);
        let mut acc = vec![0.0f64; latent];
        for (&(ex, on_card), &wt) in picked.iter().zip(weights) {
            let xa = if on_card { &x_card } else { &x_host };
            let h: Vec<f32> = (0..ff)
                .map(|r| {
                    let (mut g, mut u) = (0.0f64, 0.0f64);
                    for (c, &a) in xa.iter().enumerate() {
                        g += f64::from(ex.gate[r * latent + c]) * f64::from(a);
                        u += f64::from(ex.up[r * latent + c]) * f64::from(a);
                    }
                    (g / (1.0 + (-g).exp()) * u) as f32
                })
                .collect();
            let ha = round(&h, on_card)?;
            let z: Vec<f64> = (0..latent)
                .map(|c| {
                    let row = &ex.down[c * ff..(c + 1) * ff];
                    let d: f64 = row
                        .iter()
                        .zip(&ha)
                        .map(|(&w, &a)| f64::from(w) * f64::from(a))
                        .sum();
                    d * f64::from(post[c])
                })
                .collect();
            let ms = z.iter().map(|v| v * v).sum::<f64>() / latent as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            for (a, zc) in acc.iter_mut().zip(&z) {
                *a += f64::from(wt) * zc * inv;
            }
        }
        Ok(acc.into_iter().map(|v| v as f32).collect())
    }

    /// The routed experts and the router's weights are the card's own rule:
    /// at every position of the 16-token walk and every layer the picked
    /// experts' weights are the sigmoid scores of the body's own logits over
    /// their sum (times `expert_weights_scale`), and its `ffn_moe_out` is the
    /// host's f64 chain of the same rule on the body's own `ffn_latent_down`,
    /// ids and weights, to [`RULE_REL_BAND`] (all but [`FLIPS_PER_LAYER`]
    /// positions a layer, which stay within [`FLIP_REL_BAND`]). No oracle: the bands of the two
    /// oracle clauses are loose enough (the activation rules' difference from
    /// mainline's) that a few-percent defect in the experts or the weights
    /// would sit inside them, and this clause is what catches it.
    fn moe_is_its_rule(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "moe_is_its_rule";
        let hp = &case.hp;
        let Some((_, steps)) = node_walk(m, &case.walk, None, Ok)? else {
            println!("{CLAUSE}: the walk faulted FAIL");
            return Ok(false);
        };
        let mut pass = true;
        let mut experts: BTreeMap<usize, Expert> = BTreeMap::new();
        let checked = steps.len().min(case.sample.rule_positions);
        for &l in &case.sample.rule {
            // One layer's experts at a time: a real expert's three matrices
            // dequantized are 200 MB.
            experts.clear();
            let n_card = case.layout.n_card[l];
            let gain = |what: &str| {
                split_f32(
                    &case.file,
                    &format!("blk.{l}.ffn_{what}_exp_norm.weight"),
                    hp.latent,
                )
            };
            let (pre, post) = (gain("pre")?, gain("post")?);
            let (mut w_worst, mut out_worst, mut flips) = (0.0f64, 0.0f64, 0usize);
            for (p, step) in steps.iter().enumerate().take(checked) {
                let n = step
                    .get(l)
                    .ok_or_else(|| format!("the body holds no layer {l}"))?;
                let ids: Vec<usize> = n.ffn_moe_topk.iter().map(|&i| i as usize).collect();
                let scores: Option<Vec<f64>> = ids
                    .iter()
                    .map(|&i| {
                        n.ffn_moe_logits
                            .get(i)
                            .map(|&z| 1.0 / (1.0 + (-f64::from(z)).exp()))
                    })
                    .collect();
                let Some(scores) = scores.filter(|_| {
                    ids.len() == hp.n_used
                        && n.ffn_moe_weights.len() == hp.n_used
                        && n.ffn_moe_logits.len() == hp.n_expert
                }) else {
                    println!(
                        "{CLAUSE}: layer {l} position {p}: the picked ids {ids:?} do not index the router's {} logits FAIL",
                        hp.n_expert
                    );
                    pass = false;
                    continue;
                };
                let sum: f64 = scores.iter().sum();
                let want: Vec<f32> = scores
                    .iter()
                    .map(|s| (s / sum * f64::from(hp.weights_scale)) as f32)
                    .collect();
                let w_rel = n
                    .ffn_moe_weights
                    .iter()
                    .zip(&want)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs() / f64::from(b).abs())
                    .fold(0.0f64, |acc, e| {
                        if acc.is_nan() || e.is_nan() {
                            f64::NAN
                        } else {
                            acc.max(e)
                        }
                    });
                // The experts' input: the latent row through the weighted RMS
                // and the pre-expert gain.
                let ld = &n.ffn_latent_down;
                let ms = ld.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / ld.len() as f64;
                let inv = 1.0 / (ms + f64::from(hp.rms_eps)).sqrt();
                let x: Vec<f32> = ld
                    .iter()
                    .zip(&pre)
                    .map(|(&v, &g)| (f64::from(v) * inv * f64::from(g)) as f32)
                    .collect();
                for &e in &ids {
                    if let std::collections::btree_map::Entry::Vacant(slot) = experts.entry(e) {
                        slot.insert(expert(&case.file, hp, l, e)?);
                    }
                }
                // The plan keeps each layer's id prefix `[0, n_card)` on the card.
                let picked: Vec<(&Expert, bool)> =
                    ids.iter().map(|&e| (&experts[&e], e < n_card)).collect();
                let host = host_experts(
                    &picked,
                    (hp.latent, hp.expert_ff[l], hp.post_eps),
                    &post,
                    &x,
                    &n.ffn_moe_weights,
                )?;
                let out_rel = crate::e2e::rel(&host, &n.ffn_moe_out);
                let w_ok = w_rel <= RULE_REL_BAND;
                if !w_ok {
                    println!(
                        "{CLAUSE}: layer {l} position {p} router weights rel {w_rel:.3e} > band \
                         {RULE_REL_BAND:.0e} FAIL"
                    );
                    pass = false;
                }
                if out_rel > RULE_REL_BAND || out_rel.is_nan() {
                    if out_rel <= FLIP_REL_BAND {
                        flips += 1;
                    } else {
                        println!(
                            "{CLAUSE}: layer {l} position {p} ffn_moe_out rel {out_rel:.3e} > flip \
                             band {FLIP_REL_BAND:.0e} FAIL"
                        );
                        pass = false;
                    }
                }
                w_worst = w_worst.max(w_rel);
                out_worst = out_worst.max(out_rel);
            }
            if flips > FLIPS_PER_LAYER {
                println!(
                    "{CLAUSE}: layer {l}: {flips} of {checked} positions sit over the band {RULE_REL_BAND:.0e} \
                     (at most {FLIPS_PER_LAYER}) FAIL"
                );
                pass = false;
            }
            println!(
                "{CLAUSE}: layer {l} ({n_card} of {} experts on the card), {checked} positions: \
                 router weights worst rel {w_worst:.3e} (band {RULE_REL_BAND:.0e}), ffn_moe_out \
                 worst rel {out_worst:.3e}, {flips} over {RULE_REL_BAND:.0e} (flip band \
                 {FLIP_REL_BAND:.0e}, at most {FLIPS_PER_LAYER})",
                hp.n_expert
            );
        }
        println!("{CLAUSE}: {}", verdict(pass));
        Ok(pass)
    }

    /// The body's nodes at every position of one walk and its last logits,
    /// from a reset over `ids` in eager mode with the taps armed, each
    /// position's nodes kept as `keep` projects them (a whole position's
    /// nodes are ~40 KB; the long window walk keeps one layer's rows alone).
    /// `force` names a layer whose input every step replaces with the
    /// oracle's `l_out` of the layer before it, so that layer's nodes carry
    /// its own rounding alone and not the rounding the layers before it
    /// handed on (layer 0 reads the embedding and is never forced). Eager is
    /// set before the taps arm: a capture from an earlier clause holds the
    /// untapped body, and arming refuses while one exists. The taps come off
    /// on the way out, so the later clauses capture the production graph.
    fn node_walk<T>(
        m: &mut Model,
        ids: &[u32],
        force: Option<(usize, &Taps)>,
        mut keep: impl FnMut(Vec<LayerNodes>) -> Result<T, GateError>,
    ) -> Walk<T> {
        m.set_mode(StepMode::Eager);
        m.set_taps(true)?;
        // A forced layer's nodes need that layer alone: the layers before it
        // are replaced by the oracle's row and the ones after it are not
        // compared, so the step runs the layer (and, for the last layer, the
        // head after it). The span comes off with the taps below.
        if let Some((l, _)) = force {
            let n = m.layers().len();
            m.set_layer_span(Some(l..if l + 1 == n { n } else { l + 1 }))?;
        }
        // stderr is unbuffered, so a signal mid-walk still names the position.
        eprintln!("walk: {} positions", ids.len());
        let walk = (|| {
            m.reset()?;
            let mut logits = Vec::new();
            let mut nodes = Vec::with_capacity(ids.len());
            for (i, &t) in ids.iter().enumerate() {
                if i.is_multiple_of(256) {
                    eprintln!("walk position {i}");
                }
                if let Some((l, taps)) = force
                    && l > 0
                {
                    let row = taps
                        .row("l_out", l - 1, i)?
                        .ok_or_else(|| format!("l_out-{} holds no row {i}", l - 1))?;
                    m.set_input_override(l, Some(row))?;
                }
                match or_fault("mainline runs", "the walk", m.step(&[t]))? {
                    Some(_) => logits = m.logits()?,
                    None => return Ok(None),
                }
                // Every compared buffer is rewritten by the next step, so
                // the nodes are read back per position.
                nodes.push(keep(m.nodes()?)?);
            }
            eprintln!("walk position {}", ids.len());
            Ok(Some((logits, nodes)))
        })();
        m.set_taps(false)?;
        walk
    }

    /// How many times its band each row of layer `l` keeps, by projection
    /// node ([`PROJECTIONS`]): the layer's median gain `|y| / |x|` over
    /// mainline's rows against the row's own, where the row's is the smaller,
    /// else 1 (and 1 for every other node). An input difference that is a
    /// fraction of `|x|` moves `y` by about that fraction of `|x|` times the
    /// median gain, so a row whose gain is `r` of the median reads it `1 / r`
    /// times larger against its own `|y|`. A position the dump does not hold
    /// keeps 1; a row of no norm is refused by name.
    fn gain_scales(
        taps: &Taps,
        l: usize,
        n_pos: usize,
    ) -> Result<BTreeMap<&'static str, Vec<f64>>, GateError> {
        let norm = |v: &[f32]| v.iter().map(|&a| f64::from(a).powi(2)).sum::<f64>().sqrt();
        let mut out = BTreeMap::new();
        for &(node, y, x) in PROJECTIONS {
            let mut gains = Vec::with_capacity(n_pos);
            for p in 0..n_pos {
                let gain = match (taps.row(y, l, p)?, taps.row(x, l, p)?) {
                    (Some(yr), Some(xr)) => {
                        let (ny, nx) = (norm(yr), norm(xr));
                        if !(nx.is_finite() && ny.is_finite() && nx > 0.0 && ny > 0.0) {
                            return Err(format!(
                                "layer {l} position {p}: {x} norm {nx:e}, {y} norm {ny:e}: no gain"
                            )
                            .into());
                        }
                        Some(ny / nx)
                    }
                    _ => None,
                };
                gains.push(gain);
            }
            let mut held: Vec<f64> = gains.iter().flatten().copied().collect();
            held.sort_by(f64::total_cmp);
            let median = held.get(held.len() / 2).copied();
            let scales = gains
                .iter()
                .map(|g| match (g, median) {
                    (Some(g), Some(m)) => (m / g).max(1.0),
                    _ => 1.0,
                })
                .collect();
            out.insert(node, scales);
        }
        Ok(out)
    }

    /// The relative band of a node: the experts' for the nodes the router's
    /// choice feeds, the dense one for the rest.
    fn band_of(node: &str) -> f64 {
        if after_router(node) {
            EXPERT_REL_BAND
        } else {
            DENSE_REL_BAND
        }
    }

    /// Hold one activation node's values to its relative band times the
    /// row's `scale` ([`gain_scales`]), printing the failure; `worst` keeps
    /// the largest distance over the scale of the dense nodes and of the
    /// nodes the experts feed.
    fn hold_node(
        clause: &str,
        (l, p): (usize, usize),
        node: &str,
        (ours, theirs): (&[f32], &[f32]),
        scale: f64,
        worst: &mut [f64; 2],
    ) -> bool {
        let e = crate::e2e::rel(ours, theirs);
        let kind = usize::from(after_router(node));
        worst[kind] = worst[kind].max(e / scale);
        let band = band_of(node) * scale;
        let pass = e <= band;
        if !pass {
            println!(
                "{clause}: layer {l} position {p} {node} rel {e:.3e} > band {band:.3} \
                 ({scale:.2} x the flat band, the row's gain) FAIL"
            );
        }
        pass
    }

    /// A layer's nodes by the oracle's `cb()` name.
    fn node_of<'a>(n: &'a LayerNodes, name: &str) -> &'a [f32] {
        match name {
            "attn_gate_proj" => &n.attn_gate_proj,
            "attn_gate_sig" => &n.attn_gate_sig,
            "attn_gated" => &n.attn_gated,
            "attn_o_norm" => &n.attn_o_norm,
            "attn_o_proj" => &n.attn_o_proj,
            "attn_post_norm" => &n.attn_post_norm,
            "ffn_inp" => &n.ffn_inp,
            "ffn_latent_down" => &n.ffn_latent_down,
            "ffn_moe_out" => &n.ffn_moe_out,
            "ffn_latent_up" => &n.ffn_latent_up,
            "l_out" => &n.l_out,
            _ => unreachable!("the compared nodes are named above"),
        }
    }

    /// The nodes the router's choice feeds: a position whose kept set differs
    /// from mainline's runs these on other experts' outputs, so they are not
    /// compared there.
    fn after_router(node: &str) -> bool {
        matches!(node, "ffn_moe_out" | "ffn_latent_up" | "l_out")
    }

    /// Set a, one layer at a time on the oracle's own input: layer `l`'s
    /// walk replaces its input with mainline's `l_out` of the layer before
    /// it, so every node of layer `l` carries that layer's rounding alone —
    /// the activation rule's difference from mainline's (Q8_K blocks of 256
    /// under the Q6_K and IQ2_XXS weights, Q8_0 blocks of 32 under the Q8_0
    /// ones, against the card's 128-value form and f32) — and not the
    /// rounding the layers before it handed on, which the fixture's
    /// attention (logit spread 2) and routers (16 close scores) multiply by
    /// 2 to 4 a layer. Every node of every layer and position is held to the
    /// band, the router's logits to the cosine floor, the kept ids to
    /// mainline's set or to the crossing band; the nodes the router's choice
    /// feeds are not compared at a position whose set crossed. The head's
    /// logits come from the last layer's walk.
    fn nodes_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "nodes_near_mainlines";
        let Some(sets) = case.sets.as_ref() else {
            return Err("the node clauses ran with no set opened".into());
        };
        let (hp, man) = (&case.hp, &sets.a);
        let n_layer = hp.n_layer;
        let tokens = prompt_tables::tokens(man)?;
        let (w0, rows0, geom0) = stitch(man, "l_out-0")?;
        let n_pos = rows0.len() / w0;
        if n_pos == 0 || n_pos > tokens.len() {
            println!(
                "{CLAUSE}: l_out-0 ({geom0}) has {n_pos} rows, the prompt holds {} ids FAIL",
                tokens.len()
            );
            return Ok(false);
        }
        let taps = match Taps::of(
            man,
            n_layer,
            0..n_layer,
            &NODES
                .iter()
                .copied()
                .chain(["ffn_norm", "ffn_moe_logits", "ffn_moe_probs"])
                .collect::<Vec<_>>(),
            n_pos,
        ) {
            Ok(t) => t,
            Err(e) => {
                println!("{CLAUSE}: {e} FAIL");
                return Ok(false);
            }
        };
        let ids = &tokens[..n_pos];
        // The oracle's kept ids. The last layer holds the output token only.
        let mut oracle_topk = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let ids = topk_ids_logical_within(
                man,
                man.tensor(&format!("ffn_moe_topk-{l}"), 0)?,
                hp.n_expert as u32,
            )?;
            if !ids.len().is_multiple_of(hp.n_used) {
                println!(
                    "{CLAUSE}: ffn_moe_topk-{l} holds {} ids, not a multiple of {} FAIL",
                    ids.len(),
                    hp.n_used
                );
                return Ok(false);
            }
            let rows = ids.len() / hp.n_used;
            let at = match row_at(n_pos, rows, l + 1 == n_layer) {
                Ok(at) => at,
                Err(e) => {
                    println!("{CLAUSE}: ffn_moe_topk-{l} {e} FAIL");
                    return Ok(false);
                }
            };
            oracle_topk.push(IdRows {
                ids,
                at,
                n_used: hp.n_used,
            });
        }
        let mut pass = true;
        // The largest distance of the dense nodes and of the expert-fed ones.
        let mut worst = [0.0f64; 2];
        // A layer-position whose set differs: the layer, the position, the
        // body's logits and kept ids and mainline's ids.
        type Crossing = (usize, usize, Vec<f32>, Vec<u32>, Vec<i32>);
        let mut crossings: Vec<Crossing> = Vec::new();
        let mut kept = 0usize;
        // The largest router-logit delta over the layer-positions that kept
        // the set, relative to the peak logit.
        let mut eps = 0.0f32;
        let mut head_logits = Vec::new();
        for (l, topk) in oracle_topk.iter().enumerate() {
            let keep = |per: Vec<LayerNodes>| {
                per.into_iter()
                    .nth(l)
                    .ok_or_else(|| format!("the body holds no layer {l}").into())
            };
            let Some((logits, nodes)) = node_walk(m, ids, Some((l, &taps)), keep)? else {
                println!("{CLAUSE}: the layer {l} walk faulted FAIL");
                return Ok(false);
            };
            if l + 1 == n_layer {
                head_logits = logits;
            }
            let scales = gain_scales(&taps, l, n_pos)?;
            for (p, ours) in nodes.iter().enumerate() {
                let same = topk.at(p).map(|row| {
                    row.len() == ours.ffn_moe_topk.len()
                        && row.iter().all(|&i| ours.ffn_moe_topk.contains(&(i as u32)))
                });
                // Mainline divides its kept scores by their sum clamped at
                // `RENORM_CLAMP` (`ffn_moe_weights_sum_clamped`, llama-graph.cpp's
                // build_moe_ffn), the body by the sum itself
                // (`route_core::renorm_divisor`). Where mainline's kept scores sum
                // under the clamp, its experts' sum and latent up are the body's
                // times that sum over the clamp; the post-FFN norm, scale then RMS,
                // takes the factor out of `l_out`.
                let clamped = match (same, topk.at(p), taps.row("ffn_moe_probs", l, p)?) {
                    (Some(true), Some(row), Some(probs)) => {
                        let sum: f32 = row.iter().map(|&i| probs[i as usize]).sum();
                        (sum < RENORM_CLAMP).then_some(sum / RENORM_CLAMP)
                    }
                    _ => None,
                };
                if let Some(c) = clamped {
                    println!(
                        "{CLAUSE}: layer {l} position {p}: mainline's kept scores sum to {:.4e}, \
                         under its clamp {RENORM_CLAMP:e}: the body's ffn_moe_out and \
                         ffn_latent_up are held at {c:.4} of their values",
                        c * RENORM_CLAMP
                    );
                }
                for name in NODES {
                    if after_router(name) && same == Some(false) {
                        continue;
                    }
                    if let Some(theirs) = taps.row(name, l, p)? {
                        let scaled: Vec<f32>;
                        let row = match clamped {
                            Some(c) if matches!(*name, "ffn_moe_out" | "ffn_latent_up") => {
                                scaled = node_of(ours, name).iter().map(|v| v * c).collect();
                                &scaled[..]
                            }
                            _ => node_of(ours, name),
                        };
                        let scale = scales.get(name).and_then(|s| s.get(p)).copied();
                        let scale = scale.unwrap_or(1.0);
                        pass &= hold_node(CLAUSE, (l, p), name, (row, theirs), scale, &mut worst);
                    }
                }
                // The router's logits are a logits-kind node. A position the
                // dump does not hold (the last layer's non-output tokens)
                // is not a comparison.
                let Some(want) = taps.row("ffn_moe_logits", l, p)? else {
                    continue;
                };
                let cos = cosine(&ours.ffn_moe_logits, want);
                if cos < COSINE_FLOOR {
                    println!(
                        "{CLAUSE}: layer {l} ffn_moe_logits cosine {cos:.6} < floor \
                         {COSINE_FLOOR} at position {p} FAIL"
                    );
                    pass = false;
                }
                let mut delta = 0.0f64;
                for (a, b) in ours.ffn_moe_logits.iter().zip(want) {
                    delta = delta.max((f64::from(*a) - f64::from(*b)).abs());
                }
                let peak = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                match (same, topk.at(p)) {
                    (Some(true), _) => {
                        kept += 1;
                        eps = eps.max((delta / f64::from(peak.max(1e-6))) as f32);
                    }
                    (Some(false), Some(row)) => crossings.push((
                        l,
                        p,
                        ours.ffn_moe_logits.clone(),
                        ours.ffn_moe_topk.clone(),
                        row.to_vec(),
                    )),
                    _ => {}
                }
            }
        }
        // The eps is the layers' pooled one: every layer ran on mainline's
        // own input, so the layers share one rounding.
        if eps == 0.0 {
            println!("{CLAUSE}: no layer-position kept mainline's experts, no band FAIL");
            return Ok(false);
        }
        println!("{CLAUSE}: eps {eps:.4} (max |Δlogit| / peak |logit|, {kept} kept sets)");
        case.eps.set((eps, kept));
        // Pass two: a crossed set — every id mainline keeps and the body
        // does not inside the crossing band of the body's own selection
        // order.
        let mut crossed = 0usize;
        for (l, p, logits, topk, want) in &crossings {
            let Selection {
                scores,
                rank,
                z_max,
                ..
            } = selection_order(&case.bias[*l], logits);
            let last = rank[topk.len() - 1];
            let last_score = scores[last.0 as usize];
            let mut ok = true;
            for &id in want.iter().filter(|&&i| !topk.contains(&(i as u32))) {
                let Some(at) = rank.iter().position(|&(e, _)| e == id as u32) else {
                    ok = false;
                    break;
                };
                let gap = last.1 - rank[at].1;
                let band = cross_band(eps, z_max, last_score, scores[id as usize]);
                if !band.is_finite() || gap.is_nan() || gap > band {
                    println!(
                        "{CLAUSE}: layer {l} position {p}: mainline keeps expert {id} the \
                         body does not, gap {gap:.4} > band {band:.4} FAIL"
                    );
                    ok = false;
                }
            }
            if ok {
                crossed += 1;
            } else {
                pass = false;
            }
        }
        println!(
            "{CLAUSE}: {n_pos} positions x {n_layer} layers: {kept} kept sets, {crossed} crossed \
             inside the band; worst rel over the row's gain scale dense {:.3e} (band \
             {DENSE_REL_BAND}), expert-fed {:.3e} (band {EXPERT_REL_BAND}) {}",
            worst[0],
            worst[1],
            verdict(pass)
        );
        // The head: the last position's softcapped logits (the last layer's
        // walk, on mainline's input) against the dump's own last row of
        // `result_output`.
        let vocab = hp.n_vocab;
        let result = ref_tensor_logical_in(&man.dir, man.tensor("result_output", 0)?)?;
        let at = result
            .len()
            .checked_sub(vocab)
            .ok_or("result_output holds no row")?;
        let logits = &head_logits;
        let cos = cosine(logits, &result[at..]);
        let mut delta = 0.0f64;
        for (a, b) in logits.iter().zip(&result[at..]) {
            delta = delta.max((f64::from(*a) - f64::from(*b)).abs());
        }
        let own = argmax(logits);
        let top = crate::e2e::top_ids(&result[at..], 2);
        let margin = f64::from(result[at + top[0]]) - f64::from(result[at + top[1]]);
        let head_pass = cos >= COSINE_FLOOR && (margin <= 2.0 * delta || own as usize == top[0]);
        println!(
            "{CLAUSE}: result_output cosine {cos:.6} (floor {COSINE_FLOOR}), argmax body {own} \
             mainline {} margin {margin:.4} vs 2 x max |Δ| {:.4} {}",
            top[0],
            2.0 * delta,
            verdict(head_pass)
        );
        Ok(pass & head_pass)
    }

    /// Set e: `l_out` at every layer and position, the context past the
    /// sliding window among them, one layer at a time on the oracle's own
    /// input as [`nodes_near_mainlines`] runs them. The set holds no ids, so
    /// a row over the band is excused only where the body's own selection
    /// order puts its last kept expert and the next one inside the crossing
    /// band, at the node clause's eps grown by the expected maximum of the
    /// longer walk's draws (`sqrt(ln N_e / ln N_a)`: the maximum of `N`
    /// Gaussian draws grows as `sqrt(2 ln N)`); a layer where the excuse
    /// covers more than half the rows fails by name.
    fn window_past_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "window_past_mainlines";
        let Some(sets) = case.sets.as_ref() else {
            return Err("the window clauses ran with no set opened".into());
        };
        let (hp, man) = (&case.hp, &sets.e);
        let n_layer = hp.n_layer;
        let tokens = prompt_tables::tokens(man)?;
        let (w0, rows0, geom0) = stitch(man, "l_out-0")?;
        let n_pos = rows0.len() / w0;
        if n_pos == 0 || n_pos > tokens.len() {
            println!(
                "{CLAUSE}: l_out-0 ({geom0}) has {n_pos} rows, the prompt holds {} ids FAIL",
                tokens.len()
            );
            return Ok(false);
        }
        // A forced layer reads the `l_out` of the one before it and compares its own.
        let mut tapped: Vec<usize> = case
            .sample
            .window
            .iter()
            .flat_map(|&l| [l.saturating_sub(1), l])
            .collect();
        tapped.sort_unstable();
        tapped.dedup();
        let taps = match Taps::of(man, n_layer, tapped, &["l_out"], n_pos) {
            Ok(t) => t,
            Err(e) => {
                println!("{CLAUSE}: {e} FAIL");
                return Ok(false);
            }
        };
        let ids = &tokens[..n_pos];
        let past = hp.window;
        if n_pos <= past {
            println!(
                "{CLAUSE}: l_out-0 ({geom0}) has {n_pos} positions, none past the window {past} FAIL"
            );
            return Ok(false);
        }
        let (eps_a, n_a) = case.eps.get();
        if eps_a == 0.0 || n_a < 2 {
            println!("{CLAUSE}: the node clause's eps is not set (it runs first) FAIL");
            return Ok(false);
        }
        let eps = eps_a * ((n_pos as f64).ln() / (n_a as f64).ln()).sqrt() as f32;
        let mut pass = true;
        let mut worst = 0.0f64;
        let walked = &case.sample.window;
        let mut past_rows = vec![0usize; walked.len()];
        let (mut compared, mut excused) = (0usize, 0usize);
        for (w, &l) in walked.iter().enumerate() {
            let keep = |per: Vec<LayerNodes>| {
                per.into_iter()
                    .nth(l)
                    .map(|n| (n.l_out, n.ffn_moe_logits))
                    .ok_or_else(|| format!("the body holds no layer {l}").into())
            };
            let Some((_last, rows)) = node_walk(m, ids, Some((l, &taps)), keep)? else {
                println!("{CLAUSE}: the layer {l} walk faulted FAIL");
                return Ok(false);
            };
            let (mut n_rows, mut n_excused) = (0usize, 0usize);
            for (p, (ours, logits)) in rows.iter().enumerate() {
                let Some(theirs) = taps.row("l_out", l, p)? else {
                    continue;
                };
                n_rows += 1;
                if p >= past {
                    past_rows[w] += 1;
                }
                let e = crate::e2e::rel(ours, theirs);
                worst = worst.max(e);
                if e <= EXPERT_REL_BAND {
                    continue;
                }
                let Selection {
                    scores,
                    rank,
                    z_max,
                    ..
                } = selection_order(&case.bias[l], logits);
                let (last, next) = (rank[hp.n_used - 1], rank[hp.n_used]);
                let band = cross_band(eps, z_max, scores[last.0 as usize], scores[next.0 as usize]);
                if last.1 - next.1 <= band {
                    n_excused += 1;
                    continue;
                }
                println!(
                    "{CLAUSE}: layer {l} position {p} l_out rel {e:.3e} > band \
                     {EXPERT_REL_BAND:.3}, the body's last kept expert is {:.4} above the next \
                     (band {band:.4}) FAIL",
                    last.1 - next.1
                );
                pass = false;
            }
            if n_excused * 2 > n_rows {
                println!(
                    "{CLAUSE}: layer {l}: the crossing excuse covers {n_excused} of {n_rows} rows FAIL"
                );
                pass = false;
            }
            compared += n_rows;
            excused += n_excused;
        }
        if past_rows.contains(&0) {
            println!(
                "{CLAUSE}: {n_pos} positions, window {past}, rows past it per walked layer \
                 {past_rows:?} (layers {walked:?}, {geom0}), a layer has no windowed row FAIL"
            );
            return Ok(false);
        }
        let gated: usize = past_rows.iter().sum();
        println!(
            "{CLAUSE}: {n_pos} positions x layers {walked:?} of {n_layer}, {compared} rows, {gated} \
             past the window {past}, {excused} excused inside the crossing band (eps {eps:.4}), \
             worst l_out rel {worst:.3e} (band {EXPERT_REL_BAND:.3}) {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The greedy continuation of sets a, b and c, teacher-forced on the
    /// oracle's own ids: the body's argmax is the oracle's wherever the
    /// oracle's margin exceeds twice the run's max |Δ| — the Δ the body's
    /// own last-prompt logits carry against `topk.tsv`'s logit values, the
    /// one place the dump states the oracle's logits. Below the band the two
    /// ids and the margin are printed, not gated. On every step the run's
    /// recorded id is the argmax of its own logits row.
    fn greedy_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "greedy_is_mainlines";
        let Some(sets) = case.sets.as_ref() else {
            return Err("the greedy clauses ran with no set opened".into());
        };
        let hp = &case.hp;
        let vocab = hp.n_vocab;
        let mut ok = true;
        for (name, man) in [("a", &sets.a), ("b", &sets.b), ("c", &sets.c)] {
            let tokens = prompt_tables::tokens(man)?;
            let n_pos =
                ref_tensor_logical_in(&man.dir, man.tensor("l_out-0", 0)?)?.len() / hp.n_embd;
            let ids = &tokens[..n_pos];
            let greedy: Greedy = prompt_tables::greedy(man)?;
            // The prompt walk, its last logits the first greedy step's own.
            m.set_mode(StepMode::Graph);
            m.reset()?;
            let mut ours = Vec::new();
            let mut logits = Vec::new();
            for &t in ids {
                match or_fault(CLAUSE, &format!("set {name} prompt"), m.step(&[t]))? {
                    Some(id) => {
                        ours.push(id);
                        logits.push(m.logits()?);
                    }
                    None => return Ok(false),
                }
            }
            let mut next = m.logits()?;
            // The Δ of the band: the body's own last-prompt logits against
            // the top-k table's logit values.
            let top: Vec<Top> = prompt_tables::topk(man)?;
            let delta = top
                .iter()
                .map(|t| (f64::from(next[t.id as usize]) - f64::from(t.logit)).abs())
                .fold(0.0, f64::max);
            // Teacher-forced on the oracle's own ids: the body's argmax at
            // step s is the oracle's wherever its margin passes the band.
            let band = 2.0 * delta;
            let (mut gated, mut set_ok) = (0usize, true);
            for (s, (&want, &margin)) in greedy.ids.iter().zip(&greedy.margins).enumerate() {
                let own = argmax(&next);
                if f64::from(margin) > band {
                    gated += 1;
                    if own != want {
                        println!(
                            "{CLAUSE}: set {name} step {s} body argmax {own} mainline {want}, \
                             margin {margin:.4} > band {band:.4} FAIL"
                        );
                        set_ok = false;
                    }
                } else {
                    println!(
                        "{CLAUSE}: set {name} step {s} body argmax {own} mainline {want}, \
                         margin {margin:.4} <= band {band:.4}: not gated"
                    );
                }
                if s + 1 == greedy.ids.len() {
                    break;
                }
                match or_fault(CLAUSE, &format!("set {name} step {s}"), m.step(&[want]))? {
                    Some(id) => {
                        ours.push(id);
                        next = m.logits()?;
                        logits.push(next.clone());
                    }
                    None => return Ok(false),
                }
            }
            // Every recorded id is its own logits row's argmax.
            let self_consistent = ours.len() == logits.len()
                && ours
                    .iter()
                    .zip(&logits)
                    .all(|(id, row)| *id == argmax(row) && row.len() == vocab);
            let pass = set_ok && self_consistent;
            ok &= pass;
            println!(
                "{CLAUSE}: set {name} {n_pos} prompt ids, {} greedy steps, {gated} gated at the \
                 band, Δ {delta:.4} (topk.tsv) {}",
                greedy.ids.len(),
                verdict(pass)
            );
        }
        Ok(ok)
    }

    fn fault_word_clean(m: &mut Model, _case: &Case) -> Result<bool, GateError> {
        let fault = m.gpu().fault()?;
        let pass = fault.is_none() && m.poisoned().is_none();
        println!(
            "fault_word_clean: word {fault:?}, poisoned {:?} {}",
            m.poisoned(),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- placed arms

    /// Layer 0's nodes after the first token of the walk, from a reset in
    /// eager mode with the taps armed. Layer 0 reads the embedding, so a load
    /// that serves the layer's experts another way keeps the router's kept ids
    /// and holds the experts' sum and the layer's output to
    /// [`EXPERT_REL_BAND`].
    fn first_layer(m: &mut Model, tok: u32) -> Result<Option<LayerNodes>, GateError> {
        m.set_mode(StepMode::Eager);
        m.set_taps(true)?;
        let nodes: Result<Option<LayerNodes>, GateError> = (|| {
            m.reset()?;
            match or_fault(PLACED_CLAUSE, "the first token", m.step(&[tok]))? {
                Some(_) => Ok(m.nodes()?.into_iter().next()),
                None => Ok(None),
            }
        })();
        m.set_taps(false)?;
        nodes
    }

    /// What the placed arms are held against: the load the clauses ran on —
    /// its walk, its first token's layer 0 and its context.
    struct Baseline {
        run: Run,
        first: LayerNodes,
        ctx: usize,
    }

    /// The baseline's runs, `None` when one raised the fault word.
    fn baseline(m: &mut Model, case: &Case) -> Result<Option<Baseline>, GateError> {
        m.set_mode(StepMode::Graph);
        let Some(run) = fresh(PLACED_CLAUSE, "the baseline walk", m, &case.walk)? else {
            return Ok(None);
        };
        let Some(&tok) = case.walk.first() else {
            return Err("the walk is empty".into());
        };
        let Some(first) = first_layer(m, tok)? else {
            return Ok(None);
        };
        Ok(Some(Baseline {
            run,
            first,
            ctx: m.ctx_max(),
        }))
    }

    /// The placed arms (module doc): the baseline's load has been released,
    /// each arm opens the file as a placed load of its plan. The fixture tier's
    /// arms are `card` (every expert on the card: the whole-card load's bits),
    /// `mid` (half the card's experts) and `host` (the floor budget: every
    /// layer hosted, the card keeping only what its dense granules' slack
    /// holds); the real
    /// tier's is `host`, against the placed load of the gate plan the clauses
    /// ran on. Each arm holds the captured step's entries to its layout, eager
    /// and replay bit-identical and finite, the fault word clean, and layer 0
    /// of the first token — the layer that reads the embedding, so both loads
    /// route alike — to the baseline's router ids and
    /// [`EXPERT_REL_BAND`] on the experts' sum and the layer's output. Later
    /// layers run on this arm's own rounding, which a sigmoid router with
    /// near-equal scores turns into other picks: the last logits' cosine and
    /// the two ids are printed, not gated.
    fn placed_arms(case: &Case, base: &Baseline, tier_now: tier::Tier) -> Result<bool, GateError> {
        const CLAUSE: &str = PLACED_CLAUSE;
        let file = open_split(Arch::MuseSpark, "gate-gpu-musespark-e2e")?;
        let inputs = PlanInputs::read(&file)?;
        let spec = gate_card::card()?;
        let machine = workstation::placed_on(spec, case.hp.n_layer, workstation::SCRATCH);
        let ctx = u64::try_from(base.ctx)?;
        let all = inputs.plan(&machine, ctx, &PlanLevers::default())?;
        let c = &all.cards[0];
        let budget = |bytes: u64| PlanLevers {
            card_budget_bytes: Some(bytes).into(),
            ..PlanLevers::default()
        };
        // The least budget the plan takes, at which the card keeps no routed
        // expert: the floor its zero-budget probe is refused with (the dense
        // granules alone, the stores, context, scratch, reserves and margin).
        // The whole-card plan's rounding is not it: its expert buffers' granules
        // round on their own, so its dense-plus-rounding term sits above or
        // below the dense granules alone, by the file.
        let no_experts = match inputs.plan(&machine, ctx, &budget(0)) {
            Err(PlaceError::Placement(PlacementError::CardBudgetFloor { floor, .. })) => floor,
            Err(e) => return Err(format!("{CLAUSE}: the zero card budget's probe: {e}").into()),
            Ok(_) => {
                return Err(format!(
                    "{CLAUSE}: the zero card budget's probe planned: the card holds routed \
                     experts at no budget"
                )
                .into());
            }
        };
        let arms: Vec<(&str, PlanLevers)> = match tier_now {
            tier::Tier::Fixture => vec![
                ("card", PlanLevers::default()),
                ("mid", budget(no_experts + c.expert_bytes / 2)),
                ("host", budget(no_experts)),
            ],
            tier::Tier::Real => vec![("host", budget(no_experts))],
        };
        let (Some(want_ids), Some(want)) = (base.run.ids.last(), base.run.logits.last()) else {
            return Err("the baseline walk is empty".into());
        };
        let Some(&tok) = case.walk.first() else {
            return Err("the walk is empty".into());
        };
        let mut pass = true;
        for (name, levers) in &arms {
            let plan = inputs.plan(&machine, ctx, levers)?;
            let layout = Layout::of_plan(&plan, &case.hp)?;
            let with_card = layout.count(|n| n > 0);
            println!(
                "{CLAUSE}: arm {name} plan: {} experts on the host, {with_card} of {} layers keep \
                 card experts, {} hosted, card experts by layer {:?}",
                plan.host.experts,
                case.hp.n_layer,
                layout.hosted(),
                layout.n_card
            );
            // The arm is the one it names: `card` hosts nothing, `host` hosts
            // every layer at the floor budget (the card keeps only the experts
            // its dense granules' slack holds), `mid` hosts and keeps.
            let shaped = match *name {
                "card" => layout.hosted() == 0,
                "host" => layout.hosted() == case.hp.n_layer,
                _ => layout.hosted() > 0 && with_card > 0,
            };
            let mut m = Model::open_placed(
                open_split(Arch::MuseSpark, "gate-gpu-musespark-e2e")?,
                &plan,
                0,
                HOST,
            )?;
            m.set_mode(StepMode::Eager);
            let eager = fresh(CLAUSE, "eager", &mut m, &case.walk)?;
            let (entries, nodes, _, listed, _) = graph_entries(&mut m, &case.hp, &layout)?;
            let replay = fresh(CLAUSE, "replay", &mut m, &case.walk)?;
            let (Some(eager), Some(replay)) = (eager, replay) else {
                pass = false;
                continue;
            };
            let fine = finite(&replay) && same(&eager, &replay);
            let clean = m.gpu().fault()?.is_none() && m.poisoned().is_none();
            let Some(first) = first_layer(&mut m, tok)? else {
                pass = false;
                continue;
            };
            let sorted = |ids: &[u32]| {
                let mut v = ids.to_vec();
                v.sort_unstable();
                v
            };
            let ids_kept = sorted(&first.ffn_moe_topk) == sorted(&base.first.ffn_moe_topk);
            let out_rel = crate::e2e::rel(&first.ffn_moe_out, &base.first.ffn_moe_out);
            let l_out_rel = crate::e2e::rel(&first.l_out, &base.first.l_out);
            let layer0 = ids_kept && out_rel <= EXPERT_REL_BAND && l_out_rel <= EXPERT_REL_BAND;
            let (Some(got_ids), Some(logits)) = (replay.ids.last(), replay.logits.last()) else {
                return Err("the placed walk is empty".into());
            };
            // The card arm is the baseline's load of the same bytes.
            let bits = *name != "card" || same(&base.run, &replay);
            let arm_pass = shaped && entries && fine && clean && layer0 && bits;
            println!(
                "{CLAUSE}: arm {name}: graph_nodes={nodes} entries (got/want) {listed}; eager == \
                 replay and finite {fine}, fault clean {clean}; layer 0 first token: ids kept \
                 {ids_kept}, ffn_moe_out rel {out_rel:.3e}, l_out rel {l_out_rel:.3e} (band \
                 {EXPERT_REL_BAND}); bits equal the baseline's {bits}; last id {got_ids} vs {want_ids}, \
                 cosine {:.6}, max|Δ| {:.3e} (not gated); arm shaped as named {shaped} {}",
                cosine(want, logits),
                crate::e2e::max_delta(want, logits),
                verdict(arm_pass)
            );
            pass &= arm_pass;
        }
        Ok(pass)
    }

    // ---------------------------------------------------------------- run

    /// The tier's oracle sets, opened before the card loads: the fixture's
    /// mirrors (`fx_ref_musespark_*`) in the fixture tier, the real file's
    /// `ref_musespark_*` in the real one. One missing, stale, foreign or
    /// unfinished ends the gate by name.
    fn open_sets(tier_now: tier::Tier) -> Result<Sets, GateError> {
        let open = |clause: &str, mirror: &str, real: &str| -> Result<RefManifest, GateError> {
            Ok(match tier_now {
                tier::Tier::Fixture => {
                    let path = tier::fixture_set(clause, &fixture::LCPP, mirror)?;
                    RefManifest::open(&path, &fixture::LCPP)?
                }
                tier::Tier::Real => {
                    RefManifest::open(&real_sets::LCPP.path(real), &real_sets::LCPP)?
                }
            })
        };
        Ok(Sets {
            a: open("the node clauses", fixture::A, real_sets::A)?,
            b: open("the greedy clauses", fixture::B, real_sets::B)?,
            c: open("the greedy clauses", fixture::C, real_sets::C)?,
            e: open("the window clauses", fixture::E, real_sets::E)?,
        })
    }

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::MuseSpark, "gate-gpu-musespark-e2e")?;
        let hp = Hparams::read(&file).map_err(|e| format!("muse-spark header: {e}"))?;
        let tier_now = tier::Tier::from_env()?;
        // The sets open before the card loads: one the tier runs, missing or
        // stale, ends the gate here.
        let sets = open_sets(tier_now)?;
        let longest = [&sets.a, &sets.e]
            .into_iter()
            .filter_map(|man| man.header.tokens.as_ref().map(|t| t.len()))
            .max()
            .unwrap_or(0)
            .max(PROMPT_LEN);
        let ctx = longest + CTX_SPARE;
        let walk = walk(PROMPT_LEN, hp.n_vocab);
        let bias = (0..hp.n_layer)
            .map(|l| split_f32(&file, &format!("blk.{l}.exp_probs_b.bias"), hp.n_expert))
            .collect::<Result<Vec<_>, _>>()?;
        let t = std::time::Instant::now();
        // The fixture is one load on the card; the real file is the gate plan's
        // placed load (module doc).
        let (mut m, layout, sample) = match tier_now {
            tier::Tier::Fixture => (
                Model::open(Gpu::new()?, file, ctx)?,
                Layout::whole(&hp),
                Sample::all(&hp),
            ),
            tier::Tier::Real => {
                let inputs = PlanInputs::read(&file)?;
                let spec = gate_card::card()?;
                let machine = workstation::placed_on(spec, hp.n_layer, workstation::SCRATCH);
                let plan = inputs.plan(&machine, u64::try_from(ctx)?, &PlanLevers::default())?;
                let layout = Layout::of_plan(&plan, &hp)?;
                println!(
                    "plan: ctx={ctx} card experts {} host experts {} host expert bytes {} (plan \
                     values), {} of {} layers keep card experts",
                    plan.cards[0].experts,
                    plan.host.experts,
                    plan.host.expert_bytes,
                    layout.count(|n| n > 0),
                    hp.n_layer
                );
                (
                    Model::open_placed(file, &plan, 0, HOST)?,
                    layout,
                    Sample::real(&hp)?,
                )
            }
        };
        println!(
            "load resident_bytes={} ctx={ctx} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let case = Case {
            hp,
            file: open_split(Arch::MuseSpark, "gate-gpu-musespark-e2e")?,
            walk,
            bias,
            layout,
            sample,
            sets: Some(sets),
            eps: std::cell::Cell::new((0.0, 0)),
        };
        let mut ok = true;
        for (name, tag, clause) in CLAUSES {
            if !tier::run_clause(name, *tag)? {
                continue;
            }
            eprintln!("clause {name}");
            let pass = clause(&mut m, &case)?;
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if !pass {
                println!("clause {name} FAIL");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            ok &= pass;
        }
        // The placed arms open loads of their own: the clauses' load is
        // released first, its walk kept. A walk that raised the fault word
        // leaves the arms nothing to be held against, and the clause fails.
        let arms = tier::run_clause(PLACED_CLAUSE, Tag::SelfConsistency)?;
        let held = if arms { baseline(&mut m, &case)? } else { None };
        drop(m);
        if arms {
            eprintln!("clause {PLACED_CLAUSE}");
            let pass = match &held {
                Some(base) => placed_arms(&case, base, tier_now)?,
                None => false,
            };
            if !pass {
                println!("clause {PLACED_CLAUSE} FAIL");
            }
            ok &= pass;
        }
        println!("{}", tier::tally_line());
        tier::expect_fixture_oracle(NAME, FIXTURE_ORACLE)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the captured step holds the entries its layers and their placement \
             derive; eager and replay give bit-identical logits; a reset rewinds the ids and the \
             logits bits; the K/V rows carry; a NaN router lane raises the fault word; the routed \
             experts and the router's weights are the rule of the side that serves each expert; \
             every compared node keeps its band at every layer and position (the real tier's \
             window clause on a sample of the layers), the sliding layers read a window of real \
             cells past 2,048 and the greedy continuation keeps the oracle's argmax above the \
             band; the placed arms keep the router's picks and the band at layer 0; the fault \
             word stays clean"
        );
        Ok(())
    }
}
