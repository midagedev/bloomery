//! The Mistral Large 4 end-to-end gate: the whole one-token decode chain of
//! the placed load (`Body::open_placed` on the gate plan: the gate card's
//! bytes, one stage card running every layer and the head, the routed experts
//! the card does not hold on the host tier; in the fixture tier a budget
//! lever keeps half of the card's experts so both legs run) — the absorbed
//! MLA with SSMax on the joined query,
//! the YaRN rope at the LeRoPE factors, the lightning indexer whose keys are
//! projected from the layer input and whose list picks the positions the
//! attention reads (the file's `[rope | nope]` index head moved to the
//! kernel's tail order), the KV-shared NoPE layers on the 512-wide walk, the
//! softmax top-4 router over the picked experts beside the shared expert,
//! the dense lead and the softcapped head — against mainline llama.cpp's
//! node dumps: the fixture's (`fx_ref_mistrallarge4e128_{a,b}`) in the
//! fixture tier, the real file's (`ref_mistrallarge4_{a,b}`, or the
//! Mix-256G file's `ref_mistrallarge4_mix256g_{a,b}` when that file is the
//! one in view) in the real tier, each oracle clause tagged for its tier.
//!
//! The clauses are a list ([`CLAUSES`]); each prints lines ending in PASS or
//! FAIL, and a step that raises the card's fault word is a FAIL naming the
//! fault, never a pass. Each carries its tier tag (`tier::run_clause`):
//!
//! - `structure` (self-consistency): the captured step holds the entries
//!   each layer's kind derives (the kinds read from the header's arrays) and
//!   the kernels each tensor's file type launches, which the plan's site
//!   table names (`sites::launch`, read per tensor and per layer from the
//!   header: the dense gemv by its entry with the q8_1 quantization its
//!   entry reads, the per-head pair, a gated block by its three types, the
//!   head by its entry) — per layer the norm (fused with the q8_1 form where
//!   the query's entry reads one), the query and output projections, the rope
//!   and its columns on a roped layer, the latent projection and the fused
//!   norm-rope-append (the bare latent append on a NoPE owner, none on a
//!   reader), the LayerNorm, the index query projection, its permutation, its
//!   SSMax, the key projection, its permutation, the key append, the head
//!   weights, the score and the top-k (none of the indexer's on a shared
//!   layer), the per-head absorption, the rope-span assembly and the main
//!   SSMax, the row gather and the flash (the walk's segment and merge on
//!   NoPE), the value up-projection, the residual add; the dense lead's norm,
//!   block and add; a routed layer's norm, fused softmax router, handoff and
//!   go, the shared expert's block, the wait and the one join; a layer with
//!   card experts the card leg's four launches (its three slot launches named
//!   by the stacks' types) and its sum; the head's norm, quantization,
//!   projection, softcap and argmax. The memory operations are each routed
//!   layer's go and wait, no host node; the span is the node count (the
//!   chain is one path: no launch runs beside another). The embedding row is
//!   the host's copy before the step, no node.
//! - `eager_equals_replay`, `reset_rewinds`, `state_carries`
//!   (self-consistency): as `gate_musespark_e2e`'s, over a 40-token walk —
//!   past the fixture's top-k 16, so every step past position 16 selects;
//!   the carried state is the latent and index-key rows.
//! - `seeded_replay` (self-consistency): a step graph captured before a
//!   `seed_depth` and replayed after it — the seed refills the resident
//!   caches the capture recorded, never new allocations — equals an eager
//!   step at the seeded depth bit for bit.
//! - `router_lane_nan` (self-consistency): as `gate_musespark_e2e`'s.
//! - `selection_is_its_rule` (self-consistency): at every position past the
//!   top-k of the walk, every full indexer layer's picked set equals the
//!   host's f64 top-k of the body's own index scores (ties to the lower
//!   row, the kernel's own rule, named where they cut), and every shared
//!   layer's list is its source's, entry for entry.
//! - `layer_span_refusals` (self-consistency): with the taps armed,
//!   `Body::set_layer_span` refuses an empty span, a reversed one and one
//!   that ends or starts past the model's last layer by name (none of them
//!   is the whole chain), and takes the spans that fit.
//! - `nodes_near_mainlines` (fixture oracle, sets a and b): every compared
//!   node at every layer and prompt position, one layer at a time, the prompt
//!   teacher-forced from a reset: the span (`Body::set_layer_span`) runs from
//!   the first layer whose list or cache the compared one reads to the
//!   compared layer, each layer it reads and the compared one on mainline's
//!   own input at every position (`Body::set_input_override`
//!   with the dump's `l_out` of the layer before it), so every cache row,
//!   index key and list the layer reads was built from mainline's rows: the
//!   attention norm, the scaled queries, the key path's rows, the indexer's
//!   queries, key and weights, the list, the router's logits, ids and
//!   renormalized weights, the experts' sum (the host's sum and the card's,
//!   the join's own terms), the shared expert, the block's output and the
//!   residual. Logit-kind nodes keep a cosine floor; the
//!   attention's value row keeps [`ATTN_REL_BAND`]; the activation nodes the
//!   routed experts feed keep [`EXPERT_REL_BAND`], the others
//!   [`DENSE_REL_BAND`] (Muse's bands: the same projections' and experts'
//!   activation rules differ from mainline's here). A position whose list
//!   or router set crossed inside its own band skips the nodes that choice
//!   feeds.
//! - `greedy_is_mainlines` (fixture oracle, sets a and b): the greedy 32
//!   continuation from a reset, teacher-forced on the oracle's own ids — the
//!   argmax of step s's logits (the prompt's last for step 0, then those
//!   after the oracle's id s − 1) is the oracle's id s wherever its margin
//!   exceeds the set's band ([`GREEDY_MARGIN_A`], [`GREEDY_MARGIN_B`]);
//!   below it the two ids and the margin are printed, not gated. On every
//!   step the run's recorded id is the argmax of its own logits row. Each set
//!   prints how many steps its band gates, the prompt's last logits row
//!   against the dump's whole `result_output` row, and the drift of the
//!   oracle id's log-probability on the run's row at every step (printed
//!   readings: no band reads them).
//! - `head_near_mainlines` (fixture oracle, sets a and b): the head alone on
//!   mainline's own input. The last layer runs as the node clause runs it
//!   (the span from the layer its list or cache lives on, each layer it
//!   reads and the last on the dump's `l_out` of the layer before it at every
//!   prompt position), and the head reads that layer's output. The prompt's
//!   last position is compared with the dump's `result_norm` and
//!   `result_output` rows: the normed row's relative distance and its gain
//!   (the least-squares scale of ours on theirs), the logits row's cosine, and
//!   the gain over the dump's top [`HEAD_TOP`] ids. The greedy clause cannot
//!   see a scale error of the final norm or of the softcap — both keep the
//!   argmax — and the gains read them where the row's per-value distances
//!   hide them in the rounding of the layer beneath. The last layer's own
//!   output distance is printed beside them, so a distance the head did not
//!   add reads as the layer's.
//! - `nodes_near_mainlines_real`, `greedy_is_mainlines_real`,
//!   `head_near_mainlines_real` (oracle, the real file's sets): the three
//!   fixture clauses over the real file — every layer of set a one at a
//!   time, sets a and b's greedy 32 with the same printed readings, and the
//!   head of both sets — at the real tier's own bands and margins, and the
//!   last prompt position's logits at `topk.tsv`'s ids above the real tier's
//!   cosine floor. The real tier's node bands are derived per row from the
//!   fixture's constants (the gate's error-model section): each constant at
//!   the rounding ratio of its projection's input against a Gaussian row's,
//!   plus the distance the node's inputs carry; the normed latent takes a
//!   band per value, and a shared expert past its band has its down's input
//!   read on the host, with the line that names each term.
//! - `nodes_near_mainlines_deep` (oracle, the real file's deep set): the
//!   engine's own walk of the deep set's 3,000 prefill ids, then the one step
//!   at position 3,000 — past the indexer's top-k 2,048, where the list is a
//!   cut of thousands of rows — with every layer on mainline's input. Two
//!   contracts on the one walk: the selection rule (`selection_is_its_rule`'s
//!   check) after every step from the first past the top-k, and the node
//!   clause's comparisons at the deep step, every layer against the dump's
//!   rows, the caches and index keys the layers read being the walk's own (a
//!   list that moved a few rows at its threshold excuses nothing here; the
//!   moved fraction is printed), with the step's logits row held to the real
//!   tier's cosine floor against `result_output`. Bands: the real tier's
//!   derived ones, [`DEEP_BANDS`], with the attention's cache-row term not in
//!   view (the deep set holds the step's rows alone).
//! - `fault_word_clean` (self-consistency).

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_e2e", gate::run())
}

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the fixture arm reads its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "mistrallarge4")]
mod gate {
    use crate::e2e::{
        argmax, cosine, feed, finite, fresh, histogram, ik_last, or_fault, outcome, refused_by,
        same, tap_at, walk,
    };
    use crate::gate_card;
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, ref_model_path, verdict};
    use bloomery_gpu_mistrallarge4::body::{Body, LayerNodes};
    use cuda_core::sys;
    use gguf::quant::{dequant_row, f32_to_f16_bits, half_to_f32, quantize_row_q8_0_roundtrip};
    use gguf::{GgmlType, Split};
    use model::arch::Arch;
    use model::arch::mistrallarge4::hparams::Hparams;
    use model::arch::mistrallarge4::names as stems;
    use model::arch::mistrallarge4::place::PlanInputs;
    use model::arch::mistrallarge4::sites::{self, Launch, SelUp, UpProj};
    use model::arch::names::blk;
    use model::placement::kernels::{HeadEntry, SiteAct, SiteEntry};
    use model::placement::{Machine, PlanLevers, workstation};
    use refset::arch::mistrallarge4::{self as real_sets, fixture};
    use refset::arch::prompt_tables::{self, Greedy};
    use refset::family::Family;
    use refset::ik::{Layout, RefManifest, RowKind, ref_ints};
    use std::collections::BTreeMap;

    type Model = GpuModel<Body>;

    /// The model's body, for its own instruments (the graph rule is the
    /// caller's: no capture holds where these are called).
    fn body_of(m: &mut Model) -> Result<(&bloomery_gpu::Gpu, &mut Body), GateError> {
        const WHAT: &str = "gate_mistrallarge4_e2e";
        let (gpu, _w, body) = m.body_parts(WHAT)?;
        Ok((gpu, body))
    }

    const NAME: &str = "gate_mistrallarge4_e2e";

    /// Rows of the caches beyond the longest prompt the gate runs.
    const CTX_SPARE: usize = 8;

    /// The host tier's configuration the load opens with: no lever is read.
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// Tokens of the walk every own-contract clause runs: past the fixture's
    /// top-k 16, so every step past position 16 selects.
    const PROMPT_LEN: usize = 40;

    /// The fixture-oracle clauses the gate declares.
    const FIXTURE_ORACLE: usize = 3;

    /// The cosine the logits-kind nodes keep against mainline's
    /// (`gate_musespark_e2e`'s floor).
    const COSINE_FLOOR: f64 = 0.999;

    /// The relative distance a node keeps that the routed experts do not
    /// reach (Muse's band: the same q8_1-vs-q8_K and layer-rounding
    /// differences sit under it).
    // PIN(2026-10-31): Muse's EXPERT_REL_BAND (0.055, cosine 0.9985), this gate's first
    // run; the fixture keeps the same projection types and expert formats.
    const EXPERT_REL_BAND: f64 = 0.055;

    /// The relative distance a node keeps that the routed experts do not
    /// reach (Muse's band).
    // PIN(2026-10-31): Muse's DENSE_REL_BAND (0.03), this gate's first run.
    const DENSE_REL_BAND: f64 = 0.03;

    /// The relative distance the attention's value row (`fattn_mla`) keeps
    /// where the layer's selection is the dump's own: the card's Q6_K query
    /// and Q8_0 key projections round differently from mainline's
    /// dequantizing mul_mat (the scaled query and key rows sit inside
    /// [`DENSE_REL_BAND`]), and the softmax over the selected rows amplifies
    /// that onto the weighted sum.
    // PIN(2026-11-01): measured frontier 0.0494 (set b, layer 2, p28) over
    // both sets' positions whose selection is the dump's own; 1.2x it, the
    // margin DENSE_REL_BAND keeps over Muse's projection nodes (2.5 %).
    const ATTN_REL_BAND: f64 = 0.06;

    /// The oracle margin a greedy step may differ from mainline's id under,
    /// set by set: the head's Q6_K rows and the layers' rounding move the
    /// fixture's flat logits, so a step whose top two sit closer than the
    /// noise at that step may flip. The prompt's last position, whose top 40
    /// logits `topk.tsv` states, is one step of the 32: its max |Δ| does not
    /// bound the others, so a band is the frontier of the flips a run shows,
    /// not a bound derived from that |Δ|. The flips are rounding draws the
    /// engine's bits fix; a change that moves the bits redraws them.
    // PIN(2026-11-02): the 128-expert fixture's clean run, the largest margin at a step whose id
    // differs: set a 0.5917 (step 20), 1.2x; set b 0.4372 (step 29), under its band, which stays
    // at 1.2x the 16-expert cut's 0.5465 (step 10).
    const GREEDY_MARGIN_A: f64 = 0.71;
    const GREEDY_MARGIN_B: f64 = 0.66;

    /// The bands a node clause reads its distances against.
    #[derive(Clone, Copy)]
    struct Bands {
        expert: f64,
        dense: f64,
        attn: f64,
        cosine: f64,
        /// The constants are the own-operation terms of rows that sit as the
        /// fixture's do (unit gains, Gaussian rows), and each comparison's band
        /// is derived from them per row (the error-model section): scaled by
        /// how much louder the int8 rounding of the projection's input is, plus
        /// the distance the node's inputs carry. Off, each band is its
        /// constant.
        derived: bool,
    }

    /// The fixture's bands.
    const FIXTURE_BANDS: Bands = Bands {
        expert: EXPERT_REL_BAND,
        dense: DENSE_REL_BAND,
        attn: ATTN_REL_BAND,
        cosine: COSINE_FLOOR,
        derived: false,
    };

    // PIN(2026-11-02): the real tier's node bands are derived per row from the fixture's
    // constants (the error-model section: a band is its own-operation term at the input's
    // rounding ratio plus the distance its inputs carry), not from a run's worst. The fixture's
    // constants are the terms of its unit gains and Gaussian rows; the real file's norm gains
    // and heavy-tailed rows move the distances in ways a constant does not follow (the first
    // real-tier run, tree 11d9d26b4: 201 node lines past the fixture's constants, the greedy and
    // head clauses green). TODO-lead: the greedy margins below stay the fixture's until a
    // real-tier run's flips are read (1.2x the largest margin at a flip).
    /// The real file's bands.
    const REAL_BANDS: Bands = Bands {
        expert: EXPERT_REL_BAND,
        dense: DENSE_REL_BAND,
        attn: ATTN_REL_BAND,
        cosine: COSINE_FLOOR,
        derived: true,
    };
    const REAL_GREEDY_MARGIN_A: f64 = GREEDY_MARGIN_A;
    const REAL_GREEDY_MARGIN_B: f64 = GREEDY_MARGIN_B;

    // PIN(2026-11-02): TODO-lead, the deep step's bands below are the real tier's derived ones;
    // its layers read the caches and index keys of the engine's own walk of 3,000 positions,
    // whose rows carry the rounding of every earlier position's layers, and the deep set holds
    // mainline's rows of the step alone, so the attention's cache-row term (the error-model
    // section's `rows_term`) is not in view there: the deep attention's band is the one term
    // this derivation leaves open. The run prints each layer's frontier (the
    // "nodes_near_mainlines_deep: layer L" lines, with the "top_k moved" fraction of the list's
    // rows beside the nodes' distances) and the logits' cosine.
    /// The bands the deep step's node and logits comparisons read.
    const DEEP_BANDS: Bands = REAL_BANDS;

    /// The ids of the dump's logits row whose gain the head clause reads:
    /// the width of `topk.tsv`'s table.
    const HEAD_TOP: usize = 40;

    /// The most the head clause lets a gain (the least-squares scale of our
    /// row on the dump's, minus one) differ from zero, for the normed row and
    /// for the logits over the dump's top [`HEAD_TOP`] ids.
    // PIN(2026-11-02): TODO-lead, provisional [derived]: a final-norm gain 5 % off scales the normed
    // row by 0.05; the softcap bound at 29 for 30 moves a logit of 10 by -0.27 %, one of 20 by
    // -1.15 % and one of 25 by -1.9 % (c·tanh(x/c) at the same x), so the gain over the top ids
    // falls in -0.3 % .. -1.9 %; the rounding of one layer moves a gain at second order only.
    // The first run of each tier prints both gains (the "head_near_mainlines" lines); each band
    // becomes 1.2x the run's worst of its kind.
    const HEAD_GAIN_BAND: f64 = 0.003;

    /// The bands the head clause reads its rows against.
    #[derive(Clone, Copy)]
    struct HeadBands {
        /// The normed row's relative distance ([`rel`]) to `result_norm`.
        norm_rel: f64,
        /// The normed row's gain.
        norm_gain: f64,
        /// The logits row's cosine floor against `result_output`.
        cosine: f64,
        /// The logits' gain over the dump's top ids.
        top_gain: f64,
    }

    // PIN(2026-11-02): TODO-lead, the head bands below are provisional until the first run of each
    // tier: the normed row keeps the bands of the rows the last layer's routed experts feed
    // ([`EXPERT_REL_BAND`]) and the logits the cosine floor of the logit-kind nodes. The run prints
    // the measured distance, gain and cosine of every set; each band becomes 1.2x the worst of
    // its kind, with this line rewritten to name the run.
    /// The fixture's head bands.
    const FIXTURE_HEAD: HeadBands = HeadBands {
        norm_rel: EXPERT_REL_BAND,
        norm_gain: HEAD_GAIN_BAND,
        cosine: COSINE_FLOOR,
        top_gain: HEAD_GAIN_BAND,
    };
    /// The real file's head bands.
    const REAL_HEAD: HeadBands = FIXTURE_HEAD;

    /// What a layer's kind gives the clauses, read from the header's arrays.
    #[derive(Clone, Debug)]
    struct LayerCase {
        name: String,
        /// The layer ropes its queries and keys.
        ropes: bool,
        /// The layer runs its own indexer.
        full: bool,
        /// The layer owns its cache.
        owns: bool,
        /// The dense lead.
        dense: bool,
        /// The layer whose list it reads (itself on a full layer) and whose
        /// cache (itself on an owner).
        indexer: usize,
        cache: usize,
    }

    /// Every layer's [`LayerCase`].
    fn layer_cases(hp: &Hparams) -> Vec<LayerCase> {
        (0..hp.n_layer)
            .map(|l| {
                let (ropes, full, owns) = (hp.rope_layer[l], hp.indexer_full[l], hp.owns_cache(l));
                let dense = l < hp.dense_lead;
                let name = if dense {
                    "dense lead".to_string()
                } else {
                    format!(
                        "{} {} {}",
                        if ropes { "roped" } else { "nope" },
                        if full { "full" } else { "shared" },
                        if owns { "owner" } else { "reader" }
                    )
                };
                LayerCase {
                    name,
                    ropes,
                    full,
                    owns,
                    dense,
                    indexer: if full { l } else { hp.prev_full_indexer(l) },
                    cache: if owns { l } else { hp.kv_source[l] },
                }
            })
            .collect()
    }

    /// The `top_k-<j>` tensor holding layer `l`'s list in `man`: a list is
    /// dumped once, under the last name it was given, so layer `l` reads the
    /// first `j >= l` the dump holds.
    fn dumped_list(man: &RefManifest, l: usize, n_layer: usize) -> Result<usize, GateError> {
        (l..n_layer)
            .find(|j| man.tensor(&format!("top_k-{j}"), 0).is_ok())
            .ok_or_else(|| format!("no top_k-<j> with j >= {l} in the dump").into())
    }

    /// The header, the walk, the file's sites and the oracle sets of the tier.
    struct Case {
        hp: Hparams,
        walk: Vec<u32>,
        sites: Sites,
        sets: Sets,
        gains: Gains,
        /// The model file, a second mapping beside the load's: the derived
        /// bands' host FFN reads its weight rows.
        weights: Split,
    }

    /// The tier's sets: the fixture's two prompts, or the real file's two prompts and its
    /// deep set (a decode step past the indexer's top-k; the fixture's top-k is 16, which
    /// its own walk passes, so the fixture tier has none).
    struct Sets {
        a: RefManifest,
        b: RefManifest,
        deep: Option<RefManifest>,
    }

    type Clause = fn(&mut Model, &Case) -> Result<bool, GateError>;

    const CLAUSES: &[(&str, Tag, Clause)] = &[
        ("structure", Tag::SelfConsistency, structure),
        (
            "eager_equals_replay",
            Tag::SelfConsistency,
            eager_equals_replay,
        ),
        ("reset_rewinds", Tag::SelfConsistency, reset_rewinds),
        ("seeded_replay", Tag::SelfConsistency, seeded_replay),
        ("state_carries", Tag::SelfConsistency, state_carries),
        ("router_lane_nan", Tag::SelfConsistency, router_lane_nan),
        (
            "selection_is_its_rule",
            Tag::SelfConsistency,
            selection_is_its_rule,
        ),
        (
            "layer_span_refusals",
            Tag::SelfConsistency,
            layer_span_refusals,
        ),
        (
            "nodes_near_mainlines",
            Tag::FixtureOracle,
            nodes_near_mainlines,
        ),
        (
            "greedy_is_mainlines",
            Tag::FixtureOracle,
            greedy_is_mainlines,
        ),
        (
            "head_near_mainlines",
            Tag::FixtureOracle,
            head_near_mainlines,
        ),
        (
            "nodes_near_mainlines_real",
            Tag::Oracle,
            nodes_near_mainlines_real,
        ),
        (
            "greedy_is_mainlines_real",
            Tag::Oracle,
            greedy_is_mainlines_real,
        ),
        (
            "head_near_mainlines_real",
            Tag::Oracle,
            head_near_mainlines_real,
        ),
        (
            "nodes_near_mainlines_deep",
            Tag::Oracle,
            nodes_near_mainlines_deep,
        ),
        ("fault_word_clean", Tag::SelfConsistency, fault_word_clean),
    ];

    /// The relative distance of `got` to `want`: the largest absolute
    /// difference against the row's own peak (`max |want|`), the form every
    /// projection's rounding sits under — a per-value division would blow up
    /// at the row's near-zero entries. A length mismatch or any non-finite
    /// value on either side is an infinite distance: `f64::max` drops a NaN,
    /// so a NaN tap would otherwise read as a perfect match, and `zip` would
    /// pass a short row.
    fn rel(got: &[f32], want: &[f32]) -> f64 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let peak = want
            .iter()
            .map(|w| f64::from(w.abs()))
            .fold(0.0_f64, f64::max)
            .max(1e-3);
        got.iter()
            .zip(want)
            .map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs())
            .fold(0.0_f64, f64::max)
            / peak
    }

    /// Each node's three largest distances over a layer's positions, with the
    /// positions they sit at, in the order the nodes were first compared:
    /// the frontier a band is pinned from, and whether its largest stands
    /// alone.
    fn frontier(seen: &[(String, usize, f64)]) -> String {
        let mut names: Vec<&str> = Vec::new();
        for (n, _, _) in seen {
            if !names.contains(&n.as_str()) {
                names.push(n.as_str());
            }
        }
        names
            .iter()
            .map(|&n| {
                let mut at: Vec<(f64, usize)> = seen
                    .iter()
                    .filter(|(m, _, _)| m == n)
                    .map(|&(_, p, e)| (e, p))
                    .collect();
                at.sort_by(|a, b| b.0.total_cmp(&a.0));
                let top: Vec<String> = at
                    .iter()
                    .take(3)
                    .map(|(e, p)| format!("{e:.4}@p{p}"))
                    .collect();
                format!("{n} {}", top.join(" "))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Two four-value windows a side of a failing comparison — the head and
    /// a boundary 64 in — to read by eye.
    fn print_samples(clause: &str, got: &[f32], want: &[f32]) {
        let w = |v: &[f32], at: usize| {
            format!(
                "@{at}: [{:.4} {:.4} {:.4} {:.4}]",
                v.get(at).copied().unwrap_or(f32::NAN),
                v.get(at + 1).copied().unwrap_or(f32::NAN),
                v.get(at + 2).copied().unwrap_or(f32::NAN),
                v.get(at + 3).copied().unwrap_or(f32::NAN),
            )
        };
        println!(
            "{clause}:   got  {} {} {} len {}",
            w(got, 0),
            w(got, 64),
            w(got, 512),
            got.len()
        );
        println!(
            "{clause}:   want {} {} {} len {}",
            w(want, 0),
            w(want, 64),
            w(want, 512),
            want.len()
        );
    }

    /// A dump row of the reference's `[latent | rope]` order as the card's
    /// `[rope | latent]` (the flash row's own): the rope span from the row's
    /// tail first, then the latent.
    /// A dump tensor's per-position width and position count: positions are
    /// its last dimension over 1 (`[1024,18]`, `[576,16,18]`), a tensor no
    /// wider than its row holds one position (`[1024,1,1,1]`, the last
    /// layer's head-bound residual).
    fn dump_rows(ne: &[u64; 4]) -> (usize, usize) {
        match (1..4).rev().find(|&i| ne[i] > 1) {
            Some(k) => (ne[..k].iter().product::<u64>() as usize, ne[k] as usize),
            None => (ne[0] as usize, 1),
        }
    }

    fn rope_last(dump: &[f32], head: usize, rope: usize) -> Vec<f32> {
        // One row holds every head; the rope part moves within each head.
        assert!(head > rope && dump.len().is_multiple_of(head));
        dump.chunks(head)
            .flat_map(|h| {
                let (latent, r) = h.split_at(head - rope);
                r.iter().chain(latent)
            })
            .copied()
            .collect()
    }

    /// The k best rows of `scores` by the kernel's own rule: descending
    /// score, a tie to the lower row.
    fn rule_top(scores: &[f32], k: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..scores.len()).collect();
        ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        ids.truncate(k);
        ids
    }

    /// One full layer's list against the rule at one position: the first `n`
    /// rows of `list`, as a set (the kernel writes row order, the rule ranks),
    /// are [`rule_top`]'s. A tie at the cut goes to the lower row, so a set
    /// that differs is wrong whether or not the rows it took tie the k-th
    /// score; the two sets come back as the refusal. `Ok(true)` names a cut
    /// that ran through a tie the lower rows won.
    fn selection_check(
        rows: &[f32],
        list: &[u32],
        n: usize,
    ) -> Result<bool, (Vec<usize>, Vec<usize>)> {
        let rank = rule_top(rows, n);
        let mut want = rank.clone();
        let mut got: Vec<usize> = list[..n].iter().map(|&r| r as usize).collect();
        got.sort_unstable();
        want.sort_unstable();
        if got != want {
            return Err((got, want));
        }
        let Some(&last) = rank.last() else {
            return Ok(false);
        };
        let kth = rows[last];
        Ok(rows
            .iter()
            .enumerate()
            .any(|(r, v)| v.total_cmp(&kth).is_eq() && want.binary_search(&r).is_err()))
    }

    /// [`selection_check`]'s own calibration, on a cut through 17 equal
    /// scores at k = 16: the lower rows' list passes and names the tie, the
    /// list of rows 1..=16 and a list of one row sixteen times are refused,
    /// and a cut through no tie names none.
    fn selection_check_holds() -> bool {
        let tied = [1.0_f32; 17];
        let lower: Vec<u32> = (0..16).collect();
        let higher: Vec<u32> = (1..17).collect();
        let repeated = [16_u32; 16];
        let distinct = [3.0_f32, 1.0, 2.0];
        selection_check(&tied, &lower, 16) == Ok(true)
            && selection_check(&tied, &higher, 16).is_err()
            && selection_check(&tied, &repeated, 16).is_err()
            && selection_check(&distinct, &[0, 2], 2) == Ok(false)
    }

    // ------------------------------------------------------ the error model
    //
    // Both engines run the same weight bits, so a weight's own quantization
    // error cancels in every node distance: what the distances hold is the two
    // engines' activation rules and the distances a node's inputs already carry.
    // A Q8_0 projection reads the f32 row here (`q8_0_gemv`, the shared
    // expert's `ds41_shexp_gate_up`, the heads' `q8_0_gemv_heads`) and mainline's
    // CPU build quantizes it to q8_0 blocks of 32; a Q6_K projection reads q8_1
    // blocks of 128 here (`SiteAct::Q8x128`) and q8_K blocks of 256 there; the
    // routed experts meet mainline's q8_K on every leg (the host tier reads q8_K
    // itself). An int8 block of `B` values rounds each value by at most half of
    // `amax/127`, a noise of variance `(amax/127)²/12` a value, which a
    // projection carries onto its output at the input's RMS-relative size. The
    // fixture's rows are Gaussian with unit gains, so its constants are each
    // operation's term at a Gaussian row's block maxima; a row of the real file
    // scales that term by its own block maxima ([`rounding_ratio`]), and adds the
    // distance its inputs carry: an RMS norm's through its gains
    // ([`norm_carry`]), the cache rows' through the attention ([`rows_term`]),
    // the FFN input's through the SwiGLU block ([`SWIGLU_CARRY`]).

    /// The norm gains the node clauses read, from the model file before the
    /// load takes it: each cache owner's `attn_kv_a_norm`, each layer's
    /// `ffn_norm`, each full indexer layer's query LayerNorm (weight, bias).
    struct Gains {
        kv: Vec<Option<Vec<f32>>>,
        ffn: Vec<Vec<f32>>,
        idx_q: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    }

    impl Gains {
        fn read(file: &Split, hp: &Hparams) -> Result<Gains, GateError> {
            let f32s = |l: usize, stem: &str, n: usize| -> Result<Vec<f32>, GateError> {
                let name = blk(l, stem);
                let (_, t) = file
                    .find(&name)
                    .ok_or_else(|| format!("{name} is not in the model file"))?;
                if t.dims.iter().product::<u64>() != n as u64 {
                    return Err(format!("{name} is {:?}, want {n} values", t.dims).into());
                }
                let mut v = vec![0.0f32; n];
                dequant_row(t.ty, file.data(&name)?, &mut v)?;
                Ok(v)
            };
            let mut g = Gains {
                kv: Vec::with_capacity(hp.n_layer),
                ffn: Vec::with_capacity(hp.n_layer),
                idx_q: Vec::with_capacity(hp.n_layer),
            };
            for l in 0..hp.n_layer {
                g.kv.push(if hp.owns_cache(l) {
                    Some(f32s(l, stems::ATTN_KV_A_NORM, hp.kv_lora)?)
                } else {
                    None
                });
                g.ffn.push(f32s(l, stems::FFN_NORM, hp.n_embd)?);
                g.idx_q.push(if hp.indexer_full[l] {
                    Some((
                        f32s(l, stems::INDEXER_Q_NORM_W, hp.n_embd)?,
                        f32s(l, stems::INDEXER_Q_NORM_B, hp.n_embd)?,
                    ))
                } else {
                    None
                });
            }
            Ok(g)
        }

        /// Layer `l`'s latent norm gain: an owner's, refused by name elsewhere.
        fn kv(&self, l: usize) -> Result<&[f32], GateError> {
            match self.kv.get(l) {
                Some(Some(g)) => Ok(g.as_slice()),
                _ => Err(format!("layer {l} holds no attn_kv_a_norm: not a cache owner").into()),
            }
        }
    }

    /// The unit roundoff of f32.
    const U: f64 = 1.0 / (1u64 << 24) as f64;

    /// `n·u/(1 − n·u)`: the relative bound of an `n`-term f32 sum.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// Half an f16 ulp at magnitude `a` (the subnormal spacing below 2^-14).
    fn half_ulp16(a: f64) -> f64 {
        let ex = if a < f64::from(2.0f32.powi(-14)) {
            -14
        } else {
            a.log2().floor() as i32
        };
        2.0f64.powi(ex - 10) / 2.0
    }

    /// The RMS norm's rule in f64: `g_i·v_i / √(mean(v²) + eps)`.
    fn rms_normed(v: &[f32], g: &[f32], eps: f32) -> Vec<f64> {
        let sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let s = 1.0 / (sq / v.len() as f64 + f64::from(eps)).sqrt();
        v.iter()
            .zip(g)
            .map(|(&x, &g)| f64::from(g) * f64::from(x) * s)
            .collect()
    }

    /// The LayerNorm's rule in f64: `w_i·(v_i − mean) / √(var + eps) + b_i`.
    fn layer_normed(v: &[f32], (w, b): (&[f32], &[f32]), eps: f32) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = v
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n;
        let s = 1.0 / (var + f64::from(eps)).sqrt();
        v.iter()
            .zip(w)
            .zip(b)
            .map(|((&x, &w), &b)| f64::from(w) * (f64::from(x) - mean) * s + f64::from(b))
            .collect()
    }

    /// The activation blocks of a Q8_0 projection's two rules (mainline's q8_0;
    /// the f32 row here rounds nothing).
    const Q8_0_RULES: &[usize] = &[32];
    /// A Q6_K projection's: the q8_1 blocks of 128 here, mainline's q8_K of 256.
    const Q6_K_RULES: &[usize] = &[128, 256];
    /// The routed experts': mainline's q8_K, which every leg meets.
    const EXPERT_RULES: &[usize] = &[256];

    /// `E[max_{i<B} Z_i²]` over `B` standard normal values, the square of a
    /// Gaussian row's block maximum the fixture's constants sit at [derived:
    /// `∫₀^∞ 2t·(1 − erf(t/√2)^B) dt`, 5.7095 at 32, 8.1483 at 128, 9.4038 at
    /// 256].
    fn gauss_block_max2(b: usize) -> Result<f64, GateError> {
        match b {
            32 => Ok(5.7095),
            128 => Ok(8.1483),
            256 => Ok(9.4038),
            _ => Err(format!("no Gaussian block maximum for blocks of {b}").into()),
        }
    }

    /// How much louder the int8 rounding of `x`'s blocks under `rules` is
    /// than a Gaussian row's at the same RMS: `√(Σ_B mean_b amax_b² / (mean x²
    /// · Σ_B E_B))`, each rule's noise power `mean_b (amax_b/127)²/12` a value
    /// against the Gaussian row's [`gauss_block_max2`]. Floored at 1: the
    /// constant it scales also holds the terms no block maximum moves (the
    /// sums' order, the f16 rows). A zero row rounds nothing (1).
    fn rounding_ratio(x: &[f64], rules: &[usize]) -> Result<f64, GateError> {
        let n = x.len();
        let ms = x.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
        if ms <= 0.0 || !ms.is_finite() {
            return Ok(1.0);
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for &b in rules {
            if n == 0 || !n.is_multiple_of(b) {
                return Err(format!("a row of {n} values is not whole blocks of {b}").into());
            }
            let sum: f64 = x
                .chunks(b)
                .map(|c| c.iter().fold(0.0f64, |a, v| a.max(v.abs())).powi(2))
                .sum();
            num += sum / (n / b) as f64;
            den += gauss_block_max2(b)?;
        }
        Ok((num / (ms * den)).sqrt().max(1.0))
    }

    /// The RMS-relative distance `‖got − want‖₂ / ‖want‖₂`: infinite on a
    /// length mismatch, a non-finite value or a zero `want`.
    fn rel2(got: &[f64], want: &[f64]) -> f64 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let d: f64 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
        let w: f64 = want.iter().map(|b| b * b).sum();
        if w > 0.0 {
            (d / w).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// [`rel`] over f64 rows: the widest difference against `want`'s peak.
    fn rel64(got: &[f64], want: &[f64]) -> f64 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let peak = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-3);
        got.iter()
            .zip(want)
            .fold(0.0f64, |a, (g, w)| a.max((g - w).abs()))
            / peak
    }

    /// The largest magnitude of a row.
    fn peak(v: &[f32]) -> f64 {
        v.iter().fold(0.0f64, |a, x| a.max(f64::from(x.abs())))
    }

    /// The normed latent's band per value: the RMS norm's rule (f64) on each
    /// engine's own latent, `ours` and `theirs` (the projection's rows), moves
    /// value `i` by `|rule(ours)_i − rule(theirs)_i|` — the input's distance
    /// carried through the norm exactly, each value by its own gain — plus
    /// each engine's rounding of the rule, `2(γ_K + 4u)·|rule_i|` (the
    /// `gate_dspark_kv` norm bound's), plus half an f16 ulp where ours is the
    /// f16 cache's row. A band per value: the gains' spread moves a value of a
    /// high gain and a small latent far against the row's peak, which one band
    /// against the peak does not follow.
    fn latent_band(ours: &[f32], theirs: &[f32], g: &[f32], eps: f32, f16: bool) -> Vec<f64> {
        let (o, t) = (rms_normed(ours, g, eps), rms_normed(theirs, g, eps));
        let round = 2.0 * (gamma(ours.len()) + 4.0 * U);
        o.iter()
            .zip(&t)
            .map(|(&a, &b)| {
                (a - b).abs()
                    + round * (a.abs() + b.abs())
                    + if f16 { half_ulp16(a.abs()) } else { 0.0 }
            })
            .collect()
    }

    /// The largest magnitude of a row's distance against its peak, per head:
    /// `max_h ‖got_h − want_h‖₂ / max_h ‖want_h‖₂` over heads of `w` values.
    /// A per-head dense map (the value up-projection) carries it onto its
    /// output's peak-relative distance [derived: a dense map of incoherent
    /// rows gives its outputs one crest, so the output's peak follows the
    /// largest head norm and its widest difference the largest head
    /// distance].
    fn head_rel(got: &[f64], want: &[f64], w: usize) -> f64 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (mut d, mut p) = (0.0f64, 0.0f64);
        for (g, t) in got.chunks(w).zip(want.chunks(w)) {
            let diff: Vec<f64> = g.iter().zip(t).map(|(a, b)| a - b).collect();
            d = d.max(norm(&diff));
            p = p.max(norm(t));
        }
        if p > 0.0 { d / p } else { f64::INFINITY }
    }

    /// The share of `bound` the widest value of `got` against `want` uses
    /// (`max_i |got_i − want_i| / bound_i`; 1 is the bound): infinite on a
    /// length mismatch or a non-finite value, and where a bound of 0 meets a
    /// difference.
    fn bound_used(got: &[f32], want: &[f64], bound: &[f64]) -> f64 {
        if got.len() != want.len()
            || got.len() != bound.len()
            || got.iter().any(|v| !v.is_finite())
            || want.iter().any(|v| !v.is_finite())
        {
            return f64::INFINITY;
        }
        got.iter()
            .zip(want)
            .zip(bound)
            .fold(0.0f64, |a, ((&g, &w), &b)| {
                let d = (f64::from(g) - w).abs();
                a.max(if b > 0.0 {
                    d / b
                } else if d > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                })
            })
    }

    /// A normed latent `got` against the RMS norm's rule on the latent `raw`
    /// it was normed from: the share of the rule's rounding bound (`2(γ_K +
    /// 4u)·|rule|`, plus half an f16 ulp where `got` is an f16 cache row) its
    /// widest value uses. Read on our rows it holds our norm to its own rule;
    /// on mainline's two `kv_cmpr` rows it holds the oracle's normed row to the
    /// latent it dumped beside it.
    fn rule_used(got: &[f32], raw: &[f32], g: &[f32], eps: f32, f16: bool) -> f64 {
        let rule = rms_normed(raw, g, eps);
        let round = 2.0 * (gamma(raw.len()) + 4.0 * U);
        let bound: Vec<f64> = rule
            .iter()
            .map(|r| round * r.abs() + if f16 { half_ulp16(r.abs()) } else { 0.0 })
            .collect();
        bound_used(got, &rule, &bound)
    }

    /// The SwiGLU block's carry of an RMS-relative input distance onto its
    /// output's: `√(1 + σ²·E[silu′(σZ)²] / E[silu(σZ)²])` at its widest over
    /// the gate's scale σ, 1.4376 near σ = 1 and √2 at both ends [derived:
    /// gate and up moved independently by incoherent maps, the down
    /// incoherent].
    const SWIGLU_CARRY: f64 = 1.44;

    /// The FFN's normed input on both engines (`ffn_norm` of each engine's
    /// `ffn_inp` at the file's gain) and what the derived bands read of it.
    struct FfnIn {
        x_o: Vec<f64>,
        x_m: Vec<f64>,
        /// Mainline's rounding ratio of the row under the Q8_0 rules (the
        /// shared expert's and the dense lead's projections) and under the
        /// routed experts'.
        r_q8: f64,
        r_exp: f64,
        /// The RMS-relative distance of our normed input to mainline's.
        eps: f64,
    }

    fn ffn_in(ours: &[f32], theirs: &[f32], g: &[f32], eps: f32) -> Result<FfnIn, GateError> {
        let x_m = rms_normed(theirs, g, eps);
        let x_o = rms_normed(ours, g, eps);
        Ok(FfnIn {
            r_q8: rounding_ratio(&x_m, Q8_0_RULES)?,
            r_exp: rounding_ratio(&x_m, EXPERT_RULES)?,
            eps: rel2(&x_o, &x_m),
            x_o,
            x_m,
        })
    }

    /// A row as f64.
    fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    /// One node's distance ([`rel`]) against its band, counted into `tally`
    /// and the layer's `seen`, a FAIL line past the band; returns the
    /// distance.
    fn judge(
        clause: &str,
        (what, l, p): (&str, usize, usize),
        (got, want): (&[f32], &[f32]),
        band: f64,
        tally: &mut Tally,
        seen: &mut Vec<(String, usize, f64)>,
    ) -> f64 {
        let e = rel(got, want);
        tally.compared += 1;
        if e > band {
            println!("{clause}: {what}-{l} p{p} rel {e:.4} > {band:.4} FAIL");
            tally.ok = false;
        }
        tally.worst = tally.worst.max(e);
        seen.push((what.to_string(), p, e));
        e
    }

    /// The latent attention of every head on the host in f64: head `h`'s
    /// query row `q[h·w..]` (the flash row, SSMax applied) against `rows`
    /// (each `w` values in the flash row's `[rope | latent]` order), the
    /// softmax of `scale·q·k` over the rows, then the weighted sum of the
    /// rows' latent (their last `latent` values) — the body's attention with
    /// no rounding.
    fn host_attention(
        q: &[f32],
        rows: &[Vec<f32>],
        (w, latent, scale): (usize, usize, f64),
    ) -> Vec<f64> {
        let mut o = Vec::with_capacity(q.len() / w * latent);
        for qh in q.chunks(w) {
            let s: Vec<f64> = rows
                .iter()
                .map(|k| {
                    scale
                        * qh.iter()
                            .zip(k)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for i in w - latent..w {
                o.push(
                    rows.iter()
                        .zip(&e)
                        .map(|(k, p)| p * f64::from(k[i]))
                        .sum::<f64>()
                        / z,
                );
            }
        }
        o
    }

    /// What the host's f64 attention ([`host_attention`]) of our query reads
    /// at this position: `carry`, the distance the layer's cache rows carry
    /// into its output ([`head_rel`] of the attention over our rows against
    /// it over mainline's, `Kcur` through f16 as its cache holds them), and
    /// `own`, our tapped output's distance to the attention over our own rows
    /// ([`rel64`]; the flash's own rule, read where it fails).
    struct RowsTerm {
        carry: Option<f64>,
        own: f64,
    }

    /// [`RowsTerm`] of layer `l` at `at`: every row the position reads, the
    /// whole prefix below the top-k, the list's rows past it. Mainline's rows
    /// are in view only in a set of every position (the carry is `None` in a
    /// one-row set, whose earlier rows the dump does not hold); `None` on a
    /// layer that reads another layer's cache (the tap holds an owner's).
    fn rows_term(
        hp: &Hparams,
        (l, kind): (usize, &LayerCase),
        node: &LayerNodes,
        at: &At<'_>,
    ) -> Result<Option<RowsTerm>, GateError> {
        if kind.cache != l {
            return Ok(None);
        }
        let (latent, rope) = (hp.kv_lora, if kind.ropes { hp.rope_dims } else { 0 });
        let w = latent + rope;
        let n = at.pos + 1;
        let picked: Vec<usize> = if n <= hp.indexer_top_k {
            (0..n).collect()
        } else {
            node.list
                .get(..hp.indexer_top_k)
                .ok_or_else(|| format!("layer {l}: a list of {} rows", node.list.len()))?
                .iter()
                .map(|&r| r as usize)
                .collect()
        };
        if picked.iter().any(|&j| j >= n)
            || node.cache16.len() < n * w
            || !node.q_cur.len().is_multiple_of(w)
        {
            return Err(format!(
                "layer {l}: the tapped cache ({} values), query ({}) or list is not of {n} rows of {w}",
                node.cache16.len(),
                node.q_cur.len()
            )
            .into());
        }
        let ours: Vec<Vec<f32>> = picked
            .iter()
            .map(|&j| {
                node.cache16[j * w..(j + 1) * w]
                    .iter()
                    .map(|&b| half_to_f32(b))
                    .collect()
            })
            .collect();
        let scale = 1.0 / (hp.head_k as f64).sqrt();
        let o_ours = host_attention(&node.q_cur, &ours, (w, latent, scale));
        let att: Vec<f64> = node.att.iter().map(|&v| f64::from(v)).collect();
        let own = rel64(&att, &o_ours);
        if at.rows == 1 || n > hp.indexer_top_k {
            return Ok(Some(RowsTerm { carry: None, own }));
        }
        let name = format!("Kcur-{l}");
        let (kw, rows) = at.dims(&at.man.tensor(&name, 0)?.ne);
        let dump = tap_at(at.man, &name, 0)?;
        if rows != at.rows || dump.len() != kw * rows || kw < w {
            return Err(format!(
                "{name}: {rows} rows of {kw} values, want {} of {w}",
                at.rows
            )
            .into());
        }
        let theirs: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                let r = &dump[j * kw..(j + 1) * kw];
                let row = if kind.ropes {
                    rope_last(r, kw, rope)
                } else {
                    r[..latent].to_vec()
                };
                row.iter()
                    .map(|&v| half_to_f32(f32_to_f16_bits(v)))
                    .collect()
            })
            .collect();
        let o_theirs = host_attention(&node.q_cur, &theirs, (w, latent, scale));
        Ok(Some(RowsTerm {
            carry: Some(head_rel(&o_ours, &o_theirs, latent)),
            own,
        }))
    }

    /// One FFN block on the host over three inputs, every weight row
    /// dequantized from the file once: the f64 block on our normed input
    /// (`ours`) and on mainline's (`exact`), and mainline's own rule on its
    /// input (`rule`: its CPU build's q8_0 rows of the input and of the
    /// SwiGLU output, the products summed exactly), with `h`, mainline's
    /// SwiGLU output before its q8_0 rows (the down projection's input).
    struct FfnHost {
        ours: Vec<f64>,
        exact: Vec<f64>,
        rule: Vec<f64>,
        h: Vec<f64>,
    }

    /// Tensor `name`'s type, columns, rows and bytes; a tensor of more than
    /// two dimensions is refused by name.
    fn matrix<'f>(
        file: &'f Split,
        name: &str,
    ) -> Result<(GgmlType, usize, usize, usize, &'f [u8]), GateError> {
        let (_, info) = file
            .find(name)
            .ok_or_else(|| format!("{name}: not in the file"))?;
        if info.dims.len() < 2 || info.dims[2..].iter().any(|&d| d != 1) {
            return Err(format!("{name}: dims {:?}, want a matrix", info.dims).into());
        }
        let (ty, cols, rows) = (info.ty, info.dims[0] as usize, info.dims[1] as usize);
        let blck = ty
            .blck_size()
            .ok_or_else(|| format!("{name}: no block size"))? as usize;
        let tsz = ty
            .type_size()
            .ok_or_else(|| format!("{name}: no type size"))? as usize;
        let bytes = file.data(name)?;
        if blck == 0 || !cols.is_multiple_of(blck) || bytes.len() < cols / blck * tsz * rows {
            return Err(format!("{name}: {} bytes for {rows} rows of {cols}", bytes.len()).into());
        }
        Ok((ty, cols, rows, cols / blck * tsz, bytes))
    }

    /// `x`'s q8_0 round trip (mainline's CPU rows of a Q8_0 matmul's input).
    fn q8_0_rows(x: &[f64]) -> Vec<f64> {
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut o = vec![0.0f32; xf.len()];
        quantize_row_q8_0_roundtrip(&xf, &mut o);
        o.iter().map(|&v| f64::from(v)).collect()
    }

    /// [`FfnHost`] of the FFN whose gate, up and down weights are `names`,
    /// on our normed input `x_o` and mainline's `x_m`.
    fn ffn_host(
        file: &Split,
        names: [String; 3],
        x_o: &[f64],
        x_m: &[f64],
    ) -> Result<FfnHost, GateError> {
        let [gate, up, down] = names;
        let (gt, gc, gr, gb, gw) = matrix(file, &gate)?;
        let (ut, uc, ur, ub, uw) = matrix(file, &up)?;
        let (dt, dc, dr, db, dw) = matrix(file, &down)?;
        if gc != x_m.len() || uc != gc || ur != gr || dc != gr || dr != gc || x_o.len() != gc {
            return Err(format!(
                "{gate} {gc}x{gr}, {up} {uc}x{ur}, {down} {dc}x{dr}: not one FFN over {} values",
                x_m.len()
            )
            .into());
        }
        fn dot(w: &[f32], x: &[f64]) -> f64 {
            w.iter().zip(x).map(|(&a, &b)| f64::from(a) * b).sum()
        }
        let silu = |v: f64| v / (1.0 + (-v).exp());
        let xq = q8_0_rows(x_m);
        let (mut wg, mut wu) = (vec![0.0f32; gc], vec![0.0f32; gc]);
        let (mut h_o, mut h_m, mut h) = (
            Vec::with_capacity(gr),
            Vec::with_capacity(gr),
            Vec::with_capacity(gr),
        );
        for r in 0..gr {
            dequant_row(gt, &gw[r * gb..(r + 1) * gb], &mut wg)?;
            dequant_row(ut, &uw[r * ub..(r + 1) * ub], &mut wu)?;
            h_o.push(silu(dot(&wg, x_o)) * dot(&wu, x_o));
            h_m.push(silu(dot(&wg, x_m)) * dot(&wu, x_m));
            h.push(silu(dot(&wg, &xq)) * dot(&wu, &xq));
        }
        let hq = q8_0_rows(&h);
        let mut wd = vec![0.0f32; dc];
        let mut out = FfnHost {
            ours: Vec::with_capacity(dr),
            exact: Vec::with_capacity(dr),
            rule: Vec::with_capacity(dr),
            h,
        };
        for r in 0..dr {
            dequant_row(dt, &dw[r * db..(r + 1) * db], &mut wd)?;
            out.ours.push(dot(&wd, &h_o));
            out.exact.push(dot(&wd, &h_m));
            out.rule.push(dot(&wd, &hq));
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- sites

    /// The quantizer of an activation row to the q8_1 form every K-quant gemv
    /// reads.
    const QUANTIZE: &str = "q3k_quantize_q8_1";

    /// The entries whose count follows the file's types: each is counted
    /// (zero where the file's types launch none), so a launch the table does
    /// not expect is a mismatch, never an uncounted entry.
    const TYPED: [&str; 20] = [
        "norm_quant",
        "rms_norm",
        QUANTIZE,
        "q3k_gemv",
        "q4k_gemv",
        "q4k_gemv_prefetch",
        "q4k_gemv_sel",
        "q4k_gemv_sel_prefetch",
        "q5k_gemv_sel",
        "q6k_gemv",
        "q6k_gemv_sel",
        "iq4_xs_gemv_sel",
        "q8_0_gemv",
        "q8_0_gemv_heads",
        "f32_gemv",
        "ds41_shexp_gate_up",
        "kq_gate_up_act_q4k",
        "kq_gate_up_act_q5k",
        "gemm_swiglu_quant",
        "swiglu_rule_rows",
    ];

    /// One tensor as the step launches it: the launch the plan's site table
    /// names for its type, the type, and the width of the rows it reads.
    #[derive(Clone, Copy, Debug)]
    struct Site {
        launch: Launch,
        ty: GgmlType,
        k: usize,
    }

    /// The sites of one layer that the structure clause counts by type.
    #[derive(Clone, Debug)]
    struct LayerSites {
        q: Site,
        o: Site,
        /// `attn_kv_a_mqa`, on a layer that owns its cache.
        kv_a: Option<Site>,
        k_b: Site,
        v_b: Site,
        /// A full indexer layer's query projection, key projection and head
        /// weights.
        idx: Option<[Site; 3]>,
        /// The gate, up and down of the block the layer's feed-forward half
        /// runs whole: the dense lead's FFN, or a routed layer's shared expert.
        block: [Site; 3],
    }

    /// Every layer's sites and the head's, read from the file's header before
    /// the load takes the file.
    struct Sites {
        layers: Vec<LayerSites>,
        head: Site,
    }

    /// `names`, each counted once more in `out`.
    fn add(out: &mut BTreeMap<String, usize>, names: impl IntoIterator<Item = String>) {
        for name in names {
            *out.entry(name).or_default() += 1;
        }
    }

    /// The dense entry the site table names for `site`; any other launch is
    /// refused by name.
    fn dense(what: &str, site: &Site) -> Result<SiteEntry, GateError> {
        match site.launch {
            Launch::Dense(entry) => Ok(entry),
            other => Err(format!(
                "{what} is {}: the site table names {other:?}, a dense gemv site was wanted",
                site.ty
            )
            .into()),
        }
    }

    /// The entry the gemv of a dense site of `entry` over rows of `k` values
    /// launches (`site::kgemv` for the K-quants and the i-quant over the q8_1
    /// rows, `site::gemv` for the f32-row forms); `prefetch` is the card's
    /// `row_prefetch`, which names the Q4_K entries. An entry this gate's
    /// table has no name for is refused by name.
    fn gemv_name(entry: SiteEntry, k: usize, prefetch: bool) -> Result<String, GateError> {
        let name = match entry {
            // An odd super-block count runs the stack-of-one entry.
            SiteEntry::Q6k if (k / 256) % 2 == 1 => "q6k_gemv_sel",
            SiteEntry::Q6k => "q6k_gemv",
            SiteEntry::Q4k if prefetch => "q4k_gemv_prefetch",
            SiteEntry::Q4k => "q4k_gemv",
            SiteEntry::Q3k => "q3k_gemv",
            SiteEntry::Q5k => "q5k_gemv_sel",
            SiteEntry::Iq4xs => "iq4_xs_gemv_sel",
            SiteEntry::Q8_0 => "q8_0_gemv",
            SiteEntry::F32 => "f32_gemv",
            other => {
                return Err(format!(
                    "the dense site entry {other:?} has no kernel name in this gate's table"
                )
                .into());
            }
        };
        Ok(name.to_string())
    }

    /// The launches of `Site::project` for a dense site of `entry` over rows
    /// of `k` values: the q8_1 quantization the entry reads, then its gemv.
    fn project(entry: SiteEntry, k: usize, prefetch: bool) -> Result<Vec<String>, GateError> {
        match entry.reads_at(k as u64) {
            SiteAct::Q8x128 => Ok(vec![QUANTIZE.to_string(), gemv_name(entry, k, prefetch)?]),
            SiteAct::Q8x32 | SiteAct::F32 => Ok(vec![gemv_name(entry, k, prefetch)?]),
            SiteAct::Q8Blocks32 => Err(format!(
                "the dense site entry {entry:?} reads Q8Blocks32 rows: no kernel names in this \
                 gate's table"
            )
            .into()),
        }
    }

    /// The entry the down `_sel` launch of `family` runs over the heads'
    /// stack of file type `ty` (`attn_v_b` read as experts).
    fn sel_name(family: SelUp, ty: GgmlType, prefetch: bool) -> Result<String, GateError> {
        Ok(match family {
            SelUp::Q4k if prefetch => "q4k_gemv_sel_prefetch".to_string(),
            SelUp::Q4k => "q4k_gemv_sel".to_string(),
            SelUp::Q5k => "q5k_gemv_sel".to_string(),
            SelUp::Q6k => "q6k_gemv_sel".to_string(),
            SelUp::IQuant => format!(
                "{}_gemv_sel",
                ty.name()
                    .ok_or_else(|| format!("{ty:?} has no name"))?
                    .to_ascii_lowercase()
            ),
        })
    }

    /// The launches of a gated block (a dense lead's FFN or a shared expert)
    /// by its three sites' entries, in the order the common block enqueues
    /// them: the q8_0 pair (`ds41_shexp_gate_up`), or the row's q8_1 form and
    /// either the fused `_sel` (one Q4_K or Q5_K type for gate and up) or a
    /// gemv each and the SwiGLU (written into the down's q8_1 blocks where the
    /// down reads them, else into f32 rows), then the down by its type.
    fn block_launches(block: &[Site; 3], prefetch: bool) -> Result<Vec<String>, GateError> {
        let [gate, up, down] = [
            dense("the block's gate", &block[0])?,
            dense("the block's up", &block[1])?,
            dense("the block's down", &block[2])?,
        ];
        let [gk, uk, dk] = [block[0].k, block[1].k, block[2].k];
        let q128 = |entry: SiteEntry| entry.reads() == SiteAct::Q8x128;
        let mut out: Vec<String> = Vec::new();
        match (gate, up) {
            (SiteEntry::Q8_0, SiteEntry::Q8_0) => out.push("ds41_shexp_gate_up".to_string()),
            (g, u) if q128(g) && q128(u) => {
                out.push(QUANTIZE.to_string());
                match (g == u, g) {
                    (true, SiteEntry::Q4k) => out.push("kq_gate_up_act_q4k".to_string()),
                    (true, SiteEntry::Q5k) => out.push("kq_gate_up_act_q5k".to_string()),
                    _ => {
                        out.push(gemv_name(g, gk, prefetch)?);
                        out.push(gemv_name(u, uk, prefetch)?);
                        if q128(down) {
                            out.push("gemm_swiglu_quant".to_string());
                            out.push(gemv_name(down, dk, prefetch)?);
                            return Ok(out);
                        }
                        out.push("swiglu_rule_rows".to_string());
                    }
                }
            }
            (g, u) => {
                return Err(format!(
                    "the block's gate {g:?} and up {u:?} are no pair of this gate's table (two \
                     Q8_0 or two sites over the q8_1 rows)"
                )
                .into());
            }
        }
        if q128(down) {
            out.push(QUANTIZE.to_string());
            out.push(gemv_name(down, dk, prefetch)?);
        } else if down == SiteEntry::Q8_0 {
            out.push("q8_0_gemv".to_string());
        } else {
            return Err(format!(
                "the block's down {down:?} has no kernel name in this gate's table"
            )
            .into());
        }
        Ok(out)
    }

    impl Sites {
        /// The sites of every layer of `hp` and the head, in `file`: each
        /// tensor's type from the header, its launch from the site table
        /// ([`sites::launch`]); a tensor the table refuses ends the gate by
        /// name.
        fn read(file: &Split, hp: &Hparams) -> Result<Sites, GateError> {
            let site = |name: String| -> Result<Site, GateError> {
                let (_, t) = file
                    .find(&name)
                    .ok_or_else(|| format!("{name} is not in the file"))?;
                let k = *t
                    .dims
                    .first()
                    .ok_or_else(|| format!("{name} has no dimension"))?;
                Ok(Site {
                    launch: sites::launch(&name, t.ty)?,
                    ty: t.ty,
                    k: usize::try_from(k)?,
                })
            };
            let cases = layer_cases(hp);
            let mut layers = Vec::with_capacity(hp.n_layer);
            for (l, kind) in cases.iter().enumerate() {
                let at = |stem: &str| site(blk(l, stem));
                let block = if kind.dense {
                    [
                        at(stems::FFN_GATE)?,
                        at(stems::FFN_UP)?,
                        at(stems::FFN_DOWN)?,
                    ]
                } else {
                    [
                        at(stems::FFN_GATE_SHEXP)?,
                        at(stems::FFN_UP_SHEXP)?,
                        at(stems::FFN_DOWN_SHEXP)?,
                    ]
                };
                layers.push(LayerSites {
                    q: at(stems::ATTN_Q)?,
                    o: at(stems::ATTN_OUTPUT)?,
                    kv_a: if kind.owns {
                        Some(at(stems::ATTN_KV_A_MQA)?)
                    } else {
                        None
                    },
                    k_b: at(stems::ATTN_K_B)?,
                    v_b: at(stems::ATTN_V_B)?,
                    idx: if kind.full {
                        Some([
                            at(stems::INDEXER_ATTN_Q_B)?,
                            at(stems::INDEXER_ATTN_K)?,
                            at(stems::INDEXER_PROJ)?,
                        ])
                    } else {
                        None
                    },
                    block,
                });
            }
            Ok(Sites {
                layers,
                head: site(stems::output())?,
            })
        }

        /// The kernel entries the file's types decide, with their counts in
        /// one step: every [`TYPED`] name (zero where nothing launches it) and
        /// any other the types name. The attention norm is `norm_quant` on a
        /// layer whose query reads the q8_1 rows and `rms_norm` otherwise; the
        /// rest of the walk is the module doc's.
        fn kernels(&self, prefetch: bool) -> Result<BTreeMap<String, usize>, GateError> {
            let mut out: BTreeMap<String, usize> =
                TYPED.iter().map(|n| ((*n).to_string(), 0)).collect();
            for ls in &self.layers {
                let q = dense("attn_q", &ls.q)?;
                if q.reads_at(ls.q.k as u64) == SiteAct::Q8x128 {
                    add(
                        &mut out,
                        ["norm_quant".to_string(), gemv_name(q, ls.q.k, prefetch)?],
                    );
                } else {
                    add(&mut out, ["rms_norm".to_string()]);
                    add(&mut out, project(q, ls.q.k, prefetch)?);
                }
                if let Some(kv_a) = &ls.kv_a {
                    add(
                        &mut out,
                        project(dense("attn_kv_a_mqa", kv_a)?, kv_a.k, prefetch)?,
                    );
                }
                if let Some(idx) = &ls.idx {
                    for (what, s) in ["indexer.attn_q_b", "indexer.attn_k", "indexer.proj"]
                        .into_iter()
                        .zip(idx)
                    {
                        add(&mut out, project(dense(what, s)?, s.k, prefetch)?);
                    }
                }
                if ls.k_b.launch != Launch::KUp {
                    return Err(format!(
                        "attn_k_b is {}: the site table names {:?}, the per-head gemv was wanted",
                        ls.k_b.ty, ls.k_b.launch
                    )
                    .into());
                }
                add(&mut out, ["q8_0_gemv_heads".to_string()]);
                match ls.v_b.launch {
                    Launch::VUp(UpProj::Planes) => add(&mut out, ["q8_0_gemv_heads".to_string()]),
                    Launch::VUp(UpProj::Sel(family)) => add(
                        &mut out,
                        [QUANTIZE.to_string(), sel_name(family, ls.v_b.ty, prefetch)?],
                    ),
                    other => {
                        return Err(format!(
                            "attn_v_b is {}: the site table names {other:?}, an up-projection \
                             was wanted",
                            ls.v_b.ty
                        )
                        .into());
                    }
                }
                add(
                    &mut out,
                    project(dense("attn_output", &ls.o)?, ls.o.k, prefetch)?,
                );
                add(&mut out, block_launches(&ls.block, prefetch)?);
            }
            let Launch::Head(head) = self.head.launch else {
                return Err(format!(
                    "output is {}: the site table names {:?}, the head was wanted",
                    self.head.ty, self.head.launch
                )
                .into());
            };
            let gemv = match head {
                HeadEntry::Q6k => "q6k_gemv",
                HeadEntry::Q5k => "q5k_gemv_sel",
                HeadEntry::Q4k if prefetch => "q4k_gemv_prefetch",
                HeadEntry::Q4k => "q4k_gemv",
                other => {
                    return Err(format!(
                        "the head entry {other:?} has no kernel names in this gate's table"
                    )
                    .into());
                }
            };
            add(
                &mut out,
                [
                    QUANTIZE.to_string(),
                    gemv.to_string(),
                    "argmax_fault".to_string(),
                ],
            );
            Ok(out)
        }
    }

    // ------------------------------------------------------------ structure

    /// The captured step's kernel entries against the list the layers derive
    /// (the module doc's) and the file's types name, and the span: one path,
    /// no node beside another.
    fn structure(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        let hp = &case.hp;
        let (card, prefetch) = {
            let (gpu, b) = body_of(m)?;
            (b.card_layers().len(), gpu.row_prefetch())
        };
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let list = m.step_graph_nodes()?;
        let span = m.step_graph_span()?;
        let hist = histogram(&list);
        let ([host, memop], _) = count_kinds(
            &list,
            [
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            ],
        );
        let kinds = layer_cases(hp);
        let count = |f: &dyn Fn(&LayerCase) -> bool| kinds.iter().filter(|k| f(k)).count();
        let layers = hp.n_layer;
        let full = count(&|k| k.full);
        let owners = count(&|k| k.owns);
        let rope_layers = count(&|k| k.ropes);
        let dense = count(&|k| k.dense);
        let routed = layers - dense;
        let got = |name: &str| hist.get(name).copied().unwrap_or(0);
        // The entries the file's types decide (the site table's, per tensor and
        // layer), then those a layer kind holds by the body's own launch list
        // (its module doc) and the common routed block's: the counts are per
        // full/shared/owner/reader/roped/dense/routed/card layer and the head's
        // own few.
        let mut counts = case.sites.kernels(prefetch)?;
        let mut put = |name: &str, n: usize| *counts.entry(name.to_string()).or_default() += n;
        // every layer's ffn_norm and the head's final norm, beside the
        // attention norms of the layers whose query does not read the q8_1
        // rows (counted by type).
        put("rms_norm", layers + 1);
        // the card leg's normed row a card layer.
        put(QUANTIZE, card);
        put("rope", rope_layers);
        put("kv_norm_rope_append", count(&|k| k.owns && k.ropes));
        put("latent_rms_append", count(&|k| k.owns && !k.ropes));
        put("ln_rows", full);
        // the index head's two permutations a roped full layer (a NoPE
        // index head has no rope rows to move), the flash row's rope
        // spans a roped one.
        put(
            "gather_pairs",
            2 * count(&|k| k.ropes && k.full) + rope_layers,
        );
        // the index queries' pair a full layer, the joined row's every.
        put("ssmax_rows", full + layers);
        put("ds41_index_key_input", full);
        put("ds41_indexer_score", full);
        put("ds41_indexer_topk", full);
        put("kv_gather_sel", rope_layers);
        // The flash over the gathered rows by the body's own rule
        // (`flash::segments_for` of the list's width): one launch where the
        // list fits one segment, else the split pair. All three names are
        // listed, so a step that runs the other form is red either way.
        let one_segment = bloomery_gpu::flash::segments_for(hp.indexer_top_k) == 1;
        let (single, split) = if one_segment {
            (rope_layers, 0)
        } else {
            (0, rope_layers)
        };
        put("flash_latent", single);
        put("flash_latent_mma", split);
        put("flash_merge", split);
        put("ds41_attn_seg_sel", layers - rope_layers);
        put("ds41_attn_merge", layers - rope_layers);
        // the attention's residual a layer, the dense lead's block add.
        put("add", layers + dense);
        put("softmax_router_fused", routed);
        put("ds41_ffn_handoff_4", routed);
        put("routed_shared_residual", routed);
        put("ds41_ffn_card_acc_4", card);
        put("softcap_rows", 1);
        let want: Vec<(String, usize)> = counts.into_iter().collect();
        // The card leg's three slot launches, whatever entries the stacks'
        // types name (the activation's quantization by the down's block, the
        // gate·up and the down): every `_sel` entry the list above does not
        // name, three a layer with card experts.
        let card_sel: usize = hist
            .iter()
            .filter(|(k, _)| k.contains("_sel") && !want.iter().any(|(n, _)| n == *k))
            .map(|(_, n)| *n)
            .sum();
        // Every kernel node is a listed entry or a card slot launch: an entry
        // the list does not name (a step that grew a launch, or a list that
        // lost one) is red, not passed over.
        let unlisted: Vec<String> = hist
            .iter()
            .filter(|(k, _)| !k.contains("_sel") && !want.iter().any(|(n, _)| n == *k))
            .map(|(k, n)| format!("{k}={n}"))
            .collect();
        let mut pass =
            host == 0 && memop == 2 * routed && card_sel == 3 * card && unlisted.is_empty();
        for (name, n) in &want {
            pass &= got(name) == *n;
        }
        pass &= span == nodes;
        let listed = want
            .iter()
            .filter(|(name, n)| *n != 0 || got(name) != 0)
            .map(|(name, n)| format!("{name}={}/{n}", got(name)))
            .chain([
                format!("card_sel={card_sel}/{}", 3 * card),
                format!("batch_mem_op={memop}/{}", 2 * routed),
                format!("unlisted={}/0 [{}]", unlisted.len(), unlisted.join(" ")),
                format!("span={span}/{nodes}"),
            ])
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "structure: layers={layers} ({dense} dense, {routed} routed, {card} with card \
             experts, {full} full indexer, {owners} cache owners, {rope_layers} roped), \
             graph_nodes={nodes} host={host}, entries (got/want) {listed} {}",
            verdict(pass)
        );
        let kernels = hist
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("structure: kernel entries {kernels}");
        Ok(pass)
    }

    // ------------------------------------------------- the own-contract ones

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
        let pass = same(&first, &second);
        println!(
            "{CLAUSE}: walk{PROMPT_LEN} twice, logits bit-identical {pass} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// A capture made before a `seed_depth` replays over the seeded rows:
    /// the seed refills the resident caches the capture recorded instead of
    /// replacing them, so the replayed step at the seeded depth equals an
    /// eager one there bit for bit. The depth sits past the fixture's top-k
    /// so the step selects its rows from the seeded pattern.
    fn seeded_replay(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "seeded_replay";
        let rows = PROMPT_LEN;
        let token = case.walk[0];
        m.set_mode(StepMode::Eager);
        m.reset()?;
        m.seed_depth(rows)?;
        let Some(eager) = or_fault(
            CLAUSE,
            "the eager step at the seeded depth",
            feed(m, &[token]),
        )?
        else {
            return Ok(false);
        };
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let nodes = m.capture_step()?;
        m.seed_depth(rows)?;
        let Some(replay) = or_fault(
            CLAUSE,
            "the replayed step over the seeded rows",
            feed(m, &[token]),
        )?
        else {
            return Ok(false);
        };
        let identical = same(&eager, &replay);
        let pass = identical && finite(&eager);
        println!(
            "{CLAUSE}: depth {rows}, a {nodes}-node capture seeded after the record, the replay's \
             token and logits bit-identical to the eager step's {identical}, finite {} {}",
            finite(&eager),
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
        let carries = same(&after_a, &after_b);
        let once = !same(&alone, &after_a);
        let pass = carries && once;
        println!(
            "{CLAUSE}: the last token after the walk and alone-from-reset differ {once}, the two \
             walks leave it bit-identical {carries} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    fn router_lane_nan(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "router_lane_nan";
        // The first routed layer, past the dense lead.
        let layer = case.hp.dense_lead;
        let (gpu, b) = body_of(m)?;
        let (raised, defined) = b.router_lane_nan(gpu, layer)?;
        let pass = raised && !defined;
        println!(
            "{CLAUSE}: layer {layer}'s routing over a planted NaN lane: raised {raised}, a \
             defined pick {defined} (want none) {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The selection rule: every full indexer layer's list is the host's f64
    /// top-k of the body's own scores (the kernel's own tie rule, a tie
    /// named where it cut), and every shared layer's list is its source's.
    fn selection_is_its_rule(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "selection_is_its_rule";
        let hp = &case.hp;
        let top_k = hp.indexer_top_k;
        // The check first: a walk read through a check that passes a tied cut
        // the higher rows took means nothing.
        let sound = selection_check_holds();
        println!(
            "{CLAUSE}: the check refuses a tied cut the higher rows took and names the one the \
             lower rows won {}",
            verdict(sound)
        );
        if !sound {
            return Ok(false);
        }
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        let mut named = 0usize;
        for at in top_k..PROMPT_LEN {
            // The walk to the position first: the selection the step at `at`
            // runs is over its own `at + 1` rows.
            m.reset()?;
            if or_fault(
                CLAUSE,
                "the walk to the position",
                feed(m, &case.walk[..at]),
            )?
            .is_none()
            {
                return Ok(false);
            }
            let ids = [case.walk[at]];
            let got = or_fault(CLAUSE, "a selecting step", feed(m, &ids))?;
            if got.is_none() {
                return Ok(false);
            }
            check_selection(m, hp, (CLAUSE, at), (&mut ok, &mut named))?;
        }
        let past = PROMPT_LEN.saturating_sub(top_k);
        if past == 0 {
            // A prompt no longer than top-k cuts nothing: the real file's cut is the deep clause's.
            println!(
                "{CLAUSE}: no position past top-k {top_k} in a {PROMPT_LEN}-id prompt, nothing \
                 cut here; nodes_near_mainlines_deep reads a cut"
            );
            return Ok(ok);
        }
        println!(
            "{CLAUSE}: {past} positions past top-k {top_k}, every full layer's set the rule's and \
             every shared layer's its source's ({named} threshold ties named) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// The selection rule over the step just run at position `at` (the
    /// caller's step, its lists and scores still resident): every full
    /// indexer layer's list is the host's f64 top-`min(at + 1, top_k)` of
    /// that layer's index scores over the rows `0..=at` as a set, the list
    /// being in row order and the rule's in rank order, and every shared
    /// layer's list is its source's, entry for entry. A set that differs is
    /// printed and clears `ok`, whether or not the rows it took tie the k-th
    /// score (a tie at the cut goes to the lower row); a cut that ran through
    /// a tie the lower rows won is counted in `named`.
    fn check_selection(
        m: &mut Model,
        hp: &Hparams,
        (clause, at): (&str, usize),
        (ok, named): (&mut bool, &mut usize),
    ) -> Result<(), GateError> {
        let (top_k, n_layer) = (hp.indexer_top_k, hp.n_layer);
        let (gpu, b) = body_of(m)?;
        let lists = b.lists(gpu, 0..n_layer)?;
        let mut scores: Vec<Option<Vec<f32>>> = vec![None; n_layer];
        for (l, score) in scores.iter_mut().enumerate() {
            if hp.indexer_full[l] {
                *score = Some(b.index_scores(gpu, l)?);
            }
        }
        for l in 0..n_layer {
            let (src, list) = &lists[l];
            let n = (at + 1).min(top_k);
            if hp.indexer_full[l] {
                let scores = scores[l].as_ref().unwrap();
                let rows = &scores[..(at + 1).min(scores.len())];
                match selection_check(rows, list, n) {
                    Ok(cut_a_tie) => *named += usize::from(cut_a_tie),
                    Err((got, want)) => {
                        // What differs: the lists where they are short enough to
                        // read, else the rows of each side the other lacks, and
                        // how many of them score the threshold's own value (the
                        // k-th ranked score), the mark of a tie cut the other way.
                        let lacks = |a: &[usize], b: &[usize]| -> Vec<usize> {
                            a.iter()
                                .filter(|r| b.binary_search(r).is_err())
                                .copied()
                                .collect()
                        };
                        let (missing, extra) = (lacks(&want, &got), lacks(&got, &want));
                        let ranked = rule_top(rows, n);
                        let threshold = rows[ranked[ranked.len() - 1]];
                        let at_threshold = missing
                            .iter()
                            .chain(&extra)
                            .filter(|&&r| rows[r] == threshold)
                            .count();
                        let detail = if n <= 64 {
                            format!("{got:?} vs {want:?}")
                        } else {
                            format!(
                                "{} rows of {n} missing, {} extra, {at_threshold} of them at the \
                                 threshold's score",
                                missing.len(),
                                extra.len()
                            )
                        };
                        println!(
                            "{clause}: layer {l} at position {at}: the sets differ: {detail} FAIL"
                        );
                        *ok = false;
                    }
                }
            } else if list[..n] != lists[*src].1[..n] {
                println!(
                    "{clause}: shared layer {l} at position {at}: its list is not source \
                     {src}'s FAIL"
                );
                *ok = false;
            }
        }
        Ok(())
    }

    /// The layer span's refusals, with the taps armed: an empty span, a
    /// reversed one and one that ends or starts past the model's last layer
    /// are each refused by name (a span read as the whole chain would run a
    /// node clause over layers it did not ask for), and the spans that fit
    /// are taken.
    fn layer_span_refusals(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "layer_span_refusals";
        const WHAT: &str = "mistrallarge4::set_layer_span";
        let n = case.hp.n_layer;
        let last = n - 1;
        m.set_mode(StepMode::Eager);
        if m.has_capture() {
            println!("{CLAUSE}: a capture exists FAIL");
            return Ok(false);
        }
        let (gpu, b) = body_of(m)?;
        b.arm_taps(gpu, true)?;
        let mut ok = true;
        let refused = [
            ("empty", last..last, "empty or reversed"),
            ("reversed", last..0, "empty or reversed"),
            ("past-the-end", 0..n + 1, "past the model's"),
            ("starting-past-the-end", n..n + 1, "past the model's"),
        ];
        for (what, span, says) in refused {
            let r = b.set_layer_span(Some(span.clone()));
            let pass = refused_by(&r, WHAT, says);
            println!(
                "{CLAUSE}: the {what} span {span:?}: {} {}",
                outcome(&r),
                verdict(pass)
            );
            ok &= pass;
        }
        // The last of these leaves the whole chain.
        for span in [Some(0..1), Some(0..n), Some(last..n), None] {
            let r = b.set_layer_span(span.clone());
            let pass = r.is_ok();
            println!(
                "{CLAUSE}: the span {span:?}: {} {}",
                outcome(&r),
                verdict(pass)
            );
            ok &= pass;
        }
        b.arm_taps(gpu, false)?;
        Ok(ok)
    }

    fn fault_word_clean(m: &mut Model, _case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "fault_word_clean";
        let fault = m.gpu().fault()?;
        let pass = fault.is_none() && m.poisoned().is_none();
        println!(
            "{CLAUSE}: word {fault:?}, poisoned {:?} {}",
            m.poisoned(),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- the oracle ones

    /// The fixture's node clause: sets a and b at the fixture's bands.
    fn nodes_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        nodes_against(m, case, ("nodes_near_mainlines", FIXTURE_BANDS), 2)
    }

    /// The real file's node clause: set a at the real tier's bands (set b's
    /// longer prompt stays under the real file's top-k as set a's does, so
    /// it adds no selected row).
    fn nodes_near_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        nodes_against(m, case, ("nodes_near_mainlines_real", REAL_BANDS), 1)
    }

    /// Where a comparison reads: the dump's rows of the per-position nodes and the
    /// position of the step in the walk.
    struct At<'a> {
        man: &'a RefManifest,
        /// The positions the dump holds a row of (a decode-step set holds one).
        rows: usize,
        /// This position's row in the dump.
        row: usize,
        /// The step's position in the walk: the cache row it appended and the
        /// rows its selection ran over.
        pos: usize,
    }

    impl At<'_> {
        /// A dump tensor's per-position width and row count. A one-row set
        /// holds the whole tensor as its row (`[576,64,1]` is one position of
        /// 36,864 values, not 64 of 576).
        fn dims(&self, ne: &[u64; 4]) -> (usize, usize) {
            if self.rows == 1 {
                (ne.iter().product::<u64>() as usize, 1)
            } else {
                dump_rows(ne)
            }
        }
    }

    /// What a node clause counts over its comparisons.
    struct Tally {
        ok: bool,
        compared: usize,
        worst: f64,
        worst_cos: f64,
    }

    impl Tally {
        fn new() -> Self {
            Tally {
                ok: true,
                compared: 0,
                worst: 0.0,
                worst_cos: 1.0,
            }
        }
    }

    /// Layer `l`'s `l_out` row at row `p` of the dump.
    fn l_out_row(
        man: &RefManifest,
        hidden: usize,
        l: usize,
        p: usize,
    ) -> Result<Vec<f32>, GateError> {
        let rows = tap_at(man, &format!("l_out-{l}"), 0)?;
        let at = p * hidden;
        Ok(rows
            .get(at..at + hidden)
            .ok_or("a row past the dump")?
            .to_vec())
    }

    /// One layer's tapped values (`node`, read after the step at `at`)
    /// against the dump's rows at `at`, read against `bands`: every
    /// comparison counts into `tally` and the layer's `seen` (node, position,
    /// distance). `deep` is the deep step: a list that crossed inside the
    /// indexer's band does not excuse the nodes it feeds.
    fn compare_layer(
        m: &mut Model,
        (clause, bands): (&str, Bands),
        case: &Case,
        (l, kind): (usize, &LayerCase),
        node: &LayerNodes,
        at: &At<'_>,
        (deep, tally, seen): (bool, &mut Tally, &mut Vec<(String, usize, f64)>),
    ) -> Result<(), GateError> {
        let hp = &case.hp;
        let Bands {
            expert: expert_band,
            dense: dense_band,
            attn: attn_band,
            cosine: cosine_floor,
            derived,
        } = bands;
        let (man, p, top_k) = (at.man, at.pos, hp.indexer_top_k);
        let (ropes, full, owns, dense) = (kind.ropes, kind.full, kind.owns, kind.dense);
        // A dump row by its own width and row count (`ne[0]` wide, the
        // file's values over it): the last layer's FFN chain (the
        // residual the head reads) the dump holds at the output
        // position alone, so every other position compares nothing
        // there (an empty want).
        let row = |what: &str, occ: u32| -> Result<Vec<f32>, GateError> {
            let name = format!("{what}-{l}");
            let (width, rows) = at.dims(&man.tensor(&name, occ)?.ne);
            let all = tap_at(man, &name, occ)?;
            assert_eq!(all.len(), width * rows, "{name}: {rows} rows");
            if rows == 1 {
                return Ok(if at.row + 1 == at.rows {
                    all[..width].to_vec()
                } else {
                    Vec::new()
                });
            }
            assert_eq!(rows, at.rows, "{name}: {rows} rows");
            Ok(all[at.row * width..(at.row + 1) * width].to_vec())
        };
        // An i32 node's logical twin (the expert ids), this position.
        let ids_row = |what: &str| -> Result<Vec<u32>, GateError> {
            let name = format!("{what}-{l}");
            let (width, rows) = at.dims(&man.tensor(&name, 0)?.ne);
            let all = ref_ints(man, &name, 0, RowKind::Tensor, Layout::Logical)?;
            assert_eq!(all.len(), width * rows, "{name}: {rows} rows");
            let ids = |v: i64| u32::try_from(v).unwrap_or(u32::MAX);
            if rows == 1 {
                return Ok(if at.row + 1 == at.rows {
                    all[..width].iter().map(|&v| ids(v)).collect()
                } else {
                    Vec::new()
                });
            }
            assert_eq!(rows, at.rows, "{name}: {rows} rows");
            Ok(all[at.row * width..(at.row + 1) * width]
                .iter()
                .map(|&v| ids(v))
                .collect())
        };
        // The selection against the dump's own `top_k`, past the
        // boundary where the node stops being the argsort's identity
        // padding (a shared layer's is its source's list, scored on
        // the source's head): a set that crossed inside the
        // indexer's own band (the k-th gap under the scores' peak
        // times it) excuses the attention-fed nodes at this position —
        // except in the deep step, whose cut moves a few rows at its
        // threshold on any rounding: there the nodes stay compared and
        // the moved count is the frontier line's.
        let mut idx_crossed = false;
        if at.pos + 1 > top_k {
            let tk = dumped_list(man, l, hp.n_layer)?;
            let name = format!("top_k-{tk}");
            let (width, rows) = at.dims(&man.tensor(&name, 0)?.ne);
            let all = ref_ints(man, &name, 0, RowKind::Tensor, Layout::Flat)?;
            assert_eq!(all.len(), width * rows, "{name}: {rows} rows");
            let want: Vec<u32> = all[at.row * width..(at.row + 1) * width]
                .iter()
                .map(|&v| u32::try_from(v).unwrap_or(u32::MAX))
                .collect();
            let mut got = node.list[..top_k].to_vec();
            got.sort_unstable();
            let mut want = want;
            want.sort_unstable();
            tally.compared += 1;
            // The dump's rows the run's list lacks (both are sorted).
            let outside: Vec<u32> = want
                .iter()
                .filter(|id| got.binary_search(id).is_err())
                .copied()
                .collect();
            seen.push((
                "top_k moved".to_string(),
                p,
                outside.len() as f64 / top_k as f64,
            ));
            if got != want {
                let scores = {
                    let (gpu, b) = body_of(m)?;
                    b.index_scores(gpu, kind.indexer)?
                };
                let live = &scores[..at.pos + 1];
                let peak = live.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
                let mut order = live.to_vec();
                order.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let kth = order[top_k - 1];
                let band = dense_band as f32 * peak;
                let excused =
                    !outside.is_empty() && outside.iter().all(|&r| kth - live[r as usize] <= band);
                // The lists themselves where they are short enough to read.
                let sets = if top_k <= 64 {
                    format!("{got:?} != {want:?}")
                } else {
                    format!("{} of {top_k} rows moved", outside.len())
                };
                if excused {
                    idx_crossed = !deep;
                    println!(
                        "{clause}: top_k-{tk} p{p} {sets} inside the indexer's band (k-th gap \
                         {:.2e} ≤ {:.2e}): the attention-fed nodes {}",
                        kth - live[outside[0] as usize],
                        band,
                        if deep { "stay compared" } else { "skip here" }
                    );
                } else {
                    println!("{clause}: top_k-{tk} p{p} {sets} FAIL");
                    tally.ok = false;
                }
            }
        }
        // The derived bands' inputs (the error-model section): mainline's
        // normed row at this position and the rounding ratios of the
        // projections that read it, each 1 where the bands are constants.
        // The latent's and the index key's Q8_0 projections read the normed
        // row; the query's Q6_K projection reads it, and its Q8_0 absorption
        // the projected query's blocks.
        let ratio = |x: &[f64], rules: &[usize]| -> Result<f64, GateError> {
            if derived {
                rounding_ratio(x, rules)
            } else {
                Ok(1.0)
            }
        };
        let attn_m = if derived {
            row("attn_norm", 0)?
        } else {
            Vec::new()
        };
        let x_attn = f64s(&attn_m);
        let r_q8 = ratio(&x_attn, Q8_0_RULES)?;
        let r_q = ratio(&x_attn, Q6_K_RULES)?.max(ratio(&f64s(&node.q), Q8_0_RULES)?);
        // The FFN's normed inputs ([`FfnIn`]), where the bands are derived and
        // the dump holds the row.
        let ffn = || -> Result<Option<FfnIn>, GateError> {
            let inp = row("ffn_inp", 0)?;
            if !derived || inp.is_empty() {
                return Ok(None);
            }
            let g = case
                .gains
                .ffn
                .get(l)
                .ok_or("a layer past the ffn_norm gains")?;
            Ok(Some(ffn_in(&node.ffn_inp, &inp, g, hp.rms_eps)?))
        };
        let pairs: Vec<(&str, Vec<f32>, f64)> = vec![
            ("attn_norm", node.attn_norm.clone(), dense_band),
            ("ffn_inp", node.ffn_inp.clone(), dense_band),
        ];
        // The attention's own output against the reference's
        // `fattn_mla` (the heads' values up-projected): the dump's
        // `[head_v, n_pos, heads]` layout, head-major a position.
        // The card's Q6_K query and Q8_0 key projections round
        // differently from mainline's, and the softmax amplifies
        // that onto this row. A position whose selection crossed
        // compares nothing here. Derived, the constant is read at the
        // louder of the up-projection's input rounding (its Q8_0 blocks of
        // the attention row) and the query's, plus the distance the cache
        // rows carry ([`rows_term`]; not in view in a one-row set).
        if !idx_crossed {
            let all = tap_at(man, &format!("fattn_mla-{l}"), 0)?;
            let (hv, tokens, n_head) = (hp.head_v, at.rows, hp.n_head);
            assert_eq!(all.len(), hv * tokens * n_head);
            let want: Vec<f32> = (0..n_head * hv)
                .map(|i| {
                    let (h, d) = (i / hv, i % hv);
                    all[h * hv * tokens + at.row * hv + d]
                })
                .collect();
            // A set of every position reads the rows for the band; a
            // one-row set (the deep step, a cut of thousands of rows) reads
            // its own-rule line only where the row passes the constant.
            let rows = if derived && (at.rows > 1 || rel(&node.av, &want) > attn_band) {
                rows_term(hp, (l, kind), node, at)?
            } else {
                None
            };
            let carry = rows.as_ref().and_then(|t| t.carry).unwrap_or(0.0);
            let band = attn_band * ratio(&f64s(&node.att), Q8_0_RULES)?.max(r_q) + carry;
            let e = judge(
                clause,
                ("fattn_mla", l, p),
                (&node.av, &want),
                band,
                tally,
                seen,
            );
            if e > band {
                print_samples(clause, &node.av, &want);
            }
            if let Some(t) = rows.as_ref().filter(|_| e > attn_band) {
                println!(
                    "{clause}: fattn_mla-{l} p{p} rel {e:.4} (band {band:.4}): the cache rows \
                     carry {}, the flash against its own rule over our rows {:.4}",
                    t.carry
                        .map_or_else(|| "(not in view)".to_string(), |c| format!("{c:.4}")),
                    t.own
                );
            }
        }
        for (what, got, band) in pairs {
            let want = row(what, 0)?;
            if want.is_empty() || (idx_crossed && what == "ffn_inp") {
                continue;
            }
            let e = rel(&got, &want);
            tally.compared += 1;
            if e > band {
                println!("{clause}: {what}-{l} p{p} rel {e:.4} > {band} FAIL");
                print_samples(clause, &got, &want);
                tally.ok = false;
            }
            tally.worst = tally.worst.max(e);
            seen.push((what.to_string(), p, e));
        }
        // The scaled query: a rope layer's card row is `[rope |
        // latent]` a head, the reference's joined `[latent | rope]`;
        // a NoPE layer's dump row is the latent with a zero rope part
        // after it (`pad_rope`), the card's row the latent alone.
        {
            let head = hp.kv_lora + hp.rope_dims;
            let want = if ropes {
                rope_last(&row("Qcur_ssmax", 0)?, head, hp.rope_dims)
            } else {
                // The NoPE row is the absorbed alone, the dump's
                // zero rope part its tail.
                row("Qcur_ssmax", 0)?
                    .chunks(head)
                    .flat_map(|h| h[..hp.kv_lora].iter().copied())
                    .collect()
            };
            let band = dense_band * r_q;
            let e = judge(
                clause,
                ("Qcur_ssmax", l, p),
                (&node.q_cur, &want),
                band,
                tally,
                seen,
            );
            if e > band {
                print_samples(clause, &node.q_cur, &want);
            }
        }
        if full {
            // The index queries and the key's projection stage: the
            // dump holds the file's `[rope | nope]` head order, a
            // roped layer's card row the kernel's tail order (the
            // load-time move's), a NoPE layer's the file's own.
            // Derived, the queries' Q6_K projection reads the query
            // LayerNorm's row of mainline's normed row, the key's Q8_0
            // projection the normed row itself.
            let x_ln = match (derived, case.gains.idx_q.get(l)) {
                (false, _) => Vec::new(),
                (true, Some(Some((w, b)))) => layer_normed(&attn_m, (w, b), hp.idx_norm_eps),
                (true, _) => {
                    return Err(format!("layer {l}: a full layer with no indexer q_norm").into());
                }
            };
            let taps: Vec<(&str, Vec<f32>, f64)> = vec![
                (
                    "indexer_q_ssmax",
                    node.qi.clone(),
                    dense_band * ratio(&x_ln, Q6_K_RULES)?,
                ),
                ("indexer_k", node.kidx.clone(), dense_band * r_q8),
            ];
            for (what, got, band) in taps {
                let rope = if ropes { hp.rope_dims } else { 0 };
                let mut want = row(what, 0)?;
                model::arch::mistrallarge4::rows::rope_rows_to_tail(&mut want, hp.indexer_d, rope);
                let e = judge(clause, (what, l, p), (&got, &want), band, tally, seen);
                if e > band {
                    print_samples(clause, &got, &want);
                }
            }
            {
                let (what, band) = ("indexer_weights", dense_band);
                let want = row(what, 0)?;
                let e = rel(&node.wi, &want);
                tally.compared += 1;
                if e > band {
                    println!("{clause}: {what}-{l} p{p} rel {e:.4} > {band} FAIL");
                    tally.ok = false;
                }
                tally.worst = tally.worst.max(e);
                seen.push((what.to_string(), p, e));
            }
        }
        if owns {
            let (rope_dims, latent) = (hp.rope_dims, hp.kv_lora);
            // A NoPE owner's normed latent and appended row live only
            // in its f16 cache: the tap's whole-cache copy, row p.
            let cache_row = || -> Vec<f32> {
                node.cache16[p * latent..(p + 1) * latent]
                    .iter()
                    .map(|&b| half_to_f32(b))
                    .collect()
            };
            // `kv_cmpr`'s two occurrences: the projection's latent
            // (the dump's occ-0 row is the latent alone, 512 wide;
            // the card's joined row is `[latent | rope]`, and the
            // rope span beside it is the `Kcur` row's) and its
            // normed latent. [`rel`] refuses a length mismatch, so
            // the tap must hand it the span the oracle holds.
            let normed = if ropes {
                node.kv_s[..latent].to_vec()
            } else {
                cache_row()
            };
            let raw = node.kv_a[..latent].to_vec();
            if derived {
                // PIN(2026-11-02): derived, the projection's latent (its Q8_0
                // projection of the normed row) takes the constant at the row's
                // rounding ratio, and the normed latent a band per value
                // ([`latent_band`]: the RMS norm's rule on each engine's own
                // latent, so the latent's distance is carried through each
                // value's own gain, plus each engine's rounding of the rule);
                // the rows whose peak-relative distance passes the constant
                // print the reading that names the side. `Kcur`'s latent part
                // is that same row (both engines write the normed latent into
                // it unchanged), so `Kcur` compares its rope part alone, the
                // projection's rope values roped, at the latent's band.
                let y_m = row("kv_cmpr", 0)?;
                let z_m = row("kv_cmpr", 1)?;
                let g = case.gains.kv(l)?;
                if y_m.len() != latent || z_m.len() != latent || g.len() != latent {
                    return Err(format!(
                        "kv_cmpr-{l}: rows of {} and {} values and a gain of {}, want {latent}",
                        y_m.len(),
                        z_m.len(),
                        g.len()
                    )
                    .into());
                }
                let band0 = dense_band * r_q8;
                let e0 = judge(
                    clause,
                    ("kv_cmpr#0", l, p),
                    (&raw, &y_m),
                    band0,
                    tally,
                    seen,
                );
                if e0 > band0 {
                    print_samples(clause, &raw, &y_m);
                }
                let band = latent_band(&raw, &y_m, g, hp.rms_eps, !ropes);
                let want = f64s(&z_m);
                let used = bound_used(&normed, &want, &band);
                let e1 = rel(&normed, &z_m);
                tally.compared += 1;
                if used > 1.0 {
                    tally.ok = false;
                }
                tally.worst = tally.worst.max(e1);
                seen.push(("kv_cmpr#1".to_string(), p, e1));
                seen.push(("kv_cmpr#1 band share".to_string(), p, used));
                if used > 1.0 || e1 > dense_band {
                    // The widest value against its band, and each engine's
                    // normed row against the rule on its own latent: a side
                    // past its own rule's rounding is the side that moved.
                    let share = |i: usize| (f64::from(normed[i]) - want[i]).abs() / band[i];
                    let i = (0..latent).fold(0, |b, i| if share(i) > share(b) { i } else { b });
                    println!(
                        "{clause}: kv_cmpr-{l} occ 1 p{p} rel {e1:.4} (peak {:.3e}), {used:.3} of its \
                         per-value band {}: widest at {i}, ours {:.4e} mainline {:.4e} band {:.2e}, \
                         gain {:.4e}, latent {:.4e} ours {:.4e}; own rule ours {:.3}, mainline's {:.3}",
                        peak(&z_m),
                        verdict(used <= 1.0),
                        normed[i],
                        z_m[i],
                        band[i],
                        g[i],
                        y_m[i],
                        raw[i],
                        rule_used(&normed, &raw, g, hp.rms_eps, !ropes),
                        rule_used(&z_m, &y_m, g, hp.rms_eps, false),
                    );
                }
                if ropes {
                    let want = rope_last(&row("Kcur", 0)?, latent + rope_dims, rope_dims);
                    let (got, want) = (&node.kvr[..rope_dims], &want[..rope_dims]);
                    let e = judge(clause, ("Kcur rope", l, p), (got, want), band0, tally, seen);
                    if e > band0 {
                        print_samples(clause, got, want);
                    }
                }
            } else {
                let taps: Vec<(&str, Vec<f32>, u32, f64)> = vec![
                    ("kv_cmpr", raw, 0, dense_band),
                    ("kv_cmpr", normed, 1, dense_band),
                ];
                for (what, got, occ, band) in taps {
                    let want = row(what, occ)?;
                    let e = rel(&got, &want);
                    tally.compared += 1;
                    if e > band {
                        println!("{clause}: {what}-{l} occ {occ} p{p} rel {e:.4} FAIL");
                        print_samples(clause, &got, &want);
                        tally.ok = false;
                    }
                    tally.worst = tally.worst.max(e);
                    seen.push((format!("{what}#{occ}"), p, e));
                }
                // `Kcur`: a rope layer's appended row is `[k_rope |
                // kv_compressed]`, the reference's concat the other way;
                // a NoPE layer's row is the latent, which the cache
                // holds, with a zero rope part after it.
                let (got, want) = if ropes {
                    (
                        node.kvr.clone(),
                        rope_last(&row("Kcur", 0)?, latent + rope_dims, rope_dims),
                    )
                } else {
                    (cache_row(), row("Kcur", 0)?[..latent].to_vec())
                };
                let e = judge(
                    clause,
                    ("Kcur", l, p),
                    (&got, &want),
                    dense_band,
                    tally,
                    seen,
                );
                if e > dense_band {
                    print_samples(clause, &got, &want);
                }
            }
        }
        if dense && !idx_crossed {
            // The dense lead's own block output, fed by its
            // attention.
            // Derived, the constant at its Q8_0 projections' input rounding
            // ratio, plus the input's distance carried through the SwiGLU
            // block (its down's own input ratio is the shared expert's
            // reading, which the routed layers print).
            let want = row("ffn_out", 0)?;
            if !want.is_empty() {
                let band =
                    ffn()?.map_or(expert_band, |f| expert_band * f.r_q8 + SWIGLU_CARRY * f.eps);
                let e = judge(
                    clause,
                    ("ffn_out", l, p),
                    (&node.ffn_out, &want),
                    band,
                    tally,
                    seen,
                );
                if e > band {
                    print_samples(clause, &node.ffn_out, &want);
                }
            }
        }
        let mut crossed = false;
        if !dense && !idx_crossed {
            // The router's logits (cosine), the ids (a
            // set: the kernel writes them row-ascending, mainline in
            // rank order), the weights and the experts' outputs. A
            // set that crossed inside the logits' own noise (the
            // flip's gap under twice the run's max |Δ| against the
            // dump's own probabilities) excuses the nodes the
            // router's choice feeds at this position.
            let want_logits = row("ffn_moe_logits", 0)?;
            if !want_logits.is_empty() {
                let c = cosine(&node.route_logits, &want_logits);
                tally.worst_cos = tally.worst_cos.min(c);
                seen.push(("ffn_moe_logits 1-cos".to_string(), p, 1.0 - c));
                tally.compared += 1;
                if c < cosine_floor {
                    println!("{clause}: ffn_moe_logits-{l} p{p} cosine {c:.6} FAIL");
                    tally.ok = false;
                }
            }
            let mut got_ids = node.route_ids.clone();
            let mut want_ids = ids_row("ffn_moe_topk")?;
            if !want_ids.is_empty() {
                got_ids.sort_unstable();
                want_ids.sort_unstable();
                tally.compared += 1;
                if got_ids != want_ids {
                    let want_probs = row("ffn_moe_probs", 0)?;
                    let dp = node
                        .route_probs
                        .iter()
                        .zip(&want_probs)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    let weakest = got_ids
                        .iter()
                        .map(|&i| node.route_probs[i as usize])
                        .fold(f32::INFINITY, f32::min);
                    let outside: Vec<u32> = want_ids
                        .iter()
                        .filter(|id| !got_ids.contains(id))
                        .copied()
                        .collect();
                    let band = 2.0 * dp;
                    let excused = !outside.is_empty()
                        && outside
                            .iter()
                            .all(|&id| node.route_probs[id as usize] - weakest <= band);
                    if excused {
                        crossed = true;
                        println!(
                            "{clause}: ffn_moe_topk-{l} p{p} {got_ids:?} != {want_ids:?} \
                         inside the logits' noise (Δp {dp:.2e}): the router-fed \
                         nodes skip here"
                        );
                    } else {
                        println!(
                            "{clause}: ffn_moe_topk-{l} p{p} {got_ids:?} != {want_ids:?} FAIL",
                        );
                        tally.ok = false;
                    }
                }
            }
            // PIN(2026-11-01): the router's renormalised weights
            // and the shared expert keep the dense band, measured
            // 0.0164 and 0.0244 at their frontiers over both sets
            // (the shared expert's at a router crossing, which it
            // does not skip); the experts' sum and the block's
            // output, which the routed experts feed, the routed
            // band.
            // PIN(2026-11-02): derived (the error-model section), the experts'
            // sum takes the routed constant at mainline's q8_K ratio of the
            // normed input, the shared expert the dense constant at its Q8_0
            // ratio, each plus the input's distance carried through the SwiGLU
            // block ([`SWIGLU_CARRY`]). A shared expert past that band has its
            // down's input read on the host ([`ffn_host`], mainline's rule on
            // its own input): the band takes the louder of the two Q8_0 inputs'
            // ratios, and the line beside it names each term — our block
            // against its exact rule, mainline's against its own rule, the
            // carried input, mainline's int8 rounding. The block's output is
            // the two sums' add, so its band is theirs against its peak.
            let fin = ffn()?;
            let carry = fin.as_ref().map_or(0.0, |f| SWIGLU_CARRY * f.eps);
            let moe_band = fin
                .as_ref()
                .map_or(expert_band, |f| expert_band * f.r_exp + carry);
            let mut sh_band = fin
                .as_ref()
                .map_or(dense_band, |f| dense_band * f.r_q8 + carry);
            let want_w = row("ffn_moe_weights_norm", 0)?;
            let want_moe = row("ffn_moe_out", 0)?;
            let want_sh = row("ffn_shexp", 0)?;
            let want_out = row("ffn_out", 0)?;
            if !crossed && !want_w.is_empty() {
                let got = &node.route_weights;
                judge(
                    clause,
                    ("ffn_moe_weights_norm", l, p),
                    (got, &want_w),
                    dense_band,
                    tally,
                    seen,
                );
            }
            if !crossed && !want_moe.is_empty() {
                judge(
                    clause,
                    ("ffn_moe_out", l, p),
                    (&node.moe, &want_moe),
                    moe_band,
                    tally,
                    seen,
                );
            }
            if !want_sh.is_empty() {
                let e = rel(&node.sh, &want_sh);
                if let Some(f) = fin.as_ref().filter(|_| e > sh_band) {
                    let names = [
                        blk(l, stems::FFN_GATE_SHEXP),
                        blk(l, stems::FFN_UP_SHEXP),
                        blk(l, stems::FFN_DOWN_SHEXP),
                    ];
                    let host = ffn_host(&case.weights, names, &f.x_o, &f.x_m)?;
                    let r_h = rounding_ratio(&host.h, Q8_0_RULES)?;
                    sh_band = dense_band * f.r_q8.max(r_h) + carry;
                    println!(
                        "{clause}: ffn_shexp-{l} p{p} rel {e:.4}: input ratios {:.2} (gate, up) {r_h:.2} \
                         (down), carried input {:.4}, band {sh_band:.4}; ours against its exact rule \
                         {:.4}, mainline's against its own {:.4}, exact on the two inputs {:.4}, \
                         mainline's int8 rounding {:.4}",
                        f.r_q8,
                        f.eps,
                        rel64(&f64s(&node.sh), &host.ours),
                        rel64(&f64s(&want_sh), &host.rule),
                        rel64(&host.ours, &host.exact),
                        rel64(&host.rule, &host.exact),
                    );
                }
                judge(
                    clause,
                    ("ffn_shexp", l, p),
                    (&node.sh, &want_sh),
                    sh_band,
                    tally,
                    seen,
                );
            }
            if !crossed && !want_out.is_empty() {
                let band = if derived {
                    let floor = |v: &[f32]| peak(v).max(1e-3);
                    (moe_band * floor(&want_moe) + sh_band * floor(&want_sh)) / floor(&want_out)
                        + 2.0 * U
                } else {
                    expert_band
                };
                judge(
                    clause,
                    ("ffn_out", l, p),
                    (&node.ffn_out, &want_out),
                    band,
                    tally,
                    seen,
                );
            }
        }
        // The block's output, the residual plus the router-fed FFN:
        // it skips where the attention or the router's set crossed
        // (`gate_musespark_e2e`'s `after_router` rule).
        let want = row("l_out", 0)?;
        if !want.is_empty() && !(idx_crossed || crossed) {
            let e = rel(&node.l_out, &want);
            tally.compared += 1;
            if e > expert_band {
                println!("{clause}: l_out-{l} p{p} rel {e:.4} > {expert_band} FAIL");
                print_samples(clause, &node.l_out, &want);
                tally.ok = false;
            }
            tally.worst = tally.worst.max(e);
            seen.push(("l_out".to_string(), p, e));
        }
        Ok(())
    }

    /// One layer's compared values against the dump's, at every prompt
    /// position of the tier's first `n_sets` sets, the layer alone on
    /// mainline's own input, read against `bands`.
    fn nodes_against(
        m: &mut Model,
        case: &Case,
        (clause, bands): (&str, Bands),
        n_sets: usize,
    ) -> Result<bool, GateError> {
        let sets = &case.sets;
        let hp = &case.hp;
        let hidden = hp.n_embd;
        let mut tally = Tally::new();
        let mut positions = 0usize;
        m.set_mode(StepMode::Eager);
        if m.has_capture() {
            println!("{clause}: a capture exists FAIL");
            return Ok(false);
        }
        let (gpu, b) = body_of(m)?;
        b.arm_taps(gpu, true)?;
        // Set b's longer prompt walks past the top-k for most of its
        // positions, set a's for its last two: the selected attention's
        // rows come mostly from b.
        for (set, man) in [("a", &sets.a), ("b", &sets.b)].into_iter().take(n_sets) {
            let ids = prompt_tables::tokens(man)?.to_vec();
            let n_pos = ids.len();
            positions += n_pos;
            let l_out = |l: usize, p: usize| l_out_row(man, hidden, l, p);
            for (l, kind) in layer_cases(hp).iter().enumerate() {
                let name = &kind.name;
                // The span starts at the first layer the compared one reads
                // (its list's or its cache's), and each layer it reads runs
                // on mainline's input as the compared one does.
                let source = kind.indexer.min(kind.cache);
                let mut forced = vec![kind.indexer, kind.cache, l];
                forced.sort_unstable();
                forced.dedup();
                // The prompt teacher-forced from a reset, the span from the layer
                // the list and cache live on to the compared layer: at every
                // position each of the two runs on mainline's own input (the
                // dump's `l_out` of the layer before it; the embedding at layer
                // 0), so every cache row, index key and list the compared layer
                // reads was built from mainline's rows, as the reference's own
                // layer reads its own.
                m.reset()?;
                {
                    let (_gpu, b) = body_of(m)?;
                    b.set_layer_span(Some(source..l + 1))?;
                }
                // Every comparison of the layer (node, position, distance): the
                // frontier line its bands are read from.
                let mut seen: Vec<(String, usize, f64)> = Vec::new();
                for (p, &id) in ids[..n_pos].iter().enumerate() {
                    for &at in &forced {
                        if at > 0 {
                            let (gpu, b) = body_of(m)?;
                            b.set_input_override(gpu, at, &l_out(at - 1, p)?)?;
                        }
                    }
                    if or_fault(clause, "the forced layer", feed(m, &[id]))?.is_none() {
                        let (gpu, b) = body_of(m)?;
                        b.arm_taps(gpu, false)?;
                        return Ok(false);
                    }
                    let nodes = {
                        let (gpu, b) = body_of(m)?;
                        b.nodes(gpu, l..l + 1)?
                    };
                    compare_layer(
                        m,
                        (clause, bands),
                        case,
                        (l, kind),
                        &nodes[0],
                        &At {
                            man,
                            rows: n_pos,
                            row: p,
                            pos: p,
                        },
                        (false, &mut tally, &mut seen),
                    )?;
                }
                println!(
                    "{clause}: set {set} layer {l} ({name}) frontier: {}",
                    frontier(&seen)
                );
                {
                    let (_gpu, b) = body_of(m)?;
                    b.set_layer_span(None)?;
                    b.clear_input_override();
                }
                eprintln!("clause {clause}: set {set} layer {l} ({name})");
            }
        }
        {
            let (gpu, b) = body_of(m)?;
            b.set_layer_span(None)?;
            b.arm_taps(gpu, false)?;
        }
        println!(
            "{clause}: {} comparisons over {} layers x {positions} positions ({n_sets} sets), \
             worst rel {:.4}, worst logit cosine {:.6} {}",
            tally.compared,
            hp.n_layer,
            tally.worst,
            tally.worst_cos,
            verdict(tally.ok)
        );
        Ok(tally.ok)
    }

    /// The deep set's clause (the real file, past the indexer's top-k): the engine
    /// walks the deep set's prefill from a reset, 3,000 ids on its own layers, then
    /// takes the one step on the set's last id with every layer on mainline's own
    /// input (the dump's `l_out` of the layer before it, at the step) and reads
    /// every layer against the dump's rows. Two contracts, one walk:
    /// - the selection rule (`selection_is_its_rule`'s check) after every step from
    ///   the first past the top-k to the deep step, where the list is a cut of
    ///   thousands of rows and not all of them;
    /// - the node clause (`nodes_against`'s comparisons) at the deep step, each
    ///   layer reading the caches and index keys the walk built. A list that
    ///   differs from the dump's inside the indexer's band does not excuse the nodes
    ///   it feeds here: the cut moves rows at its threshold on any rounding and
    ///   the "top_k moved" fraction of each layer is printed beside the distances.
    ///
    /// The step's logits row is read against the dump's `result_output` at the
    /// real tier's cosine floor.
    fn nodes_near_mainlines_deep(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "nodes_near_mainlines_deep";
        let man = case
            .sets
            .deep
            .as_ref()
            .ok_or("the deep set is not open (the real tier opens it)")?;
        let hp = &case.hp;
        let (top_k, hidden) = (hp.indexer_top_k, hp.n_embd);
        let (pos, step, prefill) = man.step()?;
        let (pos, token) = (
            pos as usize,
            *step.first().ok_or("the deep set holds no step id")?,
        );
        if pos <= top_k {
            return Err(format!(
                "{CLAUSE}: the deep set's step is at position {pos}, not past the indexer's \
                 top-k {top_k}: its selection is every position, not a cut"
            )
            .into());
        }
        m.set_mode(StepMode::Eager);
        if m.has_capture() {
            println!("{CLAUSE}: a capture exists FAIL");
            return Ok(false);
        }
        m.reset()?;
        let t = std::time::Instant::now();
        let (mut selection_ok, mut named) = (true, 0usize);
        for (at, &id) in prefill.iter().enumerate() {
            if or_fault(CLAUSE, "the prefill walk", m.step(&[id]))?.is_none() {
                return Ok(false);
            }
            if at >= top_k {
                check_selection(m, hp, (CLAUSE, at), (&mut selection_ok, &mut named))?;
            }
            if (at + 1) % 500 == 0 {
                eprintln!(
                    "clause {CLAUSE}: walked {} of {} ids, {:.0} s",
                    at + 1,
                    prefill.len(),
                    t.elapsed().as_secs_f64()
                );
            }
        }
        let walked = t.elapsed().as_secs_f64();
        // The step: the taps armed now (the walk ran without their copies) and
        // every layer but the first on mainline's own input.
        {
            let (gpu, b) = body_of(m)?;
            b.arm_taps(gpu, true)?;
            for l in 1..hp.n_layer {
                b.set_input_override(gpu, l, &l_out_row(man, hidden, l - 1, 0)?)?;
            }
        }
        let disarm = |m: &mut Model| -> Result<(), GateError> {
            let (gpu, b) = body_of(m)?;
            b.clear_input_override();
            b.arm_taps(gpu, false)?;
            Ok(())
        };
        if or_fault(CLAUSE, "the deep step", m.step(&[token]))?.is_none() {
            disarm(m)?;
            return Ok(false);
        }
        // The deep step is a selecting step too: the body's own scores and
        // its list, whatever input the layers ran on.
        check_selection(m, hp, (CLAUSE, pos), (&mut selection_ok, &mut named))?;
        println!(
            "{CLAUSE}: selection_is_its_rule: {} positions past top-k {top_k}, every full layer's \
             set the rule's and every shared layer's its source's ({named} threshold ties named) {}",
            pos + 1 - top_k,
            verdict(selection_ok)
        );
        let logits = m.logits()?;
        let mut tally = Tally::new();
        for (l, kind) in layer_cases(hp).iter().enumerate() {
            let nodes = {
                let (gpu, b) = body_of(m)?;
                b.nodes(gpu, l..l + 1)?
            };
            let mut seen: Vec<(String, usize, f64)> = Vec::new();
            compare_layer(
                m,
                (CLAUSE, DEEP_BANDS),
                case,
                (l, kind),
                &nodes[0],
                &At {
                    man,
                    rows: 1,
                    row: 0,
                    pos,
                },
                (true, &mut tally, &mut seen),
            )?;
            println!(
                "{CLAUSE}: layer {l} ({}) frontier: {}",
                kind.name,
                frontier(&seen)
            );
            eprintln!("clause {CLAUSE}: layer {l} ({})", kind.name);
        }
        disarm(m)?;
        // The step's logits at `topk.tsv`'s ids against the table's logits, as
        // the greedy clause reads a prompt's last row (the layers here on
        // mainline's input), and the whole row against the dump's
        // `result_output` over every id of the head, printed.
        let dumped = ik_last(man, hp.n_vocab)?;
        if dumped.len() != logits.len() {
            return Err(format!(
                "{CLAUSE}: the step's logits row holds {} values, result_output's {}",
                logits.len(),
                dumped.len()
            )
            .into());
        }
        let top = prompt_tables::topk(man)?;
        let got_k: Vec<f32> = top
            .iter()
            .map(|t| logits.get(t.id as usize).copied().unwrap_or(f32::NAN))
            .collect();
        let want_k: Vec<f32> = top.iter().map(|t| t.logit).collect();
        let c = cosine(&got_k, &want_k);
        let head_ok = c >= DEEP_BANDS.cosine;
        let row_delta: Vec<f64> = logits
            .iter()
            .zip(&dumped)
            .map(|(&g, &w)| f64::from(g) - f64::from(w))
            .collect();
        let (row_at, row_max) = widest(&row_delta);
        println!(
            "{CLAUSE}: the step's top-{} logits' cosine {c:.6} against topk, floor {}; the row \
             against result_output over {} ids: rms {:.4}, max |Δ| {row_max:.4} at id {row_at}; \
             argmax ours {} theirs {:?} {}",
            top.len(),
            DEEP_BANDS.cosine,
            row_delta.len(),
            rms(&row_delta),
            argmax(&logits),
            top.first().map(|t| t.id),
            verdict(head_ok)
        );
        let ok = selection_ok && tally.ok && head_ok;
        println!(
            "{CLAUSE}: {} comparisons over {} layers at position {pos} after a walk of {} ids \
             ({walked:.1} s, runtime value), worst rel {:.4}, worst logit cosine {:.6} {}",
            tally.compared,
            hp.n_layer,
            prefill.len(),
            tally.worst,
            tally.worst_cos,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The log-softmax of `row` at `id` in f64 (the same quantity as the
    /// `logprob` column of `greedy.tsv` on mainline's row); NaN where `id` is
    /// outside the row.
    fn logprob_at(row: &[f32], id: u32) -> f64 {
        let Some(&z) = row.get(id as usize) else {
            return f64::NAN;
        };
        let peak = row
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let sum: f64 = row.iter().map(|&v| (f64::from(v) - peak).exp()).sum();
        f64::from(z) - peak - sum.ln()
    }

    /// The root mean square of `xs`; NaN for none.
    fn rms(xs: &[f64]) -> f64 {
        (xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64).sqrt()
    }

    /// The index and the size of the entry of `xs` largest in magnitude.
    fn widest(xs: &[f64]) -> (usize, f64) {
        xs.iter().enumerate().fold(
            (0, 0.0),
            |best, (i, x)| {
                if x.abs() > best.1 { (i, x.abs()) } else { best }
            },
        )
    }

    /// The fixture's greedy clause: sets a and b at the fixture's margins.
    fn greedy_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        greedy_against(
            m,
            case,
            "greedy_is_mainlines",
            (GREEDY_MARGIN_A, GREEDY_MARGIN_B),
            None,
        )
    }

    /// The real file's greedy clause: sets a and b at the real tier's
    /// margins, and the last prompt position's top-40 logits against
    /// `topk.tsv` above the real tier's cosine floor.
    fn greedy_is_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        greedy_against(
            m,
            case,
            "greedy_is_mainlines_real",
            (REAL_GREEDY_MARGIN_A, REAL_GREEDY_MARGIN_B),
            Some(REAL_BANDS.cosine),
        )
    }

    /// The greedy 32 of sets a and b, teacher-forced on the oracle's own ids,
    /// each set's flips read against its margin; with a floor, the last
    /// prompt position's logits at `topk.tsv`'s ids keep that cosine with
    /// the oracle's.
    fn greedy_against(
        m: &mut Model,
        case: &Case,
        clause: &str,
        (margin_a, margin_b): (f64, f64),
        cosine_floor: Option<f64>,
    ) -> Result<bool, GateError> {
        let sets = &case.sets;
        let vocab = case.hp.n_vocab;
        let mut ok = true;
        for (label, man, band) in [("a", &sets.a, margin_a), ("b", &sets.b, margin_b)] {
            let ids = prompt_tables::tokens(man)?.to_vec();
            let Greedy {
                ids: want,
                margins,
                logprobs,
            } = prompt_tables::greedy(man)?;
            m.set_mode(StepMode::Graph);
            m.reset()?;
            let Some(prompt) = or_fault(clause, "the prompt", feed(m, &ids))? else {
                return Ok(false);
            };
            // Step s's logits are the ones that pick the oracle's s-th id:
            // the prompt's last for step 0, then each step's after the
            // oracle's own id before it. The last id is compared, not fed.
            let (Some(&first_id), Some(first)) = (prompt.ids.last(), prompt.logits.last()) else {
                return Err(format!("set {label}: an empty prompt").into());
            };
            let mut run = vec![first_id];
            let mut logits = vec![first.clone()];
            for &t in &want[..want.len().saturating_sub(1)] {
                let Some(r) = or_fault(clause, "a greedy step", feed(m, &[t]))? else {
                    return Ok(false);
                };
                run.push(r.ids[0]);
                logits.push(r.logits[0].clone());
            }
            // The run's own ids are its own argmax on every step.
            let self_argmax = logits
                .iter()
                .zip(&run)
                .all(|(row, &id)| row.len() == vocab && argmax(row) == id);
            // The last prompt position's logits against `topk.tsv`'s logit
            // values (the one place the dump states the oracle's logits),
            // printed beside the band it is read against.
            let topk = prompt_tables::topk(man)?;
            let mut set_ok = true;
            let mut delta = 0.0_f64;
            let (mut got_k, mut want_k) = (Vec::new(), Vec::new());
            for t in &topk {
                let got = first.get(t.id as usize).copied().unwrap_or(f32::NAN);
                delta = delta.max((f64::from(got) - f64::from(t.logit)).abs());
                got_k.push(got);
                want_k.push(t.logit);
            }
            if let Some(floor) = cosine_floor {
                let c = cosine(&got_k, &want_k);
                let held = c >= floor;
                println!(
                    "{clause}: set {label}: the top-{} logits' cosine {c:.6} against topk, floor \
                     {floor} {}",
                    topk.len(),
                    verdict(held)
                );
                set_ok &= held;
            }
            // The prompt's last row against the dump's whole `result_output`
            // row: every id of the head, not the 40 `topk.tsv` states.
            let dumped = ik_last(man, vocab)?;
            if dumped.len() != first.len() {
                return Err(format!(
                    "set {label}: the run's last row holds {} logits, result_output's {}",
                    first.len(),
                    dumped.len()
                )
                .into());
            }
            let row_delta: Vec<f64> = first
                .iter()
                .zip(&dumped)
                .map(|(&g, &w)| f64::from(g) - f64::from(w))
                .collect();
            let (row_at, row_max) = widest(&row_delta);
            println!(
                "{clause}: set {label}: the prompt's last logits row against result_output over \
                 {} ids: rms {:.4}, max |Δ| {row_max:.4} at id {row_at}",
                row_delta.len(),
                rms(&row_delta)
            );
            let mut agree = 0usize;
            let mut excused = 0usize;
            let mut gated = 0usize;
            for (s, (&got, (&want_id, &margin))) in
                run.iter().zip(want.iter().zip(margins.iter())).enumerate()
            {
                if f64::from(margin) > band {
                    gated += 1;
                }
                if got == want_id {
                    agree += 1;
                } else if f64::from(margin) <= band {
                    excused += 1;
                    println!(
                        "{clause}: set {label} step {s}: ids {got}/{} margin {:.4} below the band \
                         {band:.4}",
                        want_id,
                        f64::from(margin)
                    );
                } else {
                    println!(
                        "{clause}: set {label} step {s}: id {got} != {} at margin {:.4} FAIL",
                        want_id,
                        f64::from(margin)
                    );
                    set_ok = false;
                }
            }
            // The oracle id's log-probability on the run's own row against the
            // dump's, at every step: the rounding the run puts on the id
            // mainline chose at each continuation step, where the |Δ| above
            // is the prompt's last position alone.
            let drift: Vec<f64> = logits
                .iter()
                .zip(want.iter().zip(&logprobs))
                .map(|(row, (&id, &lp))| logprob_at(row, id) - lp)
                .collect();
            let (mut moved, mut kept) = (Vec::new(), Vec::new());
            for ((&got, &id), &d) in run.iter().zip(&want).zip(&drift) {
                if got == id {
                    kept.push(d);
                } else {
                    moved.push(d);
                }
            }
            let (drift_at, drift_max) = widest(&drift);
            println!(
                "{clause}: set {label}: the oracle id's log-probability on the run's row against \
                 the dump's: rms {:.4} over {} steps, max {drift_max:.4} at step {drift_at}; the \
                 {} differing steps rms {:.4}, the {} agreeing {:.4}",
                rms(&drift),
                drift.len(),
                moved.len(),
                rms(&moved),
                kept.len(),
                rms(&kept)
            );
            set_ok &= self_argmax;
            ok &= set_ok;
            println!(
                "{clause}: set {label}: {agree}/{} agree, {excused} below the band {band}, \
                 {gated} above it gated, max |Δ| {delta:.4} against topk, the run's own ids its \
                 argmax {self_argmax} {}",
                want.len(),
                verdict(set_ok)
            );
        }
        Ok(ok)
    }

    // ------------------------------------------------------------------ head

    /// The fixture's head clause: sets a and b at the fixture's head bands.
    fn head_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        head_against(m, case, ("head_near_mainlines", FIXTURE_HEAD))
    }

    /// The real file's head clause: sets a and b at the real tier's head bands.
    fn head_near_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        head_against(m, case, ("head_near_mainlines_real", REAL_HEAD))
    }

    /// The least-squares scale of `got` on `want` through the origin, minus
    /// one (`Σ got·want / Σ want² − 1`): a row that is a constant multiple of
    /// the other reads that multiple, and rounding that is independent of the
    /// row's values averages out of it. NaN for rows of unequal length, a
    /// non-finite value on either side, or a `want` of all zeros.
    fn gain(got: &[f32], want: &[f32]) -> f64 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        let (dot, norm) = got
            .iter()
            .zip(want)
            .fold((0.0_f64, 0.0_f64), |(d, n), (&g, &w)| {
                (
                    d + f64::from(g) * f64::from(w),
                    n + f64::from(w) * f64::from(w),
                )
            });
        dot / norm - 1.0
    }

    /// The values of `row` at `ids`.
    fn picked(row: &[f32], ids: &[usize]) -> Vec<f32> {
        ids.iter().map(|&i| row[i]).collect()
    }

    /// The head alone on mainline's own input, sets a and b: the last layer
    /// runs on the dump's `l_out` of the layer before it at every prompt
    /// position (the node clause's span and overrides, so every cache row,
    /// index key and list it reads is built from mainline's rows), the head
    /// reads that layer's output, and the prompt's last position is compared
    /// with the dump's `result_norm` and `result_output`: the normed row's
    /// relative distance and gain, the logits row's cosine, and the logits'
    /// gain over the dump's top [`HEAD_TOP`] ids. Every number is printed
    /// whether or not it holds, with the last layer's own output distance
    /// beside them.
    fn head_against(
        m: &mut Model,
        case: &Case,
        (clause, bands): (&str, HeadBands),
    ) -> Result<bool, GateError> {
        let hp = &case.hp;
        let (hidden, vocab) = (hp.n_embd, hp.n_vocab);
        let last = hp.n_layer - 1;
        let kind = layer_cases(hp).swap_remove(last);
        // The span starts on the first layer the last one reads (its list's or
        // its cache's); each of those and the last run on mainline's input.
        let source = kind.indexer.min(kind.cache);
        let mut forced = vec![kind.indexer, kind.cache, last];
        forced.sort_unstable();
        forced.dedup();
        m.set_mode(StepMode::Eager);
        if m.has_capture() {
            println!("{clause}: a capture exists FAIL");
            return Ok(false);
        }
        {
            let (gpu, b) = body_of(m)?;
            b.arm_taps(gpu, true)?;
        }
        let mut ok = true;
        for (set, man) in [("a", &case.sets.a), ("b", &case.sets.b)] {
            let ids = prompt_tables::tokens(man)?.to_vec();
            let n_pos = ids.len();
            if n_pos == 0 {
                return Err(format!("{clause}: set {set} holds no prompt").into());
            }
            // The dump's input row of each forced layer at every position: the
            // output of the layer before it (the embedding feeds layer 0).
            let mut inputs: Vec<(usize, Vec<f32>)> = Vec::new();
            for &at in forced.iter().filter(|&&at| at > 0) {
                inputs.push((at, tap_at(man, &format!("l_out-{}", at - 1), 0)?));
            }
            m.reset()?;
            {
                let (_gpu, b) = body_of(m)?;
                b.set_layer_span(Some(source..last + 1))?;
            }
            let mut logits: Vec<f32> = Vec::new();
            for (p, &token) in ids.iter().enumerate() {
                for (at, rows) in &inputs {
                    let row = rows
                        .get(p * hidden..(p + 1) * hidden)
                        .ok_or("a row past the dump")?;
                    let (gpu, b) = body_of(m)?;
                    b.set_input_override(gpu, *at, row)?;
                }
                let Some(run) = or_fault(
                    clause,
                    "the last layer on the oracle's input",
                    feed(m, &[token]),
                )?
                else {
                    let (gpu, b) = body_of(m)?;
                    b.arm_taps(gpu, false)?;
                    return Ok(false);
                };
                if p + 1 == n_pos {
                    logits = run
                        .logits
                        .into_iter()
                        .next()
                        .ok_or("a step without logits")?;
                }
            }
            let up = {
                let (gpu, b) = body_of(m)?;
                b.nodes(gpu, last..last + 1)?.swap_remove(0).l_out
            };
            let normed = m.head_normed()?;
            // The last position's row of a dump tensor that holds whole rows of
            // `width` (the last layer's output and the head's norm are dumped
            // for the output position alone).
            let last_row = |what: &str, width: usize| -> Result<Vec<f32>, GateError> {
                let all = tap_at(man, what, 0)?;
                let at = all
                    .len()
                    .checked_sub(width)
                    .filter(|_| all.len().is_multiple_of(width))
                    .ok_or_else(|| {
                        format!("{what}: {} values, not whole rows of {width}", all.len())
                    })?;
                Ok(all[at..].to_vec())
            };
            let want_up = last_row(&format!("l_out-{last}"), hidden)?;
            let want_norm = last_row("result_norm", hidden)?;
            let want_logits = ik_last(man, vocab)?;
            if normed.len() != hidden || logits.len() != vocab {
                println!(
                    "{clause}: set {set}: the run's normed row holds {} values (want {hidden}) \
                     and its logits {} (want {vocab}) FAIL",
                    normed.len(),
                    logits.len()
                );
                ok = false;
                continue;
            }
            let (e_up, e_norm, e_logit) = (
                rel(&up, &want_up),
                rel(&normed, &want_norm),
                rel(&logits, &want_logits),
            );
            let (c_norm, c_logit) = (cosine(&normed, &want_norm), cosine(&logits, &want_logits));
            let g_norm = gain(&normed, &want_norm);
            let top = rule_top(&want_logits, HEAD_TOP);
            let g_top = gain(&picked(&logits, &top), &picked(&want_logits, &top));
            let g_all = gain(&logits, &want_logits);
            let (ours_id, want_id) = (argmax(&logits), argmax(&want_logits));
            // A NaN on any side fails: every comparison below is false for it.
            let norm_held = e_norm <= bands.norm_rel && g_norm.abs() <= bands.norm_gain;
            let logits_held = c_logit >= bands.cosine && g_top.abs() <= bands.top_gain;
            let set_ok = norm_held && logits_held;
            println!(
                "{clause}: set {set} position {}: result_norm rel {e_norm:.6} (band {}), gain \
                 {g_norm:+.6} (band ±{}), cosine {c_norm:.8}; result_output over {vocab} ids: \
                 rel {e_logit:.6}, cosine {c_logit:.8} (floor {}), gain over the dump's top \
                 {HEAD_TOP} {g_top:+.6} (band ±{}), over every id {g_all:+.6}, argmax \
                 {ours_id}/{want_id}; the last layer's own output rel {e_up:.6} {}",
                n_pos - 1,
                bands.norm_rel,
                bands.norm_gain,
                bands.cosine,
                bands.top_gain,
                verdict(set_ok)
            );
            if !norm_held {
                print_samples(clause, &normed, &want_norm);
            }
            ok &= set_ok;
            let (_gpu, b) = body_of(m)?;
            b.clear_input_override();
        }
        {
            let (gpu, b) = body_of(m)?;
            b.set_layer_span(None)?;
            b.arm_taps(gpu, false)?;
        }
        println!(
            "{clause}: the head on the oracle's last-layer input, both sets' last position {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------------------------ run

    /// The tier's sets, opened before the card loads: the fixture's in the
    /// fixture tier, in the real tier the family of the file in view (the
    /// UD-IQ1_S file's or the Mix-256G file's: each family refuses by name a
    /// set dumped from another file). One missing, stale or foreign ends the
    /// gate by name.
    fn open_sets(tier_now: tier::Tier) -> Result<Sets, GateError> {
        let open = |clause: &str, family: &'static Family, set: &str| -> Result<_, GateError> {
            Ok(match tier_now {
                tier::Tier::Fixture => {
                    RefManifest::open(&tier::fixture_set(clause, family, set)?, family)?
                }
                tier::Tier::Real => RefManifest::open(&family.path(set), family)?,
            })
        };
        let (family, a, b, deep): (&'static Family, &str, &str, Option<&str>) = match tier_now {
            tier::Tier::Fixture => (&fixture::LCPP, fixture::A, fixture::B, None),
            tier::Tier::Real
                if ref_model_path()? == std::path::Path::new(real_sets::MODEL_MIX256G) =>
            {
                (
                    &real_sets::LCPP_MIX256G,
                    real_sets::A_MIX256G,
                    real_sets::B_MIX256G,
                    Some(real_sets::D3K_MIX256G),
                )
            }
            tier::Tier::Real => (
                &real_sets::LCPP,
                real_sets::A,
                real_sets::B,
                Some(real_sets::D3K),
            ),
        };
        let deep = match deep {
            Some(name) => {
                let man = open("the deep clause", family, name)?;
                real_sets::check_deep(&man, name)?;
                Some(man)
            }
            None => None,
        };
        Ok(Sets {
            a: open("the node clauses", family, a)?,
            b: open("the greedy clauses", family, b)?,
            deep,
        })
    }

    /// The plan's levers: in the fixture tier the card keeps half of the
    /// experts the gate card would hold (so the host leg serves the rest and
    /// both legs run), in the real tier the card's own bytes, which already
    /// leave most of the experts to the host.
    fn levers(
        inputs: &PlanInputs,
        machine: &Machine,
        (ctx, tier_now): (u64, tier::Tier),
    ) -> Result<PlanLevers, GateError> {
        if tier_now == tier::Tier::Real {
            return Ok(PlanLevers::default());
        }
        let all = inputs.plan(machine, ctx, &PlanLevers::default())?;
        let c = &all.cards[0];
        let card = &machine.cards[0];
        let kept = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + inputs.prompt_reserve
            + c.expert_bytes / 2;
        Ok(PlanLevers {
            card_budget_bytes: Some(kept).into(),
        })
    }

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::MistralLarge4, "gate-gpu-mistrallarge4-e2e")?;
        let hp = Hparams::read(&file).map_err(|e| format!("mistral-large4 header: {e}"))?;
        let tier_now = tier::Tier::from_env()?;
        // The sets open before the card loads: one the tier runs, missing or
        // stale, ends the gate here.
        let sets = open_sets(tier_now)?;
        // The greedy clause teacher-forces a set's whole prompt and its
        // greedy ids but the last from a reset: the cache covers the longer
        // set's demand.
        let mut longest = PROMPT_LEN;
        for man in [&sets.a, &sets.b] {
            let ids = prompt_tables::tokens(man)?.len();
            let greedy = prompt_tables::greedy(man)?.ids.len();
            longest = longest.max(ids + greedy);
        }
        // The deep clause walks the whole deep set, the step on its last id.
        if let Some(man) = &sets.deep {
            longest = longest.max(prompt_tables::tokens(man)?.len());
        }
        let ctx = longest + CTX_SPARE;
        let walk = walk(PROMPT_LEN, hp.n_vocab);
        // The gate plan (module doc): the file as it is described, whatever
        // the coverage list holds, on the gate card, one stage card running
        // every layer and the head.
        let inputs = PlanInputs::describe(&file)?;
        // The sites the structure clause counts by type, read from the header
        // before the load takes the file.
        let sites = Sites::read(&file, &hp)?;
        // The norm gains the node clauses' rules and derived bands read.
        let gains = Gains::read(&file, &hp)?;
        let spec = gate_card::card()?;
        let machine = workstation::placed_on(spec, hp.n_layer, workstation::SCRATCH);
        let ctx64 = u64::try_from(ctx)?;
        let plan = inputs.plan(
            &machine,
            ctx64,
            &levers(&inputs, &machine, (ctx64, tier_now))?,
        )?;
        println!(
            "plan: ctx={ctx} card experts {} ({} B) host experts {} ({} B) (plan values)",
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes
        );
        let t = std::time::Instant::now();
        let mut m = Body::open_placed(file, &plan, &inputs, 0, HOST_CFG)?;
        let card_layers = {
            let (_gpu, b) = body_of(&mut m)?;
            b.card_layers().len()
        };
        println!(
            "load resident_bytes={} ctx={ctx} layers={} ({card_layers} with card experts) in \
             {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let case = Case {
            hp,
            walk,
            sites,
            sets,
            gains,
            weights: open_split(Arch::MistralLarge4, "gate-gpu-mistrallarge4-e2e")?,
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
            }
            ok &= pass;
        }
        println!("{}", tier::tally_line());
        tier::expect_fixture_oracle(NAME, FIXTURE_ORACLE)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the captured step of the placed load holds the entries its layer \
             kinds derive; \
             eager and replay give bit-identical logits; a reset rewinds; a capture seeded after \
             the record replays over the seeded rows; the cache rows carry; a \
             NaN router lane raises the fault word and is picked by no slot; every full indexer \
             layer's list is the rule's own top-k and every shared layer's its source's; a layer \
             span that is empty, reversed or past the model is refused by name; every \
             compared node of the tier's sets keeps its band at every layer and position; the \
             greedy continuation keeps the oracle's argmax above the band; the head's normed row \
             and logits keep their bands and gains against the oracle's; the real file's deep \
             step past the indexer's top-k keeps its selection rule at every position and its \
             nodes and logits their bands; the fault word stays clean"
        );
        Ok(())
    }
}
