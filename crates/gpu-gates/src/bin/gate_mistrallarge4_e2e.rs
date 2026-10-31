//! The Mistral Large 4 end-to-end gate: the whole one-token decode chain of
//! the five fixture layers — the absorbed MLA with SSMax on the joined query,
//! the YaRN rope at the LeRoPE factors, the lightning indexer whose keys are
//! projected from the layer input and whose list picks the positions the
//! attention reads (the file's `[rope | nope]` index head moved to the
//! kernel's tail order), the KV-shared NoPE layers on the 512-wide walk, the
//! softmax top-4 router over the picked experts beside the shared expert,
//! the dense lead and the softcapped head — against mainline llama.cpp's
//! node dumps on the fixture (`fx_ref_mistrallarge4_{a,b}`). The real tier
//! (the real file's placed load and its sets) is R5's: this gate's oracle
//! clauses are tagged so the real tier runs only the own-contract ones.
//!
//! The clauses are a list ([`CLAUSES`]); each prints lines ending in PASS or
//! FAIL, and a step that raises the card's fault word is a FAIL naming the
//! fault, never a pass. Each carries its tier tag (`tier::run_clause`):
//!
//! - `structure` (self-consistency): the captured step holds the entries the
//!   layer's kind derives — the embedding lookup and the live count it
//!   writes; per layer the fused norm+quant, the Q6_K query and output
//!   projections with their quantizations, the rope and its columns on a
//!   roped layer, the latent projection and the fused norm-rope-append (the
//!   bare latent append on a NoPE owner, none on the f4 reader), the
//!   LayerNorm, the index query projection, its permutation, its SSMax, the
//!   key projection, its permutation, the key append, the head weights, the
//!   score and the top-k (none of the indexer's on the f2 shared layer), the
//!   per-head absorption, the rope-span assembly and the main SSMax, the
//!   row gather and the flash (the walk's segment and merge on NoPE), the
//!   value up-projection, the residual add; the dense lead's three
//!   projections, swiglu and residual add, or the routed block's router
//!   gemv, routing, two quantizations, the gate·up and down of the file's
//!   own i-quant type, the weighted sum, the shared expert's three, two adds
//!   and the router gemv; the head's norm, quantization, projection, softcap
//!   and argmax. The span is the node count (the chain is one path: no
//!   launch runs beside another).
//! - `eager_equals_replay`, `reset_rewinds`, `state_carries`
//!   (self-consistency): as `gate_musespark_e2e`'s, over a 40-token walk —
//!   past the fixture's top-k 16, so every step past position 16 selects;
//!   the carried state is the latent and index-key rows.
//! - `router_lane_nan` (self-consistency): as `gate_musespark_e2e`'s.
//! - `selection_is_its_rule` (self-consistency): at every position past the
//!   top-k of the walk, every full indexer layer's picked set equals the
//!   host's f64 top-k of the body's own index scores (ties to the lower
//!   row, the kernel's own rule, named where they cut), and every shared
//!   layer's list is its source's, entry for entry.
//! - `nodes_near_mainlines` (fixture oracle, sets a and b): every compared
//!   node at every layer and prompt position, one layer at a time, the prompt
//!   teacher-forced from a reset: the span (`Body::set_layer_span`) runs the
//!   layer its list and cache live on and the compared layer, each on
//!   mainline's own input at every position (`Body::set_input_override`
//!   with the dump's `l_out` of the layer before it), so every cache row,
//!   index key and list the layer reads was built from mainline's rows: the
//!   attention norm, the scaled queries, the key path's rows, the indexer's
//!   queries, key and weights, the list, the router's logits, ids and
//!   renormalized weights, the experts' sum, the shared expert, the block's
//!   output and the residual. Logit-kind nodes keep a cosine floor; the
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
//!   step the run's recorded id is the argmax of its own logits row.
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
mod gate {
    use crate::e2e::{
        argmax, cosine, feed, finite, fresh, histogram, or_fault, same, tap_at, walk,
    };
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use bloomery_gpu_mistrallarge4::body::Body;
    use cuda_core::sys;
    use gguf::Split;
    use gguf::quant::half_to_f32;
    use model::arch::Arch;
    use model::arch::mistrallarge4::hparams::Hparams;
    use refset::arch::mistrallarge4::fixture;
    use refset::arch::prompt_tables::{self, Greedy};
    use refset::ik::{Layout, RefManifest, RowKind, ref_ints};

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

    /// Tokens of the walk every own-contract clause runs: past the fixture's
    /// top-k 16, so every step past position 16 selects.
    const PROMPT_LEN: usize = 40;

    /// The fixture-oracle clauses the gate declares.
    const FIXTURE_ORACLE: usize = 2;

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
    /// fixture's flat logits by up to the run's max |Δ| against `topk.tsv`,
    /// so a step whose top two sit closer than about twice it may flip.
    // PIN(2026-11-01): the clean run's largest margin at a step whose id
    // differs, set a 0.3966 (step 14), set b 0.5465 (step 10), each 1.2x;
    // both under twice the run's max |Δ| (0.3840 and 0.5741), the bound a
    // flip needs.
    const GREEDY_MARGIN_A: f64 = 0.48;
    const GREEDY_MARGIN_B: f64 = 0.66;

    /// The layers the fixture holds, by kind (the fixture table's own).
    const KINDS: [(&str, bool, bool, bool, bool, usize, usize); 5] = [
        // (name, ropes, full indexer, owns its cache, dense lead, the layer
        //  its list and cache live on — itself on an owner, the dump's own
        //  `top_k` index: a layer reads the first `top_k-<j>` with j >= il)
        ("dense lead", true, true, true, true, 0, 0),
        ("roped full", true, true, true, false, 1, 2),
        ("roped shared", true, false, true, false, 1, 2),
        ("nope owner", false, true, true, false, 3, 3),
        ("nope reader", false, true, false, false, 3, 4),
    ];

    /// The walk, the model file (for the layer types the structure clause
    /// counts), the oracle sets of the tier.
    struct Case {
        hp: Hparams,
        walk: Vec<u32>,
        file: Split,
        sets: Option<Sets>,
    }

    /// The fixture's two sets (the real tier holds none of its own: R5's).
    struct Sets {
        a: RefManifest,
        b: RefManifest,
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
        ("state_carries", Tag::SelfConsistency, state_carries),
        ("router_lane_nan", Tag::SelfConsistency, router_lane_nan),
        (
            "selection_is_its_rule",
            Tag::SelfConsistency,
            selection_is_its_rule,
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
        ("fault_word_clean", Tag::SelfConsistency, fault_word_clean),
    ];

    /// The relative distance of `got` to `want` over their common length:
    /// the largest absolute difference against the row's own peak
    /// (`max |want|`), the form every projection's rounding sits under — a
    /// per-value division would blow up at the row's near-zero entries.
    fn rel(got: &[f32], want: &[f32]) -> f64 {
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

    /// A dump row of the file's `[rope | nope]` index-head order as the
    /// kernel's tail order (the load-time move's own): within each head, the
    /// nope part first.
    fn head_moved(dump: &[f32], head: usize, rope: usize) -> Vec<f32> {
        dump.chunks(head)
            .flat_map(|h| {
                let (r, nope) = h.split_at(rope);
                nope.iter().chain(r).copied()
            })
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

    // ------------------------------------------------------------ structure

    /// The captured step's kernel entries against the list the layers derive
    /// (the module doc's), and the span: one path, no node beside another.
    fn structure(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        let hp = &case.hp;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let list = m.step_graph_nodes()?;
        let span = m.step_graph_span()?;
        let hist = histogram(&list);
        let ([host], _) = count_kinds(&list, [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST]);
        let (dense, roped_full, roped_shared, nope_owner, nope_reader) = (0usize, 1, 2, 3, 4);
        let full = [dense, roped_full, nope_owner, nope_reader];
        let owners = [dense, roped_full, roped_shared, nope_owner];
        let routed = [roped_full, roped_shared, nope_owner, nope_reader];
        let got = |name: &str| hist.get(name).copied().unwrap_or(0);
        // One entry a name a layer kind holds, by the body's own launch list
        // (its module doc): the counts are per full/shared/owner/reader/dense
        // layer and the head's own five.
        let (layers, rope_layers) = (hp.n_layer, [dense, roped_full, roped_shared].len());
        let want: Vec<(String, usize)> = vec![
            ("embed_rows_q6k".into(), 1),
            ("norm_quant".into(), layers),
            // q and o a layer, the index queries a full one, the head its own.
            ("q6k_gemv".into(), 2 * layers + full.len() + 1),
            // the o form and the index queries' a full layer, the experts'
            // two and the head's own quantizations.
            (
                "q3k_quantize_q8_1".into(),
                layers + full.len() + 2 * routed.len() + 1,
            ),
            ("f32_gemv".into(), full.len() + routed.len()),
            // kv_a an owner, the index key a full layer, the dense lead's
            // three and a shared expert's three a routed layer.
            (
                "q8_0_gemv".into(),
                owners.len() + full.len() + 3 * (1 + routed.len()),
            ),
            ("rope".into(), rope_layers),
            ("kv_norm_rope_append".into(), rope_layers),
            ("latent_rms_append".into(), [nope_owner].len()),
            ("ln_rows".into(), full.len()),
            // the index head's two permutations a roped full layer (a NoPE
            // index head has no rope rows to move), the flash row's rope
            // spans a roped one.
            (
                "gather_pairs".into(),
                2 * [dense, roped_full].len() + rope_layers,
            ),
            // the index queries' pair a full layer, the joined row's every.
            ("ssmax_rows".into(), full.len() + layers),
            ("ds41_index_key_input".into(), full.len()),
            ("ds41_indexer_score".into(), full.len()),
            ("ds41_indexer_topk".into(), full.len()),
            ("q8_0_gemv_heads".into(), 2 * layers),
            ("kv_gather_sel".into(), rope_layers),
            ("flash_latent".into(), rope_layers),
            ("ds41_attn_seg_sel".into(), [nope_owner, nope_reader].len()),
            ("ds41_attn_merge".into(), [nope_owner, nope_reader].len()),
            ("add".into(), 3 * routed.len() + 2 * (layers - routed.len())),
            // every layer's ffn_norm and the head's final norm.
            ("rms_norm".into(), layers + 1),
            ("qwen3moe_router_route".into(), routed.len()),
            ("iq1_s_gate_up_sel".into(), ty_count(case, "iq1_s")),
            ("iq1_s_gemv_sel".into(), ty_count(case, "iq1_s_down")),
            ("iq2_xxs_gate_up_sel".into(), ty_count(case, "iq2_xxs")),
            ("iq2_xxs_gemv_sel".into(), ty_count(case, "iq2_xxs_down")),
            ("weighted_sum".into(), routed.len()),
            ("swiglu".into(), 1 + routed.len()),
            ("softcap_rows".into(), 1),
            ("argmax_fault".into(), 1),
        ];
        let mut pass = host == 0;
        for (name, n) in &want {
            pass &= got(name) == *n;
        }
        pass &= span == nodes;
        let listed = want
            .iter()
            .map(|(name, n)| format!("{name}={}/{n}", got(name)))
            .chain(std::iter::once(format!("span={span}/{nodes}")))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "structure: layers={} (dense, roped full, roped shared, nope owner, nope reader), \
             graph_nodes={nodes} host={host}, entries (got/want) {listed} {}",
            hp.n_layer,
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

    /// How many of the fixture's routed layers hold an IQ1_S (the `gate_up`
    /// and `down` forms) or an IQ2_XXS (the `2` forms) stack — the counts the
    /// structure clause wants, read from the file's own types.
    fn ty_count(case: &Case, what: &str) -> usize {
        let hp = &case.hp;
        let (want, stem) = match what {
            "iq1_s" => (gguf::quant::GgmlType::IQ1_S, "ffn_gate_exps"),
            "iq1_s_down" => (gguf::quant::GgmlType::IQ1_S, "ffn_down_exps"),
            "iq2_xxs" => (gguf::quant::GgmlType::IQ2_XXS, "ffn_gate_exps"),
            "iq2_xxs_down" => (gguf::quant::GgmlType::IQ2_XXS, "ffn_down_exps"),
            _ => return 0,
        };
        (1..hp.n_layer)
            .filter(|&l| case_file_type(case, &format!("blk.{l}.{stem}.weight")) == Some(want))
            .count()
    }

    /// The file's type of tensor `name`, held open by the case.
    fn case_file_type(case: &Case, name: &str) -> Option<gguf::quant::GgmlType> {
        case.file.find(name).map(|(_, t)| t.ty)
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

    fn router_lane_nan(m: &mut Model, _case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "router_lane_nan";
        let layer = 1;
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
        let (top_k, n_layer) = (hp.indexer_top_k, hp.n_layer);
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
            let (gpu, b) = body_of(m)?;
            let lists = b.lists(gpu, 0..n_layer)?;
            let mut scores: Vec<Option<Vec<f32>>> = vec![None; n_layer];
            for l in 0..n_layer {
                if KINDS[l].2 {
                    scores[l] = Some(b.index_scores(gpu, l)?);
                }
            }
            for l in 0..n_layer {
                let (src, list) = &lists[l];
                let n = (at + 1).min(top_k);
                if KINDS[l].2 {
                    let scores = scores[l].as_ref().unwrap();
                    let rows = &scores[..(at + 1).min(scores.len())];
                    let want = rule_top(rows, n);
                    // The list is in row order (the kernel's own), the
                    // rule's in rank order: the SET is the contract.
                    let mut got: Vec<usize> = list[..n].iter().map(|&r| r as usize).collect();
                    let mut want = want;
                    got.sort_unstable();
                    want.sort_unstable();
                    if got != want {
                        // A tie at the threshold cuts by row order; name it.
                        let kth = rows[want[want.len() - 1]];
                        let tied = rows
                            .iter()
                            .enumerate()
                            .filter(|&(r, &v)| v == kth && !want.contains(&r))
                            .count();
                        if tied > 0 && got.iter().all(|g| rows[*g] == kth) {
                            named += 1;
                            continue;
                        }
                        println!(
                            "{CLAUSE}: layer {l} at position {at}: the sets differ: {got:?} vs \
                             {want:?} FAIL"
                        );
                        ok = false;
                    }
                } else if list[..n] != lists[*src].1[..n] {
                    println!(
                        "{CLAUSE}: shared layer {l} at position {at}: its list is not source \
                         {src}'s FAIL"
                    );
                    ok = false;
                }
            }
        }
        println!(
            "{CLAUSE}: {} positions past top-k {top_k}, every full layer's set the rule's and \
             every shared layer's its source's ({named} threshold ties named) {}",
            PROMPT_LEN - top_k,
            verdict(ok)
        );
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

    /// One layer's compared values against the dump's, at every prompt
    /// position of sets a and b, the layer alone on mainline's own input.
    fn nodes_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "nodes_near_mainlines";
        let Some(sets) = case.sets.as_ref() else {
            return Err("the fixture sets are open in the fixture tier only".into());
        };
        let hp = &case.hp;
        let hidden = hp.n_embd;
        let (top_k, used) = (hp.indexer_top_k, hp.n_used);
        let mut worst = 0.0_f64;
        let mut worst_cos = 1.0_f64;
        let mut ok = true;
        let mut compared = 0usize;
        let mut positions = 0usize;
        m.set_mode(StepMode::Eager);
        if m.has_capture() {
            println!("{CLAUSE}: a capture exists FAIL");
            return Ok(false);
        }
        let (gpu, b) = body_of(m)?;
        b.arm_taps(gpu, true)?;
        // Set b's longer prompt walks past the top-k for most of its
        // positions, set a's for its last two: the selected attention's
        // rows come mostly from b.
        for (set, man) in [("a", &sets.a), ("b", &sets.b)] {
            let ids = prompt_tables::tokens(man)?.to_vec();
            let n_pos = ids.len();
            positions += n_pos;
            let l_out = |l: usize, p: usize| -> Result<Vec<f32>, GateError> {
                let rows = tap_at(man, &format!("l_out-{l}"), 0)?;
                let at = p * hidden;
                Ok(rows
                    .get(at..at + hidden)
                    .ok_or("a row past the dump")?
                    .to_vec())
            };
            for (l, &(name, ropes, full, owns, dense, source, tk)) in
                KINDS.iter().enumerate().take(hp.n_layer)
            {
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
                for p in 0..n_pos {
                    let forced: &[usize] = if source < l { &[source, l] } else { &[l] };
                    for &at in forced {
                        if at > 0 {
                            let (gpu, b) = body_of(m)?;
                            b.set_input_override(gpu, at, &l_out(at - 1, p)?)?;
                        }
                    }
                    if or_fault(CLAUSE, "the forced layer", feed(m, &[ids[p]]))?.is_none() {
                        let (gpu, b) = body_of(m)?;
                        b.arm_taps(gpu, false)?;
                        return Ok(false);
                    }
                    let nodes = {
                        let (gpu, b) = body_of(m)?;
                        b.nodes(gpu, l..l + 1)?
                    };
                    let node = &nodes[0];
                    // A dump row by its own width and row count (`ne[0]` wide, the
                    // file's values over it): the last layer's FFN chain (the
                    // residual the head reads) the dump holds at the output
                    // position alone, so every other position compares nothing
                    // there (an empty want).
                    let row = |what: &str, occ: u32| -> Result<Vec<f32>, GateError> {
                        let name = format!("{what}-{l}");
                        let (width, rows) = dump_rows(&man.tensor(&name, occ)?.ne);
                        let all = tap_at(man, &name, occ)?;
                        assert_eq!(all.len(), width * rows, "{name}: {rows} rows");
                        if rows == 1 {
                            return Ok(if p + 1 == n_pos {
                                all[..width].to_vec()
                            } else {
                                Vec::new()
                            });
                        }
                        assert_eq!(rows, n_pos, "{name}: {rows} rows");
                        Ok(all[p * width..(p + 1) * width].to_vec())
                    };
                    // An i32 node's logical twin (the expert ids), this position.
                    let ids_row = |what: &str| -> Result<Vec<u32>, GateError> {
                        let name = format!("{what}-{l}");
                        let (width, rows) = dump_rows(&man.tensor(&name, 0)?.ne);
                        let all = ref_ints(man, &name, 0, RowKind::Tensor, Layout::Logical)?;
                        assert_eq!(all.len(), width * rows, "{name}: {rows} rows");
                        let ids = |v: i64| u32::try_from(v).unwrap_or(u32::MAX);
                        if rows == 1 {
                            return Ok(if p + 1 == n_pos {
                                all[..width].iter().map(|&v| ids(v)).collect()
                            } else {
                                Vec::new()
                            });
                        }
                        assert_eq!(rows, n_pos, "{name}: {rows} rows");
                        Ok(all[p * width..(p + 1) * width]
                            .iter()
                            .map(|&v| ids(v))
                            .collect())
                    };
                    // The selection against the dump's own `top_k`, past the
                    // boundary where the node stops being the argsort's identity
                    // padding (a shared layer's is its source's list, scored on
                    // the source's head): a set that crossed inside the
                    // indexer's own band (the k-th gap under the scores' peak
                    // times it) excuses the attention-fed nodes at this position.
                    let mut idx_crossed = false;
                    if p + 1 > top_k {
                        let name = format!("top_k-{tk}");
                        let (width, rows) = dump_rows(&man.tensor(&name, 0)?.ne);
                        let all = ref_ints(man, &name, 0, RowKind::Tensor, Layout::Flat)?;
                        assert_eq!(all.len(), width * rows, "{name}: {rows} rows");
                        let want: Vec<u32> = all[p * width..(p + 1) * width]
                            .iter()
                            .map(|&v| u32::try_from(v).unwrap_or(u32::MAX))
                            .collect();
                        let mut got = node.list[..top_k].to_vec();
                        got.sort_unstable();
                        let mut want = want;
                        want.sort_unstable();
                        compared += 1;
                        if got != want {
                            let scores = {
                                let (gpu, b) = body_of(m)?;
                                b.index_scores(gpu, if full { l } else { source })?
                            };
                            let live = &scores[..p + 1];
                            let peak = live.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
                            let mut order = live.to_vec();
                            order.sort_by(|a, b| b.partial_cmp(a).unwrap());
                            let kth = order[top_k - 1];
                            let outside: Vec<u32> = want
                                .iter()
                                .filter(|id| !got.contains(id))
                                .copied()
                                .collect();
                            let band = DENSE_REL_BAND as f32 * peak;
                            let excused = !outside.is_empty()
                                && outside.iter().all(|&r| kth - live[r as usize] <= band);
                            if excused {
                                idx_crossed = true;
                                println!(
                                    "{CLAUSE}: top_k-{tk} p{p} {got:?} != {want:?} inside the \
                                 indexer's band (k-th gap {:.2e} ≤ {:.2e}): the attention-fed \
                                 nodes skip here",
                                    kth - live[outside[0] as usize],
                                    band
                                );
                            } else {
                                println!("{CLAUSE}: top_k-{tk} p{p} {got:?} != {want:?} FAIL");
                                ok = false;
                            }
                        }
                    }
                    let pairs: Vec<(&str, Vec<f32>, f64)> = vec![
                        ("attn_norm", node.attn_norm.clone(), DENSE_REL_BAND),
                        ("ffn_inp", node.ffn_inp.clone(), DENSE_REL_BAND),
                    ];
                    // The attention's own output against the reference's
                    // `fattn_mla` (the heads' values up-projected): the dump's
                    // `[head_v, n_pos, heads]` layout, head-major a position.
                    // The card's Q6_K query and Q8_0 key projections round
                    // differently from mainline's, and the softmax amplifies
                    // that onto this row. A position whose selection crossed
                    // compares nothing here.
                    if !idx_crossed {
                        let all = tap_at(man, &format!("fattn_mla-{l}"), 0)?;
                        let (hv, tokens, n_head) = (hp.head_v, n_pos, hp.n_head);
                        assert_eq!(all.len(), hv * tokens * n_head);
                        let want: Vec<f32> = (0..n_head * hv)
                            .map(|i| {
                                let (h, d) = (i / hv, i % hv);
                                all[h * hv * tokens + p * hv + d]
                            })
                            .collect();
                        let e = rel(&node.av, &want);
                        compared += 1;
                        if e > ATTN_REL_BAND {
                            println!(
                                "{CLAUSE}: fattn_mla-{l} p{p} rel {e:.4} > {ATTN_REL_BAND} FAIL"
                            );
                            print_samples(CLAUSE, &node.av, &want);
                            ok = false;
                        }
                        worst = worst.max(e);
                        seen.push(("fattn_mla".to_string(), p, e));
                    }
                    for (what, got, band) in pairs {
                        let want = row(what, 0)?;
                        if want.is_empty() || (idx_crossed && what == "ffn_inp") {
                            continue;
                        }
                        let e = rel(&got, &want);
                        compared += 1;
                        if e > band {
                            println!("{CLAUSE}: {what}-{l} p{p} rel {e:.4} > {band} FAIL");
                            print_samples(CLAUSE, &got, &want);
                            ok = false;
                        }
                        worst = worst.max(e);
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
                        let e = rel(&node.q_cur, &want);
                        compared += 1;
                        if e > DENSE_REL_BAND {
                            println!("{CLAUSE}: Qcur_ssmax-{l} p{p} rel {e:.4} FAIL");
                            print_samples(CLAUSE, &node.q_cur, &want);
                            ok = false;
                        }
                        worst = worst.max(e);
                        seen.push(("Qcur_ssmax".to_string(), p, e));
                    }
                    if full {
                        // The index queries and the key's projection stage: the
                        // dump holds the file's `[rope | nope]` head order, a
                        // roped layer's card row the kernel's tail order (the
                        // load-time move's), a NoPE layer's the file's own.
                        let taps: Vec<(&str, Vec<f32>)> = vec![
                            ("indexer_q_ssmax", node.qi.clone()),
                            ("indexer_k", node.kidx.clone()),
                        ];
                        for (what, got) in taps {
                            let rope = if ropes { hp.rope_dims } else { 0 };
                            let want = head_moved(&row(what, 0)?, hp.indexer_d, rope);
                            let e = rel(&got, &want);
                            compared += 1;
                            if e > DENSE_REL_BAND {
                                println!("{CLAUSE}: {what}-{l} p{p} rel {e:.4} FAIL");
                                print_samples(CLAUSE, &got, &want);
                                ok = false;
                            }
                            worst = worst.max(e);
                            seen.push((what.to_string(), p, e));
                        }
                        {
                            let (what, band) = ("indexer_weights", DENSE_REL_BAND);
                            let want = row(what, 0)?;
                            let e = rel(&node.wi, &want);
                            compared += 1;
                            if e > band {
                                println!("{CLAUSE}: {what}-{l} p{p} rel {e:.4} > {band} FAIL");
                                ok = false;
                            }
                            worst = worst.max(e);
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
                        // `kv_cmpr`'s two occurrences: the projection (`[latent |
                        // rope]`, the file's own row) and its normed latent.
                        let normed = if ropes {
                            node.kv_s[..latent].to_vec()
                        } else {
                            cache_row()
                        };
                        let taps: Vec<(&str, Vec<f32>, u32, f64)> = vec![
                            ("kv_cmpr", node.kv_a.clone(), 0, DENSE_REL_BAND),
                            ("kv_cmpr", normed, 1, DENSE_REL_BAND),
                        ];
                        for (what, got, occ, band) in taps {
                            let want = row(what, occ)?;
                            let e = rel(&got, &want);
                            compared += 1;
                            if e > band {
                                println!("{CLAUSE}: {what}-{l} occ {occ} p{p} rel {e:.4} FAIL");
                                print_samples(CLAUSE, &got, &want);
                                ok = false;
                            }
                            worst = worst.max(e);
                            seen.push((format!("{what}#{occ}"), p, e));
                        }
                        // `Kcur`: a rope layer's appended row is `[k_rope |
                        // kv_compressed]`, the reference's concat the other way;
                        // a NoPE layer's row is the latent, which the cache
                        // holds, with a zero rope part after it.
                        if ropes {
                            let want = rope_last(&row("Kcur", 0)?, latent + rope_dims, rope_dims);
                            let e = rel(&node.kvr, &want);
                            compared += 1;
                            if e > DENSE_REL_BAND {
                                println!("{CLAUSE}: Kcur-{l} p{p} rel {e:.4} FAIL");
                                print_samples(CLAUSE, &node.kvr, &want);
                                ok = false;
                            }
                            worst = worst.max(e);
                            seen.push(("Kcur".to_string(), p, e));
                        } else {
                            let want = row("Kcur", 0)?[..latent].to_vec();
                            let got = cache_row();
                            let e = rel(&got, &want);
                            compared += 1;
                            if e > DENSE_REL_BAND {
                                println!("{CLAUSE}: Kcur-{l} p{p} rel {e:.4} FAIL");
                                print_samples(CLAUSE, &got, &want);
                                ok = false;
                            }
                            worst = worst.max(e);
                            seen.push(("Kcur".to_string(), p, e));
                        }
                    }
                    if dense && !idx_crossed {
                        // The dense lead's own block output, fed by its
                        // attention.
                        let want = row("ffn_out", 0)?;
                        if !want.is_empty() {
                            let e = rel(&node.ffn_out, &want);
                            compared += 1;
                            if e > EXPERT_REL_BAND {
                                println!("{CLAUSE}: ffn_out-{l} p{p} rel {e:.4} FAIL");
                                print_samples(CLAUSE, &node.ffn_out, &want);
                                ok = false;
                            }
                            worst = worst.max(e);
                            seen.push(("ffn_out".to_string(), p, e));
                        }
                    }
                    let mut crossed = false;
                    if !dense && !idx_crossed {
                        // The router's logits (cosine, the real lanes alone: the
                        // routing body runs its pads at f32::MIN), the ids (a
                        // set: the kernel writes them row-ascending, mainline in
                        // rank order), the weights and the experts' outputs. A
                        // set that crossed inside the logits' own noise (the
                        // flip's gap under twice the run's max |Δ| against the
                        // dump's own probabilities) excuses the nodes the
                        // router's choice feeds at this position.
                        let want_logits = row("ffn_moe_logits", 0)?;
                        if !want_logits.is_empty() {
                            let c = cosine(&node.route_logits[..hp.n_expert], &want_logits);
                            worst_cos = worst_cos.min(c);
                            seen.push(("ffn_moe_logits 1-cos".to_string(), p, 1.0 - c));
                            compared += 1;
                            if c < COSINE_FLOOR {
                                println!("{CLAUSE}: ffn_moe_logits-{l} p{p} cosine {c:.6} FAIL");
                                ok = false;
                            }
                        }
                        let mut got_ids = node.route_ids.clone();
                        let mut want_ids = ids_row("ffn_moe_topk")?;
                        if !want_ids.is_empty() {
                            got_ids.sort_unstable();
                            want_ids.sort_unstable();
                            compared += 1;
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
                                        "{CLAUSE}: ffn_moe_topk-{l} p{p} {got_ids:?} != {want_ids:?} \
                                     inside the logits' noise (Δp {dp:.2e}): the router-fed \
                                     nodes skip here"
                                    );
                                } else {
                                    println!(
                                        "{CLAUSE}: ffn_moe_topk-{l} p{p} {got_ids:?} != {want_ids:?} FAIL",
                                    );
                                    ok = false;
                                }
                            }
                        }
                        // PIN(2026-11-01): the router's renormalised weights
                        // and the shared expert keep DENSE_REL_BAND, measured
                        // 0.0164 and 0.0244 at their frontiers over both sets
                        // (the shared expert's at a router crossing, which it
                        // does not skip); the experts' sum and the block's
                        // output, which the routed experts feed, the routed
                        // band.
                        let taps: Vec<(&str, Vec<f32>, f64)> = vec![
                            (
                                "ffn_moe_weights_norm",
                                node.route_weights.clone(),
                                DENSE_REL_BAND,
                            ),
                            ("ffn_moe_out", node.moe.clone(), EXPERT_REL_BAND),
                            ("ffn_shexp", node.sh.clone(), DENSE_REL_BAND),
                            ("ffn_out", node.ffn_out.clone(), EXPERT_REL_BAND),
                        ];
                        for (what, got, band) in taps {
                            let want = row(what, 0)?;
                            if want.is_empty() || (crossed && what != "ffn_shexp") {
                                continue;
                            }
                            let e = rel(&got, &want);
                            compared += 1;
                            if e > band {
                                println!("{CLAUSE}: {what}-{l} p{p} rel {e:.4} > {band} FAIL");
                                ok = false;
                            }
                            worst = worst.max(e);
                            seen.push((what.to_string(), p, e));
                        }
                    }
                    // The block's output, the residual plus the router-fed FFN:
                    // it skips where the attention or the router's set crossed
                    // (`gate_musespark_e2e`'s `after_router` rule).
                    let want = row("l_out", 0)?;
                    if !want.is_empty() && !(idx_crossed || crossed) {
                        let e = rel(&node.l_out, &want);
                        compared += 1;
                        if e > EXPERT_REL_BAND {
                            println!(
                                "{CLAUSE}: l_out-{l} p{p} rel {e:.4} > {EXPERT_REL_BAND} FAIL"
                            );
                            print_samples(CLAUSE, &node.l_out, &want);
                            ok = false;
                        }
                        worst = worst.max(e);
                        seen.push(("l_out".to_string(), p, e));
                    }
                    let _ = (top_k, used, full);
                }
                println!(
                    "{CLAUSE}: set {set} layer {l} ({name}) frontier: {}",
                    frontier(&seen)
                );
                {
                    let (_gpu, b) = body_of(m)?;
                    b.set_layer_span(None)?;
                    b.clear_input_override();
                }
                eprintln!("clause nodes_near_mainlines: set {set} layer {l} ({name})");
            }
        }
        {
            let (gpu, b) = body_of(m)?;
            b.set_layer_span(None)?;
            b.arm_taps(gpu, false)?;
        }
        println!(
            "{CLAUSE}: {} comparisons over {} layers x {positions} positions (sets a and b), \
             worst rel {worst:.4}, worst logit cosine {worst_cos:.6} {}",
            compared,
            hp.n_layer,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The greedy 32 of sets a and b, teacher-forced on the oracle's own ids.
    fn greedy_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "greedy_is_mainlines";
        let Some(sets) = case.sets.as_ref() else {
            return Err("the fixture sets are open in the fixture tier only".into());
        };
        let vocab = case.hp.n_vocab;
        let mut ok = true;
        for (label, man, band) in [
            ("a", &sets.a, GREEDY_MARGIN_A),
            ("b", &sets.b, GREEDY_MARGIN_B),
        ] {
            let ids = prompt_tables::tokens(man)?.to_vec();
            let Greedy {
                ids: want, margins, ..
            } = prompt_tables::greedy(man)?;
            m.set_mode(StepMode::Graph);
            m.reset()?;
            let Some(prompt) = or_fault(CLAUSE, "the prompt", feed(m, &ids))? else {
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
                let Some(r) = or_fault(CLAUSE, "a greedy step", feed(m, &[t]))? else {
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
            let mut delta = 0.0_f64;
            for t in &topk {
                let got = first.get(t.id as usize).copied().unwrap_or(f32::NAN);
                delta = delta.max((f64::from(got) - f64::from(t.logit)).abs());
            }
            let mut agree = 0usize;
            let mut excused = 0usize;
            for (s, (&got, (&want_id, &margin))) in
                run.iter().zip(want.iter().zip(margins.iter())).enumerate()
            {
                if got == want_id {
                    agree += 1;
                } else if f64::from(margin) <= band {
                    excused += 1;
                    println!(
                        "{CLAUSE}: set {label} step {s}: ids {got}/{} margin {:.4} below the band \
                         {band:.4}",
                        want_id,
                        f64::from(margin)
                    );
                } else {
                    println!(
                        "{CLAUSE}: set {label} step {s}: id {got} != {} at margin {:.4} FAIL",
                        want_id,
                        f64::from(margin)
                    );
                    ok = false;
                }
            }
            ok &= self_argmax;
            println!(
                "{CLAUSE}: set {label}: {}/{} agree, {excused} below the band {band}, max |Δ| \
                 {delta:.4} against topk, the run's own ids its argmax {self_argmax} {}",
                agree,
                want.len(),
                verdict(ok)
            );
        }
        Ok(ok)
    }

    // ------------------------------------------------------------------ run

    fn open_sets(tier_now: tier::Tier) -> Result<Option<Sets>, GateError> {
        Ok(match tier_now {
            tier::Tier::Fixture => {
                let a = RefManifest::open(
                    &tier::fixture_set("the node clauses", &fixture::LCPP, fixture::A)?,
                    &fixture::LCPP,
                )?;
                let b = RefManifest::open(
                    &tier::fixture_set("the greedy clauses", &fixture::LCPP, fixture::B)?,
                    &fixture::LCPP,
                )?;
                Some(Sets { a, b })
            }
            tier::Tier::Real => None,
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
        // set's demand (the fixture keeps the real file's trained context,
        // so the load's own bound never cuts it).
        let mut longest = PROMPT_LEN;
        if let Some(s) = sets.as_ref() {
            for man in [&s.a, &s.b] {
                let ids = prompt_tables::tokens(man)?.len();
                let greedy = prompt_tables::greedy(man)?.ids.len();
                longest = longest.max(ids + greedy);
            }
        }
        let ctx = longest + CTX_SPARE;
        let walk = walk(PROMPT_LEN, hp.n_vocab);
        let t = std::time::Instant::now();
        let mut m = Body::open(Gpu::new()?, file, ctx)?;
        println!(
            "load resident_bytes={} ctx={ctx} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let case = Case {
            hp,
            walk,
            file: open_split(Arch::MistralLarge4, "gate-gpu-mistrallarge4-e2e")?,
            sets,
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
            "PASSED: {NAME} the captured step holds the entries its five layer kinds derive; \
             eager and replay give bit-identical logits; a reset rewinds; the cache rows carry; a \
             NaN router lane raises the fault word and is picked by no slot; every full indexer \
             layer's list is the rule's own top-k and every shared layer's its source's; every \
             compared node keeps its band at every layer and position; the greedy continuation \
             keeps the oracle's argmax above the band; the fault word stays clean"
        );
        Ok(())
    }
}
