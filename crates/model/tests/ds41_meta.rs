//! V4.1 metadata gate: what `arch::deepseek41::hparams` resolves from the
//! served file's headers, against what ik resolves from the same file. Three
//! contracts, one test each; each prints what it compared, then fails with
//! the whole list of what differs.
//!
//! 1. `hw_ds41_hparams_match_ik` — every `Hparams` field equals the value ik
//!    takes from this file. A value ik prints at load is pinned from that
//!    printout (`$BLOOMERY_DATA/ikppl/after-idxkey.log`, a full ik load of the
//!    served file); a value it does not print is pinned from the header, with
//!    the ik line that reads it; a derived value says so.
//! 2. `hw_ds41_layer_kinds_match_ik_graph` — every `LayerKind` flag and source
//!    against the graph ik built from this file, read off two oracle manifests
//!    by node name (ik lines are `src/graphs/build_deepseek4.cpp`):
//!
//!    | `LayerKind` | node of layer L | line | sets |
//!    |---|---|---|---|
//!    | no stream (window only) | `attn_raw-L` | 1404 | both |
//!    | stream in the first / second segment | `csa_k_all-L` / `hca_k_all-L` | 1322 | both |
//!    | owns a compressor | `csa_state_compress-L` / `hca_state_compress-L` | 951 | both |
//!    | its compressor is gated | a node reads `blk.L.attn_compressor_gate.weight` | — | both |
//!    | owns index keys | `lid_k_write-L` | 1171 | both |
//!    | engram site | `engram_out-L` | 1045 | both |
//!    | `kv_source` | the layer owning a compressor whose `*_k-S` views the cache leaf `*_k-L` views | 1321 | shared |
//!    | runs the indexer | `lid_top_k-L` | 885, 907 | step |
//!    | `topk_source` | the `lid_top_k-S` layer L's nodes read | — | step |
//!    | `index_key_source` (indexer layers) | the owner of index keys whose `lid_k-S` views the leaf `lid_k-L` views | 861 | step |
//!    | `Indexer::top_k` under the set's `--override-kv` | `lid_top_k-L`'s width | 879 | step |
//!
//!    The shared set (`ref_deepseek41`, a 5-token prefill) builds no indexer:
//!    ik builds it only when `top_k` is below the padded key count
//!    (1344-1348), which 5 tokens never reach. The step set
//!    (`ref_deepseek41_d1_every_node`, one decode step at position 301 with the
//!    top-k overridden to 64) builds it, and there the compressed rows reach a
//!    layer as the top-k's gather, so the stream's cache leaf is read from the
//!    shared set only. A node belongs to the layer whose `l_out-L` follows it.
//! 3. `hw_ds41_names_in_file` — every name `names` gives exists on exactly the
//!    layers its kind says, and every tensor of the file is one of them or one
//!    the step does not read.
//!
//! `hw_`: needs the V4.1 shards and the oracle sets on the box
//! (`just gate-ds41-meta`); `BLOOMERY_V41_MODEL` names another first shard.
//! Headers and manifests only: seconds.

use std::collections::BTreeSet;
use std::fmt::{Debug, Write as _};
use std::path::Path;

use gguf::Split;
use model::arch::deepseek41::hparams::{
    CandidateMask, Collapse, Hparams, LayerKind, Model, Rope, Score,
};
use model::arch::deepseek41::names;
use model::placement::workstation;
use refset::arch::deepseek41::IK;
use refset::ik::RefManifest;

/// The oracle set of the shared prefill and the decode-step set with the
/// indexer built.
const SHARED_SET: &str = refset::arch::deepseek41::BATCH;
const STEP_SET: &str = refset::arch::deepseek41::D1;

/// The one tensor family the file carries and the text step does not read:
/// the router bias ik swaps in for image tokens (build_deepseek4.cpp:1584-1588).
const UNREAD: &[&str] = &["exp_probs_b_vl.bias"];

fn open() -> (Split, Hparams) {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    (split, hp)
}

/// One compared value: a line of the printed table, and an entry of `bad`
/// when it differs.
fn row<T: PartialEq + Debug>(
    out: &mut String,
    bad: &mut Vec<String>,
    what: &str,
    got: T,
    want: T,
    source: &str,
) {
    let mark = if got == want { "ok " } else { "BAD" };
    let _ = writeln!(out, "  {mark} {what:<34} {got:?}  [{source}]");
    if got != want {
        bad.push(format!("{what}: ours {got:?}, ik {want:?} [{source}]"));
    }
}

/// [`row`] for one of many per-layer values: printed only when it differs.
fn layer_row<T: PartialEq + Debug>(
    out: &mut String,
    bad: &mut Vec<String>,
    what: &str,
    got: T,
    want: T,
    source: &str,
) -> bool {
    if got == want {
        return true;
    }
    let _ = writeln!(out, "  BAD {what:<34} {got:?}  [{source}]");
    bad.push(format!("{what}: ours {got:?}, ik {want:?} [{source}]"));
    false
}

/// `layers` as the compact list `2-7,9`.
fn ranges(layers: impl IntoIterator<Item = usize>) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut run: Option<(usize, usize)> = None;
    for l in layers {
        run = match run {
            Some((a, b)) if l == b + 1 => Some((a, l)),
            Some((a, b)) => {
                out.push(if a == b {
                    a.to_string()
                } else {
                    format!("{a}-{b}")
                });
                Some((l, l))
            }
            None => Some((l, l)),
        };
    }
    if let Some((a, b)) = run {
        out.push(if a == b {
            a.to_string()
        } else {
            format!("{a}-{b}")
        });
    }
    out.join(",")
}

fn fail_if_bad(out: &str, bad: &[String]) {
    println!("{out}");
    assert!(
        bad.is_empty(),
        "{} value(s) differ from ik:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

#[test]
#[ignore = "hw: needs the V4.1 shards on the box"]
fn hw_ds41_hparams_match_ik() {
    let (split, hp) = open();
    let mut out = String::new();
    let mut bad = Vec::new();
    let o = &mut out;
    let b = &mut bad;

    // PIN(2026-09-23): ik's load print of the served file (after-idxkey.log) and the file's header; ik lines at the oracle tree 49ef19d0.
    let _ = writeln!(o, "shape");
    row(o, b, "n_layer", hp.n_layer, 40, "ik print n_layer = 40");
    row(o, b, "n_embd", hp.n_embd, 5120, "ik print n_embd = 5120");
    row(o, b, "n_head", hp.n_head, 64, "ik print n_head = 64");
    row(o, b, "n_head_kv", hp.n_head_kv, 1, "ik print n_head_kv = 1");
    row(
        o,
        b,
        "head_dim",
        hp.head_dim,
        512,
        "ik print n_embd_head_k = n_embd_head_v = 512",
    );
    row(
        o,
        b,
        "q_lora_rank",
        hp.q_lora_rank,
        1280,
        "header q_lora_rank; llama-hparams.cpp:1985",
    );
    row(
        o,
        b,
        "o_groups",
        hp.o_groups,
        8,
        "header output_group_count; llama-hparams.cpp:2043",
    );
    row(
        o,
        b,
        "o_lora_rank",
        hp.o_lora_rank,
        1024,
        "header output_lora_rank; llama-hparams.cpp:2047",
    );
    row(o, b, "rope_dims", hp.rope_dims, 64, "ik print n_rot = 64");
    row(o, b, "window", hp.window, 128, "ik print n_swa = 128");
    row(
        o,
        b,
        "rms_eps",
        hp.rms_eps,
        1e-20,
        "ik print f_norm_rms_eps = 1.0e-20",
    );
    row(
        o,
        b,
        "n_vocab",
        hp.n_vocab,
        129_280,
        "ik print n_vocab = 129280 (no vocab_size key)",
    );
    let head_rows = split
        .find(&names::output())
        .map(|(_, t)| t.dims[1] as usize);
    row(
        o,
        b,
        "n_vocab = output rows",
        Some(hp.n_vocab),
        head_rows,
        "the file's output.weight",
    );
    row(
        o,
        b,
        "n_ctx_train",
        hp.n_ctx_train,
        1_048_576,
        "ik print n_ctx_train = 1048576",
    );
    row(o, b, "csa_ratio", hp.csa_ratio, 2, "ik print csa ratio 2");
    row(o, b, "hca_ratio", hp.hca_ratio, 1, "ik print hca ratio 1");

    let _ = writeln!(o, "indexer");
    row(
        o,
        b,
        "indexer.n_head",
        hp.indexer.n_head,
        32,
        "header indexer.head_count; llama-hparams.cpp:2024",
    );
    row(
        o,
        b,
        "indexer.head_dim",
        hp.indexer.head_dim,
        128,
        "header indexer.key_length; llama-hparams.cpp:2025",
    );
    row(
        o,
        b,
        "indexer.top_k",
        hp.indexer.top_k,
        512,
        "header indexer.top_k; llama-hparams.cpp:2026",
    );
    let over = hp.clone().with_indexer_top_k(64);
    let mut rest = over.clone();
    rest.indexer.top_k = hp.indexer.top_k;
    row(
        o,
        b,
        "override: top_k",
        over.indexer.top_k,
        64,
        "with_indexer_top_k(64)",
    );
    row(
        o,
        b,
        "override: every other field",
        rest == hp,
        true,
        "with_indexer_top_k(64)",
    );
    row(
        o,
        b,
        "candidates (no candidate key)",
        hp.candidates,
        Some(CandidateMask {
            source_layer: 20,
            block_size: 8,
            topk_blocks: 2048,
        }),
        "ik's defaults when absent; llama-hparams.cpp:2318-2320 (81af2772)",
    );
    row(
        o,
        b,
        "candidate_free_positions",
        hp.candidate_free_positions(),
        16_384,
        "2048 blocks x 8 rows x layer 20's ratio 1",
    );

    let _ = writeln!(o, "hyper-connections");
    row(
        o,
        b,
        "hc.streams",
        hp.hc.streams,
        4,
        "header hyper_connection.count; llama-hparams.cpp:2056",
    );
    row(
        o,
        b,
        "hc.sinkhorn_iters",
        hp.hc.sinkhorn_iters,
        20,
        "header sinkhorn_iterations; llama-hparams.cpp:2067",
    );
    row(
        o,
        b,
        "hc.eps",
        hp.hc.eps,
        1e-6,
        "header hyper_connection.epsilon; llama-hparams.cpp:2070",
    );
    row(
        o,
        b,
        "collapse",
        hp.collapse,
        Collapse::Lagged,
        "no output_hc_* tensor; ik dsv4_hc_lag for deepseek41, llama-hparams.cpp:2191",
    );
    row(
        o,
        b,
        "model / q_head_norm",
        (hp.model, hp.q_head_norm),
        (Model::Deepseek41, false),
        "the file's architecture string; ik dsv4_q_head_norm for deepseek41, llama-hparams.cpp:2192",
    );

    let _ = writeln!(o, "experts");
    let e = &hp.experts;
    row(
        o,
        b,
        "experts.n_expert",
        e.n_expert,
        384,
        "ik print n_expert = 384",
    );
    row(
        o,
        b,
        "experts.n_used",
        e.n_used,
        6,
        "ik print n_expert_used = 6",
    );
    row(
        o,
        b,
        "experts.n_shared",
        e.n_shared,
        1,
        "header expert_shared_count; llama-hparams.cpp:1970",
    );
    row(
        o,
        b,
        "experts.ff",
        e.ff,
        2304,
        "header expert_feed_forward_length; llama-hparams.cpp:1955",
    );
    row(
        o,
        b,
        "experts.routed_scale",
        e.routed_scale,
        1.5,
        "header expert_weights_scale; llama-hparams.cpp:1972",
    );
    row(
        o,
        b,
        "experts.weights_norm",
        e.weights_norm,
        true,
        "header expert_weights_norm; llama-hparams.cpp:1973",
    );
    row(
        o,
        b,
        "experts.score",
        e.score,
        Score::SqrtSoftplus,
        "header expert_gating_func = 4; llama-hparams.cpp:2214, .h:18",
    );
    row(
        o,
        b,
        "experts.dense_lead",
        e.dense_lead,
        0,
        "derived: every layer has ffn_gate_inp; no leading_dense_block_count key, ik's optional read :1971 leaves 0",
    );
    row(
        o,
        b,
        "experts.hash_layers",
        e.hash_layers,
        0,
        "header hash_layer_count; llama-hparams.cpp:2073",
    );

    let _ = writeln!(o, "engram");
    let g = hp.engram.as_ref().expect("a V4.1 file has engram sites");
    row(
        o,
        b,
        "engram.layer_ids",
        g.layer_ids.clone(),
        vec![1, 14],
        "header engram.layer_ids; llama-hparams.cpp:2076-2083",
    );
    row(
        o,
        b,
        "engram.n_head",
        g.n_head,
        8,
        "header engram.head_count; llama-hparams.cpp:2085",
    );
    row(
        o,
        b,
        "engram.key_length",
        g.key_length,
        256,
        "header engram.key_length; llama-hparams.cpp:2086",
    );
    row(
        o,
        b,
        "engram.max_ngram",
        g.max_ngram,
        4,
        "header engram.max_ngram_size; llama-hparams.cpp:2087",
    );
    row(
        o,
        b,
        "engram.rows_per_token",
        g.rows_per_token(),
        24,
        "derived (4-1)*8; build_deepseek4.cpp:996",
    );
    for &l in &g.layer_ids {
        let input = split
            .find(&names::engram_wkv(l))
            .map(|(_, t)| t.dims[0] as usize);
        let what = format!("engram.embedding_width, layer {l}");
        row(
            o,
            b,
            &what,
            Some(g.embedding_width()),
            input,
            "derived 24*256 = the file's engram_wkv input width",
        );
    }

    let _ = writeln!(o, "rope");
    let window = Rope {
        base: 10_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        n_ctx_orig: 0,
    };
    let yarn = hp.layers[hp.n_layer - 1].rope;
    row(
        o,
        b,
        "compressed rope base",
        yarn.base,
        160_000.0,
        "header compress_rope_freq_base; llama-hparams.cpp:2051",
    );
    row(
        o,
        b,
        "compressed rope freq_scale",
        yarn.freq_scale,
        0.0625,
        "ik print freq_scale_train = 0.0625",
    );
    row(
        o,
        b,
        "compressed rope ext_factor",
        yarn.ext_factor,
        1.0,
        "derived: ik print rope scaling = yarn; llama.cpp:9059-9060",
    );
    // Derived: 1/(1 + 0.1 ln(1/freq_scale)) in f64 from the pinned factor; ours is ik's f32 op order.
    let want = 1.0 / (1.0 + 0.1 * 16f64.ln());
    let within = ((f64::from(yarn.attn_factor) - want) / want).abs() < 1e-6;
    let what = format!("compressed rope attn_factor {:?}", yarn.attn_factor);
    let source =
        format!("derived: within 1e-6 of 1/(1+0.1 ln 16) = {want:.9}; build_deepseek4.cpp:23-29");
    row(o, b, &what, within, true, &source);
    row(
        o,
        b,
        "compressed rope beta_fast",
        yarn.beta_fast,
        32.0,
        "header yarn_beta_fast; ik does not read it here, its default 32 (llama-hparams.h:88) is the same",
    );
    row(
        o,
        b,
        "compressed rope beta_slow",
        yarn.beta_slow,
        1.0,
        "header yarn_beta_slow; ik does not read it here, its default 1 (llama-hparams.h:89) is the same",
    );
    row(
        o,
        b,
        "compressed rope n_ctx_orig",
        yarn.n_ctx_orig,
        65_536,
        "ik print n_ctx_orig_yarn = 65536",
    );

    let _ = writeln!(o, "per layer (a layer that differs prints its own line)");
    let ratio_src =
        "header compress_ratios, the first 40 of arr[i32,43]; llama-hparams.cpp:2115-2121";
    let window_src = "ik print freq_base_train = 10000.0; build_deepseek4.cpp:1094-1100";
    let clamp_src = "header swiglu_clamp_exp/_shexp, arr[f32,40]; llama-hparams.cpp:1976-1977";
    let ratio = |l: usize| match l {
        0..=1 => 0,
        2..=19 => 2,
        _ => 1,
    };
    let (mut ratios_ok, mut ropes_ok, mut clamps_ok) = (true, true, true);
    for (l, k) in hp.layers.iter().enumerate() {
        let got = k.stream.map_or(0, |s| s.ratio);
        ratios_ok &= layer_row(o, b, &format!("layer {l} ratio"), got, ratio(l), ratio_src);
        let (want, source) = if ratio(l) == 0 {
            (window, window_src)
        } else {
            (yarn, "the compressed rope above")
        };
        ropes_ok &= layer_row(o, b, &format!("layer {l} rope"), k.rope, want, source);
        let clamps = (k.swiglu_limit, k.swiglu_limit_shared);
        let what = format!("layer {l} swiglu clamps");
        clamps_ok &= layer_row(o, b, &what, clamps, (10.0, 10.0), clamp_src);
    }
    let mark = |ok: bool| if ok { "ok " } else { "BAD" };
    let _ = writeln!(
        o,
        "  {} ratios 0-1: 0, 2-19: 2, 20-39: 1  [{ratio_src}]",
        mark(ratios_ok)
    );
    let _ = writeln!(
        o,
        "  {} rope 0-1: {window:?}  [{window_src}]",
        mark(ropes_ok)
    );
    let _ = writeln!(
        o,
        "  {} rope 2-39: the compressed rope above",
        mark(ropes_ok)
    );
    let _ = writeln!(
        o,
        "  {} swiglu clamps (10.0, 10.0) on 0-39  [{clamp_src}]",
        mark(clamps_ok)
    );

    fail_if_bad(&out, &bad);
}

/// A manifest's `tensor` row: the columns this gate reads, and the layer it
/// belongs to.
struct Node {
    name: String,
    occurrence: u32,
    ne0: u64,
    src: [String; 2],
    layer: usize,
}

/// One oracle set's manifest: its dumper's flags and its tensor rows, in graph
/// order.
struct Manifest {
    set: &'static str,
    flags: Option<String>,
    nodes: Vec<Node>,
}

impl Manifest {
    /// The set `set` of the V4.1 file the tree runs, through the node dumps'
    /// family check (`refset::ik`: complete, of the file the tree runs, the
    /// family's architecture and ik tree), and its `# model_file` the name of
    /// the file `split` opens.
    fn open(set: &'static str, split: &Split) -> Manifest {
        let man = RefManifest::open(&IK.path(set), &IK).unwrap_or_else(|e| {
            panic!(
                "{e}. The lead produces it (tools/ref/dump.sh). \
                 Do not run it yourself and do not skip this test."
            )
        });
        let mut nodes = Vec::new();
        let mut layer = 0;
        for r in &man.tensors {
            let src = |s: &Option<String>| {
                s.clone().unwrap_or_else(|| {
                    panic!(
                        "{set}: tensor row {}#{} has no source columns (a v1 row)",
                        r.name, r.occurrence
                    )
                })
            };
            let node = Node {
                name: r.name.clone(),
                occurrence: r.occurrence,
                ne0: r.ne[0],
                src: [src(&r.src0), src(&r.src1)],
                layer,
            };
            if node.occurrence == 0
                && let Some(l) = node
                    .name
                    .strip_prefix("l_out-")
                    .and_then(|n| n.parse::<usize>().ok())
            {
                layer = l + 1;
            }
            nodes.push(node);
        }
        let ours = split
            .shard_path(0)
            .and_then(Path::file_name)
            .map(|n| n.to_string_lossy().into_owned());
        assert_eq!(
            man.header.model_file, ours,
            "{set} was dumped from another file"
        );
        Manifest {
            set,
            flags: man.header.flags,
            nodes,
        }
    }

    /// The node `<stem>-<layer>` (its first occurrence), if ik built it.
    fn node(&self, stem: &str, layer: usize) -> Option<&Node> {
        let name = format!("{stem}-{layer}");
        self.nodes
            .iter()
            .find(|n| n.occurrence == 0 && n.name == name)
    }

    fn has(&self, stem: &str, layer: usize) -> bool {
        self.node(stem, layer).is_some()
    }

    /// Whether any node reads `tensor`.
    fn reads(&self, tensor: &str) -> bool {
        self.nodes.iter().any(|n| n.src.iter().any(|s| s == tensor))
    }

    /// The leaf `<stem>-<layer>` views: the cache it reads.
    fn leaf(&self, stem: &str, layer: usize) -> Option<&str> {
        self.node(stem, layer).map(|n| n.src[0].as_str())
    }

    /// The one layer below `n_layer` that owns (`owner(S)`) the leaf `stems`
    /// view at `layer`, if exactly one does.
    fn owner_of_leaf(
        &self,
        stems: &[&str],
        layer: usize,
        n_layer: usize,
        owner: impl Fn(usize) -> bool,
    ) -> Option<usize> {
        let leaf_at = |l: usize| stems.iter().find_map(|s| self.leaf(s, l));
        let leaf = leaf_at(layer)?;
        let owners: Vec<usize> = (0..n_layer)
            .filter(|&s| owner(s) && leaf_at(s) == Some(leaf))
            .collect();
        match owners[..] {
            [s] => Some(s),
            _ => None,
        }
    }

    /// The layers whose `lid_top_k` layer `layer`'s nodes read.
    fn topk_read(&self, layer: usize) -> BTreeSet<usize> {
        self.nodes
            .iter()
            .filter(|n| n.layer == layer)
            .flat_map(|n| n.src.iter())
            .filter_map(|s| s.strip_prefix("lid_top_k-")?.parse().ok())
            .collect()
    }

    /// The top-k this set's ik run used, if its flags override the file's.
    fn top_k_override(&self, split: &Split) -> Option<usize> {
        let flags = self.flags.as_deref()?;
        let mut words = flags.split_whitespace();
        let key = split.arch_key("attention.indexer.top_k");
        while let Some(w) = words.next() {
            if w != "--override-kv" {
                continue;
            }
            let (k, v) = words.next()?.split_once('=')?;
            if k == key {
                let n = v.strip_prefix("int:")?;
                return Some(
                    n.parse()
                        .unwrap_or_else(|e| panic!("{}: {v}: {e}", self.set)),
                );
            }
        }
        None
    }
}

/// The columns both sets show.
fn common_columns(g: &Manifest, hp: &Hparams, l: usize, k: &LayerKind) -> Vec<(String, bool)> {
    let segment = |ratio: u32| {
        if ratio == hp.csa_ratio { "csa" } else { "hca" }
    };
    let ours_segment = k.stream.map(|s| segment(s.ratio));
    let graph_segment = if g.has("csa_k_all", l) {
        Some("csa")
    } else if g.has("hca_k_all", l) {
        Some("hca")
    } else {
        None
    };
    let owns = g.has("csa_state_compress", l) || g.has("hca_state_compress", l);
    let gate = format!("blk.{l}.attn_compressor_gate.weight");
    vec![
        (
            format!(
                "window only: ours {}, ik {}",
                k.stream.is_none(),
                g.has("attn_raw", l)
            ),
            k.stream.is_none() == g.has("attn_raw", l),
        ),
        (
            format!("segment: ours {ours_segment:?}, ik {graph_segment:?}"),
            ours_segment == graph_segment,
        ),
        (
            format!(
                "owns a compressor: ours {}, ik {owns}",
                k.compressor.is_some()
            ),
            k.compressor.is_some() == owns,
        ),
        (
            format!(
                "gated compressor: ours {}, ik {}",
                k.compressor.is_some_and(|c| c.gated),
                g.reads(&gate)
            ),
            k.compressor.is_some_and(|c| c.gated) == g.reads(&gate),
        ),
        (
            format!(
                "owns index keys: ours {}, ik {}",
                k.index_keys,
                g.has("lid_k_write", l)
            ),
            k.index_keys == g.has("lid_k_write", l),
        ),
        (
            format!(
                "engram site: ours {}, ik {}",
                k.engram.is_some(),
                g.has("engram_out", l)
            ),
            k.engram.is_some() == g.has("engram_out", l),
        ),
    ]
}

#[test]
#[ignore = "hw: needs the V4.1 shards and the V4.1 oracle sets on the box"]
fn hw_ds41_layer_kinds_match_ik_graph() {
    let (split, hp) = open();
    let shared = Manifest::open(SHARED_SET, &split);
    let step = Manifest::open(STEP_SET, &split);
    let n = hp.n_layer;
    let top_k = step.top_k_override(&split);
    let stepped = match top_k {
        Some(k) => hp.clone().with_indexer_top_k(k),
        None => hp.clone(),
    };
    let mut out = String::new();
    let mut bad = Vec::new();
    let _ = writeln!(
        out,
        "{STEP_SET} overrides top_k: {top_k:?}; our model for it keeps {}",
        stepped.indexer.top_k
    );
    let _ = writeln!(
        out,
        "layer  stream(ratio kv key topk)  compressor  keys  indexer  engram"
    );
    for (l, k) in hp.layers.iter().enumerate() {
        let stream = k.stream.map_or("-".to_string(), |s| {
            format!(
                "{} {} {} {}",
                s.ratio, s.kv_source, s.index_key_source, s.topk_source
            )
        });
        let compressor = k
            .compressor
            .map_or("-", |c| if c.gated { "gated" } else { "plain" });
        let _ = writeln!(
            out,
            "{l:>5}  {stream:<24}  {compressor:<10}  {:<4}  {:<7}  {:?}",
            k.index_keys, k.indexer, k.engram
        );
        let mut columns = Vec::new();
        for g in [&shared, &step] {
            for (what, ok) in common_columns(g, &hp, l, k) {
                columns.push((g.set, what, ok));
            }
        }
        let kv = shared.owner_of_leaf(&["csa_k", "hca_k"], l, n, |s| {
            shared.has("csa_state_compress", s) || shared.has("hca_state_compress", s)
        });
        let ours_kv = k.stream.map(|s| s.kv_source);
        columns.push((
            SHARED_SET,
            format!("kv_source: ours {ours_kv:?}, ik {kv:?}"),
            ours_kv == kv,
        ));
        let runs = step.has("lid_top_k", l);
        columns.push((
            STEP_SET,
            format!("runs the indexer: ours {}, ik {runs}", k.indexer),
            k.indexer == runs,
        ));
        let read: Vec<usize> = step.topk_read(l).into_iter().collect();
        let ours_topk: Vec<usize> = k.stream.map(|s| s.topk_source).into_iter().collect();
        columns.push((
            STEP_SET,
            format!("topk_source: ours {ours_topk:?}, ik {read:?}"),
            ours_topk == read,
        ));
        if k.indexer {
            let key = step.owner_of_leaf(&["lid_k"], l, n, |s| step.has("lid_k_write", s));
            let ours_key = k.stream.map(|s| s.index_key_source);
            columns.push((
                STEP_SET,
                format!("index_key_source: ours {ours_key:?}, ik {key:?}"),
                ours_key == key,
            ));
            let width = step.node("lid_top_k", l).map(|t| t.ne0 as usize);
            // ik keeps min(visible rows, top_k) (build_deepseek4.cpp:879); at position 301 every
            // stream shows more than 64 rows [derived: 151 at ratio 2, 302 at ratio 1].
            columns.push((
                STEP_SET,
                format!("top-k width: ours {}, ik {width:?}", stepped.indexer.top_k),
                width == Some(stepped.indexer.top_k),
            ));
        }
        for (set, what, ok) in columns {
            if !ok {
                bad.push(format!("layer {l}: {what} ({set})"));
            }
        }
    }
    let _ = writeln!(
        out,
        "compressors {} (gated {}), index keys {}, indexers {}, engram {}",
        ranges((0..n).filter(|&l| hp.layers[l].compressor.is_some())),
        ranges((0..n).filter(|&l| hp.layers[l].compressor.is_some_and(|c| c.gated))),
        ranges((0..n).filter(|&l| hp.layers[l].index_keys)),
        ranges((0..n).filter(|&l| hp.layers[l].indexer)),
        ranges((0..n).filter(|&l| hp.layers[l].engram.is_some())),
    );
    fail_if_bad(&out, &bad);
}

type Name = fn(usize) -> String;

/// The names every layer carries.
const EVERY_LAYER: &[Name] = &[
    names::attn_norm,
    names::attn_q_a,
    names::attn_q_a_norm,
    names::attn_q_b,
    names::attn_kv,
    names::attn_kv_a_norm,
    names::attn_sinks,
    names::attn_output_a,
    names::attn_output_b,
    names::hc_attn_fn,
    names::hc_attn_base,
    names::hc_attn_scale,
    names::hc_ffn_fn,
    names::hc_ffn_base,
    names::hc_ffn_scale,
    names::ffn_norm,
];

/// The names a routed layer carries.
const ROUTED: &[Name] = &[
    names::ffn_gate_inp,
    names::exp_probs_b,
    names::ffn_gate_exps,
    names::ffn_up_exps,
    names::ffn_down_exps,
    names::ffn_gate_shexp,
    names::ffn_up_shexp,
    names::ffn_down_shexp,
];

const COMPRESSOR: &[Name] = &[names::attn_compressor_kv, names::attn_compressor_norm];
const GATE: &[Name] = &[names::attn_compressor_gate];
const INDEX_KEYS: &[Name] = &[names::indexer_attn_k, names::indexer_k_norm];
const INDEXER: &[Name] = &[names::indexer_attn_q_b, names::indexer_proj];
const ENGRAM: &[Name] = &[
    names::engram_embd,
    names::engram_k,
    names::engram_q,
    names::engram_wkv,
];

#[test]
#[ignore = "hw: needs the V4.1 shards on the box"]
fn hw_ds41_names_in_file() {
    let (split, hp) = open();
    let mut bad = Vec::new();
    let mut named = BTreeSet::new();
    let model = [names::token_embd(), names::output_norm(), names::output()];
    for name in model {
        if split.find(&name).is_none() {
            bad.push(format!("{name} is not in the file"));
        }
        named.insert(name);
    }
    for (l, k) in hp.layers.iter().enumerate() {
        let groups: [(&str, &[Name], bool); 7] = [
            ("every layer", EVERY_LAYER, true),
            ("routed", ROUTED, k.routed),
            ("compressor", COMPRESSOR, k.compressor.is_some()),
            ("gated", GATE, k.compressor.is_some_and(|c| c.gated)),
            ("index keys", INDEX_KEYS, k.index_keys),
            ("indexer", INDEXER, k.indexer),
            ("engram", ENGRAM, k.engram.is_some()),
        ];
        for (group, list, carries) in groups {
            for name in list.iter().map(|f| f(l)) {
                if split.find(&name).is_some() != carries {
                    let is = if carries { "is not" } else { "is" };
                    bad.push(format!(
                        "layer {l} ({group} {carries}): {name} {is} in the file"
                    ));
                }
                named.insert(name);
            }
        }
    }
    let unnamed: Vec<String> = split
        .iter_tensors()
        .map(|(_, t)| t.name.clone())
        .filter(|t| !named.contains(t) && !UNREAD.iter().any(|u| t.ends_with(&format!(".{u}"))))
        .collect();
    if !unnamed.is_empty() {
        bad.push(format!(
            "{} tensors of the file are no name of `names` and not unread: {}",
            unnamed.len(),
            unnamed.join(", ")
        ));
    }
    println!(
        "{} tensors in the file; {} named by `names`, the rest {}",
        split.tensor_count(),
        named.iter().filter(|n| split.find(n).is_some()).count(),
        UNREAD.join(", ")
    );
    assert!(
        bad.is_empty(),
        "{} name(s) wrong:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

#[path = "common/spec_view.rs"]
mod spec_view;

#[path = "common/spec_fail_first.rs"]
mod spec_fail_first;

// PIN(2026-09-27): the served file's description as `arch::deepseek41::spec` reads it, line by line
// (model-wide fields, then each run of layers that read alike) — docs/research/modelspec-design.md
// §3 and §4 are the table these lines were checked against.
// PIN(2026-09-27): the `hc` and `engram` lines re-pinned when `HcSpec` gained its kind (mHC or
// gated-residual) and `EngramSpec` its rule (engram or PLE); the values are the same.
const V41_VIEW: &[&str] = &[
    "arch Deepseek41",
    "hidden 5120 vocab 129280 ctx_train 1048576",
    "rms_eps bits 0x1e3ce508",
    "layers 40 mtp 0",
    "hc Some(HcSpec { streams: 4, kind: Mhc { sinkhorn: 20, eps: 1e-6, mix: Lagged, collapse: LastMix } })",
    "engram Some(EngramSpec { heads: 8, max_ngram: 4, key_length: 256, rule: Engram })",
    "chat pre joyai-llm template bytes 6945 tools Some(Dsml) reasoning Some(ThinkSpan)",
    "[0] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 10000 yarn - qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[1] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 10000 yarn - qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc [Engram]",
    "[2,8] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r2 Own{g1 a0 o0} | sel 32x128 k512 keys Own list Own cand None || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[3-7] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r2 From2 | sel 32x128 k512 keys From2 list From2 cand None || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[9-13] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r2 From8 | sel 32x128 k512 keys From8 list From8 cand None || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[14] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r2 Own{g1 a0 o0} | sel 32x128 k512 keys Own list Own cand None || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc [Engram]",
    "[15-19] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r2 From14 | sel 32x128 k512 keys From14 list From14 cand None || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[20] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 Own{g0 a0 o0} | sel 32x128 k512 keys Own list Own cand None || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[21-23] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 From20 | sel 32x128 k512 keys From20 list From20 cand Some(Candidates { source: 20, blocks: 2048, block: 8 }) || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[24,28,32,36] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 From20 | sel 32x128 k512 keys From20 list Own cand Some(Candidates { source: 20, blocks: 2048, block: 8 }) || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[25-27] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 From20 | sel 32x128 k512 keys From20 list From24 cand Some(Candidates { source: 20, blocks: 2048, block: 8 }) || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[29-31] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 From20 | sel 32x128 k512 keys From20 list From28 cand Some(Candidates { source: 20, blocks: 2048, block: 8 }) || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[33-35] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 From20 | sel 32x128 k512 keys From20 list From32 cand Some(Candidates { source: 20, blocks: 2048, block: 8 }) || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
    "[37-39] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 160000 yarn 16/65536/32/1 qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true | cmp r1 From20 | sel 32x128 k512 keys From20 list From36 cand Some(Candidates { source: 20, blocks: 2048, block: 8 }) || moe 384/6 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
];

// PIN(2026-09-27): the DSpark draft's description (`spec::draft_of`), same form; §3's DraftSpec row.
// PIN(2026-09-27): the `hc` line re-pinned when `HcSpec` gained its kind; the values are the same.
const DSPARK_VIEW: &[&str] = &[
    "hidden 5120 vocab 129280 rms_eps bits 0x1e3ce508",
    "hc HcSpec { streams: 4, kind: Mhc { sinkhorn: 20, eps: 1e-6, mix: Lagged, collapse: LastMix } }",
    "width 5 target_layers [37, 38, 39] mask_token 128799 markov_rank 256",
    "[0-2] latent h 64 q 1280 kv 512 KeqV rope NormTail 64 base 10000 yarn - qhn false out Grouped { groups: 8, rank: 1024 } win Some(128) sinks true || moe 128/3 ff 2304 swiglu Some(10.0) SqrtSoftplus bias true norm true x1.5 hash false shared 2304 swiglu Some(10.0) gate false || hc",
];

/// 4. `hw_ds41_spec` — the typed description of the served file and of the
///    DSpark draft: (a) every field pinned; (b) the three facts the prompt
///    call's triangle reads (owns rows, owns keys, sources) equal the layer
///    table's on every layer, and so does whether the layer's stores make it
///    run every position (`runtime::state::every_position` against the layer
///    table's compressor or index keys); (c) the coverage check lists nothing, and with
///    the router's row taken out of its table it lists that router on every
///    layer.
#[test]
#[ignore = "hw: needs the V4.1 shards and the DSpark draft on the box"]
fn hw_ds41_spec() {
    use model::arch::deepseek41::{roles, spec};
    use model::arch::dspark::DraftHparams;
    use model::arch::models::DraftSpec;
    let (split, hp) = open();
    let mut o = String::new();
    let mut b = Vec::new();
    let read = spec::read(&split).unwrap_or_else(|e| panic!("spec: {e}"));
    let s = &read.spec;
    spec_view::compare(
        &mut o,
        &mut b,
        "V4.1 description",
        &spec_view::view(s),
        V41_VIEW,
    );
    let tensors = roles::classify(&split, &hp).unwrap_or_else(|e| panic!("classify: {e}"));
    let _ = writeln!(o, "the triangle's facts, layer by layer");
    for (l, (k, ls)) in hp.layers.iter().zip(&s.layers).enumerate() {
        let at = l as u32;
        let kind = (
            k.compressor.is_some(),
            k.index_keys,
            k.stream
                .map(|st| [st.kv_source, st.index_key_source, st.topk_source]),
        );
        let from_spec = (
            ls.owns_rows(),
            ls.owns_keys(),
            ls.sources(at).map(|x| x.map(|v| v as usize)),
        );
        layer_row(
            &mut o,
            &mut b,
            &format!("layer {l} ced facts"),
            from_spec,
            kind,
            "LayerKind",
        );
        layer_row(
            &mut o,
            &mut b,
            &format!("layer {l} every position"),
            runtime::state::every_position(ls),
            k.compressor.is_some() || k.index_keys,
            "LayerKind compressor || index_keys",
        );
    }
    let items = spec_view::items(&model::arch::coverage::check(s, &tensors));
    row(
        &mut o,
        &mut b,
        "coverage items",
        items,
        Vec::new(),
        "none: the chain runs it",
    );
    spec_fail_first::fail_first(
        &mut o,
        &mut b,
        s,
        &tensors,
        "models/src/shape.rs ROUTERS, the Ds41 body",
        "router: sqrt-softplus, 384 experts, top 6, with a selection bias: 0-39",
    );
    let dpath = std::env::var("BLOOMERY_DSPARK_MODEL").unwrap_or_else(|_| {
        panic!("BLOOMERY_DSPARK_MODEL is unset: run through `just gate-ds41-meta`")
    });
    let dsplit = Split::open(&dpath).unwrap_or_else(|e| panic!("open {dpath}: {e}"));
    let dhp = DraftHparams::read(&dsplit).unwrap_or_else(|e| panic!("draft hparams: {e}"));
    let DraftSpec::Block(d) = spec::draft_of(&dhp).unwrap_or_else(|e| panic!("draft spec: {e}"))
    else {
        panic!("draft spec: not a block draft")
    };
    let mut dview = vec![
        format!(
            "hidden {} vocab {} rms_eps bits {:#010x}",
            d.hidden,
            d.vocab,
            d.rms_eps.to_bits()
        ),
        format!("hc {:?}", d.hc),
        format!(
            "width {} target_layers {:?} mask_token {} markov_rank {}",
            d.width, d.target_layers, d.mask_token, d.markov_rank
        ),
    ];
    dview.extend(spec_view::layer_lines(&d.layers));
    spec_view::compare(&mut o, &mut b, "DSpark description", &dview, DSPARK_VIEW);
    fail_if_bad(&o, &b);
}
