//! Inkling 2's DSpark draft: its target's own `config.json` states it under `mtp_config`, beside the
//! target's `text_config`, and `mtp.safetensors` in the target's folder holds it, every tensor BF16.
//! [`read`] takes the two configs into the common shape ([`DsparkShape`]), the layers' numbers
//! ([`Backbone`]) and the target's facts the draft is held to ([`Target`]); [`tensors`] states every
//! tensor the draft reads, [`check`] holds a file's header to that statement and [`plan`] states what
//! a load builds beside the target's head. The reference is llama.cpp PR #32218
//! (`src/models/dspark.cpp`: `load_tensors_inkling2`, `graph_inkling2`, `build_inkling2_head`).
//!
//! The file in the common shape's terms:
//! - **Taps.** `target_layer_ids` count the target's layer applications in run order, a looped
//!   core's layers once a pass; the last id is the application count, the row past the output mix
//!   and norm ([`TapRow::Final`]). They stand as the file writes them: the reference taps visit
//!   `il`'s input row as `t_layer_inp[il]` and the normed final row as `t_layer_inp[n_layer]`.
//! - **The block.** Row 0 is the anchor: the target's embedding row of the token at the target's
//!   position, RMS-normed by the target's embedding norm. Rows 1.. are the file's `mask_embed` row,
//!   which the layers embed; no mask token is read, so `mask_id` is 0, as for Mistral Large 4's
//!   draft. A block row attends to every row of the block.
//! - **The context.** No `fc`: `fuse_weight` holds a row of one weight a tap for each draft layer
//!   and one more for the head ([`Context::Fuse`]). A mix is its row's softmax over the tapped rows,
//!   RMS-normed by its layer's `dflare_norm` (the head's: `markov_head.dflare_norm`). Each layer
//!   projects its mix to keys and values of its own (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`;
//!   [`Backbone::ctx_kv`]), roped at the position.
//! - **The layers.** Qwen3-shaped GQA with q and k norms and the whole head rotated at `rope_theta`;
//!   the layers `local_layer_ids` names read `sliding_window_size` positions back (the reference's
//!   mask reads the anchor's own position on top: its `n_swa` is the window plus one), the others
//!   the whole context; a SiLU-gated MLP.
//! - **The head.** `norm`, the target's output matrix over the block's rows, then the recurrent
//!   Markov head of `markov_rank` in the anchor form ([`RnnRow`]): `rnn_proj` over `[state |
//!   markov_w1[prev] | normed row | the head's mix]` to three `markov_rank` chunks plus its bias;
//!   the correction `hidden_out · tanh(e) + b` through the target's output matrix, which the load
//!   folds ([`runtime::dspark::Fold`]); the target's `final_logit_softcapping` on the sum. The ids
//!   are the target's unpadded vocabulary (`unpadded_vocab_size`); `markov_w1` holds a row for each
//!   padded id (`vocab_size`).

use models::{LoopSpec, Visit, visits};
use runtime::dspark::{
    BaseHead, Context, DraftVocab, DsparkShape, Fold, HeadKind, HeadMatrix, HeadPlace, RnnCap,
    RnnOut, RnnRow, RnnStart, TapRow,
};
use safetensors::{Dtype, Safetensors};
use serde_json::Value;

use super::{Backbone, Error, field, flag, ids, kind, size, uint};

/// The head `markov_head_type` names, the one this reader takes.
const HEAD_TYPE: &str = "rnn_hidden";

/// The recurrent row of the file's head (the module doc's head).
const ROW: RnnRow = RnnRow {
    start: RnnStart::Anchor,
    ctx: true,
    cap: RnnCap::Sum,
    out: RnnOut::Hidden,
};

/// The target's facts the draft is held to, from `text_config`.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    /// The hidden width: a tapped row's, and that of the draft's rows the target's head reads.
    pub hidden: usize,
    /// `vocab_size`: the output matrix's rows, and `markov_w1`'s.
    pub padded_vocab: usize,
    /// `final_logit_softcapping`: the cap on a drafted row's sum.
    pub softcap: f32,
    /// The trunk's layer applications in run order ([`models::visits`]).
    pub applications: Vec<Visit>,
}

/// The draft as its target's config states it.
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub shape: DsparkShape,
    /// The recurrent head's rank, `markov_rank`: the shape's head's.
    pub rank: usize,
    pub backbone: Backbone,
    pub target: Target,
    /// Each tap against the target's applications ([`DsparkShape::tap_rows`]).
    pub taps: Vec<TapRow>,
}

/// What a load builds of the draft beside the target's resident output matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The target's output matrix, read in place: the block's base projection and the fold's.
    pub base: BaseHead,
    /// `W2 = output · hidden_out` and `c = output · b`, over the unpadded ids.
    pub fold: Fold,
    /// `fuse_weight`'s rows and row length: a mix a draft layer and the head's, a weight a tap.
    pub fuse: [usize; 2],
    /// `rnn_proj`'s rows and row length.
    pub joint: [usize; 2],
    /// `markov_w1`'s rows: one a padded id.
    pub w1_rows: usize,
}

fn section<'a>(root: &'a Value, key: &'static str) -> Result<&'a Value, Error> {
    let v = field(root, key)?;
    if v.is_object() {
        Ok(v)
    } else {
        Err(Error::Type {
            key,
            got: kind(v),
            want: "an object",
        })
    }
}

fn number(obj: &Value, key: &'static str) -> Result<f64, Error> {
    let v = field(obj, key)?;
    v.as_f64().ok_or_else(|| Error::Type {
        key,
        got: kind(v),
        want: "a number",
    })
}

fn word(obj: &Value, key: &'static str) -> Result<u32, Error> {
    let n = uint(obj, key)?;
    u32::try_from(n).map_err(|_| Error::Type {
        key,
        got: n.to_string(),
        want: "a 32-bit count",
    })
}

/// The target's facts: its width, its two vocabularies, its cap and its applications, the loop of
/// `loop_config` expanded when the config states one.
fn target(text: &Value) -> Result<Target, Error> {
    let n_layer = size(text, "num_hidden_layers")?;
    let loop_ = match text.get("loop_config").filter(|v| !v.is_null()) {
        None => None,
        Some(lp) => {
            if flag(lp, "is_varloop", false)? {
                return Err(Error::Unsupported(
                    "`loop_config.is_varloop` is true; the taps count a fixed loop's applications"
                        .into(),
                ));
            }
            let l = LoopSpec {
                core_start: word(lp, "core_start")?,
                core_end: word(lp, "core_end")?,
                passes: word(lp, "num_passes")?,
                gate_rank: word(lp, "gate_rank")?,
            };
            if l.core_start >= l.core_end || l.core_end as usize > n_layer || l.passes == 0 {
                return Err(Error::Unsupported(format!(
                    "a loop over layers {}..{} of {n_layer}, {} passes",
                    l.core_start, l.core_end, l.passes
                )));
            }
            Some(l)
        }
    };
    let softcap = number(text, "final_logit_softcapping")? as f32;
    if !softcap.is_finite() || softcap <= 0.0 {
        return Err(Error::Unsupported(format!(
            "`final_logit_softcapping` is {softcap}; the head caps its sum by a finite positive cap"
        )));
    }
    Ok(Target {
        hidden: size(text, "hidden_size")?,
        padded_vocab: size(text, "vocab_size")?,
        softcap,
        applications: visits(n_layer, loop_),
    })
}

/// The draft `config_json` (the target's whole config) states, its shape checked and its taps
/// planned against the target's applications. A head type other than `rnn_hidden`, a projection
/// bias, a width other than the target's and a layer list past the layers are refused by name.
pub fn read(config_json: &str) -> Result<Draft, Error> {
    let root: Value = serde_json::from_str(config_json).map_err(|e| Error::Json(e.to_string()))?;
    let mtp = section(&root, "mtp_config")?;
    let text = section(&root, "text_config")?;
    let target = target(text)?;
    let head = field(mtp, "markov_head_type")?;
    if head.as_str() != Some(HEAD_TYPE) {
        return Err(Error::Unsupported(format!(
            "`markov_head_type` is {}; the reader takes {HEAD_TYPE:?}",
            kind(head)
        )));
    }
    for key in ["q_bias", "o_bias"] {
        if flag(mtp, key, false)? {
            return Err(Error::Unsupported(format!(
                "`{key}` is true; the draft's projections carry no bias"
            )));
        }
    }
    let hidden = size(mtp, "hidden_size")?;
    if hidden != target.hidden {
        return Err(Error::Unsupported(format!(
            "a draft {hidden} wide over a target {} wide: the target's head reads the draft's rows",
            target.hidden
        )));
    }
    let layers = size(mtp, "num_hidden_layers")?;
    let local = ids(mtp, "local_layer_ids")?;
    if let Some(l) = local.iter().find(|&&l| l >= layers) {
        return Err(Error::Unsupported(format!(
            "`local_layer_ids` names layer {l} of {layers}"
        )));
    }
    let (heads, kv_heads) = (
        size(mtp, "num_attention_heads")?,
        size(mtp, "num_key_value_heads")?,
    );
    if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(Error::Unsupported(format!(
            "{heads} query heads over {kv_heads} key/value heads: a whole group a key/value head"
        )));
    }
    let unpadded = size(text, "unpadded_vocab_size")?;
    if unpadded == 0 || unpadded > target.padded_vocab {
        return Err(Error::Unsupported(format!(
            "`unpadded_vocab_size` is {unpadded} of a padded {}",
            target.padded_vocab
        )));
    }
    let window = size(mtp, "sliding_window_size")?;
    let windowed: Vec<bool> = (0..layers).map(|l| local.contains(&l)).collect();
    let rank = size(mtp, "markov_rank")?;
    let shape = DsparkShape {
        block: size(mtp, "block_size")?,
        row0_drafts: false,
        taps: ids(mtp, "target_layer_ids")?,
        hidden,
        context: Context::Fuse { mixes: layers + 1 },
        vocab: DraftVocab::Full { n: unpadded },
        mask_id: 0,
        window: windowed.iter().all(|&w| w).then_some(window),
        head: HeadKind::MarkovRnn { rank, row: ROW },
    };
    shape.check()?;
    let taps = shape.tap_rows(&target.applications)?;
    let backbone = Backbone {
        layers,
        hidden,
        heads,
        kv_heads,
        head_dim: size(mtp, "head_dim")?,
        intermediate: size(mtp, "intermediate_size")?,
        eps: number(mtp, "rms_norm_eps")? as f32,
        rope_theta: number(mtp, "rope_theta")?,
        window: Some(window),
        windowed,
        block_causal: false,
        ctx_kv: true,
    };
    Ok(Draft {
        shape,
        rank,
        backbone,
        target,
        taps,
    })
}

/// Every tensor the draft reads, by the file's names, each with its shape rows first.
#[must_use]
pub fn tensors(d: &Draft) -> Vec<(String, Vec<usize>)> {
    let b = &d.backbone;
    let (h, hd, ff, r) = (b.hidden, b.head_dim, b.intermediate, d.rank);
    let (q, kv) = (b.heads * hd, b.kv_heads * hd);
    let model = |name: &str, shape: &[usize]| (name.to_string(), shape.to_vec());
    let mut out = vec![
        (
            "fuse_weight".to_string(),
            d.shape.fuse_dims().map_or_else(Vec::new, Vec::from),
        ),
        model("mask_embed", &[h]),
        model("norm.weight", &[h]),
        model("markov_head.dflare_norm.weight", &[h]),
        model("markov_head.markov_w1.weight", &[d.target.padded_vocab, r]),
        model("markov_head.rnn_proj.weight", &[3 * r, ROW.joint_in(r, h)]),
        model("markov_head.rnn_proj.bias", &[3 * r]),
        model("markov_head.hidden_out.weight", &[h, r]),
        model("markov_head.hidden_out.bias", &[h]),
    ];
    for l in 0..b.layers {
        let at = |name: &str, shape: &[usize]| (format!("layers.{l}.{name}"), shape.to_vec());
        out.extend([
            at("dflare_norm.weight", &[h]),
            at("input_layernorm.weight", &[h]),
            at("post_attention_layernorm.weight", &[h]),
            at("self_attn.q_proj.weight", &[q, h]),
            at("self_attn.k_proj.weight", &[kv, h]),
            at("self_attn.v_proj.weight", &[kv, h]),
            at("self_attn.o_proj.weight", &[h, q]),
            at("self_attn.q_norm.weight", &[hd]),
            at("self_attn.k_norm.weight", &[hd]),
            at("self_attn.k_proj_ctx.weight", &[kv, h]),
            at("self_attn.v_proj_ctx.weight", &[kv, h]),
            at("self_attn.k_norm_ctx.weight", &[hd]),
            at("mlp.gate_proj.weight", &[ff, h]),
            at("mlp.up_proj.weight", &[ff, h]),
            at("mlp.down_proj.weight", &[h, ff]),
        ]);
    }
    out
}

/// `file`'s header against [`tensors`]: each tensor BF16 at its shape, none missing and none the
/// draft does not read, the first that differs refused by name.
pub fn check(d: &Draft, file: &Safetensors) -> Result<(), Error> {
    let want = tensors(d);
    for (name, shape) in &want {
        let e = file.entry(name).ok_or_else(|| Error::Tensor {
            name: name.clone(),
            detail: "is missing".into(),
        })?;
        if e.dtype != Dtype::Bf16 || e.shape != *shape {
            return Err(Error::Tensor {
                name: name.clone(),
                detail: format!(
                    "is {} {:?}; the draft reads BF16 {shape:?}",
                    e.dtype.name(),
                    e.shape
                ),
            });
        }
    }
    match file.names().find(|n| !want.iter().any(|(w, _)| w == n)) {
        Some(extra) => Err(Error::Tensor {
            name: extra.to_string(),
            detail: "is not one the draft reads".into(),
        }),
        None => Ok(()),
    }
}

/// What a load builds beside the target's resident output matrix of type `matrix`, read in place.
pub fn plan(d: &Draft, matrix: HeadMatrix) -> Result<Plan, Error> {
    let base = BaseHead::plan(matrix, HeadPlace::InPlace)?;
    let fold = d
        .shape
        .fold()
        .ok_or_else(|| Error::Unsupported("a head with no fold".into()))?;
    let fuse = d
        .shape
        .fuse_dims()
        .ok_or_else(|| Error::Unsupported("a context with no fuse".into()))?;
    Ok(Plan {
        base,
        fold,
        fuse,
        joint: [3 * d.rank, ROW.joint_in(d.rank, d.backbone.hidden)],
        w1_rows: d.target.padded_vocab,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small looped target (4 layers, the core 1..3 run twice: 6 applications) and a draft of 2
    /// layers over 4 taps, the last the final row; the fuse is then 3 mixes of 4 weights.
    const CONFIG: &str = r#"{
        "text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 96,
            "unpadded_vocab_size": 90, "final_logit_softcapping": 30.0,
            "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 2, "gate_rank": 8,
                            "is_varloop": false}
        },
        "mtp_config": {
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "local_layer_ids": [0],
            "sliding_window_size": 32, "block_size": 4, "markov_rank": 32,
            "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 4, 5, 6],
            "q_bias": false, "o_bias": false
        }
    }"#;

    /// A file of `tensors` in BF16, zero-filled.
    fn file(tensors: &[(String, Vec<usize>)]) -> Safetensors {
        let data: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, s)| vec![0; 2 * s.iter().product::<usize>()])
            .collect();
        let rows: Vec<(&str, &str, &[usize], &[u8])> = tensors
            .iter()
            .zip(&data)
            .map(|((n, s), d)| (n.as_str(), "BF16", s.as_slice(), d.as_slice()))
            .collect();
        Safetensors::parse(safetensors::write(&rows)).expect("a written file parses")
    }

    /// The config reads as the anchor form over the target's loop: taps by application, the last
    /// the final row, the fused context, the per-layer windows and the context's own projections.
    #[test]
    fn the_draft_reads_as_the_anchor_form_over_the_loop() {
        let d = read(CONFIG).expect("the config reads");
        let at = |application, layer, pass| TapRow::Entering {
            application,
            visit: Visit { layer, pass },
        };
        assert_eq!(
            d.taps,
            vec![at(1, 1, 0), at(4, 2, 1), at(5, 3, 0), TapRow::Final]
        );
        let s = &d.shape;
        assert_eq!(
            (s.block, s.row0_drafts, s.width(8), s.rows(3)),
            (4, false, 3, 4)
        );
        assert_eq!(s.context, Context::Fuse { mixes: 3 });
        assert_eq!(s.vocab, DraftVocab::Full { n: 90 });
        assert_eq!(s.head, HeadKind::MarkovRnn { rank: 32, row: ROW });
        assert_eq!(s.window, None);
        let b = &d.backbone;
        assert_eq!(
            (b.window, b.windowed.as_slice(), b.ctx_kv, b.block_causal),
            (Some(32), [true, false].as_slice(), true, false)
        );
        assert_eq!(
            (b.rope_theta, d.target.padded_vocab, d.target.softcap),
            (1e6, 96, 30.0)
        );
        let past = CONFIG.replace("[1, 4, 5, 6]", "[1, 4, 5, 7]");
        assert!(matches!(
            read(&past),
            Err(Error::Shape(runtime::dspark::DsparkError::TapPast {
                tap: 7,
                applications: 6
            }))
        ));
    }

    /// The header is held to the statement: `fuse_weight` a row of one weight a tap for each mix,
    /// so its transpose is refused by name, as are a missing tensor and one the draft does not read.
    #[test]
    fn the_header_is_held_to_the_fuse_of_a_row_a_mix() {
        let d = read(CONFIG).expect("the config reads");
        let want = tensors(&d);
        assert_eq!(want.len(), 9 + 15 * 2);
        assert_eq!(want[0], ("fuse_weight".to_string(), vec![3, 4]));
        assert!(check(&d, &file(&want)).is_ok());
        let mut turned = want.clone();
        turned[0].1 = vec![4, 3];
        assert!(matches!(
            check(&d, &file(&turned)),
            Err(Error::Tensor { name, .. }) if name == "fuse_weight"
        ));
        let missing = &want[1..];
        assert!(matches!(
            check(&d, &file(missing)),
            Err(Error::Tensor { name, .. }) if name == "fuse_weight"
        ));
        let mut extra = want.clone();
        extra.push(("fc.weight".to_string(), vec![64, 256]));
        assert!(matches!(
            check(&d, &file(&extra)),
            Err(Error::Tensor { name, .. }) if name == "fc.weight"
        ));
    }

    /// The plan reads the target's Q8_0 or BF16 head in place and folds the correction over the
    /// unpadded ids; `markov_w1` keeps a row for each padded id.
    #[test]
    fn the_plan_reads_the_targets_head_in_place() {
        let d = read(CONFIG).expect("the config reads");
        for matrix in [HeadMatrix::Q8_0, HeadMatrix::Bf16] {
            let p = plan(&d, matrix).expect("the plan");
            assert_eq!(
                p,
                Plan {
                    base: BaseHead {
                        matrix,
                        place: HeadPlace::InPlace
                    },
                    fold: Fold { rows: 90, rank: 32 },
                    fuse: [3, 4],
                    joint: [96, 192],
                    w1_rows: 96,
                }
            );
        }
    }

    /// The Inkling 2 Mini's folder (`$BLOOMERY_INKLING2_HF`): its config reads, `mtp.safetensors`'s
    /// header holds every tensor the draft reads and nothing else, and the plan is the design's.
    #[test]
    #[ignore = "needs the Inkling 2 Mini folder's config.json and mtp.safetensors at $BLOOMERY_INKLING2_HF"]
    fn hw_the_inkling2_draft_header_plans() {
        let dir = std::path::PathBuf::from(
            std::env::var_os("BLOOMERY_INKLING2_HF")
                .expect("$BLOOMERY_INKLING2_HF names the folder"),
        );
        let text = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let d = read(&text).expect("the config reads");
        let file = Safetensors::open(&dir.join("mtp.safetensors")).expect("mtp.safetensors");
        check(&d, &file).expect("the header holds the draft");
        let at = |application, layer, pass| TapRow::Entering {
            application,
            visit: Visit { layer, pass },
        };
        assert_eq!(d.target.applications.len(), 48);
        assert_eq!(
            d.taps,
            vec![at(5, 5, 0), at(23, 11, 1), at(41, 17, 2), TapRow::Final]
        );
        assert_eq!((d.shape.width(8), d.shape.rows(7)), (7, 8));
        assert_eq!(d.backbone.windowed, vec![true, false, true]);
        assert_eq!((d.backbone.rope_theta, d.target.softcap), (1e6, 30.0));
        for matrix in [HeadMatrix::Q8_0, HeadMatrix::Bf16] {
            let p = plan(&d, matrix).expect("the plan");
            assert_eq!(
                (p.fuse, p.joint, p.w1_rows, p.base.own_bytes(201_024, 2304)),
                ([4, 4], [768, 5120], 201_024, 0)
            );
            assert_eq!(
                p.fold,
                Fold {
                    rows: 200_058,
                    rank: 256
                }
            );
            assert_eq!((p.fold.w2_bytes(), p.fold.c_bytes()), (54_415_776, 800_232));
        }
        let want = tensors(&d);
        assert_eq!(want.len(), 9 + 15 * 3);
        let bytes: usize = want
            .iter()
            .map(|(_, s)| 2 * s.iter().product::<usize>())
            .sum();
        println!(
            "inkling2 draft: {} tensors, {bytes} BF16 bytes, taps {:?}",
            want.len(),
            d.shape.taps
        );
    }
}
