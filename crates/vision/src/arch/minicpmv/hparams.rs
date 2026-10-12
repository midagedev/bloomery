//! Every hyperparameter a `minicpmv4_7` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The network is one fixed one, so each count key must hold exactly its value; the slice size,
//! the slice cap and the merger layer are the file's own (mainline reads them from the same keys),
//! checked for what the plan and the tower need of them.

use gguf::{Gguf, Value};

use super::PROJECTOR_TYPE;
use crate::VisionError;
use crate::arch::header::{
    KEY_HAS_VISION, Meta, check_architecture, check_eps, check_mean_std, check_projector,
    check_true, count_multiple, counts, need, refusal,
};

const KEY_GELU: &str = "clip.use_gelu";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";
const KEY_IMAGE_SIZE: &str = "clip.vision.image_size";
const KEY_SCALE_FACTOR: &str = "clip.vision.projector.scale_factor";
const KEY_MAX_SLICES: &str = "clip.vision.max_slice_nums";
const KEY_MERGER_LAYERS: &str = "clip.vision.wa_layer_indexes";

/// The tile width of the encoder's GEMM: the projection width is a multiple of it.
pub const TILE_N: u64 = 64;

/// Pixels per merged token side: a patch (14) times the merge (4). Slice and overview sizes are
/// multiples of it whatever the merge, as in mtmd (`get_slice_align`: always `patch · 4`).
pub const ALIGN: usize = 56;

/// `(key, the value the module runs, where that value comes from)` for the counts.
const COUNTS: [(&str, u64, &str); 5] = [
    ("clip.vision.block_count", 27, "the mmproj header"),
    ("clip.vision.embedding_length", 1152, "the mmproj header"),
    ("clip.vision.attention.head_count", 16, "the mmproj header"),
    ("clip.vision.feed_forward_length", 4304, "the mmproj header"),
    ("clip.vision.patch_size", 14, "the mmproj header"),
];

/// The vision tower's LayerNorm epsilon, as the f32 the file stores.
const EPS: f32 = 1e-6;

/// The hyperparameters of one `minicpmv4_7` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Attention heads.
    pub n_head: usize,
    /// The MLP's hidden width.
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// `clip.vision.image_size`: the slice size in pixels, the side one tile is planned to. The
    /// file says 448 where Hugging Face's config says 980 (that is the position table's 70×70
    /// grid, `v.position_embd.weight`'s 4900 rows); mainline plans slices from the file's key
    /// (`mtmd-image.cpp`, `get_slice_instructions`).
    pub slice_size: usize,
    /// `clip.vision.projector.scale_factor`: patches per token side, 2×2 in the ViT merger and
    /// 2×2 in the final merger.
    pub merge: usize,
    /// `clip.vision.max_slice_nums`: the most slices an image is cut into.
    pub max_slices: usize,
    /// The block after which the ViT merger folds 2×2 patches (`clip.vision.wa_layer_indexes`,
    /// borrowed for it as in mainline's `insert_layer_id`).
    pub merger_after: usize,
    /// The final merger's output width: the text model's embedding width.
    pub out_dim: usize,
    /// LayerNorm epsilon.
    pub eps: f32,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this crate does not run.
    pub fn read(gguf: &Gguf) -> Result<Hparams, VisionError> {
        check_architecture(gguf)?;
        Hparams::from_meta(gguf)
    }

    /// The file pairs with a text model of embedding width `n_embd`; any other width is refused.
    pub fn check_text_width(&self, n_embd: usize) -> Result<(), VisionError> {
        if self.out_dim == n_embd {
            return Ok(());
        }
        Err(refusal(
            KEY_PROJECTION_DIM,
            format!(
                "is {}; the text model's embedding width is {n_embd}, so this encoder file is \
                 not its own",
                self.out_dim
            ),
        ))
    }

    /// The width of the merged rows both mergers fold to: `dim · 4`, the 2×2 window of the ViT
    /// merger and the 2×2 window of the final merger.
    #[must_use]
    pub const fn merged_dim(&self) -> usize {
        self.dim * 4
    }

    /// Pixels per output token side.
    #[must_use]
    pub const fn token_px(&self) -> usize {
        self.patch * self.merge
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        let why = "the ViT and its GELU MLP";
        for key in [KEY_HAS_VISION, KEY_GELU] {
            check_true(m, key, why)?;
        }
        let [n_layer, dim, n_head, ff, patch] = counts(m, PROJECTOR_TYPE, COUNTS)?;
        let slice_size = count_multiple(
            m,
            KEY_IMAGE_SIZE,
            ALIGN as u64,
            "the slice size is planned in merged tokens of 56 pixels, so it must be",
        )?;
        let merge = scale_factor(m)?;
        let max_slices = count_multiple(m, KEY_MAX_SLICES, 1, "the slice cap is")?;
        let merger_after = merger_layer(m, n_layer)?;
        let out_dim = count_multiple(
            m,
            KEY_PROJECTION_DIM,
            TILE_N,
            "the projector's output width is a GEMM tile count: it must be",
        )?;
        let eps = check_eps(m, EPS, "the vision tower's LayerNorm")?;
        check_mean_std(
            m,
            "the preprocessing normalizes (x - 0.5) / 0.5 per channel",
        )?;
        Ok(Hparams {
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            slice_size,
            merge,
            max_slices,
            merger_after,
            out_dim,
            eps,
        })
    }
}

/// `clip.vision.projector.scale_factor`: 4, the 16-to-1 downsample with the ViT merger. Mainline
/// also runs 2 (no ViT merger, `merger_required = n_merge == 4`); this crate has no tower for it.
fn scale_factor(m: &impl Meta) -> Result<usize, VisionError> {
    let v = need(m, KEY_SCALE_FACTOR)?;
    match v.as_unsigned() {
        Some(4) => Ok(4),
        Some(2) => Err(refusal(
            KEY_SCALE_FACTOR,
            "is 2, the 4x mode with no ViT merger, which mainline runs; this crate runs 4",
        )),
        _ => Err(refusal(
            KEY_SCALE_FACTOR,
            format!("is {v:?}; this crate runs 4"),
        )),
    }
}

/// `clip.vision.wa_layer_indexes` holds one block index, below the block count: the block after
/// which the ViT merger runs (mainline reads its first entry, `clip.cpp` `insert_layer_id`).
fn merger_layer(m: &impl Meta, n_layer: usize) -> Result<usize, VisionError> {
    let v = need(m, KEY_MERGER_LAYERS)?;
    let Value::Array(items) = v else {
        return Err(refusal(
            KEY_MERGER_LAYERS,
            format!("is {v:?}, not an array"),
        ));
    };
    let [item] = items.as_slice() else {
        return Err(refusal(
            KEY_MERGER_LAYERS,
            format!("has {} entries; the ViT merger runs once", items.len()),
        ));
    };
    match item.as_unsigned().map(usize::try_from) {
        Some(Ok(layer)) if layer < n_layer => Ok(layer),
        _ => Err(refusal(
            KEY_MERGER_LAYERS,
            format!("holds {item:?}; a block index below {n_layer} is needed"),
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Hparams;
    use crate::arch::header::testing::{Table, expect_refusals};
    use gguf::Value;

    /// The keys this module reads, with the values of the 35B-A3B mmproj header.
    pub(crate) fn minicpm() -> Table {
        let half = || Value::Array(vec![Value::F32(0.5); 3]);
        Table(vec![
            ("clip.projector_type", Value::String("minicpmv4_7".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_gelu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(27)),
            ("clip.vision.embedding_length", Value::U32(1152)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(4304)),
            ("clip.vision.patch_size", Value::U32(14)),
            ("clip.vision.projection_dim", Value::U32(2048)),
            ("clip.vision.image_size", Value::U32(448)),
            ("clip.vision.projector.scale_factor", Value::U32(4)),
            ("clip.vision.max_slice_nums", Value::U32(9)),
            (
                "clip.vision.wa_layer_indexes",
                Value::Array(vec![Value::I32(6)]),
            ),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
            ("clip.vision.image_mean", half()),
            ("clip.vision.image_std", half()),
        ])
    }

    /// The figures of the 35B-A3B file.
    pub(crate) fn hp() -> Hparams {
        Hparams::from_meta(&minicpm()).expect("the 35B-A3B header reads")
    }

    /// The header of the 35B-A3B mmproj reads to its figures: the slice size is the file's 448,
    /// not Hugging Face's 980.
    #[test]
    fn the_35b_header_reads_to_its_figures() {
        let hp = Hparams::from_meta(&minicpm()).expect("read");
        assert_eq!(
            (hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch),
            (27, 1152, 16, 4304, 14)
        );
        assert_eq!((hp.slice_size, hp.merge, hp.max_slices), (448, 4, 9));
        assert_eq!((hp.merger_after, hp.out_dim), (6, 2048));
        assert_eq!((hp.merged_dim(), hp.token_px()), (4608, 56));
        assert!(hp.check_text_width(2048).is_ok());
        assert!(hp.check_text_width(4096).is_err());
    }

    /// Every value this crate does not run, and every absent key, is refused by its key.
    #[test]
    fn a_value_not_run_or_an_absent_key_is_refused_by_name() {
        let t = minicpm;
        let u = Value::U32;
        expect_refusals(
            vec![
                (
                    t().without("clip.projector_type"),
                    "metadata clip.projector_type: is absent",
                ),
                (
                    t().with(
                        "clip.projector_type",
                        Value::String("qwen3vl_merger".into()),
                    ),
                    "metadata clip.projector_type: is \"qwen3vl_merger\"",
                ),
                (
                    t().with("clip.vision.block_count", u(24)),
                    "metadata clip.vision.block_count: is 24",
                ),
                (
                    t().with("clip.vision.patch_size", u(16)),
                    "metadata clip.vision.patch_size: is 16",
                ),
                (
                    t().with("clip.vision.image_size", u(980)),
                    "metadata clip.vision.image_size: is 980",
                ),
                (
                    t().without("clip.vision.image_size"),
                    "metadata clip.vision.image_size: is absent",
                ),
                (
                    t().with("clip.vision.projector.scale_factor", u(2)),
                    "metadata clip.vision.projector.scale_factor: is 2",
                ),
                (
                    t().with("clip.vision.projector.scale_factor", u(8)),
                    "metadata clip.vision.projector.scale_factor: is U32(8)",
                ),
                (
                    t().with("clip.vision.max_slice_nums", u(0)),
                    "metadata clip.vision.max_slice_nums: is 0",
                ),
                (
                    t().with("clip.vision.wa_layer_indexes", Value::Array(vec![])),
                    "metadata clip.vision.wa_layer_indexes: has 0 entries",
                ),
                (
                    t().with(
                        "clip.vision.wa_layer_indexes",
                        Value::Array(vec![Value::I32(27)]),
                    ),
                    "metadata clip.vision.wa_layer_indexes: holds",
                ),
                (
                    t().with("clip.vision.projection_dim", u(2050)),
                    "metadata clip.vision.projection_dim: is 2050",
                ),
                (
                    t().with(
                        "clip.vision.image_mean",
                        Value::Array(vec![Value::F32(0.48); 3]),
                    ),
                    "metadata clip.vision.image_mean:",
                ),
                (
                    t().with("clip.use_gelu", Value::Bool(false)),
                    "metadata clip.use_gelu:",
                ),
            ],
            Hparams::from_meta,
        );
    }
}
