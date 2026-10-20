//! An M3 file's context-sized bytes and the inputs a plan of it is made from. A layer's cache is
//! its K and V planes, a position's `2 · kv_heads · head_dim` values (f16, or q8_0's two-plane
//! layout), and a sparse layer also keeps an f16 index key of the indexer's width for every
//! position ([`KvLayout`]). [`PlanInputs::read`] reads the headers once and refuses a file with
//! a feature no program runs, every one listed.

use gguf::Split;
use models::ModelSpec;
use runtime::stores::{F16_BYTES, F32_BYTES, kv_plane_bytes};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::{KvBytes, ModelTensors, PlacementError, Unimplemented};

/// An M3 file's per-layer cache.
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    layers: usize,
    dense_lead: usize,
    kv_heads: usize,
    head_dim: usize,
    /// The index key's values: `attention.indexer.key_length`, stored f16 a position.
    index_dim: usize,
    /// The rotated values of the rope row the body copies to its card a position, in f32.
    rope_dims: usize,
    q8: bool,
}

impl KvLayout {
    /// The layout `hp` describes, with the K and V planes in f16.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        KvLayout {
            layers: hp.n_layer,
            dense_lead: hp.dense_lead,
            kv_heads: hp.n_head_kv,
            head_dim: hp.head_dim,
            index_dim: hp.msa.d,
            rope_dims: hp.rope.dims,
            q8: false,
        }
    }

    /// The same layout with the K and V planes in q8_0's two-plane layout; the index keys stay
    /// f16.
    #[must_use]
    pub const fn in_q8(mut self) -> KvLayout {
        self.q8 = true;
        self
    }

    /// The K and V planes' bytes a position on one layer.
    #[must_use]
    pub const fn plane_row_bytes(&self) -> u64 {
        kv_plane_bytes(self.kv_heads, self.head_dim, self.q8, 1)
    }

    /// The index keys' bytes a position on one sparse layer.
    #[must_use]
    pub const fn index_row_bytes(&self) -> u64 {
        self.index_dim as u64 * F16_BYTES
    }

    /// Layers that keep index keys: the sparse ones.
    #[must_use]
    pub const fn sparse_layers(&self) -> usize {
        self.layers - self.dense_lead
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        if layer >= self.layers {
            return 0;
        }
        let planes = kv_plane_bytes(self.kv_heads, self.head_dim, self.q8, ctx_max);
        if layer < self.dense_lead {
            planes
        } else {
            planes + ctx_max * self.index_row_bytes()
        }
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        self.rope_dims as u64 * F32_BYTES * ctx_max
    }
}

/// What a plan of an M3 file is made from, read from its headers.
#[derive(Clone, Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's cache bytes.
    pub kv: KvLayout,
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its description, the first that
    /// fails being the error; then a file with a feature the engine does not run
    /// ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the coverage refusal: the file as it is, for a gate that
    /// describes a file whatever the coverage list holds.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, None, None)?;
        let spec = super::spec::spec_of(&hp, chat)?;
        let kv = KvLayout::of(&hp);
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
        })
    }

    /// Every feature of the file the engine does not run, as [`coverage::check`] lists them.
    #[must_use]
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// Whether layer `l` keeps index keys.
    #[must_use]
    pub fn sparse(&self, l: usize) -> bool {
        self.hp.kind(l) == Kind::Sparse
    }
}

#[cfg(test)]
mod tests {
    use super::super::hparams::tests::{keys, real, tensors};
    use super::{KvBytes, KvLayout, PlanInputs};
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::PlacementError;

    const POSITIONS_32K: u64 = 32_768;
    const POSITIONS_128K: u64 = 131_072;

    /// The bytes of every layer at `ctx` positions.
    fn total(kv: &KvLayout, ctx: u64) -> u64 {
        (0..64).map(|l| kv.layer_bytes(l, ctx)).sum()
    }

    /// The released file's cache at its real widths: a position is 122,880 B of f16 K and V over
    /// 60 layers (65,280 B in q8_0) and 14,592 B of f16 index keys over the 57 sparse layers; at
    /// 32k and 128k positions the planes are 4.03 and 16.11 GB (2.14 and 8.56 GB in q8_0) and
    /// the index keys 0.48 and 1.91 GB. Mutant: the index keys on the dense layers, or in q8.
    #[test]
    fn the_cache_is_the_released_files_bytes() {
        let kv = KvLayout::of(&real());
        assert_eq!((kv.plane_row_bytes(), kv.index_row_bytes()), (2048, 256));
        assert_eq!(kv.sparse_layers(), 57);
        assert_eq!(total(&kv, 1), 60 * 2048 + 57 * 256);
        assert_eq!(60 * kv.plane_row_bytes(), 122_880);
        assert_eq!(57 * kv.index_row_bytes(), 14_592);
        assert_eq!(total(&kv, POSITIONS_32K), 4_026_531_840 + 478_150_656);
        assert_eq!(total(&kv, POSITIONS_128K), 16_106_127_360 + 1_912_602_624);
        let q8 = kv.in_q8();
        assert_eq!(60 * q8.plane_row_bytes(), 65_280);
        assert_eq!(total(&q8, POSITIONS_32K), 2_139_095_040 + 478_150_656);
        assert_eq!(total(&q8, POSITIONS_128K), 8_556_380_160 + 1_912_602_624);
    }

    /// A dense layer keeps planes alone, a sparse one planes and index keys, and a layer past the
    /// model nothing; the stage keeps one rope row of the rotated width in f32 a position.
    #[test]
    fn dense_layers_keep_no_index_keys() {
        let kv = KvLayout::of(&real());
        for l in 0..3 {
            assert_eq!(kv.layer_bytes(l, 10), 10 * 2048, "layer {l}");
        }
        for l in [3, 30, 59] {
            assert_eq!(kv.layer_bytes(l, 10), 10 * (2048 + 256), "layer {l}");
        }
        assert_eq!(kv.layer_bytes(60, 10), 0);
        assert_eq!(kv.stage_position_bytes(10), 64 * 4 * 10);
        assert_eq!(kv.in_q8().stage_position_bytes(10), 64 * 4 * 10);
    }

    /// A header reads into the plan's inputs; the file is refused, every missing feature listed,
    /// until a program runs it. A file the reader refuses is that refusal, not a coverage one.
    #[test]
    fn the_inputs_describe_a_file_and_the_coverage_refuses_it() {
        let global = [
            ("tokenizer.ggml.pre", V::Str("minimax-m2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped("mm3-place", "minimax-m3", &keys(), &global, &tensors());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let described = PlanInputs::describe(&split).expect("the header describes");
        let refused = PlanInputs::read(&split);
        let _ = std::fs::remove_file(&path);
        assert_eq!(described.model.layers, 5);
        assert!(described.sparse(2) && !described.sparse(1));
        assert_eq!(described.kv.sparse_layers(), 3);
        match refused {
            Err(PlacementError::Unimplemented(items)) => {
                assert!(
                    items.iter().any(|u| u.feature.contains("swigluoai")),
                    "{items:?}"
                );
            }
            other => panic!("the coverage refusal: {other:?}"),
        }
        let bad = header_shaped(
            "mm3-place-bad",
            "minimax-m3",
            &[("block_count", V::U32(5))],
            &global,
            &tensors(),
        );
        let split = gguf::Split::open(&bad).expect("the synthetic header opens");
        let err = PlanInputs::read(&split).expect_err("a header without keys");
        let _ = std::fs::remove_file(&bad);
        assert!(matches!(err, PlacementError::Metadata { .. }), "{err:?}");
    }
}
