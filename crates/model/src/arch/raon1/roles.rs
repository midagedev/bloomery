//! The role of every tensor in a Raon-1 file, by exact name: the model-level names, then the
//! per-layer stems by the groups a layer's kind, the file's dialect ([`Dialect`]) and its
//! per-layer input carry ([`stems_of`]), which the header reader's tensor check reads too. A
//! name the layer's groups do not hold — the other dialect's fused `attn_qkv` beside a split
//! `attn_q`, the other kind's stem, a block past the model — fails the file with every such name
//! listed, and every tensor of a next-token layer is [`Role::Unused`]: carried, never loaded.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::names;
use crate::arch::classify::{Counts, block_stem, classify_with, model_level_role};
use crate::arch::kda;
use crate::arch::stems::{Stem, join, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The two gates an RDA layer adds to the delta rule's common mixer: the output gate's
/// projection and the write gate's offset.
const GATES: &[Stem] = &[
    req("ssm_g.weight", Role::Attention),
    req("ssm_rda.weight", Role::Attention),
];

/// An RDA layer's mixer of the split dialect: the common [`kda`] head and tail around the two
/// gates.
const RDA: &[Stem] = &join::<16>(&[kda::HEAD, GATES, kda::TAIL]);

/// An RDA layer's mixer of the fused dialect: one `attn_qkv` projection of `q | k | v | g` rows,
/// one conv over the fused channels, the erase and write gates as the last rows of `ssm_f_a`,
/// and the gated output's norm and projection (raon1nano.cpp:113-121).
const RDA_FUSED: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req(names::ATTN_QKV, Role::Attention),
    req(names::SSM_CONV1D, Role::Attention),
    req("ssm_f_a.weight", Role::Attention),
    req("ssm_f_b.weight", Role::Attention),
    req("ssm_dt.bias", Role::Attention),
    req("ssm_norm.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// A latent layer's mixer, the common `delta_mla` owner's, both dialects'.
use crate::arch::delta_mla::MLA;

/// A dense layer's fused feed-forward block: the SwiGLU's gate and up rows in one `ffn_up`
/// (raon1nano.cpp:152-154).
const DENSE_FUSED: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_up.weight", Role::DenseFfn),
    req("ffn_down.weight", Role::DenseFfn),
];

/// A dense layer's split feed-forward block and every layer's hyper-connection tensors, the
/// common tables'.
use crate::arch::stems::{DENSE, HC};

/// The hyper-connection tensors of the fused dialect: the folded chart in place of the split
/// dialect's scale scalars (conversion/raon1_nano.py: the fn rows pre-multiplied by alpha, the
/// chart holding the Helmert mix).
const HC_CHART: &[Stem] = &[
    req("hc_attn_fn.weight", Role::HyperConnection),
    req("hc_attn_base.weight", Role::HyperConnection),
    req("hc_attn_chart.weight", Role::HyperConnection),
    req("hc_ffn_fn.weight", Role::HyperConnection),
    req("hc_ffn_base.weight", Role::HyperConnection),
    req("hc_ffn_chart.weight", Role::HyperConnection),
];

/// The per-layer input a trunk layer of a file that carries one holds: its gate, its projection
/// back to the residual width and that projection's gain (raon1nano.cpp:146-149).
const PLE: &[Stem] = &[
    req(names::INP_GATE, Role::EngramDense),
    req(names::PROJ, Role::EngramDense),
    req(names::POST_NORM, Role::EngramDense),
];

/// A routed layer's feed-forward block, the common noaux owner's.
use crate::arch::noaux::BLOCK as ROUTED;

/// A next-token layer's own stems: the two-norm input projection and the shared head's norm
/// (raon1nano.cpp:156-161).
const NEXTN: &[Stem] = &[
    req("nextn.eh_proj.weight", Role::Unused),
    req("nextn.enorm.weight", Role::Unused),
    req("nextn.hnorm.weight", Role::Unused),
    req("nextn.shared_head_norm.weight", Role::Unused),
];

/// Every kind of trunk layer a file may hold: a mixer, and a dense (`true`) or routed feed-forward
/// block.
pub(super) const KINDS: [(Kind, bool); 4] = [
    (Kind::Rda, true),
    (Kind::Rda, false),
    (Kind::Mla, true),
    (Kind::Mla, false),
];

/// The tensor dialect a file speaks, as its loader creates it: the released Mini's split stems
/// or Nano's fused ones. A file that mixes the two is refused by name ([`Dialect::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// `attn_q` beside `attn_k` and `attn_v`, three convs, the gates' own tensors, `hc_*_scale`.
    Split,
    /// `attn_qkv` rows `q | k | v | g`, one conv, the gates folded into `ssm_f_a`'s rows,
    /// `ffn_up` rows `gate | up`, `hc_*_chart`.
    Fused,
}

impl Dialect {
    /// The dialect the file's delta layers speak, by the projection they carry: a fused
    /// `attn_qkv` marks Nano's, a split `attn_q` Mini's. A file that carries both mixes the
    /// dialects and one that carries neither says nothing; both are refused naming the tensors.
    pub(super) fn of(split: &Split, kinds: &[Kind]) -> Result<Dialect, PlacementError> {
        let first = kinds
            .iter()
            .position(|k| matches!(k, Kind::Rda))
            .expect("the kinds read holds both kinds");
        let has = |stem: &str| split.find(&format!("blk.{first}.{stem}")).is_some();
        let (fused, split_stem) = (has(names::ATTN_QKV), has("attn_q.weight"));
        let name = |stem: &str| format!("blk.{first}.{stem}");
        match (fused, split_stem) {
            (true, true) => Err(PlacementError::Tensor {
                name: name(names::ATTN_QKV),
                detail: format!(
                    "is in the file, and so is {}: a raon1 file speaks one dialect, split stems \
                     or fused ones, never both",
                    name("attn_q.weight")
                ),
            }),
            (true, false) => Ok(Dialect::Fused),
            (false, true) => Ok(Dialect::Split),
            (false, false) => Err(PlacementError::Tensor {
                name: name(names::ATTN_QKV),
                detail: format!(
                    "is not in the file, and neither is {}: a delta layer carries one of them",
                    name("attn_q.weight")
                ),
            }),
        }
    }

    /// A delta layer's mixer stems of this dialect.
    fn rda(self) -> &'static [Stem] {
        match self {
            Dialect::Split => RDA,
            Dialect::Fused => RDA_FUSED,
        }
    }

    /// A dense layer's feed-forward stems of this dialect.
    fn dense(self) -> &'static [Stem] {
        match self {
            Dialect::Split => DENSE,
            Dialect::Fused => DENSE_FUSED,
        }
    }

    /// The hyper-connection stems of this dialect.
    fn hc(self) -> &'static [Stem] {
        match self {
            Dialect::Split => HC,
            Dialect::Fused => HC_CHART,
        }
    }
}

/// The stem groups a trunk layer of `mixer` and `dense` carries in `dialect`.
pub(super) fn stems_of(dialect: Dialect, (mixer, dense): (Kind, bool)) -> Vec<&'static [Stem]> {
    let mixer = match mixer {
        Kind::Rda => dialect.rda(),
        Kind::Mla => MLA,
    };
    vec![
        mixer,
        if dense { dialect.dense() } else { ROUTED },
        dialect.hc(),
    ]
}

/// The stem groups trunk layer `l` of `hp`'s file carries: its kind's mixer, its feed-forward
/// block, its hyper-connection mix, and the per-layer input when the file carries one.
pub(super) fn trunk_groups(hp: &Hparams, l: usize) -> Vec<&'static [Stem]> {
    let mut groups = stems_of(hp.dialect, (hp.kind(l), hp.dense(l)));
    if hp.ple.is_some() {
        groups.push(PLE);
    }
    groups
}

/// The stem groups a next-token layer of `kind` carries: its kind's mixer, a dense block, the
/// mix, and its own stems — no per-layer input (raon1nano.cpp:91, :146).
fn nextn_groups(hp: &Hparams, kind: Kind) -> Vec<&'static [Stem]> {
    let mixer = match kind {
        Kind::Rda => hp.dialect.rda(),
        Kind::Mla => MLA,
    };
    vec![mixer, hp.dialect.dense(), hp.dialect.hc(), NEXTN]
}

/// The stem groups block `l` carries: its own when it is a trunk layer, a next-token layer's
/// when it is not.
pub(super) fn block_groups(hp: &Hparams, l: usize) -> Vec<&'static [Stem]> {
    if l < hp.n_trunk {
        trunk_groups(hp, l)
    } else {
        nextn_groups(hp, hp.kind(l))
    }
}

/// Every stem a block may carry, both dialects' with the per-layer input's and the next-token
/// block's, for the walk that refuses a stem on the block that does not own it — so a stem of
/// the dialect the file does not speak is refused wherever it sits.
pub(super) fn every_stem() -> Vec<&'static str> {
    let mut all: Vec<&'static str> = Vec::new();
    let mut push = |groups: &[&[Stem]]| {
        for stem in groups.iter().flat_map(|g| g.iter()) {
            if !all.contains(&stem.name) {
                all.push(stem.name);
            }
        }
    };
    for dialect in [Dialect::Split, Dialect::Fused] {
        for kind in KINDS {
            push(&stems_of(dialect, kind));
        }
    }
    push(&[PLE]);
    push(&[NEXTN]);
    all
}

/// The role and layer of `name` in `hp`'s file: a model-level name's, a per-layer-input name's
/// when the file carries one, else its block's groups'; a next-token block's every stem is
/// [`Role::Unused`]. `None` for any other name.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    if let Some(model_level) = model_level_role(name) {
        return Some(model_level);
    }
    if hp.ple.is_some() {
        match name {
            names::PER_LAYER_TOKEN_EMBD => return Some((Role::EngramTable, None)),
            names::PER_LAYER_MODEL_PROJ | names::PER_LAYER_PROJ_NORM => {
                return Some((Role::EngramDense, None));
            }
            _ => {}
        }
    }
    let (layer, stem) = block_stem(name, hp.n_layer)?;
    let trunk = layer < hp.n_trunk;
    block_groups(hp, layer)
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| (if trunk { s.role } else { Role::Unused }, Some(layer)))
}

/// Every tensor of `split` with its role and header facts. The per-layer input's table gathers
/// one row a token, like the embedding; its gain and projections are dense weights every token
/// reads in full.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let experts = hp.experts.as_ref();
    let mut model = classify_with(
        split,
        |name| role(name, hp),
        Counts::new(
            hp.n_trunk,
            experts.map_or(0, |e| e.n_expert),
            experts.map_or(0, |e| e.n_used),
        ),
    )?;
    if hp.ple.is_some() {
        for t in &mut model.tensors {
            if t.name == names::PER_LAYER_TOKEN_EMBD {
                t.gathered_rows = Some(1);
            }
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::super::hparams::tests::{keys_nano, read_nano, read_nano_mtp};
    use super::super::names;
    use super::{Dialect, KINDS, MLA, RDA, every_stem, stems_of};
    use crate::arch::classify::{assert_one_role_per_stem, role_by_layer};
    use crate::arch::raon1::hparams::Kind;
    use crate::arch::synthetic::header_shaped;
    use crate::placement::Role;

    /// A tensor's role and layer by its name in a 25-layer file whose layers 0 and 1 are dense RDA
    /// layers, 2 to 23 routed RDA layers and 24 a routed latent one.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 25, |l| {
            stems_of(
                Dialect::Split,
                (if l == 24 { Kind::Mla } else { Kind::Rda }, l < 2),
            )
        })
    }

    /// The same, in a nano file: every layer a dense block, a per-layer input on the trunk.
    /// A stem in two groups names one role (the two mixers both hold `attn_norm` and
    /// `attn_output`, the two feed-forward blocks `ffn_norm`), and no group lists a stem twice.
    /// Mutant: `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        for dialect in [Dialect::Split, Dialect::Fused] {
            let groups: Vec<_> = KINDS
                .into_iter()
                .flat_map(|k| stems_of(dialect, k))
                .collect();
            assert_one_role_per_stem(&groups);
        }
    }

    /// The split table is exact: a stem of each group has its role on a layer of its kind, and a
    /// name the layer's kind does not hold — the other kind's stem, a fused projection, a layer
    /// past the model, a next-token tensor, a bias, a KDA low-rank gate, an indexer — has none,
    /// so `classify` lists it. Mutant: every layer given the stems of every kind.
    #[test]
    fn the_split_table_is_exact() {
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        assert_eq!(
            role("blk.24.attn_gate.weight"),
            Some((Role::Attention, Some(24)))
        );
        assert_eq!(
            role("blk.1.ssm_rda.weight"),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.2.hc_ffn_scale.weight"),
            Some((Role::HyperConnection, Some(2)))
        );
        assert_eq!(
            role("blk.24.exp_probs_b.bias"),
            Some((Role::Router, Some(24)))
        );
        assert_eq!(
            role("blk.1.ffn_gate.weight"),
            Some((Role::DenseFfn, Some(1)))
        );
        assert_eq!((RDA.len(), MLA.len()), (16, 10));
        for name in [
            "blk.24.ssm_rda.weight",
            "blk.3.attn_gate.weight",
            "blk.1.exp_probs_b.bias",
            "blk.2.ffn_gate.weight",
            "blk.25.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_g_a.weight",
            "blk.3.indexer.proj.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.0.inp_gate.weight",
            "blk.3.attn_q.bias",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// The fused table is exact on a whole nano file: every trunk tensor of the small header has
    /// its role on its layer, the per-layer input's names have theirs, a next-token layer's every
    /// stem is unused, and a split stem (`attn_q`, `hc_*_scale`, `ffn_gate`) has no role in it.
    /// Mutant: the fused dialect given the split stems.
    #[test]
    fn the_fused_table_is_exact_on_the_small_header() {
        let (hp, tensors) = read_nano("roles-fused");
        let want: Vec<(String, Option<Role>, Option<usize>)> = tensors
            .iter()
            .map(|(name, _)| {
                let (r, l) = super::role(name, &hp).unwrap_or((Role::Unread, None));
                (name.clone(), Some(r), l)
            })
            .collect();
        let of = |name: &str| {
            want.iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, r, l)| (*r, *l))
        };
        assert_eq!(
            of("token_embd.weight"),
            Some((Some(Role::TokenEmbedding), None))
        );
        assert_eq!(of("output_norm.weight"), Some((Some(Role::Head), None)));
        assert_eq!(
            of(names::PER_LAYER_TOKEN_EMBD),
            Some((Some(Role::EngramTable), None))
        );
        assert_eq!(
            of("blk.0.inp_gate.weight"),
            Some((Some(Role::EngramDense), Some(0)))
        );
        assert_eq!(
            of("blk.3.attn_qkv.weight"),
            Some((Some(Role::Attention), Some(3)))
        );
        assert_eq!(
            of("blk.3.hc_attn_chart.weight"),
            Some((Some(Role::HyperConnection), Some(3)))
        );
        for name in [
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.hc_attn_scale.weight",
            "blk.0.ssm_rda.weight",
        ] {
            assert_eq!(
                of(name),
                None,
                "{name}: a split stem has no role in a fused file"
            );
        }
    }

    /// A next-token layer's every stem is unused, and a per-layer input is none of its stems: the
    /// MTP file's small shape (a seventh block, a latent one, carrying the trunk's latent mixer,
    /// a dense block, the mix and its own nextn stems). Mutant: the nextn groups given PLE.
    #[test]
    fn a_next_token_block_carries_its_stems_unused() {
        let (hp, tensors) = read_nano_mtp("roles-nextn");
        let role_of = |name: &str| super::role(name, &hp);
        for name in [
            "blk.6.attn_kv_a_mqa.weight",
            "blk.6.ffn_up.weight",
            "blk.6.hc_attn_chart.weight",
            "blk.6.nextn.eh_proj.weight",
            "blk.6.nextn.shared_head_norm.weight",
        ] {
            assert!(
                tensors.iter().any(|(n, _)| n == name),
                "{name}: the MTP header carries it"
            );
            assert_eq!(role_of(name), Some((Role::Unused, Some(6))), "{name}");
        }
        for name in ["blk.6.inp_gate.weight", "blk.6.proj.weight"] {
            assert!(
                !tensors.iter().any(|(n, _)| n == name),
                "{name}: the MTP layer carries no per-layer input"
            );
            assert_eq!(role_of(name), None, "{name}");
        }
    }

    /// The walk that refuses a misplaced stem lists every stem a block may carry, both dialects'
    /// and the per-layer input's and the nextn block's.
    #[test]
    fn every_stem_lists_both_dialects() {
        let pool = every_stem();
        for stem in [
            "attn_q.weight",
            super::names::ATTN_QKV,
            "ffn_gate.weight",
            "hc_attn_scale.weight",
            "hc_attn_chart.weight",
            super::names::INP_GATE,
            "nextn.eh_proj.weight",
            "exp_probs_b.bias",
        ] {
            assert!(pool.contains(&stem), "{stem} not in the pool");
        }
    }

    /// A file that carries both dialects' markers is refused naming both tensors, and one that
    /// carries neither naming the absent pair. Mutant: the both arm returning a dialect, or the
    /// neither arm reading one stem only.
    #[test]
    fn the_dialect_is_one_of_two() {
        let (hp, mut tensors) = read_nano("dialect");
        tensors.push(("blk.0.attn_q.weight".to_string(), vec![64, 256]));
        let path = header_shaped(
            "raon1-roles-dialect-mixed",
            "raon1nano",
            &keys_nano(),
            &[],
            &tensors,
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        let err = Dialect::of(&split, &hp.kinds).expect_err("a file of both dialects");
        let text = err.to_string();
        assert!(text.contains("blk.0.attn_qkv.weight"), "{text}");
        assert!(
            text.contains("blk.0.attn_q.weight"),
            "the refusal names both: {text}"
        );
        assert_eq!(hp.dialect, Dialect::Fused);
    }
}
