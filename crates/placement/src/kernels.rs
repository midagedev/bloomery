//! The card kernels by weight type: for each operation, which entry of the
//! common card kernel crate (`crates/gpu`) runs a file tensor of a given
//! [`GgmlType`], so a type's card support is stated in one place. This crate
//! is pure and cannot link the card code: an entry is a variant whose doc
//! names the launcher and its file.
//!
//! A type has a row for an operation only if a `pub` launcher of `crates/gpu`
//! that any family can call runs it, in the operation's shape:
//! - [`gate_up`]: one launch for every slot, the activation rule (`Act`) a
//!   launch argument, over `Q8Act` columns and device-side expert ids;
//! - [`down`]: one launch for every slot, one activation column a slot, in
//!   the form [`DownAct`] names. Where two launchers run one type, the row
//!   names the one a model body launches;
//! - [`gemm`]: the prompt path's GEMM over a route table (`GemmWeight`,
//!   `Gemm32Weight`);
//! - [`dense`]: `crates/gpu/src/site.rs`'s dense gemv sites, named by the file
//!   tensor's own type;
//! - [`head`]: `crates/gpu/src/head.rs`'s lm_head;
//! - [`embed`]: the card's embedding row lookups.
//!
//! Every `match` here names every [`GgmlType`] variant, with no `_` arm, so a
//! new type is decided in each operation. Each function maps a type to its
//! row or `None`; a type with no row is not an error here (a routed stack of
//! it stays on the host tier), and a tensor a role puts on a card is refused
//! by the plan ([`crate::placement::PlacementError::NoCardFormat`]).
//!
//! Weights. The routed `_sel`, gate·up and file-layout gemm entries read a
//! stack as the file's blocks in the [`crate::placement::CardFormat::KQuant`]
//! word stream; `q5_0_gemv_sel` reads the packed
//! [`crate::placement::CardFormat::Q5_0`] layout, and the Q8_0 dense, head
//! and embedding entries read [`crate::placement::CardFormat::Q8_0Planes`].
//! Every type a row names has a card layout:
//! [`crate::placement::CardFormat::of`] gives one, or
//! [`crate::placement::CardFormat::holds`] names the type for `KQuant`.
//!
//! Not rows, because no launcher of that shape is common:
//! - Q3_K gate·up. `ds41_expert_gate_up` in `crates/gpu-deepseek41` is that
//!   family's own. `crates/gpu/src/moe_fused.rs` holds a Q3_K gate·up with the
//!   SiLU rule fixed and one token. `Gpu::enqueue_gemv_q3k_sel` in
//!   `crates/gpu/src/lib.rs` is a gemv over one shared column with no rule.
//! - Q4_K gate·up in `crates/gpu/src/arch/qwen3moe/experts.rs`
//!   (`qwen3moe_gate_up_swiglu_q4k`): the whole-card qwen3moe body's own, with
//!   the SiLU rule fixed. The common Q4_K gate·up is the row below.
//! - Q8_0 `_sel` over f32 activations
//!   (`Q8F32Kernels::enqueue_q8_0_gemv_sel_f32`, the MTP draft's experts): an
//!   activation form [`DownAct`] does not name.
//! - MXFP4 gate·up and down: `crates/gpu-deepseek41/src/experts_mxfp4.rs` is
//!   V4.1's own; `crates/gpu/src/mxfp4.rs` holds a device row core and no
//!   launcher.
//! - `crates/gpu/src/iq.rs`'s `IqFormat` plane row cores (IQ2_XS, IQ3_XXS,
//!   IQ4_XS, Q2_K) are run by gates only; IQ3_XXS and IQ4_XS reach a launch
//!   through `crates/gpu/src/iq_sel.rs`, the rows below.
//! - V4.1's BF16 and Q3_K embedding rows (`chain/glue.rs` in
//!   `crates/gpu-deepseek41`): the BF16 rows are that family's own; the
//!   Q3_K rows have the common row below.
//! - The grouped tile entries (`crates/gpu/src/kquant/tiles.rs`, and
//!   `grouped_tiles` and `q4k_gemv_tiles` in `crates/gpu/src/q4k_sel.rs`) run
//!   the Q4_K and Q5_K stacks the `_sel` rows already name and add no type.
//!
//! BF16 and a dense site. A BF16 file tensor loads on a card as f32 values
//! decoded at load ([`crate::placement::CardFormat::Bf16AsF32`],
//! `crates/gpu/src/weights.rs`) and is then an F32 weight, but
//! `site::file_site` names a site by the file's own type and has no BF16 arm,
//! so [`dense`] has no BF16 row. Qwen3.8's indexer projections are where a BF16
//! tensor runs as a dense projection: `program38.rs` and `wide38.rs` in
//! `crates/gpu/src/arch/qwen3moe` call the f32 gemv and tile on the decoded
//! weight directly, outside `site.rs`.

use gguf::GgmlType;

/// A common card gate·up `_sel` entry: slot `s` writes the activation rule
/// over the rows of expert `sel[s]` of the gate and up stacks, dotted with
/// column `s / slots_per_col` of a `Q8Act`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateUpEntry {
    /// `KquantKernels::enqueue_gate_up_q4k` (`kq_gate_up_act_q4k`, Walk A) in
    /// `crates/gpu/src/kquant/sel.rs`.
    Q4k,
    /// `KquantKernels::enqueue_gate_up_q5k` (`kq_gate_up_act_q5k`) in
    /// `crates/gpu/src/kquant/sel.rs`.
    Q5k,
    /// `KquantKernels::enqueue_gate_up_q8_0` (`kq_gate_up_act_q8_0`, the
    /// file's `block_q8_0` stream) in `crates/gpu/src/kquant/sel.rs`.
    Q8_0,
    /// `IqSelKernels::enqueue_gate_up` (`iq3_xxs_gate_up_sel`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq3xxs,
    /// `IqSelKernels::enqueue_gate_up` (`iq4_xs_gate_up_sel`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq4xs,
}

/// A common card down `_sel` entry: slot `s` writes the rows of expert
/// `sel[s]` dotted with activation column `s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownEntry {
    /// `Q4kSelKernels::enqueue_gemv_q4k_sel` (`q4k_gemv_sel`) in
    /// `crates/gpu/src/q4k_sel.rs`.
    Q4k,
    /// `KquantKernels::enqueue_gemv_q5k_sel` (`q5k_gemv_sel`, Walk A) in
    /// `crates/gpu/src/kquant/sel.rs`.
    Q5k,
    /// `Q6kSelKernels::enqueue_gemv_q6k_sel` (`q6k_gemv_sel`) in
    /// `crates/gpu/src/q6k_sel.rs`.
    Q6k,
    /// `Q5Kernels::enqueue_gemv_q5_0_sel` (`q5_0_gemv_sel`, the packed q5_0
    /// layout) in `crates/gpu/src/q5.rs`.
    Q5_0,
    /// `Q51SelKernels::enqueue_gemv_q5_1_sel` (`q5_1_gemv_sel`) in
    /// `crates/gpu/src/q5_1_sel.rs`.
    Q5_1,
    /// `Q80SelKernels::enqueue_gemv_q8_0_sel32` (`q8_0_gemv_sel32`, any K a
    /// multiple of 32) in `crates/gpu/src/q8_0_sel32.rs`. The K-quant family
    /// also has `KquantKernels::enqueue_gemv_q8_0_sel` (`q8_0_gemv_sel`, whole
    /// 256-value super-blocks, `Q8Act`) in `crates/gpu/src/kquant/sel.rs`,
    /// which no model body launches.
    Q8_0,
    /// `IqSelKernels::enqueue_down` (`iq4_nl_gemv_sel32`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq4nl,
}

/// The activation form a down `_sel` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownAct {
    /// `Q8Act` (`crates/gpu/src/tensor.rs`): q8_1 columns of 128-value
    /// blocks, quantized by `Q4kSelKernels::enqueue_quantize_sel`.
    Q8Act,
    /// `Q8Blocks32` (`crates/gpu/src/q5.rs`): q8_1 columns of 32-value
    /// blocks, quantized by `Q5Kernels::enqueue_quantize_q8_sel`.
    Q8Blocks32,
}

/// A prompt-path GEMM entry over a route table: the grouped K-quant GEMM
/// (`GemmWeight`, `GemmKernels::enqueue_gemm` in
/// `crates/gpu/src/gemm/grouped.rs`) or the 32-value-block GEMM
/// (`Gemm32Weight`, `Gemm32Kernels::enqueue_gemm32` in
/// `crates/gpu/src/gemm/gemm32.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmEntry {
    /// `GemmWeight::Q3K` (`gemm_q3k` in `crates/gpu/src/gemm/kernels.rs`).
    Q3k,
    /// `GemmWeight::Q4K` (`gemm_q4k`).
    Q4k,
    /// `GemmWeight::Q5K` (`gemm_q5k`).
    Q5k,
    /// `GemmWeight::Q6K` (`gemm_q6k`).
    Q6k,
    /// `GemmWeight::Iq3Xxs` (`gemm_iq3xxs`).
    Iq3xxs,
    /// `GemmWeight::Iq4Xs` (`gemm_iq4xs`).
    Iq4xs,
    /// `Gemm32Weight::Q8_0File` (`gemm_q8_0f`, the file's `block_q8_0` stream)
    /// for a stack read as `KQuant`, or `Gemm32Weight::Q8_0Plane` (`gemm_q8_0p`,
    /// the q8f32 planes) for a dense projection (`SiteEntry::Q8_0`); both in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Q8_0,
    /// `Gemm32Weight::Q5_1File` (`gemm_q5_1`) in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Q5_1,
    /// `Gemm32Weight::Iq4NlFile` (`gemm_iq4nl`) in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Iq4nl,
}

/// A dense gemv site's launch family (`SiteTy` in `crates/gpu/src/site.rs`):
/// `site::gemv` over the f32 rows, `site::kgemv` over the q8_1 rows, and
/// `site::gemm` for a wide unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteEntry {
    /// `Gpu::enqueue_gemv_q3k`; wide, the grouped GEMM.
    Q3k,
    /// `Gpu::enqueue_gemv_q4k`; wide, the grouped GEMM.
    Q4k,
    /// The K-quant down `_sel` over a stack of one expert
    /// (`KquantKernels::enqueue_gemv_q5k_sel`); wide, the grouped GEMM.
    Q5k,
    /// `Gpu::enqueue_gemv_q6k`; wide, the grouped GEMM.
    Q6k,
    /// `Q8F32Kernels::enqueue_q8_0_gemv` (`enqueue_q8_0_gemv_mcol` past one
    /// column) over the q8f32 planes and f32 rows; wide, `Gemm32Weight::Q8_0Plane`.
    Q8_0,
    /// `Gemm32Kernels::enqueue_f32_tile` over the f32 rows, narrow and wide.
    F32,
}

/// The lm_head launch family (`OutW` in `crates/gpu/src/head.rs`,
/// `Head::enqueue`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadEntry {
    /// The q8_1 quantizer, `Gpu::enqueue_gemv_q6k`, then `argmax_fault`.
    Q6k,
    /// The q8_1 quantizer, `Gpu::enqueue_gemv_q4k`, then `argmax_fault`.
    Q4k,
    /// `Q8F32Kernels::enqueue_q8_0_gemv` over the q8f32 planes and the normed
    /// f32 rows, then `argmax_finite_fault`.
    Q8_0,
}

/// A card embedding row lookup: the rows of the ids dequantized token-major,
/// with each row's position and live key count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedEntry {
    /// `ElemKernels::enqueue_embed_rows_kquant` (`embed_rows_q3k`, over
    /// `Q3kRows`) in `crates/gpu/src/elem.rs`.
    Q3k,
    /// `ElemKernels::enqueue_embed_rows_q4k` (`embed_rows_q4k`) in
    /// `crates/gpu/src/elem.rs`.
    Q4k,
    /// `ElemKernels::enqueue_embed_rows_kquant` (`embed_rows_q5k`, over
    /// `Q5kRows`) in `crates/gpu/src/elem.rs`.
    Q5k,
    /// `ElemKernels::enqueue_embed_rows_kquant` (`embed_rows_q6k`, over
    /// `Q6kRows`) in `crates/gpu/src/elem.rs`.
    Q6k,
    /// `Q38Kernels::enqueue_embed_rows` (`embed_rows_q8_0`, the q8f32 planes)
    /// in `crates/gpu/src/q38.rs`.
    Q8_0,
}

/// The common card gate·up `_sel` entry that runs a stack of type `ty`.
#[must_use]
pub const fn gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    match ty {
        GgmlType::Q4_K => Some(GateUpEntry::Q4k),
        GgmlType::Q5_K => Some(GateUpEntry::Q5k),
        GgmlType::Q8_0 => Some(GateUpEntry::Q8_0),
        GgmlType::IQ3_XXS => Some(GateUpEntry::Iq3xxs),
        GgmlType::IQ4_XS => Some(GateUpEntry::Iq4xs),
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q6_K
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ1_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => None,
    }
}

/// The common card down `_sel` entry that runs a stack of type `ty`, and the
/// activation form it reads.
#[must_use]
pub const fn down(ty: GgmlType) -> Option<(DownEntry, DownAct)> {
    match ty {
        GgmlType::Q4_K => Some((DownEntry::Q4k, DownAct::Q8Act)),
        GgmlType::Q5_K => Some((DownEntry::Q5k, DownAct::Q8Act)),
        GgmlType::Q6_K => Some((DownEntry::Q6k, DownAct::Q8Act)),
        GgmlType::Q5_0 => Some((DownEntry::Q5_0, DownAct::Q8Blocks32)),
        GgmlType::Q5_1 => Some((DownEntry::Q5_1, DownAct::Q8Blocks32)),
        GgmlType::Q8_0 => Some((DownEntry::Q8_0, DownAct::Q8Blocks32)),
        GgmlType::IQ4_NL => Some((DownEntry::Iq4nl, DownAct::Q8Blocks32)),
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => None,
    }
}

/// The prompt-path GEMM entry that runs a stack of type `ty`.
#[must_use]
pub const fn gemm(ty: GgmlType) -> Option<GemmEntry> {
    match ty {
        GgmlType::Q3_K => Some(GemmEntry::Q3k),
        GgmlType::Q4_K => Some(GemmEntry::Q4k),
        GgmlType::Q5_K => Some(GemmEntry::Q5k),
        GgmlType::Q6_K => Some(GemmEntry::Q6k),
        GgmlType::IQ3_XXS => Some(GemmEntry::Iq3xxs),
        GgmlType::IQ4_XS => Some(GemmEntry::Iq4xs),
        GgmlType::Q8_0 => Some(GemmEntry::Q8_0),
        GgmlType::Q5_1 => Some(GemmEntry::Q5_1),
        GgmlType::IQ4_NL => Some(GemmEntry::Iq4nl),
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Q5_0
        | GgmlType::Q2_K
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ1_S
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => None,
    }
}

/// The dense gemv site launch family for a file tensor of type `ty`.
#[must_use]
pub const fn dense(ty: GgmlType) -> Option<SiteEntry> {
    match ty {
        GgmlType::Q3_K => Some(SiteEntry::Q3k),
        GgmlType::Q4_K => Some(SiteEntry::Q4k),
        GgmlType::Q5_K => Some(SiteEntry::Q5k),
        GgmlType::Q6_K => Some(SiteEntry::Q6k),
        GgmlType::Q8_0 => Some(SiteEntry::Q8_0),
        GgmlType::F32 => Some(SiteEntry::F32),
        GgmlType::F16
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q2_K
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => None,
    }
}

/// The lm_head launch family for an `output.weight` of type `ty`.
#[must_use]
pub const fn head(ty: GgmlType) -> Option<HeadEntry> {
    match ty {
        GgmlType::Q6_K => Some(HeadEntry::Q6k),
        GgmlType::Q4_K => Some(HeadEntry::Q4k),
        GgmlType::Q8_0 => Some(HeadEntry::Q8_0),
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q5_K
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => None,
    }
}

/// The card embedding row lookup for a token table of type `ty`.
#[must_use]
pub const fn embed(ty: GgmlType) -> Option<EmbedEntry> {
    match ty {
        GgmlType::Q3_K => Some(EmbedEntry::Q3k),
        GgmlType::Q4_K => Some(EmbedEntry::Q4k),
        GgmlType::Q5_K => Some(EmbedEntry::Q5k),
        GgmlType::Q6_K => Some(EmbedEntry::Q6k),
        GgmlType::Q8_0 => Some(EmbedEntry::Q8_0),
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q2_K
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::CardFormat;

    /// Every `GgmlType` variant in declaration order, then one id the enum
    /// does not name.
    const ALL: [GgmlType; 29] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::BF16,
        GgmlType::MXFP4,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ1_S,
        GgmlType::IQ4_NL,
        GgmlType::IQ3_S,
        GgmlType::IQ2_S,
        GgmlType::IQ4_XS,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::IQ1_M,
        GgmlType::TQ1_0,
        GgmlType::TQ2_0,
        GgmlType::Unknown(99),
    ];

    /// A variant's place in [`ALL`]. Exhaustive: a type added to `GgmlType`
    /// does not compile here until it is placed.
    const fn place(ty: GgmlType) -> usize {
        match ty {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q5_0 => 2,
            GgmlType::Q5_1 => 3,
            GgmlType::Q8_0 => 4,
            GgmlType::Q2_K => 5,
            GgmlType::Q3_K => 6,
            GgmlType::Q4_K => 7,
            GgmlType::Q5_K => 8,
            GgmlType::Q6_K => 9,
            GgmlType::BF16 => 10,
            GgmlType::MXFP4 => 11,
            GgmlType::IQ2_XXS => 12,
            GgmlType::IQ2_XS => 13,
            GgmlType::IQ3_XXS => 14,
            GgmlType::IQ1_S => 15,
            GgmlType::IQ4_NL => 16,
            GgmlType::IQ3_S => 17,
            GgmlType::IQ2_S => 18,
            GgmlType::IQ4_XS => 19,
            GgmlType::I8 => 20,
            GgmlType::I16 => 21,
            GgmlType::I32 => 22,
            GgmlType::I64 => 23,
            GgmlType::F64 => 24,
            GgmlType::IQ1_M => 25,
            GgmlType::TQ1_0 => 26,
            GgmlType::TQ2_0 => 27,
            GgmlType::Unknown(_) => 28,
        }
    }

    /// The `(type, row)` pairs an operation admits over [`ALL`].
    fn rows<T>(op: fn(GgmlType) -> Option<T>) -> Vec<(GgmlType, T)> {
        ALL.iter()
            .filter_map(|&ty| op(ty).map(|row| (ty, row)))
            .collect()
    }

    /// Whether each operation has a row for `ty`, by the operation's name.
    fn has_row_in(ty: GgmlType) -> [(&'static str, bool); 6] {
        [
            ("gate_up", gate_up(ty).is_some()),
            ("down", down(ty).is_some()),
            ("gemm", gemm(ty).is_some()),
            ("dense", dense(ty).is_some()),
            ("head", head(ty).is_some()),
            ("embed", embed(ty).is_some()),
        ]
    }

    /// The pin tests list their pairs in [`ALL`]'s order, so the list has to
    /// be every variant, each once.
    #[test]
    fn the_type_list_is_every_variant() {
        for (i, ty) in ALL.iter().enumerate() {
            assert_eq!(place(*ty), i, "{ty:?} is not at its place in ALL");
        }
    }

    #[test]
    fn gate_up_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(gate_up),
            [
                (GgmlType::Q8_0, GateUpEntry::Q8_0),
                (GgmlType::Q4_K, GateUpEntry::Q4k),
                (GgmlType::Q5_K, GateUpEntry::Q5k),
                (GgmlType::IQ3_XXS, GateUpEntry::Iq3xxs),
                (GgmlType::IQ4_XS, GateUpEntry::Iq4xs),
            ]
        );
    }

    #[test]
    fn down_admits_exactly_the_pinned_pairs() {
        let entries: Vec<_> = ALL
            .iter()
            .filter_map(|&ty| down(ty).map(|(entry, _)| (ty, entry)))
            .collect();
        assert_eq!(
            entries,
            [
                (GgmlType::Q5_0, DownEntry::Q5_0),
                (GgmlType::Q5_1, DownEntry::Q5_1),
                (GgmlType::Q8_0, DownEntry::Q8_0),
                (GgmlType::Q4_K, DownEntry::Q4k),
                (GgmlType::Q5_K, DownEntry::Q5k),
                (GgmlType::Q6_K, DownEntry::Q6k),
                (GgmlType::IQ4_NL, DownEntry::Iq4nl),
            ]
        );
    }

    /// The K-quant downs read the 128-value `Q8Act`; every other down reads
    /// the 32-value `Q8Blocks32`.
    #[test]
    fn down_reads_the_activation_form_of_its_entry() {
        let forms: Vec<_> = ALL
            .iter()
            .filter_map(|&ty| down(ty).map(|(_, act)| (ty, act)))
            .collect();
        assert_eq!(
            forms,
            [
                (GgmlType::Q5_0, DownAct::Q8Blocks32),
                (GgmlType::Q5_1, DownAct::Q8Blocks32),
                (GgmlType::Q8_0, DownAct::Q8Blocks32),
                (GgmlType::Q4_K, DownAct::Q8Act),
                (GgmlType::Q5_K, DownAct::Q8Act),
                (GgmlType::Q6_K, DownAct::Q8Act),
                (GgmlType::IQ4_NL, DownAct::Q8Blocks32),
            ]
        );
    }

    #[test]
    fn gemm_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(gemm),
            [
                (GgmlType::Q5_1, GemmEntry::Q5_1),
                (GgmlType::Q8_0, GemmEntry::Q8_0),
                (GgmlType::Q3_K, GemmEntry::Q3k),
                (GgmlType::Q4_K, GemmEntry::Q4k),
                (GgmlType::Q5_K, GemmEntry::Q5k),
                (GgmlType::Q6_K, GemmEntry::Q6k),
                (GgmlType::IQ3_XXS, GemmEntry::Iq3xxs),
                (GgmlType::IQ4_NL, GemmEntry::Iq4nl),
                (GgmlType::IQ4_XS, GemmEntry::Iq4xs),
            ]
        );
    }

    #[test]
    fn dense_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(dense),
            [
                (GgmlType::F32, SiteEntry::F32),
                (GgmlType::Q8_0, SiteEntry::Q8_0),
                (GgmlType::Q3_K, SiteEntry::Q3k),
                (GgmlType::Q4_K, SiteEntry::Q4k),
                (GgmlType::Q5_K, SiteEntry::Q5k),
                (GgmlType::Q6_K, SiteEntry::Q6k),
            ]
        );
    }

    #[test]
    fn head_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(head),
            [
                (GgmlType::Q8_0, HeadEntry::Q8_0),
                (GgmlType::Q4_K, HeadEntry::Q4k),
                (GgmlType::Q6_K, HeadEntry::Q6k),
            ]
        );
    }

    #[test]
    fn embed_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(embed),
            [
                (GgmlType::Q8_0, EmbedEntry::Q8_0),
                (GgmlType::Q3_K, EmbedEntry::Q3k),
                (GgmlType::Q4_K, EmbedEntry::Q4k),
                (GgmlType::Q5_K, EmbedEntry::Q5k),
                (GgmlType::Q6_K, EmbedEntry::Q6k),
            ]
        );
    }

    /// A card entry reads a layout the card loader can build: every type a
    /// row names has a [`CardFormat`] of its own, or is one of the file-block
    /// types a program's routed stack keeps as `KQuant`.
    #[test]
    fn every_row_type_has_a_card_layout() {
        for ty in ALL {
            for (name, has_row) in has_row_in(ty) {
                assert!(
                    !has_row || CardFormat::of(ty).is_some() || CardFormat::KQuant.holds(ty),
                    "{name} has a row for {ty:?}, which no card layout holds"
                );
            }
        }
    }

    /// A type id the enum does not name has no row in any operation.
    #[test]
    fn an_unnamed_type_has_no_row() {
        for id in [0, 2, 3, 99, u32::MAX] {
            let ty = GgmlType::Unknown(id);
            assert_eq!(gate_up(ty), None, "gate_up({ty:?})");
            assert_eq!(down(ty), None, "down({ty:?})");
            assert_eq!(gemm(ty), None, "gemm({ty:?})");
            assert_eq!(dense(ty), None, "dense({ty:?})");
            assert_eq!(head(ty), None, "head({ty:?})");
            assert_eq!(embed(ty), None, "embed({ty:?})");
        }
    }
}
