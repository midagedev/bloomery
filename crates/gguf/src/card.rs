//! The card columns of a type's row ([`crate::QRow::card`], a [`Card`]): the layout a file
//! tensor of the type loads in on a card ([`CardFormat::of`]), the layout a routed stack of it
//! loads in under the default expert rule ([`CardFormat::of_routed`]), and the entry of the
//! common card kernel crate (`crates/gpu`) each card operation runs it with, `None` where it has
//! none. A row decides these columns with the rest of its facts, so a new type is decided in every
//! column at once; `placement::kernels` reads them by operation. This crate cannot link the card
//! code: an entry is a variant whose doc names the launcher and its file.
//!
//! A type has an entry for an operation only if a `pub` launcher of `crates/gpu`
//! that any family can call runs it, in the operation's shape:
//! - [`Card::gate_up`]: one launch for every slot, the activation rule (`Act`) a
//!   launch argument, over `Q8Act` columns and device-side expert ids;
//! - [`Card::down`]: one launch for every slot, one activation column a slot, in
//!   the form [`DownAct`] names. Where two launchers run one type, the entry
//!   is the one a model body launches;
//! - `placement::kernels::up`: the up of an expert with no gate, which is a
//!   [`Card::down`] entry of the 32-value form over columns that repeat the
//!   token's row;
//! - [`Card::gemm`]: the prompt path's GEMM over a route table (`GemmWeight`,
//!   `Gemm32Weight`);
//! - [`Card::dense`]: `crates/gpu/src/site.rs`'s dense gemv sites, named by the file
//!   tensor's own type; the activation each reads is [`SiteEntry::reads`];
//! - [`Card::head`]: `crates/gpu/src/head.rs`'s lm_head;
//! - [`Card::embed`]: the card's embedding row lookups.
//!
//! A type with no entry is not an error here (a routed stack of it stays on the host tier),
//! and a tensor a role puts on a card is refused by the plan
//! (`placement::PlacementError::NoCardFormat`). A type's host facts (the activation form a CPU
//! matmul quantizes to, and so whether a host kernel runs it) are its row's other columns.
//!
//! Weights. The routed `_sel`, gate·up and file-layout gemm entries read a
//! stack as the file's blocks in the [`CardFormat::KQuant`] word stream;
//! `q5_0_gemv_sel`, the Q5_0 gemm entry and the Q5_0 dense site
//! ([`SiteEntry::Q5_0`]) read the packed [`CardFormat::Q5_0`] layout, the
//! Q8_0 dense, head and embedding
//! entries read [`CardFormat::Q8_0Planes`], and the NVFP4 entries read
//! [`CardFormat::Nvfp4Planes`] (a routed stack's with one tensor scale an
//! expert). A routed entry's layout is its `layout`, which
//! [`CardFormat::holds`] reads, so every type an entry runs has a card layout.
//!
//! Not entries, because no launcher of that shape is common:
//! - Q3_K gate·up. `ds41_expert_gate_up` in `crates/gpu-deepseek41` is that
//!   family's own. `crates/gpu/src/moe_fused.rs` holds a Q3_K gate·up with the
//!   SiLU rule fixed and one token. `Gpu::enqueue_gemv_q3k_sel` in
//!   `crates/gpu/src/lib.rs` is a gemv over one shared column with no rule.
//! - Q4_K gate·up in `crates/gpu/src/arch/qwen3moe/experts.rs`
//!   (`qwen3moe_gate_up_swiglu_q4k`): the whole-card qwen3moe body's own, with
//!   the SiLU rule fixed. The common Q4_K gate·up is its entry.
//! - Q8_0 and FP8_B128 `_sel` over f32 activations
//!   (`Q8F32Kernels::enqueue_q8_0_gemv_sel_f32` and `enqueue_e4m3_gemv_sel_f32`
//!   through `site::resident_sel_f32`, the MTP draft's experts): an
//!   activation form [`DownAct`] does not name.
//! - `dflash_expert_gate_up` and `dflash_expert_down` in
//!   `crates/gpu/src/mxfp4_sel.rs`: the V4.1 DSpark draft's route-plan entries
//!   over two weight planes, with a fused weighted combine. The MXFP4 entries
//!   are that file's per-slot pair over the file's own blocks.
//! - `crates/gpu/src/iq.rs`'s `IqFormat` plane row cores (IQ2_XS, IQ3_XXS,
//!   IQ4_XS, Q2_K) are run by gates only; IQ2_XS, IQ2_S, IQ1_S, IQ1_M, IQ3_XXS,
//!   IQ3_S and IQ4_XS reach a launch through `crates/gpu/src/iq_sel.rs`, their
//!   entries (IQ2_S, IQ1_S, IQ1_M and IQ3_S have no plane row core). Q2_K has
//!   no launcher of that shape.
//! - V4.1's BF16, Q3_K and Q5_K embedding rows and its IQ4_NL Engram rows
//!   (`chain/glue.rs` in `crates/gpu-deepseek41`): the BF16 and IQ4_NL rows
//!   are that family's own; the Q3_K and Q5_K rows have the common entry.
//! - The grouped tile entries (`crates/gpu/src/kquant/tiles.rs`, and
//!   `grouped_tiles` and `q4k_gemv_tiles` in `crates/gpu/src/q4k_sel.rs`) run
//!   the Q4_K and Q5_K stacks the `_sel` entries already name and add no type.
//!
//! BF16 matrices a program's dense site, head or embedding lookup reads load
//! in their words ([`CardFormat::Bf16Raw`]), selected per tensor by
//! `placement::ModelTensors::card_format`. These entries widen values to f32
//! as they read them. Other BF16 tensors decode to f32 at load
//! ([`CardFormat::Bf16AsF32`]).

use crate::qtype::{OWN_ROWS, QROWS, QRow};
use crate::quant::GgmlType;

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
    /// `Q6kSelKernels::enqueue_gate_up` (`q6k_gate_up_sel`, the file's 210-byte
    /// `block_q6_K` stream over the `Q8Act`'s q6 plane) in
    /// `crates/gpu/src/q6k_sel.rs`.
    Q6k,
    /// `QuantSelKernels::enqueue_gate_up` (`iq2_xxs_gate_up_sel`, the file's
    /// 66-byte `block_iq2_xxs` stream through the block's shared table) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq2xxs,
    /// `QuantSelKernels::enqueue_gate_up` (`iq2_xs_gate_up_sel`, the file's
    /// 74-byte `block_iq2_xs` stream) in `crates/gpu/src/iq_sel.rs`.
    Iq2xs,
    /// `QuantSelKernels::enqueue_gate_up` (`iq2_s_gate_up_sel`, the file's
    /// 82-byte `block_iq2_s` stream) in `crates/gpu/src/iq_sel.rs`.
    Iq2s,
    /// `QuantSelKernels::enqueue_gate_up` (`iq1_s_gate_up_sel`, the file's
    /// 50-byte `block_iq1_s` stream) in `crates/gpu/src/iq_sel.rs`.
    Iq1s,
    /// `QuantSelKernels::enqueue_gate_up` (`iq1_m_gate_up_sel`, the file's
    /// 56-byte `block_iq1_m` stream) in `crates/gpu/src/iq_sel.rs`.
    Iq1m,
    /// `QuantSelKernels::enqueue_gate_up` (`iq3_xxs_gate_up_sel`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq3xxs,
    /// `QuantSelKernels::enqueue_gate_up` (`iq3_s_gate_up_sel`, the file's
    /// 110-byte `block_iq3_s` stream) in `crates/gpu/src/iq_sel.rs`.
    Iq3s,
    /// `QuantSelKernels::enqueue_gate_up` (`iq4_xs_gate_up_sel`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq4xs,
    /// `Mxfp4SelKernels::enqueue_gate_up` (`mxfp4_gate_up_sel`, the file's
    /// 17-byte `block_mxfp4` stream) in `crates/gpu/src/mxfp4_sel.rs`.
    Mxfp4,
    /// `Nvfp4SelKernels::enqueue_gate_up` (`nvfp4_gate_up_sel`, the two
    /// planes and a tensor scale an expert) in `crates/gpu/src/mxfp4_sel.rs`.
    Nvfp4,
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
    /// `QuantSelKernels::enqueue_gemv_sel_cols` (`iq2_xxs_gemv_sel`, over
    /// `Q8Act` through the block's shared table, `slots_per_col` 1 for a
    /// down) in `crates/gpu/src/iq_sel.rs`.
    Iq2xxs,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq2_xs_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq2xs,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq2_s_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq2s,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq1_s_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq1s,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq1_m_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq1m,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq3_xxs_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq3xxs,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq3_s_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq3s,
    /// `QuantSelKernels::enqueue_gemv_sel` (`iq4_xs_gemv_sel`, over `Q8Act`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq4xs,
    /// `QuantSelKernels::enqueue_down` (`iq4_nl_gemv_sel32`) in
    /// `crates/gpu/src/iq_sel.rs`.
    Iq4nl,
    /// `Mxfp4SelKernels::enqueue_down` (`mxfp4_gemv_sel`, the file's 17-byte
    /// `block_mxfp4` stream) in `crates/gpu/src/mxfp4_sel.rs`.
    Mxfp4,
    /// `Nvfp4SelKernels::enqueue_sel32` (`nvfp4_gemv_sel32`, the two planes
    /// and a tensor scale an expert) in `crates/gpu/src/mxfp4_sel.rs`.
    Nvfp4,
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
    /// `GemmWeight::Iq2Xxs` (`gemm_iq2xxs`).
    Iq2xxs,
    /// `GemmWeight::Iq2Xs` (`gemm_iq2xs`).
    Iq2xs,
    /// `GemmWeight::Iq2S` (`gemm_iq2s`).
    Iq2s,
    /// `GemmWeight::Iq1S` (`gemm_iq1s`).
    Iq1s,
    /// `GemmWeight::Iq1M` (`gemm_iq1m`).
    Iq1m,
    /// `GemmWeight::Iq3Xxs` (`gemm_iq3xxs`).
    Iq3xxs,
    /// `GemmWeight::Iq3S` (`gemm_iq3s`).
    Iq3s,
    /// `GemmWeight::Iq4Xs` (`gemm_iq4xs`).
    Iq4xs,
    /// `Gemm32Weight::Q8_0File` (`gemm_q8_0f`, the file's `block_q8_0` stream)
    /// for a stack read as `KQuant`, or `Gemm32Weight::Q8_0Plane` (`gemm_q8_0p`,
    /// the q8f32 planes) for a dense projection (`SiteEntry::Q8_0`); both in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Q8_0,
    /// `Gemm32Weight::Q5_0Packed` (`gemm_q5_0p`, the `pack_q5_0` layout a
    /// stack's `q5_0_gemv_sel` reads) in `crates/gpu/src/gemm/kernels32.rs`.
    Q5_0,
    /// `Gemm32Weight::Q5_1File` (`gemm_q5_1`) in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Q5_1,
    /// `Gemm32Weight::Iq4NlFile` (`gemm_iq4nl`) in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Iq4nl,
    /// `Gemm32Weight::Nvfp4Planes` (`gemm_nvfp4p`, the two planes and a
    /// tensor scale an expert of the route's stack) in
    /// `crates/gpu/src/gemm/kernels32.rs`.
    Nvfp4,
}

/// A dense gemv site's launch family (`SiteTy` in `crates/gpu/src/site.rs`):
/// `site::gemv` over the f32 rows, `site::kgemv` over the q8_1 rows,
/// `site::blocks32_gemv` over the `Q8Blocks32` rows, and `site::gemm` for a
/// wide unit.
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
    /// `Q5Kernels::enqueue_gemv_q5_0` (`crates/gpu/src/q5.rs`) over the packed
    /// Q5_0 layout, the site as one expert (`row0 = 0`), through
    /// `site::blocks32_gemv` on the `Q8Blocks32` rows; wide, the 32-value GEMM
    /// over the same blocks in `GemmAct32` (`gemm_q5_0p`).
    Q5_0,
    /// `Q8F32Kernels::enqueue_q8_0_gemv` (`enqueue_q8_0_gemv_mcol` past one
    /// column) over the q8f32 planes and f32 rows; wide, `Gemm32Weight::Q8_0Plane`.
    Q8_0,
    /// `Gemm32Kernels::enqueue_f32_tile` over the f32 rows, narrow and wide.
    F32,
    /// `Nvfp4Kernels::enqueue_gemv` (`nvfp4_gemv`, `crates/gpu/src/mxfp4.rs`)
    /// over the NVFP4 planes and the q8_1 rows, times the tensor's scale,
    /// through `site::kgemv`; at a width that is no multiple of 256
    /// (`SiteEntry::reads_at`) `Nvfp4SelKernels::enqueue_sel32` (`nvfp4_gemv_sel32`)
    /// over the planes as one expert and the `Q8Blocks32` rows, through
    /// `site::nvfp4_sel32_gemv`; wide, `Gemm32Weight::Nvfp4Planes` at any
    /// width of the format's 64-value blocks.
    Nvfp4,
    /// `Q8F32Kernels::enqueue_e4m3_gemv` (`mxfp8_gemv`, `mxfp8_gemv_mcol`,
    /// `crates/gpu/src/q8f32.rs`) over bloomery's MXFP8 planes
    /// (`GgmlType::MXFP8`) and the f32 rows; wide, `Gemm32Kernels::enqueue_e4m3_tile`
    /// (`mxfp8_tile_gemm`) over the f32 rows.
    Mxfp8,
    /// The same launchers' FP8_B128 entries (`fp8_b128_gemv`,
    /// `fp8_b128_gemv_mcol`; wide, `fp8_b128_tile_gemm`) over bloomery's
    /// FP8_B128 planes (`GgmlType::FP8_B128`) and the f32 rows.
    Fp8B128,
    /// `Q8F32Kernels::enqueue_bf16_gemv` over BF16 words and f32 rows;
    /// wide, `Q8F32Kernels::enqueue_bf16_tile` over the same words and rows.
    Bf16,
    /// `QuantSelKernels::enqueue_gemv_sel` over one matrix in the file's
    /// IQ4_XS blocks and `Q8Act` columns; wide, the grouped IQ4_XS GEMM.
    Iq4xs,
}

/// The values of one q8_1 super-block, the unit a [`SiteAct::Q8x128`] column holds whole.
const Q8_1_SUPER_BLOCK: u64 = 256;

/// The activation a dense site's launch reads; `crates/gpu/src/site.rs`
/// names it `Form` and a body quantizes a unit's rows into exactly the forms
/// its sites read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteAct {
    /// q8_1 blocks of 128 values (the K-quant gemv's `Q8Act`, the grouped
    /// GEMM's `GemmAct`).
    Q8x128,
    /// The f32 rows on the gemv arm; q8 blocks of 32 values (`GemmAct32`) on
    /// the wide arm.
    Q8x32,
    /// The f32 rows on both arms.
    F32,
    /// q8 blocks of 32 values in the q5 gemv's `Q8Blocks32`: per column
    /// `q_stride` code words, `k/32` block sums and `k/32` scales, written
    /// by `Q5Kernels::enqueue_quantize_q8`; `site::blocks32_gemv` reads them.
    Q8Blocks32,
}

impl SiteEntry {
    /// The activation this site's launch reads.
    #[must_use]
    pub const fn reads(self) -> SiteAct {
        match self {
            SiteEntry::Q3k
            | SiteEntry::Q4k
            | SiteEntry::Q5k
            | SiteEntry::Q6k
            | SiteEntry::Iq4xs => SiteAct::Q8x128,
            SiteEntry::Q8_0 => SiteAct::Q8x32,
            SiteEntry::F32 => SiteAct::F32,
            SiteEntry::Q5_0 => SiteAct::Q8Blocks32,
            SiteEntry::Nvfp4 => SiteAct::Q8x128,
            SiteEntry::Mxfp8 | SiteEntry::Fp8B128 | SiteEntry::Bf16 => SiteAct::F32,
        }
    }

    /// The activation this site's wide prompt GEMM reads: what
    /// [`SiteEntry::reads`] says, except NVFP4, whose GEMM reads the q8 blocks
    /// of 32 ([`SiteAct::Q8x32`]) at any width of its 64-value blocks while its
    /// gemv reads the q8_1 rows.
    #[must_use]
    pub const fn reads_wide(self) -> SiteAct {
        match self {
            SiteEntry::Nvfp4 => SiteAct::Q8x32,
            _ => self.reads(),
        }
    }

    /// The activation this site's launch reads over rows of `k` values: what
    /// [`SiteEntry::reads`] says, except NVFP4 at a width that is no multiple
    /// of the q8_1 super-block's 256 values. Its gemv reads whole super-blocks;
    /// the one-expert `_sel` gemv (`nvfp4_gemv_sel32`) reads the 32-value
    /// [`SiteAct::Q8Blocks32`] and takes every width the format's 64-value block
    /// divides, which every NVFP4 row's is.
    #[must_use]
    pub fn reads_at(self, k: u64) -> SiteAct {
        match self {
            SiteEntry::Nvfp4 if !k.is_multiple_of(Q8_1_SUPER_BLOCK) => SiteAct::Q8Blocks32,
            _ => self.reads(),
        }
    }

    /// The file type whose dense site this entry is: the type of the one row
    /// that names it ([`Card::dense`]), which the build checks for every
    /// entry.
    #[must_use]
    pub const fn ty(self) -> GgmlType {
        DENSE_TYPE[self as usize]
    }
}

/// One past the last [`SiteEntry`]: the span [`DENSE_TYPE`] covers.
const DENSE_SPAN: usize = SiteEntry::Iq4xs as usize + 1;

/// Dense entry to the file type of the row that names it. The build refuses
/// an entry two rows name, an entry no row names, and an entry past
/// [`DENSE_SPAN`].
const DENSE_TYPE: [GgmlType; DENSE_SPAN] = {
    const fn note(slot: &mut [Option<GgmlType>; DENSE_SPAN], rows: &[QRow]) {
        let mut i = 0;
        while i < rows.len() {
            if let Some(entry) = rows[i].card.dense {
                assert!(
                    slot[entry as usize].is_none(),
                    "two rows name one dense entry"
                );
                slot[entry as usize] = Some(rows[i].ty);
            }
            i += 1;
        }
    }
    let mut slot = [None; DENSE_SPAN];
    note(&mut slot, QROWS);
    note(&mut slot, OWN_ROWS);
    let mut ty = [GgmlType::Unknown(0); DENSE_SPAN];
    let mut e = 0;
    while e < DENSE_SPAN {
        match slot[e] {
            Some(t) => ty[e] = t,
            None => panic!("a dense entry no row names"),
        }
        e += 1;
    }
    ty
};

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
    /// `Q8F32Kernels::enqueue_bf16_gemv` over BF16 words and normed f32 rows,
    /// then `argmax_finite_fault`.
    Bf16,
    /// The q8_1 quantizer, the `_sel` Q5_K gemv over the matrix as one expert
    /// (`QuantSelKernels::enqueue_gemv_q5k_sel`, a slot each of the `m`
    /// rows), then `argmax_fault`; past one row the sel gemv's token-major
    /// logits are copied into the layout the rows' argmax reads
    /// (`token_major::TokenMajor`).
    Q5k,
    /// The NVFP4 dense gemv over the matrix's planes and the tensor's scale
    /// (`site::kgemv` over the q8_1 rows, or `site::nvfp4_sel32_gemv` over the
    /// `Q8Blocks32` rows at a width that is no multiple of 256: [`SiteEntry::reads_at`]),
    /// its token-major logits copied into the layout the rows' argmax reads
    /// past one row (`token_major::TokenMajor`), then `argmax_fault`.
    Nvfp4,
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
    /// `Q38Kernels::enqueue_embed_rows` (`embed_rows_bf16`) over BF16 words.
    Bf16,
}

impl DownEntry {
    /// The card layout this entry reads a stack in: the packed q5_0 layout,
    /// NVFP4's planes, and the file's blocks for every other entry.
    pub(crate) const fn layout(self) -> CardFormat {
        match self {
            DownEntry::Q5_0 => CardFormat::Q5_0,
            DownEntry::Nvfp4 => CardFormat::Nvfp4Planes,
            DownEntry::Q4k
            | DownEntry::Q5k
            | DownEntry::Q6k
            | DownEntry::Q5_1
            | DownEntry::Q8_0
            | DownEntry::Iq2xxs
            | DownEntry::Iq2xs
            | DownEntry::Iq2s
            | DownEntry::Iq1s
            | DownEntry::Iq1m
            | DownEntry::Iq3xxs
            | DownEntry::Iq3s
            | DownEntry::Iq4xs
            | DownEntry::Iq4nl
            | DownEntry::Mxfp4 => CardFormat::KQuant,
        }
    }
}

/// A type's card columns, `None` where it has none. A row names every column
/// (`card!`), and the struct has no `Default`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Card {
    /// The layout a file tensor of the type loads in ([`CardFormat::of`]).
    pub layout: Option<CardFormat>,
    /// The layout a routed stack of the type loads in under the default
    /// expert rule ([`CardFormat::of_routed`]): the `layout` column less
    /// q5_K, which that rule does not admit (a program that runs it names it
    /// in its own `RoutedFormat`), and less mxfp8, whose planes have no card
    /// expert entry.
    pub routed: Option<CardFormat>,
    /// The common card gate·up `_sel` entry that runs a stack of the type.
    pub gate_up: Option<GateUpEntry>,
    /// The common card down `_sel` entry that runs a stack of the type, and
    /// the activation form it reads.
    pub down: Option<(DownEntry, DownAct)>,
    /// The prompt-path GEMM entry that runs a stack of the type.
    pub gemm: Option<GemmEntry>,
    /// The dense gemv site launch family for a file tensor of the type.
    pub dense: Option<SiteEntry>,
    /// The lm_head launch family for an `output.weight` of the type.
    pub head: Option<HeadEntry>,
    /// The card embedding row lookup for a token table of the type.
    pub embed: Option<EmbedEntry>,
}

impl Card {
    /// No layout and no entry: a type the card neither loads nor runs.
    pub(crate) const NONE: Card = Card {
        layout: None,
        routed: None,
        gate_up: None,
        down: None,
        gemm: None,
        dense: None,
        head: None,
        embed: None,
    };
}

/// A [`Card`] from its columns in field order (`layout routed gate_up down
/// gemm dense head embed`), `-` for a column the type has none in and
/// `(entry, form)` for a down.
macro_rules! card {
    (@one $E:ident -) => {
        None
    };
    (@one $E:ident $v:ident) => {
        Some($crate::card::$E::$v)
    };
    (@down -) => {
        None
    };
    (@down ($e:ident, $a:ident)) => {
        Some(($crate::card::DownEntry::$e, $crate::card::DownAct::$a))
    };
    ($layout:tt $routed:tt $gate_up:tt $down:tt $gemm:tt $dense:tt $head:tt $embed:tt) => {
        $crate::card::Card {
            layout: $crate::card::card!(@one CardFormat $layout),
            routed: $crate::card::card!(@one CardFormat $routed),
            gate_up: $crate::card::card!(@one GateUpEntry $gate_up),
            down: $crate::card::card!(@down $down),
            gemm: $crate::card::card!(@one GemmEntry $gemm),
            dense: $crate::card::card!(@one SiteEntry $dense),
            head: $crate::card::card!(@one HeadEntry $head),
            embed: $crate::card::card!(@one EmbedEntry $embed),
        }
    };
}
pub(crate) use card;

/// The values of one block of [`CardFormat::Nvfp4Planes`]: NVFP4's block, 32 code bytes and four
/// UE4M3 scales, whichever NVFP4 file type the planes were made from.
const NVFP4_PLANES_BLOCK: u64 = 64;

impl GgmlType {
    /// The card columns of the type's row; `Card::NONE` for a tag with no
    /// row, which has no layout and no entry.
    #[must_use]
    pub const fn card(self) -> Card {
        match self.row() {
            Some(row) => row.card,
            None => Card::NONE,
        }
    }
}

/// A card's layout for a file tensor: the GPU loader's `DevWeight` formats
/// (`crates/gpu/src/weights.rs`). [`CardFormat::buffer_bytes`] is the one
/// owner of their file → device arithmetic: the plan sums it and counts its
/// buffers through the allocator, and the loader refuses and checks every
/// upload by it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardFormat {
    /// The file's rows as they are stored, their byte stream as u32 words,
    /// zero-padded at its end to `rows · ceil(words / rows)` words: every
    /// q3_K/q4_K/q5_K/q6_K tensor, and a routed q5_1, q8_0, iq3_xxs, iq3_s,
    /// iq4_xs, iq4_nl or mxfp4 stack a
    /// program's card experts read in the file's `block_q5_1`, `block_q8_0`,
    /// `block_iq3_xxs`, `block_iq3_s`, `block_iq4_xs`, `block_iq4_nl` or
    /// `block_mxfp4`
    /// bytes (a program's `placement::RoutedFormat` names it; [`CardFormat::of`] keeps
    /// q5_1 in [`CardFormat::Q5_1`] and q8_0 in [`CardFormat::Q8_0Planes`]).
    /// The kernels address a row by its byte offset in the stream, so a row
    /// need not start on a word (q6_K at k = 768 is 630 bytes); the padding
    /// only makes the buffer a whole number of words per row.
    KQuant,
    /// q5_0 in the gemv's layout: per row the file's 22 bytes a 32-value
    /// block regrouped into planes — four `qs` words and one `qh` word a
    /// block, then the f16 scales two to a word.
    Q5_0,
    /// q5_1 in the gemv's layout: per row one byte per 5-bit code in
    /// 1024-value windows, then one f32 scale and one f32 min per block.
    Q5_1,
    /// q8_0 in two planes: per 32-value block 8 code words and the block's
    /// f16 scale bits as the file stores them.
    Q8_0Planes,
    /// nvfp4 in two planes (`crates/gpu/src/mxfp4.rs`, `repack`): a row's
    /// code bytes as `k / 8` words, then its UE4M3 scale bytes four to a
    /// word, `k / 64` words — the file's 36 bytes a 64-value block, moved.
    /// A tensor's f32 `.scale` is its own F32 row; a routed stack (three
    /// dims) adds a third buffer, one f32 scale an expert it holds
    /// (`placement::card_buffers`).
    Nvfp4Planes,
    /// bloomery's mxfp8 (`GgmlType::MXFP8`) in two planes
    /// (`crates/gpu/src/q8f32.rs`, `e4m3_planes`): a row's value bytes as
    /// `k / 4` words, then its E8M0 scale bytes four to a word, `k / 128`
    /// words — the file's 33 bytes a 32-value block, moved.
    Mxfp8Planes,
    /// bloomery's fp8_b128 (`GgmlType::FP8_B128`) in the same two planes
    /// (`e4m3_planes`): a row's value bytes as `k / 4` words, then its
    /// blocks' f32 scales, one word a 128 values — the file's 132 bytes a
    /// block, moved.
    Fp8B128Planes,
    /// f32: the file's values.
    F32,
    /// Decoded to f32 at load (`dequant_row`): bf16 by its type, and a
    /// K-quant that `placement::card_format_of_role` gives it by its role.
    Bf16AsF32,
    /// bf16 as the file stores it, two values to a little-endian u32 word
    /// (value `2j` in the low half of word `j`), for a kernel that widens on
    /// the card. No file type loads in it by [`CardFormat::of`]: a reader
    /// picks it for its own tensors, including dense sites, the head and
    /// embedding rows (`placement::ModelTensors::card_format`).
    Bf16Raw,
}

impl CardFormat {
    /// The format a file tensor of type `ty` loads in on a card, its row's
    /// [`Card::layout`]; `None` when the loader has none. `None` is not a
    /// refusal by itself: a routed stack of such a type stays on the host,
    /// and a tensor its role puts on a card is refused by the plan with
    /// `placement::PlacementError::NoCardFormat`. IQ2_XXS, IQ2_XS, IQ2_S,
    /// IQ1_S and IQ1_M load in the file's blocks ([`CardFormat::KQuant`]),
    /// which `iq2_xxs_gemv_sel` and its kin read for a routed stack and for a
    /// dense matrix as a stack of one expert (the latent MoE body's shared
    /// expert).
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<CardFormat> {
        ty.card().layout
    }

    /// The device buffers of `rows` rows of `k` values of file type `ty` in
    /// this format, in bytes, in the order the upload allocates them: q8_0's
    /// code plane, then its scale plane; one buffer in every other format —
    /// the upload's own arithmetic. `None` exactly where the upload refuses: a
    /// zero dimension, `k` off the type's block (off the planes' 64-value block
    /// for [`CardFormat::Nvfp4Planes`]), or a `ty` this format is not the
    /// loader's format for.
    #[must_use]
    pub fn buffer_bytes(self, ty: GgmlType, k: u64, rows: u64) -> Option<Vec<u64>> {
        let blck = ty.blck_size()?;
        if !self.holds(ty) || k == 0 || rows == 0 || !k.is_multiple_of(blck) {
            return None;
        }
        let blocks = k / blck;
        // The Q5_1 code section: one byte per code, in whole 1024-value windows.
        let window = || k.div_ceil(1024).checked_mul(1024);
        Some(match self {
            CardFormat::KQuant => {
                let words = ty
                    .type_size()?
                    .checked_mul(blocks)?
                    .checked_mul(rows)?
                    .div_ceil(4);
                vec![words.div_ceil(rows).checked_mul(rows)?.checked_mul(4)?]
            }
            CardFormat::Q5_0 => {
                // Per block four `qs` words and one `qh` word, then the f16
                // scales two to a word (`q5::q5_0_row_words`).
                let words = blocks.checked_mul(5)?.checked_add(blocks.div_ceil(2))?;
                vec![rows.checked_mul(words.checked_mul(4)?)?]
            }
            CardFormat::Q5_1 => {
                vec![rows.checked_mul(window()?.checked_add(blocks.checked_mul(8)?)?)?]
            }
            CardFormat::Q8_0Planes => {
                vec![
                    rows.checked_mul(k)?,
                    rows.checked_mul(blocks.checked_mul(2)?)?,
                ]
            }
            CardFormat::Nvfp4Planes => {
                // The planes hold NVFP4's 64-value blocks whatever the file's block (NVFP4_ROWS
                // groups 16): a code nibble a value and a UE4M3 scale a 16 values.
                if !k.is_multiple_of(NVFP4_PLANES_BLOCK) {
                    return None;
                }
                vec![rows.checked_mul(k / 2)?, rows.checked_mul(k / 16)?]
            }
            CardFormat::Mxfp8Planes => {
                // Whole scale words a row: a row's four blocks a word.
                if !k.is_multiple_of(128) {
                    return None;
                }
                vec![rows.checked_mul(k)?, rows.checked_mul(k / 32)?]
            }
            CardFormat::Fp8B128Planes => {
                vec![
                    rows.checked_mul(k)?,
                    rows.checked_mul(blocks.checked_mul(4)?)?,
                ]
            }
            CardFormat::F32 | CardFormat::Bf16AsF32 => vec![rows.checked_mul(k.checked_mul(4)?)?],
            CardFormat::Bf16Raw => {
                // Whole words per row: a row never shares a word with the next.
                if !k.is_multiple_of(2) {
                    return None;
                }
                vec![rows.checked_mul(k.checked_mul(2)?)?]
            }
        })
    }

    /// The format an expert stack of type `ty` loads in on a card under the
    /// default expert rule (`placement::plan_with`), its row's
    /// [`Card::routed`]. A q8_0 or fp8_b128 stack loads in its planes, which
    /// the f32 expert-select gemv reads (`site::resident_sel_f32`, the MTP
    /// draft's experts). A card expert kernel does read q5_K
    /// (`kq_gate_up_act_q5k`, `q5k_gemv_sel`), which that column leaves out,
    /// and the program that runs them names q5_K in its own `RoutedFormat`
    /// (`placement::accepts::routed_rule`). An nvfp4 stack loads in its planes, which
    /// the card leg's nvfp4 pair reads (`placement::kernels::card_planes_runs`).
    /// A stack of a type with none stays on the host.
    #[must_use]
    pub fn of_routed(ty: GgmlType) -> Option<CardFormat> {
        ty.card().routed
    }

    /// A file tensor of type `ty` can be laid out in this format: the
    /// format [`CardFormat::of`] names for `ty`, [`CardFormat::Bf16AsF32`] for
    /// a K-quant (`placement::card_format_of_role`'s engram gain),
    /// [`CardFormat::Bf16Raw`] for bf16, which only a reader that picks it
    /// uses, or the layout a routed card entry of `ty` reads a stack in
    /// (`DownEntry::layout`): the file's blocks
    /// ([`CardFormat::KQuant`]) of q5_1, q8_0, iq2_xs, iq2_s, iq1_s, iq1_m, iq3_xxs, iq3_s,
    /// iq4_xs, iq4_nl and mxfp4, which only a program's
    /// `placement::RoutedFormat` picks.
    #[must_use]
    pub fn holds(self, ty: GgmlType) -> bool {
        CardFormat::of(ty) == Some(self)
            || (self == CardFormat::Bf16AsF32 && CardFormat::of(ty) == Some(CardFormat::KQuant))
            || (matches!(self, CardFormat::Bf16Raw | CardFormat::Mxfp8Planes)
                && ty == GgmlType::BF16)
            || ty.card().down.is_some_and(|(e, _)| e.layout() == self)
    }

    /// The format's name in a plan's rows and a loader's table.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            CardFormat::KQuant => "kquant",
            CardFormat::Q5_0 => "q5_0",
            CardFormat::Q5_1 => "q5_1",
            CardFormat::Q8_0Planes => "q8_0_planes",
            CardFormat::Nvfp4Planes => "nvfp4_planes",
            CardFormat::Mxfp8Planes => "mxfp8_planes",
            CardFormat::Fp8B128Planes => "fp8_b128_planes",
            CardFormat::F32 => "f32",
            CardFormat::Bf16AsF32 => "bf16_as_f32",
            CardFormat::Bf16Raw => "bf16_raw",
        }
    }

    /// Device bytes of `rows` rows of `k` values of file type `ty` in this
    /// format: the sum of its [`CardFormat::buffer_bytes`], `None` where they
    /// are.
    #[must_use]
    pub fn resident_bytes(self, ty: GgmlType, k: u64, rows: u64) -> Option<u64> {
        self.buffer_bytes(ty, k, rows)?
            .into_iter()
            .try_fold(0u64, u64::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::CardFormat;
    use crate::quant::GgmlType;

    /// NVFP4's planes hold a code nibble a value and a UE4M3 scale a 16 values, whichever NVFP4
    /// file type they were made from: NVFP4's 64-value blocks and NVFP4_ROWS' 16-value groups
    /// count the same bytes, and a width off the planes' 64-value block is refused for both.
    /// Mutant, the count before NVFP4_ROWS loaded in planes: blocks of the file type's own block
    /// (NVFP4_ROWS: four times the codes and scales).
    #[test]
    fn nvfp4_planes_count_from_the_planes_geometry() {
        let planes = CardFormat::Nvfp4Planes;
        let (k, rows) = (512, 3);
        let want = Some(vec![rows * k / 2, rows * k / 16]);
        assert_eq!(planes.buffer_bytes(GgmlType::NVFP4, k, rows), want);
        assert_eq!(planes.buffer_bytes(GgmlType::NVFP4_ROWS, k, rows), want);
        for ty in [GgmlType::NVFP4_ROWS, GgmlType::NVFP4] {
            assert_eq!(planes.buffer_bytes(ty, 48, rows), None, "{ty}");
        }
        assert_eq!(planes.buffer_bytes(GgmlType::NVFP4_ROWS, 96, rows), None);
    }
}
