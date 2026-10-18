//! One row per ggml tensor type: the facts a type owns, in one table.
//!
//! [`QROWS`] holds a literal for every type [`GgmlType`] names, with no `Default`, so a new
//! type is decided field by field here. [`GgmlType::row`] finds a type's row by its tag. The
//! size, name, decode and activation questions `quant.rs` answers ([`GgmlType::blck_size`],
//! [`GgmlType::type_size`], [`GgmlType::name`], [`GgmlType::has_dequant`],
//! [`crate::activation_format`]) and [`crate::ggml_type_info`] are reads of a row, not lists of
//! their own. A tag with no row (every [`GgmlType::Unknown`]) answers `None` or a named
//! refusal from each of them, never a default row.
//!
//! The numbers are ggml's. `blck` and `bytes` are `blck_size` and `type_size` of the type_traits
//! table (ik_llama.cpp ggml.c:620, mainline's for the types ik lacks), `bytes` as the block
//! struct's `static_assert` in ggml-common.h gives it; the comment above each row names the
//! lines. `name` is `ggml_type_name`, which also names the oracle dumps
//! `$BLOOMERY_DATA/ref/<name>.raw`. The tests in `quant.rs` hold every row to the table
//! compiled from ggml and to the dispatchers a row describes.

use crate::quant::GgmlType;

/// The activation form this engine's CPU matmul quantizes to before a dot with a weight of the
/// row's type: ggml's `vec_dot_type` for the types a kernel here takes
/// ([`crate::activation_format`] cites each). `F32` is a weight whose activations stay f32;
/// `None` is a type no CPU matmul here takes, so the form is refused by name, not guessed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HostAct {
    Q8K,
    Q8_2X4,
    Q8_0,
    F32,
    None,
}

/// The facts one ggml type owns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QRow {
    pub ty: GgmlType,
    /// `ggml_type_name`, lower case except the K of the K-quants.
    pub name: &'static str,
    /// Values per block (`blck_size`).
    pub blck: u64,
    /// Bytes per block (`type_size`).
    pub bytes: u64,
    /// [`crate::dequant_row`] decodes the type. A row without it is sized: a file holding the
    /// type opens and its bytes move, but every decode of it is `QuantError::Unsupported`.
    pub dequant: bool,
    pub host_act: HostAct,
}

use GgmlType as T;
use HostAct as H;

/// Every type [`GgmlType`] names, in tag order.
#[rustfmt::skip]
pub const QROWS: &[QRow] = &[
    // f32: ggml.c:657
    QRow { ty: T::F32, name: "f32", blck: 1, bytes: 4, dequant: true, host_act: H::F32 },
    // f16: ggml.c:667
    QRow { ty: T::F16, name: "f16", blck: 1, bytes: 2, dequant: true, host_act: H::F32 },
    // q4_0 and q4_1: ggml-common.h:167-171 and :174-178 (two f16 halves for q4_1). Sized and named so
    // a file holding them opens; no decoder and no CPU matmul here, so every use is refused by name.
    QRow { ty: T::Q4_0, name: "q4_0", blck: 32, bytes: 18, dequant: false, host_act: H::None },
    QRow { ty: T::Q4_1, name: "q4_1", blck: 32, bytes: 20, dequant: false, host_act: H::None },
    // q5_0: ggml.c:756, ggml-common.h:196-216; q5_1: ggml.c:777; q8_0: ggml.c:819, ggml-common.h:233-238
    // (one f16 d and 32 int8 codes)
    QRow { ty: T::Q5_0, name: "q5_0", blck: 32, bytes: 22, dequant: true, host_act: H::Q8_2X4 },
    QRow { ty: T::Q5_1, name: "q5_1", blck: 32, bytes: 24, dequant: true, host_act: H::Q8_2X4 },
    QRow { ty: T::Q8_0, name: "q8_0", blck: 32, bytes: 34, dequant: true, host_act: H::Q8_2X4 },
    // The K-quants, QK_K = 256 (ggml-common.h:79): block_q2_K :309-314, q3_K :327-332, q4_K :348-353,
    // q5_K :367-373, q6_K :388-393; ggml.c:912/938/998.
    QRow { ty: T::Q2_K, name: "q2_K", blck: 256, bytes: 84, dequant: true, host_act: H::Q8K },
    QRow { ty: T::Q3_K, name: "q3_K", blck: 256, bytes: 110, dequant: true, host_act: H::Q8K },
    QRow { ty: T::Q4_K, name: "q4_K", blck: 256, bytes: 144, dequant: true, host_act: H::Q8_2X4 },
    QRow { ty: T::Q5_K, name: "q5_K", blck: 256, bytes: 176, dequant: true, host_act: H::Q8_2X4 },
    QRow { ty: T::Q6_K, name: "q6_K", blck: 256, bytes: 210, dequant: true, host_act: H::Q8_2X4 },
    // The i-quants, QK_K = 256 (ggml.c:1055-1293): block_iq2_xxs ggml-common.h:443, iq2_xs :458, iq3_xxs :492,
    // iq1_s :526, iq3_s :510, iq2_s :474, iq4_xs :608. iq4_nl is QK4_NL = 32 (ggml.c:1276), :590.
    QRow { ty: T::IQ2_XXS, name: "iq2_xxs", blck: 256, bytes: 66, dequant: true, host_act: H::Q8K },
    QRow { ty: T::IQ2_XS, name: "iq2_xs", blck: 256, bytes: 74, dequant: true, host_act: H::Q8K },
    QRow { ty: T::IQ3_XXS, name: "iq3_xxs", blck: 256, bytes: 98, dequant: true, host_act: H::Q8K },
    QRow { ty: T::IQ1_S, name: "iq1_s", blck: 256, bytes: 50, dequant: false, host_act: H::None },
    QRow { ty: T::IQ4_NL, name: "iq4_nl", blck: 32, bytes: 18, dequant: true, host_act: H::Q8_2X4 },
    QRow { ty: T::IQ3_S, name: "iq3_s", blck: 256, bytes: 110, dequant: true, host_act: H::Q8K },
    QRow { ty: T::IQ2_S, name: "iq2_s", blck: 256, bytes: 82, dequant: false, host_act: H::None },
    QRow { ty: T::IQ4_XS, name: "iq4_xs", blck: 256, bytes: 136, dequant: true, host_act: H::Q8K },
    // The C scalars: ggml.c:621-649.
    QRow { ty: T::I8, name: "i8", blck: 1, bytes: 1, dequant: false, host_act: H::None },
    QRow { ty: T::I16, name: "i16", blck: 1, bytes: 2, dequant: false, host_act: H::None },
    QRow { ty: T::I32, name: "i32", blck: 1, bytes: 4, dequant: false, host_act: H::None },
    QRow { ty: T::I64, name: "i64", blck: 1, bytes: 8, dequant: false, host_act: H::None },
    QRow { ty: T::F64, name: "f64", blck: 1, bytes: 8, dequant: false, host_act: H::None },
    // iq1_m: ggml-common.h:540. bf16 is one `uint16_t` (ggml.h:380, ggml.c:1485); ggml quantizes its
    // activations to bf16, a form this engine does not encode.
    QRow { ty: T::IQ1_M, name: "iq1_m", blck: 256, bytes: 56, dequant: false, host_act: H::None },
    QRow { ty: T::BF16, name: "bf16", blck: 1, bytes: 2, dequant: true, host_act: H::None },
    // Mainline's alone (llama.cpp 930e2fa59): block_tq1_0 ggml-common.h:281, block_tq2_0 :288, ggml.c:912/920.
    QRow { ty: T::TQ1_0, name: "tq1_0", blck: 256, bytes: 54, dequant: false, host_act: H::None },
    QRow { ty: T::TQ2_0, name: "tq2_0", blck: 256, bytes: 66, dequant: false, host_act: H::None },
    // mxfp4: QK_MXFP4 = 32, one E8M0 byte then 16 bytes of 4-bit codes (ggml.c:1308, ggml-common.h:182-187);
    // ggml pairs it with Q8_2_X4 on this AVX2 IQK build (ggml.c:1316).
    QRow { ty: T::MXFP4, name: "mxfp4", blck: 32, bytes: 17, dequant: true, host_act: H::Q8_2X4 },
    // nvfp4: mainline's (llama.cpp-mainline ggml.c:760-764, ggml-common.h:221-227), four UE4M3 sub-block scales
    // then 32 bytes of 4-bit codes; it has a reference decode and no CPU matmul here.
    QRow { ty: T::NVFP4, name: "nvfp4", blck: 64, bytes: 36, dequant: true, host_act: H::None },
    // q2_0: mainline's (llama.cpp f2cc7282c, ggml.c:685-687, ggml-common.h:187-192), one f16 d then 16 bytes of
    // 2-bit codes; its dot takes Q8_0 (ggml-cpu.c:237).
    QRow { ty: T::Q2_0, name: "q2_0", blck: 64, bytes: 18, dequant: true, host_act: H::Q8_0 },
];

/// The slot a tag has in [`QROWS`], or `NO_ROW`.
const NO_ROW: u8 = u8::MAX;

/// One past the largest tag a row has.
const TAG_SPAN: usize = {
    let mut span = 0;
    let mut i = 0;
    while i < QROWS.len() {
        let tag = QROWS[i].ty.as_u32() as usize;
        if tag >= span {
            span = tag + 1;
        }
        i += 1;
    }
    span
};

/// Tag to slot in [`QROWS`]. The build refuses a table with two rows for one tag, a row whose
/// type `from_u32` does not name, and a table the `u8` slot cannot index.
const SLOT: [u8; TAG_SPAN] = {
    assert!(QROWS.len() < NO_ROW as usize, "QROWS outgrew its u8 slots");
    let mut slot = [NO_ROW; TAG_SPAN];
    let mut i = 0;
    while i < QROWS.len() {
        let ty = QROWS[i].ty;
        assert!(
            !matches!(GgmlType::from_u32(ty.as_u32()), GgmlType::Unknown(_)),
            "a QROWS row's type is not named by GgmlType::from_u32"
        );
        let tag = ty.as_u32() as usize;
        assert!(slot[tag] == NO_ROW, "two QROWS rows share one tag");
        slot[tag] = i as u8;
        i += 1;
    }
    slot
};

impl GgmlType {
    /// The row of a type this enum names; `None` for [`GgmlType::Unknown`], whatever tag it holds.
    #[must_use]
    #[inline]
    pub const fn row(self) -> Option<&'static QRow> {
        if matches!(self, GgmlType::Unknown(_)) {
            return None;
        }
        let tag = self.as_u32() as usize;
        if tag >= TAG_SPAN || SLOT[tag] == NO_ROW {
            return None;
        }
        Some(&QROWS[SLOT[tag] as usize])
    }
}
