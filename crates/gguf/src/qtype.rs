//! One row per tensor type: the facts a type owns, in one table for ggml's types and one for
//! bloomery's own.
//!
//! [`QROWS`] holds a literal for every ggml type [`GgmlType`] names, and `OWN_ROWS` one for
//! each of bloomery's own value types, with no `Default`, so a new type is decided field by
//! field here. [`GgmlType::row`] finds a type's row by its tag. The
//! size, name, decode and activation questions `quant.rs` answers ([`GgmlType::blck_size`],
//! [`GgmlType::type_size`], [`GgmlType::name`], [`GgmlType::has_dequant`],
//! [`crate::activation_format`]) and [`crate::ggml_type_info`] are reads of a row, not lists of
//! their own. A tag with no row (every [`GgmlType::Unknown`]) answers `None` or a named
//! refusal from each of them, never a default row. A row's `card` is the type's card columns
//! ([`crate::card`]): the layouts it loads in on a card and the entry each card operation runs
//! it with, which `placement::kernels` reads by operation.
//!
//! The numbers of [`QROWS`] are ggml's. `blck` and `bytes` are `blck_size` and `type_size` of the type_traits
//! table (ik_llama.cpp ggml.c:620, mainline's for the types ik lacks), `bytes` as the block
//! struct's `static_assert` in ggml-common.h gives it; the comment above each row names the
//! lines. `name` is `ggml_type_name`, which also names the oracle dumps
//! `$BLOOMERY_DATA/ref/<name>.raw`. The tests in `quant.rs` hold every row to the table
//! compiled from ggml and to the dispatchers a row describes. `OWN_ROWS` cite the decode
//! that defines each block.

use crate::card::{Card, card};
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
    /// The type's card columns ([`crate::card`]): the layouts it loads in on a card and the
    /// card kernel entry each operation runs it with.
    pub card: Card,
}

use GgmlType as T;
use HostAct as H;

/// Every type [`GgmlType`] names, in tag order.
#[rustfmt::skip]
pub const QROWS: &[QRow] = &[
    // `card!` takes the card columns in `Card`'s field order:
    //                 layout        routed        gate_up down                gemm   dense   head embed
    // f32: ggml.c:657
    QRow { ty: T::F32, name: "f32", blck: 1, bytes: 4, dequant: true, host_act: H::F32,
           card: card!(F32           F32           -       -                   -      F32     -    -) },
    // f16: ggml.c:667
    // An f16 tensor loads widened to f32 (exact), the layout a BF16 one loads in; no dense entry names
    // it, so a site over it is the f32 site of its widened rows.
    QRow { ty: T::F16, name: "f16", blck: 1, bytes: 2, dequant: true, host_act: H::F32,
           card: card!(Bf16AsF32     Bf16AsF32     -       -                   -      -       -    -) },
    // q4_0 and q4_1: ggml-common.h:167-171 and :174-178 (two f16 halves for q4_1). Sized and named so
    // a file holding them opens; no decoder and no CPU matmul here, so every use is refused by name.
    QRow { ty: T::Q4_0, name: "q4_0", blck: 32, bytes: 18, dequant: false, host_act: H::None, card: Card::NONE },
    QRow { ty: T::Q4_1, name: "q4_1", blck: 32, bytes: 20, dequant: false, host_act: H::None, card: Card::NONE },
    // q5_0: ggml.c:756, ggml-common.h:196-216; q5_1: ggml.c:777; q8_0: ggml.c:819, ggml-common.h:233-238
    // (one f16 d and 32 int8 codes)
    QRow { ty: T::Q5_0, name: "q5_0", blck: 32, bytes: 22, dequant: true, host_act: H::Q8_2X4,
           card: card!(Q5_0          Q5_0          -       (Q5_0, Q8Blocks32)  Q5_0   Q5_0    -    -) },
    QRow { ty: T::Q5_1, name: "q5_1", blck: 32, bytes: 24, dequant: true, host_act: H::Q8_2X4,
           card: card!(Q5_1          Q5_1          -       (Q5_1, Q8Blocks32)  Q5_1   -       -    -) },
    QRow { ty: T::Q8_0, name: "q8_0", blck: 32, bytes: 34, dequant: true, host_act: H::Q8_2X4,
           card: card!(Q8_0Planes    Q8_0Planes    Q8_0    (Q8_0, Q8Blocks32)  Q8_0   Q8_0    Q8_0 Q8_0) },
    // q8_1: ggml's activation form (llama.cpp 36a73916 ggml-common.h:258-269, ggml.c:745): two f16, d and s = d * sum(qs),
    // then 32 int8 codes. ggml gives it neither to_float nor vec_dot, so every use here is refused by name.
    QRow { ty: T::Q8_1, name: "q8_1", blck: 32, bytes: 36, dequant: false, host_act: H::None, card: Card::NONE },
    // The K-quants, QK_K = 256 (ggml-common.h:79): block_q2_K :309-314, q3_K :327-332, q4_K :348-353,
    // q5_K :367-373, q6_K :388-393; ggml.c:912/938/998.
    QRow { ty: T::Q2_K, name: "q2_K", blck: 256, bytes: 84, dequant: true, host_act: H::Q8K, card: Card::NONE },
    QRow { ty: T::Q3_K, name: "q3_K", blck: 256, bytes: 110, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        KQuant        -       -                   Q3k    Q3k     -    Q3k) },
    QRow { ty: T::Q4_K, name: "q4_K", blck: 256, bytes: 144, dequant: true, host_act: H::Q8_2X4,
           card: card!(KQuant        KQuant        Q4k     (Q4k, Q8Act)        Q4k    Q4k     Q4k  Q4k) },
    QRow { ty: T::Q5_K, name: "q5_K", blck: 256, bytes: 176, dequant: true, host_act: H::Q8_2X4,
           card: card!(KQuant        -             Q5k     (Q5k, Q8Act)        Q5k    Q5k     Q5k  Q5k) },
    QRow { ty: T::Q6_K, name: "q6_K", blck: 256, bytes: 210, dequant: true, host_act: H::Q8_2X4,
           card: card!(KQuant        KQuant        Q6k     (Q6k, Q8Act)        Q6k    Q6k     Q6k  Q6k) },
    // q8_K: ggml's activation form, mainline's block (llama.cpp 36a73916 ggml-common.h:371-376, ggml.c:880): float d, 256 int8
    // codes, 16 int16 bsums = 292 B. ik's block adds a float sum, 296 B (c10fbbcc ggml-common.h:405-411): a divergence.
    QRow { ty: T::Q8_K, name: "q8_K", blck: 256, bytes: 292, dequant: false, host_act: H::None, card: Card::NONE },
    // The i-quants, QK_K = 256 (ggml.c:1055-1293): block_iq2_xxs ggml-common.h:443, iq2_xs :458, iq3_xxs :492,
    // iq1_s :526, iq3_s :510, iq2_s :474, iq4_xs :608. iq4_nl is QK4_NL = 32 (ggml.c:1276), :590.
    QRow { ty: T::IQ2_XXS, name: "iq2_xxs", blck: 256, bytes: 66, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        KQuant        Iq2xxs  (Iq2xxs, Q8Act)      Iq2xxs -       -    -) },
    QRow { ty: T::IQ2_XS, name: "iq2_xs", blck: 256, bytes: 74, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        KQuant        Iq2xs   (Iq2xs, Q8Act)      Iq2xs  -       -    -) },
    QRow { ty: T::IQ3_XXS, name: "iq3_xxs", blck: 256, bytes: 98, dequant: true, host_act: H::Q8K,
           card: card!(-             -             Iq3xxs  (Iq3xxs, Q8Act)     Iq3xxs -       -    -) },
    QRow { ty: T::IQ1_S, name: "iq1_s", blck: 256, bytes: 50, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        KQuant        Iq1s    (Iq1s, Q8Act)       Iq1s   -       -    -) },
    QRow { ty: T::IQ4_NL, name: "iq4_nl", blck: 32, bytes: 18, dequant: true, host_act: H::Q8_2X4,
           card: card!(-             -             -       (Iq4nl, Q8Blocks32) Iq4nl  -       -    -) },
    QRow { ty: T::IQ3_S, name: "iq3_s", blck: 256, bytes: 110, dequant: true, host_act: H::Q8K,
           card: card!(-             -             Iq3s    (Iq3s, Q8Act)       Iq3s   -       -    -) },
    QRow { ty: T::IQ2_S, name: "iq2_s", blck: 256, bytes: 82, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        KQuant        Iq2s    (Iq2s, Q8Act)       Iq2s   -       -    -) },
    QRow { ty: T::IQ4_XS, name: "iq4_xs", blck: 256, bytes: 136, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        -             Iq4xs   (Iq4xs, Q8Act)      Iq4xs  Iq4xs  -    -) },
    // The C scalars: ggml.c:621-649.
    QRow { ty: T::I8, name: "i8", blck: 1, bytes: 1, dequant: false, host_act: H::None, card: Card::NONE },
    QRow { ty: T::I16, name: "i16", blck: 1, bytes: 2, dequant: false, host_act: H::None, card: Card::NONE },
    QRow { ty: T::I32, name: "i32", blck: 1, bytes: 4, dequant: false, host_act: H::None, card: Card::NONE },
    QRow { ty: T::I64, name: "i64", blck: 1, bytes: 8, dequant: false, host_act: H::None, card: Card::NONE },
    QRow { ty: T::F64, name: "f64", blck: 1, bytes: 8, dequant: false, host_act: H::None, card: Card::NONE },
    // iq1_m: ggml-common.h:540. bf16 is one `uint16_t` (ggml.h:380, ggml.c:1485); ggml quantizes its
    // activations to bf16, a form this engine does not encode.
    QRow { ty: T::IQ1_M, name: "iq1_m", blck: 256, bytes: 56, dequant: true, host_act: H::Q8K,
           card: card!(KQuant        KQuant        Iq1m    (Iq1m, Q8Act)       Iq1m   -       -    -) },
    QRow { ty: T::BF16, name: "bf16", blck: 1, bytes: 2, dequant: true, host_act: H::None,
           card: card!(Bf16AsF32     Bf16AsF32     -       -                   -      Bf16    Bf16 Bf16) },
    // Mainline's alone (llama.cpp 930e2fa59): block_tq1_0 ggml-common.h:281, block_tq2_0 :288, ggml.c:912/920.
    QRow { ty: T::TQ1_0, name: "tq1_0", blck: 256, bytes: 54, dequant: false, host_act: H::None, card: Card::NONE },
    QRow { ty: T::TQ2_0, name: "tq2_0", blck: 256, bytes: 66, dequant: false, host_act: H::None, card: Card::NONE },
    // mxfp4: QK_MXFP4 = 32, one E8M0 byte then 16 bytes of 4-bit codes (ggml.c:1308, ggml-common.h:182-187);
    // ggml pairs it with Q8_2_X4 on this AVX2 IQK build (ggml.c:1316).
    QRow { ty: T::MXFP4, name: "mxfp4", blck: 32, bytes: 17, dequant: true, host_act: H::Q8_2X4,
           card: card!(-             -             Mxfp4   (Mxfp4, Q8Act)      -      -       -    -) },
    // nvfp4: mainline's (llama.cpp-mainline ggml.c:760-764, ggml-common.h:221-227), four UE4M3 sub-block scales
    // then 32 bytes of 4-bit codes. Mainline pairs it with Q8_0; qdot's dot reads the Q8_2_X4 column MXFP4's does, and
    // leaves the tensor's `.scale` to its caller.
    QRow { ty: T::NVFP4, name: "nvfp4", blck: 64, bytes: 36, dequant: true, host_act: H::Q8_2X4,
           card: card!(Nvfp4Planes   Nvfp4Planes   Nvfp4   (Nvfp4, Q8Blocks32) Nvfp4  Nvfp4   Nvfp4 -) },
    // q1_0: mainline's (llama.cpp 36a73916 ggml-common.h:180-185, ggml.c:677), one f16 d then 16 bytes of 1-bit codes; ik's
    // q1_0_g128 is the same block at the same id (ggml-common.h:553-558). Its dot takes Q8_0 (ggml-cpu.c:228); sized only here.
    QRow { ty: T::Q1_0, name: "q1_0", blck: 128, bytes: 18, dequant: false, host_act: H::None, card: Card::NONE },
    // q2_0: mainline's (llama.cpp f2cc7282c, ggml.c:685-687, ggml-common.h:187-192), one f16 d then 16 bytes of
    // 2-bit codes; its dot takes Q8_0 (ggml-cpu.c:237).
    QRow { ty: T::Q2_0, name: "q2_0", blck: 64, bytes: 18, dequant: true, host_act: H::Q8_0, card: Card::NONE },
];

/// bloomery's own value types, ids 2000 up, which no ggml table holds: a file holding them
/// opens in bloomery's readers alone.
#[rustfmt::skip]
pub(crate) const OWN_ROWS: &[QRow] = &[
    // `card!` takes the card columns in `Card`'s field order:
    //                 layout        routed        gate_up down                gemm   dense   head embed
    // f8e4m3_rows: one F8_E4M3 byte a value times its tensor's one f32 `<stem>.scale`
    // (`quant::E4m3Rows`), so `dequant_row`, which has no tensor scale, does not decode it.
    QRow { ty: T::F8E4M3_ROWS, name: "f8e4m3_rows", blck: 1, bytes: 1, dequant: false, host_act: H::None, card: Card::NONE },
    // mxfp8: OCP MX with F8_E4M3 elements, the block's E8M0 scale byte then its 32 F8_E4M3 bytes
    // (`quant::check_mxfp8`, `dequant_mxfp8`); no CPU matmul here.
    QRow { ty: T::MXFP8, name: "mxfp8", blck: 32, bytes: 33, dequant: true, host_act: H::None,
           card: card!(Mxfp8Planes   -             -       -                   -      Mxfp8   -    -) },
    // nvfp4_rows: ModelOpt's NVFP4 groups of 16 values, the group's UE4M3 scale byte then its 8
    // code bytes in ModelOpt's order (value 2i in the low nibble of byte i, 2i + 1 in the high),
    // each value e2m1 x scale times its tensor's one f32 `<stem>.scale` (`quant::Nvfp4Rows`), so
    // `dequant_row` does not decode it. A row of any multiple of 16 values is whole groups, where
    // GGUF's NVFP4 block needs 64. Its host dot is qdot's (`dot_row`, or `dot_nvfp4_planes_row` over a
    // row's code and scale planes), against Q8_2_X4 as MXFP4's is, and leaves the tensor scale to the caller.
    // A routed stack loads on the card as NVFP4 does, its nibbles moved into GGUF's order by the part source:
    // NVFP4's routed columns. No dense site reads it.
    QRow { ty: T::NVFP4_ROWS, name: "nvfp4_rows", blck: 16, bytes: 9, dequant: false, host_act: H::Q8_2X4,
           card: card!(Nvfp4Planes   Nvfp4Planes   Nvfp4   (Nvfp4, Q8Blocks32) Nvfp4  -       -    -) },
    // fp8_b128: F8_E4M3 values in blocks of 128 along the row, the block's f32 scale (little-endian)
    // then its 128 value bytes (`quant::check_fp8_b128`, `dequant_fp8_b128`); a checkpoint's one scale
    // a 128 by 128 square is written into each row's block of it. No CPU matmul here.
    QRow { ty: T::FP8_B128, name: "fp8_b128", blck: 128, bytes: 132, dequant: true, host_act: H::None,
           card: card!(Fp8B128Planes Fp8B128Planes -       -                   -      Fp8B128 -    -) },
    // mlx_a4: MLX's affine 4-bit groups of 64 (`quant::check_mlx_affine`, `dequant_mlx_a4`): the
    // group's 8 little-endian U32 words of codes — value i the 4 bits at bit 4·(i mod 8) of word
    // i/8, low nibble first — then its BF16 scale and BF16 bias, the bytes an mlx-community
    // checkpoint holds as `.weight`/`.scales`/`.biases` interleaved per group and moved verbatim.
    // No CPU matmul here.
    QRow { ty: T::MLX_A4, name: "mlx_a4", blck: 64, bytes: 36, dequant: true, host_act: H::None,
           card: Card::NONE },
    // mlx_a8: the same block with one-byte codes (`dequant_mlx_a8`): 16 U32 words, value i byte i,
    // then the BF16 scale and bias.
    QRow { ty: T::MLX_A8, name: "mlx_a8", blck: 64, bytes: 68, dequant: true, host_act: H::None,
           card: Card::NONE },
    // mlx_a4h and mlx_a8h: mlx_a4's and mlx_a8's blocks with the scale and the bias stored as F16
    // bits, where a checkpoint holds `.scales` and `.biases` as F16 (`dequant_mlx_a4`,
    // `dequant_mlx_a8`): the same sizes and codes, a pair `quant::half_to_f32` widens.
    QRow { ty: T::MLX_A4H, name: "mlx_a4h", blck: 64, bytes: 36, dequant: true, host_act: H::None,
           card: Card::NONE },
    QRow { ty: T::MLX_A8H, name: "mlx_a8h", blck: 64, bytes: 68, dequant: true, host_act: H::None,
           card: Card::NONE },
];

/// The build refuses an own row whose type `from_u32` does not name, whose tag ggml's table
/// could hold (below 2000), or whose tag another own row has.
const _: () = {
    let mut i = 0;
    while i < OWN_ROWS.len() {
        let tag = OWN_ROWS[i].ty.as_u32();
        assert!(tag >= 2000, "an own row's tag is in ggml's range");
        assert!(
            !matches!(GgmlType::from_u32(tag), GgmlType::Unknown(_)),
            "an own row's type is not named by GgmlType::from_u32"
        );
        let mut j = 0;
        while j < i {
            assert!(OWN_ROWS[j].ty.as_u32() != tag, "two own rows share one tag");
            j += 1;
        }
        i += 1;
    }
};

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
    /// The row of a type this enum names, ggml's or bloomery's own; `None` for
    /// [`GgmlType::Unknown`], whatever tag it holds.
    #[must_use]
    #[inline]
    pub const fn row(self) -> Option<&'static QRow> {
        if matches!(self, GgmlType::Unknown(_)) {
            return None;
        }
        let tag = self.as_u32() as usize;
        if tag < TAG_SPAN && SLOT[tag] != NO_ROW {
            return Some(&QROWS[SLOT[tag] as usize]);
        }
        let mut i = 0;
        while i < OWN_ROWS.len() {
            if OWN_ROWS[i].ty.as_u32() as usize == tag {
                return Some(&OWN_ROWS[i]);
            }
            i += 1;
        }
        None
    }
}
