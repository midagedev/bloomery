//! IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS and Q2_K weights against q8_1 activations: the
//! card layout of each format, one lane's share of a row's dot
//! (device-callable, the [`crate::mxfp4`] shape), and the same rule on the
//! host. IQ3_S, IQ2_S, IQ1_S and IQ1_M have the host rule and the file format
//! below only: their card layout is the file's own blocks ([`crate::iq_sel`]),
//! so they have no planes, no repack and no row core.
//!
//! File formats (ggml's blocks, 256 values each; `gguf::quant` ports their
//! dequant bit for bit):
//! - IQ2_XXS, 66 B: f16 `d` @0, then two u32 words a 32-value sub-block `ib`
//!   @`2 + 8ib`: `idx`, whose byte `l` indexes `iq2xxs_grid` (eight
//!   magnitudes, bytes of a u64) for values `8l .. 8l + 8`, and `aux`, whose
//!   bits `7l .. 7l + 7` are group `l`'s sign index (as IQ2_XS's) and bits
//!   28.. the scale `s`; the factor is `d·(0.5 + s)/4`.
//! - IQ2_XS, 74 B: f16 `d` @0, 32 u16 codes @2, 8 scale bytes @66. Code `g`
//!   of 32-value sub-block `ib` covers values `8g .. 8g + 8` of the block: its
//!   low 9 bits index `iq2xs_grid` (eight magnitudes, bytes of a u64), its
//!   high 7 bits are a sign index `i`, whose eight sign bits are `i` with bit
//!   7 the parity of the seven (ggml's `ksigns_iq2xs[i]`). Sub-block `ib`'s
//!   scale byte `s` gives codes `4ib, 4ib+1` the factor `d·(0.5 + (s & 15))/4`
//!   and codes `4ib+2, 4ib+3` `d·(0.5 + (s >> 4))/4`.
//! - IQ2_S, 82 B: f16 `d` @0, 32 grid-index bytes @2 (four a sub-block), 32 raw
//!   sign bytes @34 (four a sub-block), 8 high-bit bytes `qh` @66 and 8 scale
//!   bytes @74 (one a sub-block). Group `l` (values `8l ..`) of sub-block
//!   `ib` has the index byte `4ib + l` plus bits `2l ..` of `qh[ib]` as bits
//!   8 and 9, into `iq2s_grid` (eight magnitudes, bytes of a u64), and the
//!   sign byte `4ib + l`, whose bit `j` negates value `j`. The scale byte `s`
//!   gives groups 0, 1 the factor `d·(0.5 + (s & 15))/4` and groups 2, 3
//!   `d·(0.5 + (s >> 4))/4`, as IQ2_XS's.
//! - IQ1_S, 50 B: f16 `d` @0, 32 grid-index bytes @2 (four a sub-block), 8 u16
//!   `qh` @34 (one a sub-block). Sub-block `ib`'s `h = qh[ib]`: group `l`
//!   (values `8l ..`) has the index byte `4ib + l` plus bits `3l ..` of `h` as
//!   bits 8..10, into `iq1s_grid` (eight values −1, 0 or 1, bytes of a u64);
//!   bits 12..14 of `h` are the scale `s` and bit 15 the sign of the delta
//!   `±1/8` added to every grid value: a value is `d·(2s + 1)·(g ± 1/8)`,
//!   which is `d/8·(2s + 1)·(8g ± 1)`.
//! - IQ1_M, 56 B: 32 grid-index bytes @0 (four a sub-block), 16 `qh` bytes @32
//!   (two a sub-block), four u16 scales @48. There is no `d` field: its f16
//!   is the top nibbles of the four scale halfwords, the first halfword's the
//!   lowest. Sub-block `ib` has two 3-bit scales, bits `6(ib & 1)` and
//!   `6(ib & 1) + 3` of halfword `ib/2` (the factors `2s + 1` of groups 0, 1
//!   and of groups 2, 3). Group `l` has the index byte `4ib + l` plus the low
//!   three bits (even `l`) or bits 4..6 (odd `l`) of `qh[2ib + l/2]` as bits
//!   8..10, and the delta sign in bit 3 (even `l`) or bit 7 (odd `l`) of that
//!   byte; the group's values are `d/8·(2s + 1)·(8g ± 1)` as IQ1_S's.
//! - IQ3_XXS, 98 B: f16 `d` @0, 64 grid-index bytes @2, eight u32 words @66.
//!   Sub-block `ib` has index bytes `8ib .. 8ib + 8` (each four magnitudes of
//!   `iq3xxs_grid`) and word `aux`: bits 28.. the scale `s`, bits `7l .. 7l+7`
//!   the sign index of group `l` (values `8l ..`, index bytes `2l, 2l + 1`);
//!   the factor is `d·(0.5 + s)/2`.
//! - IQ3_S, 110 B: f16 `d` @0, 64 index bytes @2, 8 high-bit bytes `qh` @66, 32
//!   sign bytes @74, 4 scale bytes @106. Sub-block `ib` has index bytes `8ib ..
//!   8ib + 8`; index `m` (values `4m .. 4m + 4`) is its byte plus bit `m` of
//!   `qh[ib]` as bit 8, into `iq3s_grid` (four magnitudes). The signs are raw:
//!   bit `(4m + j) % 8` of `signs[4ib + m / 2]` negates value `4m + j`. The
//!   scale `s` is nibble `ib & 1` of `scales[ib / 2]`, the factor `d·(2s + 1)`.
//! - IQ4_XS, 136 B: f16 `d` @0, u16 `scales_h` @2, 4 bytes `scales_l` @4, 128
//!   code bytes @8. Sub-block `ib`: 6-bit `ls` = nibble `ib & 1` of
//!   `scales_l[ib/2]` | bits `2ib ..` of `scales_h` << 4, factor `d·(ls − 32)`;
//!   byte `j` of its 16 holds value `j` (low nibble) and `16 + j` (high),
//!   each a `kvalues_iq4nl` entry.
//! - Q2_K, 84 B: 16 scale bytes @0 (low nibble scale, high nibble min, one
//!   per 16 values), 64 code bytes @16, f16 `d` @80, f16 `dmin` @82. Sub-block
//!   `ib = 4c + f` (values `128c + 32f ..`) reads bits `2f ..` of code bytes
//!   `32c .. 32c + 32`, scale bytes `8c + 2f` (its first 16 values) and
//!   `8c + 2f + 1`; a value is `d·(sc & 15)·q − dmin·(sc >> 4)`.
//!
//! The card layout ([`IqFormat::repack`]) keeps every file byte and moves
//! none in: word planes with a row's sub-blocks in order, so sub-block `g`
//! of the stack (row `r`, sub-block `b`: `g = 8·n_sb·r + b`, super-block
//! `h = n_sb·r + b/8`) is found by index alone.
//!
//! | format | plane 0 | plane 1 | plane 2 |
//! |---|---|---|---|
//! | IQ2_XXS | `idx`, `aux`: 2 words per sub-block `g` | — | `d` halves, as IQ2_XS |
//! | IQ2_XS | codes, 2 words per sub-block `g` | scale bytes, 4 per word (`g/4`, byte `g % 4`) | `d` halves, 2 per word (`h/2`, half `h % 2`) |
//! | IQ3_XXS | index bytes, 2 words per `g` | `aux`, 1 word per `g` | `d` halves, as IQ2_XS |
//! | IQ4_XS | code bytes, 4 words per `g` | `d \| scales_h << 16`, `scales_l`: 2 words per `h` | — |
//! | Q2_K | code bytes, 16 words per `h` | scale bytes, 4 words per `h` | `d \| dmin << 16`, 1 word per `h` |
//!
//! The repack refuses a non-finite `d` (or `dmin`), so the card may convert
//! the halves in hardware ([`crate::flash`]'s `half_bits_to_f32` equals
//! `gguf::quant::half_to_f32` everywhere but on NaN).
//!
//! The codebooks are `gguf::iq_tables`' constants, read through a reference:
//! cuda-oxide materializes such a constant as one immutable device global
//! read with `ld.global.nc`, the lanes' divergent entries served from L1 —
//! the placement ik's CUDA build uses (`GGML_TABLE_BEGIN` is `static const
//! __device__` there), where constant memory would serialize a warp's
//! distinct addresses. `kvalues_iq4nl` is a register table read by `prmt`, as
//! MXFP4's is. Signs are computed, not looked up: the parity bit and a byte
//! mask built by one multiply. IQ2_XXS is the exception on both counts: each
//! block copies its grid, recoded as `prmt` selectors, and the sign
//! selectors of all 128 sign indices (computed by the same parity rule) into
//! shared memory once ([`Iq2XxsTab`]), and a group's eight signed values are
//! two table words or'd and two `prmt`s over the magnitudes and their
//! negatives — the same value words, so the same bits. IQ1_S and IQ1_M keep
//! their grid in the global as selectors too ([`iq1_group`]): a group's eight
//! int8 weights `8g ± 1` are two `prmt`s over the delta sign's three-byte
//! table, no per-value work.
//!
//! Activations are the q8_1 quantizer's, as for MXFP4: per 128-value block a
//! scale `d_x` and int8 codes in the Q4_K gemv's permutation, word `i` of
//! 32-value sub-block `b` at `256·(b >> 5) + 32·i + (b & 31)`.
//!
//! The rule, one row against one column, both sides:
//! - lane `L` of the row's warp takes sub-blocks `b = L, L + 32, …` ascending;
//! - integer sums over the sub-block's 32 values, exact in i32, by `dp4a` on
//!   the device: IQ2_XS `S0` over values 0..16 and `S1` over 16..32 of the
//!   signed grid values times the codes,
//!   `I = (2(s & 15) + 1)·S0 + (2(s >> 4) + 1)·S1`, IQ2_S and IQ1_M the same
//!   with their own two factors (IQ1 over the int8 weights `8g ± 1`);
//!   IQ2_XXS and IQ3_XXS `I = (2s + 1)·S` over all 32, IQ1_S too (over `8g ±
//!   1`); IQ4_XS `I = (ls − 32)·S`;
//!   Q2_K, per 16-value half `t`, `A_t = Σ q·x` and `B_t = Σ x`, `SA = Σ_t
//!   (sc_t & 15)·A_t`, `SM = Σ_t (sc_t >> 4)·B_t`;
//! - from `f = 0`: IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M `f = fma((d/8)·d_x, I, f)`, IQ3_XXS `f = fma((d/4)·d_x,
//!   I, f)`, IQ3_S `f = fma(d·d_x, I, f)` with `I = (2s + 1)·S` over all 32,
//!   IQ4_XS `f = fma(d·d_x, I, f)`, Q2_K `f = fma(d·d_x, SA, f)` then
//!   `f = fma(−(dmin·d_x), SM, f)`, `d_x` the scale of the 128-value block
//!   holding `b` (`d/8` and `d/4` are exact);
//! - the caller sums the 32 lane values with the warp butterfly
//!   (`warp::reduce_sum_f32`).
//!
//! ik's CUDA dots for IQ2_XS and IQ3_XXS fold the half-step `0.5 + s` into
//! the integer sum with a truncating `/ 2`; this rule keeps `2s + 1` whole and
//! moves the division into the exact f32 scale, so no term is dropped.

use crate::flash::half_bits_to_f32;
use crate::{GpuError, cores, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::dotprod::dp4a_s32;
use cuda_device::prmt::prmt;
use cuda_device::vector::{U32x2, U32x4, as_vectors};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use gguf::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use gguf::quant::{GgmlType, half_to_f32};
use std::sync::Arc;

/// Values per super-block (every format here).
pub const SB_VALUES: usize = 256;
/// Values per sub-block, a lane's unit of work.
pub const SUB_VALUES: usize = 32;
/// Sub-blocks per super-block.
const SUBS: usize = SB_VALUES / SUB_VALUES;

/// The IQ2_XS grid behind a reference: one device global (module doc).
const GRID2: &[u64; 512] = &IQ2XS_GRID;
/// The IQ2_XXS grid as `prmt` selectors ([`grid_sel`]), likewise: 256
/// entries, a byte index. The decode reads it from the block's copy
/// ([`Iq2XxsTab`]).
const GRID2XXS_SEL: &[u32; 256] = &grid_sel(&IQ2XXS_GRID);
/// The IQ2_XXS grid raw, likewise: `iq2xxs_grid_half`'s reads (the grouped
/// GEMM's mma fragments hold values, not selectors).
const GRID2XXS: &[u64; 256] = &IQ2XXS_GRID;
/// The IQ3_XXS grid, likewise.
pub(crate) const GRID3: &[u32; 256] = &IQ3XXS_GRID;
/// The IQ3_S grid, likewise: 512 entries, a 9-bit index.
pub(crate) const GRID3S: &[u32; 512] = &IQ3S_GRID;
/// The IQ2_S grid, likewise: 1,024 entries, a 10-bit index (eight magnitudes,
/// bytes of a u64, as IQ2_XS's).
pub(crate) const GRID2S: &[u64; 1024] = &IQ2S_GRID;
/// The IQ1 grid (IQ1_S's and IQ1_M's) as `prmt` selectors ([`grid1_sel`]),
/// likewise: 2,048 entries, an 11-bit index.
const GRID1_SEL: &[u32; 2048] = &grid1_sel(&IQ1S_GRID);

/// `KVALUES_IQ4NL` as the shared nibble decode's table words
/// ([`cores::table16`]).
const IQ4NL: [u32; 4] = cores::table16(&KVALUES_IQ4NL);

/// The nine formats this module takes. [`IqFormat::Iq3S`], [`IqFormat::Iq2S`],
/// [`IqFormat::Iq1S`] and [`IqFormat::Iq1M`] have the host rule only
/// ([`IqFormat::plane_words`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IqFormat {
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq1S,
    Iq1M,
    Iq3Xxs,
    Iq3S,
    Iq4Xs,
    Q2K,
}

impl IqFormat {
    /// The file type.
    #[must_use]
    pub fn ggml(self) -> GgmlType {
        match self {
            IqFormat::Iq2Xxs => GgmlType::IQ2_XXS,
            IqFormat::Iq2Xs => GgmlType::IQ2_XS,
            IqFormat::Iq2S => GgmlType::IQ2_S,
            IqFormat::Iq1S => GgmlType::IQ1_S,
            IqFormat::Iq1M => GgmlType::IQ1_M,
            IqFormat::Iq3Xxs => GgmlType::IQ3_XXS,
            IqFormat::Iq3S => GgmlType::IQ3_S,
            IqFormat::Iq4Xs => GgmlType::IQ4_XS,
            IqFormat::Q2K => GgmlType::Q2_K,
        }
    }

    /// File bytes per 256-value block, the card layout's too
    /// (`GgmlType::type_size` owns the number).
    #[must_use]
    pub fn block_bytes(self) -> usize {
        let bytes = self
            .ggml()
            .type_size()
            .expect("every IqFormat is a sized GgmlType");
        usize::try_from(bytes).expect("a block size fits in usize")
    }

    /// Whether the format has the plane layout and a row core (every format
    /// but IQ3_S, IQ2_S, IQ1_S and IQ1_M).
    #[must_use]
    pub fn has_planes(self) -> bool {
        self.plane_words(1).is_some()
    }

    /// Words of the three planes for `n` super-blocks, in plane order (the
    /// module doc's table; halves round up to whole words); `None` for a
    /// format with no planes (IQ3_S, IQ2_S, IQ1_S, IQ1_M).
    #[must_use]
    pub fn plane_words(self, n: usize) -> Option<[usize; 3]> {
        match self {
            IqFormat::Iq2Xxs => Some([16 * n, 0, n.div_ceil(2)]),
            IqFormat::Iq2Xs => Some([16 * n, 2 * n, n.div_ceil(2)]),
            IqFormat::Iq3Xxs => Some([16 * n, 8 * n, n.div_ceil(2)]),
            IqFormat::Iq3S | IqFormat::Iq2S | IqFormat::Iq1S | IqFormat::Iq1M => None,
            IqFormat::Iq4Xs => Some([32 * n, 2 * n, 0]),
            IqFormat::Q2K => Some([16 * n, 4 * n, n]),
        }
    }

    /// The card layout of `rows` rows of `k` values from their file bytes.
    /// `k` must be a positive multiple of 256 and `bytes` exactly the rows'
    /// `block_bytes · k / 256 · rows`; a non-finite `d` (or Q2_K `dmin`) is
    /// refused, naming its row and super-block. A format with no planes
    /// ([`IqFormat::plane_words`]) is refused by name.
    pub fn repack(self, bytes: &[u8], rows: usize, k: usize) -> Result<[Vec<u32>; 3], GpuError> {
        let what = "iq::repack";
        if k == 0 || !k.is_multiple_of(SB_VALUES) {
            return Err(GpuError::shape(
                what,
                format!("k {k} is not a positive multiple of 256"),
            ));
        }
        let n_sb = k / SB_VALUES;
        let bb = self.block_bytes();
        if bytes.len() != rows * n_sb * bb {
            return Err(GpuError::shape(
                what,
                format!(
                    "{self:?}: {} bytes for {rows} rows of {k} values, want {}",
                    bytes.len(),
                    rows * n_sb * bb
                ),
            ));
        }
        let n = rows * n_sb;
        let Some([w0, w1, w2]) = self.plane_words(n) else {
            return Err(GpuError::shape(
                what,
                format!("{self:?} has no plane layout: its card layout is the file's blocks"),
            ));
        };
        let (mut p0, mut p1, mut p2) = (
            Vec::with_capacity(w0),
            Vec::with_capacity(w1),
            vec![0u32; w2],
        );
        let word =
            |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
        let half = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
        for (h, blk) in bytes.chunks_exact(bb).enumerate() {
            let halves: &[usize] = match self {
                IqFormat::Q2K => &[80, 82],
                _ => &[0],
            };
            for &at in halves {
                if !half_to_f32(half(blk, at)).is_finite() {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "{self:?} row {} super-block {}: the f16 scale at byte {at} is not finite",
                            h / n_sb,
                            h % n_sb
                        ),
                    ));
                }
            }
            match self {
                IqFormat::Iq2Xxs => {
                    p0.extend((0..16).map(|i| word(blk, 2 + 4 * i)));
                    p2[h / 2] |= u32::from(half(blk, 0)) << (16 * (h % 2));
                }
                IqFormat::Iq2Xs => {
                    p0.extend((0..16).map(|i| word(blk, 2 + 4 * i)));
                    p1.extend((0..2).map(|i| word(blk, 66 + 4 * i)));
                    p2[h / 2] |= u32::from(half(blk, 0)) << (16 * (h % 2));
                }
                IqFormat::Iq3Xxs => {
                    p0.extend((0..16).map(|i| word(blk, 2 + 4 * i)));
                    p1.extend((0..8).map(|i| word(blk, 66 + 4 * i)));
                    p2[h / 2] |= u32::from(half(blk, 0)) << (16 * (h % 2));
                }
                IqFormat::Iq3S | IqFormat::Iq2S | IqFormat::Iq1S | IqFormat::Iq1M => {
                    unreachable!("plane_words refused a format with no planes above")
                }
                IqFormat::Iq4Xs => {
                    p0.extend((0..32).map(|i| word(blk, 8 + 4 * i)));
                    p1.push(word(blk, 0));
                    p1.push(word(blk, 4));
                }
                IqFormat::Q2K => {
                    p0.extend((0..16).map(|i| word(blk, 16 + 4 * i)));
                    p1.extend((0..4).map(|i| word(blk, 4 * i)));
                    p2[h] = word(blk, 80);
                }
            }
        }
        Ok([p0, p1, p2])
    }
}

// ---------------------------------------------------------------- device

/// Byte masks: byte `j` of the result is `0xff` when bit `j` (`j < 4`) of
/// `s` is set. `(s & 15) · 0x0020_4081` puts bit `j` at bit `8j` with no
/// carry (the four shifted copies do not overlap).
#[inline(always)]
pub(crate) fn sign_mask4(s: u32) -> u32 {
    ((s & 15).wrapping_mul(0x0020_4081) & 0x0101_0101).wrapping_mul(0xff)
}

/// The eight sign bits of a 7-bit sign index: the index with bit 7 its
/// parity, ggml's `ksigns_iq2xs`.
#[inline(always)]
pub(crate) fn signs_of(i: u32) -> u32 {
    let mut p = i ^ (i >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    i | ((p & 1) << 7)
}

/// Four unsigned magnitudes negated where `m` is `0xff`: `(g ^ m) + (m & 1)`
/// per byte, with no carry between bytes because every grid magnitude is at
/// least 1 (`!g + 1` stays below 256).
#[inline(always)]
pub(crate) fn signed4(g: u32, m: u32) -> u32 {
    (g ^ m).wrapping_add(m & 0x0101_0101)
}

/// One IQ2_XS code decoded to its eight signed values, as two value-order
/// words.
#[inline(always)]
pub(crate) fn iq2_group(code: u32) -> (u32, u32) {
    // SAFETY: code & 511 < 512, GRID2's length.
    let g = unsafe { *GRID2.get_unchecked((code & 511) as usize) };
    let s = signs_of(code >> 9);
    (
        signed4(g as u32, sign_mask4(s)),
        signed4((g >> 32) as u32, sign_mask4(s >> 4)),
    )
}

/// One IQ2_S group: grid index `i` (ten bits) under the raw sign byte in the
/// low 8 bits of `sb` (its low nibble the first four values), as two
/// value-order words. IQ3_S's shape over IQ2_S's eight-magnitude grid.
#[inline(always)]
pub(crate) fn iq2s_group(i: u32, sb: u32) -> (u32, u32) {
    // SAFETY: i & 1023 < 1024, GRID2S's length.
    let g = unsafe { *GRID2S.get_unchecked((i & 1023) as usize) };
    (
        signed4(g as u32, sign_mask4(sb)),
        signed4((g >> 32) as u32, sign_mask4(sb >> 4)),
    )
}

/// IQ2_XS grid entry `i` (nine bits)'s magnitudes at nibble `n` (bytes `4n ..
/// 4n + 4` of its u64): one group's half, the four values an `mma` A-fragment
/// lane holds (`grouped.rs`'s IQ2_XS decode).
#[inline(always)]
pub(crate) fn iq2xs_grid_half(i: u32, n: u32) -> u32 {
    // SAFETY: i & 511 < 512, GRID2's length.
    unsafe { (*GRID2.get_unchecked((i & 511) as usize) >> (32 * (n & 1))) as u32 }
}

/// IQ2_S grid entry `i` (ten bits)'s magnitudes at nibble `n`, as
/// [`iq2xs_grid_half`].
#[inline(always)]
pub(crate) fn iq2s_grid_half(i: u32, n: u32) -> u32 {
    // SAFETY: i & 1023 < 1024, GRID2S's length.
    unsafe { (*GRID2S.get_unchecked((i & 1023) as usize) >> (32 * (n & 1))) as u32 }
}

/// An IQ1 weight is `8·g + 1` where its group's delta is positive and `8·g -
/// 1` where it is negative, `g` in −1, 0, 1 (the grid): ggml's `d·(2s +
/// 1)·(g ± 1/8)` with the eighth moved into the exact scale `d/8`, so the
/// weights are int8 and every dot is an integer sum. [`IQ1_POS`] holds the
/// three positive-delta weights (−7, 1, 9) as bytes 0..3 of a `prmt` word
/// and [`IQ1_NEG`] the negative-delta ones (−9, −1, 7); selector nibble `c =
/// g + 1` picks weight `c`.
const IQ1_POS: u32 = 0x0009_01f9;
const IQ1_NEG: u32 = 0x0007_fff7;

/// Each IQ1 grid entry as a `prmt` selector over [`IQ1_POS`] or [`IQ1_NEG`]:
/// nibble `j` is value `j`'s `g + 1`. A grid value that is none of −1, 0, 1
/// stops the build.
const fn grid1_sel(g: &[u64; 2048]) -> [u32; 2048] {
    let mut t = [0u32; 2048];
    let mut e = 0;
    while e < 2048 {
        let mut j = 0;
        while j < 8 {
            let c = match (g[e] >> (8 * j)) as u8 {
                0xff => 0,
                0x00 => 1,
                0x01 => 2,
                _ => panic!("an IQ1 grid value is not -1, 0 or 1"),
            };
            t[e] |= c << (4 * j);
            j += 1;
        }
        e += 1;
    }
    t
}

/// The weight table [`IQ1_POS`] or [`IQ1_NEG`] a delta sign picks.
#[inline(always)]
fn iq1_table(neg: bool) -> u32 {
    if neg { IQ1_NEG } else { IQ1_POS }
}

/// One IQ1 group: grid index `i` (eleven bits) under the delta sign `neg`, as
/// two value-order words of int8 weights `8·g ± 1`.
#[inline(always)]
pub(crate) fn iq1_group(i: u32, neg: bool) -> (u32, u32) {
    // SAFETY: i & 2047 < 2048, GRID1_SEL's length.
    let sel = unsafe { *GRID1_SEL.get_unchecked((i & 2047) as usize) };
    let tab = iq1_table(neg);
    (prmt(tab, 0, sel), prmt(tab, 0, sel >> 16))
}

/// IQ1 grid entry `i`'s weights at nibble `n` (value bytes `4n .. 4n + 4`):
/// one group's half, the four values an `mma` A-fragment lane holds, as
/// [`iq2xs_grid_half`] holds its magnitudes.
#[inline(always)]
pub(crate) fn iq1_group_half(i: u32, neg: bool, n: u32) -> u32 {
    // SAFETY: i & 2047 < 2048, GRID1_SEL's length.
    let sel = unsafe { *GRID1_SEL.get_unchecked((i & 2047) as usize) };
    prmt(iq1_table(neg), 0, sel >> (16 * (n & 1)))
}

/// The IQ2_XXS grid's three magnitudes as bytes 0..3 of a `prmt` word (byte
/// 3 unused) and their negatives as bytes 0..3 of the second: selector
/// nibble `c` picks magnitude `c` (0, 1, 2), `c | 4` its negative.
const IQ2XXS_POS: u32 = 0x002b_1908;
const IQ2XXS_NEG: u32 = 0x00d5_e7f8;

/// Each grid entry as a `prmt` selector over [`IQ2XXS_POS`]: nibble `j` the
/// byte of value `j`'s magnitude. A magnitude that is none of the three
/// stops the build.
const fn grid_sel(g: &[u64; 256]) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut e = 0;
    while e < 256 {
        let mut j = 0;
        while j < 8 {
            let c = match (g[e] >> (8 * j)) as u8 {
                0x08 => 0,
                0x19 => 1,
                0x2b => 2,
                _ => panic!("an IQ2_XXS grid magnitude is not 8, 25 or 43"),
            };
            t[e] |= c << (4 * j);
            j += 1;
        }
        e += 1;
    }
    t
}

/// Eight sign bits (a [`signs_of`] word) as selector bit 2 of eight `prmt`
/// nibbles: bit `j` of `s` at bit `4j + 2`, the bit that turns a grid
/// selector's magnitude into its negative ([`IQ2XXS_NEG`]).
#[inline(always)]
pub(crate) fn sign_sel(s: u32) -> u32 {
    let t = (s | (s << 12)) & 0x000f_000f;
    let t = (t | (t << 6)) & 0x0303_0303;
    ((t | (t << 3)) & 0x1111_1111) << 2
}

/// Words of a block's IQ2_XXS decode table: the 256 grid selectors, then the
/// 128 sign selectors.
pub const IQ2XXS_TAB_WORDS: usize = 384;

/// A block's IQ2_XXS decode table in shared memory: word `e < 256` is grid
/// entry `e`'s selector ([`grid_sel`]), word `256 + i` sign index `i`'s
/// [`sign_sel`] of [`signs_of`]. A sub-block's eight values are then two
/// table words a group, or'd, and two `prmt`s: no per-value work, and the
/// table's bank conflicts cost fewer L1 wavefronts than a warp's divergent
/// reads of the global grid.
#[derive(Clone, Copy)]
pub struct Iq2XxsTab(*const u32);

impl Iq2XxsTab {
    /// Fill the table at `t` and wait for the block: thread `i` of the
    /// 256-thread block writes grid word `i` and, below 128, sign word `256
    /// + i`. Ends in the block barrier.
    ///
    /// # Safety
    ///
    /// `t` is a block-shared array of [`IQ2XXS_TAB_WORDS`] words that nothing
    /// else writes; the block is 256 threads and every one of them calls this
    /// once, before any return.
    #[inline(always)]
    pub unsafe fn fill(t: *mut u32) -> Iq2XxsTab {
        let i = thread::threadIdx_x() as usize;
        // SAFETY: the block has 256 threads, so i < GRID2XXS_SEL.len().
        let grid = unsafe { *GRID2XXS_SEL.get_unchecked(i) };
        // SAFETY: t holds 384 shared words; thread i alone writes word i < 256.
        unsafe { *t.add(i) = grid };
        if i < 128 {
            let sign = sign_sel(signs_of(i as u32));
            // SAFETY: 256 + i < 384; this thread alone writes this sign word.
            unsafe { *t.add(256 + i) = sign };
        }
        thread::sync_threads();
        Iq2XxsTab(t)
    }
}

/// The `iq2xxs_grid` entry `i`'s magnitudes at nibble `n` (bytes `4n .. 4n + 4`
/// of its u64): one group's half, the four values an `mma` A-fragment lane
/// holds (`grouped.rs`'s IQ2_XXS decode).
#[inline(always)]
pub(crate) fn iq2xxs_grid_half(i: u32, n: u32) -> u32 {
    // SAFETY: i & 255 < 256, GRID2XXS's length.
    unsafe { (*GRID2XXS.get_unchecked((i & 255) as usize) >> (32 * (n & 1))) as u32 }
}

/// One IQ2_XXS sub-block from its two words (module doc): its eight signed
/// value words in value order and the factor `2s + 1`. Group `l`'s selector
/// is grid word `idx` byte `l` or'd with sign word `aux` bits `7l .. 7l + 7`;
/// its low and high halves pick values `8l .. 8l + 4` and `8l + 4 .. 8l + 8`.
/// The plane core and `iq_sel`'s raw-block reader both decode through it.
#[inline(always)]
pub(crate) fn iq2xxs_sub(idx: u32, aux: u32, tab: Iq2XxsTab) -> ([u32; 8], i32) {
    let group = |l: u32| {
        let (e, si) = ((idx >> (8 * l)) & 255, (aux >> (7 * l)) & 127);
        // SAFETY: e < 256; fill's barrier published this grid word.
        let grid = unsafe { *tab.0.add(e as usize) };
        // SAFETY: 256 + si < 384; fill's barrier published this sign word.
        let sign = unsafe { *tab.0.add(256 + si as usize) };
        let sel = grid | sign;
        (
            prmt(IQ2XXS_POS, IQ2XXS_NEG, sel),
            prmt(IQ2XXS_POS, IQ2XXS_NEG, sel >> 16),
        )
    };
    let (w0, w1) = group(0);
    let (w2, w3) = group(1);
    let (w4, w5) = group(2);
    let (w6, w7) = group(3);
    (
        [w0, w1, w2, w3, w4, w5, w6, w7],
        (2 * (aux >> 28) + 1) as i32,
    )
}

/// One IQ3_XXS group: grid index bytes `i0` (values 0..4) and `i1` (4..8)
/// under the sign index in the low 7 bits of `si`, as two value-order words.
#[inline(always)]
pub(crate) fn iq3_group(i0: u32, i1: u32, si: u32) -> (u32, u32) {
    // SAFETY: both indices are bytes (& 255), below GRID3's length 256.
    let (g0, g1) = unsafe {
        (
            *GRID3.get_unchecked((i0 & 255) as usize),
            *GRID3.get_unchecked((i1 & 255) as usize),
        )
    };
    let s = signs_of(si & 127);
    (signed4(g0, sign_mask4(s)), signed4(g1, sign_mask4(s >> 4)))
}

/// One IQ3_S group: grid indices `m` and `m + 1` (values `4m .. 4m + 8`) of a
/// sub-block, each byte `m & 3` and `(m + 1) & 3` of the index word `c` with
/// bit `m` and `m + 1` of the high-bit byte `qh` as bit 8, under the sign byte
/// in the low 8 bits of `sb` (its low nibble the first four values), as two
/// value-order words. `m` is even and a literal after inlining.
#[inline(always)]
pub(crate) fn iq3s_group(c: u32, qh: u32, m: u32, sb: u32) -> (u32, u32) {
    let idx = |m: u32| ((c >> (8 * (m & 3))) & 255) | (((qh >> m) & 1) << 8);
    // SAFETY: both indices are below 512, GRID3S's length.
    let (g0, g1) = unsafe {
        (
            *GRID3S.get_unchecked(idx(m) as usize),
            *GRID3S.get_unchecked(idx(m + 1) as usize),
        )
    };
    (
        signed4(g0, sign_mask4(sb)),
        signed4(g1, sign_mask4(sb >> 4)),
    )
}

/// The eight IQ4_XS codes of one word decoded to int8 values: the low
/// nibbles of its four bytes and the high nibbles, each as four bytes in byte
/// order — the shared nibble decode ([`cores::table16_word`]) over
/// `kvalues_iq4nl`.
#[inline(always)]
pub(crate) fn iq4_word(qs: u32) -> (u32, u32) {
    cores::table16_word(qs, IQ4NL)
}

/// `Σ dp4a(w[i], x[i])` over words `0 .. 4` and `4 .. 8`, each from 0. The
/// indices are literals so neither array leaves registers.
#[inline(always)]
pub(crate) fn dp4_halves(w: &[u32; 8], x: &[u32; 8]) -> (i32, i32) {
    let lo = dp4a_s32(
        w[3],
        x[3],
        dp4a_s32(w[2], x[2], dp4a_s32(w[1], x[1], dp4a_s32(w[0], x[0], 0))),
    );
    let hi = dp4a_s32(
        w[7],
        x[7],
        dp4a_s32(w[6], x[6], dp4a_s32(w[5], x[5], dp4a_s32(w[4], x[4], 0))),
    );
    (lo, hi)
}

/// `Σ dp4a(w[i], x[i])` over all eight words, from 0, in word order.
#[inline(always)]
pub(crate) fn dp4_all(w: &[u32; 8], x: &[u32; 8]) -> i32 {
    let mut s = dp4a_s32(w[0], x[0], 0);
    s = dp4a_s32(w[1], x[1], s);
    s = dp4a_s32(w[2], x[2], s);
    s = dp4a_s32(w[3], x[3], s);
    s = dp4a_s32(w[4], x[4], s);
    s = dp4a_s32(w[5], x[5], s);
    s = dp4a_s32(w[6], x[6], s);
    dp4a_s32(w[7], x[7], s)
}

/// `Σ x[i]`'s bytes over words `0 .. 4` and `4 .. 8`: `dp4a` against ones.
#[inline(always)]
fn byte_sum_halves(x: &[u32; 8]) -> (i32, i32) {
    const ONE: u32 = 0x0101_0101;
    let lo = dp4a_s32(
        ONE,
        x[3],
        dp4a_s32(ONE, x[2], dp4a_s32(ONE, x[1], dp4a_s32(ONE, x[0], 0))),
    );
    let hi = dp4a_s32(
        ONE,
        x[7],
        dp4a_s32(ONE, x[6], dp4a_s32(ONE, x[5], dp4a_s32(ONE, x[4], 0))),
    );
    (lo, hi)
}

/// Column `c`'s eight q8_1 words of sub-block `b` and its block scale.
///
/// # Safety
///
/// `q.len() >= q0 + c · q_col + 256 · ceil((b + 1) / 32)` and `d8.len() >
/// d0 + c · d_col + b / 4`.
#[inline(always)]
pub(crate) unsafe fn x_words(q: &[u32], d8: &[f32], qb: usize, db: usize) -> ([u32; 8], f32) {
    // SAFETY: qb = q0 + c·q_col + 256·(b >> 5) + (b & 31), so qb + 224 is
    // inside column c's words, and db inside its scales, by this fn's
    // contract.
    unsafe {
        (
            [
                *q.get_unchecked(qb),
                *q.get_unchecked(qb + 32),
                *q.get_unchecked(qb + 64),
                *q.get_unchecked(qb + 96),
                *q.get_unchecked(qb + 128),
                *q.get_unchecked(qb + 160),
                *q.get_unchecked(qb + 192),
                *q.get_unchecked(qb + 224),
            ],
            *d8.get_unchecked(db),
        )
    }
}

/// The per-column body of a core: `$body` runs once for every column
/// `$c < $m` whose bit is set in `$mask`, with `$c` a constant, so the
/// accumulator index never leaves registers.
macro_rules! for_cols {
    ($m:expr, $mask:expr, $c:ident => $body:block) => {
        for_cols!(@each $m, $mask, $c, $body, 0 1 2 3 4 5 6 7);
    };
    (@each $m:expr, $mask:expr, $c:ident, $body:block, $($n:literal)*) => {
        $(
            if $m > $n && ($mask >> $n) & 1 != 0 {
                const $c: usize = $n;
                $body
            }
        )*
    };
}

/// The activation arguments every core takes: column `c` reads its q8_1
/// words from `q0 + c · q_col` and its scales from `d0 + c · d_col`.
#[derive(Clone, Copy)]
pub struct ActCols<'a> {
    pub q: &'a [u32],
    pub d8: &'a [f32],
    pub q0: usize,
    pub q_col: usize,
    pub d0: usize,
    pub d_col: usize,
}

impl ActCols<'_> {
    /// Column `c`'s words and scale for sub-block `b` (see [`x_words`]).
    ///
    /// # Safety
    ///
    /// As [`x_words`] for column `c`.
    #[inline(always)]
    pub(crate) unsafe fn col(&self, c: usize, b: usize) -> ([u32; 8], f32) {
        let qb = self.q0 + c * self.q_col + 256 * (b >> 5) + (b & 31);
        let db = self.d0 + c * self.d_col + (b >> 2);
        // SAFETY: this fn's contract.
        unsafe { x_words(self.q, self.d8, qb, db) }
    }
}

/// Lane `lane`'s partial sums of one IQ2_XS row against `M` q8_1 columns
/// (the module doc's rule), each sub-block decoded once and dotted with
/// every column whose bit is set in `mask`; columns `M..8` and columns
/// outside `mask` stay 0.0.
///
/// # Safety
///
/// `qs` holds at least `8 · n_sb · (row + 1)` vectors, `sc`
/// `2 · n_sb · (row + 1)` words, `dh` `ceil(n_sb · (row + 1) / 2)` words
/// (plane layout of the module doc); for every column `c < M`, `a.q.len() >= a.q0 + c · a.q_col +
/// 256 · ceil(n_sb / 4)` and `a.d8.len() >= a.d0 + c · a.d_col + 2 · n_sb`;
/// `lane < 32`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn iq2_xs_lane_partials<const M: usize>(
    qs: &[U32x2],
    sc: &[u32],
    dh: &[u32],
    row: usize,
    n_sb: usize,
    a: ActCols<'_>,
    mask: u32,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let mut f = [0.0f32; 8];
    let n_sub = SUBS * n_sb;
    let (g0, h0) = (row * n_sub, row * n_sb);
    let mut b = lane;
    while b < n_sub {
        let (g, h) = (g0 + b, h0 + (b >> 3));
        // SAFETY: b < n_sub, so g is a sub-block of the row and h its
        // super-block, inside the three planes by this fn's contract.
        let (v, sw, dw) = unsafe {
            (
                *qs.get_unchecked(g),
                *sc.get_unchecked(g >> 2),
                *dh.get_unchecked(h >> 1),
            )
        };
        let d = half_bits_to_f32((dw >> (16 * (h & 1))) as u16) * 0.125;
        let s = (sw >> (8 * (g & 3))) & 0xff;
        let (a0, a1) = ((2 * (s & 15) + 1) as i32, (2 * (s >> 4) + 1) as i32);
        let [c0, c1] = v.0;
        let (w0, w1) = iq2_group(c0 & 0xffff);
        let (w2, w3) = iq2_group(c0 >> 16);
        let (w4, w5) = iq2_group(c1 & 0xffff);
        let (w6, w7) = iq2_group(c1 >> 16);
        let w = [w0, w1, w2, w3, w4, w5, w6, w7];
        for_cols!(M, mask, C => {
            // SAFETY: b < n_sub and C < M, inside the activation bounds of
            // this fn's contract.
            let (x, dx) = unsafe { a.col(C, b) };
            let (s0, s1) = dp4_halves(&w, &x);
            let i = a0 * s0 + a1 * s1;
            f[C] = (d * dx).mul_add(i as f32, f[C]);
        });
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one IQ2_XXS row, as
/// [`iq2_xs_lane_partials`], decoding through the block's table `tab`.
///
/// # Safety
///
/// `qs` holds at least `8 · n_sb · (row + 1)` vectors and `dh`
/// `ceil(n_sb · (row + 1) / 2)` words; the activation bounds and `lane` as
/// [`iq2_xs_lane_partials`].
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn iq2_xxs_lane_partials<const M: usize>(
    qs: &[U32x2],
    dh: &[u32],
    row: usize,
    n_sb: usize,
    a: ActCols<'_>,
    mask: u32,
    lane: usize,
    tab: Iq2XxsTab,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let mut f = [0.0f32; 8];
    let n_sub = SUBS * n_sb;
    let (g0, h0) = (row * n_sub, row * n_sb);
    let mut b = lane;
    while b < n_sub {
        let (g, h) = (g0 + b, h0 + (b >> 3));
        // SAFETY: b < n_sub, so g and h are inside the planes by this fn's
        // contract.
        let (v, dw) = unsafe { (*qs.get_unchecked(g), *dh.get_unchecked(h >> 1)) };
        let d = half_bits_to_f32((dw >> (16 * (h & 1))) as u16) * 0.125;
        let [idx, aux] = v.0;
        let (w, ai) = iq2xxs_sub(idx, aux, tab);
        for_cols!(M, mask, C => {
            // SAFETY: as in iq2_xs_lane_partials.
            let (x, dx) = unsafe { a.col(C, b) };
            let i = ai * dp4_all(&w, &x);
            f[C] = (d * dx).mul_add(i as f32, f[C]);
        });
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one IQ3_XXS row, as
/// [`iq2_xs_lane_partials`].
///
/// # Safety
///
/// `qs` holds at least `8 · n_sb · (row + 1)` vectors, `aux` as many words,
/// `dh` `ceil(n_sb · (row + 1) / 2)` words; the activation bounds and `lane`
/// as [`iq2_xs_lane_partials`].
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn iq3_xxs_lane_partials<const M: usize>(
    qs: &[U32x2],
    aux: &[u32],
    dh: &[u32],
    row: usize,
    n_sb: usize,
    a: ActCols<'_>,
    mask: u32,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let mut f = [0.0f32; 8];
    let n_sub = SUBS * n_sb;
    let (g0, h0) = (row * n_sub, row * n_sb);
    let mut b = lane;
    while b < n_sub {
        let (g, h) = (g0 + b, h0 + (b >> 3));
        // SAFETY: b < n_sub, so g and h are inside the planes by this fn's
        // contract.
        let (v, ax, dw) = unsafe {
            (
                *qs.get_unchecked(g),
                *aux.get_unchecked(g),
                *dh.get_unchecked(h >> 1),
            )
        };
        let d = half_bits_to_f32((dw >> (16 * (h & 1))) as u16) * 0.25;
        let ai = (2 * (ax >> 28) + 1) as i32;
        let [c0, c1] = v.0;
        let (w0, w1) = iq3_group(c0, c0 >> 8, ax);
        let (w2, w3) = iq3_group(c0 >> 16, c0 >> 24, ax >> 7);
        let (w4, w5) = iq3_group(c1, c1 >> 8, ax >> 14);
        let (w6, w7) = iq3_group(c1 >> 16, c1 >> 24, ax >> 21);
        let w = [w0, w1, w2, w3, w4, w5, w6, w7];
        for_cols!(M, mask, C => {
            // SAFETY: as in iq2_xs_lane_partials.
            let (x, dx) = unsafe { a.col(C, b) };
            let i = ai * dp4_all(&w, &x);
            f[C] = (d * dx).mul_add(i as f32, f[C]);
        });
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one IQ4_XS row, as
/// [`iq2_xs_lane_partials`].
///
/// # Safety
///
/// `qs` holds at least `8 · n_sb · (row + 1)` vectors and `meta` `n_sb ·
/// (row + 1)` vectors; the activation bounds and `lane` as
/// [`iq2_xs_lane_partials`].
#[inline(always)]
pub unsafe fn iq4_xs_lane_partials<const M: usize>(
    qs: &[U32x4],
    meta: &[U32x2],
    row: usize,
    n_sb: usize,
    a: ActCols<'_>,
    mask: u32,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let mut f = [0.0f32; 8];
    let n_sub = SUBS * n_sb;
    let (g0, h0) = (row * n_sub, row * n_sb);
    let mut b = lane;
    while b < n_sub {
        let (g, h) = (g0 + b, h0 + (b >> 3));
        // SAFETY: b < n_sub, so g and h are inside the planes by this fn's
        // contract.
        let (v, mt) = unsafe { (*qs.get_unchecked(g), *meta.get_unchecked(h)) };
        let [m0, sl] = mt.0;
        let d = half_bits_to_f32(m0 as u16);
        let ib = (b & 7) as u32;
        let ls = ((sl >> (8 * (ib >> 1) + 4 * (ib & 1))) & 15) | (((m0 >> (16 + 2 * ib)) & 3) << 4);
        let ai = ls as i32 - 32;
        let [q0, q1, q2, q3] = v.0;
        let (l0, h0v) = iq4_word(q0);
        let (l1, h1) = iq4_word(q1);
        let (l2, h2) = iq4_word(q2);
        let (l3, h3) = iq4_word(q3);
        let w = [l0, l1, l2, l3, h0v, h1, h2, h3];
        for_cols!(M, mask, C => {
            // SAFETY: as in iq2_xs_lane_partials.
            let (x, dx) = unsafe { a.col(C, b) };
            let i = ai * dp4_all(&w, &x);
            f[C] = (d * dx).mul_add(i as f32, f[C]);
        });
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one Q2_K row, as [`iq2_xs_lane_partials`].
///
/// # Safety
///
/// `qs` holds at least `4 · n_sb · (row + 1)` vectors, `sc`
/// `4 · n_sb · (row + 1)` words and `dm` `n_sb · (row + 1)` words; the
/// activation bounds and `lane` as [`iq2_xs_lane_partials`].
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn q2_k_lane_partials<const M: usize>(
    qs: &[U32x4],
    sc: &[u32],
    dm: &[u32],
    row: usize,
    n_sb: usize,
    a: ActCols<'_>,
    mask: u32,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let mut f = [0.0f32; 8];
    let n_sub = SUBS * n_sb;
    let h0 = row * n_sb;
    let mut b = lane;
    while b < n_sub {
        let h = h0 + (b >> 3);
        let (c, fld) = ((b >> 2) & 1, (b & 3) as u32);
        // SAFETY: b < n_sub, so h is the row's super-block and 4h + 2c + 1,
        // 4h + 2c + fld/2 and h are inside the planes by this fn's contract.
        let (va, vb, sw, dw) = unsafe {
            (
                *qs.get_unchecked(4 * h + 2 * c),
                *qs.get_unchecked(4 * h + 2 * c + 1),
                *sc.get_unchecked(4 * h + 2 * c + (fld >> 1) as usize),
                *dm.get_unchecked(h),
            )
        };
        let (d, dmin) = (
            half_bits_to_f32(dw as u16),
            half_bits_to_f32((dw >> 16) as u16),
        );
        let sh = 16 * (fld & 1);
        let (s0, s1) = ((sw >> sh) & 0xff, (sw >> (sh + 8)) & 0xff);
        let (k0, k1, m0, m1) = (
            (s0 & 15) as i32,
            (s1 & 15) as i32,
            (s0 >> 4) as i32,
            (s1 >> 4) as i32,
        );
        let [a0, a1, a2, a3] = va.0;
        let [a4, a5, a6, a7] = vb.0;
        let fs = 2 * fld;
        let two = 0x0303_0303;
        let w = [
            (a0 >> fs) & two,
            (a1 >> fs) & two,
            (a2 >> fs) & two,
            (a3 >> fs) & two,
            (a4 >> fs) & two,
            (a5 >> fs) & two,
            (a6 >> fs) & two,
            (a7 >> fs) & two,
        ];
        for_cols!(M, mask, C => {
            // SAFETY: as in iq2_xs_lane_partials.
            let (x, dx) = unsafe { a.col(C, b) };
            let (q0, q1) = dp4_halves(&w, &x);
            let (b0, b1) = byte_sum_halves(&x);
            let sa = k0 * q0 + k1 * q1;
            let sm = m0 * b0 + m1 * b1;
            f[C] = (d * dx).mul_add(sa as f32, f[C]);
            f[C] = (-(dmin * dx)).mul_add(sm as f32, f[C]);
        });
        b += 32;
    }
    f
}

// ------------------------------------------------------------------ host

/// The f16 bits of an IQ1_M block's `d`: the top nibbles of its four scale
/// halfwords `sc`, `sc[0]`'s the lowest.
#[inline(always)]
pub fn iq1m_scale_bits(sc: [u16; 4]) -> u16 {
    (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
}

/// A file half as f32 (`gguf::quant::half_to_f32`).
fn half_at(blk: &[u8], at: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([blk[at], blk[at + 1]]))
}

/// `±1` for sign bit `j` of a `ksigns_iq2xs` entry.
fn sign(signs: u8, j: usize) -> i32 {
    if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 }
}

/// Sub-block `ib` of one file block against its 32 int8 activation codes in
/// value order, the rule's integer sums: `(I, 0)` for the i-quants, `(SA,
/// SM)` for Q2_K. Scalar, from the file bytes and `gguf`'s ggml tables.
#[must_use]
pub fn sub_sums_host(fmt: IqFormat, blk: &[u8], ib: usize, xq: &[i8; SUB_VALUES]) -> (i32, i32) {
    let x = |j: usize| i32::from(xq[j]);
    match fmt {
        IqFormat::Iq2Xxs => {
            let at = 2 + 8 * ib;
            let aux = u32::from_le_bytes(std::array::from_fn(|i| blk[at + 4 + i]));
            let mut sum = 0i32;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[usize::from(blk[at + l])].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                for (j, &g) in grid.iter().enumerate() {
                    sum += i32::from(g) * sign(signs, j) * x(8 * l + j);
                }
            }
            ((2 * (aux >> 28) as i32 + 1) * sum, 0)
        }
        IqFormat::Iq2Xs => {
            let s = i32::from(blk[66 + ib]);
            let mut half = [0i32; 2];
            for l in 0..4 {
                let at = 2 + 2 * (4 * ib + l);
                let code = u16::from_le_bytes([blk[at], blk[at + 1]]);
                let grid = IQ2XS_GRID[usize::from(code & 511)].to_le_bytes();
                let signs = KSIGNS_IQ2XS[usize::from(code >> 9)];
                for (j, &g) in grid.iter().enumerate() {
                    half[l / 2] += i32::from(g) * sign(signs, j) * x(8 * l + j);
                }
            }
            (
                (2 * (s & 15) + 1) * half[0] + (2 * (s >> 4) + 1) * half[1],
                0,
            )
        }
        IqFormat::Iq2S => {
            let (qh, s) = (blk[66 + ib], i32::from(blk[74 + ib]));
            let mut half = [0i32; 2];
            for l in 0..4 {
                let index =
                    usize::from(blk[2 + 4 * ib + l]) | ((usize::from(qh) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[index].to_le_bytes();
                let signs = blk[34 + 4 * ib + l];
                for (j, &g) in grid.iter().enumerate() {
                    half[l / 2] += i32::from(g) * sign(signs, j) * x(8 * l + j);
                }
            }
            (
                (2 * (s & 15) + 1) * half[0] + (2 * (s >> 4) + 1) * half[1],
                0,
            )
        }
        IqFormat::Iq1S => {
            let h = u16::from_le_bytes([blk[34 + 2 * ib], blk[35 + 2 * ib]]);
            let fac = 2 * i32::from((h >> 12) & 7) + 1;
            let delta = if h & 0x8000 != 0 { -1 } else { 1 };
            let mut sum = 0i32;
            for l in 0..4 {
                let index =
                    usize::from(blk[2 + 4 * ib + l]) | (usize::from((h >> (3 * l)) & 7) << 8);
                for (j, &g) in IQ1S_GRID[index].to_le_bytes().iter().enumerate() {
                    sum += (8 * i32::from(g as i8) + delta) * x(8 * l + j);
                }
            }
            (fac * sum, 0)
        }
        IqFormat::Iq1M => {
            let sc = |i: usize| u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]);
            let three = i32::from(sc(ib / 2) >> (6 * (ib % 2)));
            let fac = [2 * (three & 7) + 1, 2 * ((three >> 3) & 7) + 1];
            let mut half = [0i32; 2];
            for l in 0..4 {
                let h = blk[32 + 2 * ib + l / 2];
                let (high, flag) = if l % 2 == 0 {
                    (h & 7, 0x08)
                } else {
                    ((h >> 4) & 7, 0x80)
                };
                let index = usize::from(blk[4 * ib + l]) | (usize::from(high) << 8);
                let delta = if h & flag != 0 { -1 } else { 1 };
                for (j, &g) in IQ1S_GRID[index].to_le_bytes().iter().enumerate() {
                    half[l / 2] += (8 * i32::from(g as i8) + delta) * x(8 * l + j);
                }
            }
            (fac[0] * half[0] + fac[1] * half[1], 0)
        }
        IqFormat::Iq3Xxs => {
            let aux = u32::from_le_bytes(std::array::from_fn(|i| blk[66 + 4 * ib + i]));
            let mut sum = 0i32;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[usize::from(blk[2 + 8 * ib + 2 * l])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(blk[2 + 8 * ib + 2 * l + 1])].to_le_bytes();
                for j in 0..4 {
                    sum += i32::from(g1[j]) * sign(signs, j) * x(8 * l + j);
                    sum += i32::from(g2[j]) * sign(signs, j + 4) * x(8 * l + 4 + j);
                }
            }
            ((2 * (aux >> 28) as i32 + 1) * sum, 0)
        }
        IqFormat::Iq3S => {
            let qh = blk[66 + ib];
            let scale = i32::from((blk[106 + ib / 2] >> (4 * (ib % 2))) & 0x0f);
            let mut sum = 0i32;
            for m in 0..8 {
                let index = usize::from(blk[2 + 8 * ib + m]) | (usize::from((qh >> m) & 1) << 8);
                let grid = IQ3S_GRID[index].to_le_bytes();
                let signs = blk[74 + 4 * ib + m / 2];
                for (j, &g) in grid.iter().enumerate() {
                    let term = i32::from(g) * x(4 * m + j);
                    sum += if (signs >> (4 * (m % 2) + j)) & 1 != 0 {
                        -term
                    } else {
                        term
                    };
                }
            }
            ((2 * scale + 1) * sum, 0)
        }
        IqFormat::Iq4Xs => {
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let ls = i32::from((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0x0f)
                | (i32::from((scales_h >> (2 * ib)) & 3) << 4);
            let kv = |c: u8| i32::from(KVALUES_IQ4NL[usize::from(c)]);
            let mut sum = 0i32;
            for j in 0..16 {
                let byte = blk[8 + 16 * ib + j];
                sum += kv(byte & 0x0f) * x(j) + kv(byte >> 4) * x(16 + j);
            }
            ((ls - 32) * sum, 0)
        }
        IqFormat::Q2K => {
            let (c, fld) = (ib / 4, ib % 4);
            let (mut sa, mut sm) = (0i32, 0i32);
            for t in 0..2 {
                let scale = blk[8 * c + 2 * fld + t];
                let (mut qa, mut qb) = (0i32, 0i32);
                for j in 16 * t..16 * t + 16 {
                    let q = i32::from((blk[16 + 32 * c + j] >> (2 * fld)) & 3);
                    qa += q * x(j);
                    qb += x(j);
                }
                sa += i32::from(scale & 15) * qa;
                sm += i32::from(scale >> 4) * qb;
            }
            (sa, sm)
        }
    }
}

/// The f32 scales of one file block that the rule's integer sums of
/// [`sub_sums_host`] are multiplied by (with the column's `d_x`): `(d/8, 0)`
/// for IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M, `(d/4, 0)` for IQ3_XXS, `(d,
/// 0)` for IQ3_S and IQ4_XS, `(d, dmin)` for Q2_K (the second scales `SM` and
/// is subtracted). `d/8` and `d/4` are exact. IQ1_M has no `d` field: its f16
/// is spread over the top nibbles of its four scale halfwords, low nibble
/// first ([`iq1m_scale_bits`]).
#[must_use]
pub fn sub_scales_host(fmt: IqFormat, blk: &[u8]) -> (f32, f32) {
    match fmt {
        IqFormat::Iq2Xxs | IqFormat::Iq2Xs | IqFormat::Iq2S | IqFormat::Iq1S => {
            (half_at(blk, 0) * 0.125, 0.0)
        }
        IqFormat::Iq1M => {
            let sc: [u16; 4] =
                std::array::from_fn(|i| u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]));
            (half_to_f32(iq1m_scale_bits(sc)) * 0.125, 0.0)
        }
        IqFormat::Iq3Xxs => (half_at(blk, 0) * 0.25, 0.0),
        IqFormat::Iq3S | IqFormat::Iq4Xs => (half_at(blk, 0), 0.0),
        IqFormat::Q2K => (half_at(blk, 80), half_at(blk, 82)),
    }
}

/// Lane `lane`'s partial of one row on the host, the module doc's rule:
/// `row` the row's file bytes, `xq` the column's int8 codes in value order,
/// `d8` its 128-value block scales.
#[must_use]
pub fn lane_partial_host(fmt: IqFormat, row: &[u8], xq: &[i8], d8: &[f32], lane: usize) -> f32 {
    let bb = fmt.block_bytes();
    let codes = xq.as_chunks::<SUB_VALUES>().0;
    let n_sub = SUBS * row.len() / bb;
    let mut f = 0.0f32;
    let mut b = lane;
    while b < n_sub {
        let blk = &row[bb * (b / SUBS)..bb * (b / SUBS + 1)];
        let (i, m) = sub_sums_host(fmt, blk, b % SUBS, &codes[b]);
        let dx = d8[b / 4];
        let (si, sm) = sub_scales_host(fmt, blk);
        f = (si * dx).mul_add(i as f32, f);
        if fmt == IqFormat::Q2K {
            f = (-(sm * dx)).mul_add(m as f32, f);
        }
        b += 32;
    }
    f
}

// ---------------------------------------------------------------- kernels

/// Threads per block of the row kernels: eight warps, a row each.
const BLOCK: u32 = 256;

/// The rows' stores after a core: lane 0 writes `y[c · rows + row]` for
/// every column `c < m_cols`, each the butterfly of the lanes' partials.
///
/// # Safety
///
/// `row < rows`, `m_cols <= 8`, `y.len() >= m_cols · rows`, and all 32 lanes
/// of the warp enter (the butterfly is warp-wide).
#[inline(always)]
unsafe fn store_cols(
    f: &[f32; 8],
    m_cols: usize,
    rows: usize,
    row: usize,
    lane: usize,
    y: &mut DisjointSlice<f32>,
) {
    for_cols!(m_cols, 0xffu32, C => {
        let s = warp::reduce_sum_f32(f[C]);
        if lane == 0 {
            // SAFETY: C < m_cols and row < rows, so C·rows + row < m_cols·rows
            // <= y.len(); lane 0 alone writes it.
            unsafe {
                *y.get_unchecked_mut(C * rows + row) = s;
            }
        }
    });
}

#[cuda_module]
mod iq_kernels {
    use super::*;

    // The five row kernels: warp `r` (eight to a 256-thread block) runs its
    // format's core over row `r` of the planes (the module doc's layout)
    // against `m_cols` q8_1 columns and stores `y[c · rows + r]`. Column `c`
    // has `256 · n_grp` q words at `c · 256 · n_grp` and `2 · n_sb` scales at
    // `c · 2 · n_sb`. `n_grp` is the host's `ceil(n_sb / 4)` and `n_dh` its
    // `ceil(rows · n_sb / 2)`: the contract grammar has no division, so the
    // ceils travel as arguments. They are the probe entries `ptx-scan` reads
    // each core's registers from.

    /// IQ2_XS rows (`iq2_xs_lane_partials`).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= rows * 16 * n_sb,
            sc.len() >= rows * 2 * n_sb,
            dh.len() >= n_dh,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= m_cols * rows
        )
    )]
    pub fn iq2_xs_rows(
        qs: &[u32],
        sc: &[u32],
        dh: &[u32],
        rows: u32,
        n_sb: u32,
        n_dh: u32,
        n_grp: u32,
        m_cols: u32,
        q: &[u32],
        d8: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let row = (tid / 256) * 8 + (tid % 256) / 32;
        if row >= rows as usize || m_cols == 0 || m_cols > 8 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let mask = (1u32 << m_cols) - 1;
        let a = ActCols {
            q,
            d8,
            q0: 0,
            q_col: 256 * n_grp as usize,
            d0: 0,
            d_col: 2 * n_sb as usize,
        };
        let Some(v) = as_vectors::<U32x2>(qs) else {
            return;
        };
        // n_dh only bounds dh in the launch contract.
        let _ = n_dh;
        // SAFETY: row < rows; the contract gives 8·n_sb vectors and 2·n_sb
        // scale words a row, n_dh = ceil(rows·n_sb/2) halves words, and
        // column c < m_cols its 256·n_grp words and 2·n_sb scales; the
        // returns above are warp-uniform.
        let f = unsafe { iq2_xs_lane_partials::<8>(v, sc, dh, row, n_sb as usize, a, mask, lane) };
        // SAFETY: row < rows and m_cols <= 8 (checked above), y by the contract.
        unsafe { store_cols(&f, m_cols as usize, rows as usize, row, lane, &mut y) };
    }

    /// IQ2_XXS rows (`iq2_xxs_lane_partials`).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= rows * 16 * n_sb,
            dh.len() >= n_dh,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= m_cols * rows
        )
    )]
    pub fn iq2_xxs_rows(
        qs: &[u32],
        dh: &[u32],
        rows: u32,
        n_sb: u32,
        n_dh: u32,
        n_grp: u32,
        m_cols: u32,
        q: &[u32],
        d8: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        static mut TAB: SharedArray<u32, IQ2XXS_TAB_WORDS> = SharedArray::UNINIT;
        // SAFETY: TAB is this block's alone; every thread of the 256-thread
        // block fills it here, before any return.
        let tab = unsafe { Iq2XxsTab::fill(SharedArray::as_raw_mut_ptr(&raw mut TAB)) };
        let tid = thread::index_1d().get();
        let row = (tid / 256) * 8 + (tid % 256) / 32;
        if row >= rows as usize || m_cols == 0 || m_cols > 8 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let mask = (1u32 << m_cols) - 1;
        let a = ActCols {
            q,
            d8,
            q0: 0,
            q_col: 256 * n_grp as usize,
            d0: 0,
            d_col: 2 * n_sb as usize,
        };
        let Some(v) = as_vectors::<U32x2>(qs) else {
            return;
        };
        // n_dh only bounds dh in the launch contract.
        let _ = n_dh;
        // SAFETY: row < rows; the contract gives 8·n_sb word pairs a row,
        // n_dh = ceil(rows·n_sb/2) halves words, and column c < m_cols its
        // 256·n_grp words and 2·n_sb scales; the returns above are
        // warp-uniform.
        let f =
            unsafe { iq2_xxs_lane_partials::<8>(v, dh, row, n_sb as usize, a, mask, lane, tab) };
        // SAFETY: row < rows and m_cols <= 8 (checked above), y by the contract.
        unsafe { store_cols(&f, m_cols as usize, rows as usize, row, lane, &mut y) };
    }

    /// IQ3_XXS rows (`iq3_xxs_lane_partials`).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= rows * 16 * n_sb,
            aux.len() >= rows * 8 * n_sb,
            dh.len() >= n_dh,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= m_cols * rows
        )
    )]
    pub fn iq3_xxs_rows(
        qs: &[u32],
        aux: &[u32],
        dh: &[u32],
        rows: u32,
        n_sb: u32,
        n_dh: u32,
        n_grp: u32,
        m_cols: u32,
        q: &[u32],
        d8: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let row = (tid / 256) * 8 + (tid % 256) / 32;
        if row >= rows as usize || m_cols == 0 || m_cols > 8 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let mask = (1u32 << m_cols) - 1;
        let a = ActCols {
            q,
            d8,
            q0: 0,
            q_col: 256 * n_grp as usize,
            d0: 0,
            d_col: 2 * n_sb as usize,
        };
        let Some(v) = as_vectors::<U32x2>(qs) else {
            return;
        };
        // n_dh only bounds dh in the launch contract.
        let _ = n_dh;
        // SAFETY: as iq2_xs_rows, with 8·n_sb aux words a row.
        let f =
            unsafe { iq3_xxs_lane_partials::<8>(v, aux, dh, row, n_sb as usize, a, mask, lane) };
        // SAFETY: row < rows and m_cols <= 8 (checked above), y by the contract.
        unsafe { store_cols(&f, m_cols as usize, rows as usize, row, lane, &mut y) };
    }

    /// IQ4_XS rows (`iq4_xs_lane_partials`).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= rows * 32 * n_sb,
            meta.len() >= rows * 2 * n_sb,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= m_cols * rows
        )
    )]
    pub fn iq4_xs_rows(
        qs: &[u32],
        meta: &[u32],
        rows: u32,
        n_sb: u32,
        n_grp: u32,
        m_cols: u32,
        q: &[u32],
        d8: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let row = (tid / 256) * 8 + (tid % 256) / 32;
        if row >= rows as usize || m_cols == 0 || m_cols > 8 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let mask = (1u32 << m_cols) - 1;
        let a = ActCols {
            q,
            d8,
            q0: 0,
            q_col: 256 * n_grp as usize,
            d0: 0,
            d_col: 2 * n_sb as usize,
        };
        let (Some(v), Some(mt)) = (as_vectors::<U32x4>(qs), as_vectors::<U32x2>(meta)) else {
            return;
        };
        // SAFETY: row < rows; 8·n_sb code vectors and n_sb meta vectors a
        // row by the contract; the activation bounds as iq2_xs_rows.
        let f = unsafe { iq4_xs_lane_partials::<8>(v, mt, row, n_sb as usize, a, mask, lane) };
        // SAFETY: row < rows and m_cols <= 8 (checked above), y by the contract.
        unsafe { store_cols(&f, m_cols as usize, rows as usize, row, lane, &mut y) };
    }

    /// Q2_K rows (`q2_k_lane_partials`).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= rows * 16 * n_sb,
            sc.len() >= rows * 4 * n_sb,
            dm.len() >= rows * n_sb,
            q.len() >= m_cols * 256 * n_grp,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= m_cols * rows
        )
    )]
    pub fn q2_k_rows(
        qs: &[u32],
        sc: &[u32],
        dm: &[u32],
        rows: u32,
        n_sb: u32,
        n_grp: u32,
        m_cols: u32,
        q: &[u32],
        d8: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let row = (tid / 256) * 8 + (tid % 256) / 32;
        if row >= rows as usize || m_cols == 0 || m_cols > 8 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let mask = (1u32 << m_cols) - 1;
        let a = ActCols {
            q,
            d8,
            q0: 0,
            q_col: 256 * n_grp as usize,
            d0: 0,
            d_col: 2 * n_sb as usize,
        };
        let Some(v) = as_vectors::<U32x4>(qs) else {
            return;
        };
        // SAFETY: row < rows; 4·n_sb code vectors, 4·n_sb scale words and
        // n_sb dm words a row by the contract; the activation bounds as
        // iq2_xs_rows.
        let f = unsafe { q2_k_lane_partials::<8>(v, sc, dm, row, n_sb as usize, a, mask, lane) };
        // SAFETY: row < rows and m_cols <= 8 (checked above), y by the contract.
        unsafe { store_cols(&f, m_cols as usize, rows as usize, row, lane, &mut y) };
    }
}

/// A stack of rows of one format on the card, in the module doc's layout.
pub struct IqRows {
    fmt: IqFormat,
    planes: [DeviceBuffer<u32>; 3],
    rows: usize,
    n_sb: usize,
}

impl IqRows {
    /// Repack `rows` rows of `k` values of `fmt` from their file bytes and
    /// upload them. Load-time only.
    pub fn upload(
        stream: &CudaStream,
        fmt: IqFormat,
        bytes: &[u8],
        rows: usize,
        k: usize,
    ) -> Result<IqRows, GpuError> {
        let [p0, p1, p2] = fmt.repack(bytes, rows, k)?;
        // An empty plane still gets one word, so every buffer is a valid
        // launch argument.
        let up = |p: Vec<u32>| {
            let p = if p.is_empty() { vec![0] } else { p };
            DeviceBuffer::from_host(stream, &p)
        };
        Ok(IqRows {
            fmt,
            planes: [up(p0)?, up(p1)?, up(p2)?],
            rows,
            n_sb: k / SB_VALUES,
        })
    }

    /// The format.
    #[must_use]
    pub fn format(&self) -> IqFormat {
        self.fmt
    }

    /// Rows in the stack.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// The loaded row kernels and their enqueue API. Owns no context and no
/// stream.
pub struct IqKernels {
    module: iq_kernels::LoadedModule,
}

impl IqKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<IqKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(iq_kernels, ctx)? };
        Ok(IqKernels { module })
    }

    /// Enqueue `y[c · rows + r] = row r · column c` for every row of `w` and
    /// `m` q8_1 columns: `q` holds `m` columns of `256 · ceil(n_sb / 4)`
    /// words in the Q4_K permutation and `d8` their `2 · n_sb` block scales
    /// each (a `Q8Act`'s `q4` and `d8`). Asynchronous, capturable.
    pub fn enqueue_rows(
        &self,
        stream: &CudaStream,
        w: &IqRows,
        q: &DeviceBuffer<u32>,
        d8: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "iq::enqueue_rows";
        match w.fmt {
            IqFormat::Iq2Xs | IqFormat::Iq2Xxs | IqFormat::Iq3Xxs => {
                crate::tensor::check_alignment::<U32x2, _>(what, "qs", &w.planes[0])?;
            }
            IqFormat::Iq4Xs => {
                crate::tensor::check_alignment::<U32x4, _>(what, "qs", &w.planes[0])?;
                crate::tensor::check_alignment::<U32x2, _>(what, "meta", &w.planes[1])?;
            }
            IqFormat::Q2K => {
                crate::tensor::check_alignment::<U32x4, _>(what, "qs", &w.planes[0])?;
            }
            IqFormat::Iq3S | IqFormat::Iq2S | IqFormat::Iq1S | IqFormat::Iq1M => {}
        }
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(what, format!("1 <= m <= 8, got {m}")));
        }
        let n_grp = w.n_sb.div_ceil(4);
        if q.len() < m * 256 * n_grp || d8.len() < m * 2 * w.n_sb || y.len() < m * w.rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} columns of {} q words and {} scales into {} outputs need q {} d8 {} y {}",
                    256 * n_grp,
                    2 * w.n_sb,
                    m * w.rows,
                    q.len(),
                    d8.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", w.rows.div_ceil(8))?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        let rows = launch_u32(what, "rows", w.rows)?;
        let n_sb = launch_u32(what, "n_sb", w.n_sb)?;
        let n_grp = launch_u32(what, "n_grp", n_grp)?;
        let m = launch_u32(what, "m", m)?;
        let [p0, p1, p2] = &w.planes;
        let md = &self.module;
        match w.fmt {
            IqFormat::Iq2Xxs => {
                let n_dh = launch_u32(what, "n_dh", p2.len())?;
                let prep = md.prepare_iq2_xxs_rows(cfg)?;
                md.iq2_xxs_rows(stream, &prep, p0, p2, rows, n_sb, n_dh, n_grp, m, q, d8, y)?;
            }
            IqFormat::Iq2Xs => {
                let n_dh = launch_u32(what, "n_dh", p2.len())?;
                let prep = md.prepare_iq2_xs_rows(cfg)?;
                md.iq2_xs_rows(
                    stream, &prep, p0, p1, p2, rows, n_sb, n_dh, n_grp, m, q, d8, y,
                )?;
            }
            IqFormat::Iq3Xxs => {
                let n_dh = launch_u32(what, "n_dh", p2.len())?;
                let prep = md.prepare_iq3_xxs_rows(cfg)?;
                md.iq3_xxs_rows(
                    stream, &prep, p0, p1, p2, rows, n_sb, n_dh, n_grp, m, q, d8, y,
                )?;
            }
            IqFormat::Iq4Xs => {
                let prep = md.prepare_iq4_xs_rows(cfg)?;
                md.iq4_xs_rows(stream, &prep, p0, p1, rows, n_sb, n_grp, m, q, d8, y)?;
            }
            IqFormat::Q2K => {
                let prep = md.prepare_q2_k_rows(cfg)?;
                md.q2_k_rows(stream, &prep, p0, p1, p2, rows, n_sb, n_grp, m, q, d8, y)?;
            }
            IqFormat::Iq3S | IqFormat::Iq2S | IqFormat::Iq1S | IqFormat::Iq1M => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "{:?} has no row core: its card layout is the file's blocks (iq_sel)",
                        w.fmt
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GRID2XXS_SEL, IQ2XS_GRID, IQ2XXS_GRID, IQ2XXS_NEG, IQ2XXS_POS, IQ3S_GRID, IQ3XXS_GRID,
        IqFormat, IqKernels, IqRows, KSIGNS_IQ2XS, sign_mask4, sign_sel, signed4, signs_of,
    };
    use cuda_core::DeviceBuffer;

    /// `prmt`'s generic mode on the host: byte `k` of the result is byte
    /// `sel` nibble `k` of `b:a` (bytes 0..4 `a`, 4..8 `b`). The decode never
    /// sets a nibble's sign-replicate bit, so this refuses it.
    fn prmt_host(a: u32, b: u32, sel: u32) -> u32 {
        let bytes = (u64::from(b) << 32 | u64::from(a)).to_le_bytes();
        (0..4).fold(0u32, |w, k| {
            let n = (sel >> (4 * k)) & 15;
            assert!(
                n < 8,
                "selector {sel:#x} nibble {k} asks for sign replication"
            );
            w | u32::from(bytes[n as usize]) << (8 * k)
        })
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn hw_iq_refuses_unaligned_vector_planes() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA stream");
        let kernels = IqKernels::load(&ctx).expect("IQ kernels");
        let q = DeviceBuffer::zeroed(&stream, 256).expect("activation codes");
        let d8 = DeviceBuffer::zeroed(&stream, 2).expect("activation scales");
        let mut y = DeviceBuffer::zeroed(&stream, 1).expect("output");
        let mut failures = Vec::new();
        for (fmt, plane, align) in [
            (IqFormat::Iq2Xs, 0, 8),
            (IqFormat::Iq2Xxs, 0, 8),
            (IqFormat::Iq3Xxs, 0, 8),
            (IqFormat::Iq4Xs, 0, 16),
            (IqFormat::Iq4Xs, 1, 8),
            (IqFormat::Q2K, 0, 16),
        ] {
            let mut w = std::mem::ManuallyDrop::new(
                IqRows::upload(&stream, fmt, &vec![0; fmt.block_bytes()], 1, 256).expect("IQ rows"),
            );
            let len = w.planes[plane].len();
            let parent = DeviceBuffer::<u32>::zeroed(&stream, len + 1).expect("parent");
            // SAFETY: the u32-aligned window covers len words of parent, which
            // lives through the launch and read; into_raw_parts prevents its free.
            let view = unsafe {
                DeviceBuffer::from_raw_parts(parent.cu_deviceptr() + 4, len, ctx.clone())
            };
            let owned = std::mem::replace(&mut w.planes[plane], view);
            let result = kernels.enqueue_rows(&stream, &w, &q, &d8, 1, &mut y);
            let view = std::mem::replace(&mut w.planes[plane], owned);
            let _parts = view.into_raw_parts();
            let w = std::mem::ManuallyDrop::into_inner(w);
            stream.synchronize().expect("launch finished");
            let name = if plane == 0 { "qs" } else { "meta" };
            println!("{fmt:?} {name}: {result:?}");
            if !matches!(&result, Err(err) if err.to_string().contains(name)
                && err.to_string().contains(&align.to_string()))
            {
                failures.push(format!("{fmt:?} {name}: {result:?}"));
            }
            kernels
                .enqueue_rows(&stream, &w, &q, &d8, 1, &mut y)
                .expect("aligned launch");
            assert_eq!(y.to_host_vec(&stream).expect("output"), [0.0]);
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The IQ2_XXS table decode gives the computed decode's value words: for
    /// every grid entry under every sign index, the two `prmt`s of the or'd
    /// selector equal `signed4` of the grid word under `sign_mask4` of
    /// `signs_of`, both halves — so the cores' dots, and every output bit,
    /// are the computed decode's. `sign_sel` sets bit `4j + 2` from bit `j`
    /// and nothing else.
    #[test]
    fn iq2xxs_table_decode_matches_the_computed_one() {
        for s in 0..256u32 {
            let want = (0..8).fold(0u32, |w, j| w | ((s >> j) & 1) << (4 * j + 2));
            assert_eq!(sign_sel(s), want, "sign bits {s:#x}");
        }
        for (e, &g) in IQ2XXS_GRID.iter().enumerate() {
            for i in 0..128u32 {
                let s = signs_of(i);
                let sel = GRID2XXS_SEL[e] | sign_sel(s);
                let got = (
                    prmt_host(IQ2XXS_POS, IQ2XXS_NEG, sel),
                    prmt_host(IQ2XXS_POS, IQ2XXS_NEG, sel >> 16),
                );
                let want = (
                    signed4(g as u32, sign_mask4(s)),
                    signed4((g >> 32) as u32, sign_mask4(s >> 4)),
                );
                assert_eq!(got, want, "grid entry {e} sign index {i}");
            }
        }
    }

    /// The device's computed signs equal ggml's table, and the SWAR negation
    /// equals the scalar one on every grid word under every sign pattern.
    #[test]
    fn device_signs_match_the_tables() {
        for (i, &s) in KSIGNS_IQ2XS.iter().enumerate() {
            assert_eq!(signs_of(i as u32), u32::from(s), "sign index {i}");
        }
        let words: Vec<u32> = IQ2XS_GRID
            .iter()
            .chain(&IQ2XXS_GRID)
            .flat_map(|g| [*g as u32, (*g >> 32) as u32])
            .chain(IQ3XXS_GRID)
            .chain(IQ3S_GRID)
            .collect();
        for &g in &words {
            for s in 0..16u32 {
                let got = signed4(g, sign_mask4(s)).to_le_bytes();
                for (j, &byte) in g.to_le_bytes().iter().enumerate() {
                    let want = if s >> j & 1 != 0 {
                        -i32::from(byte)
                    } else {
                        i32::from(byte)
                    };
                    assert_eq!(
                        i32::from(got[j] as i8),
                        want,
                        "grid word {g:#x} signs {s:#x}"
                    );
                }
            }
        }
    }
}
