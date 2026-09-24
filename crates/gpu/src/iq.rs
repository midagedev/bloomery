//! IQ2_XS, IQ3_XXS, IQ4_XS and Q2_K weights against q8_1 activations: the
//! card layout of each format, one lane's share of a row's dot
//! (device-callable, the [`crate::mxfp4`] shape), and the same rule on the
//! host.
//!
//! File formats (ggml's blocks, 256 values each; `gguf::quant` ports their
//! dequant bit for bit):
//! - IQ2_XS, 74 B: f16 `d` @0, 32 u16 codes @2, 8 scale bytes @66. Code `g`
//!   of 32-value sub-block `ib` covers values `8g .. 8g + 8` of the block: its
//!   low 9 bits index `iq2xs_grid` (eight magnitudes, bytes of a u64), its
//!   high 7 bits are a sign index `i`, whose eight sign bits are `i` with bit
//!   7 the parity of the seven (ggml's `ksigns_iq2xs[i]`). Sub-block `ib`'s
//!   scale byte `s` gives codes `4ib, 4ib+1` the factor `d·(0.5 + (s & 15))/4`
//!   and codes `4ib+2, 4ib+3` `d·(0.5 + (s >> 4))/4`.
//! - IQ3_XXS, 98 B: f16 `d` @0, 64 grid-index bytes @2, eight u32 words @66.
//!   Sub-block `ib` has index bytes `8ib .. 8ib + 8` (each four magnitudes of
//!   `iq3xxs_grid`) and word `aux`: bits 28.. the scale `s`, bits `7l .. 7l+7`
//!   the sign index of group `l` (values `8l ..`, index bytes `2l, 2l + 1`);
//!   the factor is `d·(0.5 + s)/2`.
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
//! mask built by one multiply.
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
//!   `I = (2(s & 15) + 1)·S0 + (2(s >> 4) + 1)·S1`; IQ3_XXS `I = (2s + 1)·S` over all 32; IQ4_XS `I = (ls − 32)·S`;
//!   Q2_K, per 16-value half `t`, `A_t = Σ q·x` and `B_t = Σ x`, `SA = Σ_t
//!   (sc_t & 15)·A_t`, `SM = Σ_t (sc_t >> 4)·B_t`;
//! - from `f = 0`: IQ2_XS `f = fma((d/8)·d_x, I, f)`, IQ3_XXS `f = fma((d/4)·d_x,
//!   I, f)`, IQ4_XS `f = fma(d·d_x, I, f)`, Q2_K `f = fma(d·d_x, SA, f)` then
//!   `f = fma(−(dmin·d_x), SM, f)`, `d_x` the scale of the 128-value block
//!   holding `b` (`d/8` and `d/4` are exact);
//! - the caller sums the 32 lane values with the warp butterfly
//!   (`warp::reduce_sum_f32`).
//!
//! ik's CUDA dots for IQ2_XS and IQ3_XXS fold the half-step `0.5 + s` into
//! the integer sum with a truncating `/ 2`; this rule keeps `2s + 1` whole and
//! moves the division into the exact f32 scale, so no term is dropped.

use crate::GpuError;
use crate::flash::half_bits_to_f32;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::dotprod::dp4a_s32;
use cuda_device::prmt::prmt;
use cuda_device::vector::{U32x2, U32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use gguf::iq_tables::{IQ2XS_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL};
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
/// The IQ3_XXS grid, likewise.
const GRID3: &[u32; 256] = &IQ3XXS_GRID;

/// `KVALUES_IQ4NL` bytes `4i .. 4i + 4` as a little-endian word: the four
/// `prmt` table words.
const fn kv_word(i: usize) -> u32 {
    let k = KVALUES_IQ4NL;
    (k[4 * i] as u8 as u32)
        | ((k[4 * i + 1] as u8 as u32) << 8)
        | ((k[4 * i + 2] as u8 as u32) << 16)
        | ((k[4 * i + 3] as u8 as u32) << 24)
}
const KV0: u32 = kv_word(0);
const KV1: u32 = kv_word(1);
const KV2: u32 = kv_word(2);
const KV3: u32 = kv_word(3);

/// The four formats this module takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IqFormat {
    Iq2Xs,
    Iq3Xxs,
    Iq4Xs,
    Q2K,
}

impl IqFormat {
    /// The format of a file tensor type, `None` for every other type.
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<IqFormat> {
        match ty {
            GgmlType::IQ2_XS => Some(IqFormat::Iq2Xs),
            GgmlType::IQ3_XXS => Some(IqFormat::Iq3Xxs),
            GgmlType::IQ4_XS => Some(IqFormat::Iq4Xs),
            GgmlType::Q2_K => Some(IqFormat::Q2K),
            _ => None,
        }
    }

    /// The file type.
    #[must_use]
    pub fn ggml(self) -> GgmlType {
        match self {
            IqFormat::Iq2Xs => GgmlType::IQ2_XS,
            IqFormat::Iq3Xxs => GgmlType::IQ3_XXS,
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

    /// Words of the three planes for `n` super-blocks, in plane order (the
    /// module doc's table; halves round up to whole words).
    #[must_use]
    pub fn plane_words(self, n: usize) -> [usize; 3] {
        match self {
            IqFormat::Iq2Xs => [16 * n, 2 * n, n.div_ceil(2)],
            IqFormat::Iq3Xxs => [16 * n, 8 * n, n.div_ceil(2)],
            IqFormat::Iq4Xs => [32 * n, 2 * n, 0],
            IqFormat::Q2K => [16 * n, 4 * n, n],
        }
    }

    /// The card layout of `rows` rows of `k` values from their file bytes.
    /// `k` must be a positive multiple of 256 and `bytes` exactly the rows'
    /// `block_bytes · k / 256 · rows`; a non-finite `d` (or Q2_K `dmin`) is
    /// refused, naming its row and super-block.
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
        let [w0, w1, w2] = self.plane_words(n);
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
fn sign_mask4(s: u32) -> u32 {
    ((s & 15).wrapping_mul(0x0020_4081) & 0x0101_0101).wrapping_mul(0xff)
}

/// The eight sign bits of a 7-bit sign index: the index with bit 7 its
/// parity, ggml's `ksigns_iq2xs`.
#[inline(always)]
fn signs_of(i: u32) -> u32 {
    let mut p = i ^ (i >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    i | ((p & 1) << 7)
}

/// Four unsigned magnitudes negated where `m` is `0xff`: `(g ^ m) + (m & 1)`
/// per byte, with no carry between bytes because every grid magnitude is at
/// least 4 (`!g + 1` stays below 256).
#[inline(always)]
fn signed4(g: u32, m: u32) -> u32 {
    (g ^ m).wrapping_add(m & 0x0101_0101)
}

/// One IQ2_XS code decoded to its eight signed values, as two value-order
/// words.
#[inline(always)]
fn iq2_group(code: u32) -> (u32, u32) {
    // SAFETY: code & 511 < 512, GRID2's length.
    let g = unsafe { *GRID2.get_unchecked((code & 511) as usize) };
    let s = signs_of(code >> 9);
    (
        signed4(g as u32, sign_mask4(s)),
        signed4((g >> 32) as u32, sign_mask4(s >> 4)),
    )
}

/// One IQ3_XXS group: grid index bytes `i0` (values 0..4) and `i1` (4..8)
/// under the sign index in the low 7 bits of `si`, as two value-order words.
#[inline(always)]
fn iq3_group(i0: u32, i1: u32, si: u32) -> (u32, u32) {
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

/// The eight IQ4_XS codes of one word decoded to int8 values: the low
/// nibbles of its four bytes and the high nibbles, each as four bytes in byte
/// order — `mxfp4`'s register table read by `prmt`, over `kvalues_iq4nl`.
#[inline(always)]
fn iq4_word(qs: u32) -> (u32, u32) {
    let sel = qs & 0x7777_7777;
    let pick = 0x3210_3210 | ((qs & 0x8888_8888) >> 1);
    let a = prmt(prmt(KV0, KV1, sel), prmt(KV2, KV3, sel), pick);
    let b = prmt(
        prmt(KV0, KV1, sel >> 16),
        prmt(KV2, KV3, sel >> 16),
        pick >> 16,
    );
    (prmt(a, b, 0x6420), prmt(a, b, 0x7531))
}

/// `Σ dp4a(w[i], x[i])` over words `0 .. 4` and `4 .. 8`, each from 0. The
/// indices are literals so neither array leaves registers.
#[inline(always)]
fn dp4_halves(w: &[u32; 8], x: &[u32; 8]) -> (i32, i32) {
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
fn dp4_all(w: &[u32; 8], x: &[u32; 8]) -> i32 {
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
unsafe fn x_words(q: &[u32], d8: &[f32], qb: usize, db: usize) -> ([u32; 8], f32) {
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
    unsafe fn col(&self, c: usize, b: usize) -> ([u32; 8], f32) {
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
        f = match fmt {
            IqFormat::Iq2Xs => (half_at(blk, 0) * 0.125 * dx).mul_add(i as f32, f),
            IqFormat::Iq3Xxs => (half_at(blk, 0) * 0.25 * dx).mul_add(i as f32, f),
            IqFormat::Iq4Xs => (half_at(blk, 0) * dx).mul_add(i as f32, f),
            IqFormat::Q2K => {
                let f = (half_at(blk, 80) * dx).mul_add(i as f32, f);
                (-(half_at(blk, 82) * dx)).mul_add(m as f32, f)
            }
        };
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

    // The four row kernels: warp `r` (eight to a 256-thread block) runs its
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
        let module = unsafe { iq_kernels::load(ctx)? };
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
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IQ2XS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, sign_mask4, signed4, signs_of};

    /// The device's computed signs equal ggml's table, and the SWAR negation
    /// equals the scalar one on every grid word under every sign pattern.
    #[test]
    fn device_signs_match_the_tables() {
        for (i, &s) in KSIGNS_IQ2XS.iter().enumerate() {
            assert_eq!(signs_of(i as u32), u32::from(s), "sign index {i}");
        }
        let words: Vec<u32> = IQ2XS_GRID
            .iter()
            .flat_map(|g| [*g as u32, (*g >> 32) as u32])
            .chain(IQ3XXS_GRID)
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
