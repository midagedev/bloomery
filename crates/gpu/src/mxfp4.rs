//! FP4 weights against q8_1 activations — MXFP4 and NVFP4, one owner: the
//! card layout of each, one lane's share of a row's dot (device-callable),
//! the same rule on the host, and NVFP4's dense gemv ([`Nvfp4Kernels`]).
//!
//! The two formats share their codes: a nibble is an E2M1 value, doubled
//! (`KVALUES_MXFP4`, mainline's `kvalues_fp4`), decoded on the card by the
//! one nibble-table decode ([`crate::cores::table16_word`]) and on the host
//! by `gguf::quant::fp4_pair`. They differ in the scale and in which values a
//! code byte holds:
//! - MXFP4 (ggml's `block_mxfp4`): 17 bytes per 32 values, byte 0 the E8M0
//!   scale `E`, bytes 1..17 the codes; value `j < 16` is the low nibble of
//!   code byte `j` and value `j + 16` its high nibble; code `c` means
//!   `KVALUES_MXFP4[c] · 2^(E − 128)` (`gguf::quant::e8m0_to_f32_half`).
//! - NVFP4 (mainline's `block_nvfp4`, ggml-common.h:221-227): 36 bytes per 64
//!   values, bytes 0..4 the UE4M3 scales of the four 16-value sub-blocks,
//!   bytes 4..36 the codes; sub-block `s`'s value `j < 8` is the low nibble
//!   of code byte `8s + j` and value `j + 8` its high nibble; code `c` means
//!   `KVALUES_MXFP4[c] · ue4m3(d[s]) / 2` (`gguf::quant::ue4m3_to_f32_half`).
//!   The tensor's F32 `<name>.scale` multiplies the product
//!   (llama-graph.cpp `build_lora_mm`: `ggml_mul(res, w_s)`).
//!
//! The card layout ([`repack`]) splits a tensor into two word planes, the
//! file's bytes moved and none added: `qs`, the code bytes in row order as
//! little-endian words, so 32-value unit `g` of the stack (row `r`, unit `u`:
//! `g = r · n_units + u`) is the 16-byte vector `g` — an MXFP4 block, or
//! one half of an NVFP4 block (its sub-blocks `2(g & 1)`, `2(g & 1) + 1`);
//! and `e`, the scale bytes in the same order packed four to a word (an
//! MXFP4 block's one byte, an NVFP4 block's four: one word a block). A
//! warp's lanes take consecutive units, so each lane's code load is one
//! 16-byte vector and the warp's 32 loads are 512 contiguous bytes; the
//! scale loads are lane-consecutive words. The repack refuses a scale byte
//! the card rule does not take: an MXFP4 byte outside [`E_MIN`]..=[`E_MAX`],
//! so the kernel's scale is a normal f32 power of two; an NVFP4 byte that is
//! no UE4M3 scale (0x7F or bit 7 set, `ue4m3_to_f32_half`'s refusals).
//!
//! The MXFP4 file layout is also read as it is ([`lane_partial_file`]): a
//! block `g` of a stream of whole rows starts at byte `17g` and is read
//! through the five words that cover it, so the stack stays the file's
//! blocks (`CardFormat::KQuant`). That path takes every scale byte by ggml's
//! half rule ([`scale_half`]); the planes path takes the normal range only.
//!
//! Activations are the q8_1 quantizer's (`q8_1_quant_block`): per 128-value
//! block a scale `d_x` and int8 codes, stored in the Q4_K gemv's permutation,
//! where value-order word `v` of a column sits at slot `256·(v >> 8) +
//! 32·(v & 7) + 8·((v >> 6) & 3) + ((v >> 3) & 7)`. Word `i` of 32-value
//! unit `u` (`v = 8u + i`) is then at `256·(u >> 5) + 32·i + (u & 31)`: a
//! lane that owns unit `32g + L` reads its eight words lane-consecutive.
//!
//! The rule, one row against one column, both sides ([`Fp4`]):
//! - lane `L` of the row's warp takes units `u = L, L + 32, …` ascending;
//! - MXFP4: unit sum `s_u = Σ_j KVALUES_MXFP4[c_j] · q_j` over the 32 values,
//!   an exact i32 (`|s_u| ≤ 32 · 12 · 127`), by `dp4a`; then
//!   `f = fma(d_w · d_x, s_u as f32, f)`, `d_w = 2^(E − 128)`;
//! - NVFP4: per sub-block `s` of the unit in order, the exact i32 `s_s` over
//!   its 16 values (`|s_s| ≤ 16 · 12 · 127`), then `f = fma(d_s · s_s, d_x,
//!   f)` with `d_s = ue4m3(d[s]) / 2` — a 4-bit significand times a 15-bit
//!   integer, so `d_s · s_s` is exact and each sub-block rounds once;
//! - `f` starts at 0, `d_x` is the scale of the 128-value block holding the
//!   unit (of the unit's own 32 values in the 32-value form,
//!   `crate::q5::Q8Blocks32`), and the caller sums the 32 lane values with the warp butterfly
//!   (`warp::reduce_sum_f32`); NVFP4's gemv then multiplies by the tensor's
//!   scale.
//!
//! Mainline's NVFP4 dot (`vec_dot_nvfp4_q8_1`, ggml-cuda/vecdotq.cuh:334-362
//! at cd0a5b6a1) is the same per-sub-block rule — an exact `dp4a` sum over
//! 16 values, scaled by `ue4m3(d[s]) · d_x` — against its 32-value q8_1
//! blocks, and `mul_mat_vec_q` multiplies the reduced row by the tensor's
//! scale (mmvq.cu:811-818); here the activation is the engine's 128-value
//! q8_1, and the sub-block product `d_s · s_s` is formed exactly before the
//! one rounding.

use crate::cores::{table16, table16_word};
use crate::kquant::q8_0::q8_0_code;
use crate::tensor::Q8Act;
use crate::upload::pack_planes;
use crate::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::dotprod::dp4a_s32;
use cuda_device::vector::{U32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use gguf::quant::{KVALUES_MXFP4, e8m0_to_f32_half, fp4_pair, ue4m3_to_f32_half};
use std::sync::Arc;

/// Values per MXFP4 block.
pub const BLOCK_VALUES: usize = 32;
/// File bytes per MXFP4 block: the scale byte and 16 code bytes.
pub const BLOCK_BYTES: usize = 17;
/// Code words per 32-value unit in the card layout's `qs` plane: one vector.
pub const QS_WORDS: usize = 4;
/// The smallest scale byte the card layout takes: `E = 0` and `1` are the
/// subnormal scales of ggml's half rule.
pub const E_MIN: u8 = 2;
/// The largest: `E = 255` is NaN in the OCP MX format (ggml decodes it as
/// `2^127`).
pub const E_MAX: u8 = 254;

/// Values per NVFP4 block.
pub const NV_BLOCK_VALUES: usize = 64;
/// File bytes per NVFP4 block: four UE4M3 scale bytes and 32 code bytes.
pub const NV_BLOCK_BYTES: usize = 36;

/// Values per lane unit, both formats: one `qs` vector, one 32-value block
/// of the q8_1 activation.
pub const UNIT_VALUES: usize = 32;

/// `KVALUES_MXFP4` as the shared nibble decode's table words.
pub(crate) const E2M1: [u32; 4] = table16(&KVALUES_MXFP4);

/// The eight codes of one `qs` word decoded to int8 values through the E2M1
/// table: the low nibbles of its four bytes, then the high nibbles, each as
/// four bytes in byte order ([`table16_word`]).
#[inline(always)]
fn decode_word(qs: u32) -> (u32, u32) {
    table16_word(qs, E2M1)
}

/// `2^(E − 128)` for a scale byte `E` in [`E_MIN`]..=[`E_MAX`]: the f32 whose
/// exponent field is `E − 1` — ggml's half rule on its normal range.
#[inline(always)]
fn scale_of(e: u32) -> f32 {
    f32::from_bits((e - 1) << 23)
}

/// `2^(E − 128)` for any scale byte `E`: ggml's half rule
/// (`gguf::quant::e8m0_to_f32_half`), `E >= 2` the f32 whose exponent field
/// is `E − 1` (255 is `2^127`), and `E = 0`, `1` the subnormals
/// `2^-129`, `2^-128`.
#[inline(always)]
fn scale_half(e: u32) -> f32 {
    f32::from_bits(if e >= 2 {
        (e - 1) << 23
    } else {
        0x0020_0000 << e
    })
}

/// The eight int8 words one block's four `qs` words decode to, in value
/// order: the low-nibble words of `qs[0..4]`, then their high-nibble words
/// ([`decode_word`]). The first half of the block rule.
#[inline(always)]
pub fn decode_block(qs: [u32; 4]) -> [u32; 8] {
    let (l0, h0) = decode_word(qs[0]);
    let (l1, h1) = decode_word(qs[1]);
    let (l2, h2) = decode_word(qs[2]);
    let (l3, h3) = decode_word(qs[3]);
    [l0, l1, l2, l3, h0, h1, h2, h3]
}

/// The second half of the block rule, one block against one column: the
/// exact i32 block sum of the decoded words `dec` ([`decode_block`]) with the
/// column's eight activation words `x` (value order), by eight `dp4a`, then
/// `fma(d_w · d_x, s, f)`.
#[inline(always)]
pub fn block_fma(dec: &[u32; 8], x: &[u32; 8], dw: f32, dx: f32, f: f32) -> f32 {
    let s = dp4a_s32(dec[0], x[0], 0);
    let s = dp4a_s32(dec[1], x[1], s);
    let s = dp4a_s32(dec[2], x[2], s);
    let s = dp4a_s32(dec[3], x[3], s);
    let s = dp4a_s32(dec[4], x[4], s);
    let s = dp4a_s32(dec[5], x[5], s);
    let s = dp4a_s32(dec[6], x[6], s);
    let s = dp4a_s32(dec[7], x[7], s);
    (dw * dx).mul_add(s as f32, f)
}

/// `ue4m3(b) / 2` for a UE4M3 scale byte `b` the repack took (`b <= 0x7E`):
/// `gguf::quant::ue4m3_to_f32_half` on its defined domain. Exponent 0 is the
/// subnormal `m · 2^-9`, halved: `m · 2^-10`, exact from the integer. Any
/// other byte, `(1 + m/8) · 2^(e − 7)` halved, is the f32 whose exponent
/// field is `e + 119` and whose top three mantissa bits are `m`: the byte
/// shifted to bit 20 plus `119 << 23`.
#[inline(always)]
pub fn ue4m3_half(b: u32) -> f32 {
    let sub = (b as f32) * (1.0 / 1024.0);
    let normal = f32::from_bits((b << 20) + (119 << 23));
    if b < 8 { sub } else { normal }
}

/// One FP4 format's rule over a 32-value lane unit (module doc): where the
/// unit's scale bytes sit in the `e` plane, how its codes and scales decode,
/// how the decoded unit meets one activation column, and the same on the
/// host over the file's bytes. [`Mxfp4`] and [`Nvfp4`] are the two; the lane
/// loop ([`lane_partials`]), the repack ([`repack`]) and the host rule
/// ([`lane_partial_host`]) are each one function over both.
pub trait Fp4 {
    /// The format's name in a refusal.
    const NAME: &'static str;
    /// The `what` of the repack's refusals.
    const REPACK: &'static str;
    /// Values per file block.
    const BLOCK_VALUES: usize;
    /// File bytes per block: the scale bytes, then the code bytes.
    const BLOCK_BYTES: usize;
    /// Scale bytes at the head of a block.
    const SCALE_BYTES: usize;
    /// The planes' row width is a positive multiple of this: whole `e` words
    /// a row.
    const K_STEP: usize;
    /// A unit decoded: its eight value-order int8 words and its scales.
    type Dec;

    /// Why scale byte `byte` is no scale the card rule takes, or `Ok`.
    ///
    /// # Errors
    ///
    /// The reason, for the repack's refusal.
    fn scale_ok(byte: u8) -> Result<(), String>;

    /// The `e` word holding the scale bytes of unit `g` of the stack.
    fn scale_word(g: usize) -> usize;

    /// Unit `u` of a row (the row's first unit an `e`-word boundary, which
    /// [`Fp4::K_STEP`] keeps) from its `qs` vector and its `e` word.
    fn decode(qs: [u32; 4], ew: u32, u: usize) -> Self::Dec;

    /// The decoded unit against one column's eight activation words `x`
    /// (value order) of block scale `dx`, into `f`.
    fn fma(dec: &Self::Dec, x: &[u32; 8], dx: f32, f: f32) -> f32;

    /// [`Fp4::decode`] then [`Fp4::fma`] on the host: unit `u` of a row of
    /// file bytes `row` against its 32 activation codes `xq` of scale `dx`,
    /// into `f`. A scale byte the repack refuses panics by name.
    fn unit_host(row: &[u8], u: usize, xq: &[i8], dx: f32, f: f32) -> f32;
}

/// MXFP4: a unit is a block, one E8M0 scale byte (module doc).
pub struct Mxfp4;

impl Fp4 for Mxfp4 {
    const NAME: &'static str = "MXFP4";
    const REPACK: &'static str = "mxfp4::repack";
    const BLOCK_VALUES: usize = BLOCK_VALUES;
    const BLOCK_BYTES: usize = BLOCK_BYTES;
    const SCALE_BYTES: usize = 1;
    const K_STEP: usize = 128;
    type Dec = ([u32; 8], f32);

    fn scale_ok(byte: u8) -> Result<(), String> {
        if (E_MIN..=E_MAX).contains(&byte) {
            Ok(())
        } else {
            Err(format!("{byte} is outside {E_MIN}..={E_MAX}"))
        }
    }

    #[inline(always)]
    fn scale_word(g: usize) -> usize {
        g >> 2
    }

    #[inline(always)]
    fn decode(qs: [u32; 4], ew: u32, u: usize) -> ([u32; 8], f32) {
        let dw = scale_of((ew >> (8 * (u & 3))) & 0xff);
        (decode_block(qs), dw)
    }

    #[inline(always)]
    fn fma(dec: &([u32; 8], f32), x: &[u32; 8], dx: f32, f: f32) -> f32 {
        block_fma(&dec.0, x, dec.1, dx, f)
    }

    fn unit_host(row: &[u8], u: usize, xq: &[i8], dx: f32, f: f32) -> f32 {
        let blk: &[u8; BLOCK_BYTES] = row[BLOCK_BYTES * u..][..BLOCK_BYTES]
            .try_into()
            .unwrap_or_else(|_| panic!("mxfp4 unit {u}: a 17-byte block"));
        let xq: &[i8; BLOCK_VALUES] = xq[..BLOCK_VALUES]
            .try_into()
            .unwrap_or_else(|_| panic!("mxfp4 unit {u}: 32 activation codes"));
        let dw = e8m0_to_f32_half(blk[0]);
        (dw * dx).mul_add(block_sum_host(blk, xq) as f32, f)
    }
}

/// NVFP4: a unit is half a block, its two 16-value sub-blocks, each with its
/// UE4M3 scale byte (module doc).
pub struct Nvfp4;

impl Fp4 for Nvfp4 {
    const NAME: &'static str = "NVFP4";
    const REPACK: &'static str = "nvfp4::repack";
    const BLOCK_VALUES: usize = NV_BLOCK_VALUES;
    const BLOCK_BYTES: usize = NV_BLOCK_BYTES;
    const SCALE_BYTES: usize = 4;
    const K_STEP: usize = NV_BLOCK_VALUES;
    type Dec = ([u32; 8], f32, f32);

    fn scale_ok(byte: u8) -> Result<(), String> {
        ue4m3_to_f32_half(byte)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    #[inline(always)]
    fn scale_word(g: usize) -> usize {
        g >> 1
    }

    /// The unit's code words are its sub-blocks' two words each; a word's
    /// low nibbles are its sub-block's values `0..8` (four a word), its high
    /// nibbles `8..16`. The value order is the low words, then the high
    /// words, of each sub-block in turn. Its scales are bytes `2(u & 1)` and
    /// `2(u & 1) + 1` of the block's word.
    #[inline(always)]
    fn decode(qs: [u32; 4], ew: u32, u: usize) -> ([u32; 8], f32, f32) {
        let (l0, h0) = decode_word(qs[0]);
        let (l1, h1) = decode_word(qs[1]);
        let (l2, h2) = decode_word(qs[2]);
        let (l3, h3) = decode_word(qs[3]);
        let sh = 16 * (u & 1) as u32;
        (
            [l0, l1, h0, h1, l2, l3, h2, h3],
            ue4m3_half((ew >> sh) & 0xff),
            ue4m3_half((ew >> (sh + 8)) & 0xff),
        )
    }

    /// Each sub-block's exact sum by four `dp4a`, scaled exactly by its
    /// `d_s`, then one fused multiply-add by `d_x`: the first sub-block, then
    /// the second. This order is the gate.
    #[inline(always)]
    fn fma(dec: &([u32; 8], f32, f32), x: &[u32; 8], dx: f32, f: f32) -> f32 {
        let (w, d0, d1) = dec;
        let s0 = dp4a_s32(w[0], x[0], 0);
        let s0 = dp4a_s32(w[1], x[1], s0);
        let s0 = dp4a_s32(w[2], x[2], s0);
        let s0 = dp4a_s32(w[3], x[3], s0);
        let s1 = dp4a_s32(w[4], x[4], 0);
        let s1 = dp4a_s32(w[5], x[5], s1);
        let s1 = dp4a_s32(w[6], x[6], s1);
        let s1 = dp4a_s32(w[7], x[7], s1);
        let f = (d0 * s0 as f32).mul_add(dx, f);
        (d1 * s1 as f32).mul_add(dx, f)
    }

    fn unit_host(row: &[u8], u: usize, xq: &[i8], dx: f32, f: f32) -> f32 {
        let blk = &row[NV_BLOCK_BYTES * (u / 2)..][..NV_BLOCK_BYTES];
        let mut f = f;
        for half in 0..2 {
            let sub = 2 * (u & 1) + half;
            let dw = ue4m3_to_f32_half(blk[sub])
                .unwrap_or_else(|why| panic!("nvfp4 unit {u} sub-block {sub}: {why}"));
            let mut s = 0i32;
            for (j, &byte) in blk[4 + 8 * sub..][..8].iter().enumerate() {
                let [lo, hi] = fp4_pair(byte);
                s += i32::from(lo) * i32::from(xq[16 * half + j])
                    + i32::from(hi) * i32::from(xq[16 * half + 8 + j]);
            }
            f = (dw * s as f32).mul_add(dx, f);
        }
        f
    }
}

/// Block `g` of a file-layout stream (the rows' 17-byte blocks back to back as
/// little-endian words): its scale by [`scale_half`] and its four `qs` words.
///
/// The block starts at byte `17g`, word `17g >> 2`, `17g & 3 = g & 3` bytes
/// in, and its last byte is in the fifth word from there for every `g`, so
/// five words cover it and none lies past the block's own last byte. The
/// scale is byte `g & 3` of the first word; code word `i` is the
/// little-endian pair of words `i`, `i + 1` shifted right by `8 · (g & 3) + 8`
/// ([`q8_0_code`], whose shift reaches 32 for `g & 3 = 3`).
///
/// The caller keeps `w` holding the stream's word of block `g`'s last byte,
/// `(17g + 16) >> 2` (rust-quality R28: a crate-internal fn over unchecked
/// reads under a caller-kept boundary).
#[inline(always)]
pub(crate) fn file_block(w: &[u32], g: usize) -> (f32, [u32; 4]) {
    let at = BLOCK_BYTES * g;
    let i = at >> 2;
    let o = (at & 3) as u32;
    // SAFETY: words i .. i + 4 end at (17g + 16) >> 2, inside `w` by this
    // fn's contract.
    let v = unsafe {
        [
            *w.get_unchecked(i),
            *w.get_unchecked(i + 1),
            *w.get_unchecked(i + 2),
            *w.get_unchecked(i + 3),
            *w.get_unchecked(i + 4),
        ]
    };
    let sh = 8 * o + 8;
    (
        scale_half((v[0] >> (8 * o)) & 0xff),
        [
            q8_0_code(v[0], v[1], sh),
            q8_0_code(v[1], v[2], sh),
            q8_0_code(v[2], v[3], sh),
            q8_0_code(v[3], v[4], sh),
        ],
    )
}

/// Lane `lane`'s partial of one row of a file-layout stream against one
/// q8_1 column, the module doc's rule: row `row`'s block `b` is block `row ·
/// n_blk + b` of the stream ([`file_block`]), the column's words are read as
/// in [`lane_partials`] from `q0` and its scales from `d0`.
///
/// The caller keeps `n_blk` a multiple of 4 (so a row is whole words), `w`
/// holding the stream through the last block of row `row` ([`file_block`]'s
/// boundary), `q.len() >= q0 + 256 · ceil(n_blk / 32)`, `d8.len() >= d0 +
/// n_blk / 4` and `lane < 32` (rust-quality R28).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) fn lane_partial_file(
    w: &[u32],
    row: usize,
    n_blk: usize,
    q: &[u32],
    d8: &[f32],
    q0: usize,
    d0: usize,
    lane: usize,
) -> f32 {
    let g0 = row * n_blk;
    let mut f = 0.0f32;
    let mut b = lane;
    while b < n_blk {
        // b < n_blk, so block g0 + b is inside the row, inside `w` by this
        // fn's contract.
        let (dw, qs) = file_block(w, g0 + b);
        let dec = decode_block(qs);
        let qb = q0 + 256 * (b >> 5) + (b & 31);
        // SAFETY: qb + 224 < q0 + 256·ceil(n_blk/32) for b < n_blk, so the
        // eight words qb + 32i are inside the column, and scale d0 + b/4
        // inside its scales, both by this fn's contract.
        let (x, dx) = unsafe {
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
                *d8.get_unchecked(d0 + (b >> 2)),
            )
        };
        f = block_fma(&dec, &x, dw, dx, f);
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one row of format `F` in the card layout
/// against `M` q8 columns, the module doc's rule: each unit's codes and
/// scales are loaded and decoded once and dotted with every column whose bit
/// is set in `mask`. Column `c` reads its q8 words from `q0 + c · q_col`
/// and its scales from `d0 + c · d_col`, unit `u`'s at `u >> D_SHIFT`: 2 for
/// a `Q8Act` column (one scale a 128-value block), 0 for a
/// `crate::q5::Q8Blocks32` column (one a unit); both lay a unit's eight
/// words at `256 · (u >> 5) + 32 · i + (u & 31)`. Columns `M..8` and columns
/// outside `mask` stay 0.0.
///
/// # Safety
///
/// `n_units` is the row's 32-value units, a multiple of `1 << D_SHIFT` and of
/// the units an `e` word covers; `qs` holds at least `(row + 1) · n_units`
/// vectors and `e` at least the word `F::scale_word((row + 1) · n_units −
/// 1)`; for every column `c < M`, `q.len() >= q0 + c · q_col + 256 ·
/// ceil(n_units / 32)` and `d8.len() >= d0 + c · d_col + (n_units >>
/// D_SHIFT)`; `lane < 32`; every scale byte of the row passes `F::scale_ok`
/// (the repack's check).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn lane_partials<const M: usize, F: Fp4, const D_SHIFT: usize>(
    qs: &[U32x4],
    e: &[u32],
    row: usize,
    n_units: usize,
    q: &[u32],
    d8: &[f32],
    q0: usize,
    q_col: usize,
    d0: usize,
    d_col: usize,
    mask: u32,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
        assert!(D_SHIFT == 0 || D_SHIFT == 2);
    };
    let mut f = [0.0f32; 8];
    let blk0 = row * n_units;
    let mut b = lane;
    while b < n_units {
        // SAFETY: b < n_units, so vector blk0 + b and its scale word are
        // inside the row, inside qs and e by this fn's contract.
        let (v, ew) = unsafe {
            (
                *qs.get_unchecked(blk0 + b),
                *e.get_unchecked(F::scale_word(blk0 + b)),
            )
        };
        let dec = F::decode(v.0, ew, b);
        let slot = 256 * (b >> 5) + (b & 31);
        macro_rules! col {
            ($c:literal) => {
                if M > $c && (mask >> $c) & 1 != 0 {
                    let qb = q0 + $c * q_col + slot;
                    // SAFETY: slot + 224 < 256·ceil(n_units/32) for b <
                    // n_units, so the eight words qb + 32i are inside column
                    // $c's words, and scale d0 + $c·d_col + (b >> D_SHIFT)
                    // inside its scales, both inside q and d8 by this fn's
                    // contract.
                    let (x, dx) = unsafe {
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
                            *d8.get_unchecked(d0 + $c * d_col + (b >> D_SHIFT)),
                        )
                    };
                    f[$c] = F::fma(&dec, &x, dx, f[$c]);
                }
            };
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
        b += 32;
    }
    f
}

// ------------------------------------------------------------------ host

/// The MXFP4 block sum on the host: the 32 codes of a file block
/// (`blk[1..17]`, the scale byte ignored) against 32 int8 activation codes
/// in value order.
#[must_use]
pub fn block_sum_host(blk: &[u8; BLOCK_BYTES], xq: &[i8; BLOCK_VALUES]) -> i32 {
    let mut s = 0i32;
    for (j, &byte) in blk[1..].iter().enumerate() {
        let [lo, hi] = fp4_pair(byte);
        s += i32::from(lo) * i32::from(xq[j]) + i32::from(hi) * i32::from(xq[j + 16]);
    }
    s
}

/// Lane `lane`'s partial of one row of format `F` on the host, the module
/// doc's rule ([`Fp4::unit_host`] per unit) and [`lane_partials`]' column:
/// `row` the row's file bytes, `xq` the column's int8 codes in value order
/// (the row's values), `d8` its scales, unit `u`'s at `u >> D_SHIFT`.
#[must_use]
pub fn lane_partial_host<F: Fp4, const D_SHIFT: usize>(
    row: &[u8],
    xq: &[i8],
    d8: &[f32],
    lane: usize,
) -> f32 {
    let n_units = row.len() / F::BLOCK_BYTES * (F::BLOCK_VALUES / UNIT_VALUES);
    let mut f = 0.0f32;
    let mut u = lane;
    while u < n_units {
        f = F::unit_host(row, u, &xq[UNIT_VALUES * u..], d8[u >> D_SHIFT], f);
        u += 32;
    }
    f
}

/// An FP4 tensor in the card layout (the module doc): `qs` holds `k / 8`
/// words per row (`4 · n_units`), `e` the row's scale bytes four to a word
/// (MXFP4 `k / 128`, NVFP4 `k / 64` words per row).
pub struct Planes {
    pub qs: Vec<u32>,
    pub e: Vec<u32>,
}

/// The card layout of `rows` rows of format `F` of `k` values from their
/// file bytes. `k` must be a positive multiple of `F::K_STEP` (whole scale
/// words per row) and `bytes` exactly the rows' file bytes; a scale byte
/// `F::scale_ok` refuses is refused, naming its row, block and byte.
pub fn repack<F: Fp4>(bytes: &[u8], rows: usize, k: usize) -> Result<Planes, GpuError> {
    let what = F::REPACK;
    if k == 0 || !k.is_multiple_of(F::K_STEP) {
        return Err(GpuError::shape(
            what,
            format!("k {k} is not a positive multiple of {}", F::K_STEP),
        ));
    }
    let n_blk = k / F::BLOCK_VALUES;
    if bytes.len() != rows * n_blk * F::BLOCK_BYTES {
        return Err(GpuError::shape(
            what,
            format!(
                "{} bytes for {rows} rows of {k} {} values, want {}",
                bytes.len(),
                F::NAME,
                rows * n_blk * F::BLOCK_BYTES
            ),
        ));
    }
    let code_words = (F::BLOCK_BYTES - F::SCALE_BYTES) / 4;
    let widths = [n_blk * code_words, n_blk * F::SCALE_BYTES / 4];
    let (qs, e) = pack_planes(
        what,
        bytes,
        (rows, n_blk * F::BLOCK_BYTES),
        widths,
        |first, src, qs, e| {
            for (j, blk) in src.chunks_exact(F::BLOCK_BYTES).enumerate() {
                let (scales, codes) = blk.split_at(F::SCALE_BYTES);
                for (s, &byte) in scales.iter().enumerate() {
                    F::scale_ok(byte).map_err(|why| {
                        let sub = if F::SCALE_BYTES > 1 {
                            format!(" sub-block {s}")
                        } else {
                            String::new()
                        };
                        let i = first * n_blk + j;
                        format!(
                            "row {} block {}{sub}: scale byte {why}",
                            i / n_blk,
                            i % n_blk
                        )
                    })?;
                    let at = j * F::SCALE_BYTES + s;
                    e[at / 4] |= u32::from(byte) << (8 * (at % 4));
                }
                for (w, c) in qs[j * code_words..][..code_words]
                    .iter_mut()
                    .zip(codes.as_chunks::<4>().0)
                {
                    *w = u32::from_le_bytes(*c);
                }
            }
            Ok(())
        },
    )?;
    Ok(Planes { qs, e })
}

// ------------------------------------------------------- NVFP4 dense gemv

/// Threads per block of the NVFP4 gemv: eight warps, a row each.
const NV_BLOCK: u32 = 256;
/// Rows a block computes: one per warp.
const NV_ROWS_PER_BLOCK: usize = NV_BLOCK as usize / 32;

// The entry's launch attributes spell the block as a literal.
const _: () = assert!(NV_BLOCK == 256);

#[cuda_module]
mod nvfp4_kernels {
    use super::*;

    /// `y[c · n_rows + r] = scale · Σ` row `r` of an NVFP4 tensor in the card
    /// layout against q8_1 column `c < m` (token-major): one warp per row,
    /// eight rows per block, the lanes' partials ([`lane_partials`] under
    /// [`Nvfp4`]) summed by the warp butterfly, then one multiply by the
    /// tensor's scale. Every row below `n_rows` writes every column below
    /// `m`.
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
            qs.len() >= n_rows * 32 * n_sb,
            e.len() >= n_rows * 4 * n_sb,
            q.len() >= m * 256 * iters,
            d8.len() >= m * 2 * n_sb,
            4 * iters >= n_sb,
            m >= 1,
            m <= 8,
            y.len() >= m * n_rows
        )
    )]
    pub fn nvfp4_gemv(
        qs: &[u32],
        e: &[u32],
        q: &[u32],
        d8: &[f32],
        n_rows: u32,
        m: u32,
        n_sb: u32,
        iters: u32,
        scale: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % NV_BLOCK as usize;
        let row = (thread::index_1d().get() / NV_BLOCK as usize) * NV_ROWS_PER_BLOCK + t / 32;
        let n_rows = n_rows as usize;
        if row >= n_rows {
            return;
        }
        // The launcher hands a plane of whole vectors (`k / 8` words a row,
        // a multiple of 32), so the view exists; the return is grid-uniform.
        let Some(qv) = as_vectors::<U32x4>(qs) else {
            return;
        };
        let lane = warp::lane_id() as usize;
        let (m, n_sb) = (m as usize, n_sb as usize);
        let mask = (1u32 << m) - 1;
        // SAFETY: n_units = 8·n_sb is a multiple of 4; row < n_rows, so the
        // row's 8·n_sb vectors and its 4·n_sb scale words (one a block, word
        // g >> 1 for unit g) are inside qs and e (the first two bounds); each
        // column c < m <= 8 has 256·iters >= 256·ceil(n_units/32) q words at
        // c·256·iters and 2·n_sb = n_units/4 scales at c·2·n_sb (the q, d8
        // and iters bounds); lane < 32; the repack took every scale byte.
        let f = unsafe {
            lane_partials::<8, Nvfp4, 2>(
                qv,
                e,
                row,
                8 * n_sb,
                q,
                d8,
                0,
                256 * iters as usize,
                0,
                2 * n_sb,
                mask,
                lane,
            )
        };
        macro_rules! col {
            ($c:literal) => {
                if $c < m {
                    let s = warp::reduce_sum_f32(f[$c]);
                    if lane == 0 {
                        // SAFETY: $c·n_rows + row < m·n_rows <= y.len() by
                        // the launch contract; lane 0 of the row's warp is
                        // its only writer.
                        unsafe { *y.get_unchecked_mut($c * n_rows + row) = s * scale };
                    }
                }
            };
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
    }
}

/// An NVFP4 weight resident in the card layout: the two planes of
/// [`repack`]`::<`[`Nvfp4`]`>` (`qs` `k / 8` words a row, `e` `k / 64`) and
/// the tensor's scale, read from the file at load.
#[derive(Clone, Copy)]
pub struct Nvfp4Weight<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub e: &'a DeviceTensor<u32>,
    pub scale: f32,
}

/// The loaded NVFP4 dense gemv and its launcher. Owns no stream: every
/// enqueue takes the engine stream, so launches order with the step and are
/// capturable.
pub struct Nvfp4Kernels {
    module: nvfp4_kernels::LoadedModule,
}

impl Nvfp4Kernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Nvfp4Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(nvfp4_kernels, ctx)? };
        Ok(Nvfp4Kernels { module })
    }

    /// Enqueue `y = scale · W · act` over the `act.m()` (1..=8) columns of
    /// the q8_1 rows `act`, token-major (`y[c · rows + r]`), `W` the planes
    /// of `w` with `act.k()` values a row. Another plane width, a row count
    /// the planes disagree on, more than eight columns or a short `y` is a
    /// named error. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv(
        &self,
        stream: &CudaStream,
        w: Nvfp4Weight<'_>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Nvfp4Kernels::enqueue_gemv";
        crate::tensor::check_alignment::<U32x4, _>(what, "qs", w.qs.buf())?;
        let (k, m, rows) = (act.k(), act.m(), w.qs.rows());
        if w.qs.cols() != k / 8 || w.e.cols() != k / NV_BLOCK_VALUES || w.e.rows() != rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "NVFP4 planes at K={k} are {} and {} words a row over one row count; got \
                     {rows} x {} and {} x {}",
                    k / 8,
                    k / NV_BLOCK_VALUES,
                    w.qs.cols(),
                    w.e.rows(),
                    w.e.cols()
                ),
            ));
        }
        if !(1..=8).contains(&m) || y.len() < m * rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "1 <= m <= 8 columns, got {m}; y.len() {} >= m*rows = {}",
                    y.len(),
                    m * rows
                ),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", rows.div_ceil(NV_ROWS_PER_BLOCK))?;
        let prep = self
            .module
            .prepare_nvfp4_gemv(LaunchConfig1D::new(grid, NV_BLOCK, 0))?;
        self.module.nvfp4_gemv(
            stream,
            &prep,
            w.qs.buf(),
            w.e.buf(),
            &act.q4,
            &act.d8,
            launch_u32(what, "n_rows", rows)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            w.scale,
            y,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Fp4, Mxfp4, Nvfp4, Nvfp4Kernels, Nvfp4Weight, repack, ue4m3_half};
    use crate::tensor::{DeviceTensor, Q8Act};
    use cuda_core::DeviceBuffer;
    use gguf::quant::ue4m3_to_f32_half;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn hw_nvfp4_refuses_an_unaligned_code_plane() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA stream");
        let kernels = Nvfp4Kernels::load(&ctx).expect("NVFP4 kernels");
        let parent = DeviceBuffer::from_host(&stream, &[0u32; 33]).expect("code parent");
        let qs = DeviceTensor::window_of(&parent, 4, 1, 32).expect("u32 window");
        let e = DeviceTensor::upload(&stream, &[0u32; 4], 1, 4).expect("scales");
        let act = Q8Act::with_k(&stream, 1, 256).expect("activation");
        let mut y = DeviceBuffer::from_host(&stream, &[123.0f32]).expect("output");
        let result = kernels.enqueue_gemv(
            &stream,
            Nvfp4Weight {
                qs: &qs,
                e: &e,
                scale: 1.0,
            },
            &act,
            &mut y,
        );
        let output = y.to_host_vec(&stream).expect("output read");
        println!("unaligned NVFP4: {result:?}, y={output:?}");
        let err = result
            .expect_err("unaligned codes must be refused")
            .to_string();
        assert!(err.contains("qs") && err.contains("16"), "{err}");
        assert_eq!(output, [123.0]);
        let qs = DeviceTensor::window_of(&parent, 0, 1, 32).expect("aligned window");
        kernels
            .enqueue_gemv(
                &stream,
                Nvfp4Weight {
                    qs: &qs,
                    e: &e,
                    scale: 1.0,
                },
                &act,
                &mut y,
            )
            .expect("aligned launch");
        assert_eq!(y.to_host_vec(&stream).expect("aligned output"), [0.0]);
    }

    /// The card's UE4M3 decode is the host's on every byte the repack takes,
    /// bit for bit, and the repack takes exactly the bytes the host defines.
    #[test]
    fn the_card_ue4m3_decode_is_the_hosts_on_every_defined_byte() {
        for b in 0..=u8::MAX {
            match ue4m3_to_f32_half(b) {
                Ok(want) => {
                    assert!(Nvfp4::scale_ok(b).is_ok(), "{b:#04x}");
                    assert_eq!(
                        ue4m3_half(u32::from(b)).to_bits(),
                        want.to_bits(),
                        "{b:#04x}"
                    );
                }
                Err(_) => assert!(Nvfp4::scale_ok(b).is_err(), "{b:#04x}"),
            }
        }
    }

    /// The NVFP4 repack moves the file's bytes into the planes — the four
    /// scale bytes of a block as one word, its 32 code bytes as eight — and
    /// refuses an undefined scale byte naming its row, block and sub-block.
    /// MXFP4's keeps its one byte a block and its range.
    #[test]
    fn the_repack_moves_the_bytes_and_refuses_an_undefined_scale_by_name() {
        let k = 128;
        let mut bytes: Vec<u8> = (0..2 * 2 * 36).map(|i| (i % 0x7F) as u8).collect();
        let p = repack::<Nvfp4>(&bytes, 2, k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.qs.len(), p.e.len()), (2 * k / 8, 2 * k / 64));
        for (blk, chunk) in bytes.as_chunks::<36>().0.iter().enumerate() {
            assert_eq!(p.e[blk].to_le_bytes(), chunk[..4], "block {blk}");
            for w in 0..8 {
                assert_eq!(
                    p.qs[8 * blk + w].to_le_bytes(),
                    chunk[4 + 4 * w..8 + 4 * w],
                    "block {blk} word {w}"
                );
            }
        }
        for byte in [0x7Fu8, 0x80, 0xFF] {
            bytes[3 * 36 + 2] = byte;
            match repack::<Nvfp4>(&bytes, 2, k) {
                Err(e) => {
                    let msg = e.to_string();
                    assert!(
                        msg.contains("row 1 block 1 sub-block 2") && msg.contains("UE4M3"),
                        "{msg}"
                    );
                }
                Ok(_) => panic!("scale byte {byte:#04x} repacked"),
            }
        }
        let mut mx = vec![100u8; 2 * 4 * 17];
        mx[17 * 5] = 255;
        match repack::<Mxfp4>(&mx, 2, 128) {
            Err(e) => assert!(e.to_string().contains("row 1 block 1"), "{e}"),
            Ok(_) => panic!("MXFP4 scale 255 repacked"),
        }
    }
}
