//! What the V4.1 step runs outside its sub-layers: the embedding broadcast
//! into the hyper-connection streams, the engram, the streams' collapse
//! before the head, and the head with its argmax. See [`super`] for the piece
//! contract.
//!
//! The piece, in step order, one token a step:
//! - [`Glue::enqueue_embed`], one launch at the step's start: the token's
//!   embedding row, in the step image as the file stores it (bf16, or Q3_K or
//!   Q5_K blocks), decoded to f32 into every stream and into layer 0's
//!   attention input. The first sub-layer's `pre` is one-hot on stream 0, so
//!   that fold is the row itself and does not run.
//! - [`Glue::enqueue_engram_kv_at`], three launches per engram site (four
//!   when `engram_wkv` is a K-quant), none of which reads a stream: the
//!   site's gathered rows (the table's Q8_0, Q3_K, Q6_K, IQ4_NL or MXFP8 bytes in the
//!   image) dequantized, `engram_wkv` over them ([`crate::dense`]: `q8_0_gemv`
//!   on the f32 rows, or the rows' q8_1 form and the K-quant gemv), and the key
//!   norm. They depend on the token alone and are first read by the site
//!   layer's engram step, so the step hands them ([`EngramKv`]) to the MoE
//!   sub-layer of the layer before the site, whose host-leg shadow runs them
//!   ([`ShadowWork`]).
//! - [`Glue::enqueue_engram`], two launches at each engram layer: the gate
//!   over the streams the previous ffn's HC_POST left (that ffn ends with
//!   HC_POST alone), then the fold of the gated streams that the layer's
//!   attention reads, by the previous sub-layer's `pre` (the lag).
//! - [`Glue::enqueue_head`], after the last layer: the fold of the last
//!   ffn's streams by that ffn's `pre` (`hc_out`), written into the head's
//!   input, then the head's own launches.
//!
//! [`StepRows`] is the host half: the rows a step's image carries — each
//! token's embedding row from the file, and each site's rows, hashed from the
//! token's n-gram by the engram crate and read out of the site's table by a
//! helper thread ([`RowsLevers::helper`]), so the calling thread never touches
//! the table's mapping.
//!
//! Numbers: the bf16 widening and a Q8_0 or IQ4_NL row's dequantization (an
//! f16 scale times an 8-bit code or table value) are exact; a Q3_K or Q5_K
//! value is `gguf::quant::dequant_row`'s bit for bit
//! (`elem::EmbedSb::value`, its roundings in its order); everything else is
//! an op this piece composes and does not change.
//!
//! The resident weights come in at each enqueue, not at [`Glue::new`]: the
//! body owns the pieces and its model owns the weights beside it, so a piece
//! that kept a borrow of them would borrow its owner's sibling. Every name is
//! resolved at `new`; an enqueue looks the names up (at capture, once per
//! graph) and allocates nothing.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu::elem::{
    EmbedSb, Q3kRows, Q5kRows, iq4nl_embed_value, mxfp8_embed_value, q3k_embed_value,
};
use bloomery_gpu::fault::FaultSink;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::rank::Sums;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Gpu, GpuError, Q8Act, StagedRows, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{cvt_f32_f16x2_hi, cvt_f32_f16x2_lo, cvt_f32x2_bf16x2};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use engram::hash::Hash;
use engram::prefetch::{FillMode, HelperOptions, Prefetcher, caller_sibling};
use engram::rows::{PageAdvice, PlanarTable};
use engram::{Engram, Faults, RowTable, RowTables};
use gguf::Split;
use gguf::engram::EngramLayout;
use gguf::quant::GgmlType;
use model::arch::deepseek41::hparams::{EngramGains, Hparams};
use model::arch::deepseek41::names;
use model::arch::deepseek41::plan::StepPlan;
use model::embed::RowLayout;
use model::part::StoredRows;

use crate::chain::ffn::ShadowWork;
use crate::chain::join;
use crate::dense::{Dense, DenseKernels};
use crate::engram_gate::{EngramGateKernels, GateArgs, KeyNormArgs, ROW};
use crate::hc::{HC_MIX, HC_STREAMS, HcKernels};
use crate::params::ImageLayout;
use crate::span::{SpanMut, span_mut};

mod batch;
pub use batch::{GlueBatch, PromptRows};

const WHAT: &str = "deepseek41 Glue";

/// Threads of every glue launch.
const THREADS: u32 = 256;

/// Values of one Q8_0 block, and its bytes: an f16 scale, then the codes.
const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 34;

/// Values of one Q3_K super-block, and its bytes.
const Q3_K_BLOCK: usize = 256;
const Q3_K_BYTES: usize = Q3kRows::BYTES;

/// Values of one Q5_K super-block.
const Q5_K_BLOCK: usize = 256;

/// Values of one IQ4_NL block, and its bytes: an f16 scale, then 16 bytes of
/// two codes each.
const IQ4_NL_BLOCK: usize = 32;
const IQ4_NL_BYTES: usize = 18;

/// Values of one MXFP8 block, and its bytes: an E8M0 scale, then one F8_E4M3
/// byte a value.
const MXFP8_BLOCK: usize = 32;
const MXFP8_BYTES: usize = 33;

/// Values of one Q6_K super-block, and its bytes: 128 bytes of low nibbles,
/// 64 of high bit pairs, 16 signed scales, then the f16 `d`.
const Q6_K_BLOCK: usize = 256;
const Q6_K_BYTES: usize = 210;

/// How a token's embedding row sits in the image: `token_embd`'s type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbdRows {
    /// Two bf16 to a word, the low half first.
    Bf16,
    /// Q3_K super-blocks, 110 bytes each, from the row's first word; the
    /// last word of a row with an odd super-block count is zero-padded.
    Q3K,
    /// Q5_K super-blocks, 176 bytes (whole words) each, from the row's first
    /// word.
    Q5K,
}

impl EmbdRows {
    /// The row format of a `token_embd` of type `ty` with rows of `n_embd`
    /// values; `None` for any other type, and for a K-quant row that is not
    /// whole super-blocks.
    #[must_use]
    pub fn of(ty: GgmlType, n_embd: usize) -> Option<EmbdRows> {
        match ty {
            GgmlType::BF16 => Some(EmbdRows::Bf16),
            GgmlType::Q3_K if n_embd.is_multiple_of(Q3_K_BLOCK) => Some(EmbdRows::Q3K),
            GgmlType::Q5_K if n_embd.is_multiple_of(Q5_K_BLOCK) => Some(EmbdRows::Q5K),
            _ => None,
        }
    }

    /// The file type these rows are.
    #[must_use]
    pub fn ty(self) -> GgmlType {
        match self {
            EmbdRows::Bf16 => GgmlType::BF16,
            EmbdRows::Q3K => GgmlType::Q3_K,
            EmbdRows::Q5K => GgmlType::Q5_K,
        }
    }

    /// The file bytes of one row of `n_embd` values.
    #[must_use]
    pub fn row_bytes(self, n_embd: usize) -> usize {
        let ty = self.ty();
        match (ty.blck_size(), ty.type_size()) {
            (Some(blck), Some(size)) => size as usize * (n_embd / blck as usize),
            _ => unreachable!("bf16, Q3_K and Q5_K are sized types"),
        }
    }

    /// The u32 words one row takes in a buffer or an image.
    #[must_use]
    pub fn row_words(self, n_embd: usize) -> usize {
        self.row_bytes(n_embd).div_ceil(4)
    }
}

/// How a site's gathered rows sit in the image: the engram tables' type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableRows {
    /// Q8_0 rows in whole words.
    Q8_0,
    /// Q3_K rows of whole super-blocks, packed at 110 bytes each: every
    /// other row starts on a half word.
    Q3K,
    /// IQ4_NL rows of whole blocks, packed at 18 bytes each: every other
    /// block starts on a half word.
    Iq4Nl,
    /// MXFP8 rows of whole blocks, packed at 33 bytes each, read a byte at a
    /// time: a block starts at any byte.
    Mxfp8,
    /// Q6_K rows of whole super-blocks, packed at 210 bytes each: every other
    /// row starts on a half word.
    Q6K,
}

impl TableRows {
    /// The format of a table of type `ty` with rows of `key_len` values, and
    /// the bytes of a row.
    fn of(ty: GgmlType, key_len: usize) -> Option<(TableRows, usize)> {
        match ty {
            GgmlType::Q8_0 if key_len.is_multiple_of(Q8_0_BLOCK) => {
                Some((TableRows::Q8_0, key_len / Q8_0_BLOCK * Q8_0_BYTES))
            }
            GgmlType::Q3_K if key_len.is_multiple_of(Q3_K_BLOCK) => {
                Some((TableRows::Q3K, key_len / Q3_K_BLOCK * Q3_K_BYTES))
            }
            GgmlType::IQ4_NL if key_len.is_multiple_of(IQ4_NL_BLOCK) => {
                Some((TableRows::Iq4Nl, key_len / IQ4_NL_BLOCK * IQ4_NL_BYTES))
            }
            GgmlType::MXFP8 if key_len.is_multiple_of(MXFP8_BLOCK) => {
                Some((TableRows::Mxfp8, key_len / MXFP8_BLOCK * MXFP8_BYTES))
            }
            GgmlType::Q6_K if key_len.is_multiple_of(Q6_K_BLOCK) => {
                Some((TableRows::Q6K, key_len / Q6_K_BLOCK * Q6_K_BYTES))
            }
            _ => None,
        }
    }

    /// Whether the format's blocks start on a half word, not a word: the
    /// kernels read a block's f16 scale from one half of its covering word.
    fn on_half_words(self) -> bool {
        matches!(self, TableRows::Q3K | TableRows::Iq4Nl | TableRows::Q6K)
    }

    /// Enqueue the dequantization of `rows` gathered rows of `key_len`
    /// values (`row_bytes` bytes each, from image byte `at_byte`) of `params`
    /// into `y`: one launch, a thread a value. An MXFP8 row's undefined byte
    /// raises into `fault` (`mxfp8_embed_value`).
    fn enqueue(
        self,
        module: &glue_kernels::LoadedModule,
        stream: &CudaStream,
        shape: TableShape,
        (params, at_byte): (&DeviceBuffer<u32>, u32),
        fault: FaultSink,
        y: &mut SpanMut<'_, f32>,
    ) -> Result<(), GpuError> {
        let TableShape {
            rows,
            key_len,
            row_bytes,
        } = shape;
        let grid = (rows * key_len).div_ceil(THREADS);
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        match self {
            TableRows::Q8_0 => {
                let prep = module.prepare_ds41_glue_engram_rows(cfg)?;
                module.ds41_glue_engram_rows(
                    stream,
                    &prep,
                    params,
                    at_byte / 4,
                    rows,
                    row_bytes / 4,
                    key_len,
                    y,
                )?;
            }
            TableRows::Q3K => {
                let prep = module.prepare_ds41_glue_engram_rows_q3k(cfg)?;
                module.ds41_glue_engram_rows_q3k(
                    stream,
                    &prep,
                    params,
                    at_byte,
                    rows,
                    key_len / Q3_K_BLOCK as u32,
                    y,
                )?;
            }
            TableRows::Iq4Nl => {
                let prep = module.prepare_ds41_glue_engram_rows_iq4nl(cfg)?;
                module.ds41_glue_engram_rows_iq4nl(
                    stream,
                    &prep,
                    params,
                    at_byte,
                    rows,
                    key_len / IQ4_NL_BLOCK as u32,
                    y,
                )?;
            }
            TableRows::Q6K => {
                let prep = module.prepare_ds41_glue_engram_rows_q6k(cfg)?;
                module.ds41_glue_engram_rows_q6k(
                    stream,
                    &prep,
                    params,
                    at_byte,
                    rows,
                    key_len / Q6_K_BLOCK as u32,
                    y,
                )?;
            }
            TableRows::Mxfp8 => {
                let prep = module.prepare_ds41_glue_engram_rows_mxfp8(cfg)?;
                module.ds41_glue_engram_rows_mxfp8(
                    stream,
                    &prep,
                    params,
                    at_byte,
                    rows,
                    key_len / MXFP8_BLOCK as u32,
                    fault,
                    y,
                )?;
            }
        }
        Ok(())
    }
}

/// The gather a site makes per token: rows, values a row, bytes a row.
#[derive(Clone, Copy)]
struct TableShape {
    rows: u32,
    key_len: u32,
    row_bytes: u32,
}

/// Enqueue the embedding broadcast of the row at word `at` of `params`, `n`
/// values in `rows`' format, into every one of the four streams of `streams`
/// (`[s][n]`) and into `input`: one launch, owned here for the step, the
/// prompt batch and the draft alike.
pub(crate) fn enqueue_embed_row(
    module: &glue_kernels::LoadedModule,
    stream: &CudaStream,
    rows: EmbdRows,
    (params, at): (&DeviceBuffer<u32>, u32),
    n: usize,
    (streams, input): (&mut SpanMut<'_, f32>, &mut SpanMut<'_, f32>),
) -> Result<(), GpuError> {
    const WHAT: &str = "enqueue_embed_row";
    let hc = HC_STREAMS as u32;
    let grid = launch_u32(WHAT, "grid", n.div_ceil(THREADS as usize))?;
    match rows {
        EmbdRows::Bf16 => {
            let half = launch_u32(WHAT, "half a row", n / 2)?;
            let cfg = LaunchConfig1D::new(half.div_ceil(THREADS), THREADS, 0);
            let prep = module.prepare_ds41_glue_embed(cfg)?;
            module.ds41_glue_embed(stream, &prep, params, at, half, hc, streams, input)?;
        }
        EmbdRows::Q3K => {
            let n_sb = launch_u32(WHAT, "super-blocks", n / Q3_K_BLOCK)?;
            let prep = module.prepare_ds41_glue_embed_q3k(LaunchConfig1D::new(grid, THREADS, 0))?;
            module.ds41_glue_embed_q3k(stream, &prep, params, at, n_sb, hc, streams, input)?;
        }
        EmbdRows::Q5K => {
            let n_sb = launch_u32(WHAT, "super-blocks", n / Q5_K_BLOCK)?;
            let prep = module.prepare_ds41_glue_embed_q5k(LaunchConfig1D::new(grid, THREADS, 0))?;
            module.ds41_glue_embed_q5k(stream, &prep, params, at, n_sb, hc, streams, input)?;
        }
    }
    Ok(())
}

/// Byte `x` of the image `w`, from its covering word.
///
/// # Safety
///
/// `x < 4 * w.len()`.
#[inline(always)]
unsafe fn image_byte(w: &[u32], x: usize) -> u32 {
    // SAFETY: x < 4·w.len() by this fn's `# Safety`, so word x / 4 is inside w.
    (unsafe { *w.get_unchecked(x >> 2) } >> (8 * (x & 3))) & 0xff
}

/// The f32 value of Q6_K weight `v` (0..256) of the super-block at byte
/// `base` of the image `w`, in `gguf::quant::dequant_row`'s op order (ggml's
/// `dequantize_row_q6_K`): half `h = v / 128`, quarter `q = (v % 128) / 32`,
/// lane `l = v % 32`; the 6-bit code is the nibble `q / 2` of ql byte `64h +
/// 32(q % 2) + l` plus the bit pair `q` of qh byte `128 + 32h + l` shifted up
/// four, the signed scale is byte `192 + 8h + l / 16 + 2q`, `d` the f16 that
/// closes the block (bytes 208..209), and the value is `(d · scale) ·
/// (code − 32)` — two plain products, bit-identical to the host's. A
/// super-block starts on an even byte (210 is even), so `d` is one half of
/// a word.
///
/// # Safety
///
/// `base + 210 <= 4 * w.len()`, `base` even, and `v < 256`: every byte read
/// is below `base + 210`.
#[inline(always)]
unsafe fn q6k_row_value(w: &[u32], base: usize, v: usize) -> f32 {
    let (h, q, l) = (v >> 7, (v >> 5) & 3, v & 31);
    // SAFETY: the ql byte is below base + 128, the qh byte below base + 192
    // and the scale byte below base + 208, inside the super-block by this
    // fn's `# Safety`.
    let (ql, qh, sc) = unsafe {
        (
            image_byte(w, base + 64 * h + 32 * (q & 1) + l),
            image_byte(w, base + 128 + 32 * h + l),
            image_byte(w, base + 192 + 8 * h + l / 16 + 2 * q),
        )
    };
    let code = ((ql >> (4 * (q >> 1))) & 0xf) | (((qh >> (2 * q)) & 3) << 4);
    let x = base + 208;
    // SAFETY: x + 2 <= base + 210 <= 4·w.len() by this fn's `# Safety`.
    let dw = unsafe { *w.get_unchecked(x >> 2) };
    let d = if x.is_multiple_of(4) {
        cvt_f32_f16x2_lo(dw)
    } else {
        cvt_f32_f16x2_hi(dw)
    };
    (d * f32::from(sc as u8 as i8)) * (code as i32 - 32) as f32
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
pub(crate) mod glue_kernels {
    use super::*;

    /// The embedding broadcast: thread `i` widens image word `at + i` — two
    /// bf16, the low half first — into values `2i` and `2i + 1` of each of
    /// the `hc` streams (`[s][2·half]`) and of `input`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            params.len() >= at + half,
            streams.len() >= hc * 2 * half,
            input.len() >= 2 * half
        )
    )]
    pub fn ds41_glue_embed(
        params: &[u32],
        at: u32,
        half: u32,
        hc: u32,
        mut streams: DisjointSlice<f32>,
        mut input: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let half = half as usize;
        if i >= half {
            return;
        }
        // SAFETY: i < half, so at + i < at + half <= params.len() by the
        // launch contract.
        let word = unsafe { *params.get_unchecked(at as usize + i) };
        let (lo, hi) = cvt_f32x2_bf16x2(word);
        let n = 2 * half;
        for s in 0..hc as usize {
            let b = s * n + 2 * i;
            // SAFETY: s < hc and 2i + 1 < n, so b + 1 < hc·n <= streams.len()
            // by the launch contract; the two values are this thread's alone.
            unsafe {
                *streams.get_unchecked_mut(b) = lo;
                *streams.get_unchecked_mut(b + 1) = hi;
            }
        }
        // SAFETY: 2i + 1 < n <= input.len() by the launch contract; the two
        // values are this thread's alone.
        unsafe {
            *input.get_unchecked_mut(2 * i) = lo;
            *input.get_unchecked_mut(2 * i + 1) = hi;
        }
    }

    /// One site's gathered rows, dequantized: thread `i` is value `v = i %
    /// key_len` of row `r = i / key_len`, code `v % 32` of block `v / 32`.
    /// The row's bytes start at byte `4·(at + r·row_words)` of the image; a
    /// block is its f16 scale, then 32 codes. `y[i] = scale · code`, exact.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            params.len() >= at + rows * row_words,
            y.len() >= rows * key_len
        )
    )]
    pub fn ds41_glue_engram_rows(
        params: &[u32],
        at: u32,
        rows: u32,
        row_words: u32,
        key_len: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let key_len = key_len as usize;
        if i >= rows as usize * key_len {
            return;
        }
        let (r, v) = (i / key_len, i % key_len);
        let base = at as usize + r * row_words as usize;
        let block = Q8_0_BYTES * (v / Q8_0_BLOCK);
        let code = block + 2 + v % Q8_0_BLOCK;
        // SAFETY: `Glue::new` sets 4·row_words = key_len/32·34, the bytes of
        // a row, so byte `code` of row r lies in words base .. base +
        // row_words, and r < rows gives base + row_words <= at +
        // rows·row_words <= params.len() by the launch contract.
        let (dw, cw) = unsafe {
            (
                *params.get_unchecked(base + block / 4),
                *params.get_unchecked(base + code / 4),
            )
        };
        // A block starts on an even byte: its scale is one half of a word.
        let d = if block.is_multiple_of(4) {
            cvt_f32_f16x2_lo(dw)
        } else {
            cvt_f32_f16x2_hi(dw)
        };
        let q = (cw >> (8 * (code % 4))) as u8 as i8;
        // SAFETY: i < rows·key_len <= y.len() by the launch contract; one
        // thread per value.
        unsafe {
            *y.get_unchecked_mut(i) = d * f32::from(q);
        }
    }

    /// The embedding broadcast of a Q3_K row: thread `i` decodes value `i`
    /// (super-block `i / 256` of the row at image word `at`, value `i %
    /// 256`, `q3k_embed_value`) into value `i` of each of the `hc` streams
    /// (`[s][256·n_sb]`) and of `input`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * params.len() >= 4 * at + 110 * n_sb,
            streams.len() >= hc * 256 * n_sb,
            input.len() >= 256 * n_sb
        )
    )]
    pub fn ds41_glue_embed_q3k(
        params: &[u32],
        at: u32,
        n_sb: u32,
        hc: u32,
        mut streams: DisjointSlice<f32>,
        mut input: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n = 256 * n_sb as usize;
        if i >= n {
            return;
        }
        // The super-block starts 0 or 2 mod 4 (a word, plus an even byte
        // count), the two alignments the core funnels.
        // SAFETY: i < n (the guard above) gives i / 256 < n_sb, so the
        // super-block ends at byte 4·at + 110·(i/256 + 1) <= 4·at + 110·n_sb
        // <= 4·params.len() (the launch contract); i % 256 < 256.
        let v = unsafe { q3k_embed_value(params, 4 * at as usize + 110 * (i / 256), i % 256) };
        for s in 0..hc as usize {
            // SAFETY: s < hc and i < n, so s·n + i < hc·n <= streams.len() by
            // the launch contract; the value is this thread's alone.
            unsafe { *streams.get_unchecked_mut(s * n + i) = v };
        }
        // SAFETY: i < n <= input.len() by the launch contract; one thread per
        // value.
        unsafe { *input.get_unchecked_mut(i) = v };
    }

    /// The embedding broadcast of a Q5_K row, as `ds41_glue_embed_q3k` for
    /// Q5_K super-blocks: thread `i` decodes value `i` (super-block `i / 256`
    /// of the row at image word `at`, value `i % 256`, `Q5kRows`) into value
    /// `i` of each of the `hc` streams (`[s][256·n_sb]`) and of `input`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * params.len() >= 4 * at + 176 * n_sb,
            streams.len() >= hc * 256 * n_sb,
            input.len() >= 256 * n_sb
        )
    )]
    pub fn ds41_glue_embed_q5k(
        params: &[u32],
        at: u32,
        n_sb: u32,
        hc: u32,
        mut streams: DisjointSlice<f32>,
        mut input: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n = 256 * n_sb as usize;
        if i >= n {
            return;
        }
        // The super-block is whole words, so it starts on a word.
        // SAFETY: i < n (the guard above) gives i / 256 < n_sb, so the
        // super-block ends at byte 4·at + 176·(i/256 + 1) <= 4·at + 176·n_sb
        // <= 4·params.len() (the launch contract); i % 256 < 256.
        let v = unsafe { Q5kRows::value(params, 4 * at as usize + 176 * (i / 256), i % 256) };
        for s in 0..hc as usize {
            // SAFETY: s < hc and i < n, so s·n + i < hc·n <= streams.len() by
            // the launch contract; the value is this thread's alone.
            unsafe { *streams.get_unchecked_mut(s * n + i) = v };
        }
        // SAFETY: i < n <= input.len() by the launch contract; one thread per
        // value.
        unsafe { *input.get_unchecked_mut(i) = v };
    }

    /// One site's gathered Q3_K rows, dequantized: thread `i` is value `v =
    /// i % (256·n_sb)` of row `r = i / (256·n_sb)`, super-block `v / 256` of
    /// the rows packed from image byte `at_byte` at `110·n_sb` bytes each
    /// (`q3k_embed_value`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * params.len() >= at_byte + rows * 110 * n_sb,
            y.len() >= rows * 256 * n_sb
        )
    )]
    pub fn ds41_glue_engram_rows_q3k(
        params: &[u32],
        at_byte: u32,
        rows: u32,
        n_sb: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n_sb = n_sb as usize;
        if i >= rows as usize * 256 * n_sb {
            return;
        }
        let sb = i / 256;
        // The super-block starts 0 or 2 mod 4 (`Glue::new` checks at_byte
        // even; 110 is even), the two alignments the core funnels.
        // SAFETY: i < rows·256·n_sb (the guard above) gives sb < rows·n_sb, so
        // the super-block ends at byte at_byte + 110·(sb + 1) <= at_byte +
        // rows·110·n_sb <= 4·params.len() (the launch contract); i % 256 < 256.
        let v = unsafe { q3k_embed_value(params, at_byte as usize + 110 * sb, i % 256) };
        // SAFETY: i < rows·256·n_sb <= y.len() by the launch contract; one
        // thread per value.
        unsafe { *y.get_unchecked_mut(i) = v };
    }

    /// One site's gathered Q6_K rows, dequantized: thread `i` is value `v =
    /// i % (256·n_sb)` of row `r = i / (256·n_sb)`, super-block `v / 256` of
    /// the rows packed from image byte `at_byte` at `210·n_sb` bytes each
    /// ([`q6k_row_value`]).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * params.len() >= at_byte + rows * 210 * n_sb,
            y.len() >= rows * 256 * n_sb
        )
    )]
    pub fn ds41_glue_engram_rows_q6k(
        params: &[u32],
        at_byte: u32,
        rows: u32,
        n_sb: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n_sb = n_sb as usize;
        if i >= rows as usize * 256 * n_sb {
            return;
        }
        let sb = i / 256;
        // SAFETY: i < rows·256·n_sb (the guard above) gives sb < rows·n_sb, so
        // the super-block ends at byte at_byte + 210·(sb + 1) <= at_byte +
        // rows·210·n_sb <= 4·params.len() (the launch contract); it starts on
        // an even byte (`Glue::new` checks at_byte even; 210 is even); i % 256
        // < 256.
        let v = unsafe { q6k_row_value(params, at_byte as usize + 210 * sb, i % 256) };
        // SAFETY: i < rows·256·n_sb <= y.len() by the launch contract; one
        // thread per value.
        unsafe { *y.get_unchecked_mut(i) = v };
    }

    /// One site's gathered IQ4_NL rows, dequantized: thread `i` is value `v =
    /// i % (32·n_blk)` of row `r = i / (32·n_blk)`, value `v % 32` of block
    /// `v / 32` of the rows packed from image byte `at_byte` at `18·n_blk`
    /// bytes each (`iq4nl_embed_value`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * params.len() >= at_byte + rows * 18 * n_blk,
            y.len() >= rows * 32 * n_blk
        )
    )]
    pub fn ds41_glue_engram_rows_iq4nl(
        params: &[u32],
        at_byte: u32,
        rows: u32,
        n_blk: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n_blk = n_blk as usize;
        if i >= rows as usize * 32 * n_blk {
            return;
        }
        let blk = i / 32;
        // The block starts on an even byte (`Glue::new` checks at_byte even;
        // 18 is even).
        // SAFETY: i < rows·32·n_blk (the guard above) gives blk < rows·n_blk,
        // so the block ends at byte at_byte + 18·(blk + 1) <= at_byte +
        // rows·18·n_blk <= 4·params.len() (the launch contract); i % 32 < 32.
        let v = unsafe { iq4nl_embed_value(params, at_byte as usize + 18 * blk, i % 32) };
        // SAFETY: i < rows·32·n_blk <= y.len() by the launch contract; one
        // thread per value.
        unsafe { *y.get_unchecked_mut(i) = v };
    }

    /// One site's gathered MXFP8 rows, dequantized: thread `i` is value `v =
    /// i % (32·n_blk)` of row `r = i / (32·n_blk)`, value `v % 32` of block
    /// `v / 32` of the rows packed from image byte `at_byte` at `33·n_blk`
    /// bytes each (`mxfp8_embed_value`, which raises an undefined byte into
    /// `fault`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * params.len() >= at_byte + rows * 33 * n_blk,
            y.len() >= rows * 32 * n_blk
        )
    )]
    pub fn ds41_glue_engram_rows_mxfp8(
        params: &[u32],
        at_byte: u32,
        rows: u32,
        n_blk: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n_blk = n_blk as usize;
        if i >= rows as usize * 32 * n_blk {
            return;
        }
        let blk = i / 32;
        // SAFETY: i < rows·32·n_blk (the guard above) gives blk < rows·n_blk,
        // so the block ends at byte at_byte + 33·(blk + 1) <= at_byte +
        // rows·33·n_blk <= 4·params.len() (the launch contract); i % 32 < 32.
        let v = unsafe { mxfp8_embed_value(params, at_byte as usize + 33 * blk, i % 32, fault) };
        // SAFETY: i < rows·32·n_blk <= y.len() by the launch contract; one
        // thread per value.
        unsafe { *y.get_unchecked_mut(i) = v };
    }
}

// -------------------------------------------------------------- the piece

/// A gain an engram launch reads: a resident f32 weight of `hc` rows, or the
/// piece's rows of ones ([`Glue::with_rows`] holds them for a file whose
/// gains are folded into the query's: `v·1` is `v`, so the key side then
/// rounds once, the norm's product alone).
enum Gain {
    Weight(String),
    Ones,
}

/// One engram site: its layer, its weights' names, where its rows lie in the
/// image, and per row the scratch its launches leave for the layer's gate.
struct EngramSite {
    layer: usize,
    wkv: String,
    gain_k: Gain,
    gain_q: Gain,
    /// Image byte of the site's first row.
    at_byte: u32,
    /// Per row ([`Glue::with_rows`]): a pass whose rows run one layer apart
    /// enqueues the other row's token-only work between a row's and its
    /// gate.
    bufs: Vec<SiteRow>,
}

/// One row's scratch at a site.
struct SiteRow {
    /// The rows dequantized: the projection's input.
    x: DeviceBuffer<f32>,
    /// `x` in q8_1, for a K-quant `engram_wkv`.
    act: Q8Act,
    /// `engram_wkv`'s output: `hc` keys, then the value.
    kv: DeviceBuffer<f32>,
    /// The normalized keys.
    kn: DeviceBuffer<f32>,
    /// The gates, one per stream.
    gate: DeviceBuffer<f32>,
}

impl SiteRow {
    fn device_bytes(&self) -> usize {
        self.x.num_bytes()
            + q8act_bytes(self.act.m(), self.x.len())
            + self.kv.num_bytes()
            + self.kn.num_bytes()
            + self.gate.num_bytes()
    }
}

/// Device bytes of a [`Q8Act`] of `m` columns of `k`: its allocation's
/// formula (`Q8Act::with_k`), which exposes no size of its own.
fn q8act_bytes(m: usize, k: usize) -> usize {
    let n_sb = k / 256;
    m * (64 * n_sb.div_ceil(2) * 8
        + 256 * n_sb.div_ceil(4) * 4
        + 128 * n_sb.div_ceil(2) * 4
        + 8 * n_sb * 4
        + 2 * n_sb * 4)
}

/// A site's scratch as the gate reads it back.
pub struct SiteBuffers<'a> {
    /// The rows dequantized from the image.
    pub rows: &'a DeviceBuffer<f32>,
    /// `engram_wkv`'s output.
    pub kv: &'a DeviceBuffer<f32>,
    /// The gates of the last engram step at the site.
    pub gate: &'a DeviceBuffer<f32>,
}

/// [`Glue::enqueue_engram`]'s buffers.
pub struct EngramStep<'a> {
    /// The streams the previous ffn's HC_POST left.
    pub streams: &'a DeviceBuffer<f32>,
    /// The previous sub-layer's HC_PRE result ([`HC_MIX`] values): its `pre`
    /// folds the gated streams.
    pub pre: &'a DeviceBuffer<f32>,
    /// The gated streams.
    pub out: &'a mut DeviceBuffer<f32>,
    /// The layer's attention input: the fold of `out`.
    pub input: &'a mut DeviceBuffer<f32>,
}

/// The token-only work of the engram site at `layer`
/// ([`Glue::enqueue_engram_kv_at`]) as shadow work: what the step hands the
/// MoE sub-layer of the layer before the site. On the step's first engram
/// site it carries the rows' arrival, which it enqueues first.
pub struct EngramKv<'a> {
    pub glue: &'a mut Glue,
    /// The resident weights.
    pub w: &'a Weights,
    /// The step image's device copy, whose rows section the arrival writes.
    pub params: &'a mut DeviceBuffer<u32>,
    /// The site's layer.
    pub layer: usize,
    /// The row whose scratch it writes.
    pub row: usize,
    /// The row's engram rows arrive here: the first reader of any of them.
    pub arrive: Option<&'a RowsArrival>,
    /// The rank's side of the ranks' sums, on a rank of a tensor-parallel load.
    pub join: Option<&'a mut Sums>,
}

impl ShadowWork for EngramKv<'_> {
    fn enqueue(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if let Some(arrive) = self.arrive {
            arrive.enqueue(gpu.stream(), self.row, self.params)?;
        }
        self.glue.enqueue_engram_kv_of(
            gpu,
            self.w,
            self.params,
            (self.layer, self.row),
            self.join.as_deref_mut(),
        )
    }
}

/// How a step's engram rows reach the card after its launch: per row of
/// the pass, the rows section of that row's image staged for its arrival
/// ([`StagedRows`]: a page-locked staging and a host flag).
///
/// The host clears row `r`'s flag before the launch ([`RowsArrival::clear`]),
/// writes the rows into the staging once they are read and raises the flag
/// ([`RowsArrival::deliver`]). The chain's first reader of the rows
/// ([`EngramKv`] of the step's first site) first enqueues the arrival
/// ([`RowsArrival::enqueue`]): the wait on the flag, then one host-to-device
/// copy of the staging over the rows section of the row's image copy. So
/// the card computes up to the first site's token-only work while the host
/// reads the rows, and never reads a row before the host wrote it.
///
/// Contract: [`StagedRows`]'s. The host raises a row's flag on every path
/// after the launch, an error included; the card then runs that step on the
/// rows the staging held before, and the caller condemns the step (the body
/// refuses the next one until a reset). A row's staging is written only
/// while its flag is lowered, after the refresh that lowered it
/// synchronized the stream.
pub struct RowsArrival {
    rows: StagedRows<u32>,
    /// The rows section's first word in the image, and the bytes of rows it
    /// carries.
    at: usize,
    bytes: usize,
}

impl RowsArrival {
    /// The arrival of `rows` rows of images laid out as `layout`, in `ctx`.
    /// Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        rows: usize,
        layout: &ImageLayout,
    ) -> Result<RowsArrival, GpuError> {
        let dims = layout.dims();
        let at = layout.engram_at(0);
        let words = layout.words() - at;
        let bytes = dims.tokens * dims.engram_bytes;
        if words * 4 < bytes {
            return Err(GpuError::Shape {
                what: "RowsArrival::new",
                detail: format!(
                    "a rows section of {words} words from word {at} for {} tokens of {} engram bytes",
                    dims.tokens, dims.engram_bytes
                ),
            });
        }
        Ok(RowsArrival {
            rows: StagedRows::new(
                ctx,
                rows,
                words,
                "cuMemAllocHost (the engram rows' staging)",
            )?,
            at,
            bytes,
        })
    }

    /// Lower row `row`'s flag, before the launch whose arrival it gates.
    pub fn clear(&self, row: usize) -> Result<(), GpuError> {
        self.rows.clear(row)
    }

    /// Write `rows`, the rows section's bytes as [`StepRows::engram`] holds
    /// them, into row `row`'s staging, four bytes to a little-endian word,
    /// then raise its flag. The flag is raised on every path: a `rows` of
    /// the wrong length leaves the staging as it was and returns the error.
    pub fn deliver(&mut self, row: usize, rows: &[u8]) -> Result<(), GpuError> {
        let bytes = self.bytes;
        self.rows.deliver(row, |dst| {
            if rows.len() != bytes {
                return Err(GpuError::Shape {
                    what: "RowsArrival::deliver",
                    detail: format!(
                        "{} bytes into row {row}'s staging: a row carries {bytes}",
                        rows.len()
                    ),
                });
            }
            for (w, b) in dst.iter_mut().zip(rows.chunks(4)) {
                let mut le = [0u8; 4];
                le[..b.len()].copy_from_slice(b);
                *w = u32::from_le_bytes(le);
            }
            Ok(())
        })
    }

    /// Enqueue row `row`'s arrival on `stream`: the wait on its flag, then the
    /// staging copied over the rows section of `params`, the row's image
    /// copy ([`StagedRows::enqueue`]). Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        row: usize,
        params: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.rows.enqueue(stream, row, params, self.at)
    }
}

/// The glue piece: see the module comment.
pub struct Glue {
    module: glue_kernels::LoadedModule,
    hc: HcKernels,
    engram: EngramGateKernels,
    dense: DenseKernels,
    n_embd: usize,
    eps: f32,
    /// Image word of the embedding row, the words it takes, and its format.
    embd_at: u32,
    embd_words: u32,
    embd: EmbdRows,
    /// Rows a site gathers, their values, their bytes and their format.
    shape: TableShape,
    table: TableRows,
    sites: Vec<EngramSite>,
    /// `hc` rows of ones, the key gain of a file whose gains are folded
    /// ([`EngramGains::Folded`]); `None` otherwise.
    ones: Option<DeviceBuffer<f32>>,
    /// Whether a site's rows are a rank's columns of them: its `engram_wkv` output is then the
    /// rank's part of a sum over the ranks, whose key norm waits for the sum.
    split: bool,
}

impl Glue {
    /// The piece for the model `hp` describes, reading the step image laid
    /// out by `layout`: its launch geometry, every name its launches read,
    /// and each engram site's scratch. The kernels hold a stream row of
    /// [`ROW`] values and four streams; a step runs one token, and the image
    /// carries the embedding row as bf16, Q3_K or Q5_K and each site's rows as
    /// Q8_0, Q3_K, Q6_K, IQ4_NL or MXFP8 (the file's types, [`Hparams::rows`]). One row of
    /// site scratch.
    pub fn new(gpu: &Gpu, hp: &Hparams, layout: &ImageLayout) -> Result<Glue, GpuError> {
        Glue::with_rows(gpu, hp, layout, 1)
    }

    /// [`Glue::new`] with `rows` rows of site scratch, for a pass whose rows
    /// run one layer apart.
    pub fn with_rows(
        gpu: &Gpu,
        hp: &Hparams,
        layout: &ImageLayout,
        n_rows: usize,
    ) -> Result<Glue, GpuError> {
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        if n_rows == 0 {
            return Err(refuse("a piece of no rows".to_string()));
        }
        let dims = layout.dims();
        if hp.n_embd != ROW || dims.n_embd != ROW || hp.hc.streams != HC_STREAMS {
            return Err(refuse(format!(
                "a model of rows of {} values in {} streams and an image of rows of {}; the \
                 kernels take {ROW} and {HC_STREAMS}",
                hp.n_embd, hp.hc.streams, dims.n_embd
            )));
        }
        if dims.tokens != 1 {
            return Err(refuse(format!(
                "an image of {} tokens; the piece runs one token a step",
                dims.tokens
            )));
        }
        let (en, table_ty) = (hp.engram()?, hp.rows.engram()?);
        let table = TableRows::of(table_ty, en.key_length)
            .filter(|&(t, b)| t != TableRows::Q8_0 || b.is_multiple_of(4));
        // A rank of a tensor-parallel load carries its columns of each site's rows alone.
        let site_cols = |b: usize| {
            let site = en.layer_ids.len() * b;
            (site > 0 && dims.engram_bytes.is_multiple_of(site))
                .then(|| dims.engram_bytes / site)
                .filter(|&c| c > 0 && c <= en.rows_per_token())
        };
        let Some((table, row_bytes, rows)) = table
            .filter(|_| en.key_length > 0)
            .and_then(|(t, b)| Some((t, b, site_cols(b)?)))
        else {
            return Err(refuse(format!(
                "engram rows of {} values of {}, {} bytes a token in the image: not {} sites of \
                 up to {} Q8_0 rows in whole words, Q3_K or Q6_K rows of whole super-blocks, or \
                 IQ4_NL or MXFP8 rows of whole blocks",
                en.key_length,
                table_ty,
                dims.engram_bytes,
                en.layer_ids.len(),
                en.rows_per_token()
            )));
        };
        let Some(embd) =
            EmbdRows::of(hp.rows.token_embd, ROW).filter(|e| dims.embd_bytes == e.row_bytes(ROW))
        else {
            return Err(refuse(format!(
                "an embedding row of {}, {} bytes in the image: the broadcast decodes bf16, Q3_K \
                 and Q5_K rows of {ROW} values",
                hp.rows.token_embd, dims.embd_bytes
            )));
        };
        let (embd_at, embd_words, engram_at) = row_offsets(layout)?;
        let site_bytes = rows * row_bytes;
        let stream = gpu.stream();
        let ones = match en.layout() {
            EngramLayout::PerSite => None,
            EngramLayout::Concatenated => Some(DeviceBuffer::from_host(
                stream,
                &vec![1.0f32; HC_STREAMS * ROW],
            )?),
        };
        let mut sites = Vec::with_capacity(en.layer_ids.len());
        for (s, &layer) in en.layer_ids.iter().enumerate() {
            if hp.layers.get(layer).and_then(|k| k.engram) != Some(s) {
                return Err(refuse(format!(
                    "engram site {s} names layer {layer}, whose kind does not name the site"
                )));
            }
            let at_byte = launch_u32(WHAT, "site rows", 4 * engram_at as usize + s * site_bytes)?;
            if table.on_half_words() && !at_byte.is_multiple_of(2) {
                return Err(refuse(format!(
                    "engram site {s}'s {table_ty} rows start at image byte {at_byte}: not on a half \
                     word"
                )));
            }
            let (gain_k, gain_q) = match en.gains(layer) {
                EngramGains::Pair { key, query } => (Gain::Weight(key), Gain::Weight(query)),
                EngramGains::Folded { gate } => (Gain::Ones, Gain::Weight(gate)),
            };
            sites.push(EngramSite {
                layer,
                wkv: en.wkv(layer),
                gain_k,
                gain_q,
                at_byte,
                bufs: (0..n_rows)
                    .map(|_| {
                        Ok(SiteRow {
                            x: DeviceBuffer::zeroed(stream, rows * en.key_length)?,
                            act: Q8Act::with_k(stream, 1, rows * en.key_length)?,
                            kv: DeviceBuffer::zeroed(stream, (HC_STREAMS + 1) * ROW)?,
                            kn: DeviceBuffer::zeroed(stream, HC_STREAMS * ROW)?,
                            gate: DeviceBuffer::zeroed(stream, HC_STREAMS)?,
                        })
                    })
                    .collect::<Result<Vec<_>, GpuError>>()?,
            });
        }
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { bloomery_gpu::shared_module!(glue_kernels, gpu.context())? };
        Ok(Glue {
            module,
            hc: HcKernels::load(gpu.context())?,
            engram: EngramGateKernels::load(gpu.context())?,
            dense: DenseKernels::load(gpu.context())?,
            n_embd: hp.n_embd,
            eps: hp.rms_eps,
            embd_at,
            embd_words,
            embd,
            shape: TableShape {
                rows: launch_u32(WHAT, "rows", rows)?,
                key_len: launch_u32(WHAT, "key_length", en.key_length)?,
                row_bytes: launch_u32(WHAT, "row bytes", row_bytes)?,
            },
            table,
            sites,
            ones,
            split: rows < en.rows_per_token(),
        })
    }

    /// Device bytes of the piece's scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let ones = self.ones.as_ref().map_or(0, |o| 4 * o.len());
        self.sites
            .iter()
            .flat_map(|s| &s.bufs)
            .map(SiteRow::device_bytes)
            .sum::<usize>()
            + ones
    }

    /// Site `layer`'s projection of its gathered rows, by the file's name.
    #[must_use]
    pub fn wkv(&self, layer: usize) -> Option<&str> {
        self.sites
            .iter()
            .find(|s| s.layer == layer)
            .map(|s| s.wkv.as_str())
    }

    /// Device bytes of one row's scratch, every site's.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.sites
            .iter()
            .filter_map(|s| s.bufs.first())
            .map(SiteRow::device_bytes)
            .sum()
    }

    /// The layers that carry an engram site, in site order.
    pub fn engram_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.sites.iter().map(|s| s.layer)
    }

    /// Row 0's scratch of the site at `layer`; `None` for a layer without
    /// one.
    #[must_use]
    pub fn site_buffers(&self, layer: usize) -> Option<SiteBuffers<'_>> {
        self.site_buffers_of(layer, 0)
    }

    /// Row `row`'s scratch of the site at `layer`; `None` for a layer
    /// without one or a row the piece does not hold.
    #[must_use]
    pub fn site_buffers_of(&self, layer: usize, row: usize) -> Option<SiteBuffers<'_>> {
        self.sites
            .iter()
            .find(|s| s.layer == layer)
            .and_then(|s| s.bufs.get(row))
            .map(|b| SiteBuffers {
                rows: &b.x,
                kv: &b.kv,
                gate: &b.gate,
            })
    }

    /// Enqueue the embedding broadcast: the row the image `params` carries
    /// into every stream of `streams` and into `input`, layer 0's attention
    /// input. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed(
        &self,
        gpu: &Gpu,
        params: &DeviceBuffer<u32>,
        streams: &mut DeviceBuffer<f32>,
        input: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_embed";
        let n = self.n_embd;
        need(
            WHAT,
            "params",
            params.len(),
            self.embd_at as usize + self.embd_words as usize,
        )?;
        need(WHAT, "streams", streams.len(), HC_STREAMS * n)?;
        need(WHAT, "input", input.len(), n)?;
        let mut streams = span_mut(WHAT, streams, 0, HC_STREAMS * n)?;
        let mut input = span_mut(WHAT, input, 0, n)?;
        enqueue_embed_row(
            &self.module,
            gpu.stream(),
            self.embd,
            (params, self.embd_at),
            n,
            (&mut streams, &mut input),
        )
    }

    /// Enqueue the token-only work of the engram site at `layer`: its rows
    /// from the image `params` dequantized, `engram_wkv` over them (with the
    /// rows' q8_1 form first when it is a K-quant), and the key norm. Three
    /// or four launches, reading no stream. Asynchronous, allocation-free,
    /// capturable. Row 0's scratch.
    pub fn enqueue_engram_kv_at(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        params: &DeviceBuffer<u32>,
        layer: usize,
    ) -> Result<(), GpuError> {
        self.enqueue_engram_kv_of(gpu, w, params, (layer, 0), None)
    }

    /// [`Glue::enqueue_engram_kv_at`] into row `row`'s scratch, from that
    /// row's image `params`. On a rank of a tensor-parallel load the rows are
    /// the rank's columns and `engram_wkv`'s output the rank's part of the
    /// sum: `join` posts it, and the key norm runs after the sum
    /// ([`Glue::enqueue_engram_of`]). A `join` without split rows, or split
    /// rows without one, is refused by name.
    pub fn enqueue_engram_kv_of(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        params: &DeviceBuffer<u32>,
        (layer, row): (usize, usize),
        join: Option<&mut Sums>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_engram_kv_at";
        let stream = gpu.stream();
        let site_at = self.joined(layer, join.is_some(), WHAT)?;
        let SiteRef {
            at_byte,
            wkv,
            gain_k,
            row: site,
            ..
        } = site_row(&mut self.sites, layer, row, WHAT)?;
        need(
            WHAT,
            "params",
            4 * params.len(),
            at_byte as usize + self.shape.rows as usize * self.shape.row_bytes as usize,
        )?;
        {
            let mut x = span_mut(
                WHAT,
                &mut site.x,
                0,
                (self.shape.rows * self.shape.key_len) as usize,
            )?;
            self.table.enqueue(
                &self.module,
                stream,
                self.shape,
                (params, at_byte),
                gpu.layer_sink(layer)?,
                &mut x,
            )?;
        }
        let wkv = Dense::of(w, wkv, site.x.len(), site.kv.len())?;
        let act = if wkv.reads_q8_1() {
            gpu.enqueue_quantize_q8_1_layer(&site.x, &mut site.act, layer)?;
            Some(&site.act)
        } else {
            None
        };
        self.dense.enqueue(gpu, wkv, &site.x, act, &mut site.kv)?;
        if let Some(join) = join {
            return join.post(gpu, join::site(site_at), &site.kv);
        }
        key_norm(
            &self.engram,
            gpu,
            w,
            (gain_k, self.ones.as_ref(), self.eps),
            layer,
            site,
        )
    }

    /// The index of the site at `layer`, checked against the rows it reads: a
    /// rank's columns go with the rank's side of the sums, and every column
    /// with none.
    fn joined(&self, layer: usize, join: bool, what: &'static str) -> Result<usize, GpuError> {
        let at = self
            .sites
            .iter()
            .position(|s| s.layer == layer)
            .ok_or_else(|| GpuError::Shape {
                what,
                detail: format!("layer {layer} carries no engram site"),
            })?;
        if join != self.split {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a site whose rows are a rank's columns: {}; a sum over ranks given: {join}",
                    self.split
                ),
            });
        }
        Ok(at)
    }

    /// Enqueue the engram step at `layer`: the gate over `a.streams` into
    /// `a.out`, then the fold of `a.out` by `a.pre` into `a.input`. Two
    /// launches. The site's token-only work ([`Glue::enqueue_engram_kv_at`])
    /// must precede it on the stream.
    /// Asynchronous, allocation-free, capturable. Row 0's scratch.
    pub fn enqueue_engram(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        a: EngramStep<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_engram_of(gpu, w, (layer, 0), a, None)
    }

    /// [`Glue::enqueue_engram`] on row `row`'s scratch, which that row's
    /// token-only work wrote. On a rank of a tensor-parallel load `join` first
    /// sums the ranks' parts of `engram_wkv`'s output, which the token-only
    /// work posted, and the key norm runs on the sum.
    pub fn enqueue_engram_of(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (layer, row): (usize, usize),
        a: EngramStep<'_>,
        join: Option<&mut Sums>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_engram";
        let stream = gpu.stream();
        need(WHAT, "pre", a.pre.len(), HC_MIX)?;
        let site_at = self.joined(layer, join.is_some(), WHAT)?;
        let SiteRef {
            gain_k,
            gain_q,
            row: site,
            ..
        } = site_row(&mut self.sites, layer, row, WHAT)?;
        if let Some(join) = join {
            join.sum(gpu, join::site(site_at), &mut site.kv)?;
            key_norm(
                &self.engram,
                gpu,
                w,
                (gain_k, self.ones.as_ref(), self.eps),
                layer,
                site,
            )?;
        }
        self.engram.enqueue_gate(
            stream,
            GateArgs {
                x: a.streams,
                kn: &site.kn,
                kv: &site.kv,
                gain: gain(w, gain_q, self.ones.as_ref())?,
                eps: self.eps,
                hc: HC_STREAMS,
                m: 1,
                out: &mut *a.out,
                gate: &mut site.gate,
                fault: gpu.layer_sink(layer)?,
            },
        )?;
        self.hc
            .enqueue_fold(stream, a.out, a.pre, self.n_embd, 1, a.input)
    }

    /// Enqueue the head end: `streams`, the last ffn's HC_POST output, folded
    /// by that ffn's HC_PRE result `pre` into the head's input (`hc_out`),
    /// then the head. One launch and the head's. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_head(
        &self,
        gpu: &Gpu,
        w: &Weights,
        streams: &DeviceBuffer<f32>,
        pre: &DeviceBuffer<f32>,
        head: &mut Head,
    ) -> Result<(), GpuError> {
        if head.hidden() != self.n_embd {
            return Err(GpuError::Shape {
                what: "Glue::enqueue_head",
                detail: format!(
                    "a head of {} values, the streams hold rows of {}",
                    head.hidden(),
                    self.n_embd
                ),
            });
        }
        self.hc
            .enqueue_fold(gpu.stream(), streams, pre, self.n_embd, 1, head.input_mut())?;
        head.enqueue(gpu, w)
    }
}

/// The site at `layer` as one enqueue reads it: its image byte, its
/// weights' names and row `row`'s scratch.
struct SiteRef<'s> {
    at_byte: u32,
    wkv: &'s str,
    gain_k: &'s Gain,
    gain_q: &'s Gain,
    row: &'s mut SiteRow,
}

/// Enqueue the key norm of `site`'s `engram_wkv` output at `layer`, by the
/// gain `gain_k` (the piece's `ones` for [`Gain::Ones`]) and the epsilon
/// `eps`.
fn key_norm(
    k: &EngramGateKernels,
    gpu: &Gpu,
    w: &Weights,
    (gain_k, ones, eps): (&Gain, Option<&DeviceBuffer<f32>>, f32),
    layer: usize,
    site: &mut SiteRow,
) -> Result<(), GpuError> {
    k.enqueue_key_norm(
        gpu.stream(),
        KeyNormArgs {
            kv: &site.kv,
            gain: gain(w, gain_k, ones)?,
            eps,
            hc: HC_STREAMS,
            m: 1,
            kn: &mut site.kn,
            fault: gpu.layer_sink(layer)?,
        },
    )
}

/// The site at `layer` of `sites`, with row `row`'s scratch; `what` names
/// the caller in the error for a layer without a site or a row the piece
/// does not hold.
fn site_row<'s>(
    sites: &'s mut [EngramSite],
    layer: usize,
    row: usize,
    what: &'static str,
) -> Result<SiteRef<'s>, GpuError> {
    let EngramSite {
        at_byte,
        wkv,
        gain_k,
        gain_q,
        bufs,
        ..
    } = sites
        .iter_mut()
        .find(|s| s.layer == layer)
        .ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("layer {layer} carries no engram site"),
        })?;
    let n = bufs.len();
    let row = bufs.get_mut(row).ok_or_else(|| GpuError::Shape {
        what,
        detail: format!("row {row} of a piece of {n} rows"),
    })?;
    Ok(SiteRef {
        at_byte: *at_byte,
        wkv,
        gain_k,
        gain_q,
        row,
    })
}

/// Token 0's embedding row and engram rows in `layout`'s image, as image
/// words: the row's first word and its word count, and the first engram
/// word. Read through the layout's own view of an image whose word `i`
/// holds `i`, so the offsets are the layout's and are written nowhere else.
fn row_offsets(layout: &ImageLayout) -> Result<(u32, u32, u32), GpuError> {
    let index = (0..layout.words())
        .map(|i| launch_u32(WHAT, "image word", i))
        .collect::<Result<Vec<u32>, _>>()?;
    let view = layout.view(&index)?;
    let (embd, engram) = (view.embd(0), view.engram(0));
    match (embd.first(), engram.first()) {
        (Some(&at), Some(&engram_at)) => {
            Ok((at, launch_u32(WHAT, "row words", embd.len())?, engram_at))
        }
        _ => Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "an image of {} embedding words and {} engram words a token",
                embd.len(),
                engram.len()
            ),
        }),
    }
}

/// The `hc` rows of `g`: its resident f32 values, refused unless there are
/// `HC_STREAMS·ROW`, or `ones`.
fn gain<'w>(
    w: &'w Weights,
    g: &Gain,
    ones: Option<&'w DeviceBuffer<f32>>,
) -> Result<&'w DeviceBuffer<f32>, GpuError> {
    let len = HC_STREAMS * ROW;
    let name = match g {
        Gain::Weight(name) => name,
        Gain::Ones => {
            return ones.ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: "a key gain of ones in a piece that holds none".to_string(),
            });
        }
    };
    match w.get(name) {
        Some(DevWeight::F32 { w: t, .. }) if t.buf().len() == len => Ok(t.buf()),
        Some(DevWeight::F32 { w: t, .. }) => Err(GpuError::Shape {
            what: WHAT,
            detail: format!("{name} holds {} values, want {len}", t.buf().len()),
        }),
        found => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: if found.is_some() { "F32" } else { "resident" },
        }),
    }
}

/// Refuse a buffer `name` of `len` elements shorter than `want`.
fn need(what: &'static str, name: &str, len: usize, want: usize) -> Result<(), GpuError> {
    if len < want {
        return Err(GpuError::Shape {
            what,
            detail: format!("{name}.len() {len} < {want}"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------- the host half

/// The levers of [`StepRows`], which it opens with ([`StepRows::open`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowsLevers {
    /// The engram rows are read by a helper thread: the calling thread hashes
    /// the ids, hands them over, reads the embedding row meanwhile and waits.
    /// Off (`BLOOMERY_ENGRAM_HELPER=0`), the calling thread reads them itself.
    pub helper: bool,
    /// Keep [`EngramStats`] (`BLOOMERY_STEP_STATS=1`). Off, a fill reads no
    /// clock and makes no classifying syscall.
    pub stats: bool,
}

impl RowsLevers {
    /// The levers of a binary's one parse (`bloomery_levers::at_main`).
    #[must_use]
    pub fn from_levers(levers: &bloomery_levers::Levers) -> RowsLevers {
        RowsLevers {
            helper: levers.engram_helper(),
            stats: levers.step_stats(),
        }
    }
}

/// What the fills since [`StepRows`] opened read, when it keeps stats
/// ([`RowsLevers::stats`]); all zero otherwise. A row is warm when every page
/// of it was in the page cache before the read, cold otherwise
/// ([`engram::Site::resident_rows`]); both are counted on either path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngramStats {
    pub warm: u64,
    pub cold: u64,
    /// Rows the calling thread read itself: the direct path's.
    pub direct: u64,
    /// The calling thread's time in the engram read: the wait on the helper
    /// in [`StepRows::finish`] — in the engine, after the step's launch — or
    /// the direct path's copy, before it.
    pub wait_ns: u64,
    /// The helper's own time on the rows it served: advise, fill and copy.
    pub helper_ns: u64,
    /// The classification's time ([`engram::Site::resident_rows`]): on the
    /// helper before its advise, so inside `wait_ns`, or on the calling thread
    /// before the direct copy, outside it. The stats' own tax.
    pub classify_ns: u64,
}

/// The rows one step's image carries ([`crate::params::StepImage::build`]'s
/// `embd` and `engram`), for a step of a fixed token count. A token's
/// embedding row is the file's `token_embd` row, its bytes as the file
/// stores them (bf16, Q3_K or Q5_K). A site's rows are the
/// ones the engram crate's hash names for the token's n-gram — the plan's
/// window mapped as the port maps it: the current token, then each older one
/// up to the first missing, the pad id from there on — read out of the
/// site's table by the helper ([`Prefetcher`], advise every row first, then
/// copy), or on the calling thread with the helper off. Either way the bytes
/// are the table's.
pub struct StepRows {
    engram: Arc<Tables>,
    /// The helper, when [`RowsLevers::helper`] is on.
    helper: Option<Prefetcher>,
    /// Kept when [`RowsLevers::stats`] is on.
    stats: Option<EngramStats>,
    /// `token_embd`'s layout in the file.
    embd: RowLayout,
    embd_bytes: usize,
    n_vocab: usize,
    /// Rows a site gathers per token, and bytes a row.
    n_cols: usize,
    row_bytes: usize,
    /// The columns of each site's rows the step reads: every one, or a rank's.
    cols: Range<usize>,
    tokens: usize,
    ctx: Vec<u64>,
    /// One token's ids of every column of every site, as the hash gives them.
    all: Vec<u32>,
    /// `[token][site][n_cols]`: the read columns' ids.
    ids: Vec<u32>,
    embd_rows: Vec<u8>,
    engram_rows: Vec<u8>,
    /// The token whose engram rows are on the helper, begun and not
    /// finished.
    in_flight: Option<usize>,
}

impl StepRows {
    /// The host half for steps of `tokens` tokens of `file`, whose
    /// hyperparameters are `hp`, under `levers`: the embedding table checked,
    /// the engram tables opened (headers only) and checked against `hp`, and
    /// every buffer a fill writes allocated. Each site's rows are read in its
    /// columns `cols`: all of them, or a rank's of a tensor-parallel load. An
    /// empty range or one past the columns is refused by name. The tables are the split's
    /// (`stored` `None`), or a checkpoint folder's row tables where it stores them, each site's
    /// two MXFP8 planes ([`PlanarTable`]) under the hash `file`'s keys give.
    ///
    /// A helper is pinned to the SMT sibling of the calling thread's core when
    /// that thread is pinned ([`caller_sibling`]) — the caller is the pool's
    /// dispatcher, which waits while the helper reads, so that core is idle
    /// then and no worker's core is shared — and floats otherwise.
    pub fn open(
        file: &Split,
        hp: &Hparams,
        tokens: usize,
        levers: RowsLevers,
        cols: Range<usize>,
        stored: Option<&BTreeMap<String, StoredRows>>,
    ) -> Result<StepRows, GpuError> {
        const WHAT: &str = "StepRows::open";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let name = names::token_embd();
        let embd = RowLayout::open(file, &name, hp.n_embd)?;
        let embd_ok = EmbdRows::of(embd.ty(), hp.n_embd).is_some()
            && embd.ty() == hp.rows.token_embd
            && embd.n_vocab() == hp.n_vocab;
        if !embd_ok {
            return Err(refuse(format!(
                "{name} is {} of {} rows, want bf16, Q3_K or Q5_K [{}, {}]",
                embd.ty(),
                embd.n_vocab(),
                hp.n_embd,
                hp.n_vocab
            )));
        }
        let embd_bytes = embd.row_bytes();
        let engram = Tables::open(file, stored).map_err(|e| GpuError::plan(WHAT, e))?;
        let hash = &engram.hash;
        hash.check_vocab(hp.n_vocab)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let (en, table_ty) = (hp.engram()?, hp.rows.engram()?);
        let (_, row_bytes) = TableRows::of(table_ty, en.key_length).ok_or_else(|| {
            refuse(format!(
                "engram tables of {} rows of {} values: the card reads Q8_0, Q3_K, Q6_K, IQ4_NL and MXFP8 rows",
                table_ty, en.key_length
            ))
        })?;
        let layers_agree = hash.layer_ids().len() == en.layer_ids.len()
            && hash
                .layer_ids()
                .iter()
                .zip(&en.layer_ids)
                .all(|(&a, &b)| usize::try_from(a).is_ok_and(|a| a == b));
        if !layers_agree
            || hash.n_gram() != en.max_ngram
            || hash.n_cols() != en.rows_per_token()
            || usize::try_from(hash.key_length()).ok() != Some(en.key_length)
            || engram
                .sites()
                .any(|t| usize::try_from(t.row_bytes()).ok() != Some(row_bytes))
        {
            return Err(refuse(format!(
                "the engram tables hash sites {:?} of {}-grams into {} rows of {} values; the \
                 file's hyperparameters say {:?}, {}, {} and {} in {} rows of {row_bytes} bytes",
                hash.layer_ids(),
                hash.n_gram(),
                hash.n_cols(),
                hash.key_length(),
                en.layer_ids,
                en.max_ngram,
                en.rows_per_token(),
                en.key_length,
                table_ty
            )));
        }
        let all = hash.n_cols();
        if cols.is_empty() || cols.end > all {
            return Err(refuse(format!(
                "columns {cols:?} of a site's rows; the hash gives {all}"
            )));
        }
        let n_cols = cols.len();
        let sites = engram.count();
        let engram = Arc::new(engram);
        let helper = levers
            .helper
            .then(|| {
                Prefetcher::over(
                    Arc::clone(&engram) as Arc<dyn RowTables>,
                    &vec![n_cols; sites],
                    HelperOptions::spread(FillMode::Touch, caller_sibling(), levers.stats),
                )
            })
            .transpose()
            .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(StepRows {
            helper,
            stats: levers.stats.then(EngramStats::default),
            embd,
            embd_bytes,
            n_vocab: hp.n_vocab,
            n_cols,
            row_bytes,
            cols,
            tokens,
            ctx: vec![0; en.max_ngram],
            all: vec![0; sites * all],
            ids: vec![0; tokens * sites * n_cols],
            embd_rows: vec![0; tokens * embd_bytes],
            engram_rows: vec![0; tokens * sites * n_cols * row_bytes],
            in_flight: None,
            engram,
        })
    }

    /// Bytes of one engram row.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Fill the rows of `plan`'s step from `file`, the file `open` read:
    /// [`StepRows::begin`], then [`StepRows::finish`].
    pub fn fill(&mut self, file: &Split, plan: &StepPlan) -> Result<(), GpuError> {
        self.begin(file, plan)?;
        self.finish()
    }

    /// Start the rows of `plan`'s step: every token's ids hashed and its
    /// embedding row read from `file`; with the helper on, the last token's
    /// engram rows handed to the helper and left in flight — the embedding
    /// rows are read meanwhile — until [`StepRows::finish`] takes them back
    /// (every earlier token's are taken back here); with it off, every row
    /// read on this thread. A begin while a token is in flight is refused.
    pub fn begin(&mut self, file: &Split, plan: &StepPlan) -> Result<(), GpuError> {
        const WHAT: &str = "StepRows::begin";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let n_gram = self.ctx.len();
        if plan.tokens.len() != self.tokens || plan.engram_window.len() != self.tokens * n_gram {
            return Err(refuse(format!(
                "a plan of {} tokens and {} n-gram slots; the rows hold {} tokens of {n_gram}",
                plan.tokens.len(),
                plan.engram_window.len(),
                self.tokens
            )));
        }
        if self.in_flight.is_some() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the finish of the rows in flight (StepRows::finish) before the next begin",
            });
        }
        check_tokens(plan, self.n_vocab)?;
        if self.helper.is_none() {
            embd_rows_into(&self.embd, &mut self.embd_rows, file, plan)?;
        }
        let site_bytes = self.n_cols * self.row_bytes;
        let token_ids = self.engram.count() * self.n_cols;
        for (k, window) in plan.engram_window.chunks_exact(n_gram).enumerate() {
            let ids = &mut self.ids[k * token_ids..][..token_ids];
            token_rows(
                &self.engram,
                (&mut self.ctx, window),
                &mut self.all,
                ids,
                &self.cols,
            )?;
            let ids = &self.ids[k * token_ids..][..token_ids];
            match self.helper.as_mut() {
                Some(helper) => {
                    helper
                        .submit(ids.chunks_exact(self.n_cols))
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                    self.in_flight = Some(k);
                    // Only the first token overlaps the embedding rows; a
                    // step of one token, the engine's, is all of it.
                    if k == 0 {
                        let embd = embd_rows_into(&self.embd, &mut self.embd_rows, file, plan);
                        // The helper holds a job: take it back before an
                        // error leaves, or the next begin's submit is refused.
                        if let Err(e) = embd {
                            let _ = self.finish();
                            return Err(e);
                        }
                    }
                    if k + 1 < self.tokens {
                        self.finish()?;
                    }
                }
                None => {
                    let bytes = &mut self.engram_rows[k * token_ids * self.row_bytes..]
                        [..token_ids * self.row_bytes];
                    for ((site, site_ids), out) in self
                        .engram
                        .sites()
                        .zip(ids.chunks_exact(self.n_cols))
                        .zip(bytes.chunks_exact_mut(site_bytes))
                    {
                        let t0 = match self.stats.as_mut() {
                            Some(st) => {
                                let tc = Instant::now();
                                let warm = site
                                    .resident_rows(site_ids)
                                    .map_err(|e| GpuError::plan(WHAT, e))?;
                                st.classify_ns += tc.elapsed().as_nanos() as u64;
                                st.warm += warm;
                                st.cold += site_ids.len() as u64 - warm;
                                st.direct += site_ids.len() as u64;
                                Some(Instant::now())
                            }
                            None => None,
                        };
                        site.copy_rows(site_ids, out)
                            .map_err(|e| GpuError::plan(WHAT, e))?;
                        if let (Some(st), Some(t0)) = (self.stats.as_mut(), t0) {
                            st.wait_ns += t0.elapsed().as_nanos() as u64;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a token's engram rows are in flight on the helper: begun and
    /// not finished.
    #[must_use]
    pub fn in_flight(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Take the rows in flight back from the helper into
    /// [`StepRows::engram`]; nothing to do when none are. A helper that
    /// failed leaves that token's rows as the previous fill left them and
    /// returns its error; either way nothing is in flight after.
    pub fn finish(&mut self) -> Result<(), GpuError> {
        const WHAT: &str = "StepRows::finish";
        let (Some(k), Some(helper)) = (self.in_flight.take(), self.helper.as_mut()) else {
            return Ok(());
        };
        let token_bytes = self.engram.count() * self.n_cols * self.row_bytes;
        let t0 = self.stats.map(|_| Instant::now());
        helper.wait().map_err(|e| GpuError::plan(WHAT, e))?;
        let wait_ns = t0.map_or(0, |t| t.elapsed().as_nanos() as u64);
        let filled = helper.filled();
        if filled.len() != token_bytes {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the helper filled {} bytes of a token's {token_bytes}",
                    filled.len()
                ),
            });
        }
        self.engram_rows[k * token_bytes..][..token_bytes].copy_from_slice(filled);
        if let Some(st) = self.stats.as_mut() {
            let t = helper.last();
            st.warm += t.resident_rows;
            st.cold += t.rows - t.resident_rows;
            st.wait_ns += wait_ns;
            st.helper_ns += t.submit_ns + t.fill_ns + t.copy_ns;
            st.classify_ns += t.classify_ns;
        }
        Ok(())
    }

    /// The page faults the helper thread has taken since it started (its own
    /// `getrusage(RUSAGE_THREAD)`, as of the last token it finished); `None`
    /// with the helper off, when the calling thread reads the rows itself.
    #[must_use]
    pub fn helper_faults(&self) -> Option<Faults> {
        self.helper.as_ref().map(|h| {
            let t = h.last();
            Faults {
                major: t.major_faults,
                minor: t.minor_faults,
            }
        })
    }

    /// What the fills since open read; all zero unless the rows keep stats
    /// ([`RowsLevers::stats`]).
    #[must_use]
    pub fn engram_stats(&self) -> EngramStats {
        self.stats.unwrap_or_default()
    }

    /// Whether a helper reads the engram rows, and the cpu it is pinned to:
    /// `None` with the helper off, `Some(None)` for a floating helper.
    #[must_use]
    pub fn helper_cpu(&self) -> Option<Option<usize>> {
        self.helper.as_ref().map(Prefetcher::pinned_cpu)
    }

    /// The step's embedding rows, `tokens` rows of `token_embd`'s bytes.
    #[must_use]
    pub fn embd(&self) -> &[u8] {
        &self.embd_rows
    }

    /// The step's engram rows: per token, every site's rows in site order.
    #[must_use]
    pub fn engram(&self) -> &[u8] {
        &self.engram_rows
    }

    /// The row ids site `site` gathered for token `token` of the last fill.
    #[must_use]
    pub fn ids(&self, token: usize, site: usize) -> &[u32] {
        let sites = self.engram.count();
        &self.ids[(token * sites + site) * self.n_cols..][..self.n_cols]
    }
}

/// The row ids every site gathers for the token whose n-gram `window` holds,
/// newest first (a plan's [`StepPlan::engram_window`] run of one token), into
/// `ids`, the columns `cols` of each site in site order: the window mapped as
/// the port maps it — the token, then each older one up to the first missing,
/// the pad id from there on — into `ctx`, then hashed per site into `all`,
/// every column of every site, whose `cols` are kept. A token past the
/// table's token map is not a token of the model and is refused by name.
fn token_rows(
    engram: &Tables,
    (ctx, window): (&mut [u64], &[Option<u32>]),
    all: &mut [u32],
    ids: &mut [u32],
    cols: &Range<usize>,
) -> Result<(), GpuError> {
    let hash = &engram.hash;
    let mut blocked = false;
    for (slot, &token) in ctx.iter_mut().zip(window) {
        blocked |= token.is_none();
        *slot = match token {
            Some(t) if !blocked => hash
                .map_token(t)
                .map_err(|e| GpuError::plan("StepRows::begin", e))?,
            _ => hash.pad_id(),
        };
    }
    for (s, (site_all, site_ids)) in all
        .chunks_exact_mut(hash.n_cols())
        .zip(ids.chunks_exact_mut(cols.len()))
        .enumerate()
    {
        hash.rows_into(s, ctx, site_all)
            .map_err(|e| GpuError::plan("StepRows::begin", e))?;
        site_ids.copy_from_slice(&site_all[cols.clone()]);
    }
    Ok(())
}

/// Every token of `plan` below `n_vocab`, checked before any row is read: an id
/// past the vocabulary is not a token of this model. The one refusal for it on
/// every path — the engram helper hashes a token before the embedding row is
/// read, so without this the helper and the direct path would name the same
/// input differently. It carries the embedding read's name, the fill's.
fn check_tokens(plan: &StepPlan, n_vocab: usize) -> Result<(), GpuError> {
    match plan.tokens.iter().find(|&&t| t as usize >= n_vocab) {
        Some(t) => Err(GpuError::Shape {
            what: "StepRows::fill",
            detail: format!("token {t} is past the vocabulary of {n_vocab} tokens"),
        }),
        None => Ok(()),
    }
}

/// The engram hash and each site's row table, in site order, which a step's rows and a prompt's
/// read ([`StepRows`], [`PromptRows`]).
pub(crate) struct Tables {
    hash: Hash,
    sites: Box<dyn RowTables>,
}

impl Tables {
    /// The split `file`'s own tables (`stored` `None`), or the folder's row tables `stored`
    /// holds by name, each site's MXFP8 values beside its block scales ([`PlanarTable`]), under
    /// the hash `file`'s keys give ([`Hash::from_keys`]). A site the folder does not store as
    /// MXFP8 planes is refused by name.
    fn open(
        file: &Split,
        stored: Option<&BTreeMap<String, StoredRows>>,
    ) -> Result<Tables, engram::EngramError> {
        let Some(stored) = stored else {
            let paths = (0..file.shard_count()).filter_map(|i| file.shard_path(i));
            let (hash, sites) = Engram::open(paths)?.into_parts();
            return Ok(Tables {
                hash,
                sites: Box::new(sites),
            });
        };
        let hash = Hash::from_keys(file)?;
        let sites = hash
            .layer_ids()
            .iter()
            .map(|&l| {
                let name = names::engram_embd(l as usize);
                match stored.get(&name) {
                    Some(StoredRows {
                        values,
                        block_scales: Some(scales),
                        ..
                    }) => PlanarTable::open(
                        &name,
                        &values.runs,
                        &scales.runs,
                        values.row_bytes,
                        scales.row_bytes,
                        PageAdvice::Random,
                    ),
                    _ => Err(engram::EngramError::MissingSite { name }),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tables {
            hash,
            sites: Box::new(sites),
        })
    }

    /// The sites' tables, in site order.
    fn sites(&self) -> impl Iterator<Item = &dyn RowTable> {
        (0..self.count()).filter_map(|i| self.sites.table(i))
    }
}

impl RowTables for Tables {
    fn count(&self) -> usize {
        self.sites.count()
    }

    fn table(&self, i: usize) -> Option<&dyn RowTable> {
        self.sites.table(i)
    }
}

/// Every token's embedding row of `plan` into `out` (`embd.row_bytes()`
/// bytes a token, as the file stores them), from `file`'s `token_embd` laid
/// out as `embd`.
fn embd_rows_into(
    embd: &RowLayout,
    out: &mut [u8],
    file: &Split,
    plan: &StepPlan,
) -> Result<(), GpuError> {
    for (&token, dst) in plan
        .tokens
        .iter()
        .zip(out.chunks_exact_mut(embd.row_bytes()))
    {
        let bytes = embd.row(file, token).map_err(|e| GpuError::Shape {
            what: "StepRows::fill",
            detail: e.to_string(),
        })?;
        dst.copy_from_slice(bytes);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomery_gpu::fault::{Fault, FaultSite, LAYER_NONE};
    use gguf::quant::dequant_row;

    /// A xorshift64 stream: the test's bytes, the same on every run.
    struct Rng(u64);

    impl Rng {
        fn word(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.word() as u8).collect()
        }
    }

    /// `n_blocks` blocks of `ty` of random bytes, but for each f16 scale: its
    /// exponent's top bit is cleared, so every value is finite and a decode
    /// that differs shows as different bits, never as NaN against NaN.
    fn blocks(ty: GgmlType, n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
        let size = ty.type_size().expect("a sized type") as usize;
        let scales: &[usize] = match ty {
            GgmlType::Q3_K => &[108],
            GgmlType::Q5_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            GgmlType::MXFP8 => return mxfp8_blocks(n_blocks, rng),
            _ => &[0],
        };
        let mut out = Vec::with_capacity(n_blocks * size);
        for _ in 0..n_blocks {
            let mut b = rng.bytes(size);
            for &at in scales {
                b[at + 1] &= 0xbf;
            }
            out.extend(b);
        }
        out
    }

    /// `n_blocks` MXFP8 blocks of random bytes the file decode takes: each
    /// scale byte on 112..=136 (`2^-15` to `2^9`, so a value times it stays a
    /// finite f32) and no E4M3 NaN value (0x7F, 0xFF).
    fn mxfp8_blocks(n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
        let mut out = rng.bytes(n_blocks * MXFP8_BYTES);
        for (i, b) in out.iter_mut().enumerate() {
            if i % MXFP8_BYTES == 0 {
                *b = 112 + *b % 25;
            } else if *b & 0x7F == 0x7F {
                *b &= 0xFE;
            }
        }
        out
    }

    /// An image whose `at_byte`-th byte starts `rows`, between random bytes.
    fn image(at_byte: usize, rows: &[u8], rng: &mut Rng) -> Vec<u32> {
        let mut bytes = rng.bytes(at_byte);
        bytes.extend_from_slice(rows);
        bytes.extend(rng.bytes(8));
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    fn differs(got: &[f32], want: &[f32]) -> Option<String> {
        got.iter()
            .zip(want)
            .position(|(g, w)| g.to_bits() != w.to_bits())
            .map(|i| {
                format!(
                    "value {i}: {:#010x} against {:#010x}",
                    got[i].to_bits(),
                    want[i].to_bits()
                )
            })
            .or_else(|| {
                (got.len() != want.len())
                    .then(|| format!("{} values, want {}", got.len(), want.len()))
            })
    }

    /// Every row format the glue takes, decoded on the card out of an image,
    /// is the bits `gguf::quant::dequant_row` gives the same bytes: a token's
    /// embedding row into the four streams and the layer's input, and a
    /// site's gathered table rows (Q8_0, Q3_K, IQ4_NL, MXFP8 and Q6_K — its
    /// rows from both half-word starts), none of them raising; an MXFP8 byte
    /// the format leaves undefined decodes to NaN under its named fault.
    /// Mutants: the decoder without its check gives 480 for the E4M3 NaN; a
    /// Q6_K decode that drops its signed scale differs from the file's.
    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_glue_rows_decode_as_the_file_decodes_them() {
        const N: usize = 768;
        const KEY_LEN: usize = 256;
        const ROWS: usize = 6;
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { bloomery_gpu::shared_module!(glue_kernels, gpu.context()) }
            .expect("glue kernels");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut failures = Vec::new();
        for embd in [EmbdRows::Bf16, EmbdRows::Q3K, EmbdRows::Q5K] {
            let (ty, bytes) = (embd.ty(), embd.row_bytes(N));
            let row = if embd == EmbdRows::Bf16 {
                (0..N)
                    .flat_map(|_| (rng.word() as u16 & 0xbfff).to_le_bytes())
                    .collect()
            } else {
                blocks(ty, N / 256, &mut rng)
            };
            let mut want = vec![0f32; N];
            if embd == EmbdRows::Bf16 {
                for (w, b) in want.iter_mut().zip(row.as_chunks::<2>().0) {
                    *w = f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16);
                }
            } else {
                dequant_row(ty, &row, &mut want).expect("the file decode");
            }
            assert_eq!(row.len(), bytes, "{embd:?}");
            let at = 5usize;
            let params =
                DeviceBuffer::from_host(stream, &image(4 * at, &row, &mut rng)).expect("image");
            let mut streams = DeviceBuffer::<f32>::zeroed(stream, HC_STREAMS * N).expect("streams");
            let mut input = DeviceBuffer::<f32>::zeroed(stream, N).expect("input");
            {
                let mut s = span_mut("test", &mut streams, 0, HC_STREAMS * N).expect("streams");
                let mut x = span_mut("test", &mut input, 0, N).expect("input");
                enqueue_embed_row(
                    &module,
                    stream,
                    embd,
                    (&params, at as u32),
                    N,
                    (&mut s, &mut x),
                )
                .expect("embed launch");
            }
            let got = streams.to_host_vec(stream).expect("streams");
            for (s, got) in got.as_chunks::<N>().0.iter().enumerate() {
                failures.extend(differs(got, &want).map(|d| format!("{embd:?} stream {s}: {d}")));
            }
            let got = input.to_host_vec(stream).expect("input");
            failures.extend(differs(&got, &want).map(|d| format!("{embd:?} input: {d}")));
        }
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q3_K,
            GgmlType::IQ4_NL,
            GgmlType::MXFP8,
            GgmlType::Q6_K,
        ] {
            let (table, row_bytes) = TableRows::of(ty, KEY_LEN).expect("a table format");
            let n_blocks = ROWS * row_bytes / ty.type_size().expect("sized") as usize;
            let bytes = blocks(ty, n_blocks, &mut rng);
            let mut want = vec![0f32; ROWS * KEY_LEN];
            for (b, w) in bytes
                .chunks_exact(row_bytes)
                .zip(want.as_chunks_mut::<KEY_LEN>().0)
            {
                dequant_row(ty, b, w).expect("the file decode");
            }
            // Q8_0 rows are read by word; Q3_K, IQ4_NL and Q6_K start on a
            // half word, which 14 puts at 2 mod 4 (a Q6_K row's 210 bytes
            // put the next at 0 mod 4); MXFP8 is read by byte, from an odd
            // one.
            let at_byte = match table {
                TableRows::Q8_0 => 12,
                TableRows::Mxfp8 => 13,
                _ => 14,
            };
            let params =
                DeviceBuffer::from_host(stream, &image(at_byte, &bytes, &mut rng)).expect("image");
            let mut y = DeviceBuffer::<f32>::zeroed(stream, ROWS * KEY_LEN).expect("rows");
            let shape = TableShape {
                rows: ROWS as u32,
                key_len: KEY_LEN as u32,
                row_bytes: row_bytes as u32,
            };
            {
                let mut span = span_mut("test", &mut y, 0, ROWS * KEY_LEN).expect("rows");
                table
                    .enqueue(
                        &module,
                        stream,
                        shape,
                        (&params, at_byte as u32),
                        gpu.unlabelled_sink(),
                        &mut span,
                    )
                    .expect("rows launch");
            }
            let got = y.to_host_vec(stream).expect("rows");
            failures.extend(differs(&got, &want).map(|d| format!("{table:?} rows: {d}")));
            let fault = gpu.take_fault().expect("fault word");
            failures.extend(fault.map(|f| format!("{table:?} rows raised {f:?}")));
        }
        // An MXFP8 byte the format leaves undefined, or a product past f32,
        // decodes to NaN and raises its named fault: value 5 of block 0 at a
        // NaN scale, an E4M3 NaN, and 448 at the top scale.
        let (table, row_bytes) = TableRows::of(GgmlType::MXFP8, KEY_LEN).expect("MXFP8 rows");
        for (case, scale, value) in [
            ("scale", 0xFF, 0x10),
            ("value", 127, 0x7F),
            ("product", 0xFE, 0x7E),
        ] {
            let mut bytes = mxfp8_blocks(row_bytes / MXFP8_BYTES, &mut rng);
            bytes[0] = scale;
            bytes[1 + 5] = value;
            let params =
                DeviceBuffer::from_host(stream, &image(13, &bytes, &mut rng)).expect("image");
            let mut y = DeviceBuffer::<f32>::zeroed(stream, KEY_LEN).expect("row");
            let shape = TableShape {
                rows: 1,
                key_len: KEY_LEN as u32,
                row_bytes: row_bytes as u32,
            };
            {
                let mut span = span_mut("test", &mut y, 0, KEY_LEN).expect("row");
                table
                    .enqueue(
                        &module,
                        stream,
                        shape,
                        (&params, 13),
                        gpu.unlabelled_sink(),
                        &mut span,
                    )
                    .expect("row launch");
            }
            let got = y.to_host_vec(stream).expect("row");
            if !got[5].is_nan() {
                failures.push(format!("MXFP8 {case}: value 5 decoded to {}", got[5]));
            }
            let fault = gpu.take_fault().expect("fault word");
            if fault != Some(Fault::at(LAYER_NONE, FaultSite::F32Product)) {
                failures.push(format!("MXFP8 {case}: raised {fault:?}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
