//! The launch of decode GQA attention: 16 query heads, 2 key heads, head
//! dim 256. One threadgroup is one key head and one segment of eight SIMD
//! groups, one query head a group, the keys streamed from the cache into
//! registers.
//!
//! One segment is the flash: it normalizes, applies the gate and writes the
//! output. More than one writes partials, and the dispatch's last
//! threadgroup — the ticket's — folds them, applies the gate and writes the
//! output in the same launch. The ticket is one word the caller holds
//! zeroed; the fold resets it, so the next launch finds it zero. There is
//! no merge dispatch.
//!
//! [`Kernels::attention_rows`] is the same attention over a chunk of rows,
//! each row a causal query at its own position: the append of the chunk's
//! keys and values, then the flash with a row grid dimension. Each row is the
//! bits of the decode launch at its position with the same segment count.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{Library, Pipeline};

use super::rows::check_strided;
use super::{Kernels, args_bytes, check_window_len, pack, word};

use super::attention_geom::{FLASH_THREADS, PARTIAL_ELEMS, SIMD_WIDTH};
pub use super::attention_geom::{
    HEAD_DIM, KV_HEADS, QG_STRIDE, QUERY_HEADS, ROT, SCALE, SEGMENT_CAP, SEGMENT_KEYS, TILE,
    segments_for_keys,
};

/// The f16 cache or the q8_0 planes a launch reads and appends.
pub enum AttnCache<'w> {
    /// `[kv][ctx][256]` f16 bits.
    F16 {
        /// Key rows.
        k: WindowMut<'w, u16, MetalDevice>,
        /// Value rows.
        v: WindowMut<'w, u16, MetalDevice>,
    },
    /// q8_0 planes: 64 code words and 8 scales per row.
    Q8 {
        /// Key codes, little-endian bytes within each word.
        k_codes: WindowMut<'w, u32, MetalDevice>,
        /// Key scales, f16 bits.
        k_scales: WindowMut<'w, u16, MetalDevice>,
        /// Value codes.
        v_codes: WindowMut<'w, u32, MetalDevice>,
        /// Value scales, f16 bits.
        v_scales: WindowMut<'w, u16, MetalDevice>,
    },
}

/// One attention launch. A key count of 0 or past `ctx`, and an append
/// position at or past `ctx`, are the kernel's faults, not refusals.
///
/// `q` is `[query heads, 512]`: the query, then the gate, per head.
pub struct AttentionParams<'w> {
    /// Head dimension. The kernel takes [`HEAD_DIM`].
    pub head_dim: usize,
    /// Query heads. The kernel takes [`QUERY_HEADS`].
    pub query_heads: usize,
    /// Key and value heads. The kernel takes [`KV_HEADS`].
    pub kv_heads: usize,
    /// Split-K segments. One finishes in the flash.
    pub segments: usize,
    /// Cache rows per key head.
    pub ctx: usize,
    /// RMS epsilon.
    pub eps: f32,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// Append position.
    pub pos: u32,
    /// Live key count, including the appended row when it is live.
    pub n_keys: u32,
    /// The query projection, [`qg_len`] elements: query, then gate, per head.
    pub q: Window<'w, f32, MetalDevice>,
    /// Keys, one raw row per key head.
    pub k: Window<'w, f32, MetalDevice>,
    /// Values, one raw row per key head.
    pub v: Window<'w, f32, MetalDevice>,
    /// Query RMS gain, 256 values.
    pub q_gain: Window<'w, f32, MetalDevice>,
    /// Key RMS gain, 256 values.
    pub k_gain: Window<'w, f32, MetalDevice>,
    /// Rope table, `ctx` rows of [`ROT`] interleaved `(cos, sin)` pairs.
    pub rope: Window<'w, f32, MetalDevice>,
    /// The cache the launch reads and appends.
    pub cache: AttnCache<'w>,
    /// Segment partials, [`partials_len`] elements.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The layer output, [`out_len`] elements, after the gate.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word.
    pub fault: &'w MetalFaultWord,
    /// The fold's ticket, one word. Held zeroed; the fold resets it, so the
    /// next launch finds it zero. Bound only past one segment.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
}

/// The partial buffer's length: one partial per head and segment.
#[must_use]
pub fn partials_len(segments: usize) -> Option<usize> {
    QUERY_HEADS
        .checked_mul(segments)?
        .checked_mul(PARTIAL_ELEMS)
}

/// The attention output's length.
#[must_use]
pub fn out_len() -> Option<usize> {
    QUERY_HEADS.checked_mul(HEAD_DIM)
}

/// The query projection's length: the query and the gate, per head.
#[must_use]
pub fn qg_len() -> Option<usize> {
    QUERY_HEADS.checked_mul(QG_STRIDE)
}

/// f16 cache rows, `kv * ctx * 256` bits.
#[must_use]
pub fn f16_len(ctx: usize) -> Option<usize> {
    KV_HEADS.checked_mul(ctx)?.checked_mul(HEAD_DIM)
}

/// q8_0 code words per cache, `kv * ctx * 64`.
#[must_use]
pub fn q8_code_len(ctx: usize) -> Option<usize> {
    KV_HEADS.checked_mul(ctx)?.checked_mul(HEAD_DIM / 4)
}

/// q8_0 scales per cache, `kv * ctx * 8`.
#[must_use]
pub fn q8_scale_len(ctx: usize) -> Option<usize> {
    KV_HEADS.checked_mul(ctx)?.checked_mul(HEAD_DIM / 32)
}

pub(crate) struct AttentionPipelines {
    f16: Pipeline,
    q8: Pipeline,
    append_rows: Pipeline,
    flash_rows: Pipeline,
}

impl AttentionPipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            f16: library.pipeline("gqa256_flash_f16")?,
            q8: library.pipeline("gqa256_flash_q8")?,
            append_rows: library.pipeline("attn_norm_rope_rows")?,
            flash_rows: library.pipeline("gqa256_flash_f16_rows")?,
        })
    }
}

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "attention",
        detail,
    }
}

struct Checked {
    args: [[u8; 4]; 7],
}

fn checked(p: &AttentionParams<'_>) -> Result<Checked, DeviceError> {
    if p.head_dim != HEAD_DIM {
        return Err(refuse(format!(
            "head dim {}, the kernel takes {HEAD_DIM}",
            p.head_dim
        )));
    }
    if p.query_heads != QUERY_HEADS {
        return Err(refuse(format!(
            "query heads {}, the kernel takes {QUERY_HEADS}",
            p.query_heads
        )));
    }
    if p.kv_heads != KV_HEADS {
        return Err(refuse(format!(
            "kv heads {}, the kernel takes {KV_HEADS}",
            p.kv_heads
        )));
    }
    if p.segments == 0 {
        return Err(refuse("segments of 0".to_string()));
    }
    if p.ctx == 0 {
        return Err(refuse("ctx of 0".to_string()));
    }
    let q_len = qg_len().ok_or_else(|| refuse("the query length wraps".to_string()))?;
    let kv_len = KV_HEADS
        .checked_mul(HEAD_DIM)
        .ok_or_else(|| refuse("the key length wraps".to_string()))?;
    let rope_len = p
        .ctx
        .checked_mul(ROT)
        .ok_or_else(|| refuse("the rope length wraps".to_string()))?;
    let partials =
        partials_len(p.segments).ok_or_else(|| refuse("the partial length wraps".to_string()))?;
    if p.segments > SEGMENT_CAP {
        return Err(refuse(format!(
            "segments of {} past the kernel's {SEGMENT_CAP}",
            p.segments
        )));
    }
    check_window_len("attention", "ticket", p.ticket.len(), 1)?;
    let ticket = p
        .ticket
        .byte_range()
        .map_err(|e| refuse(format!("the ticket window: {e}")))?;
    if ticket.start.get() % 4 != 0 {
        return Err(refuse("the ticket word is not 4-byte aligned".to_string()));
    }
    let out = out_len().ok_or_else(|| refuse("the output length wraps".to_string()))?;
    check_window_len("attention", "q", p.q.len(), q_len)?;
    check_window_len("attention", "k", p.k.len(), kv_len)?;
    check_window_len("attention", "v", p.v.len(), kv_len)?;
    check_window_len("attention", "q gain", p.q_gain.len(), HEAD_DIM)?;
    check_window_len("attention", "k gain", p.k_gain.len(), HEAD_DIM)?;
    check_window_len("attention", "rope", p.rope.len(), rope_len)?;
    check_window_len("attention", "partials", p.partials.len(), partials)?;
    check_window_len("attention", "out", p.out.len(), out)?;
    match &p.cache {
        AttnCache::F16 { k, v } => {
            let n = f16_len(p.ctx).ok_or_else(|| refuse("the cache length wraps".to_string()))?;
            check_window_len("attention", "k cache", k.len(), n)?;
            check_window_len("attention", "v cache", v.len(), n)?;
            // The kernel reads each cache row as 16-byte words, one a lane:
            // the windows must start where those words align.
            for (name, w) in [("k cache", k), ("v cache", v)] {
                let range = w
                    .byte_range()
                    .map_err(|e| refuse(format!("the {name} window: {e}")))?;
                if range.start.get() % 16 != 0 {
                    return Err(refuse(format!(
                        "the {name} starts {start} bytes in, not on the 16-byte word the kernel                          reads",
                        start = range.start.get()
                    )));
                }
            }
        }
        AttnCache::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            let codes =
                q8_code_len(p.ctx).ok_or_else(|| refuse("the code length wraps".to_string()))?;
            let scales =
                q8_scale_len(p.ctx).ok_or_else(|| refuse("the scale length wraps".to_string()))?;
            check_window_len("attention", "k codes", k_codes.len(), codes)?;
            check_window_len("attention", "k scales", k_scales.len(), scales)?;
            check_window_len("attention", "v codes", v_codes.len(), codes)?;
            check_window_len("attention", "v scales", v_scales.len(), scales)?;
        }
    }
    // Each count is validated to fit the kernel's word; the flash's grid
    // takes the segment count itself.
    let segments = word("attention", "segments", p.segments)?;
    let ctx = word("attention", "ctx", p.ctx)?;
    let args = args_bytes([
        segments,
        ctx,
        p.layer,
        p.eps.to_bits(),
        SCALE.to_bits(),
        p.pos,
        p.n_keys,
    ]);
    Ok(Checked { args })
}

fn bind_flash(
    encoder: &mut ComputeEncoder<'_>,
    p: &AttentionParams<'_>,
    args: &[[u8; 4]; 7],
) -> Result<(), DeviceError> {
    encoder.set_binding(0, Binding::of(&p.q)?)?;
    encoder.set_binding(1, Binding::of(&p.k)?)?;
    encoder.set_binding(2, Binding::of(&p.v)?)?;
    encoder.set_binding(3, Binding::of(&p.q_gain)?)?;
    encoder.set_binding(4, Binding::of(&p.k_gain)?)?;
    encoder.set_binding(5, Binding::of(&p.rope)?)?;
    match &p.cache {
        AttnCache::F16 { k, v } => {
            encoder.set_binding(6, Binding::of_mut(k)?)?;
            encoder.set_binding(7, Binding::of_mut(v)?)?;
            encoder.set_binding(8, Binding::of_mut(&p.partials)?)?;
            encoder.set_binding(9, Binding::of_mut(&p.out)?)?;
            encoder.set_binding(10, Binding::bytes(pack(args))?)?;
            encoder.set_binding(11, p.fault.binding())?;
            encoder.set_binding(12, Binding::of_mut(&p.ticket)?)?;
        }
        AttnCache::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            encoder.set_binding(6, Binding::of_mut(k_codes)?)?;
            encoder.set_binding(7, Binding::of_mut(k_scales)?)?;
            encoder.set_binding(8, Binding::of_mut(v_codes)?)?;
            encoder.set_binding(9, Binding::of_mut(v_scales)?)?;
            encoder.set_binding(10, Binding::of_mut(&p.partials)?)?;
            encoder.set_binding(11, Binding::of_mut(&p.out)?)?;
            encoder.set_binding(12, Binding::bytes(pack(args))?)?;
            encoder.set_binding(13, p.fault.binding())?;
            encoder.set_binding(14, Binding::of_mut(&p.ticket)?)?;
        }
    }
    Ok(())
}

impl Kernels {
    /// Encode the flash. One dispatch, which past one segment also folds
    /// the partials and writes the gated output. A shape the kernel does
    /// not take, or a window that is not that shape, is refused by name and
    /// nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_flash(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionParams<'_>,
    ) -> Result<(), DeviceError> {
        let checked = checked(&p)?;
        let pipeline = match &p.cache {
            AttnCache::F16 { .. } => &self.attention.f16,
            AttnCache::Q8 { .. } => &self.attention.q8,
        };
        let groups = KV_HEADS
            .checked_mul(p.segments)
            .ok_or_else(|| refuse("the flash grid wraps".to_string()))?;
        encoder.set_pipeline(pipeline);
        bind_flash(encoder, &p, &checked.args)?;
        encoder.dispatch_groups(groups, 1, FLASH_THREADS)
    }
}

/// One attention launch over a chunk of `rows` causal queries, on the f16
/// cache. Row `r` sits at position `from + r`, appends its key and value
/// there and attends the `from + r + 1` keys up to and including it.
///
/// The windows with a stride hold `rows` rows that far apart; each may be
/// longer than the last row's end.
pub struct AttentionRowsParams<'w> {
    /// Head dimension. The kernel takes [`HEAD_DIM`].
    pub head_dim: usize,
    /// Query heads. The kernel takes [`QUERY_HEADS`].
    pub query_heads: usize,
    /// Key and value heads. The kernel takes [`KV_HEADS`].
    pub kv_heads: usize,
    /// Split-K segments of every row's flash, the same count the decode
    /// launch a row is held to uses.
    pub segments: usize,
    /// Cache rows per key head.
    pub ctx: usize,
    /// RMS epsilon.
    pub eps: f32,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// The position of row 0.
    pub from: u32,
    /// Rows in the chunk. Zero is refused, and so is a chunk that ends past
    /// `ctx`.
    pub rows: usize,
    /// The query projections, [`qg_len`] elements a row, `q_stride` apart.
    pub q: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `q`.
    pub q_stride: usize,
    /// Raw keys, [`KV_HEADS`] rows of [`HEAD_DIM`] a row, `k_stride` apart.
    pub k: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `k`.
    pub k_stride: usize,
    /// Raw values, laid out as `k`.
    pub v: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `v`.
    pub v_stride: usize,
    /// Query RMS gain, 256 values.
    pub q_gain: Window<'w, f32, MetalDevice>,
    /// Key RMS gain, 256 values.
    pub k_gain: Window<'w, f32, MetalDevice>,
    /// Rope table, `ctx` rows of [`ROT`] interleaved `(cos, sin)` pairs.
    pub rope: Window<'w, f32, MetalDevice>,
    /// The cache the launch appends and reads. The rows launches take the
    /// f16 cache; the q8_0 planes are refused by name.
    pub cache: AttnCache<'w>,
    /// Segment partials, [`partials_len`] elements a row.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The outputs after the gate, [`out_len`] elements a row, `out_stride`
    /// apart.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `out`.
    pub out_stride: usize,
    /// The fault word.
    pub fault: &'w MetalFaultWord,
    /// The folds' tickets, one word a row. Held zeroed; each row's fold
    /// resets its word, so the next launch finds them zero.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
}

const ROWS_WHAT: &str = "attention rows";

fn refuse_rows(detail: String) -> DeviceError {
    DeviceError::Range {
        what: ROWS_WHAT,
        detail,
    }
}

fn checked_rows(p: &AttentionRowsParams<'_>) -> Result<[[u8; 4]; 11], DeviceError> {
    if p.head_dim != HEAD_DIM {
        return Err(refuse_rows(format!(
            "head dim {}, the kernel takes {HEAD_DIM}",
            p.head_dim
        )));
    }
    if p.query_heads != QUERY_HEADS {
        return Err(refuse_rows(format!(
            "query heads {}, the kernel takes {QUERY_HEADS}",
            p.query_heads
        )));
    }
    if p.kv_heads != KV_HEADS {
        return Err(refuse_rows(format!(
            "kv heads {}, the kernel takes {KV_HEADS}",
            p.kv_heads
        )));
    }
    if p.segments == 0 {
        return Err(refuse_rows("segments of 0".to_string()));
    }
    if p.segments > SEGMENT_CAP {
        return Err(refuse_rows(format!(
            "segments of {} past the kernel's {SEGMENT_CAP}",
            p.segments
        )));
    }
    if p.ctx == 0 {
        return Err(refuse_rows("ctx of 0".to_string()));
    }
    if p.rows == 0 {
        return Err(refuse_rows("rows of 0".to_string()));
    }
    let rows = word(ROWS_WHAT, "rows", p.rows)?;
    let ctx = word(ROWS_WHAT, "ctx", p.ctx)?;
    match p.from.checked_add(rows) {
        Some(end) if end <= ctx => {}
        _ => {
            return Err(refuse_rows(format!(
                "positions {}..+{rows} pass the cache's ctx {ctx}",
                p.from
            )));
        }
    }
    let q_len = qg_len().ok_or_else(|| refuse_rows("the query length wraps".to_string()))?;
    let kv_len = KV_HEADS
        .checked_mul(HEAD_DIM)
        .ok_or_else(|| refuse_rows("the key length wraps".to_string()))?;
    let out = out_len().ok_or_else(|| refuse_rows("the output length wraps".to_string()))?;
    let rope_len = p
        .ctx
        .checked_mul(ROT)
        .ok_or_else(|| refuse_rows("the rope length wraps".to_string()))?;
    let partials_row = partials_len(p.segments)
        .ok_or_else(|| refuse_rows("the partial length wraps".to_string()))?;
    check_strided(ROWS_WHAT, "q", p.q.len(), p.rows, p.q_stride, q_len)?;
    check_strided(ROWS_WHAT, "k", p.k.len(), p.rows, p.k_stride, kv_len)?;
    check_strided(ROWS_WHAT, "v", p.v.len(), p.rows, p.v_stride, kv_len)?;
    check_strided(ROWS_WHAT, "out", p.out.len(), p.rows, p.out_stride, out)?;
    check_strided(
        ROWS_WHAT,
        "partials",
        p.partials.len(),
        p.rows,
        partials_row,
        partials_row,
    )?;
    check_strided(ROWS_WHAT, "ticket", p.ticket.len(), p.rows, 1, 1)?;
    let ticket = p
        .ticket
        .byte_range()
        .map_err(|e| refuse_rows(format!("the ticket window: {e}")))?;
    if ticket.start.get() % 4 != 0 {
        return Err(refuse_rows(
            "the ticket word is not 4-byte aligned".to_string(),
        ));
    }
    check_window_len(ROWS_WHAT, "q gain", p.q_gain.len(), HEAD_DIM)?;
    check_window_len(ROWS_WHAT, "k gain", p.k_gain.len(), HEAD_DIM)?;
    check_window_len(ROWS_WHAT, "rope", p.rope.len(), rope_len)?;
    match &p.cache {
        AttnCache::F16 { k, v } => {
            let n =
                f16_len(p.ctx).ok_or_else(|| refuse_rows("the cache length wraps".to_string()))?;
            check_window_len(ROWS_WHAT, "k cache", k.len(), n)?;
            check_window_len(ROWS_WHAT, "v cache", v.len(), n)?;
            for (name, w) in [("k cache", k), ("v cache", v)] {
                let range = w
                    .byte_range()
                    .map_err(|e| refuse_rows(format!("the {name} window: {e}")))?;
                if range.start.get() % 16 != 0 {
                    return Err(refuse_rows(format!(
                        "the {name} starts {} bytes in, not on the 16-byte word the kernel reads",
                        range.start.get()
                    )));
                }
            }
        }
        AttnCache::Q8 { .. } => {
            return Err(refuse_rows(
                "the rows launches take the f16 cache, not the q8_0 planes".to_string(),
            ));
        }
    }
    Ok(args_bytes([
        word(ROWS_WHAT, "segments", p.segments)?,
        ctx,
        p.layer,
        p.eps.to_bits(),
        SCALE.to_bits(),
        p.from,
        rows,
        word(ROWS_WHAT, "the q stride", p.q_stride)?,
        word(ROWS_WHAT, "the k stride", p.k_stride)?,
        word(ROWS_WHAT, "the v stride", p.v_stride)?,
        word(ROWS_WHAT, "the out stride", p.out_stride)?,
    ]))
}

impl Kernels {
    /// Encode the attention over a chunk of rows: the append of every row's
    /// key and value, a memory barrier, and the flash with a row grid
    /// dimension. A shape the kernels do not take, or a window that is not
    /// that shape, is refused by name and nothing is encoded. The two
    /// dispatches are one group each, the barrier the flash's order after the
    /// appends.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        let args = checked_rows(&p)?;
        let AttnCache::F16 {
            k: cache_k,
            v: cache_v,
        } = &p.cache
        else {
            unreachable!("checked_rows refused the q8_0 planes");
        };
        encoder.set_pipeline(&self.attention.append_rows);
        encoder.set_binding(0, Binding::of(&p.k)?)?;
        encoder.set_binding(1, Binding::of(&p.v)?)?;
        encoder.set_binding(2, Binding::of(&p.k_gain)?)?;
        encoder.set_binding(3, Binding::of(&p.rope)?)?;
        encoder.set_binding(4, Binding::of_mut(cache_k)?)?;
        encoder.set_binding(5, Binding::of_mut(cache_v)?)?;
        encoder.set_binding(6, Binding::bytes(pack(&args))?)?;
        encoder.set_binding(7, p.fault.binding())?;
        encoder.dispatch_groups(KV_HEADS, p.rows, SIMD_WIDTH)?;

        encoder.memory_barrier();
        let groups = KV_HEADS
            .checked_mul(p.segments)
            .ok_or_else(|| refuse_rows("the flash grid wraps".to_string()))?;
        encoder.set_pipeline(&self.attention.flash_rows);
        encoder.set_binding(0, Binding::of(&p.q)?)?;
        encoder.set_binding(1, Binding::of(&p.q_gain)?)?;
        encoder.set_binding(2, Binding::of(&p.rope)?)?;
        encoder.set_binding(3, Binding::of_mut(cache_k)?)?;
        encoder.set_binding(4, Binding::of_mut(cache_v)?)?;
        encoder.set_binding(5, Binding::of_mut(&p.partials)?)?;
        encoder.set_binding(6, Binding::of_mut(&p.out)?)?;
        encoder.set_binding(7, Binding::bytes(pack(&args))?)?;
        encoder.set_binding(8, p.fault.binding())?;
        encoder.set_binding(9, Binding::of_mut(&p.ticket)?)?;
        encoder.dispatch_groups(groups, p.rows, FLASH_THREADS)
    }
}
