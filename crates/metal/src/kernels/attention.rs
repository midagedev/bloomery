//! The launch of decode GQA attention: 16 query heads, 2 key heads, head
//! dim 256. One threadgroup is one key head and one segment. It stages one
//! tile of 16 K/V rows and four SIMD groups read that tile.
//!
//! One segment is the flash: it normalizes, applies the gate and writes the
//! output. More than one segment is the flash, which writes partials, and
//! then the merge, which normalizes, applies the gate and writes the output.
//! The caller places the barrier between those two dispatches. Both read
//! [`attention_dispatches`].

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{Library, Pipeline};

use super::{Kernels, args_bytes, check_window_len, pack, word};

use super::attention_geom::{FLASH_THREADS, MERGE_THREADS, PARTIAL_ELEMS};
pub use super::attention_geom::{
    HEAD_DIM, KV_HEADS, QG_STRIDE, QUERY_HEADS, ROT, SCALE, SEGMENT_KEYS, TILE,
    attention_dispatches, segments_for_keys,
};

/// Every f16 bit pattern, one f32 each.
const F16_PATTERNS: usize = 65_536;
/// Threads in the pattern dispatch.
const EXPAND_THREADS: usize = 256;

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
    merge: Pipeline,
    expand: Pipeline,
}

impl AttentionPipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            f16: library.pipeline("gqa256_flash_f16")?,
            q8: library.pipeline("gqa256_flash_q8")?,
            merge: library.pipeline("gqa256_merge")?,
            expand: library.pipeline("gqa256_f16_expand")?,
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
    segments: u32,
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
    Ok(Checked { segments, args })
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
        }
    }
    Ok(())
}

impl Kernels {
    /// Encode the flash. One dispatch. A shape the kernel does not take, or
    /// a window that is not that shape, is refused by name and nothing is
    /// encoded. One segment writes the gated output; more than one writes
    /// partials and leaves the output to [`Self::attention_merge`].
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

    /// Encode the merge. One dispatch. One segment is refused by name: that
    /// launch finishes in the flash, and this call encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_merge(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionParams<'_>,
    ) -> Result<(), DeviceError> {
        let checked = checked(&p)?;
        if attention_dispatches(checked.segments) == 1 {
            return Err(refuse(
                "one segment finishes in the flash, and the merge is not launched".to_string(),
            ));
        }
        let groups = QUERY_HEADS;
        encoder.set_pipeline(&self.attention.merge);
        encoder.set_binding(0, Binding::of_mut(&p.partials)?)?;
        encoder.set_binding(1, Binding::of_mut(&p.out)?)?;
        encoder.set_binding(2, Binding::of(&p.q)?)?;
        encoder.set_binding(3, Binding::bytes(pack(&checked.args))?)?;
        encoder.set_binding(4, p.fault.binding())?;
        encoder.dispatch_groups(groups, 1, MERGE_THREADS)
    }

    /// Encode the f16-pattern expansion. The window holds every pattern.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_f16_expand(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        out: WindowMut<'_, f32, MetalDevice>,
    ) -> Result<(), DeviceError> {
        check_window_len("attention", "f16 patterns", out.len(), F16_PATTERNS)?;
        encoder.set_pipeline(&self.attention.expand);
        encoder.set_binding(0, Binding::of_mut(&out)?)?;
        encoder.dispatch_groups(F16_PATTERNS / EXPAND_THREADS, 1, EXPAND_THREADS)
    }
}
