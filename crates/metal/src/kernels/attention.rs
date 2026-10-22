//! The launch of decode GQA attention: 16 query heads, 2 key heads, head
//! dim 256. One threadgroup is one key head and one segment. It stages one
//! tile of 16 K/V rows and four SIMD groups read that tile.
//!
//! The re-fetch arm is function constant 0 (`attn_reread` in the kernel
//! header). The constant the host sets is the mat-vec SIMD-group slot: this
//! kernel's specialization uses 0 for the threadgroup tile, the shipped
//! default, and 1 for the re-fetch. Both arms are pipelines here.
//!
//! F′ is `queries = 1`. M is the same kernel with `queries = 2`. Split-K
//! writes a partial per segment and `gqa256_merge` folds them, including
//! when there is one segment.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{Library, Pipeline};

use super::{Kernels, check_window_len, word};

/// Values in one head.
pub const HEAD_DIM: usize = 256;
/// Query heads.
pub const QUERY_HEADS: usize = 16;
/// Key and value heads.
pub const KV_HEADS: usize = 2;
/// The partial rope: the first 64 values turn, in 32 pairs.
pub const ROT: usize = 64;
/// Keys in one K/V tile.
pub const TILE: usize = 16;
/// `1/sqrt(256)`, exact.
pub const SCALE: f32 = 0.0625;
/// One segment partial: the max, the weight sum, then the head.
pub const PARTIAL_ELEMS: usize = HEAD_DIM + 2;

const FLASH_THREADS: usize = 128;
const MERGE_THREADS: usize = 32;

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

/// One attention launch. `queries` is 1 or 2. `pos[1]` and `n_keys[1]` are
/// the second query's, read when `queries` is 2. A key count of 0 or past
/// `ctx`, and an append position at or past `ctx`, are the kernel's faults,
/// not refusals.
pub struct AttentionParams<'w> {
    /// Head dimension. The kernel takes [`HEAD_DIM`].
    pub head_dim: usize,
    /// Query heads. The kernel takes [`QUERY_HEADS`].
    pub query_heads: usize,
    /// Key and value heads. The kernel takes [`KV_HEADS`].
    pub kv_heads: usize,
    /// Query positions in the tile: 1 (F′) or 2 (M).
    pub queries: usize,
    /// Split-K segments. One still writes a partial and merges it.
    pub segments: usize,
    /// Cache rows per key head.
    pub ctx: usize,
    /// RMS epsilon.
    pub eps: f32,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// Append position of each query.
    pub pos: [u32; 2],
    /// Live key count of each query, including the appended row when it is live.
    pub n_keys: [u32; 2],
    /// The re-fetch arm. The shipped launch leaves this false.
    pub reread: bool,
    /// Queries, `[queries, 16, 256]`.
    pub q: Window<'w, f32, MetalDevice>,
    /// Keys, `[queries, 2, 256]`, raw.
    pub k: Window<'w, f32, MetalDevice>,
    /// Values, `[queries, 2, 256]`, raw.
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
    /// Attention, [`out_len`] elements. The context, not a gated output.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word.
    pub fault: &'w MetalFaultWord,
}

/// The partial buffer's length: one partial per query, head and segment.
#[must_use]
pub fn partials_len(queries: usize, segments: usize) -> Option<usize> {
    queries
        .checked_mul(QUERY_HEADS)?
        .checked_mul(segments)?
        .checked_mul(PARTIAL_ELEMS)
}

/// The attention buffer's length.
#[must_use]
pub fn out_len(queries: usize) -> Option<usize> {
    queries.checked_mul(QUERY_HEADS)?.checked_mul(HEAD_DIM)
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
    f16_reread: Pipeline,
    q8: Pipeline,
    q8_reread: Pipeline,
    merge: Pipeline,
}

impl AttentionPipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            f16: library.pipeline_with_simdgroups("gqa256_flash_f16", 0)?,
            f16_reread: library.pipeline_with_simdgroups("gqa256_flash_f16", 1)?,
            q8: library.pipeline_with_simdgroups("gqa256_flash_q8", 0)?,
            q8_reread: library.pipeline_with_simdgroups("gqa256_flash_q8", 1)?,
            merge: library.pipeline("gqa256_merge")?,
        })
    }
}

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "attention",
        detail,
    }
}

fn pack_args(words: [u32; 10]) -> [u8; 40] {
    let mut out = [0u8; 40];
    for (i, word) in words.into_iter().enumerate() {
        let at = i * 4;
        out[at..at + 4].copy_from_slice(&word.to_ne_bytes());
    }
    out
}

impl Kernels {
    /// Encode the flash dispatch and the merge. A shape the kernel does not
    /// take, or a window that is not that shape, is refused by name and
    /// nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionParams<'_>,
    ) -> Result<(), DeviceError> {
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
        if p.queries != 1 && p.queries != 2 {
            return Err(refuse(format!(
                "{} queries, the kernel takes 1 or 2",
                p.queries
            )));
        }
        if p.segments == 0 {
            return Err(refuse("segments of 0".to_string()));
        }
        if p.ctx == 0 {
            return Err(refuse("ctx of 0".to_string()));
        }
        let q_len =
            out_len(p.queries).ok_or_else(|| refuse("the query length wraps".to_string()))?;
        let kv_len = p
            .queries
            .checked_mul(KV_HEADS)
            .and_then(|n| n.checked_mul(HEAD_DIM))
            .ok_or_else(|| refuse("the key length wraps".to_string()))?;
        let rope_len = p
            .ctx
            .checked_mul(ROT)
            .ok_or_else(|| refuse("the rope length wraps".to_string()))?;
        let partials = partials_len(p.queries, p.segments)
            .ok_or_else(|| refuse("the partial length wraps".to_string()))?;
        check_window_len("attention", "q", p.q.len(), q_len)?;
        check_window_len("attention", "k", p.k.len(), kv_len)?;
        check_window_len("attention", "v", p.v.len(), kv_len)?;
        check_window_len("attention", "q gain", p.q_gain.len(), HEAD_DIM)?;
        check_window_len("attention", "k gain", p.k_gain.len(), HEAD_DIM)?;
        check_window_len("attention", "rope", p.rope.len(), rope_len)?;
        check_window_len("attention", "partials", p.partials.len(), partials)?;
        check_window_len("attention", "out", p.out.len(), q_len)?;
        match &p.cache {
            AttnCache::F16 { k, v } => {
                let n =
                    f16_len(p.ctx).ok_or_else(|| refuse("the cache length wraps".to_string()))?;
                check_window_len("attention", "k cache", k.len(), n)?;
                check_window_len("attention", "v cache", v.len(), n)?;
            }
            AttnCache::Q8 {
                k_codes,
                k_scales,
                v_codes,
                v_scales,
            } => {
                let codes = q8_code_len(p.ctx)
                    .ok_or_else(|| refuse("the code length wraps".to_string()))?;
                let scales = q8_scale_len(p.ctx)
                    .ok_or_else(|| refuse("the scale length wraps".to_string()))?;
                check_window_len("attention", "k codes", k_codes.len(), codes)?;
                check_window_len("attention", "k scales", k_scales.len(), scales)?;
                check_window_len("attention", "v codes", v_codes.len(), codes)?;
                check_window_len("attention", "v scales", v_scales.len(), scales)?;
            }
        }

        let queries = word("attention", "queries", p.queries)?;
        let segments = word("attention", "segments", p.segments)?;
        let ctx = word("attention", "ctx", p.ctx)?;
        let args = pack_args([
            queries,
            segments,
            ctx,
            p.layer,
            p.eps.to_bits(),
            SCALE.to_bits(),
            p.pos[0],
            p.pos[1],
            p.n_keys[0],
            p.n_keys[1],
        ]);
        let flash = match (&p.cache, p.reread) {
            (AttnCache::F16 { .. }, false) => &self.attention.f16,
            (AttnCache::F16 { .. }, true) => &self.attention.f16_reread,
            (AttnCache::Q8 { .. }, false) => &self.attention.q8,
            (AttnCache::Q8 { .. }, true) => &self.attention.q8_reread,
        };
        let flash_threads = KV_HEADS
            .checked_mul(p.segments)
            .and_then(|n| n.checked_mul(FLASH_THREADS))
            .ok_or_else(|| refuse("the flash grid wraps".to_string()))?;
        let merge_threads = p
            .queries
            .checked_mul(QUERY_HEADS)
            .and_then(|n| n.checked_mul(MERGE_THREADS))
            .ok_or_else(|| refuse("the merge grid wraps".to_string()))?;

        encoder.set_pipeline(flash);
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
                encoder.set_binding(10, Binding::bytes(&args)?)?;
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
                encoder.set_binding(12, Binding::bytes(&args)?)?;
                encoder.set_binding(13, p.fault.binding())?;
            }
        }
        encoder.dispatch(flash_threads, FLASH_THREADS)?;

        encoder.set_pipeline(&self.attention.merge);
        encoder.set_binding(0, Binding::of_mut(&p.partials)?)?;
        encoder.set_binding(1, Binding::of_mut(&p.out)?)?;
        encoder.set_binding(2, Binding::bytes(&args)?)?;
        encoder.dispatch(merge_threads, MERGE_THREADS)
    }
}
