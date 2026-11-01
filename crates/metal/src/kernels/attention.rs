//! The launch of decode GQA attention at the library's instance geometry
//! (`AttnGeom`: its query heads over its key heads, its values a head).
//!
//! One segment of the f16 cache is the one-pass flash: one threadgroup a
//! query head, its SIMD groups taking the keys strided, merged in
//! threadgroup memory; it normalizes, applies the gate when the layer's query
//! carries one (`AttnGeom::qg_stride` is twice the head) and writes the
//! output, and writes no partial. One segment of the q8_0 cache is the
//! segmented flash below with one segment, which ends the same way.
//!
//! Past one segment, one threadgroup is one key head and one segment, one
//! query head a SIMD group, the keys streamed from the cache into registers.
//! Each writes partials, and the last threadgroup of each key head — its
//! ticket word's — folds that key head's partials, applies the gate and writes
//! its query heads' output in the same launch, the key heads at once. The
//! ticket is one word a key head the caller holds zeroed; each fold resets its
//! word, so the next launch finds them zero. There is no merge dispatch.
//!
//! [`Kernels::attention_rows`] is the same attention over a chunk of rows,
//! each row a causal query at its own position: the append of the chunk's
//! keys and values, then the flash with a row grid dimension. Each row is the
//! bits of the decode launch at its position with the same segment count.
//!
//! [`Kernels::attention_prefill`] is the same chunk run by the prefill
//! kernel: simdgroup matrices over tiles of keys, held to the decode launch by
//! a band rather than by bits.
//!
//! What differs layer to layer inside one instance is [`AttnLayer`]: the
//! window the layer attends (the last `window` of its live keys; a layer with
//! none attends every key) and, for an instance whose output gate is one value
//! a query head, the gate's window. The launches without a layer argument
//! launch [`AttnLayer::full`]. The keys of a windowed layer are the last
//! `window` of the live ones, so its split-K segments are those of
//! [`segments_for_window`]: a segment holding no key of the window is never
//! launched, and one a caller counts anyway is empty and skipped by the fold.
//! The store is the planar cache of every position; the window only cuts what
//! the launch reads.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;
use std::sync::OnceLock;

use crate::library::{Library, Pipeline};

use super::rows::check_strided;
use super::{Kernels, args_bytes, check_window_len, word};
use crate::geom::{AttnGeom, HOST_GEOMETRY};

pub use super::attention_geom::segments_for_window;
pub use super::attention_geom::{
    HEAD_DIM, KV_HEADS, ONE_PASS_KEYS, QG_STRIDE, QUERY_HEADS, ROT, SCALE, SEGMENT_CAP,
    SEGMENT_KEYS, TILE, segments_for_keys,
};
use super::attention_geom::{ONE_PASS_THREADS, PREFILL_POSITIONS, PREFILL_THREADS, SIMD_WIDTH};

/// The f16 cache or the q8_0 planes a launch reads and appends.
pub enum AttnCache<'w> {
    /// `[kv][ctx][head]` f16 bits.
    F16 {
        /// Key rows.
        k: WindowMut<'w, u16, MetalDevice>,
        /// Value rows.
        v: WindowMut<'w, u16, MetalDevice>,
    },
    /// q8_0 planes: `head / 4` code words and `head / 32` scales per row.
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

/// What one layer adds to an attention launch beyond the instance's shape.
pub struct AttnLayer<'w> {
    /// The layer attends the last `window` of its live keys; `None` attends
    /// every key. Only an instance whose flash row cuts a window
    /// (`AttnGeom::windowed`) takes one, and a window of 0 keys is refused.
    pub window: Option<u32>,
    /// The raw per-head gate values, `[rows][query heads]`, before the
    /// sigmoid. Held exactly when the instance's output gate is one value a
    /// query head (`AttnGeom::head_gate`).
    pub head_gate: Option<Window<'w, f32, MetalDevice>>,
}

impl AttnLayer<'_> {
    /// The layer of a launch with no layer argument: every key, no per-head
    /// gate.
    #[must_use]
    pub const fn full() -> Self {
        AttnLayer {
            window: None,
            head_gate: None,
        }
    }
}

/// One attention launch. A key count of 0 or past `ctx`, and an append
/// position at or past `ctx`, are the kernel's faults, not refusals.
///
/// `q` is `[query heads, qg_stride]`: the query, then the gate when the
/// layer has one, per head.
pub struct AttentionParams<'w> {
    /// Head dimension. The kernel takes the instance's (`AttnGeom::head`).
    pub head_dim: usize,
    /// Query heads. The kernel takes the instance's (`AttnGeom::query_heads`).
    pub query_heads: usize,
    /// Key and value heads. The kernel takes the instance's
    /// (`AttnGeom::kv_heads`).
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
    /// The query projection, `AttnGeom::qg_len` elements: query, then any
    /// gate, per head.
    pub q: Window<'w, f32, MetalDevice>,
    /// Keys, one raw row per key head.
    pub k: Window<'w, f32, MetalDevice>,
    /// Values, one raw row per key head.
    pub v: Window<'w, f32, MetalDevice>,
    /// Query RMS gain, one value per head value.
    pub q_gain: Window<'w, f32, MetalDevice>,
    /// Key RMS gain, one value per head value.
    pub k_gain: Window<'w, f32, MetalDevice>,
    /// Rope table, `ctx` rows of `AttnGeom::rot` interleaved `(cos, sin)`
    /// pairs.
    pub rope: Window<'w, f32, MetalDevice>,
    /// The cache the launch reads and appends.
    pub cache: AttnCache<'w>,
    /// Segment partials, `AttnGeom::partials_len` elements.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The layer output, `AttnGeom::out_len` elements, after any gate.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word.
    pub fault: &'w MetalFaultWord,
    /// The folds' tickets, one word a key head (`AttnGeom::kv_heads`). Held
    /// zeroed; each key head's fold resets its word, so the next launch finds
    /// them zero. Bound only past one segment.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
}

impl AttnGeom {
    /// The partial buffer's length: one partial per head and segment.
    #[must_use]
    pub fn partials_len(&self, segments: usize) -> Option<usize> {
        self.query_heads
            .checked_mul(segments)?
            .checked_mul(self.partial_elems)
    }

    /// The attention output's length.
    #[must_use]
    pub fn out_len(&self) -> Option<usize> {
        self.query_heads.checked_mul(self.head)
    }

    /// The query projection's length: the query and any gate, per head.
    #[must_use]
    pub fn qg_len(&self) -> Option<usize> {
        self.query_heads.checked_mul(self.qg_stride)
    }

    /// f16 cache rows, `kv * ctx * head` bits.
    #[must_use]
    pub fn f16_len(&self, ctx: usize) -> Option<usize> {
        self.kv_heads.checked_mul(ctx)?.checked_mul(self.head)
    }

    /// q8_0 code words per cache, `kv * ctx * head / 4`.
    #[must_use]
    pub fn q8_code_len(&self, ctx: usize) -> Option<usize> {
        self.kv_heads.checked_mul(ctx)?.checked_mul(self.head / 4)
    }

    /// q8_0 scales per cache, `kv * ctx * head / 32`.
    #[must_use]
    pub fn q8_scale_len(&self, ctx: usize) -> Option<usize> {
        self.kv_heads.checked_mul(ctx)?.checked_mul(self.head / 32)
    }
}

/// The partial buffer's length: one partial per head and segment. A shim: the
/// build's host instance's (`AttnGeom::partials_len`). Callers left: the arch,
/// the kernel gate and the probes.
#[must_use]
pub fn partials_len(segments: usize) -> Option<usize> {
    HOST_GEOMETRY.attn.partials_len(segments)
}

/// The attention output's length. A shim: the build's host instance's
/// (`AttnGeom::out_len`). Callers left: the arch, the kernel gate and the
/// probes.
#[must_use]
pub fn out_len() -> Option<usize> {
    HOST_GEOMETRY.attn.out_len()
}

/// The query projection's length: the query and any gate, per head. A shim:
/// the build's host instance's (`AttnGeom::qg_len`). Callers left: the arch,
/// the kernel gate and the probes.
#[must_use]
pub fn qg_len() -> Option<usize> {
    HOST_GEOMETRY.attn.qg_len()
}

/// f16 cache rows, `kv * ctx * head` bits. A shim: the build's host
/// instance's (`AttnGeom::f16_len`). Callers left: the arch, the kernel gate
/// and the probes.
#[must_use]
pub fn f16_len(ctx: usize) -> Option<usize> {
    HOST_GEOMETRY.attn.f16_len(ctx)
}

/// q8_0 code words per cache, `kv * ctx * head / 4`. A shim: the build's host
/// instance's (`AttnGeom::q8_code_len`). Callers left: the kernel gate.
#[must_use]
pub fn q8_code_len(ctx: usize) -> Option<usize> {
    HOST_GEOMETRY.attn.q8_code_len(ctx)
}

/// q8_0 scales per cache, `kv * ctx * head / 32`. A shim: the build's host
/// instance's (`AttnGeom::q8_scale_len`). Callers left: the kernel gate.
#[must_use]
pub fn q8_scale_len(ctx: usize) -> Option<usize> {
    HOST_GEOMETRY.attn.q8_scale_len(ctx)
}

pub(crate) struct AttentionPipelines {
    f16: Pipeline,
    q8: Pipeline,
    append_rows: Pipeline,
    flash_rows: Pipeline,
    /// The f16 launch of one segment: one threadgroup a query head.
    one: Pipeline,
    /// The rows launch of one segment on the one-pass flash.
    one_rows: Pipeline,
    /// The prefill flash is built on its first launch: its registers spill to
    /// a scratch buffer the device allocates when the pipeline first runs, and
    /// a decode-only session never pays it.
    prefill: OnceLock<Pipeline>,
    library: Library,
}

impl AttentionPipelines {
    /// The prefill flash's pipeline, built by the first call.
    fn prefill(&self) -> Result<&Pipeline, DeviceError> {
        if let Some(pipeline) = self.prefill.get() {
            return Ok(pipeline);
        }
        let built = self.library.pipeline("attn_prefill_f16")?;
        Ok(self.prefill.get_or_init(|| built))
    }

    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            f16: library.pipeline("gqa256_flash_f16")?,
            q8: library.pipeline("gqa256_flash_q8")?,
            append_rows: library.pipeline("attn_norm_rope_rows")?,
            flash_rows: library.pipeline("gqa256_flash_f16_rows")?,
            one: library.pipeline("attn_flash_one_f16")?,
            one_rows: library.pipeline("attn_flash_one_f16_rows")?,
            prefill: OnceLock::new(),
            library: library.clone(),
        })
    }
}

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "attention",
        detail,
    }
}

/// The window word `layer` launches, refused by name when the instance cannot
/// take the layer: a window its flash row does not cut or of no keys, a
/// per-head gate window the instance has none for, or none where it gates per
/// head, or one of another length than `rows` rows of the query heads.
fn check_layer(
    what: &'static str,
    g: &AttnGeom,
    layer: &AttnLayer<'_>,
    rows: usize,
) -> Result<u32, DeviceError> {
    let refuse = |detail: String| refuse_in(what, detail);
    let window = match layer.window {
        None => 0,
        Some(0) => {
            return Err(refuse(
                "a window of 0 keys: a layer attends its own key at least, and none is None"
                    .to_string(),
            ));
        }
        Some(w) if g.windowed => w,
        Some(w) => {
            return Err(refuse(format!(
                "a window of {w} keys, the instance's flash row cuts no window"
            )));
        }
    };
    match (&layer.head_gate, g.head_gate) {
        (Some(gate), true) => {
            let need = rows
                .checked_mul(g.query_heads)
                .ok_or_else(|| refuse("the head gate length wraps".to_string()))?;
            check_window_len(what, "head gate", gate.len(), need)?;
        }
        (None, true) => {
            return Err(refuse(
                "the instance gates per head, the launch holds no head gate window".to_string(),
            ));
        }
        (Some(_), false) => {
            return Err(refuse(
                "a head gate window, the instance has no per-head gate".to_string(),
            ));
        }
        (None, false) => {}
    }
    Ok(window)
}

/// The most argument words an attention launch sends: the rows launches'
/// eleven and the layer's window.
const MAX_ARG_WORDS: usize = 12;

/// An attention launch's argument words, each in the machine's byte order:
/// the launch sends `words[..len]`.
struct LaunchArgs {
    words: [[u8; 4]; MAX_ARG_WORDS],
    len: usize,
}

impl LaunchArgs {
    /// The bytes the launch binds.
    fn bytes(&self) -> &[u8] {
        self.words[..self.len].as_flattened()
    }
}

/// The launch's argument words, then the layer's window when the instance's
/// flash takes one.
fn with_window<const N: usize>(words: [[u8; 4]; N], g: &AttnGeom, window: u32) -> LaunchArgs {
    const { assert!(N < MAX_ARG_WORDS, "no room for the window word") };
    let mut out = [[0; 4]; MAX_ARG_WORDS];
    out[..N].copy_from_slice(&words);
    let mut len = N;
    if g.windowed {
        out[len] = window.to_ne_bytes();
        len += 1;
    }
    LaunchArgs { words: out, len }
}

struct Checked {
    args: LaunchArgs,
}

fn checked(
    g: &AttnGeom,
    p: &AttentionParams<'_>,
    layer: &AttnLayer<'_>,
) -> Result<Checked, DeviceError> {
    if p.head_dim != g.head {
        return Err(refuse(format!(
            "head dim {}, the kernel takes {}",
            p.head_dim, g.head
        )));
    }
    if p.query_heads != g.query_heads {
        return Err(refuse(format!(
            "query heads {}, the kernel takes {}",
            p.query_heads, g.query_heads
        )));
    }
    if p.kv_heads != g.kv_heads {
        return Err(refuse(format!(
            "kv heads {}, the kernel takes {}",
            p.kv_heads, g.kv_heads
        )));
    }
    if p.segments == 0 {
        return Err(refuse("segments of 0".to_string()));
    }
    if p.ctx == 0 {
        return Err(refuse("ctx of 0".to_string()));
    }
    let q_len = g
        .qg_len()
        .ok_or_else(|| refuse("the query length wraps".to_string()))?;
    let kv_len = g
        .kv_heads
        .checked_mul(g.head)
        .ok_or_else(|| refuse("the key length wraps".to_string()))?;
    let rope_len = p
        .ctx
        .checked_mul(g.rot)
        .ok_or_else(|| refuse("the rope length wraps".to_string()))?;
    let partials = g
        .partials_len(p.segments)
        .ok_or_else(|| refuse("the partial length wraps".to_string()))?;
    if p.segments > SEGMENT_CAP {
        return Err(refuse(format!(
            "segments of {} past the kernel's {SEGMENT_CAP}",
            p.segments
        )));
    }
    check_window_len("attention", "ticket", p.ticket.len(), g.kv_heads)?;
    let ticket = p
        .ticket
        .byte_range()
        .map_err(|e| refuse(format!("the ticket window: {e}")))?;
    if ticket.start.get() % 4 != 0 {
        return Err(refuse("the ticket word is not 4-byte aligned".to_string()));
    }
    let out = g
        .out_len()
        .ok_or_else(|| refuse("the output length wraps".to_string()))?;
    check_window_len("attention", "q", p.q.len(), q_len)?;
    check_window_len("attention", "k", p.k.len(), kv_len)?;
    check_window_len("attention", "v", p.v.len(), kv_len)?;
    check_window_len("attention", "q gain", p.q_gain.len(), g.head)?;
    check_window_len("attention", "k gain", p.k_gain.len(), g.head)?;
    check_window_len("attention", "rope", p.rope.len(), rope_len)?;
    check_window_len("attention", "partials", p.partials.len(), partials)?;
    check_window_len("attention", "out", p.out.len(), out)?;
    match &p.cache {
        AttnCache::F16 { k, v } => {
            let n = g
                .f16_len(p.ctx)
                .ok_or_else(|| refuse("the cache length wraps".to_string()))?;
            check_window_len("attention", "k cache", k.len(), n)?;
            check_window_len("attention", "v cache", v.len(), n)?;
            // The kernels read each cache row as lane words of up to 16 bytes
            // (the prefill's are 16): the windows must start where those
            // words align.
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
            let codes = g
                .q8_code_len(p.ctx)
                .ok_or_else(|| refuse("the code length wraps".to_string()))?;
            let scales = g
                .q8_scale_len(p.ctx)
                .ok_or_else(|| refuse("the scale length wraps".to_string()))?;
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
    let window = check_layer("attention", g, layer, 1)?;
    let args = args_bytes([
        segments,
        ctx,
        p.layer,
        p.eps.to_bits(),
        g.scale.to_bits(),
        p.pos,
        p.n_keys,
    ]);
    Ok(Checked {
        args: with_window(args, g, window),
    })
}

fn bind_flash(
    encoder: &mut ComputeEncoder<'_>,
    p: &AttentionParams<'_>,
    args: &[u8],
    layer: &AttnLayer<'_>,
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
            encoder.set_binding(10, Binding::bytes(args)?)?;
            encoder.set_binding(11, p.fault.binding())?;
            encoder.set_binding(12, Binding::of_mut(&p.ticket)?)?;
            if let Some(gate) = &layer.head_gate {
                encoder.set_binding(13, Binding::of(gate)?)?;
            }
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
            encoder.set_binding(12, Binding::bytes(args)?)?;
            encoder.set_binding(13, p.fault.binding())?;
            encoder.set_binding(14, Binding::of_mut(&p.ticket)?)?;
            if let Some(gate) = &layer.head_gate {
                encoder.set_binding(15, Binding::of(gate)?)?;
            }
        }
    }
    Ok(())
}

impl Kernels {
    /// The pipelines this object builds on first launch, built now: the
    /// move proof's scan lists every pipeline a run can need.
    #[cfg(feature = "pipeline-scan")]
    pub fn build_deferred(&self) -> Result<(), DeviceError> {
        self.attention.prefill().map(|_| ())
    }

    /// Encode the flash of a layer that attends every key. An instance that
    /// gates per head refuses it by name: its launches name the gate
    /// ([`Kernels::attention_flash_layer`]).
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_flash(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionParams<'_>,
    ) -> Result<(), DeviceError> {
        self.attention_flash_layer(encoder, p, AttnLayer::full())
    }

    /// Encode the flash. One dispatch, which past one segment also folds
    /// the partials and writes the gated output. A shape the kernel does
    /// not take, or a window that is not that shape, is refused by name and
    /// nothing is encoded. `layer` names the keys the layer attends and its
    /// per-head gate.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_flash_layer(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionParams<'_>,
        layer: AttnLayer<'_>,
    ) -> Result<(), DeviceError> {
        let g = &self.geom.attn;
        let checked = checked(g, &p, &layer)?;
        let split = || {
            g.kv_heads
                .checked_mul(p.segments)
                .ok_or_else(|| refuse("the flash grid wraps".to_string()))
        };
        // One segment of the f16 cache is the one-pass flash, a threadgroup a
        // query head; the rest are a threadgroup a (key head, segment).
        let (pipeline, groups, threads) = match &p.cache {
            AttnCache::F16 { .. } if p.segments == 1 => {
                (&self.attention.one, g.query_heads, ONE_PASS_THREADS)
            }
            AttnCache::F16 { .. } => (&self.attention.f16, split()?, g.flash_threads),
            AttnCache::Q8 { .. } => (&self.attention.q8, split()?, g.flash_threads),
        };
        encoder.set_pipeline(pipeline);
        bind_flash(encoder, &p, checked.args.bytes(), &layer)?;
        encoder.dispatch_groups(groups, 1, threads)
    }
}

/// One attention launch over a chunk of `rows` causal queries, on the f16
/// cache. Row `r` sits at position `from + r`, appends its key and value
/// there and attends the `from + r + 1` keys up to and including it.
///
/// The windows with a stride hold `rows` rows that far apart; each may be
/// longer than the last row's end.
pub struct AttentionRowsParams<'w> {
    /// Head dimension. The kernel takes the instance's (`AttnGeom::head`).
    pub head_dim: usize,
    /// Query heads. The kernel takes the instance's (`AttnGeom::query_heads`).
    pub query_heads: usize,
    /// Key and value heads. The kernel takes the instance's
    /// (`AttnGeom::kv_heads`).
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
    /// The query projections, `AttnGeom::qg_len` elements a row, `q_stride`
    /// apart.
    pub q: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `q`.
    pub q_stride: usize,
    /// Raw keys, `AttnGeom::kv_heads` rows of the head's values a row,
    /// `k_stride` apart.
    pub k: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `k`.
    pub k_stride: usize,
    /// Raw values, laid out as `k`.
    pub v: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `v`.
    pub v_stride: usize,
    /// Query RMS gain, one value per head value.
    pub q_gain: Window<'w, f32, MetalDevice>,
    /// Key RMS gain, one value per head value.
    pub k_gain: Window<'w, f32, MetalDevice>,
    /// Rope table, `ctx` rows of `AttnGeom::rot` interleaved `(cos, sin)`
    /// pairs.
    pub rope: Window<'w, f32, MetalDevice>,
    /// The cache the launch appends and reads. The rows launches take the
    /// f16 cache; the q8_0 planes are refused by name.
    pub cache: AttnCache<'w>,
    /// Segment partials, `AttnGeom::partials_len` elements a row.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The outputs after any gate, `AttnGeom::out_len` elements a row,
    /// `out_stride` apart.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `out`.
    pub out_stride: usize,
    /// The fault word.
    pub fault: &'w MetalFaultWord,
    /// The folds' tickets, one word a key head a row (`AttnGeom::kv_heads`
    /// words a row, row-major). Held zeroed; each fold resets its word, so the
    /// next launch finds them zero.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
}

const ROWS_WHAT: &str = "attention rows";
const PREFILL_WHAT: &str = "attention prefill";

fn refuse_in(what: &'static str, detail: String) -> DeviceError {
    DeviceError::Range { what, detail }
}

/// What the two chunk launches share: the shape and the windows of a chunk
/// of causal queries on the f16 cache.
struct ChunkView<'a, 'w> {
    what: &'static str,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    ctx: usize,
    from: u32,
    rows: usize,
    q: &'a Window<'w, f32, MetalDevice>,
    q_stride: usize,
    k: &'a Window<'w, f32, MetalDevice>,
    k_stride: usize,
    v: &'a Window<'w, f32, MetalDevice>,
    v_stride: usize,
    q_gain: &'a Window<'w, f32, MetalDevice>,
    k_gain: &'a Window<'w, f32, MetalDevice>,
    rope: &'a Window<'w, f32, MetalDevice>,
    cache: &'a AttnCache<'w>,
    out: &'a WindowMut<'w, f32, MetalDevice>,
    out_stride: usize,
}

impl ChunkView<'_, '_> {
    /// The rows launches' argument words, in `AttnRowsArgs`' order, then the
    /// layer's window when the instance's flash takes one.
    fn words(
        &self,
        g: &AttnGeom,
        segments: usize,
        layer: u32,
        eps: f32,
        window: u32,
    ) -> Result<LaunchArgs, DeviceError> {
        let what = self.what;
        let words = args_bytes([
            word(what, "segments", segments)?,
            word(what, "ctx", self.ctx)?,
            layer,
            eps.to_bits(),
            g.scale.to_bits(),
            self.from,
            word(what, "rows", self.rows)?,
            word(what, "the q stride", self.q_stride)?,
            word(what, "the k stride", self.k_stride)?,
            word(what, "the v stride", self.v_stride)?,
            word(what, "the out stride", self.out_stride)?,
        ]);
        Ok(with_window(words, g, window))
    }
}

/// The shape, the windows and the cache of a chunk, refused by name.
fn check_chunk(g: &AttnGeom, c: &ChunkView<'_, '_>) -> Result<(), DeviceError> {
    let what = c.what;
    let refuse = |detail: String| refuse_in(what, detail);
    if c.head_dim != g.head {
        return Err(refuse(format!(
            "head dim {}, the kernel takes {}",
            c.head_dim, g.head
        )));
    }
    if c.query_heads != g.query_heads {
        return Err(refuse(format!(
            "query heads {}, the kernel takes {}",
            c.query_heads, g.query_heads
        )));
    }
    if c.kv_heads != g.kv_heads {
        return Err(refuse(format!(
            "kv heads {}, the kernel takes {}",
            c.kv_heads, g.kv_heads
        )));
    }
    if c.ctx == 0 {
        return Err(refuse("ctx of 0".to_string()));
    }
    if c.rows == 0 {
        return Err(refuse("rows of 0".to_string()));
    }
    let rows = word(what, "rows", c.rows)?;
    let ctx = word(what, "ctx", c.ctx)?;
    match c.from.checked_add(rows) {
        Some(end) if end <= ctx => {}
        _ => {
            return Err(refuse(format!(
                "positions {}..+{rows} pass the cache's ctx {ctx}",
                c.from
            )));
        }
    }
    let q_len = g
        .qg_len()
        .ok_or_else(|| refuse("the query length wraps".to_string()))?;
    let kv_len = g
        .kv_heads
        .checked_mul(g.head)
        .ok_or_else(|| refuse("the key length wraps".to_string()))?;
    let out = g
        .out_len()
        .ok_or_else(|| refuse("the output length wraps".to_string()))?;
    let rope_len = c
        .ctx
        .checked_mul(g.rot)
        .ok_or_else(|| refuse("the rope length wraps".to_string()))?;
    check_strided(what, "q", c.q.len(), c.rows, c.q_stride, q_len)?;
    check_strided(what, "k", c.k.len(), c.rows, c.k_stride, kv_len)?;
    check_strided(what, "v", c.v.len(), c.rows, c.v_stride, kv_len)?;
    check_strided(what, "out", c.out.len(), c.rows, c.out_stride, out)?;
    check_window_len(what, "q gain", c.q_gain.len(), g.head)?;
    check_window_len(what, "k gain", c.k_gain.len(), g.head)?;
    check_window_len(what, "rope", c.rope.len(), rope_len)?;
    match c.cache {
        AttnCache::F16 { k, v } => {
            let n = g
                .f16_len(c.ctx)
                .ok_or_else(|| refuse("the cache length wraps".to_string()))?;
            check_window_len(what, "k cache", k.len(), n)?;
            check_window_len(what, "v cache", v.len(), n)?;
            for (name, w) in [("k cache", k), ("v cache", v)] {
                let range = w
                    .byte_range()
                    .map_err(|e| refuse(format!("the {name} window: {e}")))?;
                if range.start.get() % 16 != 0 {
                    return Err(refuse(format!(
                        "the {name} starts {} bytes in, not on the 16-byte word the kernel reads",
                        range.start.get()
                    )));
                }
            }
        }
        AttnCache::Q8 { .. } => {
            return Err(refuse(
                "the rows launches take the f16 cache, not the q8_0 planes".to_string(),
            ));
        }
    }
    Ok(())
}

fn rows_view<'a, 'w>(p: &'a AttentionRowsParams<'w>) -> ChunkView<'a, 'w> {
    ChunkView {
        what: ROWS_WHAT,
        head_dim: p.head_dim,
        query_heads: p.query_heads,
        kv_heads: p.kv_heads,
        ctx: p.ctx,
        from: p.from,
        rows: p.rows,
        q: &p.q,
        q_stride: p.q_stride,
        k: &p.k,
        k_stride: p.k_stride,
        v: &p.v,
        v_stride: p.v_stride,
        q_gain: &p.q_gain,
        k_gain: &p.k_gain,
        rope: &p.rope,
        cache: &p.cache,
        out: &p.out,
        out_stride: p.out_stride,
    }
}

fn checked_rows(
    g: &AttnGeom,
    p: &AttentionRowsParams<'_>,
    layer: &AttnLayer<'_>,
) -> Result<LaunchArgs, DeviceError> {
    if p.segments == 0 {
        return Err(refuse_in(ROWS_WHAT, "segments of 0".to_string()));
    }
    if p.segments > SEGMENT_CAP {
        return Err(refuse_in(
            ROWS_WHAT,
            format!("segments of {} past the kernel's {SEGMENT_CAP}", p.segments),
        ));
    }
    let view = rows_view(p);
    check_chunk(g, &view)?;
    let partials_row = g
        .partials_len(p.segments)
        .ok_or_else(|| refuse_in(ROWS_WHAT, "the partial length wraps".to_string()))?;
    check_strided(
        ROWS_WHAT,
        "partials",
        p.partials.len(),
        p.rows,
        partials_row,
        partials_row,
    )?;
    check_strided(
        ROWS_WHAT,
        "ticket",
        p.ticket.len(),
        p.rows,
        g.kv_heads,
        g.kv_heads,
    )?;
    let ticket = p
        .ticket
        .byte_range()
        .map_err(|e| refuse_in(ROWS_WHAT, format!("the ticket window: {e}")))?;
    if ticket.start.get() % 4 != 0 {
        return Err(refuse_in(
            ROWS_WHAT,
            "the ticket word is not 4-byte aligned".to_string(),
        ));
    }
    let window = check_layer(ROWS_WHAT, g, layer, p.rows)?;
    view.words(g, p.segments, p.layer, p.eps, window)
}

/// One prefill attention launch over a chunk of `rows` causal queries on the
/// f16 cache: the rows launch's chunk, run by the simdgroup-matrix kernel
/// ([`Kernels::attention_prefill`]). Row `r` sits at position `from + r`,
/// appends its key and value there and attends the `from + r + 1` keys up to
/// and including it. The windows are the rows launch's, less the split-K
/// partials and tickets the kernel does not use.
pub struct AttentionPrefillParams<'w> {
    /// Head dimension. The kernel takes the instance's (`AttnGeom::head`).
    pub head_dim: usize,
    /// Query heads. The kernel takes the instance's (`AttnGeom::query_heads`).
    pub query_heads: usize,
    /// Key and value heads. The kernel takes the instance's
    /// (`AttnGeom::kv_heads`).
    pub kv_heads: usize,
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
    /// The query projections, `AttnGeom::qg_len` elements a row, `q_stride`
    /// apart.
    pub q: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `q`.
    pub q_stride: usize,
    /// Raw keys, `AttnGeom::kv_heads` rows of the head's values a row,
    /// `k_stride` apart.
    pub k: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `k`.
    pub k_stride: usize,
    /// Raw values, laid out as `k`.
    pub v: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `v`.
    pub v_stride: usize,
    /// Query RMS gain, one value per head value.
    pub q_gain: Window<'w, f32, MetalDevice>,
    /// Key RMS gain, one value per head value.
    pub k_gain: Window<'w, f32, MetalDevice>,
    /// Rope table, `ctx` rows of `AttnGeom::rot` interleaved `(cos, sin)`
    /// pairs.
    pub rope: Window<'w, f32, MetalDevice>,
    /// The cache the launch appends and reads: the f16 cache; the q8_0
    /// planes are refused by name.
    pub cache: AttnCache<'w>,
    /// The outputs after any gate, `AttnGeom::out_len` elements a row,
    /// `out_stride` apart.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `out`.
    pub out_stride: usize,
    /// The fault word.
    pub fault: &'w MetalFaultWord,
}

fn checked_prefill(
    g: &AttnGeom,
    p: &AttentionPrefillParams<'_>,
    layer: &AttnLayer<'_>,
) -> Result<LaunchArgs, DeviceError> {
    let view = ChunkView {
        what: PREFILL_WHAT,
        head_dim: p.head_dim,
        query_heads: p.query_heads,
        kv_heads: p.kv_heads,
        ctx: p.ctx,
        from: p.from,
        rows: p.rows,
        q: &p.q,
        q_stride: p.q_stride,
        k: &p.k,
        k_stride: p.k_stride,
        v: &p.v,
        v_stride: p.v_stride,
        q_gain: &p.q_gain,
        k_gain: &p.k_gain,
        rope: &p.rope,
        cache: &p.cache,
        out: &p.out,
        out_stride: p.out_stride,
    };
    check_chunk(g, &view)?;
    let window = check_layer(PREFILL_WHAT, g, layer, p.rows)?;
    // The append dispatch's span rule takes one segment: the prefill has no split.
    view.words(g, 1, p.layer, p.eps, window)
}

impl Kernels {
    /// Encode the rows launch of a layer that attends every key. An instance
    /// that gates per head refuses it by name
    /// ([`Kernels::attention_rows_layer`]).
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        self.attention_rows_layer(encoder, p, AttnLayer::full())
    }

    /// Encode the attention over a chunk of rows: the append of every row's
    /// key and value, a memory barrier, and the flash with a row grid
    /// dimension. A shape the kernels do not take, or a window that is not
    /// that shape, is refused by name and nothing is encoded. The two
    /// dispatches are one group each, the barrier the flash's order after the
    /// appends. `layer` names the keys the layer attends and its per-head
    /// gate, one row of the query heads' values a row of the chunk.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_rows_layer(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionRowsParams<'_>,
        layer: AttnLayer<'_>,
    ) -> Result<(), DeviceError> {
        let g = &self.geom.attn;
        let args = checked_rows(g, &p, &layer)?;
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
        encoder.set_binding(6, Binding::bytes(args.bytes())?)?;
        encoder.set_binding(7, p.fault.binding())?;
        encoder.dispatch_groups(g.kv_heads, p.rows, SIMD_WIDTH)?;

        encoder.memory_barrier();
        // One segment is the one-pass flash, a threadgroup a (query head, row).
        let (pipeline, groups, threads) = if p.segments == 1 {
            (&self.attention.one_rows, g.query_heads, ONE_PASS_THREADS)
        } else {
            let groups = g
                .kv_heads
                .checked_mul(p.segments)
                .ok_or_else(|| refuse_in(ROWS_WHAT, "the flash grid wraps".to_string()))?;
            (&self.attention.flash_rows, groups, g.flash_threads)
        };
        encoder.set_pipeline(pipeline);
        encoder.set_binding(0, Binding::of(&p.q)?)?;
        encoder.set_binding(1, Binding::of(&p.q_gain)?)?;
        encoder.set_binding(2, Binding::of(&p.rope)?)?;
        encoder.set_binding(3, Binding::of_mut(cache_k)?)?;
        encoder.set_binding(4, Binding::of_mut(cache_v)?)?;
        encoder.set_binding(5, Binding::of_mut(&p.partials)?)?;
        encoder.set_binding(6, Binding::of_mut(&p.out)?)?;
        encoder.set_binding(7, Binding::bytes(args.bytes())?)?;
        encoder.set_binding(8, p.fault.binding())?;
        encoder.set_binding(9, Binding::of_mut(&p.ticket)?)?;
        if let Some(gate) = &layer.head_gate {
            encoder.set_binding(10, Binding::of(gate)?)?;
        }
        encoder.dispatch_groups(groups, p.rows, threads)
    }

    /// Build the prefill attention's pipeline if no launch has built it yet. A
    /// pass calls it before it commits its first op, so a refused build
    /// leaves nothing of the pass on the device.
    pub fn prepare_prefill(&self) -> Result<(), DeviceError> {
        self.attention.prefill().map(|_| ())
    }

    /// Encode the prefill launch of a layer that attends every key. An
    /// instance that gates per head refuses it by name
    /// ([`Kernels::attention_prefill_layer`]).
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_prefill(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionPrefillParams<'_>,
    ) -> Result<(), DeviceError> {
        self.attention_prefill_layer(encoder, p, AttnLayer::full())
    }

    /// Encode the prefill attention over a chunk of rows: the append of every
    /// row's key and value (the rows launch's own), a memory barrier, and the
    /// simdgroup-matrix flash, one threadgroup a key head and
    /// [`PREFILL_POSITIONS`] positions. A shape the kernels do not take, or a
    /// window that is not that shape, is refused by name and nothing is
    /// encoded. The rows are within a band of the decode launch's, not its
    /// bits. `layer` names the keys the layer attends and its per-head gate,
    /// one row of the query heads' values a row of the chunk. The flash starts
    /// at the tile of the first window of each block of positions, so the
    /// tiles before every window of the block are never read.
    #[must_use = "a refused launch encoded nothing"]
    pub fn attention_prefill_layer(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: AttentionPrefillParams<'_>,
        layer: AttnLayer<'_>,
    ) -> Result<(), DeviceError> {
        let g = &self.geom.attn;
        let args = checked_prefill(g, &p, &layer)?;
        let flash = self.attention.prefill()?;
        let AttnCache::F16 {
            k: cache_k,
            v: cache_v,
        } = &p.cache
        else {
            unreachable!("check_chunk refused the q8_0 planes");
        };
        encoder.set_pipeline(&self.attention.append_rows);
        encoder.set_binding(0, Binding::of(&p.k)?)?;
        encoder.set_binding(1, Binding::of(&p.v)?)?;
        encoder.set_binding(2, Binding::of(&p.k_gain)?)?;
        encoder.set_binding(3, Binding::of(&p.rope)?)?;
        encoder.set_binding(4, Binding::of_mut(cache_k)?)?;
        encoder.set_binding(5, Binding::of_mut(cache_v)?)?;
        encoder.set_binding(6, Binding::bytes(args.bytes())?)?;
        encoder.set_binding(7, p.fault.binding())?;
        encoder.dispatch_groups(g.kv_heads, p.rows, SIMD_WIDTH)?;

        encoder.memory_barrier();
        encoder.set_pipeline(flash);
        encoder.set_binding(0, Binding::of(&p.q)?)?;
        encoder.set_binding(1, Binding::of(&p.q_gain)?)?;
        encoder.set_binding(2, Binding::of(&p.rope)?)?;
        encoder.set_binding(3, Binding::of_mut(cache_k)?)?;
        encoder.set_binding(4, Binding::of_mut(cache_v)?)?;
        encoder.set_binding(5, Binding::of_mut(&p.out)?)?;
        encoder.set_binding(6, Binding::bytes(args.bytes())?)?;
        encoder.set_binding(7, p.fault.binding())?;
        if let Some(gate) = &layer.head_gate {
            encoder.set_binding(8, Binding::of(gate)?)?;
        }
        let blocks = p.rows.div_ceil(PREFILL_POSITIONS);
        encoder.dispatch_groups(blocks, g.kv_heads, PREFILL_THREADS)
    }
}
