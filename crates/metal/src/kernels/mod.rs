//! The quantized kernels' host side: the pipelines of the Q4_K and Q6_K
//! mat-vecs and the Q4_K embedding, and the launch of each.
//!
//! A launch takes a parameter struct of windows and the shape they are for,
//! checks every window's length against the shape before it encodes anything
//! (a window that is not the shape's is a named [`bloomery_device::DeviceError`], never a
//! partial read or write), and encodes one dispatch on the caller's encoder.
//! The kernels themselves are `kernels/{norm,matvec_q4k,matvec_q6k,embed}.metal`.
//! Decode attention is `kernels/attention.metal`.
//!
//! Three mat-vecs live here, one concept each: the plain mat-vec of one
//! tensor ([`matvec`]), the segmented mat-vec of up to four tensors of one
//! type and one width in one dispatch ([`seg`]), and the head-normed residual
//! mat-vec of a gated delta net's out projection ([`headnorm`]).
//! The router ([`router`]) is one more pipeline: `kernels/router.metal`, its ticket `kernels/ticket.metal`.
//! The routed gate-up pair and the folded down are one dispatch each.
//! The gated delta step ([`delta`]) writes the numerator and the partials that
//! mat-vec reads. Its kernel is `kernels/delta.metal`.
//! The head ([`head`]) is one more pipeline: `kernels/head.metal`, the same ticket.

mod attention;
use crate::attention_geom;
mod delta;
mod embed;
mod experts;
mod geometry;
mod head;
mod headnorm;
mod matvec;
pub mod router;
mod seg;

pub use attention::{
    AttentionParams, AttnCache, HEAD_DIM, KV_HEADS, QG_STRIDE, QUERY_HEADS, ROT, SCALE,
    SEGMENT_KEYS, TILE, attention_dispatches, f16_len, out_len, partials_len, q8_code_len,
    q8_scale_len, qg_len, segments_for_keys,
};
pub use delta::CONV_TAPS;
pub use delta::DeltaParams;
pub use delta::HeadMap;
pub use delta::PASS_ROWS;
pub use delta::Q_SCALE;
pub use delta::RING_ROWS;
pub use delta::channels;
pub use delta::key_of;
pub use delta::ring_elems;
pub use delta::state_elems;
pub use delta::state_index;
pub use embed::EmbedParams;
pub use experts::{
    EXPERT_K, EXPERT_SLOTS, FoldParams, PairParams, ROUTED_IDS, SHARED_SLOT, SharedPlan,
};
pub use geometry::{
    BLOCK_VALUES, HEAD_COLS, MAX_SEGMENTS, PARTIALS_PER_HEAD, ROWS_PER_GROUP, ROWS_PER_SIMD_GROUP,
    SIMD_GROUPS_PER_GROUP, THREADS_PER_GROUP, VALUE_HEADS,
};
pub use head::{HEAD_GROUPS, HEAD_K, HEAD_SIMD_GROUPS, HEAD_THREADS, HEAD_VOCAB, HeadParams};
pub use headnorm::HeadNormParams;
pub use matvec::{MatvecParams, Mode, NormOnLoad, Quant};
pub use seg::{SegParams, Segment};

use bloomery_device::DeviceError;

use crate::device::MetalDevice;
use crate::library::{Library, Pipeline};

/// The pipelines of one mat-vec type: the plain entry, the entry that norms its
/// input as it loads it, the one that adds into its output, and the segmented
/// entry.
struct MatvecPipelines {
    store: Pipeline,
    norm: Pipeline,
    residual: Pipeline,
    seg: Pipeline,
}

impl MatvecPipelines {
    fn new(library: &Library, names: [&str; 4], simdgroups: u32) -> Result<Self, DeviceError> {
        Ok(Self {
            store: library.pipeline_with_simdgroups(names[0], simdgroups)?,
            norm: library.pipeline_with_simdgroups(names[1], simdgroups)?,
            residual: library.pipeline_with_simdgroups(names[2], simdgroups)?,
            seg: library.pipeline_with_simdgroups(names[3], simdgroups)?,
        })
    }

    /// The segmented entry of one type.
    pub(crate) fn seg(&self) -> &Pipeline {
        &self.seg
    }
}

/// The compute pipelines of the quantized kernels.
pub struct Kernels {
    q4k: MatvecPipelines,
    q6k: MatvecPipelines,
    headnorm: Pipeline,
    embed: Pipeline,
    router: Pipeline,
    experts: experts::Pipelines,
    /// The gated delta step.
    delta: Pipeline,
    attention: attention::AttentionPipelines,
    head: Pipeline,
    /// The SIMD groups in a mat-vec's threadgroup: the kernels' function
    /// constant, and the geometry of every launch.
    simdgroups: usize,
}

impl Kernels {
    /// The pipelines of the device's own library.
    pub fn new(device: &MetalDevice) -> Result<Self, DeviceError> {
        Self::from_library(device.library())
    }

    /// The pipelines of `library`: a missing kernel is a named error.
    pub fn from_library(library: &Library) -> Result<Self, DeviceError> {
        Self::with_simdgroups(library, SIMD_GROUPS_PER_GROUP)
    }

    /// The pipelines of `library` with `simdgroups` SIMD groups in a
    /// threadgroup. The count is the kernels' function constant, so each count
    /// is its own pipeline, and every launch's geometry follows it.
    pub fn with_simdgroups(library: &Library, simdgroups: usize) -> Result<Self, DeviceError> {
        let count = u32::try_from(simdgroups).map_err(|_| DeviceError::Range {
            what: "matvec",
            detail: format!("{simdgroups} SIMD groups do not fit the kernel's word"),
        })?;
        if count == 0 {
            return Err(DeviceError::Range {
                what: "matvec",
                detail: "a threadgroup of no SIMD groups".to_string(),
            });
        }
        // A threadgroup holds at most the device's 1024 threads, and a SIMD
        // group is 32 of them.
        if simdgroups > geometry::MAX_SIMD_GROUPS_PER_GROUP {
            return Err(DeviceError::Range {
                what: "matvec",
                detail: format!(
                    "{simdgroups} SIMD groups in a threadgroup, the most is {}",
                    geometry::MAX_SIMD_GROUPS_PER_GROUP
                ),
            });
        }
        Ok(Self {
            q4k: MatvecPipelines::new(
                library,
                [
                    "q4k_matvec",
                    "q4k_matvec_norm",
                    "q4k_matvec_resid",
                    "q4k_matvec_seg",
                ],
                count,
            )?,
            q6k: MatvecPipelines::new(
                library,
                [
                    "q6k_matvec",
                    "q6k_matvec_norm",
                    "q6k_matvec_resid",
                    "q6k_matvec_seg",
                ],
                count,
            )?,
            headnorm: library.pipeline_with_simdgroups("q4k_matvec_headnorm_resid", count)?,
            embed: library.pipeline("embed_q4k")?,
            router: library.pipeline("router35")?,
            experts: experts::Pipelines::new(library)?,
            delta: library.pipeline("gdn_step")?,
            attention: attention::AttentionPipelines::new(library)?,
            // The head's SIMD-group count, not the mat-vec count this function
            // was given. The constant is the pipeline's function constant.
            head: library.pipeline_with_simdgroups(
                "q6k_head_argmax",
                u32::try_from(head::HEAD_SIMD_GROUPS)
                    .expect("the head's SIMD-group count fits u32"),
            )?,
            simdgroups,
        })
    }

    /// The SIMD groups in a mat-vec's threadgroup.
    #[must_use]
    pub fn simdgroups(&self) -> usize {
        self.simdgroups
    }

    /// The threads in a mat-vec's threadgroup.
    #[must_use]
    pub fn threads_per_group(&self) -> usize {
        geometry::SIMD_WIDTH * self.simdgroups
    }
}

/// A named range error. Launch modules build every refusal through this.
pub(crate) fn range(what: &'static str, detail: impl Into<String>) -> DeviceError {
    DeviceError::Range {
        what,
        detail: detail.into(),
    }
}

/// How a kernel reads a weight window.
pub(crate) enum WeightAlign {
    /// The kernel loads `float4` from the window.
    Float4,
    /// The kernel loads 16-bit words from the window.
    Half,
}

/// `start` is the window's first byte in its buffer. A misaligned window is
/// refused by name: the message names the word the kernel loads.
pub(crate) fn check_weights_align(
    what: &'static str,
    name: &str,
    start: usize,
    kind: WeightAlign,
) -> Result<(), DeviceError> {
    match kind {
        WeightAlign::Float4 => {
            if start.is_multiple_of(16) {
                return Ok(());
            }
            Err(range(
                what,
                format!(
                    "the {name} window starts {start} bytes into its buffer and is not 16-byte aligned"
                ),
            ))
        }
        WeightAlign::Half => {
            if start.is_multiple_of(2) {
                return Ok(());
            }
            Err(range(
                what,
                format!(
                    "the {name} weights start {start} bytes into their buffer: the kernels read 16-bit words"
                ),
            ))
        }
    }
}

/// The bytes of `rows` rows of `nblocks` quantized blocks. `None` when the
/// product does not fit. An expert stack multiplies the result by the expert
/// count.
pub(crate) fn weight_bytes(quant: Quant, rows: usize, nblocks: usize) -> Option<usize> {
    rows.checked_mul(nblocks)?.checked_mul(quant.block_bytes())
}

/// The bytes a kernel reads: `words` in order, each already in the machine's
/// byte order.
pub(crate) fn pack<const N: usize>(words: &[[u8; 4]; N]) -> &[u8] {
    words.as_flattened()
}

/// A kernel's argument words, each in the machine's byte order. [`pack`] is
/// the bytes.
pub(crate) fn args_bytes<const N: usize>(words: [u32; N]) -> [[u8; 4]; N] {
    words.map(u32::to_ne_bytes)
}

/// A count as the kernel's 32-bit word, refused by name when it does not fit.
pub(crate) fn word(what: &'static str, name: &str, n: usize) -> Result<u32, DeviceError> {
    u32::try_from(n).map_err(|_| {
        range(
            what,
            format!("{name} of {n} does not fit the kernel's 32-bit word"),
        )
    })
}

/// A kernel window's length against the length the launch's shape needs,
/// refused by name when they differ: `name` says which window. (The
/// contract's [`bloomery_device::check_len`] is the other rule: a host slice
/// against the window it is copied to or from.)
pub(crate) fn check_window_len(
    what: &'static str,
    name: &str,
    got: bloomery_device::Elems,
    want: usize,
) -> Result<(), DeviceError> {
    if got.get() == want {
        return Ok(());
    }
    Err(range(
        what,
        format!(
            "the {name} window holds {} elements, the shape needs {want}",
            got.get()
        ),
    ))
}
