//! The quantized kernels' host side: the pipelines of the Q4_K and Q6_K
//! mat-vecs and the Q4_K embedding, and the launch of each.
//!
//! A launch takes a parameter struct of windows and the shape they are for,
//! checks every window's length against the shape before it encodes anything
//! (a window that is not the shape's is a named [`bloomery_device::DeviceError`], never a
//! partial read or write), and encodes one dispatch on the caller's encoder.
//! The kernels themselves are `kernels/{norm,matvec_q4k,matvec_q6k,embed}.metal`.
//!
//! Three mat-vecs live here, one concept each: the plain mat-vec of one
//! tensor ([`matvec`]), the segmented mat-vec of up to four tensors of one
//! type and one width in one dispatch ([`seg`]), and the head-normed residual
//! mat-vec of a gated delta net's out projection ([`headnorm`]).
//! The routed gate-up pair and the folded down are one dispatch each. // K3-REGISTER

mod embed;
mod experts; // K3-REGISTER
mod geometry;
mod headnorm;
mod matvec;
mod seg;

pub use embed::EmbedParams;
pub use experts::{
    // K3-REGISTER
    EXPERT_K,     // K3-REGISTER
    EXPERT_SLOTS, // K3-REGISTER
    FoldParams,   // K3-REGISTER
    PairParams,   // K3-REGISTER
    ROUTED_IDS,   // K3-REGISTER
    SHARED_SLOT,  // K3-REGISTER
    SharedPlan,   // K3-REGISTER
}; // K3-REGISTER
pub use geometry::{
    BLOCK_VALUES, HEAD_COLS, MAX_SEGMENTS, PARTIALS_PER_HEAD, ROWS_PER_GROUP, ROWS_PER_SIMD_GROUP,
    SIMD_GROUPS_PER_GROUP, THREADS_PER_GROUP, VALUE_HEADS,
};
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
    pair_q4: Pipeline, // K3-REGISTER
    pair_q6: Pipeline, // K3-REGISTER
    fold_q4: Pipeline, // K3-REGISTER
    fold_q6: Pipeline, // K3-REGISTER
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
            pair_q4: library.pipeline("q4k_pair_swiglu")?, // K3-REGISTER
            pair_q6: library.pipeline("q6k_pair_swiglu")?, // K3-REGISTER
            fold_q4: library.pipeline("q4k_down_fold")?,   // K3-REGISTER
            fold_q6: library.pipeline("q6k_down_fold")?,   // K3-REGISTER
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

/// A kernel's argument block, as the bytes the kernel reads: the words in
/// order, each in the machine's byte order.
pub(crate) fn args_bytes(words: [u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_ne_bytes();
    }
    out
}

/// A count as the kernel's 32-bit word, refused by name when it does not fit.
pub(crate) fn word(what: &'static str, name: &str, n: usize) -> Result<u32, DeviceError> {
    u32::try_from(n).map_err(|_| DeviceError::Range {
        what,
        detail: format!("{name} of {n} does not fit the kernel's 32-bit word"),
    })
}

/// A count the kernel only indexes by, refused by name when it does not fit
/// its 32-bit word: no argument carries it, so there is no value to take.
pub(crate) fn check_word(what: &'static str, name: &str, n: usize) -> Result<(), DeviceError> {
    word(what, name, n).map(drop)
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
    Err(DeviceError::Range {
        what,
        detail: format!(
            "the {name} window holds {} elements, the shape needs {want}",
            got.get()
        ),
    })
}
