//! The quantized kernels' host side: the pipelines of the Q4_K and Q6_K
//! mat-vecs and the Q4_K embedding, and the launch of each.
//!
//! A launch takes a parameter struct of windows and the shape they are for,
//! checks every window's length against the shape before it encodes anything
//! (a window that is not the shape's is a named [`bloomery_device::DeviceError`], never a
//! partial read or write), and encodes one dispatch on the caller's encoder.
//! The kernels themselves are `kernels/{norm,matvec_q4k,matvec_q6k,embed}.metal`.

mod embed;
mod geometry;
mod matvec;

pub use embed::EmbedParams;
pub use geometry::{BLOCK_VALUES, ROWS_PER_GROUP, THREADS_PER_GROUP};
pub use matvec::{MatvecParams, Mode, NormOnLoad, Quant};

use bloomery_device::DeviceError;

use crate::device::MetalDevice;
use crate::library::{Library, Pipeline};

/// The pipelines of one mat-vec type: the plain entry, the entry that norms its
/// input as it loads it, and the one that adds into its output.
struct MatvecPipelines {
    store: Pipeline,
    norm: Pipeline,
    residual: Pipeline,
}

impl MatvecPipelines {
    fn new(library: &Library, names: [&str; 3]) -> Result<Self, DeviceError> {
        Ok(Self {
            store: library.pipeline(names[0])?,
            norm: library.pipeline(names[1])?,
            residual: library.pipeline(names[2])?,
        })
    }
}

/// The compute pipelines of the quantized kernels.
pub struct Kernels {
    q4k: MatvecPipelines,
    q6k: MatvecPipelines,
    embed: Pipeline,
}

impl Kernels {
    /// The pipelines of the device's own library.
    pub fn new(device: &MetalDevice) -> Result<Self, DeviceError> {
        Self::from_library(device.library())
    }

    /// The pipelines of `library`: a missing kernel is a named error.
    pub fn from_library(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            q4k: MatvecPipelines::new(
                library,
                ["q4k_matvec", "q4k_matvec_norm", "q4k_matvec_resid"],
            )?,
            q6k: MatvecPipelines::new(
                library,
                ["q6k_matvec", "q6k_matvec_norm", "q6k_matvec_resid"],
            )?,
            embed: library.pipeline("embed_q4k")?,
        })
    }
}

/// A word of a kernel's argument block, as the bytes the kernel reads.
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
