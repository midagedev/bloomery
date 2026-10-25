//! The pipelines the engine builds, recorded as they are built: the list the
//! move proof's pipeline hash (`bin/metal_scan.rs`) is taken over.
//!
//! The one place a compute pipeline is made is [`Library::pipeline`] and
//! [`Library::pipeline_with_simdgroups`], so a record there is every pipeline
//! the `Kernels` constructors and the lazily built ones make, with the
//! function-constant values they gave it. The scan keeps no list of its own.
//! The record is the calling thread's.

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLFunction};

use crate::library::{Library, Pipeline};

/// One pipeline the engine built.
pub struct Built {
    /// The address of the library object the function came from
    /// ([`Library::scan_id`]).
    pub library: usize,
    /// The function's name.
    pub entry: String,
    /// The function constants it was specialized with: index and `uint` value.
    pub constants: Vec<(usize, u32)>,
    /// The specialized function, the input of a binary archive.
    pub function: Retained<ProtocolObject<dyn MTLFunction>>,
    /// The most threads a group of the built pipeline may hold.
    pub max_threads: usize,
    /// The pipeline's SIMD width.
    pub execution_width: usize,
    /// The pipeline's static threadgroup memory, in bytes.
    pub static_threadgroup_bytes: usize,
}

thread_local! {
    static BUILT: RefCell<Option<Vec<Built>>> = const { RefCell::new(None) };
}

/// Start recording, dropping what an earlier start recorded.
pub fn start() {
    BUILT.with(|built| *built.borrow_mut() = Some(Vec::new()));
}

/// What was built since [`start`], in build order; recording stops.
///
/// # Panics
/// When [`start`] was not called on this thread: a list of nothing would read
/// as an engine that builds no pipeline.
#[must_use]
pub fn take() -> Vec<Built> {
    BUILT
        .with(|built| built.borrow_mut().take())
        .expect("scan::take without scan::start on this thread")
}

/// Note a pipeline just made from `function` of `library`, while recording.
pub(crate) fn record(
    library: &Library,
    function: &Retained<ProtocolObject<dyn MTLFunction>>,
    constants: &[(usize, u32)],
    pipeline: &Pipeline,
) {
    BUILT.with(|built| {
        if let Some(list) = built.borrow_mut().as_mut() {
            let state = pipeline.metal();
            list.push(Built {
                library: library.scan_id(),
                entry: function.name().to_string(),
                constants: constants.to_vec(),
                function: function.clone(),
                max_threads: pipeline.max_threads_per_group(),
                execution_width: state.threadExecutionWidth(),
                static_threadgroup_bytes: state.staticThreadgroupMemoryLength(),
            });
        }
    });
}
