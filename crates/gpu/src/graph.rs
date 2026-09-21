//! CUDA graph capture and replay over the driver bindings cuda-core re-exports
//! as `cuda_core::sys` (cutile-rs `cuda-bindings`, bindgen of `^cu.*` against
//! the toolkit headers, resolved through `dlopen("libcuda.so.1")` at first
//! call). cuda-core 0.3.1 has `begin_capture`/`end_capture` only on its legacy
//! `runtime::Stream`, not on the `simt::CudaStream` the generated launchers
//! take, and no instantiate/launch wrapper at all — so this file is the
//! bloomery-side wrapper and the shape of the cutile-rs contribution
//! (`docs/upstream/nvlabs-ledger.md` #3).
//!
//! Why graphs at all: ik's 3090 decode on this model measured +10.6% with
//! CUDA graphs on vs `GGML_CUDA_DISABLE_GRAPHS=1` (docs/gpu-design.md §미정),
//! so replay is the common starting line, not an optimization to earn later.

use crate::GpuError;
use cuda_core::{CudaStream, sys};
use std::ffi::CStr;
use std::ptr;

/// Map a driver result to `GpuError` with the driver's own string.
pub(crate) fn cu(result: sys::CUresult, what: &str) -> Result<(), GpuError> {
    if result == sys::cudaError_enum_CUDA_SUCCESS {
        return Ok(());
    }
    let mut msg: *const std::os::raw::c_char = ptr::null();
    // SAFETY: cuGetErrorString only writes the out-pointer; a failure leaves
    // it null, which the branch below handles.
    let rc = unsafe { sys::cuGetErrorString(result, &mut msg) };
    let text = if rc == sys::cudaError_enum_CUDA_SUCCESS && !msg.is_null() {
        // SAFETY: the driver returns a NUL-terminated static string.
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    } else {
        format!("CUresult {result}")
    };
    Err(format!("{what}: {text}").into())
}

/// A captured kernel sequence, instantiated for replay.
///
/// Replaying launches the recorded work with the recorded buffer addresses —
/// every buffer the captured closure touched must stay alive and in place for
/// as long as the graph is launched (decision 4: weights, KV and scratch are
/// allocated once at load, so this holds by construction in `GpuModel`).
pub struct Graph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    nodes: usize,
}

// SAFETY: the handles are opaque driver objects valid across threads while the
// owning context lives; nothing here has interior mutability.
unsafe impl Send for Graph {}

impl Graph {
    /// Capture everything `body` enqueues on `stream` into a graph and
    /// instantiate it.
    ///
    /// `stream` must be a real stream (`CudaContext::new_stream`): the driver
    /// refuses capture on the legacy null stream, which is what
    /// `default_stream()` hands out, so that case is rejected here with a
    /// readable error instead of `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. The
    /// body must not allocate, free or synchronize (no `DeviceBuffer::zeroed`
    /// / `from_host` / `to_host_vec`, no drops of device buffers) — those are
    /// not capturable and abort the capture.
    pub fn capture<F>(stream: &CudaStream, body: F) -> Result<Graph, GpuError>
    where
        F: FnOnce(&CudaStream) -> Result<(), GpuError>,
    {
        let hs = stream.cu_stream();
        if hs.is_null() {
            return Err(
                "Graph::capture: refused on the legacy default (null) stream; \
                        capture needs CudaContext::new_stream()"
                    .into(),
            );
        }
        // SAFETY: hs is a live stream of the current context and not capturing
        // (a nested capture fails here with the driver's own error).
        let begun = unsafe {
            sys::cuStreamBeginCapture_v2(
                hs,
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        cu(begun, "cuStreamBeginCapture_v2")?;
        let body_result = body(stream);
        let mut graph: sys::CUgraph = ptr::null_mut();
        // SAFETY: the stream is capturing (begun above); end_capture always
        // leaves the stream un-captured, whether or not the body failed.
        let end = unsafe { sys::cuStreamEndCapture(hs, &mut graph) };
        body_result?;
        cu(end, "cuStreamEndCapture")?;

        let mut nodes = 0usize;
        // SAFETY: a null node array asks only for the count.
        let rc = unsafe { sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut nodes) };
        if let Err(e) = cu(rc, "cuGraphGetNodes") {
            // SAFETY: graph is a valid handle returned by end_capture.
            unsafe { sys::cuGraphDestroy(graph) };
            return Err(e);
        }

        let mut exec: sys::CUgraphExec = ptr::null_mut();
        // SAFETY: graph is valid; flags 0 is the plain instantiation. cuda.h
        // 12+ maps `cuGraphInstantiate` to this 3-argument entry point, which
        // is the one the bindings expose.
        let rc = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if let Err(e) = cu(rc, "cuGraphInstantiateWithFlags") {
            // SAFETY: as above.
            unsafe { sys::cuGraphDestroy(graph) };
            return Err(e);
        }
        Ok(Graph { graph, exec, nodes })
    }

    /// Enqueue one replay on `stream`. Asynchronous; the caller synchronizes.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), GpuError> {
        // SAFETY: exec is a live instantiated graph; the stream belongs to the
        // same context.
        cu(
            unsafe { sys::cuGraphLaunch(self.exec, stream.cu_stream()) },
            "cuGraphLaunch",
        )
    }

    /// Number of nodes the capture recorded — kernel launches plus any memset
    /// or copy nodes. This is the "launch count" column of the first lease
    /// measurement (docs/gpu-design.md §런치 수).
    pub fn node_count(&self) -> usize {
        self.nodes
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: both handles were created in `capture` and are destroyed
        // exactly once here; the exec first, then the template it came from.
        // Errors on the drop path are unreportable and ignored.
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}
