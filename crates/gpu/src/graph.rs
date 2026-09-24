//! CUDA graph capture and replay over the driver bindings cuda-core re-exports
//! as `cuda_core::sys` (cutile-rs `cuda-bindings`, bindgen of `^cu.*` against
//! the toolkit headers, resolved through `dlopen("libcuda.so.1")` at first
//! call). cuda-core 0.3.1 has `begin_capture`/`end_capture` only on its legacy
//! `runtime::Stream`, not on the `simt::CudaStream` the generated launchers
//! take, and no instantiate/launch wrapper at all — so this file is the
//! bloomery-side wrapper and the shape of the cutile-rs contribution
//! (`docs/upstream/nvlabs-ledger.md` #3).
//!
//! Why graphs at all: the reference engine's decode on this model is faster
//! with CUDA graphs on than with them disabled (docs/gpu-design.md §미정), so
//! replay is the common starting line, not an optimization to earn later.

use crate::GpuError;
use cuda_core::{CudaStream, DriverError, sys};
use std::ptr;

/// Map a driver result to `GpuError::Driver`, tagged with the driver entry
/// point we called. `DriverError` carries the code and asks the driver for
/// its own string when it is printed.
pub(crate) fn cu(result: sys::CUresult, what: &'static str) -> Result<(), GpuError> {
    if result == sys::cudaError_enum_CUDA_SUCCESS {
        return Ok(());
    }
    Err(GpuError::Driver {
        op: Some(what),
        source: DriverError(result),
    })
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
    /// not capturable and abort the capture. A body that returns `Err` or
    /// panics still ends the capture and frees the partial template: the
    /// stream is capturable again afterwards.
    pub(crate) fn capture<F>(stream: &CudaStream, body: F) -> Result<Graph, GpuError>
    where
        F: FnOnce(&CudaStream) -> Result<(), GpuError>,
    {
        let hs = stream.cu_stream();
        if hs.is_null() {
            return Err(GpuError::state(
                "Graph::capture",
                "refused on the legacy default (null) stream; capture needs \
                 CudaContext::new_stream()",
            ));
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
        // The body may fail or unwind; either way the stream must leave
        // capture mode and the half-recorded template must not leak, or every
        // later capture on this stream fails.
        let body_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(stream)));
        let mut graph: sys::CUgraph = ptr::null_mut();
        // SAFETY: the stream is capturing (begun above); end_capture leaves
        // the stream un-captured on every path.
        let ended = cu(
            unsafe { sys::cuStreamEndCapture(hs, &mut graph) },
            "cuStreamEndCapture",
        );
        let discard = |graph: sys::CUgraph| {
            if !graph.is_null() {
                // SAFETY: a non-null handle from end_capture is a valid
                // template, destroyed exactly once here.
                unsafe { sys::cuGraphDestroy(graph) };
            }
        };
        match body_result {
            Err(payload) => {
                discard(graph);
                std::panic::resume_unwind(payload);
            }
            Ok(Err(e)) => {
                discard(graph);
                return Err(e);
            }
            Ok(Ok(())) => {}
        }
        ended?;

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
            // SAFETY: graph is still the valid handle end_capture returned;
            // the failed instantiation left it untouched.
            unsafe { sys::cuGraphDestroy(graph) };
            return Err(e);
        }
        // From here the handles are owned: a failed upload drops them.
        let captured = Graph { graph, exec, nodes };
        // The upload the first launch of an exec would otherwise do inside
        // its own call, done once here on the stream the replays use.
        // SAFETY: exec is the live instantiation above; hs is the live stream
        // it was captured on.
        let rc = unsafe { sys::cuGraphUpload(captured.exec, hs) };
        cu(rc, "cuGraphUpload")?;
        Ok(captured)
    }

    /// Enqueue one replay on `stream`. Asynchronous; the caller synchronizes.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), GpuError> {
        // SAFETY: exec is a live instantiated graph; the stream belongs to the
        // same context.
        unsafe { launch_exec(self.exec, stream.cu_stream()) }
    }

    /// The instantiated graph, for a launch issued from another thread while
    /// this graph is kept alive ([`launch_exec`]).
    pub(crate) fn exec(&self) -> sys::CUgraphExec {
        self.exec
    }

    /// Number of nodes the capture recorded — kernel launches plus any memset
    /// or copy nodes. This is the "launch count" column of the first lease
    /// measurement (docs/gpu-design.md §런치 수).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Every node the capture recorded, in the order `cuGraphGetNodes`
    /// returns them: what an enqueue with more than one graph form (a host
    /// function with a sync mode, a stream memory operation) became.
    pub fn nodes(&self) -> Result<Vec<NodeInfo>, GpuError> {
        let mut handles: Vec<sys::CUgraphNode> = vec![ptr::null_mut(); self.nodes];
        let mut n = self.nodes;
        // SAFETY: the array holds `n` slots, the count the driver reported
        // for this template at capture; it writes at most `n` handles.
        let rc = unsafe { sys::cuGraphGetNodes(self.graph, handles.as_mut_ptr(), &mut n) };
        cu(rc, "cuGraphGetNodes")?;
        handles.truncate(n);
        handles
            .into_iter()
            .map(|node| {
                let mut kind: sys::CUgraphNodeType = 0;
                // SAFETY: `node` is a handle of this live template.
                let rc = unsafe { sys::cuGraphNodeGetType(node, &mut kind) };
                cu(rc, "cuGraphNodeGetType")?;
                let host_sync = if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST {
                    // SAFETY: all-zero is a valid value of this plain C
                    // struct (integers, pointers, a nullable function
                    // pointer), which the driver then fills for `node`.
                    let mut params: sys::CUgraphNodeParams = unsafe { std::mem::zeroed() };
                    // SAFETY: `node` is a live host node of this template and
                    // `params` is a writable struct of the type the call fills.
                    let rc = unsafe { sys::cuGraphNodeGetParams(node, &mut params) };
                    cu(rc, "cuGraphNodeGetParams")?;
                    // SAFETY: for a host node the driver fills the `host`
                    // member of the union.
                    Some(unsafe { params.__bindgen_anon_1.host.syncMode })
                } else {
                    None
                };
                Ok(NodeInfo { kind, host_sync })
            })
            .collect()
    }
}

/// `cuGraphLaunch` of `exec` on `stream`, mapped to [`GpuError`].
///
/// # Safety
///
/// `exec` is a live instantiated graph (or null, which the driver refuses with
/// an error) that stays alive until the call returns, and `stream` a live
/// stream of the context current on the calling thread.
pub(crate) unsafe fn launch_exec(
    exec: sys::CUgraphExec,
    stream: sys::CUstream,
) -> Result<(), GpuError> {
    // SAFETY: the caller's contract above.
    cu(unsafe { sys::cuGraphLaunch(exec, stream) }, "cuGraphLaunch")
}

/// One node of a captured graph as the driver reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeInfo {
    /// The node's `CUgraphNodeType` (kernel, memcpy, host, batch memop, …).
    pub kind: sys::CUgraphNodeType,
    /// For a host node, the `CUhostTaskSyncMode` it carries.
    pub host_sync: Option<u32>,
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

#[cfg(test)]
mod tests {
    use super::{Graph, cu};
    use cuda_core::{CudaContext, DeviceBuffer, sys};
    use std::ffi::c_void;

    /// The host function the test graph records. The graph is never
    /// launched, so it never runs.
    unsafe extern "C" fn nothing(_: *mut c_void) {}

    /// A structure gate pins "no host node" by counting the nodes `nodes`
    /// reports as `CU_GRAPH_NODE_TYPE_HOST`; that count means something only
    /// if a captured host function does come back as one. A memset and a
    /// host function, captured: two nodes, exactly one of them a host node
    /// carrying its sync mode.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_nodes_report_a_captured_host_function() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
        let mut buf = DeviceBuffer::<u32>::zeroed(&stream, 64).expect("a device buffer");
        stream
            .synchronize()
            .expect("the allocation lands before capture");
        let graph = Graph::capture(&stream, |s| {
            buf.zero_async(s)?;
            // SAFETY: `s` is the live stream being captured; `nothing` takes
            // any pointer and never reads it, and the graph is never launched.
            let rc = unsafe {
                sys::cuLaunchHostFunc(s.cu_stream(), Some(nothing), std::ptr::null_mut())
            };
            cu(rc, "cuLaunchHostFunc")
        })
        .expect("the capture");
        let nodes = graph.nodes().expect("the node list");
        let host: Vec<_> = nodes
            .iter()
            .filter(|n| n.kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST)
            .collect();
        assert_eq!(nodes.len(), 2, "{nodes:?}");
        assert_eq!(host.len(), 1, "{nodes:?}");
        assert!(host[0].host_sync.is_some(), "{nodes:?}");
        assert!(
            nodes
                .iter()
                .any(|n| n.kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET),
            "{nodes:?}"
        );
    }
}
