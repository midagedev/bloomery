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
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError, PinnedHostBuffer,
    sys,
};
use std::ffi::c_void;
use std::os::unix::fs::OpenOptionsExt;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

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

/// A driver result as cuda-core's error, for [`CudaContext::record_err`] on a
/// drop path, where nothing can return it.
fn driver(result: sys::CUresult) -> Result<(), DriverError> {
    if result == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DriverError(result))
    }
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
    /// The context the graph was captured in: held so it outlives the
    /// handles, and bound when they are destroyed, as cuda-core's owners do.
    ctx: Arc<CudaContext>,
}

// SAFETY: the handles are opaque driver objects valid across threads while the
// owning context lives, which `ctx` guarantees; nothing here has interior
// mutability.
unsafe impl Send for Graph {}

/// Begin a capture of everything `body` enqueues on `stream`, run the body,
/// and end the capture: the one place in the workspace
/// `cuStreamBeginCapture` is called, so the device's capture lock and this
/// thread's capture note — which keep a context-wide synchronize out of
/// the capture ([`crate::capsync`]) — live here, over the whole
/// begin..end. Returns the raw template for the caller to instantiate
/// ([`Graph::capture`]) or read (`gpu-gates`' node walk) and destroy.
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
pub fn capture_template<F>(stream: &CudaStream, body: F) -> Result<sys::CUgraph, GpuError>
where
    F: FnOnce(&CudaStream) -> Result<(), GpuError>,
{
    let hs = stream.cu_stream();
    if hs.is_null() {
        return Err(GpuError::state(
            "capture_template",
            "refused on the legacy default (null) stream; capture needs \
             CudaContext::new_stream()",
        ));
    }
    crate::capsync::capture_hold(stream.context().ordinal(), move || {
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
        Ok(graph)
    })
}

impl Graph {
    /// Capture everything `body` enqueues on `stream` into a graph and
    /// instantiate it: [`capture_template`] over the begin..end, then the
    /// node count, the instantiation and the once-for-all upload. See
    /// [`capture_template`] for what the body may not do.
    pub(crate) fn capture<F>(stream: &CudaStream, body: F) -> Result<Graph, GpuError>
    where
        F: FnOnce(&CudaStream) -> Result<(), GpuError>,
    {
        let graph = capture_template(stream, body)?;

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
        let captured = Graph {
            graph,
            exec,
            nodes,
            ctx: Arc::clone(stream.context()),
        };
        // The upload the first launch of an exec would otherwise do inside
        // its own call, done once here on the stream the replays use.
        // SAFETY: exec is the live instantiation above; the stream is the
        // live one it was captured on.
        let rc = unsafe { sys::cuGraphUpload(captured.exec, stream.cu_stream()) };
        cu(rc, "cuGraphUpload")?;
        Ok(captured)
    }

    /// Enqueue one replay on `stream`. Asynchronous; the caller synchronizes.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), GpuError> {
        // SAFETY: exec is a live instantiated graph; the stream belongs to the
        // same context.
        unsafe { launch_exec(self.exec, stream.cu_stream()) }
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
    /// function with a sync mode, a stream memory operation) became, and for
    /// a kernel node the entry it launches and its geometry.
    pub fn nodes(&self) -> Result<Vec<NodeInfo>, GpuError> {
        self.handles()?
            .into_iter()
            // SAFETY: every handle is a node of this template, which lives
            // as long as `self`.
            .map(|node| unsafe { node_info(node) })
            .collect()
    }

    /// The most nodes on one dependency path of the template: the node count
    /// of a capture on one stream, less what each fork's shorter side
    /// ([`Branch`]) runs beside its longer one.
    pub fn span(&self) -> Result<usize, GpuError> {
        const WHAT: &str = "Graph::span";
        let handles = self.handles()?;
        let mut n_edges = 0usize;
        // SAFETY: with both node arrays null the call writes only the edge
        // count of this live template.
        let rc = unsafe {
            sys::cuGraphGetEdges_v2(
                self.graph,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut n_edges,
            )
        };
        cu(rc, "cuGraphGetEdges_v2")?;
        let mut from: Vec<sys::CUgraphNode> = vec![ptr::null_mut(); n_edges];
        let mut to: Vec<sys::CUgraphNode> = vec![ptr::null_mut(); n_edges];
        // SAFETY: both arrays hold `n_edges` slots, the count just reported
        // for this template; the call writes at most that many edges.
        let rc = unsafe {
            sys::cuGraphGetEdges_v2(
                self.graph,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                ptr::null_mut(),
                &mut n_edges,
            )
        };
        cu(rc, "cuGraphGetEdges_v2")?;
        let index: std::collections::HashMap<sys::CUgraphNode, usize> =
            handles.iter().enumerate().map(|(i, &h)| (h, i)).collect();
        let at = |h: &sys::CUgraphNode| {
            index.get(h).copied().ok_or(GpuError::state(
                WHAT,
                "an edge between the template's nodes",
            ))
        };
        let mut after: Vec<Vec<usize>> = vec![Vec::new(); handles.len()];
        let mut waits = vec![0usize; handles.len()];
        for (f, t) in from.iter().zip(&to).take(n_edges) {
            let (f, t) = (at(f)?, at(t)?);
            after[f].push(t);
            waits[t] += 1;
        }
        // Longest path by topological order: a node's depth is one more than
        // its deepest dependency's.
        let mut depth = vec![1usize; handles.len()];
        let mut ready: Vec<usize> = (0..handles.len()).filter(|&i| waits[i] == 0).collect();
        let mut done = 0usize;
        while let Some(v) = ready.pop() {
            done += 1;
            for &w in &after[v] {
                depth[w] = depth[w].max(depth[v] + 1);
                waits[w] -= 1;
                if waits[w] == 0 {
                    ready.push(w);
                }
            }
        }
        if done != handles.len() {
            return Err(GpuError::state(WHAT, "an acyclic template"));
        }
        Ok(depth.into_iter().max().unwrap_or(0))
    }

    /// The template's node handles, in the order `cuGraphGetNodes` returns them.
    fn handles(&self) -> Result<Vec<sys::CUgraphNode>, GpuError> {
        let mut handles: Vec<sys::CUgraphNode> = vec![ptr::null_mut(); self.nodes];
        let mut n = self.nodes;
        // SAFETY: the array holds `n` slots, the count the driver reported
        // for this template at capture; it writes at most `n` handles.
        let rc = unsafe { sys::cuGraphGetNodes(self.graph, handles.as_mut_ptr(), &mut n) };
        cu(rc, "cuGraphGetNodes")?;
        handles.truncate(n);
        Ok(handles)
    }
}

/// What the driver reports of `node`: its type, a host node's sync mode, and
/// a kernel node's entry name and geometry. The one reader of these facts:
/// [`Graph::nodes`] and the gates' template walk both call it.
///
/// # Safety
///
/// `node` is a node of a live graph template.
pub unsafe fn node_info(node: sys::CUgraphNode) -> Result<NodeInfo, GpuError> {
    let mut kind: sys::CUgraphNodeType = 0;
    // SAFETY: `node` is a node of a live template (the caller's contract).
    let rc = unsafe { sys::cuGraphNodeGetType(node, &mut kind) };
    cu(rc, "cuGraphNodeGetType")?;
    let host_sync = if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST {
        // SAFETY: all-zero is a valid value of this plain C struct
        // (integers, pointers, a nullable function pointer), which the
        // driver then fills for `node`.
        let mut params: sys::CUgraphNodeParams = unsafe { std::mem::zeroed() };
        // SAFETY: `node` is a live host node and `params` is a writable
        // struct of the type the call fills.
        let rc = unsafe { sys::cuGraphNodeGetParams(node, &mut params) };
        cu(rc, "cuGraphNodeGetParams")?;
        // SAFETY: for a host node the driver fills the `host` member of the
        // union.
        Some(unsafe { params.__bindgen_anon_1.host.syncMode })
    } else {
        None
    };
    let kernel = if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL {
        // SAFETY: `node` is a live kernel node (the caller's contract and
        // the type just read).
        Some(unsafe { kernel_node(node) }?)
    } else {
        None
    };
    Ok(NodeInfo {
        kind,
        host_sync,
        kernel,
    })
}

/// A kernel node's entry name and geometry. A node whose function the driver
/// cannot name, or names in bytes that are not UTF-8, is an error by name,
/// never an empty or lossy name.
///
/// # Safety
///
/// `node` is a kernel node of a live graph template.
unsafe fn kernel_node(node: sys::CUgraphNode) -> Result<KernelNode, GpuError> {
    const WHAT: &str = "Graph::nodes";
    // SAFETY: all-zero is a valid value of this plain C struct (integers and
    // nullable pointers), which the call then fills.
    let mut p: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    // SAFETY: `node` is a live kernel node and `p` the struct the call fills.
    let rc = unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut p) };
    cu(rc, "cuGraphKernelNodeGetParams_v2")?;
    let mut name: *const std::ffi::c_char = ptr::null();
    if p.func.is_null() {
        if p.kern.is_null() {
            return Err(GpuError::state(WHAT, "a kernel node with no function"));
        }
        // SAFETY: the node launches the library kernel `p.kern`, a live
        // handle; `name` is a local the call writes.
        let rc = unsafe { sys::cuKernelGetName(&mut name, p.kern) };
        cu(rc, "cuKernelGetName")?;
    } else {
        // SAFETY: the node launches the module function `p.func`, a live
        // handle; `name` is a local the call writes.
        let rc = unsafe { sys::cuFuncGetName(&mut name, p.func) };
        cu(rc, "cuFuncGetName")?;
    }
    if name.is_null() {
        return Err(GpuError::state(
            WHAT,
            "a kernel node whose function has no name",
        ));
    }
    // SAFETY: the driver hands back a NUL-terminated name it owns for the
    // function's lifetime; it is copied out at once.
    let name = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| GpuError::state(WHAT, "a kernel entry name that is not UTF-8"))?
        .to_owned();
    Ok(KernelNode {
        name,
        grid: [p.gridDimX, p.gridDimY, p.gridDimZ],
        block: [p.blockDimX, p.blockDimY, p.blockDimZ],
    })
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

/// A second stream that work can run on beside one main stream, captured or
/// eager alike: [`Branch::fork`] makes the branch wait for everything
/// enqueued on the main stream so far, and [`Forked::join`] makes the main
/// stream wait for everything enqueued on the branch since.
///
/// Inside a capture on the main stream the fork's wait pulls the branch into
/// the same capture, and the fork then sets the branch's capture
/// dependencies to the main stream's, so each fork's work is a parallel path
/// of the graph: an edge from the fork point and one into the join point, no
/// node of its own, on every fork of one capture. Outside a capture the same
/// two event waits order the eager run. The branch's stream and both events
/// are made once, at load: `fork` and `join` only record, wait and set
/// dependencies, which are capturable and allocate nothing. A branch forks
/// only from the main stream it was made for, and one fork is open at a
/// time: [`Forked`] borrows the branch until it is joined or dropped.
pub struct Branch {
    main: Arc<CudaStream>,
    stream: Arc<CudaStream>,
    forked: CudaEvent,
    joined: CudaEvent,
}

impl Branch {
    /// A branch of `main`, on a stream of `main`'s context. Load-time only.
    pub fn new(main: Arc<CudaStream>) -> Result<Branch, GpuError> {
        let ctx = main.context();
        Ok(Branch {
            stream: crate::role_stream(ctx, crate::StreamRole::Engine)?,
            forked: ctx.new_event(None)?,
            joined: ctx.new_event(None)?,
            main,
        })
    }

    /// Start the branch after everything enqueued on the main stream so far.
    /// The branch's work goes on [`Forked::stream`]; the fork must be joined
    /// before the main stream's next reader of that work is enqueued.
    pub fn fork(&mut self) -> Result<Forked<'_>, GpuError> {
        self.forked.record(&self.main)?;
        self.stream.wait(&self.forked)?;
        // The wait adds the fork point to the branch's capture dependencies
        // and keeps the branch's last node from an earlier fork of the same
        // capture. That fork was joined into this main stream before this
        // fork point (its `Forked` held the branch until its join or drop),
        // so setting the branch's set to main's drops no ordering.
        let mut status: sys::CUstreamCaptureStatus = 0;
        let mut deps: *const sys::CUgraphNode = ptr::null();
        let mut data: *const sys::CUgraphEdgeData = ptr::null();
        let mut n = 0usize;
        // SAFETY: `main` is a live stream and the four outputs are locals the
        // call writes.
        let rc = unsafe {
            sys::cuStreamGetCaptureInfo_v3(
                self.main.cu_stream(),
                &mut status,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut deps,
                &mut data,
                &mut n,
            )
        };
        cu(rc, "cuStreamGetCaptureInfo_v3")?;
        match status {
            sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE => {}
            sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                // SAFETY: `deps` and `data` are the driver's arrays of `n`
                // entries for `main`, valid until the next call on `main`;
                // this call is on the branch and only reads them.
                let rc = unsafe {
                    sys::cuStreamUpdateCaptureDependencies_v2(
                        self.stream.cu_stream(),
                        deps.cast_mut(),
                        data,
                        n,
                        sys::CUstreamUpdateCaptureDependencies_flags_enum_CU_STREAM_SET_CAPTURE_DEPENDENCIES,
                    )
                };
                cu(rc, "cuStreamUpdateCaptureDependencies_v2")?;
            }
            _ => {
                return Err(GpuError::state(
                    "Branch::fork",
                    "the main stream's capture is invalidated",
                ));
            }
        }
        Ok(Forked {
            branch: self,
            open: true,
        })
    }
}

/// A [`Branch`] forked from its main stream and not joined yet.
///
/// Dropped without [`Forked::join`] (an early error return between the two),
/// it joins anyway and ignores the driver's answer: the error already on its
/// way out is the one reported, and a capture is never left with a branch
/// its end would refuse as unjoined.
#[must_use = "a fork is joined before its work is read"]
pub struct Forked<'a> {
    branch: &'a mut Branch,
    open: bool,
}

impl Forked<'_> {
    /// The branch stream the forked work is enqueued on.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.branch.stream
    }

    /// Make the main stream wait for everything enqueued on the branch.
    pub fn join(mut self) -> Result<(), GpuError> {
        self.open = false;
        self.branch.joined.record(&self.branch.stream)?;
        self.branch.main.wait(&self.branch.joined)?;
        Ok(())
    }
}

impl Drop for Forked<'_> {
    fn drop(&mut self) {
        if self.open {
            let _ = self.branch.joined.record(&self.branch.stream);
            let _ = self.branch.main.wait(&self.branch.joined);
        }
    }
}

/// Whether `stream` is recording a capture right now.
pub fn capturing(stream: &CudaStream) -> Result<bool, GpuError> {
    let mut status: sys::CUstreamCaptureStatus = 0;
    // SAFETY: the stream is live and `status` is a local the call writes.
    let rc = unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut status) };
    cu(rc, "cuStreamIsCapturing")?;
    Ok(status != sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE)
}

/// The flags beside `PORTABLE | DEVICEMAP` of memory the card and the CPU
/// both read ([`MappedHost::read_in_place`]): `WRITECOMBINED` on an
/// integrated device, whose card reads such pages at the rate of its own
/// memory; none on a card across PCIe, whose CPU reads write-combined pages
/// uncached.
fn in_place_flags(integrated: bool) -> u32 {
    if integrated {
        sys::CU_MEMHOSTALLOC_WRITECOMBINED
    } else {
        0
    }
}

/// Pinned host memory mapped into the device's address space
/// (`cuMemHostAlloc` with `PORTABLE | DEVICEMAP`, and `WRITECOMBINED` from
/// [`MappedHost::read_in_place`] on an integrated device): the host reaches
/// it at `host`, kernels, copies and stream memory operations at `dev`, from
/// every context of the process. Zeroed by [`MappedHost::new`], freed on
/// drop. Or a shared file's mapping registered the same way
/// ([`MappedHost::shared_file`]), which another process that maps the file
/// reaches too. The one owner of mapped host memory: the host flags
/// ([`HostFlags`]) and the host tier's page are views over it. Its owner
/// keeps every stream and graph that addresses it from running past the
/// drop: it drains the stream, drops the graph first, or ends the process
/// without dropping it.
pub struct MappedHost {
    host: *mut u8,
    dev: sys::CUdeviceptr,
    bytes: usize,
    /// The context that allocated the page: held so it outlives the
    /// allocation, and bound when the page is freed, as cuda-core's owners do.
    ctx: Arc<CudaContext>,
    /// The shared file's mapping, unmapped after the drop unregisters it;
    /// `None` for the driver's own allocation.
    file: Option<memmap2::MmapMut>,
}

// SAFETY: this value alone frees or unregisters the allocation, once, in its
// drop (a shared file's other process holds its own mapping and
// registration); the host pointer is plain process memory any thread may
// reach, and every access to it goes through the methods below.
unsafe impl Send for MappedHost {}
// SAFETY: a host access through `&self` is an atomic word
// ([`MappedHost::atomic_u32`]) or a plain read of bytes the protocol that
// owns the span orders before it — the host tier reads a handoff image only
// after an Acquire load of the generation word saw the go that published it,
// and nothing writes it again before the host's Release signal; a host write
// takes `&mut self`.
unsafe impl Sync for MappedHost {}

impl MappedHost {
    /// `bytes` zeroed bytes of `ctx`'s mapped host memory. `what` names the
    /// allocation in a driver error. Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        what: &'static str,
    ) -> Result<MappedHost, GpuError> {
        let page = MappedHost::alloc(ctx, bytes, 0, what)?;
        // SAFETY: the allocation holds `bytes` writable bytes and nothing else
        // references it yet.
        unsafe { ptr::write_bytes(page.host, 0, bytes) };
        Ok(page)
    }

    /// [`MappedHost::new`] not zeroed, for bytes the card and the CPU both
    /// read in place: write-combined on an integrated device alone
    /// ([`in_place_flags`]). Its owner reads no byte it did not write.
    /// Load-time only.
    pub(crate) fn read_in_place(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        what: &'static str,
    ) -> Result<MappedHost, GpuError> {
        let flags = in_place_flags(crate::raw_device_integrated(ctx.cu_device())?);
        MappedHost::alloc(ctx, bytes, flags, what)
    }

    /// The one `cuMemHostAlloc` of mapped host memory, with `flags` beside
    /// `PORTABLE | DEVICEMAP`, its bytes as the driver gave them.
    fn alloc(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        flags: u32,
        what: &'static str,
    ) -> Result<MappedHost, GpuError> {
        ctx.bind_to_thread()?;
        let mut host: *mut c_void = ptr::null_mut();
        // SAFETY: the context is current on this thread (bound above) and
        // `host` is a live local the call writes.
        let rc = unsafe {
            sys::cuMemHostAlloc(
                &mut host,
                bytes,
                sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP | flags,
            )
        };
        cu(rc, what)?;
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `host` is the mapped allocation just made; the flags must be 0.
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0) };
        if let Err(e) = cu(rc, "cuMemHostGetDevicePointer_v2") {
            // SAFETY: `host` came from cuMemHostAlloc and is freed once, here.
            unsafe { sys::cuMemFreeHost(host) };
            return Err(e);
        }
        Ok(MappedHost {
            host: host.cast(),
            dev,
            bytes,
            ctx: Arc::clone(ctx),
            file: None,
        })
    }

    /// The `bytes` of the file at `path` (under `/dev/shm`, so host memory)
    /// mapped shared and registered as `ctx`'s mapped host memory
    /// (`cuMemHostRegister` with `PORTABLE | DEVICEMAP`): this process
    /// reaches them as it reaches [`MappedHost::new`]'s, and every other
    /// process that maps the file reaches the same bytes. With `create` the
    /// file is made `bytes` long and zeroed, and a file already at `path` is
    /// refused; without, the file must be `bytes` long. `bytes` is a whole
    /// number of host pages, as registration takes. `what` names the page in
    /// an error. A file made here is readable and writable by its user alone,
    /// and is removed again when the call fails. Load-time only.
    pub(crate) fn shared_file(
        ctx: &Arc<CudaContext>,
        path: &std::path::Path,
        bytes: usize,
        create: bool,
        what: &'static str,
    ) -> Result<MappedHost, GpuError> {
        if bytes == 0 || !bytes.is_multiple_of(SHARED_PAGE) {
            return Err(GpuError::shape(
                what,
                format!("a shared page of {bytes} bytes: a whole number of {SHARED_PAGE}"),
            ));
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(create)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .map_err(|e| GpuError::link(what, format!("{}: {e}", path.display())))?;
        let r = MappedHost::map_file(ctx, &file, path, bytes, create, what);
        if r.is_err() && create {
            drop(std::fs::remove_file(path));
        }
        r
    }

    /// [`MappedHost::shared_file`] once `file`, at `path`, is open.
    fn map_file(
        ctx: &Arc<CudaContext>,
        file: &std::fs::File,
        path: &std::path::Path,
        bytes: usize,
        create: bool,
        what: &'static str,
    ) -> Result<MappedHost, GpuError> {
        let io = |e: std::io::Error| GpuError::link(what, format!("{}: {e}", path.display()));
        if create {
            file.set_len(bytes as u64).map_err(io)?;
        }
        let len = file.metadata().map_err(io)?.len();
        if len != bytes as u64 {
            return Err(GpuError::link(
                what,
                format!("{} holds {len} bytes, the page {bytes}", path.display()),
            ));
        }
        // SAFETY: the process that made the file and the one that opened it
        // reach its bytes only through their registered mappings, by the
        // page's protocol (atomic words, and spans one side writes before the
        // word that hands them over). The file is made by its user alone
        // (mode 0600) under a name a tier server opens alone; that no other
        // process of that user maps it is assumed, not enforced.
        let mut map = unsafe { memmap2::MmapMut::map_mut(file) }.map_err(io)?;
        let host = map.as_mut_ptr();
        ctx.bind_to_thread()?;
        // SAFETY: the context is current on this thread (bound above);
        // `host` is the start of a live, page-aligned mapping of `bytes`
        // bytes, which outlives the registration (the drop unregisters it
        // before the map goes).
        let rc = unsafe {
            sys::cuMemHostRegister_v2(
                host.cast(),
                bytes,
                sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP,
            )
        };
        cu(rc, what)?;
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: the context is current (bound above), `dev` is a live local
        // the call writes, and `host` is the mapping registered just above;
        // the flags must be 0.
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host.cast(), 0) };
        if let Err(e) = cu(rc, "cuMemHostGetDevicePointer_v2") {
            // SAFETY: `host` was registered above and is unregistered once,
            // here.
            unsafe { sys::cuMemHostUnregister(host.cast()) };
            return Err(e);
        }
        Ok(MappedHost {
            host,
            dev,
            bytes,
            ctx: Arc::clone(ctx),
            file: Some(map),
        })
    }

    /// The device address of byte `off`, which the caller's layout keeps
    /// inside the allocation.
    pub fn dev_at(&self, off: usize) -> sys::CUdeviceptr {
        self.dev + off as u64
    }

    /// The allocation's length in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The host address of byte `off`. Computing it touches nothing; each
    /// access proves its own span.
    pub fn host_at(&self, off: usize) -> *mut u8 {
        self.host.wrapping_add(off)
    }

    /// Whether the `len` f32 at byte `off` lie inside the allocation, with no
    /// wrap for an `off` or `len` near `usize::MAX`.
    fn holds(&self, off: usize, len: usize) -> bool {
        len.checked_mul(4)
            .and_then(|n| off.checked_add(n))
            .is_some_and(|end| end <= self.bytes)
    }

    /// The u32 at byte `off` as an atomic; `None` when its address is not
    /// 4-aligned or it passes the allocation.
    pub fn atomic_u32(&self, off: usize) -> Option<&AtomicU32> {
        let word = self.host_at(off).cast::<AtomicU32>();
        if !word.is_aligned() || off.checked_add(4).is_none_or(|end| end > self.bytes) {
            return None;
        }
        // SAFETY: the word's address is aligned and the word is inside the
        // allocation (both checked above); `AtomicU32` has `u32`'s layout,
        // and every host access to a flag word goes through this view.
        Some(unsafe { &*word })
    }

    /// The `len` f32 at byte `off`, for the host to write; `None` when not
    /// 4-aligned or past the allocation. `&mut self` makes it the only host
    /// reference; the owner's protocol keeps the card off the span meanwhile.
    pub(crate) fn f32_mut(&mut self, off: usize, len: usize) -> Option<&mut [f32]> {
        if !off.is_multiple_of(4) || !self.holds(off, len) {
            return None;
        }
        // SAFETY: the span is inside the allocation and 4-aligned (checked
        // above); `&mut self` makes it the only host reference.
        Some(unsafe { std::slice::from_raw_parts_mut(self.host_at(off).cast::<f32>(), len) })
    }

    /// A copy of the `len` f32 at byte `off`, when inside the allocation.
    pub(crate) fn f32_copy(&self, off: usize, len: usize) -> Option<Vec<f32>> {
        if !off.is_multiple_of(4) || !self.holds(off, len) {
            return None;
        }
        let mut out = vec![0.0f32; len];
        // SAFETY: the span is inside the allocation and 4-aligned (checked
        // above); `out` is a distinct host buffer of `len` f32.
        unsafe {
            ptr::copy_nonoverlapping(self.host_at(off).cast::<f32>(), out.as_mut_ptr(), len);
        }
        Some(out)
    }
}

impl Drop for MappedHost {
    fn drop(&mut self) {
        // Freed under the context that allocated it, not whichever context
        // the thread last bound; an error is kept on that context.
        self.ctx.record_err(self.ctx.bind_to_thread());
        let rc = match self.file {
            // SAFETY: `host` is the mapping `shared_file` registered, whose
            // map is dropped after this body; it is unregistered once, here,
            // after every stream operation and graph that names it (the
            // owner's contract on the type).
            Some(_) => unsafe { sys::cuMemHostUnregister(self.host.cast()) },
            // SAFETY: `host` came from cuMemHostAlloc and is freed once, here,
            // after every stream operation and graph that names it (the
            // owner's contract on the type); the engine declares its graphs
            // before the bodies that own the views over it.
            None => unsafe { sys::cuMemFreeHost(self.host.cast()) },
        };
        self.ctx.record_err(driver(rc));
    }
}

/// The unit a shared page's bytes come in ([`MappedHost::shared_file`]): a
/// whole number of host pages on every host the engine runs on.
pub(crate) const SHARED_PAGE: usize = 1 << 16;

// ------------------------------------------------------ stream memory ops

/// An all-zero batch operation.
fn op_zero() -> sys::CUstreamBatchMemOpParams {
    // SAFETY: every member of the union is a plain C struct of integers, so
    // all-zero bytes are a valid value.
    unsafe { std::mem::zeroed() }
}

/// Wait until the u32 at `addr` is at least `value` in the driver's cyclic
/// order, `(int32_t)(*addr - value) >= 0`, not a plain unsigned `>=`.
pub fn op_wait_geq(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.waitValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WAIT_VALUE_32,
        address: addr,
        __bindgen_anon_1:
            sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
                value,
            },
        flags: sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
        alias: 0,
    };
    p
}

/// Write `value` to the u32 at `addr`.
pub fn op_write(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.writeValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WRITE_VALUE_32,
        address: addr,
        __bindgen_anon_1:
            sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
                value,
            },
        flags: sys::CUstreamWriteValue_flags_enum_CU_STREAM_WRITE_VALUE_DEFAULT,
        alias: 0,
    };
    p
}

/// A system-scope memory barrier: everything the stream wrote before it is
/// visible system-wide before anything after it.
pub fn op_barrier_sys() -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.memoryBarrier = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_BARRIER,
        flags: sys::CUstreamMemoryBarrier_flags_enum_CU_STREAM_MEMORY_BARRIER_TYPE_SYS,
    };
    p
}

/// Atomic reduction `*addr += value` on a u32 (wrapping).
pub fn op_add(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.atomicReduction = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpAtomicReductionParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_ATOMIC_REDUCTION,
        flags: 0,
        reductionOp: sys::CUstreamAtomicReductionOpType_enum_CU_STREAM_ATOMIC_REDUCTION_OP_ADD,
        dataType: sys::CUstreamAtomicReductionDataType_enum_CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_32,
        address: addr,
        value: u64::from(value),
        alias: 0,
    };
    p
}

/// Enqueue `ops` on `stream` as one batch of stream memory operations: one
/// graph node when captured.
///
/// # Safety
///
/// - Every operation's address must point at a 4-aligned u32 of memory the
///   device reaches in `stream`'s context.
/// - That memory must stay allocated and mapped until the batch has run, and
///   for a captured batch until every launch of the graph has run: the graph
///   bakes the addresses in.
pub unsafe fn mem_batch(
    stream: &CudaStream,
    ops: &mut [sys::CUstreamBatchMemOpParams],
    what: &'static str,
) -> Result<(), GpuError> {
    let n = u32::try_from(ops.len()).map_err(|_| GpuError::shape(what, "batch too long"))?;
    // SAFETY: `ops` is a live array of `n` initialized operations the driver
    // reads during the call (a capture copies them into the node), whose
    // addresses are the caller's contract; the flags must be 0.
    let rc = unsafe { sys::cuStreamBatchMemOp_v2(stream.cu_stream(), n, ops.as_mut_ptr(), 0) };
    cu(rc, what)
}

// ------------------------------------------------------------- host flags

/// Bytes between two flags of a [`HostFlags`]: each has a cache line of its
/// own, so a host store to one never shares a line with another.
const FLAG_STRIDE: usize = 64;

/// Operations in the batch [`HostFlags::enqueue_wait`] enqueues: the one
/// wait. A gate tells this batch from the host tier's go and wait by it.
pub const FLAG_WAIT_OPS: u32 = 1;

/// Flag words the host raises and a stream waits on: a view over mapped host
/// memory ([`MappedHost`]), one word per flag on its own line, zero at
/// allocation.
///
/// [`HostFlags::enqueue_wait`] holds a stream until the host raises a flag —
/// one batch memory-operation node when captured, a wait until the word is
/// non-zero. A captured wait carries its value fixed, so the host, not the
/// graph, rearms the flag: [`HostFlags::clear`] before the launch whose wait
/// it gates, [`HostFlags::raise`] once what the wait guards is in place.
/// A launched wait on a flag the host never raises waits forever: whoever
/// launches a gated wait raises its flag on every path, errors included.
pub struct HostFlags {
    page: MappedHost,
    len: usize,
}

impl HostFlags {
    /// `len` cleared flags of `ctx`. Load-time only.
    pub fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<HostFlags, GpuError> {
        const WHAT: &str = "HostFlags::new";
        if len == 0 {
            return Err(GpuError::shape(WHAT, "no flags"));
        }
        let bytes = len
            .checked_mul(FLAG_STRIDE)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{len} flags pass usize bytes")))?;
        Ok(HostFlags {
            page: MappedHost::new(ctx, bytes, "cuMemHostAlloc (host flags)")?,
            len,
        })
    }

    /// Flags held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Never: [`HostFlags::new`] refuses zero flags.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Flag `i`'s host word, or the refusal of an index past the flags.
    fn word(&self, i: usize, what: &'static str) -> Result<&AtomicU32, GpuError> {
        let refuse = || GpuError::shape(what, format!("flag {i} of {} flags", self.len));
        if i >= self.len {
            return Err(refuse());
        }
        self.page.atomic_u32(i * FLAG_STRIDE).ok_or_else(refuse)
    }

    /// Lower flag `i`: a wait on it enqueued from here on holds its stream.
    /// Call it only while no launched wait on the flag is pending.
    pub fn clear(&self, i: usize) -> Result<(), GpuError> {
        self.word(i, "HostFlags::clear")?
            .store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Raise flag `i`, after every host store the wait guards (release
    /// order): a stream held on it goes on.
    pub fn raise(&self, i: usize) -> Result<(), GpuError> {
        self.word(i, "HostFlags::raise")?
            .store(1, Ordering::Release);
        Ok(())
    }

    /// Whether flag `i` is raised.
    pub fn raised(&self, i: usize) -> Result<bool, GpuError> {
        Ok(self.word(i, "HostFlags::raised")?.load(Ordering::Acquire) != 0)
    }

    /// Enqueue on `stream` a wait until flag `i` is raised: one batch of
    /// [`FLAG_WAIT_OPS`] stream memory operation, one graph node when
    /// captured. Capturable, allocation-free.
    pub fn enqueue_wait(&self, stream: &CudaStream, i: usize) -> Result<(), GpuError> {
        self.word(i, "HostFlags::enqueue_wait")?;
        let mut ops = [op_wait_geq(self.page.dev_at(i * FLAG_STRIDE), 1)];
        const _: () = assert!(FLAG_WAIT_OPS == 1);
        // SAFETY: flag `i`'s word lies in the page, 4-aligned (checked
        // above); the page goes only with the flags, whose owner keeps every
        // wait on them from running past that ([`MappedHost`]'s contract).
        unsafe { mem_batch(stream, &mut ops, "cuStreamBatchMemOp_v2 (host flag wait)") }
    }
}

/// Rows the host writes after the launch of the graph that reads them: a
/// page-locked staging of rows of `width` values and a host flag a row
/// ([`HostFlags`]). The graph's first reader of a row enqueues its arrival
/// ([`StagedRows::enqueue`]): the wait on the row's flag, then one copy of
/// the row into a device buffer. So the card runs the work before that
/// reader while the host writes the row, and never reads the row before the
/// host wrote it.
///
/// Contract: the host lowers a row's flag before the launch whose arrival it
/// gates ([`StagedRows::clear`]) and writes the row only while its flag is
/// lowered and no copy of the row is in flight, that is after the stream
/// synchronized past the launch before. It raises the flag on every path
/// after the launch, an error included ([`StagedRows::deliver`]): a launched
/// arrival waits until then, and the card then reads the row as the staging
/// held it.
pub struct StagedRows<T: DeviceCopy> {
    flags: HostFlags,
    staging: PinnedHostBuffer<T>,
    width: usize,
}

impl<T: DeviceCopy> StagedRows<T> {
    /// `rows` zeroed rows of `width` values in `ctx`, each with its flag
    /// lowered; `op` names the staging's allocation in an error. Load-time
    /// only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        rows: usize,
        width: usize,
        op: &'static str,
    ) -> Result<StagedRows<T>, GpuError> {
        let len = rows.checked_mul(width).ok_or_else(|| {
            GpuError::shape("StagedRows::new", format!("{rows} rows of {width} values"))
        })?;
        let staging = PinnedHostBuffer::zeroed(ctx, len).map_err(|source| GpuError::Driver {
            op: Some(op),
            source,
        })?;
        Ok(StagedRows {
            flags: HostFlags::new(ctx, rows)?,
            staging,
            width,
        })
    }

    /// Lower row `row`'s flag, before the launch whose arrival it gates.
    pub fn clear(&self, row: usize) -> Result<(), GpuError> {
        self.flags.clear(row)
    }

    /// Row `row`'s staging written by `write`, then its flag raised. The
    /// flag is raised on every path: a `write` that fails, or a row past the
    /// staging, returns its error with the flag raised.
    pub fn deliver(
        &mut self,
        row: usize,
        write: impl FnOnce(&mut [T]) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let (width, held) = (self.width, self.staging.len());
        let staged = self
            .staging
            .get_mut(row * width..(row + 1) * width)
            .ok_or_else(|| {
                GpuError::shape(
                    "StagedRows::deliver",
                    format!("row {row} of {width} values in a staging of {held}"),
                )
            })
            .and_then(write);
        let raised = self.flags.raise(row);
        staged.and(raised)
    }

    /// Enqueue row `row`'s arrival on `stream`: the wait on its flag, then
    /// the row copied over `dst`'s values from value `at` on. Two nodes when
    /// captured (a batch memory operation, a copy). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        row: usize,
        dst: &mut DeviceBuffer<T>,
        at: usize,
    ) -> Result<(), GpuError> {
        self.enqueue_prefix(stream, row, self.width, dst, at)
    }

    /// [`StagedRows::enqueue`] of the row's first `len` values: the arrival
    /// of a walk that reads fewer values than a row holds.
    pub fn enqueue_prefix(
        &self,
        stream: &CudaStream,
        row: usize,
        len: usize,
        dst: &mut DeviceBuffer<T>,
        at: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "StagedRows::enqueue";
        let width = self.width;
        if len > width {
            return Err(GpuError::shape(
                WHAT,
                format!("{len} values of a row of {width}"),
            ));
        }
        let src = self
            .staging
            .get(row * width..row * width + len)
            .ok_or_else(|| {
                let held = self.staging.len();
                GpuError::shape(
                    WHAT,
                    format!("row {row} of {width} values in a staging of {held}"),
                )
            })?;
        if at.checked_add(len).is_none_or(|end| end > dst.len()) {
            return Err(GpuError::shape(
                WHAT,
                format!("{len} values of a row at {at} of {}", dst.len()),
            ));
        }
        self.flags.enqueue_wait(stream, row)?;
        // SAFETY: the destination is values `at .. at + len` of `dst`
        // (checked above), borrowed mutably for the enqueue; the source is
        // the row in page-locked memory this value owns, whose owner drops
        // the graphs that captured the copy before it, and the host writes
        // the row only while the copy waits on its lowered flag or none is
        // in flight (the type's contract).
        let rc = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst.cu_deviceptr() + (at * size_of::<T>()) as u64,
                src.as_ptr().cast(),
                size_of_val(src),
                stream.cu_stream(),
            )
        };
        cu(rc, "cuMemcpyHtoDAsync_v2 (a staged row's arrival)")
    }
}

/// One node of a captured graph as the driver reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInfo {
    /// The node's `CUgraphNodeType` (kernel, memcpy, host, batch memop, …).
    pub kind: sys::CUgraphNodeType,
    /// For a host node, the `CUhostTaskSyncMode` it carries.
    pub host_sync: Option<u32>,
    /// For a kernel node, the entry it launches and its geometry.
    pub kernel: Option<KernelNode>,
}

/// A kernel node's launch as the driver reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelNode {
    /// The entry's name (`cuFuncGetName`): the PTX `.entry` the node runs.
    pub name: String,
    /// The grid, in blocks: x, y, z.
    pub grid: [u32; 3],
    /// The block, in threads: x, y, z.
    pub block: [u32; 3],
}

impl Drop for Graph {
    fn drop(&mut self) {
        // Destroyed under the context it was captured in; an error is kept on
        // that context.
        self.ctx.record_err(self.ctx.bind_to_thread());
        // SAFETY: both handles were created in `capture` and are destroyed
        // exactly once here; the exec first, then the template it came from.
        let (exec, graph) = unsafe {
            (
                sys::cuGraphExecDestroy(self.exec),
                sys::cuGraphDestroy(self.graph),
            )
        };
        self.ctx.record_err(driver(exec));
        self.ctx.record_err(driver(graph));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Branch, FLAG_WAIT_OPS, Graph, HostFlags, StagedRows, capturing, cu, in_place_flags,
    };
    use crate::{DeviceTensor, Gpu, GpuError, Q8Act, capsync};
    use cuda_core::{DeviceBuffer, PinnedHostBuffer, sys};
    use std::ffi::c_void;
    use std::panic::AssertUnwindSafe;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Memory the card and the CPU both read is write-combined on an
    /// integrated device alone: on a card across PCIe the CPU's reads of
    /// such pages go uncached.
    #[test]
    fn read_in_place_is_write_combined_on_an_integrated_device_alone() {
        assert_eq!(
            in_place_flags(true),
            sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            "an integrated device"
        );
        assert_eq!(in_place_flags(false), 0, "a card across PCIe");
    }

    /// A captured wait on a flag holds the copy behind it until the host
    /// raises the flag: the graph is launched before the host writes the
    /// staging, the host sleeps, writes it and raises, and the copy lands
    /// what the host wrote after the launch. A replay after a clear waits
    /// again. Without the wait the copy reads the staging as it was at the
    /// launch.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_host_flag_holds_a_copy_until_raised() {
        const N: usize = 1024;
        let (ctx, stream) = capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let flags = HostFlags::new(&ctx, 2).expect("two host flags");
        let mut staging = PinnedHostBuffer::<u32>::zeroed(&ctx, N).expect("pinned staging");
        let dst = DeviceBuffer::<u32>::zeroed(&stream, N).expect("a device buffer");
        stream
            .synchronize()
            .expect("the allocation lands before capture");
        let src = staging.as_ptr();
        let graph = Graph::capture(&stream, |s| {
            flags.enqueue_wait(s, 1)?;
            // SAFETY: `src` is the pinned staging of N words, `dst` a device
            // buffer of N words; both outlive every launch below.
            let rc = unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst.cu_deviceptr(),
                    src.cast(),
                    N * size_of::<u32>(),
                    s.cu_stream(),
                )
            };
            cu(rc, "cuMemcpyHtoDAsync_v2")
        })
        .expect("the capture");
        let nodes = graph.nodes().expect("the node list");
        assert_eq!(nodes.len(), 2, "{nodes:?}");
        assert_eq!(
            nodes[0].kind,
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            "{nodes:?}"
        );
        for round in 1..=2u32 {
            flags.clear(1).expect("clear");
            graph.launch(&stream).expect("a launch");
            std::thread::sleep(Duration::from_millis(20));
            for (i, w) in staging.as_mut_slice().iter_mut().enumerate() {
                *w = round * 0x1000_0000 + i as u32;
            }
            flags.raise(1).expect("raise");
            stream.synchronize().expect("the replay");
            let got = dst.to_host_vec(&stream).expect("readback");
            let want: Vec<u32> = (0..N as u32).map(|i| round * 0x1000_0000 + i).collect();
            let stale = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            assert_eq!(
                stale, 0,
                "round {round}: {stale} of {N} words are not what the host wrote after the \
                 launch — the copy ran before the flag was raised"
            );
        }
        assert!(!flags.raised(0).expect("flag 0"), "flag 0 was never raised");
        assert_eq!(FLAG_WAIT_OPS, 1);
    }

    /// A row the host writes after the launch arrives where the graph reads
    /// it: the captured arrival of row 1 of two (the wait, the copy to value
    /// `AT` on) is launched, the host sleeps, then writes and delivers the
    /// row, and the buffer holds what the host wrote after the launch from
    /// `AT` on and nothing around it. A replay after a clear waits again. A
    /// write that fails is returned and raises the flag all the same: the
    /// replay ends. A row past the staging and a copy past the buffer are
    /// refused by name before anything is enqueued. Without the wait the
    /// copy reads the staging as it stood at the launch.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_staged_row_arrives_after_the_launch() {
        const W: usize = 4096;
        const AT: usize = 5;
        let (ctx, stream) = capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let mut rows =
            StagedRows::<f32>::new(&ctx, 2, W, "cuMemAllocHost (test)").expect("two staged rows");
        let mut dst = DeviceBuffer::<f32>::zeroed(&stream, AT + W + 7).expect("a device buffer");
        stream
            .synchronize()
            .expect("the allocation lands before capture");
        let graph =
            Graph::capture(&stream, |s| rows.enqueue(s, 1, &mut dst, AT)).expect("the capture");
        for round in 1..=2usize {
            rows.clear(1).expect("clear");
            graph.launch(&stream).expect("a launch");
            std::thread::sleep(Duration::from_millis(20));
            let want: Vec<f32> = (0..W).map(|i| (round * W + i) as f32).collect();
            rows.deliver(1, |row| {
                row.copy_from_slice(&want);
                Ok(())
            })
            .expect("the delivery");
            stream.synchronize().expect("the replay");
            let got = dst.to_host_vec(&stream).expect("readback");
            let stale = got[AT..AT + W]
                .iter()
                .zip(&want)
                .filter(|(g, w)| g != w)
                .count();
            assert_eq!(
                stale, 0,
                "round {round}: {stale} of {W} values are not what the host delivered after \
                 the launch"
            );
            let around = got[..AT].iter().chain(&got[AT + W..]);
            assert!(
                around.copied().all(|v| v == 0.0),
                "round {round}: around the row"
            );
        }
        rows.clear(1).expect("clear");
        graph.launch(&stream).expect("a launch");
        let failed = rows.deliver(1, |_| {
            Err(GpuError::state(
                "hw_staged_row_arrives_after_the_launch",
                "a failed write",
            ))
        });
        assert!(failed.is_err(), "a failed write is returned");
        stream
            .synchronize()
            .expect("the replay ends: the flag is raised on a failed write too");
        assert!(rows.deliver(2, |_| Ok(())).is_err(), "row 2 of two");
        assert!(
            rows.enqueue(&stream, 1, &mut dst, AT + 8).is_err(),
            "a copy past the buffer"
        );
        let nodes = graph.nodes().expect("the node list");
        assert_eq!(nodes.len(), 2, "{nodes:?}");
    }

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
        let (_ctx, stream) = capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
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
        assert!(nodes.iter().all(|n| n.kernel.is_none()), "{nodes:?}");
        assert!(
            nodes
                .iter()
                .any(|n| n.kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET),
            "{nodes:?}"
        );
    }

    /// A branch forked three times in one capture is three parallel paths:
    /// each fork's work depends on its fork point alone, and it and main's
    /// node beside it lead into one join point, the next round's fork point.
    /// A branch that kept its node of the round before as a dependency would
    /// carry one edge more on every fork after the first. The middle round
    /// drops its fork instead of joining it, and the drop records the join.
    /// The span counts one of each round's two parallel nodes.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_branch_forks_one_at_a_time_from_its_fork_point() {
        const ROUNDS: usize = 3;
        const CAP: usize = 64;
        let (_ctx, stream) = capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let mut branch = Branch::new(Arc::clone(&stream)).expect("a branch");
        let mut on_main = DeviceBuffer::<u32>::zeroed(&stream, 64).expect("a device buffer");
        let mut beside = DeviceBuffer::<u32>::zeroed(&stream, 64).expect("a device buffer");
        stream
            .synchronize()
            .expect("the allocations land before capture");
        let graph = Graph::capture(&stream, |s| {
            on_main.zero_async(s)?;
            for r in 0..ROUNDS {
                let fork = branch.fork()?;
                beside.zero_async(fork.stream())?;
                on_main.zero_async(s)?;
                if r == 1 {
                    drop(fork);
                } else {
                    fork.join()?;
                }
                on_main.zero_async(s)?;
            }
            Ok(())
        })
        .expect("the capture");
        let mut from = [std::ptr::null_mut(); CAP];
        let mut to = [std::ptr::null_mut(); CAP];
        let mut n = CAP;
        // SAFETY: `graph.graph` is the live template and both arrays hold
        // `n` slots; the call writes at most `n` edges and their count.
        let rc = unsafe {
            sys::cuGraphGetEdges_v2(
                graph.graph,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut n,
            )
        };
        cu(rc, "cuGraphGetEdges_v2").expect("the edges");
        assert_eq!(graph.node_count(), 1 + 3 * ROUNDS, "the captured nodes");
        assert_eq!(
            graph.span().expect("the span"),
            1 + 2 * ROUNDS,
            "a round's two parallel nodes add one to the span"
        );
        assert_eq!(
            n,
            4 * ROUNDS,
            "{n} edges for {ROUNDS} rounds, want 4 a round"
        );
        let edges: Vec<_> = from.into_iter().zip(to).take(n).collect();
        let deps = |v| -> Vec<_> {
            let mut d: Vec<_> = edges.iter().filter(|e| e.1 == v).map(|e| e.0).collect();
            d.sort();
            d
        };
        let dependents = |v| -> Vec<_> {
            let mut d: Vec<_> = edges.iter().filter(|e| e.0 == v).map(|e| e.1).collect();
            d.sort();
            d
        };
        let mut point = edges
            .iter()
            .map(|e| e.0)
            .find(|&v| deps(v).is_empty())
            .expect("main's first node");
        for r in 0..ROUNDS {
            let paths = dependents(point);
            assert_eq!(paths.len(), 2, "round {r}: the fork point starts two paths");
            for &v in &paths {
                assert_eq!(
                    deps(v),
                    [point],
                    "round {r}: a path depends on its fork point alone"
                );
            }
            let join = dependents(paths[0]);
            assert_eq!(
                join,
                dependents(paths[1]),
                "round {r}: both paths lead into one node"
            );
            let [join] = join[..] else {
                panic!(
                    "round {r}: the paths lead into {} nodes, want one",
                    join.len()
                );
            };
            assert_eq!(
                deps(join),
                paths,
                "round {r}: the join point waits on the two paths alone"
            );
            point = join;
        }
    }

    /// The step shape of the capture tests below: one [`ROWS`] × [`K`] Q4_K
    /// weight against one f32 column.
    const K: usize = 2048;
    const ROWS: usize = 2048;

    /// `rows` Q4_K rows of `n_sb` super-blocks as u32 words, from a
    /// fixed-seed xorshift64: word 0 of each super-block carries `d` and
    /// `dmin` as positive normal f16 (no NaN or Inf enters), and any pattern
    /// of the other 35 words is a valid Q4_K block. `seed` must be nonzero.
    fn q4k_words(rows: usize, n_sb: usize, seed: u64) -> Vec<u32> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let half = |r: u64| -> u32 { ((1 + (r % 9) as u32) << 10) | ((r >> 32) as u32 & 0x3ff) };
        let mut out = Vec::with_capacity(rows * n_sb * 36);
        for _ in 0..rows * n_sb {
            let (d, dmin) = (half(next()), half(next()));
            out.push(d | (dmin << 16));
            for _ in 1..36 {
                out.push((next() >> 32) as u32);
            }
        }
        out
    }

    /// Resident weight, column, q8_1 scratch and output of a two-kernel
    /// sequence: quantize the column, then the Q4_K gemv over it.
    struct Step {
        w: DeviceTensor<u32>,
        x: DeviceBuffer<f32>,
        act: Q8Act,
        y: DeviceBuffer<f32>,
    }

    impl Step {
        fn new(gpu: &Gpu) -> Step {
            let s = gpu.stream();
            let n_sb = K / 256;
            let words = q4k_words(ROWS, n_sb, 0x9e37_79b9_7f4a_7c15);
            let w = DeviceTensor::upload(s, &words, ROWS, 36 * n_sb).expect("the weight");
            let x: Vec<f32> = (0..K).map(|i| (i * 37 % 97) as f32 / 48.0 - 1.0).collect();
            let x = DeviceBuffer::from_host(s, &x).expect("the column");
            let act = Q8Act::with_k(s, 1, K).expect("the q8_1 scratch");
            let y = DeviceBuffer::<f32>::zeroed(s, ROWS).expect("the output");
            s.synchronize()
                .expect("the uploads land before any capture");
            Step { w, x, act, y }
        }

        fn enqueue(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
            gpu.enqueue_quantize_q8_1(&self.x, &mut self.act)?;
            gpu.enqueue_gemv_q4k(&self.w, &self.act, &mut self.y)
        }

        /// The sequence enqueued eagerly: what it wrote to `y`.
        fn eager(&mut self, gpu: &Gpu) -> Vec<f32> {
            self.enqueue(gpu).expect("the eager sequence");
            gpu.stream().synchronize().expect("the eager run");
            let y = self.y.to_host_vec(gpu.stream()).expect("readback");
            assert!(
                y.iter().all(|v| v.is_finite()) && y.iter().any(|&v| v != 0.0),
                "the eager output is not finite and nonzero, so a replay that runs nothing \
                 could match it"
            );
            y
        }

        /// `y` zeroed, the sequence captured and replayed once: the graph's
        /// node count and what the replay wrote.
        fn replayed(&mut self, gpu: &Gpu) -> (usize, Vec<f32>) {
            let s = gpu.stream();
            self.y.zero_async(s).expect("zero y");
            s.synchronize().expect("the zero lands before the capture");
            let graph = gpu.capture(|_| self.enqueue(gpu)).expect("the capture");
            graph.launch(s).expect("a replay");
            s.synchronize().expect("the replay");
            (graph.node_count(), self.y.to_host_vec(s).expect("readback"))
        }

        /// A capture of the sequence records its two launches and replays
        /// `eager` bit for bit; `after` names what ran on the stream before.
        fn assert_replays(&mut self, gpu: &Gpu, eager: &[f32], after: &str) {
            let (nodes, got) = self.replayed(gpu);
            assert_eq!(
                nodes, 2,
                "after {after}: the capture recorded {nodes} nodes"
            );
            let off = eager
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                off, 0,
                "after {after}: {off} of {ROWS} replayed outputs differ from the eager bits"
            );
        }
    }

    /// Eager = replay: the two-kernel sequence captured into a graph records
    /// two nodes, and one replay over a zeroed output writes the bytes the
    /// same sequence enqueued eagerly wrote — the same launches on the same
    /// addresses.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_two_kernel_capture_replays_the_eager_bytes() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let mut step = Step::new(&gpu);
        let eager = step.eager(&gpu);
        step.assert_replays(&gpu, &eager, "an eager run");
    }

    /// `nodes` names each kernel node's entry and geometry in the order the
    /// launches were enqueued: the two-kernel sequence comes back as the
    /// quantizer (one block of 32 threads per half super-block of the one
    /// column) and then the Q4_K gemv (eight rows per block of 256 threads).
    /// A structure gate that compares launch lists by name reads this order.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_nodes_name_each_kernel_in_enqueue_order() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let mut step = Step::new(&gpu);
        let graph = gpu.capture(|_| step.enqueue(&gpu)).expect("the capture");
        let got: Vec<(String, [u32; 3], [u32; 3])> = graph
            .nodes()
            .expect("the node list")
            .into_iter()
            .map(|n| {
                let k = n.kernel.expect("every node of the sequence is a kernel");
                (k.name, k.grid, k.block)
            })
            .collect();
        let quant_grid = u32::try_from(2 * (K / 256)).expect("fits");
        let gemv_grid = u32::try_from(ROWS / 8).expect("fits");
        let want = [
            (
                "q3k_quantize_q8_1".to_owned(),
                [quant_grid, 1, 1],
                [32, 1, 1],
            ),
            (
                if gpu.row_prefetch() {
                    "q4k_gemv_prefetch"
                } else {
                    "q4k_gemv"
                }
                .to_owned(),
                [gemv_grid, 1, 1],
                [256, 1, 1],
            ),
        ];
        assert_eq!(got.len(), want.len(), "node list {got:?}");
        if let Some(i) = (0..want.len()).find(|&i| got[i] != want[i]) {
            panic!(
                "node {i}: expected {:?}, got {:?} (whole list {got:?})",
                want[i], got[i]
            );
        }
    }

    /// A capture body that returns `Err` or panics ends the capture it ran
    /// in: the error or the panic comes back to the caller, the stream is not
    /// left capturing, and the next capture on it records the two nodes and
    /// replays the eager bytes. The test owns its `Gpu` and so its stream;
    /// the context is the device's primary one, shared with every test of
    /// this binary, and a hw test that runs after this one inherits it.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_capture_body_that_fails_or_panics_leaves_the_stream_capturable() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let mut step = Step::new(&gpu);
        let eager = step.eager(&gpu);

        let failed = gpu.capture(|_| {
            step.enqueue(&gpu)?;
            Err(GpuError::state("graph test", "injected body failure"))
        });
        assert!(
            matches!(
                failed,
                Err(GpuError::State {
                    missing: "injected body failure",
                    ..
                })
            ),
            "the body's own error does not come back from the capture"
        );
        assert!(
            !capturing(gpu.stream()).expect("the capture status"),
            "the stream is still capturing after a body that returned Err"
        );
        step.assert_replays(&gpu, &eager, "a body that returned Err");

        let panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
            gpu.capture(|_| -> Result<(), GpuError> {
                step.enqueue(&gpu)?;
                panic!("injected body panic")
            })
        }));
        let payload = panicked
            .err()
            .expect("the body's panic does not reach the caller");
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"injected body panic"),
            "the caller's panic is not the body's"
        );
        assert!(
            !capturing(gpu.stream()).expect("the capture status"),
            "the stream is still capturing after a body that panicked"
        );
        step.assert_replays(&gpu, &eager, "a body that panicked");
    }

    /// How long a race test's capture body holds the window open for thread
    /// B: B's path from the signal to the context-wide synchronize of a
    /// fresh handle's first stream is two handle creations and a priority
    /// query — the synchronize is the first thing a fresh handle does after
    /// its constructor — so this bound covers it many times over, and B
    /// reaches its synchronize while the capture is still open on code that
    /// does not hold it out. It cannot pass red-side on the wait alone: the
    /// capture closes before B's answer only on this timeout, and B always
    /// gets here first. On code that does hold B out (the owner's exclusive
    /// side), the timeout is what the green path pays.
    const RACE_WAIT: Duration = Duration::from_secs(5);

    /// A one-entry PTX module the naming test JIT-loads inside another
    /// thread's capture: the smallest load that reaches the driver's module
    /// path. Its target is at or below every card the gates run on, so the
    /// JIT accepts it everywhere.
    const NOP_PTX: &[u8] = b"\
        .version 7.0\n\
        .target sm_80\n\
        .address_size 64\n\
        .visible .entry capsync_probe_nop()\n\
        {\n\
        \tret;\n\
        }\n";

    /// A fresh handle's first stream waits for a capture on its device:
    /// while thread A captures, thread B opens a second `Gpu`, and B's
    /// open, A's capture and the capture's replay all succeed — the
    /// context-wide synchronize a fresh handle's first stream runs cannot
    /// land inside the capture. The outcome is pinned whole (both sides
    /// succeed), not the mechanism.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_capsync_race_fresh_first_stream_waits_for_a_capture() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let mut step = Step::new(&gpu);
        let eager = step.eager(&gpu);
        let (opened_tx, opened_rx) = mpsc::channel::<Result<(), String>>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        // A `Gpu` never crosses threads: B opens it, drops it and sends only
        // the answer.
        let b = std::thread::spawn(move || {
            go_rx
                .recv_timeout(RACE_WAIT * 2)
                .expect("A signals inside its capture");
            let opened = match Gpu::with_device(0) {
                Ok(_gpu) => Ok(()),
                Err(e) => Err(e.to_string()),
            };
            opened_tx.send(opened).expect("A still collects the open");
        });
        let graph = gpu
            .capture(|_| {
                step.enqueue(&gpu)?;
                go_tx.send(()).expect("B waits for the signal");
                // B answers inside the capture only when nothing held it
                // back; held out, it answers once this capture has ended,
                // and the wait closes on its timeout either way.
                let _ = opened_rx.recv_timeout(RACE_WAIT);
                Ok(())
            })
            .expect("the capture with a second Gpu opening under it");
        assert_eq!(
            graph.node_count(),
            2,
            "the capture recorded what the body enqueued"
        );
        let opened = opened_rx
            .recv_timeout(RACE_WAIT)
            .expect("B finishes once the capture has ended");
        opened.expect("the second Gpu's open under the capture");
        b.join().expect("B does not panic");
        step.assert_replays(
            &gpu,
            &eager,
            "a capture raced by a fresh handle's first stream",
        );
    }

    /// The call that breaks a capture is the context-wide synchronize, not
    /// the work around it: a handle that already has its stream — made
    /// before the capture began — allocates a device buffer and JIT-loads
    /// a module inside the capture, and B's work, the capture and the
    /// replay all succeed, because no path either call takes synchronizes
    /// the whole context. The buffer and the module are dropped after the
    /// capture has ended, so their drops stay out of the claim.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_capsync_alloc_and_module_load_leave_a_capture_whole() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let mut step = Step::new(&gpu);
        let eager = step.eager(&gpu);
        let (ctx, stream) = capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (end_tx, end_rx) = mpsc::channel::<()>();
        let b = std::thread::spawn(move || {
            go_rx
                .recv_timeout(RACE_WAIT * 2)
                .expect("A signals inside its capture");
            let work: Result<_, GpuError> = (|| {
                // cuda-core's `DeviceBuffer::zeroed` allocates without binding
                // the stream's context, so this fresh thread binds it first
                // (docs/upstream/nvlabs-ledger.md); the bind is no
                // context-wide call.
                ctx.bind_to_thread()?;
                let bytes = DeviceBuffer::<u32>::zeroed(&stream, 4096)?;
                let module = ctx.load_module_from_image(NOP_PTX)?;
                Ok((bytes, module))
            })();
            let answer = match &work {
                Ok(_) => Ok(()),
                Err(e) => Err(e.to_string()),
            };
            done_tx.send(answer).expect("A still collects the work");
            // The buffer and module drop when this closure returns, which
            // the wait below holds past the capture's end: a drop is not
            // part of what this test claims is safe inside a capture.
            end_rx
                .recv_timeout(RACE_WAIT * 2)
                .expect("A ends its capture");
        });
        let graph = gpu
            .capture(|_| {
                step.enqueue(&gpu)?;
                go_tx.send(()).expect("B waits for the signal");
                // B holds no lock, so its answer lands inside the capture.
                done_rx
                    .recv_timeout(RACE_WAIT)
                    .expect("B's work needs no lock, so it answers inside")
                    .expect("B's allocation and module load inside the capture");
                Ok(())
            })
            .expect("the capture with an allocation and a module load under it");
        assert_eq!(
            graph.node_count(),
            2,
            "the capture recorded what the body enqueued"
        );
        end_tx.send(()).expect("B drops after the capture");
        b.join().expect("B does not panic");
        step.assert_replays(
            &gpu,
            &eager,
            "a capture raced by an allocation and a module load",
        );
    }
}
