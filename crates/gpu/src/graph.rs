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
use cuda_core::{CudaContext, CudaEvent, CudaStream, DriverError, sys};
use std::ffi::c_void;
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
        let mut handles: Vec<sys::CUgraphNode> = vec![ptr::null_mut(); self.nodes];
        let mut n = self.nodes;
        // SAFETY: the array holds `n` slots, the count the driver reported
        // for this template at capture; it writes at most `n` handles.
        let rc = unsafe { sys::cuGraphGetNodes(self.graph, handles.as_mut_ptr(), &mut n) };
        cu(rc, "cuGraphGetNodes")?;
        handles.truncate(n);
        handles
            .into_iter()
            // SAFETY: every handle is a node of this template, which lives
            // as long as `self`.
            .map(|node| unsafe { node_info(node) })
            .collect()
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

/// A second stream that work can run on beside the stream it forks from,
/// captured or eager alike: [`Branch::fork`] makes the branch wait for
/// everything enqueued on the main stream so far, and [`Forked::join`] makes
/// the main stream wait for everything enqueued on the branch since.
///
/// Inside a capture on the main stream the fork's wait pulls the branch into
/// the same capture, so the branch's work becomes a parallel path of the
/// graph — an edge from the fork point and one into the join point, no node
/// of its own. Outside a capture the same two event waits order the eager
/// run. The stream and both events are made once, at load: `fork` and `join`
/// only record and wait, which are capturable and allocate nothing.
pub struct Branch {
    stream: Arc<CudaStream>,
    forked: CudaEvent,
    joined: CudaEvent,
}

impl Branch {
    /// A branch of streams of `ctx`. Load-time only.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Branch, GpuError> {
        Ok(Branch {
            stream: crate::role_stream(ctx, crate::StreamRole::Engine)?,
            forked: ctx.new_event(None)?,
            joined: ctx.new_event(None)?,
        })
    }

    /// Start the branch after everything enqueued on `main` so far. The
    /// branch's work goes on [`Forked::stream`]; the fork must be joined
    /// before `main`'s next reader of that work is enqueued.
    pub fn fork<'a>(&'a self, main: &'a CudaStream) -> Result<Forked<'a>, GpuError> {
        self.forked.record(main)?;
        self.stream.wait(&self.forked)?;
        Ok(Forked {
            branch: self,
            main,
            open: true,
        })
    }
}

/// A [`Branch`] forked from a main stream and not joined yet.
///
/// Dropped without [`Forked::join`] (an early error return between the two),
/// it joins anyway and ignores the driver's answer: the error already on its
/// way out is the one reported, and a capture is never left with a branch
/// its end would refuse as unjoined.
#[must_use = "a fork is joined before its work is read"]
pub struct Forked<'a> {
    branch: &'a Branch,
    main: &'a CudaStream,
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
        self.main.wait(&self.branch.joined)?;
        Ok(())
    }
}

impl Drop for Forked<'_> {
    fn drop(&mut self) {
        if self.open {
            let _ = self.branch.joined.record(&self.branch.stream);
            let _ = self.main.wait(&self.branch.joined);
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

/// Pinned host memory mapped into the device's address space
/// (`cuMemHostAlloc` with `PORTABLE | DEVICEMAP`): the host reaches it at
/// `host`, kernels, copies and stream memory operations at `dev`, from every
/// context of the process. Zeroed at allocation, freed on drop. The one owner
/// of mapped host memory: the host flags ([`HostFlags`]) and the host tier's
/// page are views over it.
pub(crate) struct MappedHost {
    host: *mut u8,
    dev: sys::CUdeviceptr,
    bytes: usize,
}

// SAFETY: the allocation is owned by this value alone and freed once, in its
// drop; the host pointer is plain process memory any thread may reach, and
// every access to it goes through the methods below.
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
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        bytes: usize,
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
                sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
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
        // SAFETY: the allocation holds `bytes` writable bytes and nothing else
        // references it yet.
        unsafe { ptr::write_bytes(host.cast::<u8>(), 0, bytes) };
        Ok(MappedHost {
            host: host.cast(),
            dev,
            bytes,
        })
    }

    /// The device address of byte `off`, which the caller's layout keeps
    /// inside the allocation.
    pub(crate) fn dev_at(&self, off: usize) -> sys::CUdeviceptr {
        self.dev + off as u64
    }

    /// The host address of byte `off`. Computing it touches nothing; each
    /// access proves its own span.
    pub(crate) fn host_at(&self, off: usize) -> *mut u8 {
        self.host.wrapping_add(off)
    }

    /// The u32 at byte `off` as an atomic; `None` when it is not 4-aligned
    /// or passes the allocation.
    pub(crate) fn atomic_u32(&self, off: usize) -> Option<&AtomicU32> {
        if !off.is_multiple_of(4) || off + 4 > self.bytes {
            return None;
        }
        // SAFETY: the word is 4-aligned and inside the allocation (checked
        // above; the allocation is page-aligned); `AtomicU32` has `u32`'s
        // layout, and every host access to a flag word goes through this
        // view.
        Some(unsafe { &*self.host_at(off).cast::<AtomicU32>() })
    }

    /// The `len` f32 at byte `off`, for the host to write; `None` when not
    /// 4-aligned or past the allocation. `&mut self` makes it the only host
    /// reference; the owner's protocol keeps the card off the span meanwhile.
    pub(crate) fn f32_mut(&mut self, off: usize, len: usize) -> Option<&mut [f32]> {
        if !off.is_multiple_of(4) || off + 4 * len > self.bytes {
            return None;
        }
        // SAFETY: the span is inside the allocation and 4-aligned (checked
        // above); `&mut self` makes it the only host reference.
        Some(unsafe { std::slice::from_raw_parts_mut(self.host_at(off).cast::<f32>(), len) })
    }

    /// A copy of the `len` f32 at byte `off`, when inside the allocation.
    pub(crate) fn f32_copy(&self, off: usize, len: usize) -> Option<Vec<f32>> {
        if !off.is_multiple_of(4) || off + 4 * len > self.bytes {
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
        // SAFETY: `host` came from cuMemHostAlloc and is freed once, here —
        // after every graph that names it, since the graphs are declared
        // before the bodies that own the views over it. A failure on the drop
        // path is unreportable and ignored.
        unsafe { sys::cuMemFreeHost(self.host.cast()) };
    }
}

// ------------------------------------------------------ stream memory ops

/// An all-zero batch operation.
fn op_zero() -> sys::CUstreamBatchMemOpParams {
    // SAFETY: every member of the union is a plain C struct of integers, so
    // all-zero bytes are a valid value.
    unsafe { std::mem::zeroed() }
}

/// Wait until the u32 at `addr` is at least `value`.
pub(crate) fn op_wait_geq(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
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
pub(crate) fn op_write(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
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
pub(crate) fn op_barrier_sys() -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.memoryBarrier = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_BARRIER,
        flags: sys::CUstreamMemoryBarrier_flags_enum_CU_STREAM_MEMORY_BARRIER_TYPE_SYS,
    };
    p
}

/// Atomic reduction `*addr += value` on a u32 (wrapping).
pub(crate) fn op_add(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
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

/// Enqueue `ops` on `stream` as one batch of stream memory operations — one
/// graph node when captured.
pub(crate) fn mem_batch(
    stream: &CudaStream,
    ops: &mut [sys::CUstreamBatchMemOpParams],
    what: &'static str,
) -> Result<(), GpuError> {
    let n = u32::try_from(ops.len()).map_err(|_| GpuError::shape(what, "batch too long"))?;
    // SAFETY: `ops` is a live array of `n` initialized operations the driver
    // reads during the call (a capture copies them into the node); the flags
    // must be 0.
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
        mem_batch(stream, &mut ops, "cuStreamBatchMemOp_v2 (host flag wait)")
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
    use super::{FLAG_WAIT_OPS, Graph, HostFlags, capturing, cu};
    use crate::{DeviceTensor, Gpu, GpuError, Q8Act};
    use cuda_core::{CudaContext, DeviceBuffer, PinnedHostBuffer, sys};
    use std::ffi::c_void;
    use std::panic::AssertUnwindSafe;
    use std::time::Duration;

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
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
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
        assert!(nodes.iter().all(|n| n.kernel.is_none()), "{nodes:?}");
        assert!(
            nodes
                .iter()
                .any(|n| n.kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET),
            "{nodes:?}"
        );
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
            ("q4k_gemv".to_owned(), [gemv_grid, 1, 1], [256, 1, 1]),
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
}
