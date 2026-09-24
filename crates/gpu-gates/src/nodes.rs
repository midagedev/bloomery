//! The nodes of a captured graph by type, as the gates count and print them,
//! and a capture's nodes in stream order with the edges between them.

use crate::GateError;
use bloomery_gpu::{GpuError, NodeInfo};
use cuda_core::CudaStream;
use cuda_core::sys::{self, CUgraphNodeType};
use std::collections::HashMap;

/// The node types a gate prints by name, with the name: the driver's
/// `CU_GRAPH_NODE_TYPE_*`, lower case.
const NAMES: [(CUgraphNodeType, &str); 7] = [
    (
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
        "kernel",
    ),
    (
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
        "memcpy",
    ),
    (
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
        "memset",
    ),
    (sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST, "host"),
    (sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EMPTY, "empty"),
    (
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_WAIT_EVENT,
        "wait_event",
    ),
    (
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
        "batch_mem_op",
    ),
];

/// The name a gate prints for node type `kind`; `type<n>` for a type with
/// no name here.
pub fn kind_name(kind: CUgraphNodeType) -> String {
    NAMES
        .iter()
        .find(|k| k.0 == kind)
        .map_or_else(|| format!("type{kind}"), |k| k.1.to_string())
}

/// `nodes` counted by type: one count per entry of `kinds`, in its order,
/// and the nodes of every other type together.
pub fn count_kinds<const N: usize>(
    nodes: &[NodeInfo],
    kinds: [CUgraphNodeType; N],
) -> ([usize; N], usize) {
    let mut counts = [0usize; N];
    let mut other = 0;
    for n in nodes {
        match kinds.iter().position(|&k| k == n.kind) {
            Some(i) => counts[i] += 1,
            None => other += 1,
        }
    }
    (counts, other)
}

/// One node of a captured template, as a gate reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepNode {
    /// A kernel launch and its entry name.
    Kernel(String),
    /// A stream memory-operation batch and its operation count.
    Memop(u32),
    /// Any other node, by [`kind_name`].
    Other(String),
}

/// A capture's nodes in the order they were enqueued, and its edges as
/// indices into that order.
#[derive(Debug)]
pub struct Captured {
    /// The nodes, in enqueue order.
    pub nodes: Vec<StepNode>,
    /// Per node, the nodes it depends on, ascending.
    pub deps: Vec<Vec<usize>>,
    /// Per node, the nodes that depend on it, ascending.
    pub dependents: Vec<Vec<usize>>,
}

impl Captured {
    /// The kernel name of node `i`, or `None` for another kind.
    #[must_use]
    pub fn kernel(&self, i: usize) -> Option<&str> {
        match self.nodes.get(i) {
            Some(StepNode::Kernel(k)) => Some(k.as_str()),
            _ => None,
        }
    }
}

fn drv(rc: sys::CUresult, what: &str) -> Result<(), GateError> {
    if rc == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: CUresult {rc}").into())
    }
}

/// Capture what `enqueue` records on `stream` (and on any stream it forks
/// into the capture) and read the template back, then destroy it without
/// instantiating it.
///
/// The order is `cuGraphGetNodes`', which lists a captured template's nodes
/// in the order they were created, that is the host's enqueue order across
/// every stream of the capture. That is taken only when it is also a
/// topological order: an edge from a later node to an earlier one is refused
/// by name rather than read as stream order. When it holds, it is the order
/// Kahn's walk takes with ties broken by that index, and on a one-stream
/// capture (a chain) it is the chain's one order — the order a walk from the
/// root along each node's single successor reads.
pub fn capture_order(
    stream: &CudaStream,
    enqueue: impl FnOnce() -> Result<(), GpuError>,
) -> Result<Captured, GateError> {
    let hs = stream.cu_stream();
    // SAFETY: `hs` is a live stream that is not capturing (every capture
    // before this one ended).
    let rc = unsafe {
        sys::cuStreamBeginCapture_v2(
            hs,
            sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    };
    drv(rc, "cuStreamBeginCapture_v2")?;
    let enqueued = enqueue();
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    // SAFETY: the stream is capturing (begun above); this ends it on every
    // path and writes the template, or null, into `graph`.
    let ended = unsafe { sys::cuStreamEndCapture(hs, &mut graph) };
    let order = match enqueued {
        Err(e) => Err(e.into()),
        Ok(()) => drv(ended, "cuStreamEndCapture").and_then(|()| read_template(graph)),
    };
    if !graph.is_null() {
        // SAFETY: a non-null handle from the end of the capture is a
        // template, destroyed exactly once here.
        unsafe { sys::cuGraphDestroy(graph) };
    }
    order
}

/// `graph`'s nodes in `cuGraphGetNodes` order and its edges; see
/// [`capture_order`].
fn read_template(graph: sys::CUgraph) -> Result<Captured, GateError> {
    let mut total = 0usize;
    // SAFETY: a null array asks only for the count.
    let rc = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut total) };
    drv(rc, "cuGraphGetNodes")?;
    let mut handles: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); total];
    let mut n = total;
    // SAFETY: the array holds `n` slots, the count just reported.
    let rc = unsafe { sys::cuGraphGetNodes(graph, handles.as_mut_ptr(), &mut n) };
    drv(rc, "cuGraphGetNodes")?;
    if n != total {
        return Err(format!("cuGraphGetNodes listed {n} of the {total} nodes it counted").into());
    }
    let index: HashMap<usize, usize> = handles
        .iter()
        .enumerate()
        .map(|(i, h)| (*h as usize, i))
        .collect();
    let mut nodes = Vec::with_capacity(total);
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); total];
    for (i, &node) in handles.iter().enumerate() {
        nodes.push(describe(node)?);
        let mut k = 0usize;
        // SAFETY: `node` is a node of the live template; null arrays ask
        // only for the count.
        let rc = unsafe {
            sys::cuGraphNodeGetDependentNodes_v2(
                node,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut k,
            )
        };
        drv(rc, "cuGraphNodeGetDependentNodes_v2")?;
        let mut next: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); k];
        if k > 0 {
            // SAFETY: `k` slots, the count just reported; the edge data is
            // not asked for.
            let rc = unsafe {
                sys::cuGraphNodeGetDependentNodes_v2(
                    node,
                    next.as_mut_ptr(),
                    std::ptr::null_mut(),
                    &mut k,
                )
            };
            drv(rc, "cuGraphNodeGetDependentNodes_v2")?;
            next.truncate(k);
        }
        for h in next {
            let j = *index
                .get(&(h as usize))
                .ok_or_else(|| format!("node {i} has a dependent the template does not list"))?;
            if j <= i {
                return Err(format!(
                    "the edge from node {i} to node {j} runs backward in cuGraphGetNodes order: \
                     that order is not the enqueue order and the walk cannot name one"
                )
                .into());
            }
            dependents[i].push(j);
            deps[j].push(i);
        }
    }
    for v in dependents.iter_mut().chain(deps.iter_mut()) {
        v.sort_unstable();
    }
    Ok(Captured {
        nodes,
        deps,
        dependents,
    })
}

/// A node's kind, with its kernel's entry name or its batch's operation
/// count.
fn describe(node: sys::CUgraphNode) -> Result<StepNode, GateError> {
    let mut kind: sys::CUgraphNodeType = 0;
    // SAFETY: `node` is a node of a live template.
    let rc = unsafe { sys::cuGraphNodeGetType(node, &mut kind) };
    drv(rc, "cuGraphNodeGetType")?;
    if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL {
        // SAFETY: all-zero is a valid value of this plain C struct (integers
        // and nullable pointers), which the call then fills.
        let mut p: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        // SAFETY: `node` is a kernel node and `p` the struct the call fills.
        let rc = unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut p) };
        drv(rc, "cuGraphKernelNodeGetParams_v2")?;
        let mut name: *const std::ffi::c_char = std::ptr::null();
        if p.func.is_null() {
            // SAFETY: the node launches the library kernel `p.kern`, a live
            // handle; `name` is a local the call writes.
            let rc = unsafe { sys::cuKernelGetName(&mut name, p.kern) };
            drv(rc, "cuKernelGetName")?;
        } else {
            // SAFETY: the node launches the module function `p.func`, a live
            // handle; `name` is a local the call writes.
            let rc = unsafe { sys::cuFuncGetName(&mut name, p.func) };
            drv(rc, "cuFuncGetName")?;
        }
        if name.is_null() {
            return Err("a kernel node whose function has no name".into());
        }
        // SAFETY: the driver hands back a NUL-terminated name it owns for the
        // function's lifetime; it is copied out at once.
        let name = unsafe { std::ffi::CStr::from_ptr(name) };
        Ok(StepNode::Kernel(name.to_string_lossy().into_owned()))
    } else if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP {
        // SAFETY: all-zero is a valid value of this plain C struct (a
        // context, a count, a nullable array pointer, flags), which the call
        // then fills.
        let mut p: sys::CUDA_BATCH_MEM_OP_NODE_PARAMS = unsafe { std::mem::zeroed() };
        // SAFETY: `node` is a batch memory-operation node and `p` the struct
        // the call fills.
        let rc = unsafe { sys::cuGraphBatchMemOpNodeGetParams(node, &mut p) };
        drv(rc, "cuGraphBatchMemOpNodeGetParams")?;
        Ok(StepNode::Memop(p.count))
    } else {
        Ok(StepNode::Other(kind_name(kind)))
    }
}
