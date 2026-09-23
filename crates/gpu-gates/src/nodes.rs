//! The nodes of a captured graph by type, as the gates count and print them.

use bloomery_gpu::NodeInfo;
use cuda_core::sys::{self, CUgraphNodeType};

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
