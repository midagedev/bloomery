//! The Apart clauses: a shared expert read in a type of its own beside its
//! stacks' (the Q8_0 shared expert of a file whose stacks are Q4_K, Q5_K and
//! Q6_K). The pair, the fold and the grouped GEMM each run every pair of
//! their table on synthetic rows at the experts' real shapes against the f64
//! oracle, with the shared slot held to the shared type's band; the
//! families' own clauses hold the stacks' types. The bodies are the pair's,
//! the fold's and the grouped GEMM's own (`clauses_experts`, `clauses_route`),
//! taken with a second type.

use super::{ClauseCtx, ClauseRun, clauses_experts, clauses_route};

/// The clauses of the whole table.
pub fn table<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    vec![
        (
            "apart_pair",
            Box::new(|| clauses_experts::apart_pair(ctx.harness)) as ClauseRun<'a>,
        ),
        (
            "apart_fold",
            Box::new(|| clauses_experts::apart_fold(ctx.harness)),
        ),
        (
            "apart_grouped",
            Box::new(|| clauses_route::grouped_apart(ctx.harness)),
        ),
    ]
}
