//! The tensor-op GEMMs (`kernels/gemm_tensor.metal`), forced: every clause
//! here runs them on the device that runs the gate, whatever its matrix-unit
//! rule picks for the engine, through a library whose GEMMs come from the
//! tensor-op program (`Library::with_tensor_gemms`). A device whose runtime
//! does not build that program fails each clause by name; none is skipped.
//! Each clause starts from a clear fault word, whatever the clause before
//! it left raised.
//!
//! - `tensor_gemm_q4k`, `tensor_gemm_q6k`, `tensor_gemm_a4`: the dense GEMM's
//!   cases of `gemm_*_store` and `gemm_*_resid`, both epilogues in both
//!   operands, within the tensor path's band (`clauses_gemm`'s doc).
//! - `tensor_gemm_seg`: `gemm_seg` on these kernels — a segment is the same
//!   bits as the same tensor run alone.
//! - `tensor_gemm_range`: `gemm_range` on these kernels — the f16 operand's
//!   range check raises the quantizer's site and makes the row NaN.
//! - `tensor_grouped_q4k`, `tensor_grouped_q6k`, `tensor_grouped_a4`: the
//!   routed pair and down over the route table's tiles, both operands,
//!   within the tensor path's band; an activation past the half range the
//!   fault and NaN; a tile past the tile's slots the expert-id site with
//!   nothing stored.
//! - `tensor_q8_0_*`, `tensor_q5k_*`: the dense, segmented and grouped GEMMs
//!   of the owners no fixture holds yet, on synthetic rows
//!   (`clauses_synth`).
//!
//! A row that runs its A4 sites only has `tensor_gemm_a4`, `tensor_gemm_seg`
//! (its A4 group) and `tensor_grouped_a4`, at its own sites.

use bloomery_device::conformance::Clause;

use super::clauses_a4;
use super::fixture::Fixture;
use super::{ClauseRun, Harness, clauses_gemm, clauses_q5k, clauses_q8_0, clauses_route};

/// `clause` on the tensor-op harness over `fix`: the harness's fault word
/// cleared first, or the reason the tensor-op GEMMs or the fixture are not
/// there.
fn on<'a>(
    tensor: &'a Result<Harness<'a>, String>,
    fix: &'a Result<Fixture, String>,
    clause: impl Fn(&Harness<'_>, &Fixture) -> Clause + 'a,
) -> ClauseRun<'a> {
    Box::new(move || match (tensor, fix) {
        (Err(e), _) => Err(format!("the tensor-op GEMMs do not build here: {e}")),
        (_, Err(e)) => Err(e.clone()),
        (Ok(h), Ok(fix)) => {
            h.clear_fault()?;
            clause(h, fix)
        }
    })
}

/// The tensor-op clauses over `tensor`, the harness of the tensor-op GEMMs
/// (or why this device has none).
pub fn table<'a>(
    tensor: &'a Result<Harness<'a>, String>,
    fixture: &'a Result<Fixture, String>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    let sites = &clauses_a4::QWEN35_SITES;
    let mut clauses = vec![
        (
            "tensor_gemm_q4k",
            on(tensor, fixture, clauses_gemm::tensor_gemm_q4k),
        ),
        (
            "tensor_gemm_q6k",
            on(tensor, fixture, clauses_gemm::tensor_gemm_q6k),
        ),
        (
            "tensor_gemm_a4",
            on(tensor, mlx, move |h, f| {
                clauses_gemm::tensor_gemm_a4(h, f, sites)
            }),
        ),
        (
            "tensor_gemm_seg",
            Box::new(move || match (tensor, fixture, mlx) {
                (Err(e), _, _) => Err(format!("the tensor-op GEMMs do not build here: {e}")),
                (_, Err(e), _) | (_, _, Err(e)) => Err(e.clone()),
                (Ok(h), Ok(fix), Ok(split)) => {
                    h.clear_fault()?;
                    clauses_gemm::gemm_seg(h, fix, split, sites)
                }
            }),
        ),
        (
            "tensor_gemm_range",
            on(tensor, fixture, clauses_gemm::gemm_range),
        ),
        (
            "tensor_grouped_q4k",
            on(tensor, fixture, clauses_route::tensor_grouped_q4k),
        ),
        (
            "tensor_grouped_q6k",
            on(tensor, fixture, clauses_route::tensor_grouped_q6k),
        ),
        (
            "tensor_grouped_a4",
            on(tensor, mlx, clauses_route::tensor_grouped_a4),
        ),
    ];
    clauses.extend(clauses_q8_0::tensor_table(tensor));
    clauses.extend(clauses_q5k::tensor_table(tensor));
    clauses
}

/// The tensor-op clauses of a row that runs its A4 sites only: [`table`]'s A4
/// clauses at that row's sites.
pub fn shaped<'a>(
    tensor: &'a Result<Harness<'a>, String>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    let sites = &clauses_a4::QWEN3_SITES;
    vec![
        (
            "tensor_gemm_a4",
            on(tensor, mlx, move |h, f| {
                clauses_gemm::tensor_gemm_a4(h, f, sites)
            }),
        ),
        (
            "tensor_gemm_seg",
            on(tensor, mlx, move |h, f| {
                clauses_gemm::gemm_seg_a4(h, f, sites)
            }),
        ),
        (
            "tensor_grouped_a4",
            on(tensor, mlx, clauses_route::tensor_grouped_a4),
        ),
    ]
}
