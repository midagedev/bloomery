//! The route trace of a qwen4exp run (`BLOOMERY_ROUTE_TRACE`), for
//! `generate_qwen3moe` and the Qwen3.8 serve seat, each including this file:
//! the header a trace carries and its directory made ([`create`]), and the
//! words of the two refusals both binaries make, so the trace's contract has
//! one owner. It lives beside the bins, not in the lib, because it names the
//! qwen3moe body's plan inputs.

use std::path::Path;

use bloomery_gpu::arch::qwen3moe::Prompt38;
use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
use bloomery_gpu_gates::GateError;
use model::arch::qwen35moe::place::{Experts, PlanInputs};

/// The name the `plan` line and the trace's header print for `experts`.
pub fn experts_name(experts: Experts) -> &'static str {
    match experts {
        Experts::Host => "host",
        Experts::Card => "card",
    }
}

/// What a trace's header says of the run that writes it, besides the file's
/// own counts.
pub struct TraceRun<'a> {
    /// The model's first shard, its full path.
    pub model: &'a Path,
    /// The binary, as the header's `build` line names it.
    pub build: &'a str,
    /// The placement's name.
    pub place: &'a str,
    pub experts: Experts,
    /// The prompt path the run feeds by.
    pub prefill: Prompt38,
    /// The positions every context of the run holds, for the header's `chunk`
    /// line, when they all hold the same count; `None` writes no line (the
    /// replay tool then refuses the set's unknown contexts by name).
    pub chunk: Option<usize>,
}

/// The trace into `dir`, which is created here as a new directory
/// ([`RouteTrace::create`]) before the load, with the header of `run` over
/// the counts `inputs` describes.
pub fn create(
    dir: &Path,
    inputs: &PlanInputs,
    run: &TraceRun<'_>,
) -> Result<RouteTrace, GateError> {
    let hp = &inputs.hp;
    let mut extra = vec![
        ("place".to_owned(), run.place.to_owned()),
        ("experts".to_owned(), experts_name(run.experts).to_owned()),
        ("prefill".to_owned(), run.prefill.name().to_owned()),
    ];
    if let Some(n) = run.chunk {
        extra.push(("chunk".to_owned(), n.to_string()));
    }
    let header = TraceHeader {
        model: run.model.to_path_buf(),
        arch: "qwen4exp".to_owned(),
        build: run.build.to_owned(),
        n_expert: hp.n_expert,
        n_used: hp.n_used,
        first_layer: 0,
        n_layer: inputs.spec.layers.len(),
        extra,
    };
    Ok(RouteTrace::create(dir, header)?)
}

/// The refusal of the MTP draft beside a trace: a drafted run verifies
/// several rows in one pass, and the trace records one-row steps.
pub fn draft_refusal() -> GateError {
    "BLOOMERY_ROUTE_TRACE records the plain run's routing, one step a position; it is refused \
     beside BLOOMERY_DRAFT=mtp"
        .into()
}

/// The refusal of a residency word other than `off` (`word`) beside a trace:
/// the trace is the input the residency model replays under a fixed seed, and
/// under the machine its slot files would record the machine's own moves.
pub fn residency_refusal(word: &str) -> GateError {
    format!(
        "BLOOMERY_ROUTE_TRACE records a fixed placement's routing; BLOOMERY_RESIDENCY={word} \
         moves the slot map under it"
    )
    .into()
}
