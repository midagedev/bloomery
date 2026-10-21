//! The Solar-Open2 end-to-end gate: the whole decode chain — 36 KDA layers (low-rank forget and output gates, depthwise conv,
//! beta = 2 sigmoid, softplus decay) and 12 NoPE gated GQA layers (64 query heads over 8 key heads, the gate by its own
//! projection), every layer with a 320-expert routed block and one shared expert, the head and the argmax — on the gate card,
//! against the PAI tree's CPU dump of the same file (`refset::arch::solar2`, `tools/ref/solar2-dump.sh`: llama.cpp 6ea215d17 plus
//! prometheusAIR's solar_open2 patch, the only faithful implementation; ik has none, LRR's port omits beta = 2 sigmoid).
//!
//! The prompts are the dump's four (`tools/ref/data/solar2`): a an English paragraph (41 ids), b a Korean one (40), c a Rust
//! function (114), d the first 1,536 ids of wikitext-2; each is fed one id a step from a reset, and the oracle's reads are the
//! set's own: the last position's whole logits row (`result_output`), each layer's output (`l_out-L`, sets a, b and c), the
//! 40 largest logprobs (`topk.tsv`) and the 64 greedy tokens with their margins (`greedy.tsv`, sets a, b and c).
//!
//! The clauses are `shared/lcpp_e2e.rs`'s (`finite_and_fault_clean`, `reset_rewinds`, `layer_outputs_near_pai` and its fixture
//! twin `layer_outputs_near_pai_fx`, `top_logprobs_in_band`, `forced_greedy_count_pin`, `prefill_near_pai`); this file holds the
//! model's part: the PAI family and its fixture mirror, and the seat (`shared/solar2_seat.rs`: the load, the tap ceilings by
//! tier — tight over the first [`seat::TAP_EARLY`] layers, wide after; fixture: wide on every layer). The first layer past its
//! ceiling is named: a beta that is not doubled, or a decay of another form, moves layer 1's output (the first KDA layer) by
//! the order of its own contribution.
//!
//! `--arm a|bp` picks the placement arm (default `a`: the largest card the stage; `bp`: the next-largest card an expert tier under
//! the host tier). The clauses are the arm's own: the same bands hold on both.
//!
//! FAIL-first (box, on the first run of the body): the engine's beta scale set to 1 puts `layer_outputs_near_pai` red at layer
//! 1 and `top_logprobs_in_band` red; a decay kernel of the GLM form (sigmoid, lower bound) the same; a prompt that skips the
//! shared expert's add, the layer outputs red at layer 0.

#[cfg(feature = "solar2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with(gate::NAME, gate::run())
}

#[cfg(feature = "solar2")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/solar2_seat.rs"]
mod seat;

#[cfg(feature = "solar2")]
mod gate {
    use crate::lcpp_e2e::{self, Gate};
    use crate::seat;
    use bloomery_gpu_gates::GateError;
    use refset::arch::solar2::{self, fixture};

    pub const NAME: &str = "gate_solar2_e2e";

    /// The model's part of the gate: the PAI tree's sets on the real file and on the fixture, the seat and its ceilings.
    static GATE: Gate = Gate {
        name: NAME,
        recipe: "gate-gpu-solar2-e2e",
        arch: solar2::ARCH,
        slug: "pai",
        who: "PAI",
        real: &solar2::PAI,
        fixture: Some(&fixture::PAI),
        tap_ceil: seat::tap_ceil,
        tap_ceils: seat::tap_ceils,
        open: seat::open,
        has_prefill: true,
        // The Solar-Open2 body runs no layer alone yet, so the teacher-forced arm is not declared
        // for it; the common clause is there for the family that has one.
        has_forced: false,
        n_expert: 320,
        free_ceil: None,
        top_deferred: None,
    };

    pub fn run() -> Result<(), GateError> {
        lcpp_e2e::run(&GATE)
    }
}
