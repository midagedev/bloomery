//! The K-EXAONE-2.0 end-to-end gate: the whole decode chain of the composed head128 program —
//! the dense lead's two blocks (layer 0 a full-attention NoPE one, layer 1 the 4096-window dense
//! one), four routed blocks of 256 experts top-8 with one shared expert each (layers 2 and 4
//! window-128 roped, 3 and 5 full NoPE, 4 and 5 gate-first clamped at 7), the head and the argmax —
//! on the gate card, against the patched llama.cpp tree's CPU node dumps of the same file
//! (`refset::arch::exaone_moe`, `just dump-ref-exaone-moe`: mainline 7bcbc04fb plus the exaone-moe
//! oracle patch, the only faithful implementation; ik has no exaone-moe).
//!
//! The prompts are the dump's four (`tools/ref/data/exaone-moe`): a an English paragraph (24 ids),
//! b a Korean one (18), c a Rust function head (30), d a five-id question; each is fed one id a
//! step from a reset, and the oracle's reads are the set's own: the last position's whole logits
//! row (`result_output`), each layer's output at every position (`l_out-L`), the 40 largest
//! logprobs (`topk.tsv`) and the 32 greedy tokens with their margins (`greedy.tsv`).
//!
//! The clauses are `shared/lcpp_e2e.rs`'s (`finite_and_fault_clean`, `reset_rewinds`,
//! `layer_outputs_near_lcpp`, `top_logprobs_in_band`, `forced_greedy_count_pin`,
//! `prefill_near_lcpp`, `head_near_lcpp`); this file holds the model's part: the family and the
//! seat (`shared/exaone_seat.rs`: the load, the tap ceilings). The family has no fixture tier —
//! its file is the fixture GGUF and the sets are its own, so the fixture twin clause is not
//! declared and the fixture tier is refused by name. The teacher-forced layer arm is not declared
//! either: the body runs no layer alone on caller's rows.
//!
//! `--arm a|bp` picks the placement arm (default `a`: the largest card the stage; `bp`: the
//! next-largest card an expert tier under the host tier).
//!
//! FAIL-first (the mutants the design names, applied on the box):
//! - rope on a NoPE layer (`crates/gpu/src/attention_h128.rs`'s `Binding::load` reading
//!   `rotate: true` whatever the description): layer 0's output leaves its ceiling — the first
//!   layer past it is named — and `top_logprobs_in_band` goes red with it.
//! - the clamp order swapped (the gate-first act `crates/gpu/src/kquant/act.rs` run as the
//!   act-first one): the two orders differ by at most 7 − silu(7) ≈ 0.0014 on a gated factor, two
//!   orders under every layer ceiling [derived], so this gate holds the act by the
//!   `ToolFormat`-independent owner instead: `gate-iq-sel`/`gate-gemm`'s host comparison
//!   (`gpu-gates/src/swiglu_host.rs`) pins the gate-first rule on values past the limit; a
//!   swap is red there, and this gate's clauses name the layers whose outputs move.
//! - window 4096→128 at depth: the sets hold at most 62 positions, all under the 128-keys bound
//!   both windows share, so no clause of this gate separates the two at depth; what holds the
//!   window here is the load's own store count (the plan counts layer 1's 4608-row ring, and
//!   `Binding::capacity` refuses a mismatch) and the meta gate's `ROWS_AT` pin. A clause past
//!   4096 tokens needs either an oracle set dumped at that depth (the patch refuses it) or the
//!   engine's own f64 simulation of the rule, neither of which exists yet; the gap is the lead's.

#[cfg(feature = "flint")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with(gate::NAME, gate::run())
}

#[cfg(feature = "flint")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/exaone_seat.rs"]
mod seat;

#[cfg(feature = "flint")]
mod gate {
    use crate::lcpp_e2e::{self, Gate};
    use crate::seat;
    use bloomery_gpu_gates::GateError;
    use refset::arch::exaone_moe;

    pub const NAME: &str = "gate_exaone_e2e";

    /// The model's part of the gate: the patched tree's sets on the fixture GGUF (the family has
    /// no fixture tier of its own), the seat and its ceilings.
    static GATE: Gate = Gate {
        name: NAME,
        recipe: "gate-gpu-exaone-e2e",
        arch: exaone_moe::ARCH,
        slug: "lcpp",
        who: "the patched llama.cpp",
        real: &exaone_moe::LCPP,
        fixture: None,
        tap_ceil: seat::tap_ceil,
        tap_ceils: seat::tap_ceils,
        open: seat::open,
        has_prefill: true,
        // The body runs no layer alone on caller's rows, so the teacher-forced arm is not
        // declared for it; the common clause is there for the family that has one.
        has_forced: false,
        has_head: true,
        n_expert: 256,
        free_ceil: None,
        top_deferred: None,
        greedy_deferred: None,
        sigma_logit: seat::SIGMA_LOGIT,
    };

    pub fn run() -> Result<(), GateError> {
        lcpp_e2e::run(&GATE)
    }
}
