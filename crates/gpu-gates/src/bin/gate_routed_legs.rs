//! gate_routed_legs — every family's routed card layers are layers its card
//! leg runs (qwen3moe, qwen35moe, qwen4exp, glm5-next, mimo2, deepseek41),
//! planned from real users' file headers (the census stubs,
//! `refset::census`) through each family's `--place`-unset production
//! defaults at derived card rooms, and every plan the planner makes is one
//! the loader's plan-time stack table takes.
//!
//! The clause table — each clause of the gate's contract, the code that
//! asserts it, and its FAIL-first evidence (the round report holds the runs):
//!
//! | clause | asserted by | FAIL-first |
//! |---|---|---|
//! | census: a planned set's stubs check against the manifest, or the gate is red naming the set | `routed_legs::run`'s `StubSet::check` refusal | F3: a manifest copy with one `header_sha256` flipped turns the gate red naming that set |
//! | arms: the CLI's and the serve seat's `--place`-unset defaults, as each family's seat calls its planner (MiMo, which ships no CLI, has the seat at two slots and at one) | the adapters `routed_legs::{qwen3moe,qwen35moe,qwen38,glm5next,mimo2,deepseek41}` | the arms are the production call chains themselves; the checks below are what is FAIL-first |
//! | rooms: the card budget floor from a `CardBudgetFloor` refusal of the placed planner, then `floor + f·E` for f ∈ {1/4, 1/2, 3/4} | `routed_legs::rooms_of`, `E` from the unbudgeted plan | the floor probe names the card whose trunk alone overflows a zero budget |
//! | L1: every layer a plan gives card experts (`n_l`, any `tier_n_l`) is one `leg_runs(program_of(<the file's arch>), …)` runs, the types read from the header | `routed_legs::l1` | F1: `accepts::routed_rule` returning the per-type union for the three qwen bodies makes at least one L1 FAIL line per family, naming layer and types; F1b: the same for Glm5nextBody and Mimo2Body, and for Deepseek41Chain |
//! | L2: a named planner refusal is an outcome, counted by kind per family | `routed_legs::outcome_of` and the per-family counts line | any other error is red with its text |
//! | L3: each family holds at least one partial plan over its sets, arms and rooms | the per-family counts line's `partial` term | F2: with the L3 check commented out the same counts line would print `partial 0` green — the reasoning is the round report's |
//! | L4: the retype sweep — one set and one carded layer per family, the layer's stacks set to each sweep triple — holds L1 at every room | `routed_legs::sweep` | the sweep's trap triple is the same mechanism F1 mutates |
//! | L5: the sweep holds a triple a per-type union rule would card but `leg_runs` refuses | `routed_legs::trap_count`, pure from `leg_arms` | F1's mutant is that rule; the pure count proves the sweep carries the trap |
//! | L6: the family's loader-side stack table takes the header of every plan the planner makes that L1 passes (Qwen3.8's `swap38::stacks`, MiMo's `mimo2::swap::stacks`); a family that names a table must have asked it | `routed_legs::plan_line` over `PlanView::loader`, set by the adapters' `planned` | F4: MiMo's old `card_types` check restored in `swap.rs` turns the gate red on exactly the gate≠up triples whose three types pass `kernels::card_runs` |
//!
//! No weight is uploaded and no card is opened: the planners read headers
//! and the device census only, so the fixture tier runs it as the real tier
//! does. The run plans on the card `tools/gpu-gate.sh` put in view — the
//! recipe pins the 3090, the gate machine's bytes. The sets, and then the
//! swept headers, are planned on up to eight worker threads and printed in
//! the manifest's order; a `legs: time <family>` line per family and phase
//! names its plan calls, its seconds of work and the seconds a call, and the
//! last `legs: wall` line the run's wall: the gate holds the card's lock the
//! whole run, and its recipe's bound is 900 s.

#[cfg(all(feature = "glm5next", feature = "mimo2"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_routed_legs", routed_legs::run())
}

#[cfg(all(feature = "glm5next", feature = "mimo2"))]
#[path = "shared/routed_legs.rs"]
mod routed_legs;
