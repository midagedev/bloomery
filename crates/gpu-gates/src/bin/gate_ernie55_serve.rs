//! `gate_ernie55_serve` — the ERNIE 5.5 Flash seat of `bloomery-serve` against the model it serves,
//! on the ernie5_5 folder of the profile (`BLOOMERY_REF_MODEL`; the fixture cut under the fixture
//! tier): the seat starts, opens the checkpoint folder once and loads it on the placed body,
//! answers one chat request and keeps the prefix a request shares with the slot by the body's own
//! rule.
//!
//!     gate_ernie55_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name). Every
//! server is `bloomery-serve --model ernie55 -m <folder> …` beside this binary,
//! `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in `<dir>/<name>/`
//! (`server.err`, the answer).
//!
//! The clauses are the shared ones (`shared/serve_gate.rs`): the seat refuses a second slot, loads
//! the folder at the context it was asked for (the `load` record names arch `ernie5_5`, one slot,
//! the folder's layers and that context), answers a chat turn, and runs the prefix clause
//! `live_slot_keeps_the_prefix` at its own cut's fact: the body takes a checkpoint of the
//! window rings and the compressor state at each prompt call's start and end
//! (`app::arch::ernie55`'s `Batch::cut`), so a request that extends the last prompt keeps all of
//! it and answers as the same request on an empty slot does ([`serve_gate::Cut::Checkpoint`]; the
//! first reply is long enough to wrap both rings, so a restore that leaves one out answers
//! differently). The reasoning split and the turn end read what the model writes, which the fixture cut does not
//! write; they are the real folder's (`Tag::FileBound`), with the think span and the turn end as
//! the model's vocabulary spells them.
//!
//! - `parallel_above_one_is_refused`, `load_and_listen`, `chat_answers_a_turn`,
//!   `chat_splits_the_reasoning_at_the_models_tags`, `chat_ends_at_the_turn_end`,
//!   `live_slot_keeps_the_prefix`: the shared clauses, see `shared/serve_gate.rs`. FAIL-first: a
//!   seat that opens the folder as another family's (`.mutants/e55seat-wrong-family.patch`) is
//!   refused before its load and `load_and_listen` is red; a seat that takes a second slot prints
//!   its plan and `parallel_above_one_is_refused` is red.

#[cfg(all(feature = "deepseek41", feature = "ernie55"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_serve", gate::run())
}

#[cfg(all(feature = "deepseek41", feature = "ernie55"))]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(all(feature = "deepseek41", feature = "ernie55"))]
mod gate {
    use bloomery_gpu_gates::{GateError, model_file, record, ref_model_path};
    use model::arch::ernie55::place::PlanInputs;

    use crate::serve_gate;

    /// The slot's positions the server is asked for.
    const CTX: u64 = 4096;

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "<|ernie:im_end|>";

    pub fn run() -> Result<(), GateError> {
        let file = ref_model_path()?;
        let layers = model_file::with_header(&file, |split| {
            PlanInputs::describe(split).map(|inputs| inputs.model.layers)
        })??;
        let c = serve_gate::Clauses {
            gate: "gate_ernie55_serve",
            model: "ernie55",
            kinds: record::BLOOMERY_SERVE_HYBRID,
            load: &record::LOAD_HYBRID,
            listening: &record::LISTENING_HYBRID,
            arch: gguf::arch::ernie55::ARCH,
            layers: u64::try_from(layers)?,
            // The think span's open and close, as the vocabulary spells them.
            think: Some(["<ernie:think>", "</ernie:think>"]),
            turn_end: Some(TURN_END),
            ctx: CTX,
            // A cut keeps the newest checkpoint at or below the shared prefix: the body takes one
            // at each prompt call's start and end (`app::arch::ernie55`'s `Batch::cut`).
            cut: serve_gate::Cut::Checkpoint,
        };
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, serve_gate::Ask::THINK_SPAN)?;
        serve_gate::finish(&c, ok)
    }
}
