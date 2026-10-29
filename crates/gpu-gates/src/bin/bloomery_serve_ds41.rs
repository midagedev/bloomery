//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1
//! engine: [`ds41`]'s seat, whose module doc holds the contract (the flags,
//! the records, the prompt cache and the exit codes). `bloomery-serve --model
//! ds41` serves the same seat. llama-server's placement flags, their `LLAMA_ARG_*` variables and
//! `--list-devices` are read as `bloomery-serve` reads them (`bloomery_gpu_gates::llama_launch`), and
//! its events and exit codes are `bloomery-serve`'s (`bloomery_gpu_gates::seat_bin`).

#[cfg(feature = "deepseek41")]
const NAME: &str = "bloomery-serve-ds41";

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    // Before any x86-64-v3 code: `seat_bin::main` is not inlined, so `main` holds nothing but the
    // two calls.
    #[cfg(target_arch = "x86_64")]
    serve::preflight::cpu_gate(NAME);
    bloomery_gpu_gates::seat_bin::main(&bloomery_gpu_gates::seat_bin::Seat {
        name: NAME,
        word: "ds41",
        check: |args, read_levers, from_repo| ds41::check(args, read_levers, from_repo).map(drop),
        run: ds41::run,
    })
}

#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/ds41.rs"]
mod ds41;

// The seats' one owner of a round run as passes, the seat naming it beside
// itself (`super::rounds`) as it does under `serve_seats` in the one-binary
// server.
#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/rounds.rs"]
#[allow(
    dead_code,
    reason = "the seat steps its rows plain; the drafted half serves the drafted seats"
)]
mod rounds;

// The per-slot draft table `rounds` names for its drafted half
// (`super::drafted`).
#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/drafted.rs"]
#[allow(
    dead_code,
    reason = "only rounds' drafted half names it, which this seat does not run"
)]
mod drafted;

// The owner of the draft the seat serves (`super::served`).
#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/served.rs"]
mod served;

// The shared files the seat names at the crate root (`crate::dspark` and
// kin), as `generate_ds41` includes them.
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_draft.rs"]
mod draft;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_place.rs"]
mod place;

// The seat's `--place` word, the reader `ds41`'s argument parse calls (`place::parse`; the parse
// itself reads the census). It is tested here, in the per-model binary's root: a test inside the
// seat would also build in `bloomery-serve`, which includes it.
#[cfg(all(test, feature = "deepseek41"))]
mod tests {
    use super::place::parse;
    use model::placement::workstation::{A6000, RTX_3090};

    /// The V4.1 seat reads `--place` as the shared grammar with the one tier
    /// card its body serves: each word it took before parses to the placement
    /// it names, a list of more tier cards is refused by name at parse, and a
    /// word no card or ordinal spells is too.
    #[test]
    fn the_place_word_is_the_shared_grammar_with_the_v41_tier_count() {
        for (word, name, cards) in [
            ("a", "a", vec![A6000.name]),
            ("gate", "gate", vec![RTX_3090.name]),
            ("bp", "bp", vec![A6000.name, RTX_3090.name]),
            ("a6000+3090", "bp", vec![A6000.name, RTX_3090.name]),
            ("cuda1+0", "cuda1+0", vec!["cuda1", "cuda0"]),
            ("tp", "tp", vec![A6000.name]),
        ] {
            let p = parse(word).unwrap_or_else(|e| panic!("{word}: {e}"));
            assert_eq!((p.name(), p.cards()), (name, cards), "{word}");
        }
        let e = parse("cuda0+cuda1+cuda2")
            .expect_err("two tier cards")
            .to_string();
        assert_eq!(
            e,
            "--place cuda0+cuda1+cuda2: 2 expert tier cards, and the deepseek41 body serves 1 at most"
        );
        let e = parse("b2").expect_err("b2").to_string();
        assert!(e.starts_with("--place is a, gate, bp"), "{e}");
    }
}
