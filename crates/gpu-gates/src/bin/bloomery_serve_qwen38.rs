//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine: [`qwen38`]'s seat, whose module doc
//! holds the contract (the flags, the records, the keep rule and the exit
//! codes). `bloomery-serve --model qwen38` serves the same seat.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "bloomery-serve-qwen38: built without the `deepseek41` feature; see `just gate-gpu-qwen38-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match qwen38::run(&args) {
        // EX_SOFTWARE: the engine, not the listener or the load, ended the run.
        Ok(serve::ServeError::Engine(f)) => {
            eprintln!("bloomery-serve-qwen38: {f}");
            std::process::ExitCode::from(70)
        }
        Ok(e) => bloomery_gpu_gates::exit_with("bloomery-serve-qwen38", Err(e.into())),
        Err(e) => bloomery_gpu_gates::exit_with("bloomery-serve-qwen38", Err(e)),
    }
}

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/qwen38.rs"]
mod qwen38;

// The seats' MTP drafts, one a resident slot, a sibling of the seat as in
// `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/drafted.rs"]
mod drafted;

// The seats' one owner of a round run as passes, a sibling of the seat as
// in `bloomery-serve`: this seat's drafted rounds and — on a load that
// drafts nothing — its plain rounds run through it (`step_round`); the
// qwen3 seat's the plain ones on every load.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/rounds.rs"]
pub mod rounds;

// The seat's refusal and cut rules, pure functions of its feed (the seat's module
// doc holds the contract). They are tested here, in the per-model binary's root: a
// test inside the seat would also build in `bloomery-serve`, which includes it.
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use bloomery_gpu::arch::qwen3moe::Prompt38;

    use super::qwen38::{ACTS_ON, Feed38, prompt_cuts, refuse_feed};

    const STEPS: Feed38 = Feed38 {
        prompt: Prompt38::Step,
        traced: false,
    };
    const TRACED: Feed38 = Feed38 {
        prompt: Prompt38::Step,
        traced: true,
    };
    const BATCH_TRACED: Feed38 = Feed38 {
        prompt: Prompt38::Auto,
        traced: true,
    };
    const BATCH: Feed38 = Feed38 {
        prompt: Prompt38::Auto,
        traced: false,
    };

    /// The refusal `refuse_feed` gives, or a panic naming the call that
    /// passed.
    fn refused(feed: Feed38, run: (usize, bool), residency: Option<&str>) -> String {
        match refuse_feed(feed, run, residency) {
            Err(e) => e.to_string(),
            Ok(()) => panic!("refused nothing: slots {}, draft {}", run.0, run.1),
        }
    }

    #[test]
    fn a_route_trace_takes_only_the_step_feed() {
        let why = refused(BATCH_TRACED, (1, false), None);
        assert!(why.contains("set BLOOMERY_PREFILL=steps"), "{why}");
        refuse_feed(TRACED, (1, false), None).expect("a trace on the step feed of one plain slot");
    }

    #[test]
    fn a_route_trace_takes_no_mtp_draft() {
        let why = refused(TRACED, (1, true), None);
        assert!(why.contains("refused beside BLOOMERY_DRAFT=mtp"), "{why}");
    }

    #[test]
    fn a_route_trace_takes_one_slot() {
        let why = refused(TRACED, (2, false), None);
        assert!(
            why.contains("--parallel 2 puts another sequence's steps"),
            "{why}"
        );
    }

    #[test]
    fn a_route_trace_takes_no_residency_word_but_off() {
        let why = refused(TRACED, (1, false), Some("mid-p40-s1"));
        assert!(
            why.contains("BLOOMERY_RESIDENCY=mid-p40-s1 moves the slot map"),
            "{why}"
        );
    }

    #[test]
    fn the_step_feed_takes_no_mtp_draft() {
        let why = refused(STEPS, (1, true), None);
        assert!(
            why.contains("BLOOMERY_PREFILL=steps feeds a prompt one step an id"),
            "{why}"
        );
        assert!(why.contains("set BLOOMERY_DRAFT=off"), "{why}");
    }

    #[test]
    fn the_step_feed_takes_no_residency_word_but_off() {
        let why = refused(STEPS, (1, false), Some("mid-p40-s1"));
        assert!(
            why.contains("BLOOMERY_RESIDENCY=mid-p40-s1 beside BLOOMERY_PREFILL=steps"),
            "{why}"
        );
    }

    #[test]
    fn a_step_fed_prompt_is_never_cut() {
        let marks = [10, 40];
        assert_eq!(prompt_cuts(Prompt38::Auto, 0, 60, &marks), marks);
        assert!(prompt_cuts(Prompt38::Step, 0, 60, &marks).is_empty());
    }

    #[test]
    fn the_batch_feed_without_a_trace_refuses_nothing() {
        refuse_feed(BATCH, (2, true), Some("mid-p40-s1"))
            .expect("the seat's own feed beside every other lever");
    }

    /// Every accessor the seat calls on `Levers`, with the levers it reads.
    const READS: &[(&str, &[&str])] = &[
        ("draft", &[bloomery_levers::DRAFT]),
        (
            "host",
            &[
                bloomery_levers::HOST_POPULATE,
                bloomery_levers::HOST_LOCK,
                bloomery_levers::CARD_DONTNEED,
                bloomery_levers::R8,
            ],
        ),
        ("mtp_draft", &[bloomery_levers::MTP_DRAFT]),
        ("mtp_head_rows", &[bloomery_levers::MTP_HEAD_ROWS]),
        ("mtp_width", &[bloomery_levers::MTP_WIDTH]),
        ("pin_main", &[bloomery_levers::PIN_MAIN]),
        ("prefill", &[bloomery_levers::PREFILL]),
        ("qwen38_experts", &[bloomery_levers::QWEN38_EXPERTS]),
        ("residency", &[bloomery_levers::RESIDENCY]),
        ("route_trace", &[bloomery_levers::ROUTE_TRACE]),
        ("step_stats", &[bloomery_levers::STEP_STATS]),
        ("xstream", &[bloomery_levers::XSTREAM]),
    ];

    /// The lever `PlanLevers::from_levers` reads for the seat's plans.
    const PLAN_READS: &[&str] = &[bloomery_levers::CARD_BUDGET];

    #[test]
    fn the_levers_list_is_the_levers_the_seat_reads() {
        let src = include_str!("shared/serve_seats/qwen38.rs");
        let mut called = std::collections::BTreeSet::new();
        for (at, _) in src.match_indices("levers.") {
            let own = src[..at]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
            let rest = &src[at + "levers.".len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if own && rest[name.len()..].starts_with('(') {
                called.insert(name);
            }
        }
        let mut read = std::collections::BTreeSet::new();
        for name in &called {
            let Some((_, levers)) = READS.iter().find(|(a, _)| a == name) else {
                panic!("the seat calls levers.{name}(): add its levers to READS and to ACTS_ON");
            };
            read.extend(levers.iter().copied());
        }
        for (accessor, _) in READS {
            assert!(
                called.contains(*accessor),
                "READS names levers.{accessor}(), which the seat no longer calls"
            );
        }
        read.extend(PLAN_READS.iter().copied());
        for lever in &read {
            assert!(
                ACTS_ON.contains(lever),
                "{lever} is read by the seat but not in ACTS_ON"
            );
        }
        for lever in ACTS_ON {
            assert!(
                read.contains(lever),
                "{lever} is in ACTS_ON but the seat reads no such lever"
            );
        }
    }
}
