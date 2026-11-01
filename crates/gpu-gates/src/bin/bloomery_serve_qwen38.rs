//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine: [`qwen38`]'s seat, whose module doc
//! holds the contract (the flags, the records, the keep rule and the exit
//! codes). `bloomery-serve --model qwen38` serves the same seat. llama-server's placement flags,
//! their `LLAMA_ARG_*` variables and `--list-devices` are read as `bloomery-serve` reads them
//! (`bloomery_gpu_gates::llama_launch`), and its events and exit codes are `bloomery-serve`'s
//! (`bloomery_gpu_gates::seat_bin`).

#[cfg(feature = "gpu")]
const NAME: &str = "bloomery-serve-qwen38";

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    // Before any x86-64-v3 code: `seat_bin::main` is not inlined, so `main` holds nothing but the
    // two calls.
    #[cfg(target_arch = "x86_64")]
    serve::preflight::cpu_gate(NAME);
    bloomery_gpu_gates::seat_bin::main(&bloomery_gpu_gates::seat_bin::Seat {
        name: NAME,
        word: "qwen38",
        check: |args, read_levers, from_repo| qwen38::check(args, read_levers, from_repo).map(drop),
        run: qwen38::run,
    })
}

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/qwen38.rs"]
mod qwen38;

// The expert stream's rule the seat runs, its sibling as in `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/xstream_rule.rs"]
mod xstream_rule;

// The seats' one owner of a checkpoint folder's open, a sibling of the seat as in `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/folder.rs"]
#[allow(
    dead_code,
    reason = "this binary opens a Qwen3.8 folder alone: the checked open is what it reads"
)]
mod folder;

// A Qwen3.8 checkpoint folder, which the seat opens, a sibling of the seat as in `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/qwen38_folder.rs"]
mod qwen38_folder;

// The seats' MTP drafts, one a resident slot, a sibling of the seat as in
// `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/drafted.rs"]
mod drafted;

// The seats' one owner of a round run as passes, a sibling of the seat as
// in `bloomery-serve`: this seat's drafted rounds run through it
// (`step_round`), and so do its plain rounds on a load that drafts nothing
// or drafts with the MTP draft; the qwen3 seat's plain ones on every load.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/rounds.rs"]
pub mod rounds;

// The seat's refusal and cut rules, pure functions of its feed (the seat's module
// doc holds the contract). They are
// tested here, in the per-model binary's root: a test inside the seat would also
// build in `bloomery-serve`, which includes it.
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use bloomery_gpu::arch::qwen3moe::Prompt38;
    use bloomery_gpu::kv_format::{KvChoice, KvFrom, KvQ8};
    use bloomery_gpu_gates::GateError;
    use bloomery_levers::Draft38Off;
    use model::placement::workstation::{A6000, RTX_3090};

    use super::qwen38::{
        ACTS_ON, Feed38, Kind38, Place38, folder_draft_for, prompt_cuts, refuse_feed,
        refuse_lookup_feed, refuse_media_feed, setup38,
    };

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

    /// The seat reads `--place` as the shared grammar with the one tier card
    /// its body serves. The words it took before (`a`, `gate`, `bp`) parse to
    /// the same placement and kind; a card list of this workstation's cards
    /// spelled as an alias is that alias, and a list of other cards or of
    /// ordinals now parses, its kind read by structure (a tier card is `bp`,
    /// the word `gate` the gate card, any other single card `a` on that
    /// card's own bytes). What the body cannot serve (`tp`, more than one tier card, a word no card or
    /// ordinal spells) is refused by name at parse.
    #[test]
    fn the_place_word_is_the_shared_grammar_with_the_qwen38_tier_count() {
        for (word, kind, cards) in [
            ("a", Kind38::A, vec![A6000.name]),
            ("gate", Kind38::Gate, vec![RTX_3090.name]),
            ("bp", Kind38::Bp, vec![A6000.name, RTX_3090.name]),
        ] {
            let p = Place38::parse(word).unwrap_or_else(|e| panic!("{word}: {e}"));
            assert_eq!(
                (p.kind, p.name(), p.cards.cards()),
                (kind, word, cards),
                "{word}"
            );
            assert_eq!(p.tiered(), kind == Kind38::Bp, "{word}");
            assert_eq!(p.stage_a(), kind != Kind38::Gate, "{word}");
        }
        for (spelled, alias) in [("a6000", "a"), ("3090", "gate"), ("a6000+3090", "bp")] {
            let p = Place38::parse(spelled).unwrap_or_else(|e| panic!("{spelled}: {e}"));
            let q = Place38::parse(alias).expect(alias);
            assert_eq!((p.kind, p.name()), (q.kind, q.name()), "{spelled}");
        }
        for (word, kind, cards) in [
            ("cuda1", Kind38::A, vec!["cuda1"]),
            ("cuda0", Kind38::A, vec!["cuda0"]),
            ("cuda1+cuda0", Kind38::Bp, vec!["cuda1", "cuda0"]),
            ("0+1", Kind38::Bp, vec!["cuda0", "cuda1"]),
            ("3090+a6000", Kind38::Bp, vec![RTX_3090.name, A6000.name]),
        ] {
            let p = Place38::parse(word).unwrap_or_else(|e| panic!("{word}: {e}"));
            assert_eq!(
                (p.kind, p.name(), p.cards.cards()),
                (kind, word, cards),
                "{word}"
            );
            assert_eq!(p.tiered(), kind == Kind38::Bp, "{word}");
            assert_eq!(p.stage_a(), kind != Kind38::Gate, "{word}");
        }
        for (word, why) in [
            ("tp", "--place tp: a tensor-parallel run"),
            (
                "cuda0+cuda1+cuda2",
                "--place cuda0+cuda1+cuda2: 2 expert tier cards, and the qwen4exp body serves 1 at most",
            ),
            ("b2", "--place is a, gate, bp"),
        ] {
            let e = Place38::parse(word).expect_err(word).to_string();
            assert!(e.starts_with(why), "{word}: {e}");
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
    fn the_lookup_draft_takes_only_the_batched_feed_untraced() {
        for feed in [STEPS, TRACED, BATCH_TRACED] {
            let why = refuse_lookup_feed(feed)
                .expect_err("the lookup beside a one-row feed")
                .to_string();
            assert!(why.contains("BLOOMERY_DRAFT=lookup verifies"), "{why}");
        }
        refuse_lookup_feed(BATCH).expect("the lookup on the batched feed");
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

    /// An image load enters a span's rows only through the ubatch walk's media call, so the step
    /// feed is refused beside `--mmproj`, by name and with its way out.
    #[test]
    fn a_media_load_takes_no_step_feed() {
        let why = refuse_media_feed(STEPS)
            .expect_err("the step feed beside --mmproj")
            .to_string();
        assert!(
            why.contains("--mmproj with BLOOMERY_PREFILL=steps"),
            "{why}"
        );
        assert!(why.contains("set BLOOMERY_PREFILL=batch"), "{why}");
        refuse_media_feed(BATCH).expect("the ubatch walk takes a media call");
    }

    // PIN(2026-10-30): the two cases that held a media load refusing a set `mtp` draft and turning
    // an unset one off are gone with that rule: an image load drafts by the text load's rule, its
    // draft walks reading each sequence's rotation table and a span's rows (`Mtp38::store_rows`).

    /// A checkpoint folder's own MTP layer drafts when the draft's reader takes it. A folder
    /// without one, a layer its preparation refuses, and a prepared layer the reader refuses
    /// each refuse a set `mtp` naming the refusal, turn an unset draft off with the refusal as
    /// its reason (the target served), and keep a set `off` as set. Mutant: the reader's refusal
    /// dropped (the prepared layer drafts) or propagated (the load refused).
    #[test]
    fn a_folder_load_refuses_a_set_mtp_draft_and_turns_an_unset_one_off() {
        let layout = gguf::write::Layout::new(&[], Vec::new()).unwrap();
        let header = || gguf::Split::header_only("mtp".into(), &layout).unwrap();
        let ask = |layer: &Result<Option<gguf::Split>, String>,
                   reader: Result<(), &str>,
                   word: Option<&str>,
                   run: (bool, Option<Draft38Off>)| {
            folder_draft_for(layer, |_| reader.map_err(GateError::from), word, run)
        };
        let (f32, missing) = (
            "mtp.fc_embedding.weight is stored F32",
            "blk.48.attn_q.weight is not in the draft file",
        );
        for (layer, reader, why) in [
            (Ok(None), Ok(()), Draft38Off::Folder),
            (
                Err(f32.to_string()),
                Ok(()),
                Draft38Off::FolderMtp { why: f32.into() },
            ),
            (
                Ok(Some(header())),
                Err(missing),
                Draft38Off::FolderMtp {
                    why: missing.into(),
                },
            ),
        ] {
            let e = ask(&layer, reader, Some("mtp"), (true, None))
                .expect_err("BLOOMERY_DRAFT=mtp without a folder MTP layer the draft reads")
                .to_string();
            assert!(
                e.contains("BLOOMERY_DRAFT=mtp with a checkpoint folder"),
                "{e}"
            );
            if let Draft38Off::FolderMtp { why } = &why {
                assert!(e.contains(why.as_str()), "{e}");
            }
            let off = ask(&layer, reader, None, (true, None)).unwrap();
            assert_eq!(off, (false, Some(why)));
            let set = ask(&layer, reader, Some("off"), (false, Some(Draft38Off::Set))).unwrap();
            assert_eq!(set, (false, Some(Draft38Off::Set)));
        }
        let layer = Ok(Some(header()));
        for word in [None, Some("mtp")] {
            assert_eq!(
                ask(&layer, Ok(()), word, (true, None)).unwrap(),
                (true, None)
            );
        }
    }

    /// The `setup` line names the context's rule, the cache's format with what chose it, and a
    /// draft a rule turned off with its reason: the yield's card bytes and the positions a slot
    /// gets either way.
    #[test]
    fn setup_names_the_context_rule_and_why_the_draft_is_off() {
        use bloomery_levers::{Reserve, RungCost};
        use model::placement::ctx_ask::{Rule, Short, Side};
        let fitted = Rule::Fitted {
            from: 262_144,
            short: Short {
                on: Side::Card,
                need: 9,
                room: 8,
            },
        };
        let y = RungCost::of(Reserve::Draft, 4096, 69_632, 2_842_413_228).expect("a yield");
        let off = Draft38Off::Yield(y);
        let change = "--parallel N sets the slots, --ctx-size N their context and \
                      --cache-type-k f16|q8_0 the KV cache";
        assert_eq!(
            setup38(
                (false, false, false),
                Some(&off),
                (1, 69_632, fitted),
                KvChoice {
                    kv: KvQ8::F16,
                    from: KvFrom::Default
                }
            )
            .line(),
            format!(
                "setup draft=off slots=1 ctx=69632 ctx_rule=fitted cache=f16 \
                 cache_from=default (the draft is off, unset: the draft's 2842413228 B on the \
                 card leave a slot 4096 positions; without the draft a slot takes 69632; \
                 BLOOMERY_DRAFT=mtp drafts with the MTP layer; {change})"
            )
        );
        assert_eq!(
            setup38(
                (true, false, true),
                None,
                (1, 262_144, Rule::Target),
                KvChoice {
                    kv: KvQ8::Q8,
                    from: KvFrom::Flag
                }
            )
            .line(),
            format!(
                "setup draft=mtp draft_from=folder slots=1 ctx=262144 ctx_rule=target cache=q8_0 \
                 cache_from=flag (BLOOMERY_DRAFT=off serves without the draft, \
                 BLOOMERY_MTP_DRAFT=FILE drafts from a draft file; {change})"
            )
        );
    }

    // PIN(2026-10-29): the unset draft drafts on any slot count (a sampled request's window
    // rides the round's one pass), so the slot rule `draft_for_slots`, its `Draft38Off::Slots`
    // and their test are gone; gate_qwen38_serve's `draft_unset_drafts_on_two_slots` holds the
    // two-slot load.

    #[test]
    fn the_batch_feed_without_a_trace_refuses_nothing() {
        refuse_feed(BATCH, (2, true), Some("mid-p40-s1"))
            .expect("the seat's own feed beside every other lever");
    }

    // PIN(2026-10-23): no round falls back to keep slot 0's refresh (a pass without slot 0
    // walks into its first slot's own rows, `Body38::enqueue_slots`), so the arena model of
    // that rule is gone; gate_qwen4exp_mtp's (z) plain rounds pin slot 0's draft on the card.

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
        ("host_lanes", &[bloomery_levers::HOST_LANES]),
        ("mtp_draft", &[bloomery_levers::MTP_DRAFT]),
        ("mtp_head_rows", &[bloomery_levers::MTP_HEAD_ROWS]),
        ("mtp_width", &[bloomery_levers::MTP_WIDTH]),
        ("pin_main", &[bloomery_levers::PIN_MAIN]),
        ("prefill", &[bloomery_levers::PREFILL]),
        ("qwen38_experts", &[bloomery_levers::QWEN38_EXPERTS]),
        ("qwen3_kv_set", &[bloomery_levers::QWEN3_KV]),
        ("residency", &[bloomery_levers::RESIDENCY]),
        ("route_trace", &[bloomery_levers::ROUTE_TRACE]),
        ("schema_jump", &[bloomery_levers::SCHEMAJUMP]),
        ("screen", &[bloomery_levers::SCREEN]),
        ("step_stats", &[bloomery_levers::STEP_STATS]),
        ("xstream", &[bloomery_levers::XSTREAM]),
    ];

    /// The lever `PlanLevers::from_levers` reads for the seat's plans.
    const PLAN_READS: &[&str] = &[bloomery_levers::CARD_BUDGET];

    /// Each call of the seat that reads a Direct lever in place, with the lever.
    const DIRECT_READS: &[(&str, &str)] = &[(
        "ubatch_rule(a.",
        bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
    )];

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
        for (call, lever) in DIRECT_READS {
            assert!(
                src.contains(call),
                "the seat no longer calls {call}: drop {lever}"
            );
            read.insert(*lever);
        }
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
