//! The MiniMax-M3 end-to-end gate: the whole decode chain — every layer a grouped-query attention layer (64 query heads over 4 key
//! heads of 128 values, a per-head q/k norm, a partial NEOX rope), the sparse layers attending the blocks their indexer picks,
//! the dense lead's feed-forward blocks and the sparse layers' routed blocks with one shared expert, the head and the argmax —
//! on the gate card, against mainline llama.cpp's CPU dump of the same file (`refset::arch::minimaxm3`,
//! `tools/ref/minimaxm3-dump.sh`: llama.cpp `cd0a5b6a1`, no patch; the oracle and the public baseline are one engine).
//!
//! The prompts are the dump's five (`tools/ref/data/minimaxm3`): a an English paragraph (41 ids), b a Korean one (55), c a Rust
//! function (116), d the first 1,536 ids of wikitext-2, e the first 2,560; each is fed one id a step from a reset, and the
//! oracle's reads are the set's own: the last position's whole logits row (`result_output`), each layer's output (`l_out-L`, the
//! sets that hold it), the 40 largest logprobs (`topk.tsv`) and the greedy tokens with their margins (`greedy.tsv`, sets a, b
//! and c; d and e hold none and are left out of the teacher-forced clause). Set e is the one context past the 2,048 positions
//! 16 blocks of 128 cover, so its last position drops a block.
//!
//! The clauses are `shared/lcpp_e2e.rs`'s (`finite_and_fault_clean`, `reset_rewinds`, `layer_outputs_near_lcpp`,
//! `top_logprobs_in_band`, `forced_greedy_count_pin`); the body has no prompt batch, so the prefill clause is not declared, and the
//! family has no fixture file, so the fixture tier is refused by name before anything loads. This file holds the model's part: the
//! family and the seat (`shared/minimaxm3_seat.rs`: the load, the tap ceilings — provisional, [`seat::TAP_EARLY_CEIL`] over the
//! dense lead and [`seat::TAP_CEIL`] after).
//!
//! Two modes, parsed before anything else and not clauses:
//! - `--plan`: host only, no card opened. The file `BLOOMERY_REF_MODEL` names, planned at `PLAN_CTX` positions on the gate card's
//!   bytes as the seat plans it; one line per run of sparse layers (card and host experts a layer), the card's bytes by term and the
//!   host's.
//! - `--smoke`: the file loaded through the seat at `SMOKE_CTX` positions, the ids of reference prompt a fed a step each (the
//!   file's pre-tokenizer is not the tokenizer crate's yet), then `SMOKE_TOKENS` tokens decoded greedily; prints the ids, their
//!   vocabulary pieces, the load wall and the fault word, and fails on a row that is not finite or a fault word that is not
//!   clean. It pins nothing.
//!
//! - `--refuse-slots`: the body asked for two resident sequences at the gate plan is refused by name before any card is opened.
//! - `--route-trace DIR SET`: the file loaded through the seat with the host tier writing a route trace into the new directory
//!   `DIR`, the ids of reference prompt `SET` (a, b or c) fed a step each; `DIR` then holds every sparse layer's routed ids and
//!   slot kinds a position, to be held against the set's `ffn_moe_topk-*` nodes (the first layer and position where the
//!   engine's selection differs from mainline's). It pins nothing.
//!
//! `--arm a|bp` picks the placement arm (default `a`); `bp` is refused by name until the body serves an expert tier card.
//!

#[cfg(feature = "minimaxm3")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with(gate::NAME, gate::main())
}

#[cfg(feature = "minimaxm3")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "minimaxm3")]
#[path = "shared/minimaxm3_seat.rs"]
mod seat;

#[cfg(feature = "minimaxm3")]
mod gate {
    use crate::e2e::argmax;
    use crate::lcpp_e2e::{self, Arm, Gate};
    use crate::seat;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::RefManifest;
    use bloomery_gpu_gates::tier::Tier;
    use refset::arch::minimaxm3;
    use refset::arch::prompt_tables;
    use std::time::Instant;

    pub const NAME: &str = "gate_minimaxm3_e2e";

    const RECIPE: &str = "gate-gpu-minimaxm3-e2e";

    /// The model's part of the gate: mainline's sets on the real file (no fixture family), the seat and its ceilings.
    static GATE: Gate = Gate {
        name: NAME,
        recipe: RECIPE,
        arch: minimaxm3::ARCH,
        slug: "lcpp",
        who: "llama.cpp",
        real: &minimaxm3::LCPP,
        fixture: None,
        tap_ceil: seat::tap_ceil,
        tap_ceils: seat::tap_ceils,
        open: seat::open,
        has_prefill: false,
    };

    /// The positions `--plan` plans at.
    const PLAN_CTX: usize = 8192;

    /// The positions `--smoke` loads.
    const SMOKE_CTX: usize = 512;

    /// The tokens `--smoke` decodes.
    const SMOKE_TOKENS: usize = 16;

    /// What the command line asks for.
    enum Mode {
        Gate,
        Plan,
        Smoke,
        RefuseSlots,
        RouteTrace(std::path::PathBuf, String),
    }

    /// `--plan` or `--smoke` as the only argument, else the gate (whose own parse refuses an argument it does not know).
    fn mode() -> Result<Mode, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.as_slice() {
            [a] if a == "--plan" => Ok(Mode::Plan),
            [a] if a == "--smoke" => Ok(Mode::Smoke),
            [a] if a == "--refuse-slots" => Ok(Mode::RefuseSlots),
            [a, dir, set] if a == "--route-trace" => {
                Ok(Mode::RouteTrace(dir.into(), set.clone()))
            }
            _ if args.iter().any(|a| {
                a == "--plan" || a == "--smoke" || a == "--refuse-slots" || a == "--route-trace"
            }) =>
            {
                Err("--plan, --smoke, --refuse-slots and `--route-trace DIR SET` are each the whole command line of a run".into())
            }
            _ => Ok(Mode::Gate),
        }
    }

    pub fn main() -> Result<(), GateError> {
        match mode()? {
            Mode::Gate => lcpp_e2e::run(&GATE),
            Mode::Plan => plan(),
            Mode::Smoke => smoke(),
            Mode::RefuseSlots => refuse_slots(),
            Mode::RouteTrace(dir, set) => route_trace(&dir, &set),
        }
    }

    /// The real tier's file, the only one the modes run on.
    fn real_file() -> Result<std::path::PathBuf, GateError> {
        let tier = Tier::from_env()?;
        if tier == Tier::Fixture {
            return Err(format!(
                "{NAME}: the family has no fixture file: --plan and --smoke run in the real tier (`just {RECIPE}`)"
            )
            .into());
        }
        lcpp_e2e::model_path(tier, RECIPE, minimaxm3::ARCH)
    }

    /// `--plan`: the gate plan of the file, host only.
    fn plan() -> Result<(), GateError> {
        let path = real_file()?;
        for line in seat::plan_lines(&path, PLAN_CTX)? {
            println!("{line}");
        }
        Ok(())
    }

    /// `--refuse-slots`: the body asked for two resident sequences is refused by name before any card is opened.
    fn refuse_slots() -> Result<(), GateError> {
        let path = real_file()?;
        let message = seat::refuse_second_slot(&path, PLAN_CTX)?;
        println!("refuse_slots: two resident sequences refused: {message} PASS");
        Ok(())
    }

    /// `--route-trace DIR SET`: the ids of reference prompt `SET` through the seat, the host tier's route trace written into `DIR`.
    fn route_trace(dir: &std::path::Path, set: &str) -> Result<(), GateError> {
        let path = real_file()?;
        let name = match set {
            "a" => minimaxm3::A,
            "b" => minimaxm3::B,
            "c" => minimaxm3::C,
            other => {
                return Err(format!(
                    "{NAME}: --route-trace: prompt set {other:?} is not a, b or c"
                )
                .into());
            }
        };
        let man = RefManifest::open(&minimaxm3::LCPP.path(name), &minimaxm3::LCPP)?;
        let ids = prompt_tables::tokens(&man)?.to_vec();
        let rows = seat::route_trace(&path, SMOKE_CTX, dir, &ids)?;
        println!(
            "route_trace: {} ids of {name}, {rows} positions in {}",
            ids.len(),
            dir.display()
        );
        Ok(())
    }

    /// `--smoke`: load, feed the prompt, decode greedily, print.
    fn smoke() -> Result<(), GateError> {
        let path = real_file()?;
        let split =
            gguf::Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // The file's pre-tokenizer (`minimax-m2`) is one the tokenizer crate does not run yet: the prompt is the ids
        // of the reference set's prompt a, and the output is shown as the vocabulary's pieces.
        let man = RefManifest::open(&minimaxm3::LCPP.path(minimaxm3::A), &minimaxm3::LCPP)?;
        let prompt = prompt_tables::tokens(&man)?.to_vec();
        let pieces = pieces_of(&split)?;
        let t = Instant::now();
        let mut seat = seat::open(&path, SMOKE_CTX, Arm::A)?;
        let load = t.elapsed().as_secs_f64();
        let vocab = seat.dims().vocab;
        seat.reset()?;
        let mut finite = true;
        let mut row = Vec::new();
        for &id in &prompt {
            row = seat.step(id)?;
            finite &= is_row(&row, vocab);
        }
        let mut out = Vec::with_capacity(SMOKE_TOKENS);
        for n in 0..SMOKE_TOKENS {
            let next = argmax(&row);
            out.push(next);
            if n + 1 < SMOKE_TOKENS {
                row = seat.step(next)?;
                finite &= is_row(&row, vocab);
            }
        }
        let fault = seat.fault();
        println!(
            "smoke: prompt set {} = {} ids {prompt:?}",
            minimaxm3::A,
            prompt.len()
        );
        println!("smoke: {} ids {out:?}", out.len());
        println!("smoke: pieces {:?}", piece_text(&pieces, &out));
        println!("smoke: load {load:.1} s, fault word {fault:?}, every row finite {finite}");
        if !finite {
            return Err(format!("{NAME}: --smoke met a logits row that is not finite").into());
        }
        fault
    }

    /// The vocabulary's pieces, `tokenizer.ggml.tokens` of the file.
    fn pieces_of(split: &gguf::Split) -> Result<Vec<String>, GateError> {
        match split.value("tokenizer.ggml.tokens") {
            Some(gguf::Value::Array(items)) => Ok(items
                .iter()
                .map(|v| match v {
                    gguf::Value::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect()),
            _ => Err("the file holds no tokenizer.ggml.tokens array".into()),
        }
    }

    /// The pieces of `ids` as byte-level BPE writes them (a space is `\u{120}`, a newline `\u{10a}`), joined.
    fn piece_text(pieces: &[String], ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| {
                pieces
                    .get(i as usize)
                    .map_or("<out of range>".to_string(), |p| {
                        p.replace('\u{120}', " ").replace('\u{10a}', "\n")
                    })
            })
            .collect()
    }

    /// Whether `row` is a whole logits row of finite values.
    fn is_row(row: &[f32], vocab: usize) -> bool {
        row.len() == vocab && row.iter().all(|v| v.is_finite())
    }
}
