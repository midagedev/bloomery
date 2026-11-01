//! The MiniMax-M3.1-Flash end-to-end gate: the whole decode chain — every layer a grouped-query attention layer (64 query heads
//! over 4 key heads of 128 values, a per-head q/k norm, a partial NEOX rope) attending the blocks its indexer picks, the dense
//! lead's feed-forward blocks and the sparse layers' routed blocks with one shared expert, the head and the argmax — on the gate
//! card, read from the checkpoint folder in its stored formats (MXFP8 projections, NVFP4 routed experts, BF16 embedding and
//! head), against transformers' f32 dump of the same folder (`refset::arch::minimaxm31`, `tools/ref/mm31-dump.sh`).
//!
//! The prompts are the dump's five (`tools/ref/data/minimaxm3`): a an English paragraph (41 ids), b a Korean one (55), c a Rust
//! function (116), d the first 1,536 ids of wikitext-2, e the first 2,560; each is fed one id a step from a reset, and the
//! oracle's reads are the set's own: the last position's whole logits row (`result_output`), each layer's output (`l_out-L`, the
//! sets that hold it), the 40 largest logprobs (`topk.tsv`) and the greedy tokens with their margins (`greedy.tsv`, sets a, b
//! and c; d and e hold none and are left out of the teacher-forced clause). Set e is the one context past the 2,048 positions
//! 16 blocks of 128 cover, so its last position drops a block; it is read under the card kernel's selection rule, one top-16 per
//! index head ([`minimaxm31::E_PERHEAD`]).
//!
//! The clauses are `shared/lcpp_e2e.rs`'s (`finite_and_fault_clean`, `reset_rewinds`, `layer_outputs_near_tf`, its fixture
//! twin `layer_outputs_near_tf_fx`, `forced_layers_near_tf`, `head_near_tf`, and `top_logprobs_in_band` and
//! `forced_greedy_count_pin`, both deferred by name: the top-4 of 128 near ties carry any rounding into an exchange of
//! experts, so the layers and the head are held one at a time on the oracle's own rows); the prompt batch's own clauses are the
//! prefill gate's (`gate_minimaxm3_prefill`), so this gate declares no prefill clause. The fixture tier runs the fixture cut
//! (the folder's layers 0, 3 and 59, routed width 512) against its own dump ([`minimaxm31::fixture::TF`]). This file holds the
//! model's part: the families and the seat (`shared/minimaxm3_seat.rs`: the load, the tap ceilings — provisional,
//! [`seat::TAP_EARLY_CEIL`] over the dense lead and [`seat::TAP_CEIL`] after).
//!
//! Two modes, parsed before anything else and not clauses:
//! - `--plan`: host only, no card opened. The model `BLOOMERY_REF_MODEL` names, planned at `PLAN_CTX` positions on the gate card's
//!   bytes as the seat plans it; one line per run of sparse layers (card and host experts a layer), the card's bytes by term and the
//!   host's.
//! - `--smoke`: the model loaded through the seat at `SMOKE_CTX` positions, the ids of reference prompt a fed a step each, then
//!   `SMOKE_TOKENS` tokens decoded greedily; prints the ids, their text (a folder's tokenizer, a file's vocabulary pieces), the
//!   load wall and the fault word, and fails on a row that is not finite or a fault word that is not clean. It pins nothing.
//!
//! - `--refuse-slots`: the body asked for two resident sequences at the gate plan is refused by name before any card is opened.
//! - `--route-trace DIR SET`: the file loaded through the seat with the host tier writing a route trace into the new directory
//!   `DIR`, the ids of reference prompt `SET` (a, b or c) fed a step each; `DIR` then holds every sparse layer's routed ids and
//!   slot kinds a position, to be held against the set's `ffn_moe_topk-*` nodes (the first layer and position where the
//!   engine's selection differs from the oracle's). It pins nothing.
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
#[path = "shared/xstream_rule.rs"]
mod xstream_rule;

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
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
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
    use refset::arch::minimaxm31;
    use refset::arch::prompt_tables;
    use refset::family::Family;
    use std::time::Instant;

    pub const NAME: &str = "gate_minimaxm3_e2e";

    const RECIPE: &str = "gate-gpu-minimaxm3-e2e";

    /// The model's part of the gate: transformers' sets on the real folder and on the fixture cut, the seat and its ceilings.
    static GATE: Gate = Gate {
        name: NAME,
        recipe: RECIPE,
        arch: minimaxm31::ARCH,
        slug: "tf",
        who: "transformers",
        real: &minimaxm31::TF,
        fixture: Some(&minimaxm31::fixture::TF),
        tap_ceil: seat::tap_ceil,
        tap_ceils: seat::tap_ceils,
        open: seat::open,
        has_prefill: false,
        has_forced: true,
        has_head: true,
        n_expert: 128,
        free_ceil: Some(seat::free_ceil),
        // The shared logprob band is Solar-Open2's llama.cpp-oracle gate's, derived from two
        // implementations' logits; here a divergence is a discrete exchange of experts at a near
        // tie, not a Gaussian, so no sigma derives from it. The head clause holds the head on the
        // oracle's own row instead, and the forced arm holds the layers.
        top_deferred: Some(
            "the band is Solar-Open2's llama.cpp-oracle gate's, from two implementations' logits; a \
             divergence here is a discrete exchange of experts at a near tie, not a Gaussian, so no \
             sigma derives from it",
        ),
        // PIN(2026-10-23): deferred. Our own two paths, the prompt batch's GEMM and the step's gemv, part by sum order alone
        // (gate_minimaxm3_prefill, P=16: 2.2e-4 at layer 3, 0.498 at layer 59, |dlogprob| 10.48, z 24.65), the top-4 of 128
        // near ties flipping; any σ pin wide enough to pass that catches nothing. The real run's 19 of 192 at margins up to
        // 8.02 against the pin of 3. The forced arm, the layer outputs and head_near_tf keep the coverage (FAIL-first: the
        // Gemma norms without their +1 redden them).
        greedy_deferred: Some(
            "the top-4 of 128 near ties carry any rounding into a discrete exchange of experts, so two implementations' \
             teacher-forced argmax parts past any Gaussian band: the engine's own prompt batch and steps part the same \
             way (gate_minimaxm3_prefill); the forced arm holds every layer and head_near_tf the head on the \
             oracle's own rows instead",
        ),
        sigma_logit: seat::SIGMA_LOGIT,
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

    /// The tier's model, the family of its sets and the names of its prompt sets a, b and c.
    fn model() -> Result<(std::path::PathBuf, &'static Family, [&'static str; 3]), GateError> {
        let tier = Tier::from_env()?;
        let (family, abc) = match tier {
            Tier::Real => (
                &minimaxm31::TF,
                [minimaxm31::A, minimaxm31::B, minimaxm31::C],
            ),
            Tier::Fixture => (
                &minimaxm31::fixture::TF,
                [
                    minimaxm31::fixture::A,
                    minimaxm31::fixture::B,
                    minimaxm31::fixture::C,
                ],
            ),
        };
        let path = lcpp_e2e::model_path(tier, RECIPE, minimaxm31::ARCH)?;
        Ok((path, family, abc))
    }

    /// `--plan`: the gate plan of the file, host only.
    fn plan() -> Result<(), GateError> {
        let (path, _, _) = model()?;
        for line in seat::plan_lines(&path, PLAN_CTX)? {
            println!("{line}");
        }
        Ok(())
    }

    /// `--refuse-slots`: the body asked for two resident sequences is refused by name before any card is opened.
    fn refuse_slots() -> Result<(), GateError> {
        let (path, _, _) = model()?;
        let message = seat::refuse_second_slot(&path, PLAN_CTX)?;
        println!("refuse_slots: two resident sequences refused: {message} PASS");
        Ok(())
    }

    /// `--route-trace DIR SET`: the ids of reference prompt `SET` through the seat, the host tier's route trace written into `DIR`.
    fn route_trace(dir: &std::path::Path, set: &str) -> Result<(), GateError> {
        let (path, family, [a, b, c]) = model()?;
        let name = match set {
            "a" => a,
            "b" => b,
            "c" => c,
            other => {
                return Err(format!(
                    "{NAME}: --route-trace: prompt set {other:?} is not a, b or c"
                )
                .into());
            }
        };
        let man = RefManifest::open(&family.path(name), family)?;
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
        let (path, family, [a, _, _]) = model()?;
        // The prompt is the ids of the reference set's prompt a.
        let man = RefManifest::open(&family.path(a), family)?;
        let prompt = prompt_tables::tokens(&man)?.to_vec();
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
        println!("smoke: prompt set {a} = {} ids {prompt:?}", prompt.len());
        println!("smoke: {} ids {out:?}", out.len());
        println!("smoke: text {:?}", text_of(&path, &out)?);
        println!("smoke: load {load:.1} s, fault word {fault:?}, every row finite {finite}");
        if !finite {
            return Err(format!("{NAME}: --smoke met a logits row that is not finite").into());
        }
        fault
    }

    /// The text of `ids`: by a checkpoint folder's tokenizer ([`seat::decode`]), or as a file's vocabulary pieces.
    fn text_of(path: &std::path::Path, ids: &[u32]) -> Result<String, GateError> {
        if path.is_dir() {
            return seat::decode(path, ids);
        }
        let split = gguf::Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        Ok(piece_text(&pieces_of(&split)?, ids))
    }

    /// The vocabulary's pieces, `tokenizer.ggml.tokens` of the file.
    fn pieces_of(split: &gguf::Split) -> Result<Vec<String>, GateError> {
        match split.value("tokenizer.ggml.tokens") {
            Some(gguf::Value::Array(items)) => items
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    gguf::Value::String(s) => Ok(s.clone()),
                    _ => Err(format!("tokenizer.ggml.tokens[{i}] is not a string").into()),
                })
                .collect(),
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
