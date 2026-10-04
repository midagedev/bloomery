//! `gate_qwen3_serve` — the qwen3 seat of `bloomery-serve` against
//! `generate_qwen3moe`, on each model file given, one process at a time on
//! the card the runner pins.
//!
//!     gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir>
//!
//! For each file it starts `bloomery-serve --model qwen3 -m <file> --port 0
//! --parallel 1` (beside this binary, `BLOOMERY_REF_MODEL` removed from its
//! environment: the file is named once, by `-m`; the plain engine pinned,
//! the prefix clauses holding the one-slot path's keeps) and holds:
//!
//! - `load_and_listen`: the server prints its `load` record for the file's
//!   architecture before its `listening` record;
//! - `unplaced_default_names_itself`: with no `--place` the default is the
//!   whole-card load printing no `plan` record (a card with room; today's
//!   lines), or the auto-placed plan whose one record names
//!   `whole_does_not_fit` (FAIL-first: a default that plans without the why);
//! - the chat turn [`MESSAGES`], rendered by the server's own template
//!   (`/apply-template`, `/tokenize`), is longer than the eight ids a pass
//!   takes, so the prompt call runs the ubatch walk;
//! - `completion_ids_are_the_cli_ids`: `/completion` at temperature 0, [`N`]
//!   tokens, of those ids and of [`PROSE`] (`/tokenize`, a raw continuation,
//!   which a model's next ids follow more loosely than a chat answer, so a
//!   prompt call fed other ids moves them sooner), gives for each the ids
//!   `generate_qwen3moe --arm <ids>/N … --last-step` prints for the same
//!   file (the server's cut: the prompt less its last id, then a step) — all
//!   of them, or a prefix ending at an end-of-generation id;
//! - `chat_is_those_ids`: `/v1/chat/completions` of the same turn at
//!   temperature 0 and `max_tokens` N is a 200 whose `usage` counts the
//!   prompt's ids and the completion's tokens, and whose message text (the
//!   reasoning and the content) is inside the text of those ids
//!   (`/detokenize`);
//! - on a qwen3moe file, whose seat keeps a prefix of what the slot holds,
//!   the prefix clauses: the edit clause — the turn of [`EDIT_A`] answered,
//!   then resent with its user message changed to [`EDIT_B`], which diverges
//!   at `j` inside the rows the first run's prompt call wrote with at least
//!   [`GEMM_FROM`] ids after it — keeps `j` (`cache_n`) and its ids are the
//!   same ids fed fresh (`cache_prompt: false`: a cut call of at least
//!   `GEMM_FROM` rows is the GEMM walk wherever it is cut); and the
//!   extension clause — the turn resent with its reply and [`LATER`]'s user
//!   turn — keeps every position the slot held (`cache_n == held`), its ids
//!   printed only, for the rows the first run's steps wrote are not a fresh
//!   prompt call's rows;
//! - on a qwen35moe file, whose seat keeps a prefix back to the checkpoints
//!   a marked prompt call took (every ubatch of positions and its end), the
//!   prefix clauses: the stripped clause — the turn of [`EDIT_A`] answered,
//!   then resent with a reasoning-free reply ([`STRIPPED_35`]) in place of
//!   the reply, so the shared prefix ends at the turn's prompt end — keeps
//!   the checkpoint there (`cache_n == len(p1) - 1`) and its ids are the
//!   same ids fed fresh (a re-fed call of at least [`GEMM_FROM`] rows is the
//!   wide walk wherever it is cut); and the extension clause — the turn
//!   resent with its reply and [`LATER`]'s user turn — keeps every position
//!   the slot held (`cache_n == held`: the ask reaches the standing
//!   position, which no cut takes back), its ids printed only.
//!   FAIL-first mutants, each red on its line: a keep rule that grants the
//!   common prefix instead of the checkpoint makes the session's cut refuse
//!   by name and both clauses' requests fail; the cut's restore omitted
//!   leaves the re-fed ids a fresh run's (the stripped clause's ids red).
//! - `swap_reprefills_the_parked_ids`, on the qwen35moe arm (the park is
//!   the seat's, and the qwen3moe arm's whole-card load serves resident
//!   slots instead): a server of `--parallel 2`, a
//!   decode preempted mid-run by a second request comes back by the
//!   re-prefill fallback — the engine reset to 0 and its held ids fed again
//!   before it steps — so the first request's ids are its solo run's through
//!   the park (the same steps wrote them; past it the re-fed call's walk
//!   re-writes the rows its steps wrote, the prefill band, so the tail is
//!   printed, not held) and the second's, which never parks, are its solo
//!   run's; the second finishes before the first, the switches counter
//!   moved and the park's re-fed ids counted in `/metrics`. FAIL-first: the
//!   re-prefill omitted leaves the first request stepping from an empty
//!   engine and the re-fed count at 0.
//! - `slots_flow_together`, on the qwen3moe arm: a server of
//!   `--parallel 2` (the gate's ctx handling — no `--ctx`, the auto
//!   default), two greedy streamed requests of distinct prompts
//!   (`n_predict` [`SLOTS_PREDICT`], `ignore_eos`, no cache), each first run
//!   alone on that same server and then both together —
//!   (v1) each request's together ids are its alone ids (the slots hold
//!   separate sequences; nothing one runs feeds the other);
//!   (v2) `swaps_total` is absent or 0 (no park), and over the together run
//!   the deltas of the busy slots booked / `n_decode_total` sit at ≥ 1.5
//!   (`/metrics` carries the busy count as the running mean
//!   `n_busy_slots_per_decode`; times `n_decode_total` it is the total):
//!   the worker books one call a round carrying every running slot
//!   (`worker.rs`'s `Stats`), so two requests that decode together book 2
//!   busy a call, the two prompts and any non-overlapping head or tail
//!   rounds 1 — at 96 tokens each the ratio is (2·94 + 2 + ~2)/(94 + 2 +
//!   ~2) ≈ 1.9 — while a turn-taking server books exactly 1 busy a call
//!   (every round is one slot's), 1.0 past the two prompts;
//!   (v3) the two SSE streams' token arrivals, timestamped by a reader
//!   thread a stream: while both are live, every window of
//!   [`SLOTS_WINDOW`] = 16 consecutive arrivals holds at least
//!   [`SLOTS_EACH`] = 4 of each stream — the engine emits one token a slot
//!   a round (8 and 8 ideally), delivery batching (the server flushes per
//!   event; curl and the pipe may clump) tolerates up to 12 consecutive
//!   same-stream arrivals before the minority drops under 4, while a
//!   turn-taking server's `QUANTUM` = 64-token turns put windows of 16
//!   inside a turn holding 16 of one stream and none of the other;
//!   (v4) the load record names `slots=2` and `slot_ctx` = the auto default
//!   the one-slot arm measured over two (a cache row the granularity, the
//!   floor exact), and `/props`' `n_ctx` is that slot ctx.
//!   FAIL-first mutants, each red on its line: the seat still building the
//!   turn-taking engine leaves (v2) at ~1.0 and (v3) failing; a select that
//!   ignores the slot (both requests on one sequence) moves the together
//!   ids off the alone ones; a reply that hands row answers back in the
//!   wrong order scrambles the ids; a ctx not divided gives each slot the
//!   total, and the second sequence does not fit past the first (the auto
//!   default spent the card's free bytes on the one-sequence cache): the
//!   server never listens, which the clause names as the split's check red.
//! - `slot_ctx_too_small_is_refused`, on the qwen3moe arm: `--ctx 15`
//!   under the default two slots — a slot of 7 rows, one under the 8 rows
//!   the load's widest captured pass writes (`router::MAX_TOKENS`) — ends
//!   the process before the load, naming the total, the slot count and the
//!   slot ctx. FAIL-first: the refusal dropped loads the slot and dies in
//!   the pass capture under the launcher's own message.
//! - `placed_slots_take_turns`, `ctx_default`'s placed arm (`--place a
//!   --parallel 2`): a placed load's slots take its one sequence in turns
//!   over the whole context — its plan counts the card bytes of one
//!   sequence — so the load record names one resident sequence and the
//!   listening record two slots of the arm's `n_ctx` each. FAIL-first: a
//!   placed load made resident adds a sequence its plan never counted, and
//!   either never listens or names two in its load.
//!
//! The spawn census, every server this gate starts and its `--parallel`:
//! [`spawn`] and `ctx_default`'s default, flag and q8 arms pin
//! `--parallel 1` (the prefix and ctx clauses hold the one-slot path's
//! keeps and defaults), its placed arm passes `--parallel 2` (the placed
//! turns above); `cache_refusals`' two refusal arms pass none — they die at
//! flag parsing before the seat splits anything — and its flag-wins arm
//! pins `--parallel 1`; the swap clause passes `--parallel 2` (qwen35moe,
//! the turns); the slots clause passes `--parallel 2` (qwen3moe, resident)
//! and the refusal arm passes none (the default two are what makes the
//! split too small).
//!
//! The server is stopped by the handle this binary spawned it with before
//! the CLI loads. Logs per file in `<dir>/<n>/` (`server.err`, `gen.log`,
//! the responses).

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_qwen3_serve: built without the `deepseek41` feature; see `just gate-gpu-qwen3-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
    use bloomery_gpu_gates::serve_client::{Served, curl, ids_of, json_of, parse_ids};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use gguf::Split;
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    const USAGE: &str = "usage: gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir>";

    /// The tokens each request makes.
    const N: usize = 64;

    /// The chat turn every file answers.
    fn messages() -> Value {
        json!([{ "role": "user", "content": "Name the three primary colors of light, and say in one sentence why a screen mixes them." }])
    }

    /// The raw prompt every file continues.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and twelve, the same as every night, until the hundred and thirteenth";

    /// The edit clause's turn, and the same turn with its user message
    /// changed: the resend diverges inside the rows the first run's prompt
    /// call wrote, with the rest of the prompt after it.
    const EDIT_A: &str =
        "Name the three primary colors of light, and say in one sentence why a screen mixes them.";
    const EDIT_B: &str =
        "Name the three primary colors of ink, and say in one sentence why a page reflects them.";

    /// The whole-call clause's system message, ahead of [`EDIT_A`]'s user
    /// turn: the chat's prompt opens two messages.
    const SYSTEM_MSG: &str = "You answer in one short sentence, naming each item exactly as the \
                              question names it.";

    /// The user turn the extension clause appends after the reply, as the
    /// template renders it past the reply's end.
    const LATER: &str = "<|im_end|>\n<|im_start|>user\nAnd which of the three does a screen show when it shows none of them?\n<|im_end|>\n";

    /// The stripped clause's resend on a qwen35moe file (the module header):
    /// a reasoning-free reply in place of the reply, then a later user turn
    /// — the template's rendering of the reply with its reasoning dropped.
    /// Its first id is a word of the answer, never the reply's first id.
    const STRIPPED_35: &str = "Red, green and blue; a screen mixes them because each of its \
                               pixels emits those three lights side by side.<|im_end|>\n\
                               <|im_start|>user\nAnd which of the three does a screen show when \
                               it shows none of them?<|im_end|>\n<|im_start|>assistant\n";

    /// The prompt ids a pass takes at most: the rendered turn must be
    /// longer, so the clause covers the ubatch walk.
    const PASS_IDS: usize = 8;

    /// The swap clause's requests: the first long enough that the second's
    /// arrival cannot miss its decode (a whole quantum and more, so a late
    /// arrival still finds it mid-run), the second short enough to come back
    /// before it.
    const SWAP_A_PREDICT: usize = 96;
    const SWAP_B_PREDICT: usize = 8;

    /// The swap clause's `/slots` poll while it waits for the first
    /// request's decode.
    const SWAP_POLL: Duration = Duration::from_millis(300);

    /// The slots clause's requests: long enough that the together run holds
    /// dozens of rounds with both streams live.
    const SLOTS_PREDICT: usize = 96;

    /// The slots clause's interleave window and its floor for each stream
    /// (the module header's (v3) derivation).
    const SLOTS_WINDOW: usize = 16;
    const SLOTS_EACH: usize = 4;

    /// The slots clause's poll for the refusal arm's exit.
    const SLOTS_POLL: Duration = Duration::from_millis(500);

    /// The refusal arm's `--ctx`: two slots of one row under the seat's
    /// least slot context, the widest pass a whole-card load captures
    /// ([`MAX_TOKENS`] rows).
    const SLOTS_REFUSED_CTX: usize = 2 * MAX_TOKENS - 1;

    /// The fewest rows the seat's prompt call runs as the GEMM walk
    /// (`app::arch::qwen3moe::GEMM_FROM`): the rows a kept prefix leaves
    /// behind are a whole fresh run's.
    const GEMM_FROM: usize = app::arch::qwen3moe::GEMM_FROM;

    struct Args {
        models: Vec<PathBuf>,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut models, mut dir) = (Vec::new(), None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--model" => models.push(PathBuf::from(v)),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (models.is_empty(), dir) {
            (false, Some(dir)) => Ok(Args { models, dir }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// A binary beside this one.
    fn beside(name: &str) -> Result<PathBuf, GateError> {
        Ok(std::env::current_exe()?.with_file_name(name))
    }

    /// `/metrics`' `llamacpp:<name>` value; `None` when it carries none.
    fn metric(url: &dyn Fn(&str) -> String, name: &str) -> Result<Option<f64>, GateError> {
        let (st, body) = curl(&url("/metrics"), None, false)?;
        if st != 200 {
            return Err(format!("/metrics: HTTP {st}: {body}").into());
        }
        let key = format!("llamacpp:{name} ");
        Ok(body
            .lines()
            .find_map(|l| l.strip_prefix(&key))
            .and_then(|v| v.trim().parse().ok()))
    }

    /// The busy slots the server has booked over its `decodes` engine calls:
    /// `/metrics` carries them as the running mean `n_busy_slots_per_decode`
    /// (`serve::api`'s metrics), so the total is that mean times the calls.
    fn busy_total(url: &dyn Fn(&str) -> String, decodes: f64) -> Result<f64, GateError> {
        Ok(metric(url, "n_busy_slots_per_decode")?.unwrap_or(f64::NAN) * decodes)
    }

    /// The whole-number field ` name=…` of a record line; `None` when the
    /// line carries none.
    fn field_u64(line: &str, name: &str) -> Option<u64> {
        line.split(&format!(" {name}="))
            .nth(1)?
            .split(' ')
            .next()?
            .parse()
            .ok()
    }

    /// The `load` record's line and the `listening` record's line of a
    /// server's stderr, each empty when the server printed none.
    fn load_and_listening(log: &str) -> (&str, &str) {
        let load = log.lines().find(|l| l.starts_with("load ")).unwrap_or("");
        let listening = log
            .lines()
            .find(|l| l.contains(" listening on http://"))
            .unwrap_or("");
        (load, listening)
    }

    /// The seat's default `--ctx` against the file and this gate's card:
    /// unset, the whole-card load takes the file's trained context capped
    /// to what the card had free — at least the 4096 floor, a multiple of
    /// 1024, and past the floor when the card has room past it (a 24 GB
    /// card and a Q4_K_M 30B file leave tens of thousands of rows); the
    /// `--ctx` flag still wins; a load under `--place` takes the placed
    /// search's answer over the plan's expert split, pinned by relation
    /// (the placed arm's checks below), and its two slots take the one
    /// sequence in turns (`placed_slots_take_turns`). FAIL-first: a search
    /// that hands back the trained context uncapped makes the default arm's
    /// load a plan the card cannot hold (the spawn never listens), and one
    /// that hands back nothing leaves the default at the floor. Returns the
    /// default arm's `n_ctx` — the one-slot load's total context, what the
    /// slots clause's split halves.
    fn ctx_default(model: &Path, dir: &Path, ok: &mut bool) -> Result<Option<u64>, GateError> {
        // The trained context read from the file beside the server, not the
        // server's own echo of it.
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let trained = split
            .arch_get_u64("context_length")
            .ok_or_else(|| format!("{}: no context_length", model.display()))?;
        let props_ctx = |url: &dyn Fn(&str) -> String| -> Result<u64, GateError> {
            let (st, body) = curl(&url("/props"), None, false)?;
            let v = json_of("/props", st, &body)?;
            Ok(v["n_ctx"].as_u64().unwrap_or(u64::MAX))
        };
        // Each arm loads the model, so each takes its own directory.
        let arms: [(&str, &[&str]); 4] = [
            ("default", &["--parallel", "1"]),
            ("flag", &["--parallel", "1", "--ctx", "2048"]),
            // Two slots on the placed load: they take its one sequence in
            // turns (the module header's `placed_slots_take_turns`).
            ("placed", &["--parallel", "2", "--place", "a"]),
            // The cache lever's arm: the q8_0 planes the seat's flag names,
            // the auto context search under the halved KV term.
            ("q8", &["--parallel", "1", "--cache-type-k", "q8_0"]),
        ];
        // The default arm's answer, for the q8 arm's growth relation.
        let mut default_n = None;
        for (name, extra) in arms {
            let d = dir.join(format!("ctx-{name}"));
            std::fs::create_dir_all(&d)?;
            let err_log = d.join("server.err");
            let mut cmd = Command::new(beside("bloomery-serve")?);
            cmd.env_remove("BLOOMERY_REF_MODEL");
            let m = model.to_str().ok_or("the model path is not UTF-8")?;
            let mut args: Vec<&str> = vec!["--model", "qwen3", "--port", "0", "-m", m];
            args.extend_from_slice(extra);
            let mut s = Served::spawn_cmd(cmd, &args, &d)?;
            let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
            let url = |p: &str| format!("http://{addr}{p}");
            let n = props_ctx(&url)?;
            println!("ctx arm {name}: props n_ctx {n} of trained {trained}");
            check(ok, "ctx_never_passes_the_trained_context", n <= trained);
            match name {
                "flag" => check(ok, "ctx_flag_wins", n == 2048),
                "q8" => {
                    // The load names the format it holds, the halved KV term
                    // never shrinks the auto answer (where the card has room
                    // past the f16 answer it grows it — the fit rides the
                    // census, so the growth prints, not judged), and the
                    // residents differ by exactly the planes' derived delta:
                    // the file's layers, KV heads and head width at the
                    // served context, 15/8 B a value (f16's 4 against q8_0's
                    // 17/8, both planes). FAIL-first: a budget or allocation
                    // that ignores the flag leaves the delta wrong or 0.
                    let log = std::fs::read_to_string(&err_log)?;
                    let loads = log.lines().filter(|l| l.starts_with("load ")).count();
                    let says_q8 = log
                        .lines()
                        .any(|l| l.starts_with("load ") && l.contains("cache=q8_0"));
                    check(ok, "q8_load_names_its_cache", loads == 1 && says_q8);
                    check(
                        ok,
                        "q8_ctx_never_below_the_f16_answer",
                        n >= 4096
                            && n % 1024 == 0
                            && n <= trained
                            && n >= default_n.unwrap_or(u64::MAX),
                    );
                    let resident = |log: &str| {
                        log.lines()
                            .find(|l| l.starts_with("load "))
                            .and_then(|l| l.split("resident_bytes=").nth(1))
                            .and_then(|r| r.split(' ').next())
                            .and_then(|r| r.parse::<u64>().ok())
                    };
                    let kv_heads = split
                        .arch_get_u64("attention.head_count_kv")
                        .or_else(|| split.arch_get_u64("attention.head_count"))
                        .ok_or("no attention.head_count(_kv)")?;
                    let head_dim = split
                        .arch_get_u64("attention.key_length")
                        .ok_or("no attention.key_length")?;
                    let layers = split.arch_get_u64("block_count").ok_or("no block_count")?;
                    // A qwen3moe file's every layer carries the planes, so
                    // the delta is exact; a qwen35moe file's delta layers
                    // carry none, so the same product is only the upper
                    // bound (their exact 17/32 term is the model crate's
                    // own unit pin).
                    let ceiling = layers * n * kv_heads * head_dim * 15 / 8;
                    let f16_log =
                        std::fs::read_to_string(dir.join("ctx-default").join("server.err"))?;
                    let dropped = resident(&f16_log)
                        .zip(resident(&log))
                        .is_some_and(|(a, b)| {
                            let d = a.saturating_sub(b);
                            match split.architecture() {
                                Some("qwen3moe") => d == ceiling,
                                _ => d > 0 && d <= ceiling,
                            }
                        });
                    println!(
                        "ctx arm q8: residents f16 {:?} q8_0 {:?}, the planes' delta at most \
                         {ceiling} B",
                        resident(&f16_log),
                        resident(&log)
                    );
                    check(ok, "q8_resident_drops_by_the_planes", dropped);
                }
                "default" => {
                    check(
                        ok,
                        "ctx_default_is_capped_to_the_card",
                        n >= 4096 && n % 1024 == 0 && n > 4096,
                    );
                    let log = std::fs::read_to_string(&err_log)?;
                    let says_f16 = log
                        .lines()
                        .any(|l| l.starts_with("load ") && l.contains("cache=f16"));
                    check(ok, "default_load_names_f16", says_f16);
                    default_n = Some(n);
                }
                _ => {
                    // The placed default is searched over the plan's own
                    // expert split, and the relation is what holds on every
                    // card — the fit rides the census reading: the served
                    // context keeps the floor and the granule, is the ctx
                    // the one `plan` record was made at, and the one
                    // `--ctx defaults to` line names it, present exactly
                    // when the search stopped below the trained context
                    // (an idle A6000 holds every expert at the trained
                    // context and prints none). FAIL-first: a search that
                    // ignores its bound plans past what the card holds and
                    // the server dies before listening; a load that says
                    // nothing leaves the line's side of the biconditional
                    // red on a card where the search grows.
                    check(
                        ok,
                        "placed_ctx_keeps_the_floor_and_the_granule",
                        n >= 4096 && n % 1024 == 0,
                    );
                    let log = std::fs::read_to_string(&err_log)?;
                    let plan_ctx = log.lines().find(|l| l.starts_with("plan ")).and_then(|l| {
                        l.split("ctx_max=")
                            .nth(1)?
                            .split(' ')
                            .next()?
                            .parse::<u64>()
                            .ok()
                    });
                    println!("placed arm: plan ctx_max {plan_ctx:?}");
                    check(ok, "placed_ctx_is_the_plans_ctx_max", Some(n) == plan_ctx);
                    let said: Vec<u64> = log
                        .lines()
                        .filter(|l| l.contains("--ctx defaults to "))
                        .filter_map(|l| {
                            l.split("--ctx defaults to ")
                                .nth(1)?
                                .chars()
                                .take_while(char::is_ascii_digit)
                                .collect::<String>()
                                .parse()
                                .ok()
                        })
                        .collect();
                    println!("placed arm: --ctx default lines {said:?} of trained {trained}");
                    check(
                        ok,
                        "placed_names_its_default_ctx",
                        (n < trained) == !said.is_empty()
                            && said.len() <= 1
                            && (said.is_empty() || said[0] == n),
                    );
                    // The two slots take the placed load's one sequence in
                    // turns over the whole context (the module header).
                    let (load, listening) = load_and_listening(&log);
                    let terms = [
                        field_u64(load, "slots"),
                        field_u64(load, "slot_ctx"),
                        field_u64(listening, "slots"),
                        field_u64(listening, "slot_ctx"),
                    ];
                    println!(
                        "placed arm: load slots={:?} slot_ctx={:?}, listening slots={:?} \
                         slot_ctx={:?}",
                        terms[0], terms[1], terms[2], terms[3]
                    );
                    check(
                        ok,
                        "placed_slots_take_turns",
                        terms == [Some(1), Some(n), Some(2), Some(n)],
                    );
                }
            }
            println!("ctx arm {name}: server stopped: {}", s.stop()?);
        }
        // The default arm's answer, the one-slot load's total context: the
        // slots clause's split is half of it.
        Ok(default_n)
    }

    /// `bloomery-serve --model qwen3 -m <model> --port 0 --parallel 1` beside
    /// this binary, `BLOOMERY_REF_MODEL` removed, its logs in `dir`. The
    /// plain engine is pinned: the prefix clauses below hold the one-slot
    /// path's keeps.
    fn spawn(model: &Path, dir: &Path) -> Result<Served, GateError> {
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "-m",
                m,
            ],
            dir,
        )
    }

    /// What the server answered for one prompt.
    struct Answer {
        prompt: Vec<u32>,
        tokens: Vec<u32>,
        stop: String,
        /// `cache_n`: the positions the request kept of what the slot held.
        cache_n: u64,
    }

    /// `/completion` of `prompt` at temperature 0 — `cache` asks the server
    /// to keep the prefix the prompt shares with the slot — its body in
    /// `<dir>/<name>.json`.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        prompt: Vec<u32>,
        dir: &Path,
        name: &str,
        cache: bool,
    ) -> Result<(Answer, Value), GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": N, "temperature": 0, "return_tokens": true,
            "cache_prompt": cache,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        std::fs::write(dir.join(format!("{name}.json")), &text)?;
        let v = json_of("/completion", st, &text)?;
        let tokens = ids_of(&v["tokens"]);
        let stop = v["stop_type"].as_str().unwrap_or("").to_owned();
        let cache_n = v["timings"]["cache_n"].as_u64().unwrap_or(u64::MAX);
        println!("{name} completion tokens {tokens:?} stop {stop} cache_n {cache_n}");
        Ok((
            Answer {
                prompt,
                tokens,
                stop,
                cache_n,
            },
            v,
        ))
    }

    /// The ids of `messages` as the server's chat template renders them
    /// (`/apply-template`, then `/tokenize`).
    fn rendered(url: &dyn Fn(&str) -> String, messages: Value) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        tokenized(url, &text)
    }

    /// `/tokenize` of `text`.
    fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": text })), false)?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// The prefix clauses (the module header), on the qwen3moe arm alone:
    /// the edit clause — the turn answered, then resent with its user
    /// message changed — diverges at `j`, inside the rows the first run's
    /// prompt call wrote, with at least [`GEMM_FROM`] ids after it, keeps
    /// `j` (`cache_n`) and answers the ids of the same ids fed fresh; the
    /// extension clause — the turn resent with its reply and [`LATER`]'s
    /// user turn — keeps every position the slot held, its ids printed
    /// only; the whole-call clause — a chat whose prompt opens a system and
    /// a user message, each at least [`GEMM_FROM`] ids, on a slot holding
    /// another conversation — runs as one prompt call: no `prefill split`
    /// note (the per-position cache keeps every mark as it stands, so a cut
    /// would buy nothing and cost a walk), and the answer it returns is
    /// held to the CLI's below, a fresh run's ids.
    fn prefix(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<Answer, GateError> {
        let p = rendered(url, json!([{ "role": "user", "content": EDIT_A }]))?;
        let q = rendered(url, json!([{ "role": "user", "content": EDIT_B }]))?;
        let later = tokenized(url, LATER)?;
        let j = p.iter().zip(&q).take_while(|(a, b)| a == b).count();
        if j == 0 || j >= p.len() || q.len() < j + GEMM_FROM + 1 {
            return Err(format!(
                "the edit clause's turns diverge at {j} of {} and {} ids; the divergence must \
                 sit inside the first turn's prompt rows with at least {} ids after it",
                p.len(),
                q.len(),
                GEMM_FROM
            )
            .into());
        }
        // The edit clause: the turn answered fresh, resent with the user
        // message changed, and the changed turn fed fresh.
        let (first, _) = greedy(url, p.clone(), dir, "edit_first", false)?;
        let (edit, _) = greedy(url, q.clone(), dir, "edit", true)?;
        let (fresh, _) = greedy(url, q, dir, "edit_fresh", false)?;
        println!(
            "edit: the turns diverge at {j}; the resend kept {}; its ids {:?}; the fresh run's \
             {:?}",
            edit.cache_n, edit.tokens, fresh.tokens
        );
        check(
            ok,
            "edit_resend_keeps_the_row_where_it_diverges",
            first.cache_n == 0 && edit.cache_n == j as u64,
        );
        check(
            ok,
            "edit_resend_ids_are_a_fresh_runs",
            !fresh.tokens.is_empty() && edit.tokens == fresh.tokens && fresh.cache_n == 0,
        );

        // The extension clause: the turn answered again, then resent with
        // its reply and a later user turn.
        let (held_run, _) = greedy(url, p, dir, "extend_first", false)?;
        let held = (held_run.prompt.len() + held_run.tokens.len()).saturating_sub(1) as u64;
        let mut extend = held_run.prompt.clone();
        extend.extend_from_slice(&held_run.tokens);
        extend.extend_from_slice(&later);
        let (resend, _) = greedy(url, extend, dir, "extend", true)?;
        println!(
            "extension: the slot held {held}; the resend kept {} and answered {:?}",
            resend.cache_n, resend.tokens
        );
        check(
            ok,
            "extension_keeps_every_held_position",
            held_run.cache_n == 0 && resend.cache_n == held && !resend.tokens.is_empty(),
        );

        // The whole-call clause: the two-message chat on the slot the
        // extension left holding another conversation. Each message at
        // least GEMM_FROM ids, so the user message's start sits GEMM_FROM
        // ids inside the call either side — a mark a cutting rule could
        // take.
        let n_sys = tokenized(url, SYSTEM_MSG)?.len();
        let n_user = tokenized(url, EDIT_A)?.len();
        if n_sys < GEMM_FROM || n_user < GEMM_FROM {
            return Err(format!(
                "the whole-call clause's messages hold {n_sys} and {n_user} ids; each needs at \
                 least {GEMM_FROM} so a message-start mark sits inside the call"
            )
            .into());
        }
        let chat = rendered(
            url,
            json!([
                { "role": "system", "content": SYSTEM_MSG },
                { "role": "user", "content": EDIT_A },
            ]),
        )?;
        let (whole, _) = greedy(url, chat, dir, "whole", true)?;
        let log = std::fs::read_to_string(dir.join("server.err"))?;
        let cuts = log.lines().filter(|l| l.contains("prefill split")).count();
        println!(
            "whole-call: the chat holds {} ids; the server's log carries {cuts} prefill split \
             note(s)",
            whole.prompt.len(),
        );
        check(ok, "two_message_chat_runs_one_call", cuts == 0);
        Ok(whole)
    }

    /// The prefix clauses of the qwen35moe arm (the module header): the
    /// stripped clause — the turn answered, then resent with a
    /// reasoning-free reply in place of the reply, so the shared prefix
    /// ends at the turn's prompt end — keeps the checkpoint there and
    /// answers the ids of the same ids fed fresh; the extension clause —
    /// the turn resent with its reply and [`LATER`]'s user turn — keeps
    /// every position the slot held, its ids printed only.
    fn prefix35(url: &dyn Fn(&str) -> String, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": EDIT_A }]))?;
        let later = tokenized(url, LATER)?;
        let stripped = tokenized(url, STRIPPED_35)?;
        if stripped.len() < GEMM_FROM || later.len() < GEMM_FROM {
            return Err(format!(
                "the prefix clauses' resends hold {} and {} ids past the turn; each needs at \
                 least {GEMM_FROM}",
                stripped.len(),
                later.len()
            )
            .into());
        }
        // The stripped clause: the turn answered fresh, resent with the
        // reasoning-free reply, and the same ids fed fresh.
        let (held_run, _) = greedy(url, p1.clone(), dir, "strip35_first", false)?;
        if held_run.tokens.is_empty() || stripped.first() == held_run.tokens.first() {
            return Err(format!(
                "the reply's first id {:?} is the stripped reply's {:?}: the shared prefix \
                 would not end at the turn",
                held_run.tokens.first(),
                stripped.first()
            )
            .into());
        }
        let mut strip = p1.clone();
        strip.extend_from_slice(&stripped);
        let (resend, _) = greedy(url, strip.clone(), dir, "strip35", true)?;
        let (fresh, _) = greedy(url, strip, dir, "strip35_fresh", false)?;
        let turn_end = p1.len() as u64 - 1;
        println!(
            "strip35: the turn ends at {turn_end}; the resend kept {} and answered {:?}; the \
             fresh run's {:?}",
            resend.cache_n, resend.tokens, fresh.tokens
        );
        check(
            ok,
            "qwen35_stripped_resend_keeps_the_turns_end",
            held_run.cache_n == 0 && resend.cache_n == turn_end,
        );
        check(
            ok,
            "qwen35_stripped_resend_ids_are_a_fresh_runs",
            !resend.tokens.is_empty() && resend.tokens == fresh.tokens && fresh.cache_n == 0,
        );

        // The extension clause: the turn resent with its reply and a later
        // user turn. The stripped clause's runs left the slot holding the
        // stripped conversation, so the fresh turn runs again first — the
        // slot back at the reply's end, its prompt call's end mark standing.
        let (held_again, _) = greedy(url, p1.clone(), dir, "extend35_first", false)?;
        if held_again.tokens != held_run.tokens {
            return Err(
                "the extension's fresh turn answered ids the first one did not: the clause                  cannot hold a shared prefix"
                    .into(),
            );
        }
        let held = (held_again.prompt.len() + held_again.tokens.len()).saturating_sub(1) as u64;
        let mut extend = held_again.prompt.clone();
        extend.extend_from_slice(&held_again.tokens);
        extend.extend_from_slice(&later);
        let (resend, _) = greedy(url, extend, dir, "extend35", true)?;
        println!(
            "extension35: the slot held {held}; the resend kept {} and answered {:?}",
            resend.cache_n, resend.tokens
        );
        check(
            ok,
            "qwen35_extension_keeps_every_held_position",
            resend.cache_n == held && !resend.tokens.is_empty(),
        );
        Ok(())
    }

    /// The swap clause (module header) on a server of two slots started into
    /// `<dir>/swap`: a decode preempted mid-run by a second request comes
    /// back by the re-prefill fallback — the engine reset to 0 and its held
    /// ids fed again before it steps (`Park::Ids`; the seat holds no
    /// snapshot to park) — so both requests answer their solo runs' ids, the
    /// second finishes before the first, and the switches and the re-fed
    /// ids the park counted both moved.
    fn swap_reprefills_the_parked_ids(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let dir = dir.join("swap");
        std::fs::create_dir_all(&dir)?;
        let err_log = dir.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "2",
                "-m",
                m,
            ],
            &dir,
        )?;
        // The load reads the whole file: up to ten minutes from a cold cache.
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let body = |ids: &[u32], n: usize| {
            json!({
                "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
                "cache_prompt": false,
                // The seat runs no draft, so a banned stop id steps nothing
                // plainly (glm's clause avoids it for its draft); this holds
                // the first request's decode open for the second's arrival.
                "ignore_eos": true,
            })
        };
        let (a_ids, b_ids) = {
            let a = rendered(&url, messages())?;
            let (st, text) = curl(&url("/tokenize"), Some(&json!({ "content": PROSE })), false)?;
            let b = ids_of(&json_of("/tokenize", st, &text)?["tokens"]);
            (a, b)
        };
        // The solo runs: one request at a time takes no turn, so these are
        // the plain engine's answers.
        let mut alone = Vec::new();
        for (ids, n) in [(&a_ids, SWAP_A_PREDICT), (&b_ids, SWAP_B_PREDICT)] {
            let (st, text) = curl(&url("/completion"), Some(&body(ids, n)), false)?;
            let v = json_of("/completion", st, &text)?;
            alone.push(ids_of(&v["tokens"]));
        }
        println!("swap alone: {} and {} ids", alone[0].len(), alone[1].len());
        let swaps = metric(&url, "swaps_total")?.unwrap_or(f64::NAN);
        let refed = metric(&url, "swap_reprefill_tokens_total")?.unwrap_or(f64::NAN);
        // A request on a helper thread of its own: its handle, and its answer
        // with when it came back.
        type Answer = (Result<(u16, String), String>, Instant);
        let post = |ids: Vec<u32>,
                    n: usize|
         -> Result<(JoinHandle<()>, mpsc::Receiver<Answer>), GateError> {
            let u = url("/completion");
            let b = body(&ids, n);
            let (tx, rx) = mpsc::channel();
            let (h, _) = spawn_helper("swap-request", Placement::Float, move || {
                let r = curl(&u, Some(&b), false).map_err(|e| e.to_string());
                let _ = tx.send((r, Instant::now()));
            })
            .map_err(|e| format!("swap: {}", e.what()))?;
            Ok((h, rx))
        };
        let a_len = a_ids.len();
        let first = post(a_ids, SWAP_A_PREDICT)?;
        loop {
            if first.0.is_finished() {
                return Err(
                    "swap: the first request ended before /slots showed it decoding".into(),
                );
            }
            let (st, text) = curl(&url("/slots"), None, false)?;
            let slots = json_of("/slots", st, &text)?;
            let decoding = slots.as_array().is_some_and(|l| {
                l.iter().any(|s| {
                    s["turn"] == "running"
                        && s["next_token"]["n_decoded"].as_u64().is_some_and(|n| n > 0)
                })
            });
            if decoding {
                break;
            }
            std::thread::sleep(SWAP_POLL);
        }
        let second = post(b_ids, SWAP_B_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("swap: the {what} request's thread panicked"))?;
            let (r, at) = rx
                .recv()
                .map_err(|_| format!("swap: the {what} request's thread gave no answer"))?;
            let (st, text) = r?;
            let v = json_of("/completion", st, &text)?;
            together.push((ids_of(&v["tokens"]), at));
        }
        let after = metric(&url, "swaps_total")?.unwrap_or(f64::NAN);
        let refed_after = metric(&url, "swap_reprefill_tokens_total")?.unwrap_or(f64::NAN);
        // The park re-fed the first request's held ids: its prompt's ids plus
        // the tokens it had written before the park, so many tokens of its
        // answer. Through those its ids are its solo run's (the same steps
        // wrote them); past the park the re-fed call's walk re-writes the
        // rows its steps wrote (the prefill band: a step-written row is not
        // a fresh run's, qwen38-(a) class), so the tail is printed, not
        // held. The second request never parks: its ids are its solo run's.
        let pre_park = (refed_after as usize)
            .saturating_sub(a_len)
            .min(alone[0].len());
        let tail_same = together[0].0 == alone[0];
        println!(
            "swap together: first {} ids, second {} ids, second back {:?} before the first, \
             switches {swaps} -> {after}, re-fed ids {refed} -> {refed_after} ({} held of the \
             first), the first's ids its solo run's through {pre_park}, the whole tail the \
             same {tail_same}",
            together[0].0.len(),
            together[1].0.len(),
            together[0].1.checked_duration_since(together[1].1),
            refed_after,
        );
        check(
            ok,
            "swap_alone_ran_long_enough_to_preempt",
            alone[0].len() == SWAP_A_PREDICT && !alone[1].is_empty(),
        );
        check(
            ok,
            "swap_second_back_before_the_first",
            together[1].1 < together[0].1,
        );
        check(
            ok,
            "swap_first_ids_are_alone_through_the_park",
            together[0].0.len() >= pre_park
                && pre_park > 0
                && together[0].0[..pre_park] == alone[0][..pre_park],
        );
        check(
            ok,
            "swap_second_ids_are_alone",
            !together[1].0.is_empty() && together[1].0 == alone[1],
        );
        check(
            ok,
            "swap_switched_and_reprefilled_the_held_ids",
            after > swaps && refed_after > refed,
        );
        println!("swap server stopped: {}", s.stop()?);
        Ok(())
    }

    /// One streamed request's answer: the final event's tokens, and each
    /// token event's arrival on its reader thread's clock.
    struct Streamed {
        tokens: Vec<u32>,
        arrivals: Vec<Instant>,
    }

    /// A streamed request's answer channel: what its reader thread sends
    /// when the stream ends.
    type StreamedRx = mpsc::Receiver<Result<Streamed, String>>;

    /// `/completion` of `ids` at temperature 0, streamed: a helper thread of
    /// its own runs `curl -N` and reads the SSE lines as they land, each
    /// token event timestamped the moment it is read (the module header's
    /// (v3) reader thread a stream). The final event carries the tokens.
    fn streamed(
        addr: &str,
        ids: &[u32],
        n: usize,
    ) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        let (tx, rx) = mpsc::channel();
        let body = json!({
            "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
            "cache_prompt": false, "ignore_eos": true, "stream": true,
        });
        let url = format!("http://{addr}/completion");
        let (h, _) = spawn_helper("slots-request", Placement::Float, move || {
            let run = (|| -> Result<Streamed, String> {
                use std::io::{BufRead, BufReader, Write};
                let mut child = Command::new("curl")
                    .args([
                        "-sS",
                        "-N",
                        "--max-time",
                        "600",
                        "-H",
                        "Content-Type: application/json",
                        "--data-binary",
                        "@-",
                    ])
                    .arg(&url)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("curl {url}: {e}"))?;
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| format!("curl {url}: no stdin"))?;
                stdin
                    .write_all(body.to_string().as_bytes())
                    .map_err(|e| format!("curl {url}: {e}"))?;
                drop(stdin);
                let out = child
                    .stdout
                    .take()
                    .ok_or_else(|| format!("curl {url}: no stdout"))?;
                let mut arrivals = Vec::new();
                let mut tokens = Vec::new();
                for line in BufReader::new(out).lines() {
                    let line = line.map_err(|e| format!("curl {url}: {e}"))?;
                    let at = Instant::now();
                    let Some(v) = line
                        .strip_prefix("data: ")
                        .and_then(|d| serde_json::from_str::<Value>(d).ok())
                    else {
                        continue;
                    };
                    if v["stop"] == json!(false) && !v["content"].as_str().unwrap_or("").is_empty()
                    {
                        arrivals.push(at);
                    }
                    if v.get("tokens").is_some_and(Value::is_array) {
                        tokens = ids_of(&v["tokens"]);
                    }
                }
                let status = child.wait().map_err(|e| format!("curl {url}: {e}"))?;
                if !status.success() {
                    return Err(format!("curl {url}: {status}"));
                }
                Ok(Streamed { tokens, arrivals })
            })();
            let _ = tx.send(run);
        })
        .map_err(|e| format!("slots: {}", e.what()))?;
        Ok((h, rx))
    }

    /// The slots clause (module header) on a server of `--parallel 2` started
    /// into `<dir>/slots`: two greedy streamed requests of distinct prompts,
    /// each first run alone on that same server and then both together, the
    /// ids, the round counters and the streams' interleaving held, the load
    /// record's split and `/props`' context pinned to half the one-slot
    /// default `total`.
    fn slots_flow_together(
        model: &Path,
        dir: &Path,
        total: Option<u64>,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let dir = dir.join("slots");
        std::fs::create_dir_all(&dir)?;
        let err_log = dir.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "2",
                "-m",
                m,
            ],
            &dir,
        )?;
        let addr = match s.address(&err_log, 600, Duration::from_secs(1)) {
            Ok(a) => a,
            // A load that never listens names no split: the split's check
            // is red by name, and the clauses after this one still run.
            Err(e) => {
                println!("slots: the server never listened: {e}");
                check(ok, "slots_load_names_the_split", false);
                let _ = s.stop();
                return Ok(());
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        // (v4) The load names the split and /props the slot ctx.
        let log = std::fs::read_to_string(&err_log)?;
        let (load, _) = load_and_listening(&log);
        let field = |name: &str| field_u64(load, name);
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        let n_ctx = props["n_ctx"].as_u64().unwrap_or(u64::MAX);
        println!(
            "slots: {load}; the one-slot default {total:?}, props n_ctx {n_ctx}, record \
             slots={:?} slot_ctx={:?}",
            field("slots"),
            field("slot_ctx"),
        );
        check(
            ok,
            "slots_load_names_the_split",
            field("slots") == Some(2) && total.is_some_and(|t| field("slot_ctx") == Some(t / 2)),
        );
        check(
            ok,
            "props_names_the_slot_ctx",
            total.is_some_and(|t| n_ctx == t / 2),
        );
        // Two distinct prompts, each first run alone, then both together.
        let a_ids = rendered(&url, messages())?;
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": PROSE })), false)?;
        let b_ids = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        let mut alone = Vec::new();
        for ids in [&a_ids, &b_ids] {
            let (h, rx) = streamed(&addr, ids, SLOTS_PREDICT)?;
            h.join()
                .map_err(|_| "slots: an alone request's thread panicked")?;
            let ran = rx
                .recv()
                .map_err(|_| "slots: an alone request's thread gave no answer")??;
            println!("slots alone: {} ids", ran.tokens.len());
            alone.push(ran.tokens);
        }
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy_total(&url, decode0)?;
        // Both posted at once: the prompts serialize on the engine thread
        // (a round or two of skew), so nearly every decode round carries
        // both slots — the (v2) bound's shape.
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("slots: the {what} request's thread gave no answer"))??;
            together.push(ran);
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy_total(&url, decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        // (v1) The slots hold separate sequences: together ids are alone ids.
        check(
            ok,
            "slots_alone_ids_stay",
            !together[0].tokens.is_empty()
                && !together[1].tokens.is_empty()
                && together[0].tokens == alone[0]
                && together[1].tokens == alone[1],
        );
        // (v2) No park, and a round carries both slots (the derivation in
        // the module header): ≥ 1.5 against the turns' 1.0.
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        println!(
            "slots together: {} and {} ids; decode {decode0} -> {decode1}, busy {busy0} -> \
             {busy1} (ratio {ratio:.3}); swaps {swaps:?}",
            together[0].tokens.len(),
            together[1].tokens.len(),
        );
        check(
            ok,
            "slots_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && ratio >= 1.5,
        );
        // (v3) While both streams are live, every window of SLOTS_WINDOW
        // consecutive arrivals holds at least SLOTS_EACH of each: both live
        // from the later stream's first arrival to the earlier one's last.
        let mut merged: Vec<(usize, Instant)> = together
            .iter()
            .enumerate()
            .flat_map(|(i, s)| s.arrivals.iter().map(move |&t| (i, t)))
            .collect();
        merged.sort_by_key(|&(_, t)| t);
        let live_from = (0..2)
            .map(|i| together[i].arrivals.first().copied())
            .max()
            .flatten();
        let live_to = (0..2)
            .map(|i| together[i].arrivals.last().copied())
            .min()
            .flatten();
        let tags: Vec<usize> = merged
            .iter()
            .filter(|(_, t)| live_from.is_some_and(|a| *t >= a) && live_to.is_some_and(|b| *t < b))
            .map(|&(i, _)| i)
            .collect();
        let mut thin = 0;
        for w in tags.windows(SLOTS_WINDOW) {
            let of_first = w.iter().filter(|&&i| i == 0).count();
            if of_first < SLOTS_EACH || SLOTS_WINDOW - of_first < SLOTS_EACH {
                thin += 1;
            }
        }
        println!(
            "slots interleave: {} arrivals while both live, {} thin window(s) of {}",
            tags.len(),
            thin,
            SLOTS_WINDOW,
        );
        check(
            ok,
            "slots_streams_interleave",
            !tags.is_empty() && thin == 0,
        );
        println!("slots server stopped: {}", s.stop()?);
        Ok(())
    }

    /// The refusal clause (module header): `--ctx` [`SLOTS_REFUSED_CTX`]
    /// under the default two slots ends the process before the load,
    /// naming the total, the slot count and the slot ctx it split, one row
    /// under the seat's least. A refusal that loads instead shows as the
    /// server still running (or listening) past the polls and is stopped
    /// here, red; one that dies inside the load names another cause, red.
    fn slot_ctx_too_small_is_refused(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("refuse");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let total = SLOTS_REFUSED_CTX.to_string();
        let mut s = Served::spawn_cmd(
            cmd,
            &["--model", "qwen3", "--port", "0", "--ctx", &total, "-m", m],
            &d,
        )?;
        let mut said = String::new();
        let mut refused = false;
        for _ in 0..360 {
            said = std::fs::read_to_string(&err_log).unwrap_or_default();
            match s.child.try_wait()? {
                Some(status) => {
                    refused = !status.success()
                        && said.contains("--parallel 2")
                        && said.contains(&format!("of a --ctx of {total}:"))
                        && said.contains(&format!(
                            "a slot's context of {} rows is below the {MAX_TOKENS} rows",
                            SLOTS_REFUSED_CTX / 2
                        ));
                    break;
                }
                None if said.contains("listening on http://") => break,
                None => std::thread::sleep(SLOTS_POLL),
            }
        }
        println!(
            "slots refusal: exit {:?}, said {said}",
            s.child.try_wait()?.map(|s| s.code())
        );
        check(ok, "slot_ctx_too_small_is_refused", refused);
        let _ = s.stop();
        Ok(())
    }

    /// The server's clauses for `model`, its logs in `dir`; the answers the
    /// CLI is held to, the chat turn's first, and the file's architecture
    /// (which arm carries which clause below).
    fn served(model: &Path, dir: &Path, ok: &mut bool) -> Result<(Vec<Answer>, String), GateError> {
        let err_log = dir.join("server.err");
        let mut s = spawn(model, dir)?;
        // The load reads the whole file: up to ten minutes from a cold cache.
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let log = std::fs::read_to_string(&err_log)?;
        let load = log.lines().position(|l| l.starts_with("load arch="));
        let listen = log.lines().position(|l| l.contains("listening on http://"));
        let arch = log
            .lines()
            .find_map(|l| l.strip_prefix("load arch="))
            .and_then(|r| r.split(' ').next())
            .unwrap_or("")
            .to_owned();
        println!(
            "server {} at {addr}; load record {:?}",
            model.display(),
            load.and_then(|i| log.lines().nth(i))
        );
        check(
            ok,
            "load_and_listen",
            matches!((load, listen), (Some(a), Some(b)) if a < b),
        );
        // The server ran with no `--place`: the default is exactly one of
        // two things — the whole-card load, printing no `plan` record (a
        // card with room; today's lines), or the auto-placed plan, its one
        // record naming `whole_does_not_fit`. FAIL-first: a default that
        // plans without the why, or prints more than one plan record, turns
        // this red.
        let plans: Vec<&str> = log.lines().filter(|l| l.starts_with("plan ")).collect();
        let default_ok =
            plans.is_empty() || (plans.len() == 1 && plans[0].contains("why=whole_does_not_fit"));
        println!("unplaced default: {} plan record(s) {plans:?}", plans.len());
        check(
            ok,
            "unplaced_default_names_itself",
            load.is_some() && default_ok,
        );

        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages() })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        std::fs::write(dir.join("prompt.txt"), &text)?;
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": text })), false)?;
        let prompt = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("rendered prompt: {} ids", prompt.len());
        check(ok, "the_turn_is_past_a_pass", prompt.len() > PASS_IDS);

        let (chat_ids, v) = greedy(&url, prompt, dir, "completion", false)?;
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": PROSE })), false)?;
        let prose = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("prose prompt: {} ids", prose.len());
        check(ok, "the_prose_is_past_a_pass", prose.len() > PASS_IDS);
        let (prose_ids, _) = greedy(&url, prose, dir, "prose", false)?;

        let body = json!({
            "messages": messages(), "max_tokens": N, "temperature": 0, "stream": false,
        });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        std::fs::write(dir.join("chat.json"), &text)?;
        let chat = json_of("/v1/chat/completions", st, &text)?;
        let msg = &chat["choices"][0]["message"];
        let said = format!(
            "{}{}",
            msg["reasoning_content"].as_str().unwrap_or(""),
            msg["content"].as_str().unwrap_or("")
        );
        let (st, text) = curl(
            &url("/detokenize"),
            Some(&json!({ "tokens": chat_ids.tokens })),
            false,
        )?;
        let detok = json_of("/detokenize", st, &text)?["content"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let usage = &chat["usage"];
        let counted = usage["prompt_tokens"].as_u64() == Some(chat_ids.prompt.len() as u64)
            && usage["completion_tokens"].as_u64() == v["tokens_predicted"].as_u64();
        let parts_inside = [
            msg["reasoning_content"].as_str().unwrap_or(""),
            msg["content"].as_str().unwrap_or(""),
        ]
        .iter()
        .all(|p| detok.contains(p.trim()));
        println!(
            "chat usage {usage}; said {} chars, the ids' text {} chars",
            said.len(),
            detok.len()
        );
        check(
            ok,
            "chat_is_those_ids",
            counted && !said.trim().is_empty() && parts_inside,
        );
        // The qwen3moe arm's whole-call answer joins the CLI comparison: a
        // fresh run's ids for the two-message chat.
        let mut whole = None;
        if arch == "qwen3moe" {
            whole = Some(prefix(&url, dir, ok)?);
        }
        if arch == "qwen35moe" {
            prefix35(&url, dir, ok)?;
        }
        println!("server stopped: {}", s.stop()?);
        let mut answers = vec![chat_ids, prose_ids];
        answers.extend(whole);
        Ok((answers, arch))
    }

    /// `generate_qwen3moe --arm <prompt>/N … --last-step` on `model`, an arm
    /// a prompt; its `tokens` lines in order.
    fn cli(model: &Path, prompts: &[&[u32]], dir: &Path) -> Result<Vec<Vec<u32>>, GateError> {
        let exe = beside("generate_qwen3moe")?;
        let mut args = Vec::new();
        for p in prompts {
            let ids: Vec<String> = p.iter().map(u32::to_string).collect();
            args.push("--arm".to_owned());
            args.push(format!("{}/{N}", ids.join(",")));
        }
        args.push("--last-step".to_owned());
        let out = Command::new(&exe)
            .args(&args)
            .env("BLOOMERY_REF_MODEL", model)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        let log = dir.join("gen.log");
        std::fs::write(&log, &out.stdout)?;
        std::fs::write(dir.join("gen.err"), &out.stderr)?;
        if !out.status.success() {
            return Err(format!(
                "{} {}: {}",
                exe.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
        let text = String::from_utf8(out.stdout)?;
        let lines: Vec<Vec<u32>> = text
            .lines()
            .filter_map(|l| l.strip_prefix("tokens "))
            .map(parse_ids)
            .collect::<Result<_, _>>()?;
        if lines.len() != prompts.len() {
            return Err(format!(
                "{}: {} `tokens` lines for {} arms",
                log.display(),
                lines.len(),
                prompts.len()
            )
            .into());
        }
        Ok(lines)
    }

    /// The cache flag's own surface (`--cache-type-k`): a word outside the
    /// two spellings refused by name before anything loads, `--cache-type-v`
    /// refused by name with the why (both planes quantize together), and
    /// the flag winning over the `BLOOMERY_QWEN3_KV` lever — a spawn with
    /// the lever at `q8_0` and the flag at `f16` loads the f16 planes and
    /// says so. FAIL-first: a parser that takes any word listens; one that
    /// lets the lever override the flag loads `cache=q8_0`.
    fn cache_refusals(model: &Path, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let refuses = |name: &str, want: &str| -> Result<bool, GateError> {
            let d = dir.join(format!("cache-{name}"));
            std::fs::create_dir_all(&d)?;
            let out = Command::new(beside("bloomery-serve")?)
                .args(["--model", "qwen3", "--port", "0", "-m", m])
                .args(
                    match name {
                        "word" => vec!["--cache-type-k", "q6_0"],
                        _ => vec!["--cache-type-v", "q8_0"],
                    }
                    .as_slice(),
                )
                .env_remove("BLOOMERY_REF_MODEL")
                .output()?;
            let said = String::from_utf8_lossy(&out.stderr).into_owned();
            std::fs::write(d.join("server.err"), &said)?;
            let refused = !out.status.success() && said.contains(want);
            println!(
                "cache arm {name}: exit {:?}, said {said}",
                out.status.code()
            );
            Ok(refused)
        };
        check(
            ok,
            "cache_type_k_refuses_other_words",
            refuses("word", "--cache-type-k takes f16 or q8_0, not \"q6_0\"")?,
        );
        check(
            ok,
            "cache_type_v_refused_naming_why",
            refuses("v", "--cache-type-v does not exist")?
                && std::fs::read_to_string(dir.join("cache-v").join("server.err"))?
                    .contains("together"),
        );
        // The flag wins over the lever: the spawn listens and its load names
        // the f16 planes.
        let d = dir.join("cache-flag-wins");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL")
            .env("BLOOMERY_QWEN3_KV", "q8_0");
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "-m",
                m,
                "--cache-type-k",
                "f16",
            ],
            &d,
        )?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/props"), None, false)?;
        json_of("/props", st, &body)?;
        let log = std::fs::read_to_string(&err_log)?;
        let f16 = log
            .lines()
            .any(|l| l.starts_with("load ") && l.contains("cache=f16"));
        println!("cache arm flag-wins: server stopped: {}", s.stop()?);
        check(ok, "cache_type_k_wins_over_the_lever", f16);
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let a = parse_args()?;
        let mut ok = true;
        for (i, model) in a.models.iter().enumerate() {
            let dir = a.dir.join(i.to_string());
            std::fs::create_dir_all(&dir)?;
            println!("== {}", model.display());
            let (answers, arch) = served(model, &dir, &mut ok)?;
            let prompts: Vec<&[u32]> = answers.iter().map(|a| a.prompt.as_slice()).collect();
            let references = cli(model, &prompts, &dir)?;
            let mut agree = true;
            for (ans, reference) in answers.iter().zip(&references) {
                println!("cli tokens {reference:?}");
                agree &= match ans.stop.as_str() {
                    "eos" => !ans.tokens.is_empty() && reference.starts_with(&ans.tokens),
                    _ => ans.tokens == *reference,
                };
            }
            check(&mut ok, "completion_ids_are_the_cli_ids", agree);
            let total = ctx_default(model, &dir, &mut ok)?;
            cache_refusals(model, &dir, &mut ok)?;
            if arch == "qwen3moe" {
                // The qwen3moe arm: the resident slots and the split they
                // serve (a moved coverage clause would be lost — the swap
                // clause below is the qwen35moe arm's now).
                slots_flow_together(model, &dir, total, &mut ok)?;
                slot_ctx_too_small_is_refused(model, &dir, &mut ok)?;
            }
            if arch == "qwen35moe" {
                // The park is the seat's, and the qwen3moe arm's whole-card
                // load serves resident slots: the qwen35moe arm carries the
                // swap clause.
                swap_reprefills_the_parked_ids(model, &dir, &mut ok)?;
            }
        }
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
