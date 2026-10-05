//! `gate_ds41_serve` — `bloomery-serve-ds41` on the 3090 (placement gate),
//! driven over HTTP, as one process under the GPU gate lock.
//!
//!     gate_ds41_serve --gen <generate_ds41 log> --prompt <text> --ids <a,b,…> --dir <out>
//!                     [--plain <the plain run's --dir> | --place bp | --slots | --need-path]
//!
//! Starts the server beside this binary (`--port 0 --place gate --parallel
//! 1`, the plain engine this gate's body holds), reads its
//! address from its stderr, waits for `/health`, then checks:
//!
//! - (p4) a slot save on this server, started without `--slot-save-path`, is
//!   the server's 501 naming the flag, and `/health` answers after it;
//! - `/props`' `engine` object, printed once, against this gate's own plan of
//!   the file the server opens (placement gate, the server's default
//!   context): the server's name, argv and pid; the model's architecture,
//!   shards, bytes on disk and counts as the header states them; a card
//!   `GPU<n>` and the host `CPU`, each device's `bytes` the sum of its classes
//!   and equal to the plan's card (dense + experts) and host (experts +
//!   tables) bytes; the cards' KV bytes; `ctx_verified` the deepest reference
//!   set's (`refset::arch::deepseek41::VERIFIED_POSITIONS`); no draft;
//! - `/completion` of `--prompt` at temperature 0 with `return_tokens`: its
//!   ids are `generate_ds41 --tokens <--ids> -n 16`'s `tokens` line — all 16,
//!   or a prefix ending in the end-of-generation id when the server stopped
//!   there (`generate_ds41` does not stop at it);
//! - the same `/completion` sampled (temperature above 0), which reads the
//!   logits row every token: one seed twice gives the same ids, and `top_k` 1
//!   gives the greedy ids;
//! - the positions the server serves: `/props`' `n_ctx` is the server's
//!   default context (`workstation::CTX_MAX`); a prompt of that many ids,
//!   which leaves no position for the token it asks for, is a 400
//!   (`exceed_context_size_error`, naming it) refused before the engine, and
//!   the server stays up;
//! - `/v1/chat/completions` of one user turn at temperature 0: the streamed
//!   deltas concatenate to the non-streamed content, and the stream ends with
//!   `data: [DONE]`;
//! - `/tokenize` of `--prompt` is `--ids`;
//! - the same `/completion` after those requests gives the same ids (the
//!   engine's reset between requests leaves nothing behind), and once more
//!   right after itself it keeps at most `n − 1` positions of its `n` and at
//!   least `n − 1` in whole compression groups (a multiple of every ratio),
//!   with the same ids;
//! - prefix reuse (`cache_prompt`, default true): `--ids` then `--ids` plus
//!   its 8 greedy ids (`ignore_eos`) keeps every cached position (`timings.cache_n`), and a
//!   prompt that leaves the last cached position out takes that one back;
//!   each gives the greedy ids of the same prompt with `cache_prompt: false`,
//!   and `timings.prompt_n` is the ids evaluated, the prompt less `cache_n`;
//! - conversations at temperature 0, sent as a client sends them (each turn
//!   the earlier messages, the assistant's `content` as answered, its reasoning
//!   left out, and the next user message), each turn then run again from a
//!   reset cache as a greedy `/completion` of its rendered ids: two turns;
//!   two turns thinking on; four turns; two turns thinking on with a reply
//!   longer than the raw window (`ignore_eos`); a second conversation on the
//!   first's long system prompt, diverging inside the first's prompt call; and
//!   one conversation's second turn after another conversation's first. What a
//!   turn keeps is the engine's answer, which this gate does not model: every
//!   turn's reply is the fresh run's text and length, its counts add up to the
//!   prompt (`usage.prompt_tokens`, `cached_tokens` = `timings.cache_n`), and
//!   `cache_n` is at most the longest prefix the prompt shares with what any
//!   earlier request left. Where the template makes a turn share the previous
//!   prompt (thinking off) or all of it but its `<think>` (thinking on), that
//!   is checked, and the turn keeps at least that prefix — all of it when it
//!   is everything the most recent request left (no cut), else in whole
//!   compression groups: its cut lands in a reply or at a prompt call's end,
//!   outside every prompt call's hole. The other two print what they kept.
//! - the think-span budget (`reasoning_budget`): a thinking-on generation
//!   with the budget at 8 carries the close id as the 9th, and the
//!   continuation conditions on it (its ids after the close are a fresh
//!   run's fed the close as an ordinary prompt id): the close id really
//!   enters the engine's context. The ids before the budget, the reply's
//!   bounded-and-closed shape and the follow-up turn's prefix are
//!   `crates/serve`'s mock tests' to pin;
//! - two conversations interleaved (the prompt cache): `--ids` and its
//!   greedy ids, then the ids reversed, then `--ids` plus its greedy ids
//!   again keeps every position the first left (`cache_n` = its length − 1)
//!   and gives the ids of `cache_prompt: false`;
//! - a shared system prompt: a chat whose system prompt and user message are
//!   long enough that a single prompt call would leave the system prompt in
//!   its hole, then a chat with the same system prompt and another message
//!   keeps the system prompt (`cache_n` is the first user marker's position,
//!   or one less for the compressor's parity) and answers as with
//!   `cache_prompt: false`.
//!
//! Then the server is killed by the handle this binary spawned it with and
//! waited for. A server asked for [`SLOTS_OVER`] slots of the default context
//! (`<dir>/slots-over`), whose plan counts that many whole sequences, ends
//! before it loads (s4): no `plan` record, and the plan's own refusal names
//! the gate card and its KV term (the card's floor, or its usable bytes past
//! the plan's total) — the same context as one slot is the server above.
//! Then a second server of two resident sequences (`--parallel 2`, the
//! default context each, the residency off as the gate placement runs it,
//! the slot actions on, `BLOOMERY_STEP_STATS=1` set on the process — the
//! lever moves no computation in the round path and, on V4.1, also turns on
//! the engine's engram fill statistics (clock reads and page-classifying
//! syscalls), which change no id and no pass count but cost host time a
//! step, so a server run under it is never a timing reading) is started
//! into `<dir>/slots`:
//!
//! - its `parallel` line names the rule, the two slots, the default context
//!   a slot and their total; its `slots` line two sequences made of two
//!   planned; `/props`' `n_ctx` the default context;
//! - (s5) two greedy requests (`ignore_eos`, `cache_prompt: false`), each
//!   alone, then both posted at once: each request's ids are its alone ids,
//!   and the decode rounds over the pair carry at least [`SLOTS_BUSY`] busy
//!   slots each (`/metrics`' `n_busy_slots_per_decode` and `n_decode_total`,
//!   before and after) with no swap counted — the slots flow together, not
//!   in turns;
//! - the rounds of the pair posted at once
//!   ([`slots_rounds_run_one_pass`]): every `slots round` record the seat
//!   printed, that window's and (s5)'s together window's, holds `cmd=step`,
//!   `slots=2`, `rows=2` and `passes=1` — one pass of the busy rows, the
//!   body's two-row bound (FAIL-first: the seat left on the fallback loop
//!   prints `passes=rows`) — and the rows the clause's own window carries,
//!   with the solo rounds beside them (a select and a `next`, no record),
//!   which `/metrics`' `n_decode_total` deltas count less the two requests'
//!   prompt calls, booked at their end, sum to the tokens the two requests
//!   decoded beyond their prompts' first;
//! - (s2) `--ids` and a prompt sharing nothing with it, each on its own
//!   slot (`cache_prompt`); the second's slot erased; then `--ids` plus its
//!   greedy ids takes the first's slot and keeps all but its last id, and
//!   its ids are those of `cache_prompt: false`;
//! - (p3) a slot file, on a prompt sent as ids with no BOS whose first id
//!   starts no earlier request (so no slot and no cached state shares a
//!   prefix with it): the prompt on a slot (`cache_prompt`), the slot saved
//!   to a file, erased and restored from it; then the prompt plus its
//!   greedy ids keeps all but its last id on that slot, with the ids of
//!   `cache_prompt: false`;
//! - (p2) park and resume through the prompt cache: both slots erased (slot
//!   0 first, so it is the least recently used), then three conversations
//!   that share no prefix. A and B are each answered on a slot of its own;
//!   C takes the least recently used, A's, and a `cache save` record names
//!   A's positions. B's continuation keeps its slot (no `cache load`); then
//!   A's takes the least recently used slot, C's, and a `cache load` record
//!   of A's positions keeps all of them: the continuation keeps all but its
//!   last id. Each continuation's ids are those of `cache_prompt: false`. B
//!   runs first: A's continuation first would take B's slot, the least
//!   recently used, and park B's state instead.
//!
//! That server is killed and waited for the same way. A second server of
//! the same arguments and directory (`<dir>/slots-files`) restores the (p3)
//! file into the other slot, and the continuation keeps all but its last id
//! with the first server's ids of `cache_prompt: false`: the round trip
//! across two processes. A third, of half the default context
//! (`<dir>/slots-files-ctx`), refuses the file by name (p4): a 400
//! (`invalid_request_error`) naming both contexts; it then answers `/health`
//! and a request. Each starts after the one before it is gone: one load on
//! the gate card at a time. Last, the same
//! `--parallel 2` under `BLOOMERY_DRAFT=dspark` (set on that process alone)
//! ends before it listens, naming both: the draft's window is one
//! sequence's. Logs and the raw stream go to `--dir`, the first
//! `/completion`'s ids to `completion.ids` in it, and the greedy ids of the
//! probe — [`DRAFT_PREDICT`] ids after a long document, a prompt whose
//! continuation both keeps and rejects DSpark proposals — to `probe.ids`,
//! and the first sampled request's ids to `sampled.ids`.
//!
//! With `--slots` (no draft), the gate runs the slots clauses above alone —
//! too many slots refused, the two-slot server and the slot-file servers
//! after it, the DSpark refusal — and nothing else.
//!
//! With `--need-path` (no draft), the gate starts the first server above and
//! checks (p4)'s save without `--slot-save-path` on it, nothing else: a run
//! of that clause that loads the model once.
//!
//! With `--plain`, under `BLOOMERY_DRAFT=dspark` (refused otherwise), the gate
//! checks the server's DSpark draft instead, and nothing above:
//!
//! - `/props`' `engine.draft`: kind `dspark`, the file `$BLOOMERY_DSPARK_MODEL`
//!   names (its base name as `model`, the path as `path`), `n_max` 1, and the
//!   device `GPU<n>` of the card `BLOOMERY_DSPARK_CARD` names, whose placement
//!   row holds the class `draft`; the target's card and host rows are still
//!   the plan's bytes;
//! - `/completion` of `--prompt` at temperature 0: its ids are the plain run's
//!   (`completion.ids` in `--plain`, the plain run's directory) and
//!   `generate_ds41`'s, and its `timings` carry `draft_n` above 0 and
//!   `draft_n_accepted` at most that; the probe's ids are the plain run's
//!   (`probe.ids`), over passes that kept a proposal and passes that did not;
//! - a request that samples or sets `ignore_eos` is served by plain steps
//!   and drafts nothing (no `draft_n`): the plain run's sampled request
//!   (`sampled.ids` in `--plain`) with the plain server's ids, the
//!   `ignore_eos` one with all its ids; the server serves on;
//! - prefix reuse under the draft, each request's ids those of the same prompt
//!   with `cache_prompt: false`: a continuation of what the cache holds keeps
//!   all but its last id (the draft follows); a long prompt cut back inside
//!   its generated ids keeps no more than its shared prefix less the draft's
//!   window, and more than none; a long conversation put back from the prompt
//!   cache after another one keeps the same bound;
//! - `/metrics`' `spec_decode_num_draft_tokens_total` is above 0.
//!
//! With `--place bp`, under `BLOOMERY_DRAFT=dspark` (refused otherwise, and
//! beside `--plain`), the server runs plan (b′) — plan (a) on the A6000, the
//! 3090 an expert tier holding the draft's reserve — with the DSpark draft,
//! started as `--port 0 --place bp --parallel 1`, and the gate checks it against its own
//! plan (b′) of the file (the draft's reserve from its header and the tier's
//! prompt-batch bytes from the file's, as the server makes them) and nothing
//! above:
//!
//! - `/props`' placement names three devices: the A6000 (`GPU<n>`, every
//!   layer, the plan's stage-card bytes), the tier card (the 3090's
//!   `GPU<n>`, no `layers`, its `experts` class the plan's tier bytes and a
//!   `draft` class above 0 and at most the reserve) and the host (`CPU`, the
//!   plan's host bytes); `engine.draft` is the DSpark draft on the tier
//!   card's device — the plan's tier card, not the draft card rule the
//!   server applies; `args` are the ones this gate passed;
//! - `/completion` of `--prompt` at temperature 0: its ids are `--gen`'s
//!   `tokens` line (`generate_ds41 --place bp` under the same draft: the same
//!   plan, the same card sets, the same prompt feed — both read
//!   `BLOOMERY_PREFILL` from this environment), and its `timings` carry
//!   `draft_n` above 0 and `draft_n_accepted` at most that.
//!
//! The server inherits this binary's environment, so the levers it acts on
//! are the server's (`serve_levers::ACTS_ON`): one the server would refuse is
//! refused here, at `main`, before the server starts.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_ds41_serve: built without the `deepseek41` feature; see `just weekly-gpu-ds41-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ds41_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::mpsc::{self, Receiver};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use bloomery_gpu_gates::bind::nvidia_smi_index;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        Served, curl, ids_of, json_of, metric, parse_ids, server_log,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::{self, PlanInputs};
    use model::arch::deepseek41::plan::Planner;
    use model::placement::{Card, PlanLevers, workstation};
    use refset::arch::deepseek41::VERIFIED_POSITIONS;
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    use crate::dspark;

    const USAGE: &str = "usage: gate_ds41_serve --gen <generate_ds41 log> --prompt <text> --ids <a,b,…> --dir <out> [--plain <the plain run's --dir> | --place bp | --slots | --need-path]";
    /// Where the first `/completion`'s ids go in `--dir`, for the draft run.
    const COMPLETION_IDS: &str = "completion.ids";
    /// Where the probe's ids go in `--dir`, for the draft run.
    const PROBE_IDS: &str = "probe.ids";
    /// Where the first sampled request's ids go in `--dir`, for the draft run.
    const SAMPLED_IDS: &str = "sampled.ids";
    /// The draft run's greedy requests' length: long enough for several passes.
    const DRAFT_PREDICT: usize = 32;
    /// The server's arguments after its path; `/props` must echo them. The
    /// plain engine is pinned (`--parallel 1`): this gate's body holds the
    /// one-slot path's prompt-cache clauses, and the draft runs below would
    /// refuse the default's two slots by name; the slots clause's server
    /// runs two on its own.
    const SERVER_ARGS: [&str; 8] = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "gate",
        "--parallel",
        "1",
    ];
    /// The server's arguments under `--place bp`, the plain engine pinned as
    /// [`SERVER_ARGS`]'s (the run drafts).
    const BP_SERVER_ARGS: [&str; 8] = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "bp",
        "--parallel",
        "1",
    ];
    // PIN(2026-10-05): the swap clause (two slots taking this seat in turns through
    // serve::SwapEngine) is replaced by the slots clause below; SwapEngine stays only as the
    // fallback for bodies without Slots, and crates/serve's mock suite owns its turns
    // (tests/slots: hw_live_requests_take_turns_…, hw_slots_that_take_turns_keep_the_draft).
    /// The slots clause's server: two resident sequences of the default
    /// context each (the slot actions' directory added at the spawn).
    const SLOTS_SERVER_ARGS: [&str; 8] = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "gate",
        "--parallel",
        "2",
    ];
    /// The slots clause's pair: each request this long, so nearly every
    /// round after the second prompt call carries both slots.
    const SLOTS_PREDICT: usize = 48;
    /// (s4) Slots of the default context past what the gate card holds: each
    /// one's KV is the plan's card KV of one sequence, about 110 MB at the
    /// default context, so this many need about 28 GB beside the dense trunk
    /// on a card of 24 GB.
    const SLOTS_OVER: usize = 256;
    /// (s5) Busy slots a decode round carries over the pair posted at once:
    /// both prompt calls run before their first rounds, a round or two of
    /// skew, so the rounds carry `2·P / (P + skew)` slots each, about 1.9 at
    /// [`SLOTS_PREDICT`]; turns or one slot a round carry 1.
    const SLOTS_BUSY: f64 = 1.5;
    /// (p3, p4) The slot file the slot-file clauses save and restore.
    const SLOT_FILE: &str = "slot.bin";
    /// (p3) The slot file's prompt, sent as ids with no BOS: its first word
    /// starts no earlier request ([`fresh_ids`]).
    const FILES_TEXT: &str =
        "Slot files keep a conversation across server processes. Name two reasons to save one.";
    /// (p2) The park/resume clause's three conversations, sent as ids with
    /// no BOS: each first word starts no other request ([`fresh_ids`]).
    const PARK_TEXTS: [&str; 3] = [
        "Alpha asks: name three rivers of Europe, one line each.",
        "Bravo asks: name three mountains of Asia, one line each.",
        "Charlie asks: name three deserts of Africa, one line each.",
    ];
    /// The load takes tens of seconds; the bound is the spec's 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    const N_PREDICT: usize = 16;
    /// Greedy ids of each prefix-reuse request, run past the end-of-generation id
    /// (`ignore_eos`) so every one of them is compared.
    const REUSE_PREDICT: usize = 8;
    const CHAT: &str = "What is the capital of France? Answer in one word.";
    /// The two-turn chat's messages and its answers' length.
    const TURN1: &str = "Name three primary colors.";
    const TURN2: &str = "Which of them is the color of the sky?";
    const TURN_PREDICT: usize = 24;
    /// A reply longer than the raw window (checked against the file's).
    const WINDOW_PREDICT: usize = 160;
    const TURN3: &str = "And the color of fresh grass?";
    const TURN4: &str = "Name one color that is not primary.";
    /// Another conversation's first message.
    const OTHER: &str = "Name three farm animals.";
    /// The long conversations' shared system prompt and first message: lines
    /// enough that the prompt call runs more than two windows past the prefix
    /// they share.
    const LONG_RULES: usize = 20;
    const LONG_LINES: usize = 30;
    const LONG_PREDICT: usize = 8;
    /// The shared-system-prompt clause's messages: a system prompt of some
    /// hundreds of ids, and a first user message long enough that one prompt
    /// call would leave the system prompt in its CED hole.
    const SYSTEM_LINE: &str = "You are a careful assistant. Answer briefly and cite nothing. ";
    const SYSTEM_REPEAT: usize = 24;
    const LONG_USER_LINE: &str = "Here is a line of a long document to read before the question. ";
    const LONG_USER_REPEAT: usize = 48;
    /// The sampled requests' temperature and seed.
    const SAMPLED_TEMPERATURE: f64 = 0.8;
    const SAMPLED_SEED: u64 = 7;

    /// `ask` in whole compression groups: rounded down to a multiple of every
    /// ratio. The least a request keeps of `ask` shared positions when its cut
    /// lands outside every prompt call's hole, since the compressor state rings
    /// keep whole groups: a floor, not the engine's rule, which may keep more.
    fn whole_groups(ask: usize, ratios: &[usize]) -> usize {
        (0..=ask)
            .rev()
            .find(|k| ratios.iter().all(|&r| r == 0 || k % r == 0))
            .unwrap_or(0)
    }

    /// The file's compression ratios and raw window, from its headers.
    struct Rules {
        ratios: Vec<usize>,
        window: usize,
    }

    fn file_rules() -> Result<Rules, GateError> {
        let path = workstation::model_v41();
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&split)?;
        let planner = Planner::from_file(&split, &hp, workstation::CTX_MAX)?;
        Ok(Rules {
            ratios: planner
                .stream_ratios()
                .iter()
                .map(|&r| r as usize)
                .collect(),
            window: hp.window,
        })
    }

    /// The positions the server serves at its default context: all of them.
    fn served_positions() -> Result<usize, GateError> {
        Ok(usize::try_from(workstation::CTX_MAX)?)
    }

    /// `/completion` of `prompt` sampled at [`SAMPLED_TEMPERATURE`] with
    /// `seed` and `top_k`, from a reset cache: its ids.
    fn sampled(
        url: &dyn Fn(&str) -> String,
        prompt: &str,
        seed: u64,
        top_k: u32,
    ) -> Result<Vec<u32>, GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": SAMPLED_TEMPERATURE,
            "top_k": top_k, "seed": seed, "return_tokens": true, "cache_prompt": false,
        });
        let (st, body) = curl(&url("/completion"), Some(&body), false)?;
        let v = json_of("/completion", st, &body)?;
        let ids = ids_of(&v["tokens"]);
        println!(
            "sampled seed {seed} top_k {top_k}: {ids:?} stop_type={}",
            v["stop_type"]
        );
        Ok(ids)
    }

    /// `/props`' `n_ctx` is [`served_positions`], a prompt of that many ids is
    /// a 400 naming it, and the server answers `/health` after it.
    fn position_limit(url: &dyn Fn(&str) -> String, id: u32) -> Result<bool, GateError> {
        let served = served_positions()?;
        let (st, body) = curl(&url("/props"), None, false)?;
        let n_ctx = json_of("/props", st, &body)?["default_generation_settings"]["n_ctx"].clone();
        let long = vec![id; served];
        let (st, body) = curl(
            &url("/completion"),
            Some(&json!({"prompt": long, "n_predict": 1, "temperature": 0})),
            false,
        )?;
        let refused: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let e = &refused["error"];
        println!(
            "positions served {served}: /props n_ctx {n_ctx}; a prompt of {served} ids: HTTP {st} {e}"
        );
        let (hst, hbody) = curl(&url("/health"), None, false)?;
        let mut ok = true;
        check(
            &mut ok,
            "props_n_ctx_is_the_served_positions",
            n_ctx == json!(served),
        );
        check(
            &mut ok,
            "prompt_of_the_served_positions_is_a_400",
            st == 400
                && e["type"] == "exceed_context_size_error"
                && e["message"]
                    .as_str()
                    .is_some_and(|m| m.contains(&served.to_string())),
        );
        check(
            &mut ok,
            "health_after_the_400",
            hst == 200 && hbody.contains("\"ok\""),
        );
        Ok(ok)
    }

    /// The length of the common prefix of `a` and `b`.
    fn common(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    struct Args {
        gen_log: PathBuf,
        prompt: String,
        ids: Vec<u32>,
        dir: PathBuf,
        /// The plain run's `--dir`: the draft run.
        plain: Option<PathBuf>,
        /// `--place bp`: the two-card run.
        bp: bool,
        /// `--slots`: the slots clauses alone.
        slots: bool,
        /// `--need-path`: the main server's slot save without
        /// `--slot-save-path` alone.
        need_path: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut gen_log, mut prompt, mut ids, mut dir) = (None, None, None, None);
        let (mut plain, mut bp, mut slots, mut need_path) = (None, false, false, false);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--slots" {
                slots = true;
                continue;
            }
            if flag == "--need-path" {
                need_path = true;
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--gen" => gen_log = Some(PathBuf::from(v)),
                "--prompt" => prompt = Some(v),
                "--ids" => ids = Some(parse_ids(&v)?),
                "--dir" => dir = Some(PathBuf::from(v)),
                "--plain" => plain = Some(PathBuf::from(v)),
                "--place" => match Place::parse(&v)? {
                    p if p == Place::Bp => bp = true,
                    p if p == Place::Gate => bp = false,
                    p if p == Place::A => {
                        return Err(format!(
                            "--place a: this gate runs the server on the gate card, or on both \
                             under bp: {USAGE}"
                        )
                        .into());
                    }
                    other => {
                        return Err(format!(
                            "--place {}: this gate runs the server under gate or bp: {USAGE}",
                            other.name()
                        )
                        .into());
                    }
                },
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if slots && need_path {
            return Err(format!("--slots and --need-path each run one part alone: {USAGE}").into());
        }
        match (gen_log, prompt, ids, dir) {
            (Some(gen_log), Some(prompt), Some(ids), Some(dir)) => Ok(Args {
                gen_log,
                prompt,
                ids,
                dir,
                plain,
                bp,
                slots,
                need_path,
            }),
            _ => Err(USAGE.into()),
        }
    }

    /// The `tokens [..]` line of a `generate_ds41` log.
    fn gen_tokens(log: &Path) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", log.display()))?;
        parse_ids(line)
    }

    /// The rendered ids of a chat's `messages` under the template's `kwargs`,
    /// through the server's own template and tokenizer.
    fn rendered(
        url: &dyn Fn(&str) -> String,
        messages: &Value,
        kwargs: &Value,
    ) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({"messages": messages, "chat_template_kwargs": kwargs})),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template gave no prompt")?
            .to_owned();
        let (st, body) = curl(&url("/tokenize"), Some(&json!({"content": text})), false)?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// A JSON count as a `usize`; `None` when it is not one.
    fn as_count(v: &Value) -> Option<usize> {
        v.as_u64().and_then(|c| usize::try_from(c).ok())
    }

    /// Every id sequence a request of the conversations left in the server's
    /// cache: its prompt and all its generated ids but the last. Whichever slot
    /// or saved state the server keeps them in, a request can keep no more than
    /// the longest prefix it shares with one of them.
    #[derive(Clone)]
    struct Ledger(Vec<Vec<u32>>);

    impl Ledger {
        fn book(&mut self, p: &[u32], generated: &[u32]) {
            let mut held = p.to_vec();
            held.extend(&generated[..generated.len().saturating_sub(1)]);
            self.0.push(held);
        }

        /// The prefix `p` shares with what the most recent request left, and
        /// the longest it shares with any; each at most `p.len() − 1`, since
        /// the last id is always evaluated.
        fn shares(&self, p: &[u32]) -> (usize, usize) {
            let top = p.len().saturating_sub(1);
            let last = self.0.last().map_or(0, |h| common(h, p));
            let any = self.0.iter().map(|h| common(h, p)).max().unwrap_or(0);
            (last.min(top), any.min(top))
        }
    }

    /// What the template makes a turn share with the most recent request, the
    /// previous turn of its conversation unless another conversation came between.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Shares {
        /// Nothing checked: a conversation's first turn.
        Any,
        /// All of the previous prompt: thinking off, the recorded turn renders
        /// as its reply was generated.
        Prompt,
        /// The previous prompt but its last id: thinking on, the generation
        /// prompt ends in `<think>` and the recorded turn in `</think>`.
        ToThink,
        /// More than a window, ending more than two windows before the
        /// previous step's prompt does: inside that prompt call.
        InsideCall,
        /// All of its conversation's previous prompt with some earlier request,
        /// less with the most recent one: another conversation came between.
        Elsewhere,
    }

    struct Step {
        conv: usize,
        user: String,
        shares: Shares,
        /// The cut lands in a reply or at a prompt call's end: the turn keeps
        /// at least what it shares with the most recent request, all of it
        /// when that is everything the request left, else in whole compression
        /// groups.
        floor: bool,
    }

    struct Script {
        name: &'static str,
        /// One system message per conversation.
        systems: Vec<String>,
        steps: Vec<Step>,
        /// `chat_template_kwargs`.
        kwargs: Value,
        max_tokens: usize,
        ignore_eos: bool,
        /// Every reply must run past the raw window.
        past_window: bool,
    }

    fn step(conv: usize, user: &str, shares: Shares, floor: bool) -> Step {
        Step {
            conv,
            user: user.to_owned(),
            shares,
            floor,
        }
    }

    /// The warm message is the fresh completion's `raw` text as the chat splits
    /// it: with thinking on, its reasoning, `</think>` and its content (all of
    /// it reasoning while the span is still open); with thinking off, its content.
    fn same_text(m: &Value, raw: &str, thinking: bool) -> bool {
        let content = m["content"].as_str().unwrap_or("");
        let reasoning = m["reasoning_content"].as_str().unwrap_or("");
        if thinking {
            raw == format!("{reasoning}{}{content}", serve::reasoning::THINK_CLOSE)
                || (content.is_empty() && raw == reasoning)
        } else {
            reasoning.is_empty() && raw == content
        }
    }

    /// Runs `s` (see the module header): every turn warm, as a client sends it,
    /// then every turn again as a greedy `/completion` of its rendered ids from
    /// a reset cache. `ledger` holds what the requests before left and gains
    /// what these leave.
    fn converse(
        url: &dyn Fn(&str) -> String,
        rules: &Rules,
        ledger: &mut Ledger,
        s: &Script,
    ) -> Result<bool, GateError> {
        let mut convs: Vec<Vec<Value>> = s
            .systems
            .iter()
            .map(|t| vec![json!({"role": "system", "content": t})])
            .collect();
        let mut warm: Vec<(Vec<u32>, Value)> = Vec::new();
        for st in &s.steps {
            let msgs = convs
                .get_mut(st.conv)
                .ok_or_else(|| format!("{}: no conversation {}", s.name, st.conv))?;
            msgs.push(json!({"role": "user", "content": st.user}));
            let messages = Value::Array(msgs.clone());
            let p = rendered(url, &messages, &s.kwargs)?;
            let body = json!({
                "messages": messages, "temperature": 0, "max_tokens": s.max_tokens,
                "ignore_eos": s.ignore_eos, "chat_template_kwargs": s.kwargs,
            });
            let (code, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
            let reply = json_of("/v1/chat/completions", code, &text)?;
            let content = reply["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_owned();
            msgs.push(json!({"role": "assistant", "content": content}));
            warm.push((p, reply));
        }
        let mut fresh: Vec<(Vec<u32>, Value)> = Vec::new();
        for (p, _) in &warm {
            let body = json!({
                "prompt": p, "n_predict": s.max_tokens, "temperature": 0,
                "ignore_eos": s.ignore_eos, "return_tokens": true, "cache_prompt": false,
            });
            let (code, text) = curl(&url("/completion"), Some(&body), false)?;
            let v = json_of("/completion", code, &text)?;
            fresh.push((ids_of(&v["tokens"]), v));
        }
        let thinking = s.kwargs["thinking"] == json!(true);
        let mut before = ledger.clone();
        let mut prev: Vec<Option<usize>> = vec![None; convs.len()];
        let mut recent: Option<usize> = None;
        let mut ok = true;
        for (i, (st, ((p, reply), (g, f)))) in
            s.steps.iter().zip(warm.iter().zip(&fresh)).enumerate()
        {
            let name = format!("{}_turn{}", s.name, i + 1);
            let (last, bound) = before.shares(p);
            let (t, u) = (&reply["timings"], &reply["usage"]);
            let m = &reply["choices"][0]["message"];
            let cache_n = as_count(&t["cache_n"]);
            let most_recent = before.0.last().map_or(0, Vec::len);
            // All of what the most recent request left needs no cut.
            let floor = if last == most_recent {
                last
            } else {
                whole_groups(last, &rules.ratios)
            };
            println!(
                "{} turn {}: {} rendered ids; the most recent request left {most_recent} ids and \
                 shares {last} with it, {bound} at most with any earlier one{}; cache_n={} \
                 prompt_n={} usage={}; fresh {} ids stop_type={}; message {m}",
                s.name,
                i + 1,
                p.len(),
                if st.floor {
                    format!(", keeps at least {floor}")
                } else {
                    String::new()
                },
                t["cache_n"],
                t["prompt_n"],
                u,
                g.len(),
                f["stop_type"],
            );
            check(
                &mut ok,
                &format!("{name}_counts"),
                cache_n
                    .zip(as_count(&t["prompt_n"]))
                    .is_some_and(|(c, n)| c + n == p.len())
                    && u["prompt_tokens"] == json!(p.len())
                    && u["prompt_tokens_details"]["cached_tokens"] == t["cache_n"],
            );
            check(
                &mut ok,
                &format!("{name}_is_fresh"),
                f["timings"]["cache_n"] == json!(0)
                    && u["completion_tokens"] == json!(g.len())
                    && same_text(m, f["content"].as_str().unwrap_or(""), thinking),
            );
            let before_p = prev[st.conv];
            let shares = match (st.shares, before_p) {
                (Shares::Any, _) => true,
                (Shares::Prompt, Some(n)) => last >= n,
                (Shares::ToThink, Some(n)) => last + 1 == n,
                (Shares::InsideCall, _) => recent.is_some_and(|n| {
                    last > rules.window && n.saturating_sub(last) > 2 * rules.window
                }),
                (Shares::Elsewhere, Some(n)) => bound >= n && last < n,
                (_, None) => false,
            };
            if st.shares != Shares::Any {
                check(&mut ok, &format!("{name}_shares"), shares);
            }
            check(
                &mut ok,
                &format!("{name}_cache_n"),
                cache_n.is_some_and(|c| c <= bound && (!st.floor || c >= floor)),
            );
            if s.past_window {
                check(
                    &mut ok,
                    &format!("{name}_reply_passes_the_window"),
                    g.len() > rules.window,
                );
            }
            before.book(p, g);
            prev[st.conv] = Some(p.len());
            recent = Some(p.len());
        }
        // The warm turns, then their fresh runs, which left the same ids when
        // the turns passed.
        for _ in 0..2 {
            for ((p, _), (g, _)) in warm.iter().zip(&fresh) {
                ledger.book(p, g);
            }
        }
        Ok(ok)
    }

    /// The conversations of the module header, in order, after whatever the
    /// gate ran before them: each conversation opens with a system message of
    /// its own, so an earlier request shares at most the first id,
    /// `<｜begin▁of▁sentence｜>`, with any of them.
    fn conversations(url: &dyn Fn(&str) -> String, rules: &Rules) -> Result<bool, GateError> {
        let (code, body) = curl(&url("/props"), None, false)?;
        let bos_text = json_of("/props", code, &body)?["bos_token"]
            .as_str()
            .ok_or("/props gave no bos_token")?
            .to_owned();
        let (code, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": bos_text})),
            false,
        )?;
        let bos = ids_of(&json_of("/tokenize", code, &body)?["tokens"]);
        let mut ledger = Ledger(vec![bos]);
        let rules_text: String = (1..=LONG_RULES)
            .map(|i| {
                format!("Rule {i}: reply in plain words and keep every answer under fifty words.\n")
            })
            .collect();
        let report: String = (1..=LONG_LINES)
            .map(|i| format!("Line {i} of the report: the shipment arrived on time, complete and undamaged.\n"))
            .collect();
        let long_user = format!("{report}Summarize the report in one sentence.");
        let off = json!({});
        let on = json!({"thinking": true});
        let scripts = [
            Script {
                name: "chat",
                systems: vec!["Conversation one: answer plainly.".to_owned()],
                steps: vec![
                    step(0, TURN1, Shares::Any, false),
                    step(0, TURN2, Shares::Prompt, true),
                ],
                kwargs: off.clone(),
                max_tokens: TURN_PREDICT,
                ignore_eos: false,
                past_window: false,
            },
            Script {
                name: "think",
                systems: vec!["Conversation two: think first, then answer.".to_owned()],
                steps: vec![
                    step(0, TURN1, Shares::Any, false),
                    step(0, TURN2, Shares::ToThink, true),
                ],
                kwargs: on.clone(),
                max_tokens: TURN_PREDICT,
                ignore_eos: false,
                past_window: false,
            },
            Script {
                name: "turns",
                systems: vec!["Conversation three: one short sentence per answer.".to_owned()],
                steps: vec![
                    step(0, TURN1, Shares::Any, false),
                    step(0, TURN2, Shares::Prompt, true),
                    step(0, TURN3, Shares::Prompt, true),
                    step(0, TURN4, Shares::Prompt, true),
                ],
                kwargs: off.clone(),
                max_tokens: TURN_PREDICT,
                ignore_eos: false,
                past_window: false,
            },
            Script {
                name: "window",
                systems: vec!["Conversation four: think at length.".to_owned()],
                steps: vec![
                    step(0, TURN1, Shares::Any, false),
                    step(0, TURN2, Shares::ToThink, true),
                ],
                kwargs: on,
                max_tokens: WINDOW_PREDICT,
                ignore_eos: true,
                past_window: true,
            },
            Script {
                name: "long",
                systems: vec![rules_text.clone(), rules_text],
                steps: vec![
                    step(0, &long_user, Shares::Any, false),
                    step(1, TURN1, Shares::InsideCall, false),
                ],
                kwargs: off.clone(),
                max_tokens: LONG_PREDICT,
                ignore_eos: false,
                past_window: false,
            },
            Script {
                name: "interleave",
                systems: vec![
                    "Conversation five: colors.".to_owned(),
                    "Conversation six: animals.".to_owned(),
                ],
                steps: vec![
                    step(0, TURN1, Shares::Any, false),
                    step(1, OTHER, Shares::Any, false),
                    step(0, TURN2, Shares::Elsewhere, false),
                ],
                kwargs: off,
                max_tokens: TURN_PREDICT,
                ignore_eos: false,
                past_window: false,
            },
        ];
        let mut ok = true;
        for s in &scripts {
            ok &= converse(url, rules, &mut ledger, s)?;
        }
        Ok(ok)
    }

    /// The think-span budget (module header): a thinking-on generation with
    /// `reasoning_budget` 8 carries the close id at the 9th position, and the
    /// continuation conditions on it — fed the run's ids through the close as
    /// an ordinary prompt, the model continues as the budgeted run did. The
    /// close at the budget is the clause's precondition, not a pin of its own:
    /// without it the continuation matches whatever the run did. The model's
    /// own ids before the budget, the bounded-and-closed shape of the reply,
    /// the count that includes the close, and the follow-up turn's prefix are
    /// `crates/serve`'s mock tests' to pin; this clause holds the one they
    /// cannot — the close id really enters the engine's context.
    fn budget(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
        const THINK_BUDGET: usize = 8;
        /// The budgeted generation's length.
        const THINK_PREDICT: usize = 16;
        let on = json!({"thinking": true});
        let one = json!([{"role": "user", "content": CHAT}]);
        let p = rendered(url, &one, &on)?;
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": serve::reasoning::THINK_CLOSE})),
            false,
        )?;
        let close = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        // A run's ids, from a reset cache.
        let run = |prompt: &[u32], budget: Value, n: usize| -> Result<Vec<u32>, GateError> {
            let body = json!({
                "prompt": prompt, "n_predict": n, "temperature": 0, "ignore_eos": true,
                "return_tokens": true, "cache_prompt": false, "reasoning_budget": budget,
            });
            let (st, text) = curl(&url("/completion"), Some(&body), false)?;
            Ok(ids_of(&json_of("/completion", st, &text)?["tokens"]))
        };
        let capped = run(&p, json!(THINK_BUDGET), THINK_PREDICT)?;
        let through = THINK_BUDGET + close.len();
        let closed = capped.len() == THINK_PREDICT && capped[THINK_BUDGET..through] == close[..];
        // Fed the close as an ordinary prompt id, the model continues as the
        // capped run did: the engine's context really carries the close.
        let after = if closed {
            let mut fed = p.clone();
            fed.extend(&capped[..through]);
            run(&fed, Value::Null, THINK_PREDICT - through)?
        } else {
            Vec::new()
        };
        println!(
            "think budget {THINK_BUDGET}: capped {capped:?}, close {close:?} at {THINK_BUDGET}: \
             {closed}, fed through it {after:?}"
        );
        let mut ok = true;
        check(
            &mut ok,
            "budget_continues_on_the_close",
            closed && after == capped[through..],
        );
        Ok(ok)
    }

    /// One greedy `/completion` of `prompt` run past the end-of-generation id:
    /// its ids and `timings.cache_n`.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        prompt: &[u32],
        cache: bool,
    ) -> Result<(Vec<u32>, Value), GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": REUSE_PREDICT, "temperature": 0,
            "ignore_eos": true, "return_tokens": true, "cache_prompt": cache,
        });
        let (st, body) = curl(&url("/completion"), Some(&body), false)?;
        let v = json_of("/completion", st, &body)?;
        let ids = ids_of(&v["tokens"]);
        println!(
            "greedy prompt {} ids cache_prompt={cache}: cache_n={} tokens {ids:?}",
            prompt.len(),
            v["timings"]["cache_n"]
        );
        Ok((ids, v["timings"]["cache_n"].clone()))
    }

    /// Two conversations interleaved (module header).
    fn interleaved(url: &dyn Fn(&str) -> String, ids: &[u32]) -> Result<bool, GateError> {
        let (g1, _) = greedy(url, ids, true)?;
        let other: Vec<u32> = ids.iter().rev().copied().collect();
        greedy(url, &other, true)?;
        let cont: Vec<u32> = ids.iter().chain(&g1).copied().collect();
        let (g2, c2) = greedy(url, &cont, true)?;
        let (g3, c3) = greedy(url, &cont, false)?;
        let mut ok = true;
        check(
            &mut ok,
            "interleaved_keeps_the_first_conversation",
            c2 == json!(cont.len() - 1),
        );
        check(
            &mut ok,
            "interleaved_ids_are_fresh",
            g2 == g3 && c3 == json!(0),
        );
        Ok(ok)
    }

    /// A shared system prompt (module header).
    fn shared_system(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
        let system = SYSTEM_LINE.repeat(SYSTEM_REPEAT);
        let first = LONG_USER_LINE.repeat(LONG_USER_REPEAT);
        let one =
            json!([{"role": "system", "content": system}, {"role": "user", "content": first}]);
        let two =
            json!([{"role": "system", "content": system}, {"role": "user", "content": TURN2}]);
        // The chat requests below send no template kwargs; render them the same way.
        let none = json!({});
        let (p1, p2) = (rendered(url, &one, &none)?, rendered(url, &two, &none)?);
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": "<｜User｜>"})),
            false,
        )?;
        let marker = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        let mark = p1
            .iter()
            .position(|id| marker.as_slice() == [*id])
            .ok_or("the rendered chat holds no user marker")?;
        let chat = |messages: &Value, cache: bool| -> Result<Value, GateError> {
            let body = json!({
                "messages": messages, "temperature": 0, "max_tokens": TURN_PREDICT,
                "cache_prompt": cache,
            });
            let (st, body) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
            json_of("/v1/chat/completions", st, &body)
        };
        chat(&one, true)?;
        let warm = chat(&two, true)?;
        let fresh = chat(&two, false)?;
        let kept = warm["timings"]["cache_n"].as_u64().unwrap_or(0) as usize;
        println!(
            "shared system prompt: turn 1 {} ids, turn 2 {} ids, first user marker at {mark} \
             (shared {}): cache_n={kept}",
            p1.len(),
            p2.len(),
            common(&p1, &p2)
        );
        let mut ok = true;
        check(
            &mut ok,
            "shared_system_prompt_kept",
            common(&p1, &p2) > mark && (mark - 1..=mark).contains(&kept),
        );
        check(
            &mut ok,
            "shared_system_prompt_answer_is_fresh",
            warm["choices"][0]["message"] == fresh["choices"][0]["message"]
                && fresh["timings"]["cache_n"] == json!(0),
        );
        Ok(ok)
    }

    /// One greedy `/completion` body of the slots clause's pair.
    fn pair_body(prompt: &str) -> Value {
        json!({
            "prompt": prompt, "n_predict": SLOTS_PREDICT, "temperature": 0, "ignore_eos": true,
            "return_tokens": true, "cache_prompt": false,
        })
    }

    /// `/metrics`' decode rounds and the busy slots they carried in all; a
    /// server that counts neither is refused by name.
    fn rounds(url: &dyn Fn(&str) -> String) -> Result<(f64, f64), GateError> {
        let read = |name: &str| -> Result<f64, GateError> {
            metric(url, name)?.ok_or_else(|| format!("/metrics carries no llamacpp:{name}").into())
        };
        let decodes = read("n_decode_total")?;
        Ok((decodes, read("n_busy_slots_per_decode")? * decodes))
    }

    /// One greedy `/completion` of `prompt` ([`REUSE_PREDICT`] ids, run past
    /// the end-of-generation id): the slot that served it, its ids and
    /// `timings.cache_n`.
    fn on_slot(
        url: &dyn Fn(&str) -> String,
        prompt: &[u32],
        cache: bool,
    ) -> Result<(u64, Vec<u32>, Value), GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": REUSE_PREDICT, "temperature": 0,
            "ignore_eos": true, "return_tokens": true, "cache_prompt": cache,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        let v = json_of("/completion", st, &text)?;
        let slot = v["id_slot"].as_u64().unwrap_or(u64::MAX);
        let ids = ids_of(&v["tokens"]);
        println!(
            "slot {slot}: prompt {} ids cache_prompt={cache}: cache_n={} tokens {ids:?}",
            prompt.len(),
            v["timings"]["cache_n"]
        );
        Ok((slot, ids, v["timings"]["cache_n"].clone()))
    }

    /// (s4) [`SLOTS_OVER`] slots of the default context, which the gate
    /// card cannot hold, refused before the load (module header), into
    /// `<dir>/slots-over`.
    fn over_refused(a: &Args) -> Result<bool, GateError> {
        let dir = a.dir.join("slots-over");
        std::fs::create_dir_all(&dir)?;
        let n = SLOTS_OVER.to_string();
        let args = [
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--place",
            "gate",
            "--parallel",
            &n,
        ];
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&args, &dir)?;
        let refused = match served.address(&err_log, POLLS, POLL) {
            Ok(addr) => format!("listening on {addr}"),
            Err(e) => e.to_string(),
        };
        let planned = server_log(&err_log, record::BLOOMERY_SERVE_DS41)?
            .first(&record::PLAN)?
            .is_some();
        // The card the server's `--place gate` resolves to, as it resolves it.
        let card = Place::Gate
            .on_host()?
            .card_specs()?
            .first()
            .map(|c| c.name)
            .ok_or("--place gate resolved to no card")?;
        println!("slots over the card {card}: planned {planned}; {refused}");
        let mut ok = true;
        check(
            &mut ok,
            "slots_over_the_card_refused_before_the_load",
            refused.contains("before listening")
                && refused.contains(&format!("card {card}: "))
                && refused.contains("KV")
                && !planned,
        );
        Ok(ok)
    }

    /// The slots clause (module header) on a server of two resident slots
    /// started into `<dir>/slots`, after too many refused: the context a
    /// slot it names, (s5) the pair served together, (s2) an erase of one
    /// slot leaving the other's continuation, (p3) a slot file's round trip
    /// and (p2) park and resume through the prompt cache; then (p3) the
    /// file in a second process and (p4) refused by a third of another
    /// context.
    fn slots(a: &Args) -> Result<bool, GateError> {
        let mut ok = over_refused(a)?;
        let dir = a.dir.join("slots");
        let save = dir.join("save");
        std::fs::create_dir_all(&save)?;
        let save = save
            .to_str()
            .ok_or_else(|| format!("{}: not UTF-8", save.display()))?
            .to_owned();
        let mut args = SLOTS_SERVER_ARGS.to_vec();
        args.extend(["--slot-save-path", save.as_str()]);
        // The rounds clause counts the seat's rounds: the lever set on this
        // process alone, so the servers below run without it.
        let mut cmd = Command::new(Served::exe()?);
        cmd.env(bloomery_levers::STEP_STATS, "1");
        let mut served = Served::spawn_cmd(cmd, &args, &dir)?;
        println!("slots server pid {}", served.child.id());
        let err_log = dir.join("server.err");
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        ok &= slots_ctx(&url, &err_log)?;
        ok &= slots_together(&url, a)?;
        ok &= slots_rounds_run_one_pass(&url, &err_log, a)?;
        ok &= slots_actions(&url, a)?;
        let mut taken = taken_firsts(&url, a)?;
        let file = slots_files_save(&url, &mut taken)?;
        ok &= slots_park_resume(&url, &err_log, &mut taken)?;
        println!("slots server stopped: {}", served.stop()?);
        // One V4.1 load on the gate card at a time: the processes below start
        // after the slots server is gone.
        ok &= slots_files_round_trip(a, &save, &file)?;
        ok &= slots_files_refused(a, &save)?;
        Ok(ok)
    }

    /// The first ids of every request the slots server ran before the
    /// slot-file and park/resume clauses: the BOS a text prompt starts with,
    /// and the first ids of `--ids` and of `--ids` reversed.
    fn taken_firsts(url: &dyn Fn(&str) -> String, a: &Args) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": "", "add_special": true})),
            false,
        )?;
        let mut taken = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        taken.extend(a.ids.first().into_iter().chain(a.ids.last()));
        Ok(taken)
    }

    /// `text`'s ids through the server's tokenizer, no BOS, its first id
    /// then added to `taken`. A prompt whose first id starts no earlier
    /// request shares no prefix with any slot or cached state, so it runs
    /// from nothing and nothing but its own state can serve it again;
    /// refused by name when its first id is among `taken`.
    fn fresh_ids(
        url: &dyn Fn(&str) -> String,
        text: &str,
        taken: &mut Vec<u32>,
    ) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(&url("/tokenize"), Some(&json!({"content": text})), false)?;
        let ids = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        match ids.first() {
            Some(&first) if !taken.contains(&first) => {
                taken.push(first);
                Ok(ids)
            }
            first => Err(format!(
                "the prompt {text:?} starts with id {first:?}, which an earlier request of the \
                 slots server started with (taken {taken:?}): it would share a prefix"
            )
            .into()),
        }
    }

    /// (p3) The slots server's half of the slot-file round trip, and what it
    /// leaves for the second process.
    struct SlotFile {
        /// The slot the file was saved from.
        slot: u64,
        /// The prompt plus its greedy ids.
        cont: Vec<u32>,
        /// The continuation's ids with `cache_prompt: false`.
        fresh: Vec<u32>,
        /// The positions the file holds.
        saved: u64,
        /// Whether this half held.
        held: bool,
    }

    /// A slot action's answer, printed: its status and its JSON (`Null`
    /// when the body is not JSON).
    fn slot_action(
        url: &dyn Fn(&str) -> String,
        slot: u64,
        action: &str,
        body: &Value,
    ) -> Result<(u16, Value), GateError> {
        let (st, text) = curl(
            &url(&format!("/slots/{slot}?action={action}")),
            Some(body),
            false,
        )?;
        println!("slots {action} of slot {slot}: HTTP {st}: {text}");
        Ok((st, serde_json::from_str(&text).unwrap_or(Value::Null)))
    }

    // PIN(2026-10-05): slots_save_refused_by_name (a save answered 501) is replaced by the
    // slot-file round trip: the seat now serves save and restore (triage slotsnap, in the user's
    // 0.2.1 scope), so the contract changed from refused to served.
    /// (p3) The slots server's half (module header): the prompt on a slot,
    /// its file saved, the slot erased, the file restored, then the prompt
    /// plus its greedy ids kept but its last id, with the ids of
    /// `cache_prompt: false`.
    fn slots_files_save(
        url: &dyn Fn(&str) -> String,
        taken: &mut Vec<u32>,
    ) -> Result<SlotFile, GateError> {
        let p = fresh_ids(url, FILES_TEXT, taken)?;
        let (slot, g, c0) = on_slot(url, &p, true)?;
        if c0 != json!(0) {
            return Err(
                format!("the slot-file prompt kept {c0} positions: it is not fresh").into(),
            );
        }
        let file = json!({"filename": SLOT_FILE});
        let (st_save, saved) = slot_action(url, slot, "save", &file)?;
        let (st_erase, _) = slot_action(url, slot, "erase", &json!({}))?;
        let (st_restore, restored) = slot_action(url, slot, "restore", &file)?;
        let cont: Vec<u32> = p.iter().chain(&g).copied().collect();
        let (sc, gc, cc) = on_slot(url, &cont, true)?;
        let (_, gf, cf) = on_slot(url, &cont, false)?;
        let positions = (p.len() + REUSE_PREDICT - 1) as u64;
        println!(
            "slot file: {} positions saved from slot {slot} in {} bytes, {} restored from {} bytes; \
             the continuation on slot {sc}: cache_n={cc}",
            saved["n_saved"], saved["n_written"], restored["n_restored"], restored["n_read"]
        );
        let held = g.len() == REUSE_PREDICT
            && st_save == 200
            && saved["n_saved"] == json!(positions)
            && st_erase == 200
            && st_restore == 200
            && restored["n_restored"] == json!(positions)
            && restored["n_read"] == saved["n_written"]
            && sc == slot
            && cc == json!(cont.len() - 1)
            && gc == gf
            && cf == json!(0);
        Ok(SlotFile {
            slot,
            cont,
            fresh: gf,
            saved: positions,
            held,
        })
    }

    /// (p3) The second process (module header), into `<dir>/slots-files`:
    /// the file restored into the other slot, the continuation's ids and
    /// cache those of the first; the clause's one verdict, both halves.
    fn slots_files_round_trip(a: &Args, save: &str, file: &SlotFile) -> Result<bool, GateError> {
        let dir = a.dir.join("slots-files");
        std::fs::create_dir_all(&dir)?;
        let mut args = SLOTS_SERVER_ARGS.to_vec();
        args.extend(["--slot-save-path", save]);
        let mut served = Served::spawn(&args, &dir)?;
        println!("slot-file server pid {}", served.child.id());
        let addr = served.address(&dir.join("server.err"), POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let into = u64::from(file.slot == 0);
        let (st, restored) = slot_action(&url, into, "restore", &json!({"filename": SLOT_FILE}))?;
        let (sc, gc, cc) = on_slot(&url, &file.cont, true)?;
        println!("slot-file server stopped: {}", served.stop()?);
        let mut ok = true;
        check(
            &mut ok,
            "slots_files_round_trip",
            file.held
                && st == 200
                && restored["n_restored"] == json!(file.saved)
                && sc == into
                && cc == json!(file.cont.len() - 1)
                && gc == file.fresh,
        );
        Ok(ok)
    }

    /// (p4) The third process (module header), into `<dir>/slots-files-ctx`:
    /// the file refused by a server of another context, which serves on. Its
    /// probes are answers, not errors: a server the refusal ended reads as a
    /// failed check.
    fn slots_files_refused(a: &Args, save: &str) -> Result<bool, GateError> {
        let dir = a.dir.join("slots-files-ctx");
        std::fs::create_dir_all(&dir)?;
        let whole = served_positions()?;
        let ctx = (whole / 2).to_string();
        let mut args = SLOTS_SERVER_ARGS.to_vec();
        args.extend(["--ctx", ctx.as_str(), "--slot-save-path", save]);
        let mut served = Served::spawn(&args, &dir)?;
        println!("other-context server pid {}", served.child.id());
        let addr = served.address(&dir.join("server.err"), POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, refused) = slot_action(&url, 0, "restore", &json!({"filename": SLOT_FILE}))?;
        let e = &refused["error"];
        let names = e["message"]
            .as_str()
            .is_some_and(|m| m.contains(&whole.to_string()) && m.contains(&ctx));
        let probe = |path: &str, body: Option<&Value>| match curl(&url(path), body, false) {
            Ok((st, text)) => (st, text),
            Err(e) => (0, e.to_string()),
        };
        let (hst, health) = probe("/health", None);
        let body = json!({
            "prompt": a.ids, "n_predict": REUSE_PREDICT, "temperature": 0, "ignore_eos": true,
            "return_tokens": true, "cache_prompt": false,
        });
        let (rst, answer) = probe("/completion", Some(&body));
        let answered = rst == 200
            && serde_json::from_str::<Value>(&answer)
                .is_ok_and(|v| ids_of(&v["tokens"]).len() == REUSE_PREDICT);
        println!(
            "other context {ctx}: restore HTTP {st} {e}; health {hst} {health}; a request: HTTP \
             {rst}"
        );
        println!("other-context server stopped: {}", served.stop()?);
        let mut ok = true;
        check(
            &mut ok,
            "slots_files_refused",
            st == 400
                && e["type"] == "invalid_request_error"
                && names
                && hst == 200
                && health.contains("\"ok\"")
                && answered,
        );
        Ok(ok)
    }

    /// (p2) Park and resume through the prompt cache (module header), on
    /// the slots server: the `cache save` and `cache load` records read
    /// from its stderr, `err_log`. B's continuation runs before A's: resident
    /// slots break a tie of shared prefixes by the least recently used
    /// (`serve::sched`'s `best_slot`), so A's continuation first would take
    /// B's slot and park B's state rather than keep it.
    fn slots_park_resume(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        taken: &mut Vec<u32>,
    ) -> Result<bool, GateError> {
        let records = |kind: &record::Kind| -> Result<Vec<record::Fields>, GateError> {
            Ok(server_log(err_log, record::BLOOMERY_SERVE_DS41)?.all(kind)?)
        };
        let ram = server_log(err_log, record::BLOOMERY_SERVE_DS41)?
            .one(&record::CACHE_CONFIG)?
            .u64("ram")?;
        if ram == 0 {
            return Err(
                "the slots server's prompt cache is off (`cache ram=0`): the park/resume \
                 clause needs it"
                    .into(),
            );
        }
        // Erased in order, slot 0 first: both empty, slot 0 the least
        // recently used.
        for slot in [0, 1] {
            let (st, _) = slot_action(url, slot, "erase", &json!({}))?;
            if st != 200 {
                return Err(format!("the erase of slot {slot} answered HTTP {st}").into());
            }
        }
        let [ta, tb, tc] = PARK_TEXTS;
        let (pa, pb, pc) = (
            fresh_ids(url, ta, taken)?,
            fresh_ids(url, tb, taken)?,
            fresh_ids(url, tc, taken)?,
        );
        let held = |p: &[u32]| p.len() + REUSE_PREDICT - 1;
        let (sa, ga, _) = on_slot(url, &pa, true)?;
        let (sb, gb, _) = on_slot(url, &pb, true)?;
        let (saves, loads) = (
            records(&record::CACHE_SAVE)?.len(),
            records(&record::CACHE_LOAD)?.len(),
        );
        let (sc, _, _) = on_slot(url, &pc, true)?;
        let parked = records(&record::CACHE_SAVE)?.split_off(saves);
        let cb: Vec<u32> = pb.iter().chain(&gb).copied().collect();
        let (sb2, gb2, kb2) = on_slot(url, &cb, true)?;
        let b_loads = records(&record::CACHE_LOAD)?.len() - loads;
        let ca: Vec<u32> = pa.iter().chain(&ga).copied().collect();
        let (_, ga2, ka2) = on_slot(url, &ca, true)?;
        let resumed = records(&record::CACHE_LOAD)?.split_off(loads);
        let (_, ga3, _) = on_slot(url, &ca, false)?;
        let (_, gb3, _) = on_slot(url, &cb, false)?;
        let lines = |fs: &[record::Fields]| -> Vec<String> {
            fs.iter().map(|f| f.line().to_owned()).collect()
        };
        let (parked_lines, resumed_lines) = (lines(&parked), lines(&resumed));
        println!("park: A on slot {sa}, B on slot {sb}, C on slot {sc}; C's {parked_lines:?}");
        println!("resume: B's continuation on slot {sb2} ({b_loads} loads); A's {resumed_lines:?}");
        let names = |fs: &[record::Fields], want: &[(&str, usize)]| {
            fs.iter().any(|f| {
                want.iter()
                    .all(|&(k, v)| f.u64(k).ok() == u64::try_from(v).ok())
            })
        };
        let a_held = held(&pa);
        let mut ok = true;
        check(
            &mut ok,
            "slots_park_resume",
            sa != sb
                && sc == sa
                && names(&parked, &[("positions", a_held)])
                && sb2 == sb
                && b_loads == 0
                && kb2 == json!(cb.len() - 1)
                && gb2 == gb3
                && names(&resumed, &[("positions", a_held), ("kept", a_held)])
                && ka2 == json!(ca.len() - 1)
                && ga2 == ga3,
        );
        Ok(ok)
    }

    /// (p4) A slot save on the main server, started without
    /// `--slot-save-path`: the server's 501 naming the flag, and `/health`
    /// after it.
    fn slots_files_need_the_path(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
        let (st, text) = curl(
            &url("/slots/0?action=save"),
            Some(&json!({"filename": SLOT_FILE})),
            false,
        )?;
        let (hst, health) = curl(&url("/health"), None, false)?;
        println!("a slot save without --slot-save-path: HTTP {st}: {text}; health {hst} {health}");
        let mut ok = true;
        check(
            &mut ok,
            "slots_files_need_the_path",
            st == 501
                && text.contains("--slot-save-path")
                && hst == 200
                && health.contains("\"ok\""),
        );
        Ok(ok)
    }

    /// `--need-path`: the main server (`SERVER_ARGS`) and (p4)'s save
    /// without `--slot-save-path` on it, nothing else: one load.
    fn need_path(a: &Args) -> Result<(), GateError> {
        std::fs::create_dir_all(&a.dir)?;
        let mut served = Served::spawn(&SERVER_ARGS, &a.dir)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&a.dir.join("server.err"), POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let ok = slots_files_need_the_path(&url)?;
        println!("server stopped: {}", served.stop()?);
        if ok {
            println!("weekly-gpu-ds41-serve need-path: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// The context the slots server names: its `parallel` line (the rule,
    /// two slots, the default context a slot and their total), its `slots`
    /// line (two sequences made of two planned) and `/props`' `n_ctx`.
    fn slots_ctx(url: &dyn Fn(&str) -> String, err_log: &Path) -> Result<bool, GateError> {
        let text = std::fs::read_to_string(err_log)?;
        let line = |head: &str| {
            text.lines()
                .find(|l| l.starts_with(head))
                .unwrap_or("")
                .to_owned()
        };
        let (parallel, made) = (line("parallel rule="), line("slots made="));
        let (st, body) = curl(&url("/props"), None, false)?;
        let n_ctx = json_of("/props", st, &body)?["n_ctx"].as_u64();
        let whole = served_positions()?;
        println!("slots: {parallel}; {made}; /props n_ctx {n_ctx:?}");
        let mut ok = true;
        check(
            &mut ok,
            "slots_name_a_whole_context_each",
            parallel
                == format!(
                    "parallel rule=slots slots=2 slot_ctx={whole} total={} from=flag",
                    2 * whole
                )
                && made.starts_with("slots made=2 planned=2 ")
                && n_ctx == u64::try_from(whole).ok(),
        );
        Ok(ok)
    }

    /// (s5) The pair, each request alone, then both posted at once (module
    /// header).
    fn slots_together(url: &dyn Fn(&str) -> String, a: &Args) -> Result<bool, GateError> {
        let bodies = [pair_body(&a.prompt), pair_body(OTHER)];
        let mut alone = Vec::new();
        for body in &bodies {
            let (st, text) = curl(&url("/completion"), Some(body), false)?;
            alone.push(ids_of(&json_of("/completion", st, &text)?["tokens"]));
        }
        let (decode0, busy0) = rounds(url)?;
        type Answer = Result<(u16, String), String>;
        let post = |body: Value| -> Result<(JoinHandle<()>, Receiver<Answer>), GateError> {
            let u = url("/completion");
            let (tx, rx) = mpsc::channel();
            let (h, _) = spawn_helper("slots-request", Placement::Float, move || {
                let _ = tx.send(curl(&u, Some(&body), false).map_err(|e| e.to_string()));
            })
            .map_err(|e| format!("slots: {}", e.what()))?;
            Ok((h, rx))
        };
        let first = post(bodies[0].clone())?;
        let second = post(bodies[1].clone())?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
            let (st, text) = rx
                .recv()
                .map_err(|_| format!("slots: the {what} request's thread gave no answer"))??;
            together.push(ids_of(&json_of("/completion", st, &text)?["tokens"]));
        }
        let (decode1, busy1) = rounds(url)?;
        let swaps = metric(url, "swaps_total")?;
        let carried = (busy1 - busy0) / (decode1 - decode0);
        println!(
            "slots alone {} and {} ids, together {} and {} ids; decode rounds {decode0} -> \
             {decode1}, busy slots {busy0} -> {busy1} ({carried} a round); swaps {swaps:?}",
            alone[0].len(),
            alone[1].len(),
            together[0].len(),
            together[1].len()
        );
        let mut ok = true;
        check(
            &mut ok,
            "slots_together_ids_are_alone",
            alone.iter().all(|ids| ids.len() == SLOTS_PREDICT) && together == alone,
        );
        check(
            &mut ok,
            "slots_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && carried >= SLOTS_BUSY,
        );
        Ok(ok)
    }

    /// The rounds of the pair posted at once (module header): the two-slot
    /// server runs under `BLOOMERY_STEP_STATS=1`, so the seat counts its
    /// rounds of several slots. The (s5) pair is posted at once again — the
    /// clause's own window — and every `slots round` record the seat
    /// printed, this window's and (s5)'s together window's, holds one pass
    /// of the busy rows; the window's rows, with the solo rounds beside
    /// them, sum to the tokens the two requests decoded beyond their
    /// prompts' first. The lever moves no computation in the round path —
    /// it only prints the record — so (s5)'s together-equals-alone bits
    /// stand for the default server.
    fn slots_rounds_run_one_pass(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        a: &Args,
    ) -> Result<bool, GateError> {
        // (s5)'s together run printed its records before this point; the
        // window opens at what the log holds now, so its records are
        // exactly this pair's.
        let before = server_log(err_log, record::BLOOMERY_SERVE_DS41)?
            .all(&record::SLOTS_ROUND)?
            .len();
        let decode0 = rounds(url)?.0;
        type Answer = Result<(u16, String), String>;
        let post = |body: Value| -> Result<(JoinHandle<()>, Receiver<Answer>), GateError> {
            let u = url("/completion");
            let (tx, rx) = mpsc::channel();
            let (h, _) = spawn_helper("slots-rounds-request", Placement::Float, move || {
                let _ = tx.send(curl(&u, Some(&body), false).map_err(|e| e.to_string()));
            })
            .map_err(|e| format!("slots rounds: {}", e.what()))?;
            Ok((h, rx))
        };
        let bodies = [pair_body(&a.prompt), pair_body(OTHER)];
        let first = post(bodies[0].clone())?;
        let second = post(bodies[1].clone())?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("slots rounds: the {what} request's thread panicked"))?;
            let (st, text) = rx.recv().map_err(|_| {
                format!("slots rounds: the {what} request's thread gave no answer")
            })??;
            together.push(ids_of(&json_of("/completion", st, &text)?["tokens"]));
        }
        let decode1 = rounds(url)?.0;
        let rounds_log =
            server_log(err_log, record::BLOOMERY_SERVE_DS41)?.all(&record::SLOTS_ROUND)?;
        // The seat serves two slots, so a round of several is two rows run
        // as one pass: every record the seat printed, this window's and
        // (s5)'s.
        let every = !rounds_log.is_empty()
            && rounds_log.iter().all(|r| {
                r.word("cmd") == Ok("step")
                    && r.u64("slots") == Ok(2)
                    && r.u64("rows") == Ok(2)
                    && r.u64("passes") == Ok(1)
            });
        let window = &rounds_log[before.min(rounds_log.len())..];
        let sum: u64 = window.iter().map(|r| r.u64("rows").unwrap_or(0)).sum();
        // The solo rounds beside the window's records (a select and a
        // `next`, no record): the decode deltas less the records' rounds
        // less the two requests' prompt calls, booked at their end
        // (`worker.rs`'s `end_request`).
        let solo = decode1 - decode0 - window.len() as f64 - 2.0;
        let want = 2 * (SLOTS_PREDICT - 1) as u64;
        println!(
            "slots rounds: {} and {} ids together; {} record(s) in the window, {} in all; rows \
             sum {sum} + {solo} solo round(s) against the {want} decoded beyond the prompts' \
             first (decode {decode0} -> {decode1})",
            together[0].len(),
            together[1].len(),
            window.len(),
            rounds_log.len(),
        );
        let mut ok = true;
        check(&mut ok, "slots_rounds_run_one_pass", every);
        check(
            &mut ok,
            "slots_rounds_rows_sum_the_decoded_tokens",
            solo >= 0.0 && sum + solo as u64 == want,
        );
        Ok(ok)
    }

    /// (s2), an erase of one slot, on a pair that keeps its prompts (module
    /// header).
    fn slots_actions(url: &dyn Fn(&str) -> String, a: &Args) -> Result<bool, GateError> {
        let mut ok = true;
        let other: Vec<u32> = a.ids.iter().rev().copied().collect();
        let (sa, ga, _) = on_slot(url, &a.ids, true)?;
        let (sb, _, _) = on_slot(url, &other, true)?;
        // A POST, as llama-server's slot actions are: curl sends one with a body.
        let (st, text) = curl(
            &url(&format!("/slots/{sb}?action=erase")),
            Some(&json!({})),
            false,
        )?;
        let erased = json_of("/slots erase", st, &text)?;
        println!("slots erase of slot {sb}: {erased}");
        let cont: Vec<u32> = a.ids.iter().chain(&ga).copied().collect();
        let (sc, gc, cc) = on_slot(url, &cont, true)?;
        let (_, gf, cf) = on_slot(url, &cont, false)?;
        check(
            &mut ok,
            "slots_erase_empties_the_other_slot",
            sb != sa && erased["n_erased"].as_u64().is_some_and(|n| n > 0),
        );
        check(
            &mut ok,
            "slots_erase_leaves_the_slot_beside_it",
            sc == sa && cc == json!(cont.len() - 1) && gc == gf && cf == json!(0),
        );
        Ok(ok)
    }

    /// `--parallel 2` under the DSpark draft (module header), into
    /// `<dir>/slots-dspark`.
    fn slots_refuse_dspark(a: &Args) -> Result<bool, GateError> {
        let dir = a.dir.join("slots-dspark");
        std::fs::create_dir_all(&dir)?;
        let mut cmd = Command::new(Served::exe()?);
        cmd.env("BLOOMERY_DRAFT", "dspark");
        let mut served = Served::spawn_cmd(cmd, &SLOTS_SERVER_ARGS, &dir)?;
        let refused = match served.address(&dir.join("server.err"), POLLS, POLL) {
            Ok(addr) => format!("listening on {addr}"),
            Err(e) => e.to_string(),
        };
        println!("slots under the DSpark draft: {refused}");
        let mut ok = true;
        check(
            &mut ok,
            "slots_under_dspark_is_refused_by_name",
            refused.contains("--parallel 2 under BLOOMERY_DRAFT=dspark"),
        );
        Ok(ok)
    }

    /// `/props`' `engine` object (see the module header) against the plan of
    /// the file the server opens, made here from its headers the way the
    /// server makes it — under the placement's `levers` the server inherits
    /// and its stage card's free reading, taken from the server's own `plan`
    /// record ([`server_card_free`]); `argv` and `pid` are the process this
    /// gate spawned.
    fn props_engine(
        url: &dyn Fn(&str) -> String,
        argv: &[String],
        pid: u32,
        levers: &PlanLevers,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!("props engine {e}");
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|err| format!("open {}: {err}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let mut spec = workstation::RTX_3090;
        spec.free_bytes = Some(server_card_free(err_log)?);
        let machine = workstation::plan_on(spec, inputs.model.layers);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, levers)?;
        let (Some(card), Some(stage)) = (plan.cards.first(), machine.cards.first()) else {
            return Err("the gate's plan has no card".into());
        };
        let card_bytes = card.dense_bytes + card.expert_bytes;
        let host_bytes = plan.host.expert_bytes + plan.host.table_bytes;
        let kv: u64 = plan.cards.iter().map(|c| c.kv_bytes).sum();
        let mut file_bytes = 0u64;
        for i in 0..split.shard_count() {
            let shard = split.shard_path(i).ok_or("a shard without a path")?;
            file_bytes += std::fs::metadata(shard)?.len();
        }
        println!(
            "plan card bytes {card_bytes} (dense {} + experts {}) host bytes {host_bytes} \
             (experts {} + tables {}) kv {kv}; file {file_bytes} B in {} shards",
            card.dense_bytes,
            card.expert_bytes,
            plan.host.expert_bytes,
            plan.host.table_bytes,
            split.shard_count()
        );
        let devices = e["placement"]["devices"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let class_sum = |d: &Value| {
            d["classes"]
                .as_object()
                .map(|c| c.values().filter_map(Value::as_u64).sum::<u64>())
        };
        let layers = format!("{}-{}", stage.layers.start, stage.layers.end - 1);
        let m = &e["model"];
        let mut ok = true;
        check(
            &mut ok,
            "props_engine_identity",
            e["name"] == "bloomery"
                && e["version"]
                    .as_str()
                    .is_some_and(|v| !v.is_empty() && !v.ends_with(" mock"))
                && e["args"] == json!(argv)
                && e["server_pid"] == json!(pid),
        );
        check(
            &mut ok,
            "props_engine_model",
            m["format"] == "gguf"
                && m["arch"] == json!(split.architecture())
                && m["files"] == json!(split.shard_count())
                && m["bytes"] == json!(file_bytes)
                && m["n_layers"] == json!(inputs.model.layers)
                && m["n_experts"] == json!(inputs.model.experts)
                && m["n_experts_used"] == json!(inputs.model.experts_used)
                && m["ctx_train"] == json!(split.arch_get_u64("context_length"))
                && m["quant"].as_str().is_some_and(|q| !q.is_empty()),
        );
        check(
            &mut ok,
            "props_engine_devices",
            devices.len() == 2
                && devices[0]["device"]
                    .as_str()
                    .and_then(|d| d.strip_prefix("GPU"))
                    .is_some_and(|n| n.parse::<u32>().is_ok())
                && devices[0]["layers"] == json!(layers)
                && devices[1]["device"] == "CPU",
        );
        check(
            &mut ok,
            "props_engine_bytes_are_the_plans",
            devices
                .iter()
                .all(|d| d["bytes"].as_u64().is_some() && d["bytes"].as_u64() == class_sum(d))
                && devices
                    .first()
                    .is_some_and(|d| d["bytes"] == json!(card_bytes))
                && devices
                    .get(1)
                    .is_some_and(|d| d["bytes"] == json!(host_bytes)),
        );
        check(
            &mut ok,
            "props_engine_vram_kv",
            e["placement"]["vram_kv_bytes"] == json!(kv),
        );
        check(
            &mut ok,
            "props_engine_ctx_verified",
            e["ctx_verified"] == json!(VERIFIED_POSITIONS),
        );
        check(&mut ok, "props_engine_no_draft", e.get("draft").is_none());
        Ok(ok)
    }

    /// `ids` agree with `reference`, a run that did not stop at the
    /// end-of-generation id: all of them, or, when `stop` is `eos`, a prefix
    /// ending there.
    fn agree(ids: &[u32], stop: &str, reference: &[u32]) -> bool {
        match stop {
            "eos" => !ids.is_empty() && reference.starts_with(ids),
            _ => ids == reference,
        }
    }

    /// A greedy `/completion` of `prompt` ([`DRAFT_PREDICT`] ids, stopping at
    /// the end-of-generation id): its ids, `cache_n` when it and `prompt_n`
    /// add up to the prompt, and its timings.
    fn drafted_run(
        url: &dyn Fn(&str) -> String,
        what: &str,
        prompt: &[u32],
        cache: bool,
    ) -> Result<(Vec<u32>, Option<usize>, Value), GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": DRAFT_PREDICT, "temperature": 0,
            "return_tokens": true, "cache_prompt": cache,
        });
        let (st, body) = curl(&url("/completion"), Some(&body), false)?;
        let v = json_of("/completion", st, &body)?;
        let ids = ids_of(&v["tokens"]);
        let t = v["timings"].clone();
        println!(
            "{what}: prompt {} ids cache_prompt={cache}: cache_n={} prompt_n={} draft_n={} \
             draft_n_accepted={} tokens {ids:?}",
            prompt.len(),
            t["cache_n"],
            t["prompt_n"],
            t["draft_n"],
            t["draft_n_accepted"],
        );
        let counted = as_count(&t["cache_n"])
            .zip(as_count(&t["prompt_n"]))
            .filter(|(c, n)| c + n == prompt.len());
        Ok((ids, counted.map(|(c, _)| c), t))
    }

    /// The requests that need the logits row, served under the draft by
    /// plain steps: the plain run's sampled request ([`sampled`]'s body at
    /// [`SAMPLED_SEED`], top_k 40) is served with the plain server's ids
    /// (`plain_sampled`, [`SAMPLED_IDS`] in `--plain`) and drafts nothing,
    /// and a request that sets `ignore_eos` is served, all its 4 ids, and
    /// drafts nothing. Mutant of each: the refusal of a sampled or id-banning
    /// request under a draft restored (a 400).
    fn drafted_steps(
        url: &dyn Fn(&str) -> String,
        prompt: &str,
        plain_sampled: &[u32],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let cases = [
            (
                "sampled",
                json!({
                    "prompt": prompt, "n_predict": N_PREDICT,
                    "temperature": SAMPLED_TEMPERATURE, "top_k": 40, "seed": SAMPLED_SEED,
                    "return_tokens": true, "cache_prompt": false,
                }),
            ),
            (
                "ignore_eos",
                json!({
                    "prompt": prompt, "n_predict": 4, "temperature": 0, "ignore_eos": true,
                    "return_tokens": true, "cache_prompt": false,
                }),
            ),
        ];
        for (what, body) in cases {
            let (st, text) = curl(&url("/completion"), Some(&body), false)?;
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            let ids = ids_of(&v["tokens"]);
            println!(
                "{what} under the draft: HTTP {st} tokens {ids:?} draft_n={} error={}",
                v["timings"]["draft_n"], v["error"]
            );
            let served = st == 200 && v["timings"].get("draft_n").is_none();
            let (name, pass) = match what {
                "sampled" => (
                    "draft_serves_a_sampled_request_with_the_plain_ids",
                    served && !ids.is_empty() && ids == plain_sampled,
                ),
                _ => ("draft_serves_ignore_eos", served && ids.len() == 4),
            };
            check(&mut ok, name, pass);
        }
        Ok(ok)
    }

    /// The draft run (module header), on a server started with this
    /// process's `BLOOMERY_DRAFT=dspark`.
    fn drafted(a: &Args, plain: &Path, levers: &PlanLevers) -> Result<(), GateError> {
        let reference = gen_tokens(&a.gen_log)?;
        let read_ids = |name: &str| -> Result<Vec<u32>, GateError> {
            let path = plain.join(name);
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(serde_json::from_str(&text)?)
        };
        let plain_ids = read_ids(COMPLETION_IDS)?;
        let plain_probe = read_ids(PROBE_IDS)?;
        let (_, hp) = dspark::draft_hparams()?;
        let draft_path = dspark::draft_path()?;
        let draft_card = dspark::draft_card(Place::Gate)?;
        let draft_device = format!(
            "GPU{}",
            nvidia_smi_index(draft_card.name, draft_card.device)?
        );
        println!(
            "draft {} on {} ({draft_device}), window {}",
            draft_path.display(),
            draft_card.name,
            hp.window
        );
        std::fs::create_dir_all(&a.dir)?;
        let err_log = a.dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, &a.dir)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");
        let mut ok = true;
        ok &= drafted_props(&url, levers, &draft_path, &draft_device, &err_log)?;

        let completion = json!({
            "prompt": a.prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let ids = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        let t = &c1["timings"];
        println!("completion tokens {ids:?} stop_type={stop}");
        println!("plain server     {plain_ids:?}");
        println!("generate_ds41    {reference:?}");
        println!("completion timings {t}");
        check(
            &mut ok,
            "draft_completion_ids_are_the_plain_servers",
            ids == plain_ids,
        );
        check(
            &mut ok,
            "draft_completion_ids_are_generate_ds41",
            agree(&ids, &stop, &reference),
        );
        let (n, acc) = (as_count(&t["draft_n"]), as_count(&t["draft_n_accepted"]));
        check(
            &mut ok,
            "draft_timings_carry_the_drafts_counts",
            n.zip(acc).is_some_and(|(n, acc)| n > 0 && acc <= n),
        );
        let (probe, _, t) = drafted_run(&url, "draft probe", &probe_prompt(&url)?, false)?;
        println!("plain probe      {plain_probe:?}");
        println!("probe timings {t}");
        check(
            &mut ok,
            "draft_probe_ids_are_the_plain_servers",
            probe.len() == DRAFT_PREDICT && probe == plain_probe,
        );
        let (n, acc) = (as_count(&t["draft_n"]), as_count(&t["draft_n_accepted"]));
        check(
            &mut ok,
            "draft_probe_kept_and_rejected_proposals",
            n.zip(acc).is_some_and(|(n, acc)| 0 < acc && acc < n),
        );

        ok &= drafted_steps(&url, &a.prompt, &read_ids(SAMPLED_IDS)?)?;
        let (st, body) = curl(&url("/health"), None, false)?;
        check(
            &mut ok,
            "draft_health_after_the_stepped_requests",
            st == 200 && body.contains("\"ok\""),
        );

        ok &= drafted_reuse(&url, &a.ids, hp.window)?;

        let drafted_total = metric(&url, "spec_decode_num_draft_tokens_total")?;
        println!("metrics spec_decode_num_draft_tokens_total {drafted_total:?}");
        check(
            &mut ok,
            "draft_metrics_count_the_drafts",
            drafted_total.is_some_and(|v| v > 0.0),
        );

        println!("server stopped: {}", served.stop()?);
        if ok {
            println!("weekly-gpu-ds41-serve draft: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// A chat of a long system prompt, a long first message, a short reply
    /// and `last`.
    fn long_chat(last: &str) -> Value {
        json!([
            {"role": "system", "content": SYSTEM_LINE.repeat(SYSTEM_REPEAT)},
            {"role": "user", "content": LONG_USER_LINE.repeat(LONG_USER_REPEAT)},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": last},
        ])
    }

    /// The probe's prompt: [`long_chat`]'s rendered ids in reverse, whose
    /// greedy continuation runs [`DRAFT_PREDICT`] ids with no end-of-generation
    /// id and both keeps and rejects DSpark proposals.
    fn probe_prompt(url: &dyn Fn(&str) -> String) -> Result<Vec<u32>, GateError> {
        let mut ids = rendered(url, &long_chat(TURN3), &json!({}))?;
        ids.reverse();
        Ok(ids)
    }

    /// `/props`' `engine.draft` and the draft's placement row (module header),
    /// the plan re-derived under the server's own `card_free` reading
    /// ([`server_card_free`]).
    fn drafted_props(
        url: &dyn Fn(&str) -> String,
        levers: &PlanLevers,
        draft_path: &Path,
        draft_device: &str,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!("props engine {e}");
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|err| format!("open {}: {err}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let mut spec = workstation::RTX_3090;
        spec.free_bytes = Some(server_card_free(err_log)?);
        let machine = workstation::plan_on(spec, inputs.model.layers);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, levers)?;
        let card = plan.cards.first().ok_or("the gate's plan has no card")?;
        let card_bytes = card.dense_bytes + card.expert_bytes;
        let host_bytes = plan.host.expert_bytes + plan.host.table_bytes;
        let devices = e["placement"]["devices"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let class = |d: &Value, c: &str| d["classes"][c].as_u64();
        let row = |name: &str| devices.iter().find(|d| d["device"] == name).cloned();
        let draft_row = row(draft_device).unwrap_or(Value::Null);
        let target = devices.first().cloned().unwrap_or(Value::Null);
        let draft_bytes = class(&draft_row, "draft").unwrap_or(0);
        let target_bytes = target["bytes"].as_u64().unwrap_or(0);
        let own = if target["device"] == draft_device {
            draft_bytes
        } else {
            0
        };
        let file = draft_path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned());
        let mut ok = true;
        check(
            &mut ok,
            "props_engine_draft",
            e["draft"]["kind"] == "dspark"
                && e["draft"]["model"] == json!(file)
                && e["draft"]["path"] == json!(draft_path.display().to_string())
                && e["draft"]["n_max"] == 1
                && e["draft"]["device"] == draft_device,
        );
        check(
            &mut ok,
            "props_engine_draft_row",
            draft_bytes > 0
                && devices.last().is_some_and(|d| {
                    d["device"] == "CPU" && d["bytes"].as_u64() == Some(host_bytes)
                })
                && target_bytes == card_bytes + own,
        );
        Ok(ok)
    }

    /// Prefix reuse under the draft (module header).
    fn drafted_reuse(
        url: &dyn Fn(&str) -> String,
        ids: &[u32],
        window: usize,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        // A continuation: the draft follows the target, every position kept.
        let (g1, _, _) = drafted_run(url, "draft continue base", ids, false)?;
        let cont: Vec<u32> = ids.iter().chain(&g1).copied().collect();
        let (g2, c2, _) = drafted_run(url, "draft continue warm", &cont, true)?;
        let (g3, c3, _) = drafted_run(url, "draft continue fresh", &cont, false)?;
        check(
            &mut ok,
            "draft_continuation_cache_n",
            c2 == Some(cont.len() - 1),
        );
        check(
            &mut ok,
            "draft_continuation_ids_are_fresh",
            g2 == g3 && c3 == Some(0),
        );
        // Two chats that share a long first message and differ in their
        // second: the second chat's cut falls a window before the prefix they
        // share, which the cut rule brings to the first message's call start.
        let none = json!({});
        let (pa, pb) = (
            rendered(url, &long_chat(TURN2), &none)?,
            rendered(url, &long_chat(TURN3), &none)?,
        );
        let bound = common(&pa, &pb).saturating_sub(window);
        println!(
            "draft chats of {} and {} ids share {}: a cut keeps at most {bound}",
            pa.len(),
            pb.len(),
            common(&pa, &pb)
        );
        drafted_run(url, "draft cut base", &pa, true)?;
        let (b1, cb, _) = drafted_run(url, "draft cut warm", &pb, true)?;
        let (b2, _, _) = drafted_run(url, "draft cut fresh", &pb, false)?;
        check(
            &mut ok,
            "draft_cut_keeps_a_window_less",
            cb.is_some_and(|c| c > 0 && c <= bound),
        );
        check(&mut ok, "draft_cut_ids_are_fresh", b1 == b2);
        // The second chat put back from the prompt cache after another prompt:
        // the draft starts over there too.
        let other = probe_prompt(url)?;
        drafted_run(url, "draft other", &other, true)?;
        let bound = pb.len() - 1 - window;
        let (r1, cr, _) = drafted_run(url, "draft resume warm", &pb, true)?;
        check(
            &mut ok,
            "draft_resume_keeps_a_window_less",
            cr.is_some_and(|c| c > 0 && c <= bound),
        );
        check(&mut ok, "draft_resume_ids_are_fresh", r1 == b2);
        Ok(ok)
    }

    /// The two-card run (module header): the server under `--place bp` and
    /// this process's `BLOOMERY_DRAFT=dspark`.
    fn tiered(a: &Args, levers: &PlanLevers) -> Result<(), GateError> {
        let reference = gen_tokens(&a.gen_log)?;
        let (draft, _) = dspark::draft_hparams()?;
        let draft_path = dspark::draft_path()?;
        let path = ref_model_path()?;
        let reserve = dspark::draft_reserve(Place::Bp, &draft, &path)?
            .ok_or("plan (b′) made no draft reserve")?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let machine_of = Place::Bp.machine(Some(reserve), Some(place::tier_batch(&inputs.hp)))?;
        std::fs::create_dir_all(&a.dir)?;
        let exe = Served::exe()?;
        let err_log = a.dir.join("server.err");
        let mut served = Served::spawn(&BP_SERVER_ARGS, &a.dir)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");
        // The machine the server planned by, as close as its record carries
        // it: the placement's name-based cards with the stage card's free
        // reading taken from the server's own `plan` record
        // ([`server_card_free`]). The record carries the stage's reading
        // alone, so the tier's card plans on its usable bytes, where an idle
        // card's reading caps once the plan adds the context's own cost back.
        let mut machine = machine_of(inputs.model.layers);
        machine.cards[0].free_bytes = Some(server_card_free(&err_log)?);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, levers)?;
        let ([stage], [tier], [card, tcard]) = (
            machine.cards.as_slice(),
            machine.tiers.as_slice(),
            plan.cards.as_slice(),
        ) else {
            return Err("plan (b′) is not one stage card and one tier card".into());
        };
        let gpu = |c: &Card| -> Result<String, GateError> {
            Ok(format!("GPU{}", nvidia_smi_index(&c.name, c.device)?))
        };
        let (stage_device, tier_device) = (gpu(stage)?, gpu(tier)?);
        let stage_bytes = card.dense_bytes + card.expert_bytes;
        let tier_bytes = tcard.dense_bytes + tcard.expert_bytes;
        let host_bytes = plan.host.expert_bytes + plan.host.table_bytes;
        println!(
            "plan (b′): {} ({stage_device}) {stage_bytes} B, tier {} ({tier_device}) {} experts \
             {tier_bytes} B with the draft's reserve {reserve} B, host {host_bytes} B",
            stage.name, tier.name, tcard.experts
        );
        let mut ok = true;

        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!("props engine {e}");
        let argv: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(BP_SERVER_ARGS.iter().map(|a| (*a).to_owned()))
            .collect();
        let devices = e["placement"]["devices"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let class = |d: &Value, c: &str| d["classes"][c].as_u64();
        let layers = format!("{}-{}", stage.layers.start, stage.layers.end - 1);
        check(&mut ok, "bp_props_args", e["args"] == json!(argv));
        check(
            &mut ok,
            "bp_props_names_both_cards",
            devices.len() == 3
                && devices[0]["device"] == json!(stage_device)
                && devices[0]["layers"] == json!(layers)
                && devices[1]["device"] == json!(tier_device)
                && devices[1].get("layers").is_none()
                && devices[2]["device"] == "CPU",
        );
        let draft_bytes = devices.get(1).and_then(|d| class(d, "draft")).unwrap_or(0);
        check(
            &mut ok,
            "bp_props_bytes_are_the_plans",
            devices
                .first()
                .is_some_and(|d| d["bytes"] == json!(stage_bytes))
                && devices.get(1).is_some_and(|d| {
                    class(d, "experts") == Some(tier_bytes)
                        && d["bytes"].as_u64() == Some(tier_bytes + draft_bytes)
                })
                && devices
                    .get(2)
                    .is_some_and(|d| d["bytes"] == json!(host_bytes))
                && e["placement"]["vram_kv_bytes"]
                    == json!(plan.cards.iter().map(|c| c.kv_bytes).sum::<u64>()),
        );
        let file = draft_path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned());
        check(
            &mut ok,
            "bp_props_draft_on_the_tier_card",
            e["draft"]["kind"] == "dspark"
                && e["draft"]["model"] == json!(file)
                && e["draft"]["device"] == json!(tier_device)
                && draft_bytes > 0
                && draft_bytes <= reserve,
        );

        let completion = json!({
            "prompt": a.prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let ids = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        let t = &c1["timings"];
        println!("completion tokens {ids:?} stop_type={stop}");
        println!("generate_ds41    {reference:?}");
        println!("completion timings {t}");
        check(
            &mut ok,
            "bp_completion_ids_are_generate_ds41",
            agree(&ids, &stop, &reference),
        );
        let (n, acc) = (as_count(&t["draft_n"]), as_count(&t["draft_n_accepted"]));
        check(
            &mut ok,
            "bp_timings_carry_the_drafts_counts",
            n.zip(acc).is_some_and(|(n, acc)| n > 0 && acc <= n),
        );
        let (st, body) = curl(&url("/health"), None, false)?;
        check(
            &mut ok,
            "bp_health_ok",
            st == 200 && body.contains("\"ok\""),
        );
        println!("server stopped: {}", served.stop()?);
        if ok {
            println!("weekly-gpu-ds41-serve bp: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// The card free bytes the server's own `plan` record ([`record::PLAN`])
    /// named: the census reading its plan was capped by. This gate's
    /// re-derivations plan with the same reading — a census read here would
    /// see the loaded server's own bytes as taken and size another plan, and a
    /// census-free re-plan (`workstation::plan_gate`) misses the cap the server
    /// planned under. A server that printed no `plan` record or more than
    /// one, or a record without its `card_free`, is a named error, never an
    /// uncapped re-plan.
    fn server_card_free(err_log: &Path) -> Result<u64, GateError> {
        let free = server_log(err_log, record::BLOOMERY_SERVE_DS41)?
            .one(&record::PLAN)?
            .u64("card_free")?;
        println!("the server's plan record: card_free={free}");
        Ok(free)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(crate::serve_levers::ACTS_ON)?;
        let a = parse_args()?;
        let place = PlanLevers::from_levers(&levers)?;
        match (&a.plain, levers.draft()) {
            (None, Some("dspark")) if a.bp => return tiered(&a, &place),
            _ if a.bp => {
                return Err(format!(
                    "--place bp takes BLOOMERY_DRAFT=dspark and no --plain; got --plain {:?}, \
                     BLOOMERY_DRAFT={:?}",
                    a.plain,
                    levers.draft()
                )
                .into());
            }
            (Some(plain), Some("dspark")) => return drafted(&a, plain, &place),
            (None, None) if a.need_path => return need_path(&a),
            (None, None) if a.slots => {
                std::fs::create_dir_all(&a.dir)?;
                let ok = slots(&a)? & slots_refuse_dspark(&a)?;
                if ok {
                    println!("weekly-gpu-ds41-serve slots: PASS");
                    return Ok(());
                }
                return Err(checks_failed());
            }
            (None, None) => {}
            (plain, draft) => {
                return Err(format!(
                    "--plain {plain:?} with BLOOMERY_DRAFT={draft:?}: the draft run takes both, \
                     BLOOMERY_DRAFT=dspark, and the plain run neither"
                )
                .into());
            }
        }
        let reference = gen_tokens(&a.gen_log)?;
        let rules = file_rules()?;
        println!(
            "compression ratios {:?}, raw window {}",
            rules.ratios, rules.window
        );
        std::fs::create_dir_all(&a.dir)?;
        let exe = Served::exe()?;
        let err_log = a.dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, &a.dir)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");

        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        println!("health {st} {body}");
        check(&mut ok, "health_ok", st == 200 && body.contains("\"ok\""));
        ok &= slots_files_need_the_path(&url)?;
        let argv: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(SERVER_ARGS.iter().map(|a| (*a).to_owned()))
            .collect();
        ok &= props_engine(&url, &argv, served.child.id(), &place, &err_log)?;

        let completion = json!({
            "prompt": a.prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let first = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "completion tokens {first:?} stop_type={stop} content={}",
            c1["content"]
        );
        println!("generate_ds41    {reference:?}");
        std::fs::write(a.dir.join(COMPLETION_IDS), serde_json::to_string(&first)?)?;
        let (probe, _, _) = drafted_run(&url, "probe", &probe_prompt(&url)?, false)?;
        std::fs::write(a.dir.join(PROBE_IDS), serde_json::to_string(&probe)?)?;
        let matches = match stop.as_str() {
            // The server stopped at the end-of-generation id, which it returns.
            "eos" => !first.is_empty() && reference.starts_with(&first),
            _ => first == reference,
        };
        check(&mut ok, "completion_ids_are_generate_ds41", matches);

        let s1 = sampled(&url, &a.prompt, SAMPLED_SEED, 40)?;
        std::fs::write(a.dir.join(SAMPLED_IDS), serde_json::to_string(&s1)?)?;
        let s2 = sampled(&url, &a.prompt, SAMPLED_SEED, 40)?;
        let k1 = sampled(&url, &a.prompt, SAMPLED_SEED, 1)?;
        check(
            &mut ok,
            "sampled_same_seed_identical",
            !s1.is_empty() && s1 == s2,
        );
        check(&mut ok, "sampled_top_k_1_is_greedy", k1 == first);
        let id = a.ids.iter().copied().min().ok_or("--ids is empty")?;
        ok &= position_limit(&url, id)?;

        let chat = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": 32,
        });
        let (st, body) = curl(&url("/v1/chat/completions"), Some(&chat), false)?;
        let plain = json_of("/v1/chat/completions", st, &body)?;
        let plain_content = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        println!(
            "chat content={:?} finish_reason={} usage={}",
            plain_content, plain["choices"][0]["finish_reason"], plain["usage"]
        );
        let mut streamed = chat.clone();
        streamed["stream"] = json!(true);
        let (st, sse) = curl(&url("/v1/chat/completions"), Some(&streamed), true)?;
        std::fs::write(a.dir.join("chat-stream.sse"), &sse)?;
        let events: Vec<&str> = sse
            .split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .collect();
        let mut deltas = String::new();
        for e in &events {
            if let Ok(v) = serde_json::from_str::<Value>(e)
                && let Some(t) = v["choices"][0]["delta"]["content"].as_str()
            {
                deltas.push_str(t);
            }
        }
        println!(
            "chat stream HTTP {st}: {} events, content={deltas:?}, last event {:?}",
            events.len(),
            events.last().copied().unwrap_or("")
        );
        check(
            &mut ok,
            "chat_stream_equals_non_stream",
            st == 200 && deltas == plain_content,
        );
        check(
            &mut ok,
            "chat_stream_ends_with_done",
            events.last() == Some(&"[DONE]"),
        );

        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": a.prompt})),
            false,
        )?;
        let tok = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("tokenize {tok:?} ids {:?}", a.ids);
        check(&mut ok, "tokenize_is_the_ids", tok == a.ids);

        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let again = json_of("/completion", st, &body)?;
        println!(
            "completion again {:?} cache_n={}",
            ids_of(&again["tokens"]),
            again["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_after_reset_identical",
            ids_of(&again["tokens"]) == first,
        );
        // Right after itself: the cache holds the prompt and all but the last
        // generated id, so the prompt shares all n of its ids and the last is
        // evaluated again. The cut at n − 1 is at its prompt call's end, past
        // any hole: at least n − 1 in whole compression groups is kept.
        let n = a.ids.len();
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let repeat = json_of("/completion", st, &body)?;
        let floor = whole_groups(n - 1, &rules.ratios);
        let t = &repeat["timings"];
        println!(
            "completion repeated {:?} cache_n={} prompt_n={} (n {n}, at least {floor} kept)",
            ids_of(&repeat["tokens"]),
            t["cache_n"],
            t["prompt_n"],
        );
        let (cache_n, prompt_n) = (as_count(&t["cache_n"]), as_count(&t["prompt_n"]));
        check(
            &mut ok,
            "repeat_cache_n",
            floor > 0
                && cache_n.is_some_and(|c| (floor..n).contains(&c))
                && cache_n.zip(prompt_n).is_some_and(|(c, p)| c + p == n),
        );
        check(
            &mut ok,
            "repeat_ids_identical",
            ids_of(&repeat["tokens"]) == first,
        );

        let reuse = |prompt: &[u32], cache: bool| -> Result<(Vec<u32>, Value), GateError> {
            let body = json!({
                "prompt": prompt, "n_predict": REUSE_PREDICT, "temperature": 0,
                "ignore_eos": true, "return_tokens": true, "cache_prompt": cache,
            });
            let (st, body) = curl(&url("/completion"), Some(&body), false)?;
            let v = json_of("/completion", st, &body)?;
            let ids = ids_of(&v["tokens"]);
            let t = &v["timings"];
            println!(
                "reuse prompt {} ids cache_prompt={cache}: cache_n={} prompt_n={} tokens {ids:?}",
                prompt.len(),
                t["cache_n"],
                t["prompt_n"]
            );
            let counted = t["cache_n"]
                .as_u64()
                .zip(t["prompt_n"].as_u64())
                .is_some_and(|(c, n)| c + n == prompt.len() as u64);
            if !counted {
                println!(
                    "FAIL: prompt_n + cache_n is not the prompt's {} ids",
                    prompt.len()
                );
            }
            Ok((
                ids,
                if counted {
                    t["cache_n"].clone()
                } else {
                    Value::Null
                },
            ))
        };
        let (g1, _) = reuse(&a.ids, false)?;
        let cont: Vec<u32> = a.ids.iter().chain(&g1).copied().collect();
        let (g2, c2) = reuse(&cont, true)?;
        let (g3, c3) = reuse(&cont, false)?;
        let full = g1.len() == REUSE_PREDICT && g3.len() == REUSE_PREDICT;
        check(&mut ok, "reuse_fixture_ran_to_n_predict", full);
        // The cache held the prompt and all but the last of g1: every position is kept.
        check(
            &mut ok,
            "reuse_continuation_cache_n",
            c2 == json!(cont.len() - 1),
        );
        check(
            &mut ok,
            "reuse_continuation_ids_are_fresh",
            g2 == g3 && c3 == json!(0),
        );
        if full {
            // The cache holds `cont` and g3[..7]; this prompt shares all of it but the
            // last, so the engine takes that one position back.
            let last = g3[REUSE_PREDICT - 2];
            let alt = if a.ids[0] != last { a.ids[0] } else { a.ids[1] };
            let mut back = cont.clone();
            back.extend(&g3[..REUSE_PREDICT - 2]);
            back.push(alt);
            let (g4, c4) = reuse(&back, true)?;
            let (g5, c5) = reuse(&back, false)?;
            check(
                &mut ok,
                "reuse_rollback_cache_n",
                c4 == json!(back.len() - 1),
            );
            check(
                &mut ok,
                "reuse_rollback_ids_are_fresh",
                g4 == g5 && c5 == json!(0),
            );
        }

        ok &= conversations(&url, &rules)?;
        ok &= budget(&url)?;
        ok &= interleaved(&url, &a.ids)?;
        ok &= shared_system(&url)?;

        println!("server stopped: {}", served.stop()?);
        ok &= slots(&a)?;
        ok &= slots_refuse_dspark(&a)?;
        if ok {
            println!("weekly-gpu-ds41-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
