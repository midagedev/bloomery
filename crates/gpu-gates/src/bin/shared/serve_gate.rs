//! The serve gate's clauses: the one owner of what a seat of a body that runs
//! one sequence on one card is held to — it refuses a second slot
//! ([`parallel_above_one_is_refused`]) or plans the two a body with resident
//! sequences serves ([`plan_names_the_slots`]), it loads
//! the file at the context it was asked for, it answers a chat turn with its
//! reasoning split at the model's own tags and its turn ending at the model's
//! own end token, and a request keeps the prefix it shares with the slot the
//! server holds ([`PREFIX`], `live_slot_keeps_the_prefix`: what holds and its
//! FAIL-first are that clause's doc).
//!
//! A seat gives the clauses its own facts ([`Clauses`]): the model word, the
//! records its server prints, the load and the listening record, the
//! architecture word its load record names, the file's layers, the tags of its
//! think span (none for a seat that serves plain text), the token that closes
//! its turn (none for a seat whose model never does), the context the server
//! is asked for and what its body's cut keeps
//! of a shared prefix ([`Cut`]); and what its chat request asks beside the
//! turn ([`Ask`]).
//!
//! A seat that keeps a saved prompt cache (its server prints a `cache` line with a budget) is also
//! held to what the cache is for: a conversation another one interrupted returns from the cache
//! ([`saved_state_returns`]); a seat with none skips that clause by name.
//!
//! Every seat is held to its plan's card bytes at the most context it serves
//! ([`card_bytes_hold_at_max_ctx`], the arm `serve_client` owns).

use std::path::Path;
use std::time::Duration;

use bloomery_gpu_gates::record;
use bloomery_gpu_gates::serve_client::{
    MAX_CTX, MaxCtx, PREFIX_TEXT, SeatGate, Served, card_bytes_hold_at_max_ctx, check,
    completion_raw, curl, greedy, ids_of, json_of, repeated_ids, server_log, tokenized,
};
use bloomery_gpu_gates::tier::{self, Tag};
use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
use serde_json::json;

/// What a serve gate's seat gives the shared clauses.
pub struct Clauses {
    /// The gate's own name, as the `PASSED` line names it.
    pub gate: &'static str,
    /// The seat's model word (`--model`).
    pub model: &'static str,
    /// The records the seat's server prints.
    pub kinds: &'static [&'static record::Kind],
    /// The load record of the seat.
    pub load: &'static record::Kind,
    /// The listening record of the seat.
    pub listening: &'static record::Kind,
    /// The architecture word the load record names.
    pub arch: &'static str,
    /// The layers of the file the seat serves.
    pub layers: u64,
    /// The tags of the model's think span, as its vocabulary spells them; for a model with no
    /// span (Muse Spark's reasoning is a message to `self`), the two markers of its protocol
    /// the server must take out of the message. `None` for a model whose description names no
    /// tags and whose seat serves its reasoning inline as plain text: the message then has no
    /// tag to leak. A model whose reply has sections (Kimi K3) names the markup between them.
    pub think: Option<[&'static str; 2]>,
    /// The token that closes the model's turn, as its vocabulary spells it; `None` for a seat
    /// whose model never ends a turn inside the tokens a request asks for (a cut of the
    /// model's layers), whose [`ENDED`] clause is left by name to a model that does.
    pub turn_end: Option<&'static str>,
    /// The context the server is asked for.
    pub ctx: u64,
    /// What the seat's cut (`app::Keep`) keeps of a prefix a request shares with the slot:
    /// every held position, or the body's own smaller rule ([`Cut`]).
    pub cut: Cut,
}

/// What a seat's cut does with a prefix a request shares with the slot, as
/// the prefix clause reads it: the seat states its body's own rule
/// (`app::Keep for Body`).
pub enum Cut {
    /// Every held position: any shared prefix, shorter than what the slot
    /// holds or not, is kept whole.
    EveryPosition,
    /// The body's rule keeps a shared prefix shorter than what the slot holds
    /// only where the sequence stands — at the nearest checkpoint at or below
    /// it, and from an empty model where none is held — so a shorter shared
    /// prefix is kept only as a whole continuation of what the slot holds.
    WhereItStands,
    /// The body keeps a shared prefix shorter than what the slot holds only at a checkpoint a
    /// prompt call took (its start and its end), so a request that extends the last prompt keeps
    /// the call's end (the prompt but its last id, which the server steps), and answers as the
    /// same request on an empty slot does. The first request's reply is `CHECKPOINT_REPLY`
    /// tokens, long enough that the positions the cut takes back wrap every ring of the state (a
    /// window of 128 positions, a compressor's block of 128), and the requests after it ask
    /// `CHECKPOINT_AFTER`: a restore that leaves one ring out answers differently.
    Checkpoint,
}

/// The tokens the chat request asks for at most, and the reasoning tokens the
/// budget lets the model spend before the server writes the close.
// PIN(2026-10-21): 64 → 512. MiniMax-M3's greedy answer to `TURN` ends its turn at
// 143 tokens (16 of reasoning, the rest content), past 64, so the turn-end clause
// read a cut answer (`finish_reason "length"`); 512 holds it 3.5×, and Solar-Open2
// stops under 64 either way. The clause still asks `stop` and no turn end in the
// message: red at 64, green at 512 on the same binary.
const MAX_TOKENS: u64 = 512;
const BUDGET: u64 = 16;

/// What the chat request asks beside the turn, and what its answer must hold: the reasoning
/// budget a think-span model's request carries, the effort a model with that knob is held to,
/// and whether the answer must carry reasoning.
#[derive(Clone, Copy)]
pub struct Ask {
    /// `reasoning_budget`: the reasoning tokens the server lets the model spend before it writes
    /// the span's close; `None` for a model with no span to close.
    pub budget: Option<u64>,
    /// `reasoning_effort`, where the template reads one.
    pub effort: Option<&'static str>,
    /// Whether [`CHAT`] requires reasoning in the message: false for a model that may answer a
    /// low-effort turn with none, whose reasoning is then read from the printed answer.
    pub reasons: bool,
}

impl Ask {
    /// A think-span model's request: the budget closes the span, the effort is the template's
    /// default and the message holds reasoning.
    pub const THINK_SPAN: Ask = Ask {
        budget: Some(BUDGET),
        effort: None,
        reasons: true,
    };
}

/// Polls of a second that bound the wait for a server to listen: a cold load
/// of the whole host set.
const LISTEN_POLLS: usize = 1000;

/// The chat turn the file answers.
const TURN: &str = "Name three primary colours.";

/// The words of the refusal of a second slot.
const ONE_SEQUENCE: &str = "holds one sequence";

/// The saved-state clause's prompts: the texts the two conversations repeat to their lengths in
/// ids (the first, the one that is saved and returns; the second, the one that takes the slot
/// between), what A's second return appends, and the tokens each request makes.
const SAVED_A: &str = "The quick brown fox jumps over the lazy dog near the river bank.";
const SAVED_B: &str = "Colourless green ideas sleep furiously beside a silent grey sea.";
const SAVED_TAIL: &str = " Now answer in one short sentence.";
const SAVED_A_IDS: usize = 1024;
const SAVED_B_IDS: usize = 256;
const SAVED_N: usize = 16;
/// The tokens of the request that opens A by its head (A less its last id): one, which is never
/// fed back, so the slot holds the head's rows alone.
const SAVED_HEAD_N: usize = 1;

const SAVED: &str = "saved_state_returns";
const CHAT: &str = "chat_splits_the_reasoning_at_the_models_tags";
const ENDED: &str = "chat_ends_at_the_turn_end";
const ANSWERED: &str = "chat_answers_a_turn";
const PREFIX: &str = "live_slot_keeps_the_prefix";

/// The tokens each request of the prefix clause asks for at most.
const PREFIX_TOKENS: u64 = 8;

/// The tokens the first request of [`Cut::Checkpoint`]'s prefix clause asks for: past the 128
/// positions a window ring and a compressor's block hold, so the positions a cut takes back
/// overwrite every row of both.
const CHECKPOINT_REPLY: u64 = 144;

/// The tokens the requests that follow it ask for at most: more than the 128 positions a window
/// ring and a compressor's block hold, so the positions after the restored one pass the next
/// block's end and read the compressed row the restored residues built, and a state a restore
/// got wrong shows in the ids.
const CHECKPOINT_AFTER: u64 = 160;

/// The question the prefix clause's first prompt asks after [`PREFIX_TEXT`].
const Q1: &str = "Which page did the clerk open first?";

/// The question the second prompt asks after the first's answer.
const Q2: &str = "And what did the oldest page record?";

/// The question the third prompt asks after [`PREFIX_TEXT`]: it differs from
/// [`Q1`] in its first word, so the two prompts' ids share the fixed text and
/// part inside it.
const Q3: &str = "What page did the clerk open first?";

/// The seat of `c`, the one server the clauses drive.
fn seat(c: &Clauses) -> SeatGate {
    SeatGate {
        model: c.model,
        kinds: c.kinds,
    }
}

/// The clauses of the seat `c` names, on the file of the profile
/// (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier): the seat
/// starts, loads the file and answers one chat request whose reasoning is
/// split at the model's own tags.
pub fn run(c: &Clauses) -> Result<(), GateError> {
    let dir = dir_of(c)?;
    std::fs::create_dir_all(&dir)?;
    let mut ok = parallel_above_one_is_refused(c, &dir)?;
    ok &= served(c, &dir)?;
    finish(c, ok)
}

/// The gate's own directory, the `--dir` flag every serve gate takes.
pub fn dir_of(c: &Clauses) -> Result<std::path::PathBuf, GateError> {
    seat(c).dir(&format!("usage: {} --dir <dir>", c.gate))
}

/// The gate's end: the `PASSED` line, or the failed clauses' own error.
pub fn finish(c: &Clauses, ok: bool) -> Result<(), GateError> {
    if !ok {
        return Err(checks_failed());
    }
    println!(
        "PASSED: {} the seat refuses a second slot, loads at the context it was asked for, \
         answers a chat turn with its reasoning split at the model's tags, and keeps the prefix \
         a request shares with the slot by its cut's rule (live_slot_keeps_the_prefix)",
        c.gate
    );
    Ok(())
}

/// `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its
/// stderr names the body's one sequence, and no `plan` record is printed (the
/// refusal comes before any plan). FAIL-first: a seat that takes a second slot
/// prints its plan and the clause is red.
pub fn parallel_above_one_is_refused(c: &Clauses, dir: &Path) -> Result<bool, GateError> {
    let mut ok = true;
    let file = ref_model_path()?;
    let (status, log, text) = seat(c).plan(dir, "refuse-parallel", &file, &["--parallel", "2"])?;
    let named = text.contains(ONE_SEQUENCE);
    let no_plan = log.first(&record::PLAN)?.is_none();
    println!(
        "refuse-parallel: exit non-zero {}; names {ONE_SEQUENCE:?} {named}; no plan record {no_plan}",
        !status.success()
    );
    if !named {
        println!("refuse-parallel stderr:\n{text}");
    }
    check(
        &mut ok,
        "parallel_above_one_is_refused",
        !status.success() && named && no_plan,
    );
    Ok(ok)
}

/// `plan_names_the_slots`: `--ctx` 2·[`Clauses::ctx`] `--parallel 2 --plan` exits 0 and its
/// stderr carries the `parallel` line of two slots a slot's [`Clauses::ctx`] wide that a round
/// of several serves in turns (`pass=turns`), and `--ctx` [`Clauses::ctx`] `--plan` carries the
/// one slot's (`pass=one`): the seat's body holds resident sequences, a select and a step each.
/// The one owner of what every seat that serves a second slot is held to; a seat that holds one
/// sequence is held to [`parallel_above_one_is_refused`] instead. FAIL-first: a seat that serves
/// several slots in one pass prints `pass=one` for two and the clause is red; one that divides
/// the context between its slots prints another `slot_ctx` and the clause is red.
pub fn plan_names_the_slots(c: &Clauses, dir: &Path) -> Result<bool, GateError> {
    let mut ok = true;
    let file = ref_model_path()?;
    let seat = SeatGate {
        model: c.model,
        kinds: c.kinds,
    };
    let total = (2 * c.ctx).to_string();
    let (status, _, text) = seat.plan(
        dir,
        "plan-two-slots",
        &file,
        &["--ctx", &total, "--parallel", "2"],
    )?;
    let two = status.success()
        && text.lines().any(|l| {
            l.starts_with(&format!(
                "parallel rule=slots slots=2 slot_ctx={} total={total} ",
                c.ctx
            )) && l.ends_with(" pass=turns")
        });
    let ctx = c.ctx.to_string();
    let (status, _, text) = seat.plan(dir, "plan-one-slot", &file, &["--ctx", &ctx])?;
    let one = status.success()
        && text.lines().any(|l| {
            l.starts_with(&format!(
                "parallel rule=slots slots=1 slot_ctx={} total={} ",
                c.ctx, c.ctx
            )) && l.ends_with(" pass=one")
        });
    println!(
        "ctx {total} --parallel 2: two slots, pass=turns {two}; ctx {ctx}: one slot, pass=one {one}"
    );
    if !(two && one) {
        println!("plan-two-slots stderr:\n{text}");
    }
    check(&mut ok, "plan_names_the_slots", two && one);
    Ok(ok)
}

/// A server of two slots over the file at `--ctx` 2·[`Clauses::ctx`] in `<dir>/two-slots`,
/// `more` adding the seat's own environment and arguments (it is given that directory): its
/// address, the handle and its stderr log.
pub fn two_slot_server(
    c: &Clauses,
    dir: &Path,
    more: impl FnOnce(&Path, &mut std::process::Command, &mut Vec<String>) -> Result<(), GateError>,
) -> Result<(Served, String, std::path::PathBuf), GateError> {
    let d = dir.join("two-slots");
    std::fs::create_dir_all(&d)?;
    let file = ref_model_path()?;
    let (mut cmd, mut all) = seat(c).command(&file)?;
    let total = (2 * c.ctx).to_string();
    for a in ["--port", "0", "--ctx", &total, "--parallel", "2"] {
        all.push(a.to_owned());
    }
    more(&d, &mut cmd, &mut all)?;
    let all: Vec<&str> = all.iter().map(String::as_str).collect();
    let mut s = Served::spawn_cmd(cmd, &all, &d)?;
    let err_log = d.join("server.err");
    let addr = s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1))?;
    Ok((s, addr, err_log))
}

/// The slot a `/completion` answer ran on.
pub fn slot_of(v: &serde_json::Value) -> Result<u64, GateError> {
    Ok(v["id_slot"]
        .as_u64()
        .ok_or("a /completion answer without id_slot")?)
}

/// `two_slots_load`: the `load` and `listening` records of the server [`two_slot_server`]
/// started (its stderr log `err_log`) name two slots and a slot's [`Clauses::ctx`].
pub fn two_slots_load(c: &Clauses, err_log: &Path) -> Result<bool, GateError> {
    let mut ok = true;
    let log = server_log(err_log, c.kinds)?;
    let (load, listen) = (log.one(c.load)?, log.one(c.listening)?);
    let held = load.u64("slots")? == 2
        && load.u64("ctx")? == c.ctx
        && listen.u64("slots")? == 2
        && listen.u64("ctx")? == c.ctx;
    println!(
        "two slots: load {:?}; listening {:?} {held}",
        load.line(),
        listen.line()
    );
    check(&mut ok, "two_slots_load", held);
    Ok(ok)
}

/// The clauses on one server: `load_and_listen` and the chat; the server is
/// stopped at the end.
pub fn served(c: &Clauses, dir: &Path) -> Result<bool, GateError> {
    served_with(c, dir, Ask::THINK_SPAN)
}

/// [`served`] with the chat request `ask` says.
pub fn served_with(c: &Clauses, dir: &Path, ask: Ask) -> Result<bool, GateError> {
    let mut ok = true;
    let d = dir.join("serve");
    std::fs::create_dir_all(&d)?;
    let file = ref_model_path()?;
    let (cmd, mut all) = seat(c).command(&file)?;
    all.extend(
        ["--port", "0", "--ctx", &c.ctx.to_string()]
            .iter()
            .map(|a| (*a).to_owned()),
    );
    let all: Vec<&str> = all.iter().map(String::as_str).collect();
    let mut s = Served::spawn_cmd(cmd, &all, &d)?;
    let err_log = d.join("server.err");
    let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
        Ok(addr) => addr,
        Err(e) => {
            println!("the server never listened: {e}");
            check(&mut ok, "load_and_listen", false);
            return Ok(ok);
        }
    };
    let records = server_log(&err_log, c.kinds)?;
    let load = records.one(c.load)?;
    let listen = records.one(c.listening)?;
    let url = |p: &str| format!("http://{addr}{p}");
    let (health, _) = curl(&url("/health"), None, false)?;
    let loaded = load.word("arch")? == c.arch
        && load.u64("slots")? == 1
        && load.u64("layers")? == c.layers
        && load.u64("ctx")? == c.ctx;
    let listening = listen.u64("slots")? == 1 && listen.u64("ctx")? == c.ctx;
    let in_order = load.at() < listen.at();
    println!(
        "serve at {addr}: load {:?}; listening {:?}; load names {}, one slot, {} layers and ctx \
         {} {loaded}; listening names one slot and ctx {} {listening}; load before listening \
         {in_order}; /health {health}",
        load.line(),
        listen.line(),
        c.arch,
        c.layers,
        c.ctx,
        c.ctx
    );
    check(
        &mut ok,
        "load_and_listen",
        loaded && listening && in_order && health == 200,
    );
    // The turn is answered on any file; its reasoning and its end read what the model writes, which
    // a cut of the model's layers does not write, so those two are the real file's.
    let (answered, split, ended) = match chat(c, ask, &url, &d) {
        Ok(verdicts) => verdicts,
        Err(e) => {
            println!("{CHAT}: {e}");
            (false, false, false)
        }
    };
    check(&mut ok, ANSWERED, answered);
    if tier::run_clause(CHAT, Tag::FileBound)? {
        check(&mut ok, CHAT, split);
    }
    if tier::run_clause(ENDED, Tag::FileBound)? {
        match c.turn_end {
            Some(_) => check(&mut ok, ENDED, ended),
            None => println!(
                "skip {ENDED}: the {model} seat's model never ends a turn inside the tokens a \
                 request asks for, so the clause is left to a model that does",
                model = c.model
            ),
        }
    }
    if tier::run_clause(PREFIX, Tag::SelfConsistency)? {
        let lever = records
            .one(&record::RESIDENCY_LEVER)?
            .word("residency")?
            .to_owned();
        check(
            &mut ok,
            PREFIX,
            live_slot_keeps_the_prefix(c, &url, &d, &lever)?,
        );
    }
    ok &= saved_state_returns(c, &err_log, &url, &d)?;
    println!("server stopped: {}", s.stop()?);
    ok &= card_bytes_hold_at_max_ctx(
        MAX_CTX,
        &MaxCtx {
            dir: &dir.join("max-ctx"),
            kinds: c.kinds,
            listening: c.listening,
        },
        || {
            let (cmd, mut all) = seat(c).command(&ref_model_path()?)?;
            all.extend(["--port", "0"].iter().map(|a| (*a).to_owned()));
            Ok((cmd, all))
        },
    )?;
    Ok(ok)
}

/// The budget a server's `cache` line prints (`cache ram=<bytes> …`), `None` for a server that
/// prints none: a seat with no saved prompt cache.
fn cache_ram_of(err_log: &Path) -> Result<Option<u64>, GateError> {
    let text =
        std::fs::read_to_string(err_log).map_err(|e| format!("read {}: {e}", err_log.display()))?;
    let Some(line) = text.lines().find(|l| l.starts_with("cache ram=")) else {
        return Ok(None);
    };
    let ram = line["cache ram=".len()..]
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| format!("the cache line {line:?} names no budget"))?;
    Ok(Some(ram))
}

/// `saved_state_returns` on the server of `served_with` (its stderr `err_log`): conversation A of
/// [`SAVED_A_IDS`] ids, an unrelated conversation B that takes the slot, and A's return from the
/// saved state. A seat that prints no `cache` line, or a budget of 0, skips the clause by name.
///
/// The first half: A's head (its ids less the last) is answered, then A itself continues the live
/// slot by that one id and is answered ([`SAVED_N`] tokens: the truth); B takes the slot; A
/// returns. The return keeps every position but the one id every request feeds
/// (`timings.cache_n` at least A's ids less one: restored from the saved state, not fed again)
/// and, with the residency off (the server's `residency lever` record; any other word skips this
/// half by name, the experts the two runs find on the card may differ), answers the truth's ids.
/// The two runs are the same arithmetic on the same rows by construction: the head's rows (the
/// rows saved) and then one pass of one id, so the ids are equal on every seat. They are not
/// compared with an uncached run: the same prompt in other passes, or its answer's stepped rows
/// in a prompt pass, differ by rounding that the Muse Spark batch does not hide.
///
/// The second half: B takes the slot again and A returns extended by its reply and a tail;
/// `cache_n` at least A's ids, every position the saved state holds of A's answer.
///
/// FAIL-first: a resume that puts back one position leaves `cache_n` at 1, and the clause is red on
/// it; one that stands at the positions asked over the rows of one is red on the ids (the
/// MiniMax-M3 fixture cut answers one id whatever the rows: its ids half is the real tier's). A
/// snapshot that returns the refusal it replaced is not this clause's: the engine fails and the
/// server stops.
fn saved_state_returns(
    c: &Clauses,
    err_log: &Path,
    url: &dyn Fn(&str) -> String,
    dir: &Path,
) -> Result<bool, GateError> {
    let mut ok = true;
    match cache_ram_of(err_log)? {
        None => {
            println!("skip {SAVED}: the {} seat prints no cache line", c.model);
            return Ok(ok);
        }
        Some(0) => {
            println!("skip {SAVED}: the {} seat's cache budget is 0", c.model);
            return Ok(ok);
        }
        Some(_) => {}
    }
    let a = repeated_ids(url, SAVED_A, SAVED_A_IDS)?;
    let b = repeated_ids(url, SAVED_B, SAVED_B_IDS)?;
    let tail = tokenized(url, SAVED_TAIL)?;
    if tail.is_empty() || a[0] == b[0] {
        return Err(format!(
            "the saved-state prompts: A starts {:?} and B {:?}, the tail holds {} ids; B must \
             start where A does not",
            a[0],
            b[0],
            tail.len()
        )
        .into());
    }
    let fed = a.len() as u64 - 1;
    greedy(
        url,
        a[..a.len() - 1].to_vec(),
        SAVED_HEAD_N,
        dir,
        "saved_head",
        true,
    )?;
    let (truth, _) = greedy(url, a.clone(), SAVED_N, dir, "saved_a", true)?;
    if truth.tokens.is_empty() || truth.cache_n != fed {
        return Err(format!(
            "the saved-state setup: A after its head kept {} ids of {fed} and answered {} \
             tokens",
            truth.cache_n,
            truth.tokens.len()
        )
        .into());
    }
    greedy(url, b.clone(), SAVED_N, dir, "saved_b", true)?;
    let (back, _) = greedy(url, a.clone(), SAVED_N, dir, "saved_a_back", true)?;
    let word = server_log(err_log, c.kinds)?
        .first(&record::RESIDENCY_LEVER)?
        .map(|l| l.word("residency").map(str::to_owned))
        .transpose()?;
    let same = if word.as_deref() == Some("off") {
        println!(
            "saved: the return's ids {:?}, the truth's {:?}",
            back.tokens, truth.tokens
        );
        back.tokens == truth.tokens
    } else {
        println!(
            "skip {SAVED} answers the truth's ids: the residency lever is {word:?}, not \"off\""
        );
        true
    };
    greedy(url, b, SAVED_N, dir, "saved_b_again", true)?;
    let mut extended = a;
    extended.extend_from_slice(&truth.tokens);
    extended.extend_from_slice(&tail);
    let (longer, _) = greedy(url, extended, SAVED_N, dir, "saved_a_extended", true)?;
    println!(
        "saved: A {} ids, B {} ids took the slot; A returned kept {} (wants at least {fed}) and \
         answers the truth's ids {same}; after B again, A extended kept {} (wants at least {})",
        fed + 1,
        SAVED_B_IDS,
        back.cache_n,
        longer.cache_n,
        fed + 1
    );
    check(
        &mut ok,
        SAVED,
        back.cache_n >= fed && same && longer.cache_n > fed,
    );
    Ok(ok)
}

/// [`ANSWERED`], [`CHAT`] and [`ENDED`], from one request: the turn is counted, carries no think
/// tag and stops by its end or its length; its message holds reasoning and content; it finishes
/// at the turn end.
fn chat(
    c: &Clauses,
    ask: Ask,
    url: &dyn Fn(&str) -> String,
    dir: &Path,
) -> Result<(bool, bool, bool), GateError> {
    let mut body = json!({
        "messages": [{ "role": "user", "content": TURN }],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": false,
    });
    if let Some(budget) = ask.budget {
        body["reasoning_budget"] = json!(budget);
    }
    if let Some(effort) = ask.effort {
        body["reasoning_effort"] = json!(effort);
    }
    let (status, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
    std::fs::write(dir.join("chat.json"), &text)?;
    let answer = json_of("/v1/chat/completions", status, &text)?;
    let message = &answer["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap_or("");
    let content = message["content"].as_str().unwrap_or("");
    let leaked = c
        .think
        .iter()
        .flatten()
        .any(|tag| reasoning.contains(tag) || content.contains(tag));
    let usage = &answer["usage"];
    let counted = usage["prompt_tokens"].as_u64().is_some_and(|n| n > 0)
        && usage["completion_tokens"]
            .as_u64()
            .is_some_and(|n| (1..=MAX_TOKENS).contains(&n));
    let finish = answer["choices"][0]["finish_reason"].as_str().unwrap_or("");
    let end_in_message = |end: &str| content.contains(end) || reasoning.contains(end);
    let ended = c
        .turn_end
        .is_some_and(|end| finish == "stop" && !end_in_message(end));
    println!(
        "chat: usage {usage}; finish_reason {finish:?}; reasoning {} chars, content {} chars; a \
         think tag in the message {leaked}; the turn end in the message {}; reasoning \
         {reasoning:?}; content {content:?}",
        reasoning.trim().len(),
        content.trim().len(),
        c.turn_end.is_some_and(end_in_message)
    );
    let stopped = matches!(finish, "stop" | "length");
    Ok((
        counted && !leaked && stopped,
        counted
            && (!ask.reasons || !reasoning.trim().is_empty())
            && !content.trim().is_empty()
            && !leaked,
        ended,
    ))
}

/// One greedy `/completion` of the prefix clause: the timings' `cache_n` and
/// the sum of it and `prompt_n`, the answer's text and ids, its body in
/// `<dir>/<name>.json`.
struct Asked {
    cache_n: u64,
    whole: u64,
    content: String,
    tokens: Vec<u32>,
}

/// [`Asked`] of one request: `ids` at `tokens` tokens at most, temperature 0,
/// the prompt cache `cache` asks the server to keep the prefix the prompt
/// shares with the slot.
fn asked(
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    name: &str,
    (ids, tokens): (&[u32], u64),
    cache: bool,
) -> Result<Asked, GateError> {
    let (st, text) = completion_raw(url, ids, usize::try_from(tokens)?, cache)?;
    std::fs::write(dir.join(format!("{name}.json")), &text)?;
    let v = json_of("/completion", st, &text)?;
    let timings = &v["timings"];
    let count = |k: &str| {
        timings[k]
            .as_u64()
            .ok_or_else(|| format!("{name}: the answer's timings carry no {k}"))
    };
    let (prompt_n, cache_n) = (count("prompt_n")?, count("cache_n")?);
    Ok(Asked {
        cache_n,
        whole: prompt_n + cache_n,
        content: v["content"]
            .as_str()
            .ok_or_else(|| format!("{name}: the answer carries no content text"))?
            .to_owned(),
        tokens: ids_of(&v["tokens"])?,
    })
}

/// The length of the longest common prefix of `a` and `b`.
fn common(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// [`PREFIX`], `live_slot_keeps_the_prefix`: the slot the chat clauses left
/// standing reuses the prefix a later request shares with it. The prompts are
/// posted as the ids `/tokenize` reads at `add_special` false, and a prompt of
/// ids alone takes no special token (only a prompt's first text element gets
/// the vocabulary's specials), so `timings.prompt_n + timings.cache_n` counts
/// exactly the ids the request carried. The first prompt — [`PREFIX_TEXT`] and
/// [`Q1`] — is answered with the cache on and its whole prompt count recorded
/// as `n1`; the second — the first's text, its answer and [`Q2`] — keeps at
/// least `n1` ids and its two timings count its whole prompt; the third —
/// [`PREFIX_TEXT`] and [`Q3`], which differs from [`Q1`] in its first word —
/// keeps exactly the ids the two prompts' tokenizations share, and the same
/// prompt answered with the cache off keeps no position and writes the same
/// text. The holds past the
/// first prompt's counting are a seat whose cut keeps every held position's
/// ([`Cut::EveryPosition`]): a body that keeps a shared prefix only where its
/// state stands ([`Cut::WhereItStands`]) re-feeds a shorter shared prompt from
/// an empty model by its own rule, so for it the second prompt's counting hold
/// alone runs and the third and fourth are skipped by name. The same-answer
/// hold runs only where the server's `residency lever` record says `off`
/// (`lever`): on a server the machine serves, the placement follows the
/// history of requests, and one line names the record and skips the hold.
/// A body that keeps a checkpoint ([`Cut::Checkpoint`]) runs a second shape of
/// the clause: the first request asks [`CHECKPOINT_REPLY`] tokens, so the
/// positions the cut takes back wrap every ring of the state, the second
/// prompt extends the first and keeps its call's end (the prompt but its last
/// id, which the server steps), and the same second prompt on the cache-off
/// run answers the same ids; the shorter third prompt
/// is asked after it, keeps no more than it shares (no checkpoint stands at
/// or below the shared prefix, so the model starts over) and answers the ids
/// of its cache-off run.
/// FAIL-first: a seat whose keep query grants nothing (`Seat::keep` answering
/// 0) keeps none of the shared prefix, the second and third holds are red and
/// the clause is red; a server that reuses the slot for the cache-off request
/// too (`cache_prompt` ignored) keeps positions in the reference, and the
/// same-answer hold is red.
fn live_slot_keeps_the_prefix(
    c: &Clauses,
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    lever: &str,
) -> Result<bool, GateError> {
    let checkpointed = matches!(c.cut, Cut::Checkpoint);
    let (reply, after) = if checkpointed {
        (CHECKPOINT_REPLY, CHECKPOINT_AFTER)
    } else {
        (PREFIX_TOKENS, PREFIX_TOKENS)
    };
    let p1 = format!("{PREFIX_TEXT} {Q1}");
    let ids1 = tokenized(url, &p1)?;
    if u64::try_from(ids1.len()).unwrap_or(u64::MAX) + reply >= c.ctx / 2 {
        return Err(format!(
            "the prefix clause's first prompt holds {} ids and asks {reply} more, past \
             half of the {} positions the server serves",
            ids1.len(),
            c.ctx
        )
        .into());
    }
    let first = asked(url, dir, "prefix_first", (&ids1, reply), true)?;
    let n1 = first.whole;
    println!(
        "prefix first: {} ids, kept {}, {} tokens; the answer {} chars",
        ids1.len(),
        first.cache_n,
        first.tokens.len(),
        first.content.trim().len()
    );
    // A cut that takes back fewer positions than a ring holds restores nothing of it: the reply
    // must have wrapped the window and the compressor's block (`Cut::Checkpoint`).
    let wrapped = !checkpointed || first.tokens.len() >= 128;
    if !wrapped {
        println!(
            "prefix first: the reply is {} tokens, short of the 128 that wrap a window ring and \
             a compressor's block: the restore below would be held to nothing",
            first.tokens.len()
        );
    }
    let answer = first.content.trim().to_owned();
    let p2 = if answer.is_empty() {
        format!("{PREFIX_TEXT} {Q1} {Q2}")
    } else {
        format!("{PREFIX_TEXT} {Q1} {answer} {Q2}")
    };
    let ids2 = tokenized(url, &p2)?;
    if matches!(c.cut, Cut::EveryPosition | Cut::Checkpoint) && common(&ids1, &ids2) < n1 as usize {
        return Err(format!(
            "the first answer's text re-tokenizes into a seam inside the first prompt: the \
             second prompt shares {} of the first's {n1} ids, and the clause needs a question \
             whose ids the answer's text does not merge",
            common(&ids1, &ids2)
        )
        .into());
    }
    let second = asked(url, dir, "prefix_extended", (&ids2, after), true)?;
    let counted = second.whole == ids2.len() as u64;
    // The server runs a prompt's last id as a step of its own (`Engine::prefill_next`), so a
    // prompt call ends, and a checkpoint stands, one id short of the prompt.
    let least = if checkpointed {
        n1.saturating_sub(1)
    } else {
        n1
    };
    let kept = second.cache_n >= least;
    println!(
        "prefix extended: {} ids; kept {} (at least {least} of the first prompt's {n1}: {}), \
         counted {}",
        ids2.len(),
        second.cache_n,
        if kept { "holds" } else { "FAIL" },
        counted
    );
    let mut holds = counted && wrapped && (matches!(c.cut, Cut::WhereItStands) || kept);
    match c.cut {
        Cut::EveryPosition => {
            let p3 = format!("{PREFIX_TEXT} {Q3}");
            let ids3 = tokenized(url, &p3)?;
            let share = common(&ids1, &ids3);
            let third = asked(url, dir, "prefix_shorter", (&ids3, after), true)?;
            let exact = third.cache_n == share as u64;
            println!(
                "prefix shorter: {} ids share {} with the first prompt and kept {} (want {}); \
                 {} tokens",
                ids3.len(),
                share,
                third.cache_n,
                share,
                third.tokens.len()
            );
            holds &= exact;
            if lever == "off" {
                let fresh = asked(url, dir, "prefix_shorter_fresh", (&ids3, after), false)?;
                let same = fresh.cache_n == 0 && fresh.content == third.content;
                println!(
                    "prefix same answer: the cache-off run kept {} (want 0) and wrote the same \
                     text {}",
                    fresh.cache_n,
                    fresh.content == third.content
                );
                holds &= same;
            } else {
                println!(
                    "prefix same answer: skipped, the server's residency lever record is \
                     {lever:?}, not off"
                );
            }
        }
        Cut::Checkpoint => {
            // A prompt that shares less than any checkpoint holds starts the model over, and the
            // checkpoints above the cut go with it.
            let p3 = format!("{PREFIX_TEXT} {Q3}");
            let ids3 = tokenized(url, &p3)?;
            let share = common(&ids1, &ids3);
            let third = asked(url, dir, "prefix_shorter", (&ids3, after), true)?;
            let within = third.cache_n <= share as u64;
            let answered = !second.tokens.is_empty() && !third.tokens.is_empty();
            println!(
                "prefix shorter: {} ids share {} with the first prompt and kept {} (at most {}); \
                 {} tokens",
                ids3.len(),
                share,
                third.cache_n,
                share,
                third.tokens.len()
            );
            holds &= within && answered;
            // The extended request restored a checkpoint and fed the rest; the same request on an
            // empty slot feeds every id. Both are one arithmetic on the same rows, so the ids are
            // equal where the residency leaves the experts where the plan put them.
            if lever == "off" {
                let fresh = asked(url, dir, "prefix_extended_fresh", (&ids2, after), false)?;
                let fresh3 = asked(url, dir, "prefix_shorter_fresh", (&ids3, after), false)?;
                let same = fresh.cache_n == 0
                    && fresh.tokens == second.tokens
                    && fresh3.cache_n == 0
                    && fresh3.tokens == third.tokens;
                println!(
                    "prefix same answer: the cache-off runs kept {} and {} (want 0) and wrote \
                     the same ids: {} (extended, {} ids) and {} (shorter, {} ids)",
                    fresh.cache_n,
                    fresh3.cache_n,
                    fresh.tokens == second.tokens,
                    second.tokens.len(),
                    fresh3.tokens == third.tokens,
                    third.tokens.len()
                );
                holds &= same;
            } else {
                println!(
                    "prefix same answer: skipped, the server's residency lever record is \
                     {lever:?}, not off"
                );
            }
        }
        Cut::WhereItStands => {
            println!(
                "prefix shorter and same answer: skipped, the seat's cut keeps a shared prefix \
                 only where the sequence stands, so a shorter one re-feeds from an empty model"
            );
        }
    }
    Ok(holds)
}
