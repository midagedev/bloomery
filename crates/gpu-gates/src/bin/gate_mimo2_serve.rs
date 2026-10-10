//! `gate_mimo2_serve` — the MiMo seat of `bloomery-serve` against the session
//! it serves, on the model file of the mimo2 profile (`BLOOMERY_REF_MODEL`),
//! on the one card the runner puts in view.
//!
//!     gate_mimo2_serve --dir <dir>
//!
//! The seat serves one slot, every routed expert on the host tier, the prompt
//! fed one decode step an id, and a later request keeps any prefix it shares
//! with what the slot holds. Every server is `bloomery-serve --model mimo2 -m
//! <file> …` beside this binary, `BLOOMERY_REF_MODEL` removed from its
//! environment. Logs per server in `<dir>/<name>/` (`server.err`, the
//! responses).
//!
//! Without a load:
//!
//! - `plan_only_names_its_default`: `--plan` with no `--place`, `--ctx` or
//!   `--parallel` exits 0 and prints `place unset` (the word `a`), the plan
//!   record and the `parallel` line, in that order, and no `load` or
//!   `listening` record. The plan's `ctx_max` is the seat's default context:
//!   at least 4096, at most the file's trained context, a multiple of 1024
//!   or the trained context itself; the plan at that context stands
//!   (`--ctx <ctx_max> --plan` exits 0 with the same `ctx_max`); and, unless
//!   it is the trained context, the next step of 1024 does not
//!   (`--ctx <ctx_max + 1024> --plan` is refused). FAIL-first: a seat that
//!   defaults to 4096 regardless plans a `ctx_max` whose next step stands,
//!   and the largest-plan hold is red.
//! - `refusals_before_any_load`: `--parallel 2` exits non-zero within
//!   [`REFUSE_WITHIN`], its stderr names the one-slot refusal and no `load`
//!   record is printed. On a census of two cards, `--place bp` exits
//!   non-zero the same way with the plan's tier-card refusal; on one card
//!   the `bp` half is a named skip. FAIL-first: a seat that accepts
//!   `--parallel 2` is still running at the bound and the clause is red.
//!
//! One server, `--ctx` [`CTX`], `--parallel 1`, `--port 0`:
//!
//! - `load_and_listen`: the `load` record comes before the `listening`
//!   record; the load names `arch=mimo2`, `slots=1`, `card_experts=0`,
//!   `host_experts` [`HOST_EXPERTS`], `layers` [`LAYERS`] and `ctx` [`CTX`];
//!   the `listening` record names `ctx` [`CTX`] and `slots=1`. A server that
//!   never listens is this clause red, its stderr printed, and the clauses
//!   that need it skipped by name. FAIL-first: a seat that plans an expert on
//!   the card names `card_experts` above 0.
//! - `completion_ids`: `/completion` at temperature 0, [`N`] tokens, of the
//!   batch set's ids (`refset::arch::mimo2::BATCH`, read through its family),
//!   of [`PROSE`] (`/tokenize`) and of the chat turn the server's own
//!   template renders (`/apply-template`, `/tokenize`) each answer [`N`] ids
//!   (or a prefix ending at an end-of-generation id), each from an empty
//!   cache (`cache_n` 0, `cache_prompt: false`). FAIL-first: a server that
//!   answers fewer than [`N`] ids without an end-of-generation stop, or keeps
//!   a prefix under `cache_prompt: false`, is red.
//! - `chat_is_those_ids`: `/v1/chat/completions` of the same turn at
//!   temperature 0 and `max_tokens` [`N`] is a 200 whose `usage` counts the
//!   prompt's ids and the completion's tokens, and whose message text (the
//!   reasoning and the content) is inside the text of the chat run's ids
//!   (`/detokenize`). FAIL-first: a seat whose chat path renders another
//!   prompt than `/apply-template` counts other prompt tokens.
//! - the prefix clauses, the seat keeping any held position:
//!   `edit_resend_keeps_the_row_where_it_diverges` — the turn of [`EDIT_A`]
//!   answered, then resent with its user message changed to [`EDIT_B`], which
//!   diverges at `j` inside the prompt — keeps `j` (`cache_n`);
//!   `edit_resend_ids_are_a_fresh_runs` — its ids equal the same request with
//!   `cache_prompt: false`, bit for bit, since every position is a decode step
//!   either way; and `extension_keeps_every_held_position` — the turn resent
//!   with its reply and a later user turn (the ids the template renders past a
//!   reply, found by [`later_ids`]) — keeps every position the slot held.
//!   FAIL-first: the seat's former keep rule, every position or none, keeps 0
//!   on the edit and the first clause is red; shown by that rule restored in
//!   `app::arch::mimo2`'s `Keep for Body` (`keepable` = `if n >= pos { pos }
//!   else { 0 }`, `cut` refusing any other position by name).
//!
//! Then the server is stopped by its handle and waited for (its host set is
//! free before the next load):
//!
//! - `server_ids_are_the_session_ids`: the session opened in process at [`CTX`]
//!   on the gate card (`shared/mimo2_open.rs`), each of the three prompts
//!   fed as the server feeds it — the prompt less its last id, then a step of
//!   the last, then `N − 1` steps at argmax — answers the server's ids bit for
//!   bit; the batch prompt's first id is ik's argmax at the set's last
//!   position or a named tie inside the e2e gate's rule (`ik_last`,
//!   `tie_allowed`). On a census of two cards the clause is a named skip: the
//!   gate plan (`shared/gate_card.rs`) runs on one visible card.
//! - `ids_see_an_id_short`: the same session fed a prompt one id short does
//!   not answer the server's ids for at least one prompt, so the clause above
//!   can see a seat that feeds one id short. FAIL-first: a server fed one id
//!   short answers the session's short run, and `server_ids_are_the_session_ids`
//!   is red on the prompt whose ids move.

#[cfg(not(all(feature = "mimo2", feature = "glm5next")))]
fn main() {
    eprintln!(
        "gate_mimo2_serve: built without the `mimo2` and `glm5next` features; see `just weekly-gpu-mimo2-serve`."
    );
    std::process::exit(2);
}

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mimo2_serve", gate::run())
}

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
#[path = "shared/mimo2_open.rs"]
mod mimo2_open;

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::ExitStatus;
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Answer, PrefixTurns, Served, agree, chat_is_those_ids, check, curl, edit_and_extension,
        greedy, json_of, rendered, serve_cmd, server_log, tokenized,
    };
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, ref_model_path};
    use bloomery_levers::CARD_BUDGET;
    use gguf::Split;
    use refset::arch::mimo2::{BATCH, IK};
    use serde_json::{Value, json};

    use crate::e2e::ik_last;
    use crate::mimo2_open::{N_VOCAB, last_argmax, open};

    const USAGE: &str = "usage: gate_mimo2_serve --dir <dir>";

    /// The tokens each request makes.
    const N: usize = 16;

    /// Cache rows of the loaded server and of the session it is held to: the
    /// 4,096-position set's step at position 4,096, with a margin of whole 64
    /// positions (`gate_mimo2_e2e`'s).
    const CTX: usize = 4160;

    /// The routed layers and experts a layer holds, from the header: the
    /// dense layer 0 and 47 routed layers of 256 experts each, all on the
    /// host: 47 · 256.
    const ROUTED_LAYERS: u64 = 47;
    const EXPERTS_PER_LAYER: u64 = 256;
    const HOST_EXPERTS: u64 = ROUTED_LAYERS * EXPERTS_PER_LAYER;

    /// The model's layers, the dense one included.
    const LAYERS: u64 = 48;

    /// The seat's context floor and the step its default context searches in.
    const CTX_FLOOR: u64 = 4096;
    const CTX_STEP: u64 = 1024;

    /// How long a refused server may take to exit: it refuses before any
    /// load.
    const REFUSE_WITHIN: Duration = Duration::from_secs(30);

    /// A `--plan` reads the file's header and exits; it opens no card.
    const PLAN_WITHIN: Duration = Duration::from_secs(120);

    /// The words the one-slot refusal names: the seat serves one slot, and
    /// the prompt cache is not served either.
    const ONE_SLOT: [&str; 2] = ["serves one slot", "no prompt cache"];

    /// The words the plan's tier-card refusal names (`PlacementError::
    /// HostRoutedTier`): the tier card, and the plan that keeps every routed
    /// expert on the host.
    const TIER_REFUSAL: [&str; 2] = [
        "tier card",
        "this plan puts every routed expert on the host",
    ];

    /// The server loads the whole host set: up to this many polls a second
    /// from a cold page cache.
    const LISTEN_POLLS: usize = 900;

    /// The chat turn the file answers, as the user message of [`EDIT_A`].
    const EDIT_A: &str =
        "Name the three primary colors of light, and say in one sentence why a screen mixes them.";

    /// The edit clause's turn changed: the resend diverges inside the first
    /// turn's prompt, with the rest of the prompt after it.
    const EDIT_B: &str =
        "Name the three primary colors of ink, and say in one sentence why a page reflects them.";

    /// The raw prompt the file continues.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and twelve, the same as every night, until the hundred and thirteenth";

    /// The extension clause's reply stand-in and later user turn: the
    /// template renders them in a conversation, and the text past the reply
    /// stand-in is the later turn's ids ([`later_ids`]).
    const MARK: &str = "QXZMARKQXZ";
    const LATER_USER: &str =
        "And which of the three does a screen show when it shows none of them?";

    struct Args {
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut dir = None;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok(Args {
            dir: dir.ok_or(USAGE)?,
        })
    }

    fn utf8(p: &Path) -> Result<&str, GateError> {
        Ok(p.to_str()
            .ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
    }

    /// The chat turn the file answers.
    fn messages() -> Value {
        json!([{ "role": "user", "content": EDIT_A }])
    }

    /// `bloomery-serve --model mimo2 -m <file> <args>` beside this binary,
    /// its logs in `<dir>/<name>/`.
    fn spawn(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
    ) -> Result<(Served, PathBuf), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let mut all = vec!["--model", "mimo2", "-m", utf8(file)?];
        all.extend_from_slice(args);
        Ok((Served::spawn_cmd(serve_cmd()?, &all, &d)?, d))
    }

    /// A server run to its exit within `within`: its status (`None`: still
    /// running at the bound, killed) and its stderr's path.
    fn run_to_exit(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        within: Duration,
    ) -> Result<(Option<ExitStatus>, PathBuf), GateError> {
        let (mut s, d) = spawn(dir, name, file, args)?;
        let t = Instant::now();
        let status = loop {
            if let Some(st) = s.child.try_wait()? {
                break Some(st);
            }
            if t.elapsed() > within {
                let _ = s.stop()?;
                break None;
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        Ok((status, d.join("server.err")))
    }

    /// `--plan` of `args`: its exit status and the log of its stderr.
    fn plan_run(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
    ) -> Result<(Option<ExitStatus>, Log, String), GateError> {
        let mut all = args.to_vec();
        all.push("--plan");
        let (status, err) = run_to_exit(dir, name, file, &all, PLAN_WITHIN)?;
        let text = std::fs::read_to_string(&err).unwrap_or_default();
        println!(
            "{name}: {} --plan: {}",
            args.join(" "),
            status.map_or("still running at the bound".to_owned(), |s| s.to_string())
        );
        Ok((
            status,
            server_log(&err, record::BLOOMERY_SERVE_MIMO2)?,
            text,
        ))
    }

    /// The file's trained context, `None` when it states none.
    fn trained_ctx(file: &Path) -> Result<Option<u64>, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        Ok(split.arch_get_u64("context_length"))
    }

    /// The cards in view: the driver's census, the cards the server this
    /// gate starts sees. The census holds one context on each; a plan only
    /// names a holder in its refusal text and reads the free bytes.
    fn visible_cards() -> Result<usize, GateError> {
        Ok(bloomery_gpu::census()?.len())
    }

    /// `plan_only_names_its_default` (the module header).
    fn plan_only_names_its_default(
        dir: &Path,
        file: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let trained = trained_ctx(file)?;
        let (status, log, text) = plan_run(dir, "plan", file, &[])?;
        let place = log.first(&record::PLACE_UNSET)?;
        let plan = log.first(&record::PLAN)?;
        let parallel_at = text
            .lines()
            .position(|l| l.starts_with("parallel "))
            .map(|i| i + 1);
        let in_order = matches!(
            (&place, &plan, parallel_at),
            (Some(a), Some(b), Some(c)) if a.at() < b.at() && b.at() < c
        );
        let word_a = match &place {
            Some(p) => p.word("place")? == "a",
            None => false,
        };
        let no_load = log.first(&record::LOAD_MIMO2)?.is_none()
            && log.first(&record::LISTENING_MIMO2)?.is_none();
        let exits = status.is_some_and(|s| s.success());
        println!(
            "plan only: exit 0 {exits}; place unset, plan, parallel in order {in_order}; place word a \
             {word_a}; no load or listening record {no_load}"
        );
        let ctx_max = match &plan {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let mut default_ok = false;
        if let Some(c) = ctx_max {
            let cap = trained.unwrap_or(CTX_FLOOR);
            let grid = c % CTX_STEP == 0 || Some(c) == trained;
            let bounded = c >= CTX_FLOOR && c <= cap.max(CTX_FLOOR);
            println!(
                "plan only: ctx_max {c} of trained {trained:?}; at least {CTX_FLOOR} and at most the \
                 trained context {bounded}; a multiple of {CTX_STEP} or the trained context {grid}"
            );
            // The plan at that context stands: asked for by flag it exits 0
            // and plans the same context.
            let c_arg = c.to_string();
            let (st, at_log, _) = plan_run(dir, "plan-at-default", file, &["--ctx", &c_arg])?;
            let stands = st.is_some_and(|s| s.success())
                && match at_log.first(&record::PLAN)? {
                    Some(p) => p.u64("ctx_max")? == c,
                    None => false,
                };
            // Past it, the next step does not: the default is the largest
            // context whose plan stands, unless the file's trained context
            // ends the search.
            let largest = if trained.is_some_and(|t| c + CTX_STEP > t) {
                println!(
                    "plan only: no step of {CTX_STEP} fits under the trained context past the \
                     default"
                );
                true
            } else {
                let next = (c + CTX_STEP).to_string();
                let (st, _, _) = plan_run(dir, "plan-past-default", file, &["--ctx", &next])?;
                let refused = st.is_some_and(|s| !s.success());
                println!(
                    "plan only: ctx_max {c} stands {stands}; --ctx {next} is refused {refused}"
                );
                refused
            };
            default_ok = bounded && grid && stands && largest;
        }
        check(
            ok,
            "plan_only_names_its_default",
            exits && in_order && word_a && no_load && default_ok,
        );
        Ok(())
    }

    /// `refusals_before_any_load` (the module header); `cards` the visible
    /// cards of the census.
    fn refusals_before_any_load(
        dir: &Path,
        file: &Path,
        cards: usize,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let refusal = |name: &str, args: &[&str], want: &[&str]| -> Result<bool, GateError> {
            let (status, err) = run_to_exit(dir, name, file, args, REFUSE_WITHIN)?;
            let text = std::fs::read_to_string(&err).unwrap_or_default();
            let log = server_log(&err, record::BLOOMERY_SERVE_MIMO2)?;
            let no_load = log.first(&record::LOAD_MIMO2)?.is_none();
            let named = want.iter().all(|w| text.contains(w));
            let exited = status.is_some_and(|s| !s.success());
            println!(
                "{name}: exit non-zero within {REFUSE_WITHIN:?} {exited}; names {want:?} {named}; \
                 no load record {no_load}"
            );
            if !(exited && named && no_load) {
                println!("{name} stderr:\n{text}");
            }
            Ok(exited && named && no_load)
        };
        let mut pass = refusal(
            "refuse-parallel",
            &["--parallel", "2", "--port", "0"],
            &ONE_SLOT,
        )?;
        if cards >= 2 {
            pass &= refusal(
                "refuse-bp",
                &["--place", "bp", "--port", "0"],
                &TIER_REFUSAL,
            )?;
        } else {
            println!(
                "skip refusals_before_any_load (--place bp): {cards} card in view; the tier-card \
                 refusal needs a census of two"
            );
        }
        check(ok, "refusals_before_any_load", pass);
        Ok(())
    }

    /// The ids of the later user turn the template renders past a reply: the
    /// text after [`MARK`] in the conversation of [`EDIT_A`], a reply of
    /// [`MARK`] and [`LATER_USER`], with the generation prompt.
    fn later_ids(url: &dyn Fn(&str) -> String) -> Result<Vec<u32>, GateError> {
        let messages = json!([
            { "role": "user", "content": EDIT_A },
            { "role": "assistant", "content": MARK },
            { "role": "user", "content": LATER_USER },
        ]);
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        let at = text
            .find(MARK)
            .ok_or_else(|| format!("the template's rendering holds no {MARK}: {text:?}"))?;
        let ids = tokenized(url, &text[at + MARK.len()..])?;
        if ids.is_empty() {
            return Err("the template renders no ids past a reply".into());
        }
        Ok(ids)
    }

    /// What the completions recorded: each prompt and the server's [`N`] ids
    /// for it.
    struct Answers {
        batch: Answer,
        prose: Answer,
        chat: Answer,
    }

    /// The server's clauses on the one that listens, `load_and_listen` to the
    /// prefix clauses: the three answers, `None` when the server never
    /// listened.
    fn served(
        dir: &Path,
        file: &Path,
        batch: Vec<u32>,
        ok: &mut bool,
    ) -> Result<Option<Answers>, GateError> {
        let ctx = CTX.to_string();
        let (mut s, d) = spawn(
            dir,
            "serve",
            file,
            &["--ctx", &ctx, "--parallel", "1", "--port", "0"],
        )?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the server never listened: {e}");
                println!(
                    "server stderr:\n{}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                check(ok, "load_and_listen", false);
                for name in [
                    "completion_ids",
                    "chat_is_those_ids",
                    "edit_resend_keeps_the_row_where_it_diverges",
                    "edit_resend_ids_are_a_fresh_runs",
                    "extension_keeps_every_held_position",
                    "server_ids_are_the_session_ids",
                    "ids_see_an_id_short",
                ] {
                    println!("skip {name}: the server never listened");
                }
                return Ok(None);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let records = server_log(&err_log, record::BLOOMERY_SERVE_MIMO2)?;
        let load = records.one(&record::LOAD_MIMO2)?;
        let listen = records.one(&record::LISTENING_MIMO2)?;
        let loaded = load.word("arch")? == "mimo2"
            && load.u64("slots")? == 1
            && load.u64("card_experts")? == 0
            && load.u64("host_experts")? == HOST_EXPERTS
            && load.u64("layers")? == LAYERS
            && load.u64("ctx")? == CTX as u64;
        let listening = listen.u64("ctx")? == CTX as u64 && listen.u64("slots")? == 1;
        println!(
            "server {} at {addr}; load record {:?}; listening record {:?}",
            file.display(),
            load.line(),
            listen.line()
        );
        check(
            ok,
            "load_and_listen",
            load.at() < listen.at() && loaded && listening,
        );

        // The three prompts and the server's ids for each.
        let prose = tokenized(&url, PROSE)?;
        let chat = rendered(&url, messages())?;
        println!(
            "prompts: batch {} ids, prose {} ids, chat {} ids",
            batch.len(),
            prose.len(),
            chat.len()
        );
        let (batch, _) = greedy(&url, batch, N, &d, "batch", false)?;
        let (prose, _) = greedy(&url, prose, N, &d, "prose", false)?;
        let (chat, chat_v) = greedy(&url, chat, N, &d, "completion", false)?;
        let answered = [&batch, &prose, &chat].iter().all(|a| {
            !a.tokens.is_empty() && (a.tokens.len() == N || a.stop == "eos") && a.cache_n == 0
        });
        check(ok, "completion_ids", answered);

        let chat_ok = chat_is_those_ids(&url, &d, messages(), N, (&chat, &chat_v))?;
        check(ok, "chat_is_those_ids", chat_ok);

        let later = later_ids(&url)?;
        edit_and_extension(
            &url,
            &d,
            ok,
            &PrefixTurns {
                a: EDIT_A,
                b: EDIT_B,
                n: N,
                min_after: 1,
            },
            &later,
        )?;
        println!("server stopped: {}", s.stop()?);
        Ok(Some(Answers { batch, prose, chat }))
    }

    /// The server's cut as the session runs it: the prompt less its last id,
    /// then a step of the last, then steps at argmax up to `n` ids; the first
    /// id's logits row alongside.
    fn session_ids(
        m: &mut bloomery_gpu_mimo2::Mimo2Model,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        let (last, head) = prompt.split_last().ok_or("a prompt of no id")?;
        m.reset()?;
        if !head.is_empty() {
            m.step(head)?;
        }
        let mut ids = vec![m.step(&[*last])?];
        let first_logits = m.logits()?;
        while ids.len() < n {
            let prev = *ids.last().ok_or("no id")?;
            ids.push(m.step(&[prev])?);
        }
        Ok((ids, first_logits))
    }

    /// `server_ids_are_the_session_ids` and `ids_see_an_id_short` (the module
    /// header), the server already stopped.
    fn server_ids_are_the_session_ids(
        file: &Path,
        levers: &bloomery_levers::Levers,
        a: &Answers,
        man: &RefManifest,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        crate::gate_card::init()?;
        let (mut session, opened) = open(file, levers, CTX)?;
        let m = session.model_mut();
        // The session's plan keeps every routed expert on the host, as the
        // server's load record says.
        let planned: u64 = opened.n_l.iter().sum();
        println!("session plan: {planned} card experts");
        let mut same = planned == 0;
        let mut moved = 0usize;
        for (name, ans) in [("batch", &a.batch), ("prose", &a.prose), ("chat", &a.chat)] {
            let (ours, first) = session_ids(m, &ans.prompt, N)?;
            let equal = agree(&ans.tokens, &ans.stop, &ours);
            println!(
                "{name}: server ids {:?} (stop {}), session ids {ours:?}: {}",
                ans.tokens,
                ans.stop,
                if equal { "equal" } else { "DIFFERENT" }
            );
            same &= equal;
            if name == "batch" {
                let ik = ik_last(man, N_VOCAB)?;
                let (top_ok, _) = last_argmax("batch first id", (&first, &ik), f64::INFINITY);
                same &= top_ok;
            }
            // The same session fed one id short: its ids must move for at
            // least one prompt, or the equality above cannot see a seat that
            // feeds one id short.
            if ans.prompt.len() > 1 {
                let (short, _) = session_ids(m, &ans.prompt[..ans.prompt.len() - 1], N)?;
                let differs = !agree(&ans.tokens, &ans.stop, &short);
                println!(
                    "{name}: the session fed one id short answers {short:?}: {}",
                    if differs { "moves the ids" } else { "same ids" }
                );
                moved += usize::from(differs);
            }
        }
        check(ok, "server_ids_are_the_session_ids", same);
        check(ok, "ids_see_an_id_short", moved > 0);
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let file = ref_model_path()?;
        println!("== {}", file.display());
        let cards = visible_cards()?;
        let mut ok = true;

        plan_only_names_its_default(&a.dir, &file, &mut ok)?;
        refusals_before_any_load(&a.dir, &file, cards, &mut ok)?;

        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let batch = toks.to_vec();
        match served(&a.dir, &file, batch, &mut ok)? {
            Some(answers) if cards == 1 => {
                server_ids_are_the_session_ids(&file, &levers, &answers, &man, &mut ok)?;
            }
            Some(_) => println!(
                "skip server_ids_are_the_session_ids: {cards} cards in view; the gate plan runs on \
                 one (shared/gate_card.rs)"
            ),
            None => {}
        }
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
