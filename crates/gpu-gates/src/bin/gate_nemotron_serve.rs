//! `gate_nemotron_serve` — the nemotron seat of `bloomery-serve` against the model it serves,
//! on the nemotron_h file of the profile (`BLOOMERY_REF_MODEL`, Nemotron 3 Nano 4B Q4_K_M).
//!
//!     gate_nemotron_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name), feeds a
//! prompt one decode step an id, and keeps a prefix a request shares with the slot only where it is
//! the whole held sequence (the Mamba-2 state cannot step back). Every server is `bloomery-serve
//! --model nemotron -m <file> …` beside this binary, `BLOOMERY_REF_MODEL` removed from its
//! environment. Logs per server in `<dir>/<name>/` (`server.err`, the responses).
//!
//! Without a load:
//!
//! - `plan_only_names_its_default`: `--plan` with no `--place`, `--ctx` or `--parallel` exits 0 and
//!   prints `place unset` (the word `a`), the plan record and the `parallel` line, in that order,
//!   and no `load` or `listening` record. The `parallel` line is the default's one slot:
//!   `slots=1 slot_ctx=<ctx_max> total=<ctx_max> from=default pass=one`. The one `ctx` line holds
//!   its word (`ctx_ladder::default_line_held`: `target` or `halved`, a rung of the ladder from
//!   262,144 or the trained context down to 4096, the limit the trained context) and names the
//!   plan's `ctx_max`; the plan at it stands (`--ctx <ctx_max> --plan` exits 0 with the same
//!   `ctx_max`); and the rung above it, when there is one, does not (`--ctx <rung above> --plan` is
//!   refused). FAIL-first: a seat that defaults to 4096 regardless names a rung whose rung above
//!   stands, and the hold is red.
//! - `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its stderr names the
//!   body's one sequence, and no `plan` record is printed (the refusal comes before any plan).
//!
//! One server as a user starts it, `--port 0` and no other flag:
//!
//! - `load_and_listen`: it listens; the `load` record comes before the `listening` record; the load
//!   names `arch=nemotron_h`, `slots=1` and the file's layer count, and its `ctx` and the
//!   `listening` record's are the `ctx` line's: the context the plan chose is the context the load
//!   made. FAIL-first: a seat whose load takes a context other than the plan's (a fixed one, or the
//!   slot context halved) is red.
//! - `completion_ids`: `/completion` at temperature 0, [`N`] tokens, of [`PROSE`] (`/tokenize`)
//!   and of the chat turn the server's own template renders (`/apply-template`, `/tokenize`),
//!   each from an empty cache (`cache_prompt: false`), answers the ids `NemotronHModel` gives
//!   stepped directly on the same ids (the `gate_nemotron_e2e` way: the prompt one step an id,
//!   then each answer fed back), or a prefix of them ending at an end-of-generation id. FAIL-first:
//!   a prefill that skips the first id moves the ids and the clause is red.
//! - `chat_is_200`: `/v1/chat/completions` of the same turn at temperature 0 and `max_tokens`
//!   [`N`] is a 200 whose `usage` counts the prompt's ids and the completion's tokens, and whose
//!   message text is inside the text of the chat run's ids.
//! - `edit_resend_is_a_fresh_run`: the prose prompt answered with `cache_prompt: true`, then
//!   resent with its second half changed, answers the ids of the same request at `cache_prompt:
//!   false` and keeps nothing (`cache_n` 0): the shared prefix is shorter than the held sequence
//!   and the state cannot step back, so the slot is reset. FAIL-first: a `keep` that grants the
//!   shared prefix reaches the refused rollback, the request fails and the clause is red.
//! - `extension_keeps_the_held_sequence`: the prose prompt resent with its answer appended answers
//!   the ids of the same request at `cache_prompt: false` and keeps the held positions (`cache_n`
//!   above 0): the shared prefix is the whole held sequence.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use bloomery_gpu::{Gpu, NemotronHModel};
    use bloomery_gpu_gates::ctx_ladder::{ctx_line, default_line_held, field, rung_above};
    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Answer, Served, agree, chat_is_those_ids, check, greedy, plan_run, rendered, serve_cmd,
        server_log, tokenized,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, generate, ref_model_path};
    use gguf::Split;
    use model::arch::nemotron_h::hparams::Hparams;
    use serde_json::json;

    const USAGE: &str = "usage: gate_nemotron_serve --dir <dir>";

    /// The tokens a completion asks for.
    const N: usize = 16;

    /// Polls of a second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 900;

    /// The cache rows of the session the ids are read from: past every prompt here and its
    /// answer.
    const SESSION_CTX: usize = 1024;

    /// The default is one slot.
    const DEFAULT_SLOTS: u64 = 1;

    /// The prompt the completion and prefix clauses run.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and twelve, the same as every night, until the hundred and thirteenth";

    /// The chat turn the file answers.
    const TURN: &str = "Name three primary colours.";

    /// The words of the refusal of a second slot.
    const ONE_SEQUENCE: &str = "holds one sequence";

    fn utf8(p: &Path) -> Result<&str, GateError> {
        Ok(p.to_str()
            .ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
    }

    fn parse_args() -> Result<PathBuf, GateError> {
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
        Ok(dir.ok_or(USAGE)?)
    }

    /// `bloomery-serve --model nemotron -m <file>` beside this binary.
    fn command(file: &Path) -> Result<(std::process::Command, Vec<String>), GateError> {
        Ok((
            serve_cmd()?,
            vec![
                "--model".to_owned(),
                "nemotron".to_owned(),
                "-m".to_owned(),
                utf8(file)?.to_owned(),
            ],
        ))
    }

    /// `--plan` of `args`: its exit status and stderr's lines, the log of its records and its
    /// text.
    fn plan(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
    ) -> Result<(std::process::ExitStatus, Log, String), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let (cmd, mut all) = command(file)?;
        all.extend(args.iter().map(|a| (*a).to_owned()));
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let err = d.join("server.err");
        let (status, lines) = plan_run(cmd, &all, &err)?;
        println!("{name}: {} --plan: {status}", args.join(" "));
        Ok((
            status,
            server_log(&err, record::BLOOMERY_SERVE_NEMOTRON)?,
            lines.join("\n"),
        ))
    }

    /// The file's trained context.
    fn trained_ctx(file: &Path) -> Result<u64, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let n = generate::trained_ctx(&split)
            .ok_or_else(|| format!("{}: no context_length", file.display()))?;
        Ok(u64::try_from(n)?)
    }

    /// `plan_only_names_its_default` (the module header); the context the default named.
    fn plan_only_names_its_default(
        dir: &Path,
        file: &Path,
        ok: &mut bool,
    ) -> Result<Option<u64>, GateError> {
        let limit = trained_ctx(file)?;
        let (status, log, text) = plan(dir, "plan", file, &[])?;
        let place = log.first(&record::PLACE_UNSET)?;
        let planned = log.first(&record::PLAN)?;
        let parallel_at = text
            .lines()
            .position(|l| l.starts_with("parallel "))
            .map(|i| i + 1);
        let parallel_line = text.lines().find(|l| l.starts_with("parallel "));
        let in_order = matches!(
            (&place, &planned, parallel_at),
            (Some(a), Some(b), Some(c)) if a.at() < b.at() && b.at() < c
        );
        let word_a = match &place {
            Some(p) => p.word("place")? == "a",
            None => false,
        };
        let no_load = log.first(&record::LOAD_NEMOTRON)?.is_none()
            && log.first(&record::LISTENING_NEMOTRON)?.is_none();
        let exits = status.success();
        println!(
            "plan only: exit 0 {exits}; place unset, plan, parallel in order {in_order}; place \
             word a {word_a}; no load or listening record {no_load}"
        );
        let ctx_max = match &planned {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let mut default_ok = false;
        if let Some(c) = ctx_max {
            let want = format!(
                "parallel rule=slots slots={DEFAULT_SLOTS} slot_ctx={c} total={c} from=default \
                 pass=one"
            );
            let line_ok = parallel_line == Some(want.as_str());
            println!("plan only: the parallel line {parallel_line:?} is {want:?} {line_ok}");
            let line = ctx_line(&text);
            let held = line.is_some_and(|l| {
                default_line_held(l, DEFAULT_SLOTS, (limit, "trained")) == Some(true)
                    && field(l, "ctx") == Some(c)
            });
            println!("plan only: the ctx line {line:?} holds its word at {c} {held}");
            let c_arg = c.to_string();
            let (st, at_log, _) = plan(dir, "plan-at-default", file, &["--ctx", &c_arg])?;
            let stands = st.success()
                && match at_log.first(&record::PLAN)? {
                    Some(p) => p.u64("ctx_max")? == c,
                    None => false,
                };
            println!("plan only: --ctx {c} stands {stands}");
            let above_short = match rung_above(limit, c) {
                None => {
                    println!("plan only: the default {c} is the first rung, none above it");
                    true
                }
                Some(above) => {
                    let above_arg = above.to_string();
                    let (st, _, _) = plan(dir, "plan-past-default", file, &["--ctx", &above_arg])?;
                    println!(
                        "plan only: --ctx {above}, the rung above, is refused {}",
                        !st.success()
                    );
                    !st.success()
                }
            };
            default_ok = line_ok && held && stands && above_short;
        }
        check(
            ok,
            "plan_only_names_its_default",
            exits && in_order && word_a && no_load && default_ok,
        );
        Ok(ctx_line(&text).and_then(|l| field(l, "ctx")))
    }

    /// `parallel_above_one_is_refused` (the module header).
    fn parallel_above_one_is_refused(
        dir: &Path,
        file: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let (status, log, text) = plan(dir, "refuse-parallel", file, &["--parallel", "2"])?;
        let named = text.contains(ONE_SEQUENCE);
        let no_plan = log.first(&record::PLAN)?.is_none();
        println!(
            "refuse-parallel: exit non-zero {}; names {ONE_SEQUENCE:?} {named}; no plan record \
             {no_plan}",
            !status.success()
        );
        if !named {
            println!("refuse-parallel stderr:\n{text}");
        }
        check(
            ok,
            "parallel_above_one_is_refused",
            !status.success() && named && no_plan,
        );
        Ok(())
    }

    /// The ids `NemotronHModel` gives stepped directly: for each prompt, from an empty cache, the
    /// prompt one step an id and then each answer fed back, [`N`] tokens.
    fn session_ids(file: &Path, prompts: &[&[u32]]) -> Result<Vec<Vec<u32>>, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let mut m = NemotronHModel::open(Gpu::new()?, split, SESSION_CTX)?;
        let mut out = Vec::new();
        for ids in prompts {
            m.reset()?;
            let mut answer = vec![m.step(ids)?];
            while answer.len() < N {
                let last = *answer.last().ok_or("an empty answer")?;
                answer.push(m.step(&[last])?);
            }
            out.push(answer);
        }
        Ok(out)
    }

    /// What the server answered, held until it is stopped and the session has the card.
    struct Answers {
        prose: Vec<u32>,
        turn: Vec<u32>,
        prose_answer: Answer,
        turn_answer: Answer,
    }

    /// The prose prompt with its second half replaced by ids the file's vocabulary holds.
    fn edited(prose: &[u32]) -> Vec<u32> {
        let half = prose.len() / 2;
        let mut ids = prose[..half].to_vec();
        ids.extend(prose[half..].iter().map(|&id| id + 1));
        ids
    }

    /// The clauses on one server: `load_and_listen` through the prefix clauses; `None` when the
    /// server never listened.
    fn served(
        dir: &Path,
        file: &Path,
        default_ctx: Option<u64>,
        layers: u64,
        ok: &mut bool,
    ) -> Result<Option<Answers>, GateError> {
        let d = dir.join("serve");
        std::fs::create_dir_all(&d)?;
        let (cmd, mut all) = command(file)?;
        all.extend(["--port".to_owned(), "0".to_owned()]);
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the server never listened: {e}");
                check(ok, "load_and_listen", false);
                return Ok(None);
            }
        };
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let records = server_log(&err_log, record::BLOOMERY_SERVE_NEMOTRON)?;
        let load = records.one(&record::LOAD_NEMOTRON)?;
        let listen = records.one(&record::LISTENING_NEMOTRON)?;
        let named = ctx_line(&text).and_then(|l| field(l, "ctx"));
        let same_ctx = default_ctx.is_none_or(|c| named == Some(c));
        let loaded = load.word("arch")? == "nemotron_h"
            && load.u64("slots")? == 1
            && load.u64("layers")? == layers
            && listen.u64("slots")? == 1;
        let ctx_is_plans =
            named.is_some() && named == Some(load.u64("ctx")?) && named == Some(listen.u64("ctx")?);
        let in_order = load.at() < listen.at();
        println!(
            "serve at {addr}: ctx line {:?} (the plan run's {same_ctx}); load {:?}; listening \
             {:?}; load names arch, one slot and {layers} layers {loaded}; ctx is the line's \
             {ctx_is_plans}; load before listening {in_order}",
            ctx_line(&text),
            load.line(),
            listen.line()
        );
        check(
            ok,
            "load_and_listen",
            same_ctx && loaded && ctx_is_plans && in_order,
        );
        let url = |p: &str| format!("http://{addr}{p}");
        let prose = tokenized(&url, PROSE)?;
        let messages = json!([{ "role": "user", "content": TURN }]);
        let turn = rendered(&url, messages.clone())?;
        // The clauses that compare with the session (`completion_ids`) read these two, each from
        // an empty cache.
        let (prose_answer, _) = greedy(&url, prose.clone(), N, &d, "prose", false)?;
        let (turn_answer, turn_body) = greedy(&url, turn.clone(), N, &d, "turn", false)?;
        // A clause whose requests fail is that clause's FAIL, printed; the gate goes on.
        for (name, run) in [
            (
                "chat_is_200",
                chat_is_those_ids(&url, &d, messages, N, (&turn_answer, &turn_body)),
            ),
            ("edit_resend_is_a_fresh_run", edit_resend(&url, &d, &prose)),
            (
                "extension_keeps_the_held_sequence",
                extension(&url, &d, &prose, &prose_answer),
            ),
        ] {
            let pass = run.unwrap_or_else(|e| {
                println!("{name}: {e}");
                false
            });
            check(ok, name, pass);
        }
        println!("server stopped: {}", s.stop()?);
        Ok(Some(Answers {
            prose,
            turn,
            prose_answer,
            turn_answer,
        }))
    }

    /// `edit_resend_is_a_fresh_run` (the module header): the slot holds the prose prompt and its
    /// answer, and the same prompt with its second half changed is sent.
    fn edit_resend(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        prose: &[u32],
    ) -> Result<bool, GateError> {
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "prose-held", true)?;
        let changed = edited(prose);
        let (fresh, _) = greedy(url, changed.clone(), N, dir, "edit-fresh", false)?;
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "prose-held-again", true)?;
        let (resent, _) = greedy(url, changed, N, dir, "edit-resent", true)?;
        println!(
            "edit resend: ids equal the fresh run's {}; cache_n {} is 0",
            resent.tokens == fresh.tokens,
            resent.cache_n
        );
        Ok(resent.tokens == fresh.tokens && resent.cache_n == 0)
    }

    /// `extension_keeps_the_held_sequence` (the module header): the slot holds the prose prompt
    /// and its answer (`held`'s tokens, from an empty cache), and the prompt with that answer
    /// appended is sent.
    fn extension(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        prose: &[u32],
        held: &Answer,
    ) -> Result<bool, GateError> {
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "extension-held", true)?;
        let mut longer = prose.to_vec();
        longer.extend(&held.tokens);
        let (fresh, _) = greedy(url, longer.clone(), N, dir, "extension-fresh", false)?;
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "extension-held-again", true)?;
        let (resent, _) = greedy(url, longer, N, dir, "extension-resent", true)?;
        println!(
            "extension: ids equal the fresh run's {}; cache_n {} is above 0",
            resent.tokens == fresh.tokens,
            resent.cache_n
        );
        Ok(resent.tokens == fresh.tokens && resent.cache_n > 0)
    }

    /// `completion_ids` (the module header), the server stopped and the session on the card.
    fn completion_ids(file: &Path, a: &Answers, ok: &mut bool) -> Result<(), GateError> {
        let ids = session_ids(file, &[a.prose.as_slice(), a.turn.as_slice()])?;
        let prose = agree(&a.prose_answer.tokens, &a.prose_answer.stop, &ids[0]);
        let turn = agree(&a.turn_answer.tokens, &a.turn_answer.stop, &ids[1]);
        println!(
            "completion ids: the prose prompt's {} tokens are the session's {prose}; the chat \
             turn's {} tokens are the session's {turn}",
            a.prose_answer.tokens.len(),
            a.turn_answer.tokens.len()
        );
        check(ok, "completion_ids", prose && turn);
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let dir = parse_args()?;
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(
            Hparams::read(&split)
                .map_err(|e| format!("{}: {e}", file.display()))?
                .n_layer,
        )?;
        drop(split);
        let mut ok = true;
        let default_ctx = plan_only_names_its_default(&dir, &file, &mut ok)?;
        parallel_above_one_is_refused(&dir, &file, &mut ok)?;
        let answers = served(&dir, &file, default_ctx, layers, &mut ok)?;
        if let Some(a) = answers {
            completion_ids(&file, &a, &mut ok)?;
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_nemotron_serve the seat plans its default and says so, refuses a second \
             slot, loads at the context it named, answers the session's ids, serves the chat turn, \
             and keeps a prefix only where the sequence stands"
        );
        Ok(())
    }
}
