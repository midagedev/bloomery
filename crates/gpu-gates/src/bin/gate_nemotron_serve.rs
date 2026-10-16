//! `gate_nemotron_serve` — the nemotron seat of `bloomery-serve` against the model it serves,
//! on the nemotron file of the profile (`BLOOMERY_REF_MODEL`): a `nemotron_h` file (Nemotron 3
//! Nano 4B Q4_K_M) or a `nemotron_h_moe` file (Nemotron 3.5 Bolt, the fixture under the fixture
//! tier). The file's `general.architecture` picks the body once, at `run` ([`Body`], the gate's
//! one dispatch point); any other architecture is refused by name, and every clause below is one
//! path over both bodies.
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
//!   names the file's `arch` word, `slots=1` and the file's layer count (the body's hparams), and
//!   its `ctx` and the `listening` record's are the `ctx` line's: the context the plan chose is the
//!   context the load made. FAIL-first: a seat whose load takes a context other than the plan's (a
//!   fixed one, or the slot context halved) is red.
//! - `completion_ids`: `/completion` at temperature 0, [`N`] tokens, of [`PROSE`] (`/tokenize`)
//!   and of the chat turn the server's own template renders (`/apply-template`, `/tokenize`),
//!   each from an empty cache (`cache_prompt: false`), answers the ids the file's body gives
//!   directly on the same split the server feeds the seat — the Nano's several-id head through
//!   its batched prompt call and the last id as a step, the Bolt (no batched prefill) every id a
//!   step — then each answer fed back, or a prefix of them ending at an end-of-generation id.
//!   FAIL-first: a prefill that skips the first id moves the ids and the clause is red.
//! - `chat_is_200`: `/v1/chat/completions` of the same turn at temperature 0 and `max_tokens`
//!   [`N`] is a 200 whose `usage` counts the prompt's ids and the completion's tokens, and whose
//!   message text is inside the text of the chat run's ids.
//! - `edit_resend_is_a_fresh_run`: the prose prompt answered with `cache_prompt: true`, then
//!   resent with its second half changed, answers the ids of the same request at `cache_prompt:
//!   false` and keeps nothing (`cache_n` 0): the shared prefix is shorter than the held sequence
//!   and the state cannot step back, so the slot is reset. FAIL-first: a `keep` that grants the
//!   shared prefix reaches the refused rollback, the request fails and the clause is red.
//! - `extension_keeps_the_held_sequence`: the prose prompt resent with its answer appended keeps
//!   the held positions (`cache_n` above 0): the shared prefix is the whole held sequence.
//! - `extension_is_the_held_run`: that resent answer is the body's run of the same split, the
//!   server stopped: the prose prompt and its answer held, then the answer's last id and [`N`]
//!   more. The Nano's held split (its batched prompt call, then steps) differs from a fresh one
//!   pass over the longer prompt by the q8 rounding of the activations at the Q8_0 sites, so the
//!   split is the reference; the Bolt holds and fresh-runs the same steps, equal exactly.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu::{Gpu, GpuError, GpuModel, NemotronHModel, NemotronHMoeModel};
    use bloomery_gpu_gates::ctx_ladder::{ctx_line, default_line_held, field, rung_above};
    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Answer, Served, agree, chat_is_those_ids, check, greedy, plan_run, rendered, serve_cmd,
        server_log, tokenized,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, generate, ref_model_path};
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::nemotron_h::hparams::Hparams as NanoHparams;
    use model::arch::nemotron_h_moe::hparams::Hparams as BoltHparams;
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
    ) -> Result<(Option<u64>, bool), GateError> {
        let mut ok = true;
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
            &mut ok,
            "plan_only_names_its_default",
            exits && in_order && word_a && no_load && default_ok,
        );
        Ok((ctx_line(&text).and_then(|l| field(l, "ctx")), ok))
    }

    /// `parallel_above_one_is_refused` (the module header).
    fn parallel_above_one_is_refused(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let mut ok = true;
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
            &mut ok,
            "parallel_above_one_is_refused",
            !status.success() && named && no_plan,
        );
        Ok(ok)
    }

    /// `n` greedy ids from where `m` stands after `first`, each answer fed back.
    fn continue_from<B: ChainBody>(
        m: &mut GpuModel<B>,
        first: u32,
        n: usize,
    ) -> Result<Vec<u32>, GateError> {
        let mut answer = vec![first];
        while answer.len() < n {
            let last = *answer.last().ok_or("an empty answer")?;
            answer.push(m.step(&[last])?);
        }
        Ok(answer)
    }

    /// The prompt `ids` as the server feeds the seat, from where `m` stands: every id but the
    /// last through the body's prompt call (`feed_head`: the Nano's batched prefill, the Bolt a
    /// step an id), the last as the first `next` step; then [`N`] greedy ids.
    fn seat_answer<B: ChainBody>(
        m: &mut GpuModel<B>,
        ids: &[u32],
        feed_head: fn(&mut GpuModel<B>, &[u32]) -> Result<(), GpuError>,
    ) -> Result<Vec<u32>, GateError> {
        let (&last, head) = ids.split_last().ok_or("an empty prompt")?;
        feed_head(m, head)?;
        let first = m.step(&[last])?;
        continue_from(m, first, N)
    }

    /// The Nano's head feed: the seat's prompt call — one id a step, several the batched
    /// [`GpuModel::prefill`].
    fn nano_head(m: &mut NemotronHModel, head: &[u32]) -> Result<(), GpuError> {
        match head.len() {
            0 => Ok(()),
            1 => m.step(head).map(|_| ()),
            _ => m.prefill(head).map(|_| ()),
        }
    }

    /// The Bolt's head feed: the body has no batched prefill, so the seat steps every prompt id.
    /// An empty head is nothing fed.
    fn bolt_head<B: ChainBody>(m: &mut GpuModel<B>, head: &[u32]) -> Result<(), GpuError> {
        for &id in head {
            m.step(&[id])?;
        }
        Ok(())
    }

    /// What the body's model gives directly, the server stopped: for each prompt from an empty
    /// cache, its [`N`] answer ids (the prompt through `feed_head`, the way the seat feeds that
    /// body, then each answer fed back); and the extension's run, the prose prompt and its answer
    /// held then that answer's last id fed and [`N`] more ids, the split the server's kept slot
    /// runs.
    fn session_ids<B: ChainBody>(
        file: &Path,
        prompts: &[&[u32]],
        prose: &[u32],
        open: fn(Gpu, Split, usize) -> Result<GpuModel<B>, GpuError>,
        feed_head: fn(&mut GpuModel<B>, &[u32]) -> Result<(), GpuError>,
    ) -> Result<(Vec<Vec<u32>>, Vec<u32>), GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let mut m = open(Gpu::new()?, split, SESSION_CTX)?;
        let mut out = Vec::new();
        for ids in prompts {
            m.reset()?;
            out.push(seat_answer(&mut m, ids, feed_head)?);
        }
        m.reset()?;
        let held = seat_answer(&mut m, prose, feed_head)?;
        let last = *held.last().ok_or("an empty held answer")?;
        let first = m.step(&[last])?;
        Ok((out, continue_from(&mut m, first, N)?))
    }

    /// The file's body, read once from its `general.architecture` at `run` — the gate's one
    /// dispatch point. Every clause is one path over both; what the body picks is the session's
    /// model, the way the seat feeds it a prompt, and the hparams the layer count reads.
    #[derive(Clone, Copy)]
    enum Body {
        /// `nemotron_h`, a Nemotron 3 Nano file: the seat's prefill one `GpuModel::step` over the
        /// ids.
        Nano,
        /// `nemotron_h_moe`, a Nemotron 3.5 Bolt file: the body has no batched prefill, so the
        /// seat steps every id.
        Bolt,
    }

    impl Body {
        /// The body `file` declares, any other architecture refused by name.
        fn of(file: &Path, split: &Split) -> Result<Body, GateError> {
            let first = split
                .shard(0)
                .ok_or_else(|| format!("{} opened with no shard", file.display()))?;
            match Arch::detect(first).map_err(|e| format!("{}: {e}", file.display()))? {
                Arch::NemotronH => Ok(Body::Nano),
                Arch::NemotronHMoe => Ok(Body::Bolt),
                arch => Err(format!(
                    "{} is a {} file; the nemotron seat serves nemotron_h and nemotron_h_moe \
                     files",
                    file.display(),
                    arch.name(),
                )
                .into()),
            }
        }

        /// The architecture's name, the `load` record's `arch` word.
        fn name(&self) -> &'static str {
            match self {
                Body::Nano => "nemotron_h",
                Body::Bolt => "nemotron_h_moe",
            }
        }

        /// The file's layer count, the body's own hparams.
        fn layers(&self, file: &Path, split: &Split) -> Result<u64, GateError> {
            let in_file = |e| format!("{}: {e}", file.display());
            let n = match self {
                Body::Nano => NanoHparams::read(split).map_err(in_file)?.n_layer,
                Body::Bolt => BoltHparams::read(split).map_err(in_file)?.n_layer,
            };
            Ok(u64::try_from(n)?)
        }

        /// The ids the body's model gives directly ([`session_ids`]), the prompt fed the way the
        /// seat feeds that body, and the extension's run of the held split.
        fn session_ids(
            &self,
            file: &Path,
            prompts: &[&[u32]],
            prose: &[u32],
        ) -> Result<(Vec<Vec<u32>>, Vec<u32>), GateError> {
            match self {
                Body::Nano => session_ids(file, prompts, prose, NemotronHModel::open, nano_head),
                Body::Bolt => session_ids(file, prompts, prose, NemotronHMoeModel::open, bolt_head),
            }
        }
    }

    /// What the server answered, held until it is stopped and the session has the card.
    struct Answers {
        prose: Vec<u32>,
        turn: Vec<u32>,
        prose_answer: Answer,
        turn_answer: Answer,
        /// The extension's resent answer, held to the session's run of the same split.
        resent: Option<Answer>,
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
        body: Body,
        layers: u64,
    ) -> Result<(Option<Answers>, bool), GateError> {
        let mut ok = true;
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
                check(&mut ok, "load_and_listen", false);
                return Ok((None, ok));
            }
        };
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let records = server_log(&err_log, record::BLOOMERY_SERVE_NEMOTRON)?;
        let load = records.one(&record::LOAD_NEMOTRON)?;
        let listen = records.one(&record::LISTENING_NEMOTRON)?;
        let named = ctx_line(&text).and_then(|l| field(l, "ctx"));
        let same_ctx = default_ctx.is_none_or(|c| named == Some(c));
        let loaded = load.word("arch")? == body.name()
            && load.u64("slots")? == 1
            && load.u64("layers")? == layers
            && listen.u64("slots")? == 1;
        let ctx_is_plans =
            named.is_some() && named == Some(load.u64("ctx")?) && named == Some(listen.u64("ctx")?);
        let in_order = load.at() < listen.at();
        println!(
            "serve at {addr}: ctx line {:?} (the plan run's {same_ctx}); load {:?}; listening \
             {:?}; load names arch={}, one slot and {layers} layers {loaded}; ctx is the line's \
             {ctx_is_plans}; load before listening {in_order}",
            ctx_line(&text),
            load.line(),
            listen.line(),
            body.name()
        );
        check(
            &mut ok,
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
        ] {
            let pass = run.unwrap_or_else(|e| {
                println!("{name}: {e}");
                false
            });
            check(&mut ok, name, pass);
        }
        let resent = match extension(&url, &d, &prose, &prose_answer) {
            Ok(r) => Some(r),
            Err(e) => {
                println!("extension_keeps_the_held_sequence: {e}");
                None
            }
        };
        check(
            &mut ok,
            "extension_keeps_the_held_sequence",
            resent.as_ref().is_some_and(|r| r.cache_n > 0),
        );
        println!("server stopped: {}", s.stop()?);
        Ok((
            Some(Answers {
                prose,
                turn,
                prose_answer,
                turn_answer,
                resent,
            }),
            ok,
        ))
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

    // PIN(2026-10-16): the kept slot's answer is held to the run of its held split
    // (`extension_is_the_held_run`), no longer to a fresh run of the longer prompt: since the
    // prompt path runs passes, a fresh run is one pass over all of it, and the held split's steps
    // differ from it by the q8 rounding of the activations at the Q8_0 sites (the prose prompt's
    // resent and fresh answers part on the batched path).
    /// `extension_keeps_the_held_sequence` (the module header): the slot holds the prose prompt
    /// and its answer (`held`'s tokens, from an empty cache), and the prompt with that answer
    /// appended is sent. Its answer, kept for `extension_is_the_held_run`.
    fn extension(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        prose: &[u32],
        held: &Answer,
    ) -> Result<Answer, GateError> {
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "extension-held", true)?;
        let mut longer = prose.to_vec();
        longer.extend(&held.tokens);
        let (resent, _) = greedy(url, longer, N, dir, "extension-resent", true)?;
        println!("extension: cache_n {} is above 0", resent.cache_n);
        Ok(resent)
    }

    /// `completion_ids` (the module header), the server stopped and the session on the card.
    fn completion_ids(file: &Path, body: Body, a: &Answers) -> Result<bool, GateError> {
        let mut ok = true;
        let (ids, extended) =
            body.session_ids(file, &[a.prose.as_slice(), a.turn.as_slice()], &a.prose)?;
        let prose = agree(&a.prose_answer.tokens, &a.prose_answer.stop, &ids[0]);
        let turn = agree(&a.turn_answer.tokens, &a.turn_answer.stop, &ids[1]);
        println!(
            "completion ids: the prose prompt's {} tokens are the session's {prose}; the chat \
             turn's {} tokens are the session's {turn}",
            a.prose_answer.tokens.len(),
            a.turn_answer.tokens.len()
        );
        check(&mut ok, "completion_ids", prose && turn);
        let held = a
            .resent
            .as_ref()
            .is_some_and(|r| agree(&r.tokens, &r.stop, &extended));
        println!(
            "extension: the resent answer is the session's run of the held split (the prose \
             prompt and its answer, the answer's last id, {N} more) {held}"
        );
        check(&mut ok, "extension_is_the_held_run", held);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let dir = parse_args()?;
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let body = Body::of(&file, &split)?;
        let layers = body.layers(&file, &split)?;
        drop(split);
        let mut ok = true;
        let (default_ctx, pass) = plan_only_names_its_default(&dir, &file)?;
        ok &= pass;
        ok &= parallel_above_one_is_refused(&dir, &file)?;
        let (answers, pass) = served(&dir, &file, default_ctx, body, layers)?;
        ok &= pass;
        if let Some(a) = answers {
            ok &= completion_ids(&file, body, &a)?;
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
