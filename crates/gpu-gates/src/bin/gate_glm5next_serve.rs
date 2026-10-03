//! `gate_glm5next_serve` — the GLM seat of `bloomery-serve` (`--model glm`)
//! on the 3090 (placement gate), driven over HTTP, against
//! `generate_glm5next` on the same card, as one process under the GPU gate
//! lock.
//!
//!     gate_glm5next_serve --arm plain|drafted --dir <out>
//!
//! Each arm starts the server beside this binary (`--model glm --host
//! 127.0.0.1 --port 0 --place gate --ctx 2048 --slot-save-path /tmp
//! --parallel 1`, the plain engine its clauses hold; the levers are set
//! here), reads
//! its address from its stderr, waits for `/health`, takes the prompt of one
//! chat turn ([`CHAT`]) as the server renders it (`/apply-template`, then
//! `/tokenize` of that text without BOS: the ids `/v1/chat/completions`
//! encodes), checks the server, kills it by the handle this binary spawned it
//! with and waits for it, then runs `generate_glm5next --place gate --ctx
//! 2048 --tokens <those ids> -n 16` beside this binary under the same levers
//! and holds the ids the server served against it. The loads run one after
//! the other (three in `plain`, two in `drafted`); the recipe runs the arms
//! as two processes, each under its own bound.
//!
//! `plain` (`BLOOMERY_DRAFT=off BLOOMERY_RESIDENCY=off`), after three runs of
//! the seat with both levers unset and `--plan`, which print the unset rule's
//! records and the plan and exit 0 before any card is opened: under `--place
//! a` and under `--place bp` (plan (b′), the 3090 an expert tier) a `draft
//! unset draft=mtp` and a `residency unset residency=mid-p0-s1` record and
//! the `residency host` record of the word — its pinned count 0 and the
//! headroom past the churn pool non-negative, the pool inside the plan's
//! host terms — every placement's `ctx` line naming a default context at or
//! past the floor; under `--place gate` both `off`, no `residency host`:
//!
//! - the seat's `--ctx` rule ([`ctx`], on the gate card): the default's `ctx`
//!   line against this gate's own plans of the file — the rule's context,
//!   its fit and its margin — the flag winning below the floor, and a
//!   context past `place::ORACLE_POSITIONS` refused by name before it
//!   listens;
//!
//! - the load prints `load draft=off (BLOOMERY_DRAFT=off)` and the word as a
//!   `residency lever` record (`why=set`), no `residency host` and no verify
//!   capture; `/props`' `engine` carries no `draft` and no card `draft` class;
//! - the chat turn at temperature 0 is a 200;
//! - `/completion` of the prompt's ids at temperature 0 with `return_tokens`:
//!   its ids are the CLI's `tokens` line — all 16, or a prefix ending in the
//!   end-of-generation id when the server stopped there (the CLI does not
//!   stop at it). Exact: the server feeds the prompt less its last id, then
//!   one step, and so does the CLI (`--last-step`) — the same batches, so
//!   the same bits (a batch past a chunk runs the GEMM, not the steps'
//!   gemvs);
//! - the server's `load` record and the CLI's, both under
//!   `BLOOMERY_PREFILL_GROUP` unset, print `group=` the group the plan
//!   reserves the prompt units for (`place::PROMPT_GROUP`);
//! - `POST /residency/reset` is the server's 501 (no machine);
//! - `cache` ([`cache`]): two sessions through the prompt cache, A's state
//!   saved when B takes the slot and put back when A returns — every held
//!   position kept and the ids of the same requests with no switch, and a
//!   resend that shares only A's turn keeps the checkpoint the state carried
//!   there, with the no-switch run's ids;
//! - then a third load, the server under `BLOOMERY_DRAFT=mtp
//!   BLOOMERY_RESIDENCY=off`: it prints `load draft=mtp` and no `residency
//!   host`, the same `/completion` carries the draft's counts, and its ids
//!   are the plain CLI's, all of them; a sampled `/completion` cut to
//!   `top_k` 1 (temperature 0.8, a fixed seed) is served through plain steps,
//!   drafting nothing, each step's row read before the step is told to the
//!   draft, and its ids are the plain CLI's too. Green-only for the row's
//!   read: the NextN walk writes no row of the target's head, so a `step_row`
//!   left at the default reads the same row; the clause holds the sampled
//!   path's ids end to end; and `cache` again under the draft, where the
//!   draft also rejoins A where A left it, its counts the no-switch run's.
//!   Exact: with no residency nothing
//!   moves between the host and the card, every kept token is the target's
//!   own argmax, and a verify's rows are its steps' bits — the window's
//!   wiring end to end, which the drafted arm's clause holds only up to the
//!   first landing.
//!
//! `drafted` (`BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=mid-p0-s1`, the clip's
//! levers; the word set explicitly, one the gate plan's card slots take):
//!
//! - the load prints `load draft=mtp`, the verify's capture
//!   (`pair_graph_nodes`), the word as a `residency lever` record (`why=set`)
//!   and a `residency host` record of it; `/props`' `engine.draft` is `mtp`
//!   of width 1 naming the target file, and the card's `draft` class is the
//!   bytes this gate's own NextN plan of the file reserves for the layer
//!   (its card terms and arena);
//! - `/completion` of the prompt's ids from the seed (`cache_prompt` off)
//!   carries the draft's counts (`draft_n` and `draft_n_accepted` above 0)
//!   and ends, in its `residency pass` records, a prompt pass, a step pass
//!   (the prompt's last id) and at least one `pair` pass (a window that
//!   proposed) (mutants: the seat drives plain steps on the NextN load — no
//!   counts, no `pair` pass, no `engine.draft`; the seat opens the load
//!   without the word — no `residency pass` record, and the reset below is a
//!   501);
//! - `POST /residency/reset` is a 200 whose `diff` is 0, and the server
//!   prints its `residency reset` record with the same counts; the same
//!   `/completion` after it gives the first one's ids (from the seed, one
//!   history lands the same flips at the same passes, so two runs of it are
//!   bit for bit);
//! - the chat turn at temperature 0 drafts (`draft_n` above 0);
//! - a state put back runs on the slot map as it stands
//!   ([`resume_under_residency`]): kept whole, its passes logged, the draft
//!   rejoining;
//! - a decode preempted mid-run on `--parallel 2` is put back the same way
//!   ([`swap_rejoins_the_draft`]): both requests answer their solo runs' ids,
//!   their drafts run and skip nothing, the switches counter moved and the
//!   second request came back before the first;
//! - against the CLI under the same levers: the ids agree, all of them or a
//!   prefix ending in the end-of-generation id — the CLI takes the server's
//!   cut (`--last-step`), so both run the same batches, count the rule's
//!   steps the same and land their flips at the same passes; through which
//!   pass the first landing sits is printed, not held (the reset clause
//!   above holds the seed's determinism).
//!
//! A request that extends the held sequence (the prompt and the ids it
//! generated) is printed with its `cache_n` and the draft's `mtp prompt`
//! records, not held: the checkpoint rule keeps every fed position only for
//! a request that extends all of them.
//!
//! Logs and the CLI's output go to `--dir`. The server and the CLI inherit
//! this binary's environment, so the levers they act on are the seat's
//! (`ACTS_ON`, the same list): one the seat would refuse is refused here, at
//! `main`, before anything starts; `BLOOMERY_DRAFT` and `BLOOMERY_RESIDENCY`
//! set in this binary's environment are refused by name (the arm sets both).

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_serve: built without the `glm5next` feature; see `just gate-gpu-glm5next-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_serve", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::fs::File;
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::serve_client::{curl, ids_of, json_of, parse_ids};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::glm5next::place::{NextnInputs, ORACLE_POSITIONS, PROMPT_GROUP, PlanInputs};
    use model::placement::{PlanLevers, workstation};
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    /// The levers the seat acts on — the same list `serve_seats::glm`
    /// parses, kept one with it: this gate starts the server with its own
    /// environment, so a lever the server would refuse is refused here first.
    const ACTS_ON: &[&str] = &[
        bloomery_levers::CARD_BUDGET,
        bloomery_levers::HOST_POPULATE,
        bloomery_levers::HOST_LOCK,
        bloomery_levers::CARD_DONTNEED,
        bloomery_levers::R8,
        bloomery_levers::PIN_MAIN,
        bloomery_levers::DRAFT,
        bloomery_levers::RESIDENCY,
    ];

    const USAGE: &str = "usage: gate_glm5next_serve --arm plain|drafted --dir <out>";

    /// The stores both engines size, the seat's floor and the CLI's default.
    const CTX: usize = 2048;
    /// The seat's default context's multiple (its `CTX_STEP`).
    const CTX_STEP: usize = 256;
    /// The flag arm's context, below the seat's floor: a flag seen to win,
    /// not to coincide with a default the floor already takes.
    const FLAG_CTX: usize = 1536;
    /// A context past `ORACLE_POSITIONS`, the refusal the clause asks for.
    const OVER_CTX: usize = 99_999;
    /// The `ctx` clause's polls for the seat's pre-load lines, 1 s apart:
    /// the lines print at plan time, before the load's first-start JIT.
    const LINE_POLLS: usize = 90;
    const LINE_POLL: Duration = Duration::from_secs(1);
    /// The server's arguments after its path. The slot actions need a save
    /// directory; the gate asks only for `erase`, which writes nothing. The
    /// plain engine is pinned (`--parallel 1`): this gate's clauses hold the
    /// one-slot path's prompt cache, and the swap clause's server runs the
    /// turns on its own.
    const SERVER_ARGS: [&str; 14] = [
        "--model",
        "glm",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "gate",
        "--ctx",
        "2048",
        "--slot-save-path",
        "/tmp",
        "--parallel",
        "1",
    ];
    /// The load takes a minute or two; the bound is 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    /// The greedy requests' length, the CLI's `-n`.
    const N_PREDICT: usize = 16;
    /// The swap clause's requests: the first long enough to be preempted
    /// mid-decode, the second short enough to come back before it.
    const SWAP_A_PREDICT: usize = 48;
    const SWAP_B_PREDICT: usize = 8;
    /// The swap clause's `/slots` poll while it waits for the first request's
    /// decode.
    const SWAP_POLL: Duration = Duration::from_millis(300);
    /// The one chat turn.
    const CHAT: &str = "What is the capital of France? Answer in one word.";
    /// The `cache` clause's sessions: A's first turn, B's turn, the text A
    /// sends after its reply, and the text A resends in place of its reply
    /// (its first id is not the reply's).
    const TURN_A: &str = "Name three rivers that flow through Germany and say in one sentence \
                          which of them is the longest.";
    const TURN_B: &str = "Write a haiku about a lighthouse in winter.";
    const LATER_A: &str = " And which of them reaches the sea first?";
    const STRIPPED_A: &str = "Rhine, Danube and Elbe; the Danube is the longest. And which of \
                              them reaches the sea first?";
    /// The drafted arm's residency word: no seed expert pinned, one spare a
    /// layer.
    const RESIDENCY_WORD: &str = "mid-p0-s1";

    /// The arms, one a process.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        Plain,
        Drafted,
    }

    impl Arm {
        fn parse(v: &str) -> Result<Arm, GateError> {
            match v {
                "plain" => Ok(Arm::Plain),
                "drafted" => Ok(Arm::Drafted),
                other => Err(format!("--arm is plain or drafted, not {other}").into()),
            }
        }
    }

    /// The levers a process runs under: `BLOOMERY_DRAFT` and
    /// `BLOOMERY_RESIDENCY` set (both, on the server and the CLI alike), or
    /// both unset ([`UNSET`]).
    type Levers = &'static [(&'static str, &'static str)];
    /// The plain arm's server and CLI.
    const PLAIN: Levers = &[
        (bloomery_levers::DRAFT, "off"),
        (bloomery_levers::RESIDENCY, "off"),
    ];
    /// The plain arm's drafted server: the draft with no residency.
    const DRAFT_ONLY: Levers = &[
        (bloomery_levers::DRAFT, "mtp"),
        (bloomery_levers::RESIDENCY, "off"),
    ];
    /// The drafted arm's server and CLI.
    const DRAFTED: Levers = &[
        (bloomery_levers::DRAFT, "mtp"),
        (bloomery_levers::RESIDENCY, RESIDENCY_WORD),
    ];
    /// Both levers unset: the seat's own rule.
    const UNSET: Levers = &[];

    /// The server this binary started; killed and reaped on every way out.
    /// `serve_client::Served`'s core with this gate's server's name — that
    /// module's `exe` names `bloomery-serve-ds41`.
    struct Served {
        child: Child,
    }

    impl Served {
        /// `bloomery-serve` beside this binary with `args` and the arm's
        /// levers, stdout to `<dir>/server.out` and stderr to
        /// `<dir>/server.err`. The child is killed when this process dies, so
        /// a runner's bound that ends this process does not leave the server
        /// holding a card.
        fn spawn(args: &[&str], dir: &Path, levers: Levers) -> Result<Served, GateError> {
            let exe = beside("bloomery-serve")?;
            let mut cmd = Command::new(&exe);
            cmd.args(args)
                .env_remove(bloomery_levers::DRAFT)
                .env_remove(bloomery_levers::RESIDENCY)
                .envs(levers.iter().copied())
                .stdin(Stdio::null())
                .stdout(File::create(dir.join("server.out"))?)
                .stderr(File::create(dir.join("server.err"))?);
            // SAFETY: the closure runs in the child between fork and exec and
            // calls only `prctl`, which is async-signal-safe and touches no
            // memory of ours.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
            let child = cmd
                .spawn()
                .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
            Ok(Served { child })
        }

        /// Waits for the `listening on http://<addr>` line in the server's
        /// stderr, `polls` reads `poll` apart.
        fn address(&mut self, err_log: &Path) -> Result<String, GateError> {
            for _ in 0..POLLS {
                let text = std::fs::read_to_string(err_log).unwrap_or_default();
                if let Some(addr) = text
                    .lines()
                    .find_map(|l| l.split_once("listening on http://").map(|(_, a)| a.trim()))
                {
                    return Ok(addr.to_owned());
                }
                if let Some(status) = self.child.try_wait()? {
                    return Err(format!(
                        "the server exited ({status}) before listening; {}:\n{text}",
                        err_log.display()
                    )
                    .into());
                }
                std::thread::sleep(POLL);
            }
            Err(format!("the server did not listen within {POLLS} polls").into())
        }

        fn stop(&mut self) -> Result<String, GateError> {
            self.child.kill()?;
            Ok(format!("{}", self.child.wait()?))
        }
    }

    impl Drop for Served {
        fn drop(&mut self) {
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// The binary `name` beside this one.
    fn beside(name: &str) -> Result<PathBuf, GateError> {
        Ok(std::env::current_exe()?.with_file_name(name))
    }

    struct Args {
        arm: Arm,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut arm, mut dir) = (None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--arm" => arm = Some(Arm::parse(&v)?),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (arm, dir) {
            (Some(arm), Some(dir)) => Ok(Args { arm, dir }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// The lines of the server's stderr from line `from` on.
    fn lines_from(err_log: &Path, from: usize) -> Result<Vec<String>, GateError> {
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .skip(from)
            .map(str::to_owned)
            .collect())
    }

    /// The first line of `lines` that starts with `head` and a space.
    fn record<'a>(lines: &'a [String], head: &str) -> Option<&'a str> {
        let prefix = format!("{head} ");
        lines
            .iter()
            .find(|l| l.starts_with(&prefix))
            .map(String::as_str)
    }

    /// The `group=` of the `load` record in `lines` (the generator's, which
    /// opens with `resident_bytes=`).
    fn load_group(lines: &[String]) -> Option<&str> {
        lines
            .iter()
            .find(|l| l.starts_with("load resident_bytes="))
            .and_then(|l| field(l, "group"))
    }

    /// The value of `key=` in a record line.
    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(key)?.strip_prefix('='))
    }

    /// The `residency pass` records of `lines`: each pass's kind, the rows
    /// it kept and the flips that went live at its boundary.
    fn passes(lines: &[String]) -> Vec<(String, u64, u64)> {
        lines
            .iter()
            .filter(|l| l.starts_with("residency pass "))
            .filter_map(|l| {
                let n = |k| field(l, k).and_then(|v| v.parse::<u64>().ok());
                Some((field(l, "pass")?.to_owned(), n("kept")?, n("landed")?))
            })
            .collect()
    }

    /// The tokens a run produced through the first pass whose boundary
    /// landed a flip (every token, when none did): `first` the tokens before
    /// the first pass's rows count — with the server's cut both runs' prompt
    /// calls give no token, their step token 0 — then each pass's kept rows,
    /// that pass's included — it ran before its flips went live.
    fn before_landing(passes: &[(String, u64, u64)], first: u64) -> u64 {
        let mut n = first;
        for (_, kept, landed) in passes {
            n += kept;
            if *landed > 0 {
                return n;
            }
        }
        u64::MAX
    }

    /// The prompt of the chat turn as the server renders and encodes it.
    fn chat_ids(url: &dyn Fn(&str) -> String) -> Result<Vec<u32>, GateError> {
        let messages = json!({"messages": [{"role": "user", "content": CHAT}]});
        let (st, body) = curl(&url("/apply-template"), Some(&messages), false)?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": text, "add_special": false})),
            false,
        )?;
        let ids = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("chat prompt {text:?} ids {ids:?}");
        if ids.is_empty() {
            return Err("the chat prompt tokenizes to no id".into());
        }
        Ok(ids)
    }

    /// One greedy `/completion` of `ids`, `n` tokens: the response.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        ids: &[u32],
        n: usize,
        cache: bool,
    ) -> Result<Value, GateError> {
        let body = json!({
            "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
            "cache_prompt": cache,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        json_of("/completion", st, &text)
    }

    /// The chat turn at temperature 0, from a reset (`cache_prompt` off: a
    /// prefix kept at a checkpoint below the held sequence's end leaves the
    /// draft nothing to join, and the request would step plainly): its status
    /// and `timings`.
    fn chat(url: &dyn Fn(&str) -> String) -> Result<(u16, Value), GateError> {
        let body = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": N_PREDICT, "cache_prompt": false,
        });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        println!(
            "chat HTTP {st}: message {} finish_reason {} usage {} timings {}",
            v["choices"][0]["message"], v["choices"][0]["finish_reason"], v["usage"], v["timings"]
        );
        Ok((st, v["timings"].clone()))
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

    /// `generate_glm5next --place gate --ctx 2048 --tokens <ids> -n 16`,
    /// with `--last-step` when `last_step` (the server's cut: the prompt less
    /// its last id, then a step), beside this binary under the arm's levers,
    /// stdout to `<dir>/gen.log` and stderr to `<dir>/gen.err`: its `tokens`
    /// line and its stdout.
    fn cli(
        dir: &Path,
        levers: Levers,
        ids: &[u32],
        last_step: bool,
    ) -> Result<(Vec<u32>, Vec<String>), GateError> {
        let exe = beside("generate_glm5next")?;
        let tokens = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let out = dir.join("gen.log");
        let mut args = vec![
            "--place", "gate", "--ctx", "2048", "--tokens", &tokens, "-n", "16",
        ];
        if last_step {
            args.push("--last-step");
        }
        let status = Command::new(&exe)
            .args(&args)
            .envs(levers.iter().copied())
            .stdin(Stdio::null())
            .stdout(File::create(&out)?)
            .stderr(File::create(dir.join("gen.err"))?)
            .status()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        if !status.success() {
            return Err(format!(
                "{} exited {status}; see {}",
                exe.display(),
                dir.join("gen.err").display()
            )
            .into());
        }
        let lines: Vec<String> = std::fs::read_to_string(&out)?
            .lines()
            .map(str::to_owned)
            .collect();
        let line = lines
            .iter()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", out.display()))?;
        let got = parse_ids(line)?;
        if let Some(s) = record(&lines, "mtp summary") {
            println!("generate_glm5next {s}");
        }
        Ok((got, lines))
    }

    /// `/props`' `engine` object: the draft it names, and the card's `draft`
    /// class, against `bytes`, this gate's NextN plan's bytes for the layer
    /// (`None`: no draft).
    fn props_draft(url: &dyn Fn(&str) -> String, bytes: Option<u64>) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!(
            "props engine draft {} placement {}",
            e["draft"], e["placement"]
        );
        let classes: Vec<Option<u64>> = e["placement"]["devices"]
            .as_array()
            .map(|d| {
                d.iter()
                    .filter(|d| d["device"].as_str().is_some_and(|n| n.starts_with("GPU")))
                    .map(|d| d["classes"]["draft"].as_u64())
                    .collect()
            })
            .unwrap_or_default();
        let path = ref_model_path()?.display().to_string();
        let mut ok = true;
        match bytes {
            None => check(
                &mut ok,
                "plain_props_carry_no_draft",
                e.get("draft").is_none()
                    && !classes.is_empty()
                    && classes.iter().all(Option::is_none),
            ),
            Some(b) => check(
                &mut ok,
                "drafted_props_name_the_nextn_draft",
                e["draft"]["kind"] == "mtp"
                    && e["draft"]["n_max"] == json!(1)
                    && e["draft"]["path"].as_str() == Some(path.as_str())
                    && classes == [Some(b)],
            ),
        }
        Ok(ok)
    }

    /// The bytes this gate's NextN plan of the file reserves for the layer
    /// on the gate card at [`CTX`]: its card terms and its arena.
    fn nextn_plan_bytes(levers: &bloomery_levers::Levers) -> Result<u64, GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let nextn = NextnInputs::read(&inputs)?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let plan = inputs.plan_nextn(
            &machine,
            u64::try_from(CTX)?,
            &PlanLevers::from_levers(levers)?,
            &nextn,
        )?;
        Ok(plan.nextn_card_bytes() + plan.arena_bytes)
    }

    /// `bloomery-serve --model glm --place <place> --plan` with both levers
    /// unset and no `--ctx`, so the default context rule runs: its exit
    /// status and stderr. It plans and exits before the load, so it opens no
    /// card.
    fn plan_only(dir: &Path, place: &str) -> Result<(bool, Vec<String>), GateError> {
        let exe = beside("bloomery-serve")?;
        let err = dir.join(format!("plan-{place}.err"));
        let status = Command::new(&exe)
            .args(["--model", "glm", "--place", place, "--plan"])
            .env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::RESIDENCY)
            .envs(UNSET.iter().copied())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(File::create(&err)?)
            .status()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        let lines = lines_from(&err, 0)?;
        println!("--place {place} --plan with the levers unset: {status}");
        for l in &lines {
            println!("  {l}");
        }
        Ok((status.success(), lines))
    }

    /// The unset rule's lines (the module header): `--place a` and `bp` draft
    /// and run the residency's default word, `--place gate` neither.
    fn unset_rule(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let picks = |lines: &[String], draft: &str, residency: &str| {
            let d = format!("draft unset draft={draft} why=");
            let r = format!("residency unset residency={residency} why=");
            lines.iter().any(|l| l.starts_with(&d))
                && lines.iter().any(|l| l.starts_with(&r))
                && !lines.iter().any(|l| l.contains("listening on"))
        };
        let (a_ok, a) = plan_only(dir, "a")?;
        check(
            &mut ok,
            "unset_place_a_drafts_and_runs_mid_p0_s1",
            a_ok && picks(&a, "mtp", RESIDENCY_WORD) && record(&a, "residency host").is_some(),
        );
        let (bp_ok, bp) = plan_only(dir, "bp")?;
        check(
            &mut ok,
            "unset_place_bp_drafts_and_runs_mid_p0_s1",
            bp_ok && picks(&bp, "mtp", RESIDENCY_WORD) && record(&bp, "residency host").is_some(),
        );
        let (g_ok, g) = plan_only(dir, "gate")?;
        check(
            &mut ok,
            "unset_place_gate_runs_neither",
            g_ok && picks(&g, "off", "off") && record(&g, "residency host").is_none(),
        );
        // Every placement's `ctx` line: a rule word of the default rule's,
        // its context at or past the floor while the card holds it (a card
        // that holds less plans fewer, `card`, as qwen38 does). FAIL-first:
        // a seat that keeps the floor silently, or prints no line, turns
        // this red.
        let ctx_line = |lines: &[String]| -> Option<(String, usize)> {
            let l = lines.iter().find(|l| l.starts_with("ctx rule="))?;
            Some((field(l, "rule")?.to_owned(), field(l, "ctx")?.parse().ok()?))
        };
        for (place, lines) in [("a", &a), ("bp", &bp), ("gate", &g)] {
            let named = ctx_line(lines).is_some_and(|(rule, ctx)| {
                matches!(rule.as_str(), "base" | "margin" | "fit" | "card") && ctx >= CTX
            });
            println!("--place {place}: a default ctx line, rule and context in {named}");
            check(&mut ok, "unset_plans_take_the_ctx_default", named);
        }
        // The default word's churn pool inside the plan's host terms: the
        // `residency host` record of a default load carries the word's
        // pinned count (0: no seed expert pinned, the pool every card expert
        // lands in) and the headroom left past it. FAIL-first: a default that
        // pins seed experts, or a pool past the headroom, turns this red.
        let pool = |lines: &[String]| -> Option<(u64, i128)> {
            let l = record(lines, "residency host")?;
            Some((
                field(l, "pinned")?.parse().ok()?,
                field(l, "headroom_after")?.parse().ok()?,
            ))
        };
        check(
            &mut ok,
            "unset_default_pool_is_p0_inside_the_headroom",
            [&a, &bp]
                .into_iter()
                .all(|lines| pool(lines).is_some_and(|(pinned, after)| pinned == 0 && after >= 0)),
        );
        // Every plan line names its stage card's free bytes at plan time, at
        // most its usable bytes — the census term the expert rule filled
        // within (memguard). FAIL-first: a plan line that drops it, or names
        // it past the card's usable bytes, turns this red.
        for (place, lines) in [("a", &a), ("bp", &bp), ("gate", &g)] {
            let named = record(lines, "plan").is_some_and(|l| {
                let free = l
                    .split("card_free=")
                    .nth(1)
                    .and_then(|t| t.split(' ').next())
                    .and_then(|n| n.parse::<u64>().ok());
                let usable = l
                    .split("devices=")
                    .nth(1)
                    .and_then(|d| d.split([',', ' ', ']']).next())
                    .and_then(|c| c.rsplit(':').next())
                    .and_then(|n| n.parse::<u64>().ok());
                free.is_some_and(|f| usable.is_some_and(|u| f <= u))
            });
            println!("--place {place}: card_free named and within usable {named}");
            check(&mut ok, "plan_names_the_cards_free_bytes", named);
        }
        Ok(ok)
    }

    /// The stage card's free bytes as a server's own `plan` line read them
    /// (the census reading its rule searched against).
    fn server_card_free(err_log: &Path) -> Result<Option<u64>, GateError> {
        Ok(lines_from(err_log, 0)?
            .into_iter()
            .find(|l| l.starts_with("plan "))
            .and_then(|l| {
                l.split("card_free=")
                    .nth(1)
                    .and_then(|t| t.split(' ').next())
                    .and_then(|n| n.parse::<u64>().ok())
            }))
    }

    /// The plan's stage-card expert bytes on the gate card at `ctx`, plain,
    /// as the server makes it; `None` when no plan takes the context.
    fn card_at(
        inputs: &PlanInputs,
        levers: &PlanLevers,
        ctx: usize,
        free: Option<u64>,
    ) -> Result<Option<u64>, GateError> {
        let mut machine = workstation::plan_gate(inputs.model.layers);
        machine.cards[0].free_bytes = free;
        Ok(inputs
            .plan(&machine, u64::try_from(ctx)?, levers)
            .ok()
            .and_then(|p| p.cards.first().map(|c| c.expert_bytes)))
    }

    /// The `ctx` clause (the module header): a plain server with no `--ctx`
    /// prints its `ctx` and `plan` lines before any load and is stopped
    /// there, against this gate's own plans of the file: its context is the
    /// rule's — [`CTX`] under `base`, the largest within the plan's margin
    /// under `margin`, the plan's largest under `fit`, the card's largest
    /// under `card`; the line's `trained` is the file's; its `fit` the
    /// largest context whose plan stands up to the trained context capped to
    /// `ORACLE_POSITIONS` — this gate's own plan taken, the one past it
    /// refused when the card bounds it, else the cap itself — with that
    /// plan's stage-card expert bytes; its `margin_ctx` at most
    /// `workstation::MARGIN` fewer of them than the plan at [`CTX`], and the
    /// next multiple of [`CTX_STEP`] past it more (or it is the fit). A
    /// server asked for [`FLAG_CTX`] takes it (below the floor, a value no
    /// default can coincide with); one asked past the oracle is refused by
    /// name before it listens. Every unset rule word must hold the relation
    /// it claims between the line's `ctx`, its `margin_ctx` and its `fit` —
    /// a default that keeps the floor while its margin reaches further
    /// names the wrong word. Mutants: the default taking the trained
    /// context past the fit (the server's own plan refuses it by name
    /// before any line); the search giving up (the line's `margin_ctx`
    /// keeps the floor while a step past it stays within it).
    fn ctx(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let own = dir.join("ctx");
        std::fs::create_dir_all(&own)?;
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let ctx_at = args
            .iter()
            .position(|&f| f == "--ctx")
            .ok_or("SERVER_ARGS names no --ctx")?;
        args.drain(ctx_at..ctx_at + 2);
        let wait_for = |served: &mut Served,
                        err_log: &Path,
                        want: &[&str]|
         -> Result<Vec<String>, GateError> {
            for _ in 0..LINE_POLLS {
                let text = std::fs::read_to_string(err_log).unwrap_or_default();
                if want.iter().all(|w| text.lines().any(|l| l.starts_with(w))) {
                    return Ok(text.lines().map(str::to_owned).collect());
                }
                if let Some(status) = served.child.try_wait()? {
                    return Err(format!(
                        "the server exited ({status}) before its lines; {}:\n{text}",
                        err_log.display()
                    )
                    .into());
                }
                std::thread::sleep(LINE_POLL);
            }
            Err(format!(
                "the server printed no {want:?} within {LINE_POLLS} polls; {}",
                err_log.display()
            )
            .into())
        };

        // The default arm: the rule's own line, the server stopped before
        // its load. Each arm's server takes its own directory — `spawn`
        // writes `server.err` inside the one it is given.
        let default = own.join("default");
        std::fs::create_dir_all(&default)?;
        let err_log = default.join("server.err");
        let mut served = Served::spawn(&args, &default, PLAIN)?;
        println!("ctx server pid {}", served.child.id());
        let lines = wait_for(&mut served, &err_log, &["ctx rule=", "plan "])?;
        println!("ctx: the default server stopped: {}", served.stop()?);
        let line = lines
            .iter()
            .find(|l| l.starts_with("ctx rule="))
            .map(String::as_str)
            .ok_or("the default server printed no `ctx` line")?
            .to_owned();
        println!("ctx: {line}");
        let num = |k: &str| field(&line, k).and_then(|v| v.parse::<usize>().ok());
        let bytes = |k: &str| field(&line, k).and_then(|v| v.parse::<u64>().ok());
        let (rule, ctx, trained, fit) = (
            field(&line, "rule").map(str::to_owned),
            num("ctx"),
            num("trained"),
            num("fit"),
        );
        let (fit_bytes, margin_ctx) = (bytes("fit_card_expert_bytes"), num("margin_ctx"));
        let (Some(rule), Some(ctx), Some(trained), Some(fit), Some(fit_bytes), Some(margin_ctx)) =
            (rule, ctx, trained, fit, fit_bytes, margin_ctx)
        else {
            return Err(format!(
                "a `ctx` line without its rule, ctx, trained, fit, fit_card_expert_bytes and \
                 margin_ctx: {line}"
            )
            .into());
        };

        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let plan_levers = PlanLevers::from_levers(levers)?;
        let free = server_card_free(&err_log)?;
        let at = |c: usize| card_at(&inputs, &plan_levers, c, free);
        let base = at(CTX)?.ok_or("no plan at the base context")?;
        let lost = |c: usize| -> Result<Option<u64>, GateError> {
            Ok(at(c)?.map(|e| base.saturating_sub(e)))
        };
        let margin = model::placement::workstation::MARGIN;
        let here = lost(margin_ctx)?;
        let past = lost(margin_ctx + CTX_STEP)?;
        let (at_fit, past_fit) = (at(fit)?, at(fit + 1)?);
        let cap = inputs
            .hp
            .n_ctx_train
            .min(usize::try_from(ORACLE_POSITIONS)?);
        let card_bounds = fit < cap;
        println!(
            "ctx: the gate's plans: lost at {margin_ctx} {here:?}, at {} {past:?}, fit {fit} \
             {at_fit:?}, past it {past_fit:?}; the cap {cap} ({})",
            margin_ctx + CTX_STEP,
            if card_bounds {
                "the card bounds the fit"
            } else {
                "the cap bounds the fit"
            }
        );
        let mut ok = true;
        check(
            &mut ok,
            "ctx_default_is_the_rules",
            match rule.as_str() {
                "set" => false,
                "base" => ctx == CTX && margin_ctx == CTX && CTX <= fit,
                "card" => ctx == fit && margin_ctx == fit && fit < CTX,
                "margin" => ctx == margin_ctx && CTX < margin_ctx && margin_ctx < fit,
                "fit" => ctx == fit && margin_ctx == fit && fit > CTX,
                _ => false,
            },
        );
        check(
            &mut ok,
            "ctx_line_prints_the_fit_and_the_margin",
            trained == inputs.hp.n_ctx_train
                && fit <= cap
                && at_fit == Some(fit_bytes)
                && (if card_bounds {
                    past_fit.is_none()
                } else {
                    fit == cap
                })
                && here.is_some_and(|l| l <= margin)
                && margin_ctx <= fit
                && (margin_ctx == fit
                    || (margin_ctx.is_multiple_of(CTX_STEP) && past.is_none_or(|l| l > margin))),
        );

        // The flag arm: below the floor, the flag wins.
        let flag_ctx = FLAG_CTX.to_string();
        let mut flag = args.clone();
        flag.extend_from_slice(&["--ctx", flag_ctx.as_str()]);
        let flagdir = own.join("flag");
        std::fs::create_dir_all(&flagdir)?;
        let err_log = flagdir.join("server.err");
        let mut served = Served::spawn(&flag, &flagdir, PLAIN)?;
        let lines = wait_for(&mut served, &err_log, &["ctx rule="])?;
        println!("ctx: the flag server stopped: {}", served.stop()?);
        let line = lines
            .iter()
            .find(|l| l.starts_with("ctx rule="))
            .map(String::as_str)
            .ok_or("the flag server printed no `ctx` line")?
            .to_owned();
        println!("ctx: {line}");
        check(
            &mut ok,
            "ctx_flag_wins",
            field(&line, "rule") == Some("set")
                && field(&line, "ctx").and_then(|v| v.parse::<usize>().ok()) == Some(FLAG_CTX),
        );

        // A context past the oracle: refused by name before it listens.
        let over_ctx = OVER_CTX.to_string();
        let err = own.join("oracle.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.args([
            "--model",
            "glm",
            "--place",
            "gate",
            "--ctx",
            over_ctx.as_str(),
            "--plan",
        ])
        .env_remove(bloomery_levers::DRAFT)
        .env_remove(bloomery_levers::RESIDENCY)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(File::create(&err)?);
        let status = cmd
            .status()
            .map_err(|e| format!("spawn bloomery-serve: {e}"))?;
        let text = std::fs::read_to_string(&err)?;
        let refused = text
            .lines()
            .find(|l| l.contains(&format!("ctx_max {OVER_CTX}")))
            .unwrap_or("")
            .to_owned();
        println!("ctx: --ctx {OVER_CTX} --plan: {status}; refused: {refused:?}");
        check(
            &mut ok,
            "ctx_past_the_oracle_is_refused_by_name",
            !status.success()
                && text.contains(&format!("ctx_max {OVER_CTX}"))
                && text.contains(&ORACLE_POSITIONS.to_string()),
        );
        Ok(ok)
    }

    /// The plain arm's drafted server, `BLOOMERY_DRAFT=mtp` with no
    /// residency (the module header): the prompt's greedy ids and the
    /// draft's counts.
    fn draft_only(dir: &Path, ids: &[u32], reference: &[u32]) -> Result<bool, GateError> {
        let dir = dir.join("mtp");
        std::fs::create_dir_all(&dir)?;
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, &dir, DRAFT_ONLY)?;
        println!("draft-only server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let load = lines_from(&err_log, 0)?;
        check(
            &mut ok,
            "draft_only_loads_the_draft_and_no_residency",
            record(&load, "load draft=mtp").is_some()
                && record(&load, "residency lever")
                    == Some("residency lever residency=off why=set")
                && record(&load, "residency host").is_none(),
        );
        let c = greedy(&url, ids, N_PREDICT, false)?;
        let got = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        let t = &c["timings"];
        println!(
            "draft-only completion tokens {got:?} stop_type={stop} draft_n={} draft_n_accepted={}",
            t["draft_n"], t["draft_n_accepted"]
        );
        check(
            &mut ok,
            "draft_only_carries_the_draft_counts",
            t["draft_n"].as_u64().is_some_and(|n| n > 0)
                && t["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        check(
            &mut ok,
            "draft_only_ids_are_the_plain_generate_glm5next",
            agree(&got, &stop, reference),
        );
        let top1 = json!({
            "prompt": ids, "n_predict": N_PREDICT, "temperature": 0.8, "top_k": 1, "seed": 42,
            "return_tokens": true, "cache_prompt": false,
        });
        let (st, text) = curl(&url("/completion"), Some(&top1), false)?;
        let k1 = json_of("/completion", st, &text)?;
        let sampled = ids_of(&k1["tokens"]);
        let k1_stop = k1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "draft-only top_k 1 sample tokens {sampled:?} stop_type={k1_stop} draft_n={}",
            k1["timings"]["draft_n"]
        );
        check(
            &mut ok,
            "draft_only_top1_sample_is_the_plain_generate_glm5next",
            k1["timings"].get("draft_n").is_none() && agree(&sampled, &k1_stop, reference),
        );
        ok &= cache(&url, &err_log, true)?;
        println!("draft-only server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The ids of `messages` as the server's chat template renders them
    /// (`/apply-template`, then `/tokenize` without BOS).
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

    /// `/tokenize` of `text` without BOS.
    fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({ "content": text, "add_special": false })),
            false,
        )?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// The slot dropped and not saved (`POST /slots/0?action=erase`): the
    /// prompt cache keeps what it held, and nothing more.
    fn erase(url: &dyn Fn(&str) -> String) -> Result<(), GateError> {
        let (st, body) = curl(&url("/slots/0?action=erase"), Some(&json!({})), false)?;
        json_of("/slots/0?action=erase", st, &body)?;
        Ok(())
    }

    /// One greedy request of the `cache` clause as the clause reads it: its
    /// ids, `cache_n`, the draft's counts and stop, and the server's lines it
    /// printed (`cache` notes, `mtp prompt` records, `residency pass`
    /// records).
    struct Turn {
        tokens: Vec<u32>,
        cache_n: u64,
        draft_n: u64,
        accepted: u64,
        stop: String,
        lines: Vec<String>,
    }

    impl Turn {
        fn run(
            url: &dyn Fn(&str) -> String,
            err_log: &Path,
            ids: &[u32],
            cache: bool,
        ) -> Result<Turn, GateError> {
            let from = lines_from(err_log, 0)?.len();
            let c = greedy(url, ids, N_PREDICT, cache)?;
            let t = &c["timings"];
            Ok(Turn {
                tokens: ids_of(&c["tokens"]),
                cache_n: t["cache_n"].as_u64().unwrap_or(u64::MAX),
                draft_n: t["draft_n"].as_u64().unwrap_or(0),
                accepted: t["draft_n_accepted"].as_u64().unwrap_or(0),
                stop: c["stop_type"].as_str().unwrap_or("").to_owned(),
                lines: lines_from(err_log, from)?,
            })
        }

        fn show(&self, what: &str) {
            let picked: Vec<&String> = self
                .lines
                .iter()
                .filter(|l| l.contains(" cache ") || l.starts_with("mtp prompt "))
                .collect();
            println!(
                "{what}: tokens {:?} stop={} cache_n={} draft_n={} accepted={} lines {picked:?}",
                self.tokens, self.stop, self.cache_n, self.draft_n, self.accepted
            );
        }

        /// The `cache load` note this request printed: a state put back.
        fn loaded(&self) -> Option<&str> {
            self.lines
                .iter()
                .find(|l| l.contains(": cache load "))
                .map(String::as_str)
        }

        /// The `mtp prompt` records this request printed that say the draft
        /// skips.
        fn skips(&self) -> usize {
            self.lines
                .iter()
                .filter(|l| l.starts_with("mtp prompt ") && !l.ends_with(" skipped=none"))
                .count()
        }
    }

    /// The `cache` clause on one server (`drafted`: its MTP draft runs; the
    /// residency off, so a sequence's bits are its own): session A's first
    /// turn fed fresh, B's turn, then A again — its state saved when B took
    /// the slot (`glm5next::seq_save`) and put back (`seq_resume`) — against
    /// the same requests with no switch between, run first:
    /// - (a) A's turn, its reply and a later text: it keeps every position A
    ///   held (a `cache load` note, `cache_n` the run with no switch's), and
    ///   its ids are that run's; under the draft the draft rejoins where A
    ///   left it — no `mtp prompt` record says it skips (the parked rows and
    ///   the NextN store's positions came back), it drafts and accepts, and
    ///   its counts are that run's: counts, which hold the store's and the
    ///   arena rows' bits only as far as a moved proposal moves an accept;
    /// - (b) A's turn with another text in place of its reply, so the shared
    ///   prefix ends at the turn: it keeps the checkpoint at the turn's
    ///   prompt call's end (the point the state carried), and its ids are the
    ///   run with no switch's, which cut to the model's own checkpoint there.
    ///
    /// Mutants: the state carrying no point ((b) keeps nothing); the KDA
    /// stores zeroed after the put-back ((a)'s ids move); the latent rows
    /// zeroed after it ((a)'s and (b)'s ids move); the seat's rule granting
    /// nothing of a saved state (no `cache load`: (a) and (b) keep nothing);
    /// under the draft its side not put back (the draft skips by name).
    fn cache(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        drafted: bool,
    ) -> Result<bool, GateError> {
        let label = if drafted { "drafted" } else { "plain" };
        let p1 = rendered(url, json!([{ "role": "user", "content": TURN_A }]))?;
        let pb = rendered(url, json!([{ "role": "user", "content": TURN_B }]))?;
        let later = tokenized(url, LATER_A)?;
        let stripped = tokenized(url, STRIPPED_A)?;
        let with = |a: &[u32], b: &[u32], c: &[u32]| -> Vec<u32> {
            a.iter().chain(b).chain(c).copied().collect()
        };
        let mut ok = true;

        // (a) with no switch, then the slot dropped unsaved, so no cached
        // state holds A when the switched run asks for it.
        let r1 = Turn::run(url, err_log, &p1, false)?;
        r1.show(&format!("cache {label}: A's turn fresh"));
        if r1.stop != "limit" || r1.tokens.len() != N_PREDICT {
            return Err(format!(
                "the cache clause's A turn stopped at {} tokens ({}): the clause needs {N_PREDICT}",
                r1.tokens.len(),
                r1.stop
            )
            .into());
        }
        let resend = with(&p1, &r1.tokens, &later);
        let r2 = Turn::run(url, err_log, &resend, true)?;
        r2.show(&format!("cache {label}: A resent, no switch"));
        erase(url)?;
        let a1 = Turn::run(url, err_log, &p1, false)?;
        a1.show(&format!("cache {label}: A's turn fresh again"));
        let b1 = Turn::run(url, err_log, &pb, true)?;
        b1.show(&format!("cache {label}: B's turn"));
        let a2 = Turn::run(url, err_log, &resend, true)?;
        a2.show(&format!("cache {label}: A resent after B"));
        let least = (p1.len() + N_PREDICT - 1) as u64;
        check(
            &mut ok,
            &format!("cache_{label}_a_keeps_every_held_position_after_the_switch"),
            a1.tokens == r1.tokens
                && !b1.tokens.is_empty()
                && a2.loaded().is_some()
                && r2.loaded().is_none()
                && a2.cache_n == r2.cache_n
                && a2.cache_n >= least,
        );
        check(
            &mut ok,
            &format!("cache_{label}_a_ids_are_the_run_with_no_switch"),
            !a2.tokens.is_empty() && a2.tokens == r2.tokens,
        );
        if drafted {
            check(
                &mut ok,
                "cache_drafted_a_draft_rejoins_where_a_left_it",
                a2.skips() == 0
                    && r2.skips() == 0
                    && a2.draft_n > 0
                    && a2.accepted > 0
                    && (a2.draft_n, a2.accepted) == (r2.draft_n, r2.accepted),
            );
        }

        // (b): the shared prefix ends at the turn, a checkpoint the prompt
        // call took at its end (the turn less its last id, which the first
        // step feeds).
        if stripped.first() == r1.tokens.first() {
            return Err(format!(
                "the replacement text's first id {:?} is the reply's: the shared prefix would not \
                 end at the turn",
                stripped.first()
            )
            .into());
        }
        let strip = with(&p1, &stripped, &[]);
        erase(url)?;
        let c1 = Turn::run(url, err_log, &p1, false)?;
        let s1 = Turn::run(url, err_log, &strip, true)?;
        s1.show(&format!("cache {label}: A's turn replaced, no switch"));
        erase(url)?;
        let c2 = Turn::run(url, err_log, &p1, false)?;
        let b2 = Turn::run(url, err_log, &pb, true)?;
        let s2 = Turn::run(url, err_log, &strip, true)?;
        s2.show(&format!("cache {label}: A's turn replaced after B"));
        let turn_end = p1.len() as u64 - 1;
        check(
            &mut ok,
            &format!("cache_{label}_b_keeps_the_turns_end"),
            c1.tokens == r1.tokens
                && c2.tokens == r1.tokens
                && !b2.tokens.is_empty()
                && s2.loaded().is_some()
                && s1.cache_n == turn_end
                && s2.cache_n == turn_end,
        );
        check(
            &mut ok,
            &format!("cache_{label}_b_ids_are_the_run_with_no_switch"),
            !s2.tokens.is_empty() && s2.tokens == s1.tokens,
        );
        Ok(ok)
    }

    /// Under the residency, a state put back runs on the slot map as it
    /// stands (the state carries none): A's turn, B's, A resent — put back (a
    /// `cache load` note), every held position kept, its passes printing
    /// their `residency pass` records, and the draft rejoining (no `mtp
    /// prompt` record says it skips, and it drafts). Its ids are printed, not
    /// held: B's passes land flips, and a flip moves an expert between the
    /// host and the card, whose sums round another way. Mutant: the draft's
    /// side not put back (the draft skips by name).
    fn resume_under_residency(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": TURN_A }]))?;
        let pb = rendered(url, json!([{ "role": "user", "content": TURN_B }]))?;
        let later = tokenized(url, LATER_A)?;
        let a1 = Turn::run(url, err_log, &p1, false)?;
        a1.show("residency resume: A's turn fresh");
        let b1 = Turn::run(url, err_log, &pb, true)?;
        b1.show("residency resume: B's turn");
        let resend: Vec<u32> = p1.iter().chain(&a1.tokens).chain(&later).copied().collect();
        let a2 = Turn::run(url, err_log, &resend, true)?;
        a2.show("residency resume: A resent after B");
        let a2_passes = passes(&a2.lines);
        println!("residency resume: A resent's passes (kind, kept, landed) {a2_passes:?}");
        let mut ok = true;
        check(
            &mut ok,
            "drafted_resume_runs_on_the_map_as_it_stands",
            a1.tokens.len() == N_PREDICT
                && !b1.tokens.is_empty()
                && a2.loaded().is_some()
                && a2.cache_n >= (p1.len() + N_PREDICT - 1) as u64
                && !a2.tokens.is_empty()
                && !a2_passes.is_empty()
                && a2.skips() == 0
                && a2.draft_n > 0,
        );
        Ok(ok)
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

    /// The swap clause (module header) on a server of two slots started into
    /// `<dir>/swap`: a decode preempted mid-run is put back with its draft
    /// rejoining, so both requests answer their solo runs' ids, their drafts
    /// run and skip nothing while they take turns (the proposal count rides
    /// the map as it stands at the resume, so it is not the solo run's).
    fn swap_rejoins_the_draft(dir: &Path) -> Result<bool, GateError> {
        let dir = dir.join("swap");
        std::fs::create_dir_all(&dir)?;
        // SERVER_ARGS ends in the plain engine's `--parallel 1`; this
        // clause's server takes the two-slot value.
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let parallel = args.len() - 1;
        args[parallel] = "2";
        let err_log = dir.join("server.err");
        // The draft-only levers: with the residency moving experts the map a
        // resumed pass runs on is the one the turns left, and a near tie may
        // flip between the solo and the swapped run; this clause holds the
        // draft's park and rejoin, which the map does not touch.
        let mut served = Served::spawn(&args, &dir, DRAFT_ONLY)?;
        println!("swap server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        // No `ignore_eos`: its banned stop ids step a request plainly (a
        // pass applies no logit bias), and this clause holds the draft.
        let body = |ids: &[u32], n: usize| {
            json!({
                "prompt": ids, "n_predict": n, "temperature": 0,
                "return_tokens": true, "cache_prompt": false,
            })
        };
        let (a_ids, b_ids) = (chat_ids(&url)?, tokenized(&url, LATER_A)?);
        let mut alone = Vec::new();
        for (ids, n) in [(&a_ids, SWAP_A_PREDICT), (&b_ids, SWAP_B_PREDICT)] {
            let (st, text) = curl(&url("/completion"), Some(&body(ids, n)), false)?;
            let v = json_of("/completion", st, &text)?;
            alone.push((
                ids_of(&v["tokens"]),
                v["timings"]["draft_n"].as_u64().unwrap_or(0),
            ));
        }
        println!(
            "swap alone: {} and {} ids, drafts {} and {}",
            alone[0].0.len(),
            alone[1].0.len(),
            alone[0].1,
            alone[1].1
        );
        let swaps = metric(&url, "swaps_total")?.unwrap_or(f64::NAN);
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
        let from = lines_from(&err_log, 0)?.len();
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
            together.push((
                ids_of(&v["tokens"]),
                v["timings"]["draft_n"].as_u64().unwrap_or(0),
                at,
            ));
        }
        let skips = lines_from(&err_log, from)?
            .iter()
            .filter(|l| l.starts_with("mtp prompt ") && !l.ends_with(" skipped=none"))
            .count();
        let after = metric(&url, "swaps_total")?.unwrap_or(f64::NAN);
        println!(
            "swap together: first {} ids of {} drafts, second {} ids of {} drafts, second back \
             {:?} before the first, switches {swaps} -> {after}, draft skips {skips}",
            together[0].0.len(),
            together[0].1,
            together[1].0.len(),
            together[1].1,
            together[0].2.checked_duration_since(together[1].2)
        );
        let mut ok = true;
        check(
            &mut ok,
            "swap_alone_ran_long_enough_to_preempt",
            alone[0].0.len() >= SWAP_B_PREDICT && !alone[1].0.is_empty(),
        );
        check(
            &mut ok,
            "swap_second_back_before_the_first",
            together[1].2 < together[0].2,
        );
        check(
            &mut ok,
            "swap_ids_are_alone_and_the_drafts_ran",
            together[0].0 == alone[0].0
                && together[1].0 == alone[1].0
                && together[0].1 > 0
                && together[1].1 > 0,
        );
        check(
            &mut ok,
            "swap_no_draft_skip_and_switched",
            skips == 0 && after > swaps,
        );
        println!("swap server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The plain arm (the module header).
    fn plain(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let mut ok = unset_rule(dir)?;
        ok &= ctx(dir, levers)?;
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, dir, PLAIN)?;
        println!("plain server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/health"), None, false)?;
        check(
            &mut ok,
            "plain_health_ok",
            st == 200 && body.contains("\"ok\""),
        );
        let load = lines_from(&err_log, 0)?;
        let lever = record(&load, "residency lever");
        println!(
            "plain records: {:?} {lever:?}",
            record(&load, "load draft=off")
        );
        check(
            &mut ok,
            "plain_loads_no_draft_and_no_residency",
            load.iter()
                .any(|l| l == "load draft=off (BLOOMERY_DRAFT=off)")
                && lever == Some("residency lever residency=off why=set")
                && record(&load, "residency host").is_none()
                && !load.iter().any(|l| l.contains("pair_graph_nodes=")),
        );
        ok &= props_draft(&url, None)?;
        let ids = chat_ids(&url)?;
        let (st, _) = chat(&url)?;
        check(&mut ok, "plain_chat_turn_served", st == 200);
        let c = greedy(&url, &ids, N_PREDICT, false)?;
        let got = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        println!("plain completion tokens {got:?} stop_type={stop}");
        let (st, body) = curl(&url("/residency/reset"), Some(&json!({})), false)?;
        println!("plain residency reset: HTTP {st} {body}");
        check(&mut ok, "plain_residency_reset_is_501", st == 501);
        ok &= cache(&url, &err_log, false)?;
        println!("plain server stopped: {}", served.stop()?);
        let (reference, cli_lines) = cli(dir, PLAIN, &ids, true)?;
        println!("generate_glm5next tokens {reference:?}");
        check(
            &mut ok,
            "plain_ids_are_generate_glm5next",
            agree(&got, &stop, &reference),
        );
        let (seat, cli_group) = (load_group(&load), load_group(&cli_lines));
        println!("plain load groups: the seat's {seat:?}, generate_glm5next's {cli_group:?}");
        let want = PROMPT_GROUP.to_string();
        check(
            &mut ok,
            "plain_loads_run_the_reserved_group",
            seat == Some(want.as_str()) && cli_group == Some(want.as_str()),
        );
        ok &= draft_only(dir, &ids, &reference)?;
        Ok(ok)
    }

    /// The drafted arm (the module header).
    fn drafted(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let bytes = nextn_plan_bytes(levers)?;
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, dir, DRAFTED)?;
        println!("drafted server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        check(
            &mut ok,
            "drafted_health_ok",
            st == 200 && body.contains("\"ok\""),
        );
        let load = lines_from(&err_log, 0)?;
        let named = format!("residency={RESIDENCY_WORD}");
        let (draft, lever, host) = (
            record(&load, "load draft=mtp"),
            record(&load, "residency lever"),
            record(&load, "residency host"),
        );
        println!("drafted records: {draft:?} {lever:?} {host:?}");
        check(
            &mut ok,
            "drafted_loads_the_draft_and_the_word",
            draft.is_some()
                && load.iter().any(|l| l.contains("pair_graph_nodes="))
                && lever.is_some_and(|l| l.contains(&named) && l.ends_with(" why=set"))
                && host.is_some_and(|l| l.contains(&named)),
        );
        ok &= props_draft(&url, Some(bytes))?;
        let ids = chat_ids(&url)?;

        let before = lines_from(&err_log, 0)?.len();
        let c = greedy(&url, &ids, N_PREDICT, false)?;
        let first = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        let t = &c["timings"];
        println!(
            "drafted completion tokens {first:?} stop_type={stop} draft_n={} draft_n_accepted={}",
            t["draft_n"], t["draft_n_accepted"]
        );
        let served_passes = passes(&lines_from(&err_log, before)?);
        println!("drafted completion passes (kind, kept, landed) {served_passes:?}");
        check(
            &mut ok,
            "drafted_completion_carries_the_draft_counts",
            t["draft_n"].as_u64().is_some_and(|n| n > 0)
                && t["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        let kind = |k: &str| served_passes.iter().any(|p| p.0 == k);
        check(
            &mut ok,
            "drafted_completion_ends_prompt_step_and_pair_passes",
            kind("prompt") && kind("step") && kind("pair"),
        );

        let (st, body) = curl(&url("/residency/reset"), Some(&json!({})), false)?;
        let reset: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let reset_line = record(&lines_from(&err_log, 0)?, "residency reset").map(str::to_owned);
        println!("drafted residency reset: HTTP {st} {reset}; record {reset_line:?}");
        let counts = ["cancelled", "copies", "diff"].map(|k| format!("{k}={}", reset[k]));
        check(
            &mut ok,
            "drafted_reset_is_a_200_with_its_record",
            st == 200
                && reset["diff"] == json!(0)
                && reset_line.is_some_and(|l| counts.iter().all(|c| l.contains(&format!(" {c} ")))),
        );
        let again = ids_of(&greedy(&url, &ids, N_PREDICT, false)?["tokens"]);
        println!("drafted completion after the reset tokens {again:?}");
        check(
            &mut ok,
            "drafted_ids_after_the_reset_are_the_first_requests",
            !first.is_empty() && again == first,
        );

        let (st, timings) = chat(&url)?;
        check(
            &mut ok,
            "drafted_chat_turn_drafts",
            st == 200 && timings["draft_n"].as_u64().is_some_and(|n| n > 0),
        );
        let mut extended = ids.clone();
        extended.extend_from_slice(&first);
        let joins_from = lines_from(&err_log, 0)?.len();
        let x = greedy(&url, &extended, 8, true)?;
        let joins: Vec<String> = lines_from(&err_log, joins_from)?
            .into_iter()
            .filter(|l| l.starts_with("mtp prompt "))
            .collect();
        println!(
            "drafted continuation (not held): cache_n={} draft_n={} tokens {:?} joins {joins:?}",
            x["timings"]["cache_n"],
            x["timings"]["draft_n"],
            ids_of(&x["tokens"])
        );
        ok &= resume_under_residency(&url, &err_log)?;

        println!("drafted server stopped: {}", served.stop()?);
        ok &= swap_rejoins_the_draft(dir)?;

        let (reference, cli_lines) = cli(dir, DRAFTED, &ids, true)?;
        let cli_passes = passes(&cli_lines);
        println!("generate_glm5next tokens {reference:?}");
        println!("generate_glm5next passes (kind, kept, landed) {cli_passes:?}");
        let landing = before_landing(&served_passes, 0).min(before_landing(&cli_passes, 0));
        let same_passes = served_passes == cli_passes;
        println!(
            "drafted ids: the whole run agrees {}; the first landing sits through pass {} of \
             {}, the passes the same lists {same_passes}",
            agree(&first, &stop, &reference),
            landing,
            first.len()
        );
        check(
            &mut ok,
            "drafted_ids_are_generate_glm5next",
            agree(&first, &stop, &reference),
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        for (name, set) in [
            (bloomery_levers::DRAFT, levers.draft()),
            (bloomery_levers::RESIDENCY, levers.residency()),
        ] {
            if let Some(word) = set {
                return Err(format!(
                    "{name}={word}: the gate sets it on every process it starts (the plain arm \
                     off, the drafted arm mtp and {RESIDENCY_WORD})"
                )
                .into());
            }
        }
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let ok = match a.arm {
            Arm::Plain => plain(&a.dir, &levers)?,
            Arm::Drafted => drafted(&a.dir, &levers)?,
        };
        if ok {
            println!("gate-gpu-glm5next-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
