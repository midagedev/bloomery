//! `gate_glm5next_serve` — the GLM seat of `bloomery-serve` (`--model glm`)
//! on the 3090 (placement gate), driven over HTTP, against
//! `generate_glm5next` on the same card, as one process under the GPU gate
//! lock.
//!
//!     gate_glm5next_serve --arm plain|drafted --dir <out>
//!
//! Each arm starts the server beside this binary (`--model glm --host
//! 127.0.0.1 --port 0 --place gate --ctx 2048`, its levers set here), reads
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
//! the `residency host` record of the word; under `--place gate` both `off`,
//! no `residency host`:
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
//! - `POST /residency/reset` is the server's 501 (no machine);
//! - then a third load, the server under `BLOOMERY_DRAFT=mtp
//!   BLOOMERY_RESIDENCY=off`: it prints `load draft=mtp` and no `residency
//!   host`, the same `/completion` carries the draft's counts, and its ids
//!   are the plain CLI's, all of them; a sampled `/completion` cut to
//!   `top_k` 1 (temperature 0.8, a fixed seed) is served through plain steps,
//!   drafting nothing, each step's row read before the step is told to the
//!   draft, and its ids are the plain CLI's too. Green-only for the row's
//!   read: the NextN walk writes no row of the target's head, so a `step_row`
//!   left at the default reads the same row; the clause holds the sampled
//!   path's ids end to end. Exact: with no residency nothing
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
//! - against the CLI under the same levers: the ids agree through the first
//!   pass whose boundary landed a flip in either run, by their `residency
//!   pass` records. Not the whole run: the server's prompt call stops one id
//!   short and steps it, the CLI's takes it, so the server's rule counts one
//!   more step and its flips may land at other passes; a flip moves an
//!   expert between the host and the card, whose sums round another way, so
//!   a greedy id past a landing can move at a near tie. Before any landing
//!   both run on the same seed, the plan's card experts less the spares, and
//!   a verify's rows are its steps' bits. Whether the whole run agrees is
//!   printed, not held.
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
    use std::time::Duration;

    use bloomery_gpu_gates::serve_client::{curl, ids_of, json_of, parse_ids};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::glm5next::place::{NextnInputs, PlanInputs};
    use model::placement::{PlanLevers, workstation};
    use serde_json::{Value, json};

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

    /// The stores both engines size, the seat's and the CLI's default.
    const CTX: usize = 2048;
    /// The server's arguments after its path.
    const SERVER_ARGS: [&str; 10] = [
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
    ];
    /// The load takes a minute or two; the bound is 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    /// The greedy requests' length, the CLI's `-n`.
    const N_PREDICT: usize = 16;
    /// The one chat turn.
    const CHAT: &str = "What is the capital of France? Answer in one word.";
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
    /// the first pass's rows count (the CLI's prompt call gives token 0 and
    /// keeps no row; the server's prompt call gives none, its step token 0),
    /// then each pass's kept rows, that pass's included — it ran before its
    /// flips went live.
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
    /// unset: its exit status and stderr. It plans and exits before the
    /// load, so it opens no card.
    fn plan_only(dir: &Path, place: &str) -> Result<(bool, Vec<String>), GateError> {
        let exe = beside("bloomery-serve")?;
        let err = dir.join(format!("plan-{place}.err"));
        let status = Command::new(&exe)
            .args([
                "--model", "glm", "--place", place, "--ctx", "2048", "--plan",
            ])
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
        println!("draft-only server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The plain arm (the module header).
    fn plain(dir: &Path) -> Result<bool, GateError> {
        let mut ok = unset_rule(dir)?;
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
        println!("plain server stopped: {}", served.stop()?);
        let (reference, _) = cli(dir, PLAIN, &ids, true)?;
        println!("generate_glm5next tokens {reference:?}");
        check(
            &mut ok,
            "plain_ids_are_generate_glm5next",
            agree(&got, &stop, &reference),
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

        println!("drafted server stopped: {}", served.stop()?);

        let (reference, cli_lines) = cli(dir, DRAFTED, &ids, false)?;
        let cli_passes = passes(&cli_lines);
        println!("generate_glm5next tokens {reference:?}");
        println!("generate_glm5next passes (kind, kept, landed) {cli_passes:?}");
        let exact = before_landing(&served_passes, 0)
            .min(before_landing(&cli_passes, 1))
            .min(first.len() as u64);
        let exact = usize::try_from(exact)?;
        println!(
            "drafted ids before the first landing: {exact} of {}; the whole run agrees: {}",
            first.len(),
            agree(&first, &stop, &reference)
        );
        check(
            &mut ok,
            "drafted_ids_are_generate_glm5next_through_the_first_landing",
            exact >= 1 && reference.len() >= exact && first[..exact] == reference[..exact],
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
            Arm::Plain => plain(&a.dir)?,
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
