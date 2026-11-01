//! GPU gate for the qwen3 seat's served drafts: a qwen35moe target held whole
//! on the card with a DSpark draft beside it, for a reduced-vocabulary draft
//! (RedHatAI's Qwen3.6 speculator, its own head and table) and a full-vocabulary
//! one (MiniCPM-V 4.7's, borrowing the target's Q6_K `output.weight` in place and its
//! `token_embd.weight` rows), and the n-gram lookup draft on the same target.
//!
//!     gate_qwen3_dspark --qwen <gguf> --qwen-draft <dir> --sage <gguf> \
//!         --ornith <gguf> --minicpm <gguf> --minicpm-draft <dir> \
//!         --qwen3moe <gguf> --dir <dir>
//!
//! For each DSpark case, one server with no draft and one with it, both greedy
//! from the same prompt, hold:
//!
//! - `draft_is_loaded`: the drafted server printed its `load_draft` record. A seat whose
//!   `open_dspark` returns no draft refuses the drafted server before listening, so a
//!   missing record cannot pass as a plain run;
//! - `draft_proposes`: the drafted answer's `timings.draft_n` is above 0. A draft that
//!   loaded and then proposed nothing (a plain step wearing the draft's name) is not a
//!   draft, and its ids would be the plain run's for the wrong reason;
//! - `ids_are_the_plain_ids`: the drafted run's first [`N`] ids are the plain run's. The
//!   target verifies every kept token, so this holds only when the draft's taps, mask and
//!   borrowed head leave the target's own path alone; a fault there moves the first kept
//!   id;
//! - `chat_is_the_draft_ids`: `/v1/chat/completions` at temperature 0 answers the chat
//!   template's rendered ids with the same completion the drafted server's `/completion`
//!   gives, its usage counts and its text inside the ids' text;
//!
//! then the lookup draft on the qwen3.6 file: one server with the lookup and one without,
//! the lookup's ids are the plain run's (`ids_are_the_plain_ids`) and `/props` names the
//! lookup as the draft the seat serves (`lookup_is_served`), and a prompt that repeats
//! a phrase gets proposals from it (`lookup_proposes`: `timings.draft_n` above 0, so a
//! seat that names the lookup and steps plain does not pass); then the refusals,
//! each named on the server's stderr:
//!
//! - `cross_target_is_refused`: each of the two base targets with the other case's draft
//!   exits before listening, its line naming the draft's ids beside the target's;
//! - `placed_load_is_refused`: `--place a` under `BLOOMERY_CARD_BUDGET=12G` exits before
//!   listening with the whole-card sentence;
//! - `parallel_two_is_refused`: `--parallel 2` exits before listening with the one-slot
//!   sentence;
//! - `qwen3moe_lookup_is_refused`: a qwen3moe file with the lookup draft exits before
//!   listening, naming the body's own reason;
//! - `qwen3moe_dspark_is_refused`: a qwen3moe file with the DSpark draft exits before
//!   listening, naming the family the DSpark draft serves.
//!
//! The gate reads no oracle set: the plain server is the reference.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3_dspark", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/dspark_hf.rs"]
#[allow(
    dead_code,
    reason = "the serve seat and every draft gate share this reader; this gate reads the shape only"
)]
mod dspark_hf;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        Served, chat_is_those_ids, check, completion, curl, greedy, ids_of, json_of, rendered,
        serve_cmd, server_log, tokenized, utf8,
    };
    use bloomery_gpu_gates::{GateError, checks_failed};
    use serde_json::json;

    /// The prompt every arm is fed.
    const PROMPT: &str = "The capital of France is";
    /// A prompt whose phrase repeats, so the lookup's n-grams match from its first step.
    const REPEATS: &str =
        "one two three four five one two three four five one two three four five one two";
    /// The ids compared and the tokens the any-draft arms decode.
    const N: usize = 16;
    /// How long a server gets to listen or to refuse, and the poll's gap.
    const WAIT: Duration = Duration::from_secs(300);
    const POLL: Duration = Duration::from_millis(100);

    /// The draft a case's second server serves.
    enum Arm {
        /// The DSpark draft in this directory (`$BLOOMERY_DSPARK_MODEL`).
        Dspark(PathBuf),
        /// The n-gram lookup draft: no file, no weights.
        Lookup,
    }

    /// A target and the DSpark draft beside it.
    struct Case {
        name: &'static str,
        model: PathBuf,
        draft: PathBuf,
    }

    fn arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, GateError> {
        args.next()
            .map(PathBuf::from)
            .ok_or_else(|| format!("{flag} needs a path").into())
    }

    fn parse_args() -> Result<(Vec<Case>, PathBuf, PathBuf), GateError> {
        let mut qwen = None;
        let mut qwen_draft = None;
        let mut sage = None;
        let mut ornith = None;
        let mut minicpm = None;
        let mut minicpm_draft = None;
        let mut qwen3moe = None;
        let mut dir = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--qwen" => qwen = Some(arg(&mut args, "--qwen")?),
                "--qwen-draft" => qwen_draft = Some(arg(&mut args, "--qwen-draft")?),
                "--sage" => sage = Some(arg(&mut args, "--sage")?),
                "--ornith" => ornith = Some(arg(&mut args, "--ornith")?),
                "--minicpm" => minicpm = Some(arg(&mut args, "--minicpm")?),
                "--minicpm-draft" => minicpm_draft = Some(arg(&mut args, "--minicpm-draft")?),
                "--qwen3moe" => qwen3moe = Some(arg(&mut args, "--qwen3moe")?),
                "--dir" => dir = Some(arg(&mut args, "--dir")?),
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        let usage = "usage: gate_qwen3_dspark --qwen <gguf> --qwen-draft <dir> --sage <gguf> \
                     --ornith <gguf> --minicpm <gguf> --minicpm-draft <dir> \
                     --qwen3moe <gguf> --dir <dir>";
        let redhat = qwen_draft.ok_or(usage)?;
        // The RedHat draft fits a target by its facts, not by the file's name: the
        // two fine-tunes below carry Qwen3.6's shapes, so they take it too.
        let cases = vec![
            Case {
                name: "qwen3.6",
                model: qwen.ok_or(usage)?,
                draft: redhat.clone(),
            },
            Case {
                name: "sage",
                model: sage.ok_or(usage)?,
                draft: redhat.clone(),
            },
            Case {
                name: "ornith",
                model: ornith.ok_or(usage)?,
                draft: redhat,
            },
            Case {
                name: "minicpm",
                model: minicpm.ok_or(usage)?,
                draft: minicpm_draft.ok_or(usage)?,
            },
        ];
        Ok((cases, qwen3moe.ok_or(usage)?, dir.ok_or(usage)?))
    }

    /// The server of one arm: the seat's model and the draft the arm serves,
    /// the run's own directory, the extra arguments and the extra environment.
    /// The one setup every arm of this gate uses, so the plain and the drafted
    /// server differ by the draft alone.
    fn spawn(
        case: &Case,
        arm: Option<&Arm>,
        run: &Path,
        extra: &[&str],
        env: &[(&str, &str)],
    ) -> Result<(Served, PathBuf), GateError> {
        std::fs::create_dir_all(run)?;
        let mut cmd = serve_cmd()?;
        cmd.env_remove("BLOOMERY_DRAFT")
            .env_remove("BLOOMERY_DSPARK_MODEL");
        match arm {
            Some(Arm::Dspark(dir)) => {
                cmd.env("BLOOMERY_DRAFT", "dspark");
                cmd.env("BLOOMERY_DSPARK_MODEL", dir);
            }
            Some(Arm::Lookup) => {
                cmd.env("BLOOMERY_DRAFT", "lookup");
            }
            None => {}
        }
        cmd.envs(env.iter().copied());
        let mut args = vec![
            "--model",
            "qwen3",
            "-m",
            utf8(&case.model)?,
            "--port",
            "0",
            "--parallel",
            "1",
        ];
        args.extend_from_slice(extra);
        Ok((Served::spawn_cmd(cmd, &args, run)?, run.join("server.err")))
    }

    /// The server's stderr once it exited, or an error naming that it was still up.
    fn refusal(
        case: &Case,
        arm: Option<&Arm>,
        run: &Path,
        extra: &[&str],
        env: &[(&str, &str)],
    ) -> Result<String, GateError> {
        let (mut served, err) = spawn(case, arm, run, extra, env)?;
        let started = Instant::now();
        while started.elapsed() < WAIT {
            if let Some(_status) = served.child.try_wait()? {
                return Ok(std::fs::read_to_string(&err).unwrap_or_default());
            }
            std::thread::sleep(POLL);
        }
        let _ = served.child.kill();
        Err(format!(
            "{}: the server neither listened nor exited within {WAIT:?}; {}",
            case.name,
            std::fs::read_to_string(&err).unwrap_or_default()
        )
        .into())
    }

    fn hint(text: &str, want: &str) -> bool {
        text.contains(want)
    }

    /// One arm's ids: the plain arm's, the drafted arm's, and the drafted arm's
    /// own evidence (its load record, its proposals and its chat answers).
    fn one(
        case: &Case,
        root: &Path,
        arm: Option<&Arm>,
        tag: &str,
        ok: &mut bool,
    ) -> Result<Vec<u32>, GateError> {
        let run = root.join(format!("{}-{tag}", case.name));
        let (mut served, err) = spawn(case, arm, &run, &[], &[])?;
        let addr = served.address(&err, 3000, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let ids = tokenized(&url, PROMPT)?;
        let answer = completion(&url, &ids, N, false)?;
        let tokens = ids_of(&answer["tokens"])?;
        println!("{tag} {} tokens {tokens:?}", case.name);
        match arm {
            Some(Arm::Dspark(_)) => {
                let log = server_log(&err, &[&record::LOAD_DRAFT])?;
                check(ok, "draft_is_loaded", log.one(&record::LOAD_DRAFT).is_ok());
                check(
                    ok,
                    "draft_proposes",
                    answer["timings"]["draft_n"].as_u64().is_some_and(|n| n > 0),
                );
                let messages = json!([{"role": "user", "content": "Name the capital of France."}]);
                let chat_ids = rendered(&url, messages.clone())?;
                let (ids_answer, v) = greedy(&url, chat_ids, N, &run, "chat-greedy", false)?;
                check(
                    ok,
                    "chat_is_the_draft_ids",
                    chat_is_those_ids(&url, &run, messages, N, (&ids_answer, &v))?,
                );
            }
            Some(Arm::Lookup) => {
                // The seat's own word for the draft it serves: the lookup's
                // row in `/props`, the one the seat that serves it names.
                let (st, body) = curl(&url("/props"), None, false)?;
                let props = json_of("/props", st, &body)?;
                check(
                    ok,
                    "lookup_is_served",
                    props["engine"]["draft"]["kind"].as_str() == Some("lookup"),
                );
                let repeats = completion(&url, &tokenized(&url, REPEATS)?, N, false)?;
                check(
                    ok,
                    "lookup_proposes",
                    repeats["timings"]["draft_n"]
                        .as_u64()
                        .is_some_and(|n| n > 0),
                );
            }
            None => {}
        }
        drop(served);
        Ok(tokens)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV])?;
        let (cases, qwen3moe, root) = parse_args()?;
        std::fs::create_dir_all(&root)?;
        let mut ok = true;
        let mut plains = Vec::new();
        for case in &cases {
            let plain = one(case, &root, None, "plain", &mut ok)?;
            let drafted = one(
                case,
                &root,
                Some(&Arm::Dspark(case.draft.clone())),
                "draft",
                &mut ok,
            )?;
            check(&mut ok, "ids_are_the_plain_ids", drafted == plain);
            plains.push(plain);
        }
        // The lookup draft on the qwen3.6 file: the same target, the other draft.
        let qwen = &cases[0];
        let lookup = one(qwen, &root, Some(&Arm::Lookup), "lookup", &mut ok)?;
        check(&mut ok, "ids_are_the_plain_ids", lookup == plains[0]);
        for (case, wrong) in [(&cases[0], &cases[3]), (&cases[3], &cases[0])] {
            let text = refusal(
                case,
                Some(&Arm::Dspark(wrong.draft.clone())),
                &root.join(format!("{}-cross", case.name)),
                &[],
                &[],
            )?;
            check(
                &mut ok,
                "cross_target_is_refused",
                hint(&text, "ids; the target holds"),
            );
        }
        let placed = refusal(
            qwen,
            Some(&Arm::Dspark(qwen.draft.clone())),
            &root.join("placed"),
            &["--place", "a"],
            &[("BLOOMERY_CARD_BUDGET", "12G")],
        )?;
        check(
            &mut ok,
            "placed_load_is_refused",
            hint(&placed, "BLOOMERY_DRAFT=dspark needs the whole-card load"),
        );
        let parallel = refusal(
            qwen,
            Some(&Arm::Dspark(qwen.draft.clone())),
            &root.join("parallel"),
            &["--parallel", "2"],
            &[],
        )?;
        check(
            &mut ok,
            "parallel_two_is_refused",
            hint(&parallel, "--parallel 2 under BLOOMERY_DRAFT=dspark"),
        );
        // The qwen3moe body runs no pass of several rows: the lookup draft needs
        // one, and the DSpark draft serves a qwen35moe file.
        let qwen3moe = Case {
            name: "qwen3moe",
            model: qwen3moe,
            draft: PathBuf::new(),
        };
        let lookup_refused = refusal(
            &qwen3moe,
            Some(&Arm::Lookup),
            &root.join("qwen3moe-lookup"),
            &[],
            &[],
        )?;
        check(
            &mut ok,
            "qwen3moe_lookup_is_refused",
            hint(&lookup_refused, "BLOOMERY_DRAFT=lookup on a qwen3moe file"),
        );
        let dspark_refused = refusal(
            &qwen3moe,
            Some(&Arm::Dspark(cases[0].draft.clone())),
            &root.join("qwen3moe-dspark"),
            &[],
            &[],
        )?;
        check(
            &mut ok,
            "qwen3moe_dspark_is_refused",
            hint(
                &dspark_refused,
                "BLOOMERY_DRAFT=dspark: the DSpark draft serves a qwen35moe",
            ),
        );
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
