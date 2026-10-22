//! GPU gate for `BLOOMERY_DRAFT=dspark` on the qwen3 serve seat: a qwen35moe target held
//! whole on the card with a DSpark draft beside it, for a reduced-vocabulary draft
//! (RedHatAI's Qwen3.6 speculator, its own head and table) and a full-vocabulary one
//! (MiniCPM-V 4.7's, borrowing the target's Q6_K `output.weight` in place and its
//! `token_embd.weight` rows).
//!
//!     gate_qwen3_dspark --qwen <gguf> --qwen-draft <dir> --minicpm <gguf> \
//!         --minicpm-draft <dir> --dir <dir>
//!
//! For each case, one server with no draft and one with it, both greedy from the same
//! prompt, hold:
//!
//! - `draft_is_loaded`: the drafted server printed its `load_draft` record. A seat whose
//!   `open_dspark` returns no draft refuses the drafted server before listening, so a
//!   missing record cannot pass as a plain run;
//! - `ids_are_the_plain_ids`: the drafted run's first [`N`] ids are the plain run's. The
//!   target verifies every kept token, so this holds only when the draft's taps, mask and
//!   borrowed head leave the target's own path alone; a fault there moves the first kept
//!   id;
//! - `chat_is_the_draft_ids`: `/v1/chat/completions` at temperature 0 answers the chat
//!   template's rendered ids with the same completion the drafted server's `/completion`
//!   gives, its usage counts and its text inside the ids' text;
//!
//! then the refusals, each named on the server's stderr:
//!
//! - `cross_target_is_refused`: each target with the other case's draft exits before
//!   listening, its line naming the draft's ids beside the target's;
//! - `placed_load_is_refused`: `--place a` under `BLOOMERY_CARD_BUDGET=12G` exits before
//!   listening with the whole-card sentence;
//! - `parallel_two_is_refused`: `--parallel 2` exits before listening with the one-slot
//!   sentence.
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
        Served, chat_is_those_ids, check, completion, greedy, ids_of, rendered, serve_cmd,
        server_log, tokenized, utf8,
    };
    use bloomery_gpu_gates::{GateError, checks_failed};
    use serde_json::json;

    /// The prompt every arm is fed.
    const PROMPT: &str = "The capital of France is";
    /// The ids compared and the tokens the any-draft arms decode.
    const N: usize = 16;
    /// How long a server gets to listen or to refuse, and the poll's gap.
    const WAIT: Duration = Duration::from_secs(300);
    const POLL: Duration = Duration::from_millis(100);

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

    fn parse_args() -> Result<(Case, Case, PathBuf), GateError> {
        let mut qwen = None;
        let mut qwen_draft = None;
        let mut minicpm = None;
        let mut minicpm_draft = None;
        let mut dir = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--qwen" => qwen = Some(arg(&mut args, "--qwen")?),
                "--qwen-draft" => qwen_draft = Some(arg(&mut args, "--qwen-draft")?),
                "--minicpm" => minicpm = Some(arg(&mut args, "--minicpm")?),
                "--minicpm-draft" => minicpm_draft = Some(arg(&mut args, "--minicpm-draft")?),
                "--dir" => dir = Some(arg(&mut args, "--dir")?),
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        let usage = "usage: gate_qwen3_dspark --qwen <gguf> --qwen-draft <dir> \
                     --minicpm <gguf> --minicpm-draft <dir> --dir <dir>";
        Ok((
            Case {
                name: "qwen3.6",
                model: qwen.ok_or(usage)?,
                draft: qwen_draft.ok_or(usage)?,
            },
            Case {
                name: "minicpm",
                model: minicpm.ok_or(usage)?,
                draft: minicpm_draft.ok_or(usage)?,
            },
            dir.ok_or(usage)?,
        ))
    }

    fn start(
        case: &Case,
        draft: Option<&Path>,
        run: &Path,
        extra: &[&str],
    ) -> Result<(Served, PathBuf, String), GateError> {
        std::fs::create_dir_all(run)?;
        let mut cmd = serve_cmd()?;
        cmd.env_remove("BLOOMERY_DRAFT")
            .env_remove("BLOOMERY_DSPARK_MODEL");
        if let Some(draft) = draft {
            cmd.env("BLOOMERY_DRAFT", "dspark")
                .env("BLOOMERY_DSPARK_MODEL", draft);
        }
        let mut args = vec![
            "--model".to_owned(),
            "qwen3".to_owned(),
            "-m".to_owned(),
            utf8(&case.model)?.to_owned(),
            "--port".to_owned(),
            "0".to_owned(),
            "--parallel".to_owned(),
            "1".to_owned(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_owned()));
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut served = Served::spawn_cmd(cmd, &refs, run)?;
        let err = run.join("server.err");
        let addr = served.address(&err, 3000, POLL)?;
        Ok((served, err, addr))
    }

    /// The server's stderr once it exited, or an error naming that it was still up.
    fn refusal(
        case: &Case,
        draft: Option<&Path>,
        run: &Path,
        extra: &[&str],
        env: &[(&str, &str)],
    ) -> Result<String, GateError> {
        std::fs::create_dir_all(run)?;
        let mut cmd = serve_cmd()?;
        cmd.env_remove("BLOOMERY_DRAFT")
            .env_remove("BLOOMERY_DSPARK_MODEL");
        if let Some(draft) = draft {
            cmd.env("BLOOMERY_DRAFT", "dspark")
                .env("BLOOMERY_DSPARK_MODEL", draft);
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        let args = [
            "--model".to_owned(),
            "qwen3".to_owned(),
            "-m".to_owned(),
            utf8(&case.model)?.to_owned(),
            "--port".to_owned(),
            "0".to_owned(),
            "--parallel".to_owned(),
            "1".to_owned(),
        ];
        let mut args: Vec<&str> = args.iter().map(String::as_str).collect();
        args.extend_from_slice(extra);
        let mut served = Served::spawn_cmd(cmd, &args, run)?;
        let err = run.join("server.err");
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

    fn one(case: &Case, root: &Path, draft: bool, ok: &mut bool) -> Result<Vec<u32>, GateError> {
        let tag = if draft { "draft" } else { "plain" };
        let run = root.join(format!("{}-{tag}", case.name));
        let (served, err, addr) = start(case, draft.then_some(case.draft.as_path()), &run, &[])?;
        let url = |p: &str| format!("http://{addr}{p}");
        let ids = tokenized(&url, PROMPT)?;
        let answer = completion(&url, &ids, N, false)?;
        let tokens = ids_of(&answer["tokens"]);
        println!("{tag} {} tokens {tokens:?}", case.name);
        if draft {
            let log = server_log(&err, &[&record::LOAD_DRAFT])?;
            check(ok, "draft_is_loaded", log.one(&record::LOAD_DRAFT).is_ok());
            let messages = json!([{"role": "user", "content": "Name the capital of France."}]);
            let chat_ids = rendered(&url, messages.clone())?;
            let (ids_answer, v) = greedy(&url, chat_ids, N, &run, "chat-greedy", false)?;
            check(
                ok,
                "chat_is_the_draft_ids",
                chat_is_those_ids(&url, &run, messages, N, (&ids_answer, &v))?,
            );
        }
        drop(served);
        Ok(tokens)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let (qwen, minicpm, root) = parse_args()?;
        std::fs::create_dir_all(&root)?;
        let mut ok = true;
        let cases = [&qwen, &minicpm];
        for case in cases {
            let plain = one(case, &root, false, &mut ok)?;
            let drafted = one(case, &root, true, &mut ok)?;
            check(&mut ok, "ids_are_the_plain_ids", drafted == plain);
        }
        for (case, wrong) in [(&qwen, &minicpm), (&minicpm, &qwen)] {
            let text = refusal(
                case,
                Some(wrong.draft.as_path()),
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
            &qwen,
            Some(qwen.draft.as_path()),
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
            &qwen,
            Some(qwen.draft.as_path()),
            &root.join("parallel"),
            &["--parallel", "2"],
            &[],
        )?;
        check(
            &mut ok,
            "parallel_two_is_refused",
            hint(&parallel, "--parallel 2 under BLOOMERY_DRAFT=dspark"),
        );
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
