//! GPU gate for the qwen3 seat's served drafts: the target `BLOOMERY_REF_MODEL` names (the
//! qwen35moe fixture in the fixture tier, whose `dspark/` directory holds the synthetic drafts
//! `fixture dspark` writes) held whole on the card with a DSpark draft beside it, for the
//! reduced-vocabulary synthetic draft (its own head and table, the speculator's shape), the
//! same draft beside the renamed copy of the target (`renamed/`, a file whose `general.name`
//! differs: a draft fits a target by its shapes, not by its name), and the full-vocabulary
//! synthetic draft (borrowing the target's Q6_K `output.weight` in place and its
//! `token_embd.weight` rows), and the n-gram lookup draft on the same target.
//!
//!     gate_qwen3_dspark --dir <dir>
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
//! then the lookup draft on the target: one server with the lookup and one without,
//! the lookup's ids are the plain run's (`ids_are_the_plain_ids`) and `/props` names the
//! lookup as the draft the seat serves (`lookup_is_served`), and a prompt that repeats
//! a phrase gets proposals from it (`lookup_proposes`: `timings.draft_n` above 0, so a
//! seat that names the lookup and steps plain does not pass; file-bound: a fixture's
//! random-weight target repeats no n-gram of its own, so the real tier runs it, and the
//! headline seat's `lookup_{i}_verifies_proposals` in `gate_qwen38_serve` pins the same
//! path); then the refusals,
//! each named on the server's stderr:
//!
//! - `cross_target_is_refused`: the mismatched synthetic draft (`dspark/mismatched`, whose
//!   stated vocabulary falls one id short of the target's) exits before listening, its line
//!   naming the draft's ids beside the target's — the same refusal a draft trained for
//!   another target meets, on files the box still holds;
//! - `placed_load_is_refused`: `--place a` at [`PLACED_CTX`] under the smallest card budget
//!   the planner keeps half the routed experts on the card at (the other half on the host)
//!   exits before listening with the whole-card sentence;
//! - `parallel_two_is_refused`: `--parallel 2` exits before listening with the one-slot
//!   sentence;
//! - `qwen3moe_lookup_is_refused`: a qwen3moe file (the family's fixture) with the lookup
//!   draft exits before listening, naming the body's own reason;
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
    reason = "the serve seat and every draft gate share this reader; this gate reads the draft directory only"
)]
mod dspark_hf;

#[cfg(feature = "deepseek41")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the placed refusal reads the planner's expert budget only"
)]
mod q3place;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use bloomery_gpu::arch::qwen3moe::Open35;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        Served, chat_is_those_ids, check, completion, curl, greedy, ids_of, json_of, rendered,
        serve_cmd, server_log, tokenized, utf8,
    };
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, model_arch, ref_model_path};
    use gguf::Split;
    use serde_json::json;

    /// The prompt every arm is fed.
    const PROMPT: &str = "The capital of France is";
    /// A prompt whose phrase repeats, so the lookup's n-grams match from its first step.
    const REPEATS: &str =
        "one two three four five one two three four five one two three four five one two";
    /// The ids compared and the tokens the any-draft arms decode.
    const N: usize = 16;

    /// The context the placed refusal's load is planned at.
    const PLACED_CTX: usize = 4096;

    /// The smallest card budget the placed planner keeps half of `model`'s routed experts on the
    /// card at, [`PLACED_CTX`] positions in one slot: a split load, the other half on the host,
    /// where the draft's verifying load cannot follow.
    fn half_placed_budget(model: &Path) -> Result<u64, GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let read = |suffix: &str| {
            split
                .arch_get_u64(suffix)
                .ok_or_else(|| format!("{}: no {}", model.display(), split.arch_key(suffix)))
        };
        let want = read("block_count")? * read("expert_count")? / 2;
        let o = Open35 {
            ctx: PLACED_CTX,
            mma: true,
            ubatch: ubatch_size()?,
            kv: KvQ8::F16,
            media: false,
        };
        let q = crate::q3place::PlaceQ3::qwen35(&split, Place::parse("cuda0")?, o)?;
        let ceiling = 2 * std::fs::metadata(model)?.len();
        let budget = q.budget_for_card_experts(PLACED_CTX, want, ceiling)?;
        println!(
            "placed budget on {}: {budget} B keeps {want} routed experts on the card",
            model.display()
        );
        Ok(budget)
    }

    /// How long a server gets to listen or to refuse, and the poll's gap.
    const WAIT: Duration = Duration::from_secs(300);
    const POLL: Duration = Duration::from_millis(100);

    /// The draft a case's second server serves.
    enum Arm {
        /// The DSpark draft in this directory, handed to the server as
        /// `BLOOMERY_DSPARK_MODEL`.
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

    /// The gate's files: the target `BLOOMERY_REF_MODEL` names, the drafts of its `dspark/`
    /// directory (the one `$BLOOMERY_DSPARK_MODEL` names, else the reduced one) and the
    /// renamed copy beside it, and the qwen3moe fixture. Under the fixture tier the target is
    /// held to a whole fixture by [`ref_model_path`], the renamed copy by [`model_arch`], and
    /// each draft and the copy is refused by name when `fixture dspark` has not written it.
    fn files() -> Result<(Vec<Case>, PathBuf, PathBuf, PathBuf), GateError> {
        let target = ref_model_path()?;
        let beside = |name: &str| target.parent().unwrap_or(Path::new(".")).join(name);
        // The reduced draft honors a caller's BLOOMERY_DSPARK_MODEL; its siblings and the
        // renamed copy are the fixture's own, beside the target's shards.
        let reduced = crate::dspark_hf::dir_or(|| Ok(beside(model::fixture::dspark::REDUCED_DIR)))?;
        let full = beside(model::fixture::dspark::FULL_DIR);
        let mismatched = beside(model::fixture::dspark::MISMATCHED_DIR);
        let renamed = beside(model::fixture::dspark::RENAMED_DIR).join(
            target
                .file_name()
                .ok_or_else(|| format!("{} names no file", target.display()))?,
        );
        for (what, dir) in [("reduced draft", &reduced), ("full draft", &full)] {
            for name in ["config.json", "model.safetensors"] {
                if !dir.join(name).is_file() {
                    return Err(format!(
                        "the {what} {} holds no {name}: `fixture dspark <fixture first shard>` \
                         writes it beside the fixture's shards",
                        dir.display()
                    )
                    .into());
                }
            }
        }
        if !mismatched.join("config.json").is_file()
            || !mismatched.join("model.safetensors").is_file()
        {
            return Err(format!(
                "the mismatched draft {} holds no HF draft; `fixture dspark <fixture first \
                 shard>` writes it beside the fixture's shards",
                mismatched.display()
            )
            .into());
        }
        if !renamed.is_file() {
            return Err(format!(
                "the renamed target copy {} is not there; `fixture dspark <fixture first shard>` \
                 writes it beside the fixture's shards",
                renamed.display()
            )
            .into());
        }
        if Tier::from_env()? == Tier::Fixture {
            model_arch(Tier::Fixture, &renamed)?;
        }
        let qwen3moe = PathBuf::from(refset::fixture::first_shard("qwen3moe")?);
        let cases = vec![
            Case {
                name: "target",
                model: target.clone(),
                draft: reduced.clone(),
            },
            Case {
                name: "renamed",
                model: renamed,
                draft: reduced.clone(),
            },
            Case {
                name: "full",
                model: target,
                draft: full,
            },
        ];
        Ok((cases, mismatched, qwen3moe, reduced))
    }

    /// The server of one arm: the seat's model and the draft the arm serves,
    /// the run's own directory, the extra arguments and the extra environment.
    /// The one setup every arm of this gate uses, so the plain and the drafted
    /// server differ by the draft alone. The whole-card seat refuses a card
    /// budget, which the fixture tier's box command exports for the gates that
    /// plan a fixture, so no arm inherits one; the placed arm sets its own.
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
            .env_remove("BLOOMERY_DSPARK_MODEL")
            .env_remove(bloomery_levers::CARD_BUDGET);
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
            &utf8(&case.model)?,
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
                // The lookup proposes only where the text repeats an n-gram of its own: a
                // fixture's random-weight target writes none, so the file answers it.
                if tier::run_clause("lookup_proposes", Tag::FileBound)? {
                    let repeats = completion(&url, &tokenized(&url, REPEATS)?, N, false)?;
                    check(
                        ok,
                        "lookup_proposes",
                        repeats["timings"]["draft_n"]
                            .as_u64()
                            .is_some_and(|n| n > 0),
                    );
                }
            }
            None => {}
        }
        drop(served);
        Ok(tokens)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV])?;
        let mut args = std::env::args().skip(1);
        let mut root = None;
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--dir" => {
                    root = Some(args.next().map(PathBuf::from).ok_or("--dir needs a path")?);
                }
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        let root = root.ok_or("usage: gate_qwen3_dspark --dir <dir>")?;
        std::fs::create_dir_all(&root)?;
        let (cases, mismatched, qwen3moe_file, reduced) = files()?;
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
        // The lookup draft on the target: the same target, the other draft.
        let qwen = &cases[0];
        let lookup = one(qwen, &root, Some(&Arm::Lookup), "lookup", &mut ok)?;
        check(&mut ok, "ids_are_the_plain_ids", lookup == plains[0]);
        // The mismatched draft beside the target: the ids line of a draft written for another
        // vocabulary, refused before any weight is read.
        let wrong = Case {
            name: "mismatched",
            model: qwen.model.clone(),
            draft: mismatched,
        };
        let text = refusal(
            &wrong,
            Some(&Arm::Dspark(wrong.draft.clone())),
            &root.join("mismatched"),
            &[],
            &[],
        )?;
        check(
            &mut ok,
            "cross_target_is_refused",
            hint(&text, "ids; the target holds"),
        );
        let budget = half_placed_budget(&qwen.model)?.to_string();
        let placed_ctx = PLACED_CTX.to_string();
        let placed = refusal(
            qwen,
            Some(&Arm::Dspark(reduced.clone())),
            &root.join("placed"),
            &["--place", "a", "--ctx", &placed_ctx],
            &[(bloomery_levers::CARD_BUDGET, &budget)],
        )?;
        check(
            &mut ok,
            "placed_load_is_refused",
            hint(&placed, "BLOOMERY_DRAFT=dspark needs the whole-card load"),
        );
        let parallel = refusal(
            qwen,
            Some(&Arm::Dspark(reduced.clone())),
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
            model: qwen3moe_file,
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
            Some(&Arm::Dspark(reduced)),
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
