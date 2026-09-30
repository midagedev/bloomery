//! `gate_qwen38_serve` — `bloomery-serve-qwen38` on the 3090 (placement
//! gate), driven over HTTP, as one process under the GPU gate lock.
//!
//!     gate_qwen38_serve --gen <generate_qwen3moe log> --prompt <text>
//!                       --ids <a,b,…> --dir <out>
//!
//! Starts the server beside this binary (`--host 127.0.0.1 --port 0 --place
//! gate`), reads its address from its stderr, waits for `/health`, then
//! checks:
//!
//! - `/props`' `engine` object, printed once, against this gate's own plan of
//!   the file the server opens (the gate card, the server's default context,
//!   the expert rule the inherited levers name): the server's name, argv and
//!   pid; the model's architecture (`qwen4exp`), shards, bytes on disk and
//!   counts as the header states them; a card `GPU<n>` and the host `CPU`,
//!   each device's `bytes` the sum of its classes and equal to the plan's
//!   card (dense + experts) and host (experts + tables) bytes; the cards' KV
//!   bytes; no draft;
//! - `/completion` of `--prompt` at temperature 0 with `return_tokens`: its
//!   ids are `generate_qwen3moe --tokens <--ids> -n 16`'s `tokens` line — all
//!   16, or a prefix ending in the end-of-generation id when the server
//!   stopped there (`generate_qwen3moe` does not stop at it). The prompt must
//!   hold at most eight ids: the passes both engines take below
//!   [`GEMM_FROM`](bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM) are bit
//!   for bit steps, so the server's prompt feed (the prompt less its last id)
//!   and `generate_qwen3moe`'s (the whole prompt) leave the same state, while
//!   past that each runs the prompt's last position through a different arm;
//! - the same `/completion` again, and once more after the other requests
//!   below: the same ids each time (a request keeps no prefix of another's —
//!   every request prefills from a reset, never a wrong state);
//! - `/v1/chat/completions` of one user turn at temperature 0: the streamed
//!   deltas concatenate to the non-streamed content, and the stream ends with
//!   `data: [DONE]`;
//! - `/tokenize` of `--prompt` is `--ids`;
//! - the positions the server serves: `/props`' `n_ctx` is the server's
//!   default context; a prompt of that many ids is a 400
//!   (`exceed_context_size_error`, naming it) and the server stays up.
//!
//! Then the server is killed by the handle this binary spawned it with and
//! waited for. Logs and the raw stream go to `--dir`.
//!
//! The server inherits this binary's environment, so the levers it acts on
//! are the server's (`ACTS_ON`, the same list): one the server would refuse
//! is refused here, at `main`, before the server starts.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen38_serve: built without the `deepseek41` feature; see `just gate-gpu-qwen38-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_serve", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::fs::File;
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::time::Duration;

    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu_gates::serve_client::{curl, ids_of, json_of, parse_ids};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, PlanInputs, machine_for_experts, read_head_rows,
    };
    use model::placement::PlanLevers;
    use model::placement::workstation::RTX_3090;
    use refset::arch::qwen4exp::mtp::DRAFT;
    use serde_json::{Value, json};

    /// The levers the server acts on — the same list
    /// `bloomery_serve_qwen38` parses, kept one with it: this gate starts the
    /// server with its own environment, so a lever the server would refuse is
    /// refused here first.
    const ACTS_ON: &[&str] = &[
        bloomery_levers::QWEN38_EXPERTS,
        bloomery_levers::HOT_LIST,
        bloomery_levers::CARD_BUDGET,
        bloomery_levers::PIN_MAIN,
        bloomery_levers::HOST_POPULATE,
        bloomery_levers::HOST_LOCK,
        bloomery_levers::CARD_DONTNEED,
        bloomery_levers::R8,
        bloomery_levers::DRAFT,
        bloomery_levers::MTP_HEAD_ROWS,
    ];

    const USAGE: &str = "usage: gate_qwen38_serve --gen <generate_qwen3moe log> --prompt <text> \
                         --ids <a,b,…> --dir <out>";

    /// The server's arguments after its path; `/props` must echo them.
    const SERVER_ARGS: [&str; 6] = ["--host", "127.0.0.1", "--port", "0", "--place", "gate"];
    /// The load takes tens of seconds; the bound is the spec's 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    /// The greedy requests' length, `generate_qwen3moe -n 16`'s.
    const N_PREDICT: usize = 16;
    /// The server's default context, the stores it sizes.
    const CTX: usize = 4096;
    /// The one chat turn the chat clauses send.
    const CHAT: &str = "What is the capital of France? Answer in one word.";
    /// The chat's reply length.
    const CHAT_PREDICT: usize = 32;

    /// `BLOOMERY_QWEN38_EXPERTS` as the plan's expert rule — the server's
    /// reading of the inherited environment.
    fn experts38(levers: &bloomery_levers::Levers) -> Result<Experts, GateError> {
        match levers.qwen38_experts() {
            "host" => Ok(Experts::Host),
            "card" => Ok(Experts::Card),
            other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
        }
    }

    /// The server this binary started; killed and reaped on every way out.
    /// `serve_client::Served`'s core with this gate's server's name — that
    /// module's `exe` names `bloomery-serve-ds41`.
    struct Served38 {
        child: Child,
    }

    impl Served38 {
        /// The server's path: `bloomery-serve-qwen38` beside this binary.
        fn exe() -> Result<PathBuf, GateError> {
            Ok(std::env::current_exe()?.with_file_name("bloomery-serve-qwen38"))
        }

        /// Starts `bloomery-serve-qwen38` beside this binary with `args`,
        /// stdout to `<dir>/server.out` and stderr to `<dir>/server.err`. The
        /// child is killed when this process dies, so a runner's bound that
        /// ends this process does not leave the server holding a card.
        fn spawn(args: &[&str], dir: &Path) -> Result<Served38, GateError> {
            let exe = Self::exe()?;
            let mut cmd = Command::new(&exe);
            cmd.args(args)
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
            Ok(Served38 { child })
        }

        /// Waits for the `listening on http://<addr>` line in the server's
        /// stderr, `polls` reads `poll` apart.
        fn address(
            &mut self,
            err_log: &Path,
            polls: usize,
            poll: Duration,
        ) -> Result<String, GateError> {
            for _ in 0..polls {
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
                std::thread::sleep(poll);
            }
            Err(format!("the server did not listen within {polls} polls").into())
        }

        fn stop(&mut self) -> Result<String, GateError> {
            self.child.kill()?;
            Ok(format!("{}", self.child.wait()?))
        }
    }

    impl Drop for Served38 {
        fn drop(&mut self) {
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// The `tokens [..]` line of a `generate_qwen3moe` log.
    fn gen_tokens(log: &Path) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", log.display()))?;
        parse_ids(line)
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

    struct Args {
        gen_log: PathBuf,
        prompt: String,
        ids: Vec<u32>,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut gen_log, mut prompt, mut ids, mut dir) = (None, None, None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--gen" => gen_log = Some(PathBuf::from(v)),
                "--prompt" => prompt = Some(v),
                "--ids" => ids = Some(parse_ids(&v)?),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (gen_log, prompt, ids, dir) {
            (Some(gen_log), Some(prompt), Some(ids), Some(dir)) => Ok(Args {
                gen_log,
                prompt,
                ids,
                dir,
            }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// `/props`' `engine` object (the module header) against the plan of the
    /// file the server opens, made here from its headers the way the server
    /// makes it, under the levers the server inherits; `argv` and `pid` are
    /// the process this gate spawned.
    fn props_engine(
        url: &dyn Fn(&str) -> String,
        argv: &[String],
        pid: u32,
        levers: &bloomery_levers::Levers,
    ) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!("props engine {e}");
        let mtp = levers.draft() == Some("mtp");
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|err| format!("open {}: {err}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let experts = experts38(levers)?;
        let ub = ubatch_for(CTX)?;
        let machine = machine_for_experts(
            RTX_3090,
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            experts,
        );
        // Under the draft the plan carries it (its granules, its store, its
        // row map and its program's arena beside the target's card terms),
        // and `/props` files its bytes as the card's `draft` class.
        let levers_plan = PlanLevers::from_levers(levers)?;
        let rows = match levers.mtp_head_rows() {
            Some(p) => Some(read_head_rows(p, &split, inputs.spec.vocab)?),
            None => None,
        };
        let terms = |dense: u64, experts_at: u64, host: u64, tables: u64, kv: u64, draft: u64| {
            (dense + experts_at + draft, host + tables, kv, draft)
        };
        let (card_bytes, host_bytes, kv, draft_bytes) = match mtp {
            false => {
                let plan =
                    inputs.plan_with(&machine, u64::try_from(CTX)?, &levers_plan, experts)?;
                let c = plan.cards.first().ok_or("the gate's plan has no card")?;
                terms(
                    c.dense_bytes,
                    c.expert_bytes,
                    plan.host.expert_bytes,
                    plan.host.table_bytes,
                    plan.cards.iter().map(|c| c.kv_bytes).sum(),
                    0,
                )
            }
            true => {
                let rows = rows.unwrap_or(HeadRows::Full);
                let draft =
                    Split::open(DRAFT).map_err(|e| format!("open the MTP draft {DRAFT}: {e}"))?;
                let mtp = MtpInputs::read(&draft, &split, &inputs, rows)?;
                let with = inputs.plan_mtp_with(
                    &machine,
                    u64::try_from(CTX)?,
                    &levers_plan,
                    &mtp,
                    experts,
                )?;
                let bytes = with.draft_card_bytes() + with.arena_bytes;
                let c = with
                    .plan
                    .cards
                    .first()
                    .ok_or("the gate's plan has no card")?;
                terms(
                    c.dense_bytes,
                    c.expert_bytes,
                    with.plan.host.expert_bytes,
                    with.plan.host.table_bytes,
                    with.plan.cards.iter().map(|c| c.kv_bytes).sum(),
                    bytes,
                )
            }
        };
        let mut file_bytes = 0u64;
        for i in 0..split.shard_count() {
            let shard = split.shard_path(i).ok_or("a shard without a path")?;
            file_bytes += std::fs::metadata(shard)?.len();
        }
        println!(
            "plan card bytes {card_bytes} (the draft's {draft_bytes} of it) host bytes \
             {host_bytes} kv {kv}; file {file_bytes} B in {} shards",
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
        let layers = format!("{}-{}", 0, inputs.spec.layers.len().saturating_sub(1));
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
                .all(|d| d["bytes"].as_u64().is_some_and(|b| Some(b) == class_sum(d)))
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
        match mtp {
            false => check(&mut ok, "props_engine_no_draft", e.get("draft").is_none()),
            true => {
                let d = &e["draft"];
                check(
                    &mut ok,
                    "props_engine_names_the_draft",
                    d["kind"] == json!("mtp")
                        && d["n_max"] == json!(3)
                        && d["model"]
                            .as_str()
                            .is_some_and(|m| m.starts_with("mtp-Qwen3.8"))
                        && devices
                            .first()
                            .is_some_and(|d| d["classes"]["draft"].as_u64() == Some(draft_bytes)),
                );
            }
        }
        Ok(ok)
    }

    /// `/props`' `n_ctx` is the server's default context, a prompt of that
    /// many ids is a 400 naming it, and the server answers `/health` after it.
    fn position_limit(url: &dyn Fn(&str) -> String, id: u32) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let n_ctx = json_of("/props", st, &body)?["default_generation_settings"]["n_ctx"].clone();
        let long = vec![id; CTX];
        let (st, body) = curl(
            &url("/completion"),
            Some(&json!({"prompt": long, "n_predict": 1, "temperature": 0})),
            false,
        )?;
        let refused: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let e = &refused["error"];
        println!(
            "positions served {CTX}: /props n_ctx {n_ctx}; a prompt of {CTX} ids: HTTP {st} {e}"
        );
        let (hst, hbody) = curl(&url("/health"), None, false)?;
        let mut ok = true;
        check(
            &mut ok,
            "props_n_ctx_is_the_default_context",
            n_ctx == json!(CTX),
        );
        check(
            &mut ok,
            "prompt_of_the_served_positions_is_a_400",
            st == 400
                && e["type"] == "exceed_context_size_error"
                && e["message"]
                    .as_str()
                    .is_some_and(|m| m.contains(&CTX.to_string())),
        );
        check(
            &mut ok,
            "health_after_the_400",
            hst == 200 && hbody.contains("\"ok\""),
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        let a = parse_args()?;
        let reference = gen_tokens(&a.gen_log)?;
        std::fs::create_dir_all(&a.dir)?;
        let exe = Served38::exe()?;
        let err_log = a.dir.join("server.err");
        let mut served = Served38::spawn(&SERVER_ARGS, &a.dir)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");

        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        println!("health {st} {body}");
        check(&mut ok, "health_ok", st == 200 && body.contains("\"ok\""));
        let argv: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(SERVER_ARGS.iter().map(|s| (*s).to_owned()))
            .collect();
        ok &= props_engine(&url, &argv, served.child.id(), &levers)?;

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
        println!("generate_qwen3moe {reference:?}");
        check(
            &mut ok,
            "completion_ids_are_generate_qwen3moe",
            agree(&first, &stop, &reference),
        );
        if levers.draft() == Some("mtp") {
            // The drafted server's own clause: the greedy ids above are the
            // plain run's (a draft changes which passes run, never a token),
            // and the pass carried its counts.
            let d = &c1["timings"];
            check(
                &mut ok,
                "drafted_timings_carry_the_draft_counts",
                d["draft_n"].as_u64().is_some_and(|n| n > 0)
                    && d["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
            );
        }

        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c2 = json_of("/completion", st, &body)?;
        println!(
            "completion again {:?} cache_n={}",
            ids_of(&c2["tokens"]),
            c2["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_again_identical",
            ids_of(&c2["tokens"]) == first,
        );

        let chat = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": CHAT_PREDICT,
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
        let c3 = json_of("/completion", st, &body)?;
        println!(
            "completion after the other requests {:?} cache_n={}",
            ids_of(&c3["tokens"]),
            c3["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_after_other_requests_identical",
            ids_of(&c3["tokens"]) == first,
        );

        let id = a.ids.iter().copied().min().ok_or("--ids is empty")?;
        ok &= position_limit(&url, id)?;

        println!("server stopped: {}", served.stop()?);
        if ok {
            println!("gate-gpu-qwen38-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
