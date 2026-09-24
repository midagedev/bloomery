//! `gate_ds41_serve` — `bloomery-serve-ds41` on the 3090 (placement gate),
//! driven over HTTP, as one process under the GPU gate lock.
//!
//!     gate_ds41_serve --gen <generate_ds41 log> --prompt <text> --ids <a,b,…> --dir <out>
//!
//! Starts the server beside this binary (`--port 0 --place gate`), reads its
//! address from its stderr, waits for `/health`, then checks:
//!
//! - `/completion` of `--prompt` at temperature 0 with `return_tokens`: its
//!   ids are `generate_ds41 --tokens <--ids> -n 16`'s `tokens` line — all 16,
//!   or a prefix ending in the end-of-generation id when the server stopped
//!   there (`generate_ds41` does not stop at it);
//! - `/v1/chat/completions` of one user turn at temperature 0: the streamed
//!   deltas concatenate to the non-streamed content, and the stream ends with
//!   `data: [DONE]`;
//! - `/tokenize` of `--prompt` is `--ids`;
//! - the same `/completion` after those requests gives the same ids (the
//!   engine's reset between requests leaves nothing behind);
//! - prefix reuse (`cache_prompt`, default true): `--ids` then `--ids` plus
//!   its 8 greedy ids (`ignore_eos`) keeps every cached position (`timings.cache_n`), and a
//!   prompt that leaves the last cached position out takes that one back;
//!   each gives the greedy ids of the same prompt with `cache_prompt: false`.
//!
//! Then the server is killed by the handle this binary spawned it with and
//! waited for. Logs and the raw stream go to `--dir`.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_ds41_serve: built without the `deepseek41` feature; see `just gate-gpu-ds41-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ds41_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::time::Duration;

    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use serde_json::{Value, json};

    const USAGE: &str = "usage: gate_ds41_serve --gen <generate_ds41 log> --prompt <text> --ids <a,b,…> --dir <out>";
    /// The load takes tens of seconds; the bound is the spec's 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    const N_PREDICT: usize = 16;
    /// Greedy ids of each prefix-reuse request, run past the end-of-generation id
    /// (`ignore_eos`) so every one of them is compared.
    const REUSE_PREDICT: usize = 8;
    const CHAT: &str = "What is the capital of France? Answer in one word.";

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

    /// `a,b,c` or `[a, b, c]` as ids.
    fn parse_ids(s: &str) -> Result<Vec<u32>, GateError> {
        s.trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| {
                t.parse::<u32>()
                    .map_err(|e| format!("id {t:?}: {e}").into())
            })
            .collect()
    }

    /// The `tokens [..]` line of a `generate_ds41` log.
    fn gen_tokens(log: &Path) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", log.display()))?;
        parse_ids(line)
    }

    /// The server this gate started; killed and reaped on every way out.
    struct Served {
        child: Child,
    }

    impl Served {
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

    /// One request through curl: the status and the body.
    fn curl(url: &str, body: Option<&Value>, stream: bool) -> Result<(u16, String), GateError> {
        let mut c = Command::new("curl");
        c.args(["-sS", "--max-time", "600", "-w", "\n%{http_code}"]);
        if stream {
            c.arg("-N");
        }
        if let Some(b) = body {
            c.args(["-H", "Content-Type: application/json", "-d", &b.to_string()]);
        }
        let out = c.arg(url).output()?;
        if !out.status.success() {
            return Err(format!(
                "curl {url}: {} {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
        let text = String::from_utf8(out.stdout)?;
        let (body, code) = text
            .rsplit_once('\n')
            .ok_or_else(|| format!("curl {url}: no status line"))?;
        Ok((code.trim().parse()?, body.to_owned()))
    }

    fn json_of(what: &str, status: u16, body: &str) -> Result<Value, GateError> {
        if status != 200 {
            return Err(format!("{what}: HTTP {status}: {body}").into());
        }
        Ok(serde_json::from_str(body).map_err(|e| format!("{what}: {e}: {body}"))?)
    }

    fn ids_of(v: &Value) -> Vec<u32> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().and_then(|i| u32::try_from(i).ok()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Waits for the `listening on http://<addr>` line in the server's stderr.
    fn address(served: &mut Served, err_log: &Path) -> Result<String, GateError> {
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(err_log).unwrap_or_default();
            if let Some(addr) = text
                .lines()
                .find_map(|l| l.split_once("listening on http://").map(|(_, a)| a.trim()))
            {
                return Ok(addr.to_owned());
            }
            if let Some(status) = served.child.try_wait()? {
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

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        let reference = gen_tokens(&a.gen_log)?;
        std::fs::create_dir_all(&a.dir)?;
        let exe = std::env::current_exe()?.with_file_name("bloomery-serve-ds41");
        let err_log = a.dir.join("server.err");
        let child = Command::new(&exe)
            .args(["--host", "127.0.0.1", "--port", "0", "--place", "gate"])
            .stdin(Stdio::null())
            .stdout(File::create(a.dir.join("server.out"))?)
            .stderr(File::create(&err_log)?)
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        let mut served = Served { child };
        println!("server pid {}", served.child.id());
        let addr = address(&mut served, &err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");

        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        println!("health {st} {body}");
        check(&mut ok, "health_ok", st == 200 && body.contains("\"ok\""));

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
        println!("generate_ds41    {reference:?}");
        let matches = match stop.as_str() {
            // The server stopped at the end-of-generation id, which it returns.
            "eos" => !first.is_empty() && reference.starts_with(&first),
            _ => first == reference,
        };
        check(&mut ok, "completion_ids_are_generate_ds41", matches);

        let chat = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": 32,
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
        let again = json_of("/completion", st, &body)?;
        println!(
            "completion again {:?} cache_n={}",
            ids_of(&again["tokens"]),
            again["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_after_reset_identical",
            ids_of(&again["tokens"]) == first,
        );

        let reuse = |prompt: &[u32], cache: bool| -> Result<(Vec<u32>, Value), GateError> {
            let body = json!({
                "prompt": prompt, "n_predict": REUSE_PREDICT, "temperature": 0,
                "ignore_eos": true, "return_tokens": true, "cache_prompt": cache,
            });
            let (st, body) = curl(&url("/completion"), Some(&body), false)?;
            let v = json_of("/completion", st, &body)?;
            let ids = ids_of(&v["tokens"]);
            println!(
                "reuse prompt {} ids cache_prompt={cache}: cache_n={} tokens {ids:?}",
                prompt.len(),
                v["timings"]["cache_n"]
            );
            Ok((ids, v["timings"]["cache_n"].clone()))
        };
        let (g1, _) = reuse(&a.ids, false)?;
        let cont: Vec<u32> = a.ids.iter().chain(&g1).copied().collect();
        let (g2, c2) = reuse(&cont, true)?;
        let (g3, c3) = reuse(&cont, false)?;
        let full = g1.len() == REUSE_PREDICT && g3.len() == REUSE_PREDICT;
        check(&mut ok, "reuse_fixture_ran_to_n_predict", full);
        // The cache held the prompt and all but the last of g1: every position is kept.
        check(
            &mut ok,
            "reuse_continuation_cache_n",
            c2 == json!(cont.len() - 1),
        );
        check(
            &mut ok,
            "reuse_continuation_ids_are_fresh",
            g2 == g3 && c3 == json!(0),
        );
        if full {
            // The cache holds `cont` and g3[..7]; this prompt shares all of it but the
            // last, so the engine takes that one position back.
            let last = g3[REUSE_PREDICT - 2];
            let alt = if a.ids[0] != last { a.ids[0] } else { a.ids[1] };
            let mut back = cont.clone();
            back.extend(&g3[..REUSE_PREDICT - 2]);
            back.push(alt);
            let (g4, c4) = reuse(&back, true)?;
            let (g5, c5) = reuse(&back, false)?;
            check(
                &mut ok,
                "reuse_rollback_cache_n",
                c4 == json!(back.len() - 1),
            );
            check(
                &mut ok,
                "reuse_rollback_ids_are_fresh",
                g4 == g5 && c5 == json!(0),
            );
        }

        println!("server stopped: {}", served.stop()?);
        if ok {
            println!("gate-gpu-ds41-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
