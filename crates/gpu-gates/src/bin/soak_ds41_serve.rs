//! `soak_ds41_serve` — `bloomery-serve-ds41` at placement (a) on the A6000,
//! driven by a seeded request mix for `--minutes`, its process sampled every
//! 30 s. Memory creep is the failure it catches.
//!
//!     soak_ds41_serve --minutes <M> --dir <out> [--seed <n>]
//!
//! Starts the server beside this binary (`--port 0 --place a`, the hot list
//! it inherits through `BLOOMERY_HOT_LIST`), waits for it to listen, then
//! until the deadline sends a mix drawn from `--seed` (printed): turns of a
//! multi-turn chat that reuses the cached prefix and starts over past
//! [`CHAT_CAP`] positions, `/completion` with `cache_prompt: false`, streamed
//! and plain requests at temperature 0 and above, a stream dropped after
//! [`DROP_AFTER`] events, a second generation sent while one runs, and a
//! light endpoint after each generation. Prompts are ids of
//! `$BLOOMERY_DATA/engram/corpus-{prose,code}.ids`, chat text their
//! `/detokenize`. Every request stays under [`POSITION_CAP`] positions,
//! refused by name before it is sent. A drift probe (a fixed
//! temperature-0 `/completion`) runs first and every [`probe_every`].
//!
//! The warm-up is a fixed block that runs every kind at its largest shape; its
//! end is the `warm` sample. Samples (`<dir>/samples.tsv`) are taken at start,
//! at warm, every [`SAMPLE_EVERY`] and at end, between requests: `/metrics`,
//! then the server's `/proc` status and fd table read until three reads in a
//! row agree (settled), then its card memory by pid. Each request is a line of
//! `<dir>/requests.tsv`.
//!
//! The verdict, from warm on: card memory of the server pid equal to warm;
//! settled fds, sockets and threads equal to warm; anonymous RSS
//! (`RssAnon + RssShmem`) at most warm plus [`AnonBound`]; the file-backed
//! mappings as many as at warm; no 5xx, no other non-200 status, no
//! transport error except the deliberate drops, no SSE `error` event; the server alive at the
//! end; every drift probe equal to the first. One line per quantity, then
//! `soak: PASS` or `soak: FAIL <reasons>` and a non-zero exit.
//!
//! File-backed RSS is printed, not judged: the host set and the engram tables
//! are file mappings the kernel reclaims and faults back as it likes (the
//! host set is not locked unless `BLOOMERY_HOST_LOCK=1`), so their resident
//! size says nothing about a leak. A leak on that side is a mapping that is
//! never unmapped, which the mapping count catches.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("soak_ds41_serve: built without the `deepseek41` feature; see `just soak-ds41`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("soak_ds41_serve", soak::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
mod soak {
    use std::fs::File;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdout, Command, Stdio};
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::serve_client::{Served, curl, ids_of, json_of};
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::{HotList, PlanLevers};
    use serde_json::{Value, json};

    const USAGE: &str = "usage: soak_ds41_serve --minutes <M> --dir <out> [--seed <n>]";
    const SERVER_ARGS: [&str; 6] = ["--host", "127.0.0.1", "--port", "0", "--place", "a"];
    /// The load's bound: the serve gate's 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    const SAMPLE_EVERY: Duration = Duration::from_secs(30);
    /// A request's prompt plus its predictions stay under this many positions,
    /// well inside the positions the body computes (past them a step is an
    /// engine error, which ends the server).
    const POSITION_CAP: usize = 8192;
    /// A conversation starts over when its next turn would pass this.
    const CHAT_CAP: usize = 4096;
    /// Ids of a fresh `/completion` prompt: from 256 to this; the warm-up runs
    /// this length, four prompt batches.
    const FRESH_MAX: usize = 2048;
    /// Ids of a chat turn's new user message: from 64 to this.
    const USER_MAX: usize = 512;
    const DROP_AFTER: usize = 8;
    const PROBE_IDS: usize = 256;
    const PROBE_PREDICT: usize = 32;
    /// Settling: reads this far apart until this many in a row agree, at most
    /// `SETTLE_READS` reads.
    const SETTLE_GAP: Duration = Duration::from_millis(100);
    const SETTLE_AGREE: usize = 3;
    const SETTLE_READS: usize = 30;

    // ------------------------------------------------------------ arguments

    struct Args {
        minutes: u64,
        dir: PathBuf,
        seed: u64,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut minutes, mut dir, mut seed) = (None, None, 0x50ac_d541_u64);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--minutes" => minutes = Some(v.parse::<u64>()?),
                "--dir" => dir = Some(PathBuf::from(v)),
                "--seed" => seed = v.parse()?,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (minutes, dir) {
            (Some(m), _) if m < 2 => Err(format!(
                "--minutes {m}: the warm-up alone takes about a minute; give 2 or more"
            )
            .into()),
            (Some(minutes), Some(dir)) => Ok(Args { minutes, dir, seed }),
            _ => Err(USAGE.into()),
        }
    }

    /// The drift probe's period: a tenth of the run, from 30 s to 3 min.
    fn probe_every(minutes: u64) -> Duration {
        Duration::from_secs((minutes * 6).clamp(30, 180))
    }

    // ------------------------------------------------------------ the seeded draw

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// A value in `lo..=hi`.
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() % (hi - lo + 1) as u64) as usize
        }

        fn coin(&mut self) -> bool {
            self.next() & 1 == 1
        }

        /// Temperature 0 or 0.8, and a seed for the sampler.
        fn temp(&mut self) -> (f64, u64) {
            (if self.coin() { 0.8 } else { 0.0 }, self.next() >> 33)
        }
    }

    /// The generation kinds and their weights in the draw after the warm-up.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Kind {
        Chat,
        Fresh,
        Drop,
        Pair,
        Probe,
    }

    const DRAW: [(Kind, u64); 4] = [
        (Kind::Chat, 5),
        (Kind::Fresh, 2),
        (Kind::Drop, 1),
        (Kind::Pair, 1),
    ];

    fn draw(rng: &mut Rng) -> Kind {
        let total: u64 = DRAW.iter().map(|&(_, w)| w).sum();
        let mut x = rng.next() % total;
        for &(k, w) in &DRAW {
            if x < w {
                return k;
            }
            x -= w;
        }
        unreachable!("the weights sum to `total`")
    }

    const LIGHT: [&str; 8] = [
        "/tokenize",
        "/detokenize",
        "/health",
        "/slots",
        "/props",
        "/v1/models",
        "/apply-template",
        "/metrics",
    ];

    // ------------------------------------------------------------ corpus

    struct Corpus {
        prose: Vec<u32>,
        code: Vec<u32>,
    }

    fn read_ids(path: &Path, need: usize) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ids = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.trim()
                    .parse::<u32>()
                    .map_err(|e| format!("{}: id {l:?}: {e}", path.display()))
            })
            .collect::<Result<Vec<u32>, _>>()?;
        if ids.len() < need {
            return Err(format!(
                "{}: {} ids, the mix needs {need}",
                path.display(),
                ids.len()
            )
            .into());
        }
        Ok(ids)
    }

    impl Corpus {
        fn open() -> Result<Corpus, GateError> {
            let d = data_dir().join("engram");
            Ok(Corpus {
                prose: read_ids(&d.join("corpus-prose.ids"), 4 * USER_MAX)?,
                code: read_ids(&d.join("corpus-code.ids"), 4 * FRESH_MAX)?,
            })
        }

        fn slice(ids: &[u32], rng: &mut Rng, len: usize) -> Vec<u32> {
            let at = rng.range(0, ids.len() - len);
            ids[at..at + len].to_vec()
        }
    }

    // ------------------------------------------------------------ the derived bounds

    /// The growth of `RssAnon + RssShmem` from warm the server may show with
    /// no leak: every term a request can leave resident after warm without
    /// holding it.
    struct AnonBound {
        n_vocab: u64,
    }

    impl AnonBound {
        /// Threads that allocate on a request's path at once: main, the accept
        /// loop, the engine thread, and three connections (the pair and a
        /// `/metrics` read).
        const ARENAS: u64 = 6;
        const CONNECTIONS: u64 = 3;
        /// Rust's default thread stack.
        const STACK: u64 = 2 << 20;
        /// A connection's read buffer (`BufReader`'s default).
        const READ_BUF: u64 = 8 << 10;
        /// The largest body the mix sends; a body on the command line of curl
        /// stays under the kernel's one-argument limit.
        const BODY_MAX: u64 = 128 << 10;

        /// glibc keeps a freed top of each arena below its trim threshold,
        /// twice the largest freed mapped chunk: the sampler's candidates, 8 B
        /// an id, after one capacity doubling.
        fn arenas(&self) -> u64 {
            Self::ARENAS * 2 * 16 * self.n_vocab
        }

        /// Stacks of connection threads, glibc's cache reuses them.
        fn stacks() -> u64 {
            Self::CONNECTIONS * Self::STACK
        }

        /// The slot's held ids, at most the position cap, after one doubling.
        fn cache() -> u64 {
            2 * 4 * POSITION_CAP as u64
        }

        fn streams() -> u64 {
            Self::CONNECTIONS * (Self::READ_BUF + Self::BODY_MAX)
        }

        fn total(&self) -> u64 {
            self.arenas() + Self::stacks() + Self::cache() + Self::streams()
        }

        fn terms(&self) -> String {
            format!(
                "arenas {} ({} x 2 x 16 x n_vocab {}) + stacks {} + cache {} + streams {}",
                self.arenas(),
                Self::ARENAS,
                self.n_vocab,
                Self::stacks(),
                Self::cache(),
                Self::streams()
            )
        }
    }

    // ------------------------------------------------------------ the client and its counts

    #[derive(Default)]
    struct Counts {
        requests: u64,
        generations: u64,
        server_5xx: u64,
        other_status: u64,
        transport: u64,
        sse_errors: u64,
        drops: u64,
        drops_ran_out: u64,
        pairs: u64,
        pairs_deferred: u64,
        conversations: u64,
    }

    struct Client {
        base: String,
        t0: Instant,
        log: File,
        n: Counts,
    }

    /// What a request returned, when it returned at all.
    type Reply = Option<(u16, String)>;

    impl Client {
        fn url(&self, path: &str) -> String {
            format!("{}{path}", self.base)
        }

        fn note(&mut self, kind: &str, path: &str, status: &str, ms: u128, note: &str) {
            let t = self.t0.elapsed().as_secs_f64();
            let _ = writeln!(self.log, "{t:.1}\t{kind}\t{path}\t{status}\t{ms}\t{note}");
        }

        /// Counts a status: 200 is the only one this mix expects.
        fn status(&mut self, st: u16) {
            match st {
                200 => {}
                500..=599 => self.n.server_5xx += 1,
                _ => self.n.other_status += 1,
            }
        }

        /// One request through curl, counted and logged; a transport error is
        /// counted and returns `None`.
        fn send(&mut self, kind: &str, path: &str, body: Option<&Value>, stream: bool) -> Reply {
            self.n.requests += 1;
            let t = Instant::now();
            let r = curl(&self.url(path), body, stream);
            let ms = t.elapsed().as_millis();
            match r {
                Ok((st, text)) => {
                    self.status(st);
                    if stream {
                        self.n.sse_errors += sse_errors(&text);
                    }
                    let note = if st == 200 {
                        String::new()
                    } else {
                        first_line(&text)
                    };
                    self.note(kind, path, &st.to_string(), ms, &note);
                    Some((st, text))
                }
                Err(e) => {
                    self.n.transport += 1;
                    self.note(kind, path, "transport", ms, &first_line(&e.to_string()));
                    None
                }
            }
        }

        /// A 200 response's JSON; anything else is counted and gives `None`.
        fn json(&mut self, kind: &str, path: &str, body: Option<&Value>) -> Option<Value> {
            let (st, text) = self.send(kind, path, body, false)?;
            json_of(path, st, &text).ok()
        }
    }

    fn first_line(s: &str) -> String {
        s.lines().next().unwrap_or("").chars().take(200).collect()
    }

    /// The `data:` payloads of an SSE body.
    fn events(sse: &str) -> Vec<&str> {
        sse.split("\n\n")
            .filter_map(|e| e.trim_start().strip_prefix("data: "))
            .collect()
    }

    fn sse_errors(sse: &str) -> u64 {
        events(sse)
            .iter()
            .filter(|e| serde_json::from_str::<Value>(e).is_ok_and(|v| v.get("error").is_some()))
            .count() as u64
    }

    /// Refuses by name a request that would pass the position cap: the mix
    /// is wrong, and sending it would end the server.
    fn capped(what: &str, prompt: usize, predict: usize) -> Result<(), GateError> {
        if prompt + predict > POSITION_CAP {
            return Err(format!(
                "{what}: {prompt} prompt ids + {predict} predictions pass the soak's cap of \
                 {POSITION_CAP} positions — the mix is wrong, and the request would end the server"
            )
            .into());
        }
        Ok(())
    }

    // ------------------------------------------------------------ the requests

    struct Conversation {
        messages: Vec<Value>,
        /// The rendered ids of the last turn's prompt and its answer's length.
        held: usize,
    }

    struct Mix {
        c: Client,
        rng: Rng,
        corpus: Corpus,
        conv: Conversation,
        probe: Option<Vec<u32>>,
        probes: u64,
        probes_equal: u64,
        light_next: usize,
    }

    /// Text of `ids` through the server's `/detokenize`.
    fn text_of(c: &mut Client, kind: &str, ids: &[u32]) -> Option<String> {
        let v = c.json(kind, "/detokenize", Some(&json!({"tokens": ids})))?;
        v["content"].as_str().map(str::to_owned)
    }

    /// The rendered ids of `messages`, through the server's template and
    /// tokenizer.
    fn rendered(c: &mut Client, messages: &[Value]) -> Option<usize> {
        let v = c.json(
            "chat",
            "/apply-template",
            Some(&json!({"messages": messages})),
        )?;
        let text = v["prompt"].as_str()?.to_owned();
        let v = c.json("chat", "/tokenize", Some(&json!({"content": text})))?;
        Some(ids_of(&v["tokens"]).len())
    }

    impl Mix {
        /// One chat turn: a new user message of `user` ids; a streamed or plain
        /// answer of `predict` ids at `temp`. Starts the conversation over when
        /// the turn would pass [`CHAT_CAP`].
        fn chat(
            &mut self,
            user: usize,
            predict: usize,
            stream: bool,
            temp: (f64, u64),
        ) -> Result<(), GateError> {
            let ids = Corpus::slice(&self.corpus.prose, &mut self.rng, user);
            let Some(text) = text_of(&mut self.c, "chat", &ids) else {
                return Ok(());
            };
            if self.conv.held + user + predict + 64 > CHAT_CAP {
                self.conv.messages.clear();
                self.conv.held = 0;
            }
            if self.conv.messages.is_empty() {
                self.c.n.conversations += 1;
            }
            self.conv
                .messages
                .push(json!({"role": "user", "content": text}));
            let Some(n) = rendered(&mut self.c, &self.conv.messages) else {
                self.conv.messages.pop();
                return Ok(());
            };
            capped("chat turn", n, predict)?;
            let body = json!({
                "messages": self.conv.messages, "max_tokens": predict, "temperature": temp.0,
                "seed": temp.1, "stream": stream, "cache_prompt": true,
            });
            self.c.n.generations += 1;
            let kind = if stream { "chat-stream" } else { "chat" };
            let Some((st, text)) = self
                .c
                .send(kind, "/v1/chat/completions", Some(&body), stream)
            else {
                self.conv.messages.pop();
                return Ok(());
            };
            let answer = if st != 200 {
                None
            } else if stream {
                let mut s = String::new();
                for e in events(&text) {
                    if let Ok(v) = serde_json::from_str::<Value>(e)
                        && let Some(t) = v["choices"][0]["delta"]["content"].as_str()
                    {
                        s.push_str(t);
                    }
                }
                Some(s)
            } else {
                serde_json::from_str::<Value>(&text).ok().and_then(|v| {
                    v["choices"][0]["message"]["content"]
                        .as_str()
                        .map(str::to_owned)
                })
            };
            match answer {
                Some(a) => {
                    self.conv
                        .messages
                        .push(json!({"role": "assistant", "content": a}));
                    self.conv.held = n + predict;
                }
                None => {
                    self.conv.messages.pop();
                }
            }
            Ok(())
        }

        /// `/completion` of `len` code ids with `cache_prompt: false`.
        fn fresh(&mut self, len: usize, stream: bool, temp: (f64, u64)) -> Result<(), GateError> {
            const PREDICT: usize = 32;
            capped("fresh completion", len, PREDICT)?;
            let ids = Corpus::slice(&self.corpus.code, &mut self.rng, len);
            let body = json!({
                "prompt": ids, "n_predict": PREDICT, "temperature": temp.0, "seed": temp.1,
                "cache_prompt": false, "stream": stream, "return_tokens": true,
            });
            self.c.n.generations += 1;
            let kind = if stream { "fresh-stream" } else { "fresh" };
            self.c.send(kind, "/completion", Some(&body), stream);
            Ok(())
        }

        /// The drift probe: its ids against the first probe's.
        fn probe(&mut self) -> Result<(), GateError> {
            capped("probe", PROBE_IDS, PROBE_PREDICT)?;
            let body = json!({
                "prompt": &self.corpus.prose[..PROBE_IDS], "n_predict": PROBE_PREDICT,
                "temperature": 0, "cache_prompt": false, "return_tokens": true,
            });
            self.c.n.generations += 1;
            let ids = self
                .c
                .json("probe", "/completion", Some(&body))
                .map(|v| ids_of(&v["tokens"]))
                .unwrap_or_default();
            self.probes += 1;
            match &self.probe {
                None => {
                    println!("soak: probe reference {ids:?}");
                    self.probes_equal += u64::from(!ids.is_empty());
                    self.probe = Some(ids);
                }
                Some(first) => {
                    let same = !ids.is_empty() && &ids == first;
                    self.probes_equal += u64::from(same);
                    if !same {
                        println!("soak: probe {} differs: {ids:?}", self.probes);
                    }
                }
            }
            Ok(())
        }

        /// A streamed `/completion` the client drops after [`DROP_AFTER`]
        /// events: curl is killed with the stream open.
        fn drop_stream(&mut self, temp: (f64, u64)) -> Result<(), GateError> {
            const PREDICT: usize = 256;
            capped("dropped stream", PROBE_IDS, PREDICT)?;
            let ids = Corpus::slice(&self.corpus.prose, &mut self.rng, PROBE_IDS);
            let body = json!({
                "prompt": ids, "n_predict": PREDICT, "temperature": temp.0, "seed": temp.1,
                "cache_prompt": false, "stream": true,
            });
            self.c.n.requests += 1;
            self.c.n.generations += 1;
            let t = Instant::now();
            let mut s = Stream::open(&self.c.url("/completion"), &body)?;
            s.read(Some(DROP_AFTER));
            let _ = s.child.kill();
            let _ = s.child.wait();
            let ms = t.elapsed().as_millis();
            let (status, seen, ended) = (s.status, s.events, s.ended);
            match status {
                Some(st) => self.c.status(st),
                None => self.c.n.transport += 1,
            }
            self.c.n.sse_errors += s.errors;
            if ended {
                self.c.n.drops_ran_out += 1;
            } else {
                self.c.n.drops += 1;
            }
            let st = status.map_or("transport".to_owned(), |s| s.to_string());
            self.c.note(
                "drop",
                "/completion",
                &st,
                ms,
                &format!("events={seen} ended={ended}"),
            );
            Ok(())
        }

        /// Two generations at once: a streamed `/completion` and, once its
        /// first event is out, a plain chat, which waits for the one slot.
        fn pair(&mut self) -> Result<(), GateError> {
            capped("pair", PROBE_IDS, 64)?;
            let ids = Corpus::slice(&self.corpus.prose, &mut self.rng, PROBE_IDS);
            let a_body = json!({
                "prompt": ids, "n_predict": 64, "temperature": 0, "cache_prompt": false, "stream": true,
            });
            let ids = Corpus::slice(&self.corpus.prose, &mut self.rng, 128);
            let Some(text) = text_of(&mut self.c, "pair", &ids) else {
                return Ok(());
            };
            let b_body = json!({
                "messages": [{"role": "user", "content": text}], "max_tokens": 32,
                "temperature": 0, "cache_prompt": false,
            });
            self.c.n.requests += 2;
            self.c.n.generations += 2;
            self.c.n.pairs += 1;
            let t = Instant::now();
            let mut a = Stream::open(&self.c.url("/completion"), &a_body)?;
            a.read(Some(1));
            let b = Command::new("curl")
                .args(["-sS", "--max-time", "600", "-w", "\n%{http_code}"])
                .args([
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    &b_body.to_string(),
                ])
                .arg(self.c.url("/v1/chat/completions"))
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            let mut deferred = false;
            for _ in 0..50 {
                std::thread::sleep(Duration::from_millis(100));
                if let Some(m) = self.c.send("pair", "/metrics", None, false).map(|r| r.1)
                    && metric(&m, "requests_deferred").is_some_and(|d| d >= 1.0)
                {
                    deferred = true;
                    break;
                }
            }
            a.read(None);
            let a_rc = a.child.wait()?;
            let (a_status, a_events, a_done) = (a.status, a.events, a.ended);
            match a_status {
                Some(st) if a_rc.success() => self.c.status(st),
                Some(st) => {
                    self.c.status(st);
                    self.c.n.transport += 1;
                }
                None => self.c.n.transport += 1,
            }
            self.c.n.sse_errors += a.errors;
            let out = b.wait_with_output()?;
            let b_status = String::from_utf8_lossy(&out.stdout)
                .rsplit_once('\n')
                .and_then(|(_, code)| code.trim().parse::<u16>().ok());
            match b_status {
                Some(st) if out.status.success() => self.c.status(st),
                _ => self.c.n.transport += 1,
            }
            self.c.n.pairs_deferred += u64::from(deferred);
            let ms = t.elapsed().as_millis();
            let note = format!(
                "a={a_status:?} a_events={a_events} a_done={a_done} b={b_status:?} deferred_seen={deferred}"
            );
            self.c
                .note("pair", "/completion+/v1/chat/completions", "-", ms, &note);
            Ok(())
        }

        /// One light endpoint, in turn.
        fn light(&mut self) {
            let path = LIGHT[self.light_next % LIGHT.len()];
            self.light_next += 1;
            let text = "The soak asks the server for this sentence's ids.";
            let (get, body) = match path {
                "/tokenize" => (false, json!({"content": text})),
                "/detokenize" => (false, json!({"tokens": &self.corpus.prose[..64]})),
                "/apply-template" => (
                    false,
                    json!({"messages": [{"role": "user", "content": text}]}),
                ),
                _ => (true, Value::Null),
            };
            self.c.send("light", path, (!get).then_some(&body), false);
        }

        fn run_kind(&mut self, k: Kind) -> Result<(), GateError> {
            match k {
                Kind::Chat => {
                    let user = self.rng.range(64, USER_MAX);
                    let predict = if self.rng.coin() { 64 } else { 32 };
                    let stream = self.rng.coin();
                    let temp = self.rng.temp();
                    self.chat(user, predict, stream, temp)?;
                }
                Kind::Fresh => {
                    let len = self.rng.range(256, FRESH_MAX);
                    let stream = self.rng.coin();
                    let temp = self.rng.temp();
                    self.fresh(len, stream, temp)?;
                }
                Kind::Drop => {
                    let temp = self.rng.temp();
                    self.drop_stream(temp)?;
                }
                Kind::Pair => self.pair()?,
                Kind::Probe => self.probe()?,
            }
            self.light();
            Ok(())
        }

        /// Every kind at its largest shape, in a fixed order: the probe, a
        /// fresh prompt of [`FRESH_MAX`] ids plain at temperature 0 and a
        /// streamed one at 0.8, a conversation of [`USER_MAX`]-id turns up to
        /// the cap (streamed and plain, both temperatures), a dropped stream,
        /// a pair, and every light endpoint.
        fn warm_up(&mut self) -> Result<(), GateError> {
            self.probe()?;
            self.fresh(FRESH_MAX, false, (0.0, 1))?;
            self.fresh(FRESH_MAX / 2, true, (0.8, 2))?;
            let mut turn = 0u64;
            self.conv.messages.clear();
            self.conv.held = 0;
            while self.conv.held + USER_MAX + 64 + 64 <= CHAT_CAP {
                let temp = if turn.is_multiple_of(2) {
                    (0.0, turn)
                } else {
                    (0.8, turn)
                };
                let before = self.c.n.generations;
                self.chat(USER_MAX, 64, turn % 2 == 1, temp)?;
                turn += 1;
                if self.c.n.generations == before || turn > 16 {
                    break;
                }
            }
            self.drop_stream((0.8, 3))?;
            self.pair()?;
            for _ in 0..LIGHT.len() {
                self.light();
            }
            Ok(())
        }
    }

    /// A streamed request through curl, its head on stdout (`-D -`), read as
    /// it arrives: the status (the last head's), the `data:` events seen, the
    /// `error` events among them, and whether the stream ended (the
    /// connection closed or `data: [DONE]`).
    struct Stream {
        child: Child,
        r: BufReader<ChildStdout>,
        status: Option<u16>,
        events: usize,
        errors: u64,
        ended: bool,
    }

    impl Stream {
        fn open(url: &str, body: &Value) -> Result<Stream, GateError> {
            let mut child = Command::new("curl")
                .args(["-sS", "-N", "--max-time", "600", "-D", "-"])
                .args(["-H", "Content-Type: application/json"])
                .args(["-d", &body.to_string()])
                .arg(url)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()?;
            let out = child.stdout.take().ok_or("curl gave no stdout")?;
            Ok(Stream {
                child,
                r: BufReader::new(out),
                status: None,
                events: 0,
                errors: 0,
                ended: false,
            })
        }

        /// Reads until `stop` events have been seen in all, or the end.
        fn read(&mut self, stop: Option<usize>) {
            let mut line = String::new();
            while !self.ended && !stop.is_some_and(|n| self.events >= n) {
                line.clear();
                if !matches!(self.r.read_line(&mut line), Ok(n) if n > 0) {
                    self.ended = true;
                    break;
                }
                let l = line.trim_end();
                if l.starts_with("HTTP/") {
                    self.status = l.split_whitespace().nth(1).and_then(|s| s.parse().ok());
                } else if let Some(e) = l.strip_prefix("data: ") {
                    self.events += 1;
                    if e == "[DONE]" {
                        self.ended = true;
                    } else if serde_json::from_str::<Value>(e)
                        .is_ok_and(|v| v.get("error").is_some())
                    {
                        self.errors += 1;
                    }
                }
            }
        }
    }

    /// A `llamacpp:<name>` value of a `/metrics` text.
    fn metric(text: &str, name: &str) -> Option<f64> {
        let key = format!("llamacpp:{name} ");
        text.lines()
            .find_map(|l| l.strip_prefix(&key))
            .and_then(|v| v.trim().parse().ok())
    }

    // ------------------------------------------------------------ samples

    /// One read of the server's `/proc/<pid>/status` (bytes, and its thread
    /// count) and fd table.
    #[derive(Clone, Copy, Default, PartialEq, Eq)]
    struct Proc {
        vm_rss: u64,
        vm_hwm: u64,
        rss_anon: u64,
        rss_file: u64,
        rss_shmem: u64,
        threads: u64,
        fds: u64,
        sockets: u64,
        /// Mappings of a file outside `/dev` (`/proc/<pid>/maps`).
        file_maps: u64,
    }

    impl Proc {
        fn read(pid: u32) -> Result<Proc, GateError> {
            let status = std::fs::read_to_string(format!("/proc/{pid}/status"))?;
            let field = |name: &str| -> Result<u64, GateError> {
                let v = status
                    .lines()
                    .find_map(|l| l.strip_prefix(name).and_then(|r| r.strip_prefix(':')))
                    .ok_or_else(|| format!("/proc/{pid}/status has no {name}"))?;
                Ok(v.trim().trim_end_matches("kB").trim().parse()?)
            };
            let (mut fds, mut sockets) = (0u64, 0u64);
            for e in std::fs::read_dir(format!("/proc/{pid}/fd"))? {
                fds += 1;
                if std::fs::read_link(e?.path())
                    .is_ok_and(|l| l.to_string_lossy().starts_with("socket:"))
                {
                    sockets += 1;
                }
            }
            let maps = std::fs::read_to_string(format!("/proc/{pid}/maps"))?;
            let file_maps = maps
                .lines()
                .filter(|l| {
                    l.split_whitespace()
                        .nth(5)
                        .is_some_and(|p| p.starts_with('/') && !p.starts_with("/dev/"))
                })
                .count() as u64;
            Ok(Proc {
                vm_rss: field("VmRSS")? * 1024,
                vm_hwm: field("VmHWM")? * 1024,
                rss_anon: field("RssAnon")? * 1024,
                rss_file: field("RssFile")? * 1024,
                rss_shmem: field("RssShmem")? * 1024,
                threads: field("Threads")?,
                fds,
                sockets,
                file_maps,
            })
        }

        /// What a connection that has not ended yet moves.
        fn open_parts(&self) -> (u64, u64, u64) {
            (self.threads, self.fds, self.sockets)
        }

        /// Reads until [`SETTLE_AGREE`] reads in a row agree on the open
        /// parts, [`SETTLE_READS`] at most; the last read and whether it
        /// settled.
        fn settled(pid: u32) -> Result<(Proc, bool), GateError> {
            let mut last = Proc::read(pid)?;
            let mut agree = 1usize;
            for _ in 0..SETTLE_READS {
                if agree >= SETTLE_AGREE {
                    return Ok((last, true));
                }
                std::thread::sleep(SETTLE_GAP);
                let r = Proc::read(pid)?;
                agree = if r.open_parts() == last.open_parts() {
                    agree + 1
                } else {
                    1
                };
                last = r;
            }
            Ok((last, agree >= SETTLE_AGREE))
        }
    }

    #[derive(Clone, Default)]
    struct Sample {
        t: f64,
        phase: &'static str,
        p: Proc,
        settled: bool,
        card_mib: Option<u64>,
        positions: u64,
        prompt_total: u64,
        predicted_total: u64,
        kv_tokens: u64,
        requests: u64,
        generations: u64,
        errors: u64,
    }

    const HEADER: &str = "t_s\tphase\tvm_rss_kb\tvm_hwm_kb\trss_anon_kb\trss_file_kb\trss_shmem_kb\t\
                          threads\tfds\tsockets\tfile_maps\tsettled\tcard_mib\tprompt_tokens_total\t\
                          tokens_predicted_total\tkv_cache_tokens\trequests\tgenerations\terrors";

    impl Sample {
        fn row(&self) -> String {
            let card = self.card_mib.map_or("none".to_owned(), |m| m.to_string());
            format!(
                "{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{card}\t{}\t{}\t{}\t{}\t{}\t{}",
                self.t,
                self.phase,
                self.p.vm_rss / 1024,
                self.p.vm_hwm / 1024,
                self.p.rss_anon / 1024,
                self.p.rss_file / 1024,
                self.p.rss_shmem / 1024,
                self.p.threads,
                self.p.fds,
                self.p.sockets,
                self.p.file_maps,
                u8::from(self.settled),
                self.prompt_total,
                self.predicted_total,
                self.kv_tokens,
                self.requests,
                self.generations,
                self.errors
            )
        }

        fn heap(&self) -> u64 {
            self.p.rss_anon + self.p.rss_shmem
        }
    }

    /// The server's card memory on `uuid`, by pid (`None` when nvidia-smi
    /// does not list it there).
    fn card_mib(pid: u32, uuid: &str) -> Result<Option<u64>, GateError> {
        let out = Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()?;
        if !out.status.success() {
            return Err(format!("nvidia-smi: {}", out.status).into());
        }
        Ok(String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            (f.len() == 3 && f[0] == pid.to_string() && f[1] == uuid)
                .then(|| f[2].parse().ok())
                .flatten()
        }))
    }

    /// The card `CUDA_VISIBLE_DEVICES` names, refused unless it is one A6000.
    fn a6000_uuid() -> Result<String, GateError> {
        let want = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
        let out = Command::new("nvidia-smi")
            .args(["--query-gpu=uuid,name", "--format=csv,noheader"])
            .output()?;
        let listed = String::from_utf8_lossy(&out.stdout).into_owned();
        listed
            .lines()
            .find_map(|l| {
                let (u, name) = l.split_once(',')?;
                (u.trim() == want && name.contains("A6000")).then(|| u.trim().to_owned())
            })
            .ok_or_else(|| {
                format!(
                    "the soak runs on the A6000 (BLOOMERY_CARD=a6000 through tools/box.sh); \
                     CUDA_VISIBLE_DEVICES is {want:?}, and nvidia-smi lists:\n{listed}"
                )
                .into()
            })
    }

    struct Sampler {
        pid: u32,
        uuid: String,
        out: File,
        rows: Vec<Sample>,
    }

    impl Sampler {
        /// `/metrics`, then the settled `/proc` reads, then the card.
        fn take(&mut self, c: &mut Client, phase: &'static str) -> Result<(), GateError> {
            let m = c
                .send("sample", "/metrics", None, false)
                .map(|r| r.1)
                .unwrap_or_default();
            let get = |n: &str| metric(&m, n).map_or(0, |v| v as u64);
            let (p, settled) = Proc::settled(self.pid)?;
            let n = &c.n;
            let (prompt_total, predicted_total) =
                (get("prompt_tokens_total"), get("tokens_predicted_total"));
            let s = Sample {
                t: c.t0.elapsed().as_secs_f64(),
                phase,
                p,
                settled,
                card_mib: card_mib(self.pid, &self.uuid)?,
                positions: prompt_total + predicted_total,
                prompt_total,
                predicted_total,
                kv_tokens: get("kv_cache_tokens"),
                requests: n.requests,
                generations: n.generations,
                errors: n.server_5xx + n.other_status + n.transport + n.sse_errors,
            };
            writeln!(self.out, "{}", s.row())?;
            self.rows.push(s);
            Ok(())
        }
    }

    // ------------------------------------------------------------ the verdict

    fn mib(b: u64) -> f64 {
        b as f64 / f64::from(1 << 20)
    }

    /// The verdict lines, collecting the names of those that failed.
    struct Verdict(Vec<&'static str>);

    impl Verdict {
        fn line(&mut self, name: &'static str, pass: bool, text: &str) {
            println!("soak: {name} {text} -> {}", verdict(pass));
            if !pass {
                self.0.push(name);
            }
        }
    }

    /// The smallest and largest of `f` over `rows`.
    fn range(rows: &[Sample], f: impl Fn(&Sample) -> u64) -> (u64, u64) {
        let min = rows.iter().map(&f).min().unwrap_or(0);
        let max = rows.iter().map(&f).max().unwrap_or(0);
        (min, max)
    }

    /// Card memory, fds and threads: each equal to warm in every sample from
    /// warm on, and every one of those samples settled.
    fn judge_process(v: &mut Verdict, warm: &Sample, after: &[Sample]) {
        let card: Vec<Option<u64>> = after.iter().map(|s| s.card_mib).collect();
        v.line(
            "card_mib",
            warm.card_mib.is_some() && card.iter().all(|c| *c == warm.card_mib),
            &format!(
                "warm={:?} min={:?} max={:?} samples={}",
                warm.card_mib,
                card.iter().min().copied().flatten(),
                card.iter().max().copied().flatten(),
                after.len()
            ),
        );
        let unsettled = after.iter().filter(|s| !s.settled).count();
        let (fmin, fmax) = range(after, |s| s.p.fds);
        let (smin, smax) = range(after, |s| s.p.sockets);
        v.line(
            "fds",
            unsettled == 0
                && after
                    .iter()
                    .all(|s| s.p.fds == warm.p.fds && s.p.sockets == warm.p.sockets),
            &format!(
                "warm={} min={fmin} max={fmax}; sockets warm={} min={smin} max={smax}; \
                 unsettled samples {unsettled}",
                warm.p.fds, warm.p.sockets
            ),
        );
        let (tmin, tmax) = range(after, |s| s.p.threads);
        v.line(
            "threads",
            unsettled == 0 && after.iter().all(|s| s.p.threads == warm.p.threads),
            &format!("warm={} min={tmin} max={tmax}", warm.p.threads),
        );
    }

    /// Anonymous RSS against [`AnonBound`], the file mappings against warm,
    /// and file RSS and VmRSS printed.
    fn judge_memory(v: &mut Verdict, warm: &Sample, after: &[Sample], anon: &AnonBound) {
        let (_, hmax) = range(after, Sample::heap);
        let growth = hmax.saturating_sub(warm.heap());
        v.line(
            "rss_anon",
            growth <= anon.total(),
            &format!(
                "(RssAnon + RssShmem) warm={} B max={hmax} B growth={growth} B ({:.2} MiB), \
                 bound {} B ({:.2} MiB)",
                warm.heap(),
                mib(growth),
                anon.total(),
                mib(anon.total()),
            ),
        );
        let (mmin, mmax) = range(after, |s| s.p.file_maps);
        v.line(
            "file_maps",
            after.iter().all(|s| s.p.file_maps == warm.p.file_maps),
            &format!("warm={} min={mmin} max={mmax}", warm.p.file_maps),
        );
        let last = &after[after.len() - 1];
        let (fmin, fmax) = range(after, |s| s.p.rss_file);
        println!(
            "soak: rss_file warm={} B min={fmin} B max={fmax} B end={} B over {} positions evaluated \
             since warm — reclaimable file pages, printed, not judged",
            warm.p.rss_file,
            last.p.rss_file,
            last.positions.saturating_sub(warm.positions)
        );
        let (_, rmax) = range(after, |s| s.p.vm_rss);
        let rgrowth = rmax.saturating_sub(warm.p.vm_rss);
        println!(
            "soak: vm_rss warm={} B max={rmax} B growth={rgrowth} B ({:.1} MiB) — anon + file + shmem, \
             anon judged above; vm_hwm warm={} B end={} B",
            warm.p.vm_rss,
            mib(rgrowth),
            warm.p.vm_hwm,
            last.p.vm_hwm
        );
    }

    /// How the server ended.
    struct Ending {
        alive: bool,
        health: String,
    }

    /// Requests, the server's end, and the drift probes.
    fn judge_requests(v: &mut Verdict, mix: &Mix, end: &Ending) {
        let k = &mix.c.n;
        v.line(
            "requests",
            k.server_5xx == 0 && k.other_status == 0 && k.transport == 0 && k.sse_errors == 0,
            &format!(
                "total={} generations={} conversations={} 5xx={} other_status={} transport={} \
                 sse_errors={} drops={} (streams that ended before the drop {}) pairs={} \
                 (second request seen deferred in {})",
                k.requests,
                k.generations,
                k.conversations,
                k.server_5xx,
                k.other_status,
                k.transport,
                k.sse_errors,
                k.drops,
                k.drops_ran_out,
                k.pairs,
                k.pairs_deferred
            ),
        );
        v.line("alive", end.alive, &format!("health {}", end.health));
        v.line(
            "drift",
            mix.probes > 1 && mix.probes_equal == mix.probes,
            &format!("probes={} equal_to_first={}", mix.probes, mix.probes_equal),
        );
    }

    fn judge(mix: &Mix, rows: &[Sample], anon: &AnonBound, end: &Ending) -> Vec<&'static str> {
        let mut v = Verdict(Vec::new());
        match rows.iter().position(|s| s.phase == "warm") {
            Some(w) => {
                judge_process(&mut v, &rows[w], &rows[w..]);
                judge_memory(&mut v, &rows[w], &rows[w..], anon);
            }
            None => v.line("warm", false, "no warm sample was taken"),
        }
        judge_requests(&mut v, mix, end);
        v.0
    }

    // ------------------------------------------------------------ the run

    /// Starts the server and waits for it to listen: the server, its address,
    /// and its `plan` line, refused unless it is placement (a) on the A6000
    /// with the hot list this process was given.
    fn start(dir: &Path, hot: Option<&str>, uuid: &str) -> Result<(Served, String), GateError> {
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, dir)?;
        let t = Instant::now();
        let addr = served.address(&err_log, POLLS, POLL)?;
        let plan = std::fs::read_to_string(&err_log)?
            .lines()
            .find(|l| l.starts_with("plan "))
            .unwrap_or("")
            .to_owned();
        println!(
            "soak: server pid {} on {uuid} listening on {addr} after {:.1} s",
            served.child.id(),
            t.elapsed().as_secs_f64()
        );
        println!("soak: {plan}");
        let want = format!("hot_list={}", hot.unwrap_or("none"));
        if !plan.contains("place=a ") || !plan.contains("card=A6000") || !plan.contains(&want) {
            return Err(format!(
                "the server's plan is not placement (a) on the A6000 with {want}: {plan:?}"
            )
            .into());
        }
        Ok((served, addr))
    }

    /// The warm-up, then the draw until `minutes` have passed since `t0`,
    /// sampling and probing on their periods; stops early when the server
    /// exits.
    fn drive(
        mix: &mut Mix,
        sampler: &mut Sampler,
        served: &mut Served,
        minutes: u64,
    ) -> Result<(), GateError> {
        let t0 = mix.c.t0;
        let deadline = Duration::from_secs(minutes * 60);
        sampler.take(&mut mix.c, "start")?;
        mix.warm_up()?;
        sampler.take(&mut mix.c, "warm")?;
        println!(
            "soak: warm-up done at {:.1} s: {} requests, {} generations",
            t0.elapsed().as_secs_f64(),
            mix.c.n.requests,
            mix.c.n.generations
        );
        let mut next_sample = t0.elapsed() + SAMPLE_EVERY;
        let mut next_probe = t0.elapsed() + probe_every(minutes);
        while t0.elapsed() < deadline {
            if let Some(status) = served.child.try_wait()? {
                println!(
                    "soak: the server exited ({status}) at {:.1} s",
                    t0.elapsed().as_secs_f64()
                );
                return Ok(());
            }
            if t0.elapsed() >= next_sample {
                sampler.take(&mut mix.c, "run")?;
                next_sample += SAMPLE_EVERY;
            }
            let k = if t0.elapsed() >= next_probe {
                next_probe += probe_every(minutes);
                Kind::Probe
            } else {
                draw(&mut mix.rng)
            };
            mix.run_kind(k)?;
        }
        mix.probe()?;
        sampler.take(&mut mix.c, "end")
    }

    pub fn run() -> Result<(), GateError> {
        // The server starts with this process's environment: a lever it
        // would refuse is refused here first.
        let levers = bloomery_levers::at_main(crate::serve_levers::ACTS_ON)?;
        let a = parse_args()?;
        let place = PlanLevers::from_levers(&levers)?;
        let hot = place.hot.as_ref().map(HotList::path);
        let uuid = a6000_uuid()?;
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = Hparams::read(&split)?;
        let anon = AnonBound {
            n_vocab: hp.n_vocab as u64,
        };
        drop(split);
        let corpus = Corpus::open()?;
        std::fs::create_dir_all(&a.dir)?;
        println!(
            "soak: seed {:#x}, {} minutes, probe every {} s, sample every {} s, draw {DRAW:?} after \
             a fixed warm-up; position cap {POSITION_CAP}, conversation cap {CHAT_CAP}",
            a.seed,
            a.minutes,
            probe_every(a.minutes).as_secs(),
            SAMPLE_EVERY.as_secs()
        );
        println!("soak: rss_anon bound {} B = {}", anon.total(), anon.terms());

        let (mut served, addr) = start(&a.dir, hot, &uuid)?;
        let mut log = File::create(a.dir.join("requests.tsv"))?;
        writeln!(log, "t_s\tkind\tpath\tstatus\tms\tnote")?;
        let mut out = File::create(a.dir.join("samples.tsv"))?;
        writeln!(out, "{HEADER}")?;
        let mut sampler = Sampler {
            pid: served.child.id(),
            uuid,
            out,
            rows: Vec::new(),
        };
        let mut mix = Mix {
            c: Client {
                base: format!("http://{addr}"),
                t0: Instant::now(),
                log,
                n: Counts::default(),
            },
            rng: Rng(a.seed),
            corpus,
            conv: Conversation {
                messages: Vec::new(),
                held: 0,
            },
            probe: None,
            probes: 0,
            probes_equal: 0,
            light_next: 0,
        };
        let driven = drive(&mut mix, &mut sampler, &mut served, a.minutes);
        let health = mix.c.send("end", "/health", None, false);
        let end = Ending {
            alive: served.child.try_wait()?.is_none()
                && health.as_ref().is_some_and(|h| h.0 == 200),
            health: health.map_or("transport error".to_owned(), |(st, b)| {
                format!("{st} {}", first_line(&b))
            }),
        };
        if end.alive {
            println!("soak: server stopped: {}", served.stop()?);
        }
        driven?;
        let failed = judge(&mix, &sampler.rows, &anon, &end);
        if failed.is_empty() {
            println!("soak: PASS");
            Ok(())
        } else {
            println!("soak: FAIL {}", failed.join(" "));
            Err(checks_failed())
        }
    }
}
