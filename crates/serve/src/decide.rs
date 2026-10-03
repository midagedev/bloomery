//! A decision model's HTTP server over a [`Decide`]: one prompt pass per request, an answer body
//! out, no generation and no chat; and the decide seat's choice of a model ([`Row`], [`pick`]).
//!
//! Routes: `POST` on each of the row's routes (the request body to [`Decide::decide`]; the answer
//! with a `timings` object appended as its last key), `GET /props` ([`Decide::props`], read once at
//! bind, with the `engine` object the generative server gives), `GET /v1/models` and `/models` (the
//! seated model's listing, as the generative server's), `GET /health`, `OPTIONS` on any path (the
//! CORS answer [`crate::Server`] gives), and a 404 error object for anything else. A refused request
//! is a 400 carrying the decider's message, a request the engine cannot answer (an image) a 501
//! `not_supported_error`, an engine fault a 500. The decider runs one request at a time behind a mutex; connections are
//! served one thread each, at most [`MAX_CONNECTIONS`] at once, with the same keep-alive and read
//! bounds as [`crate::Server`].
//!
//! A decision model is named by its head, not by its backbone's architecture: `--head` given, or a
//! `--hf` repo whose model card names a row's head repo as the model it quantizes, seats it
//! ([`pick`]); the head's config picks the row ([`row_of_config`]).

use std::io::{self, BufReader};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::api::{
    JSON, Permit, closing, cors_preflight, engine_object, error_body, next_request_arrives, refuse,
};
use crate::http::{self, Request};
use crate::{EngineProps, MAX_CONNECTIONS};

/// One answered request.
#[derive(Clone, Debug)]
pub struct Decided {
    /// The response JSON object, as the model's reference writes it.
    pub body: String,
    /// The prompt's token count.
    pub prompt_n: usize,
    /// The backbone's prompt pass.
    pub prompt_ms: f64,
    /// Everything after the prompt pass: the head, its rows and the answer body.
    pub head_ms: f64,
}

/// Why a request got no answer.
#[derive(Debug, thiserror::Error)]
pub enum DecideError {
    /// The request itself was refused (a 400): malformed, invalid, or too long.
    #[error("{0}")]
    Refused(String),
    /// A valid request this engine cannot answer (a 501 `not_supported_error`, as llama.cpp's
    /// server answers an image to a text-only decision model).
    #[error("{0}")]
    NotSupported(String),
    /// The engine failed on a valid request (a 500).
    #[error("{0}")]
    Engine(String),
}

/// What a decide server says of the model it seats, beside the decider.
#[derive(Clone, Debug)]
pub struct Seated {
    /// The routes its requests are posted to (its row's).
    pub routes: &'static [&'static str],
    /// The model's name: `/v1/models`' id, and the `model` the answers carry.
    pub name: String,
    /// The context the backbone was sized for (`/v1/models`' `meta.n_ctx`).
    pub n_ctx: usize,
}

/// What every connection reads: the routes, `/props` and `/v1/models`, fixed at bind.
struct Fixed {
    routes: &'static [&'static str],
    props: Value,
    models: Value,
}

/// A decision model: a request body in, an answer out. The one seam between the server and a
/// model; the model's prompt, head and answer shape stay on its side.
pub trait Decide: Send {
    /// The answer to one request body.
    fn decide(&mut self, body: &str) -> Result<Decided, DecideError>;
    /// What `/props` says about the model: a JSON object, to which the server adds the `engine`
    /// object (its name, version, argv and pid).
    fn props(&self) -> Value;
}

/// The `--model` word of the decide seat.
pub const WORD: &str = "decide";

/// One decision model the decide seat serves: the facts it reads before it opens anything, and the
/// model's own part, `open`, which the seat calls.
#[derive(Debug)]
pub struct Row<O> {
    /// The row's name, as refusals and `/props` give it.
    pub name: &'static str,
    /// The routes its requests are posted to.
    pub routes: &'static [&'static str],
    /// The context unless `--ctx` says.
    pub ctx: usize,
    /// The file architectures whose hidden states the head reads.
    pub backbones: &'static [&'static str],
    /// The repo the head is fetched from under `--hf`.
    pub head_repo: &'static str,
    /// A repo whose model card names [`head_repo`](Row::head_repo) as the model it
    /// quantizes, whose set seats the row under `--hf`: the `--hf` a bare-backbone
    /// refusal points at.
    pub quant_repo: &'static str,
    /// The head's weights in that repo.
    pub head_file: &'static str,
    /// The head config's file name, beside the weights.
    pub config_file: &'static str,
    /// Whether a head config's text is this row's, or why not.
    pub knows: fn(&str) -> Result<(), String>,
    /// File architectures that carry this model in a layout the seat does not read, each with what
    /// the refusal says (what the file is, and what to serve instead).
    pub unserved: &'static [(&'static str, &'static str)],
    /// The model's own part.
    pub open: O,
}

/// What the command line asks of the decide seat.
#[derive(Clone, Copy, Debug)]
pub struct Ask<'a> {
    /// `--head`.
    pub head: Option<&'a Path>,
    /// `--head-config`.
    pub head_config: Option<&'a Path>,
    /// The `--model` word: [`WORD`], a generative seat's, or none.
    pub word: Option<&'a str>,
    /// The repo (`owner/name`) the model file was fetched from under `--hf`.
    pub hf: Option<&'a str>,
    /// The model file's architecture.
    pub arch: &'a str,
    /// Whether a generative seat serves `arch`.
    pub generative: bool,
}

/// Where the decide seat's head comes from.
#[derive(Debug)]
pub enum HeadFrom<'r, O> {
    /// `--head`, and `--head-config` when given.
    Given {
        head: PathBuf,
        config: Option<PathBuf>,
    },
    /// The row's head repo, which the `--hf` repo's card names as the model it quantizes.
    Fetch(&'r Row<O>),
}

/// The head the command line names, in this order: `--head`; under `--hf`, the head repo of the
/// row whose backbones hold the file's architecture and which the repo's card (`card(repo)`: the
/// model the card says the repo quantizes) names; else none. The card is read only when the decide
/// seat is asked for (`--model decide`) or the file is one no generative seat serves and a row's
/// backbone may be, so a generative seat's run reads no card.
///
/// `Ok(None)` is no head: the file goes to the generative seats, which refuse a row's backbone by
/// [`no_head`]. Refused by name: a file a row lists as unserved (whatever the flags), `--head`
/// beside a generative seat's word, `--head-config` without `--head`, `--model decide` with no head.
pub fn pick<'r, O>(
    ask: &Ask<'_>,
    rows: &'r [Row<O>],
    card: &mut dyn FnMut(&str) -> Result<Option<String>, String>,
) -> Result<Option<HeadFrom<'r, O>>, String> {
    if let Some((_, why)) = rows
        .iter()
        .flat_map(|r| r.unserved)
        .find(|(arch, _)| *arch == ask.arch)
    {
        return Err(format!("a {} file is {why}", ask.arch));
    }
    let decide = ask.word == Some(WORD);
    if let Some(head) = ask.head {
        if let Some(w) = ask.word.filter(|_| !decide) {
            return Err(format!(
                "--model {w} with --head {}: a head is the decide seat's (--model {WORD}, or no --model)",
                head.display()
            ));
        }
        return Ok(Some(HeadFrom::Given {
            head: head.to_path_buf(),
            config: ask.head_config.map(Path::to_path_buf),
        }));
    }
    if let Some(c) = ask.head_config {
        return Err(format!("--head-config {} without --head", c.display()));
    }
    let backbone = rows.iter().any(|r| r.backbones.contains(&ask.arch));
    if !decide && (ask.word.is_some() || ask.generative || !backbone) {
        return Ok(None);
    }
    if let Some(repo) = ask.hf {
        let base = card(repo)?;
        if let Some(row) = rows
            .iter()
            .find(|r| base.as_deref() == Some(r.head_repo) && r.backbones.contains(&ask.arch))
        {
            return Ok(Some(HeadFrom::Fetch(row)));
        }
    }
    if decide {
        return Err(format!("--model {WORD}: {}", no_head(ask.arch, rows)));
    }
    Ok(None)
}

/// The refusal of a file of architecture `arch` with no head, naming what would seat it.
pub fn no_head<O>(arch: &str, rows: &[Row<O>]) -> String {
    let known: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{} (head repo {}, backbone {}; --hf {})",
                r.name,
                r.head_repo,
                r.backbones.join(" or "),
                r.quant_repo
            )
        })
        .collect();
    format!(
        "a {arch} file with no head: a decision model is named by its head. Give --head <weights> \
         (its config beside them, or --head-config <file>), or --hf <repo>[:<quant>] of a repo whose \
         model card names a row's head repo as the model it quantizes; the rows: {}",
        known.join(", ")
    )
}

/// The row the head config picks, with the config's path: `config` (`--head-config`) when given,
/// else each row's config file beside `head`. A config no row knows is refused by name, listing
/// every row and why it does not know it.
pub fn row_of_config<'r, O>(
    rows: &'r [Row<O>],
    head: &Path,
    config: Option<&Path>,
) -> Result<(&'r Row<O>, PathBuf), String> {
    let mut why = Vec::new();
    for row in rows {
        let path = config.map_or_else(|| head.with_file_name(row.config_file), Path::to_path_buf);
        match std::fs::read_to_string(&path) {
            Ok(text) => match (row.knows)(&text) {
                Ok(()) => return Ok((row, path)),
                Err(e) => why.push(format!("{}: {}: {e}", row.name, path.display())),
            },
            Err(e) => why.push(format!("{}: {}: {e}", row.name, path.display())),
        }
    }
    Err(format!(
        "no decision model knows the head {}: {}",
        head.display(),
        why.join("; ")
    ))
}

/// `--head` and `--head-config`, as given.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeadFlags {
    pub head: Option<PathBuf>,
    pub config: Option<PathBuf>,
}

/// `args` with `--head` and `--head-config` taken out: their values and the rest. A flag with no
/// value, or given twice, is refused by name.
pub fn take_head(args: &[String]) -> Result<(HeadFlags, Vec<String>), String> {
    let (mut head, mut config) = (None, None);
    let mut rest = Vec::with_capacity(args.len());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--head" => &mut head,
            "--head-config" => &mut config,
            _ => {
                rest.push(a.clone());
                continue;
            }
        };
        let v = it.next().ok_or_else(|| format!("{a} needs a value"))?;
        if let Some(old) = slot.replace(PathBuf::from(v)) {
            return Err(format!("{a} is given twice: {} and {v}", old.display()));
        }
    }
    Ok((HeadFlags { head, config }, rest))
}

/// One reply, before it is written.
#[derive(Debug, PartialEq)]
struct Reply {
    status: u16,
    ctype: &'static str,
    extra: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn json(status: u16, v: &Value) -> Reply {
        Reply {
            status,
            ctype: JSON,
            extra: Vec::new(),
            body: v.to_string().into_bytes(),
        }
    }

    fn error(status: u16, kind: &str, message: &str) -> Reply {
        Reply::json(status, &error_body(status, kind, message))
    }
}

/// `d`'s body with `timings` appended as its last key; a body that is not a JSON object is an
/// engine fault. The body's own text is kept byte for byte.
fn with_timings(d: &Decided) -> Result<String, DecideError> {
    let not_object = || DecideError::Engine("the decider's answer is not a JSON object".to_owned());
    let parsed: Value = serde_json::from_str(&d.body)
        .map_err(|e| DecideError::Engine(format!("the decider's answer is not JSON: {e}")))?;
    let Value::Object(map) = parsed else {
        return Err(not_object());
    };
    let open = d.body.trim_end().strip_suffix('}').ok_or_else(not_object)?;
    let timings = json!({
        "prompt_n": d.prompt_n,
        "prompt_ms": d.prompt_ms,
        "head_ms": d.head_ms,
        "cache_n": 0,
    });
    let comma = if map.is_empty() { "" } else { "," };
    Ok(format!("{open}{comma}\"timings\":{timings}}}"))
}

/// The reply to one request.
fn reply(decider: &Mutex<Box<dyn Decide>>, fixed: &Fixed, req: &Request) -> Reply {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => Reply::json(200, &json!({ "status": "ok" })),
        ("GET", "/props") => Reply::json(200, &fixed.props),
        ("GET", "/v1/models" | "/models") => Reply::json(200, &fixed.models),
        ("OPTIONS", _) => Reply {
            status: 204,
            ctype: "text/plain",
            extra: cors_preflight().to_vec(),
            body: Vec::new(),
        },
        ("POST", path) if fixed.routes.contains(&path) => {
            let Ok(body) = std::str::from_utf8(&req.body) else {
                return Reply::error(
                    400,
                    "invalid_request_error",
                    "the request body is not UTF-8",
                );
            };
            let Ok(mut d) = decider.lock() else {
                return Reply::error(
                    500,
                    "server_error",
                    "the decider panicked in an earlier request",
                );
            };
            match d.decide(body).and_then(|a| with_timings(&a)) {
                Ok(text) => Reply {
                    status: 200,
                    ctype: JSON,
                    extra: Vec::new(),
                    body: text.into_bytes(),
                },
                Err(DecideError::Refused(m)) => Reply::error(400, "invalid_request_error", &m),
                Err(DecideError::NotSupported(m)) => Reply::error(501, "not_supported_error", &m),
                Err(DecideError::Engine(m)) => Reply::error(500, "server_error", &m),
            }
        }
        _ => Reply::error(404, "not_found_error", "File Not Found"),
    }
}

/// A bound, not yet running decision server.
pub struct DecideServer {
    listener: TcpListener,
    decider: Arc<Mutex<Box<dyn Decide>>>,
    fixed: Arc<Fixed>,
}

impl DecideServer {
    /// Binds `addr` and takes ownership of the decider, which `seated` describes; `/props` is read
    /// from it here ([`props_of`]). No route, one that is not a path or is one of the server's own,
    /// and props that are not an object or carry their own `engine`, are refused by name before the
    /// bind.
    pub fn bind(
        addr: impl ToSocketAddrs,
        seated: &Seated,
        decider: Box<dyn Decide>,
    ) -> io::Result<DecideServer> {
        let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidInput, e);
        check_routes(seated.routes).map_err(invalid)?;
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let fixed = Fixed {
            routes: seated.routes,
            props: props_of(decider.props()).map_err(invalid)?,
            models: crate::models::listing(
                &seated.name,
                created,
                json!({ "n_ctx": seated.n_ctx }),
                Some(seated.n_ctx),
            ),
        };
        let listener = TcpListener::bind(addr)?;
        Ok(DecideServer {
            listener,
            decider: Arc::new(Mutex::new(decider)),
            fixed: Arc::new(fixed),
        })
    }

    /// The bound address (the real port when bound to port 0).
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accepts connections until `accept` fails, and returns that error.
    pub fn run(self) -> io::Error {
        let live = Arc::new(AtomicUsize::new(0));
        for conn in self.listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => return e,
            };
            let Some(permit) = Permit::take(&live) else {
                refuse(
                    stream,
                    &format!("the server is serving {MAX_CONNECTIONS} connections, its limit"),
                );
                continue;
            };
            let (decider, fixed) = (Arc::clone(&self.decider), Arc::clone(&self.fixed));
            let spawned = thread::Builder::new()
                .name("decide".to_owned())
                .spawn(move || {
                    let _permit = permit;
                    serve_conn(&decider, &fixed, stream);
                });
            if let Err(e) = spawned {
                eprintln!("bloomery-serve: cannot start a connection thread: {e}");
            }
        }
        io::Error::other("the listener's accept loop ended")
    }
}

/// `/props`: the decider's object with the `engine` object every seat of the server gives (the
/// engine reports no model, placement or draft here; the decider's own keys name its files). Props
/// that are not an object, or that name `engine` themselves, are refused by name.
fn props_of(decider: Value) -> Result<Value, String> {
    let Value::Object(mut o) = decider else {
        return Err(format!(
            "the decider's props are not a JSON object: {decider}"
        ));
    };
    if o.contains_key("engine") {
        return Err("the decider's props name `engine`, which is the server's".to_owned());
    }
    o.insert("engine".to_owned(), engine_object(&EngineProps::default()));
    Ok(Value::Object(o))
}

/// Why `routes` cannot be a decider's: none, one that is not a path, or one the server answers itself.
fn check_routes(routes: &[&str]) -> Result<(), String> {
    if routes.is_empty() {
        return Err("a decider with no route".to_owned());
    }
    for r in routes {
        if !r.starts_with('/') || matches!(*r, "/health" | "/props") {
            return Err(format!(
                "the route {r:?} is not a path, or is one the server answers itself"
            ));
        }
    }
    Ok(())
}

fn serve_conn(decider: &Mutex<Box<dyn Decide>>, fixed: &Fixed, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::new(stream);
    loop {
        if !next_request_arrives(&mut r) {
            return;
        }
        let req = match http::read_request(&mut r, &mut w) {
            Ok(Some(req)) => req,
            Ok(None) => return,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                let a = Reply::error(400, "invalid_request_error", &e.to_string());
                let _ = http::respond(&mut w, &closing(), a.status, a.ctype, &a.extra, &a.body);
                return;
            }
            Err(_) => return,
        };
        let a = reply(decider, fixed, &req);
        if http::respond(&mut w, &req, a.status, a.ctype, &a.extra, &a.body).is_err()
            || !req.keep_alive
        {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::closing;

    /// An answer `{"a":1}` to every body but `"image"`, which it cannot answer.
    struct Echo;

    impl Decide for Echo {
        fn decide(&mut self, body: &str) -> Result<Decided, DecideError> {
            if body == "\"image\"" {
                return Err(DecideError::NotSupported("no image input".to_owned()));
            }
            Ok(Decided {
                body: r#"{"a":1}"#.to_owned(),
                prompt_n: 3,
                prompt_ms: 1.0,
                head_ms: 2.0,
            })
        }

        fn props(&self) -> Value {
            json!({})
        }
    }

    fn post(path: &str) -> Request {
        Request {
            method: "POST".to_owned(),
            path: path.to_owned(),
            body: b"{}".to_vec(),
            ..closing()
        }
    }

    const ROUTES: &[&str] = &["/v1/rerank", "/rerank"];

    fn fixed() -> Fixed {
        Fixed {
            routes: ROUTES,
            props: json!({}),
            models: crate::models::listing("m.gguf", 0, json!({}), None),
        }
    }

    #[test]
    fn the_routes_are_the_deciders_and_no_other() {
        let d: Mutex<Box<dyn Decide>> = Mutex::new(Box::new(Echo));
        let (fixed, routes) = (fixed(), ROUTES);
        for path in routes {
            let r = reply(&d, &fixed, &post(path));
            assert_eq!(r.status, 200, "{path}");
            assert!(
                String::from_utf8(r.body)
                    .unwrap()
                    .starts_with(r#"{"a":1,"timings":"#)
            );
        }
        assert_eq!(reply(&d, &fixed, &post("/v1/systemone")).status, 404);
        assert!(check_routes(&[]).unwrap_err().contains("no route"));
        assert!(check_routes(&["v1/x"]).is_err() && check_routes(&["/props"]).is_err());
        assert!(check_routes(routes).is_ok());
    }

    /// A request the engine cannot answer is llama.cpp's 501; the model list is the seated one's.
    #[test]
    fn not_supported_is_501_and_the_models_are_listed() {
        let d: Mutex<Box<dyn Decide>> = Mutex::new(Box::new(Echo));
        let fixed = fixed();
        let mut req = post("/rerank");
        req.body = b"\"image\"".to_vec();
        let r = reply(&d, &fixed, &req);
        let body: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(
            (r.status, &body["error"]["type"]),
            (501, &json!("not_supported_error"))
        );
        for path in ["/v1/models", "/models"] {
            let r = reply(
                &d,
                &fixed,
                &Request {
                    method: "GET".to_owned(),
                    ..post(path)
                },
            );
            let v: Value = serde_json::from_slice(&r.body).unwrap();
            assert_eq!(
                (r.status, &v["data"][0]["id"]),
                (200, &json!("m.gguf")),
                "{path}"
            );
        }
    }

    #[test]
    fn props_carry_the_servers_engine_object() {
        let p = props_of(json!({ "model": "m.gguf", "row": "clef" })).unwrap();
        assert_eq!((&p["model"], &p["row"]), (&json!("m.gguf"), &json!("clef")));
        assert_eq!(p["engine"]["name"], "bloomery");
        assert_eq!(p["engine"]["server_pid"], json!(std::process::id()));
        assert!(p["engine"]["version"].as_str().is_some() && p["engine"]["args"].is_array());
        assert!(
            props_of(json!({ "engine": "bloomery" }))
                .unwrap_err()
                .contains("name `engine`")
        );
        assert!(
            props_of(json!([]))
                .unwrap_err()
                .contains("not a JSON object")
        );
    }

    fn knows_a(text: &str) -> Result<(), String> {
        if text.contains("\"a\"") {
            Ok(())
        } else {
            Err("no key a".to_owned())
        }
    }

    const ROWS: &[Row<()>] = &[Row {
        name: "rowa",
        routes: &["/v1/a"],
        ctx: 16,
        backbones: &["qwen35"],
        head_repo: "Org/a",
        quant_repo: "q/a-GGUF:Q4",
        head_file: "a.safetensors",
        config_file: "a.json",
        knows: knows_a,
        unserved: &[("rowa_gguf", "rowa in another layout; serve --hf q/a")],
        open: (),
    }];

    fn ask(arch: &str) -> Ask<'_> {
        Ask {
            head: None,
            head_config: None,
            word: None,
            hf: None,
            arch,
            generative: false,
        }
    }

    /// A card that names `base` for every repo, counting the reads.
    fn card<'a>(
        base: Option<&str>,
        reads: &'a mut usize,
    ) -> impl FnMut(&str) -> Result<Option<String>, String> + 'a {
        let base = base.map(str::to_owned);
        move |_| {
            *reads += 1;
            Ok(base.clone())
        }
    }

    #[test]
    fn the_head_names_the_decide_seat() {
        let mut n = 0;
        let head = Path::new("/h/a.safetensors");
        // --head seats the decide seat, with no --model or with --model decide.
        for word in [None, Some(WORD)] {
            let a = Ask {
                head: Some(head),
                word,
                ..ask("qwen3moe")
            };
            assert!(matches!(
                pick(&a, ROWS, &mut card(None, &mut n)),
                Ok(Some(HeadFrom::Given { .. }))
            ));
        }
        // A row's quantized repo seats it under --hf, by its card, with or without --model decide.
        for word in [None, Some(WORD)] {
            let a = Ask {
                hf: Some("q/a-GGUF"),
                word,
                ..ask("qwen35")
            };
            let got = pick(&a, ROWS, &mut card(Some("Org/a"), &mut n));
            assert!(matches!(got, Ok(Some(HeadFrom::Fetch(r))) if r.name == "rowa"));
        }
        assert_eq!(n, 2);
    }

    #[test]
    fn a_generative_run_reads_no_card() {
        let mut n = 0;
        for a in [
            Ask {
                hf: Some("q/x"),
                generative: true,
                ..ask("qwen3moe")
            },
            Ask {
                hf: Some("q/x"),
                word: Some("qwen3"),
                ..ask("qwen35")
            },
            Ask {
                hf: Some("q/x"),
                ..ask("llama")
            },
        ] {
            assert!(
                matches!(pick(&a, ROWS, &mut card(Some("Org/a"), &mut n)), Ok(None)),
                "{a:?}"
            );
        }
        assert_eq!(n, 0);
    }

    #[test]
    fn a_seat_asked_without_its_head_is_refused_by_name() {
        let mut n = 0;
        let head = Path::new("/h/a.safetensors");
        // --model qwen3 with --head.
        let a = Ask {
            head: Some(head),
            word: Some("qwen3"),
            ..ask("qwen35")
        };
        let e = pick(&a, ROWS, &mut card(None, &mut n)).unwrap_err();
        assert!(e.contains("--model qwen3 with --head"), "{e}");
        // --head-config alone.
        let a = Ask {
            head_config: Some(head),
            ..ask("qwen35")
        };
        assert!(
            pick(&a, ROWS, &mut card(None, &mut n))
                .unwrap_err()
                .contains("without --head")
        );
        // --model decide on a file with no head: -m, and a --hf repo whose card names another base.
        for (hf, base) in [(None, None), (Some("q/b"), Some("Org/b"))] {
            let a = Ask {
                hf,
                word: Some(WORD),
                ..ask("qwen35")
            };
            let e = pick(&a, ROWS, &mut card(base, &mut n)).unwrap_err();
            assert!(
                e.contains("--model decide: a qwen35 file with no head") && e.contains("--head"),
                "{e}"
            );
            assert!(
                e.contains("rowa (head repo Org/a, backbone qwen35; --hf q/a-GGUF:Q4)"),
                "{e}"
            );
        }
        // No --model and no head: no seat here; the generative side refuses by no_head.
        assert!(matches!(
            pick(&ask("qwen35"), ROWS, &mut card(None, &mut n)),
            Ok(None)
        ));
        // A file of a layout a row does not read is refused whatever the flags name.
        for a in [
            ask("rowa_gguf"),
            Ask {
                head: Some(head),
                ..ask("rowa_gguf")
            },
            Ask {
                word: Some("qwen3"),
                ..ask("rowa_gguf")
            },
        ] {
            let e = pick(&a, ROWS, &mut card(None, &mut n)).unwrap_err();
            assert_eq!(
                e,
                "a rowa_gguf file is rowa in another layout; serve --hf q/a"
            );
        }
        // A card read that fails is the refusal.
        let a = Ask {
            hf: Some("q/a"),
            ..ask("qwen35")
        };
        let e = pick(&a, ROWS, &mut |_| Err("offline".to_owned())).unwrap_err();
        assert_eq!(e, "offline");
    }

    #[test]
    fn the_head_config_picks_the_row() {
        let dir = std::env::temp_dir().join(format!("bloomery-decide-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let head = dir.join("a.safetensors");
        std::fs::write(dir.join("a.json"), r#"{"a": 1}"#).unwrap();
        std::fs::write(dir.join("other.json"), r#"{"labels": 2}"#).unwrap();
        let (row, path) = row_of_config(ROWS, &head, None).unwrap();
        assert_eq!((row.name, path), ("rowa", dir.join("a.json")));
        let other = dir.join("other.json");
        let e = row_of_config(ROWS, &head, Some(&other)).unwrap_err();
        assert!(
            e.contains("no decision model knows the head")
                && e.contains("rowa: ")
                && e.contains("no key a"),
            "{e}"
        );
        let e = row_of_config(ROWS, &dir.join("none/x.safetensors"), None).unwrap_err();
        assert!(e.contains("rowa: ") && e.contains("none/a.json"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_head_flags_come_out_once() {
        let args = |a: &[&str]| a.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let (h, rest) = take_head(&args(&["--port", "0", "--head", "/h", "--ctx", "8"])).unwrap();
        assert_eq!(
            h,
            HeadFlags {
                head: Some(PathBuf::from("/h")),
                config: None
            }
        );
        assert_eq!(rest, args(&["--port", "0", "--ctx", "8"]));
        assert!(
            take_head(&args(&["--head", "/a", "--head", "/b"]))
                .unwrap_err()
                .contains("given twice")
        );
        assert!(
            take_head(&args(&["--head-config"]))
                .unwrap_err()
                .contains("needs a value")
        );
    }
}
