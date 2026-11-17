//! The seats' flags as llama-server spells them: the context flag's
//! spellings ([`CTX`]) and bloomery's one request's context beside it
//! ([`SLOT_CTX`]), the prompt pass size's ([`UBATCH`]), a flag's number,
//! whose refusal names the flag and the value ([`number`]), and the API keys
//! ([`ApiKeys`]) both key flags add to and every request of both servers is
//! checked against.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::admit::{Limits, Meter};

/// Startup configuration files are read once per path, including failures and pipe inputs.
pub(crate) fn file_snapshot(path: &Path) -> Result<String, String> {
    static READ: Mutex<BTreeMap<PathBuf, Result<String, String>>> = Mutex::new(BTreeMap::new());
    let mut read = READ
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    read.entry(path.to_path_buf())
        .or_insert_with(|| std::fs::read_to_string(path).map_err(|e| e.to_string()))
        .clone()
}

/// The spellings of the context flag: `--ctx`, and llama-server's
/// `--ctx-size` and `-c`.
pub const CTX: [&str; 3] = ["--ctx", "--ctx-size", "-c"];

/// The flag that names one request's context, a slot's, apart from the slot count: `--ctx`
/// is the total the slots split, and `--slot-ctx S` beside `--parallel N` asks `S × N`.
pub const SLOT_CTX: &str = "--slot-ctx";

/// The spellings of the prompt pass's size flag: `--ubatch-size` and
/// llama-server's `-ub`. Unset, a seat's load chooses the rows from the
/// card's room; set, the seat takes exactly those rows or refuses by name
/// with the free bytes, the need and the largest size that fits.
pub const UBATCH: [&str; 2] = ["--ubatch-size", "-ub"];

/// The spellings of the key flags: `--api-key` and `--api-key-file`.
pub const KEYS: [&str; 2] = ["--api-key", "--api-key-file"];

/// The flags a caller's command line starts the server's optional parts with: the routes of a part
/// the server does not hold answer 501 and name these, and a caller that takes none names none, so
/// the answer never points at a flag its server refuses ([`Server::with_starts`]).
///
/// [`Server::with_starts`]: crate::Server::with_starts
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Starts {
    /// The flags that seat the embeddings side model (`--embd-model`); none when the caller takes
    /// no side model.
    pub side_model: &'static [&'static str],
    /// The flag that names the slot save directory (`--slot-save-path`); `None` when the caller
    /// takes none.
    pub slot_save: Option<&'static str>,
}

impl Starts {
    /// The 501 text of an absent part: llama-server's `This server does not support <what>` and,
    /// when the caller has flags that start it, `Start it with` those; with none, that the
    /// server has no `<part>`.
    pub(crate) fn not_served(what: &str, part: &str, flags: &[&str]) -> String {
        if flags.is_empty() {
            return format!("This server does not support {what}: it has no {part}");
        }
        let flags: Vec<String> = flags.iter().map(|f| format!("`{f}`")).collect();
        format!(
            "This server does not support {what}. Start it with {}",
            flags.join(" or ")
        )
    }
}

/// `v`, the value of `flag`, as a number; a value that is not one is refused
/// naming both, as `--ctx-size "12k": invalid digit found in string`.
pub fn number<T>(flag: &str, v: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    v.parse().map_err(|e| format!("{flag} {v:?}: {e}"))
}

/// The server's API keys: what `--api-key` and `--api-key-file` add to
/// ([`ApiKeys::add`]) and what every request of both servers is checked
/// against before its dispatch ([`ApiKeys::check`]). No key set checks
/// nothing, as llama-server's empty key list does.
///
/// The check is llama-server's `middleware_validate_api_key`: no key set
/// passes everything; `OPTIONS` passes (llama-server answers it in its
/// pre-routing handler, before the check); `/health` and `/v1/health` pass
/// (llama-server's public set is those and its web UI's asset paths, which
/// this server has none of); everything else needs one of the set's keys in
/// `Authorization`, else in `X-Api-Key`, with a `Bearer ` prefix stripped
/// from whichever header the key arrived in. Routes llama-server has no
/// counterpart for follow the same rule, so they need the key too: the
/// generation and tokenizer routes, `/props`, `/slots` and `/slots/{id}`'s
/// actions, `/metrics`, `/v1/models` (`/models`), `/v1/messages` and its
/// `count_tokens`, `/residency/reset`, `POST /shutdown` (loopback-only as
/// well) and the decision server's routes (`/v1/systemone`).
///
/// A key file line may narrow its key past llama-server's: `scope=` limits
/// the routes it reaches ([`Scope`]; a route outside them is a 403), and
/// `inflight=`, `rate=` and `burst=` limit its requests
/// ([`crate::admit`]; a request past them is a 429 with `Retry-After`). A
/// key with none of them reaches every route unlimited, as before.
#[derive(Debug, Default, Clone)]
pub struct ApiKeys {
    /// The keys, in the order the flags added them, each once.
    keys: Vec<Key>,
    /// The POST routes a decide server seats ([`crate::decide`]), which
    /// `scope=inference` reaches beside [`Scope::of`]'s, each a decision that
    /// takes its key's in-flight place.
    pub(crate) inference_routes: &'static [&'static str],
}

/// One key, what it reaches and its limits; clones share the limits' count.
#[derive(Debug, Clone)]
struct Key {
    key: String,
    scope: Scope,
    meter: Arc<Meter>,
}

/// The routes a key reaches, a key file line's `scope=` with one or more of
/// `inference`, `read` and `admin`, comma-separated: `inference` the model
/// routes (generation, token counts, the tokenizer, embeddings, the stored
/// responses, the decision routes) and the model list; `read` the status
/// routes (`GET /props`, `/slots`, `/metrics`) and the model list; `admin`
/// every route, as a key with no `scope=` reaches. The routes neither of the
/// first two names (slot actions, `/residency/reset`, `/shutdown`) need
/// `admin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Scope(u8);

impl Scope {
    const INFERENCE: Scope = Scope(1);
    const READ: Scope = Scope(2);
    /// The routes only `admin` reaches.
    const ADMIN: Scope = Scope(4);
    /// `admin`: every route.
    const ALL: Scope = Scope(7);

    /// `scope=`'s value, or the error naming it.
    fn parse(value: &str) -> Result<Scope, String> {
        value.split(',').try_fold(Scope(0), |s, name| match name {
            "inference" => Ok(Scope(s.0 | Scope::INFERENCE.0)),
            "read" => Ok(Scope(s.0 | Scope::READ.0)),
            "admin" => Ok(Scope::ALL),
            _ => Err(format!(
                "scope={value}: takes inference, read or admin, comma-separated"
            )),
        })
    }

    /// The scopes any one of which reaches `method` `path`.
    fn of(method: &str, path: &str) -> Scope {
        use crate::api::path;
        const MODEL_ROUTES: [&str; 17] = [
            path::COMPLETION,
            "/completions",
            path::COMPLETIONS,
            path::CHAT,
            "/chat/completions",
            "/v1/chat/completions/input_tokens",
            "/chat/completions/input_tokens",
            path::MESSAGES,
            "/v1/messages/count_tokens",
            path::RESPONSES,
            "/responses",
            path::EMBEDDINGS,
            "/embeddings",
            "/embedding",
            path::TOKENIZE,
            path::DETOKENIZE,
            "/apply-template",
        ];
        let stored = crate::http::below(path, "responses").is_some();
        let model_info = match method {
            "GET" => {
                matches!(path, "/v1/models" | "/models" | "/api/tags")
                    || crate::models::named(path).is_some()
            }
            "POST" => path == "/api/show",
            _ => false,
        };
        match method {
            _ if model_info => Scope(Scope::INFERENCE.0 | Scope::READ.0),
            "GET" if matches!(path, "/props" | "/slots" | "/metrics") => Scope::READ,
            "POST" if MODEL_ROUTES.contains(&path) || decision::request::ROUTES.contains(&path) => {
                Scope::INFERENCE
            }
            "GET" | "DELETE" | "POST" if stored => Scope::INFERENCE,
            _ => Scope::ADMIN,
        }
    }
}

/// Why the key check refused a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Denied {
    /// No key, or one the set does not hold: llama-server's 401.
    Key,
    /// A known key whose `scope=` does not reach the route: a 403.
    Scope,
}

impl ApiKeys {
    /// `flag`'s value added to the set. `--api-key` takes llama-server's
    /// comma-separated list — a quoted field may hold commas, `""` inside
    /// quotes an escaped quote — with empty items dropped; `--api-key-file`
    /// takes one key per line, skipping empty lines and `#` comments, each
    /// key optionally followed, after whitespace, by `name=value` fields:
    /// `scope=` ([`Scope`]), `inflight=`, `rate=` and `burst=`
    /// ([`crate::admit`]). A file that cannot be read is an error naming it,
    /// and so is a field it does not take, by its line; a flag whose value
    /// names no key at all is an error naming it, where llama-server would
    /// start with the check off, serving anyone who reaches the port while
    /// the user believes it locked. A key given twice is kept once; given
    /// again with other fields, it is an error naming where.
    ///
    /// A key file is a startup snapshot: the process reads each path once,
    /// and every later parse of the same flags takes what that read found,
    /// the keys or the error, so a pipe (`/dev/stdin`, a FIFO) holds its keys for a start that
    /// parses the flags more than once, and a change to the file after the
    /// first read is not seen.
    pub fn add(&mut self, flag: &str, value: &str) -> Result<(), String> {
        let unlimited = || (Scope::ALL, Limits::default());
        let added: Vec<(String, (Scope, Limits), String)> = match flag {
            "--api-key" => csv_row(value)
                .into_iter()
                .filter(|k| !k.is_empty())
                .map(|k| (k, unlimited(), flag.to_owned()))
                .collect(),
            "--api-key-file" => {
                let text = file_snapshot(Path::new(value))
                    .map_err(|e| format!("{flag} {value:?}: cannot read it: {e}"))?;
                let mut added = Vec::new();
                for (i, line) in text.lines().enumerate() {
                    let mut fields = line.split_whitespace();
                    let Some(key) = fields.next().filter(|k| !k.starts_with('#')) else {
                        continue;
                    };
                    let at = format!("{flag} {value:?} line {}", i + 1);
                    let narrowed = narrowing(fields).map_err(|e| format!("{at}: {e}"))?;
                    added.push((key.to_owned(), narrowed, at));
                }
                added
            }
            other => return Err(format!("{other} is not one of the key flags")),
        };
        if added.is_empty() {
            return Err(format!(
                "{flag} {value:?}: names no key, and a server with no key checks no request"
            ));
        }
        for (key, (scope, limits), at) in added {
            match self.keys.iter().find(|k| k.key == key) {
                Some(k) if (k.scope, k.meter.limits()) == (scope, limits) => {}
                Some(_) => {
                    return Err(format!(
                        "{at}: gives a key again with another scope or limits than it had"
                    ));
                }
                None => self.keys.push(Key {
                    key,
                    scope,
                    meter: Arc::new(Meter::new(limits)),
                }),
            }
        }
        Ok(())
    }

    /// Whether no key is set, so every request passes the check.
    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The key check of a request, its method, path and the two headers a key
    /// can ride on, against this set, as [the type](ApiKeys) states: the
    /// limits its key counts it against, `None` when it passes without a key
    /// (no key set, or a public route), or why it is refused.
    pub(crate) fn check(
        &self,
        method: &str,
        path: &str,
        authorization: Option<&str>,
        x_api_key: Option<&str>,
    ) -> Result<Option<&Arc<Meter>>, Denied> {
        if self.keys.is_empty() {
            return Ok(None);
        }
        if method == "OPTIONS" || matches!(path, "/health" | "/v1/health") {
            return Ok(None);
        }
        let sent = authorization
            .filter(|v| !v.is_empty())
            .or(x_api_key)
            .map(|v| v.strip_prefix("Bearer ").unwrap_or(v));
        let key = sent
            .and_then(|s| self.keys.iter().find(|k| k.key == s))
            .ok_or(Denied::Key)?;
        let reaching = if method == "POST" && self.inference_routes.contains(&path) {
            Scope::INFERENCE
        } else {
            Scope::of(method, path)
        };
        if key.scope.0 & reaching.0 == 0 {
            return Err(Denied::Scope);
        }
        Ok(Some(&key.meter))
    }
}

/// A key file line's fields past its key: the scope (every route unless
/// set) and the limits.
fn narrowing<'a>(fields: impl Iterator<Item = &'a str>) -> Result<(Scope, Limits), String> {
    let mut scope = Scope::ALL;
    let mut limits = Vec::new();
    let mut named = Vec::new();
    for (n, field) in fields.enumerate() {
        let name = field.split_once('=').map(|(name, _)| name);
        if let Some(name) = name.filter(|name| named.contains(name)) {
            return Err(format!("field {} after the key names {name}= again", n + 1));
        }
        named.extend(name);
        // A field not taken is named by its place alone: it may be the rest
        // of a key that holds whitespace.
        match field.split_once('=') {
            Some(("scope", value)) => scope = Scope::parse(value)?,
            Some(limit @ ("inflight" | "rate" | "burst", _)) => limits.push(limit),
            _ => {
                return Err(format!(
                    "field {} after the key is not scope=, inflight=, rate= or burst= (a key \
                     holds no whitespace)",
                    n + 1
                ));
            }
        }
    }
    Ok((scope, Limits::parse(limits)?))
}

/// One `--api-key` value's fields, llama-server's `parse_csv_row`: `"`
/// opens a quoted field (only at a field's start; elsewhere it is literal),
/// `""` inside quotes is one `"`, and a comma inside quotes joins the field.
/// The walk is over characters, so a key outside ASCII survives whole.
/// The last field is returned even when empty; the caller drops the empties.
fn csv_row(v: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let chars: Vec<char> = v.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' {
            if !quoted {
                if field.is_empty() {
                    quoted = true;
                } else {
                    field.push('"');
                }
            } else if chars.get(i + 1) == Some(&'"') {
                field.push('"');
                i += 1;
            } else {
                quoted = false;
            }
        } else if ch == ',' && !quoted {
            fields.push(std::mem::take(&mut field));
        } else {
            field.push(ch);
        }
        i += 1;
    }
    fields.push(field);
    fields
}

/// The in-crate tests' key file: `keys` written to a file of its own (a
/// path is read once a process, so each `name` is a new one) and added by
/// it; the set or the error, and the file's path.
#[cfg(test)]
pub(crate) fn key_file(name: &str, keys: &str) -> (Result<ApiKeys, String>, String) {
    let dir = std::env::temp_dir().join(format!("bloomery-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the key file's directory");
    let file = dir.join("keys");
    std::fs::write(&file, keys).expect("the key file");
    let path = file.display().to_string();
    let mut set = ApiKeys::default();
    let added = set.add("--api-key-file", &path).map(|()| set);
    std::fs::remove_dir_all(&dir).expect("the key file's directory removed");
    (added, path)
}

#[cfg(test)]
mod tests {
    use super::{ApiKeys, CTX, Denied, KEYS, UBATCH, csv_row, key_file, number};

    impl ApiKeys {
        /// Whether the key check lets the request through.
        fn allows(&self, method: &str, path: &str, auth: Option<&str>, x: Option<&str>) -> bool {
            self.check(method, path, auth, x).is_ok()
        }
    }

    /// A key file line's `scope=`: `inference` reaches the model routes and
    /// the model list, `read` the status routes and the model list, `admin`
    /// (and a key with no `scope=`) every route; a route outside the scope is
    /// [`Denied::Scope`], an unknown key still [`Denied::Key`].
    #[test]
    fn a_key_scope_reaches_its_routes_and_no_other() {
        let (keys, _) = key_file(
            "scopes",
            "sk-gen scope=inference\nsk-look   scope=read\nsk-both scope=read,inference\n\
             sk-root scope=admin\nsk-plain\n",
        );
        let keys = keys.unwrap();
        let check = |key: &str, method: &str, path: &str| {
            keys.check(method, path, Some(key), None).map(|_| ())
        };
        // Every model route `route()` serves, the aliases too.
        let model = [
            ("POST", "/completion"),
            ("POST", "/completions"),
            ("POST", "/v1/completions"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/chat/completions"),
            ("POST", "/v1/chat/completions/input_tokens"),
            ("POST", "/chat/completions/input_tokens"),
            ("POST", "/v1/messages"),
            ("POST", "/v1/messages/count_tokens"),
            ("POST", "/v1/responses"),
            ("POST", "/responses"),
            ("POST", "/v1/responses/input_tokens"),
            ("POST", "/responses/input_tokens"),
            ("GET", "/v1/responses/resp_1"),
            ("DELETE", "/responses/resp_1"),
            ("POST", "/v1/responses/resp_1/cancel"),
            ("POST", "/v1/embeddings"),
            ("POST", "/embeddings"),
            ("POST", "/embedding"),
            ("POST", "/tokenize"),
            ("POST", "/detokenize"),
            ("POST", "/apply-template"),
            ("POST", "/v1/systemone"),
        ];
        let status = [("GET", "/props"), ("GET", "/slots"), ("GET", "/metrics")];
        let admin = [
            ("POST", "/slots/0"),
            ("POST", "/residency/reset"),
            ("POST", "/shutdown"),
            ("GET", "/no-such-route"),
            ("GET", "/v1/responses/"),
        ];
        let list = [
            ("GET", "/v1/models"),
            ("GET", "/models"),
            ("GET", "/api/tags"),
            ("GET", "/v1/models/m"),
            ("GET", "/models/m"),
            ("POST", "/api/show"),
        ];
        for (key, reaches) in [
            ("sk-gen", [true, false, false]),
            ("sk-look", [false, true, false]),
            ("sk-both", [true, true, false]),
            ("sk-root", [true, true, true]),
            ("sk-plain", [true, true, true]),
        ] {
            for (routes, reach) in [&model[..], &status[..], &admin[..]]
                .into_iter()
                .zip(reaches)
            {
                for &(method, path) in routes {
                    let want = if reach { Ok(()) } else { Err(Denied::Scope) };
                    assert_eq!(check(key, method, path), want, "{key} {method} {path}");
                }
            }
            for (method, path) in list {
                assert_eq!(check(key, method, path), Ok(()), "{key} {method} {path}");
            }
        }
        // Every route the live view lists is reached by its own scope.
        for (method, path) in crate::api::LISTED {
            let key = if method == "POST" {
                "sk-gen"
            } else {
                "sk-look"
            };
            assert_eq!(check(key, method, path), Ok(()), "{key} {method} {path}");
        }
        assert_eq!(check("sk-none", "GET", "/props"), Err(Denied::Key));
        assert!(keys.allows("GET", "/health", None, None));
    }

    /// A field the line does not take is refused by its line and place, and
    /// the refusal never carries the key: a key with whitespace in it reads
    /// as a key and a field, and only the field's place is named.
    #[test]
    fn a_key_file_field_it_does_not_take_names_its_line_and_never_the_key() {
        for (i, (text, error)) in [
            (
                "ok\nsk-sec ret\n",
                "line 2: field 1 after the key is not scope=, inflight=",
            ),
            (
                "sk-secret colour=red\n",
                "line 1: field 1 after the key is not scope=",
            ),
            (
                "sk-secret scope=write\n",
                "line 1: scope=write: takes inference, read or admin",
            ),
            (
                "sk-secret rate=5/s inflight=0\n",
                "line 1: inflight=0: takes a whole number",
            ),
            (
                "sk-secret burst=2\n",
                "line 1: burst= needs a rate= beside it",
            ),
            (
                "sk-secret scope=read scope=admin\n",
                "line 1: field 2 after the key names scope= again",
            ),
            (
                "sk-secret\nsk-secret scope=read\n",
                "line 2: gives a key again with another scope or limits",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let (got, path) = key_file(&format!("fields-{i}"), text);
            let e = got.unwrap_err();
            assert!(
                e.starts_with(&format!("--api-key-file {path:?} {error}")),
                "{text:?}: {e}"
            );
            assert!(!e.contains("sk-"), "the refusal carries no key: {e}");
        }
        // The same key again with the same fields is kept once.
        let (keys, _) = key_file("again", "sk-a inflight=1\n  sk-a   inflight=1\n");
        let keys = keys.unwrap();
        let admit = || {
            let meter = keys.check("POST", "/completion", Some("sk-a"), None);
            meter
                .unwrap()
                .unwrap()
                .admit(std::time::Instant::now(), true)
        };
        let _held = admit().unwrap();
        assert!(admit().is_err(), "one key, one count");
    }

    #[test]
    fn the_context_flag_takes_llama_servers_spellings() {
        for f in ["--ctx", "--ctx-size", "-c"] {
            assert!(CTX.contains(&f), "{f}");
        }
        assert!(!CTX.contains(&"-C"));
    }

    #[test]
    fn the_ubatch_flag_takes_llama_servers_spellings() {
        for f in ["--ubatch-size", "-ub"] {
            assert!(UBATCH.contains(&f), "{f}");
        }
        assert!(!UBATCH.contains(&"-u"));
    }

    #[test]
    fn a_number_that_is_none_names_its_flag_and_value() {
        assert_eq!(number::<usize>("-c", "32768"), Ok(32768));
        assert_eq!(
            number::<usize>("--ctx-size", "12k"),
            Err("--ctx-size \"12k\": invalid digit found in string".to_owned())
        );
        assert_eq!(
            number::<u16>("--port", "70000"),
            Err("--port \"70000\": number too large to fit in target type".to_owned())
        );
        assert_eq!(
            number::<u64>("--cache-ram", ""),
            Err("--cache-ram \"\": cannot parse integer from empty string".to_owned())
        );
    }

    #[test]
    fn the_key_flags_are_llama_servers_two() {
        for f in ["--api-key", "--api-key-file"] {
            assert!(KEYS.contains(&f), "{f}");
        }
    }

    /// `--api-key`'s list: plain items split on commas, quoted items keeping
    /// theirs, `""` an escaped quote, a quote mid-item literal, the empty
    /// items dropped — and the flag given again adding to what it left.
    #[test]
    fn an_api_key_list_takes_llama_servers_quoting() {
        let mut keys = ApiKeys::default();
        keys.add("--api-key", "one").unwrap();
        keys.add("--api-key", "two,three").unwrap();
        keys.add("--api-key", r#""four, five""#).unwrap();
        keys.add("--api-key", r#""six ""quoted"" seven""#).unwrap();
        keys.add("--api-key", "eight,,nine").unwrap();
        keys.add("--api-key", "tw\"o").unwrap();
        let check = |auth: Option<&str>| keys.allows("POST", "/completion", auth, None);
        assert!(check(Some("one")));
        assert!(check(Some("two")));
        assert!(check(Some("three")));
        assert!(check(Some("four, five")));
        assert!(check(Some("six \"quoted\" seven")));
        assert!(check(Some("eight")));
        assert!(check(Some("nine")));
        assert!(check(Some("tw\"o")));
        assert!(!check(Some("")));
        assert!(!check(None));
        assert!(!check(Some("oneway")));
    }

    /// A key file: one key per line, empty lines and `#` comments skipped, a
    /// CRLF line's `\r` dropped; a file that cannot be read is an error
    /// naming it.
    #[test]
    fn a_key_file_takes_one_key_per_line() {
        let dir = std::env::temp_dir().join(format!("bloomery-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("keys");
        std::fs::write(&file, "# a comment\n\nfrom-file\r\n\n# another\nsecond\n").unwrap();
        let mut keys = ApiKeys::default();
        keys.add("--api-key", "from-flag").unwrap();
        keys.add("--api-key-file", &file.display().to_string())
            .unwrap();
        assert!(keys.allows("POST", "/completion", Some("from-file"), None));
        assert!(keys.allows("POST", "/completion", Some("second"), None));
        assert!(keys.allows("POST", "/completion", Some("Bearer from-flag"), None));
        let missing = dir.join("no-such-file");
        assert_eq!(
            keys.add("--api-key-file", &missing.display().to_string()),
            Err(format!(
                "--api-key-file {:?}: cannot read it: No such file or directory (os error 2)",
                missing.display().to_string()
            ))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A key file read that fails is part of the snapshot too: a later
    /// parse of the same path gets the same error and opens nothing, so a
    /// spent pipe cannot hang a second parse.
    #[test]
    fn a_failed_key_file_read_is_kept_like_its_keys() {
        let dir = std::env::temp_dir().join(format!("bloomery-bad-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("keys");
        std::fs::write(&file, b"\xff\n").unwrap();
        let path = file.display().to_string();
        let first = ApiKeys::default().add("--api-key-file", &path);
        assert!(
            first.as_ref().is_err_and(|e| e.contains("cannot read it")),
            "{first:?}"
        );
        std::fs::write(&file, "now-valid\n").unwrap();
        assert_eq!(ApiKeys::default().add("--api-key-file", &path), first);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A value that names no key refuses the flag by name: llama-server
    /// would start with the check off, serving anyone who reaches the port.
    #[test]
    fn a_key_flag_with_no_key_is_refused_by_name() {
        let mut keys = ApiKeys::default();
        assert_eq!(
            keys.add("--api-key", ""),
            Err(
                "--api-key \"\": names no key, and a server with no key checks no request"
                    .to_owned()
            )
        );
        assert_eq!(
            keys.add("--api-key", ","),
            Err(
                "--api-key \",\": names no key, and a server with no key checks no request"
                    .to_owned()
            )
        );
        let dir = std::env::temp_dir().join(format!("bloomery-keys-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("empty");
        std::fs::write(&file, "# only a comment\n\n").unwrap();
        assert_eq!(
            keys.add("--api-key-file", &file.display().to_string()),
            Err(format!(
                "--api-key-file {:?}: names no key, and a server with no key checks no request",
                file.display().to_string()
            ))
        );
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            keys.add("--port", "8080"),
            Err("--port is not one of the key flags".to_owned())
        );
        assert!(
            keys.allows("POST", "/completion", None, None),
            "still empty"
        );
    }

    /// The check itself, as llama-server's middleware runs it: an empty set
    /// passes everything; `OPTIONS` passes without a key; `/health` and
    /// `/v1/health` pass (any method, as the middleware matches the path
    /// alone); everything else needs one of the set's keys, from
    /// `Authorization` — `Bearer `-stripped — else from `X-Api-Key` (a
    /// `Bearer ` there is stripped too, as the middleware strips whichever
    /// header it read).
    #[test]
    fn the_check_passes_what_llama_servers_middleware_passes() {
        let empty = ApiKeys::default();
        for (method, path) in [
            ("POST", "/completion"),
            ("GET", "/props"),
            ("DELETE", "/slots/0"),
        ] {
            assert!(empty.allows(method, path, None, None), "{method} {path}");
        }

        let mut keys = ApiKeys::default();
        keys.add("--api-key", "sk-outer").unwrap();
        keys.add("--api-key", "sk-inner").unwrap();

        assert!(keys.allows("OPTIONS", "/completion", None, None));
        assert!(keys.allows("GET", "/health", None, None));
        assert!(keys.allows("POST", "/health", None, None));
        assert!(keys.allows("GET", "/v1/health", None, None));
        // The public path is matched exactly, not by prefix.
        assert!(!keys.allows("GET", "/health/liveness", None, None));

        assert!(keys.allows("POST", "/completion", Some("sk-outer"), None));
        assert!(keys.allows("POST", "/completion", Some("Bearer sk-outer"), None));
        // The prefix is llama-server's exact `Bearer `: a lowercase one is
        // not stripped, so the whole header value must itself be a key.
        assert!(!keys.allows("POST", "/completion", Some("bearer sk-outer"), None));
        assert!(!keys.allows("POST", "/completion", Some("Bearer sk-nope"), None));
        assert!(!keys.allows("POST", "/completion", Some("sk-nope"), None));
        assert!(!keys.allows("POST", "/completion", None, None));

        // The Anthropic header carries the key too, with or without the prefix.
        assert!(keys.allows("POST", "/v1/messages", None, Some("sk-inner")));
        assert!(keys.allows("POST", "/v1/messages", None, Some("Bearer sk-inner")));
        assert!(!keys.allows("POST", "/v1/messages", None, Some("sk-nope")));

        // An empty Authorization falls through to X-Api-Key, as the
        // middleware's retry does.
        assert!(keys.allows("POST", "/v1/messages", Some(""), Some("sk-inner")));
        assert!(!keys.allows("POST", "/v1/messages", Some(""), Some("sk-nope")));
        // Authorization, when it holds a value, wins over X-Api-Key.
        assert!(!keys.allows("POST", "/v1/messages", Some("sk-nope"), Some("sk-inner")));
    }

    /// The CSV walk itself, on llama-server's own example row.
    #[test]
    fn a_csv_row_splits_llama_servers_example() {
        assert_eq!(
            csv_row(r#"value1,"value, with, commas","value with ""escaped"" quotes",value4"#),
            [
                "value1",
                "value, with, commas",
                "value with \"escaped\" quotes",
                "value4"
            ]
        );
        assert_eq!(csv_row(""), [""]);
        assert_eq!(csv_row("a,"), ["a", ""]);
        // An opened quote that never closes keeps what followed it.
        assert_eq!(csv_row(r#""unterminated"#), ["unterminated"]);
        // An open and close around nothing is the empty field.
        assert_eq!(csv_row(r#""""#), [""]);
    }
}
