//! The seats' flags as llama-server spells them: the context flag's
//! spellings ([`CTX`]), a flag's number, whose refusal names the flag and
//! the value ([`number`]), and the API keys ([`ApiKeys`]) both key flags add
//! to and every request of both servers is checked against.

use std::fmt::Display;
use std::str::FromStr;

/// The spellings of the context flag: `--ctx`, and llama-server's
/// `--ctx-size` and `-c`.
pub const CTX: [&str; 3] = ["--ctx", "--ctx-size", "-c"];

/// The spellings of the key flags: `--api-key` and `--api-key-file`.
pub const KEYS: [&str; 2] = ["--api-key", "--api-key-file"];

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
/// against before its dispatch ([`ApiKeys::allows`]). No key set checks
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
#[derive(Debug, Default, Clone)]
pub struct ApiKeys {
    /// The keys, in the order the flags added them.
    keys: Vec<String>,
}

impl ApiKeys {
    /// `flag`'s value added to the set. `--api-key` takes llama-server's
    /// comma-separated list — a quoted field may hold commas, `""` inside
    /// quotes an escaped quote — with empty items dropped; `--api-key-file`
    /// takes one key per line, skipping empty lines and `#` comments. A file
    /// that cannot be read is an error naming it; a flag whose value names no
    /// key at all is an error naming it, where llama-server would start with
    /// the check off, serving anyone who reaches the port while the user
    /// believes it locked.
    pub fn add(&mut self, flag: &str, value: &str) -> Result<(), String> {
        let added: Vec<String> = match flag {
            "--api-key" => csv_row(value)
                .into_iter()
                .filter(|k| !k.is_empty())
                .collect(),
            "--api-key-file" => {
                let text = std::fs::read_to_string(value)
                    .map_err(|e| format!("{flag} {value:?}: cannot read it: {e}"))?;
                // `lines` drops a CRLF line's `\r`; llama-server's getline on
                // Linux keeps it, in a key no header value ever matches.
                text.lines()
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(str::to_owned)
                    .collect()
            }
            other => return Err(format!("{other} is not one of the key flags")),
        };
        if added.is_empty() {
            return Err(format!(
                "{flag} {value:?}: names no key, and a server with no key checks no request"
            ));
        }
        self.keys.extend(added);
        Ok(())
    }

    /// Whether the request passes the key check: the request's method, path
    /// and the two headers a key can ride on, against this set, as
    /// [the type](ApiKeys) states. A request the set is empty for passes.
    #[must_use]
    pub fn allows(
        &self,
        method: &str,
        path: &str,
        authorization: Option<&str>,
        x_api_key: Option<&str>,
    ) -> bool {
        if self.keys.is_empty() {
            return true;
        }
        if method == "OPTIONS" || matches!(path, "/health" | "/v1/health") {
            return true;
        }
        let sent = authorization
            .filter(|v| !v.is_empty())
            .or(x_api_key)
            .map(|v| v.strip_prefix("Bearer ").unwrap_or(v));
        sent.is_some_and(|k| self.keys.iter().any(|known| known == k))
    }
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

#[cfg(test)]
mod tests {
    use super::{ApiKeys, CTX, KEYS, csv_row, number};

    #[test]
    fn the_context_flag_takes_llama_servers_spellings() {
        for f in ["--ctx", "--ctx-size", "-c"] {
            assert!(CTX.contains(&f), "{f}");
        }
        assert!(!CTX.contains(&"-C"));
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
