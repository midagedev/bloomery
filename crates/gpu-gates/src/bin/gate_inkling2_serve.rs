//! `gate_inkling2_serve` — the Inkling 2 seat of `bloomery-serve` against the model it serves, on
//! the inkling2 file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier):
//! the family's serve gate. The seat holds the whole model on one card and one sequence, feeds a
//! prompt by the body's prompt rule (`app::Prompt`), keeps a prefix a request shares with the
//! slot only where it is the whole held sequence (a windowed visit's ring holds the window it
//! stands at alone) and takes no position back: a rollback is refused by name, and the seat runs
//! no draft, the one path that asks a body for one.
//!
//!     gate_inkling2_serve --dir <dir>
//!
//! Every server is `bloomery-serve --model inkling2 -m <file> …` beside this binary,
//! `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in `<dir>/<name>/`
//! (`server.err`, the responses).
//!
//! Without a load:
//!
//! - `plan_only_names_its_default`: `--plan` with no `--place`, `--ctx` or `--parallel` exits 0 and
//!   prints `place unset`, the plan record and the `parallel` line, in that order, and no `load`
//!   or `listening` record; the `parallel` line is the default's one slot. The one `ctx` line
//!   holds its word (`ctx_ladder::default_line_held`) and names the plan's `ctx_max`; that
//!   context stands (`--ctx <ctx_max> --plan` exits 0) and the step above it, when there is one,
//!   does not (the whole model no longer fits the card at it).
//! - `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its stderr names the
//!   body's one sequence and no `plan` record is printed (the refusal comes before any plan).
//!
//! One server as a user starts it, `--port 0` and no other flag:
//!
//! - `load_and_listen`: it listens; the `load` record comes before the `listening` record; the load
//!   names the body's arch word, one slot and a captured step graph, and its `ctx` and the
//!   `listening` record's are the `ctx` line's: the context the plan chose is the context the load
//!   made.
//! - `tokenize_is_the_oracles` (b): `/tokenize` of the text of each oracle prompt
//!   (`tools/ref/data/inkling2/{a..e}.txt`) at `add_special` false answers the ids the family's
//!   reference tokenizer made of it (`{a..e}.ids`).
//! - `chat_renders_through_the_template` (c): `/apply-template` of one user turn is the Inkling
//!   template's render of it byte for byte (system block with the default thinking effort, then
//!   the user block), the render the reference tree's jinja engine gave.
//! - `chat_is_200` (c): `/v1/chat/completions` of a turn at temperature 0 is a 200 whose `usage`
//!   counts the template's ids and the completion's tokens, and whose message text is inside the
//!   text of the ids the same turn gives through `/completion`.
//! - `tool_call_parses_into_tool_calls` (c): a chat request that lists a function and forces it
//!   by name answers one `tool_calls` entry of that function with its arguments as a JSON object
//!   inside the declared schema (a string drawn from the enum) and `finish_reason` `tool_calls`.
//!   The random fixture weights never write a call block of their own, so the call is the server's
//!   forced program over the model's logits, parsed by the block scan.
//! - `tool_choice_none_reads_no_call` (c): the same request with `tool_choice` `"none"` answers
//!   no `tool_calls` and a `finish_reason` other than `tool_calls`. On the fixture's random
//!   weights the reply holds no call block either way, so this half carries the served route's
//!   outcome only; the scan's refusal to read a call at `none` is held by the host test
//!   `hw_inkling2_tool_choice_none_reads_no_call` (`gate-serve`).
//! - `edit_resend_is_a_fresh_run`: a prompt answered with the prompt cache, then resent with its
//!   second half changed, answers the ids of the same request run from an empty cache and keeps
//!   nothing (`cache_n` 0): the shared prefix is shorter than the held sequence and the rings
//!   cannot step back, so the slot is reset. A `keep` that grants the shared prefix reaches the
//!   body's refused rollback, the request fails and the clause is red.
//! - `extension_keeps_the_held_sequence`: the prompt resent with its answer appended keeps the
//!   held positions (`cache_n` above 0): the shared prefix is the whole held sequence.
//!
//! The server stopped, the file's own body on the card, opened as the seat opens it and fed as
//! the seat feeds it (the body's prompt rule, then a step an id):
//!
//! - `completion_ids` (a): `/completion` at temperature 0, [`N`] tokens, of each oracle prompt
//!   that is short and of the longest (set `e`, 1,038 ids, past the windowed visits' 1,024), and
//!   of the chat turn the server's own template renders, each from an empty cache, answers the
//!   ids the body gives directly for the same ids, and the load record names the body's layers
//!   and visits.
//! - `extension_is_the_held_run`: the resent answer of `extension_keeps_the_held_sequence` is
//!   the body's run of the held split: the prose prompt and its answer, the answer's last id by
//!   the body's prompt rule, [`N`] more.
//! - `rollback_is_refused_by_name`: the body asked to take a position back from where it stands
//!   returns an error naming `inkling2::Body::rollback` and the rings, and stands where it did;
//!   and `BLOOMERY_DRAFT=dspark` (the draft is the path that asks for a rollback) is refused by
//!   name before any plan, naming the rollback and the ring, while `BLOOMERY_DRAFT=off` plans.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::time::Duration;

    use app::Prompt;
    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::ctx_ladder::{ctx_line, default_line_held, field, step_above};
    use bloomery_gpu_gates::generate;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        Answer, SeatGate, Served, agree, chat_is_those_ids, check, curl, greedy, json_of, rendered,
        server_log, tokenized,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
    use gguf::Split;
    use serde_json::{Value, json};

    const USAGE: &str = "usage: gate_inkling2_serve --dir <dir>";

    /// The tokens a completion asks for.
    const N: usize = 16;

    /// Polls of a second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 900;

    /// The cache rows of the session the ids are read from: past every prompt here and its answer.
    const SESSION_CTX: usize = 2048;

    /// The default is one slot.
    const DEFAULT_SLOTS: u64 = 1;

    /// The words of the refusal of a second slot.
    const ONE_SEQUENCE: &str = "holds one sequence";

    /// This gate's seat: the one server's model word and the kinds it prints.
    const SEAT: SeatGate = SeatGate {
        model: "inkling2",
        kinds: record::BLOOMERY_SERVE_INKLING2,
    };

    /// The model's body name in a load record.
    const ARCH: &str = "inkling2";

    /// The prompt the prefix clauses run.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and \
                         twelve, the same as every night, until the hundred and thirteenth";

    /// The chat turn the file answers.
    const TURN: &str = "Name three primary colours.";

    /// The question the tool clauses ask.
    const TOOL_ASK: &str = "What is the weather in Seoul right now?";

    /// The one user turn the template clause renders, and its render by the reference tree's jinja
    /// engine (`crates/serve/tests/fixtures/inkling2-renders.json`, the Mini's template, case "a
    /// user turn").
    const HELLO: &str = "Hello";
    const HELLO_RENDER: &str = "<|message_system|><|content_text|>Thinking effort level: \
                                0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>";

    /// The oracle prompts: the text of each and the reference tokenizer's ids of it, one decimal id
    /// a line (`tools/ref/data/inkling2`).
    const ORACLE: [(&str, &str, &str); 5] = [
        (
            "a",
            include_str!("../../../../tools/ref/data/inkling2/a.txt"),
            include_str!("../../../../tools/ref/data/inkling2/a.ids"),
        ),
        (
            "b",
            include_str!("../../../../tools/ref/data/inkling2/b.txt"),
            include_str!("../../../../tools/ref/data/inkling2/b.ids"),
        ),
        (
            "c",
            include_str!("../../../../tools/ref/data/inkling2/c.txt"),
            include_str!("../../../../tools/ref/data/inkling2/c.ids"),
        ),
        (
            "d",
            include_str!("../../../../tools/ref/data/inkling2/d.txt"),
            include_str!("../../../../tools/ref/data/inkling2/d.ids"),
        ),
        (
            "e",
            include_str!("../../../../tools/ref/data/inkling2/e.txt"),
            include_str!("../../../../tools/ref/data/inkling2/e.ids"),
        ),
    ];

    /// The ids of an oracle file, one decimal id a line.
    fn oracle_ids(text: &str) -> Result<Vec<u32>, GateError> {
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.parse::<u32>()
                    .map_err(|e| format!("oracle id {l:?}: {e}").into())
            })
            .collect()
    }

    /// The oracle prompt `label`'s ids.
    fn oracle(label: &str) -> Result<Vec<u32>, GateError> {
        let (_, _, ids) = ORACLE
            .iter()
            .find(|(l, _, _)| *l == label)
            .ok_or_else(|| format!("no oracle prompt {label:?}"))?;
        oracle_ids(ids)
    }

    /// What the server answered that the session clauses compare with.
    struct Answers {
        /// The prompts and what the server answered of each, from an empty cache.
        runs: Vec<Answer>,
        /// The extension's resent answer.
        resent: Option<Answer>,
        /// The prose prompt (the extension's first request) and its answer.
        prose: Answer,
        /// The server's `load` record: layers, visits.
        load: (u64, u64),
    }

    /// The tool the clauses list: a function with one string drawn from an enum, so the forced
    /// call's program ends in a bounded number of ids whatever the model's logits are.
    fn tools() -> Value {
        json!([{ "type": "function", "function": {
            "name": "get_weather",
            "description": "The current weather of a city.",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string", "enum": ["Seoul", "Busan"] } },
                "required": ["city"],
            },
        } }])
    }

    /// A chat request of [`TOOL_ASK`] that lists [`tools`] and asks `choice`, at temperature 0.
    fn tool_request(choice: Value) -> Value {
        json!({
            "messages": [{ "role": "user", "content": TOOL_ASK }],
            "tools": tools(),
            "tool_choice": choice,
            "temperature": 0,
            "max_tokens": 96,
            "stream": false,
        })
    }

    /// The file's trained context.
    fn trained_ctx(file: &Path) -> Result<u64, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let n = generate::trained_ctx(&split)
            .ok_or_else(|| format!("{}: no context_length", file.display()))?;
        Ok(u64::try_from(n)?)
    }

    /// `plan_only_names_its_default` (the module header).
    fn plan_only_names_its_default(
        dir: &Path,
        file: &Path,
    ) -> Result<(Option<u64>, bool), GateError> {
        let mut ok = true;
        let limit = trained_ctx(file)?;
        let (status, log, text) = SEAT.plan(dir, "plan", file, &[])?;
        let place = log.first(&record::PLACE_UNSET)?;
        let planned = log.first(&record::PLAN)?;
        let parallel_at = text
            .lines()
            .position(|l| l.starts_with("parallel "))
            .map(|i| i + 1);
        let parallel_line = text.lines().find(|l| l.starts_with("parallel "));
        let in_order = matches!(
            (&place, &planned, parallel_at),
            (Some(a), Some(b), Some(c)) if a.at() < b.at() && b.at() < c
        );
        let no_load = log.first(&record::LOAD_INKLING2)?.is_none()
            && log.first(&record::LISTENING_INKLING2)?.is_none();
        let exits = status.success();
        println!(
            "plan only: exit 0 {exits}; place unset, plan, parallel in order {in_order}; no load \
             or listening record {no_load}"
        );
        let ctx_max = match &planned {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let mut default_ok = false;
        if let Some(c) = ctx_max {
            let want = format!(
                "parallel rule=slots slots={DEFAULT_SLOTS} slot_ctx={c} total={c} from=default \
                 pass=one"
            );
            let line_ok = parallel_line == Some(want.as_str());
            println!("plan only: the parallel line {parallel_line:?} is {want:?} {line_ok}");
            let line = ctx_line(&text);
            let held = line.is_some_and(|l| {
                default_line_held(l, DEFAULT_SLOTS, (limit, "trained")) == Some(true)
                    && field(l, "ctx") == Some(c)
            });
            println!("plan only: the ctx line {line:?} holds its word at {c} {held}");
            let c_arg = c.to_string();
            let (st, at_log, _) = SEAT.plan(dir, "plan-at-default", file, &["--ctx", &c_arg])?;
            let stands = st.success()
                && match at_log.first(&record::PLAN)? {
                    Some(p) => p.u64("ctx_max")? == c,
                    None => false,
                };
            println!("plan only: --ctx {c} stands {stands}");
            let above_short = match step_above(limit, c) {
                None => {
                    println!("plan only: the default {c} is the target, none above it");
                    true
                }
                Some(above) => {
                    let above_arg = above.to_string();
                    let (st, _, _) =
                        SEAT.plan(dir, "plan-past-default", file, &["--ctx", &above_arg])?;
                    println!(
                        "plan only: --ctx {above}, the step above, is refused {}",
                        !st.success()
                    );
                    !st.success()
                }
            };
            default_ok = line_ok && held && stands && above_short;
        }
        check(
            &mut ok,
            "plan_only_names_its_default",
            exits && in_order && no_load && default_ok,
        );
        Ok((ctx_line(&text).and_then(|l| field(l, "ctx")), ok))
    }

    /// `parallel_above_one_is_refused` (the module header).
    fn parallel_above_one_is_refused(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let (status, log, text) = SEAT.plan(dir, "refuse-parallel", file, &["--parallel", "2"])?;
        let named = text.contains(ONE_SEQUENCE);
        let no_plan = log.first(&record::PLAN)?.is_none();
        println!(
            "refuse-parallel: exit non-zero {}; names {ONE_SEQUENCE:?} {named}; no plan record \
             {no_plan}",
            !status.success()
        );
        if !named {
            println!("refuse-parallel stderr:\n{text}");
        }
        check(
            &mut ok,
            "parallel_above_one_is_refused",
            !status.success() && named && no_plan,
        );
        Ok(ok)
    }

    /// `n` greedy ids after `first`, each answer fed back one step at a time.
    fn continue_from(m: &mut GpuModel<Body>, first: u32, n: usize) -> Result<Vec<u32>, GateError> {
        let mut answer = vec![first];
        while answer.len() < n {
            let last = *answer.last().ok_or("an empty answer")?;
            answer.push(m.step(&[last])?);
        }
        Ok(answer)
    }

    /// The ids `m` gives for the prompt `ids` from an empty cache: the body's prompt rule
    /// ([`Prompt::prompt`]) as the seat feeds it, then [`N`] greedy ids, each answer fed back.
    fn seat_answer(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<Vec<u32>, GateError> {
        m.reset()?;
        let first = <Body as Prompt>::prompt(m, ids)?;
        continue_from(m, first, N)
    }

    /// The body as the seat opens it: the whole model on the card at [`SESSION_CTX`] positions, in
    /// graph mode with its step captured.
    fn open_body(file: &Path) -> Result<GpuModel<Body>, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let mut m = GpuModel::<Body>::open(Gpu::new()?, split, SESSION_CTX)?;
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        Ok(m)
    }

    /// The prompt `ids` with the second half's ids moved by one: the same length and the same
    /// first half.
    fn edited(prose: &[u32]) -> Vec<u32> {
        let half = prose.len() / 2;
        let mut ids = prose[..half].to_vec();
        ids.extend(prose[half..].iter().map(|&id| id + 1));
        ids
    }

    /// `tokenize_is_the_oracles` (the module header).
    fn tokenize_is_the_oracles(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
        let mut all = true;
        for (label, text, ids) in ORACLE {
            let want = oracle_ids(ids)?;
            let got = tokenized(url, text)?;
            let same = got == want;
            println!(
                "tokenize {label}: {} ids, the reference's {} {same}",
                got.len(),
                want.len()
            );
            all &= same;
        }
        Ok(all)
    }

    /// `chat_renders_through_the_template` (the module header).
    fn chat_renders_through_the_template(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": [{ "role": "user", "content": HELLO }] })),
            false,
        )?;
        let prompt = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let same = prompt == HELLO_RENDER;
        println!("apply-template: {prompt:?} is the reference's render {same}");
        Ok(same)
    }

    /// The `tool_calls` of a chat answer's first choice and its finish reason.
    fn calls_of(v: &Value) -> (Vec<Value>, String) {
        let choice = &v["choices"][0];
        (
            choice["message"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default(),
            choice["finish_reason"].as_str().unwrap_or("").to_owned(),
        )
    }

    /// `tool_call_parses_into_tool_calls` (the module header).
    fn tool_call_parses_into_tool_calls(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let body =
            tool_request(json!({ "type": "function", "function": { "name": "get_weather" } }));
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        std::fs::write(dir.join("tool-forced.json"), &text)?;
        let v = json_of("/v1/chat/completions", st, &text)?;
        let (calls, finish) = calls_of(&v);
        let one = calls.len() == 1;
        let function = calls.first().map(|c| &c["function"]);
        let named = function.is_some_and(|f| f["name"] == "get_weather");
        let args: Option<Value> = function
            .and_then(|f| f["arguments"].as_str())
            .and_then(|a| serde_json::from_str(a).ok());
        let in_schema = args.as_ref().is_some_and(|a| {
            a.as_object().is_some_and(|o| o.len() == 1)
                && matches!(a["city"].as_str(), Some("Seoul" | "Busan"))
        });
        println!(
            "forced call: {} call(s); function {:?}; arguments {args:?}; finish {finish:?}; \
             named {named}, in schema {in_schema}",
            calls.len(),
            function.map(|f| &f["name"]),
        );
        Ok(one && named && in_schema && finish == "tool_calls")
    }

    /// `tool_choice_none_reads_no_call` (the module header).
    fn tool_choice_none_reads_no_call(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let (st, text) = curl(
            &url("/v1/chat/completions"),
            Some(&tool_request(json!("none"))),
            false,
        )?;
        std::fs::write(dir.join("tool-none.json"), &text)?;
        let v = json_of("/v1/chat/completions", st, &text)?;
        let (calls, finish) = calls_of(&v);
        println!(
            "tool_choice none: {} call(s); finish {finish:?}",
            calls.len()
        );
        Ok(calls.is_empty() && finish != "tool_calls")
    }

    /// `edit_resend_is_a_fresh_run` (the module header): the slot holds the prose prompt and its
    /// answer, and the same prompt with its second half changed is sent.
    fn edit_resend(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        prose: &[u32],
    ) -> Result<bool, GateError> {
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "prose-held", true)?;
        let changed = edited(prose);
        let (fresh, _) = greedy(url, changed.clone(), N, dir, "edit-fresh", false)?;
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "prose-held-again", true)?;
        let (resent, _) = greedy(url, changed, N, dir, "edit-resent", true)?;
        println!(
            "edit resend: ids equal the fresh run's {}; cache_n {} is 0",
            resent.tokens == fresh.tokens,
            resent.cache_n
        );
        Ok(resent.tokens == fresh.tokens && resent.cache_n == 0)
    }

    /// `extension_keeps_the_held_sequence` (the module header): the slot holds the prose prompt
    /// and its answer, and the prompt with that answer appended is sent. Its answer, kept for
    /// `extension_is_the_held_run`.
    fn extension(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        prose: &[u32],
        held: &Answer,
    ) -> Result<Answer, GateError> {
        let (_, _) = greedy(url, prose.to_vec(), N, dir, "extension-held", true)?;
        let mut longer = prose.to_vec();
        longer.extend(&held.tokens);
        let (resent, _) = greedy(url, longer, N, dir, "extension-resent", true)?;
        println!("extension: cache_n {} is above 0", resent.cache_n);
        Ok(resent)
    }

    /// The clauses on one server: `load_and_listen` through the prefix clauses; `None` when the
    /// server never listened.
    fn served(
        dir: &Path,
        file: &Path,
        default_ctx: Option<u64>,
    ) -> Result<(Option<Answers>, bool), GateError> {
        let mut ok = true;
        let d = dir.join("serve");
        std::fs::create_dir_all(&d)?;
        let (cmd, mut all) = SEAT.command(file)?;
        all.extend(["--port".to_owned(), "0".to_owned()]);
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the server never listened: {e}");
                check(&mut ok, "load_and_listen", false);
                return Ok((None, ok));
            }
        };
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let records = server_log(&err_log, record::BLOOMERY_SERVE_INKLING2)?;
        let load = records.one(&record::LOAD_INKLING2)?;
        let listen = records.one(&record::LISTENING_INKLING2)?;
        let named = ctx_line(&text).and_then(|l| field(l, "ctx"));
        let same_ctx = default_ctx.is_none_or(|c| named == Some(c));
        let loaded = load.word("arch")? == ARCH
            && load.u64("slots")? == 1
            && load.u64("graph_nodes")? > 0
            && listen.u64("slots")? == 1;
        let ctx_is_plans =
            named.is_some() && named == Some(load.u64("ctx")?) && named == Some(listen.u64("ctx")?);
        let in_order = load.at() < listen.at();
        println!(
            "serve at {addr}: ctx line {:?} (the plan run's {same_ctx}); load {:?}; listening \
             {:?}; load names arch={ARCH}, one slot and a captured graph {loaded}; ctx is the \
             line's {ctx_is_plans}; load before listening {in_order}",
            ctx_line(&text),
            load.line(),
            listen.line(),
        );
        check(
            &mut ok,
            "load_and_listen",
            same_ctx && loaded && ctx_is_plans && in_order,
        );
        let counts = (load.u64("layers")?, load.u64("visits")?);
        let url = |p: &str| format!("http://{addr}{p}");
        // A clause whose requests fail is that clause's FAIL, printed; the gate goes on.
        let mut run = |name: &str, r: Result<bool, GateError>| {
            let pass = r.unwrap_or_else(|e| {
                println!("{name}: {e}");
                false
            });
            check(&mut ok, name, pass);
        };
        run("tokenize_is_the_oracles", tokenize_is_the_oracles(&url));
        run(
            "chat_renders_through_the_template",
            chat_renders_through_the_template(&url),
        );
        let prose = tokenized(&url, PROSE)?;
        let messages = json!([{ "role": "user", "content": TURN }]);
        let turn = rendered(&url, messages.clone())?;
        // The prompts the clauses that compare with the body read, each from an empty cache.
        let mut runs = Vec::new();
        for (name, ids) in [
            ("a", oracle("a")?),
            ("d", oracle("d")?),
            ("e", oracle("e")?),
            ("turn", turn.clone()),
        ] {
            runs.push(greedy(&url, ids, N, &d, name, false)?.0);
        }
        let (prose_answer, _) = greedy(&url, prose.clone(), N, &d, "prose", false)?;
        let (turn_answer, turn_body) = greedy(&url, turn, N, &d, "turn-chat", false)?;
        run(
            "chat_is_200",
            chat_is_those_ids(&url, &d, messages, N, (&turn_answer, &turn_body)),
        );
        run(
            "tool_call_parses_into_tool_calls",
            tool_call_parses_into_tool_calls(&url, &d),
        );
        run(
            "tool_choice_none_reads_no_call",
            tool_choice_none_reads_no_call(&url, &d),
        );
        run("edit_resend_is_a_fresh_run", edit_resend(&url, &d, &prose));
        let resent = match extension(&url, &d, &prose, &prose_answer) {
            Ok(r) => Some(r),
            Err(e) => {
                println!("extension_keeps_the_held_sequence: {e}");
                None
            }
        };
        check(
            &mut ok,
            "extension_keeps_the_held_sequence",
            resent.as_ref().is_some_and(|r| r.cache_n > 0),
        );
        println!("server stopped: {}", s.stop()?);
        runs.push(turn_answer);
        Ok((
            Some(Answers {
                runs,
                resent,
                prose: prose_answer,
                load: counts,
            }),
            ok,
        ))
    }

    /// `completion_ids`, `extension_is_the_held_run` and the body half of
    /// `rollback_is_refused_by_name` (the module header): the server stopped and the body on the
    /// card.
    fn body_clauses(m: &mut GpuModel<Body>, a: &Answers) -> Result<bool, GateError> {
        let mut ok = true;
        let mut same = true;
        for (name, run) in ["a", "d", "e", "turn"].iter().zip(&a.runs) {
            let ids = seat_answer(m, &run.prompt)?;
            let agrees = agree(&run.tokens, &run.stop, &ids);
            println!(
                "completion ids {name}: the server's {} tokens are the body's {agrees}",
                run.tokens.len()
            );
            same &= agrees;
        }
        let counts = (
            m.layers().len() as u64,
            m.body("inkling2")?.visit_table().len() as u64,
        );
        let named = counts == a.load;
        println!(
            "completion ids: the load record's layers and visits {:?} are the body's {counts:?} \
             {named}",
            a.load
        );
        check(&mut ok, "completion_ids", same && named);
        // The extension's held split: the prose prompt and its answer, the answer's last id fed
        // as the seat feeds a resent prompt's new ids (the body's prompt rule), then N more.
        let held = seat_answer(m, &a.prose.prompt)?;
        let last = *held.last().ok_or("an empty held answer")?;
        let first = <Body as Prompt>::prompt(m, &[last])?;
        let extended = continue_from(m, first, N)?;
        let held_run = a
            .resent
            .as_ref()
            .is_some_and(|r| agree(&r.tokens, &r.stop, &extended));
        println!(
            "extension: the resent answer is the body's run of the held split (the prose prompt \
             and its answer, the answer's last id, {N} more) {held_run}"
        );
        check(&mut ok, "extension_is_the_held_run", held_run);
        Ok(ok)
    }

    /// `rollback_is_refused_by_name` (the module header): the body half on a model that has run a
    /// prompt, and the draft half on two `--plan` runs.
    fn rollback_is_refused_by_name(
        m: &mut GpuModel<Body>,
        dir: &Path,
        file: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        seat_answer(m, &oracle("a")?)?;
        let at = m.pos();
        let refused = m.rollback(at - 1);
        let text = refused.as_ref().err().map(ToString::to_string);
        let by_name = text.as_deref().is_some_and(|t| {
            t.contains("inkling2::Body::rollback") && t.contains("take no position back")
        });
        let stands = m.pos() == at;
        println!(
            "rollback to {} from {at}: {text:?}; names the body and the rings {by_name}; the \
             model stands at {at} {stands}",
            at - 1
        );
        let (status, log, text) = SEAT.plan_under(
            (dir, "draft-dspark"),
            file,
            &[],
            &[(bloomery_levers::DRAFT, Some("dspark"))],
        )?;
        let draft_named = text.contains("BLOOMERY_DRAFT=dspark")
            && text.contains("rollback")
            && text.contains("ring");
        let no_plan = log.first(&record::PLAN)?.is_none();
        println!(
            "BLOOMERY_DRAFT=dspark: exit non-zero {}; names the variable, the rollback and the \
             ring {draft_named}; no plan record {no_plan}",
            !status.success()
        );
        let (off, _, _) = SEAT.plan_under(
            (dir, "draft-off"),
            file,
            &[],
            &[(bloomery_levers::DRAFT, Some("off"))],
        )?;
        println!("BLOOMERY_DRAFT=off: plans (exit 0) {}", off.success());
        check(
            &mut ok,
            "rollback_is_refused_by_name",
            by_name && stands && !status.success() && draft_named && no_plan && off.success(),
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let dir = SEAT.dir(USAGE)?;
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let mut ok = true;
        let (default_ctx, pass) = plan_only_names_its_default(&dir, &file)?;
        ok &= pass;
        ok &= parallel_above_one_is_refused(&dir, &file)?;
        let (answers, pass) = served(&dir, &file, default_ctx)?;
        ok &= pass;
        // The server stopped, the body on the card: the one in-process load both clauses read.
        let mut m = open_body(&file)?;
        if let Some(a) = answers {
            ok &= body_clauses(&mut m, &a)?;
        }
        ok &= rollback_is_refused_by_name(&mut m, &dir, &file)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_inkling2_serve the seat plans its default and says so, refuses a second \
             slot, loads at the context it named, tokenizes the oracle prompts to the reference's \
             ids, renders the chat template, parses a forced call and reads none at none, answers \
             the body's ids, keeps a prefix only where the sequence stands and takes no position back"
        );
        Ok(())
    }
}
