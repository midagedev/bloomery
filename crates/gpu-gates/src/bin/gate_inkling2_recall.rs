//! `gate_inkling2_recall` — the recall draft on the Inkling 2 seat of `bloomery-serve`, on the
//! Mini's real file (real tier only). The fixture's random weights repeat no id, so a recall
//! server on it proposes nothing and `gate_inkling2_serve`'s recall clause holds only that a draft
//! that never proposes changes no token. Here the file's own weights continue a repeating prompt,
//! so the seat's recall window (the body's whole pass of [`PASS_ROWS`] rows) verifies proposals,
//! takes the rejected ones back and rolls the conv ring to the last kept row.
//!
//!     gate_inkling2_recall --dir <dir>
//!
//! Two servers, `bloomery-serve --model inkling2 -m <file>` beside this binary, on the profile's
//! file (`BLOOMERY_MODEL=inkling2`): one under `BLOOMERY_DRAFT=off`, one under `recall`. Each
//! answers the same three prompts at temperature 0, [`N`] tokens, each from an empty cache: the
//! repeating one, the short question and the repeated code ([`prompts`]). Logs per server in
//! `<dir>/<name>/`. The width chooser stays at its default, so the recall server prints its
//! `mtp width` records; a record is printed at the slot's next prompt call, so the last prompt's
//! windows (the server is killed at its stop) are not read.
//!
//! - `recall_proposes` (file-bound): the recall server's answer to the repeating prompt carries
//!   `draft_n` above 0 and the `mtp width` records count a window that kept all [`PASS_ROWS`]
//!   rows: the draft's window is the body's whole pass and one verified in full.
//! - `recall_rejection_is_taken_back` (file-bound): the answer to the short question carries a
//!   proposal the verify rejected (`draft_n_accepted` below `draft_n`) and the records count a
//!   window that kept fewer than all rows: the rows past the last kept one were taken back, the
//!   conv ring rolled to it.
//! - `recall_is_the_plain_ids` (file-bound): every prompt's ids under recall are the off server's,
//!   stopped as it stopped. A draft changes which rows a pass runs, never a token. [`N`] less the
//!   first token is not a whole number of the pass's rows, so the repeating prompt's request ends
//!   on what the budget leaves after its last whole pass: that end is part of the clause.
//!
//! Both servers run with `BLOOMERY_DSPARK_MODEL` and `BLOOMERY_MTP_WIDTH` removed from their
//! environment: the first would give the off server a draft by the folder, the second changes the
//! chooser whose records the clauses read.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_recall", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        Answer, SeatGate, Served, agree, check, draft_counts, greedy, server_log, tokenized,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
    use bloomery_record::Log;
    use runtime::stores::PASS_ROWS;
    use serde_json::Value;

    const USAGE: &str = "usage: gate_inkling2_recall --dir <dir>";

    /// The tokens a completion asks for.
    const N: usize = 256;

    /// Polls of a second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 900;

    /// The lever naming the DSpark draft's folder.
    const DSPARK_MODEL: &str = "BLOOMERY_DSPARK_MODEL";

    /// This gate's seat: the one server's model word and the kinds it prints.
    const SEAT: SeatGate = SeatGate {
        model: "inkling2",
        kinds: record::BLOOMERY_SERVE_INKLING2,
    };

    /// The function the code prompt repeats.
    const CODE: &str = "def total(items):\n    result = 0\n    for item in items:\n        result \
                        += item.price * item.count\n    return result\n\n";

    /// The prompts' labels and text, in the order they are sent: the clauses read the records
    /// of the first two.
    fn prompts() -> [(&'static str, String); 3] {
        [
            (
                "repeats",
                "one two three four five one two three four five one two three four five one two"
                    .to_owned(),
            ),
            ("capital", "The capital of France is".to_owned()),
            (
                "code",
                format!("{CODE}{CODE}def total(items):\n    result = 0\n    for item"),
            ),
        ]
    }

    /// What one server answered: each prompt's answer and response body, and the windows its
    /// `mtp width` records count by the rows they kept.
    struct Run {
        answers: Vec<(Answer, Value)>,
        /// The `kept` counts of each record: windows that kept 1 row, 2 rows, and so on.
        kept: Vec<Vec<u64>>,
    }

    impl Run {
        /// Windows that kept every row of the pass.
        fn full(&self) -> u64 {
            self.kept
                .iter()
                .filter_map(|k| k.get(PASS_ROWS - 1))
                .sum::<u64>()
        }

        /// Windows that kept fewer rows than the pass held: the rest were taken back.
        fn cut(&self) -> u64 {
            self.kept
                .iter()
                .flat_map(|k| k.iter().take(PASS_ROWS - 1))
                .sum::<u64>()
        }

        /// Every record counts one kept length for each row of the pass.
        fn window_is_the_pass(&self) -> bool {
            !self.kept.is_empty() && self.kept.iter().all(|k| k.len() == PASS_ROWS)
        }
    }

    /// The `kept` counts of every `mtp width` record in `log`.
    fn kept_of(log: &Log) -> Result<Vec<Vec<u64>>, GateError> {
        log.all(&record::MTP_WIDTH)?
            .iter()
            .map(|r| {
                r.csv("kept")?
                    .iter()
                    .map(|k| {
                        k.parse::<u64>()
                            .map_err(|e| format!("mtp width kept {k:?}: {e}").into())
                    })
                    .collect()
            })
            .collect()
    }

    /// A prompt a server is asked: its text, which the server tokenizes, or ids.
    enum Ask<'a> {
        Text(&'a str),
        Ids(Vec<u32>),
    }

    /// A server on `file` under `env`, answering `prompts` each from an empty cache; it is
    /// stopped before its log is read.
    fn serve(
        dir: &Path,
        file: &Path,
        env: &[(&str, Option<&str>)],
        name: &str,
        prompts: &[(&str, Ask)],
    ) -> Result<Run, GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let (mut cmd, mut all) = SEAT.command(file)?;
        for &(lever, value) in env {
            match value {
                Some(v) => cmd.env(lever, v),
                None => cmd.env_remove(lever),
            };
        }
        all.extend(["--port".to_owned(), "0".to_owned()]);
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut answers = Vec::with_capacity(prompts.len());
        for (label, ids) in prompts {
            let ids = match ids {
                Ask::Ids(ids) => ids.clone(),
                Ask::Text(text) => tokenized(&url, text)?,
            };
            answers.push(greedy(&url, ids, N, &d, label, false)?);
        }
        println!("{name}: server stopped: {}", s.stop()?);
        let kept = kept_of(&server_log(&err_log, SEAT.kinds)?)?;
        Ok(Run { answers, kept })
    }

    /// The draft word `word` and the two levers a caller's environment must not hand the
    /// servers: the DSpark folder would give the off server a draft, the width mode would
    /// change the chooser whose records the clauses read.
    fn env_of(word: &str) -> [(&str, Option<&str>); 3] {
        [
            (bloomery_levers::DRAFT, Some(word)),
            (DSPARK_MODEL, None),
            (bloomery_levers::MTP_WIDTH, None),
        ]
    }

    pub fn run() -> Result<(), GateError> {
        let dir = SEAT.dir(USAGE)?;
        let proposes = tier::run_clause("recall_proposes", Tag::FileBound)?;
        let rejected = tier::run_clause("recall_rejection_is_taken_back", Tag::FileBound)?;
        let same = tier::run_clause("recall_is_the_plain_ids", Tag::FileBound)?;
        if !(proposes || rejected || same) {
            println!("PASSED: gate_inkling2_recall left every clause to the real tier");
            return Ok(());
        }
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let texts = prompts();
        let by_text: Vec<(&str, Ask)> = texts
            .iter()
            .map(|(label, text)| (*label, Ask::Text(text.as_str())))
            .collect();
        let plain = serve(&dir, &file, &env_of("off"), "serve-off", &by_text)?;
        let by_ids: Vec<(&str, Ask)> = texts
            .iter()
            .zip(&plain.answers)
            .map(|((label, _), (a, _))| (*label, Ask::Ids(a.prompt.clone())))
            .collect();
        let recall = serve(&dir, &file, &env_of("recall"), "serve-recall", &by_ids)?;
        let mut ok = true;
        let at = |label: &str| texts.iter().position(|(l, _)| *l == label);
        let counts = |label: &str| -> Result<(u64, u64), GateError> {
            let i = at(label).ok_or_else(|| format!("no prompt {label:?}"))?;
            draft_counts(&recall.answers[i].1)
        };
        if proposes {
            let (proposed, accepted) = counts("repeats")?;
            println!(
                "recall repeats: {proposed} ids proposed, {accepted} accepted; {} windows kept \
                 all {PASS_ROWS} rows; each record counts one length a row of the pass {}",
                recall.full(),
                recall.window_is_the_pass()
            );
            check(
                &mut ok,
                "recall_proposes",
                proposed > 0 && recall.full() > 0 && recall.window_is_the_pass(),
            );
        }
        if rejected {
            let (proposed, accepted) = counts("capital")?;
            println!(
                "recall capital: {proposed} ids proposed, {accepted} accepted; {} windows kept \
                 fewer than {PASS_ROWS} rows",
                recall.cut()
            );
            check(
                &mut ok,
                "recall_rejection_is_taken_back",
                accepted < proposed && recall.cut() > 0,
            );
        }
        if same {
            let mut all = true;
            for ((label, _), ((off, _), (on, _))) in
                texts.iter().zip(plain.answers.iter().zip(&recall.answers))
            {
                let agreed = agree(&off.tokens, &off.stop, &on.tokens);
                println!(
                    "recall {label}: {} tokens are the off server's {agreed}",
                    on.tokens.len()
                );
                all &= agreed;
            }
            check(&mut ok, "recall_is_the_plain_ids", all);
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_inkling2_recall the recall draft on the Mini's file proposes on a \
             repeating prompt, verifies a window of the body's whole pass, takes a rejected \
             proposal back and answers the off server's ids"
        );
        Ok(())
    }
}
