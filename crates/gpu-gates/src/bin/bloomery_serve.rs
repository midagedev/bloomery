//! `bloomery-serve` — one server binary, one seat a model: `--model
//! ds41|qwen38|glm|qwen3|decide` picks the seat (`serve_seats`, the shared
//! seat modules), every other flag the seat's own, parsed by the seat's own
//! parser, so each seat's flag set is exactly its per-model binary's. The
//! word is optional with a model file: with none, the file's architecture
//! picks the seat, and a bare start or `--help` prints [`drive::MODELS`],
//! one line a seat with a working `--hf` example.
//!
//!     bloomery-serve [--model ds41|qwen38|glm|qwen3|decide]
//!                    [-m PATH | --model-file PATH | --hf <repo>[:<quant>]]
//!                    [--head <weights> [--head-config <file>]]
//!                    <the seat's own flags>
//!     bloomery-serve --version
//!
//! Every start prints one line on stderr before the load begins
//! ([`drive::FIRST_START`]): a first start on a new card compiles the GPU
//! code for it, which takes tens of seconds.
//!
//! The model file is `-m` (`--model-file`), the first shard of a split
//! set; `--hf` fetches a Hugging Face repo's GGUF set by its quant tag into
//! the cache (`$BLOOMERY_CACHE`, else `~/.cache/bloomery/hf`), its `hf`
//! records on stderr, and opens its first shard (a listing the network
//! refuses leaves the cache's one verified set of that quant to stand in,
//! after an `hf offline` record); with neither,
//! `$BLOOMERY_REF_MODEL`, the gates' variable. A model named twice — a flag
//! given twice, `-m` beside `--hf`, a flag beside a variable that names
//! another file — is refused by name (`bloomery_gpu_gates::model_file`).
//! A decision model is named by its head, not by its architecture
//! (`serve::decide::pick`): `--head`, or a `--hf` repo whose model card
//! names a decision row's head repo as the model it quantizes, picks the
//! decide seat (`serve_seats::decide`); `--head` beside a generative seat's
//! `--model` is refused by name, and so is `--model decide` with no head.
//! Otherwise, with no `--model`, the architecture of the file's first shard
//! picks the seat: deepseek41 and deepseek4 the ds41 seat, glm5next the glm
//! seat, qwen4exp the qwen38 seat, qwen3moe and qwen35moe the qwen3 seat;
//! a file of a decision row's backbone (qwen35) with no head is refused by
//! name, saying what would seat it (`--head`, or a row's `--hf` repo); a
//! file whose architecture no seat serves is refused by name listing the
//! seat words; a file whose architecture does not match an explicit
//! `--model` is refused by name before any load.
//! `--version` prints this crate's version and the build's commit and exits.
//! Each seat's flags, records, `/props` fields and exit codes are its
//! module's doc; `bloomery-serve-ds41` and `bloomery-serve-qwen38` are this
//! binary's ds41 and qwen38 seats alone.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "bloomery-serve: built without the `glm5next` feature, which links every seat's device bundle."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    drive::run()
}

// The seats this binary serves, and the shared files the ds41 seat names at
// the crate root (`crate::dspark` and kin).
#[cfg(feature = "glm5next")]
#[path = "shared/serve_seats/mod.rs"]
mod serve_seats;

// The GLM seat's placement word and tier batch bytes (`crate::glm_place`).
#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_place.rs"]
mod glm_place;

// The decide seat's qwen35 backbone open (`crate::qwen35_open`) and Clef's
// row's own part (`crate::clef_seat`).
#[cfg(feature = "clef")]
#[path = "shared/qwen35_open.rs"]
#[allow(
    dead_code,
    reason = "the seat opens the backbone only; the ids reader serves clef_hidden"
)]
mod qwen35_open;

#[cfg(feature = "clef")]
#[path = "shared/clef_seat.rs"]
mod clef_seat;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_draft.rs"]
mod draft;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_place.rs"]
mod place;

#[cfg(feature = "glm5next")]
mod drive {
    use std::process::ExitCode;

    use bloomery_gpu_gates::{GateError, exit_with, model_file, ref_model_path};
    use gguf::Split;
    use model::arch::DEEPSEEK4;
    use serve::ServeError;
    use serve::decide::Ask;

    use crate::serve_seats::{decide, ds41, glm, qwen3, qwen38};

    const NAME: &str = "bloomery-serve";

    const USAGE: &str = "usage: bloomery-serve [--model ds41|qwen38|glm|qwen3|decide] [-m PATH | \
                         --model-file PATH | --hf <repo>[:<quant>]] [--head <weights> \
                         [--head-config <file>]] <the chosen seat's own flags, as \
                         bloomery-serve-ds41, bloomery-serve-qwen38, the glm, the qwen3 or the \
                         decide seat takes them> | --version";

    /// One line a seat, printed beside [`USAGE`] on `--help` and a bare
    /// start: the seat word, what it serves, and a working `--hf` example of
    /// it. Plain lines, one seat each, so a gate can pin them.
    const MODELS: &str = "\
models (the --model word is optional with a model file: the file's architecture picks the seat)
  ds41    DeepSeek-V4.1-Flash          --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M
  qwen38  Qwen3.8-Flash-Next           --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL
  glm     GLM-5.3-Flash                --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL
  qwen3   Qwen3-30B-A3B and Qwen3.6    --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
  decide  a decision model by its head --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M";

    /// What every start prints on stderr before the load begins: the claim
    /// holds for every start (a start cannot know whether it is the card's
    /// first).
    const FIRST_START: &str = "bloomery-serve: a first start on a new card compiles the GPU \
                               code for it (tens of seconds); later starts take seconds";

    /// The spellings of the model path flag.
    const PATH_FLAGS: &[&str] = &["-m", "--model-file"];

    /// The generative seats, one a `--model` word.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Model {
        Ds41,
        Qwen38,
        Glm,
        Qwen3,
    }

    /// A `--model` word: a generative seat's, or the decide seat's.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Word {
        Seat(Model),
        Decide,
    }

    impl Word {
        fn parse(v: &str) -> Result<Word, GateError> {
            match v {
                "ds41" => Ok(Word::Seat(Model::Ds41)),
                "qwen38" => Ok(Word::Seat(Model::Qwen38)),
                "glm" => Ok(Word::Seat(Model::Glm)),
                "qwen3" => Ok(Word::Seat(Model::Qwen3)),
                serve::decide::WORD => Ok(Word::Decide),
                other => Err(format!(
                    "--model is ds41, qwen38, glm, qwen3 or {}, not {other}",
                    serve::decide::WORD
                )
                .into()),
            }
        }

        fn word(self) -> &'static str {
            match self {
                Word::Seat(m) => m.word(),
                Word::Decide => serve::decide::WORD,
            }
        }
    }

    impl Model {
        /// Every generative seat.
        const ALL: [Model; 4] = [Model::Ds41, Model::Qwen38, Model::Glm, Model::Qwen3];

        /// The seat's word, as errors and the usage name it.
        fn word(self) -> &'static str {
            match self {
                Model::Ds41 => "ds41",
                Model::Qwen38 => "qwen38",
                Model::Glm => "glm",
                Model::Qwen3 => "qwen3",
            }
        }

        /// Whether a file of architecture `arch` (its `general.architecture`
        /// string) is this seat's. `qwen4exp` is not an `Arch` name, which
        /// is why the seat of a file is read from the string.
        fn serves(self, arch: &str) -> bool {
            match self {
                Model::Ds41 => matches!(arch, "deepseek41" | DEEPSEEK4),
                Model::Qwen38 => arch == "qwen4exp",
                Model::Glm => arch == "glm5next",
                Model::Qwen3 => matches!(arch, "qwen3moe" | "qwen35moe"),
            }
        }
    }

    /// `args` with every `--model V` pair taken out: the last pair's word and
    /// the rest. `--model` with no value is refused by name.
    fn take_model(args: &[String]) -> Result<(Option<Word>, Vec<String>), GateError> {
        let mut word = None;
        let mut rest = Vec::with_capacity(args.len());
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if flag != "--model" {
                rest.push(flag.clone());
                continue;
            }
            let v = it.next().ok_or_else(|| {
                format!(
                    "--model needs a value (ds41, qwen38, glm, qwen3 or {}): {USAGE}",
                    serve::decide::WORD
                )
            })?;
            word = Some(Word::parse(v)?);
        }
        Ok((word, rest))
    }

    /// Every seat's word, in [`USAGE`]'s order, as the refusal of a file no
    /// seat serves lists them.
    fn seat_words() -> String {
        Model::ALL
            .iter()
            .map(|m| m.word())
            .chain([serve::decide::WORD])
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The seat the model file's ([`ref_model_path`]) first shard holds, with
    /// its `general.architecture` string: the one generative seat that serves
    /// the architecture ([`Model::serves`]; the string itself, not an `Arch`
    /// variant, because `qwen4exp` is no `Arch` name). A file of a decision
    /// row's backbone (`qwen35`, Clef's) here has no head: its refusal
    /// ([`serve::decide::no_head`]) says what would seat it. Any other
    /// architecture, and a shard that names none, is refused by name listing
    /// the seat words. A qwen35moe file (Qwen3.6) is the qwen3 seat's.
    fn seat_of_file() -> Result<(Model, String), GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let (seat, arch) = match split.architecture() {
            Some(a) if decide::ROWS.iter().any(|r| r.backbones.contains(&a)) => {
                return Err(format!(
                    "{}: {}",
                    path.display(),
                    serve::decide::no_head(a, decide::ROWS)
                )
                .into());
            }
            Some(a) => (
                Model::ALL
                    .iter()
                    .copied()
                    .find(|m| m.serves(a))
                    .ok_or_else(|| {
                        format!(
                            "{} is a {a} file, which no seat serves; the seats are --model {}",
                            path.display(),
                            seat_words()
                        )
                    })?,
                a.to_owned(),
            ),
            None => {
                return Err(format!(
                    "{}: the first shard names no architecture; the seats are --model {}",
                    path.display(),
                    seat_words()
                )
                .into());
            }
        };
        Ok((seat, arch))
    }

    /// The `general.architecture` of the model file's first shard
    /// ([`ref_model_path`]).
    fn file_arch() -> Result<String, GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        Ok(split.architecture().unwrap_or("<missing>").to_owned())
    }

    /// The head the arguments name ([`serve::decide::pick`]) and the rest:
    /// `--head` and `--head-config` taken out, the card of `repo` (the
    /// `--hf` repo) read when the pick needs it.
    fn head_of(
        word: Option<Word>,
        repo: Option<&str>,
        rest: &[String],
    ) -> Result<(Option<decide::Pick>, Vec<String>), GateError> {
        let (given, rest) = serve::decide::take_head(rest)?;
        let arch = file_arch()?;
        let ask = Ask {
            head: given.head.as_deref(),
            head_config: given.config.as_deref(),
            word: word.map(Word::word),
            hf: repo,
            arch: &arch,
            generative: Model::ALL.iter().any(|m| m.serves(&arch)),
        };
        let from = serve::decide::pick(&ask, decide::ROWS, &mut |r| {
            model_file::quantized_from(r).map_err(|e| e.to_string())
        })?;
        Ok((from, rest))
    }

    pub fn run() -> ExitCode {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--version") {
            println!("{}", model_file::version(NAME));
            return ExitCode::SUCCESS;
        }
        let (word, rest) = match take_model(&args) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if word.is_none() && (args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h")) {
            // No seat chosen yet; the seat's own usage follows once one is.
            return exit_with(NAME, Err(format!("{USAGE}\n{MODELS}").into()));
        }
        eprintln!("{FIRST_START}");
        // The model file is named before any seat parses: `-m`, or a `--hf`
        // set fetched into the cache.
        let (flags, rest) = match model_file::take(&rest, PATH_FLAGS) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if let Err(e) = model_file::name(&flags) {
            return exit_with(NAME, Err(e));
        }
        // A head names the decide seat; with none the file goes to the
        // generative seats.
        let repo = match flags.hf.as_deref().map(hf::RepoRef::parse).transpose() {
            Ok(r) => r.map(|r| r.repo),
            Err(e) => return exit_with(NAME, Err(e.into())),
        };
        let (from, rest) = match head_of(word, repo.as_deref(), &rest) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if let Some(from) = from {
            return ended(decide::run(&rest, from, repo.as_deref()));
        }
        let (seat, arch) = match seat_of_file() {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if let Some(Word::Seat(word)) = word
            && !word.serves(&arch)
        {
            return exit_with(
                NAME,
                Err(format!(
                    "--model {} on a {arch} file, which serves --model {}",
                    word.word(),
                    seat.word()
                )
                .into()),
            );
        }
        let r = match seat {
            Model::Ds41 => ds41::run(&rest),
            Model::Qwen38 => qwen38::run(&rest),
            Model::Glm => glm::run(&rest),
            Model::Qwen3 => qwen3::run(&rest),
        };
        ended(r)
    }

    /// The process's exit for a seat's end, the decide seat's and the
    /// generative seats' alike.
    fn ended(r: Result<ServeError, GateError>) -> ExitCode {
        match r {
            // EX_SOFTWARE: the engine, not the listener or the load, ended the run.
            Ok(ServeError::Engine(f)) => {
                eprintln!("{NAME}: {f}");
                ExitCode::from(70)
            }
            Ok(e) => exit_with(NAME, Err(e.into())),
            Err(e) => exit_with(NAME, Err(e)),
        }
    }
}
