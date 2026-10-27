//! `bloomery-serve` — one server binary, one seat a model: `--model
//! ds41|qwen38|glm|qwen3|mimo2|solar2|minimaxm3|nemotron|raon1|musespark|flint|decide` picks the seat
//! (`serve_seats`, the shared seat modules), every other flag the seat's own,
//! parsed by the seat's own parser, so each seat's flag set is exactly its
//! per-model binary's. The
//! word is optional with a model file: with none, the file's architecture
//! picks the seat, and a bare start or `--help` prints [`drive::MODELS`],
//! one line a seat with a working `--hf` example.
//!
//!     bloomery-serve [--model ds41|qwen38|glm|qwen3|mimo2|solar2|minimaxm3|nemotron|raon1|musespark|flint|decide]
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
//! refuses or the hub answers busy or down, and every start under
//! `HF_HUB_OFFLINE`, leaves the cache's one verified set of that quant to
//! stand in, after an `hf offline` record); with neither,
//! `$BLOOMERY_REF_MODEL`, the gates' variable. A model named twice — a flag
//! given twice, `-m` beside `--hf`, a flag beside a variable that names
//! another file — is refused by name (`bloomery_gpu_gates::model_file`).
//! A decision model is named by its head, not by its architecture
//! (`serve::decide::pick`): `--head`, or a `--hf` repo whose model card
//! names a decision row's head repo as the model it quantizes, picks the
//! decide seat (`serve_seats::decide`); so does a model file that carries
//! its head, by its `<arch>.decision.type` (lev: the file's own head, no
//! flag) or its `clef` architecture; `--head` beside a generative seat's
//! `--model` is refused by name, and so is `--model decide` with no head,
//! and a file whose decision type no row serves (`kev`).
//! Otherwise, with no `--model`, the architecture of the file's first shard
//! picks the seat: deepseek41 and deepseek4 the ds41 seat, glm5next and
//! glm5-next the glm seat, qwen4exp the qwen38 seat, qwen3moe and qwen35moe
//! the qwen3 seat, mimo2 the mimo2 seat, solar-open2 the solar2 seat, minimax-m3
//! the minimaxm3 seat,
//! nemotron_h and nemotron_h_moe the nemotron seat, raon1 the raon1 seat, muse-spark the musespark seat,
//! and flint the flint seat;
//! a dense qwen35 file generates through `--model qwen3`. With no model
//! word it keeps the decision selection rules: a file with no head is
//! refused by name, saying what would seat it (`--head`, a row's `--hf`
//! repo, or a file that carries its head); a
//! file whose architecture no seat serves is refused by name listing the
//! seat words, or, when the model reader describes it, listing what no
//! program runs of it (`model::arch::coverage::check`, as the plans refuse
//! it); a file whose architecture does not match an explicit
//! `--model` is refused by name before any load.
//! `--version` prints this crate's version and the build's commit and exits.
//! `--embd-model FILE` or `--embd-hf <repo>[:<quant>]` names an embedding
//! model the server holds beside a generative seat's model for its embeddings routes
//! (`bloomery_gpu_gates::side_seat`): read and seated before the chat model's plan, which holds
//! its bytes when it sits on the stage card, and loaded after the chat model. A file of an
//! embedding architecture given as `-m` is refused by name. The decide seat answers its row's
//! POST routes and no embeddings route, so an embedding model beside it is refused by name.
//! EmbeddingGemma 2's row opens the encoder on the seat card;
//! its pass owns the output projection and mean pooling, and the HTTP layer cuts and normalizes.
//!
//! llama-server's placement flags (`-dev`, `-sm`, `-mg`, `-ts`, `-ngl`, `--cpu-moe`, `-fit`, `-fitt`,
//! `-devd` and kin) and the `LLAMA_ARG_*` variables of all but `-devd` are read once the seat is known
//! (`bloomery_gpu_gates::llama_launch`, the start every launcher takes): the cards they
//! name become the `--place` word the seat parses, the levers they are are handed over in place of the
//! environment's before the seat reads its own, a line on stderr says each reading, and a flag bloomery has no
//! thing for is refused by name; `--place` beside any of them is refused. `--list-devices` prints
//! the devices as llama-server does and exits before a model is read.
//!
//! Two checks run before anything is fetched or loaded, each a named refusal
//! (`serve::preflight`): first of all, on a CPU that does not run the
//! release's x86-64-v3 host code, `cpu_gate` prints what it lacks and exits
//! 1; after `--version`, `--help` and the arguments' check below, a driver below R580 (CUDA 13), or
//! one whose `libcuda.so.1` does not load, ends the start with what it reported.
//!
//! The arguments are checked as far as that needs no model, card or network before the driver is
//! asked and before `--hf` fetches anything (`drive::early`): with `--model WORD`, `--help` or `-h`
//! prints the seat's usage, `--levers` its lever table, and a flag, value or lever the seat refuses
//! ends the start with the seat's own refusal; beside llama-server's placement flags, whose levers
//! are known once the cards are read, the levers wait for the seat's run. A `--head` selects the
//! decide seat; without it or `--model`, the model file picks the seat, so only arguments every
//! seat refuses end the start there, each
//! refusal once with the seats that give it, and the picked seat reads them again once the file is
//! known. What needs the model (its architecture, its context, its draft) is checked after the
//! fetch.
//! Each seat's flags, records, `/props` fields and exit codes are its
//! module's doc; `bloomery-serve-ds41` and `bloomery-serve-qwen38` are this
//! binary's ds41 and qwen38 seats alone.

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    // Before any x86-64-v3 code: `drive::run` is not inlined, so `main` holds
    // nothing but the two calls.
    #[cfg(target_arch = "x86_64")]
    serve::preflight::cpu_gate("bloomery-serve");
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

// lev's row's own part (`crate::lev_seat`).
#[cfg(feature = "clef")]
#[path = "shared/lev_seat.rs"]
mod lev_seat;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_draft.rs"]
mod draft;

// The qwen3 seat's draft opens through the same HF reader the dspark gates use
// (`crate::dspark_hf`) and its verify capture records nothing (`crate::quiet`).
#[cfg(feature = "deepseek41")]
#[path = "shared/dspark_hf.rs"]
#[allow(
    dead_code,
    reason = "the serve seat opens a draft; the gate-only readers stay unused here"
)]
mod dspark_hf;

#[cfg(feature = "deepseek41")]
#[path = "shared/quiet.rs"]
#[allow(
    clippy::duplicate_mod,
    reason = "each binary includes the shared log by path; a seat binary may reach it twice"
)]
mod quiet;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_place.rs"]
mod place;

#[cfg(feature = "glm5next")]
mod drive {
    use std::ffi::OsString;
    use std::process::ExitCode;

    use bloomery_gpu_gates::side_seat;
    use bloomery_gpu_gates::{
        GateError, exit_with, llama_launch, llama_place, model_file, ref_model_path,
    };
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::{Arch, DEEPSEEK4};
    use serve::ServeError;
    use serve::decide::Ask;
    use serve::preflight::Driver;

    use crate::serve_seats::{
        decide, ds41, flint, glm, mimo2, minimaxm3, musespark, nemotron, qwen3, qwen38, raon1,
        solar2,
    };

    const NAME: &str = "bloomery-serve";

    const USAGE: &str = "usage: bloomery-serve [--model ds41|qwen38|glm|qwen3|mimo2|solar2|minimaxm3|nemotron|raon1|musespark|flint|decide] \
                         [-m PATH | --model-file PATH | --hf <repo>[:<quant>]] [--head <weights> \
                         [--head-config <file>]] [--embd-model FILE | --embd-hf \
                         <repo>[:<quant>]] [--embd-ctx N] [--embd-cache-bytes N] [--embd-mmproj FILE | \
                         --embd-no-mmproj] [--embd-image-min-tokens N] [--embd-image-max-tokens N] \
                         <the chosen seat's own flags, as \
                         bloomery-serve-ds41, bloomery-serve-qwen38, the glm, the qwen3, the \
                         mimo2, the solar2, the minimaxm3, the nemotron, the raon1, the musespark, the flint or the decide seat takes \
                         them> [llama-server's placement flags: -dev, -sm, -mg, -ts, -ngl, \
                         --cpu-moe, -fit, -fitt, -devd] | --list-devices | --version";

    /// One line a seat, printed beside [`USAGE`] on `--help` and a bare
    /// start: the seat word, what it serves, and a working `--hf` example of
    /// it. Plain lines, one seat each, so a gate can pin them.
    const MODELS: &str = "\
models (the --model word is optional with a model file: the file's architecture picks the seat)
  ds41    DeepSeek-V4.1-Flash          --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M
  qwen38  Qwen3.8-Flash-Next           --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL
  glm     GLM-5.3-Flash                --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL
  qwen3   Qwen3-30B-A3B and Qwen3.6    --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
  mimo2   MiMo-V2.6-Flash              --hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4
  solar2  Solar-Open2-250B             -m <path>
  minimaxm3 MiniMax-M3-UD-IQ4_XS        -m <path>
  nemotron Nemotron 3 Nano             --hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M
  nemotron Nemotron 3.5 Bolt            -m <path>
  raon1   Raon-1-mini                  -m <path>
  musespark Muse Spark Open 816B       -m <path>
  flint   Flint                       -m <path>
  decide  a decision model by its head --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M
  decide  a decision model in its file --hf ggml-org/lev-GGUF:Q4_K_M";

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
        Mimo2,
        Solar2,
        MinimaxM3,
        Nemotron,
        Raon1,
        MuseSpark,
        Flint,
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
                "mimo2" => Ok(Word::Seat(Model::Mimo2)),
                "solar2" => Ok(Word::Seat(Model::Solar2)),
                "minimaxm3" => Ok(Word::Seat(Model::MinimaxM3)),
                "nemotron" => Ok(Word::Seat(Model::Nemotron)),
                "raon1" => Ok(Word::Seat(Model::Raon1)),
                "musespark" => Ok(Word::Seat(Model::MuseSpark)),
                "flint" => Ok(Word::Seat(Model::Flint)),
                serve::decide::WORD => Ok(Word::Decide),
                other => Err(format!(
                    "--model is ds41, qwen38, glm, qwen3, mimo2, solar2, minimaxm3, nemotron, raon1, musespark, flint or {}, not {other}",
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
        const ALL: [Model; 11] = [
            Model::Ds41,
            Model::Qwen38,
            Model::Glm,
            Model::Qwen3,
            Model::Mimo2,
            Model::Solar2,
            Model::MinimaxM3,
            Model::Nemotron,
            Model::Raon1,
            Model::MuseSpark,
            Model::Flint,
        ];

        /// The seat's word, as errors and the usage name it.
        fn word(self) -> &'static str {
            match self {
                Model::Ds41 => "ds41",
                Model::Qwen38 => "qwen38",
                Model::Glm => "glm",
                Model::Qwen3 => "qwen3",
                Model::Mimo2 => "mimo2",
                Model::Solar2 => "solar2",
                Model::MinimaxM3 => "minimaxm3",
                Model::Nemotron => "nemotron",
                Model::Raon1 => "raon1",
                Model::MuseSpark => "musespark",
                Model::Flint => "flint",
            }
        }

        /// Whether the file whose first shard is `file` is this seat's. The qwen3 seat
        /// serves a file by the program that runs its description's layers
        /// ([`bloomery_gpu_gates::qwen_seat::Family::of`]); the other seats read the file's
        /// `general.architecture` string: `qwen4exp` is not an `Arch` name, which is why
        /// the seat of such a file is read from the string, and the glm seat takes both
        /// spellings of its architecture through [`Arch::from_name`].
        fn serves(self, file: &Split) -> bool {
            let arch = file.architecture().unwrap_or("<missing>");
            match self {
                Model::Ds41 => matches!(arch, "deepseek41" | DEEPSEEK4),
                Model::Qwen38 => arch == "qwen4exp",
                Model::Glm => matches!(Arch::from_name(arch), Ok(Arch::Glm5next)),
                Model::Qwen3 => bloomery_gpu_gates::qwen_seat::Family::of(file).is_ok(),
                Model::Mimo2 => arch == "mimo2",
                Model::Solar2 => matches!(Arch::from_name(arch), Ok(Arch::SolarOpen2)),
                Model::MinimaxM3 => matches!(Arch::from_name(arch), Ok(Arch::MiniMaxM3)),
                Model::Nemotron => matches!(
                    Arch::from_name(arch),
                    Ok(Arch::NemotronH | Arch::NemotronHMoe)
                ),
                Model::Raon1 => matches!(Arch::from_name(arch), Ok(Arch::Raon1)),
                Model::MuseSpark => matches!(Arch::from_name(arch), Ok(Arch::MuseSpark)),
                Model::Flint => matches!(Arch::from_name(arch), Ok(Arch::Flint)),
            }
        }
    }

    /// `args` with every `--model V` pair taken out: the last pair's word and
    /// the rest. `--model` with no value is refused by name.
    fn take_model(args: &[String]) -> Result<(Option<Word>, Vec<String>), GateError> {
        let mut word = None;
        let (selected, rest) = llama_launch::take_serve_args(args, &["--model"]);
        let mut it = selected.iter();
        while it.next().is_some() {
            let v = it.next().ok_or_else(|| {
                format!(
                    "--model needs a value (ds41, qwen38, glm, qwen3, mimo2, solar2, minimaxm3, nemotron, raon1, musespark, flint or {}): {USAGE}",
                    serve::decide::WORD
                )
            })?;
            word = Some(Word::parse(v)?);
        }
        Ok((word, rest))
    }

    /// The model source parser receives model options without mistaking other values for flags.
    fn take_source(args: &[String]) -> Result<(hf::source::Flags, Vec<String>), GateError> {
        let flags: Vec<&str> = PATH_FLAGS.iter().copied().chain(["--hf"]).collect();
        let (selected, rest) = llama_launch::take_serve_args(args, &flags);
        let (model, _) = model_file::take(&selected, PATH_FLAGS)?;
        Ok((model, rest))
    }

    /// The head parser receives only head flags, preserving values of the seat's other options.
    fn take_head(args: &[String]) -> Result<(serve::decide::HeadFlags, Vec<String>), GateError> {
        let (selected, rest) = llama_launch::take_serve_args(args, &["--head", "--head-config"]);
        let (head, _) = serve::decide::take_head(&selected)?;
        Ok((head, rest))
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

    /// The seat the model file's ([`ref_model_path`]) first shard holds: the one
    /// generative seat that serves the file ([`Model::serves`]). A file of a decision
    /// row's backbone (`qwen35`, Clef's) here has no head: its refusal
    /// ([`serve::decide::no_head`]) says what would seat it. An explicit
    /// generative word admits a dense qwen35 file through the existing Qwen
    /// seat. Any other file, and a shard that names no architecture, is refused by name
    /// listing the seat words; an explicit seat word that is not the file's seat is
    /// refused naming the file's. A file the Qwen3.5 body runs (Qwen3.6, a dense
    /// `llama` trunk) is the qwen3 seat's.
    fn seat_of_file(word: Option<Word>) -> Result<Model, GateError> {
        let generative = matches!(word, Some(Word::Seat(Model::Qwen3)));
        let path = ref_model_path()?;
        model_file::with_header(&path, |split| {
            let seat = match split.architecture() {
                Some(a) if side_seat::row(a).is_some() => {
                    return Err(format!(
                        "{} is a {a} file, an embedding model: serve it beside a chat model with \
                     --embd-model",
                        path.display()
                    )
                    .into());
                }
                Some(a) if !generative && decide::ROWS.iter().any(|r| r.backbones.contains(&a)) => {
                    return Err(format!(
                        "{}: {}",
                        path.display(),
                        serve::decide::no_head(a, decide::ROWS)
                    )
                    .into());
                }
                Some(a) => {
                    let seat = Model::ALL
                        .iter()
                        .copied()
                        .find(|m| m.serves(split))
                        .ok_or_else(|| no_seat(&path, split, a))?;
                    if let Some(Word::Seat(word)) = word
                        && !word.serves(split)
                    {
                        return Err(format!(
                            "--model {} on a {a} file, which serves --model {}",
                            word.word(),
                            seat.word()
                        )
                        .into());
                    }
                    seat
                }
                None => {
                    return Err(format!(
                        "{}: the first shard names no architecture; the seats are --model {}",
                        path.display(),
                        seat_words()
                    )
                    .into());
                }
            };
            Ok(seat)
        })?
    }

    /// The refusal of a file of architecture `a` that no seat serves: what no
    /// program runs of it (the coverage list the plans refuse a file with)
    /// when the model reader describes it, else the seat words, with the
    /// reader's refusal when it has one of its own.
    fn no_seat(path: &std::path::Path, split: &Split, a: &str) -> String {
        let head = format!("{} is a {a} file, which no seat serves", path.display());
        match model::arch::spec(split) {
            Ok(read) => {
                let missing = model::arch::coverage::check(&read.spec, &read.tensors);
                if missing.is_empty() {
                    format!("{head}; the seats are --model {}", seat_words())
                } else {
                    format!(
                        "{head}: {}",
                        model::placement::PlacementError::Unimplemented(missing)
                    )
                }
            }
            Err(model::ModelError::UnknownArchitecture(_)) => {
                format!("{head}; the seats are --model {}", seat_words())
            }
            Err(e) => format!("{head} ({e}); the seats are --model {}", seat_words()),
        }
    }

    /// The `general.architecture` of the model file's first shard
    /// ([`ref_model_path`]), its `<arch>.decision.type` when it has one, and whether a
    /// generative seat serves the file ([`Model::serves`]).
    fn file_facts() -> Result<(String, Option<String>, bool), GateError> {
        let path = ref_model_path()?;
        model_file::with_header(&path, |split| {
            let decision = model_file::decision_type(split)?.map(str::to_owned);
            Ok((
                split.architecture().unwrap_or("<missing>").to_owned(),
                decision,
                Model::ALL.iter().any(|m| m.serves(split)),
            ))
        })?
    }

    /// The head the arguments name ([`serve::decide::pick`]) and the rest:
    /// `--head` and `--head-config` taken out, the card of `repo` (the
    /// `--hf` repo) read when the pick needs it.
    fn head_of(
        word: Option<Word>,
        repo: Option<&str>,
        rest: &[String],
    ) -> Result<(Option<decide::Pick>, Vec<String>), GateError> {
        let (given, rest) = take_head(rest)?;
        let (arch, decision, served) = file_facts()?;
        let ask = Ask {
            head: given.head.as_deref(),
            head_config: given.config.as_deref(),
            word: word.map(Word::word),
            hf: repo,
            arch: &arch,
            decision: decision.as_deref(),
            // A qwen35 repo can name a decision head in its card. Keep that
            // inference with no --model; an explicit qwen3 word opts into generation.
            generative: arch != "qwen35" && served,
        };
        let from = serve::decide::pick(&ask, decide::ROWS, &mut |r| {
            model_file::quantized_from(r).map_err(|e| e.to_string())
        })?;
        Ok((from, rest))
    }

    /// What the driver answers a start: its library's load as cuda-bindings
    /// loads it (`dyn_load`: `libcuda.so.1`, then `libcuda.so`, and its
    /// `cuDriverGetVersion` refused as too old below the CUDA major the
    /// build's toolkit is), read through that crate's own
    /// `cuda_driver_load_error`.
    fn driver_found() -> Driver {
        match sys::cuda_driver_load_error() {
            None => Driver::Loaded,
            Some(sys::DynLoadError::RuntimeTooOld {
                runtime_version, ..
            }) => i32::try_from(*runtime_version).map_or_else(
                |_| Driver::Missing(format!("a runtime version {runtime_version}")),
                Driver::Version,
            ),
            Some(e) => Driver::Missing(e.to_string()),
        }
    }

    #[inline(never)]
    pub fn run() -> ExitCode {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if llama_launch::serve_option(&args, "--version") {
            println!(
                "{NAME} {} (commit {})",
                env!("CARGO_PKG_VERSION"),
                serve::COMMIT
            );
            return ExitCode::SUCCESS;
        }
        // llama-server's `LLAMA_ARG_MODEL` and `LLAMA_ARG_HF_REPO` name the model unless the
        // command line does.
        let args = match llama_launch::model_from_env(NAME, args, PATH_FLAGS) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        let (word, rest) = match take_model(&args) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if word.is_none()
            && !llama_launch::serve_option(&args, "--head")
            && (args.is_empty()
                || llama_launch::serve_option(&args, "--help")
                || llama_launch::serve_option(&args, "-h"))
        {
            // No seat chosen yet; the seat's own usage follows once one is.
            return exit_with(NAME, Err(format!("{USAGE}\n{MODELS}").into()));
        }
        if let Err(e) = early(word, &rest, std::env::vars_os()) {
            return exit_with(NAME, Err(e));
        }
        // A driver that cannot run the CUDA 13 code ends the start before any
        // download.
        if let Err(e) = serve::preflight::driver(&driver_found()) {
            return exit_with(NAME, Err(e.into()));
        }
        if llama_launch::serve_option(&args, llama_place::LIST_DEVICES) {
            match llama_launch::answer_list_devices(&args) {
                Ok(true) => return ExitCode::SUCCESS,
                Ok(false) => {}
                Err(e) => return exit_with(NAME, Err(e)),
            }
        }
        eprintln!("{FIRST_START}");
        // The model file is named before any seat parses: `-m`, or a `--hf`
        // set fetched into the cache.
        let (flags, rest) = match take_source(&rest) {
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
        // The side model's flags come out before the seat reads its own; its file is read in the
        // seat, after the levers.
        let (side, rest) = match side_seat::Flags::take(&rest) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if let Some(from) = from {
            if let Err(e) = decide::check_side(&side) {
                return exit_with(NAME, Err(e));
            }
            // The decision seat places no card, so the placement flags' start does not run for
            // it; the variables and short spellings of the flags it does take are read as the
            // generative seats' are, `LLAMA_API_KEY` among them.
            let rest = match llama_launch::seat_from_env(NAME, rest, llama_launch::SERVE) {
                Ok(v) => v,
                Err(e) => return exit_with(NAME, Err(e)),
            };
            return ended(decide::run(&rest, from, repo.as_deref()));
        }
        let seat = match seat_of_file(word) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        // llama-server's placement flags and variables, read once the seat is known: the levers some
        // are, and the `--place` word the others come to.
        let seat_word = || -> Result<&'static str, GateError> { Ok(seat.word()) };
        let layers = || -> Result<Option<u64>, GateError> {
            let split = model_file::header(&ref_model_path()?)?;
            Ok(split.arch_get_u64("block_count"))
        };
        let launcher = llama_launch::Launcher {
            bin: NAME,
            seat: &seat_word,
            layers: &layers,
            reads: llama_launch::SERVE,
        };
        let rest = match launcher.placed(&rest) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        let (run, _) = entries(seat);
        ended(run(&rest, &side))
    }

    /// A generative seat's `run`.
    type Run = fn(&[String], &side_seat::Flags) -> Result<ServeError, GateError>;
    /// A generative seat's `check` before the fetch ([`early`]).
    type Check = fn(&[String], bool, bool) -> Result<(), GateError>;

    /// The entries of `seat`'s module: the start runs the one and [`early`] asks the other.
    fn entries(seat: Model) -> (Run, Check) {
        macro_rules! entry {
            ($seat:ident, source) => {
                ($seat::run, |args, read, from_repo| {
                    $seat::check(args, read, from_repo).map(drop)
                })
            };
            ($seat:ident) => {
                ($seat::run, |args, read, _| {
                    $seat::check(args, read).map(drop)
                })
            };
        }
        match seat {
            Model::Ds41 => entry!(ds41, source),
            Model::Qwen38 => entry!(qwen38, source),
            Model::Glm => entry!(glm, source),
            Model::Qwen3 => entry!(qwen3, source),
            Model::Mimo2 => entry!(mimo2),
            Model::Solar2 => entry!(solar2),
            Model::MinimaxM3 => entry!(minimaxm3),
            Model::Nemotron => entry!(nemotron),
            Model::Raon1 => entry!(raon1),
            Model::MuseSpark => entry!(musespark),
            Model::Flint => entry!(flint),
        }
    }

    /// The checks of `args` (the arguments less `--model`) that read no model, open no card and
    /// fetch nothing, made before the driver is asked and the model fetched: the model, head and
    /// side flags are taken out as the start takes them, then each seat the start may come to
    /// checks what it would parse: the seat `word` names or `--head` selects, else every seat
    /// ([`any_seat`]), since the model file picks it. A generative seat reads the arguments less
    /// llama-server's placement flags and variables, which a refusal of theirs ends there
    /// ([`llama_launch::seat_view`]); the decide seat, which places no card, reads them as its
    /// start does ([`llama_launch::seat_args`]). What needs the model stays after the fetch.
    /// `--list-devices` and `--records-schema` are left to the start, and so is `--levers` where no
    /// seat is named or, at a generative seat, the placement flags hand levers over once the census
    /// is read. `vars` supplies the reference model and the llama-server variables the seats read as
    /// flags.
    fn early(
        word: Option<Word>,
        args: &[String],
        vars: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Result<(), GateError> {
        let asked = |flag: &str| llama_launch::serve_option(args, flag);
        if asked(llama_place::LIST_DEVICES) || asked("--records-schema") {
            return Ok(());
        }
        let (model, rest) = take_source(args)?;
        let vars: Vec<_> = vars.into_iter().collect();
        let reference = vars
            .iter()
            .find_map(|(name, value)| {
                (name.as_os_str() == std::ffi::OsStr::new("BLOOMERY_REF_MODEL")).then_some(value)
            })
            .map(|value| value.to_str().ok_or("BLOOMERY_REF_MODEL is not UTF-8"))
            .transpose()?;
        let source = hf::source::resolve(&model, reference)?;
        let from_repo = matches!(source, Some(hf::source::Source::Hf(_)));
        let (head, rest) = take_head(&rest)?;
        serve::decide::check_head(
            word.map(Word::word),
            head.head.as_deref(),
            head.config.as_deref(),
        )?;
        let word = word.or(head.head.map(|_| Word::Decide));
        let (side, rest) = side_seat::Flags::take(&rest)?;
        let env = llama_launch::llama_env(vars)?;
        let check_decide = || -> Result<(), GateError> {
            decide::check_side(&side)?;
            let (unplaced, _) = llama_launch::seat_args(rest.clone(), &env, llama_launch::SERVE)?;
            decide::check(&unplaced, from_repo).map(drop)
        };
        if word == Some(Word::Decide) {
            check_decide()?;
            if source.is_none() {
                return Err(model_file::NONE.into());
            }
            return Ok(());
        }
        side.check_before_model()?;
        if word.is_none() && asked("--levers") {
            return Ok(());
        }
        // The arguments as a generative seat reads them, and whether it reads its levers.
        let view = llama_launch::seat_view(&rest, &env, llama_launch::SERVE)
            .map(|(placement, placed, _)| (placed, placement.is_empty()));
        if asked("--levers") && matches!(view, Ok((_, false))) {
            return Ok(());
        }
        let check = |word| match (word, &view) {
            (Word::Decide, _) => check_decide(),
            (Word::Seat(seat), Ok((placed, read_levers))) => {
                (entries(seat).1)(placed, *read_levers, from_repo)
            }
            (Word::Seat(_), Err(e)) => Err(e.to_string().into()),
        };
        match word {
            Some(word) => check(word),
            None => any_seat(check),
        }?;
        if source.is_none() {
            return Err(model_file::NONE.into());
        }
        Ok(())
    }

    /// `check` at every seat, for a start whose model file picks the seat ([`early`]): `Ok` when
    /// one seat takes the arguments, else each refusal once, with the seats that give it.
    fn any_seat(check: impl Fn(Word) -> Result<(), GateError>) -> Result<(), GateError> {
        let mut refused: Vec<(String, Vec<&str>)> = Vec::new();
        let words = Model::ALL.iter().map(|&m| Word::Seat(m));
        for word in words.chain([Word::Decide]) {
            let Err(e) = check(word) else {
                return Ok(());
            };
            let e = e.to_string();
            match refused.iter_mut().find(|(text, _)| *text == e) {
                Some((_, seats)) => seats.push(word.word()),
                None => refused.push((e, vec![word.word()])),
            }
        }
        let lines: Vec<String> = refused
            .iter()
            .map(|(e, seats)| format!("  --model {}: {e}", seats.join(", ")))
            .collect();
        Err(format!(
            "no seat takes these arguments, each refusing them as follows (--model names the seat):\n{}",
            lines.join("\n")
        )
        .into())
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

    #[cfg(test)]
    mod tests {
        use super::*;

        fn line(s: &str) -> Vec<String> {
            s.split_whitespace().map(str::to_owned).collect()
        }

        #[test]
        fn the_reference_source_comes_from_the_start_environment() {
            let word = Some(Word::Seat(Model::Qwen38));
            let e = early(
                word,
                &line("--hf o/r"),
                [(
                    OsString::from("BLOOMERY_REF_MODEL"),
                    OsString::from("other.gguf"),
                )],
            )
            .expect_err("the supplied environment names a second source");
            assert!(
                e.to_string().contains("BLOOMERY_REF_MODEL=other.gguf"),
                "{e}"
            );
            early(word, &line("--hf o/r"), []).expect("no reference was supplied");
            early(
                word,
                &[],
                [(
                    OsString::from("BLOOMERY_REF_MODEL"),
                    OsString::from("other.gguf"),
                )],
            )
            .expect("the supplied reference names a source without reading it");
            let e = early(word, &[], []).expect_err("no source was supplied");
            assert!(e.to_string().contains("no model file"), "{e}");
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                let e = early(
                    word,
                    &line("--hf o/r"),
                    [(
                        OsString::from("BLOOMERY_REF_MODEL"),
                        OsString::from_vec(vec![0xff]),
                    )],
                )
                .expect_err("the supplied reference is not UTF-8");
                assert!(
                    e.to_string().contains("BLOOMERY_REF_MODEL is not UTF-8"),
                    "{e}"
                );
            }
        }

        /// What a start checks before the fetch ([`early`]): with a seat word, the seat's usage
        /// and its refusal of a value, llama-server's placement flags out of its view; with none,
        /// arguments accepted by any seat pass, and an argument every seat
        /// refuses names each seat once. `--list-devices` is left to the start.
        #[test]
        fn a_start_checks_its_arguments_before_the_fetch() {
            let qwen38 = Some(Word::Seat(Model::Qwen38));
            let refusal = |word, s: &str| early(word, &line(s), []).expect_err(s).to_string();
            let usage = refusal(qwen38, "--hf o/r --help");
            assert!(usage.starts_with("usage: bloomery-serve-qwen38"), "{usage}");
            let value = refusal(qwen38, "--hf o/r -ngl 99 --ctx-size abc");
            assert!(value.starts_with("--ctx-size \"abc\""), "{value}");
            // Values a seat's run refuses from the arguments alone, before the model.
            let ds41 = Some(Word::Seat(Model::Ds41));
            let qwen3 = Some(Word::Seat(Model::Qwen3));
            let glm = Some(Word::Seat(Model::Glm));
            for word in [qwen38, ds41, qwen3, glm, Some(Word::Decide), None] {
                for s in [
                    "--hf o/r --mmproj p.gguf --no-mmproj",
                    "--hf o/r --no-mmproj --mmproj p.gguf",
                ] {
                    let e = refusal(word, s);
                    assert!(e.contains("the two flags contradict"), "{word:?} {s}: {e}");
                }
            }
            for (word, s, why) in [
                (
                    ds41,
                    "--hf o/r --ctx-size 0",
                    "each of the 1 slots serves 0 positions",
                ),
                (
                    ds41,
                    "--hf o/r --parallel 4 -c 7",
                    "each of the 4 slots serves 1 positions",
                ),
                (
                    qwen3,
                    "--hf o/r --cache-type-k q4_0",
                    "--cache-type-k takes f16 or q8_0",
                ),
            ] {
                let e = refusal(word, s);
                assert!(e.contains(why), "{s}: {e}");
            }
            for word in Model::ALL.iter().map(|&m| Some(Word::Seat(m))) {
                let e = refusal(word, "--hf o/r --ctx-size 1 --parallel 2");
                assert!(
                    e.contains("--parallel") || e.contains("slots"),
                    "{word:?}: {e}"
                );
            }
            let nemotron = Some(Word::Seat(Model::Nemotron));
            let e = refusal(nemotron, "--hf o/r --parallel 2");
            assert!(e.contains("--parallel 2: both nemotron bodies"), "{e}");
            for s in [
                "--hf o/r --parallel 1",
                "--hf o/r --alias --levers --parallel 1",
            ] {
                early(nemotron, &line(s), []).expect(s);
            }
            for (word, s) in [
                (qwen38, "--hf o/r --alias --cpu_moe --plan"),
                (qwen38, "--hf o/r --alias --cache-ram --ctx-size 8192"),
                (qwen38, "--hf o/r --alias -cram --ctx-size 8192"),
                (qwen38, "--hf o/r --alias --place --ctx-size 8192"),
                (qwen38, "--hf o/r --alias --hf --ctx-size 8192"),
                (qwen38, "--hf o/r --alias --model --ctx-size 8192"),
                (qwen38, "--hf o/r --alias --head --ctx-size 8192"),
                (qwen38, "--hf o/r --alias --embd-hf --ctx-size 8192"),
                (
                    qwen38,
                    "--hf nvidia/Qwen3.8-Flash-Next-FP4-Local --alias --cache-ram -ngl all",
                ),
                (ds41, "--hf o/r --parallel 4 -c 8"),
                (qwen3, "--hf o/r --cache-type-k q8_0"),
            ] {
                early(word, &line(s), []).expect(s);
            }
            if cfg!(feature = "vision") {
                // The repository may hold the projector that supplies this limit.
                early(glm, &line("--hf o/r --image-max-tokens 1024"), [])
                    .expect("a vision build may pick the repository's projector");
            }
            for taken in [
                "--hf o/r -ngl 99 --ctx-size 8192",
                "--hf o/r --prefill steps",
                "--list-devices --frobnicate",
            ] {
                early(None, &line(taken), []).expect(taken);
            }
            let config = refusal(None, "--hf o/r --head-config c");
            assert!(
                config.contains("--head-config c without --head"),
                "{config}"
            );
            let conflict = refusal(qwen38, "--hf o/r --head h");
            assert!(
                conflict.contains("--model qwen38 with --head h"),
                "{conflict}"
            );
            let head = refusal(None, "--hf o/r --head h --alias x");
            assert!(head.contains("--alias"), "{head}");
            assert!(
                head.contains("usage: bloomery-serve [--model decide]"),
                "{head}"
            );
            early(None, &line("--hf o/r --head h"), []).expect("a head picks decide");
            let every = refusal(None, "--hf o/r --frobnicate 1");
            let mut named: Vec<&str> = every
                .lines()
                .skip(1)
                .flat_map(|l| {
                    let (seats, why) = l
                        .trim_start()
                        .strip_prefix("--model ")
                        .and_then(|l| l.split_once(": "))
                        .expect(l);
                    assert!(why.contains("--frobnicate"), "{l}");
                    seats.split(", ")
                })
                .collect();
            named.sort_unstable();
            let mut words: Vec<&str> = Model::ALL
                .iter()
                .map(|m| m.word())
                .chain([serve::decide::WORD])
                .collect();
            words.sort_unstable();
            assert_eq!(named, words, "{every}");
        }

        /// llama-server's placement variables are the generative seats': the decide seat, named or
        /// picked by the file, reads its arguments without them, as its start does, so a refused
        /// split mode leaves it, and an argument it refuses names both readings.
        #[test]
        fn the_decide_seat_reads_no_placement_variable_before_the_fetch() {
            let row = || {
                [(
                    OsString::from("LLAMA_ARG_SPLIT_MODE"),
                    OsString::from("row"),
                )]
            };
            for word in [Some(Word::Decide), None] {
                early(word, &line("--hf o/r"), row()).expect("decide takes it");
            }
            let side = early(None, &line("--hf o/r --embd-hf o/embedding"), row())
                .expect_err("no seat serves embeddings with this placement")
                .to_string();
            assert!(side.contains("no row split"), "{side}");
            assert!(
                side.contains("--model decide: --embd-model: the decide seat"),
                "{side}"
            );
            let named = early(Some(Word::Seat(Model::Qwen38)), &line("--hf o/r"), row());
            assert!(named.expect_err("row").to_string().contains("no row split"));
            let both = early(None, &line("--hf o/r --frobnicate 1"), row())
                .expect_err("no seat")
                .to_string();
            let decide = format!("--model {}: ", serve::decide::WORD);
            let lines: Vec<&str> = both.lines().skip(1).map(str::trim_start).collect();
            assert_eq!(lines.len(), 2, "{both}");
            assert!(lines.iter().any(|l| l.contains("no row split")), "{both}");
            assert!(
                lines
                    .iter()
                    .any(|l| l.starts_with(&decide) && l.contains("--frobnicate")),
                "{both}"
            );
        }

        #[test]
        fn record_options_skip_alias_values_at_every_seat() {
            const CASE: &str = "SERVE_RECORD_ALIAS_SEAT";
            if let Ok(case) = std::env::var(CASE) {
                let (word, mode) = case.split_once(':').unwrap();
                let args = line(if mode == "schema" {
                    "--records-schema"
                } else {
                    "--alias --records-schema --ctx-size abc"
                });
                let result = match Word::parse(word).unwrap() {
                    Word::Seat(seat) => (entries(seat).0)(&args, &side_seat::Flags::default()),
                    Word::Decide => decide::run(
                        &args,
                        serve::decide::HeadFrom::Given {
                            head: "h".into(),
                            config: None,
                        },
                        None,
                    ),
                };
                let e = result
                    .expect_err("the seat must parse the invalid arguments")
                    .to_string();
                assert!(e.contains("--ctx-size") || e.contains("--alias"), "{e}");
                println!("seat parsed its arguments");
                return;
            }
            let mut refused = Vec::new();
            for word in Model::ALL
                .iter()
                .map(|m| m.word())
                .chain([serve::decide::WORD])
            {
                for mode in ["alias", "schema"] {
                    let output = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "drive::tests::record_options_skip_alias_values_at_every_seat",
                            "--nocapture",
                            "--skip",
                            "--records-schema",
                        ])
                        .env_clear()
                        .env(CASE, format!("{word}:{mode}"))
                        .output()
                        .unwrap();
                    let out = String::from_utf8_lossy(&output.stdout);
                    let err = String::from_utf8_lossy(&output.stderr);
                    let answered = if mode == "schema" {
                        out.lines().any(|l| l.starts_with('{'))
                            && !out.contains("seat parsed its arguments")
                    } else {
                        out.contains("seat parsed its arguments")
                    };
                    if !output.status.success() || !answered {
                        refused.push(format!("{word} {mode}: {out} {err}"));
                    }
                }
            }
            assert!(refused.is_empty(), "{}", refused.join("\n"));
        }

        /// A key file the start reads before the fetch and the seat reads again is one snapshot
        /// (`serve::flag::ApiKeys::add`): a pipe gives its keys to both, and is opened once.
        #[test]
        fn a_piped_key_file_is_read_once_for_the_check_and_the_seat() {
            let dir =
                std::env::temp_dir().join(format!("bloomery-piped-key-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let fifo = dir.join("keys");
            let made = std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap();
            assert!(made.success(), "mkfifo: {made}");
            // The writer offers one key, then an empty second stream after the release pipe opens.
            let release = dir.join("release");
            assert!(
                std::process::Command::new("mkfifo")
                    .arg(&release)
                    .status()
                    .unwrap()
                    .success()
            );
            let mut writer = bloomery_gpu_gates::serve_client::Served::spawn_cmd(
                std::process::Command::new("sh"),
                &[
                    "-c",
                    "printf 'piped\\n' > \"$1\"; read -r _ < \"$2\"; : > \"$1\"",
                    "sh",
                    fifo.to_str().unwrap(),
                    release.to_str().unwrap(),
                ],
                &dir,
            )
            .unwrap();
            let args = line(&format!("--api-key-file {}", fifo.display()));
            let early_args = [line("--hf o/r"), args.clone()].concat();
            early(Some(Word::Seat(Model::Qwen38)), &early_args, [])
                .expect("the check reads the key");
            std::fs::write(&release, b"\n").unwrap();
            (entries(Model::Qwen38).1)(&args, false, false).expect("the seat takes the same key");
            // The parse opened nothing: open the pipe for the spent writer.
            std::fs::read(&fifo).unwrap();
            assert!(writer.child.wait().unwrap().success());
            std::fs::remove_dir_all(&dir).unwrap();
        }

        /// The ubatch size the qwen38 and decide seats take is held to the engine's before the
        /// fetch (`ubatch_rule` in their checks), under both spellings and `LLAMA_ARG_UBATCH`.
        #[test]
        fn a_ubatch_size_is_checked_before_the_fetch() {
            let ub = |word, s: &str, env: &[(&str, &str)]| {
                let vars = env.iter().map(|&(k, v)| (k.into(), v.into()));
                early(Some(word), &line(s), vars)
            };
            let none: &[(&str, &str)] = &[];
            for word in [Word::Seat(Model::Qwen38), Word::Decide] {
                for (s, env) in [
                    ("--hf o/r -ub 4097", none),
                    ("--hf o/r --ubatch-size 4097", none),
                    ("--hf o/r", &[("LLAMA_ARG_UBATCH", "4097")]),
                ] {
                    let e = ub(word, s, env).expect_err(s).to_string();
                    assert!(
                        e.contains("--ubatch-size 4097 is not a ubatch size"),
                        "{s}: {e}"
                    );
                }
                ub(word, "--hf o/r -ub 48", none).expect("48 is a size");
            }
            let e = ub(Word::Seat(Model::Qwen38), "--hf o/r -ub 0", none).expect_err("0");
            assert!(
                e.to_string()
                    .contains("--ubatch-size 0 is not a ubatch size"),
                "{e}"
            );
        }
    }
}
