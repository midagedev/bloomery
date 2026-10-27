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
//! ([`drive::FIRST_START`]): a first start on a new card can compile the GPU
//! code for it, which takes tens of seconds (an archive that carries the
//! compiled code skips it).
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
//! 1; after `--version` and `--help`, a driver below R580 (CUDA 13), or one
//! whose `libcuda.so.1` does not load, ends the start with what it reported.
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
    use std::process::ExitCode;

    use bloomery_gpu_gates::side_seat;
    use bloomery_gpu_gates::{GateError, exit_with, llama_launch, model_file, ref_model_path};
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
    const FIRST_START: &str = "bloomery-serve: a first start on a new card can compile the \
                               GPU code for it (tens of seconds); later starts take seconds";

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
        let mut rest = Vec::with_capacity(args.len());
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if flag != "--model" {
                rest.push(flag.clone());
                continue;
            }
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
        let (given, rest) = serve::decide::take_head(rest)?;
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
        if args.iter().any(|a| a == "--version") {
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
        if word.is_none() && (args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h")) {
            // No seat chosen yet; the seat's own usage follows once one is.
            return exit_with(NAME, Err(format!("{USAGE}\n{MODELS}").into()));
        }
        // A driver that cannot run the CUDA 13 code ends the start before any
        // download.
        if let Err(e) = serve::preflight::driver(&driver_found()) {
            return exit_with(NAME, Err(e.into()));
        }
        match llama_launch::answer_list_devices(&args) {
            Ok(true) => return ExitCode::SUCCESS,
            Ok(false) => {}
            Err(e) => return exit_with(NAME, Err(e)),
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
        // The side model's flags come out before the seat reads its own; its file is read in the
        // seat, after the levers.
        let (side, rest) = match side_seat::Flags::take(&rest) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if let Some(from) = from {
            if side.names() {
                return exit_with(
                    NAME,
                    Err("--embd-model: the decide seat answers its row's POST routes and no embeddings route".into()),
                );
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
        let r = match seat {
            Model::Ds41 => ds41::run(&rest, &side),
            Model::Qwen38 => qwen38::run(&rest, &side),
            Model::Glm => glm::run(&rest, &side),
            Model::Qwen3 => qwen3::run(&rest, &side),
            Model::Mimo2 => mimo2::run(&rest, &side),
            Model::Solar2 => solar2::run(&rest, &side),
            Model::MinimaxM3 => minimaxm3::run(&rest, &side),
            Model::Nemotron => nemotron::run(&rest, &side),
            Model::Raon1 => raon1::run(&rest, &side),
            Model::MuseSpark => musespark::run(&rest, &side),
            Model::Flint => flint::run(&rest, &side),
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
