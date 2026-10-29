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
//!                    [-m PATH | --model-file PATH | --hf <repo>[@<rev>][:<quant>]]
//!                    [--head <weights> [--head-config <file>]]
//!                    [--config FILE] <the seat's own flags>
//!     bloomery-serve --version
//!
//! `--config FILE` reads the start's settings from one JSON object, each key a flag's name
//! (`bloomery_gpu_gates::serve_config`): its flags join the command line's before anything else
//! reads them, and a flag the command line gives replaces the file's.
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
//! a dense qwen35 file a generative seat serves generates with or without
//! `--model qwen3`. With no model word a file keeps the decision selection rules: a file with no head
//! that no generative seat serves is refused by name, saying what would seat it (`--head`, a row's `--hf`
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
//! release's x86-64-v3 host code, `cpu_gate` prints what it lacks and ends as
//! the class `no-device` (status 69, with its `error` event); after
//! `--version`, `--help` and the arguments' check below, a driver below R580
//! (CUDA 13), or one whose `libcuda.so.1` does not load, ends the start with
//! what it reported.
//!
//! The arguments are checked as far as that needs no model, card or network before the driver is
//! asked and before `--hf` fetches anything (`drive::early_named`): with `--model WORD`, `--help` or `-h`
//! prints the seat's usage, `--levers` its lever table, and a flag, value or lever the seat refuses
//! ends the start with the seat's own refusal; beside llama-server's placement flags, whose levers
//! are known once the cards are read, the levers wait for the seat's run. A `--head` selects the
//! decide seat; without it or `--model`, the model file picks the seat: arguments every seat
//! refuses end the start there, and arguments only some seats refuse read the `--hf` repo's
//! listing and the architecture of the file it would fetch (a set's first shard's header by HTTP
//! Range, a checkpoint's `config.json`) and end the start when the seats that may serve it refuse
//! them (`drive::any_seat`), each refusal once with the seats that give it; a repo that cannot be
//! read that way (`HF_HUB_OFFLINE`, no network) leaves the arguments to the picked seat, which
//! reads them again once the file is known. What needs the rest of the model (its context, its
//! draft) is checked after the fetch.
//!
//! An agent starts the server by a model's name, the way `docs/agents.md` says: `--model
//! NAME[:VARIANT]` (a catalog model by its name, an alias or its seat word, `hf::catalog`) picks
//! the row of the model that holds on this machine, fetches it at its pinned commit (`--revision
//! REV` reads another) and starts it; `--recommend` does that pick for every model and prints it,
//! `--dry-run` for one, each with `--json` as one document on stdout; `--plan --json` ends with
//! the plan as one JSON line (`bloomery_gpu_gates::start`). The start's events are JSON lines on
//! stdout (`serve::event`: `pick`, `bound`, `ready`, `error`, `stopped`), its listener answers
//! `/health` 503 `loading` from before the fetch, and every refusal ends with the status of
//! `serve::exit::Code`. `--help` and a bare start print the usage on stdout and exit 0.
//!
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

    use bloomery_gpu_gates::serve_config::SOURCE_FLAGS;
    use bloomery_gpu_gates::side_seat;
    use bloomery_gpu_gates::start::{self, Refusal};
    use bloomery_gpu_gates::{
        GateError, gpu_census, llama_launch, llama_place, model_file, ref_model_path, serve_config,
    };
    use gguf::Split;
    use hf::catalog::{self, CatalogError, CatalogRef, Outcome};
    use model::arch::{Arch, DEEPSEEK4};
    use serde_json::{Value, json};
    use serve::ServeError;
    use serve::decide::Ask;
    use serve::event::{self, Event};
    use serve::exit::Code;
    use serve::listener::{Phase, Progress};

    use crate::serve_seats::{
        decide, ds41, flint, glm, mimo2, minimaxm3, musespark, nemotron, qwen3, qwen38, raon1,
        solar2,
    };

    const NAME: &str = "bloomery-serve";

    /// The first line of `--help` and of a bare start: what an agent runs to learn what this
    /// machine serves, and how it starts one.
    const LEAD: &str = "bloomery-serve: run `bloomery-serve --recommend --json` to see what this \
                        machine can serve, then `bloomery-serve --model NAME` to start it \
                        (docs/agents.md)";

    const USAGE: &str = "usage: bloomery-serve [--model NAME[:VARIANT]|ds41|qwen38|glm|qwen3|mimo2|solar2|minimaxm3|nemotron|raon1|musespark|flint|decide] \
                         [--config FILE] [--revision REV] [--recommend] [--dry-run] [--json] \
                         [-m PATH | --model-file PATH | --hf <repo>[@<rev>][:<quant>]] [--head <weights> \
                         [--head-config <file>]] [--embd-model FILE | --embd-hf \
                         <repo>[@<rev>][:<quant>]] [--embd-ctx N] [--embd-cache-bytes N] [--embd-mmproj FILE | \
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

        /// Whether the file whose first shard is `file` is this seat's ([`Model::serves_arch`]).
        fn serves(self, file: &Split) -> bool {
            let arch = file.architecture().unwrap_or("<missing>");
            self.serves_arch(arch, Some(file)) == Some(true)
        }

        /// Whether a file of the `general.architecture` `arch` is this seat's. The qwen3 seat
        /// serves a file by the program that runs its description's layers
        /// ([`bloomery_gpu_gates::qwen_seat::Family::of`]), so it answers `None` without the
        /// first shard `file`; the other seats by the architecture alone. `qwen4exp` is not an
        /// `Arch` name, which is why the seat of such a file is read from the string, and the glm
        /// seat takes both spellings of its architecture through [`Arch::from_name`].
        fn serves_arch(self, arch: &str, file: Option<&Split>) -> Option<bool> {
            Some(match self {
                Model::Ds41 => matches!(arch, "deepseek41" | DEEPSEEK4),
                Model::Qwen38 => arch == "qwen4exp",
                Model::Glm => matches!(Arch::from_name(arch), Ok(Arch::Glm5next)),
                Model::Qwen3 => file.map(bloomery_gpu_gates::qwen_seat::Family::of)?.is_ok(),
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
            })
        }
    }

    /// The flags of the agent contract, taken out before any seat parses the rest.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Agent {
        /// `--json`: the result as one JSON document (`--recommend`, `--dry-run`) or line
        /// (`--plan`) on stdout.
        json: bool,
        /// `--dry-run`: the pick of a start by name, then stop before any download.
        dry_run: bool,
        /// `--recommend`: the pick for every model of the catalog, then stop.
        recommend: bool,
        /// `--revision`: read the picked repos at this revision instead of the rows' commits.
        revision: Option<String>,
    }

    /// `args` with the agent contract's flags taken out, and what they said.
    fn take_agent(args: &[String]) -> Result<(Agent, Vec<String>), GateError> {
        let (valued, rest) = llama_launch::take_serve_args(args, &["--revision"]);
        let mut agent = Agent::default();
        let mut kept = Vec::new();
        for (flag, value) in llama_launch::serve_args(&rest) {
            match flag.as_str() {
                "--json" => agent.json = true,
                "--dry-run" => agent.dry_run = true,
                "--recommend" => agent.recommend = true,
                _ => {
                    kept.push(flag.clone());
                    kept.extend(value.cloned());
                }
            }
        }
        let mut it = valued.iter();
        while it.next().is_some() {
            let v = it
                .next()
                .ok_or_else(|| Refusal::new(Code::Usage, "--revision needs a value"))?;
            agent.revision = Some(v.clone());
        }
        Ok((agent, kept))
    }

    /// What a `--model` value names: a seat's word, a catalog model, or both (a model by its seat
    /// word).
    type Named = (Option<Word>, Option<CatalogRef>);

    /// The `--model` value `v`: a catalog model by its name, alias or seat word with an optional
    /// `:VARIANT` ([`CatalogRef::parse`]), else a seat word. A value that is an existing path is
    /// refused naming `-m`; any other value is refused listing the catalog's names and the seat
    /// words.
    fn name_of(v: &str) -> Result<Named, GateError> {
        if std::path::Path::new(v).exists() {
            return Err(Refusal::new(
                Code::Usage,
                format!(
                    "--model {v} is an existing path: --model takes a catalog model's name or a \
                     seat word, and a model file is -m {v}"
                ),
            )
            .offering(&["-m", v])
            .into());
        }
        match CatalogRef::parse(v) {
            Ok(target) => Ok((Some(Word::parse(target.model.seat)?), Some(target))),
            Err(CatalogError::UnknownModel { names, .. }) => match Word::parse(v) {
                Ok(word) => Ok((Some(word), None)),
                Err(_) => Err(Refusal::new(
                    Code::Usage,
                    format!(
                        "--model {v} is no model of the catalog and no seat word; the catalog's \
                         models: {}; the seat words: {}",
                        names.join(", "),
                        seat_words()
                    ),
                )
                .offering(&["--recommend", "--json"])
                .into()),
            },
            Err(e) => Err(Refusal::new(Code::Usage, e.to_string()).into()),
        }
    }

    /// `args` with every `--model V` pair taken out: the last pair's word and catalog model and
    /// the rest. `--model` with no value is refused by name.
    fn take_model(args: &[String]) -> Result<(Named, Vec<String>), GateError> {
        let mut named: Named = (None, None);
        let (selected, rest) = llama_launch::take_serve_args(args, &["--model"]);
        let mut it = selected.iter();
        while it.next().is_some() {
            let v = it.next().ok_or_else(|| {
                Refusal::new(
                    Code::Usage,
                    format!(
                        "--model needs a value (a catalog model's name, or ds41, qwen38, glm, \
                         qwen3, mimo2, solar2, minimaxm3, nemotron, raon1, musespark, flint or \
                         {}): {USAGE}",
                        serve::decide::WORD
                    ),
                )
            })?;
            named = name_of(v)?;
        }
        Ok((named, rest))
    }

    /// The model source parser receives model options without mistaking other values for flags.
    fn take_source(args: &[String]) -> Result<(hf::source::Flags, Vec<String>), GateError> {
        let (selected, rest) = llama_launch::take_serve_args(args, SOURCE_FLAGS);
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
    /// row's backbone (`qwen35`, Clef's) here has no head: a generative seat
    /// that serves it takes it, saying on stderr what a decision model needs,
    /// and one no seat serves is refused by [`serve::decide::no_head`], which
    /// says what would seat it. Any other file, and a shard that names no architecture, is refused by name
    /// listing the seat words; an explicit seat word that is not the file's seat is
    /// refused naming the file's. A file the Qwen3.5 body runs (Qwen3.6, a dense
    /// `llama` trunk) is the qwen3 seat's.
    fn seat_of_file(word: Option<Word>) -> Result<Model, GateError> {
        let generative = matches!(word, Some(Word::Seat(Model::Qwen3)));
        let path = ref_model_path()?;
        model_file::with_header(&path, |split| {
            let seat = match split.architecture() {
                Some(a) if side_seat::row(a).is_some() => {
                    return Err(Refusal::new(
                        Code::Unsupported,
                        format!(
                            "{} is a {a} file, an embedding model: serve it beside a chat model \
                             with --embd-model",
                            path.display()
                        ),
                    )
                    .into());
                }
                Some(a)
                    if !generative
                        && decide::ROWS.iter().any(|r| r.backbones.contains(&a))
                        && !Model::ALL.iter().any(|m| m.serves(split)) =>
                {
                    return Err(Refusal::new(
                        Code::Unsupported,
                        format!(
                            "{}: {}",
                            path.display(),
                            serve::decide::no_head(a, decide::ROWS)
                        ),
                    )
                    .into());
                }
                Some(a) => {
                    if !generative && decide::ROWS.iter().any(|r| r.backbones.contains(&a)) {
                        eprintln!(
                            "{NAME}: {} is a {a} file with no head: it serves as a generative \
                             model; a decision model takes --head <weights> or the --hf repo \
                             that names its head",
                            path.display()
                        );
                    }
                    let seat = Model::ALL
                        .iter()
                        .copied()
                        .find(|m| m.serves(split))
                        .ok_or_else(|| {
                            Refusal::new(Code::Unsupported, no_seat(&path, split, a))
                                .offering(&["--recommend", "--json"])
                        })?;
                    if let Some(Word::Seat(word)) = word
                        && !word.serves(split)
                    {
                        return Err(Refusal::new(
                            Code::Usage,
                            format!(
                                "--model {} on a {a} file, which serves --model {}",
                                word.word(),
                                seat.word()
                            ),
                        )
                        .offering(&["--model", seat.word()])
                        .into());
                    }
                    seat
                }
                None => {
                    return Err(Refusal::new(
                        Code::Unsupported,
                        format!(
                            "{}: the first shard names no architecture; the seats are --model {}",
                            path.display(),
                            seat_words()
                        ),
                    )
                    .into());
                }
            };
            Ok(seat)
        })?
    }

    /// The seats a start with no `--model` and no `--head` may come to on a file whose
    /// `general.architecture` is `arch` and decision type `decision`, both read before the fetch:
    /// the decide seat alone for a file that carries a decision row's head
    /// ([`serve::decide::pick`]); else the generative seat the architecture names
    /// ([`Model::serves_arch`]), or the seats that take a file by its description, and
    /// beside it the decide seat when the architecture is a row's backbone, which the repo's card
    /// may seat there.
    fn seats_of(arch: &str, decision: Option<&str>) -> Vec<Word> {
        if decision.is_some() || decide::ROWS.iter().any(|r| r.in_file_arch(arch)) {
            return vec![Word::Decide];
        }
        let named = |want| {
            Model::ALL
                .into_iter()
                .filter(move |m| m.serves_arch(arch, None) == want)
        };
        let mut seats: Vec<Word> = match named(Some(true)).next() {
            Some(seat) => vec![Word::Seat(seat)],
            None => named(None).map(Word::Seat).collect(),
        };
        if decide::ROWS.iter().any(|r| r.backbones.contains(&arch)) {
            seats.push(Word::Decide);
        }
        seats
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
            let key = split.arch_key("decision.type");
            let decision = model_file::decision_type(&key, split.value(&key))?.map(str::to_owned);
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

    #[inline(never)]
    pub fn run() -> ExitCode {
        event::enable();
        let args: Vec<String> = std::env::args().skip(1).collect();
        if llama_launch::serve_option(&args, "--version") {
            println!(
                "{NAME} {} (commit {})",
                env!("CARGO_PKG_VERSION"),
                serve::COMMIT
            );
            return ExitCode::SUCCESS;
        }
        // The `--config` file's settings join the command line, which replaces them.
        let args = match serve_config::from_file(NAME, &args) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Usage, e),
        };
        // llama-server's `LLAMA_ARG_MODEL` and `LLAMA_ARG_HF_REPO` name the model unless the
        // command line does.
        let args = match llama_launch::model_from_env(NAME, args, PATH_FLAGS) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Usage, e),
        };
        let (agent, args) = match take_agent(&args) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Usage, e),
        };
        let asked = |flag: &str| llama_launch::serve_option(&args, flag);
        start::set_plan_json(agent.json && asked("--plan"));
        let ((word, name), rest) = match take_model(&args) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Usage, e),
        };
        if word.is_none()
            && name.is_none()
            && !agent.recommend
            && !asked("--head")
            && (args.is_empty() || asked("--help") || asked("-h"))
        {
            // No seat chosen yet; the seat's own usage follows once one is.
            println!("{LEAD}\n{USAGE}\n{MODELS}");
            return ExitCode::SUCCESS;
        }
        if agent.recommend {
            return recommend(&agent, name, &rest);
        }
        if agent.dry_run && name.is_none() {
            return refuse(
                Code::Usage,
                Refusal::new(
                    Code::Usage,
                    "--dry-run picks a catalog model: name one with --model NAME",
                )
                .offering(&["--recommend", "--json"])
                .into(),
            );
        }
        if agent.revision.is_some() && name.is_none() {
            return refuse(
                Code::Usage,
                Refusal::new(
                    Code::Usage,
                    "--revision reads a catalog model's repo at a revision: name the model with \
                     --model NAME, or give a --hf repo its revision as --hf <repo>@<rev>",
                )
                .into(),
            );
        }
        if let Err(e) = early_named(
            word,
            name,
            &rest,
            std::env::vars_os(),
            &model_file::repo_architecture,
        ) {
            // A seat asked for its usage answers with it as its refusal: that is the answer.
            let text = e.to_string();
            if (asked("--help") || asked("-h")) && text.starts_with("usage: ") {
                println!("{text}");
                return ExitCode::SUCCESS;
            }
            return refuse(Code::Usage, e);
        }
        // A driver that cannot run the CUDA 13 code ends the start before any download.
        if let Err(e) = serve::preflight::driver(&start::driver_found()) {
            return refuse(Code::NoDevice, e.into());
        }
        if llama_launch::serve_option(&args, llama_place::LIST_DEVICES) {
            match llama_launch::answer_list_devices(&args) {
                Ok(true) => return ExitCode::SUCCESS,
                Ok(false) => {}
                Err(e) => return refuse(Code::Refused, e),
            }
        }
        eprintln!("{FIRST_START}");
        // The model file is named before any seat parses: `-m`, or a `--hf` set fetched into the
        // cache, or a catalog model's pick turned into one.
        let (mut flags, rest) = match take_source(&rest) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Usage, e),
        };
        flags.name = name;
        let by_name = match hf::source::resolve(&flags, model_file::ref_model().as_deref()) {
            Ok(Some(hf::source::Source::Named(target))) => Some(target),
            Ok(_) => None,
            Err(e) => return refuse(Code::Usage, e.into()),
        };
        let asked_as = by_name.as_ref().map(|target| target.model.name);
        // The listener comes up before the pick and the fetch, so a port that is taken is found
        // first and a client sees the start at once; a run that ends before it serves binds none.
        let serves =
            !(agent.dry_run || asked("--plan") || asked("--levers") || asked("--records-schema"));
        let progress = if serves {
            match bind_start(&rest) {
                Ok(p) => Some(p),
                Err(e) => return refuse(Code::Refused, e),
            }
        } else {
            None
        };
        if let Some(target) = by_name {
            let row = match pick_by_name(&target, &agent, &rest) {
                Ok(Some(row)) => row,
                Ok(None) => return ExitCode::SUCCESS,
                Err(end) => return end,
            };
            flags.name = None;
            flags.hf = Some(row.hf());
            let revision = agent.revision.as_deref().unwrap_or(row.revision);
            if let Err(e) = model_file::pin_revision(row.repo, revision) {
                return refuse(Code::Refused, e);
            }
        }
        if let Some(p) = &progress {
            p.phase(Phase::Fetch);
        }
        if let Err(e) = model_file::name(&flags) {
            return refuse(Code::Refused, e);
        }
        if let Some(p) = &progress {
            p.phase(Phase::Plan);
        }
        if let Ok(path) = ref_model_path()
            && !path.exists()
        {
            let missing = Refusal::new(
                Code::NoModel,
                format!("{}: no such file or folder", path.display()),
            );
            return refuse(Code::NoModel, missing.into());
        }
        // A head names the decide seat; with none the file goes to the
        // generative seats.
        let repo = match flags.hf.as_deref().map(hf::RepoRef::parse).transpose() {
            Ok(r) => r.map(|r| r.id()),
            Err(e) => return refuse(Code::Usage, e.into()),
        };
        let (from, rest) = match head_of(word, repo.as_deref(), &rest) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Refused, e),
        };
        // The side model's flags come out before the seat reads its own; its file is read in the
        // seat, after the levers.
        let (side, rest) = match side_seat::Flags::take(&rest) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Usage, e),
        };
        if let Some(from) = from {
            if let Err(e) = decide::check_side(&side) {
                return refuse(Code::Usage, e);
            }
            // The decision seat places no card, so the placement flags' start does not run for
            // it; the variables and short spellings of the flags it does take are read as the
            // generative seats' are, `LLAMA_API_KEY` among them.
            let rest = match llama_launch::seat_from_env(NAME, rest, llama_launch::SERVE) {
                Ok(v) => v,
                Err(e) => return refuse(Code::Usage, e),
            };
            start::note_seat(serve::decide::WORD);
            return start::ended(NAME, decide::run(&rest, from, repo.as_deref()));
        }
        let seat = match seat_of_file(word) {
            Ok(v) => v,
            Err(e) => return refuse(Code::Refused, e),
        };
        start::note_seat(seat.word());
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
            Err(e) => return refuse(Code::Usage, e),
        };
        let (run, _) = entries(seat);
        start::ended(NAME, run(&served_as(rest, asked_as), &side))
    }

    /// The arguments of a start by name, serving the catalog name it asked by: `--alias NAME`
    /// joins them, since the served name is the file's `general.name`, which a repo's maker may
    /// have left as anything (NVIDIA's Nemotron file says `Hf`). An alias the start gave stays:
    /// the flag, `-a` and `LLAMA_ARG_ALIAS` are all `--alias` once the launcher has read them.
    fn served_as(mut rest: Vec<String>, name: Option<&str>) -> Vec<String> {
        if let Some(name) = name
            && !llama_launch::serve_option(&rest, "--alias")
        {
            rest.extend(["--alias".to_owned(), name.to_owned()]);
        }
        rest
    }

    /// Ends the start with `e`: its stderr lines, its `error` event and the status of its class
    /// (`start::exit_serve`). An error no type names takes the class `unnamed`, which is the
    /// start's phase: its arguments' check is a usage refusal, its driver's a missing device.
    fn refuse(unnamed: Code, e: GateError) -> ExitCode {
        start::exit_serve(NAME, Err(e), unnamed, |_| Vec::new())
    }

    /// The address a start listens on, read as the seats read it (`--host` and `--port`, under
    /// llama-server's variables), bound before the fetch ([`start::bind_early`]): the listener
    /// answers 503 `loading`, and the handle moves its phase and its download's bytes as the start
    /// goes on.
    fn bind_start(rest: &[String]) -> Result<Progress, GateError> {
        let env = llama_launch::llama_env(std::env::vars_os())?;
        let (args, _) = llama_launch::seat_args(rest.to_vec(), &env, llama_launch::SERVE)?;
        let mut host = bloomery_gpu_gates::bind::DEFAULT_HOST.to_owned();
        let mut port = bloomery_gpu_gates::bind::DEFAULT_PORT;
        for (flag, value) in llama_launch::serve_args(&args) {
            match (flag.as_str(), value) {
                ("--host", Some(v)) => host.clone_from(v),
                ("--port", Some(v)) => port = serve::flag::number("--port", v)?,
                _ => {}
            }
        }
        let progress = start::bind_early(&host, port).map_err(ServeError::Io)?;
        progress.phase(Phase::Plan);
        let written = progress.clone();
        model_file::on_transfer(move |done, total| written.bytes(done, total));
        let loading = progress.clone();
        start::on_loading(move || loading.phase(Phase::Load));
        Ok(progress)
    }

    /// What this machine is to a pick: its cards, the class of the stage card, the host's room
    /// and the cache's.
    struct Machine {
        census: Vec<model::placement::workstation::DeviceInfo>,
        class: model::placement::devices::CardClass,
        host_room: u64,
        cache_root: std::path::PathBuf,
        cache_free: u64,
    }

    impl Machine {
        /// The census, the stage card's class (the largest visible card's, which `--place a`
        /// stages on) in this build, the host's room and the cache's free bytes.
        fn read() -> Result<Machine, GateError> {
            let census = gpu_census::census()?;
            let stage = census
                .iter()
                .max_by_key(|d| d.total_bytes)
                .ok_or_else(|| Refusal::new(Code::NoDevice, "no CUDA device is visible"))?;
            let class = stage.card_class(bloomery_gpu::NATIVE_FP4).ok_or_else(|| {
                Refusal::new(
                    Code::NoDevice,
                    format!("{}: the census read no compute capability", stage.name),
                )
            })?;
            let (host_room, _) = model::placement::workstation::host_available_read()?;
            let cache_root = model_file::cache_root()?;
            let cache_free = catalog::cache_free_bytes(&cache_root)?;
            Ok(Machine {
                census,
                class,
                host_room,
                cache_root,
                cache_free,
            })
        }

        /// What the catalog's pick reads of it: the class, the cards' memory together and the
        /// cache's room.
        fn for_pick(&self) -> catalog::Machine {
            catalog::Machine {
                class: self.class,
                card_bytes: self.census.iter().map(|d| d.total_bytes).sum(),
                cache_free_bytes: self.cache_free,
            }
        }

        /// The `machine` object of the JSON contract.
        fn json(&self) -> Value {
            let cards: Vec<Value> = self
                .census
                .iter()
                .map(|d| {
                    json!({
                        "ordinal": d.ordinal,
                        "name": d.name,
                        "cc": d.compute_capability.map(|(major, minor)| format!("{major}.{minor}")),
                        "integrated": d.integrated,
                        "total": d.total_bytes,
                        "free": d.free_bytes,
                    })
                })
                .collect();
            json!({
                "build": {
                    "cuda_target": bloomery_gpu::BUILD_TARGET,
                    "fp4": bloomery_gpu::NATIVE_FP4,
                },
                "class": catalog::class_name(self.class),
                "cards": cards,
                "host": {"room": self.host_room},
                "cache": {"dir": self.cache_root.display().to_string(), "free": self.cache_free},
            })
        }
    }

    /// One model's pick as the lines a person reads: the picked variant and where it goes, or
    /// that nothing fits, then each rejected variant with its reason.
    fn model_text(entry: &Value) -> String {
        let name = entry["name"].as_str().unwrap_or_default();
        let pick = &entry["pick"];
        let mut out = if pick.is_null() {
            format!("{name}: nothing fits")
        } else {
            format!(
                "{name}: {} ({}), place {}, ctx {}, {} bytes to download",
                pick["variant"].as_str().unwrap_or_default(),
                pick["source"].as_str().unwrap_or_default(),
                pick["place"].as_str().unwrap_or_default(),
                pick["ctx"],
                pick["download"],
            )
        };
        for rejected in entry["rejected"].as_array().into_iter().flatten() {
            out.push_str(&format!(
                "\n  {}: {}",
                rejected["variant"].as_str().unwrap_or_default(),
                rejected["why"].as_str().unwrap_or_default()
            ));
        }
        out
    }

    /// The models of `entries` on stdout: the JSON contract's one document (`start` the argument
    /// list a dry run's start would run), or lines for a person.
    fn print_models(json: bool, machine: &Machine, entries: &[Value], start: Option<Vec<String>>) {
        if !json {
            for entry in entries {
                println!("{}", model_text(entry));
            }
            return;
        }
        let mut doc = json!({
            "schema": "bloomery.recommend/1",
            "machine": machine.json(),
            "models": entries,
        });
        if let Some(argv) = start {
            doc["start"] = json!(argv);
        }
        println!("{doc}");
    }

    /// `--recommend`: the pick of every catalog model (or of the one `--model` names), under the
    /// user's flags, printed; the exit is 0 whether or not a model fits (each entry says).
    fn recommend(agent: &Agent, name: Option<CatalogRef>, flags: &[String]) -> ExitCode {
        if let Err(e) = serve::preflight::driver(&start::driver_found()) {
            return refuse(Code::NoDevice, e.into());
        }
        let machine = match Machine::read() {
            Ok(m) => m,
            Err(e) => return refuse(Code::NoDevice, e),
        };
        let targets: Vec<CatalogRef> = match name {
            Some(target) => vec![target],
            None => catalog::models()
                .iter()
                .map(|model| CatalogRef {
                    model,
                    variant: None,
                })
                .collect(),
        };
        let mut entries = Vec::new();
        for target in &targets {
            match choose(target, &machine, agent, flags) {
                Ok(outcome) => entries.push(catalog::model_json(target.model, &outcome)),
                Err(e) => return refuse(Code::Refused, e),
            }
        }
        print_models(agent.json, &machine, &entries, None);
        ExitCode::SUCCESS
    }

    /// The pick of `target` on `machine` under the user's `flags`.
    fn choose(
        target: &CatalogRef,
        machine: &Machine,
        agent: &Agent,
        flags: &[String],
    ) -> Result<Outcome, GateError> {
        model_file::pick::choose(&model_file::pick::Ask {
            target,
            machine: &machine.for_pick(),
            flags,
            revision: agent.revision.as_deref(),
        })
    }

    /// A start by name: the row of `target` that holds on this machine. A real start prints the
    /// `pick` event and goes on with the row; a dry run prints the pick (and what the start would
    /// run) and stops, `Ok(None)`. A pick that finds no row ends the start as the class of its
    /// rejections says ([`no_pick`]).
    fn pick_by_name(
        target: &CatalogRef,
        agent: &Agent,
        flags: &[String],
    ) -> Result<Option<&'static catalog::Row>, ExitCode> {
        let machine = Machine::read().map_err(|e| refuse(Code::NoDevice, e))?;
        let outcome =
            choose(target, &machine, agent, flags).map_err(|e| refuse(Code::Refused, e))?;
        let entry = catalog::model_json(target.model, &outcome);
        if agent.dry_run {
            let start = outcome.picked.as_ref().map(|picked| {
                let mut argv = vec![
                    "--model".to_owned(),
                    format!("{}:{}", target.model.name, picked.row.variant),
                ];
                if let Some(revision) = &agent.revision {
                    argv.extend(["--revision".to_owned(), revision.clone()]);
                }
                argv.extend(flags.iter().cloned());
                argv
            });
            print_models(agent.json, &machine, std::slice::from_ref(&entry), start);
        }
        let Some(picked) = &outcome.picked else {
            return Err(no_pick(target, &outcome, &machine));
        };
        if agent.dry_run {
            return Ok(None);
        }
        let mut fields = catalog::pick_fields(picked);
        fields.insert("model".to_owned(), json!(target.model.name));
        fields.insert("rejected".to_owned(), entry["rejected"].clone());
        event::emit(&Event::Pick(&fields));
        eprintln!(
            "{NAME}: pick: {} is --hf {} at {} ({} bytes to download)",
            target.model.name,
            picked.row.hf(),
            agent.revision.as_deref().unwrap_or(picked.row.revision),
            picked.download_bytes
        );
        Ok(Some(picked.row))
    }

    /// The end of a start whose pick found no row: the class its rejections share
    /// ([`model_file::pick::why_none`]), every rejection with its reason, and, for a variant the
    /// start named that does not fit, the next smaller one.
    fn no_pick(target: &CatalogRef, outcome: &Outcome, machine: &Machine) -> ExitCode {
        use model_file::pick::{Why, why_none};
        let why = why_none(outcome);
        let (code, what) = match why {
            Why::NotInRepo => (Code::NoModel, "no row of the model is in its repo"),
            Why::Offline => (
                Code::Network,
                "no listing could be read and no row of the model is cached",
            ),
            Why::DiskShort => (
                Code::DiskShort,
                "the cache's drive cannot take the least row of the model",
            ),
            Why::NoFit => (
                Code::NoFit,
                "no row of the model fits this machine as asked",
            ),
            Why::PlanRefused(ref word) => (
                Code::from_name(word).unwrap_or(Code::NoFit),
                "the seat's plan refuses every row of the model, and the refusal is not a lack \
                 of room",
            ),
            Why::NamedOnly => (
                Code::Usage,
                "no row of the model starts on its name alone: each is explicit-only or \
                 unverified, so none was planned on this machine; start one by naming its \
                 variant (--model NAME:VARIANT)",
            ),
        };
        let reasons: Vec<String> = outcome
            .rejected
            .iter()
            .map(|r| format!("{}: {}", r.row.variant, r.why))
            .collect();
        let need = outcome
            .least_need_bytes()
            .map_or_else(String::new, |bytes| {
                format!(" (the least plan needs {bytes} bytes)")
            });
        let mut refusal = Refusal::new(
            code,
            format!("{target}: {what}{need}; {}", reasons.join("; ")),
        );
        if why == Why::NamedOnly {
            for rejected in &outcome.rejected {
                let named = format!("{}:{}", target.model.name, rejected.row.variant);
                refusal = refusal.offering(&["--model", &named]);
            }
        }
        if code == Code::NoFit
            && let Some(variant) = target.variant
            && let Some(row) = target.model.row(variant)
            && let Some(smaller) = catalog::next_smaller(target.model, machine.class, row)
        {
            let smaller = format!("{}:{}", target.model.name, smaller.variant);
            refusal = refusal.offering(&["--model", &smaller]);
        }
        refuse(code, refusal.into())
    }

    /// A generative seat's `run`.
    type Run = fn(&[String], &side_seat::Flags) -> Result<ServeError, GateError>;
    /// A generative seat's `check` before the fetch ([`early_named`]).
    type Check = fn(&[String], bool, bool) -> Result<(), GateError>;
    /// A `--hf` repo's model as [`early_named`] reads it: the architecture and the decision type.
    type Held = Result<(String, Option<String>), GateError>;

    /// The entries of `seat`'s module: the start runs the one and [`early_named`] asks the other.
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
    /// flags, and `held` the architecture and decision type of a `--hf` repo's model, which narrow
    /// the seats a start with no seat word may come to ([`any_seat`]).
    fn early_named(
        word: Option<Word>,
        name: Option<CatalogRef>,
        args: &[String],
        vars: impl IntoIterator<Item = (OsString, OsString)>,
        held: &dyn Fn(&hf::RepoRef) -> Held,
    ) -> Result<(), GateError> {
        let asked = |flag: &str| llama_launch::serve_option(args, flag);
        if asked(llama_place::LIST_DEVICES) || asked("--records-schema") {
            return Ok(());
        }
        let (model, rest) = take_source(args)?;
        let model = hf::source::Flags { name, ..model };
        let vars: Vec<_> = vars.into_iter().collect();
        let reference = vars
            .iter()
            .find_map(|(name, value)| {
                (name.as_os_str() == std::ffi::OsStr::new("BLOOMERY_REF_MODEL")).then_some(value)
            })
            .map(|value| value.to_str().ok_or("BLOOMERY_REF_MODEL is not UTF-8"))
            .transpose()?;
        let source = hf::source::resolve(&model, reference)?;
        // A catalog name's pick is a repo.
        let from_repo = matches!(
            source,
            Some(hf::source::Source::Hf(_) | hf::source::Source::Named(_))
        );
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
                return Err(no_model().into());
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
        let seats = || match &source {
            Some(hf::source::Source::Hf(r)) => held(r).ok().map(|(arch, decision)| {
                let seats = seats_of(&arch, decision.as_deref());
                (format!("--hf {r} holds a {arch} model"), seats)
            }),
            _ => None,
        };
        match word {
            Some(word) => check(word),
            None => any_seat(check, seats),
        }?;
        if source.is_none() {
            return Err(no_model().into());
        }
        Ok(())
    }

    /// The refusal when nothing names the model: the class `no-model`, with what lists the models.
    fn no_model() -> Refusal {
        Refusal::new(Code::NoModel, model_file::NONE).offering(&["--recommend", "--json"])
    }

    /// [`early_named`] for a start that names no catalog model.
    #[cfg(test)]
    fn early(
        word: Option<Word>,
        args: &[String],
        vars: impl IntoIterator<Item = (OsString, OsString)>,
        held: &dyn Fn(&hf::RepoRef) -> Held,
    ) -> Result<(), GateError> {
        early_named(word, None, args, vars, held)
    }

    /// `check` at every seat, for a start whose model file picks the seat ([`early_named`]): `Ok` when
    /// every seat takes the arguments. When only some do, `held` names the model and the seats it
    /// may come to ([`seats_of`]), and the start ends unless one of those takes them; a model
    /// `held` cannot name before the fetch (no `--hf`, no listing, a set it cannot read) leaves
    /// the fetch to go on, and the seat the file picks reads the arguments after it. Each refusal
    /// is said once, with the seats that give it.
    fn any_seat(
        check: impl Fn(Word) -> Result<(), GateError>,
        held: impl FnOnce() -> Option<(String, Vec<Word>)>,
    ) -> Result<(), GateError> {
        let words = Model::ALL.iter().map(|&m| Word::Seat(m));
        let refused: Vec<Refused> = words
            .chain([Word::Decide])
            .filter_map(|word| {
                check(word)
                    .err()
                    .map(|e| (word, start::class_of(&e), e.to_string()))
            })
            .collect();
        if refused.is_empty() {
            return Ok(());
        }
        if refused.len() == Model::ALL.len() + 1 {
            return Err(each_refusal("no seat takes these arguments", &refused));
        }
        let Some((model, seats)) = held() else {
            return Ok(());
        };
        let refused: Vec<Refused> = refused
            .into_iter()
            .filter(|(word, ..)| seats.contains(word))
            .collect();
        if refused.len() < seats.len() {
            return Ok(());
        }
        Err(each_refusal(
            &format!("{model}, and no seat that may serve it takes these arguments"),
            &refused,
        ))
    }

    /// A seat's refusal of the arguments: the seat, the class the refusal names, its text.
    type Refused = (Word, Option<Code>, String);

    /// `head`, then each refusal of `refused` once, with the seats that give it. The refusal is of
    /// the class the seats' refusals name when they name one (a lever's value is refused as a
    /// lever at every seat), else of no class: the start's phase gives it one.
    fn each_refusal(head: &str, refused: &[Refused]) -> GateError {
        let mut grouped: Vec<(&str, Vec<&str>)> = Vec::new();
        let mut named: Vec<Code> = Vec::new();
        for (word, class, e) in refused {
            named.extend(*class);
            match grouped.iter_mut().find(|(text, _)| text == e) {
                Some((_, seats)) => seats.push(word.word()),
                None => grouped.push((e, vec![word.word()])),
            }
        }
        let lines: Vec<String> = grouped
            .iter()
            .map(|(e, seats)| format!("  --model {}: {e}", seats.join(", ")))
            .collect();
        let text = format!(
            "{head}, each refusing them as follows (--model names the seat):\n{}",
            lines.join("\n")
        );
        match named.first() {
            Some(&code) if named.iter().all(|c| *c == code) => Refusal::new(code, text).into(),
            _ => text.into(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn line(s: &str) -> Vec<String> {
            s.split_whitespace().map(str::to_owned).collect()
        }

        /// The `--hf` repo's model of a start where none can be read before the fetch: the tests
        /// reach no hub.
        fn no_hub(r: &hf::RepoRef) -> Held {
            Err(format!("{r}: no hub").into())
        }

        /// [`super::early`] with [`no_hub`].
        fn early(
            word: Option<Word>,
            args: &[String],
            vars: impl IntoIterator<Item = (OsString, OsString)>,
        ) -> Result<(), GateError> {
            super::early(word, args, vars, &no_hub)
        }

        /// A start with no seat word on a `--hf` repo whose model is read before the fetch
        /// ([`seats_of`]): arguments some seats refuse end the start when the seats the model
        /// may come to all refuse them, the model is not read for arguments every seat takes, and
        /// a model that cannot be read leaves them to the seat after the fetch.
        #[test]
        fn a_repo_s_model_picks_the_seats_that_check_before_the_fetch() {
            let held = |arch: &'static str, decision: Option<&'static str>| {
                move |_: &hf::RepoRef| -> Held {
                    Ok((arch.to_owned(), decision.map(str::to_owned)))
                }
            };
            let start =
                |arch, decision, s: &str| super::early(None, &line(s), [], &held(arch, decision));
            for (arch, s, why) in [
                ("qwen4exp", "--hf o/r --prefill steps", "--model qwen38: "),
                (
                    "nemotron_h",
                    "--hf o/r --parallel 2",
                    "--model nemotron: --parallel 2: the nemotron body holds one sequence",
                ),
            ] {
                let e = start(arch, None, s).expect_err(s).to_string();
                assert!(
                    e.starts_with(&format!("--hf o/r holds a {arch} model")),
                    "{s}: {e}"
                );
                assert!(e.contains(why), "{s}: {e}");
                assert_eq!(e.lines().count(), 2, "{s}: one seat refuses: {e}");
                early(None, &line(s), []).expect("an unread model leaves the fetch to go on");
            }
            start("glm5next", None, "--hf o/r --prefill steps").expect("the glm seat takes it");
            start("nemotron_h", None, "--hf o/r --parallel 1").expect("one sequence");
            let backbone = start("qwen35", None, "--hf o/r --prefill steps")
                .expect_err("both the generative and card-selected decision seats refuse steps")
                .to_string();
            assert!(backbone.contains("--model qwen3: "), "{backbone}");
            assert!(backbone.contains("--model decide: "), "{backbone}");
            let decision = start("qwen35", Some("lev"), "--hf o/r --prefill steps")
                .expect_err("a file with a decision type is the decide seat's")
                .to_string();
            assert!(
                decision.lines().nth(1).is_some_and(|l| l
                    .trim_start()
                    .starts_with(&format!("--model {}: ", serve::decide::WORD))),
                "{decision}"
            );
            super::early(None, &line("--hf o/r --ctx-size 8192"), [], &|_| {
                panic!("arguments every seat takes need no model")
            })
            .expect("every seat takes it");
        }

        /// A start by name serves the catalog name it was asked by; an alias the start gave, as the
        /// flag or through the variable the launcher has already read, and a start that named no
        /// catalog model, are left as they are.
        #[test]
        fn a_start_by_name_serves_the_name_it_was_asked_by() {
            let rest = line("--host 127.0.0.1 --port 0");
            assert_eq!(
                served_as(rest.clone(), Some("nemotron-3-nano-4b")),
                line("--host 127.0.0.1 --port 0 --alias nemotron-3-nano-4b")
            );
            for given in ["--alias mine --port 0", "--port 0 --alias mine"] {
                let given = line(given);
                assert_eq!(served_as(given.clone(), Some("glm-5.3-flash")), given);
            }
            assert_eq!(served_as(rest.clone(), None), rest);
            // A value that spells `--alias` is the value of another flag, not an alias.
            let value = line("--api-key --alias");
            assert_eq!(
                served_as(value, Some("glm-5.3-flash")),
                line("--api-key --alias --alias glm-5.3-flash")
            );
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

        /// The code a refusal of the start path ends with: a [`Refusal`] in the error.
        fn code_of(e: &GateError) -> Code {
            e.downcast_ref::<Refusal>()
                .map(|r| r.code)
                .expect("a typed refusal")
        }

        /// The agent contract's flags come out wherever they stand and take no value of their
        /// own: a value that spells one (an alias named `--json`) is left to its flag.
        #[test]
        fn the_agent_flags_come_out_of_the_arguments() {
            let (agent, rest) = take_agent(&line(
                "--json --model glm --alias --json --dry-run --revision abc --ctx-size 8",
            ))
            .expect("flags");
            assert_eq!(
                agent,
                Agent {
                    json: true,
                    dry_run: true,
                    recommend: false,
                    revision: Some("abc".to_owned()),
                }
            );
            assert_eq!(rest, line("--model glm --alias --json --ctx-size 8"));
            let (agent, rest) = take_agent(&line("--recommend")).expect("flags");
            assert!(agent.recommend && rest.is_empty());
            let e = take_agent(&line("--revision")).expect_err("no value");
            assert_eq!(code_of(&e), Code::Usage);
        }

        /// A `--model` value is a catalog model by name, alias or seat word (with its variant),
        /// or a seat word the catalog has no model for; an unknown name lists both sets, and an
        /// existing path is `-m`'s.
        /// A start whose file picks the seat refuses as the class the seats' refusals name: a lever's
        /// bad value is the class `lever` whichever seats read it, a seat whose refusal names no class
        /// does not hide it, and refusals no seat classes stay unclassed.
        #[test]
        fn every_seat_refusing_in_one_class_ends_the_start_as_that_class() {
            let lever =
                || -> GateError { Refusal::new(Code::Lever, "BLOOMERY_X=\"b\" is refused").into() };
            let err = any_seat(|_| Err(lever()), || None).expect_err("every seat refuses");
            assert_eq!(start::class_of(&err), Some(Code::Lever));
            assert!(err.to_string().contains("BLOOMERY_X"), "{err}");
            let mixed = |w: Word| match w {
                Word::Decide => Err("a string refusal".into()),
                Word::Seat(_) => Err(lever()),
            };
            let err = any_seat(mixed, || None).expect_err("every seat refuses");
            assert_eq!(start::class_of(&err), Some(Code::Lever));
            let two = |w: Word| match w {
                Word::Decide => Err(Refusal::new(Code::NoFit, "no fit").into()),
                Word::Seat(_) => Err(lever()),
            };
            let err = any_seat(two, || None).expect_err("every seat refuses");
            assert_eq!(start::class_of(&err), None, "two classes name no one");
            let plain =
                any_seat(|_| Err("--ctx-size \"abc\"".into()), || None).expect_err("refused");
            assert_eq!(start::class_of(&plain), None);
            let one = |w: Word| match w {
                Word::Seat(Model::Glm) => Ok(()),
                _ => Err("not this seat".into()),
            };
            assert!(any_seat(one, || None).is_ok());
        }

        /// A start whose `--hf` repo's model narrows the seats ends as the class those seats'
        /// refusals name, as one whose every seat refuses does.
        #[test]
        fn a_models_seats_refusing_in_one_class_end_the_start_as_that_class() {
            let glm = |w: Word| match w {
                Word::Seat(Model::Glm) => {
                    Err(Refusal::new(Code::Lever, "BLOOMERY_X=\"b\" is refused").into())
                }
                _ => Ok(()),
            };
            let held = || {
                Some((
                    "--hf o/r holds a glm5next model".to_owned(),
                    vec![Word::Seat(Model::Glm)],
                ))
            };
            let err = any_seat(glm, held).expect_err("the seat the model comes to refuses");
            assert_eq!(start::class_of(&err), Some(Code::Lever));
            assert!(
                err.to_string()
                    .starts_with("--hf o/r holds a glm5next model"),
                "{err}"
            );
        }

        #[test]
        fn a_model_value_is_a_catalog_name_or_a_seat_word() {
            let seat = |word: &str| Some(Word::parse(word).expect("a seat word"));
            let (word, target) = name_of("glm").expect("a seat word that is a model's alias");
            assert_eq!(
                (word, target.map(|t| t.model.name)),
                (seat("glm"), Some("glm-5.3-flash"))
            );
            let (word, target) = name_of("qwen3.8-27b:nvfp4").expect("a model and its variant");
            assert_eq!(word, seat("qwen3"));
            assert_eq!(target.and_then(|t| t.variant), Some("nvfp4"));
            let (word, target) = name_of("decide").expect("a seat word with no model");
            assert_eq!((word, target), (Some(Word::Decide), None));
            let e = name_of("no-such-model").expect_err("unknown");
            assert_eq!(code_of(&e), Code::Usage);
            let text = e.to_string();
            assert!(
                text.contains("glm-5.3-flash") && text.contains("musespark"),
                "{text}"
            );
            let e = name_of("glm:no-such-variant").expect_err("unknown variant");
            assert_eq!(code_of(&e), Code::Usage);
            let dir =
                std::env::temp_dir().join(format!("bloomery-serve-model-{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("dir");
            let path = dir.to_str().expect("utf8");
            let e = name_of(path).expect_err("a path");
            assert_eq!(code_of(&e), Code::Usage);
            let refusal = e.downcast_ref::<Refusal>().expect("typed");
            assert_eq!(refusal.next, vec![vec!["-m".to_owned(), path.to_owned()]]);
            std::fs::remove_dir_all(dir).expect("clean");
        }

        /// A catalog name alone names the model (the pick turns it into a repo), so the start's
        /// check passes where a seat word alone is "no model file"; a variant beside `-m` is the
        /// model named twice, and nothing named is `no-model`.
        #[test]
        fn a_catalog_name_names_the_model_before_the_fetch() {
            let (word, name) = name_of("glm").expect("name");
            early_named(word, name, &[], [], &no_hub).expect("a name is a model");
            let (word, _) = name_of("glm").expect("name");
            let e = early_named(word, None, &[], [], &no_hub).expect_err("a word alone");
            assert_eq!(code_of(&e), Code::NoModel);
            let (word, name) = name_of("glm:Q8_0").expect("name");
            let e = early_named(word, name, &line("-m /x.gguf"), [], &no_hub).expect_err("twice");
            assert!(e.to_string().contains("named twice"), "{e}");
            let (word, name) = name_of("glm").expect("name");
            early_named(word, name, &line("-m /x.gguf"), [], &no_hub)
                .expect("a name beside -m picks the seat");
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
            for (seat, body) in [
                (Model::Nemotron, "nemotron"),
                (Model::Raon1, "raon1"),
                (Model::MuseSpark, "musespark"),
            ] {
                let e = refusal(Some(Word::Seat(seat)), "--hf o/r --parallel 2");
                assert!(
                    e.contains(&format!("--parallel 2: the {body} body holds one sequence")),
                    "{e}"
                );
            }
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
