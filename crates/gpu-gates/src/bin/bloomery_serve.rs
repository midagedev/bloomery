//! `bloomery-serve` — one server binary, one seat a model: `--model
//! ds41|qwen38|glm|qwen3` picks the seat (`serve_seats`, the shared seat
//! modules), every other flag the seat's own, parsed by the seat's own
//! parser, so each seat's flag set is exactly its per-model binary's.
//!
//!     bloomery-serve [--model ds41|qwen38|glm|qwen3]
//!                    [-m PATH | --model-file PATH | --hf <repo>[:<quant>]]
//!                    <the seat's own flags>
//!     bloomery-serve --version
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
//! With no `--model`, the architecture of the file's first shard picks the
//! seat: deepseek41 and deepseek4 the ds41 seat, qwen4exp the qwen38 seat,
//! qwen3moe and qwen35moe the qwen3 seat, glm5next the glm seat; a qwen35
//! file (Clef's backbone) is refused naming `bloomery_serve_clef`, which
//! serves it with its head, and a file whose architecture does not match an
//! explicit `--model` is refused by name before any load. `--version` prints this crate's version and the
//! build's commit and exits. Each seat's flags, records, `/props` fields
//! and exit codes are its module's doc; `bloomery-serve-ds41` and
//! `bloomery-serve-qwen38` are this binary's ds41 and qwen38 seats alone.

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
    use model::arch::Arch;
    use model::arch::DEEPSEEK4;
    use serve::ServeError;

    use crate::serve_seats::{ds41, glm, qwen3, qwen38};

    const NAME: &str = "bloomery-serve";

    const USAGE: &str = "usage: bloomery-serve [--model ds41|qwen38|glm|qwen3] [-m PATH | \
                         --model-file PATH | --hf <repo>[:<quant>]] <the chosen seat's own \
                         flags, as bloomery-serve-ds41, bloomery-serve-qwen38, the glm or the \
                         qwen3 seat takes them> | --version";

    /// The spellings of the model path flag.
    const PATH_FLAGS: &[&str] = &["-m", "--model-file"];

    /// The seats, one a `--model` word.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Model {
        Ds41,
        Qwen38,
        Glm,
        Qwen3,
    }

    impl Model {
        fn parse(v: &str) -> Result<Model, GateError> {
            match v {
                "ds41" => Ok(Model::Ds41),
                "qwen38" => Ok(Model::Qwen38),
                "glm" => Ok(Model::Glm),
                "qwen3" => Ok(Model::Qwen3),
                other => Err(format!("--model is ds41, qwen38, glm or qwen3, not {other}").into()),
            }
        }

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
        /// string) is this seat's. `qwen4exp` is not an [`Arch`] name, which
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
    fn take_model(args: &[String]) -> Result<(Option<Model>, Vec<String>), GateError> {
        let mut word = None;
        let mut rest = Vec::with_capacity(args.len());
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if flag != "--model" {
                rest.push(flag.clone());
                continue;
            }
            let v = it.next().ok_or_else(|| {
                format!("--model needs a value (ds41, qwen38, glm or qwen3): {USAGE}")
            })?;
            word = Some(Model::parse(v)?);
        }
        Ok((word, rest))
    }

    /// The seat the model file's ([`ref_model_path`]) first shard holds,
    /// with its `general.architecture` string: `qwen4exp` by its name (which
    /// [`Arch`] does not list), else [`Arch::detect`]'s. Any other file is
    /// refused by name — this server serves four seats; a `qwen35` file
    /// (Clef's backbone) naming the binary that serves it,
    /// `bloomery_serve_clef`. A qwen35moe file (Qwen3.6) is the qwen3 seat's:
    /// the qwen38 seat's body is qwen4exp's, whose plan refuses its geometry
    /// by name before the server listens.
    fn seat_of_file() -> Result<(Model, String), GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let arch = split.architecture().map(str::to_owned);
        let seat = match arch.as_deref() {
            Some("qwen4exp") => Model::Qwen38,
            Some("qwen35") => {
                return Err(format!(
                    "{} is a qwen35 file, Clef's backbone, which {NAME} does not serve: \
                     bloomery_serve_clef serves it with Clef's joint schema head \
                     (bloomery_serve_clef --model <gguf> --head <joint_head.safetensors>, or \
                     --hf <repo>[:<quant>], which fetches the head)",
                    path.display()
                )
                .into());
            }
            _ => {
                let first = split
                    .shard(0)
                    .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
                match Arch::detect(first) {
                    Ok(Arch::Deepseek41) => Model::Ds41,
                    Ok(Arch::Glm5next) => Model::Glm,
                    Ok(Arch::Qwen3moe | Arch::Qwen35moe) => Model::Qwen3,
                    Ok(other) => {
                        return Err(format!(
                            "{} is a {} file; {NAME} serves deepseek41, {DEEPSEEK4}, qwen4exp, \
                             glm5next, qwen3moe and qwen35moe files",
                            path.display(),
                            other.name()
                        )
                        .into());
                    }
                    Err(e) => {
                        return Err(format!("{}: {e}", path.display()).into());
                    }
                }
            }
        };
        Ok((seat, arch.unwrap_or_else(|| "<missing>".to_owned())))
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
        if word.is_none() && args.iter().any(|a| a == "--help" || a == "-h") {
            // No seat chosen yet; the seat's own usage follows once one is.
            return exit_with(NAME, Err(USAGE.into()));
        }
        // The model file is named before any seat parses: `-m`, or a `--hf`
        // set fetched into the cache.
        let rest = match model_file::from_args(&rest, PATH_FLAGS) {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        let (seat, arch) = match seat_of_file() {
            Ok(v) => v,
            Err(e) => return exit_with(NAME, Err(e)),
        };
        if let Some(word) = word
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
