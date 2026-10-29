//! A launcher's start with llama-server's placement flags and variables: the one owner every
//! launcher calls, so none carries a copy. [`crate::llama_place`] says what the flags and the
//! `LLAMA_ARG_*` variables come to; this module reads the process for them and hands the result
//! to the binary. [`Launcher::placed`] takes an argument list a server holds already;
//! [`Launcher::start`] takes the process's own and installs what is left of it for the binary's
//! parsers to read through [`args`] or [`args_os`] (a binary that looks its arguments up by name
//! has no list to pass down).
//!
//! A start reads the `LLAMA_ARG_*` variables of the environment, and `LLAMA_API_KEY`
//! ([`llama_env`]), and cuts the flags and variables out of the arguments. The variables of the
//! other flags a launcher's seat takes under llama's spelling are carried as those flags unless
//! the command line gives them ([`Reads`], [`Meet`]), the short spellings of those flags are
//! respelled as the flags ([`respell`]), and a variable bloomery does not read has a line saying
//! so. [`resolve`] says what they come to on the launcher's seat: the levers they are, handed
//! over in place of the environment's before the binary reads its own
//! (`bloomery_levers::override_with`), the `--place` word the cards come to, appended to the
//! arguments for the seat to parse, and a line for each reading, printed on stderr. A flag the seat
//! has no thing for is refused by name. `--list-devices` prints the census as llama-server does and
//! ends the start ([`answer_list_devices`]).

use std::ffi::{OsStr, OsString};
#[cfg(feature = "gpu")]
use std::sync::OnceLock;

use model::placement::workstation::DeviceInfo;

use crate::GateError;
#[cfg(feature = "gpu")]
use crate::generate::SEAT_STEPS;
use crate::llama_place::{self, Flags, Target};
#[cfg(feature = "gpu")]
use crate::llama_place::{LIST_DEVICES, take_with_env};

/// Whether a serve option consumes its following value. The seat table owns its spellings.
pub(crate) fn serve_takes_value(flag: &str) -> bool {
    if let Some(read) = SEAT_FLAGS.iter().find(|read| read.spelled.contains(&flag)) {
        return read.flag != "--mmproj" || flag != "--no-mmproj";
    }
    !matches!(
        flag,
        "--help"
            | "-h"
            | "--version"
            | "--plan"
            | "--embd-no-mmproj"
            | llama_place::LIST_DEVICES
            | "--records-schema"
            | "--levers"
            | "--recommend"
            | "--dry-run"
            | "--json"
    ) && !llama_place::CPU_MOE.contains(&llama_place::spelled(flag).as_str())
}

/// Serve options paired with their values, leaving a value's spelling untouched.
pub fn serve_args(args: &[String]) -> impl Iterator<Item = (&String, Option<&String>)> {
    let mut it = args.iter();
    std::iter::from_fn(move || {
        let flag = it.next()?;
        let value = if serve_takes_value(flag) {
            it.next()
        } else {
            None
        };
        Some((flag, value))
    })
}

/// Whether `wanted` occurs as a serve option, skipping values of other options.
pub fn serve_option(args: &[String], wanted: &str) -> bool {
    serve_args(args).any(|(flag, _)| flag == wanted)
}

/// Separates the named serve options and their values from all other option-value pairs.
pub fn take_serve_args(args: &[String], flags: &[&str]) -> (Vec<String>, Vec<String>) {
    let (mut selected, mut rest) = (Vec::new(), Vec::new());
    for (flag, value) in serve_args(args) {
        let out = if flags.contains(&flag.as_str()) {
            &mut selected
        } else {
            &mut rest
        };
        out.push(flag.clone());
        out.extend(value.cloned());
    }
    (selected, rest)
}

/// The prefix of llama-server's environment variables.
const PREFIX: &str = "LLAMA_ARG_";

/// The one variable a server seat reads that has no [`PREFIX`]: `--api-key`'s.
const API_KEY: &str = "LLAMA_API_KEY";

/// A flag a seat takes under llama's spelling and the variable llama-server reads it from.
#[derive(Clone, Copy, Debug)]
pub struct EnvFlag {
    /// The spelling the seat's parser is handed.
    pub flag: &'static str,
    /// Every spelling a command line may give it: one of them gives the flag, which [`Meet`] says
    /// the variable's value does then.
    pub spelled: &'static [&'static str],
    /// llama-server's variable: the `.set_env("...")` of the flag's `add_opt` in `common/arg.cpp`
    /// of ggml-org/llama.cpp at commit f5027fb, the line being that `.set_env`'s.
    pub var: &'static str,
    /// What the variable's value does when the command line gives the flag too.
    pub meet: Meet,
}

/// What a variable's value does beside the same flag on the command line. llama-server reads the
/// variables first and the command line after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meet {
    /// The command line's flag replaces the variable: a flag that holds one value.
    Replace,
    /// Both count: a flag that adds each value it is given to a set, the API keys.
    Add,
    /// [`Meet::Add`] for a secret: the line that says the variable was read names the flag and
    /// never the value.
    AddSecret,
}

/// The other variables a launcher reads: the flags its seat takes under llama's spelling, and
/// whether it takes the model by `LLAMA_ARG_MODEL` or `LLAMA_ARG_HF_REPO` ([`model_args`]).
#[derive(Clone, Copy, Debug)]
pub struct Reads {
    pub flags: &'static [EnvFlag],
    pub model: bool,
}

/// A launcher whose seat takes none of them: its other variables each have a line saying so.
pub const NONE: Reads = Reads {
    flags: &[],
    model: false,
};

/// A server seat on its own: the flags of [`SEAT_FLAGS`].
pub const SEAT: Reads = Reads {
    flags: SEAT_FLAGS,
    model: false,
};

/// The server with a model flag of its own: [`SEAT`] and the model by variable.
pub const SERVE: Reads = Reads {
    flags: SEAT_FLAGS,
    model: true,
};

/// The flags the server seats take under llama's spelling, each with its variable. A seat that
/// takes none of a flag refuses it by name after the line that says which variable it came from.
pub const SEAT_FLAGS: &[EnvFlag] = &[
    EnvFlag {
        flag: "--ctx-size",
        spelled: &["-c", "--ctx-size", "--ctx"],
        var: "LLAMA_ARG_CTX_SIZE", // line 1651
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--ubatch-size",
        spelled: &["-ub", "--ubatch-size"],
        var: "LLAMA_ARG_UBATCH", // line 1684
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--cache-ram",
        spelled: &["-cram", "--cache-ram"],
        var: "LLAMA_ARG_CACHE_RAM", // line 1726
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--mmproj",
        spelled: &["-mm", "--mmproj", "--no-mmproj"],
        var: "LLAMA_ARG_MMPROJ", // line 2588
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--image-min-tokens",
        spelled: &["--image-min-tokens"],
        var: "LLAMA_ARG_IMAGE_MIN_TOKENS", // line 2647
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--image-max-tokens",
        spelled: &["--image-max-tokens"],
        var: "LLAMA_ARG_IMAGE_MAX_TOKENS", // line 2654
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--parallel",
        spelled: &["-np", "--parallel"],
        var: "LLAMA_ARG_N_PARALLEL", // line 2556
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--alias",
        spelled: &["-a", "--alias"],
        var: "LLAMA_ARG_ALIAS", // line 3035
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--host",
        spelled: &["--host"],
        var: "LLAMA_ARG_HOST", // line 3349
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--port",
        spelled: &["--port"],
        var: "LLAMA_ARG_PORT", // line 3356
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--api-key",
        spelled: &["--api-key"],
        var: API_KEY, // line 3542
        meet: Meet::AddSecret,
    },
    EnvFlag {
        flag: "--api-key-file",
        spelled: &["--api-key-file"],
        var: "LLAMA_ARG_API_KEY_FILE", // line 3559
        meet: Meet::Add,
    },
    EnvFlag {
        flag: "--slot-save-path",
        spelled: &["--slot-save-path"],
        var: "LLAMA_ARG_SLOT_SAVE_PATH", // line 3667
        meet: Meet::Replace,
    },
    EnvFlag {
        flag: "--chat-template-file",
        spelled: &["--chat-template-file"],
        var: "LLAMA_ARG_CHAT_TEMPLATE_FILE", // line 3813
        meet: Meet::Replace,
    },
];

/// The variables that name the model: `-m` (line 3056) and `-hf` (line 3082).
const MODEL: &str = "LLAMA_ARG_MODEL";
const HF_REPO: &str = "LLAMA_ARG_HF_REPO";

/// What a launcher says about itself to the flags: the name its lines carry, its seat's word as
/// [`Target::seat`] takes it, and how to read the model file's layer count, which a flag compares
/// only when it is given ([`Flags::needs_layers`]).
#[cfg(feature = "gpu")]
pub struct Launcher<'a> {
    /// The binary's name, which prefixes every line the start prints.
    pub bin: &'a str,
    /// The seat's word, asked for once a flag or variable is present.
    pub seat: &'a dyn Fn() -> Result<&'static str, GateError>,
    /// The other variables the seat reads ([`NONE`], [`SEAT`], [`SERVE`]).
    pub reads: Reads,
    /// The file's `block_count`, `None` where it states none ([`file_layers`] reads it from the
    /// model file).
    pub layers: &'a dyn Fn() -> Result<Option<u64>, GateError>,
}

/// The arguments [`Launcher::start`] left, which [`args`] and [`args_os`] read: the process's
/// reading of its own command line, made once before anything parses it.
#[cfg(feature = "gpu")]
static ARGS: OnceLock<Vec<OsString>> = OnceLock::new();

#[cfg(feature = "gpu")]
impl Launcher<'_> {
    /// `args` without llama-server's placement flags and variables, and with the cards they name
    /// appended as the `--place` word the seat parses. The levers they are replace the
    /// environment's once, so this runs once a process, before any lever is read; a line on stderr
    /// says each reading, and one with none is returned as it is.
    pub fn placed<A>(&self, args: &[A]) -> Result<Vec<A>, GateError>
    where
        A: AsRef<OsStr> + Clone + From<String>,
    {
        let env = llama_env(std::env::vars_os())?;
        let (flags, mut rest, notes) = seat_view(args, &env, self.reads)?;
        self.say(&notes);
        if flags.is_empty() {
            return Ok(rest);
        }
        let target = Target {
            seat: (self.seat)()?,
            layers: if flags.needs_layers() {
                (self.layers)()?
            } else {
                None
            },
            ctx_floor: SEAT_STEPS.floor,
        };
        let resolved = resolve(&flags, &target, crate::gpu_census::census)?;
        self.say(&resolved.notes);
        if let Some(word) = resolved.word {
            rest.push(A::from("--place".to_owned()));
            rest.push(A::from(word));
        }
        let replaced = bloomery_levers::override_with(&resolved.set).map_err(|e| e.to_string())?;
        self.say(&replaced);
        Ok(rest)
    }

    /// The start of a binary whose parsers read the process's arguments: `Ok(true)` when
    /// `--list-devices` was answered and the binary ends; else the arguments less the flags and
    /// variables ([`Launcher::placed`]) are installed, argv[0] first, for [`args`] and [`args_os`].
    pub fn start(&self) -> Result<bool, GateError> {
        let mut all = std::env::args_os();
        let argv0 = all.next().unwrap_or_default();
        let given: Vec<OsString> = all.collect();
        if answer_list_devices(&given)? {
            return Ok(true);
        }
        let mut kept = vec![argv0];
        kept.extend(self.placed(&given)?);
        ARGS.set(kept)
            .map_err(|_| "llama_launch: the process's arguments were installed twice")?;
        Ok(false)
    }

    fn say(&self, lines: &[String]) {
        for line in lines {
            eprintln!("{}: {line}", self.bin);
        }
    }
}

/// `args` as a seat's parser reads them before [`Launcher::placed`] appends the `--place` word the
/// cards come to: llama-server's placement flags and the variables of `env` that are theirs taken
/// out ([`take_with_env`]) and returned, the other variables carried and the short spellings
/// respelled ([`seat_args`]), and the lines that say each reading, not printed. It reads no device.
#[cfg(feature = "gpu")]
pub fn seat_view<A>(
    args: &[A],
    env: &[(String, String)],
    reads: Reads,
) -> Result<(Flags, Vec<A>, Vec<String>), GateError>
where
    A: AsRef<OsStr> + Clone + From<String>,
{
    let (flags, rest, mut notes) = take_with_env(args, env, reads.flags)?;
    let (rest, more) = seat_args(rest, env, reads)?;
    notes.extend(more);
    Ok((flags, rest, notes))
}

/// The process's arguments as [`Launcher::start`] left them, argv[0] first. A binary that did not
/// start through it has none installed, and reading them is a bug a panic names.
#[cfg(feature = "gpu")]
pub fn args_os() -> Vec<OsString> {
    ARGS.get()
        .expect("a launcher reads its arguments through llama_launch after Launcher::start")
        .clone()
}

/// [`args_os`] as text, as `std::env::args` gives it: an argument that is no Unicode panics.
#[cfg(feature = "gpu")]
pub fn args() -> Vec<String> {
    args_os()
        .into_iter()
        .map(|a| {
            a.into_string()
                .unwrap_or_else(|a| panic!("argument {a:?} is not valid Unicode"))
        })
        .collect()
}

/// `--list-devices` among `args`: the census printed as llama-server prints it, and `true` for the
/// caller to end; `false` when it was not asked.
#[cfg(feature = "gpu")]
pub fn answer_list_devices<A: AsRef<OsStr>>(args: &[A]) -> Result<bool, GateError> {
    if !args
        .iter()
        .any(|a| a.as_ref().to_str() == Some(LIST_DEVICES))
    {
        return Ok(false);
    }
    let census = crate::gpu_census::census()?;
    print!("{}", crate::llama_place::list_devices(&census));
    Ok(true)
}

/// The model's layer count (`block_count`), which `-ngl N` and `-ncmoe N` are compared with: the
/// GGUF file's, or the header a checkpoint folder's map gives ([`crate::model_file::header`]);
/// `None` where the header states none.
pub fn file_layers() -> Result<Option<u64>, GateError> {
    let split = crate::model_file::header(&crate::ref_model_path()?)?;
    Ok(split.arch_get_u64("block_count"))
}

/// The `LLAMA_ARG_*` variables of `vars` and `LLAMA_API_KEY` as text. A name of another kind is
/// left alone; a value that is no UTF-8 is refused under its name (the value of a key is not
/// printed).
pub fn llama_env(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> Result<Vec<(String, String)>, GateError> {
    let mut out = Vec::new();
    for (name, value) in vars {
        let Some(name) = name
            .to_str()
            .filter(|n| n.starts_with(PREFIX) || *n == API_KEY)
        else {
            continue;
        };
        let value = value.into_string().map_err(|v| {
            if name == API_KEY {
                format!("{name}: the value is not UTF-8")
            } else {
                format!("{name}: the value {v:?} is not UTF-8")
            }
        })?;
        out.push((name.to_owned(), value));
    }
    Ok(out)
}

/// What the variables of `env` that are no placement flag's come to for a launcher that reads
/// `reads`, given the arguments `given` it has: each variable of a flag the seat takes is carried
/// as that flag and its value, unless the command line gives the flag and the flag holds one
/// value ([`Meet`]), and a line says which; every other variable has a line saying bloomery does
/// not read it. A key's value is in no line. The last of a name counts, as in [`take_with_env`].
pub fn carry_env(
    env: &[(String, String)],
    given: &[&str],
    reads: Reads,
) -> (Vec<(&'static str, String)>, Vec<String>) {
    let mut vars: Vec<(&str, &str)> = Vec::new();
    for (name, value) in env.iter().rev() {
        if !vars.iter().any(|(n, _)| *n == name.as_str()) {
            vars.push((name, value));
        }
    }
    vars.reverse();
    let mut carried = Vec::new();
    let mut notes = Vec::new();
    for (var, value) in vars {
        if crate::llama_place::reads_var(var) || (reads.model && (var == MODEL || var == HF_REPO)) {
            continue;
        }
        let Some(read) = reads.flags.iter().find(|e| e.var == var) else {
            notes.push(format!("{var} is set, and bloomery does not read it"));
            continue;
        };
        let on_the_line = read.spelled.iter().find(|s| given.contains(*s));
        if let (Some(on_the_line), Meet::Replace) = (on_the_line, read.meet) {
            notes.push(format!(
                "{var} is set, and the command line's {on_the_line} replaces it"
            ));
            continue;
        }
        carried.push((read.flag, value.to_owned()));
        let shown = if read.meet == Meet::AddSecret {
            "(value not printed)"
        } else {
            value
        };
        notes.push(match on_the_line {
            Some(on_the_line) => format!(
                "{var} is read as {} {shown}, added to the command line's {on_the_line}",
                read.flag
            ),
            None => format!("{var} is read as {} {shown}", read.flag),
        });
    }
    (carried, notes)
}

/// `rest`, the arguments less the placement flags, as a seat reads them under llama's spellings:
/// the variables of `env` carried as the flags they stand for ([`carry_env`]), each single-dash
/// spelling respelled as its flag ([`respell`]), and a prompt cache without a limit refused
/// ([`refuse_unbounded_cache`]); with a line for each variable.
pub fn seat_args<A>(
    rest: Vec<A>,
    env: &[(String, String)],
    reads: Reads,
) -> Result<(Vec<A>, Vec<String>), GateError>
where
    A: AsRef<OsStr> + From<String>,
{
    let (carried, notes) = {
        let mut given = Vec::new();
        let mut it = rest.iter();
        while let Some(raw) = it.next() {
            if let Some(flag) = raw.as_ref().to_str() {
                given.push(flag);
                if reads.flags.iter().any(|read| read.spelled.contains(&flag))
                    && serve_takes_value(flag)
                {
                    it.next();
                }
            }
        }
        carry_env(env, &given, reads)
    };
    let mut out = respell(rest, reads);
    for (flag, value) in carried {
        out.push(A::from(flag.to_owned()));
        out.push(A::from(value));
    }
    refuse_unbounded_cache(&out, reads)?;
    Ok((out, notes))
}

/// [`seat_args`] over the process's own variables, its lines printed on stderr under `bin`: the
/// start of a seat that places no card (the decision seat), whose arguments the placement flags'
/// start does not read, so each placement variable is named as one it does not act on
/// ([`unplaced`]).
pub fn seat_from_env(bin: &str, args: Vec<String>, reads: Reads) -> Result<Vec<String>, GateError> {
    let env = llama_env(std::env::vars_os())?;
    let (args, notes) = seat_args(args, &env, reads)?;
    for line in unplaced(&env).into_iter().chain(notes) {
        eprintln!("{bin}: {line}");
    }
    Ok(args)
}

/// A line for each placement variable of `env` ([`crate::llama_place::reads_var`]), for a seat
/// that places no card: set, it changes nothing there, and a variable set is said, not dropped.
pub fn unplaced(env: &[(String, String)]) -> Vec<String> {
    env.iter()
        .filter(|(name, _)| crate::llama_place::reads_var(name))
        .map(|(name, _)| format!("{name} is set, and this seat places no card"))
        .collect()
}

/// `args` with each single-dash spelling the table of `reads` gives a flag (`-cram`, `-a`, `-mm`,
/// `-c`) respelled as the flag: the seats' parsers read the flags' own spellings and only some of
/// the short ones, and llama-server's users type the short ones. A spelling with two dashes
/// stays, `--no-mmproj` among them: it is a switch of its own.
pub fn respell<A>(args: Vec<A>, reads: Reads) -> Vec<A>
where
    A: AsRef<OsStr> + From<String>,
{
    let mut out = Vec::with_capacity(args.len());
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let takes_value = a.as_ref().to_str().is_some_and(|arg| {
            reads.flags.iter().any(|read| read.spelled.contains(&arg)) && serve_takes_value(arg)
        });
        let flag = a.as_ref().to_str().and_then(|arg| {
            reads
                .flags
                .iter()
                .find(|e| !arg.starts_with("--") && e.spelled.contains(&arg))
                .map(|e| e.flag)
        });
        out.push(match flag {
            Some(flag) => A::from(flag.to_owned()),
            None => a,
        });
        if takes_value {
            out.extend(it.next());
        }
    }
    out
}

/// A negative `--cache-ram`, refused by name where the seat takes the flag. llama-server reads
/// `-1` as a prompt cache with no limit; the cache here is host memory beside the load's host
/// set, so it always has a budget, and a budget is MiB.
pub fn refuse_unbounded_cache<A: AsRef<OsStr>>(args: &[A], reads: Reads) -> Result<(), GateError> {
    const FLAG: &str = "--cache-ram";
    if !reads.flags.iter().any(|e| e.flag == FLAG) {
        return Ok(());
    }
    let mut it = args.iter();
    while let Some(raw) = it.next() {
        let Some(arg) = raw.as_ref().to_str() else {
            continue;
        };
        let Some(read) = reads.flags.iter().find(|read| read.spelled.contains(&arg)) else {
            continue;
        };
        if !serve_takes_value(arg) {
            continue;
        }
        let Some(raw_value) = it.next() else { continue };
        if read.flag == FLAG
            && let Some(value) = raw_value.as_ref().to_str()
            && value.starts_with('-')
        {
            return Err(format!(
                "{FLAG} {value} (also -cram and LLAMA_ARG_CACHE_RAM): llama-server reads a \
                 negative count as a prompt cache with no limit, and this cache is host memory \
                 beside the load, so it always has a budget; give MiB, 0 turns it off, unset \
                 takes half of what the host leaves past the load, at most llama-server's default"
            )
            .into());
        }
    }
    Ok(())
}

/// `args` with the model `LLAMA_ARG_MODEL` (a path, `-m`) or `LLAMA_ARG_HF_REPO` (`-hf`, the
/// server's `--hf`) names, unless the command line names a model by one of `path_flags` or
/// `--hf`, which replaces them; a line for each reading. Both variables set are refused by name:
/// two models.
pub fn model_args(
    mut args: Vec<String>,
    env: &[(String, String)],
    path_flags: &[&str],
) -> Result<(Vec<String>, Vec<String>), GateError> {
    let last = |var: &str| env.iter().rev().find(|(k, _)| k == var).map(|(_, v)| v);
    let (path, repo) = (last(MODEL), last(HF_REPO));
    if path.is_some() && repo.is_some() {
        return Err(format!("{MODEL} and {HF_REPO} are both set, and name two models").into());
    }
    let on_the_line = serve_args(&args)
        .map(|(flag, _)| flag)
        .find(|a| path_flags.contains(&a.as_str()) || a.as_str() == "--hf")
        .cloned();
    let mut notes = Vec::new();
    for (var, value, flag) in [(MODEL, path, path_flags[0]), (HF_REPO, repo, "--hf")] {
        let Some(value) = value else {
            continue;
        };
        match &on_the_line {
            Some(given) => notes.push(format!(
                "{var} is set, and the command line's {given} replaces it"
            )),
            None => {
                notes.push(format!("{var} is read as {flag} {value}"));
                args.push(flag.to_owned());
                args.push(value.clone());
            }
        }
    }
    Ok((args, notes))
}

/// [`model_args`] over the process's own variables, its lines printed on stderr under `bin`.
pub fn model_from_env(
    bin: &str,
    args: Vec<String>,
    path_flags: &[&str],
) -> Result<Vec<String>, GateError> {
    let env = llama_env(std::env::vars_os())?;
    let (args, notes) = model_args(args, &env, path_flags)?;
    for line in notes {
        eprintln!("{bin}: {line}");
    }
    Ok(args)
}

/// What a flag set comes to on one seat: the levers to hand over, the `--place` word the cards
/// come to (`None` where the flags name none) and the lines that say each reading.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Resolved {
    pub set: Vec<(&'static str, String)>,
    pub word: Option<String>,
    pub notes: Vec<String>,
}

/// `flags` on `target`: the refusals and levers that need no device ([`Flags::levers`]), then, for
/// the flags that read one, the cards ([`Flags::place`]), the card shares ([`Flags::budget`]) and
/// the margins ([`Flags::margin`]) against the census `census` reads, which is asked only then.
pub fn resolve(
    flags: &Flags,
    target: &Target<'_>,
    census: impl FnOnce() -> Result<Vec<DeviceInfo>, GateError>,
) -> Result<Resolved, GateError> {
    let levers = flags.levers(target)?;
    let mut out = Resolved {
        set: levers.set,
        word: None,
        notes: levers.notes,
    };
    if flags.needs_census() {
        let census = census()?;
        if let Some(cards) = flags.place(&census)? {
            out.word = Some(cards.word);
            out.notes.extend(cards.notes);
        }
        let budget = flags.budget(&census)?;
        out.set.extend(budget.set);
        out.notes.extend(budget.notes);
        let margin = flags.margin(&census)?;
        out.set.extend(margin.set);
        out.notes.extend(margin.notes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn serve_options_skip_values_and_keep_switches() {
        for flag in [
            "--levers",
            "--list-devices",
            "--records-schema",
            "--version",
            "--help",
        ] {
            let args = ["--alias", flag, "-ngl", "all", "--ctx-size", "abc"].map(str::to_owned);
            assert!(!serve_option(&args, flag), "{flag} is an alias");
            assert!(serve_option(&args, "--ctx-size"));
        }
        for flag in [
            "--plan",
            "--no-mmproj",
            "--embd-no-mmproj",
            "--cpu-moe",
            "--cpu_moe",
            "-cmoe",
        ] {
            let args = [flag, "--levers"].map(str::to_owned);
            assert!(serve_option(&args, "--levers"), "{flag} has no value");
        }
    }

    #[test]
    fn serve_values_survive_every_shared_reader() {
        for value in [
            "--cpu_moe",
            "--cache-ram",
            "-cram",
            "--place",
            "--model",
            "--hf",
            "--head",
            "--embd-hf",
        ] {
            let args = ["--alias", value, "--ctx-size", "8192", "--plan"]
                .map(str::to_owned)
                .to_vec();
            let (flags, rest, _) =
                take_with_env(&args, &[], SEAT_FLAGS).expect("placement leaves the alias");
            assert!(flags.is_empty(), "{value}");
            assert_eq!(rest, args, "{value}");
            let (selected, _) = take_serve_args(&args, &[value]);
            assert!(selected.is_empty(), "{value} is a value");
            let (out, _) = seat_args(rest, &env_of(&[("LLAMA_ARG_CACHE_RAM", "256")]), SERVE)
                .expect("the alias has no cache budget");
            let mut expected = args;
            expected.extend(["--cache-ram", "256"].map(str::to_owned));
            assert_eq!(out, expected, "{value}");
        }
    }

    use model::placement::workstation::{MIB, census_usable};

    use super::*;
    use crate::llama_place::take_with_env;

    /// A card of the census by driver name and total bytes, idle.
    fn card(ordinal: u32, name: &str, total: u64) -> DeviceInfo {
        DeviceInfo {
            ordinal,
            name: name.to_owned(),
            total_bytes: total,
            free_bytes: census_usable(total),
            integrated: false,
            uuid: [u8::try_from(ordinal).expect("small") + 1; 16],
            pci_bus: format!("0000:{:02x}:00.0", 0x41 + ordinal),
            held_by: None,
            compute_capability: Some((8, 6)),
        }
    }

    /// The A6000 as CUDA0 and the 3090 as CUDA1.
    fn two() -> Vec<DeviceInfo> {
        vec![
            card(0, "NVIDIA RTX A6000", 50_952_536_064),
            card(1, "NVIDIA GeForce RTX 3090", 25_351_356_416),
        ]
    }

    fn target(seat: &'static str) -> Target<'static> {
        Target {
            seat,
            layers: Some(61),
            ctx_floor: 4096,
        }
    }

    /// `line` and `env` taken and resolved on `seat` over [`two`].
    fn resolved(line: &str, env: &[(&str, &str)], seat: &'static str) -> Result<Resolved, String> {
        let args: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let (flags, rest, _) = take_with_env(&args, &env, &[]).map_err(|e| e.to_string())?;
        assert!(rest.is_empty(), "{rest:?}");
        resolve(&flags, &target(seat), || Ok(two())).map_err(|e| e.to_string())
    }

    /// Only llama-server's variable names are read from the environment, whatever other names it
    /// holds (`LLAMA_API_KEY` is the one with no `LLAMA_ARG_` prefix), and a value that is no
    /// UTF-8 is refused under its name, a key's without the bytes.
    #[test]
    fn only_llama_variables_are_read_and_a_non_utf8_value_is_named() {
        use std::os::unix::ffi::OsStringExt;

        let vars = [
            ("PATH", "/bin"),
            ("LLAMA_ARG_DEVICE", "CUDA0"),
            ("HOME", "/root"),
            ("LLAMA_API_KEY", "k1"),
            ("LLAMA_API_KEYS", "no"),
            ("LLAMA_ARG_CTX_SIZE", "4096"),
        ]
        .map(|(k, v)| (OsString::from(k), OsString::from(v)));
        assert_eq!(
            llama_env(vars).expect("text"),
            [
                ("LLAMA_ARG_DEVICE".to_owned(), "CUDA0".to_owned()),
                ("LLAMA_API_KEY".to_owned(), "k1".to_owned()),
                ("LLAMA_ARG_CTX_SIZE".to_owned(), "4096".to_owned())
            ]
        );
        let bytes = OsString::from_vec(vec![0xff]);
        let bad = [(OsString::from("LLAMA_ARG_DEVICE"), bytes)];
        let e = llama_env(bad).expect_err("bytes").to_string();
        assert!(e.starts_with("LLAMA_ARG_DEVICE: the value"), "{e}");
        let key = [(
            OsString::from("LLAMA_API_KEY"),
            OsString::from_vec(vec![b's', 0xff]),
        )];
        let e = llama_env(key).expect_err("bytes").to_string();
        assert_eq!(e, "LLAMA_API_KEY: the value is not UTF-8");
        assert!(llama_env([(OsString::from("HOME"), OsString::from_vec(vec![0xff]))]).is_ok());
    }

    /// A flag set and the variables that stand for it come to the same cards and levers: the
    /// `--place` word, the levers in the order the flags are read (experts, then the card margin)
    /// and a line for each of the three readings.
    #[test]
    fn flags_and_their_variables_resolve_alike() {
        let by_flags = resolved("-dev CUDA1 -fitt 1024 --cpu-moe", &[], "qwen38").expect("flags");
        let by_vars = resolved(
            "",
            &[
                ("LLAMA_ARG_DEVICE", "CUDA1"),
                ("LLAMA_ARG_FIT_TARGET", "1024"),
                ("LLAMA_ARG_CPU_MOE", "1"),
            ],
            "qwen38",
        )
        .expect("variables");
        for r in [&by_flags, &by_vars] {
            assert_eq!(r.word.as_deref(), Some("cuda1"));
            assert_eq!(
                r.set,
                [
                    ("BLOOMERY_QWEN38_EXPERTS", "host".to_owned()),
                    ("BLOOMERY_CARD_MARGIN", (1024 * MIB).to_string())
                ]
            );
            assert_eq!(r.notes.len(), 3, "{:?}", r.notes);
        }
    }

    /// `-ts` and `-fitt` together set two levers, the shares as the card budget and the margin as
    /// the card margin, beside the `bp` word the two shares make: neither stands in for the other.
    #[test]
    fn shares_and_margin_are_two_levers() {
        let r = resolved("-ts 3,1 -fitt 512", &[], "qwen38").expect("both");
        assert_eq!(r.word.as_deref(), Some("bp"));
        assert_eq!(
            r.set,
            [
                (
                    "BLOOMERY_CARD_BUDGET",
                    "CUDA0=750000x,CUDA1=250000x".to_owned()
                ),
                ("BLOOMERY_CARD_MARGIN", (512 * MIB).to_string())
            ]
        );
        // The cards' line, the layer rule and the shares' rule, then one line a lever.
        assert_eq!(r.notes.len(), 5, "{:?}", r.notes);
    }

    /// A seat that has no thing for a flag refuses it by name: the draft's card is a seat with a
    /// draft's, the experts' switch a seat with one.
    #[test]
    fn a_seat_refuses_the_flag_it_has_no_thing_for() {
        let on = resolved("-devd CUDA1", &[], "ds41").expect("a draft seat");
        assert_eq!(on.set, [("BLOOMERY_DSPARK_CARD", "cuda1".to_owned())]);
        assert_eq!(on.word, None);
        let e = resolved("-devd CUDA1", &[], "chat").expect_err("no draft");
        assert!(
            e.contains("this seat's draft opens on no card a flag names"),
            "{e}"
        );
        let e = resolved("", &[("LLAMA_ARG_CPU_MOE", "1")], "chat").expect_err("no switch");
        assert!(
            e.starts_with(
                "LLAMA_ARG_CPU_MOE: this seat has no switch that keeps every routed expert"
            ),
            "{e}"
        );
    }

    /// A flag that reads no device asks for none: the census is not read for `-ngl`, and nothing
    /// comes of it.
    #[test]
    fn the_census_is_read_only_for_the_flags_that_need_it() {
        let args = ["-ngl".to_owned(), "all".to_owned()];
        let (flags, _, _) = take_with_env(&args, &[], &[]).expect("-ngl");
        let none = resolve(
            &flags,
            &target("ds41"),
            || -> Result<Vec<DeviceInfo>, GateError> { unreachable!("-ngl names no card") },
        )
        .expect("nothing to resolve");
        assert_eq!(none, Resolved::default());
    }

    fn env_of(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    /// A seat's variables become its flags unless the command line gives the flag, in any of its
    /// spellings; a placement flag's variable is [`take_with_env`]'s and no line of this; every
    /// other variable has a line saying bloomery does not read it, and a launcher whose seat
    /// takes none of them says so of all.
    #[test]
    fn a_seats_variables_become_its_flags_and_the_rest_are_named() {
        let env = env_of(&[
            ("LLAMA_ARG_CTX_SIZE", "4096"),
            ("LLAMA_ARG_PORT", "8081"),
            ("LLAMA_ARG_N_PARALLEL", "2"),
            ("LLAMA_ARG_FLASH_ATTN", "on"),
            ("LLAMA_ARG_DEVICE", "CUDA0"),
            ("LLAMA_ARG_N_PARALLEL", "3"),
        ]);
        let (carried, notes) = carry_env(&env, &["--port", "9", "-np", "1"], SEAT);
        assert_eq!(carried, [("--ctx-size", "4096".to_owned())]);
        assert_eq!(
            notes,
            [
                "LLAMA_ARG_CTX_SIZE is read as --ctx-size 4096",
                "LLAMA_ARG_PORT is set, and the command line's --port replaces it",
                "LLAMA_ARG_FLASH_ATTN is set, and bloomery does not read it",
                "LLAMA_ARG_N_PARALLEL is set, and the command line's -np replaces it",
            ]
        );
        let (carried, notes) = carry_env(&env, &[], SEAT);
        assert_eq!(
            carried,
            [
                ("--ctx-size", "4096".to_owned()),
                ("--port", "8081".to_owned()),
                ("--parallel", "3".to_owned())
            ]
        );
        assert_eq!(notes.len(), 4, "{notes:?}");
        let (none, notes) = carry_env(&env, &[], NONE);
        assert!(none.is_empty());
        assert_eq!(
            notes,
            [
                "LLAMA_ARG_CTX_SIZE is set, and bloomery does not read it",
                "LLAMA_ARG_PORT is set, and bloomery does not read it",
                "LLAMA_ARG_FLASH_ATTN is set, and bloomery does not read it",
                "LLAMA_ARG_N_PARALLEL is set, and bloomery does not read it",
            ]
        );
    }

    /// The model variables are a flag's of the one launcher that takes them ([`SERVE`]) and a line
    /// of every other: they name the model unless the command line does, both set are two models,
    /// and the path goes through the seat's own path flag.
    #[test]
    fn the_model_variables_name_the_model_unless_the_command_line_does() {
        let line = |s: &str| -> Vec<String> { s.split_whitespace().map(str::to_owned).collect() };
        let flags = ["-m", "--model-file"];
        let path = env_of(&[("LLAMA_ARG_MODEL", "/m/a.gguf")]);
        let (args, notes) = model_args(line("--port 1"), &path, &flags).expect("a path");
        assert_eq!(args, line("--port 1 -m /m/a.gguf"));
        assert_eq!(notes, ["LLAMA_ARG_MODEL is read as -m /m/a.gguf"]);
        let (args, notes) = model_args(line("--model-file x.gguf"), &path, &flags).expect("given");
        assert_eq!(args, line("--model-file x.gguf"));
        assert_eq!(
            notes,
            ["LLAMA_ARG_MODEL is set, and the command line's --model-file replaces it"]
        );
        let repo = env_of(&[("LLAMA_ARG_HF_REPO", "u/m:Q4_K_M")]);
        let (args, notes) = model_args(vec![], &repo, &flags).expect("a repo");
        assert_eq!(args, line("--hf u/m:Q4_K_M"));
        assert_eq!(notes, ["LLAMA_ARG_HF_REPO is read as --hf u/m:Q4_K_M"]);
        let (args, _) = model_args(line("--hf a/b"), &repo, &flags).expect("given");
        assert_eq!(args, line("--hf a/b"));
        let both = env_of(&[("LLAMA_ARG_MODEL", "x"), ("LLAMA_ARG_HF_REPO", "u/m")]);
        let e = model_args(vec![], &both, &flags).expect_err("two models");
        assert_eq!(
            e.to_string(),
            "LLAMA_ARG_MODEL and LLAMA_ARG_HF_REPO are both set, and name two models"
        );
        let (args, notes) = model_args(vec![], &[], &flags).expect("none");
        assert!(args.is_empty() && notes.is_empty());
        let (carried, notes) = carry_env(&path, &[], SERVE);
        assert!(carried.is_empty() && notes.is_empty());
        let (_, notes) = carry_env(&path, &[], SEAT);
        assert_eq!(
            notes,
            ["LLAMA_ARG_MODEL is set, and bloomery does not read it"]
        );
    }

    /// The seat table's variables are llama-server's `LLAMA_ARG_*` names and `LLAMA_API_KEY`, each
    /// once and none a placement flag's; each flag is among its own spellings; a flag is named
    /// once; only the key's flag holds a secret.
    #[test]
    fn the_seat_table_names_each_variable_and_flag_once() {
        let mut vars: Vec<&str> = SEAT_FLAGS.iter().map(|e| e.var).collect();
        vars.sort_unstable();
        vars.dedup();
        assert_eq!(vars.len(), SEAT_FLAGS.len(), "a variable names one flag");
        let mut flags: Vec<&str> = SEAT_FLAGS.iter().map(|e| e.flag).collect();
        flags.sort_unstable();
        flags.dedup();
        assert_eq!(flags.len(), SEAT_FLAGS.len(), "a flag has one variable");
        for e in SEAT_FLAGS {
            assert!(e.var.starts_with(PREFIX) || e.var == API_KEY, "{}", e.var);
            assert_eq!(
                e.meet == Meet::AddSecret,
                e.flag == "--api-key",
                "{}",
                e.flag
            );
            assert!(
                !crate::llama_place::reads_var(e.var),
                "{}: a placement flag's",
                e.var
            );
            assert!(e.var != MODEL && e.var != HF_REPO, "{}", e.var);
            assert!(e.spelled.contains(&e.flag), "{}", e.flag);
        }
    }

    /// The keys are added, never replaced, as llama-server reads the variables first and then the
    /// command line, and a key's value is in no line: `LLAMA_API_KEY` is `--api-key`, beside the
    /// command line's own, and a launcher that takes no key says it does not read it.
    #[test]
    fn the_key_variables_add_to_the_command_lines_keys_and_print_no_key() {
        let env = env_of(&[
            ("LLAMA_API_KEY", "s3cret"),
            ("LLAMA_ARG_API_KEY_FILE", "/run/keys"),
        ]);
        let (carried, notes) = carry_env(&env, &[], SEAT);
        assert_eq!(
            carried,
            [
                ("--api-key", "s3cret".to_owned()),
                ("--api-key-file", "/run/keys".to_owned())
            ]
        );
        assert_eq!(
            notes,
            [
                "LLAMA_API_KEY is read as --api-key (value not printed)",
                "LLAMA_ARG_API_KEY_FILE is read as --api-key-file /run/keys"
            ]
        );
        let (carried, notes) =
            carry_env(&env, &["--api-key", "other", "--api-key-file", "/k"], SEAT);
        assert_eq!(carried.len(), 2, "the variables still count: {carried:?}");
        assert_eq!(
            notes,
            [
                "LLAMA_API_KEY is read as --api-key (value not printed), added to the command \
                 line's --api-key",
                "LLAMA_ARG_API_KEY_FILE is read as --api-key-file /run/keys, added to the command \
                 line's --api-key-file"
            ]
        );
        for notes in [
            carry_env(&env, &[], NONE).1,
            carry_env(&env, &["--api-key", "x"], NONE).1,
        ] {
            assert_eq!(
                notes,
                [
                    "LLAMA_API_KEY is set, and bloomery does not read it",
                    "LLAMA_ARG_API_KEY_FILE is set, and bloomery does not read it"
                ]
            );
        }
    }

    /// A single-dash spelling the table gives a flag becomes the flag, in any position a flag can
    /// be and only as a whole argument; a double-dash spelling stays (`--no-mmproj` is a switch
    /// of its own), and a launcher that reads none respells none.
    #[test]
    fn a_short_spelling_becomes_its_flag() {
        let line = |s: &str| -> Vec<String> { s.split_whitespace().map(str::to_owned).collect() };
        assert_eq!(
            respell(
                line("-cram 512 -a chat -mm p.gguf -c 4096 -np 2 -ub 512"),
                SEAT
            ),
            line(
                "--cache-ram 512 --alias chat --mmproj p.gguf --ctx-size 4096 --parallel 2 --ubatch-size 512"
            )
        );
        let kept = line("--no-mmproj --ctx 8 -cramx -cr --cache-ram 1 -m x.gguf");
        assert_eq!(respell(kept.clone(), SEAT), kept);
        let given = line("-cram 512");
        assert_eq!(respell(given.clone(), NONE), given);
    }

    /// A negative `--cache-ram` is refused by name on a seat that takes the flag, however it was
    /// given (the variable is carried as the flag first); a count passes, a seat without the flag
    /// leaves it to its own parser, and `-1` as another flag's value is not the cache's.
    #[test]
    fn an_unbounded_cache_is_refused_by_name() {
        let line = |s: &str| -> Vec<String> { s.split_whitespace().map(str::to_owned).collect() };
        for ok in [
            "--cache-ram 0",
            "--cache-ram 8192",
            "--seed -1 --cache-ram 64",
            "--alias -1",
        ] {
            assert!(refuse_unbounded_cache(&line(ok), SEAT).is_ok(), "{ok}");
        }
        assert!(refuse_unbounded_cache(&line("--cache-ram -1"), NONE).is_ok());
        let by_flag = seat_args(line("-cram -1"), &[], SEAT).expect_err("the short spelling");
        let by_variable = seat_args(
            line("--port 1"),
            &env_of(&[("LLAMA_ARG_CACHE_RAM", "-1")]),
            SEAT,
        )
        .expect_err("the variable");
        for e in [by_flag.to_string(), by_variable.to_string()] {
            assert!(
                e.starts_with("--cache-ram -1 (also -cram and LLAMA_ARG_CACHE_RAM): ")
                    && e.contains("give MiB, 0 turns it off"),
                "{e}"
            );
        }
        let (args, notes) = seat_args(
            line("-cram 512"),
            &env_of(&[("LLAMA_ARG_CTX_SIZE", "4096")]),
            SEAT,
        )
        .expect("a count");
        assert_eq!(args, line("--cache-ram 512 --ctx-size 4096"));
        assert_eq!(notes, ["LLAMA_ARG_CTX_SIZE is read as --ctx-size 4096"]);
    }

    /// A seat that places no card names each placement variable set, in the order the
    /// environment holds them, and none of the others.
    #[test]
    fn a_seat_that_places_no_card_names_the_placement_variables() {
        let env = env_of(&[
            ("LLAMA_ARG_DEVICE", "CUDA0"),
            ("LLAMA_ARG_CTX_SIZE", "4096"),
            ("LLAMA_ARG_N_GPU_LAYERS", "99"),
            ("LLAMA_API_KEY", "k"),
        ]);
        assert_eq!(
            unplaced(&env),
            [
                "LLAMA_ARG_DEVICE is set, and this seat places no card",
                "LLAMA_ARG_N_GPU_LAYERS is set, and this seat places no card"
            ]
        );
        assert!(unplaced(&[]).is_empty());
    }
}
