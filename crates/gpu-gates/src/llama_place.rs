//! llama-server's placement flags, read as bloomery's placement. Every launcher ([`crate::llama_launch`]) takes the flags a
//! llama-server run places its model with, each with llama's meaning where bloomery has the thing
//! the flag names, and refuses by name each one it has no thing for:
//!
//! | flag | bloomery |
//! |---|---|
//! | `--list-devices` | [`list_devices`]: the census in llama's shape, exit 0 |
//! | `-dev`, `-sm none`, `-mg`, `-ts` | the cards the plan may use ([`Flags::place`]); with two cards left `-ts` is also each card's share of the routed experts, `BLOOMERY_CARD_BUDGET` shares the plan turns into a budget per card ([`Flags::budget`]) |
//! | `-sm layer` (default) | the largest listed card runs every layer, the others hold experts |
//! | `-ngl` | a full offload only ([`Flags::levers`]); a smaller count is refused |
//! | `--cpu-moe`, `-ncmoe`, `-ot …=CPU` | `BLOOMERY_QWEN38_EXPERTS=host` on the qwen38 seat; refused on the rest |
//! | `-fit`, `-fitc` | the plan always fits; `-fit off` and another `-fitc` than the seats' floor are refused |
//! | `-fitt` | `BLOOMERY_CARD_MARGIN`, the bytes a card keeps free, which replace the plan's own margin ([`Flags::margin`]): one value is every device's margin, a list one per listed device; with no card named, every visible card has its margin and the load's own rule picks among them (a second card as an expert tier where the family's rule takes one) |
//! | `-devd` | `BLOOMERY_DSPARK_CARD` on the ds41 seat |
//! | `-sm row`, `-sm tensor`, `-dev none` | refused: bloomery has no row split and no CPU-only seat |
//!
//! [`take`] cuts these flags out of a seat's arguments. What a flag comes to is a lever
//! ([`Flags::levers`], no device read) or a `--place` word ([`Flags::place`], against the census):
//! the word every seat already parses, so a seat keeps one vocabulary for its cards, and the
//! alias (`a`, `bp`) wherever the cards are the alias's. `--place` beside any of these flags is
//! refused. llama-server's environment names the same flags: [`take_with_env`] reads the variable
//! of each (`ENV`; `-devd` has none there) as the flag it stands for, a flag on the command line
//! replaces its variable, and `--place` replaces the variables that name cards. The other
//! `LLAMA_ARG_*` variables are [`crate::llama_launch`]'s: it carries those a seat's flags stand for
//! and says in a line each one bloomery does not read. Every launcher takes both through it.

use std::ffi::OsStr;

use model::placement::workstation::{self, DeviceInfo, MIB, Pick, spec_of_device};

use crate::GateError;

/// `--list-devices`: print the devices ([`list_devices`]) and exit; it takes no value.
pub const LIST_DEVICES: &str = "--list-devices";

/// The lever `-devd` sets: the card the DSpark draft opens on.
pub const DRAFT_CARD: &str = "BLOOMERY_DSPARK_CARD";

/// The parts `-ts`' shares are resolved into: a share is the whole number of millionths of the
/// total it comes to.
const SHARE_PARTS: f64 = 1_000_000.0;

const DEVICE: &[&str] = &["-dev", "--device"];
const SPLIT: &[&str] = &["-sm", "--split-mode"];
const TENSOR_SPLIT: &[&str] = &["-ts", "--tensor-split"];
const MAIN_GPU: &[&str] = &["-mg", "--main-gpu"];
const GPU_LAYERS: &[&str] = &["-ngl", "--gpu-layers", "--n-gpu-layers"];
pub(crate) const CPU_MOE: &[&str] = &["-cmoe", "--cpu-moe"];
const N_CPU_MOE: &[&str] = &["-ncmoe", "--n-cpu-moe"];
const OVERRIDE_TENSOR: &[&str] = &["-ot", "--override-tensor"];
const FIT: &[&str] = &["-fit", "--fit"];
const FIT_TARGET: &[&str] = &["-fitt", "--fit-target"];
const FIT_CTX: &[&str] = &["-fitc", "--fit-ctx"];
const DEVICE_DRAFT: &[&str] = &["-devd", "--device-draft", "--spec-draft-device"];

/// Every flag [`take`] cuts out.
const FLAGS: [&[&str]; 12] = [
    DEVICE,
    SPLIT,
    TENSOR_SPLIT,
    MAIN_GPU,
    GPU_LAYERS,
    CPU_MOE,
    N_CPU_MOE,
    OVERRIDE_TENSOR,
    FIT,
    FIT_TARGET,
    FIT_CTX,
    DEVICE_DRAFT,
];

/// The patterns of `-ot <pattern>=CPU` that name every routed expert: the regex `--cpu-moe` itself
/// expands to, and the spellings llama-server's documentation gives for it. Any other pattern
/// names tensors bloomery cannot select by regex.
const ALL_EXPERTS: &[&str] = &[
    "exps",
    "ffn_.*_exps",
    ".ffn_.*_exps.",
    "\\.ffn_.*_exps\\.",
    "\\.ffn_(up|down|gate|gate_up)_(ch|)exps",
];

/// What `-sm` asks for (`row` and `tensor` are refused where they are read).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Split {
    None,
    Layer,
}

/// What `-ngl` asks for: `auto`, `all` and any negative count are every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ngl {
    All,
    Layers(u64),
}

/// Which routed experts stay off the cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Moe {
    All,
    First(u64),
}

/// The placement flags of one command line, as [`take`] read them.
#[derive(Debug, Default, PartialEq)]
pub struct Flags {
    /// Every flag with its value, in order.
    given: Vec<String>,
    /// The flags that name cards (`-dev`, `-sm`, `-ts`, `-mg`), as the note prints them.
    placing: Vec<String>,
    dev: Option<Vec<u32>>,
    split: Option<Split>,
    /// `-ts` as given, and its shares, one per listed device.
    shares: Option<(String, Vec<f64>)>,
    main_gpu: Option<i64>,
    ngl: Option<Ngl>,
    /// The expert flag as given, and what it asks.
    moe: Option<(String, Moe)>,
    fit: Option<bool>,
    /// `-fitt` as given, and its margins in MiB, one per value.
    fit_target: Option<(String, Vec<u64>)>,
    fit_ctx: Option<u64>,
    draft: Option<u32>,
}

/// The seat and file the flags place: the seat's word (`Model::word` of the server), the file's
/// layer count and the floor of an unset `--ctx`.
pub struct Target<'a> {
    /// The seat, by the word `--model` takes.
    pub seat: &'a str,
    /// The file's `block_count`, `None` where it states none or was not read.
    pub layers: Option<u64>,
    /// The least context an unset `--ctx` steps down to.
    pub ctx_floor: u64,
}

/// The levers a flag set comes to, and the lines that say so.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Levers {
    /// `(name, value)`, to hand to `bloomery_levers::override_with` before the seat reads its
    /// levers; a flag replaces a value the environment holds, as it does in llama-server.
    pub set: Vec<(&'static str, String)>,
    pub notes: Vec<String>,
}

/// The cards a flag set names: the `--place` word every seat parses, and the lines that say so.
#[derive(Debug, PartialEq, Eq)]
pub struct Cards {
    pub word: String,
    pub notes: Vec<String>,
}

/// llama-server's environment variable for each flag [`take_with_env`] reads: the `.set_env("...")`
/// of the flag's `add_opt` in `common/arg.cpp` of ggml-org/llama.cpp at commit f5027fb, the line
/// being that `.set_env`'s. `-devd` (`--spec-draft-device`) has none there, so it has no row.
const ENV: [(&[&str], &str); 11] = [
    (DEVICE, "LLAMA_ARG_DEVICE"),                   // line 2748
    (SPLIT, "LLAMA_ARG_SPLIT_MODE"),                // line 2843
    (TENSOR_SPLIT, "LLAMA_ARG_TENSOR_SPLIT"),       // line 2870
    (MAIN_GPU, "LLAMA_ARG_MAIN_GPU"),               // line 2880
    (GPU_LAYERS, "LLAMA_ARG_N_GPU_LAYERS"),         // line 2819
    (CPU_MOE, "LLAMA_ARG_CPU_MOE"),                 // line 2769
    (N_CPU_MOE, "LLAMA_ARG_N_CPU_MOE"),             // line 2779
    (OVERRIDE_TENSOR, "LLAMA_ARG_OVERRIDE_TENSOR"), // line 2762
    (FIT, "LLAMA_ARG_FIT"),                         // line 2894
    (FIT_TARGET, "LLAMA_ARG_FIT_TARGET"),           // line 2933
    (FIT_CTX, "LLAMA_ARG_FIT_CTX"),                 // line 2940
];

/// Whether `var` is one of the variables [`take_with_env`] reads as a placement flag.
#[must_use]
pub fn reads_var(var: &str) -> bool {
    ENV.iter().any(|(_, v)| *v == var)
}

/// The flags that name cards, whose variables `--place` replaces.
const NAMES_CARDS: [&[&str]; 4] = [DEVICE, SPLIT, TENSOR_SPLIT, MAIN_GPU];

/// [`take_with_env`] with no environment: the command line alone.
pub fn take(args: &[String]) -> Result<(Flags, Vec<String>), GateError> {
    let (flags, rest, _) = take_with_env(args, &[], &[])?;
    Ok((flags, rest))
}

/// `args` with every placement flag cut out, and what they and the `LLAMA_ARG_*` variables of `env`
/// (`(name, value)`, the last of a name counting) said; the third part is a line for each variable
/// something else replaced. A flag given twice, a flag with no value (or no UTF-8 one), a value
/// llama-server would refuse, a flag bloomery has no thing for (`-sm row`) and `--place` beside any
/// of them are refused by name; so is a variable whose flag would be, under the variable's name.
///
/// A variable is read as its flag with the variable's value, as llama-server reads it: it applies
/// unless the command line gives that flag (a line says so), or gives `--place` and the variable
/// names cards. The `--cpu-moe` variable is on for `on`, `enabled`, `true`, `1`, and off for `off`,
/// `disabled`, `false`, `0`; the three variables that name the routed experts' place and the flags
/// that name it are one place, named once. An argument that is no UTF-8 is not a flag and stays.
/// The value of `--place` and of each flag of `seat` (the caller's seat flags) stays with its flag,
/// so a value spelled like a placement flag is not read as one.
pub fn take_with_env<A>(
    args: &[A],
    env: &[(String, String)],
    seat: &[crate::llama_launch::EnvFlag],
) -> Result<(Flags, Vec<A>, Vec<String>), GateError>
where
    A: AsRef<OsStr> + Clone,
{
    let mut flags = Flags::default();
    let mut rest = Vec::with_capacity(args.len());
    let mut typed: Vec<String> = Vec::new();
    let mut place = false;
    let mut it = args.iter();
    while let Some(raw) = it.next() {
        let Some(arg) = raw.as_ref().to_str() else {
            rest.push(raw.clone());
            continue;
        };
        let name = spelled(arg);
        if !FLAGS.iter().any(|set| set.contains(&name.as_str())) {
            place |= arg == "--place";
            rest.push(raw.clone());
            let takes_value = arg == "--place"
                || (seat.iter().any(|read| read.spelled.contains(&arg))
                    && crate::llama_launch::serve_takes_value(arg));
            if takes_value {
                rest.extend(it.next().cloned());
            }
            continue;
        }
        if CPU_MOE.contains(&name.as_str()) {
            flags.set_moe(arg, Moe::All)?;
            flags.given.push(arg.to_owned());
        } else {
            let v = it
                .next()
                .ok_or_else(|| format!("{arg} needs a value"))?
                .as_ref()
                .to_str()
                .ok_or_else(|| format!("{arg}: the value is not UTF-8"))?;
            flags.set(arg, &name, v)?;
            flags.given.push(format!("{arg} {v}"));
        }
        typed.push(name);
    }
    if !flags.given.is_empty() && place {
        return Err(format!(
            "--place beside llama-server's placement flags ({}): name the cards one way",
            flags.given.join(" ")
        )
        .into());
    }
    let notes = read_env(&mut flags, env, &typed, place)?;
    Ok((flags, rest, notes))
}

/// The variables of `env` read into `flags` as the flags they stand for, in table order. A variable
/// is replaced, not read, when the command line gave its flag (`typed` holds the spelled names of
/// the flags it gave) or, under `place`, when it names cards; a line says each replacement.
fn read_env(
    flags: &mut Flags,
    env: &[(String, String)],
    typed: &[String],
    place: bool,
) -> Result<Vec<String>, GateError> {
    let mut notes = Vec::new();
    for (set, var) in ENV {
        let Some((_, value)) = env.iter().rev().find(|(k, _)| k.as_str() == var) else {
            continue;
        };
        let name = set[0];
        if typed.iter().any(|t| set.contains(&t.as_str())) {
            notes.push(format!(
                "{var} is set, and the command line's {name} replaces it"
            ));
        } else if place && NAMES_CARDS.contains(&set) {
            notes.push(format!(
                "{var} is set, and --place replaces it: the cards come from --place"
            ));
        } else {
            read_var(flags, var, name, value).map_err(|e| named_by(var, &e))?;
        }
    }
    Ok(notes)
}

/// One variable as its flag `name`: the flag's own reading of `value`, the variable standing in
/// for the flag's spelling in what it prints.
fn read_var(flags: &mut Flags, var: &str, name: &str, value: &str) -> Result<(), GateError> {
    if CPU_MOE.contains(&name) {
        if on_off(var, value)? {
            flags.set_moe(var, Moe::All)?;
            flags.given.push(var.to_owned());
        }
        return Ok(());
    }
    flags.set(var, name, value)?;
    flags.given.push(format!("{var} {value}"));
    Ok(())
}

/// `e`'s text under the variable's name, unless it starts with it already.
fn named_by(var: &str, e: &GateError) -> GateError {
    let text = e.to_string();
    if text.starts_with(var) {
        text.into()
    } else {
        format!("{var}: {text}").into()
    }
}

/// `--list-devices`' text: the devices as llama-server prints them, `CUDA<N>: <name> (<usable> MiB,
/// <free> MiB free)`, the first figure the bytes a plan may use of the device.
pub fn list_devices(census: &[DeviceInfo]) -> String {
    let mut text = String::from("Available devices:\n");
    if census.is_empty() {
        text.push_str("  (none)\n");
        return text;
    }
    for d in by_ordinal(census) {
        text.push_str(&format!(
            "  CUDA{}: {} ({} MiB, {} MiB free)\n",
            d.ordinal,
            d.name,
            spec_of_device(d).usable_bytes() / MIB,
            d.free_bytes / MIB
        ));
    }
    text
}

impl Flags {
    /// No placement flag was given.
    pub fn is_empty(&self) -> bool {
        self.given.is_empty()
    }

    /// A flag compares the file's layer count (`-ngl N`, `-ncmoe N`): the caller reads it.
    pub fn needs_layers(&self) -> bool {
        matches!(self.ngl, Some(Ngl::Layers(_)))
            || matches!(&self.moe, Some((_, Moe::First(n))) if *n > 0)
    }

    /// The census is read to resolve a card (`-dev`, `-sm`, `-ts`, `-mg`, `-devd`) or its bytes
    /// (`-fitt`).
    pub fn needs_census(&self) -> bool {
        self.names_cards() || self.draft.is_some() || self.fit_target.is_some()
    }

    fn names_cards(&self) -> bool {
        self.dev.is_some()
            || self.split == Some(Split::None)
            || self.shares.is_some()
            || self.main_gpu.is_some_and(|g| g != 0)
    }

    fn set(&mut self, arg: &str, name: &str, v: &str) -> Result<(), GateError> {
        let is = |set: &[&str]| set.contains(&name);
        if is(DEVICE) {
            once(&mut self.dev, arg, "the device list", devices(arg, v)?)?;
        } else if is(SPLIT) {
            once(&mut self.split, arg, "the split mode", split_mode(v)?)?;
        } else if is(TENSOR_SPLIT) {
            once(
                &mut self.shares,
                arg,
                "the split",
                (format!("{arg} {v}"), shares(v)?),
            )?;
        } else if is(MAIN_GPU) {
            let g = v
                .parse()
                .map_err(|_| format!("{arg} {v:?}: no device index"))?;
            once(&mut self.main_gpu, arg, "the main device", g)?;
        } else if is(GPU_LAYERS) {
            once(&mut self.ngl, arg, "the layer count", gpu_layers(v)?)?;
        } else if is(N_CPU_MOE) {
            let n = v
                .parse()
                .map_err(|_| format!("{arg} {v:?}: no layer count"))?;
            self.set_moe(arg, Moe::First(n))?;
        } else if is(OVERRIDE_TENSOR) {
            override_to_cpu(v)?;
            self.set_moe(&format!("{arg} {v}"), Moe::All)?;
        } else if is(FIT) {
            once(&mut self.fit, arg, "--fit", on_off(arg, v)?)?;
        } else if is(FIT_TARGET) {
            once(
                &mut self.fit_target,
                arg,
                "the fit margin",
                (format!("{arg} {v}"), margins(arg, v)?),
            )?;
        } else if is(FIT_CTX) {
            let n = v
                .parse()
                .map_err(|_| format!("{arg} {v:?}: no context size"))?;
            once(&mut self.fit_ctx, arg, "--fit-ctx", n)?;
        } else if is(DEVICE_DRAFT) {
            once(
                &mut self.draft,
                arg,
                "the draft's device",
                draft_device(arg, v)?,
            )?;
        }
        if is(DEVICE) || is(SPLIT) || is(TENSOR_SPLIT) || is(MAIN_GPU) {
            self.placing.push(format!("{arg} {v}"));
        }
        Ok(())
    }

    /// The expert flag `flag` asked for `rule`; one flag names the experts' place.
    fn set_moe(&mut self, flag: &str, rule: Moe) -> Result<(), GateError> {
        once(
            &mut self.moe,
            flag,
            "the routed experts' place",
            (flag.to_owned(), rule),
        )
    }

    /// What the flags that need no device come to: the refusals of `-fit off`, `-fitc`, `-ngl`, and
    /// the levers `--cpu-moe` (`-ncmoe`, `-ot`) and `-devd` are.
    pub fn levers(&self, t: &Target<'_>) -> Result<Levers, GateError> {
        if self.fit == Some(false) {
            return Err(
                "-fit off: bloomery plans every load to fit (each card takes the experts its \
                 budget holds and the host tier the rest), so it has no unfitted load; leave -fit \
                 at its default, on, and size the load with -dev or -sm none -mg N (the card), \
                 -fitt MiB (the margin kept free on it) and --ctx N (the context)"
                    .into(),
            );
        }
        if let Some(n) = self.fit_ctx
            && n != t.ctx_floor
        {
            return Err(format!(
                "-fitc {n}: the least context an unset --ctx steps down to is {} on every seat; \
                 set --ctx to pin a context",
                t.ctx_floor
            )
            .into());
        }
        if let Some(Ngl::Layers(n)) = self.ngl {
            full_offload(n, t.layers)?;
        }
        let mut out = Levers::default();
        self.experts(t, &mut out)?;
        self.draft_lever(t, &mut out)?;
        Ok(out)
    }

    /// `--cpu-moe` and what equals it: the whole set of routed experts on the host, which only the
    /// qwen38 seat has a lever for.
    fn experts(&self, t: &Target<'_>, out: &mut Levers) -> Result<(), GateError> {
        let Some((flag, rule)) = &self.moe else {
            return Ok(());
        };
        match *rule {
            Moe::First(0) => return Ok(()),
            Moe::First(n) => every_layer(flag, n, t.layers)?,
            Moe::All => {}
        }
        let Some(lever) = experts_lever(t.seat) else {
            return Err(format!(
                "{flag}: this seat has no switch that keeps every routed expert on the host; its \
                 plan puts on each card the experts the card's budget holds and leaves the rest \
                 to the host tier"
            )
            .into());
        };
        out.set.push((lever, "host".to_owned()));
        out.notes.push(format!(
            "llama flag {flag} sets {lever}=host: every routed expert is planned on the host tier"
        ));
        Ok(())
    }

    /// `-devd`: the card the DSpark draft opens on, where the seat has a draft that opens on a named
    /// card.
    fn draft_lever(&self, t: &Target<'_>, out: &mut Levers) -> Result<(), GateError> {
        let Some(ordinal) = self.draft else {
            return Ok(());
        };
        if !names_draft_card(t.seat) {
            return Err(format!(
                "-devd CUDA{ordinal}: this seat's draft opens on no card a flag names"
            )
            .into());
        }
        let word = format!("cuda{ordinal}");
        out.notes.push(format!(
            "llama flag -devd sets {DRAFT_CARD}={word}: the DSpark draft, where \
             BLOOMERY_DRAFT=dspark asks for one, opens on it"
        ));
        out.set.push((DRAFT_CARD, word));
        Ok(())
    }

    /// The cards the flags name, resolved on `census`: `None` where they name none (`-sm layer` and
    /// the flags that are no card's leave the seat's own rule to pick). The listed devices are
    /// `-dev`'s, else every visible one in CUDA order; `-sm none` keeps the one `-mg` indexes,
    /// `-sm layer` those with a share under `-ts`. The largest runs every layer and the rest hold
    /// experts, in size order.
    pub fn place(&self, census: &[DeviceInfo]) -> Result<Option<Cards>, GateError> {
        if let Some(o) = self.draft {
            device(census, o, "-devd")?;
        }
        let Some((chosen, split)) = self.chosen(census)? else {
            return Ok(None);
        };
        let all = ranked(census)?;
        let order: Vec<u32> = all
            .iter()
            .copied()
            .filter(|o| chosen.iter().any(|d| d.ordinal == *o))
            .collect();
        let word = match order.as_slice() {
            [_] if order == all[..1] => "a".to_owned(),
            [_, _] if order == all[..2] => "bp".to_owned(),
            _ => order
                .iter()
                .map(|o| format!("cuda{o}"))
                .collect::<Vec<_>>()
                .join("+"),
        };
        let notes = self.cards_notes(&word, &order, census, split);
        Ok(Some(Cards { word, notes }))
    }

    /// `-fitt`: the margin llama-server keeps free on each device, as the margin each card keeps
    /// free past its context and scratch (`BLOOMERY_CARD_MARGIN`, which replaces the plan's own
    /// and leaves the card's usable bytes whole): one value is every card's, a list one per
    /// listed device (`-dev`'s, else every visible one in CUDA order). It names the cards the
    /// flags place on, or every listed card where they name none, so the load's own rule picks
    /// among them, each planned with its margin.
    pub fn margin(&self, census: &[DeviceInfo]) -> Result<Levers, GateError> {
        let Some((flag, mib)) = &self.fit_target else {
            return Ok(Levers::default());
        };
        let listed = self.listed(census)?;
        if mib.len() != 1 && mib.len() != listed.len() {
            return Err(format!(
                "{flag}: {} margins for the {} listed device(s): one for every device, or one each",
                mib.len(),
                listed.len()
            )
            .into());
        }
        let cards = match self.chosen(census)? {
            Some((cards, _)) => cards,
            None => listed.clone(),
        };
        let mut per: Vec<(u32, u64)> = Vec::with_capacity(cards.len());
        let mut terms: Vec<String> = Vec::with_capacity(cards.len());
        for card in cards {
            let o = card.ordinal;
            let at = listed
                .iter()
                .position(|d| d.ordinal == o)
                .ok_or_else(|| format!("{flag}: CUDA{o} is placed on and not listed"))?;
            let m = if mib.len() == 1 { mib[0] } else { mib[at] };
            let usable = spec_of_device(card).usable_bytes();
            let margin = m.checked_mul(MIB).filter(|b| *b < usable).ok_or_else(|| {
                format!(
                    "{flag}: CUDA{o} ({}) has {} MiB a plan may use, and a margin of {m} MiB \
                     leaves none",
                    card.name,
                    usable / MIB
                )
            })?;
            per.push((o, margin));
            terms.push(format!(
                "CUDA{o} ({}) keeps {m} MiB of its {} MiB free",
                card.name,
                usable / MIB
            ));
        }
        let first = per.first().map(|&(_, bytes)| bytes);
        let margin = match first {
            Some(bytes) if mib.len() == 1 => bloomery_levers::CardBudget::Every(bytes),
            _ => bloomery_levers::CardBudget::Per(per),
        };
        let name = bloomery_levers::CARD_MARGIN;
        Ok(Levers {
            notes: vec![format!(
                "llama flag {flag} sets {name}={margin}: {}; the margin replaces the plan's own \
                 ({} MiB) and is kept free beside the card's context and scratch",
                terms.join(", "),
                workstation::MARGIN / MIB
            )],
            set: vec![(name, margin.to_string())],
        })
    }

    /// `-ts` over the cards the flags place on: the shares as `BLOOMERY_CARD_BUDGET` shares, one
    /// per placed card by CUDA ordinal in whole millionths of the total, which the plan turns
    /// into a budget per card from its own expert bytes (`card_budget::split`): the cards hold the
    /// routed experts in the proportion of the shares, as many in all as the card whose room is
    /// tightest for its share lets it, and the host tier the rest. One card left, or `-sm none`,
    /// places nothing and sets nothing.
    pub fn budget(&self, census: &[DeviceInfo]) -> Result<Levers, GateError> {
        let Some((flag, shares)) = &self.shares else {
            return Ok(Levers::default());
        };
        let Some((cards, Split::Layer)) = self.chosen(census)? else {
            return Ok(Levers::default());
        };
        if cards.len() < 2 {
            return Ok(Levers::default());
        }
        let listed = self.listed(census)?;
        let mut given: Vec<(u32, f64)> = Vec::with_capacity(cards.len());
        for card in &cards {
            let at = listed
                .iter()
                .position(|d| d.ordinal == card.ordinal)
                .ok_or_else(|| {
                    format!("{flag}: CUDA{} is placed on and not listed", card.ordinal)
                })?;
            let share = shares.get(at).copied().unwrap_or(0.0);
            given.push((card.ordinal, share));
        }
        let total: f64 = given.iter().map(|&(_, s)| s).sum();
        let mut parts: Vec<(u32, u64)> = Vec::with_capacity(given.len());
        for &(o, share) in &given {
            let part = (share / total * SHARE_PARTS).round() as u64;
            if part == 0 {
                return Err(format!(
                    "{flag}: the share {share} of CUDA{o} is under one part in a million of the \
                     {total} given, which bloomery's shares do not resolve"
                )
                .into());
            }
            parts.push((o, part));
        }
        let budget = bloomery_levers::CardBudget::Shares(parts);
        let name = bloomery_levers::CARD_BUDGET;
        let ratio: Vec<String> = given.iter().map(|&(_, s)| s.to_string()).collect();
        let names: Vec<String> = cards
            .iter()
            .map(|c| format!("CUDA{} ({})", c.ordinal, c.name))
            .collect();
        Ok(Levers {
            notes: vec![format!(
                "llama flag {flag} sets {name}={budget}: the plan holds the routed experts on {} in \
                 the proportion {}, as many in all as the card whose room is tightest for its share \
                 lets it, and the host tier the rest",
                names.join(", "),
                ratio.join(":")
            )],
            set: vec![(name, budget.to_string())],
        })
    }

    /// The devices the flags name and the split they name them under; `None` where they name none.
    fn chosen<'a>(
        &self,
        census: &'a [DeviceInfo],
    ) -> Result<Option<(Vec<&'a DeviceInfo>, Split)>, GateError> {
        if !self.names_cards() {
            return Ok(None);
        }
        let listed = self.listed(census)?;
        let split = self.split.unwrap_or(Split::Layer);
        let chosen = match split {
            Split::None => vec![self.main_card(&listed)?],
            Split::Layer => self.shared(&listed)?,
        };
        Ok(Some((chosen, split)))
    }

    /// The devices `-dev` lists, in its order; every visible device in CUDA order without it.
    fn listed<'a>(&self, census: &'a [DeviceInfo]) -> Result<Vec<&'a DeviceInfo>, GateError> {
        let listed: Vec<&DeviceInfo> = match &self.dev {
            Some(ordinals) => ordinals
                .iter()
                .map(|&o| device(census, o, "-dev"))
                .collect::<Result<_, _>>()?,
            None => by_ordinal(census),
        };
        if listed.is_empty() {
            return Err("no CUDA device is visible to place the model on".into());
        }
        Ok(listed)
    }

    /// `-sm none`: the listed device `-mg` indexes (the first by default).
    fn main_card<'a>(&self, listed: &[&'a DeviceInfo]) -> Result<&'a DeviceInfo, GateError> {
        let at = self.main_gpu.unwrap_or(0);
        usize::try_from(at)
            .ok()
            .and_then(|i| listed.get(i).copied())
            .ok_or_else(|| {
                let cpu = if at < 0 {
                    "; a negative index runs no card, and bloomery has no CPU-only seat"
                } else {
                    ""
                };
                format!(
                    "-mg {at}: the device list holds {} device(s), indexed from 0{cpu}",
                    listed.len()
                )
                .into()
            })
    }

    /// `-sm layer`: the listed devices with a share under `-ts`, all of them without it.
    fn shared<'a>(&self, listed: &[&'a DeviceInfo]) -> Result<Vec<&'a DeviceInfo>, GateError> {
        if let Some(g) = self.main_gpu
            && g != 0
        {
            return Err(format!(
                "-mg {g}: llama reads -mg under -sm none (and row) only, and names no card under \
                 -sm layer; bloomery's stage is the largest listed card (-sm none -mg {g}, or \
                 -dev CUDA<N>, runs one card alone)"
            )
            .into());
        }
        let Some((_, shares)) = &self.shares else {
            return Ok(listed.to_vec());
        };
        if shares.len() > listed.len() {
            return Err(format!(
                "-ts names {} shares for the {} listed device(s)",
                shares.len(),
                listed.len()
            )
            .into());
        }
        let kept: Vec<&DeviceInfo> = listed
            .iter()
            .enumerate()
            .filter(|&(i, _)| shares.get(i).is_some_and(|s| *s > 0.0))
            .map(|(_, d)| *d)
            .collect();
        if kept.is_empty() {
            return Err("-ts gives no listed device a share".into());
        }
        Ok(kept)
    }

    /// The lines a placement prints: what was read as what, the layer rule where cards share the
    /// model, and the proportion rule where `-ts` was given.
    fn cards_notes(
        &self,
        word: &str,
        order: &[u32],
        census: &[DeviceInfo],
        split: Split,
    ) -> Vec<String> {
        let named = |o: u32| {
            let name = census
                .iter()
                .find(|d| d.ordinal == o)
                .map_or("?", |d| d.name.as_str());
            format!("CUDA{o} ({name})")
        };
        let tiers: Vec<String> = order[1..].iter().map(|&o| named(o)).collect();
        let mut line = format!(
            "llama flags ({}): --place {word}, stage {}",
            self.placing.join(" "),
            named(order[0])
        );
        if !tiers.is_empty() {
            line.push_str(&format!(", expert tier {}", tiers.join(", ")));
        }
        let mut notes = vec![line];
        if !tiers.is_empty() {
            notes.push(
                "-sm layer: bloomery does not split layers; the stage card runs every layer and \
                 the head, and it fills the other listed cards with experts"
                    .to_owned(),
            );
        }
        if self.shares.is_some() {
            notes.push(if split == Split::None {
                "-ts: one device runs the model under -sm none, so the shares place nothing"
                    .to_owned()
            } else if order.len() < 2 {
                "-ts: one card is left with a share, so the shares place nothing".to_owned()
            } else {
                "-ts: a card with a zero or missing share is left out; the cards that stay hold \
                 the routed experts in the proportion of their shares (the line on \
                 BLOOMERY_CARD_BUDGET says how)"
                    .to_owned()
            });
        }
        notes
    }
}

/// The lever that keeps every routed expert on the host, on the seat that has one.
fn experts_lever(seat: &str) -> Option<&'static str> {
    (seat == "qwen38").then_some(bloomery_levers::QWEN38_EXPERTS)
}

/// Whether the seat opens its draft on the card `BLOOMERY_DSPARK_CARD` names.
fn names_draft_card(seat: &str) -> bool {
    seat == "ds41"
}

/// `flag`'s slot filled once; a second flag for it is refused by name.
fn once<T>(slot: &mut Option<T>, flag: &str, what: &str, value: T) -> Result<(), GateError> {
    if slot.replace(value).is_some() {
        return Err(format!("{flag}: {what} is named twice").into());
    }
    Ok(())
}

/// `arg` as llama-server matches it: a long flag's `_` is `-`.
pub(crate) fn spelled(arg: &str) -> String {
    if arg.starts_with("--") {
        arg.replace('_', "-")
    } else {
        arg.to_owned()
    }
}

/// The census in CUDA order, which is llama-server's device order.
fn by_ordinal(census: &[DeviceInfo]) -> Vec<&DeviceInfo> {
    let mut all: Vec<&DeviceInfo> = census.iter().collect();
    all.sort_by_key(|d| d.ordinal);
    all
}

/// The device `ordinal` of `census`, refused by name with every visible device when there is none.
fn device<'a>(
    census: &'a [DeviceInfo],
    ordinal: u32,
    flag: &str,
) -> Result<&'a DeviceInfo, GateError> {
    census.iter().find(|d| d.ordinal == ordinal).ok_or_else(|| {
        format!(
            "{flag} CUDA{ordinal}: no such device (visible: {})",
            workstation::visible(census)
        )
        .into()
    })
}

/// Every device's ordinal by size, the largest first ([`Pick::Rank`]'s order: the one rule the
/// placement words rank by).
fn ranked(census: &[DeviceInfo]) -> Result<Vec<u32>, GateError> {
    let picks = (0..census.len())
        .map(|k| u8::try_from(k).map(Pick::Rank))
        .collect::<Result<Vec<_>, _>>()?;
    let specs = workstation::resolve(&picks, census)?;
    Ok(specs
        .iter()
        .filter_map(|s| s.device.map(|d| d.ordinal))
        .collect())
}

/// `-dev`'s value: `CUDA<N>` names, each once. `none` (no device at all) is refused: no seat runs
/// on the CPU alone.
fn devices(flag: &str, v: &str) -> Result<Vec<u32>, GateError> {
    if v == "none" {
        return Err(format!("{flag} none: no seat runs on the CPU alone").into());
    }
    let mut out: Vec<u32> = Vec::new();
    for name in v.split(',') {
        let ordinal = name
            .strip_prefix("CUDA")
            .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|d| d.parse().ok())
            .ok_or_else(|| {
                format!(
                    "{flag} {v:?}: {name:?} is no CUDA device (CUDA0, CUDA1, ... as --list-devices \
                     prints them)"
                )
            })?;
        if out.contains(&ordinal) {
            return Err(format!("{flag} {v:?}: CUDA{ordinal} twice").into());
        }
        out.push(ordinal);
    }
    Ok(out)
}

/// `-devd`'s value: one `CUDA<N>`; the draft opens on one card.
fn draft_device(flag: &str, v: &str) -> Result<u32, GateError> {
    match devices(flag, v)?.as_slice() {
        &[one] => Ok(one),
        _ => Err(format!("{flag} {v:?}: the draft opens on one card").into()),
    }
}

/// `-sm`'s value; `row` and `tensor` are refused by name.
fn split_mode(v: &str) -> Result<Split, GateError> {
    match v {
        "none" => Ok(Split::None),
        "layer" => Ok(Split::Layer),
        "row" => Err("-sm row: bloomery has no row split; its stage card runs every layer".into()),
        "tensor" => Err("-sm tensor: bloomery's tensor parallel is --place tp over \
                         BLOOMERY_TP_NODES (one node a rank), not this process's cards"
            .into()),
        other => Err(format!("-sm {other:?}: none or layer").into()),
    }
}

/// `-ts`'s value: shares from 0 up, separated by `,` or `/`.
fn shares(v: &str) -> Result<Vec<f64>, GateError> {
    let shares = v
        .split([',', '/'])
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| format!("-ts {v:?}: {s:?} is no share (a number from 0 up)"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shares.is_empty() {
        return Err("-ts names no share".into());
    }
    Ok(shares)
}

/// `-ngl`'s value: a count, `auto` or `all`; a negative count is llama-server's auto.
fn gpu_layers(v: &str) -> Result<Ngl, GateError> {
    match v {
        "auto" | "all" => Ok(Ngl::All),
        _ => v
            .parse::<i64>()
            .map(|n| u64::try_from(n).map_or(Ngl::All, Ngl::Layers))
            .map_err(|_| format!("-ngl {v:?}: a count, auto or all").into()),
    }
}

/// `-ngl N` is a full offload: llama-server counts the output layer as one more, so every layer
/// and the output are `N >= layers + 1`. A smaller count leaves a layer on the CPU, which no seat
/// does.
fn full_offload(n: u64, layers: Option<u64>) -> Result<(), GateError> {
    let layers =
        layers.ok_or("-ngl N is compared with the file's block_count, and the file states none")?;
    if n > layers {
        return Ok(());
    }
    Err(format!(
        "-ngl {n}: llama-server offloads {n} of the file's {layers} layers plus the output layer, \
         and bloomery runs every layer on its stage card; use -ngl all, auto, or {} and more",
        layers + 1
    )
    .into())
}

/// `-ncmoe N` of every layer is `--cpu-moe`; fewer is a rule bloomery has no thing for.
fn every_layer(flag: &str, n: u64, layers: Option<u64>) -> Result<(), GateError> {
    let layers = layers.ok_or_else(|| {
        format!("{flag} is compared with the file's block_count, and the file states none")
    })?;
    if n >= layers {
        return Ok(());
    }
    Err(format!(
        "{flag}: keeps the experts of the first {n} of {layers} layers on the CPU, and bloomery's \
         plan puts each layer's experts on the cards by budget, with no first-N-layers rule; \
         --cpu-moe, or -ncmoe of {layers} and more, is the whole set"
    )
    .into())
}

/// `-ot`'s value: every `<pattern>=CPU` names all routed experts ([`ALL_EXPERTS`]); any other
/// pattern or buffer type is refused by name.
fn override_to_cpu(v: &str) -> Result<(), GateError> {
    for part in v.split(',') {
        let (pattern, buffer) = part
            .split_once('=')
            .ok_or_else(|| format!("-ot {v:?}: <pattern>=<buffer type> expected"))?;
        if buffer != "CPU" {
            return Err(format!(
                "-ot {part:?}: bloomery places tensors by its plan, and the buffer type {buffer:?} \
                 names a device; only <all experts>=CPU is read"
            )
            .into());
        }
        if !ALL_EXPERTS.contains(&pattern) {
            return Err(format!(
                "-ot {part:?}: bloomery cannot select tensors by regex; the patterns for all \
                 routed experts it reads are {ALL_EXPERTS:?}"
            )
            .into());
        }
    }
    Ok(())
}

/// `-fit`'s value, in llama-server's words.
fn on_off(flag: &str, v: &str) -> Result<bool, GateError> {
    match v {
        "on" | "enabled" | "true" | "1" => Ok(true),
        "off" | "disabled" | "false" | "0" => Ok(false),
        _ => Err(format!("{flag} {v:?}: on or off").into()),
    }
}

/// `-fitt`'s value: MiB counts split by `,` or `/` as llama-server splits them, one for every
/// device or one for each.
fn margins(flag: &str, v: &str) -> Result<Vec<u64>, GateError> {
    let mut out = Vec::new();
    for m in v.split([',', '/']) {
        out.push(
            m.parse()
                .map_err(|_| format!("{flag} {v:?}: {m:?} is no MiB count"))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use model::placement::workstation::{ALIASES, census_usable};

    use super::*;

    const A6000: &str = "NVIDIA RTX A6000";
    const RTX_3090: &str = "NVIDIA GeForce RTX 3090";

    /// The census totals the A6000 and the 3090 report: the figures of `generate`'s tests.
    fn total(name: &str) -> u64 {
        match name {
            A6000 => 50_952_536_064,
            RTX_3090 => 25_351_356_416,
            other => panic!("no figure for {other}"),
        }
    }

    /// A census of idle cards, by driver name, ordinals in the order given.
    fn census(names: &[&str]) -> Vec<DeviceInfo> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| DeviceInfo {
                ordinal: u32::try_from(i).expect("small"),
                name: (*n).to_owned(),
                total_bytes: total(n),
                free_bytes: census_usable(total(n)),
                integrated: false,
                uuid: [u8::try_from(i).expect("small") + 1; 16],
                pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
                held_by: None,
                compute_capability: Some((8, 6)),
            })
            .collect()
    }

    /// The A6000 as CUDA0 and the 3090 as CUDA1, and the other enumeration order.
    fn two() -> Vec<DeviceInfo> {
        census(&[A6000, RTX_3090])
    }

    fn flags(line: &str) -> Flags {
        let args: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        take(&args).unwrap_or_else(|e| panic!("{line}: {e}")).0
    }

    fn refused(line: &str) -> String {
        let args: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        take(&args).expect_err(line).to_string()
    }

    fn env_of(env: &[(&str, &str)]) -> Vec<(String, String)> {
        env.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    /// `line` and `env` as [`take_with_env`] reads them: the flags and the lines about replaced
    /// variables.
    fn env_flags(line: &str, env: &[(&str, &str)]) -> (Flags, Vec<String>) {
        let args: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        let env = env_of(env);
        let (flags, _, notes) =
            take_with_env(&args, &env, &[]).unwrap_or_else(|e| panic!("{line} {env:?}: {e}"));
        (flags, notes)
    }

    fn env_refused(line: &str, env: &[(&str, &str)]) -> String {
        let args: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        let env = env_of(env);
        take_with_env(&args, &env, &[]).expect_err(line).to_string()
    }

    /// A target seat: the qwen38 seat has the experts lever, the ds41 seat the draft card, any
    /// other neither; the file has 61 layers and the floor is 4096.
    fn target(seat: &str) -> Target<'_> {
        Target {
            seat,
            layers: Some(61),
            ctx_floor: 4096,
        }
    }

    fn levers(line: &str, seat: &str) -> Result<Levers, String> {
        flags(line).levers(&target(seat)).map_err(|e| e.to_string())
    }

    fn word(line: &str, census: &[DeviceInfo]) -> Option<String> {
        flags(line)
            .place(census)
            .unwrap_or_else(|e| panic!("{line}: {e}"))
            .map(|c| c.word)
    }

    fn placed_err(line: &str, census: &[DeviceInfo]) -> String {
        flags(line).place(census).expect_err(line).to_string()
    }

    /// The ordinals a word means on `census`: an alias by its picks, a list by its parts.
    fn devices_of(word: &str, census: &[DeviceInfo]) -> Vec<u32> {
        let picks = match ALIASES.iter().find(|(w, _)| *w == word) {
            Some((_, picks)) => picks.to_vec(),
            None => workstation::word_picks(word, 8).expect(word),
        };
        workstation::resolve(&picks, census)
            .expect(word)
            .iter()
            .filter_map(|s| s.device.map(|d| d.ordinal))
            .collect()
    }

    /// `--list-devices` prints llama-server's shape: a header, then a line a device in CUDA order,
    /// the usable MiB and the free MiB; with no device, `(none)`.
    #[test]
    fn list_devices_prints_llama_servers_shape() {
        assert_eq!(
            list_devices(&two()),
            "Available devices:\n  CUDA0: NVIDIA RTX A6000 (48592 MiB, 48592 MiB free)\n  \
             CUDA1: NVIDIA GeForce RTX 3090 (24176 MiB, 24176 MiB free)\n"
        );
        assert_eq!(list_devices(&[]), "Available devices:\n  (none)\n");
    }

    /// Every spelling of every flag is cut out, `_` for `-` in a long one included, and every
    /// other argument stays in its place; a command line with no placement flag is as it was.
    #[test]
    fn take_cuts_every_spelling_and_leaves_the_rest() {
        let args: Vec<String> = "-m x.gguf -dev CUDA0 --split-mode none --tensor-split 1 \
                                 --main_gpu 0 --n_gpu_layers 99 --cpu-moe --fit on --fit-ctx \
                                 4096 --spec-draft-device CUDA1 --port 1"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        let (f, rest) = take(&args).expect("every spelling");
        assert_eq!(rest, ["-m", "x.gguf", "--port", "1"]);
        assert!(!f.is_empty() && f.needs_census());
        let plain: Vec<String> = ["-m", "x.gguf", "--place", "bp", "-c", "4096"]
            .map(str::to_owned)
            .to_vec();
        let (f, rest) = take(&plain).expect("no llama flag");
        assert!(f.is_empty() && !f.needs_census() && !f.needs_layers());
        assert_eq!(rest, plain);
    }

    /// A flag with no value, one given twice (under two spellings too), and a value llama-server
    /// would refuse are named refusals.
    #[test]
    fn a_malformed_flag_is_refused_by_name() {
        assert_eq!(refused("-dev"), "-dev needs a value");
        assert_eq!(
            refused("-dev CUDA0 --device CUDA1"),
            "--device: the device list is named twice"
        );
        assert_eq!(
            refused("--cpu-moe -ncmoe 3"),
            "-ncmoe: the routed experts' place is named twice"
        );
        assert!(refused("-ts 1,x").contains("\"x\" is no share"));
        assert!(refused("-ts -1,1").contains("is no share"));
        assert!(refused("-mg one").contains("no device index"));
        assert!(refused("-ngl many").contains("a count, auto or all"));
        assert!(refused("-sm half").contains("none or layer"));
        assert!(refused("-fit maybe").contains("on or off"));
    }

    /// `--place` beside any llama placement flag is refused, whichever comes first; `--place`
    /// alone is the old spelling and stays.
    #[test]
    fn place_beside_a_llama_flag_is_refused() {
        for line in ["--place bp -ngl 99", "-dev CUDA0 --place a"] {
            assert!(
                refused(line).starts_with("--place beside llama-server's placement flags"),
                "{line}"
            );
        }
        assert!(flags("--place bp").is_empty());
    }

    /// Each flag's llama-server variable is read as that flag, by the flag's own reading of the
    /// value, and the variable stands in for the flag in what is printed; every flag but `-devd`
    /// has a variable, each named once.
    #[test]
    fn each_llama_variable_is_its_flags_value() {
        let one = |var: &str, value: &str| env_flags("", &[(var, value)]).0;
        assert_eq!(one("LLAMA_ARG_DEVICE", "CUDA1,CUDA0").dev, Some(vec![1, 0]));
        assert_eq!(one("LLAMA_ARG_SPLIT_MODE", "none").split, Some(Split::None));
        assert_eq!(
            one("LLAMA_ARG_TENSOR_SPLIT", "3/1").shares,
            Some(("LLAMA_ARG_TENSOR_SPLIT 3/1".to_owned(), vec![3.0, 1.0]))
        );
        assert_eq!(one("LLAMA_ARG_MAIN_GPU", "1").main_gpu, Some(1));
        assert_eq!(
            one("LLAMA_ARG_N_GPU_LAYERS", "99").ngl,
            Some(Ngl::Layers(99))
        );
        assert_eq!(
            one("LLAMA_ARG_N_CPU_MOE", "7").moe.map(|m| m.1),
            Some(Moe::First(7))
        );
        assert_eq!(
            one("LLAMA_ARG_CPU_MOE", "1").moe.map(|m| m.1),
            Some(Moe::All)
        );
        assert_eq!(
            one("LLAMA_ARG_OVERRIDE_TENSOR", "exps=CPU")
                .moe
                .map(|m| m.1),
            Some(Moe::All)
        );
        assert_eq!(one("LLAMA_ARG_FIT", "off").fit, Some(false));
        assert_eq!(
            one("LLAMA_ARG_FIT_TARGET", "512,1024")
                .fit_target
                .map(|t| t.1),
            Some(vec![512, 1024])
        );
        assert_eq!(one("LLAMA_ARG_FIT_CTX", "4096").fit_ctx, Some(4096));
        assert_eq!(
            one("LLAMA_ARG_N_GPU_LAYERS", "99").given,
            ["LLAMA_ARG_N_GPU_LAYERS 99"]
        );
        for set in FLAGS {
            let var = ENV.iter().find(|(s, _)| *s == set).map(|(_, v)| *v);
            assert_eq!(var.is_some(), set != DEVICE_DRAFT, "{set:?}");
        }
        let mut names: Vec<&str> = ENV.iter().map(|(_, v)| *v).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ENV.len(), "a variable names one flag");
    }

    /// A flag on the command line replaces its variable, in whichever spelling it is given, and a
    /// line says so; the replaced value is never read, so one the flag would refuse is no refusal,
    /// and the variables of other flags are still read.
    #[test]
    fn a_flag_on_the_command_line_replaces_its_variable() {
        let (f, notes) = env_flags("-ngl 99", &[("LLAMA_ARG_N_GPU_LAYERS", "1")]);
        assert_eq!(f.ngl, Some(Ngl::Layers(99)));
        assert_eq!(f.given, ["-ngl 99"]);
        assert_eq!(
            notes,
            ["LLAMA_ARG_N_GPU_LAYERS is set, and the command line's -ngl replaces it"]
        );
        let (f, notes) = env_flags("--n-gpu-layers all", &[("LLAMA_ARG_N_GPU_LAYERS", "many")]);
        assert_eq!((f.ngl, notes.len()), (Some(Ngl::All), 1));
        let (f, notes) = env_flags("-fit on", &[("LLAMA_ARG_N_GPU_LAYERS", "all")]);
        assert_eq!((f.fit, f.ngl, notes.len()), (Some(true), Some(Ngl::All), 0));
    }

    /// A variable whose flag would be refused is refused under the variable's name with the flag's
    /// own reason: a value set in the environment is no more lenient than one typed.
    #[test]
    fn a_variable_is_refused_as_its_flag_is() {
        for (var, value, why) in [
            ("LLAMA_ARG_SPLIT_MODE", "row", "no row split"),
            (
                "LLAMA_ARG_SPLIT_MODE",
                "tensor",
                "--place tp over BLOOMERY_TP_NODES",
            ),
            ("LLAMA_ARG_DEVICE", "none", "no seat runs on the CPU alone"),
            ("LLAMA_ARG_DEVICE", "", "is no CUDA device"),
            ("LLAMA_ARG_N_GPU_LAYERS", "many", "a count, auto or all"),
            ("LLAMA_ARG_TENSOR_SPLIT", "1,x", "is no share"),
            ("LLAMA_ARG_MAIN_GPU", "one", "no device index"),
            ("LLAMA_ARG_FIT", "maybe", "on or off"),
            ("LLAMA_ARG_FIT_TARGET", "1g", "is no MiB count"),
            ("LLAMA_ARG_FIT_CTX", "big", "no context size"),
            ("LLAMA_ARG_N_CPU_MOE", "-1", "no layer count"),
            ("LLAMA_ARG_OVERRIDE_TENSOR", "exps=CUDA0", "names a device"),
            ("LLAMA_ARG_CPU_MOE", "maybe", "on or off"),
        ] {
            let e = env_refused("", &[(var, value)]);
            assert!(e.starts_with(var) && e.contains(why), "{var}={value}: {e}");
        }
    }

    /// `LLAMA_ARG_CPU_MOE` is on for llama-server's four truthy words and off, reading nothing,
    /// for its four falsey ones.
    #[test]
    fn the_cpu_moe_variable_is_a_switch() {
        for on in ["on", "enabled", "true", "1"] {
            let (f, _) = env_flags("", &[("LLAMA_ARG_CPU_MOE", on)]);
            let moe = Some(("LLAMA_ARG_CPU_MOE".to_owned(), Moe::All));
            assert_eq!(f.moe, moe, "{on}");
            assert_eq!(f.given, ["LLAMA_ARG_CPU_MOE"], "{on}");
        }
        for off in ["off", "disabled", "false", "0"] {
            let (f, notes) = env_flags("", &[("LLAMA_ARG_CPU_MOE", off)]);
            assert!(f.is_empty() && notes.is_empty(), "{off}");
        }
    }

    /// The routed experts' place is named once, by flag or variable: a variable beside another
    /// source of it is refused under the variable's name, and a switched-off one names nothing.
    #[test]
    fn the_routed_experts_place_is_named_once_across_flags_and_variables() {
        assert_eq!(
            env_refused("-ncmoe 3", &[("LLAMA_ARG_CPU_MOE", "1")]),
            "LLAMA_ARG_CPU_MOE: the routed experts' place is named twice"
        );
        assert_eq!(
            env_refused(
                "",
                &[("LLAMA_ARG_CPU_MOE", "1"), ("LLAMA_ARG_N_CPU_MOE", "3")]
            ),
            "LLAMA_ARG_N_CPU_MOE: the routed experts' place is named twice"
        );
        let (f, _) = env_flags("-ncmoe 3", &[("LLAMA_ARG_CPU_MOE", "0")]);
        assert_eq!(f.moe.map(|m| m.1), Some(Moe::First(3)));
    }

    /// `--place` on the command line replaces the variables that name cards, with a line each,
    /// and no other: a variable that names no card is read beside it. `--place` beside a flag on
    /// the command line is refused as ever.
    #[test]
    fn place_replaces_the_cards_variables_only() {
        let (f, notes) = env_flags(
            "--place bp",
            &[
                ("LLAMA_ARG_DEVICE", "CUDA0"),
                ("LLAMA_ARG_MAIN_GPU", "1"),
                ("LLAMA_ARG_N_GPU_LAYERS", "all"),
            ],
        );
        assert!(f.dev.is_none() && f.main_gpu.is_none() && f.ngl == Some(Ngl::All));
        assert_eq!(
            notes,
            [
                "LLAMA_ARG_DEVICE is set, and --place replaces it: the cards come from --place",
                "LLAMA_ARG_MAIN_GPU is set, and --place replaces it: the cards come from --place"
            ]
        );
        assert!(env_refused("--place bp -ngl 99", &[]).starts_with("--place beside"));
    }

    /// The cards a variable names are the cards its flag names, mixed with flags on the command
    /// line, and the line that says what was read names the variable.
    #[test]
    fn a_variable_names_the_cards_as_its_flag_does() {
        let (f, _) = env_flags("", &[("LLAMA_ARG_DEVICE", "CUDA0,CUDA1")]);
        let cards = f.place(&two()).expect("both").expect("a pick");
        assert_eq!(cards.word, "bp");
        assert_eq!(
            cards.notes[0],
            "llama flags (LLAMA_ARG_DEVICE CUDA0,CUDA1): --place bp, stage CUDA0 (NVIDIA RTX \
             A6000), expert tier CUDA1 (NVIDIA GeForce RTX 3090)"
        );
        let (f, _) = env_flags(
            "-dev CUDA0,CUDA1",
            &[
                ("LLAMA_ARG_SPLIT_MODE", "none"),
                ("LLAMA_ARG_MAIN_GPU", "1"),
            ],
        );
        let cards = f.place(&two()).expect("one").expect("a pick");
        assert_eq!(cards.word, "cuda1");
        assert!(
            cards.notes[0].contains("LLAMA_ARG_MAIN_GPU 1"),
            "{:?}",
            cards.notes
        );
    }

    /// An argument that is no UTF-8 is no flag and stays in its place, byte for byte (a prompt of
    /// bytes); a flag's value that is none is refused by name.
    #[test]
    fn an_argument_that_is_no_utf8_is_left_alone() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let prompt = OsString::from_vec(vec![0x66, 0xff, 0x6f]);
        let args = vec![
            OsString::from("--prompt"),
            prompt.clone(),
            OsString::from("-ngl"),
            OsString::from("all"),
        ];
        let (f, rest, _) = take_with_env(&args, &[], &[]).expect("a prompt of bytes");
        assert_eq!(rest, [OsString::from("--prompt"), prompt]);
        assert_eq!(f.ngl, Some(Ngl::All));
        let bad = vec![OsString::from("-dev"), OsString::from_vec(vec![0xff])];
        let e = take_with_env(&bad, &[], &[]).expect_err("a value of bytes");
        assert_eq!(e.to_string(), "-dev: the value is not UTF-8");
    }

    /// A seat flag's value stays with it only for a caller whose seat takes the flag: a generator's
    /// prompt spelled `--ctx` leaves the placement flag after it to the placement.
    #[test]
    fn only_the_callers_seat_flags_keep_their_values() {
        let args: Vec<String> = ["--prompt", "--ctx", "-ngl", "all", "-n", "1"]
            .map(str::to_owned)
            .to_vec();
        let (f, rest, _) = take_with_env(&args, &[], &[]).expect("a generator's prompt");
        assert_eq!(f.ngl, Some(Ngl::All));
        assert_eq!(rest, ["--prompt", "--ctx", "-n", "1"]);
        let (f, rest, _) =
            take_with_env(&args, &[], crate::llama_launch::SEAT_FLAGS).expect("a server's context");
        assert!(f.is_empty());
        assert_eq!(rest, args);
    }

    /// `-dev` names CUDA devices once each: `none`, another backend's name, a repeat and a device
    /// the census lacks are refused by name.
    #[test]
    fn dev_names_cuda_devices_only() {
        assert!(refused("-dev none").contains("no seat runs on the CPU alone"));
        assert!(refused("-dev Vulkan0").contains("\"Vulkan0\" is no CUDA device"));
        assert!(refused("-dev CUDA").contains("is no CUDA device"));
        assert!(refused("-dev CUDA+1").contains("is no CUDA device"));
        assert!(refused("-dev CUDA0,CUDA0").contains("CUDA0 twice"));
        let e = placed_err("-dev CUDA2", &two());
        assert!(
            e.starts_with("-dev CUDA2: no such device (visible: cuda0 NVIDIA RTX A6000"),
            "{e}"
        );
    }

    /// The cards a flag set names, as the word every seat parses: the largest listed card is the
    /// stage whatever order the list is in, the alias where the cards are the alias's on this
    /// census, and a list of ordinals where they are not; and the word means the devices the
    /// table names, the stage first.
    #[test]
    fn the_flags_name_the_cards_by_size_and_alias() {
        let (two, swapped) = (two(), census(&[RTX_3090, A6000]));
        let three = census(&[A6000, RTX_3090, RTX_3090]);
        type Row<'a> = (&'a str, &'a Vec<DeviceInfo>, Option<(&'a str, Vec<u32>)>);
        let table: [Row<'_>; 15] = [
            ("-sm layer", &two, None),
            ("-ngl 99", &two, None),
            ("-mg 0", &two, None),
            ("-dev CUDA0", &two, Some(("a", vec![0]))),
            ("-dev CUDA1", &two, Some(("cuda1", vec![1]))),
            ("-dev CUDA0,CUDA1", &two, Some(("bp", vec![0, 1]))),
            ("-dev CUDA1,CUDA0", &two, Some(("bp", vec![0, 1]))),
            ("-sm none", &two, Some(("a", vec![0]))),
            ("-sm none -mg 1", &two, Some(("cuda1", vec![1]))),
            (
                "-sm none -mg 1 -dev CUDA1,CUDA0",
                &two,
                Some(("a", vec![0])),
            ),
            ("-sm layer -ts 1,1", &two, Some(("bp", vec![0, 1]))),
            ("-dev CUDA0,CUDA1", &swapped, Some(("bp", vec![1, 0]))),
            ("-sm none -mg 0", &swapped, Some(("cuda0", vec![0]))),
            ("-dev CUDA1", &swapped, Some(("a", vec![1]))),
            (
                "-dev CUDA1,CUDA2",
                &three,
                Some(("cuda1+cuda2", vec![1, 2])),
            ),
        ];
        for (line, census, want) in table {
            let got = word(line, census);
            assert_eq!(got.as_deref(), want.as_ref().map(|(w, _)| *w), "{line}");
            if let (Some(w), Some((_, devices))) = (got, want) {
                assert_eq!(devices_of(&w, census), devices, "{line}: what {w} means");
            }
        }
    }

    /// Cards that share the model print the layer rule beside the placement line: bloomery does not
    /// split layers; one card prints no such line.
    #[test]
    fn layer_split_notes_that_bloomery_does_not_split() {
        let both = flags("-dev CUDA0,CUDA1")
            .place(&two())
            .expect("both")
            .expect("a pick");
        assert_eq!(both.notes.len(), 2);
        assert_eq!(
            both.notes[0],
            "llama flags (-dev CUDA0,CUDA1): --place bp, stage CUDA0 (NVIDIA RTX A6000), expert \
             tier CUDA1 (NVIDIA GeForce RTX 3090)"
        );
        assert!(both.notes[1].starts_with("-sm layer: bloomery does not split layers"));
        let one = flags("-sm none -mg 1")
            .place(&two())
            .expect("one")
            .expect("a pick");
        assert_eq!(one.notes.len(), 1);
    }

    /// `-ts` selects by share and says what the shares come to: a zero share (or a missing
    /// one) leaves a card out, so one share left is one card; more shares than devices, no share
    /// and a share that is no number are refused; under `-sm none` it places nothing and says so.
    #[test]
    fn ts_selects_by_share_and_notes_the_rule() {
        assert_eq!(word("-ts 3,1", &two()).as_deref(), Some("bp"));
        assert_eq!(word("-ts 1,0", &two()).as_deref(), Some("a"));
        assert_eq!(word("-ts 1", &two()).as_deref(), Some("a"));
        assert_eq!(word("-ts 0,1", &two()).as_deref(), Some("cuda1"));
        assert_eq!(word("-ts 1/1", &two()).as_deref(), Some("bp"));
        let last = |line: &str| {
            flags(line)
                .place(&two())
                .expect("placed")
                .expect("a pick")
                .notes
                .pop()
                .expect("a note")
        };
        assert!(
            last("-ts 3,1").starts_with(
                "-ts: a card with a zero or missing share is left out; the cards that stay hold \
                 the routed experts in the proportion of their shares"
            ),
            "{}",
            last("-ts 3,1")
        );
        assert_eq!(
            last("-ts 1,0"),
            "-ts: one card is left with a share, so the shares place nothing"
        );
        assert!(placed_err("-ts 1,1,1", &two()).contains("3 shares for the 2 listed"));
        assert!(placed_err("-ts 0,0", &two()).contains("no listed device a share"));
        assert!(last("-sm none -ts 1,1").contains("under -sm none"));
    }

    /// `-mg` names a card under `-sm none` only: another index than llama-server's default under
    /// `-sm layer`, an index past the list, and a negative one (a CPU-only run) are refused.
    #[test]
    fn main_gpu_names_a_card_only_under_split_none() {
        assert!(placed_err("-mg 1", &two()).starts_with("-mg 1: llama reads -mg under -sm none"));
        assert_eq!(word("-mg 0", &two()), None);
        assert!(placed_err("-sm none -mg 2", &two()).contains("the device list holds 2 device(s)"));
        assert!(placed_err("-sm none -mg -1", &two()).contains("no CPU-only seat"));
    }

    /// The splits bloomery has no thing for are refused by name, each with what it has instead.
    #[test]
    fn row_and_tensor_splits_are_refused_by_name() {
        assert!(refused("-sm row").contains("no row split"));
        assert!(refused("-sm tensor").contains("--place tp over BLOOMERY_TP_NODES"));
    }

    /// `-ngl` is a full offload: `all`, `auto`, a negative count and a count past the layers and the
    /// output layer pass; a count that leaves a layer on the CPU (one short, none) is refused with
    /// the count that passes, and so is a count with no layer count to compare.
    #[test]
    fn ngl_is_a_full_offload_only() {
        for ok in ["all", "auto", "-1", "-2", "62", "99"] {
            assert!(levers(&format!("-ngl {ok}"), "ds41").is_ok(), "-ngl {ok}");
        }
        for short in ["61", "0", "30"] {
            let e = levers(&format!("-ngl {short}"), "ds41").expect_err(short);
            assert!(
                e.contains("use -ngl all, auto, or 62 and more"),
                "-ngl {short}: {e}"
            );
        }
        let blind = Target {
            layers: None,
            ..target("ds41")
        };
        let e = flags("-ngl 99")
            .levers(&blind)
            .expect_err("no block_count")
            .to_string();
        assert!(e.contains("file's block_count"), "{e}");
        assert!(flags("-ngl all").levers(&blind).is_ok());
    }

    /// `--cpu-moe` and what equals it set the qwen38 lever and name it; every other seat is refused
    /// by name; `-ncmoe` of every layer is `--cpu-moe`, of none is nothing, of some is refused; `-ot`
    /// reads only the all-experts patterns to the CPU.
    #[test]
    fn cpu_moe_maps_onto_the_qwen38_lever_only() {
        let host = |line: &str| levers(line, "qwen38").expect(line);
        let set = [("BLOOMERY_QWEN38_EXPERTS", "host".to_owned())];
        for line in [
            "--cpu-moe",
            "-cmoe",
            "-ncmoe 61",
            "-ncmoe 99",
            "-ot exps=CPU",
            "-ot \\.ffn_.*_exps\\.=CPU",
        ] {
            assert_eq!(host(line).set, set, "{line}");
            assert!(
                host(line).notes[0].contains("sets BLOOMERY_QWEN38_EXPERTS=host"),
                "{line}"
            );
        }
        assert_eq!(host("-ncmoe 0"), Levers::default());
        for seat in ["ds41", "glm", "qwen3"] {
            let e = levers("--cpu-moe", seat).expect_err(seat);
            assert!(
                e.starts_with(
                    "--cpu-moe: this seat has no switch that keeps every routed expert on the host"
                ),
                "{seat}: {e}"
            );
        }
        assert!(
            levers("-ncmoe 10", "qwen38")
                .expect_err("some")
                .contains("no first-N-layers rule")
        );
        assert!(refused("-ot exps=CUDA0").contains("names a device"));
        assert!(refused("-ot blk\\.1\\.ffn=CPU").contains("cannot select tensors by regex"));
        assert!(refused("-ot exps").contains("<pattern>=<buffer type> expected"));
    }

    /// `-fit on` is the plan's rule and passes; `-fit off` is refused (no unfitted load) with the
    /// flags that size a load here; `-fitc` is the seats' floor or refused with the floor.
    #[test]
    fn fit_flags_hold_to_the_plans_rule() {
        for on in ["on", "true", "1", "enabled"] {
            assert!(levers(&format!("-fit {on}"), "ds41").is_ok(), "-fit {on}");
        }
        for off in ["off", "false", "0", "disabled"] {
            let e = levers(&format!("-fit {off}"), "ds41").expect_err(off);
            assert!(e.contains("no unfitted load"), "-fit {off}: {e}");
            assert!(
                e.contains("-fitt MiB") && e.contains("--ctx N") && e.contains("-sm none -mg N"),
                "-fit {off}: {e}"
            );
        }
        assert!(levers("-fitc 4096", "ds41").is_ok());
        let e = levers("-fitc 2048", "ds41").expect_err("another floor");
        assert!(e.contains("is 4096 on every seat"), "{e}");
    }

    fn margin(line: &str, census: &[DeviceInfo]) -> Result<Levers, String> {
        flags(line).margin(census).map_err(|e| e.to_string())
    }

    fn budget(line: &str, census: &[DeviceInfo]) -> Result<Levers, String> {
        flags(line).budget(census).map_err(|e| e.to_string())
    }

    /// `-fitt` sets the margin each placed card keeps free, by CUDA ordinal: one value is every
    /// device's margin (one lever value for every card) and a list one per listed device
    /// (`-dev`'s order, else CUDA order), whichever way the cards are named; with none named
    /// every visible card gets its margin and no `--place` word is made (the load's own rule
    /// picks); no `-fitt` sets nothing. It is a margin and no budget: the card's usable bytes
    /// are whole.
    #[test]
    fn fit_target_is_each_placed_cards_margin() {
        let every = |mib: u64| bloomery_levers::CardBudget::Every(mib * MIB).to_string();
        let per = |entries: &[(u32, u64)]| {
            bloomery_levers::CardBudget::Per(
                entries.iter().map(|&(o, mib)| (o, mib * MIB)).collect(),
            )
            .to_string()
        };
        let cs = two();
        let cases: &[(&str, String)] = &[
            ("-sm none -mg 1 -fitt 2048", every(2048)),
            ("-dev CUDA1 -fitt 2048", every(2048)),
            ("-dev CUDA0 --fit-target 1024", every(1024)),
            ("-dev CUDA1,CUDA0 -ts 1,0 -fitt 512", every(512)),
            ("-dev CUDA0,CUDA1 -sm none -mg 1 -fitt 0", every(0)),
            ("-sm none -fitt 1024/1024", per(&[(0, 1024)])),
            ("-sm none -fitt 0,2048", per(&[(0, 0)])),
            ("-sm none -mg 1 -fitt 1024,2048", per(&[(1, 2048)])),
            ("-dev CUDA0,CUDA1 -fitt 1024", every(1024)),
            (
                "-dev CUDA1,CUDA0 -fitt 1024,2048",
                per(&[(1, 1024), (0, 2048)]),
            ),
            ("-fitt 1024 -ts 3,1", every(1024)),
            ("-fitt 1024", every(1024)),
            ("-fitt 512,4096", per(&[(0, 512), (1, 4096)])),
        ];
        for (line, want) in cases {
            let got = margin(line, &cs).expect(line);
            assert_eq!(got.set, [("BLOOMERY_CARD_MARGIN", want.clone())], "{line}");
            assert_eq!(got.notes.len(), 1, "{line}");
            assert_eq!(
                bloomery_levers::card_budget_value(&got.set[0].1)
                    .expect("the lever's own spelling")
                    .to_string(),
                *want,
                "{line}"
            );
        }
        let got = margin("-sm none -mg 1 -fitt 2048", &cs).expect("note");
        assert!(
            got.notes[0]
                .starts_with("llama flag -fitt 2048 sets BLOOMERY_CARD_MARGIN=2147483648: ")
                && got.notes[0].contains(
                    "CUDA1 (NVIDIA GeForce RTX 3090) keeps 2048 MiB of its 24176 MiB free"
                )
                && got.notes[0].contains("replaces the plan's own (1024 MiB)"),
            "{}",
            got.notes[0]
        );
        assert_eq!(margin("-dev CUDA0", &cs), Ok(Levers::default()));
        assert!(flags("-fitt 1024").needs_census());
        assert!(
            flags("-fitt 1024")
                .place(&cs)
                .expect("no card named")
                .is_none()
        );
    }

    /// A list of margins that is neither one value nor one per listed device, a margin a card
    /// cannot hold (named by its card) and no visible device are refused by name; a device that
    /// shares the host's memory keeps its margin as any card does.
    #[test]
    fn fit_target_refuses_what_no_card_can_hold() {
        let cs = two();
        for line in [
            "-dev CUDA0,CUDA1 -fitt 1,2,3",
            "-fitt 1,2,3",
            "-dev CUDA1 -fitt 1,2",
        ] {
            let e = margin(line, &cs).expect_err(line);
            assert!(
                e.contains("margins for the") && e.contains("one for every device, or one each"),
                "{line}: {e}"
            );
        }
        assert!(
            margin("-fitt 1024", &[])
                .expect_err("no card")
                .contains("no CUDA device is visible")
        );
        for (line, card) in [
            ("-dev CUDA1 -fitt 30000", "CUDA1"),
            ("-dev CUDA1 -fitt 18446744073709551615", "CUDA1"),
            ("-fitt 1024,30000", "CUDA1"),
        ] {
            let e = margin(line, &cs).expect_err(line);
            assert!(
                e.contains(card) && e.contains("a plan may use") && e.contains("leaves none"),
                "{line}: {e}"
            );
        }
        let mut unified = census(&[RTX_3090]);
        unified[0].integrated = true;
        let got = margin("-fitt 1024", &unified).expect("a unified card keeps a margin too");
        assert_eq!(
            got.set,
            [("BLOOMERY_CARD_MARGIN", (1024 * MIB).to_string())]
        );
    }

    /// `-ts` over two placed cards sets each card's share, by CUDA ordinal in millionths of the
    /// total and in the order the cards are listed, however the ratio is spelled; one card left
    /// (a zero or missing share, `-sm none`, `-dev` of one) and no `-ts` set nothing; a share that
    /// rounds to no part is refused with the terms.
    #[test]
    fn ts_sets_each_placed_cards_share() {
        let cs = two();
        let shares = |entries: &[(u32, u64)]| {
            vec![(
                "BLOOMERY_CARD_BUDGET",
                bloomery_levers::CardBudget::Shares(entries.to_vec()).to_string(),
            )]
        };
        let cases: &[(&str, &[(u32, u64)])] = &[
            ("-ts 3,1", &[(0, 750_000), (1, 250_000)]),
            ("-ts 1,1", &[(0, 500_000), (1, 500_000)]),
            ("-ts 1/3", &[(0, 250_000), (1, 750_000)]),
            ("-ts 0.5,1.5", &[(0, 250_000), (1, 750_000)]),
            ("-dev CUDA1,CUDA0 -ts 1,3", &[(1, 250_000), (0, 750_000)]),
            (
                "-dev CUDA0,CUDA1 --tensor-split 1,2",
                &[(0, 333_333), (1, 666_667)],
            ),
        ];
        for &(line, want) in cases {
            let got = budget(line, &cs).expect(line);
            assert_eq!(got.set, shares(want), "{line}");
            assert_eq!(got.notes.len(), 1, "{line}");
        }
        let got = budget("-ts 3,1", &cs).expect("note");
        assert_eq!(
            got.notes[0],
            "llama flag -ts 3,1 sets BLOOMERY_CARD_BUDGET=CUDA0=750000x,CUDA1=250000x: the plan \
             holds the routed experts on CUDA0 (NVIDIA RTX A6000), CUDA1 (NVIDIA GeForce RTX 3090) \
             in the proportion 3:1, as many in all as the card whose room is tightest for its \
             share lets it, and the host tier the rest"
        );
        assert_eq!(
            bloomery_levers::card_budget_value(&got.set[0].1),
            Ok(bloomery_levers::CardBudget::Shares(vec![
                (0, 750_000),
                (1, 250_000)
            ]))
        );
        for line in [
            "-ts 1,0",
            "-ts 1",
            "-ts 0,1",
            "-sm none -ts 1,1",
            "-fitt 1024",
            "-dev CUDA0,CUDA1",
        ] {
            assert_eq!(budget(line, &cs), Ok(Levers::default()), "{line}");
        }
        let e = budget("-ts 3000000,1", &cs).expect_err("under one part in a million");
        assert!(
            e.contains("the share 1 of CUDA1 is under one part in a million of the 3000001 given"),
            "{e}"
        );
        assert!(flags("-ts 3,1").needs_census());
    }

    /// A margin that is no MiB count, `-fitt` twice and `-fitt` beside `--place` are refused where
    /// the flags are read.
    #[test]
    fn fit_target_values_are_read_strictly() {
        assert!(refused("-fitt 1g").contains("\"1g\" is no MiB count"));
        assert!(refused("--fit-target 1,").contains("\"\" is no MiB count"));
        assert_eq!(
            refused("-fitt 1 --fit-target 2"),
            "--fit-target: the fit margin is named twice"
        );
        assert!(
            refused("-fitt 1024 --place bp")
                .contains("--place beside llama-server's placement flags")
        );
        assert_eq!(flags("-fitt 1,2/3").fit_target.expect("read").1, [1, 2, 3]);
    }

    /// `-devd` sets the draft's card lever on the ds41 seat and names it; another seat, a list, `none`
    /// and a device the census lacks are refused.
    #[test]
    fn devd_sets_the_draft_card_lever() {
        let on = levers("-devd CUDA1", "ds41").expect("ds41");
        assert_eq!(on.set, [("BLOOMERY_DSPARK_CARD", "cuda1".to_owned())]);
        assert_eq!(on.notes.len(), 1);
        assert!(
            on.notes[0].starts_with("llama flag -devd sets BLOOMERY_DSPARK_CARD=cuda1"),
            "{:?}",
            on.notes
        );
        let e = levers("--device-draft CUDA1", "glm").expect_err("glm");
        assert!(
            e.contains("this seat's draft opens on no card a flag names"),
            "{e}"
        );
        assert!(refused("-devd CUDA0,CUDA1").contains("the draft opens on one card"));
        assert!(refused("-devd none").contains("no seat runs on the CPU alone"));
        assert!(placed_err("-devd CUDA5", &two()).starts_with("-devd CUDA5: no such device"));
        assert_eq!(word("-devd CUDA1", &two()), None);
    }
}

/// The words reach the seats: [`generate::Place`] parses every word the flags make, and resolves
/// it on the census to the devices the note names.
#[cfg(all(test, feature = "gpu"))]
mod seat_words {
    use model::placement::workstation::census_usable;

    use super::*;
    use crate::generate::Place;

    fn census() -> Vec<DeviceInfo> {
        [
            ("NVIDIA RTX A6000", 50_952_536_064_u64),
            ("NVIDIA GeForce RTX 3090", 25_351_356_416),
        ]
        .iter()
        .enumerate()
        .map(|(i, (name, total))| DeviceInfo {
            ordinal: u32::try_from(i).expect("small"),
            name: (*name).to_owned(),
            total_bytes: *total,
            free_bytes: census_usable(*total),
            integrated: false,
            uuid: [u8::try_from(i).expect("small") + 1; 16],
            pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
            held_by: None,
            compute_capability: Some((8, 6)),
        })
        .collect()
    }

    #[test]
    fn a_seat_parses_every_word_the_flags_make() {
        let census = census();
        for (line, stage) in [
            ("-dev CUDA0", 0),
            ("-dev CUDA1", 1),
            ("-dev CUDA0,CUDA1", 0),
            ("-sm none -mg 1", 1),
        ] {
            let args: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
            let (flags, _) = take(&args).expect(line);
            let word = flags.place(&census).expect(line).expect(line).word;
            let place = Place::parse(&word).unwrap_or_else(|e| panic!("{line} → {word}: {e}"));
            let on = place
                .on(&census)
                .unwrap_or_else(|e| panic!("{line} → {word}: {e}"));
            let specs = on.card_specs().expect(line);
            assert_eq!(
                specs[0].device.map(|d| d.ordinal),
                Some(stage),
                "{line} → {word}"
            );
        }
    }
}
