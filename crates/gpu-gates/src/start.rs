//! The start of a server as the agent that runs it reads it, beside the human lines on stderr.
//!
//! A `--plan --json` run ends with one JSON line on stdout, `bloomery.plan/1`: the placement, the
//! context and the slots the plan chose, the weight bytes it leaves on the host and pages from the
//! NVMe tier, and the host need. The seats print their plan through common printers
//! ([`crate::generate::PlaceWhy::record`], [`crate::generate::choose_ctx_of`],
//! [`crate::need::staged`]), and each notes here what it printed ([`note_place`], [`note_ctx`],
//! [`note_need`]); no seat holds a branch for the line. [`plan_exit`] is the one end of a `--plan`
//! run: every seat calls it where it used to exit 0.
//!
//! A refusal is a [`Refusal`] — an exit code of `serve::exit` beside its message and the argument
//! lists that fix it — or an error the chain of which names one ([`exit_serve`] reads the types of
//! the crates the start path calls). An error no type names keeps the unnamed status 1, so nothing
//! changes class by accident.
//!
//! The listener a start binds before the fetch ([`bind_early`]) is handed to the seat's server
//! at its bind ([`take_early`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

/// The context a plan chose.
#[derive(Clone, Debug)]
pub(crate) struct CtxNote {
    slot_ctx: u64,
    slots: u64,
    rule: &'static str,
    limit: u64,
}

/// What the plan needs: the host's bytes, each card's, the routed-expert bytes the host keeps
/// (resident or paged) and of them the bytes paged from the NVMe tier.
#[derive(Clone, Debug)]
struct NeedNote {
    bytes: u64,
    cards: Vec<(String, u64)>,
    host_bytes: u64,
    paged: u64,
}

/// What the plan printed, as the seats noted it.
struct Report {
    per_slot: bool,
    seat: Option<&'static str>,
    place: Option<(String, String)>,
    ctx: Option<CtxNote>,
    need: Option<NeedNote>,
}

static REPORT: Mutex<Report> = Mutex::new(Report {
    per_slot: false,
    seat: None,
    place: None,
    ctx: None,
    need: None,
});

/// Whether this run was asked for `--plan --json`.
static PLAN_JSON: AtomicBool = AtomicBool::new(false);

/// What a run does when its plan is staged: the load is next.
static LOADING: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

fn report() -> MutexGuard<'static, Report> {
    REPORT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Names the seat this run serves, which the plan line carries.
pub fn note_seat(word: &'static str) {
    report().seat = Some(word);
}

/// Keeps the effective context flag's units for refusals after the slot parser has read it.
pub fn note_slot_ctx(per_slot: bool) {
    report().per_slot = per_slot;
}

/// Whether the effective context flag [`note_slot_ctx`] kept is `--slot-ctx`.
#[must_use]
pub fn slot_ctx_noted() -> bool {
    report().per_slot
}

/// The placement the run takes and why, as the `place unset` record prints them.
pub fn note_place(place: &str, why: &str) {
    report().place = Some((place.to_owned(), why.to_owned()));
}

/// The context a slot serves, the slots, the rule that set it and the model's limit, as the `ctx`
/// line prints them.
pub fn note_ctx(slot_ctx: u64, slots: u64, rule: &'static str, limit: u64) {
    report().ctx = Some(CtxNote {
        slot_ctx,
        slots,
        rule,
        limit,
    });
}

/// The context [`note_ctx`] kept last, for a probe that notes its own and gives the note back
/// ([`restore_ctx`]).
pub(crate) fn noted_ctx() -> Option<CtxNote> {
    report().ctx.clone()
}

/// Puts back the context note [`noted_ctx`] read.
pub(crate) fn restore_ctx(note: Option<CtxNote>) {
    report().ctx = note;
}

/// The plan's need: `bytes` of host, each card's bytes, the routed-expert bytes the host keeps
/// (`host_bytes`, the NVMe tier's `paged` among them). The load is next, so the hook that
/// [`on_loading`] installed runs.
pub fn note_need(bytes: u64, cards: &[(String, u64)], host_bytes: u64, paged: u64) {
    report().need = Some(NeedNote {
        bytes,
        cards: cards.to_vec(),
        host_bytes,
        paged,
    });
    if let Some(hook) = LOADING.get() {
        hook();
    }
}

/// Installs what runs when a plan is staged for its load (once; a second call is ignored): the
/// start moves its `/health` answer from planning to loading.
pub fn on_loading(hook: impl Fn() + Send + Sync + 'static) {
    let _ = LOADING.set(Box::new(hook));
}

/// Asks that a `--plan` run end with its `bloomery.plan/1` line ([`plan_exit`]).
pub fn set_plan_json(on: bool) {
    PLAN_JSON.store(on, Ordering::SeqCst);
}

/// Whether the run was asked for `--plan --json`.
#[must_use]
pub fn plan_json_asked() -> bool {
    PLAN_JSON.load(Ordering::SeqCst)
}

/// The placement word the plan noted, for the `ready` event.
#[must_use]
pub fn noted_place() -> Option<String> {
    report().place.as_ref().map(|(place, _)| place.clone())
}

#[cfg(any(feature = "deepseek41", feature = "clef"))]
pub use agent::{
    Refusal, bind_early, class_of, driver_found, ended, exit_serve, plan_exit, plan_line,
    refusal_json, take_early,
};

/// The agent-facing half, which the server crate's types and JSON need.
#[cfg(any(feature = "deepseek41", feature = "clef"))]
mod agent {
    use std::error::Error;
    use std::ffi::OsString;
    use std::fmt;
    use std::io::{self, Write};
    use std::path::Path;
    use std::process::ExitCode;
    use std::sync::{Mutex, PoisonError};

    use model::placement::PlacementError;
    use model::placement::ctx::PastLimit;
    use model::placement::need_hook::{self, HookError};
    use model::placement::workstation::HostShort;
    use serde_json::{Value, json};
    use serve::event::{self, Event};
    use serve::exit::{self, Code};
    use serve::preflight::Driver;
    use serve::{Listener, ServeError};

    use super::{CtxNote, NeedNote, Report, plan_json_asked, report};
    use crate::GateError;
    use crate::generate::{CtxDoesNotFit, FloorShort};

    /// The schema a plan line names.
    const PLAN_SCHEMA: &str = "bloomery.plan/1";

    /// A refusal a start names: the code that ends the process, what was refused and why, and the
    /// argument lists that fix it, each one runnable as it stands.
    #[derive(Debug)]
    pub struct Refusal {
        pub code: Code,
        pub message: String,
        pub next: Vec<Vec<String>>,
    }

    impl Refusal {
        /// A refusal of class `code` saying `message`.
        pub fn new(code: Code, message: impl Into<String>) -> Refusal {
            Refusal {
                code,
                message: message.into(),
                next: Vec::new(),
            }
        }

        /// The refusal with one more argument list that fixes it.
        #[must_use]
        pub fn offering(mut self, argv: &[&str]) -> Refusal {
            self.next
                .push(argv.iter().map(|a| (*a).to_owned()).collect());
            self
        }
    }

    impl fmt::Display for Refusal {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.message)
        }
    }

    impl Error for Refusal {}

    /// The plan the notes hold, as the `bloomery.plan/1` line says it ([`line_of`]).
    pub fn plan_line() -> Result<Value, String> {
        line_of(&report())
    }

    /// `report` as the `bloomery.plan/1` line, or what the seat did not print: a seat whose plan
    /// names no context or no placement cannot answer `--plan --json`, and says so instead of a
    /// guessed figure.
    fn line_of(report: &Report) -> Result<Value, String> {
        let CtxNote {
            slot_ctx,
            slots,
            rule,
            limit,
        } = report
            .ctx
            .clone()
            .ok_or("the seat's plan printed no context line, so it has no --plan --json answer")?;
        let (place, why) = report.place.clone().ok_or(
            "the seat's plan printed no placement record, so it has no --plan --json answer",
        )?;
        // A plan that stages no need record puts no routed expert on the host.
        let need = report.need.clone();
        let (host_bytes, paged) = need.as_ref().map_or((0, 0), |n| (n.host_bytes, n.paged));
        Ok(json!({
            "schema": PLAN_SCHEMA,
            "seat": report.seat,
            "place": place,
            "place_why": why,
            "ctx": slot_ctx,
            "ctx_rule": rule,
            "slots": slots,
            "limit": limit,
            "host_bytes": host_bytes,
            "paged": paged,
            "need": need.map(|NeedNote { bytes, cards, .. }| json!({"bytes": bytes, "cards": cards})),
        }))
    }

    /// The `bloomery.plan/1` line of a plan the run refused.
    #[must_use]
    pub fn refusal_json(code: Code, message: &str, need: Option<u64>) -> Value {
        json!({
            "schema": PLAN_SCHEMA,
            "refused": {"code": code.name(), "message": message, "need": need},
        })
    }

    /// Ends a `--plan` run where the seat's plan is printed. With `--json` it prints the plan
    /// line first; a seat that cannot answer it ends as a refusal, status and all.
    pub fn plan_exit() -> ! {
        if plan_json_asked() {
            match plan_line() {
                Ok(line) => {
                    println!("{line}");
                    let _ = io::stdout().flush();
                }
                Err(why) => {
                    let refusal: GateError = Box::new(Refusal::new(Code::Unsupported, why));
                    let _ = exit_serve(
                        &program_of(std::env::args_os().next()),
                        Err(refusal),
                        Code::Refused,
                        |_| Vec::new(),
                    );
                    std::process::exit(i32::from(Code::Unsupported.status()));
                }
            }
        }
        std::process::exit(0);
    }

    /// The name a process runs under, from its first argument: `bloomery-serve` or a stand-alone
    /// seat's binary, which prefixes the refusals it prints. A process with no first argument is
    /// `bloomery-serve`.
    fn program_of(arg0: Option<OsString>) -> String {
        arg0.as_deref()
            .map(Path::new)
            .and_then(Path::file_name)
            .map_or_else(
                || "bloomery-serve".to_owned(),
                |name| name.to_string_lossy().into_owned(),
            )
    }

    /// The class an error of the start path names: the first error of its chain that names one;
    /// `None` for a chain no type names.
    pub fn class_of(e: &GateError) -> Option<Code> {
        classify(e.as_ref()).map(|(code, _, _)| code)
    }

    /// What an error says of its class: the code, the argument lists that fix it and the bytes
    /// it asked for, where it says.
    type Named = (Code, Vec<Vec<String>>, Option<u64>);

    /// The class an error of the start path names, and the argument lists it offers: the first
    /// error of the chain that names one; `None` for a chain no type names.
    fn classify(e: &(dyn Error + 'static)) -> Option<Named> {
        let per_slot = report().per_slot;
        let mut link: Option<&(dyn Error + 'static)> = Some(e);
        while let Some(err) = link {
            if let Some(found) = named_by(err, per_slot) {
                return Some(found);
            }
            link = err.source();
        }
        None
    }

    /// What `err` itself says of its class: a [`Refusal`], or one of the types the start path
    /// calls.
    fn named_by(err: &(dyn Error + 'static), per_slot: bool) -> Option<Named> {
        let ctx_next = |ctx: u64| {
            let flag = if per_slot { "--slot-ctx" } else { "--ctx-size" };
            vec![vec![flag.to_owned(), ctx.to_string()]]
        };
        if let Some(r) = err.downcast_ref::<Refusal>() {
            return Some((r.code, r.next.clone(), None));
        }
        #[cfg(feature = "hf")]
        {
            if let Some(h) = err.downcast_ref::<hf::HfError>() {
                return Some((hf_code(h), Vec::new(), None));
            }
        }
        if let Some(p) = err.downcast_ref::<PlacementError>() {
            let need = p.card_short().map(|s| s.need);
            return Some((placement_code(p), Vec::new(), need));
        }
        if let Some(f) = err.downcast_ref::<FloorShort>() {
            return Some((Code::NoFit, Vec::new(), Some(f.short.need)));
        }
        if let Some(c) = err.downcast_ref::<CtxDoesNotFit>() {
            let largest = if per_slot {
                c.largest_slot
            } else {
                c.largest_total
            };
            let next = largest.map(ctx_next).unwrap_or_default();
            return Some((Code::NoFit, next, Some(c.short.need)));
        }
        if let Some(h) = err.downcast_ref::<HostShort>() {
            return Some((Code::NoFit, Vec::new(), h.pool_short().map(|s| s.need)));
        }
        if let Some(p) = err.downcast_ref::<PastLimit>() {
            return Some((Code::Usage, ctx_next(p.limit), None));
        }
        if err.downcast_ref::<bloomery_levers::LeverError>().is_some() {
            return Some((Code::Lever, Vec::new(), None));
        }
        if let Some(s) = err.downcast_ref::<ServeError>() {
            let code = Code::of(s);
            let next = if code == Code::PortInUse {
                vec![vec!["--port".to_owned(), "0".to_owned()]]
            } else {
                Vec::new()
            };
            return Some((code, next, None));
        }
        if matches!(
            err.downcast_ref::<HookError>(),
            Some(HookError::Refused { .. })
        ) {
            return Some((Code::Retry, Vec::new(), None));
        }
        None
    }

    /// The class of a hub refusal. A denied or missing repo is told by the text `hf::fetch` gives
    /// the status it read.
    #[cfg(feature = "hf")]
    fn hf_code(e: &hf::HfError) -> Code {
        use hf::HfError;
        match e {
            HfError::Repo(_)
            | HfError::Source(_)
            | HfError::CheckpointQuant { .. }
            | HfError::HubOffline(_)
            | HfError::NoCache => Code::Usage,
            HfError::CheckpointUnsupported { .. } => Code::Unsupported,
            HfError::NoGguf { .. }
            | HfError::NoQuant { .. }
            | HfError::NoMatch { .. }
            | HfError::TwoMatches { .. }
            | HfError::NoFile { .. }
            | HfError::IncompleteSet { .. } => Code::NoModel,
            HfError::Network { .. } | HfError::Offline { .. } => Code::Network,
            HfError::Unavailable { .. } => Code::Retry,
            HfError::Curl { why, .. } if why.contains(hf::fetch::DENIED_HINT) => Code::Denied,
            HfError::Curl { why, .. } if why.contains(hf::fetch::NOT_THERE_HINT) => Code::NoModel,
            HfError::Curl { .. }
            | HfError::Listing { .. }
            | HfError::Path { .. }
            | HfError::TwoDrafts { .. }
            | HfError::Size { .. }
            | HfError::Hash { .. }
            | HfError::Range { .. }
            | HfError::Io { .. } => Code::Refused,
        }
    }

    /// The class of a placement refusal: a memory the plan does not fit is [`Code::NoFit`], a
    /// program the seats do not run [`Code::Unsupported`], the rest the unnamed refusal.
    fn placement_code(e: &PlacementError) -> Code {
        if e.card_short().is_some()
            || e.host_floor_short().is_some()
            || matches!(
                e,
                PlacementError::TierBatchCard { .. }
                    | PlacementError::TierBatchHost { .. }
                    | PlacementError::ResidencyOverHost(_)
            )
        {
            Code::NoFit
        } else if matches!(e, PlacementError::Unimplemented(_)) {
            Code::Unsupported
        } else {
            Code::Refused
        }
    }

    /// What the driver answers a start: its library's load as cuda-bindings loads it (`dyn_load`:
    /// `libcuda.so.1`, then `libcuda.so`, and its `cuDriverGetVersion` refused as too old below the
    /// CUDA major the build's toolkit is), read through that crate's own `cuda_driver_load_error`.
    pub fn driver_found() -> Driver {
        use cuda_core::sys;
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

    /// The process's exit for a seat's end, the decide seat's and the generative seats' alike: the
    /// engine ending the run is the class `engine`; any other end is a refusal no type names, the
    /// class `refused` (an error that names its class keeps it).
    pub fn ended(name: &str, r: Result<ServeError, GateError>) -> ExitCode {
        match r {
            Ok(ServeError::Engine(f)) => {
                eprintln!("{name}: {f}");
                exit::refused(Code::Engine, &f.to_string(), &[])
            }
            Ok(e) => exit_serve(name, Err(e.into()), Code::Refused, |_| Vec::new()),
            Err(e) => exit_serve(name, Err(e), Code::Refused, |_| Vec::new()),
        }
    }

    /// Ends a start that failed with `r`: the stderr lines every exit prints
    /// ([`crate::print_refusal`]), the plan's refusal line under `--plan --json`, the `error`
    /// event, and the status of the error's class. An error no type names takes the class
    /// `unnamed` (the start's phase: the arguments' check is a usage refusal, the driver's a
    /// missing device). `more` adds argument lists to the ones the error carries, for the class it
    /// ends with (the start's own knowledge: a smaller variant).
    pub fn exit_serve(
        name: &str,
        r: Result<(), GateError>,
        unnamed: Code,
        more: impl FnOnce(Code) -> Vec<Vec<String>>,
    ) -> ExitCode {
        let Err(e) = r else {
            return ExitCode::SUCCESS;
        };
        let held_back = need_hook::refusal();
        crate::print_refusal(name, &e, held_back.as_deref());
        let (found, mut next, need) = classify(e.as_ref()).unwrap_or((unnamed, Vec::new(), None));
        let code = if held_back.is_some() {
            Code::Retry
        } else {
            found
        };
        next.extend(more(code));
        if code == Code::Usage && next.is_empty() {
            next.push(vec!["--help".to_owned()]);
        }
        let message = e.to_string();
        if plan_json_asked() {
            println!("{}", refusal_json(code, &message, need));
        }
        let argv: Vec<Vec<&str>> = next
            .iter()
            .map(|list| list.iter().map(String::as_str).collect())
            .collect();
        let lists: Vec<&[&str]> = argv.iter().map(Vec::as_slice).collect();
        exit::refused(code, &message, &lists)
    }

    /// The listener a start bound before its fetch, until the seat's server takes it.
    struct Early {
        host: String,
        port: u16,
        listener: Listener,
    }

    static EARLY: Mutex<Option<Early>> = Mutex::new(None);

    /// Binds `host:port` before the model is fetched: the listener answers 503 `loading` from
    /// here, and the `bound` event says where. A taken port is the error. The handle moves the
    /// answer's phase and its download's bytes.
    pub fn bind_early(host: &str, port: u16) -> io::Result<serve::listener::Progress> {
        let listener = Listener::early((host, port))?;
        let progress = listener.progress();
        event::emit(&Event::Bound {
            url: &event::url(listener.local_addr()),
        });
        *EARLY.lock().unwrap_or_else(PoisonError::into_inner) = Some(Early {
            host: host.to_owned(),
            port,
            listener,
        });
        Ok(progress)
    }

    /// The listener [`bind_early`] bound, for the seat whose server binds `host:port`: `None`
    /// when no start bound one. A seat that asks another address than the one bound is refused
    /// by name.
    pub fn take_early(host: &str, port: u16) -> io::Result<Option<Listener>> {
        let taken = EARLY.lock().unwrap_or_else(PoisonError::into_inner).take();
        match taken {
            None => Ok(None),
            Some(early) if early.host == host && early.port == port => Ok(Some(early.listener)),
            Some(early) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "the start listens on {}:{} and the seat binds {host}:{port}",
                    early.host, early.port
                ),
            )),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::Report;
        use super::*;

        fn start_with(code: Code) -> Refusal {
            Refusal::new(code, "m")
        }

        /// An error that wraps another and names nothing itself.
        #[derive(Debug)]
        struct Wrapped(Refusal);

        impl fmt::Display for Wrapped {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("the seat's own words")
            }
        }

        impl Error for Wrapped {
            fn source(&self) -> Option<&(dyn Error + 'static)> {
                Some(&self.0)
            }
        }

        /// A refusal names its own code whatever wraps it, and the first error of a chain that
        /// names one decides: an unnamed error on top of a named one is the named one's class;
        /// a chain no type names has no class of its own.
        #[test]
        fn a_refusal_in_the_chain_names_the_class() {
            let named = start_with(Code::NoFit).offering(&["--model", "m:q"]);
            let (code, next, _) = classify(&named).expect("a refusal names its class");
            assert_eq!(code, Code::NoFit);
            assert_eq!(next, vec![vec!["--model".to_owned(), "m:q".to_owned()]]);
            let wrapped = Wrapped(start_with(Code::DiskShort));
            let (code, ..) = classify(&wrapped).expect("the chain names a class");
            assert_eq!(code, Code::DiskShort);
            let unnamed: GateError = "the seat said no".into();
            assert!(classify(unnamed.as_ref()).is_none());
        }

        /// A context refusal offers the retry in the units of the flag the run noted:
        /// `--slot-ctx` takes one slot's largest share, `--ctx-size` the total over the slots,
        /// for both a capacity refusal and the model's limit.
        #[test]
        fn context_retry_names_the_flag_and_its_units() {
            use model::placement::ctx_ask::{Probe, Short, Side};
            let _note = super::super::slot_note_tests();
            for (total, slots, per_slot, flag, expected) in [
                (16_384, 1, true, "--slot-ctx", 8192),
                (131_072, 1, true, "--slot-ctx", 65_536),
                (65_536, 4, false, "--ctx-size", 32_768),
                (65_536, 4, true, "--slot-ctx", 8192),
            ] {
                let e = crate::generate::choose_ctx_of(
                    65_536,
                    (Some(total), slots),
                    (None, None),
                    |ctx| -> Result<Probe<()>, GateError> {
                        Ok(if ctx <= 8192 {
                            Probe::Holds(())
                        } else {
                            Probe::Short(Short {
                                on: Side::Card,
                                need: ctx,
                                room: 8192,
                            })
                        })
                    },
                )
                .err()
                .expect("capacity or model limit refusal");
                super::super::note_slot_ctx(per_slot);
                let (_, next, _) = classify(e.as_ref()).expect("a named refusal");
                assert_eq!(
                    next,
                    vec![vec![flag.to_owned(), expected.to_string()]],
                    "{total} over {slots}"
                );
            }
        }

        fn report_of(ctx: bool, place: bool, need: bool) -> Report {
            Report {
                per_slot: false,
                seat: Some("glm"),
                place: place.then(|| ("a".to_owned(), "one card".to_owned())),
                ctx: ctx.then_some(CtxNote {
                    slot_ctx: 8192,
                    slots: 2,
                    rule: "fitted",
                    limit: 32768,
                }),
                need: need.then(|| NeedNote {
                    bytes: 9,
                    cards: vec![("A6000".to_owned(), 5)],
                    host_bytes: 7,
                    paged: 3,
                }),
            }
        }

        /// A plan that noted no context or no placement is no answer: the line is an error naming
        /// what is missing, not a guessed figure; a plan with no need record leaves no expert on
        /// the host; a staged one says its host bytes, paged bytes and need.
        #[test]
        fn a_plan_line_is_what_the_seat_noted() {
            let e = line_of(&report_of(false, true, true)).expect_err("no context");
            assert!(e.contains("no context"), "{e}");
            let e = line_of(&report_of(true, false, true)).expect_err("no placement");
            assert!(e.contains("no placement"), "{e}");
            let held = line_of(&report_of(true, true, true)).expect("a plan");
            assert_eq!(held["schema"], PLAN_SCHEMA);
            assert_eq!(held["seat"], "glm");
            assert_eq!(
                (held["ctx"].as_u64(), held["slots"].as_u64()),
                (Some(8192), Some(2))
            );
            assert_eq!(held["host_bytes"], 7);
            assert_eq!(held["paged"], 3);
            assert_eq!(held["need"]["bytes"], 9);
            assert_eq!(held["need"]["cards"][0][0], "A6000");
            let whole = line_of(&report_of(true, true, false)).expect("a whole-card plan");
            assert_eq!(
                (whole["host_bytes"].as_u64(), whole["paged"].as_u64()),
                (Some(0), Some(0))
            );
            assert!(whole["need"].is_null());
        }

        /// The refusals a seat prints are prefixed by the name its binary runs under.
        #[test]
        fn a_refusal_is_prefixed_by_the_binary_it_runs_under() {
            let ran = |path: &str| program_of(Some(OsString::from(path)));
            assert_eq!(
                ran("/opt/bloomery/bloomery-serve-ds41"),
                "bloomery-serve-ds41"
            );
            assert_eq!(ran("bloomery-serve-qwen38"), "bloomery-serve-qwen38");
            assert_eq!(program_of(None), "bloomery-serve");
        }

        /// A hub refusal's class is told by what `hf::fetch` says of the status it read: a 401 or a
        /// 403 is denied, a 404 is no model, any other curl failure the unnamed refusal.
        #[cfg(feature = "hf")]
        #[test]
        fn a_hub_status_is_denied_or_missing_by_what_fetch_says() {
            let curl = |said: &str| hf::HfError::Curl {
                what: "listing".to_owned(),
                why: format!("exit 22: {said}"),
            };
            assert_eq!(hf_code(&curl(hf::fetch::DENIED_HINT)), Code::Denied);
            assert_eq!(hf_code(&curl(hf::fetch::NOT_THERE_HINT)), Code::NoModel);
            assert_eq!(
                hf_code(&curl("curl: (22) returned error: 500")),
                Code::Refused
            );
        }

        /// The refusal line carries the code's name and the bytes asked, or null.
        #[test]
        fn a_refusal_line_names_its_code_and_need() {
            let v = refusal_json(Code::NoFit, "too big", Some(7));
            assert_eq!(v["schema"], PLAN_SCHEMA);
            assert_eq!(v["refused"]["code"], "no-fit");
            assert_eq!(v["refused"]["need"], 7);
            assert!(refusal_json(Code::Usage, "m", None)["refused"]["need"].is_null());
        }

        /// A drafted plan that stands keeps its own context note, which the plain plan's probe
        /// overwrote; a yield keeps the plain plan's.
        #[test]
        fn the_plan_line_names_the_context_the_draft_choice_serves() {
            use crate::generate::{DraftYield, draft_yield};
            use model::arch::hybrid::PlaceError;

            let _note = super::super::slot_note_tests();
            let chosen = |without: usize| {
                super::super::note_ctx(4096, 1, "fitted", 16_384);
                let chosen = draft_yield::<PlaceError, _, _>(
                    true,
                    Ok(4096_usize),
                    |&c| Some(c),
                    || {
                        super::super::note_ctx(without as u64, 1, "fitted", 16_384);
                        Ok((without, without, 7))
                    },
                )
                .expect("a choice");
                let kept = matches!(chosen, DraftYield::Keep(_));
                (kept, report().ctx.as_ref().map(|c| c.slot_ctx))
            };
            assert_eq!(chosen(8191), (true, Some(4096)));
            assert_eq!(chosen(8192), (false, Some(8192)));
        }

        /// A plan refusal named with the reserves beside the plan keeps its class and its cause:
        /// the draft's host experts past the room are no fit, and the message the plan line and
        /// the error event carry names the reserve and the refusal both.
        #[test]
        fn a_reserve_named_refusal_keeps_its_class_and_cause() {
            let cause = PlacementError::DraftHostRoom { room: 8, draft: 9 };
            let text = cause.to_string();
            let e = crate::beside::refusal(
                crate::beside::Beside {
                    image: Some(7),
                    embedding: None,
                },
                cause,
            );
            let (code, _, need) = classify(e.as_ref()).expect("a named refusal");
            assert_eq!(code, Code::NoFit);
            let line = refusal_json(code, &e.to_string(), need);
            let message = line["refused"]["message"].as_str().expect("a message");
            assert!(message.contains("image encoder's 7 B"), "{message}");
            assert!(message.contains(&text), "{message}");
        }

        /// A placement refusal a family's error carries keeps its class: a host room under the
        /// NVMe tier floor is no fit whichever shared plan, the nodes' plan or the load refused
        /// it. Each architecture's own planner error carries it as its source the same way
        /// (`model::arch`'s `a_family_placement_refusal_is_its_errors_source`).
        #[test]
        fn a_family_placement_refusal_keeps_its_class() {
            use model::arch::{draft, hybrid, place_hybrid};
            let floor = || PlacementError::HostRoomFloor {
                room: 1,
                floor: 2,
                base: 0,
                layer: 0,
                need: 2,
                arena: None,
            };
            let families: [GateError; 5] = [
                Box::new(hybrid::PlaceError::from(floor())),
                Box::new(place_hybrid::PlaceError::from(floor())),
                Box::new(draft::DraftUsableError::from(floor())),
                Box::new(model::placement::nodes::NodesError::from(floor())),
                Box::new(model::ModelError::Placement(floor())),
            ];
            for e in families {
                assert_eq!(class_of(&e), Some(Code::NoFit), "{e:?}");
                assert_eq!(e.to_string(), floor().to_string(), "{e:?}");
            }
        }

        /// A seat that binds another address than the start bound is refused by name, and the
        /// listener that was bound is the one handed over.
        #[test]
        fn the_early_listener_goes_to_the_seat_that_binds_its_address() {
            assert!(take_early("127.0.0.1", 1).expect("none bound").is_none());
            bind_early("127.0.0.1", 0).expect("bind");
            let Err(e) = take_early("127.0.0.1", 9) else {
                panic!("another port is refused");
            };
            assert!(e.to_string().contains("127.0.0.1:0"), "{e}");
            bind_early("127.0.0.1", 0).expect("bind");
            let l = take_early("127.0.0.1", 0).expect("same").expect("bound");
            assert_ne!(l.local_addr().port(), 0, "the real port");
            assert!(take_early("127.0.0.1", 0).expect("taken").is_none());
        }
    }
}

/// Serializes the tests that write or read the [`note_slot_ctx`] note, which is process-wide.
#[cfg(test)]
pub(crate) fn slot_note_tests() -> MutexGuard<'static, ()> {
    static TESTS: Mutex<()> = Mutex::new(());
    TESTS.lock().unwrap_or_else(PoisonError::into_inner)
}
