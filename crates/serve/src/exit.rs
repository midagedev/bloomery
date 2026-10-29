//! The exit statuses of a bloomery server, one table: a status, a name and a hint to each code. An
//! agent that starts the server reads the status (or the `error` event's `code`) to learn which
//! class of refusal it met, and the hint to learn what to do next.
//!
//! 64 to 78 keep their sysexits meanings; 80 to 83 are bloomery's own, for refusals sysexits has no
//! word for. A refusal no code names yet stays [`Code::Refused`] (1), so nothing changes class by
//! accident: [`Code::of`] maps the server's own errors, and a seat maps the errors of the crates it
//! owns.

use std::fmt;
use std::io;
use std::process::ExitCode;

use crate::ServeError;
use crate::event::{self, Event};

/// Declares [`Code`] and its three columns from one list, so the table and the enum cannot disagree
/// and [`Code::ALL`] holds every variant.
macro_rules! codes {
    ($($variant:ident = $status:literal, $name:literal, $hint:literal, $doc:literal;)+) => {
        /// What a process ended with: the exit status and the name an `error` event carries.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Code {
            $(
                #[doc = $doc]
                $variant,
            )+
        }

        impl Code {
            /// Every code, in the table's order.
            pub const ALL: &'static [Code] = &[$(Code::$variant),+];

            /// The process's exit status.
            #[must_use]
            pub const fn status(self) -> u8 {
                match self {
                    $(Code::$variant => $status,)+
                }
            }

            /// The code's name, as an `error` event and the docs spell it.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Code::$variant => $name,)+
                }
            }

            /// What to do next, in one sentence; empty for [`Code::Ok`].
            #[must_use]
            pub const fn hint(self) -> &'static str {
                match self {
                    $(Code::$variant => $hint,)+
                }
            }
        }
    };
}

codes! {
    Ok = 0, "ok", "", "Success, an orderly stop, or a request that only prints.";
    Refused = 1, "refused", "read the message", "A refusal no other code names yet.";
    Usage = 64, "usage",
        "run with --help; the message names the flag or value; for a model with no automatic row, \
         `next` lists `--model NAME:VARIANT` for each row",
        "A flag, a value, a name or a combination the server refuses; a model whose every row is \
         unverified or explicit-only, started by its name alone.";
    Unsupported = 65, "unsupported", "run with --recommend --json for what this machine serves",
        "A model or file no seat of the server serves, among them an `--embd-model` file of an \
         architecture no side seat serves; a start by name whose every row the seat's plan refuses \
         as unsupported.";
    NoModel = 66, "no-model",
        "name a model with --model, -m or --hf; the message lists what exists",
        "No model was named, or the one named is not there.";
    NoDevice = 69, "no-device",
        "the message names the driver, device or CPU feature that is missing",
        "The machine lacks what the server needs: a driver, a card or a CPU feature.";
    Engine = 70, "engine", "restart the server; the crash block names the failure",
        "The engine ended the run.";
    Network = 74, "network",
        "check the connection, or set HF_HUB_OFFLINE=1 to start from a cached variant",
        "The hub could not be reached and no cached copy stands in.";
    Retry = 75, "retry", "retry later",
        "A refusal that clears by itself: the hub is busy, or the load was held back.";
    Denied = 77, "denied", "set HF_TOKEN and accept the repo's licence on its page",
        "The hub refused the credentials the request carried.";
    Lever = 78, "lever", "the message names the lever and the kind of value it takes",
        "A `BLOOMERY_*` value the server refuses.";
    NoFit = 80, "no-fit", "choose the next smaller variant, a smaller context or fewer slots",
        "The model does not fit this machine as asked.";
    CardBusy = 81, "card-busy", "retry when the process holding the card ends",
        "A fit refusal on a card another process holds.";
    DiskShort = 82, "disk-short", "free disk space, or set BLOOMERY_CACHE to a larger drive",
        "The cache's filesystem cannot take the download.";
    PortInUse = 83, "port-in-use", "use --port 0 or another port",
        "The address the server was asked to listen on is taken.";
}

impl Code {
    /// The code of `e`, a start or run of the server that ended in it.
    #[must_use]
    pub fn of(e: &ServeError) -> Code {
        match e {
            ServeError::Io(err) if err.kind() == io::ErrorKind::AddrInUse => Code::PortInUse,
            ServeError::Io(_) | ServeError::Template(_) | ServeError::NoStops => Code::Refused,
            ServeError::Engine(_) => Code::Engine,
            ServeError::SlotSavePath(_) | ServeError::Slots(_) => Code::Usage,
        }
    }

    /// The code that exits with `status`, if any does.
    #[must_use]
    pub fn from_status(status: u8) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| c.status() == status)
    }

    /// The code named `name`, if any is.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| c.name() == name)
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<Code> for ExitCode {
    fn from(code: Code) -> ExitCode {
        ExitCode::from(code.status())
    }
}

/// Ends a start that `code` refused: the `error` event on stdout, with `message`, the code's hint
/// and the argument lists `next` offers, then the status the process exits with. The stderr line is
/// the caller's.
#[must_use]
pub fn refused(code: Code, message: &str, next: &[&[&str]]) -> ExitCode {
    event::emit(&Event::Error {
        code,
        message,
        next,
    });
    ExitCode::from(code)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;

    use super::*;
    use crate::EngineFailure;

    /// The statuses and the names are each unique, so a status names one class of refusal and a
    /// name one status; zero is success alone and no status reaches the range a shell reads as a
    /// signal.
    #[test]
    fn every_code_has_its_own_status_and_name() {
        let mut statuses = HashSet::new();
        let mut names = HashSet::new();
        for code in Code::ALL {
            let (status, name) = (code.status(), code.name());
            assert!(
                statuses.insert(status),
                "{code:?} repeats the status {status} another code has"
            );
            assert!(
                names.insert(name),
                "{code:?} repeats the name {name:?} another code has"
            );
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
                "the name {name:?} is lowercase words joined by '-'"
            );
            assert!(
                status < 126,
                "{code:?}'s status {status} is one a shell reads as a signal or a missing command"
            );
            assert_eq!(
                status == 0,
                *code == Code::Ok,
                "{code:?}: zero is success and only success"
            );
            assert_eq!(
                code.hint().is_empty(),
                *code == Code::Ok,
                "{code:?}: every refusal names a next step and success has none"
            );
            assert_eq!(Code::from_status(status), Some(*code));
            assert_eq!(Code::from_name(name), Some(*code));
        }
        assert_eq!(Code::from_status(2), None, "2 is no code of the table");
        assert_eq!(Code::from_name("no-such-code"), None);
    }

    /// The table's numbers are the contract an agent keys on: a renumbered code is a break, not a
    /// refactor.
    #[test]
    fn the_table_holds_the_published_statuses() {
        let table: Vec<(u8, &str)> = Code::ALL.iter().map(|c| (c.status(), c.name())).collect();
        assert_eq!(
            table,
            [
                (0, "ok"),
                (1, "refused"),
                (64, "usage"),
                (65, "unsupported"),
                (66, "no-model"),
                (69, "no-device"),
                (70, "engine"),
                (74, "network"),
                (75, "retry"),
                (77, "denied"),
                (78, "lever"),
                (80, "no-fit"),
                (81, "card-busy"),
                (82, "disk-short"),
                (83, "port-in-use"),
            ]
        );
    }

    /// The server's own errors map to the code their class has; a taken port is its own code and
    /// any other listener failure stays the unnamed refusal.
    #[test]
    fn a_serve_error_maps_to_its_code() {
        let listener = |kind: io::ErrorKind| ServeError::Io(io::Error::from(kind));
        assert_eq!(
            Code::of(&listener(io::ErrorKind::AddrInUse)),
            Code::PortInUse
        );
        assert_eq!(
            Code::of(&listener(io::ErrorKind::PermissionDenied)),
            Code::Refused
        );
        assert_eq!(Code::of(&ServeError::NoStops), Code::Refused);
        assert_eq!(
            Code::of(&ServeError::Slots("--parallel 0".to_owned())),
            Code::Usage
        );
        assert_eq!(
            Code::of(&ServeError::SlotSavePath(PathBuf::from("missing"))),
            Code::Usage
        );
        let failure = EngineFailure {
            engine: "mock".to_owned(),
            error: "failed".to_owned(),
        };
        assert_eq!(Code::of(&ServeError::Engine(failure)), Code::Engine);
    }
}
