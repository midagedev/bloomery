//! The `main` of a stand-alone seat binary (`bloomery-serve-ds41`, `bloomery-serve-qwen38`): the
//! contract of `bloomery-serve` for one seat. Its events are JSON lines on stdout
//! (`serve::event`); a refusal ends with the status of its class (`serve::exit`) and an `error`
//! event; the checks that need no card come first, as a usage refusal, and a driver below R580
//! ends as `no-device`; `--help` answers with the seat's usage on stdout and status 0.

use std::process::ExitCode;

use serve::ServeError;
use serve::event;
use serve::exit::Code;

use crate::llama_launch::{self, Launcher};
use crate::{GateError, llama_place, model_file, side_seat, start};

/// What a seat binary hands the contract: the seat's name and word, its argument check and its run.
pub struct Seat {
    /// The binary's name, which prefixes every refusal.
    pub name: &'static str,
    /// The seat's `--model` word, which llama-server's placement flags are resolved against.
    pub word: &'static str,
    /// The seat's check of its arguments, reading no model and no card: the arguments, whether it
    /// reads its levers, and whether the model comes from a repo.
    pub check: fn(&[String], bool, bool) -> Result<(), GateError>,
    /// The seat's run: loads the model and serves until the listener or the engine fails.
    pub run: fn(&[String], &side_seat::Flags) -> Result<ServeError, GateError>,
}

/// `seat` started by the process's arguments, ending as the process's status.
#[inline(never)]
pub fn main(seat: &Seat) -> ExitCode {
    event::enable();
    let name = seat.name;
    let refuse =
        |unnamed: Code, e: GateError| start::exit_serve(name, Err(e), unnamed, |_| Vec::new());
    let args: Vec<String> = std::env::args().skip(1).collect();
    let asked = |flag: &str| llama_launch::serve_option(&args, flag);
    let (side, rest) = match side_seat::Flags::take(&args) {
        Ok(v) => v,
        Err(e) => return refuse(Code::Usage, e),
    };
    // `--list-devices` and `--records-schema` are left to the start, as `bloomery-serve` leaves
    // them: the seat's check would refuse a flag its own parser does not read.
    if !asked(llama_place::LIST_DEVICES)
        && !asked("--records-schema")
        && let Err(e) = early(seat, &rest, &side)
    {
        // A seat asked for its usage answers with it as its refusal: that is the answer.
        let text = e.to_string();
        if (asked("--help") || asked("-h")) && text.starts_with("usage: ") {
            println!("{text}");
            return ExitCode::SUCCESS;
        }
        return refuse(Code::Usage, e);
    }
    if let Err(e) = serve::preflight::driver(&start::driver_found()) {
        return refuse(Code::NoDevice, e.into());
    }
    if asked(llama_place::LIST_DEVICES) {
        match llama_launch::answer_list_devices(&args) {
            Ok(true) => return ExitCode::SUCCESS,
            Ok(false) => {}
            Err(e) => return refuse(Code::Refused, e),
        }
    }
    // llama-server's placement flags and variables, read once the seat's own flags are out.
    let word = || -> Result<&'static str, GateError> { Ok(seat.word) };
    let launcher = Launcher {
        bin: name,
        seat: &word,
        layers: &llama_launch::file_layers,
        reads: llama_launch::SEAT,
    };
    let rest = match launcher.placed(&rest) {
        Ok(v) => v,
        Err(e) => return refuse(Code::Usage, e),
    };
    start::ended(name, (seat.run)(&rest, &side))
}

/// The checks of the arguments that read no model, open no card and fetch nothing: the side
/// seat's flags, then the seat's own over the arguments less llama-server's placement flags.
fn early(seat: &Seat, rest: &[String], side: &side_seat::Flags) -> Result<(), GateError> {
    side.check_before_model()?;
    let env = llama_launch::llama_env(std::env::vars_os())?;
    let (placement, placed, _) = llama_launch::seat_view(rest, &env, llama_launch::SEAT)?;
    (seat.check)(
        &placed,
        placement.is_empty(),
        model_file::hf_repo().is_some(),
    )
}
