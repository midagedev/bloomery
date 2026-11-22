//! The session core: what every family's session on this backend shares —
//! the lane algebra of a recurrent state ([`lanes`]) and the room rule that
//! holds a load to the device's free bytes ([`room`]) on every OS; and, on
//! macOS, the step vocabulary and its clock ([`clock`]), the pass's
//! command-buffer machinery and vocabulary ([`pass`]), the prompt machinery over
//! the [`prompt`] family seam, and the greedy generation loop
//! ([`generate`]).
//!
//! A family's own session (the file's read, its arena, its ops walk) stays in
//! its `arch` module; what sits here is the protocol around it, so the next
//! family attaches over the same owners instead of a copy. The seams are
//! traits named for what the moved code calls: [`room::SessionBytes`] (a
//! byte plan at a cache size) and [`prompt::Family`] (what the prompt
//! machinery asks of a family's session).

pub mod lanes;
pub mod room;

#[cfg(target_os = "macos")]
pub mod clock;
#[cfg(target_os = "macos")]
pub mod generate;
#[cfg(target_os = "macos")]
pub mod pass;
#[cfg(target_os = "macos")]
pub mod prompt;
