//! The Inkling 2 family on this backend: the visits' buffers ([`layout`]) and
//! the dispatches and bytes a decoded token walks ([`count`]), host arithmetic
//! that builds on every OS, and on macOS the session that runs its tokens —
//! the check of a file ([`check`]), its buffers (`arena`), the walk (`ops`),
//! the prompt pass over rows (`pass`), the session ([`session`]), the
//! runtime's target (`target`), greedy generation ([`generate`]) and
//! `generate_metal --plan`'s verdict ([`plan`]).

#[cfg(target_os = "macos")]
mod arena;
#[cfg(target_os = "macos")]
pub mod check;
#[cfg(target_os = "macos")]
mod clock;
pub mod count;
#[cfg(target_os = "macos")]
pub mod generate;
pub mod layout;
#[cfg(target_os = "macos")]
mod ops;
#[cfg(target_os = "macos")]
mod pass;
#[cfg(target_os = "macos")]
pub mod plan;
#[cfg(target_os = "macos")]
mod profile;
#[cfg(target_os = "macos")]
pub mod session;
#[cfg(target_os = "macos")]
mod target;

#[cfg(target_os = "macos")]
pub use clock::{StepClock, StepTerms};
#[cfg(target_os = "macos")]
pub use pass::PassRoutes;
#[cfg(target_os = "macos")]
pub use profile::{GroupBytes, ProfileGroup};
#[cfg(target_os = "macos")]
pub use session::{Inputs, Routes, Session, SessionError, Tap};
