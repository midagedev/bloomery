//! The Metal backend of the device contract: the `Device` implementation
//! over `objc2-metal`.
//!
//! On another operating system this crate compiles to an empty library (and
//! its gate bin only refuses): the contract names no backend, and the
//! workspace still checks on Linux.
//!
//! The backend's own choices:
//!
//! - f32 activations;
//! - `objc2-metal` pinned by version, with no Objective-C++ shim;
//! - macOS 26.2 and above, refused by name below it;
//! - one command buffer per token, and no indirect command buffer.

#[cfg(target_os = "macos")]
pub mod buffer;
#[cfg(target_os = "macos")]
pub mod census;
#[cfg(target_os = "macos")]
pub mod device;
#[cfg(target_os = "macos")]
pub mod encode;
#[cfg(target_os = "macos")]
pub mod fault;
#[cfg(target_os = "macos")]
mod handle;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod library;
#[cfg(target_os = "macos")]
pub mod stream;

#[cfg(target_os = "macos")]
pub use buffer::{MetalBuffer, MetalRaw, MetalShared};
#[cfg(target_os = "macos")]
pub use census::MetalCensus;
#[cfg(target_os = "macos")]
pub use device::MetalDevice;
#[cfg(target_os = "macos")]
pub use library::{Library, LibraryPath, Pipeline};
#[cfg(target_os = "macos")]
pub use stream::{HostEvent, MetalEvent, MetalStream, Recording};
