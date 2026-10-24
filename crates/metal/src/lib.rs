//! The Metal backend of the device contract: the `Device` implementation
//! over `objc2-metal`.
//!
//! On another operating system the device modules are not built (and the
//! gate bin only refuses). [`arch`] is host arithmetic and [`server`] host
//! code; both build on every target, so the workspace still checks on Linux.
//!
//! The backend's own choices:
//!
//! - f32 activations;
//! - `objc2-metal` pinned by version, with no Objective-C++ shim;
//! - macOS 26.2 and above, refused by name below it;
//! - one command buffer per token, and no indirect command buffer.

pub mod arch;
/// The decode attention's geometry and segment rule, one file read by the
/// launch (macOS) and by the token count (every OS).
#[path = "kernels/attention_geom.rs"]
pub mod attention_geom;
/// The server's engine over any `runtime::Target`, host code: every OS.
pub mod server;

#[cfg(target_os = "macos")]
pub mod buffer;
#[cfg(target_os = "macos")]
pub mod census;
#[cfg(target_os = "macos")]
pub(crate) mod checkpoint;
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
pub mod residency;
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
pub use residency::{Residency, ResidencyStats};
#[cfg(target_os = "macos")]
pub use stream::{GpuInterval, GpuStamps, GpuTimes, HostEvent, MetalEvent, MetalStream, Recording};
