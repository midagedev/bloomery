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
/// The gated delta step's geometry, one file read by the launch (macOS) and
/// by `build.rs`.
#[path = "kernels/delta_geom.rs"]
pub mod delta_geom;
/// The expert pair and fold geometry, one file read by the launch (macOS) and
/// by `build.rs`.
#[path = "kernels/experts_geom.rs"]
pub mod experts_geom;
/// The launch geometry of one instance, resolved once: the owner of every
/// number the host sizes a launch by, which `build.rs` writes the kernels'
/// header from too. Host arithmetic: every OS.
#[path = "kernels/geom.rs"]
pub mod geom;
/// The mat-vec launch geometry, one file read by the launches (macOS) and by
/// `build.rs`.
#[path = "kernels/geometry.rs"]
pub mod geometry;
/// The kernels' program text: the header of one geometry in front of the
/// kernels' files, written by `build.rs` for every registered instance and by
/// the runtime at open. Host text: every OS.
#[path = "kernels/header.rs"]
pub mod header;
/// The kernels' instances: one per registered shape row, each the axes one
/// translation unit of the kernels is compiled at (a router's experts, picks
/// and hidden width, decode attention's head, rope and heads, the delta
/// step's heads and columns, the experts' width). The router and flash rows
/// are the shape table's (`models::shape`), the rows the CUDA bodies are
/// built at. `build.rs` writes and compiles every registered instance; a
/// file selects its instance by its shape, and a shape no row registers is
/// compiled at open when every kernel takes its geometry, else refused by
/// name. Host arithmetic: every OS.
pub mod instance;
/// The router's geometry, one file read by the launch (macOS) and by
/// `build.rs`.
#[path = "kernels/router_geom.rs"]
pub mod router_geom;
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
#[cfg(all(target_os = "macos", feature = "pipeline-scan"))]
pub mod scan;
#[cfg(target_os = "macos")]
pub mod stream;
#[cfg(all(target_os = "macos", feature = "encode-trace"))]
pub mod trace;

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
