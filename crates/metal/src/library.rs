//! The library the kernels come from.
//!
//! Two paths, and the gate prints which one ran:
//!
//! - the **metallib**: `build.rs` compiled the kernels with `xcrun metal`
//!   when the Metal Toolchain is installed, and the bytes are embedded.
//! - the **source**: the same source, with the fault site's table written
//!   in front of it, is compiled by the Metal runtime when it loads.
//!
//! Both paths are given the same math options (safe math, precise
//! single-precision functions, language version 3.2) and the source turns
//! contraction off, so the two are meant to hold the same program; the gate's
//! arithmetic kernel compares them bit for bit when both exist. A machine with
//! neither — no metallib and a runtime that refuses the source — is a named
//! error.

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLFunction, MTLLanguageVersion,
    MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode,
};

use bloomery_device::DeviceError;

use crate::device::MetalDevice;
use crate::handle::Handle;

/// The kernels' source, with the fault site's table in front of them.
const SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/bloomery.metal"));

/// The metallib `build.rs` compiled, when the Metal Toolchain was there.
#[cfg(metal_lib = "metallib")]
const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bloomery.metallib"));

/// Which path the library came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryPath {
    /// The metallib built at build time.
    Metallib,
    /// The source, compiled by the runtime.
    Source,
}

impl LibraryPath {
    /// The path's name, as a record prints it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Metallib => "metallib",
            Self::Source => "source",
        }
    }
}

/// A compute pipeline and the most threads a group of it may hold.
pub struct Pipeline {
    state: Handle<dyn MTLComputePipelineState>,
    max_threads_per_group: usize,
}

impl Pipeline {
    /// The pipeline's own Metal object.
    pub(crate) fn metal(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }

    /// The most threads a group of this pipeline may hold.
    #[must_use]
    pub fn max_threads_per_group(&self) -> usize {
        self.max_threads_per_group
    }
}

/// The library the kernels come from.
pub struct Library {
    library: Handle<dyn MTLLibrary>,
    device: Handle<dyn MTLDevice>,
    path: LibraryPath,
}

impl Library {
    /// The library this machine's Metal device takes: the metallib when the
    /// build made one, the source otherwise.
    pub(crate) fn compile(device: &Handle<dyn MTLDevice>) -> Result<Self, DeviceError> {
        #[cfg(metal_lib = "metallib")]
        {
            Self::metallib_of(device)
        }
        #[cfg(not(metal_lib = "metallib"))]
        {
            Self::source_of(device)
        }
    }

    /// The library the runtime compiles from the source.
    pub fn from_source(device: &MetalDevice) -> Result<Self, DeviceError> {
        Self::source_of(device.handle())
    }

    fn source_of(device: &Handle<dyn MTLDevice>) -> Result<Self, DeviceError> {
        let source = NSString::from_str(SOURCE);
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| DeviceError::backend("library", e.localizedDescription().to_string()))?;
        Ok(Self {
            library: Handle(library),
            device: device.clone(),
            path: LibraryPath::Source,
        })
    }

    /// The library of the embedded metallib, when the build made one.
    #[cfg(metal_lib = "metallib")]
    pub fn from_metallib(device: &MetalDevice) -> Result<Self, DeviceError> {
        Self::metallib_of(device.handle())
    }

    #[cfg(metal_lib = "metallib")]
    fn metallib_of(device: &Handle<dyn MTLDevice>) -> Result<Self, DeviceError> {
        let data = dispatch2::DispatchData::from_bytes(METALLIB);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|e| DeviceError::backend("library", e.localizedDescription().to_string()))?;
        Ok(Self {
            library: Handle(library),
            device: device.clone(),
            path: LibraryPath::Metallib,
        })
    }

    /// Which path the library came from.
    #[must_use]
    pub fn path(&self) -> LibraryPath {
        self.path
    }

    /// The named function of the library.
    fn function(
        &self,
        name: &str,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLFunction>>, DeviceError> {
        let wanted = NSString::from_str(name);
        self.library
            .newFunctionWithName(&wanted)
            .ok_or_else(|| DeviceError::backend("library", format!("no function named {name}")))
    }

    /// The compute pipeline of the named function.
    pub fn pipeline(&self, name: &str) -> Result<Pipeline, DeviceError> {
        let function = self.function(name)?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| DeviceError::backend("pipeline", e.localizedDescription().to_string()))?;
        let max_threads_per_group = state.maxTotalThreadsPerThreadgroup();
        Ok(Pipeline {
            state: Handle(state),
            max_threads_per_group,
        })
    }
}
