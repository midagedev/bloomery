//! The library the kernels come from: one instance's (`crate::instance`).
//! `build.rs` writes the kernels' source of every registered instance and
//! the library is the one of the instance it is opened at, which a file
//! selects by its shape.
//!
//! Two paths, and the gate prints which one ran:
//!
//! - the **metallib**: `build.rs` compiled the kernels with `xcrun metal`
//!   when the Metal Toolchain is installed, and the bytes are embedded.
//! - the **source**: the same source, with the fault site's table and the
//!   instance's geometry written in front of it, is compiled by the Metal
//!   runtime when it loads: the instance's alone.
//!
//! Both paths are given the same math options (safe math, precise
//! single-precision functions, language version 3.2) and the source turns
//! contraction off, so the two are meant to hold the same program; the gate's
//! arithmetic kernel compares them bit for bit when both exist. A machine with
//! neither — no metallib and a runtime that refuses the source — is a named
//! error.
//!
//! The prompt pass's GEMMs have a second program: the tensor-op GEMMs
//! (`kernels/gemm_tensor.metal`), compiled by the runtime from
//! [`TENSOR_SOURCE`] at language version 4.0 with the same math options, on
//! a GPU whose cores carry matrix units ([`MatrixUnits`]). The library holds
//! it beside its own, and the GEMMs' pipelines come from it
//! ([`Library::gemm_path`]); every other kernel stays the 3.2 library's.

use core::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunction,
    MTLFunctionConstantValues, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
};

use bloomery_device::DeviceError;

use crate::device::MetalDevice;
use crate::handle::Handle;
use crate::instance::{HOST, Instance};

/// One registered instance's embedded sources, as `build.rs` wrote them.
struct Built {
    /// The instance's name.
    name: &'static str,
    /// The kernels' source, the fault site's table and the instance's
    /// geometry in front of them.
    source: &'static str,
    /// The tensor-op GEMMs' source, with the same header in front.
    tensor: &'static str,
}

// `BUILT`, and `METALLIBS` when the Metal Toolchain was there.
include!(concat!(env!("OUT_DIR"), "/instances.rs"));

/// Whether two names are the same bytes.
const fn same_name(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The sources `build.rs` wrote for `inst`; none for an instance that is not
/// registered.
const fn built_of(inst: &Instance) -> Option<&'static Built> {
    let mut i = 0;
    while i < BUILT.len() {
        if same_name(BUILT[i].name, inst.name) {
            return Some(&BUILT[i]);
        }
        i += 1;
    }
    None
}

/// The sources of the host's instance ([`HOST`]): `build.rs` writes every
/// registered instance's, and the host's is registered.
const HOST_BUILT: &Built = match built_of(&HOST) {
    Some(built) => built,
    None => panic!("build.rs wrote no sources for the host's instance"),
};

/// The kernels' source of the host's instance ([`HOST`]), with the fault
/// site's table and its geometry in front of them: the text the kernel gate
/// edits its mutants into.
pub const SOURCE: &str = HOST_BUILT.source;

/// The tensor-op GEMMs' source of the host's instance
/// (`kernels/gemm_tensor.metal` after the files it reads), with the same
/// header in front: the text the runtime compiles at Metal 4, and the text
/// the kernel gate edits its tensor mutants into.
pub const TENSOR_SOURCE: &str = HOST_BUILT.tensor;

/// The sources of `inst`, or the named refusal of an instance `build.rs` did
/// not write.
fn sources(inst: &Instance) -> Result<&'static Built, DeviceError> {
    built_of(inst).ok_or_else(|| {
        DeviceError::unsupported(
            "library",
            format!(
                "no kernels are built for instance {}: build.rs writes the rows of \
                 instance::REGISTERED",
                inst.name
            ),
        )
    })
}

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

/// Which program the prompt pass's GEMMs run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmPath {
    /// The `simdgroup_matrix` kernels of the library itself.
    Simdgroup,
    /// The Metal 4 tensor-op kernels (`matmul2d`).
    TensorOps,
}

impl GemmPath {
    /// The path's name, as a record prints it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Simdgroup => "simdgroup",
            Self::TensorOps => "tensor",
        }
    }
}

/// Whether a GPU's cores carry the matrix units Metal 4's tensor ops reach:
/// the one fact the GEMM path is chosen by.
///
/// The rule is MLX's `is_nax_available` (mlx/backend/metal/device.cpp,
/// v0.32.1, :947–966 and the architecture parse at :588–601): the
/// architecture name ends in two generation digits and a class letter
/// (`applegpu_g13s`), and the units are there from generation 17 on, 18 on
/// for the phone class `p`. The macOS floor it also asks for, 26.2, is this
/// backend's own ([`crate::device`]). Metal 4's family is asked as well:
/// the tensor ops compile on no device without it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixUnits {
    /// `MTLDevice.architecture.name`.
    pub architecture: String,
    /// `supportsFamily(MTLGPUFamilyMetal4)`.
    pub metal4: bool,
}

impl MatrixUnits {
    /// The facts of `device`.
    pub(crate) fn of(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            architecture: device.architecture().name().to_string(),
            metal4: device.supportsFamily(MTLGPUFamily::Metal4),
        }
    }

    /// The architecture's generation and class, when its name ends in two
    /// digits and a letter.
    #[must_use]
    pub fn generation(&self) -> Option<(u32, char)> {
        let tail: Vec<char> = self.architecture.chars().rev().take(3).collect();
        match tail.as_slice() {
            [class, ones, tens] if class.is_ascii_alphabetic() => {
                Some((tens.to_digit(10)? * 10 + ones.to_digit(10)?, *class))
            }
            _ => None,
        }
    }

    /// The rule: Metal 4, and a generation that carries the units.
    #[must_use]
    pub fn present(&self) -> bool {
        self.metal4
            && self
                .generation()
                .is_some_and(|(generation, class)| generation >= if class == 'p' { 18 } else { 17 })
    }

    /// The GEMM path this device takes.
    #[must_use]
    pub fn path(&self) -> GemmPath {
        if self.present() {
            GemmPath::TensorOps
        } else {
            GemmPath::Simdgroup
        }
    }
}

/// A compute pipeline and the most threads a group of it may hold.
pub struct Pipeline {
    state: Handle<dyn MTLComputePipelineState>,
    max_threads_per_group: usize,
    /// The entry's name and, in angle brackets, its function-constant values:
    /// how the move proof's scan and encode trace name this pipeline.
    #[cfg(feature = "pipeline-label")]
    label: String,
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

    /// The entry's name and its function-constant values.
    #[cfg(feature = "pipeline-label")]
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// The library the kernels come from.
#[derive(Clone)]
pub struct Library {
    library: Handle<dyn MTLLibrary>,
    device: Handle<dyn MTLDevice>,
    path: LibraryPath,
    /// The instance whose sources the library is compiled from.
    instance: &'static Instance,
    rows_per_simd_group: Option<usize>,
    router_sg: Option<usize>,
    /// The tensor-op GEMMs' library, when the GEMMs run on them.
    tensor: Option<Box<Library>>,
}

#[cfg(feature = "pipeline-label")]
thread_local! {
    /// The function constants (index, `uint` value) of the pipeline the calling
    /// thread is about to make: set by the caller that specialized the
    /// function, taken by [`Library::pipeline_of`].
    static CONSTANTS: core::cell::RefCell<Vec<(usize, u32)>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

/// The label of an entry made with `constants`: the name, then the values in
/// angle brackets.
#[cfg(feature = "pipeline-label")]
fn label_of(entry: &str, constants: &[(usize, u32)]) -> String {
    if constants.is_empty() {
        return entry.to_string();
    }
    let values: Vec<String> = constants.iter().map(|(_, v)| v.to_string()).collect();
    format!("{entry}<{}>", values.join(","))
}

/// The value of `#define NAME <n>` in `text`, when one such line names a
/// whole number: the geometry the source's kernels were compiled with.
fn define_of(text: &str, name: &str) -> Option<usize> {
    let mut found = text.lines().filter_map(|line| {
        line.trim()
            .strip_prefix("#define ")?
            .strip_prefix(name)?
            .strip_prefix(' ')
            .and_then(|n| n.trim().parse().ok())
    });
    let n = found.next()?;
    found.next().is_none().then_some(n)
}

impl Library {
    /// The library of `instance` this machine's Metal device takes: the
    /// metallib when the build made one, the source otherwise, and the
    /// tensor-op GEMMs beside it on a GPU whose cores carry matrix units.
    /// An instance `build.rs` did not write is refused by name.
    pub(crate) fn compile(
        device: &Handle<dyn MTLDevice>,
        instance: &'static Instance,
    ) -> Result<Self, DeviceError> {
        let built = sources(instance)?;
        #[cfg(metal_lib = "metallib")]
        let library = Self::metallib_of(device, instance)?;
        #[cfg(not(metal_lib = "metallib"))]
        let library = Self::source_of(
            device,
            instance,
            built.source,
            MTLLanguageVersion::Version3_2,
        )?;
        match MatrixUnits::of(device).path() {
            GemmPath::Simdgroup => Ok(library),
            GemmPath::TensorOps => {
                let tensor = Self::source_of(
                    device,
                    instance,
                    built.tensor,
                    MTLLanguageVersion::Version4_0,
                )?;
                Ok(library.with_tensor_gemms(tensor))
            }
        }
    }

    /// The library the runtime compiles from the source of the device's
    /// instance.
    pub fn from_source(device: &MetalDevice) -> Result<Self, DeviceError> {
        let instance = device.library().instance;
        let built = sources(instance)?;
        Self::source_of(
            device.handle(),
            instance,
            built.source,
            MTLLanguageVersion::Version3_2,
        )
    }

    /// The library the runtime compiles from the given source text, at the
    /// device's instance. For the kernels gate alone, which is why it is
    /// public: its mutants (a defect edited into the shipped source's text)
    /// and its gate-only kernels appended to that text. The engine takes
    /// [`Library::from_source`].
    pub fn from_source_text(device: &MetalDevice, text: &str) -> Result<Self, DeviceError> {
        Self::source_of(
            device.handle(),
            device.library().instance,
            text,
            MTLLanguageVersion::Version3_2,
        )
    }

    /// The tensor-op GEMMs' library the runtime compiles at Metal 4 from
    /// `text` ([`TENSOR_SOURCE`], or the gate's mutant of it), at the
    /// device's instance, on any device that compiles it: the gate and the
    /// probe run it whatever [`MatrixUnits`] says. A device whose runtime
    /// refuses it is a named error.
    pub fn tensor_from_source_text(device: &MetalDevice, text: &str) -> Result<Self, DeviceError> {
        Self::source_of(
            device.handle(),
            device.library().instance,
            text,
            MTLLanguageVersion::Version4_0,
        )
    }

    /// The instance the library is compiled at.
    #[must_use]
    pub fn instance(&self) -> &'static Instance {
        self.instance
    }

    /// This library with its GEMMs taken from `tensor`, a library of
    /// [`Library::tensor_from_source_text`]: every other kernel stays this
    /// library's.
    #[must_use]
    pub fn with_tensor_gemms(&self, tensor: Library) -> Self {
        Self {
            tensor: Some(Box::new(tensor)),
            ..self.clone()
        }
    }

    /// This library with the simdgroup GEMMs, whatever its device picked:
    /// the kernel gate's clauses of those kernels run them on every device.
    #[must_use]
    pub fn with_simdgroup_gemms(&self) -> Self {
        Self {
            tensor: None,
            ..self.clone()
        }
    }

    /// The library the GEMMs' pipelines come from.
    pub(crate) fn gemm_library(&self) -> &Library {
        self.tensor.as_deref().unwrap_or(self)
    }

    /// Which program the GEMMs run.
    #[must_use]
    pub fn gemm_path(&self) -> GemmPath {
        match self.tensor {
            Some(_) => GemmPath::TensorOps,
            None => GemmPath::Simdgroup,
        }
    }

    /// The facts the GEMM path of this library's device is chosen by.
    #[must_use]
    pub fn matrix_units(&self) -> MatrixUnits {
        MatrixUnits::of(&self.device)
    }

    fn source_of(
        device: &Handle<dyn MTLDevice>,
        instance: &'static Instance,
        text: &str,
        version: MTLLanguageVersion,
    ) -> Result<Self, DeviceError> {
        let source = NSString::from_str(text);
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        options.setLanguageVersion(version);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| DeviceError::backend("library", e.localizedDescription().to_string()))?;
        Ok(Self {
            library: Handle(library),
            device: device.clone(),
            path: LibraryPath::Source,
            instance,
            rows_per_simd_group: define_of(text, "BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP"),
            router_sg: define_of(text, "BLOOMERY_ROUTER_SG"),
            tensor: None,
        })
    }

    /// The library of the device's instance's embedded metallib, when the
    /// build made one.
    #[cfg(metal_lib = "metallib")]
    pub fn from_metallib(device: &MetalDevice) -> Result<Self, DeviceError> {
        Self::metallib_of(device.handle(), device.library().instance)
    }

    #[cfg(metal_lib = "metallib")]
    fn metallib_of(
        device: &Handle<dyn MTLDevice>,
        instance: &'static Instance,
    ) -> Result<Self, DeviceError> {
        let built = sources(instance)?;
        let &(_, bytes) = METALLIBS
            .iter()
            .find(|(name, _)| *name == instance.name)
            .ok_or_else(|| {
                DeviceError::unsupported(
                    "library",
                    format!(
                        "build.rs compiled no metallib for instance {}",
                        instance.name
                    ),
                )
            })?;
        let data = dispatch2::DispatchData::from_static_bytes(bytes);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|e| DeviceError::backend("library", e.localizedDescription().to_string()))?;
        Ok(Self {
            library: Handle(library),
            device: device.clone(),
            path: LibraryPath::Metallib,
            instance,
            rows_per_simd_group: define_of(built.source, "BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP"),
            router_sg: define_of(built.source, "BLOOMERY_ROUTER_SG"),
            tensor: None,
        })
    }

    /// The library object's address: what tells a pipeline's library from
    /// another's in the scan.
    #[cfg(feature = "pipeline-scan")]
    #[must_use]
    pub fn scan_id(&self) -> usize {
        core::ptr::from_ref::<ProtocolObject<dyn MTLLibrary>>(&self.library).addr()
    }

    /// The rows a SIMD group owns and the router's SIMD groups, as the
    /// source's `#define`s name them; none when the source does not name a
    /// count exactly once.
    #[must_use]
    pub fn geometry(&self) -> (Option<usize>, Option<usize>) {
        (self.rows_per_simd_group, self.router_sg)
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
        self.pipeline_of(function)
    }

    /// The compute pipeline of the named function, specialized with the
    /// mat-vec's SIMD group count as its function constant. A function that
    /// reads a function constant has no pipeline without one: the count is
    /// what the kernel's geometry is.
    pub fn pipeline_with_simdgroups(
        &self,
        name: &str,
        simdgroups: u32,
    ) -> Result<Pipeline, DeviceError> {
        let values = MTLFunctionConstantValues::new();
        let mut count = simdgroups;
        // SAFETY: the constant values copy the value the pointer names, and
        // the pointer is valid for the call.
        unsafe {
            values.setConstantValue_type_atIndex(
                NonNull::from(&mut count).cast(),
                MTLDataType::UInt,
                0,
            );
        }
        let wanted = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&wanted, &values)
            .map_err(|e| DeviceError::backend("library", e.localizedDescription().to_string()))?;
        #[cfg(feature = "pipeline-label")]
        CONSTANTS.with_borrow_mut(|constants| *constants = vec![(0, simdgroups)]);
        self.pipeline_of(function)
    }

    /// The pipeline of a function.
    fn pipeline_of(
        &self,
        function: objc2::rc::Retained<ProtocolObject<dyn MTLFunction>>,
    ) -> Result<Pipeline, DeviceError> {
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| DeviceError::backend("pipeline", e.localizedDescription().to_string()))?;
        let max_threads_per_group = state.maxTotalThreadsPerThreadgroup();
        #[cfg(feature = "pipeline-label")]
        let constants = CONSTANTS.with_borrow_mut(core::mem::take);
        let pipeline = Pipeline {
            state: Handle(state),
            max_threads_per_group,
            #[cfg(feature = "pipeline-label")]
            label: label_of(&function.name().to_string(), &constants),
        };
        #[cfg(feature = "pipeline-scan")]
        crate::scan::record(self, &function, &constants, &pipeline);
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(architecture: &str, metal4: bool) -> MatrixUnits {
        MatrixUnits {
            architecture: architecture.to_string(),
            metal4,
        }
    }

    #[test]
    fn the_matrix_unit_rule_is_mlx_s() {
        // The M1 Pro and the M4 Pro this backend is gated on: no units.
        assert_eq!(units("applegpu_g13s", true).path(), GemmPath::Simdgroup);
        assert_eq!(units("applegpu_g16s", true).path(), GemmPath::Simdgroup);
        // Generation 17 carries them outside the phone class, 18 inside it.
        assert_eq!(units("applegpu_g17s", true).path(), GemmPath::TensorOps);
        assert_eq!(units("applegpu_g17p", true).path(), GemmPath::Simdgroup);
        assert_eq!(units("applegpu_g18p", true).path(), GemmPath::TensorOps);
        // No Metal 4, or a name the rule cannot read: no units.
        assert_eq!(units("applegpu_g17s", false).path(), GemmPath::Simdgroup);
        assert_eq!(units("applegpu", true).generation(), None);
        assert_eq!(units("applegpu", true).path(), GemmPath::Simdgroup);
        assert_eq!(units("applegpu_g17s", true).generation(), Some((17, 's')));
    }
}
