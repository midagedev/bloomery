//! The library the kernels come from: one instance's (`crate::instance`),
//! the one a file selects by its shape.
//!
//! Two paths, and the gate prints which one ran:
//!
//! - the **metallib**: `build.rs` compiled the kernels of every registered
//!   instance with `xcrun metal` when the Metal Toolchain is installed, and
//!   the bytes are embedded.
//! - the **source**: the kernels' files with the fault site's table and the
//!   instance's geometry written in front of them (`crate::header`, rendered
//!   at open from the instance's [`Geometry`]), compiled by the Metal runtime
//!   when it loads: the instance's alone. It is the path of a build with no
//!   Metal Toolchain and of an instance no row registers, so a file whose
//!   geometry every kernel takes opens on a build that has its metallib for
//!   other shapes only.
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
use crate::geom::Geometry;
use crate::handle::Handle;
use crate::header;
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

/// Every registered instance's embedded sources, in `instance::REGISTERED`'s
/// order (build.rs writes the table).
const BUILT: &[Built] = include!(concat!(env!("OUT_DIR"), "/built.rs"));

/// The kernels' files joined as `build.rs` joins them (`header::files`): the
/// part of the program text no instance changes. The open writes an
/// instance's header in front of it.
const MAIN_BODY: &str = include_str!(concat!(env!("OUT_DIR"), "/main_body.metal"));

/// The tensor-op GEMMs' files joined, the same way.
const TENSOR_BODY: &str = include_str!(concat!(env!("OUT_DIR"), "/tensor_body.metal"));

/// Each instance's metallib, when the Metal Toolchain was there.
#[cfg(metal_lib = "metallib")]
const METALLIBS: &[(&str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/metallibs.rs"));

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
/// edits its mutants into. A shim: the build's host instance's
/// ([`sources_of`]). Callers left: the kernel gate, the probes and the
/// gates' shared mutants.
pub const SOURCE: &str = HOST_BUILT.source;

/// The tensor-op GEMMs' source of the host's instance
/// (`kernels/gemm_tensor.metal` after the files it reads), with the same
/// header in front: the text the runtime compiles at Metal 4, and the text
/// the kernel gate edits its tensor mutants into. A shim: the build's host
/// instance's ([`sources_of`]). Callers left: the e2e and kernel gates, the
/// probes and the gates' shared mutants.
pub const TENSOR_SOURCE: &str = HOST_BUILT.tensor;

/// The kernels' source and the tensor-op GEMMs' source of `inst`, in that
/// order, each with the fault site's table and the instance's geometry in
/// front of it, as `build.rs` embedded them; the named refusal of an instance
/// `build.rs` did not write. [`rendered`] writes the same text at open, for
/// any instance.
pub fn sources_of(inst: &Instance) -> Result<(&'static str, &'static str), DeviceError> {
    let built = built_of(inst).ok_or_else(|| {
        DeviceError::unsupported(
            "library",
            format!(
                "no kernels are embedded for {inst}: build.rs writes the rows of \
                 instance::REGISTERED"
            ),
        )
    })?;
    Ok((built.source, built.tensor))
}

/// The kernels' source and the tensor-op GEMMs' source of any instance, in
/// that order: the embedded text for a registered one, [`rendered`]'s for one
/// made at open. The text the gates edit their mutants into, so a mutant can
/// run on an instance no row registers.
///
/// # Errors
///
/// [`DeviceError`] naming the geometry some kernel cannot take.
pub fn sources_for(inst: &Instance) -> Result<(String, String), DeviceError> {
    match built_of(inst) {
        Some(built) => Ok((built.source.to_string(), built.tensor.to_string())),
        None => rendered(inst),
    }
}

/// The kernels' source and the tensor-op GEMMs' source of `inst`, in that
/// order, written at open from its geometry: the header of [`Geometry::of`]
/// in front of the kernels' files. For a registered instance it is the text
/// `build.rs` embedded, byte for byte. A geometry some kernel cannot take is
/// refused by name, before any compile.
fn rendered(inst: &Instance) -> Result<(String, String), DeviceError> {
    let g = Geometry::of(inst);
    g.check()
        .map_err(|why| DeviceError::unsupported("library", format!("{inst}: {why}")))?;
    Ok((
        header::program(&g, MAIN_BODY),
        header::program(&g, TENSOR_BODY),
    ))
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

    /// The SIMD width this pipeline runs at.
    #[must_use]
    pub fn execution_width(&self) -> usize {
        self.state.threadExecutionWidth()
    }

    /// The threadgroup memory the pipeline's function declares, in bytes.
    #[must_use]
    pub fn static_group_bytes(&self) -> usize {
        self.state.staticThreadgroupMemoryLength()
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

impl Library {
    /// The library of `instance` this machine's Metal device takes: the
    /// metallib when the build made one for it, the source otherwise (a
    /// build with no Metal Toolchain, or an instance no row registers), and
    /// the tensor-op GEMMs beside it on a GPU whose cores carry matrix units.
    /// A geometry some kernel cannot take is refused by name.
    pub(crate) fn compile(
        device: &Handle<dyn MTLDevice>,
        instance: &'static Instance,
    ) -> Result<Self, DeviceError> {
        let (source, tensor_source) = rendered(instance)?;
        #[cfg(metal_lib = "metallib")]
        let library = match METALLIBS.iter().find(|(name, _)| *name == instance.name) {
            Some(&(_, bytes)) => Self::metallib_of(device, instance, bytes, &source)?,
            None => Self::source_of(device, instance, &source, MTLLanguageVersion::Version3_2)?,
        };
        #[cfg(not(metal_lib = "metallib"))]
        let library = Self::source_of(device, instance, &source, MTLLanguageVersion::Version3_2)?;
        match MatrixUnits::of(device).path() {
            GemmPath::Simdgroup => Ok(library),
            GemmPath::TensorOps => {
                let tensor = Self::source_of(
                    device,
                    instance,
                    &tensor_source,
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
        let (source, _) = rendered(instance)?;
        Self::source_of(
            device.handle(),
            instance,
            &source,
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
            rows_per_simd_group: header::define_of(text, "BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP"),
            router_sg: header::define_of(text, "BLOOMERY_ROUTER_SG"),
            tensor: None,
        })
    }

    /// The library of the device's instance's embedded metallib, when the
    /// build made one for it.
    #[cfg(metal_lib = "metallib")]
    pub fn from_metallib(device: &MetalDevice) -> Result<Self, DeviceError> {
        let instance = device.library().instance;
        let &(_, bytes) = METALLIBS
            .iter()
            .find(|(name, _)| *name == instance.name)
            .ok_or_else(|| {
                DeviceError::unsupported(
                    "library",
                    format!("build.rs compiled no metallib for {instance}"),
                )
            })?;
        let (source, _) = rendered(instance)?;
        Self::metallib_of(device.handle(), instance, bytes, &source)
    }

    /// The library of an instance's metallib `bytes`; `source` is the text
    /// the metallib was compiled from, which names the geometry the library
    /// holds.
    #[cfg(metal_lib = "metallib")]
    fn metallib_of(
        device: &Handle<dyn MTLDevice>,
        instance: &'static Instance,
        bytes: &'static [u8],
        source: &str,
    ) -> Result<Self, DeviceError> {
        let data = dispatch2::DispatchData::from_static_bytes(bytes);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|e| DeviceError::backend("library", e.localizedDescription().to_string()))?;
        Ok(Self {
            library: Handle(library),
            device: device.clone(),
            path: LibraryPath::Metallib,
            instance,
            rows_per_simd_group: header::define_of(source, "BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP"),
            router_sg: header::define_of(source, "BLOOMERY_ROUTER_SG"),
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
    use crate::instance::{REGISTERED, UNREGISTERED, open_key};

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

    /// The first byte at which two texts differ, for a failure to name.
    fn differs_at(a: &str, b: &str) -> usize {
        a.bytes()
            .zip(b.bytes())
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()))
    }

    #[test]
    fn the_rendered_sources_are_the_built_ones() {
        for inst in REGISTERED {
            let (built, built_tensor) =
                sources_of(inst).expect("a registered instance has sources");
            let (source, tensor) = rendered(inst).expect("a registered geometry is taken");
            assert!(
                source == built,
                "{inst}: the kernels' source differs from build.rs's at byte {}",
                differs_at(&source, built)
            );
            assert!(
                tensor == built_tensor,
                "{inst}: the tensor-op source differs from build.rs's at byte {}",
                differs_at(&tensor, built_tensor)
            );
        }
    }

    #[test]
    fn an_unregistered_geometry_renders_its_own_header() {
        let mut key = REGISTERED[0].key;
        key.hidden = 3072;
        let inst = open_key(key).expect("a geometry every kernel takes opens");
        assert_eq!(inst.name, UNREGISTERED);
        assert!(
            sources_of(inst).is_err(),
            "build.rs embedded no source for it"
        );
        let (source, tensor) = rendered(inst).expect("its geometry is taken");
        assert_eq!(
            header::define_of(&source, "BLOOMERY_ROUTER_K"),
            Some(3072),
            "the router's K in the header"
        );
        assert_eq!(
            header::define_of(&tensor, "BLOOMERY_ROUTER_K"),
            Some(3072),
            "the router's K in the tensor-op header"
        );
        assert!(source.ends_with(MAIN_BODY) && tensor.ends_with(TENSOR_BODY));
    }

    #[test]
    fn a_geometry_no_kernel_takes_is_refused_before_a_compile() {
        let mut inst = REGISTERED[0];
        inst.name = UNREGISTERED;
        inst.key.hidden = 3136;
        let Err(e) = rendered(&inst) else {
            panic!("hidden 3136 is not a multiple of the router's 1024");
        };
        let text = e.to_string();
        assert!(
            text.contains("router: hidden is 3136; it must be a multiple of 1024"),
            "{text}"
        );
    }
}
