//! What the driver allocates for the device code a binary carries:
//! `oxart_jit <section.bin>` reads the bytes `objcopy -O binary
//! --only-section=.oxart` wrote, loads every PTX module of the section into a
//! context on device 0 the way cuda-core loads a bundle (`cuModuleLoadData`
//! on the payload, default JIT options), and prints one line per entry,
//! module by module in section order:
//! `<entry>\t<regs>\t<local>\t<shared>\t<max_threads>` —
//! `CU_FUNC_ATTRIBUTE_NUM_REGS`, `LOCAL_SIZE_BYTES`, `SHARED_SIZE_BYTES`
//! (static) and `MAX_THREADS_PER_BLOCK` — after one `#` line that names the
//! card, its compute capability and the driver's CUDA version.
//! `tools/ptx-scan.sh` joins the first two as its `jit_regs` and `jit_local`
//! columns, and a scan pair compares the capability (`jit-cc`), never the
//! card name: the JIT columns follow the capability and the driver, not which
//! card of one capability JIT-ed the module.
//!
//! The kernels reach the card as PTX, so the registers a launch occupies are
//! the ones the driver's JIT compiler allocates, not the toolkit ptxas's the
//! scan's `regs` column reads; the two compilers need not agree.
//!
//! `oxart_jit --cubin <section.bin> <out.bin>` writes that JIT's output
//! instead: each bundle's PTX linked by the driver (`cuLinkAddData`, default
//! JIT options, the target of the context), so the cubin is the code a
//! `cuModuleLoadData` of the PTX would run on this card and driver. `out.bin`
//! is a section of the same bundles in the same order, each with its name,
//! target, compile options and entries and the cubin as its one payload;
//! cuda-core's loader takes a cubin as it is and JITs nothing. Every cubin is
//! loaded once and each entry it names is resolved before the section is
//! written. It prints one `cubin <bundle> ptx=<bytes> cubin=<bytes>
//! seconds=<s>` line a bundle. A section with a bundle that carries no PTX is refused.
//! `tools/release/build.sh` puts the section in the GB10 release binary.
//!
//! It creates a context and JIT-compiles, so it runs on a card: through
//! `tools/gpu-gate.sh` (the gate lock and its bound), never bare.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("oxart_jit", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), bloomery_gpu_gates::GateError> {
    use bloomery_gpu_gates::ptx;
    use cuda_core::{DriverError, sys};
    use std::path::PathBuf;

    const USAGE: &str =
        "usage: oxart_jit <section.bin> | oxart_jit --cubin <section.bin> <out.bin>";
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let (section, out) = match &args[..] {
        [section] => (section, None),
        [flag, section, out] if flag.as_os_str() == "--cubin" => (section, Some(out)),
        _ => return Err(USAGE.into()),
    };
    let bytes = std::fs::read(section).map_err(|e| format!("read {}: {e}", section.display()))?;
    let bundles =
        ptx::section_bundles(&bytes).map_err(|e| format!("{}: {e}", section.display()))?;
    let modules = ptx::modules(&bundles).map_err(|e| format!("{}: {e}", section.display()))?;
    if modules.is_empty() {
        return Err(format!("{}: no PTX module in the section", section.display()).into());
    }

    let ctx = bloomery_gpu::capsync::fresh_context(0)?;
    if let Some(out) = out {
        if modules.len() != bundles.len() {
            return Err(format!("{}: a bundle carries no PTX to link", section.display()).into());
        }
        let mut written = Vec::new();
        for (b, m) in bundles.iter().zip(&modules) {
            written.extend(cubin_bundle(&ctx, b, m)?);
        }
        return std::fs::write(out, written)
            .map_err(|e| format!("write {}: {e}", out.display()).into());
    }

    let mut version = 0i32;
    // SAFETY: the out-pointer is a live local the call only writes.
    let rc = unsafe { sys::cuDriverGetVersion(&mut version) };
    if rc != sys::cudaError_enum_CUDA_SUCCESS {
        return Err(DriverError(rc).into());
    }
    let (cc_major, cc_minor) = ctx.compute_capability()?;
    println!(
        "# card={} cc={}.{} cuda_driver={}.{}",
        ctx.device_name()?.replace(' ', "_"),
        cc_major,
        cc_minor,
        version / 1000,
        version % 1000 / 10
    );
    for m in &modules {
        let loaded = ctx
            .load_module_from_image(m.text())
            .map_err(|e| format!("bundle {}: the driver refused its PTX: {e}", m.bundle()))?;
        for name in ptx::entries(std::slice::from_ref(m)) {
            let at = |e: DriverError| format!("bundle {} entry {name}: {e}", m.bundle());
            let f = loaded.load_function(&name).map_err(at)?;
            println!(
                "{name}\t{}\t{}\t{}\t{}",
                f.num_registers().map_err(at)?,
                f.local_size_bytes().map_err(at)?,
                f.static_shared_memory_bytes().map_err(at)?,
                f.max_threads_per_block().map_err(at)?
            );
        }
    }
    Ok(())
}

/// Bundle `b` with its PTX module `m` replaced by the driver's cubin of it ([`cubin`]), as a
/// container blob: the same name, target, compile options and entries, the cubin its one payload.
/// The cubin is loaded and every entry `m` names resolved in it before the blob is made.
#[cfg(feature = "gpu")]
fn cubin_bundle(
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
    b: &oxide_artifacts::OwnedArtifactBundle,
    m: &bloomery_gpu_gates::ptx::Module<'_>,
) -> Result<Vec<u8>, bloomery_gpu_gates::GateError> {
    use oxide_artifacts::{
        ArtifactBundleSpec, ArtifactEntrySpec, ArtifactPayloadKind, ArtifactPayloadSpec,
    };

    let start = std::time::Instant::now();
    let cubin = cubin(ctx, m)?;
    let loaded = ctx
        .load_module_from_image(&cubin)
        .map_err(|e| format!("bundle {}: the driver refused its cubin: {e}", b.name))?;
    for name in bloomery_gpu_gates::ptx::entries(std::slice::from_ref(m)) {
        loaded
            .load_function(&name)
            .map_err(|e| format!("bundle {} entry {name}: not in its cubin: {e}", b.name))?;
    }
    let payload = format!("{}.cubin", b.name);
    let mut spec = ArtifactBundleSpec::new(&b.name, &b.target)
        .with_compile_options(b.compile_options)
        .with_payload(ArtifactPayloadSpec::new(
            ArtifactPayloadKind::Cubin,
            &payload,
            &cubin,
        ));
    for e in &b.entries {
        let entry = ArtifactEntrySpec::new(&e.symbol, e.kind);
        spec = spec.with_entry(match e.metadata {
            Some(meta) => entry.with_metadata(meta),
            None => entry,
        });
    }
    let blob = oxide_artifacts::build_artifact_blob(&spec)
        .map_err(|e| format!("bundle {}: {e}", b.name))?;
    println!(
        "cubin {} ptx={} cubin={} seconds={:.1}",
        b.name,
        m.text().len(),
        cubin.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(blob)
}

/// The driver's link of `m`'s PTX into a cubin for `ctx`'s card: what its
/// `cuModuleLoadData` of the same text compiles, kept instead of loaded.
#[cfg(feature = "gpu")]
fn cubin(
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
    m: &bloomery_gpu_gates::ptx::Module<'_>,
) -> Result<Vec<u8>, bloomery_gpu_gates::GateError> {
    use cuda_core::{DriverError, sys};
    use std::ptr::null_mut;

    let at = |call: &str, rc: sys::CUresult| -> Result<(), bloomery_gpu_gates::GateError> {
        if rc == sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("bundle {}: {call}: {}", m.bundle(), DriverError(rc)).into())
        }
    };
    ctx.bind_to_thread()?;
    // The driver reads PTX text up to its NUL; the payload carries none.
    let mut text = m.text().to_vec();
    text.push(0);
    let name = std::ffi::CString::new(format!("{}.ptx", m.bundle()))?;
    let mut state: sys::CUlinkState = null_mut();
    // SAFETY: no options are passed; the out-pointer is a live local the call only writes.
    at("cuLinkCreate", unsafe {
        sys::cuLinkCreate_v2(0, null_mut(), null_mut(), &mut state)
    })?;
    let linked: Result<Vec<u8>, bloomery_gpu_gates::GateError> = (|| {
        // SAFETY: `state` is the live link the create above made; `text` and `name` outlive the
        // call, which copies the input, and `text`'s length counts its NUL.
        at("cuLinkAddData", unsafe {
            sys::cuLinkAddData_v2(
                state,
                sys::CUjitInputType_enum_CU_JIT_INPUT_PTX,
                text.as_mut_ptr().cast(),
                text.len(),
                name.as_ptr(),
                0,
                null_mut(),
                null_mut(),
            )
        })?;
        let mut image: *mut std::ffi::c_void = null_mut();
        let mut size = 0usize;
        // SAFETY: `state` is live; both out-pointers are live locals the call only writes.
        at("cuLinkComplete", unsafe {
            sys::cuLinkComplete(state, &mut image, &mut size)
        })?;
        if image.is_null() || size == 0 {
            return Err(format!("bundle {}: the link made an empty cubin", m.bundle()).into());
        }
        // SAFETY: the driver owns `size` bytes at `image` until `cuLinkDestroy`, which runs below
        // after this copy.
        Ok(unsafe { std::slice::from_raw_parts(image.cast::<u8>(), size) }.to_vec())
    })();
    // SAFETY: `state` is the link the create made, destroyed once, after its image was copied.
    let destroyed = unsafe { sys::cuLinkDestroy(state) };
    let cubin = linked?;
    at("cuLinkDestroy", destroyed)?;
    Ok(cubin)
}
