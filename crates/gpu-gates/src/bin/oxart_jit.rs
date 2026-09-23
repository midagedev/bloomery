//! What the driver allocates for the device code a binary carries:
//! `oxart_jit <section.bin>` reads the bytes `objcopy -O binary
//! --only-section=.oxart` wrote, loads every PTX module of the section into a
//! context on device 0 the way cuda-core loads a bundle (`cuModuleLoadData`
//! on the payload, default JIT options), and prints one line per entry,
//! module by module in section order:
//! `<entry>\t<regs>\t<local>\t<shared>\t<max_threads>` —
//! `CU_FUNC_ATTRIBUTE_NUM_REGS`, `LOCAL_SIZE_BYTES`, `SHARED_SIZE_BYTES`
//! (static) and `MAX_THREADS_PER_BLOCK` — after one `#` line that names the
//! card and the driver's CUDA version. `tools/ptx-scan.sh` joins the first
//! two as its `jit_regs` and `jit_local` columns.
//!
//! The kernels reach the card as PTX, so the registers a launch occupies are
//! the ones the driver's JIT compiler allocates, not the toolkit ptxas's the
//! scan's `regs` column reads; the two compilers need not agree.
//!
//! It creates a context and JIT-compiles, so it runs on a card: through
//! `tools/gpu-gate.sh` (the gate lock and its bound), never bare.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("oxart_jit: built without the `gpu` feature; see `just ptx-scan`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("oxart_jit", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), bloomery_gpu_gates::GateError> {
    use bloomery_gpu_gates::ptx;
    use cuda_core::{CudaContext, DriverError, sys};
    use std::path::PathBuf;

    let mut args = std::env::args_os().skip(1);
    let (Some(section), None) = (args.next(), args.next()) else {
        return Err("usage: oxart_jit <section.bin>".into());
    };
    let section = PathBuf::from(section);
    let bytes = std::fs::read(&section).map_err(|e| format!("read {}: {e}", section.display()))?;
    let bundles =
        ptx::section_bundles(&bytes).map_err(|e| format!("{}: {e}", section.display()))?;
    let modules = ptx::modules(&bundles).map_err(|e| format!("{}: {e}", section.display()))?;
    if modules.is_empty() {
        return Err(format!("{}: no PTX module in the section", section.display()).into());
    }

    let ctx = CudaContext::new(0)?;
    let mut version = 0i32;
    // SAFETY: the out-pointer is a live local the call only writes.
    let rc = unsafe { sys::cuDriverGetVersion(&mut version) };
    if rc != sys::cudaError_enum_CUDA_SUCCESS {
        return Err(DriverError(rc).into());
    }
    println!(
        "# card={} cuda_driver={}.{}",
        ctx.device_name()?.replace(' ', "_"),
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
