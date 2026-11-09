//! The card squeezed until it refuses a page, for the clauses that grow a store past what the
//! card has free (`kvgrow`): the fit asks for a page, the card has none, and the body frees
//! expert bytes for it.

use bloomery_gpu::Gpu;
use bloomery_gpu_gates::GateError;
use cuda_core::DeviceBuffer;

/// Device memory held until the card refuses one more page: buffers of a gigabyte, halving to
/// one page as the card refuses them, until less than a page is free or a page-sized buffer is
/// refused for lack of memory (the driver keeps a reserve the free bytes count). Any other
/// refusal is refused by name. The memory is given back when the buffers drop.
pub fn squeeze(gpu: &Gpu) -> Result<Vec<DeviceBuffer<u32>>, GateError> {
    let page = gpu.allocation_granularity()?;
    let mut held = Vec::new();
    let mut chunk = 1usize << 30;
    loop {
        let free = gpu.mem_info()?.0;
        if free < page {
            return Ok(held);
        }
        let take = chunk.min(free / page * page);
        match DeviceBuffer::<u32>::zeroed(gpu.stream(), take / 4) {
            Ok(b) => held.push(b),
            Err(_) if chunk > page => chunk /= 2,
            Err(e) if e.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY => {
                return Ok(held);
            }
            Err(e) => {
                return Err(format!(
                    "the squeeze: a {take} B buffer refused with {free} B free: {e}"
                )
                .into());
            }
        }
    }
}
