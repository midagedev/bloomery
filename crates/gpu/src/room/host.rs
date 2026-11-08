//! A room page's memory can sit on the host: the driver's `cuMemCreate` at
//! the host NUMA node the device names (`CU_MEM_LOCATION_TYPE_HOST_NUMA`,
//! the id `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` reports), mapped into a room's
//! reserved range and granted to the device exactly as a card page, so every
//! kernel and capture over the store reads and writes it through the same
//! pointer. `cuda_core::vmm` can make only device-located allocations and
//! its `Mapping` takes a `&PhysicalAllocation`, so this module owns the
//! host-located handle ([`HostPage`]) and the mapping of a raw handle
//! ([`RawMap`], card allocations included) a room maps every page through.
//! The header's other host location, `CU_MEM_LOCATION_TYPE_HOST` ("id is
//! ignored"), asks for a CPU allocation that targets no NUMA node, while a
//! store's tail pages want the node nearest the device, which the header
//! pins to `HOST_NUMA` with the device's id (a system that reports no NUMA
//! must pass id 0 — the header's rule, which this module follows rather
//! than choosing a location of its own).

use std::sync::{Mutex, PoisonError};

use cuda_core::sys::{self, CUdevice};

use crate::GpuError;

/// Where a device's host pages sit and at what granularity the driver makes
/// them, read once a device: the NUMA id (0 on a system that reports none,
/// -1 being the driver's "no NUMA" marker), and the minimum granularity of
/// a `cuMemCreate` at that location.
#[derive(Clone, Copy)]
struct Placing {
    id: std::ffi::c_int,
    granularity: usize,
}

/// An allocation prop of the pinned type, zeroed as the C idiom's memset-0
/// descriptor and carrying nothing this module does not set.
fn pinned_prop() -> sys::CUmemAllocationProp_st {
    // SAFETY: an all-zero `CUmemAllocationProp` is the memset-0 descriptor
    // the C examples and cuda-core's own constructors build; every field
    // left zero is one the driver reads as "none".
    let mut prop: sys::CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
    prop.type_ = sys::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
    prop
}

/// The prop's location set to the host NUMA node `id`.
fn at_host_node(prop: &mut sys::CUmemAllocationProp_st, id: std::ffi::c_int) {
    prop.location.type_ = sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_HOST_NUMA;
    sys::set_mem_location_id(&mut prop.location, id);
}

/// `device`'s host placing, asked of the driver without a context and
/// cached. Refused by name where the device cannot host a VMM page: it must
/// report `CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
/// (the attribute exists for exactly this), and a NUMA id below -1 is one
/// the driver leaves undefined — a placement we were not told about is not
/// chosen for the caller.
fn placing(device: CUdevice) -> Result<Placing, GpuError> {
    static PLACINGS: Mutex<Vec<Option<Placing>>> = Mutex::new(Vec::new());
    let mut placings = PLACINGS.lock().unwrap_or_else(PoisonError::into_inner);
    let i = usize::try_from(device)
        .map_err(|_| GpuError::shape("room", format!("a device id of {device}")))?;
    if let Some(p) = placings.get(i).copied().flatten() {
        return Ok(p);
    }
    let numa_vmm = crate::raw_device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        "cuDeviceGetAttribute(HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)",
    )?;
    if numa_vmm == 0 {
        return Err(GpuError::State {
            what: "room",
            missing: "the device's host NUMA virtual memory management",
        });
    }
    let reported = crate::raw_device_attribute(
        device,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        "cuDeviceGetAttribute(HOST_NUMA_ID)",
    )?;
    let id = match reported {
        -1 => 0,
        id if id >= 0 => id,
        id => {
            return Err(GpuError::shape(
                "room",
                format!("a host NUMA id of {id}, which the driver leaves undefined"),
            ));
        }
    };
    let mut prop = pinned_prop();
    at_host_node(&mut prop, id);
    let mut granularity = 0usize;
    // SAFETY: `granularity` is a live out-pointer for the call and `prop` a
    // pinned-type prop at a host NUMA location the device reported; on
    // success the driver wrote the size, which this function returns.
    let rc = unsafe {
        sys::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            sys::CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    };
    crate::graph::cu(rc, "cuMemGetAllocationGranularity(HOST_NUMA)")?;
    let placing = Placing { id, granularity };
    if placings.len() <= i {
        placings.resize(i + 1, None);
    }
    placings[i] = Some(placing);
    Ok(placing)
}

/// The page a room that can back pages on the host takes on visible device
/// `ordinal`: the least common multiple of the card location's and the host
/// location's granularities, a size `cuMemCreate` accepts at either. Read
/// once a device ([`placing`]); a card-only room keeps the card's own
/// granularity ([`crate::allocation_granularity`]).
pub(crate) fn page_for(ordinal: usize) -> Result<usize, GpuError> {
    let device = cuda_core::Device::raw_device(ordinal)?;
    let (card, host) = (
        crate::allocation_granularity(ordinal)?,
        placing(device)?.granularity,
    );
    let (mut gcd, mut rest) = (card, host);
    while rest != 0 {
        (gcd, rest) = (rest, gcd % rest);
    }
    (card / gcd)
        .checked_mul(host)
        .ok_or_else(|| GpuError::shape("room", "the two locations' granularities pass usize"))
}

/// `bytes` of host memory on the NUMA node the device names, owned by the
/// handle `cuMemCreate` returned and freed by `cuMemRelease` on drop. The
/// calling thread holds a current context on the device, and `bytes` is a
/// multiple of the host location's granularity ([`page_for`] picks one).
pub(crate) struct HostPage {
    handle: sys::CUmemGenericAllocationHandle,
}

impl HostPage {
    /// The allocation; the driver's refusal names the call, and it leaves
    /// the device's own free bytes untouched.
    pub(crate) fn new(device: CUdevice, bytes: usize) -> Result<HostPage, GpuError> {
        let placing = placing(device)?;
        let mut prop = pinned_prop();
        at_host_node(&mut prop, placing.id);
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: `handle` is a live out-pointer for the call, and the prop
        // names the pinned type at a host NUMA location the device reported;
        // on success the driver wrote the handle, which this owner releases
        // exactly once, on drop.
        let rc = unsafe { sys::cuMemCreate(&mut handle, bytes, &prop, 0) };
        crate::graph::cu(rc, "cuMemCreate (the host)")?;
        Ok(HostPage { handle })
    }

    /// The raw allocation handle, for a mapping ([`RawMap`]).
    pub(crate) fn handle(&self) -> sys::CUmemGenericAllocationHandle {
        self.handle
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        // SAFETY: the handle is the one `cuMemCreate` returned and this
        // owner releases it exactly once; a mapping that still holds the
        // allocation keeps it alive, as the driver defers freeing while any
        // stands.
        let rc = unsafe { sys::cuMemRelease(self.handle) };
        if let Err(e) = crate::graph::cu(rc, "cuMemRelease (the host)") {
            eprintln!("room: a host page's release failed on drop: {e}");
        }
    }
}

/// A mapping of `bytes` of a raw allocation handle at `va`, unmapped on
/// drop: what a room maps a page through, be the memory a card's
/// `PhysicalAllocation` or a [`HostPage`] — `cuda_core::vmm`'s `Mapping`
/// takes only the former. A mapping must be dropped before the address
/// range it lies in is freed, and it holds the allocation it maps alive.
pub(crate) struct RawMap {
    va: u64,
    bytes: usize,
}

impl RawMap {
    /// `bytes` of `handle` mapped at `va`, from byte `offset` of the
    /// allocation; `va` must lie in a reserved range the caller keeps, and
    /// `bytes` and `offset` be multiples of the allocation's granularity.
    /// The mapping is not accessible until the caller grants a device
    /// access (`vmm::set_access`), as with any VMM mapping.
    pub(crate) fn new(
        va: u64,
        bytes: usize,
        handle: sys::CUmemGenericAllocationHandle,
        offset: usize,
    ) -> Result<RawMap, GpuError> {
        // SAFETY: `va` names `bytes` of a reserved virtual range that stands
        // until this mapping drops, and `handle` an allocation of at least
        // `offset + bytes`; the driver fills exactly that virtual range,
        // which no other mapping covers (a room unmaps a page before
        // mapping it again).
        let rc = unsafe { sys::cuMemMap(va, bytes, offset, handle, 0) };
        crate::graph::cu(rc, "cuMemMap")?;
        Ok(RawMap { va, bytes })
    }
}

impl Drop for RawMap {
    fn drop(&mut self) {
        // SAFETY: `[va, va + bytes)` is exactly the range this mapping
        // filled, and the reservation it lies in outlives it (a room drops
        // its pages before its range).
        let rc = unsafe { sys::cuMemUnmap(self.va, self.bytes) };
        if let Err(e) = crate::graph::cu(rc, "cuMemUnmap") {
            eprintln!("room: a page's unmap failed on drop: {e}");
        }
    }
}
