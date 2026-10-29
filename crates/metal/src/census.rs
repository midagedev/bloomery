//! The Metal census: the devices this backend opens, in the placement's
//! terms.
//!
//! A Metal device is integrated: its memory is the host's memory, one
//! pool. Its total is what the driver says the device may work with —
//! `recommendedMaxWorkingSetSize` — and it has no PCI bus to name. Its free
//! bytes are the host's room, the smaller of two inputs: the working set
//! less this process's own allocation (`currentAllocatedSize`), and the
//! host's available memory now (`os::host_memory`: the free and inactive
//! pages). The first alone overstates the room on a machine whose other
//! processes hold memory; the second alone ignores the driver's limit.
//! Its ordinal is its place in the driver's list of devices, and the first
//! eight bytes of its uuid are the driver's registry id: the one number a
//! Metal device has that tells two of one name apart.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCopyAllDevices, MTLDevice};

use bloomery_device::{Census, DeviceError};
use bloomery_placement::placement::devices::DeviceInfo;

/// The Metal devices of this machine.
pub struct MetalCensus;

impl MetalCensus {
    /// One device, as the census names it at `ordinal`. Its free bytes are
    /// [`free_of`] over the working set, this process's allocation and the
    /// host's available memory read now; a host read that fails is the
    /// census's error.
    pub(crate) fn info_of(
        device: &ProtocolObject<dyn MTLDevice>,
        ordinal: u32,
    ) -> Result<DeviceInfo, DeviceError> {
        let total = device.recommendedMaxWorkingSetSize();
        let used = Self::allocated_of(device)?;
        let host = os::host_memory().map(|memory| memory.available_bytes);
        let mut uuid = [0u8; 16];
        uuid[..8].copy_from_slice(&device.registryID().to_le_bytes());
        Ok(DeviceInfo {
            ordinal,
            name: device.name().to_string(),
            total_bytes: total,
            free_bytes: free_of(total, used, host)?,
            integrated: true,
            uuid,
            pci_bus: String::new(),
            held_by: None,
            compute_capability: None,
        })
    }

    /// The bytes of live device buffers this process holds on `device`: the
    /// driver's `currentAllocatedSize`, which the census's free bytes (the
    /// host's room) do not carry.
    pub(crate) fn allocated_of(device: &ProtocolObject<dyn MTLDevice>) -> Result<u64, DeviceError> {
        u64::try_from(device.currentAllocatedSize()).map_err(|_| {
            DeviceError::backend(
                "census",
                "the device's allocated size does not fit 64 bits".to_string(),
            )
        })
    }

    /// The place of `device` in the driver's list of devices, found by its
    /// registry id.
    pub(crate) fn ordinal_of(device: &ProtocolObject<dyn MTLDevice>) -> Result<u32, DeviceError> {
        let wanted = device.registryID();
        let all = MTLCopyAllDevices();
        let at = all
            .iter()
            .position(|d| d.registryID() == wanted)
            .ok_or_else(|| {
                DeviceError::backend(
                    "census",
                    "the default device is not in the driver's device list".to_string(),
                )
            })?;
        u32::try_from(at).map_err(|_| {
            DeviceError::backend(
                "census",
                "the device list is longer than 32 bits".to_string(),
            )
        })
    }
}

/// The free bytes of a device whose working set is `total` bytes, `used` of
/// them this process's allocation, on a host with `host_available` bytes of
/// memory available: the smaller of the working set's room left and the
/// host's. A working set already full or passed leaves 0. A host read that
/// failed is a named error, never the working set's figure alone.
fn free_of(
    total: u64,
    used: u64,
    host_available: Result<u64, os::OsError>,
) -> Result<u64, DeviceError> {
    let host = host_available.map_err(|e| {
        DeviceError::backend(
            "census",
            format!("the host's available memory could not be read: {e}"),
        )
    })?;
    Ok(total.saturating_sub(used).min(host))
}

impl Census for MetalCensus {
    fn census() -> Result<Vec<DeviceInfo>, DeviceError> {
        let devices = MTLCopyAllDevices();
        let mut out = Vec::with_capacity(devices.len());
        for (at, device) in devices.iter().enumerate() {
            let ordinal = u32::try_from(at).map_err(|_| {
                DeviceError::backend(
                    "census",
                    "the device list is longer than 32 bits".to_string(),
                )
            })?;
            out.push(Self::info_of(&device, ordinal)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn refused() -> os::OsError {
        os::OsError::Io(std::io::Error::other("host_statistics64 refused"))
    }

    /// The working set's room left is the free bytes while the host has more.
    #[test]
    fn working_set_room_binds_when_the_host_has_more() {
        assert_eq!(free_of(24 * GIB, 4 * GIB, Ok(30 * GIB)).unwrap(), 20 * GIB);
    }

    /// The host's available memory is the free bytes when it is the smaller.
    #[test]
    fn host_room_binds_when_the_host_has_less() {
        assert_eq!(free_of(24 * GIB, 4 * GIB, Ok(9 * GIB)).unwrap(), 9 * GIB);
    }

    /// Two equal inputs give that value; the working set already full or
    /// passed leaves 0 whatever the host has.
    #[test]
    fn equal_inputs_and_a_full_working_set() {
        assert_eq!(free_of(24 * GIB, 4 * GIB, Ok(20 * GIB)).unwrap(), 20 * GIB);
        assert_eq!(free_of(24 * GIB, 24 * GIB, Ok(9 * GIB)).unwrap(), 0);
        assert_eq!(free_of(24 * GIB, 26 * GIB, Ok(9 * GIB)).unwrap(), 0);
    }

    /// A host read that failed is the census's named error: not the working
    /// set's figure, and not 0.
    #[test]
    fn a_failed_host_read_is_a_named_error() {
        let e = free_of(24 * GIB, 4 * GIB, Err(refused())).unwrap_err();
        let DeviceError::Backend { what, source } = &e else {
            panic!("expected a backend error, got {e}");
        };
        assert_eq!(*what, "census");
        let text = source.to_string();
        assert!(
            text.contains("the host's available memory could not be read")
                && text.contains("host_statistics64 refused"),
            "the error does not name the host read and its cause: {text}"
        );
    }
}
