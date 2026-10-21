//! The Metal census: the devices this backend opens, in the placement's
//! terms.
//!
//! A Metal device is integrated: its memory is the host's memory, one
//! pool. Its total is what the driver says the device may work with —
//! `recommendedMaxWorkingSetSize` — and it has no PCI bus to name. Its
//! ordinal is its place in the driver's list of devices, and the first eight
//! bytes of its uuid are the driver's registry id: the one number a Metal
//! device has that tells two of one name apart.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCopyAllDevices, MTLDevice};

use bloomery_device::{Census, DeviceError};
use bloomery_placement::placement::devices::DeviceInfo;

/// The Metal devices of this machine.
pub struct MetalCensus;

impl MetalCensus {
    /// One device, as the census names it at `ordinal`.
    pub(crate) fn info_of(
        device: &ProtocolObject<dyn MTLDevice>,
        ordinal: u32,
    ) -> Result<DeviceInfo, DeviceError> {
        let total = device.recommendedMaxWorkingSetSize();
        let used = u64::try_from(device.currentAllocatedSize()).map_err(|_| {
            DeviceError::backend(
                "census",
                "the device's allocated size does not fit 64 bits".to_string(),
            )
        })?;
        let mut uuid = [0u8; 16];
        uuid[..8].copy_from_slice(&device.registryID().to_le_bytes());
        Ok(DeviceInfo {
            ordinal,
            name: device.name().to_string(),
            total_bytes: total,
            free_bytes: total.saturating_sub(used),
            integrated: true,
            uuid,
            pci_bus: String::new(),
            held_by: None,
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
