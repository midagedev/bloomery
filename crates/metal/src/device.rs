//! The Metal device: the contract's `Device` over `objc2-metal`.

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSProcessInfo;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLResourceOptions};

use bloomery_device::{
    Device, DeviceError, Elem, Elems, FaultWord, Stream, Window, WindowMut, check_len,
};
use bloomery_placement::placement::devices::DeviceInfo;

use crate::buffer::{MetalBuffer, MetalRaw, MetalShared};
use crate::census::MetalCensus;
use crate::fault::MetalFaultWord;
use crate::handle::Handle;
use crate::instance::Instance;
use crate::library::{Library, Pipeline};
use crate::stream::{MetalEvent, MetalStream};

/// The macOS floor this backend runs on: below it the backend is refused by
/// name.
pub(crate) const MACOS_FLOOR: (u32, u32, u32) = (26, 2, 0);

/// The operating system version this process runs on.
#[must_use]
pub(crate) fn os_version() -> (u32, u32, u32) {
    let version = NSProcessInfo::processInfo().operatingSystemVersion();
    let part = |n: isize| {
        u32::try_from(n).expect("an operating system version part is a small non-negative number")
    };
    (
        part(version.majorVersion),
        part(version.minorVersion),
        part(version.patchVersion),
    )
}

/// The refusal of a macOS below the floor this backend needs.
pub fn require_macos(major: u32, minor: u32, patch: u32) -> Result<(), DeviceError> {
    if (major, minor, patch) >= MACOS_FLOOR {
        return Ok(());
    }
    Err(DeviceError::unsupported(
        "macos",
        format!(
            "macOS {major}.{minor}.{patch} is below the floor {}.{}.{} this backend needs",
            MACOS_FLOOR.0, MACOS_FLOOR.1, MACOS_FLOOR.2
        ),
    ))
}

/// The Metal device the engine programs against.
pub struct MetalDevice {
    device: Handle<dyn MTLDevice>,
    queue: Handle<dyn MTLCommandQueue>,
    /// The device's own stream, on its queue.
    stream: MetalStream,
    library: Library,
    fault: MetalFaultWord,
    /// The device's place in the driver's list, as the census names it.
    ordinal: u32,
}

impl MetalDevice {
    /// Open the machine's default Metal device, its library compiled at
    /// `instance`: the one a file selects by its shape
    /// (`crate::instance::select`), or the host's for a run with no file.
    pub fn open(instance: &'static Instance) -> Result<Self, DeviceError> {
        let (major, minor, patch) = os_version();
        require_macos(major, minor, patch)?;

        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or_else(|| {
            DeviceError::unsupported(
                "metal device",
                "this machine has no Metal device".to_string(),
            )
        })?;
        let device = Handle(device);
        crate::stream::mark("queue", format_args!("newCommandQueue"));
        let queue = device.newCommandQueue().ok_or_else(|| {
            DeviceError::backend(
                "command queue",
                "the device made no command queue".to_string(),
            )
        })?;
        let queue = Handle(queue);
        let stream = MetalStream::new(device.clone(), queue.clone())?;
        let library = Library::compile(&device, instance)?;
        let fault = MetalFaultWord::new(&device)?;
        let ordinal = MetalCensus::ordinal_of(&device)?;
        Ok(Self {
            device,
            queue,
            stream,
            library,
            fault,
            ordinal,
        })
    }

    /// The device's own Metal object.
    #[must_use]
    pub fn metal(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The device's own handle, for the objects that outlive a borrow of it.
    pub(crate) fn handle(&self) -> &Handle<dyn MTLDevice> {
        &self.device
    }

    /// The bytes of live device buffers this process holds, as the driver
    /// counts them. The census's free bytes are the host's room, so a clause
    /// that asks what this process allocated reads this.
    pub fn allocated_bytes(&self) -> Result<u64, DeviceError> {
        MetalCensus::allocated_of(&self.device)
    }

    /// The library the kernels come from.
    #[must_use]
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// The compute pipeline of the named kernel of [`MetalDevice::library`].
    pub fn pipeline(&self, name: &str) -> Result<Pipeline, DeviceError> {
        self.library.pipeline(name)
    }

    /// The fault word this device's kernels raise, as the Metal backend's own
    /// type: the binding a kernel raises it through, and the whole word.
    #[must_use]
    pub fn fault_word(&self) -> &MetalFaultWord {
        &self.fault
    }

    /// A new stream on the device's queue, beside the device's own
    /// ([`Device::stream`]).
    pub fn new_stream(&self) -> Result<MetalStream, DeviceError> {
        MetalStream::new(self.device.clone(), self.queue.clone())
    }
}

impl Device for MetalDevice {
    type Buffer<T: Elem> = MetalBuffer<T>;
    type Shared<T: Elem> = MetalShared<T>;
    type Stream = MetalStream;
    type Event = MetalEvent;
    type Raw = MetalRaw;
    const BACKEND: &'static str = "metal";

    fn info(&self) -> Result<DeviceInfo, DeviceError> {
        MetalCensus::info_of(&self.device, self.ordinal)
    }

    fn stream(&self) -> &MetalStream {
        &self.stream
    }

    fn zeroed<T: Elem>(&self, _s: &Self::Stream, n: Elems) -> Result<Self::Buffer<T>, DeviceError> {
        let mut buffer = MetalBuffer::<T>::new(&self.device, n)?;
        buffer.zero();
        Ok(buffer)
    }

    fn upload<T: Elem>(
        &self,
        _s: &Self::Stream,
        host: &[T],
    ) -> Result<Self::Buffer<T>, DeviceError> {
        let mut buffer = MetalBuffer::<T>::new(&self.device, Elems(host.len()))?;
        buffer.write(host)?;
        Ok(buffer)
    }

    fn shared<T: Elem>(&self, n: Elems) -> Result<Self::Shared<T>, DeviceError> {
        MetalShared::<T>::new(&self.device, n)
    }

    // The copies are the host's: every buffer's pages are the host's pages,
    // so each waits for `s` and copies through the window's pages. A blit
    // would take whole 4-byte words only.

    fn read<T: Elem>(
        &self,
        s: &MetalStream,
        from: Window<'_, T, Self>,
        into: &mut [T],
    ) -> Result<(), DeviceError> {
        check_len("read", into.len(), from.len())?;
        s.sync()?;
        // SAFETY: `from` is a window of a live buffer, borrowed while the
        // pages live, and its pages are only read.
        let pages = unsafe {
            from.raw()
                .pages::<T>(from.byte_range()?.start, from.len())?
        };
        pages.read(into)
    }

    fn write<T: Elem>(
        &self,
        s: &MetalStream,
        to: WindowMut<'_, T, Self>,
        from: &[T],
    ) -> Result<(), DeviceError> {
        check_len("write", from.len(), to.len())?;
        s.sync()?;
        // SAFETY: `to` is a window of a live buffer held exclusively while the
        // pages live, so they are the only host access to its elements.
        let mut pages = unsafe { to.raw().pages::<T>(to.byte_range()?.start, to.len())? };
        pages.write(from)
    }

    fn copy<T: Elem>(
        &self,
        s: &MetalStream,
        from: Window<'_, T, Self>,
        to: WindowMut<'_, T, Self>,
    ) -> Result<(), DeviceError> {
        s.sync()?;
        // SAFETY: `from` is a window of a live buffer, borrowed while the
        // pages live, and its pages are only read.
        let source = unsafe {
            from.raw()
                .pages::<T>(from.byte_range()?.start, from.len())?
        };
        // SAFETY: `to` is a window of a live buffer held exclusively while the
        // pages live, so they are the only host access to its elements.
        let mut pages = unsafe { to.raw().pages::<T>(to.byte_range()?.start, to.len())? };
        pages.copy_from(&source)
    }

    fn fault(&self) -> &dyn FaultWord<Self::Stream> {
        &self.fault
    }
}

/// The options every buffer of this backend is made with: the host and the
/// device share the pages, and the barriers are ours to place.
pub(crate) fn buffer_options() -> MTLResourceOptions {
    MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked
}
